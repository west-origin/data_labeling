"""DB에 등록된 세션의 프라이버시 게이트 실행: 탐지 → (사람 검수) → 승인 → 블러본 렌더."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, blurred_key, sha256_file
from dlp_privacy.detection import FrameDetector
from dlp_privacy.pipeline import detect_video, model_version
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.render import render_blurred
from dlp_privacy.review import ReviewSegment
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    list_review_tasks,
    set_lifecycle,
    set_privacy_state,
)
from dlp_schema.episode import current_labels, retractions
from dlp_schema.labels import BlurTrackPayload, LabelRecord, Source, VerificationState
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTaskStatus
from dlp_schema.session import LifecycleState, PrivacyState, Session, StreamKind

VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


class PrivacyGateError(RuntimeError):
    pass


@dataclass
class DetectSummary:
    detected: dict[str, int] = field(default_factory=dict[str, int])  # 스트림 → 트랙 수
    skipped: list[str] = field(default_factory=list[str])
    missing: dict[str, str] = field(default_factory=dict[str, str])  # 대상 → 이유
    review_keys: list[str] = field(default_factory=list[str])
    changed: bool = False  # 블러 라벨 집합이 바뀌었는가 (새 라벨 또는 이전 버전 삭제)


def _fetch(store: ObjectStore, uri: str, work: Path) -> Path:
    prefix = store.uri("")
    if not uri.startswith(prefix):
        raise ValueError(f"{uri}는 저장소 {store.bucket}에 있지 않습니다")
    key = uri.removeprefix(prefix)
    dest = work / key.replace("/", "__")
    if not dest.exists():
        store.get_file(key, dest)
    return dest


def _blur_labels(conn: sa.Connection, session_id: str, stream_id: str) -> list[LabelRecord]:
    return [
        x
        for x in current_labels(get_labels(conn, session_id, kinds=["blur_track"]))
        if x.stream_id == stream_id and not x.seeded_error
    ]


def detect_session(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    detectors: dict[str, FrameDetector],
    missing_detectors: dict[str, str],
    policy: PrivacyPolicy,
    now: datetime,
    extra: Sequence[Predictor] = (),
) -> DetectSummary:
    """영상 스트림마다 블러 프리라벨을 만든다. 같은 모델 버전의 결과가 이미 있으면 건너뛴다.

    extra: 배포된 재학습 블러 모델 (blur_track을 내는 Predictor). 탐지기 결과와 합집합으로 남는다.
    """
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise PrivacyGateError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = DetectSummary()
    existing = get_labels(conn, session_id, kinds=["blur_track"])
    version = model_version(detectors, policy)
    versions = {version, *(p.version for p in extra)}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            done = {x.provenance.model_version for x in existing if x.stream_id == stream.stream_id}
            if versions <= done:
                summary.skipped.append(stream.stream_id)
                continue
            video = _fetch(raw, stream.uri, work)
            labels: list[LabelRecord] = []
            segments: list[ReviewSegment] = []
            if version not in done:
                result = detect_video(
                    video,
                    session_id=session_id,
                    stream_id=stream.stream_id,
                    detectors=detectors,
                    missing_detectors=missing_detectors,
                    policy=policy,
                    ontology_version=session.ontology_version,
                    now=now,
                )
                summary.missing.update(result.missing)
                labels += result.labels
                segments = result.segments
            for p in extra:
                if p.version not in done:
                    out = [
                        x
                        for x in p.run(Clip(session_id, stream.stream_id, video))
                        if isinstance(x.payload, BlurTrackPayload)
                    ]
                    labels += out
                    order = {r: i for i, r in enumerate(policy.review_priority)}
                    for x in out:
                        assert isinstance(x.payload, BlurTrackPayload)
                        segments.append(
                            ReviewSegment(
                                stream_id=stream.stream_id,
                                target=x.payload.target,
                                reason="trained_model",
                                t_start_ms=x.t_start_ms,
                                t_end_ms=x.t_end_ms,
                                priority=order.get("trained_model", len(order)),
                                detail=p.version,
                            )
                        )
            # 탐지기·모델 버전이 바뀌면 검수 전인 이전 버전 블러만 지운다 (검수한 블러는 남긴다)
            stale = [
                x
                for x in current_labels(existing)
                if x.stream_id == stream.stream_id
                and x.provenance.source is Source.MODEL
                and x.provenance.model_version not in versions
                and x.verification.state is VerificationState.UNREVIEWED
            ]
            changes = [*retractions(stale, version, now), *labels]
            insert_labels(conn, changes)
            summary.changed |= bool(changes)
            summary.detected[stream.stream_id] = len(labels)
            if segments or version not in done:
                key = f"sessions/{session_id}/derived/privacy_review/{stream.stream_id}.json"
                out_path = work / f"{stream.stream_id}-review.json"
                out_path.write_text(
                    json.dumps([s.model_dump(mode="json") for s in segments], ensure_ascii=False),
                    encoding="utf-8",
                )
                raw.put_file(key, out_path, sha256_file(out_path))
                summary.review_keys.append(key)
    if session.privacy_state is PrivacyState.PENDING or (
        summary.changed and session.privacy_state is PrivacyState.APPROVED
    ):
        # 승인 뒤 블러가 바뀌면 승인을 푼다: 새 블러를 사람이 검수하고 다시 승인해야 렌더한다
        set_privacy_state(conn, session_id, PrivacyState.AUTO_BLURRED)
    return summary


def last_detection(labels: Sequence[LabelRecord], stream_id: str) -> datetime | None:
    """스트림의 마지막 자동 탐지 반영 시각 (모델 출처 블러 레코드, 삭제 레코드 포함)."""
    times = [
        x.created_at
        for x in labels
        if x.stream_id == stream_id and x.provenance.source is Source.MODEL
    ]
    return max(times) if times else None


def missing_privacy_reviews(conn: sa.Connection, session: Session) -> list[str]:
    """마지막 자동 탐지 뒤에 만들어 수집까지 끝난 블러 검수 작업이 없는 영상 스트림.

    탐지 결과가 하나도 없어도(놓친 대상이 있을 수 있다) 사람이 영상 전체를 본 작업이 있어야 한다.
    운영 작업(표준·QA)만 센다. 블라인드·이중·오류 삽입 작업은 운영 블러를 검수하지 않는다.
    """
    labels = get_labels(conn, session.session_id, kinds=["blur_track"])
    tasks = [
        t
        for t in list_review_tasks(conn, session.session_id, ReviewStage.PRIVACY)
        if t.status is ReviewTaskStatus.COLLECTED and t.mode in (ReviewMode.STANDARD, ReviewMode.QA)
    ]
    missing: list[str] = []
    for s in session.streams:
        if s.kind not in VIDEO_KINDS:
            continue
        since = last_detection(labels, s.stream_id)
        if not any(
            t.stream_id == s.stream_id and (since is None or t.created_at >= since) for t in tasks
        ):
            missing.append(s.stream_id)
    return missing


def approve_session(conn: sa.Connection, session_id: str) -> Session:
    """현재 블러 라벨이 모두 사람 검수를 거쳤고, 영상 스트림마다 마지막 자동 탐지 뒤의 블러 검수
    작업을 수집했을 때만 프라이버시 승인한다."""
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise PrivacyGateError(f"{session_id}: 자동 탐지가 아직 실행되지 않았습니다")
    missing = missing_privacy_reviews(conn, session)
    if missing:
        raise PrivacyGateError(
            f"{session_id}: 마지막 자동 탐지 뒤 수집한 블러 검수 작업이 없는 스트림 {missing} "
            "(dlp review create --stage privacy → 검수 → collect)"
        )
    unreviewed = [
        x.label_id
        for s in session.streams
        if s.kind in VIDEO_KINDS
        for x in _blur_labels(conn, session_id, s.stream_id)
        if x.verification.state in (VerificationState.UNREVIEWED, VerificationState.SAMPLE_VERIFIED)
    ]
    if unreviewed:
        raise PrivacyGateError(
            f"{session_id}: 사람 검수를 거치지 않은 블러 라벨 {len(unreviewed)}개 "
            f"({unreviewed[:3]} …)"
        )
    set_privacy_state(conn, session_id, PrivacyState.APPROVED)
    if session.lifecycle_state is LifecycleState.RAW_INGESTED:
        # 블러를 고친 뒤 다시 승인할 때는 생애주기가 이미 앞에 있다 (되돌아가지 않는다)
        set_lifecycle(conn, session_id, LifecycleState.PRIVACY_APPROVED)
    return get_session(conn, session_id)


def render_hash(labels: Sequence[LabelRecord], policy: PrivacyPolicy) -> str:
    """블러본을 정하는 입력의 해시: 블러 라벨 집합(라벨은 불변이라 ID로 충분)과 렌더 정책."""
    content = json.dumps(
        {
            "labels": sorted(x.label_id for x in labels),
            "mode": policy.platform.render_mode,
            "render": policy.render.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def render_meta_key(session_id: str, stream_id: str) -> str:
    """블러본 옆에 두는 렌더 입력 기록 (라벨링 버킷). 라벨 ID 해시만 담는다."""
    return blurred_key(session_id, stream_id).removesuffix(".mp4") + ".render.json"


def _rendered_hash(
    labeling: ObjectStore, session_id: str, stream_id: str, work: Path
) -> str | None:
    key = blurred_key(session_id, stream_id)
    meta = render_meta_key(session_id, stream_id)
    if labeling.head(key) is None or labeling.head(meta) is None:
        return None
    dest = work / f"{stream_id}.render.json"
    labeling.get_file(meta, dest)
    value: object = json.loads(dest.read_text(encoding="utf-8")).get("render_hash")
    return value if isinstance(value, str) else None


def render_session(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    labeling: ObjectStore,
    policy: PrivacyPolicy,
) -> dict[str, str]:
    """승인된 세션의 블러본을 라벨링 버킷에 만든다. 스트림 → URI.

    같은 블러 라벨 집합·렌더 정책으로 만든 블러본이 이미 있으면 건너뛴다 (멱등). 검수로 블러가
    바뀌어 다시 승인했으면 해시가 달라져 새로 렌더한다.
    """
    if not policy.platform.strip_audio_in_release:
        raise PrivacyGateError(
            "오디오를 남기는 블러본은 지원하지 않습니다 (strip_audio_in_release)"
        )
    if labeling.bucket == raw.bucket:
        raise PrivacyGateError("블러본은 원본 버킷이 아닌 라벨링 버킷에 써야 합니다")
    session = get_session(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise PrivacyGateError(f"{session_id}: 프라이버시 승인 전에는 블러본을 만들 수 없습니다")
    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            key = blurred_key(session_id, stream.stream_id)
            labels = _blur_labels(conn, session_id, stream.stream_id)
            digest = render_hash(labels, policy)
            if _rendered_hash(labeling, session_id, stream.stream_id, work) != digest:
                dst = work / f"{stream.stream_id}-blurred.mp4"
                render_blurred(
                    _fetch(raw, stream.uri, work),
                    dst,
                    labels,
                    mode=policy.platform.render_mode,
                    min_block_px=policy.render.min_block_px,
                    blocks_per_box=policy.render.blocks_per_box,
                    encoder_rate=policy.render.encoder_rate,
                    crf=policy.render.crf,
                )
                labeling.put_file(key, dst, sha256_file(dst))
                meta = work / f"{stream.stream_id}.render.out.json"
                meta.write_text(json.dumps({"render_hash": digest}), encoding="utf-8")
                labeling.put_file(
                    render_meta_key(session_id, stream.stream_id), meta, sha256_file(meta)
                )
            out[stream.stream_id] = labeling.uri(key)
    return out
