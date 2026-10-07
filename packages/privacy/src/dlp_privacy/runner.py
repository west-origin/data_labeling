"""DB에 등록된 세션의 프라이버시 게이트 실행: 탐지 → (사람 검수) → 승인 → 블러본 렌더."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, sha256_file
from dlp_privacy.detection import FrameDetector
from dlp_privacy.pipeline import detect_video, model_version
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.render import render_blurred
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    set_lifecycle,
    set_privacy_state,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, VerificationState
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
) -> DetectSummary:
    """영상 스트림마다 블러 프리라벨을 만든다. 같은 모델 버전의 결과가 이미 있으면 건너뛴다."""
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise PrivacyGateError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = DetectSummary()
    existing = get_labels(conn, session_id, kinds=["blur_track"])
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            version = model_version(detectors, policy)
            if any(
                x.stream_id == stream.stream_id and x.provenance.model_version == version
                for x in existing
            ):
                summary.skipped.append(stream.stream_id)
                continue
            result = detect_video(
                _fetch(raw, stream.uri, work),
                session_id=session_id,
                stream_id=stream.stream_id,
                detectors=detectors,
                missing_detectors=missing_detectors,
                policy=policy,
                ontology_version=session.ontology_version,
                now=now,
            )
            summary.missing.update(result.missing)
            insert_labels(conn, result.labels)
            summary.detected[stream.stream_id] = len(result.labels)
            key = f"sessions/{session_id}/derived/privacy_review/{stream.stream_id}.json"
            out = work / f"{stream.stream_id}-review.json"
            out.write_text(
                json.dumps(
                    [s.model_dump(mode="json") for s in result.segments], ensure_ascii=False
                ),
                encoding="utf-8",
            )
            raw.put_file(key, out, sha256_file(out))
            summary.review_keys.append(key)
    if session.privacy_state is PrivacyState.PENDING:
        set_privacy_state(conn, session_id, PrivacyState.AUTO_BLURRED)
    return summary


def approve_session(conn: sa.Connection, session_id: str) -> Session:
    """현재 블러 라벨이 모두 사람 검수를 거쳤을 때만 프라이버시 승인한다."""
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise PrivacyGateError(f"{session_id}: 자동 탐지가 아직 실행되지 않았습니다")
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
    set_lifecycle(conn, session_id, LifecycleState.PRIVACY_APPROVED)
    return get_session(conn, session_id)


def render_session(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    labeling: ObjectStore,
    policy: PrivacyPolicy,
) -> dict[str, str]:
    """승인된 세션의 블러본을 라벨링 버킷에 만든다. 이미 있으면 건너뛴다. 스트림 → URI."""
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
            key = f"sessions/{session_id}/blurred/{stream.stream_id}.mp4"
            if labeling.head(key) is None:
                dst = work / f"{stream.stream_id}-blurred.mp4"
                render_blurred(
                    _fetch(raw, stream.uri, work),
                    dst,
                    _blur_labels(conn, session_id, stream.stream_id),
                    mode=policy.platform.render_mode,
                    min_block_px=policy.render.min_block_px,
                    blocks_per_box=policy.render.blocks_per_box,
                )
                labeling.put_file(key, dst, sha256_file(dst))
            out[stream.stream_id] = labeling.uri(key)
    return out
