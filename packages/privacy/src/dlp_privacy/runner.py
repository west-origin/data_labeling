"""DB에 등록된 세션의 프라이버시 게이트 실행: 탐지 → (사람 검수) → 승인 → 블러본 렌더.

WP5, ADR 0005·0019·0023·0024. CLI `dlp_cli.privacy_cmds`가 DB 연결과 저장소를 만들어 부른다.

단계와 진입점:
- `detect_session` (`dlp privacy detect`): 원본 영상 → blur_track 라벨(DB `label_records`) +
  검수 우선 구간(원본 버킷 JSON). 세션 `privacy_state`를 AUTO_BLURRED로.
- (사람 검수: `dlp review create --stage privacy` → CVAT → `dlp review collect`. dlp_review 담당)
- `approve_session` (`dlp privacy approve`): 승인 조건을 확인하고 APPROVED로.
- `render_session` (`dlp privacy render`): 승인된 세션의 블러본과 렌더 기록을 라벨링 버킷에 쓴다.
- `assert_render_current` / `check_fetched`: 블러본 소비자(검수 작업 생성, 행동 VLM, 큐레이션,
  내보내기)가 쓰기 전에 부르는 확인 (ADR 0024).
- `invalidate_render`: 블러가 바뀐 스트림의 렌더 기록을 무효로 만든다 (dlp_review의 블러 검수
  수집도 부른다).

버킷:
- raw(원본 버킷 `dlp-raw`): 원본 영상을 읽고 검수 우선 구간 JSON을 읽고 쓴다. CLI는 반드시
  `dlp_cli.raw_access.raw_store`로 만든 `AuditedStore`를 넘기므로 `get_file`/`put_file`마다 원본
  접근 기록(`raw_access_log`)이 남는다 (ADR 0020). `head`는 기록하지 않는다.
- labeling(라벨링 버킷 `dlp-labeling`): 블러본(`blurred_key`)과 렌더 기록(`render_meta_key`)만 쓴다.
  블러본은 원본 버킷에 쓰지 않는다 (`render_session`이 막는다).

트랜잭션: DB를 쓰는 함수는 호출자가 연 트랜잭션(`engine.begin()`) 안의 연결을 받는다. 저장소 쓰기는
트랜잭션 밖 부작용이라 DB가 되돌아가도 남는다 (다시 실행하면 같은 키에 덮어쓰거나 건너뛴다).

규칙 (CLAUDE.md, ADR 0015·0019):
- 라벨은 덮어쓰지 않는다. 버전이 바뀌면 검수 전(UNREVIEWED)인 이전 모델 블러만 `retractions`로
  지운다. 사람이 검수한 블러는 지우지 않는다.
- 다시 돌릴지는 전체 이력(`get_labels`)에 같은 모델 버전 레코드가 있는지로 정한다.
- 운영 블러(`operational_blur`)만 승인·렌더에 쓴다. 오류 삽입·측정 레코드와 그 후손은 뺀다.
"""

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
from dlp_schema.episode import current_labels, non_operational_ids, retractions
from dlp_schema.labels import BlurTrackPayload, LabelRecord, Source, VerificationState
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTaskStatus
from dlp_schema.session import LifecycleState, PrivacyState, Session, StreamKind

# 블러 대상인 영상 스트림 종류 (IMU·장갑·오디오 스트림은 게이트를 거치지 않는다)
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


# 게이트 조건을 만족하지 않아 단계를 진행할 수 없다 (승인 전 렌더, 미검수 블러 등).
# 메시지에 다음에 할 명령을 적는다.
class PrivacyGateError(RuntimeError):
    pass


class RenderNotCurrentError(PrivacyGateError):
    """블러본이 지금 승인된 블러 라벨·렌더 정책으로 만든 것이 아니다 (쓰면 안 된다)."""


@dataclass
class DetectSummary:
    """`detect_session` 결과 요약 (CLI 출력용)."""

    detected: dict[str, int] = field(default_factory=dict[str, int])  # 스트림 → 트랙 수
    # 같은 모델 버전 결과가 이미 있어 건너뛴 스트림
    skipped: list[str] = field(default_factory=list[str])
    missing: dict[str, str] = field(default_factory=dict[str, str])  # 대상 → 이유
    # 이번에 쓴 검수 우선 구간 객체 키 (원본 버킷)
    review_keys: list[str] = field(default_factory=list[str])
    changed: bool = False  # 블러 라벨 집합이 바뀌었는가 (새 라벨 또는 이전 버전 삭제)


def _fetch(store: ObjectStore, uri: str, work: Path) -> Path:
    """저장소 URI의 객체를 work 아래에 받는다 (이미 받았으면 다시 받지 않는다).

    Args:
        store: 그 URI가 속한 저장소 (원본이면 감사 저장소 → 읽기 기록이 남는다).
        uri: `store.uri(key)` 형식의 URI (Stream.uri).
        work: 임시 작업 디렉터리.

    Returns:
        로컬 파일 경로 (키의 "/"를 "__"로 바꾼 이름).

    Raises:
        ValueError: URI가 이 저장소(버킷)에 있지 않을 때.
    """
    prefix = store.uri("")
    if not uri.startswith(prefix):
        raise ValueError(f"{uri}는 저장소 {store.bucket}에 있지 않습니다")
    key = uri.removeprefix(prefix)
    dest = work / key.replace("/", "__")
    if not dest.exists():
        store.get_file(key, dest)
    return dest


def operational_blur(history: Sequence[LabelRecord], stream_id: str) -> list[LabelRecord]:
    """스트림의 운영 블러 라벨 (오류 삽입·측정 레코드와 그 후손 제외). 블러본은 이것으로 만든다.

    Args:
        history: 세션 라벨 전체 이력 (다른 종류가 섞여도 된다).
        stream_id: 영상 스트림 ID.

    Returns:
        그 스트림의 현재(최신·삭제되지 않은) 운영 blur_track 레코드.
    """
    # current_labels는 기본으로 운영 라벨만 남긴다. seeded_error 검사는 이중 안전장치다.
    return [
        x
        for x in current_labels([x for x in history if x.kind == "blur_track"])
        if x.stream_id == stream_id and not x.seeded_error
    ]


def _blur_labels(conn: sa.Connection, session_id: str, stream_id: str) -> list[LabelRecord]:
    """DB에서 그 스트림의 운영 블러 라벨을 읽는다 (`operational_blur`)."""
    return operational_blur(get_labels(conn, session_id, kinds=["blur_track"]), stream_id)


def review_key(session_id: str, stream_id: str) -> str:
    """원본 버킷의 검수 우선 구간 목록 (블러 검수 화면이 먼저 보여 줄 구간).

    원본 영상의 내용(대상 위치·시각)을 담으므로 라벨링 버킷이 아니라 원본 버킷에 둔다.
    """
    return f"sessions/{session_id}/derived/privacy_review/{stream_id}.json"


def read_review_segments(
    raw: ObjectStore, session_id: str, stream_id: str, work: Path
) -> list[ReviewSegment]:
    """저장한 검수 우선 구간. 없으면 빈 목록.

    부작용: 원본 버킷 읽기 (감사 저장소면 read 기록).
    """
    key = review_key(session_id, stream_id)
    if raw.head(key) is None:
        return []
    dest = work / f"{stream_id}-review.in.json"
    raw.get_file(key, dest)
    data: list[object] = json.loads(dest.read_text(encoding="utf-8"))
    return [ReviewSegment.model_validate(x) for x in data]


def merge_segments(
    old: Sequence[ReviewSegment],
    new: Sequence[ReviewSegment],
    *,
    detector_rerun: bool,
    rerun_models: set[str],
    live_models: set[str],
) -> list[ReviewSegment]:
    """이번 실행이 다시 만든 구간만 바꾸고 나머지 이전 구간은 남긴다.

    - 탐지기가 다시 돌았으면 탐지기 구간(trained_model 외)을 새것으로 바꾼다.
    - 다시 돈 재학습 모델(rerun_models)의 구간을 새것으로 바꾼다.
    - 더 이상 쓰지 않는 재학습 모델(live_models 밖)의 구간은 뺀다.

    회귀(ADR 0024 감사 4-8): 재학습 모델만 다시 돌았을 때 파일을 통째로 덮어써 탐지기 구간이
    사라졌다.

    Args:
        old: 저장돼 있던 구간.
        new: 이번 실행이 만든 구간.
        detector_rerun: 탐지기 파이프라인이 이번에 다시 돌았는가.
        rerun_models: 이번에 다시 돈 재학습 모델 버전들.
        live_models: 지금 배포된(이번 실행에 넘긴) 재학습 모델 버전들.

    Returns:
        중복(내용이 완전히 같은 구간)을 없애고 (우선순위, 시작, 대상) 순으로 정렬한 목록.
    """
    kept = [
        s
        for s in old
        if (
            # trained_model 구간: detail(모델 버전)이 다시 돌지 않았고 아직 배포 중이면 남긴다
            s.detail not in rerun_models and s.detail in live_models
            if s.reason == "trained_model"
            # 탐지기 구간: 탐지기가 다시 돌지 않았으면 남긴다
            else not detector_rerun
        )
    ]
    merged = {s.model_dump_json(): s for s in [*kept, *new]}
    return sorted(merged.values(), key=lambda s: (s.priority, s.t_start_ms, s.target))


def detect_session(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    detectors: dict[str, FrameDetector],
    missing_detectors: dict[str, str],
    policy: PrivacyPolicy,
    now: datetime,
    extra: Sequence[Predictor] = (),
    labeling: ObjectStore | None = None,
) -> DetectSummary:
    """영상 스트림마다 블러 프리라벨을 만든다. 같은 모델 버전의 결과가 이미 있으면 건너뛴다.

    extra: 배포된 재학습 블러 모델 (blur_track을 내는 Predictor). 탐지기 결과와 합집합으로 남는다.
    labeling: 블러가 바뀐 스트림의 렌더 기록을 무효로 만들 라벨링 버킷 (ADR 0024). 없어도
    블러본을 쓰는 쪽이 assert_render_current로 막지만, 있으면 이전 블러본을 바로 무효로 둔다.

    스트림마다:
    1. 이력에 이 탐지 버전·재학습 모델 버전의 레코드가 모두 있으면 건너뛴다 (멱등).
    2. 원본 영상을 받아(감사 기록) 아직 없는 버전만 돌린다 (탐지기 파이프라인, 재학습 모델).
    3. 지금 쓰지 않는 버전의 검수 전 모델 블러를 삭제 레코드로 지우고 새 라벨과 함께 쓴다.
    4. 블러가 바뀌었으면 렌더 기록을 무효로 하고, 검수 우선 구간 파일을 병합해 다시 쓴다.
    끝으로 PENDING이거나 (승인 상태에서 블러가 바뀌었으면) privacy_state를 AUTO_BLURRED로 둔다.

    Args:
        conn: 호출자가 연 트랜잭션의 DB 연결.
        session_id: 세션 ID (DB에 등록돼 있어야 한다).
        raw: 원본 버킷 저장소 (CLI는 감사 저장소).
        detectors, missing_detectors: `detectors.build_detectors`의 결과.
        policy: 프라이버시 정책.
        now: 새 레코드의 created_at (시간대 필수).

    Returns:
        `DetectSummary`.

    Raises:
        PrivacyGateError: 세션에 온톨로지 버전이 없을 때.
        sqlalchemy.exc.NoResultFound: 세션이 없을 때.

    부작용: DB `label_records` 추가, `sessions.privacy_state` 변경, 원본 버킷 읽기·쓰기(검수 우선
    구간), 라벨링 버킷 렌더 기록 덮어쓰기(labeling이 있을 때).
    """
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise PrivacyGateError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = DetectSummary()
    # 전체 이력(삭제·수정 레코드 포함): 재실행 여부는 현재 라벨이 아니라 이력으로 정한다
    existing = get_labels(conn, session_id, kinds=["blur_track"])
    version = model_version(detectors, policy)
    versions = {version, *(p.version for p in extra)}  # 이번에 있어야 할 모든 모델 버전
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            # 이 스트림에 이미 레코드가 있는 모델 버전 (삭제 레코드의 버전도 포함된다)
            done = {
                v
                for x in existing
                if x.stream_id == stream.stream_id and (v := x.provenance.model_version) is not None
            }
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
                    # 재학습 모델이 낸 블러는 전부 trained_model 검수 구간으로 낸다 (detail=버전)
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
            # 삭제 레코드의 출처 버전은 이번 탐지 버전이다 (재학습 모델만 바뀐 경우에도)
            changes = [*retractions(stale, version, now), *labels]
            insert_labels(conn, changes)
            summary.changed |= bool(changes)
            summary.detected[stream.stream_id] = len(labels)
            if changes and labeling is not None:
                invalidate_render(labeling, session_id, stream.stream_id, "blur_changed", work)
            rerun_models = {p.version for p in extra if p.version not in done}
            if segments or version not in done or rerun_models:
                # 재학습 모델만 다시 돌았으면 탐지기 구간을 지우지 않는다 (합친다)
                merged = merge_segments(
                    read_review_segments(raw, session_id, stream.stream_id, work),
                    segments,
                    detector_rerun=version not in done,
                    rerun_models=rerun_models,
                    live_models={p.version for p in extra},
                )
                key = review_key(session_id, stream.stream_id)
                out_path = work / f"{stream.stream_id}-review.json"
                out_path.write_text(
                    json.dumps([s.model_dump(mode="json") for s in merged], ensure_ascii=False),
                    encoding="utf-8",
                )
                # 검수 우선 구간은 파생물이라 불변 업로드(put_immutable)가 아니라 덮어쓴다
                raw.put_file(key, out_path, sha256_file(out_path))
                summary.review_keys.append(key)
    if session.privacy_state is PrivacyState.PENDING or (
        summary.changed and session.privacy_state is PrivacyState.APPROVED
    ):
        # 승인 뒤 블러가 바뀌면 승인을 푼다: 새 블러를 사람이 검수하고 다시 승인해야 렌더한다
        set_privacy_state(conn, session_id, PrivacyState.AUTO_BLURRED)
    return summary


def last_detection(labels: Sequence[LabelRecord], stream_id: str) -> datetime | None:
    """스트림의 마지막 자동 탐지 반영 시각 (모델 출처 블러 레코드, 삭제 레코드 포함).

    오류 삽입 사본·측정 레코드와 그 후손은 운영 블러가 아니므로 세지 않는다 (오류 삽입 계획이
    모델 출처 사본을 지금 시각으로 만들어도 승인 조건이 움직이지 않게).

    Args:
        labels: 세션 블러 라벨 전체 이력.
        stream_id: 영상 스트림 ID.

    Returns:
        가장 늦은 created_at. 모델 출처 운영 레코드가 없으면 None.
    """
    excluded = non_operational_ids(list(labels))
    times = [
        x.created_at
        for x in labels
        if x.stream_id == stream_id
        and x.provenance.source is Source.MODEL
        and x.label_id not in excluded
    ]
    return max(times) if times else None


def missing_privacy_reviews(conn: sa.Connection, session: Session) -> list[str]:
    """마지막 자동 탐지 뒤에 만들어 수집까지 끝난 블러 검수 작업이 없는 영상 스트림.

    탐지 결과가 하나도 없어도(놓친 대상이 있을 수 있다) 사람이 영상 전체를 본 작업이 있어야 한다.
    운영 작업(표준·QA)만 센다. 블라인드·이중·오류 삽입 작업은 운영 블러를 검수하지 않는다.

    "뒤에"는 작업 created_at ≥ 마지막 탐지 시각으로 본다 (재탐지 전에 만든 작업은 새 블러를 보지
    않았다). 탐지 기록이 없으면 수집한 운영 작업이 하나라도 있으면 된다.

    Returns:
        조건을 만족하지 못한 영상 스트림 ID 목록 (세션 스트림 순서).
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
    작업을 수집했을 때만 프라이버시 승인한다.

    조건:
    1. 자동 탐지를 한 번 이상 돌렸다 (privacy_state가 PENDING이 아니다).
    2. 영상 스트림마다 `missing_privacy_reviews` 조건을 만족한다.
    3. 운영 블러 라벨 중 UNREVIEWED·SAMPLE_VERIFIED가 없다 (블러는 표본 검증으로 승인하지 않는다:
       전수 사람 검수가 필요하다).

    Args:
        conn: 호출자가 연 트랜잭션의 DB 연결.
        session_id: 세션 ID.

    Returns:
        승인 후 다시 읽은 세션.

    Raises:
        PrivacyGateError: 조건을 만족하지 않을 때 (메시지에 다음에 할 명령을 적는다).

    부작용: `sessions.privacy_state` = APPROVED, 처음 승인이면 생애주기를 PRIVACY_APPROVED로.
    """
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
    """블러본을 정하는 입력의 해시: 블러 라벨 집합(라벨은 불변이라 ID로 충분)과 렌더 정책.

    Args:
        labels: 운영 블러 라벨 (`operational_blur`).
        policy: 렌더 정책(`render`)과 `platform.render_mode`만 쓴다.

    Returns:
        sha256 16진수 64자. 렌더 기록(`RenderMeta.render_hash`)과 비교한다.
    """
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
    """블러본 옆에 두는 렌더 입력 기록 (라벨링 버킷). 라벨 ID 해시만 담는다.

    예: `sessions/<세션>/blurred/<스트림>.render.json`.
    """
    return blurred_key(session_id, stream_id).removesuffix(".mp4") + ".render.json"


@dataclass(frozen=True)
class RenderMeta:
    """블러본 옆 렌더 기록. render_hash가 None이면 무효로 둔 기록이다 (승인 취소 등)."""

    # 렌더 당시 `render_hash` 값 (None이면 무효)
    render_hash: str | None
    blurred_sha256: str | None = None  # 렌더한 블러본 파일 해시 (이전 기록에는 없다)


def read_render_meta(
    labeling: ObjectStore, session_id: str, stream_id: str, work: Path
) -> RenderMeta | None:
    """블러본과 렌더 기록이 둘 다 있으면 기록을, 아니면 None.

    부작용: 라벨링 버킷 읽기, work 디렉터리 생성.
    """
    key = blurred_key(session_id, stream_id)
    meta = render_meta_key(session_id, stream_id)
    if labeling.head(key) is None or labeling.head(meta) is None:
        return None
    work.mkdir(parents=True, exist_ok=True)
    dest = work / f"{stream_id}.render.json"
    labeling.get_file(meta, dest)
    data: dict[str, object] = json.loads(dest.read_text(encoding="utf-8"))
    value, sha = data.get("render_hash"), data.get("blurred_sha256")
    # 문자열이 아닌 값(null 등)은 None으로 본다 (무효 기록·이전 형식)
    return RenderMeta(
        value if isinstance(value, str) else None, sha if isinstance(sha, str) else None
    )


def _rendered_hash(
    labeling: ObjectStore, session_id: str, stream_id: str, work: Path
) -> str | None:
    """지금 렌더 기록의 render_hash (기록·블러본이 없거나 무효면 None)."""
    meta = read_render_meta(labeling, session_id, stream_id, work)
    return None if meta is None else meta.render_hash


def write_render_meta(
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    meta: RenderMeta,
    work: Path,
    **extra: str,
) -> None:
    """렌더 기록 JSON을 라벨링 버킷에 (덮어)쓴다.

    extra: 기록에 덧붙일 문자열 필드 (예: invalidated="blur_changed").
    """
    path = work / f"{stream_id}.render.out.json"
    body = {"render_hash": meta.render_hash, "blurred_sha256": meta.blurred_sha256, **extra}
    path.write_text(json.dumps(body), encoding="utf-8")
    labeling.put_file(render_meta_key(session_id, stream_id), path, sha256_file(path))


def invalidate_render(
    labeling: ObjectStore, session_id: str, stream_id: str, reason: str, work: Path
) -> bool:
    """렌더 기록을 무효로 덮어쓴다 (블러본 파일은 남지만 어떤 단계도 현재 것으로 보지 않는다).

    라벨링 버킷 저장소에는 삭제가 없어 기록을 render_hash=None으로 바꾼다. 무효로 했으면 True.

    Args:
        reason: 무효 사유 (기록의 invalidated 필드, 예: "blur_changed").

    Returns:
        렌더 기록이 있어 무효로 바꿨으면 True, 기록이 없으면 False.
    """
    if labeling.head(render_meta_key(session_id, stream_id)) is None:
        return False
    write_render_meta(labeling, session_id, stream_id, RenderMeta(None), work, invalidated=reason)
    return True


def expected_render_hash(
    conn: sa.Connection, session_id: str, stream_id: str, policy: PrivacyPolicy
) -> str:
    """지금 DB 상태로 본 블러본 해시. 세션이 지금 프라이버시 승인 상태가 아니면 실패한다
    (데이터셋 스냅샷의 세션 상태가 아니라 현재 상태를 본다).

    Raises:
        RenderNotCurrentError: 세션이 지금 APPROVED가 아닐 때.
    """
    session = get_session(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise RenderNotCurrentError(
            f"{session_id}: 지금 프라이버시 승인 상태가 아닙니다 ({session.privacy_state.value}). "
            "블러 검수·승인(dlp privacy approve)·렌더(dlp privacy render)를 다시 하세요"
        )
    return render_hash(_blur_labels(conn, session_id, stream_id), policy)


def check_render_meta(
    meta: RenderMeta | None,
    expected: str,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
) -> str | None:
    """렌더 기록이 기대 해시와 같고 블러본 파일이 그 렌더의 것인지 본다.

    렌더한 블러본 파일 해시를 돌려준다 (이전 렌더 기록이면 None).

    Args:
        meta: `read_render_meta` 결과.
        expected: `expected_render_hash` 결과 (또는 내보내기가 미리 계산해 둔 값).

    Raises:
        RenderNotCurrentError: 기록이 없거나 무효, 해시가 다름, 블러본이 없음, 블러본 파일 해시가
            기록과 다름 (렌더 밖에서 덮어씀).
    """
    where = f"{session_id}/{stream_id}"
    if meta is None or meta.render_hash is None:
        raise RenderNotCurrentError(
            f"{where}: 현재 블러본 렌더 기록이 없거나 무효입니다 (dlp privacy render)"
        )
    if meta.render_hash != expected:
        raise RenderNotCurrentError(
            f"{where}: 블러본이 지금 승인된 블러 라벨·렌더 정책으로 만든 것이 아닙니다 "
            "(dlp privacy render)"
        )
    head = labeling.head(blurred_key(session_id, stream_id))
    if head is None:
        raise RenderNotCurrentError(f"{where}: 블러본이 없습니다 (dlp privacy render)")
    # 둘 다 해시가 있을 때만 비교한다 (이전 형식 기록이나 해시 메타데이터가 없는 객체는 넘어간다)
    if meta.blurred_sha256 and head.sha256 and head.sha256 != meta.blurred_sha256:
        raise RenderNotCurrentError(f"{where}: 블러본 파일이 렌더 기록과 다릅니다")
    return meta.blurred_sha256


def assert_render_current(
    conn: sa.Connection,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    policy: PrivacyPolicy,
) -> str | None:
    """블러본을 읽는 모든 단계(검수 작업·행동 VLM·큐레이션·내보내기)가 먼저 부른다 (ADR 0024).

    세션이 **지금** 프라이버시 승인 상태이고, 블러본 렌더 기록의 해시가 지금 운영 블러 라벨과
    렌더 정책의 해시와 같아야 한다. 아니면 RenderNotCurrentError. 렌더한 블러본 파일의 sha256을
    돌려준다 (이전 렌더 기록이면 None). 받은 파일과 비교해 그 사이 다시 렌더된 것을 막는다.

    사용 예::

        sha = assert_render_current(conn, labeling, sid, "bodycam", policy)
        labeling.get_file(blurred_key(sid, "bodycam"), local)
        check_fetched(local, sha, sid, "bodycam")
    """
    expected = expected_render_hash(conn, session_id, stream_id, policy)
    with tempfile.TemporaryDirectory() as tmp:
        meta = read_render_meta(labeling, session_id, stream_id, Path(tmp))
    return check_render_meta(meta, expected, labeling, session_id, stream_id)


def check_fetched(path: Path, expected_sha: str | None, session_id: str, stream_id: str) -> None:
    """받은 블러본이 렌더 기록의 파일인지 본다 (확인과 받기 사이에 다시 렌더된 경우).

    Args:
        path: 받은 로컬 블러본.
        expected_sha: `assert_render_current`의 반환값. None이면 (이전 기록) 확인하지 않는다.

    Raises:
        RenderNotCurrentError: 파일 해시가 다를 때.
    """
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RenderNotCurrentError(
            f"{session_id}/{stream_id}: 받은 블러본이 확인한 렌더 기록과 다릅니다. 다시 시도하세요"
        )


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

    Args:
        conn: DB 연결 (읽기만 한다).
        raw: 원본 버킷 저장소 (감사 저장소 → 원본 읽기 기록).
        labeling: 라벨링 버킷 저장소 (raw와 다른 버킷이어야 한다).
        policy: 프라이버시 정책.

    Returns:
        영상 스트림 ID → 블러본 URI (`s3://dlp-labeling/sessions/<세션>/blurred/<스트림>.mp4`).

    Raises:
        PrivacyGateError: 오디오를 남기는 설정, 같은 버킷, 세션이 승인 상태가 아님.

    부작용: 라벨링 버킷에 블러본과 렌더 기록을 (덮어)쓴다. 블러본을 먼저 쓰고 기록을 나중에 써서,
    중간에 실패하면 기록 해시가 맞지 않아(또는 파일 해시가 달라) 소비자가 쓰지 못한다.
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
            # 기록 해시가 같으면 다시 렌더하지 않는다 (무효 기록·기록 없음은 None이라 다시 렌더)
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
                blurred_sha = sha256_file(dst)
                labeling.put_file(key, dst, blurred_sha)
                write_render_meta(
                    labeling, session_id, stream.stream_id, RenderMeta(digest, blurred_sha), work
                )
            out[stream.stream_id] = labeling.uri(key)
    return out
