"""DB에 등록된 세션의 프라이버시 게이트 실행: 탐지 → (사람 검수) → 승인 → 블러본 렌더.

WP5, ADR 0005·0019·0023·0024. CLI `dlp_cli.privacy_cmds`가 DB 연결과 저장소를 만들어 부른다.

단계와 진입점:
- `detect_session` (`dlp privacy detect`): 원본 영상 → blur_track 라벨(DB `label_records`) +
  검수 우선 구간·탐지 표시(원본 버킷 JSON). 세션 `privacy_state`가 PENDING이거나, APPROVED인데
  새 탐지를 실행하거나 블러가 바뀌었으면 AUTO_BLURRED로 (0건 재탐지도 포함).
- (사람 검수: `dlp review create --stage privacy` → CVAT → `dlp review collect`. dlp_review 담당)
- `approve_session` (`dlp privacy approve`): 승인 조건을 확인하고 APPROVED로.
- `render_session` (`dlp privacy render`): 승인된 세션의 블러본과 렌더 기록을 라벨링 버킷에 쓴다.
- `assert_render_current` / `check_fetched`: 블러본 소비자(검수 작업 생성, 행동 VLM, 큐레이션,
  내보내기)가 쓰기 전에 부르는 확인 (ADR 0024).
- `invalidate_render`: 블러가 바뀐 스트림의 렌더 기록을 무효로 만든다 (dlp_review의 블러 검수
  수집도 부른다).
- `stale_media_tasks` / `request_media_revocation` (`revocation`): 블러가 바뀐 스트림의 이미 배포한
  라벨링 작업 매체에 회수 요청을 남긴다 (ADR 0039). 탐지는 블러가 바뀐 스트림에서 같은 트랜잭션에
  부르고, `dlp privacy render`·`dlp review` 명령도 부른다.

버킷:
- raw(원본 버킷 `dlp-raw`): 원본 영상을 읽고 검수 우선 구간 JSON과 탐지 표시(결과 0개 버전,
  `detect_marker_key`)를 읽고 쓴다. CLI는 반드시
  `dlp_cli.raw_access.raw_store`로 만든 `AuditedStore`를 넘기므로 `get_file`/`put_file`마다 원본
  접근 기록(`raw_access_log`)이 남는다 (ADR 0020). `head`는 기록하지 않는다.
- labeling(라벨링 버킷 `dlp-labeling`): 블러본(`blurred_key`)과 렌더 기록(`render_meta_key`)만 쓴다.
  블러본은 원본 버킷에 쓰지 않는다 (`render_session`이 막는다).

트랜잭션: DB를 쓰는 함수는 호출자가 연 트랜잭션(`engine.begin()`) 안의 연결을 받는다. 저장소 쓰기는
트랜잭션 밖 부작용이라 DB가 되돌아가도 남는다 (다시 실행하면 같은 키에 덮어쓰거나 건너뛴다).

규칙 (CLAUDE.md, ADR 0015·0019):
- 라벨은 덮어쓰지 않는다. 버전이 바뀌면 검수 전(UNREVIEWED)인 이전 모델 블러만 `retractions`로
  지운다. 사람이 검수한 블러는 지우지 않는다.
- 멱등 판정은 최신 완료 세대의 활성 버전 집합으로 한다. 과거 버전 복귀는 새 탐지 세대다.
  DB 이력 없는 외부 0건 표시는 복구하고 새 검수를 요구한다 (ADR 0033).
- 운영 블러(`operational_blur`)만 승인·렌더에 쓴다. 오류 삽입·측정 레코드와 그 후손은 뺀다.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, sha256_file
from dlp_privacy.detection import FrameDetector
from dlp_privacy.errors import PrivacyGateError as PrivacyGateError
from dlp_privacy.errors import RenderNotCurrentError as RenderNotCurrentError
from dlp_privacy.generation import (
    collected_privacy_reviews as collected_privacy_reviews,
)
from dlp_privacy.generation import (
    last_detection as last_detection,
)
from dlp_privacy.generation import (
    missing_privacy_reviews as missing_privacy_reviews,
)
from dlp_privacy.generation import (
    operational_blur as operational_blur,
)
from dlp_privacy.pipeline import detect_video, model_version
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import (
    RenderMeta as RenderMeta,
)
from dlp_privacy.rendering import (
    assert_render_current as assert_render_current,
)
from dlp_privacy.rendering import (
    check_fetched as check_fetched,
)
from dlp_privacy.rendering import (
    check_render_meta as check_render_meta,
)
from dlp_privacy.rendering import (
    expected_render_hash as expected_render_hash,
)
from dlp_privacy.rendering import (
    fetch_video,
)
from dlp_privacy.rendering import (
    invalidate_render as invalidate_render,
)
from dlp_privacy.rendering import (
    read_render_meta as read_render_meta,
)
from dlp_privacy.rendering import (
    render_hash as render_hash,
)
from dlp_privacy.rendering import (
    render_meta_key as render_meta_key,
)
from dlp_privacy.rendering import (
    render_session as render_session,
)
from dlp_privacy.rendering import (
    write_render_meta as write_render_meta,
)
from dlp_privacy.review import ReviewSegment
from dlp_privacy.revocation import (
    StaleMedia as StaleMedia,
)
from dlp_privacy.revocation import (
    request_media_revocation as request_media_revocation,
)
from dlp_privacy.revocation import (
    stale_media_tasks as stale_media_tasks,
)
from dlp_schema.common import derived_id
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    list_privacy_detections,
    lock_session,
    record_privacy_detection,
    set_lifecycle,
    set_privacy_state,
)
from dlp_schema.episode import current_labels, retractions
from dlp_schema.labels import BlurTrackPayload, LabelRecord, Source, VerificationState
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.session import LifecycleState, PrivacyState, Session, StreamKind

# 블러 대상인 영상 스트림 종류 (IMU·장갑·오디오 스트림은 게이트를 거치지 않는다)
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}


@dataclass
class DetectSummary:
    """`detect_session` 결과 요약 (CLI 출력용)."""

    # 스트림 → 이번에 새로 쓴 블러 트랙 수 (삭제 레코드는 세지 않는다. 0이면 결과 0개)
    detected: dict[str, int] = field(default_factory=dict[str, int])
    # 최신 완료 세대의 활성 버전 집합이 같아 건너뛴 스트림
    skipped: list[str] = field(default_factory=list[str])
    missing: dict[str, str] = field(default_factory=dict[str, str])  # 대상 → 이유
    # 이번에 쓴 검수 우선 구간 객체 키 (원본 버킷)
    review_keys: list[str] = field(default_factory=list[str])
    changed: bool = False  # 블러 라벨 집합이 바뀌었는가 (새 라벨 또는 이전 버전 삭제)
    reran: bool = False  # 탐지를 새로 완료했거나 구버전 0건 표시를 DB 기록으로 복구했는가
    # 스트림 → 이번에 쓴 블러 레코드 수 (삭제 레코드 포함). 0이면 블러 집합이 그대로다
    changes: dict[str, int] = field(default_factory=dict[str, int])
    # 이번에 회수를 요청한 라벨링 작업 키 (블러가 바뀐 스트림의 낡은 매체, ADR 0039)
    revocations: list[str] = field(default_factory=list[str])

    @property
    def changed_streams(self) -> list[str]:
        """이번에 블러 레코드를 쓴 스트림 (탐지 순서). 이 스트림의 배포 매체만 회수를 요청한다."""
        return [s for s, n in self.changes.items() if n]


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


def detect_marker_key(session_id: str, stream_id: str) -> str:
    """원본 버킷의 탐지 실행 표시 (결과가 0개였던 탐지·모델 버전 목록).

    블러 레코드가 하나도 나오지 않은 스트림은 DB 이력에 모델 버전이 남지 않아, 이 표시가 없으면
    `dlp privacy detect`를 돌릴 때마다 영상 전체를 다시 탐지했다 (CPU OWLv2는 프레임당 수 초).
    구버전과의 호환을 위해 검수 우선 구간 옆에 계속 둔다. 현재는 DB 완료 이력이 우선이며,
    표시만 남았다면 확인 시각으로 완료 기록을 복구하고 새 검수를 요구한다 (ADR 0032).
    원본 영상에서 나온 판단이라 라벨링 버킷이 아니라 원본 버킷에 둔다.
    """
    return f"sessions/{session_id}/derived/privacy_detect/{stream_id}.json"


def read_detect_marker(raw: ObjectStore, session_id: str, stream_id: str, work: Path) -> set[str]:
    """결과 0개로 끝난 적이 있는 탐지·모델 버전. 표시가 없으면 빈 집합.

    부작용: 원본 버킷 읽기 (감사 저장소면 read 기록).
    """
    key = detect_marker_key(session_id, stream_id)
    if raw.head(key) is None:
        return set()
    dest = work / f"{stream_id}-detect.in.json"
    raw.get_file(key, dest)
    data: dict[str, list[str]] = json.loads(dest.read_text(encoding="utf-8"))
    return set(data.get("empty_versions", []))


def write_detect_marker(
    raw: ObjectStore, session_id: str, stream_id: str, versions: set[str], work: Path
) -> None:
    """결과 0개로 끝난 버전 목록을 덮어쓴다 (파생물이라 불변 업로드가 아니다).

    부작용: 원본 버킷 쓰기 (감사 저장소면 write 기록).
    """
    out = work / f"{stream_id}-detect.json"
    out.write_text(json.dumps({"empty_versions": sorted(versions)}), encoding="utf-8")
    raw.put_file(detect_marker_key(session_id, stream_id), out, sha256_file(out))


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
    """영상 스트림마다 블러 프리라벨을 만든다. 최신 세대의 활성 버전 집합이 같으면 건너뛴다.

    extra: 배포된 재학습 블러 모델 (blur_track을 내는 Predictor). 탐지기 결과와 합집합으로 남는다.
    labeling: 블러 변경·새 탐지가 있는 스트림의 렌더 기록을 무효로 만들 라벨링 버킷. 없어도
    블러본을 쓰는 쪽이 assert_render_current로 막지만, 있으면 이전 블러본을 바로 무효로 둔다.

    스트림마다 (ADR 0030·0032):
    1. 최신 완료 세대의 활성 버전 집합이 같으면 건너뛴다 (멱등, 탐지 표시를 읽지 않는다).
       아니면 원본 버킷의 구버전 0건 표시(`detect_marker_key`)를 읽어 복구할 버전을 고른다.
    2. 현재 활성 세대·복구 표시에 없는 버전(`need_run`)만 원본 영상을 받아(감사 기록)
       돌린다 (탐지기 파이프라인, 재학습 모델). 이번에 돌아 결과가 0개인 버전은 탐지 표시에 더한다.
       구버전 표시로 복구할 버전은 영상을 다시 읽지 않고 완료 이력을 새로 남긴다.
    3. 지금 쓰지 않는 버전의 검수 전 모델 블러를 삭제 레코드로 지우고 새 라벨과 함께 쓴다.
    4. 쓴 레코드나 새 완료 기록이 있으면 렌더 기록을 무효로 한다. 모델이 다시 돌았으면 검수 우선
       구간 파일을 병합해 다시 쓴다.
    완료 버전·시각은 결과 수와 무관하게 DB에 남긴다 (ADR 0032). 외부 0건 표시만 남은 구버전은
    확인 시각으로 새 기록을 남겨 전수 검수를 다시 요구한다. DB 롤백 뒤 남은 표시도 같은 방식이다.
    PENDING이거나 승인 뒤 새 탐지·블러 변경이 있으면 AUTO_BLURRED로 둔다. 0건 재탐지도
    새 검수 작업이 필요하며, 멱등으로 건너뛴 실행은 탐지 시각·승인을 바꾸지 않는다.

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

    부작용: DB `label_records`·`privacy_detection_runs` 추가, `sessions.privacy_state` 변경,
    블러가 바뀐 스트림의 낡은 라벨링 작업에 `review_tasks` 회수 요청(외부 삭제는 하지 않는다,
    ADR 0039), 원본 버킷 읽기·쓰기(원본 영상
    읽기, 검수 우선 구간·탐지 표시 읽기·덮어쓰기), 라벨링 버킷 렌더 기록 덮어쓰기(labeling이
    있을 때). 저장소 쓰기는 DB 트랜잭션이 되돌아가도 남는다. 그래서 탐지 표시에는 결과 0개
    버전만 넣는다.
    """
    lock_session(conn, session_id)
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise PrivacyGateError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = DetectSummary()
    # 전체 이력은 불변 ID 중복·구버전 복구를 확인하는 데 쓴다
    existing = get_labels(conn, session_id, kinds=["blur_track"])
    runs = list_privacy_detections(conn, session_id)
    version = model_version(detectors, policy)
    versions = {version, *(p.version for p in extra)}  # 이번에 있어야 할 모든 모델 버전
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            stream_runs = [r for r in runs if r.stream_id == stream.stream_id]
            latest = stream_runs[-1] if stream_runs else None
            # 완료 이력은 과거 수행 증거다. 건너뛰기는 현재 활성 버전 집합으로만 정한다.
            if latest is not None and set(latest.model_versions) == versions:
                summary.skipped.append(stream.stream_id)
                continue
            done = (
                set(latest.model_versions)
                if latest
                else {
                    x.provenance.model_version
                    for x in current_labels(existing)
                    if x.stream_id == stream.stream_id
                    and x.provenance.source is Source.MODEL
                    and x.provenance.model_version is not None
                }
            )
            historical = {v for r in stream_runs for v in r.model_versions} | {
                x.provenance.model_version
                for x in existing
                if x.stream_id == stream.stream_id and x.provenance.model_version is not None
            }
            # 탐지기·모델 버전이 바뀌면 검수 전인 이전 버전 블러만 지운다 (검수한 블러는 남긴다)
            stale = [
                x
                for x in current_labels(existing)
                if x.stream_id == stream.stream_id
                and x.provenance.source is Source.MODEL
                and x.provenance.model_version not in versions
                and x.verification.state is VerificationState.UNREVIEWED
            ]
            # 현재 세대에 없는 과거 버전은 복귀 때 다시 추론한다. 외부 표시는 아직 DB에
            # 완료한 적 없는 버전(구 버전·롤백 복구)에만 쓴다.
            empty: set[str] = (
                read_detect_marker(raw, session_id, stream.stream_id, work)
                if versions - done
                else set[str]()
            ) - historical
            # 다시 돌릴 버전이 있을 때만 원본 영상을 받는다 (stale 삭제만 할 때는 받지 않는다)
            need_run = versions - done - empty
            video = fetch_video(raw, stream.uri, work) if need_run else None
            labels: list[LabelRecord] = []
            segments: list[ReviewSegment] = []
            ran_empty: set[str] = set()  # 이번에 돌았지만 블러를 하나도 내지 않은 버전
            if version in need_run:
                assert video is not None
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
                if not result.labels:
                    ran_empty.add(version)
            for p in extra:
                if p.version in need_run:
                    assert video is not None
                    out = [
                        x
                        for x in p.run(Clip(session_id, stream.stream_id, video))
                        if isinstance(x.payload, BlurTrackPayload)
                    ]
                    labels += out
                    if not out:
                        ran_empty.add(p.version)
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
            # 실제 추론을 마친 시각과 활성 버전 전체를 남긴다. 시작 시각(now)은 라벨 생성용이다.
            event_id = record_privacy_detection(
                conn, session_id, stream.stream_id, sorted(versions), datetime.now(UTC)
            )
            seen = {x.label_id for x in existing}
            # 정책 복귀는 같은 모델의 결정적 ID를 다시 낸다. 불변 원본을 보존하고 새 세대 ID를 쓴다.
            labels = [
                x.model_copy(update={"label_id": derived_id(x.label_id, f"g{event_id}")})
                if x.label_id in seen
                else x
                for x in labels
            ]
            changes = [*retractions(stale, version, now), *labels]
            insert_labels(conn, changes)
            summary.reran = True
            summary.changed |= bool(changes)
            summary.changes[stream.stream_id] = len(changes)
            summary.detected[stream.stream_id] = len(labels)
            if labeling is not None:
                invalidate_render(labeling, session_id, stream.stream_id, "blur_changed", work)
            if ran_empty:
                # 결과 0개 버전만 표시에 남긴다. 블러를 낸 버전은 DB 이력이 증거이고, 표시에 넣으면
                # DB 트랜잭션이 되돌아갔을 때 라벨 없이 "끝났다"고 남아 영영 다시 돌지 않는다.
                write_detect_marker(raw, session_id, stream.stream_id, empty | ran_empty, work)
            rerun_models = {p.version for p in extra if p.version in need_run}
            if segments or version in need_run or rerun_models or done - versions:
                # 재학습 모델만 다시 돌았으면 탐지기 구간을 지우지 않는다 (합친다)
                merged = merge_segments(
                    read_review_segments(raw, session_id, stream.stream_id, work),
                    segments,
                    detector_rerun=version in need_run,
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
        (summary.changed or summary.reran) and session.privacy_state is PrivacyState.APPROVED
    ):
        # 0건 재탐지도 놓친 대상이 있을 수 있다. 새 실행 뒤 영상을 검수하고 다시 승인해야 한다
        set_privacy_state(conn, session_id, PrivacyState.AUTO_BLURRED)
    # 블러가 바뀐 스트림의 이미 배포한 라벨링 매체는 같은 커밋에서 회수를 요청한다 (ADR 0039)
    summary.revocations = _request_for_changes(conn, session_id, summary, policy, now)
    return summary


def _request_for_changes(
    conn: sa.Connection,
    session_id: str,
    summary: DetectSummary,
    policy: PrivacyPolicy,
    now: datetime,
) -> list[str]:
    """탐지로 블러 레코드가 생긴 스트림의 낡은 라벨링 작업에 회수를 요청한다. 요청한 작업 키.

    블러 집합이 그대로인 스트림(0건 재탐지 포함)은 렌더 해시가 같아 요청하지 않는다. 바뀐 스트림은
    렌더 해시가 없는 구 작업도 낡은 것으로 본다 (legacy).
    """
    if not summary.changed_streams:
        return []
    requested = request_media_revocation(
        conn, session_id, policy, now, reason="privacy_detect",
        streams=set(summary.changed_streams), legacy=True,
    )  # fmt: skip
    return [t.task_key for t in requested]


def approve_session(conn: sa.Connection, session_id: str) -> Session:
    """현재 운영 블러가 모두 사람 검수를 거쳤고, 영상 스트림마다 최신 완료 탐지 세대의 운영 검수를
    수집했을 때만 프라이버시 승인한다. 완료 이력 이전 데이터는 시각 규약을 사용한다.

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
    lock_session(conn, session_id)
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise PrivacyGateError(f"{session_id}: 자동 탐지가 아직 실행되지 않았습니다")
    missing = missing_privacy_reviews(conn, session)
    if missing:
        raise PrivacyGateError(
            f"{session_id}: 마지막 자동 탐지 뒤 수집한 블러 검수 작업이 없는 스트림 {missing} "
            "(dlp review create --stage privacy → 검수 → collect)"
        )
    history = get_labels(conn, session_id, kinds=["blur_track"])
    unreviewed = [
        x.label_id
        for s in session.streams
        if s.kind in VIDEO_KINDS
        for x in operational_blur(history, s.stream_id)
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
