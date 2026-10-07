"""세션 프리라벨 실행.

1. 영상 스트림마다 Predictor를 돌린다 (원본 영상: 모델 추론은 접근 통제된 서버에서 원본으로 한다).
   같은 Predictor·모델 버전 결과가 그 스트림에 이미 있으면 건너뛴다.
2. 깊이 모델이 있으면 바디캠 손 관절·객체 박스를 카메라 좌표 3D 궤적으로 올린다 (lift3d).
3. 바디캠의 손 키포인트·객체 박스와 장갑 신호로 접촉 구간을 만들어 hand_state 라벨로 쓴다. 배포된
   재학습 접촉 모델이 있으면(replaced에 CONTACT_STEP) 이 단계는 돌지 않고, 이 단계가 냈던 검수 전
   접촉을 지운다. 2·3의 모델 버전에는 정책 해시와 입력(현재 입력 라벨 ID, 장갑 동기화, 장갑 압력
   채널 접두사 sync.yaml glove.pressure_prefixes) 해시를 넣는다. 입력이 바뀌면 (예측기 버전 변경,
   검수자 수정) 다시 돌고, 검수 전인 이전 결과만 지운다. 검수된(승인·표본 검증) 결과와 사람이
   고치거나 만든 결과는 남으므로, 새 결과 중 그와 겹치는 것은 버린다: 접촉은 같은 손에서 시간이
   겹치는 구간, 3D 궤적은 같은 (개체, 부위, 좌표계) (ADR 0015, 검수 결과 옆에 같은 대상의 모델
   출력이 중복으로 남지 않게).
4. 3인칭 영상이 있으면 바디캠 IMU와 3인칭 인물 손목 속도를 상관시켜 착용자를 찾는다. 찾은 인물의
   키포인트 트랙은 entity_id="wearer"인 새 레코드(parent=원래 트랙)로 남긴다. 이 레코드는 모델
   출력(검수 전, 출처 MODEL)이다. 원래 트랙이 사람 출처여도 출처를 물려받지 않는다. 전신 모델 버전이
   바뀌어 지워지면 새 트랙으로 다시 찾는다.
5. 프라이버시 승인 상태의 세션은 생애주기를 prelabeled로 옮긴다.

진입점: `dlp prelabel run <세션>` (`dlp_cli.prelabel_cmds`)이 어댑터·깊이 단계를 만들고, 원본
저장소를 감사 저장소(`dlp_cli.raw_access.raw_store`, ADR 0020)로 열어 `run_prelabel`을 한
트랜잭션에서 부른다.

부작용 요약: DB `labels`에 새 레코드(모델 라벨, 삭제 레코드)만 추가한다 (덮어쓰지 않음). 생애주기
기록 (`set_lifecycle`). 원본 버킷에서 영상·장갑·IMU 파일을 임시 디렉터리로 받는다 (접근 기록은
넘겨받은 저장소가 남긴다). 관련: WP8, ADR 0015, 0019, 0026.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlalchemy as sa

from dlp_media.storage import ObjectStore
from dlp_prelabel.common import input_digest, model_label
from dlp_prelabel.contact import (
    ContactInterval,
    fuse_contacts,
    glove_contact_intervals,
    video_contact_intervals,
)
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.wearer import match_wearer, wrist_speed
from dlp_schema.config import repo_root
from dlp_schema.db.repository import get_labels, get_session, insert_labels, set_lifecycle
from dlp_schema.episode import current_labels, retractions, version_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    Evidence,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    Trajectory3DPayload,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.session import LifecycleState, Session, StreamKind, SyncMethod
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import glove_series, imu_series

# Predictor를 돌리는 영상 스트림 종류
VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 배포 모델이 기본 어댑터를 대신해 지울 때 삭제 레코드에 남기는 model_version
REPLACED_VERSION = "replaced-by-deployed-model"
# 접촉(CONTACT_PREFIX)·착용자(WEARER_PREFIX) 단계 버전 접두사. 알고리즘을 바꾸면 숫자를 올려
# 다시 돌게 한다 (이력에 같은 버전이 없어진다)
CONTACT_PREFIX = "contact-heuristic-1"
WEARER_PREFIX = "wearer-xcorr-1"
# replaced에 이 이름이 있으면 배포된 재학습 접촉 모델이 기본 접촉 단계(장갑·영상 휴리스틱)를
# 대신한다 (training.yaml contact.replaces)
CONTACT_STEP = "contact"


def contact_version(policy: PrelabelPolicy, inputs: str) -> str:
    """접촉 단계 모델 버전: `contact-heuristic-1+p<contact 절 해시>+i<입력 해시>`.

    inputs: 접촉 단계 입력 해시 (손 키포인트·객체 박스 라벨 ID, 장갑 스트림 동기화).
    """
    return f"{CONTACT_PREFIX}+p{policy.digest('contact')}+i{inputs}"


def default_pressure_prefixes() -> tuple[str, ...]:
    """저장소 sync.yaml glove.pressure_prefixes (장갑 압력 채널 접두사).

    `run_prelabel`에 접두사를 넘기지 않았을 때만 쓴다 (이 파일 위치에서 저장소 루트를 찾는다).
    CLI는 실행 설정의 sync 정책에서 읽어 넘긴다 (ADR 0026).
    """
    root = repo_root(Path(__file__).parent)
    return load_sync_policy(root / "config" / "policies" / "sync.yaml").glove.pressure_prefixes


def wearer_version(policy: PrelabelPolicy) -> str:
    """착용자 매칭 모델 버전: `wearer-xcorr-1+p<wearer_matching 절 해시>`."""
    return f"{WEARER_PREFIX}+p{policy.digest('wearer_matching')}"


@dataclass
class PrelabelSummary:
    """`run_prelabel` 결과 요약 (CLI 출력용).

    produced: "스트림/예측기" → 이번에 넣은 라벨 수. skipped: 같은 버전이 이미 있어 건너뛴
    "스트림/예측기".
    contacts: 새 접촉 라벨 수. retracted: 지운 이전 버전 라벨 수. lifted: 새 3D 궤적 수.
    wearer: 착용자로 찾은 원래 트랙 라벨 ID. wearer_scores: 인물(라벨 ID)별 상관.
    """

    produced: dict[str, int] = field(default_factory=dict[str, int])  # "스트림/predictor" → 라벨 수
    skipped: list[str] = field(default_factory=list[str])
    contacts: int = 0
    retracted: int = 0  # 새 버전으로 바뀌며 지운 이전 버전 라벨
    lifted: int = 0
    wearer: str | None = None
    wearer_scores: dict[str, float] = field(default_factory=dict[str, float])


def _fetch(store: ObjectStore, uri: str, work: Path) -> Path:
    """원본 저장소의 URI를 임시 작업 디렉터리로 받는다 (이미 받았으면 다시 받지 않는다).

    키의 `/`를 `__`로 바꾼 파일 이름을 쓴다. 저장소 접근 기록은 `raw`(감사 저장소)가 남긴다.
    """
    key = uri.removeprefix(store.uri(""))
    dest = work / key.replace("/", "__")
    if not dest.exists():
        store.get_file(key, dest)
    return dest


def run_prelabel(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    predictors: list[Predictor],
    policy: PrelabelPolicy,
    ontology: Ontology,
    now: datetime,
    lifter: DepthLifter | None = None,
    replaced: Iterable[str] = (),
    pressure_prefixes: Sequence[str] | None = None,
) -> PrelabelSummary:
    """replaced: 배포된 재학습 모델이 대신하는 기본 어댑터 이름 (CONTACT_STEP이면 접촉 단계).

    그 어댑터가 냈던 검수 전 라벨을 지운다 (같은 대상이 기본 어댑터와 재학습 모델 양쪽으로 겹쳐 남지
    않게).
    pressure_prefixes: 장갑 압력 채널 접두사 (sync.yaml glove.pressure_prefixes). 없으면 저장소
    sync.yaml에서 읽는다. 접촉 단계 모델 버전에 들어가 바뀌면 접촉을 다시 만든다.

    Args:
        conn: DB 연결. 호출자가 트랜잭션을 연다 (`engine.begin()`); 중간에 실패하면 모두 되돌린다.
        session_id: 세션 ID.
        raw: 원본 버킷 저장소. 반드시 감사 저장소(`raw_store`)여야 한다 (ADR 0020).
        predictors: 돌릴 어댑터 (실제 모델, stub, 배포된 재학습 모델 `trained-<과제>`).
        policy: 프리라벨 정책. ontology: 접촉 대상 종류(tool/fixed_surface/object) 판단에 쓴다.
        now: 새 레코드의 created_at (시간대 필수).
        lifter: 깊이 3D 단계 (가중치가 없으면 None → 건너뛴다).

    Returns:
        `PrelabelSummary`.

    Raises:
        ValueError: 세션에 온톨로지 버전이 없을 때. 어댑터의 ModelUnavailableError 등은 그대로
            올린다.

    멱등: 같은 입력·버전으로 다시 부르면 아무것도 넣지 않는다 (판단은 현재 라벨이 아닌 전체
    이력).
    """
    prefixes = (
        tuple(pressure_prefixes) if pressure_prefixes is not None else default_pressure_prefixes()
    )
    # Iterable을 여러 번 보므로 집합으로 고정한다
    replaced = set(replaced)
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise ValueError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = PrelabelSummary()
    # 전체 이력 (삭제·수정된 것 포함). 다시 돌릴지는 현재 라벨이 아니라 이력으로 정한다 (ADR 0015)
    existing = get_labels(conn, session_id)
    # 원본 파일은 임시 디렉터리에만 받고 끝나면 지운다 (로컬에 남기지 않는다)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        reference = session.reference_stream.stream_id
        timeline = set(policy.timeline_predictors)
        for stream in (s for s in session.streams if s.kind in VIDEO):
            for predictor in predictors:
                key = f"{stream.stream_id}/{predictor.name}"
                on_timeline = predictor.name in timeline
                # 마스터 타임라인 구간을 내는 모델(재학습 접촉 모델 등)은 세션에 한 번, 기준 스트림
                # (바디캠)에서만 돌린다. 스트림마다 돌리면 같은 구간이 스트림 수만큼 겹쳐 남는다
                if on_timeline and stream.stream_id != reference:
                    continue
                # 멱등: 이 스트림(타임라인 모델이면 stream_id 없는 구간)에 같은 버전 라벨이 이력에
                # 있으면 건너뛴다.
                # 검수자가 그 결과를 모두 고치거나 지웠어도 되살리지 않는다
                if any(
                    (x.stream_id == stream.stream_id or (on_timeline and x.stream_id is None))
                    and x.provenance.model_version == predictor.version
                    for x in existing
                ):
                    summary.skipped.append(key)
                    continue
                labels = predictor.run(
                    Clip(session_id, stream.stream_id, _fetch(raw, stream.uri, work))
                )
                if stream.stream_id != reference:
                    # 목록에 없는 모델이 타임라인 구간(stream_id 없음)을 내도 기준 스트림 것만 쓴다
                    labels = [x for x in labels if x.stream_id is not None]
                # 같은 예측기의 이전 버전 라벨 중 아직 아무도 검수하지 않은 것은 지운다
                prefix = f"{session_id}-{stream.stream_id}-{predictor.name}-"
                stale = _stale(existing, prefix, predictor.version)
                insert_labels(conn, [*retractions(stale, predictor.version, now), *labels])
                summary.produced[key] = len(labels)
                summary.retracted += len(stale)
        # 배포 모델이 대신하는 기본 어댑터(접촉 제외)의 검수 전 라벨: 버전과 상관없이 모두 지운다
        replaced_stale = [
            x
            for stream in session.streams
            if stream.kind in VIDEO
            for name in sorted(replaced - {CONTACT_STEP})
            for x in _stale(existing, f"{session_id}-{stream.stream_id}-{name}-", "")
        ]
        if CONTACT_STEP in replaced:
            # 배포된 재학습 접촉 모델이 hand_state를 낸다. 기본 접촉 단계의 검수 전 결과는 지운다
            replaced_stale += _stale(existing, f"{session_id}-contact-", "")
        if replaced_stale:
            insert_labels(conn, retractions(replaced_stale, REPLACED_VERSION, now))
            summary.retracted += len(replaced_stale)
        # 1단계에서 넣은 라벨을 반영해 이력·현재 라벨을 다시 읽는다 (2~4단계의 입력)
        history = get_labels(conn, session_id)
        current = current_labels(history)
        if lifter is not None:
            summary.lifted = _lift(conn, session, history, current, raw, work, lifter, now)
        if CONTACT_STEP not in replaced:
            summary.contacts = _contacts(
                conn, session, history, current, raw, work, policy, ontology, now, prefixes
            )
        _wearer(conn, session, history, current, raw, work, policy, now, summary)
    if session.lifecycle_state is LifecycleState.PRIVACY_APPROVED:
        set_lifecycle(conn, session_id, LifecycleState.PRELABELED)
    return summary


def _stale(labels: list[LabelRecord], prefix: str, version: str) -> list[LabelRecord]:
    """같은 단계의 이전 버전 모델 라벨 중 현재 운영 라벨이고 아직 검수하지 않은 것.

    착용자 레코드(원래 트랙을 대체한 사본)는 원래 트랙이 검수된 것이면 지우지 않는다 (지우면 검수된
    원래 트랙도 현재 라벨에서 사라진다).

    Args:
        labels: 세션 전체 이력.
        prefix: 단계 라벨 ID 접두사 (예: `<세션>-<스트림>-hands-`, `<세션>-contact-`).
        version: 지금 버전. 이 버전 라벨은 지우지 않는다 (""이면 모든 버전이 대상).
    """
    reviewed = {
        x.label_id for x in labels if x.verification.state is not VerificationState.UNREVIEWED
    }
    return [
        x
        for x in current_labels(labels)
        if x.label_id.startswith(prefix)
        and x.provenance.source is Source.MODEL
        and x.provenance.model_version != version
        and x.verification.state is VerificationState.UNREVIEWED
        # 착용자 사본은 원래 트랙이 검수됐으면 남긴다 (사본을 지우면 parent 관계로 원래 트랙도
        # 현재에서 사라진다)
        and not (_is_wearer(x) and x.parent_label_id in reviewed)
    ]


def _is_wearer(x: LabelRecord) -> bool:
    """착용자 매칭 단계가 만든 사본 레코드인가 (모델 버전 접두사로 판단)."""
    return (x.provenance.model_version or "").startswith(WEARER_PREFIX)


def _protected(x: LabelRecord) -> bool:
    """검수자가 승인·표본 검증했거나 사람이 고치거나 만든 라벨. 어떤 단계도 지우지 않는다."""
    return (
        x.provenance.source is Source.HUMAN
        or x.verification.state is not VerificationState.UNREVIEWED
    )


def _trajectory_key(p: Trajectory3DPayload) -> tuple[str, str | None, str]:
    """3D 궤적의 동일 대상 키 (개체, 부위, 좌표계)."""
    return (p.entity_id, p.part, p.frame.value)


def drop_protected_trajectories(
    labels: list[LabelRecord], current: list[LabelRecord]
) -> list[LabelRecord]:
    """새 3D 궤적 중 같은 (개체, 부위, 좌표계)의 검수된·사람 궤적이 이미 있는 것을 버린다.

    ADR 0026: 검수된 결과는 지우지 않으므로, 같은 대상의 모델 궤적을 옆에 또 넣으면 중복이 된다.
    """
    kept = {
        _trajectory_key(x.payload)
        for x in current
        if isinstance(x.payload, Trajectory3DPayload) and _protected(x)
    }
    return [
        x
        for x in labels
        if not (isinstance(x.payload, Trajectory3DPayload) and _trajectory_key(x.payload) in kept)
    ]


def drop_protected_contacts(
    intervals: list[ContactInterval], hand: Hand, current: list[LabelRecord]
) -> list[tuple[int, ContactInterval]]:
    """새 접촉 구간 중 같은 손의 검수된·사람 손 상태 구간과 시간이 겹치는 것을 버린다.

    (원래 순번, 구간)을 돌려준다. 순번은 라벨 ID에 쓰므로 버려도 남은 구간의 ID가 바뀌지 않는다.
    겹침은 열린 구간 기준이다 (끝과 시작이 맞닿기만 하면 겹치지 않는다).
    """
    spans = [
        (x.t_start_ms, x.t_end_ms)
        for x in current
        if isinstance(x.payload, HandStatePayload) and x.payload.hand is hand and _protected(x)
    ]
    return [
        (i, c)
        for i, c in enumerate(intervals)
        # 열린 구간 겹침: 맞닿기만 하면 겹치지 않는다
        if not any(s < c.end_ms and c.start_ms < e for s, e in spans)
    ]


def _lift(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    lifter: DepthLifter,
    now: datetime,
) -> int:
    """멱등: 같은 버전을 낸 적이 있으면(검수자가 모두 고쳤어도) 다시 만들지 않는다.

    버전에는 입력 트랙(현재 라벨 ID) 해시가 들어가 입력이 바뀌면 다시 만든다.

    입력: 기준 스트림(바디캠)의 현재 키포인트·박스 트랙 (모델·사람 출처 모두). 출력 라벨 ID 접두사는
    `<세션>-<바디캠>-3d-`. Returns: 새로 넣은 3D 궤적 수.
    부작용: 원본 버킷에서 바디캠 영상을 받고, `labels`에 새 궤적과 이전 버전 삭제 레코드를
    넣는다.
    """
    body = session.reference_stream
    tracks = [
        x
        for x in current
        if x.stream_id == body.stream_id
        and isinstance(x.payload, KeypointTrackPayload | BoxTrackPayload)
    ]
    version = f"{lifter.version}+i{input_digest(tracks)}"
    if any(x.provenance.model_version == version for x in history):
        return 0
    stale = _stale(history, f"{session.session_id}-{body.stream_id}-3d-", version)
    if not tracks:
        # 입력이 모두 사라졌으면 검수 전인 이전 결과만 지운다
        if stale:
            insert_labels(conn, retractions(stale, version, now))
        return 0
    labels = lifter.run(
        _fetch(raw, body.uri, work),
        session_id=session.session_id,
        stream_id=body.stream_id,
        tracks=tracks,
        calib=session.calibration.intrinsics,
        ontology_version=session.ontology_version or "",
        version=version,
    )
    labels = drop_protected_trajectories(labels, current)
    insert_labels(conn, [*retractions(stale, version, now), *labels])
    return len(labels)


def _has_live_wearer(history: list[LabelRecord]) -> bool:
    """모델 단계가 지우지 않은 착용자 레코드가 있는가 (검수자가 고치거나 지운 것 포함)."""
    wearer = {x.label_id for x in history if _is_wearer(x) and not x.retracted}
    # 모델 단계(출처 MODEL)가 지운 착용자 레코드. 검수자(HUMAN)가 지운 것은 살아 있는 것으로 본다
    # (검수자 판단을 되살리지 않게)
    model_retracted = {
        x.parent_label_id
        for x in history
        if x.retracted and x.provenance.source is Source.MODEL and x.parent_label_id in wearer
    }
    return bool(wearer - model_retracted)


def _contact_kind(class_id: str | None, ontology: Ontology) -> str:
    """접촉 대상 클래스 → HandStatePayload.contact_target_kind.

    온톨로지 객체의 tool 정의가 있으면 "tool", surface면 "fixed_surface", 그 밖(모르는 클래스
    포함)은 "object".
    """
    obj = ontology.objects.get(class_id or "")
    if obj is None:
        return "object"
    if obj.tool is not None:
        return "tool"
    return "fixed_surface" if obj.surface else "object"


def _contacts(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    policy: PrelabelPolicy,
    ontology: Ontology,
    now: datetime,
    pressure_prefixes: tuple[str, ...],
) -> int:
    # 멱등: 이력에 이 버전이 있으면 (검수자가 모두 고쳤거나 지웠어도) 다시 만들지 않는다.
    # 정책(contact 절)이나 입력(손·객체 트랙, 장갑 동기화)이 바뀌면 버전이 바뀌어 다시 만들고,
    # 검수 전인 이전 버전 접촉만 지운다.
    """접촉 단계: 바디캠 손·박스 트랙과 동기화된 장갑으로 손마다 접촉 구간을 만들어 hand_state로
    쓴다.

    장갑이 있는 손은 `fuse_contacts`(장갑 시각 + 영상 대상), 없는 손은 영상 구간만 쓴다. 라벨은
    마스터 타임라인 구간(stream_id=None)이고 ID는 `<세션>-contact-<version_tag>-<손>-<순번
    4자리>`다. 신뢰도는 `contact.confidence.<출처>`, 증거는 영상만이면 INFERRED, 그 밖은 OBSERVED.

    Returns: 새로 넣은 접촉 라벨 수 (같은 버전이 이력에 있으면 0).
    부작용: 원본 버킷에서 장갑 Parquet을 받고, `labels`에 새 접촉과 이전 버전 삭제 레코드를
    넣는다.
    """
    body = session.reference_stream.stream_id
    hand_labels = [
        x
        for x in current
        if x.stream_id == body
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "hand21"
        and x.payload.hand is not None
    ]
    box_labels = [
        x for x in current if x.stream_id == body and isinstance(x.payload, BoxTrackPayload)
    ]
    gloves = {
        # 동기화된 장갑만 쓴다 (동기화 전 장갑 시각은 마스터와 어긋나 접촉 경계가 틀어진다)
        Hand.LEFT if s.kind is StreamKind.GLOVE_LEFT else Hand.RIGHT: s
        for s in session.streams
        if s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT)
        and s.sync_method is not SyncMethod.UNSYNCED
    }
    version = contact_version(
        policy,
        input_digest(
            [*hand_labels, *box_labels],
            *(g.model_dump_json() for _, g in sorted(gloves.items())),
            # 장갑 압력 채널 선택(sync.yaml glove.pressure_prefixes)도 접촉 결과를 바꾼다
            "pressure_prefixes=" + ",".join(pressure_prefixes),
        ),
    )
    if any(x.provenance.model_version == version for x in history):
        return 0
    stale = _stale(history, f"{session.session_id}-contact-", version)
    hands: dict[Hand, KeypointTrackPayload] = {}
    # 손마다 트랙 하나만 쓴다. 같은 손 트랙이 여럿이면 목록의 마지막 것이 남는다
    for x in hand_labels:
        assert isinstance(x.payload, KeypointTrackPayload) and x.payload.hand is not None
        hands[x.payload.hand] = x.payload
    objects = [x.payload for x in box_labels if isinstance(x.payload, BoxTrackPayload)]
    # entity_id → 클래스 (접촉 대상 종류 판단용)
    classes = {o.entity_id: o.class_id for o in objects}
    labels: list[LabelRecord] = []
    for hand in (Hand.LEFT, Hand.RIGHT):
        video = (
            video_contact_intervals(hands[hand], objects, policy.contact.video)
            if hand in hands
            else []
        )
        intervals: list[ContactInterval] = video
        if hand in gloves:
            g = gloves[hand]
            series = glove_series(_fetch(raw, g.uri, work), pressure_prefixes)
            # 장갑 시각(스트림 ms) → 마스터 ms (오프셋·수동 보정·clock_scale 드리프트 포함)
            master = np.array([g.to_master_ms(float(t)) for t in series.t_ms])
            intervals = fuse_contacts(
                glove_contact_intervals(master, series.values, policy.contact.glove), video
            )
        for i, c in drop_protected_contacts(intervals, hand, current):
            kind = _contact_kind(classes.get(c.target_id or ""), ontology)
            payload = HandStatePayload(
                hand=hand,
                contact_target_kind=kind,
                # 장갑만 잡은 접촉은 대상을 모른다 (관계·행동 단계가 이 ID를 버린다)
                target_id=c.target_id or policy.contact.unresolved_target_id,
                role="active",
            )
            labels.append(
                model_label(
                    label_id=f"{session.session_id}-contact-{version_tag(version)}-{hand.value}-{i:04d}",
                    session_id=session.session_id,
                    stream_id=None,
                    t_start_ms=c.start_ms,
                    t_end_ms=c.end_ms,
                    ontology_version=session.ontology_version or "",
                    model_version=version,
                    confidence=getattr(policy.contact.confidence, c.source),
                    payload=payload,
                    now=now,
                    evidence=Evidence.OBSERVED if c.source != "video" else Evidence.INFERRED,
                )
            )
    insert_labels(conn, [*retractions(stale, version, now), *labels])
    return len(labels)


def _wearer(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    policy: PrelabelPolicy,
    now: datetime,
    summary: PrelabelSummary,
) -> None:
    """착용자 매칭 단계 (결과를 summary에 적는다).

    조건: 3인칭 스트림이 있고 동기화됨, 공유 시계(SHARED_CLOCK) IMU 스트림이 있음, 살아 있는 착용자
    레코드가 없음, 3인칭 coco17 트랙이 있음. 하나라도 아니면 아무것도 하지 않는다. IMU 시각은 공유
    시계라 변환 없이 쓰고, 3인칭 손목 속도 시각은 `to_master_ms`로 마스터에 맞춘다.
    부작용: 원본 버킷에서 IMU Parquet을 받고, `labels`에 착용자 사본 하나를 넣는다.
    """
    third = next((s for s in session.streams if s.kind is StreamKind.THIRD_PERSON), None)
    imu = next(
        (
            s
            for s in session.streams
            if s.kind is StreamKind.IMU and s.sync_method is SyncMethod.SHARED_CLOCK
        ),
        None,
    )
    if third is None or imu is None or third.sync_method is SyncMethod.UNSYNCED:
        return
    # 착용자 레코드는 원래 트랙을 대체(parent)하므로 지우면 원래 트랙까지 사라진다.
    # 그래서 정책이 바뀌어도 이미 매칭한 세션은 다시 하지 않는다 (바꾸려면 검수자가 고친다).
    # 다만 전신 모델 버전이 바뀌어 착용자 레코드가 모델 단계에서 지워졌으면(원래 트랙도 함께 낡았다)
    # 새 트랙으로 다시 찾는다. 검수자가 지운 착용자 레코드는 되살리지 않는다.
    if _has_live_wearer(history):
        return
    people = {
        x.label_id: x
        for x in current
        if x.stream_id == third.stream_id
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "coco17"
    }
    if not people:
        return
    # IMU는 공유 시계(SHARED_CLOCK)라 시각을 그대로 기준 신호로 쓴다 (가속도 크기 - 중력 중앙값)
    ref = imu_series(_fetch(raw, imu.uri, work))
    speeds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label_id, label in people.items():
        assert isinstance(label.payload, KeypointTrackPayload)
        t, v = wrist_speed(label.payload)
        speeds[label_id] = (np.array([third.to_master_ms(float(x)) for x in t]), v)
    match = match_wearer(
        ref.t_ms,
        ref.values,
        speeds,
        rate_hz=policy.wearer_matching.rate_hz,
        min_correlation=policy.wearer_matching.min_correlation,
        min_overlap_samples=policy.wearer_matching.min_overlap_samples,
    )
    summary.wearer_scores = match.scores
    if match.entity_id is None:
        return
    original = people[match.entity_id]
    insert_labels(conn, [wearer_copy(original, match.correlation, wearer_version(policy), now)])
    summary.wearer = match.entity_id


def wearer_copy(
    original: LabelRecord, correlation: float, version: str, now: datetime
) -> LabelRecord:
    """착용자로 찾은 인물 트랙의 사본 (entity_id="wearer", parent=원래 트랙).

    모델 출력이다: 원래 트랙의 출처(사람이 고친 트랙이면 HUMAN)와 검수 상태를 물려받지 않는다.
    물려받으면 검수 전 모델 판단이 사람 라벨로 내보내지고, 모델 단계가 지울 수도 없다.

    ID는 `<원래 ID>:wearer`, 신뢰도는 상관(음수면 0), 증거는 INFERRED (ADR 0026).
    """
    assert isinstance(original.payload, KeypointTrackPayload)
    return original.model_copy(
        update={
            "label_id": f"{original.label_id}:wearer",
            "parent_label_id": original.label_id,
            "payload": original.payload.model_copy(update={"entity_id": "wearer"}),
            "provenance": Provenance(source=Source.MODEL, model_version=version),
            "confidence": round(max(correlation, 0.0), 4),
            "evidence": Evidence.INFERRED,
            "verification": Verification(),
            "created_at": now,
        }
    )
