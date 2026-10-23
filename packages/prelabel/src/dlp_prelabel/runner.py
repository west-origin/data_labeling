"""세션 프리라벨 실행.

1. 영상 스트림마다 Predictor를 돌린다 (원본 영상: 모델 추론은 접근 통제된 서버에서 원본으로 한다).
   같은 Predictor·모델 버전 결과가 그 스트림에 이미 있으면 건너뛴다.
2. 깊이 모델이 있으면 바디캠 손 관절·객체 박스를 카메라 좌표 3D 궤적으로 올린다 (lift3d).
3. 바디캠의 손 키포인트·객체 박스와 장갑 신호로 접촉 구간을 만들어 세션 [0, 길이]로 자른 뒤
   hand_state 라벨로 쓴다. 대상이 사람 박스(온톨로지 물리 유형 person)인 접촉은 부위를 알 수 없어
   라벨을 만들지 않고 요약에 센다 (검수자가 그린다). 배포된 재학습 접촉 모델이 있으면(replaced에
   CONTACT_STEP) 이 단계는 돌지 않고, 이 단계가 냈던 검수 전 접촉을 지운다. 2·3의 모델 버전에는
   정책 절 해시(depth·contact)와 입력 해시를 넣는다. 입력은 2가 (개체, 트랙 종류)마다 하나씩 고른
   바디캠 hand21 손·박스 트랙의 라벨 ID(`pick_lift_tracks`), 3이 바디캠 손·박스 트랙의 현재 라벨
   ID와 장갑 동기화, 장갑 압력 채널 접두사(sync.yaml glove.pressure_prefixes)다. 입력이 바뀌면
   (예측기 버전 변경, 검수자 수정) 다시 돌고, 검수 전인 이전 결과만 지운다. 검수된(승인·표본 검증)
   결과와 사람이 고치거나 만든 결과는 남으므로, 새 결과 중 그와 겹치는 것은 버린다: 접촉은 같은
   손에서 시간이 겹치는 구간, 3D 궤적은 같은 (개체, 부위, 좌표계) (ADR 0015, 검수 결과 옆에 같은
   대상의 모델 출력이 중복으로 남지 않게).
4. 3인칭 영상이 있으면 바디캠 IMU와 3인칭 인물 손목 속도를 상관시켜 착용자를 찾는다. 찾은 인물의
   키포인트 트랙은 entity_id="wearer"인 새 레코드(parent=원래 트랙)로 남긴다. 이 레코드는 모델
   출력(검수 전, 출처 MODEL)이다. 원래 트랙이 사람 출처여도 출처를 물려받지 않는다. 전신 모델 버전이
   바뀌어 지워지면 새 트랙으로 다시 찾는다.
5. 바디캠 자세와 월드 좌표 3D (ADR 0046, `camera_pose.enabled`): 동기화된 IMU 스트림과 카메라-IMU
   외부 파라미터가 있고 깊이 모델(또는 넘겨받은 자세 추정기)이 있으면 바디캠 6자유도 자세를 추정해
   trajectory3d(entity_id="camera", frame=world)로 남기고(`_PoseStage.camera_pose`), 현재 자세
   하나와 바디캠의 현재 카메라 좌표 궤적((개체, 부위)마다 하나)으로 월드 좌표 궤적을 만든다
   (`_PoseStage.world`). 두 단계 모두 버전에 정책 절 해시와 입력 해시를 넣고, 입력이 바뀌면 검수
   전인 이전 결과만 지운다.
   3D 궤적(손 관절·박스 중심)은 그것을 계산한 2D 트랙의 검수를 물려받는다 (ADR 0048·0050):
   3D·월드 좌표 단계 뒤 `apply_trajectory_follow`(경로 prelabel)가 이미 확인된 2D에서 만든 새 3D에
   검수 상태와 물려받기 기록을 쓰고, 출처 2D가 고쳐져 끊긴 물려받기만 한 3D를 지운다. 같은 2D에서
   3D 단계 입력(캘리브레이션·작업자 손 크기·depth 정책·추정기 판·자세 입력)이 바뀌어 새로 계산한
   3D는 물려받기만 한 이전 3D를 대신한다 (ADR 0051: 보호 판정이 막지 않고, 따르기가 이전 것을 지우고
   새 것이 같은 2D 검수를 물려받는다). 보호 판정·월드 좌표 입력은 지울 3D(`lapsed_trajectories`,
   새로 계산한 궤적을 이력에 더해 구한다)를 이중 방어로 계속 뺀다. 직접 검수·사람 출처 3D는 그대로
   새 3D를 막는다.
   단계마다 "다시 돌릴지"는 그 단계가 마지막으로 낸 버전으로 정한다 (`stage_version`, F-PAR-54):
   입력이 A → B → A로 되돌아오면 A를 다시 만든다 (이전 A의 라벨 ID와 겹치지 않게 버전 끝에
   재실행 세대 `+r<n>`). 막혀 쓸 것이 없으면 원본을 받지 않고 건너뛴다 (`summary.lift_blocked`).
6. 프라이버시 승인 상태의 세션은 생애주기를 prelabeled로 옮긴다.

진입점: `dlp prelabel run <세션>` (`dlp_cli.prelabel_cmds`)이 어댑터·깊이 단계를 만들고, 원본
저장소를 감사 저장소(`dlp_cli.raw_access.raw_store`, ADR 0020)로 열어 `run_prelabel`을 한
트랜잭션에서 부른다.

부작용 요약: DB `label_records`에 새 레코드(모델 라벨, 삭제 레코드)만 추가한다 (덮어쓰지 않음).
생애주기 기록 (`set_lifecycle`). 원본 버킷에서 영상·장갑·IMU 파일을 임시 디렉터리로 받는다.
접근 기록은 넘겨받은 저장소가 남긴다. 관련: WP8, ADR 0015, 0019, 0026.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlalchemy as sa
from numpy.typing import NDArray

from dlp_media.imu import read_imu_table
from dlp_media.storage import ObjectStore
from dlp_prelabel.campose import (
    CameraPoseEstimator,
    ImuSeries,
    ImuVisualOdometry,
    ModelDepth,
    PoseTrack,
    is_pose_label,
    pose_label,
    pose_samples,
)
from dlp_prelabel.common import input_digest, iter_frames, model_label
from dlp_prelabel.contact import (
    ContactInterval,
    clip_to_session,
    fuse_contacts,
    glove_contact_intervals,
    video_contact_intervals,
)
from dlp_prelabel.lift3d import HAND21_NAMES, DepthLifter, lift_key
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.posemath import as_rotation, rotation_problem
from dlp_prelabel.wearer import match_wearer, wrist_speed
from dlp_prelabel.worldframe import world_label
from dlp_schema.config import repo_root
from dlp_schema.db.calibration_repository import session_intrinsics, session_palm
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    get_trajectory_inheritance,
    insert_labels,
    set_lifecycle,
)
from dlp_schema.db.trajectory_repository import apply_trajectory_follow
from dlp_schema.description_follow import human_confirmed
from dlp_schema.episode import (
    current_labels,
    is_protected,
    pick_hand_track_labels,
    retractions,
    track_rank,
    version_tag,
)
from dlp_schema.labels import (
    BoxTrackPayload,
    CoordinateFrame,
    Evidence,
    Hand,
    HandStatePayload,
    IntrinsicsSource,
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
from dlp_schema.review import TrajectoryInheritance, TrajectoryInheritanceMechanism
from dlp_schema.session import LifecycleState, Session, Stream, StreamKind, SyncMethod
from dlp_schema.trajectory_follow import (
    TRAJECTORY_FOLLOW_VERSION,
    active_trajectory_inheritance,
    lapsed_trajectories,
)
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import glove_series, imu_series

# Predictor를 돌리는 영상 스트림 종류
VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 배포 모델이 기본 어댑터를 대신해 지울 때 삭제 레코드에 남기는 model_version
REPLACED_VERSION = "replaced-by-deployed-model"
# 접촉(CONTACT_PREFIX)·착용자(WEARER_PREFIX) 단계 버전 접두사. 알고리즘을 바꾸면 숫자를 올려
# 다시 돌게 한다 (이력에 같은 버전이 없어진다).
# 세션 [0, 길이] 자르기(F-PAR-02, `clip_to_session`, 잘린 조각의 최소 길이 재적용 포함)는 숫자를
# 올리지 않았다: 0 앞 구간이 생기던 세션은 예전에 트랜잭션이 롤백되어 이력이 없고, 세션 끝 뒤
# 구간이 남은 세션은 입력이 바뀌어 다시 돌 때 잘린다. 올리면 모든 세션의 검수 전 접촉이 새 ID로
# 다시 만들어지고, 평가 기준 버전 패턴(`contact-heuristic-1*`, ADR 0013·0025, training.yaml)도
# 함께 바꿔야 한다.
# 사람 박스 접촉을 라벨로 만들지 않게 한 수정(F-PAR-03)도 같은 이유로 올리지 않았다: 이미 사람
# 박스 접촉을 object로 기록한 세션은 입력(손·박스 트랙 등)이 바뀌어 다시 돌 때 검수 전 것이
# 지워진다.
CONTACT_PREFIX = "contact-heuristic-1"
WEARER_PREFIX = "wearer-xcorr-1"
# 카메라 자세(POSE_PREFIX-<추정기 판>)·월드 좌표(WORLD_PREFIX) 단계 버전 접두사 (ADR 0046)
POSE_PREFIX = "campose"
WORLD_PREFIX = "world3d-1"
# 자세 입력이 더는 성립하지 않아(IMU 동기화 해제·외부 파라미터 없음·입력 변경) 이전 자세를 지울 때
# 삭제 레코드에 남기는 model_version (어떤 자세 버전과도 같지 않아 다음 실행이 새로 만든다,
# F-PAR-56)
POSE_INVALIDATED_VERSION = f"{POSE_PREFIX}-invalidated"
# 단계 버전 끝의 재실행 세대 표시 (`+r<n>`): 이전에 낸 버전과 같은 입력이 되돌아와 다시 만들 때
# 이전 라벨 ID와 겹치지 않게 붙인다 (`stage_version`, F-PAR-54)
_REVISION = re.compile(r"\+r\d+$")
# 단계가 아닌 쪽이 그 단계 라벨을 지운 삭제 레코드의 model_version (단계 실행 이력에서 뺀다)
_NOT_STAGE_RUNS = frozenset({TRAJECTORY_FOLLOW_VERSION, REPLACED_VERSION})
# 사람 개체의 물리 유형 (온톨로지 physical_types의 ID). 접촉 단계는 이 유형 박스와의 접촉을 사람
# 접촉으로 보고 라벨을 만들지 않는다 (`contact_target_kind`, F-PAR-03)
PERSON_PHYSICAL_TYPE = "person"
# replaced에 이 이름이 있으면 배포된 재학습 접촉 모델이 기본 접촉 단계(장갑·영상 휴리스틱)를
# 대신한다 (training.yaml contact.replaces)
CONTACT_STEP = "contact"


def contact_version(policy: PrelabelPolicy, inputs: str) -> str:
    """접촉 단계 모델 버전: `contact-heuristic-1+p<contact 절 해시>+i<입력 해시>`.

    inputs: 접촉 단계 입력 해시 (손 키포인트·객체 박스 라벨 ID, 장갑 스트림 동기화, 압력 채널
    접두사).
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
    contacts: 새 접촉 라벨 수. person_contacts: 대상이 사람 박스라 부위(body_part)를 몰라 라벨을
    만들지 않은 접촉 구간 수 (검수자가 그린다, F-PAR-03).
    retracted: 예측기 단계(1)의 이전 버전과 배포 모델이 대신한 기본 어댑터·접촉 단계에서 지운 라벨
    수 (3D·접촉 단계가 새 버전으로 지운 수는 세지 않는다). lifted: 새 3D 궤적 수.
    wearer: 착용자로 찾은 원래 트랙 라벨 ID. wearer_scores: 인물(라벨 ID)별 상관.
    trajectories_inherited: 2D 트랙 검수를 물려받은 3D 궤적 수. trajectories_retracted: 출처 2D가
    바뀌어 물려받음이 끊겼거나 같은 2D의 새 계산이 대신해 지운 3D 궤적 수 (ADR 0048·0050·0051).
    lift_blocked: 3D 단계가 새로 계산했어야 하지만 같은 대상에 직접 검수·사람 출처 3D가 있어 넣지
    않은 궤적 수 (모두 막히면 원본을 받지 않고 건너뛴다).
    """

    produced: dict[str, int] = field(default_factory=dict[str, int])  # "스트림/predictor" → 라벨 수
    skipped: list[str] = field(default_factory=list[str])
    contacts: int = 0
    person_contacts: int = 0  # 사람 박스 접촉 (부위를 몰라 라벨을 만들지 않음)
    retracted: int = 0  # 예측기 단계·배포 모델 대체로 지운 라벨 (3D·접촉 단계 것은 빠진다)
    lifted: int = 0
    # 카메라 자세 표본 수(이번에 새로 만든 자세 궤적), 새 월드 좌표 궤적 수,
    # 자세 단계를 건너뛴 이유, 추적을 잃은 시각 (ADR 0046)
    poses: int = 0
    world: int = 0
    pose_note: str | None = None
    pose_lost_at_ms: int | None = None
    wearer: str | None = None
    wearer_scores: dict[str, float] = field(default_factory=dict[str, float])
    trajectories_inherited: int = 0
    trajectories_retracted: int = 0
    lift_blocked: int = 0
    # 손 크기 초점 거리 추정 (ADR 0056): "fx <px> hfov <도> n <표본> spread <퍼짐>" 또는 None
    focal_estimate: str | None = None


def _follow(conn: sa.Connection, session_id: str, now: datetime, summary: PrelabelSummary) -> None:
    """3D 궤적이 출처 2D 트랙의 검수를 따르게 한다 (ADR 0048·0050, 세션 잠금, 멱등)."""
    plan = apply_trajectory_follow(conn, session_id, now, TrajectoryInheritanceMechanism.PRELABEL)
    summary.trajectories_inherited += len(plan.approve)
    summary.trajectories_retracted += len(plan.retract)


def _lapsed(
    conn: sa.Connection,
    session_id: str,
    history: list[LabelRecord],
    candidates: Sequence[LabelRecord] = (),
) -> frozenset[str]:
    """궤적 따르기가 지울 물려받은 3D 궤적 (보호 판정·월드 좌표 입력에서 뺀다, 따르기가 지우기 전의
    이중 방어, ADR 0048·0050).

    candidates: 이 단계가 새로 계산한(아직 쓰지 않은) 궤적. 이력에 더해 계획하면, 같은 출처 2D에서
    다른 판으로 계산한 새 궤적이 대신할 물려받기만 한 궤적(대체됨, ADR 0051)도 들어가 새 궤적을
    막지 않는다.
    """
    inheritance = get_trajectory_inheritance(conn, session_id)
    return lapsed_trajectories([*history, *candidates], inheritance)


def _base_version(version: str | None) -> str:
    """재실행 세대 표시(`+r<n>`)를 뗀 단계 버전."""
    return _REVISION.sub("", version or "")


def _stage_records(history: list[LabelRecord], prefix: str) -> list[LabelRecord]:
    """단계가 쓴 이력 레코드: 그 단계 라벨 ID 접두사의 모델 라벨과 그 단계가 쓴 삭제 레코드.

    사람 출처, 궤적 따르기·배포 모델 대체가 쓴 삭제 레코드는 단계 실행이 아니라 뺀다.
    """
    return [
        x
        for x in history
        if x.provenance.source is Source.MODEL
        and x.provenance.model_version not in _NOT_STAGE_RUNS
        and (
            x.label_id.startswith(prefix)
            or (x.retracted and (x.parent_label_id or "").startswith(prefix))
        )
    ]


def stage_version(history: list[LabelRecord], prefix: str, version: str) -> str | None:
    """다시 돌릴지와 쓸 버전 (멱등·되돌림 판정, F-PAR-54). None이면 건너뛴다.

    단계가 마지막으로 쓴 레코드들(가장 늦은 created_at, `_stage_records`)에 같은 버전이 있으면
    지금 결과가 이 입력의 결과이므로 건너뛴다 (검수자가 그 결과를 고치거나 지웠어도 되살리지
    않는다). "이력 어디든 같은 버전이 있으면 건너뜀"이면 입력이 A → B → A로 되돌아올 때 B 입력의
    결과가 남았다. 이전에 낸 적 있는 버전을 다시 만들면 라벨 ID가 이전 것과 겹치지 않게 버전 끝에
    재실행 세대 `+r<n>`(n = 그 버전으로 이미 쓴 세대 수)을 붙인다.

    Args:
        history: 세션 전체 이력.
        prefix: 단계 라벨 ID 접두사 (`<세션>-<스트림>-3d-` 등).
        version: 지금 입력으로 정한 단계 버전 (세대 표시 없음).
    """
    records = _stage_records(history, prefix)
    if not records:
        return version
    latest = max(x.created_at for x in records)
    if any(
        _base_version(x.provenance.model_version) == version
        for x in records
        if x.created_at == latest
    ):
        return None
    used = {
        x.provenance.model_version
        for x in records
        if _base_version(x.provenance.model_version) == version
    }
    return version if not used else f"{version}+r{len(used)}"


def _fetch(store: ObjectStore, uri: str, work: Path) -> Path:
    """원본 저장소의 URI를 임시 작업 디렉터리로 받는다 (이미 받았으면 다시 받지 않는다).

    키의 `/`를 `__`로 바꾼 파일 이름을 쓴다. 접근 기록은 넘겨받은 `store`(감사 저장소)가 남긴다.
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
    pose_estimator: CameraPoseEstimator | None = None,
) -> PrelabelSummary:
    """세션 하나의 프리라벨 단계(모듈 docstring의 1~5)를 돌린다.

    Args:
        conn: DB 연결. 호출자가 트랜잭션을 연다 (`engine.begin()`); 중간에 실패하면 모두 되돌린다.
        session_id: 세션 ID.
        raw: 원본 버킷 저장소. 반드시 감사 저장소(`raw_store`)여야 한다 (ADR 0020).
        predictors: 돌릴 어댑터 (실제 모델, stub, 배포된 재학습 모델 `trained-<과제>`).
        policy: 프리라벨 정책. ontology: 접촉 대상 종류(tool/fixed_surface/object, 사람 박스는
            라벨 없음) 판단에 쓴다.
        now: 새 레코드의 created_at (시간대 필수).
        lifter: 깊이 3D 단계 (가중치가 없으면 None → 건너뛴다).
        replaced: 배포된 재학습 모델이 대신하는 기본 어댑터 이름 (`CONTACT_STEP`이면 접촉 단계).
            그 어댑터·단계가 냈던 검수 전 라벨을 버전과 상관없이 지운다 (같은 대상이 기본 어댑터와
            재학습 모델 양쪽으로 겹쳐 남지 않게).
        pressure_prefixes: 장갑 압력 채널 접두사 (sync.yaml glove.pressure_prefixes). None이면
            저장소 sync.yaml에서 읽는다. 접촉 단계 모델 버전에 들어가 바뀌면 접촉을 다시 만든다.
        pose_estimator: 바디캠 자세 추정기 (None이면 깊이 단계가 있을 때 내장
            `ImuVisualOdometry`, 없으면 자세 단계를 건너뛴다). 시험은 `OraclePoseEstimator`를
            넘긴다.

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
        # 1단계에서 넣은 라벨을 반영해 이력·현재 라벨을 다시 읽는다 (2~4단계의 입력).
        # 세션 이력 사본은 손 키포인트 트랙 때문에 수 GB가 될 수 있다. 다음 사본을 읽기 전에 앞
        # 사본을 놓아 둘 이상을 한꺼번에 들지 않는다 (F-PMS-36)
        del existing
        history = get_labels(conn, session_id)
        current = current_labels(history)
        if lifter is not None:
            summary.lifted = _lift(conn, session, history, current, raw, work, lifter, now, summary)
        del history, current
        # 새 카메라 좌표 3D가 확인된 2D의 검수를 물려받은 뒤 월드 좌표 단계가 입력을 고른다
        _follow(conn, session_id, now, summary)
        if policy.camera_pose.enabled:
            stage = _PoseStage(conn, session, raw, work, policy, now, summary)
            stage.camera_pose(get_labels(conn, session_id), lifter, pose_estimator)
            stage.world(get_labels(conn, session_id))
            _follow(conn, session_id, now, summary)
        history = get_labels(conn, session_id)
        current = current_labels(history)
        if CONTACT_STEP not in replaced:
            _contacts(
                conn, session, history, current, raw, work, policy, ontology, now, prefixes, summary
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


def pick_hand_tracks(hand_labels: list[LabelRecord]) -> dict[Hand, KeypointTrackPayload]:
    """손(왼·오른)마다 접촉 추정에 쓸 hand21 트랙 하나를 고른다.

    같은 손 트랙이 여럿일 수 있다 (사람이 새로 그린 트랙과 모델 트랙, 버전이 다른 모델 트랙이 함께
    현재 라벨인 경우). 예전에는 입력 순서의 마지막 것을 조용히 썼는데, 그 순서는 DB 조회 순서라
    접촉 결과가 실행마다 달라질 수 있었다. 행동 단계와 같은 공용 기준
    (`dlp_schema.episode.pick_hand_track_labels`, `track_rank`: 사람·검수 → 최신 → 긴 구간 → 라벨
    ID)으로 가장 앞서는 트랙을 고른다 (병합하지 않는다).

    Args:
        hand_labels: hand21 키포인트 트랙 라벨 (`payload.hand`가 있는 것).

    Returns:
        손 → 고른 트랙의 페이로드. 트랙이 없는 손은 키가 없다.
    """
    out: dict[Hand, KeypointTrackPayload] = {}
    for hand, x in pick_hand_track_labels(hand_labels).items():
        assert isinstance(x.payload, KeypointTrackPayload)
        out[hand] = x.payload
    return out


def pick_lift_tracks(current: list[LabelRecord], stream_id: str) -> list[LabelRecord]:
    """3D 단계 입력: 그 스트림의 현재 hand21 키포인트·박스 트랙을 (개체, 트랙 종류)마다 하나 고른다.

    같은 entity_id 트랙이 함께 현재 라벨일 수 있다 (승인된 이전 모델 버전 + 새 버전, 사람 트랙 +
    모델 트랙). 둘 다 넘기면 같은 (개체, 부위)에 같은 시각 표본이 두 번 들어가 3D 라벨 검증이
    실패하고 프리라벨 트랜잭션 전체가 롤백됐다 (F-PAR-01). `pick_hand_tracks`와 같은 순서
    (`dlp_schema.episode.track_rank`: 사람·검수 → 최신 → 긴 구간 → 라벨 ID)로 하나를 고르고, 트랙을
    섞지 않는다.
    3D로 올리지 않는 트랙(coco17 등, `lift3d.lift_key`가 None)은 빠진다.

    Args:
        current: 세션 현재 라벨.
        stream_id: 올릴 스트림 (러너는 기준 스트림 바디캠).

    Returns:
        고른 트랙 레코드 (라벨 ID 순). 3D 단계의 입력 해시도 이 목록으로 만든다 (실제로 쓰는 트랙).
    """
    best: dict[tuple[str, str], LabelRecord] = {}
    for x in current:
        key = lift_key(x.payload) if x.stream_id == stream_id else None
        if key is not None and (key not in best or track_rank(x) > track_rank(best[key])):
            best[key] = x
    return sorted(best.values(), key=lambda x: x.label_id)


def _is_wearer(x: LabelRecord) -> bool:
    """착용자 매칭 단계가 만든 사본 레코드인가 (모델 버전 접두사로 판단)."""
    return (x.provenance.model_version or "").startswith(WEARER_PREFIX)


def _trajectory_key(p: Trajectory3DPayload) -> tuple[str, str | None, str]:
    """3D 궤적의 동일 대상 키 (개체, 부위, 좌표계)."""
    return (p.entity_id, p.part, p.frame.value)


def drop_protected_trajectories(
    labels: list[LabelRecord], current: list[LabelRecord], lapsed: frozenset[str]
) -> list[LabelRecord]:
    """새 3D 궤적 중 같은 (개체, 부위, 좌표계)의 검수된·사람 궤적이 이미 있는 것을 버린다.

    ADR 0026: 검수된 결과는 지우지 않으므로, 같은 대상의 모델 궤적을 옆에 또 넣으면 중복이 된다.
    단 궤적 따르기가 지울 궤적(lapsed)은 막지 않는다: 출처 2D가 바뀌어 물려받음이 끊긴 궤적(ADR
    0048·0050)과, 같은 출처 2D에서 입력이 바뀌어 새로 계산한 궤적이 대신할 물려받기만 한 궤적(ADR
    0051, 호출자가 새 궤적을 더한 이력으로 lapsed를 구한다)이다. 이어지는 궤적 따르기가 그것을
    지우고 새 궤적이 그 자리를 이어받는다.
    """
    kept = {
        _trajectory_key(x.payload)
        for x in current
        if isinstance(x.payload, Trajectory3DPayload)
        and is_protected(x)
        and x.label_id not in lapsed
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
        if isinstance(x.payload, HandStatePayload) and x.payload.hand is hand and is_protected(x)
    ]
    return [
        (i, c)
        for i, c in enumerate(intervals)
        # 열린 구간 겹침: 맞닿기만 하면 겹치지 않는다
        if not any(s < c.end_ms and c.start_ms < e for s, e in spans)
    ]


def lift_keys(
    tracks: list[LabelRecord], hand_points: Sequence[int]
) -> set[tuple[str, str | None, str]]:
    """고른 트랙이 3D 단계에서 낼 궤적의 대상 키 (개체, 부위, 좌표계=camera).

    hand21 손 트랙은 정책 hand_points 관절마다, 박스 트랙은 중심(부위 None) 하나다.
    """
    camera = CoordinateFrame.CAMERA.value
    out: set[tuple[str, str | None, str]] = set()
    for t in tracks:
        p = t.payload
        if isinstance(p, KeypointTrackPayload):
            out |= {(p.entity_id, HAND21_NAMES[i], camera) for i in hand_points}
        elif isinstance(p, BoxTrackPayload):
            out.add((p.entity_id, None, camera))
    return out


def blocked_lift_keys(
    tracks: list[LabelRecord],
    current: list[LabelRecord],
    inheritance: Sequence[TrajectoryInheritance],
    lapsed: frozenset[str],
) -> set[tuple[str, str | None, str]]:
    """새 3D가 들어갈 수 없는 대상 키: 같은 대상에 대체할 수 없는 보호 3D가 있다.

    대체할 수 있는 것은 따르기가 지울 궤적(lapsed)과, 지금 고른 트랙에서 물려받기만 한 궤적이다
    (새로 계산하면 궤적 따르기 규칙 5가 대신하게 한다, ADR 0051). 나머지 보호 3D(직접 검수·
    기록 뒤 다시 검수·사람 출처, 다른 2D에서 물려받은 것)는 막는다.
    """
    sources = {t.label_id for t in tracks}
    out: set[tuple[str, str | None, str]] = set()
    for x in current:
        if not isinstance(x.payload, Trajectory3DPayload) or not is_protected(x):
            continue
        if x.label_id in lapsed:
            continue
        r = active_trajectory_inheritance(x, inheritance)
        if r is not None and r.source_label_id in sources:
            continue
        out.add(_trajectory_key(x.payload))
    return out


def _lift(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    lifter: DepthLifter,
    now: datetime,
    summary: PrelabelSummary,
) -> int:
    """3D 궤적 단계: 바디캠 손·박스 트랙을 메트릭 깊이로 카메라 좌표 3D 궤적으로 올린다.

    멱등: 이 단계가 마지막으로 낸 버전이 지금 버전이면(검수자가 모두 고쳤어도) 다시 만들지 않는다
    (`stage_version`, 입력이 이전 값으로 되돌아오면 다시 만든다). 버전
    (`<lifter.version>+i<입력 해시>+hp<w|p>`)에는 실제로 올리는 트랙(고른 트랙의 라벨 ID), 쓸 내부
    파라미터(`calibration_repository.session_intrinsics`)와 손 크기 사전값(`session_palm`,
    ADR 0045)의 해시가 들어가, 입력이 바뀌거나 고르는 트랙이 바뀌면(예: 다른 트랙 승인),
    캘리브레이션·작업자 손 크기를 등록하면 다시 만든다. 2D 검수를 물려받기만 한 이전 3D는 새 3D가
    대신한다 (ADR 0051). 같은 (개체, 부위, 좌표계)에 직접 검수·사람 궤적이 있으면 새 궤적은 버리고
    (`summary.lift_blocked`), 낼 궤적이 모두 막히면 원본을 받거나 깊이를 돌리지 않는다.

    입력: 기준 스트림(바디캠)의 현재 hand21 키포인트·박스 트랙 (모델·사람 출처 모두)을 (개체, 트랙
    종류)마다 하나씩 고른 것 (`pick_lift_tracks`, F-PAR-01). 출력 라벨 ID 접두사는
    `<세션>-<바디캠>-3d-`. Returns: 새로 넣은 3D 궤적 수.
    부작용: 원본 버킷에서 바디캠 영상을 받고, `label_records`에 새 궤적과 이전 버전 삭제 레코드를
    넣는다.
    """
    body = session.reference_stream
    prefix = f"{session.session_id}-{body.stream_id}-3d-"
    # 같은 entity_id 트랙이 여럿이면 하나만 올린다 (표본 시각이 겹쳐 라벨 검증이 실패하지 않게)
    tracks = pick_lift_tracks(current, body.stream_id)
    # 내부 파라미터(세션 값 → 등록된 기기 캘리브레이션 → 화각 근사)와 손 크기 사전값(작업자 등록값 →
    # 정책 기본값)도 입력이다 (ADR 0045). 등록이 바뀌면 입력 해시가 바뀌어 다시 돈다. 사전값 출처
    # 글자(+hpw 작업자, +hpp 정책)는 버전에 그대로 남긴다 (작업자 ID·값은 해시로만 들어간다)
    intr = session_intrinsics(conn, session)
    palm = session_palm(conn, session)
    digest = input_digest(tracks, intr.digest_input(), palm.digest_input())
    version = stage_version(history, prefix, f"{lifter.version}+i{digest}+hp{palm.source}")
    if version is None:
        return 0
    stale = _stale(history, prefix, version)
    # 낼 궤적이 모두 바꿀 수 없는 보호 3D에 막히면 원본을 받거나 깊이를 돌리지 않는다 (헛계산·원본
    # 접근 반복을 막는다). 입력이 모두 사라졌으면(트랙 없음) 검수 전인 이전 결과만 지운다
    lapsed = _lapsed(conn, session.session_id, history)
    inheritance = get_trajectory_inheritance(conn, session.session_id)
    wanted = lift_keys(tracks, lifter.policy.depth.hand_points)
    blocked = wanted & blocked_lift_keys(tracks, current, inheritance, lapsed)
    if not tracks or wanted == blocked:
        summary.lift_blocked += len(blocked)
        if stale:
            insert_labels(conn, retractions(stale, version, now))
        return 0
    labels = lifter.run(
        _fetch(raw, body.uri, work),
        session_id=session.session_id,
        stream_id=body.stream_id,
        tracks=tracks,
        calib=intr.intrinsics,
        ontology_version=session.ontology_version or "",
        version=version,
        palm_m=palm.palm_m,
    )
    focal = getattr(lifter, "last_focal", None)
    if focal is not None:
        summary.focal_estimate = (
            f"fx {focal.intrinsics.fx} hfov {focal.hfov_deg} n {focal.n_samples} "
            f"spread {focal.spread}"
        )
    # 새 궤적을 이력에 더해 지울 궤적을 구한다: 같은 2D에서 물려받기만 한 이전 궤적은 새 궤적이
    # 대신하므로(규칙 5) 막지 않는다
    kept = drop_protected_trajectories(
        labels, current, _lapsed(conn, session.session_id, history, labels)
    )
    summary.lift_blocked += len(labels) - len(kept)
    insert_labels(conn, [*retractions(stale, version, now), *kept])
    return len(kept)


def synced_imu(session: Session) -> Stream | None:
    """자세 단계가 쓸 IMU 스트림: 동기화된(unsynced가 아닌) 첫 IMU 스트림 (없으면 None)."""
    return next(
        (
            s
            for s in session.streams
            if s.kind is StreamKind.IMU and s.sync_method is not SyncMethod.UNSYNCED
        ),
        None,
    )


def imu_to_camera(
    session: Session, policy: PrelabelPolicy
) -> tuple[NDArray[np.float64] | None, str | None]:
    """IMU → 카메라 회전과 쓸 수 없을 때의 이유 (세션 `camera_imu_extrinsics` → 정책 기본값).

    세션 값은 계약이 모양·직교성을 검사하지 않으므로 여기서 정책 기본값과 같은 기준으로 본다
    (`posemath.rotation_problem`, 허용 오차 `camera_pose.imu.rotation_tolerance`): 3x3 또는 4x4,
    회전 부분이 직교이고 행렬식이 양수(반사 아님). 허용 오차 안이면 가장 가까운 회전으로 고친다.
    예외를 던지지 않는다 (IMU가 없는 세션의 프리라벨까지 실패하지 않게, F-PAR-49).

    Returns:
        (R_ci 3x3, None) 또는 (None, 이유 문장).
    """
    tol = policy.camera_pose.imu.rotation_tolerance
    extr = session.calibration.camera_imu_extrinsics
    if extr is not None:
        r = as_rotation(extr, tol)
        if r is None:
            return None, f"카메라-IMU 외부 파라미터를 쓸 수 없습니다: {rotation_problem(extr, tol)}"
        return r, None
    default = policy.camera_pose.imu.default_imu_to_camera
    if default is None:
        return None, "카메라-IMU 외부 파라미터가 없습니다 (camera_imu_extrinsics)"
    return as_rotation(default, tol), None


def pick_world_inputs(
    current: list[LabelRecord], stream_id: str, lapsed: frozenset[str]
) -> tuple[LabelRecord | None, list[LabelRecord]]:
    """월드 좌표 단계 입력: (현재 카메라 자세 하나, 카메라 좌표 궤적 (개체, 부위)마다 하나).

    여럿이면 `track_rank`(사람·검수 → 최신 → 긴 구간 → 라벨 ID)로 고른다. 자세는 그 스트림의
    entity_id="camera"·frame=world 궤적, 카메라 좌표 궤적은 그 스트림의 frame=camera 궤적이다.
    출처 2D가 바뀌어 물려받음이 끊긴 궤적(lapsed, ADR 0048)은 고르지 않는다 (낡은 3D다). 검수자가
    없는 사람 출처 카메라 좌표 궤적도 고르지 않는다: 그 월드 좌표는 물려받을 검수가 없어 검수
    경로가 없는 채 완료 판정을 영구히 막는다 (F-PAR-58).
    """
    pose: LabelRecord | None = None
    cams: dict[tuple[str, str | None], LabelRecord] = {}
    for x in current:
        p = x.payload
        if (
            x.stream_id != stream_id
            or not isinstance(p, Trajectory3DPayload)
            or x.label_id in lapsed
            or (x.provenance.source is Source.HUMAN and not human_confirmed(x))
        ):
            continue
        if is_pose_label(x):
            if pose is None or track_rank(x) > track_rank(pose):
                pose = x
        elif p.frame is CoordinateFrame.CAMERA:
            key = (p.entity_id, p.part)
            if key not in cams or track_rank(x) > track_rank(cams[key]):
                cams[key] = x
    return pose, sorted(cams.values(), key=lambda x: x.label_id)


@dataclass
class _PoseStage:
    """자세·월드 좌표 단계 (모듈 docstring 5).

    둘 다 멱등이다 (같은 버전이 이력에 있으면 건너뛴다).
    """

    conn: sa.Connection
    session: Session
    raw: ObjectStore
    work: Path
    policy: PrelabelPolicy
    now: datetime
    summary: PrelabelSummary

    def _sensor_digest(self, imu: Stream, r_ci: NDArray[np.float64]) -> str:
        """자세의 센서 입력 해시: IMU 스트림(위치·동기화 값), IMU → 카메라 회전, 내부 파라미터.

        깊이 모델 없이도 계산할 수 있어, 깊이 모델이 없는 실행에서도 지금 자세가 지금 동기화·
        캘리브레이션으로 만든 것인지 확인한다 (`_invalidate_changed`, F-PAR-56).
        """
        sync = json.dumps(
            {
                "uri": imu.uri, "offset_ms": imu.offset_ms, "clock_scale": imu.clock_scale,
                "manual_adjustment_ms": imu.manual_adjustment_ms, "method": imu.sync_method.value,
            },
            sort_keys=True,
        )  # fmt: skip
        intr = session_intrinsics(self.conn, self.session).digest_input()
        rot = json.dumps(np.round(r_ci, 9).tolist())
        return input_digest([], sync, rot, intr)

    def _version(self, est: CameraPoseEstimator, sensors: str, lifter: DepthLifter | None) -> str:
        """자세 단계 버전 `campose-<추정기 판>+p<camera_pose 절 해시>+s<센서 입력>+d<깊이 입력>`.

        깊이 입력은 깊이 가중치 판과 화각 근사 값(내부 파라미터가 없을 때 쓴다)·화면비 허용 오차만
        이다 (depth 절 전체 해시가 아니다: 손 맞춤 정책만 바꿔도 자세를 다시 추정하고 월드 좌표를
        지웠다 다시 만들던 것을 막는다, F-PAR-53).
        """
        d = self.policy.depth
        depth = (
            "none"
            if lifter is None
            else f"{lifter.weights_version}|hfov={d.default_hfov_deg}"
            f"|aspect={d.max_intrinsics_aspect_mismatch}"
        )
        cam = self.policy.digest("camera_pose")
        return f"{POSE_PREFIX}-{est.version}+p{cam}+s{sensors}+d{input_digest([], depth)}"

    def _skip(self, note: str) -> None:
        """자세 단계를 건너뛴 이유를 요약에 남긴다."""
        self.summary.pose_note = note

    def _retract(self, stale: list[LabelRecord], version: str) -> int:
        """검수 전 이전 자세를 지운다 (지운 수). 월드 좌표 단계·궤적 따르기가 뒤이어 정리한다."""
        if stale:
            insert_labels(self.conn, retractions(stale, version, self.now))
        return len(stale)

    def _invalidate(self, history: list[LabelRecord], note: str) -> None:
        """자세 입력이 더는 성립하지 않는다(IMU 없음·동기화 해제, 외부 파라미터 없음·잘못됨): 검수
        전 이전 자세를 지우고 이유를 남긴다 (자동 라벨이라 사람 검수가 막지 않는다, F-PAR-50·56)."""
        n = self._retract(_stale(history, self._prefix(), ""), POSE_INVALIDATED_VERSION)
        self._skip(f"{note}{f' (이전 자세 {n}개를 지움)' if n else ''}")

    def _invalidate_changed(self, history: list[LabelRecord], sensors: str, note: str) -> None:
        """다시 추정할 수 없는 실행(깊이 모델 없음): 지금 센서 입력과 다른 입력으로 만든 검수 전
        자세만 지운다 (재동기화·캘리브레이션 뒤 낡은 자세가 남지 않게, F-PAR-56). 센서 입력 표시
        (`+s`)가 없는 예전 판 자세는 비교할 수 없어 남긴다."""
        changed = [
            x
            for x in _stale(history, self._prefix(), "")
            if "+s" in (x.provenance.model_version or "")
            and f"+s{sensors}+" not in (x.provenance.model_version or "")
        ]
        n = self._retract(changed, POSE_INVALIDATED_VERSION)
        lead = "[자세 낡음] IMU 동기화·캘리브레이션이 바뀌어 " if n else ""
        self._skip(f"{lead}{f'이전 자세 {n}개를 지웠습니다. ' if n else ''}{note}")

    def _prefix(self) -> str:
        """자세 라벨 ID 접두사."""
        return f"{self.session.session_id}-{self.session.reference_stream.stream_id}-campose-"

    def camera_pose(
        self,
        history: list[LabelRecord],
        lifter: DepthLifter | None,
        estimator: CameraPoseEstimator | None,
    ) -> None:
        """바디캠 자세를 추정해 자세 궤적 라벨을 넣는다 (조건이 안 되면 이유만 남긴다).

        이전 자세 처리 (ADR 0046 3절, F-PAR-50·56): 자세는 사람 검수 대상이 아닌 자동 라벨이라,
        입력이 더는 성립하지 않으면(IMU 없음·동기화 해제, 외부 파라미터 없음·회전이 아님) 검수 전
        이전 자세를 지운다. 깊이 모델이 없어 다시 추정할 수 없으면 센서 입력이 바뀐 자세만 지운다.
        새 버전으로 추정이 실패하거나 표본이 0개여도 이전 자세를 지운다 (이유를 남긴다). 그 뒤 월드
        좌표 단계가 검수 전 월드 3D를, 궤적 따르기가 물려받기만 한 월드 3D를 정리한다.

        부작용: 원본 버킷에서 IMU Parquet과 바디캠 영상을 받는다 (넘겨받은 감사 저장소).
        """
        session, body = self.session, self.session.reference_stream
        imu = synced_imu(session)
        if imu is None:
            return self._invalidate(history, "동기화된 IMU 스트림이 없습니다")
        r_ci, problem = imu_to_camera(session, self.policy)
        if r_ci is None:
            return self._invalidate(history, problem or "")
        sensors = self._sensor_digest(imu, r_ci)
        if estimator is None and lifter is None:
            return self._invalidate_changed(
                history, sensors, "깊이 모델이 없어 이동 배율을 정할 수 없습니다"
            )
        d = self.policy.depth
        est: CameraPoseEstimator = estimator or ImuVisualOdometry(
            self.policy.camera_pose,
            hfov_deg=d.default_hfov_deg,
            max_aspect_mismatch=d.max_intrinsics_aspect_mismatch,
        )
        version = stage_version(history, self._prefix(), self._version(est, sensors, lifter))
        if version is None:
            return None
        stale = _stale(history, self._prefix(), version)
        track = self._estimate(est, imu, r_ci, lifter)
        if isinstance(track, str):
            n = self._retract(stale, version)
            return self._skip(f"자세 추정 실패{f'(이전 자세 {n}개를 지움)' if n else ''}: {track}")
        label = pose_label(
            track, session_id=session.session_id, stream_id=body.stream_id, version=version,
            ontology_version=session.ontology_version or "", now=self.now,
            intrinsics=self._intrinsics_source(est),
        )  # fmt: skip
        if label is None:
            n = self._retract(stale, version)
            gone = f" (이전 자세 {n}개를 지움)" if n else ""
            return self._skip(f"자세 표본이 0개입니다 (IMU가 영상 구간을 덮지 않음 등){gone}")
        current = current_labels(history)
        lapsed = _lapsed(self.conn, session.session_id, history, [label])
        new = drop_protected_trajectories([label], current, lapsed)
        insert_labels(self.conn, [*retractions(stale, version, self.now), *new])
        self.summary.poses = len(track.samples) if new else 0
        self.summary.pose_lost_at_ms = track.lost_at_ms
        return None

    def _estimate(
        self,
        est: CameraPoseEstimator,
        imu: Stream,
        r_ci: NDArray[np.float64],
        lifter: DepthLifter | None,
    ) -> PoseTrack | str:
        """IMU(마스터 시각으로 바꿈)·바디캠 영상·깊이로 자세를 추정한다. 실패하면 그 이유 문장."""
        body = self.session.reference_stream
        data = read_imu_table(_fetch(self.raw, imu.uri, self.work))
        # IMU 시각(스트림 ms) → 마스터 ms (오프셋·수동 보정·clock_scale 드리프트 포함)
        series = ImuSeries(
            np.array([imu.to_master_ms(float(t)) for t in data.t_ms]), data.acc, data.gyro
        )
        try:
            return est.estimate(
                iter_frames(_fetch(self.raw, body.uri, self.work)),
                imu=series,
                imu_to_camera=r_ci,
                intrinsics=session_intrinsics(self.conn, self.session).intrinsics,
                depth=ModelDepth(lifter.depth_model()) if lifter else None,
            )
        except ValueError as exc:
            return str(exc)

    def _intrinsics_source(self, est: CameraPoseEstimator) -> IntrinsicsSource:
        """자세 라벨에 남길 내부 파라미터 출처 (F-PAR-51).

        내장 오도메트리는 실제로 캘리브레이션을 썼는지(화면비가 달라 화각 근사로 떨어졌는지)를
        안다. 다른 추정기는 세션에 내부 파라미터가 있으면 calibrated로 본다.
        """
        if isinstance(est, ImuVisualOdometry):
            used = est.calibrated
        else:
            used = session_intrinsics(self.conn, self.session).intrinsics is not None
        return IntrinsicsSource.CALIBRATED if used else IntrinsicsSource.HFOV_APPROX

    def world(self, history: list[LabelRecord]) -> None:
        """현재 자세와 카메라 좌표 궤적으로 월드 좌표 궤적을 넣는다.

        입력(자세나 카메라 좌표 궤적)이 없으면 검수 전 이전 결과만 지운다.
        """
        session, body = self.session, self.session.reference_stream.stream_id
        current = current_labels(history)
        lapsed = _lapsed(self.conn, session.session_id, history)
        pose, cams = pick_world_inputs(current, body, lapsed)
        inputs = [pose, *cams] if pose is not None else []
        prefix = f"{session.session_id}-{body}-world3d-"
        version = stage_version(
            history,
            prefix,
            f"{WORLD_PREFIX}+p{self.policy.digest('world_frame')}+i{input_digest(inputs)}",
        )
        if version is None:
            return
        stale = _stale(history, prefix, version)
        labels: list[LabelRecord] = []
        if pose is not None and cams:
            poses = pose_samples(pose)
            gap = self.policy.world_frame.max_pose_gap_ms
            for cam in cams:
                lbl = world_label(cam, pose, poses, version=version, max_gap_ms=gap, now=self.now)
                if lbl is not None:
                    labels.append(lbl)
        labels = drop_protected_trajectories(
            labels, current, _lapsed(self.conn, session.session_id, history, labels)
        )
        if stale or labels:
            insert_labels(self.conn, [*retractions(stale, version, self.now), *labels])
        self.summary.world = len(labels)


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


def contact_target_kind(class_id: str | None, ontology: Ontology) -> str | None:
    """접촉 대상 클래스 → HandStatePayload.contact_target_kind. None이면 접촉 라벨을 만들지 않는다.

    - 물리 유형이 `PERSON_PHYSICAL_TYPE`인 클래스(COCO person 박스 → 온톨로지 person): None.
      온톨로지상 사람 접촉(person)은 body_part가 필수인데(hand.yaml, HandStatePayload 검증)
      subjects.yaml body_parts에는 "모름" 값이 없고, 박스로는 대상자의 어느 부위인지 알 수 없다.
      바디캠 속 사람 박스는 착용자 자신의 팔(self)일 수도 있다. 예전처럼 object로 쓰면 틀린 종류가
      관계·행동·내보내기로 흘러가므로(F-PAR-03) 라벨을 만들지 않고, 검수자가 부위와 함께 그린다.
    - tool 정의가 있으면 "tool", surface면 "fixed_surface", 그 밖(모르는 클래스, 장갑만 잡아 대상이
      없는 접촉 포함)은 "object".
    """
    obj = ontology.objects.get(class_id or "")
    if obj is None:
        return "object"
    if obj.physical_type == PERSON_PHYSICAL_TYPE:
        return None
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
    summary: PrelabelSummary,
) -> None:
    """접촉 단계: 바디캠 손·박스 트랙과 동기화된 장갑으로 손마다 접촉 구간을 만들어 hand_state로
    쓴다 (결과 수를 summary에 적는다).

    멱등: 이력에 이 버전이 있으면 (검수자가 모두 고쳤거나 지웠어도) 다시 만들지 않는다. 정책(contact
    절)이나 입력(손·객체 트랙, 장갑 동기화, 압력 채널 접두사)이 바뀌면 버전이 바뀌어 다시 만들고,
    검수 전인 이전 버전 접촉만 지운다. 같은 손의 검수된·사람 손 상태와 겹치는 새 구간은 버린다.

    장갑이 있는 손은 `fuse_contacts`(장갑 시각 + 영상 대상), 없는 손은 영상 구간만 쓴다. 구간은
    세션 [0, `session.duration_ms`]로 자르고, 세션 밖 구간과 잘려서 출처의 최소 길이보다 짧아진
    조각은 버린다 (`clip_to_session`, F-PAR-02: 오프셋이 음수인 장갑). 라벨은 마스터 타임라인
    구간(stream_id=None)이고 ID는 `<세션>-contact-<version_tag>-<손>-<순번 4자리>`다 (순번은 자른
    뒤 목록 기준). 신뢰도는 `contact.confidence.<출처>`, 증거는 영상만이면 INFERRED, 그 밖은
    OBSERVED. 대상 종류는 `contact_target_kind`로 정한다. 대상이 사람 박스(물리 유형 person)인
    구간은 부위(body_part)를 알 수 없어 라벨을 만들지 않고 `summary.person_contacts`로 센다
    (F-PAR-03, 검수자가 부위와 함께 그린다).

    결과: `summary.contacts`(새로 넣은 접촉 라벨 수), `summary.person_contacts`(라벨을 만들지 않은
    사람 박스 접촉 수). 같은 버전이 이력에 있으면 둘 다 0 그대로다.
    부작용: 원본 버킷에서 장갑 Parquet을 받고, `label_records`에 새 접촉과 이전 버전 삭제 레코드를
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
        return
    stale = _stale(history, f"{session.session_id}-contact-", version)
    # 손마다 트랙 하나만 쓴다. 같은 손 트랙이 여럿이면 정해진 순서로 하나를 고른다
    # (`pick_hand_tracks`)
    hands = pick_hand_tracks(hand_labels)
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
        # 마스터로 바꾼 장갑 시각은 세션 밖일 수 있다 (오프셋이 음수면 0 앞). 세션 [0, 길이]로
        # 자르고, 세션 밖 구간과 잘려서 최소 길이보다 짧아진 조각은 버린다 (음수 시각 라벨은
        # 검증에서 실패해 트랜잭션 전체가 롤백된다)
        intervals = clip_to_session(intervals, session.duration_ms, policy.contact)
        for i, c in drop_protected_contacts(intervals, hand, current):
            kind = contact_target_kind(classes.get(c.target_id or ""), ontology)
            if kind is None:
                # 사람 박스 접촉: 부위를 몰라 라벨을 만들지 않고 센다 (검수자가 그린다, F-PAR-03).
                # 순번 i는 건너뛰어도 남은 구간의 라벨 ID가 바뀌지 않는다
                summary.person_contacts += 1
                continue
            labels.append(contact_label(session, version, hand, i, c, kind, policy, now))
    insert_labels(conn, [*retractions(stale, version, now), *labels])
    summary.contacts = len(labels)


def contact_label(
    session: Session,
    version: str,
    hand: Hand,
    index: int,
    c: ContactInterval,
    kind: str,
    policy: PrelabelPolicy,
    now: datetime,
) -> LabelRecord:
    """접촉 구간 하나 → 손 상태 라벨 (모델 출처, 마스터 타임라인 구간, stream_id=None).

    ID는 `<세션>-contact-<version_tag>-<손>-<순번 4자리>`, 신뢰도는 `contact.confidence.<출처>`다.
    증거는 영상만 잡은 접촉이면 INFERRED, 장갑이 관여한 접촉(장갑만·융합)이면 OBSERVED다. AI 1차
    검수(`dlp_schema.aireview.glove_derived`)는 이 표시로 장갑 센서에서 나온 접촉을 판정 대상에서
    뺀다. 장갑만 잡은 접촉은 대상을 몰라 `contact.unresolved_target_id`를 쓴다 (관계·행동 단계가
    버린다).

    Args:
        session: 세션. version: 접촉 단계 모델 버전. hand: 손. index: 그 손의 순번.
        c: 접촉 구간. kind: 접촉 대상 종류 (`contact_target_kind`, None이 아닌 값).
        policy: 프리라벨 정책. now: 기록 시각.
    """
    payload = HandStatePayload(
        hand=hand,
        contact_target_kind=kind,
        target_id=c.target_id or policy.contact.unresolved_target_id,
        role="active",
    )
    return model_label(
        label_id=f"{session.session_id}-contact-{version_tag(version)}-{hand.value}-{index:04d}",
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
    부작용: 원본 버킷에서 IMU Parquet을 받고, `label_records`에 착용자 사본 하나를 넣는다.
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
    # IMU는 공유 시계(SHARED_CLOCK)라 시각을 그대로 기준 신호로 쓴다 (|가속도 크기 - 그 중앙값|,
    # `dlp_sync.signals.imu_series`)
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
