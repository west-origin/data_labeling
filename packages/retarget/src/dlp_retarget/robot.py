"""로봇 등록: `config/robots/<ID>.yaml` 로더, 모델 파일 sha256 확인 (ADR 0047·0068).

로봇 설정 파일 하나가 로봇 하나다. 고객 로봇의 설계 파일(URDF)과 설정은 저장소에 넣지 않는다:
`DLP_ROBOTS_DIR` 환경 변수의 디렉터리(저장소 밖)를 먼저 찾고, 없으면 저장소 `config/robots/`(공개
시험 모델만, `model.public: true`)를 찾는다. 모델 경로는 설정 파일 디렉터리 기준 상대 경로나 절대
경로다. 모델 파일은 읽기 전에 sha256을 확인한다 (다른 파일이면 거부, 내보내기 manifest·버전에 해시가
들어간다).

로봇은 두 가지다 (어느 쪽이든 팔 하나가 사람 손 하나를 따라간다). 설정의 `arm` 절에 `gripper`가
있는지 `dexterous_hand`가 있는지로 가른다. 둘을 함께 적으면 거부한다.

평행 그리퍼 로봇 (`RobotConfig` → `LoadedRobot`, `arm.gripper`):
- robot_id, display_name, model(format urdf, path, sha256, license, source, public).
- base_link: 로봇 기준 링크. arm: 사람 손 하나를 따라가는 팔 (손, 끝 링크, 움직일 관절 순서, 끝 링크
  좌표의 접근 축·닫힘 축, 그리퍼, 배치). ik: 허용 오차·가중치·반복 수.

다지 손 로봇 (`DexterousRobotConfig` → `LoadedDexterousRobot`, `arm.dexterous_hand`, ADR 0068):
- arm: 손, 기준 링크 → 손목 링크 연쇄의 관절 순서, 배치, 다지 손 절(`DexterousHandConfig`: 사람
  손목에 대응하는 손목 링크, 사람 손가락 → 로봇 손가락 끝 링크 대응 1~5개, 손 배율, 손가락 끝 허용
  오차). 끝 링크·접근 축·닫힘 축·그리퍼는 없다. ik: `HandIkConfig` (방향 항목 없음, 손목 가중치).
- 손가락 관절은 설정에 적지 않고 URDF에서 손목 링크 → 끝 링크 경로의 움직이는 관절로 읽는다.
  내보내는 관절 순서는 팔 관절 + 손가락 관절(대응 순서)이다.
- 거부하는 구조 (지어내지 않는다): 손가락 경로의 mimic 관절(독립 자유도로 풀면 결합 조건을 어긴다),
  손가락끼리 관절을 함께 쓰는 손(목표가 없는 손가락의 관절만 따로 유지할 수 없다), 손목 링크 아래에
  있지 않은 끝 링크, 움직이는 관절이 없는 손가락, URDF에 없는 링크.
- 손목 링크 아래의 움직이는 관절 가운데 어느 손가락 경로에도 없는 것(대응하지 않은 손가락 등)은
  풀지도 내보내지도 않는다. `LoadedDexterousRobot.unmapped_joints`로 드러낸다.

읽는 함수: `load_robot`·`load_robot_file`은 평행 그리퍼 로봇만 읽는다 (돌려주는 타입이
`LoadedRobot`, 다지 손 설정이면 거부). 두 가지를 다 읽으려면 `load_any_robot`·
`load_any_robot_file`을 쓴다 (내보내기와 `dlp robot check`). `arm.gripper.type: dexterous`는
거부하고 `arm.dexterous_hand`를 안내한다. 양팔(손 둘) 로봇은 아직 없다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from dlp_retarget.urdf import (
    MOVING,
    Chain,
    Joint,
    Tree,
    UrdfError,
    build_chain,
    build_tree,
    load_joints,
)
from dlp_schema.common import Contract, Identifier, OntologyId
from dlp_schema.labels import Hand

# 고객 로봇 설정·모델을 두는 저장소 밖 디렉터리를 가리키는 환경 변수
ROBOTS_DIR_ENV = "DLP_ROBOTS_DIR"
# 로봇 ID 형식 (`dlp_schema.common.Identifier`와 같다). 첫 글자가 점·슬래시가 아니고 슬래시가 없어
# `../` 같은 경로 탈출이 생기지 않는다
_ROBOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
# 접근·닫힘 축의 단위 길이·직교 검사 허용 오차. 정책 값이 아니라 입력 확인용이다: 설정에는 정확한
# 단위 벡터(예: [0, 0, 1], [0.7071068, 0.7071068, 0])를 적는다
_AXIS_TOLERANCE = 1e-6


class RobotError(ValueError):
    """로봇 설정·모델을 찾거나 확인할 수 없을 때."""


class RobotModel(Contract):
    """로봇 설계 파일.

    path: 설정 파일 디렉터리 기준 상대 경로 또는 절대 경로. sha256: 파일 해시 (확인 후에만 읽는다).
    license / source: 라이선스와 출처 (공개 모델은 받은 곳·날짜, 고객 모델은 계약 ID 등).
    public: 공개 시험 모델인가. 저장소 안 설정은 true여야 한다 (고객 모델은 저장소에 넣지 않는다).
    """

    format: Literal["urdf"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str
    source: str
    public: bool


class GripperType(StrEnum):
    """그리퍼 종류.

    - `parallel`: 평행 그리퍼. 엄지-검지 끝 거리를 그리퍼 관절 하나로 옮긴다
      (`targets.gripper_values`).
    - `dexterous`: 다지 손. 값은 있지만 `GripperConfig` 검증이 거부한다. 다지 손은 그리퍼 관절
      하나로 적을 수 없어 `arm.gripper` 대신 `arm.dexterous_hand` 절(`DexterousHandConfig`)에
      적는다 (ADR 0068).
    """

    PARALLEL = "parallel"
    DEXTEROUS = "dexterous"


class GripperConfig(Contract):
    """그리퍼. 평행 그리퍼는 엄지-검지 끝 거리를 [closed_distance_m, open_distance_m] → [closed,
    open] 관절 값으로 선형 대응한다 (범위 밖 거리는 끝값, 대응 범위일 뿐 도달 가능성과 무관)."""

    type: GripperType
    joint: str
    closed: float
    open: float
    closed_distance_m: float = Field(ge=0)
    open_distance_m: float = Field(gt=0)

    @model_validator(mode="after")
    def _check(self) -> GripperConfig:
        """다지 손 거부(다른 절에 적는다), 거리 범위 순서 검사. Raises: ValueError."""
        if self.type is GripperType.DEXTEROUS:
            raise ValueError(
                "다지 손(dexterous)은 arm.gripper가 아니라 arm.dexterous_hand 절에 적습니다 "
                "(docs/handover/robot-retargeting.md, ADR 0068)"
            )
        if self.open_distance_m <= self.closed_distance_m:
            raise ValueError("open_distance_m은 closed_distance_m보다 커야 합니다")
        return self


class PlacementMode(StrEnum):
    """세션 월드 좌표의 손 점을 로봇 기준 좌표로 옮기는 방식 (식은 `dlp_retarget.targets`).

    - `world_fixed`: 로봇 기준 링크를 세션 월드 좌표의 고정 자세(`base_xyz` m, `base_yaw_rad` rad,
      z축 둘레 요만)에 둔다. 로봇이 실제 촬영 공간의 한 자리에 서 있다고 보는 방식이다.
    - `episode_centroid`: 에피소드의 손 잡기 점 평균이 로봇 기준 좌표 `workspace_center`(m)에
      오도록 옮긴다. 회전은 없다 (월드 X = 로봇 X, 월드 Z = 로봇 Z). 세션마다 월드 원점이 달라도
      손 동작이 로봇 작업 공간 가운데에 놓인다.

    옮기는 점은 평행 그리퍼 로봇이면 잡기 점(엄지·검지 끝 가운데), 다지 손 로봇이면 손목이다
    (`episode_centroid`의 평균도 그 점의 평균).
    """

    WORLD_FIXED = "world_fixed"
    EPISODE_CENTROID = "episode_centroid"


class Placement(Contract):
    """사람 손 → 로봇 기준 좌표 배치.

    position_scale: 위치 배율 (사람 팔 길이 대비 로봇, 1이면 그대로). world_fixed는 로봇 기준 원점,
    episode_centroid는 손 잡기 점 평균을 중심으로 줄인다. 방향(접근·닫힘 축)은 배율과 무관하다.
    base_xyz (m) / base_yaw_rad (rad): world_fixed에서 세션 월드 좌표의 로봇 기준 자세.
    workspace_center (m): episode_centroid에서 손 잡기 점 평균을 놓을 로봇 기준 좌표.
    쓰지 않는 방식의 값은 무시한다.
    """

    mode: PlacementMode
    position_scale: float = Field(gt=0)
    base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_yaw_rad: float = 0.0
    workspace_center: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ArmConfig(Contract):
    """사람 손 하나를 따라가는 팔.

    tip_link: 끝 링크 (도구 중심점).
    joints: 움직일 관절 이름 (연쇄의 움직이는 관절과 같은 순서·집합).
    approach_axis / closing_axis: 끝 링크 좌표의 접근 축(손목 → 손가락 끝 방향)과 그리퍼 닫힘 축.
        둘 다 단위 벡터이고 서로 직교해야 한다 (회전 오차 = 두 단위 벡터 사이 각이라, 길이가 1이
        아니면 오차와 도달 판정이 틀린다). 조용히 정규화하지 않는다: 적은 값과 쓰는 값(설정 해시)이
        같아야 한다.
    """

    hand: Hand
    tip_link: str
    joints: tuple[str, ...] = Field(min_length=1)
    approach_axis: tuple[float, float, float]
    closing_axis: tuple[float, float, float]
    gripper: GripperConfig
    placement: Placement

    @model_validator(mode="after")
    def _axes(self) -> ArmConfig:
        """접근·닫힘 축이 단위 길이이고 서로 직교하는지. Raises: ValueError."""
        for name, v in (("approach_axis", self.approach_axis), ("closing_axis", self.closing_axis)):
            if abs(math.sqrt(sum(x * x for x in v)) - 1.0) > _AXIS_TOLERANCE:
                raise ValueError(f"{name} {list(v)}는 단위 벡터여야 합니다")
        dot = sum(a * c for a, c in zip(self.approach_axis, self.closing_axis, strict=True))
        if abs(dot) > _AXIS_TOLERANCE:
            raise ValueError("approach_axis와 closing_axis는 서로 직교해야 합니다")
        return self


class OrientationMode(StrEnum):
    """IK가 맞추는 끝단 방향.

    어느 것을 고를지는 관절 수가 아니라 팔 구조로 정한다 (`dlp robot check`의 영점 축으로 본다):
    - full: 접근 축 둘레 롤을 바꾸는 관절(손목 끝 wrist roll 등)이 있으면 쓴다. 그 관절이 사람 손의
      롤(엄지 → 검지 닫힘 축)을 따라가게 된다. SO-101처럼 pan + pitch 연쇄 끝에 wrist roll이 있는
      5축 팔도 full이다 (이 구조는 접근 축의 요를 따로 맞추지 못할 뿐 롤은 맞춘다. 요 오차는 가중
      최소제곱이 위치와 나눠 지고 보고 오차에 드러난다).
    - approach: 접근 축 둘레 롤을 바꿀 관절이 없을 때만 쓴다 (롤이 목표 없이 남아도 내보낼 관절이
      없다). 롤 관절이 있는데 approach를 쓰면 그 관절 값이 사람 손과 무관한 IK 부산물이 된다.
    - none: 위치만 (방향을 맞출 수 없는 3축 팔 등). 내보내는 회전 오차는 -1(재지 않음)이다.
    """

    NONE = "none"  # 위치만 맞춘다
    APPROACH = "approach"  # 위치 + 접근 축 (축 둘레 롤은 자유)
    FULL = "full"  # 위치 + 접근·닫힘 축 (롤까지)


class IkConfig(Contract):
    """IK 설정 (로봇마다).

    position_tolerance_m / orientation_tolerance_rad: 위치 오차가 앞의 값 이하이고,
    (orientation_mode가 none이 아니면) 회전 오차가 뒤의 값 이하일 때 그 프레임은 도달 가능이다.
    orientation_mode: 맞출 끝단 방향 (`OrientationMode`).
    orientation_weight: 회전 오차 1 rad를 위치 몇 m로 볼지. smoothness_weight: 목표가 이어지는
    프레임에서 반복 걸음을 줄이는 추가 감쇠 μ (직전 해에서 시작하므로 가까운 자세 가지로 수렴한다.
    다시 시작하는 프레임에는 쓰지 않는다). damping: 감쇠 최소제곱 λ.
    max_iterations: 프레임당 최대 반복 수.
    restarts: 첫 목표 프레임·목표가 끊긴 뒤 다시 시작할 때 쓰는 시작점 수 (관절 범위 가운데 1개 +
    결정적 난수 restarts - 1개, 직전 해가 있으면 그것을 하나 더 넣는다, `ik.seeds`).
    velocity_scale: URDF 속도 한계에 곱하는 값 (0 초과 1 이하, 보수적으로 낮춘다).
    """

    position_tolerance_m: float = Field(gt=0)
    orientation_mode: OrientationMode
    orientation_tolerance_rad: float = Field(gt=0)
    orientation_weight: float = Field(ge=0)
    smoothness_weight: float = Field(ge=0)
    damping: float = Field(gt=0)
    max_iterations: int = Field(ge=1)
    restarts: int = Field(ge=1)
    velocity_scale: float = Field(gt=0, le=1)


class RobotConfig(Contract):
    """`config/robots/<ID>.yaml` 전체 (평행 그리퍼 로봇)."""

    robot_id: Identifier
    display_name: str
    model: RobotModel
    base_link: str
    arm: ArmConfig
    ik: IkConfig

    def digest(self) -> str:
        """설정 값 해시 (sha256 앞 12자, 리타기팅 버전에 들어간다)."""
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


class FingerConfig(Contract):
    """사람 손가락 하나 → 로봇 손가락 끝 링크 (다지 손, ADR 0068).

    human_part: 사람 손가락 끝의 3D 궤적 부위 (온톨로지 hand_joints의 ID, 예: thumb_tip).
        내보내기가 세션 월드 좌표 궤적(개체 `<손>_hand`, 이 부위)에서 그 손가락 끝 점을 읽고,
        내보내기 시작 때 온톨로지와 대조한다.
    tip_link: 그 손가락 끝에 대응하는 로봇 링크 (링크 좌표의 원점이 목표를 따라간다). 손목 링크
        아래에 있어야 한다. 손목 링크 → 이 링크 경로의 움직이는 관절이 이 손가락의 관절이다.
    """

    human_part: OntologyId
    tip_link: str


class DexterousHandConfig(Contract):
    """다지 손: 손가락 관절이 있는 로봇 손 (`arm.gripper` 대신 `arm.dexterous_hand`, ADR 0068).

    wrist_link: 사람 손목에 대응하는 로봇 링크 (링크 좌표의 원점이 손목 목표를 따라간다).
    hand_scale: 사람 (손가락 끝 - 손목) 벡터에 곱하는 손 크기 배율 (로봇 손 크기 / 사람 손 크기).
        팔 작업 공간 배율 `placement.position_scale`과 따로다: 팔 궤적을 줄여도 손가락 벌림은 로봇
        손 크기에 맞춘다. `dlp robot check`가 보이는 손목 → 손가락 끝 길이를 보고 정한다.
    fingertip_tolerance_m: 손가락 끝 위치 오차가 이 이하면 그 손가락은 그 프레임에 도달이다.
    fingers: 사람 손가락 → 로봇 손가락 끝 링크 대응 (1~5개). 이 순서가 내보내는 손가락 관절
        순서다. 같은 사람 손가락이나 같은 끝 링크를 두 번 쓸 수 없다.
    """

    wrist_link: str
    hand_scale: float = Field(gt=0)
    fingertip_tolerance_m: float = Field(gt=0)
    fingers: tuple[FingerConfig, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _distinct(self) -> DexterousHandConfig:
        """대응이 겹치지 않는지, 손가락 끝 링크가 손목 링크가 아닌지. Raises: ValueError."""
        parts = [f.human_part for f in self.fingers]
        links = [f.tip_link for f in self.fingers]
        for name, values in (("human_part", parts), ("tip_link", links)):
            repeated = sorted({v for v in values if values.count(v) > 1})
            if repeated:
                raise ValueError(f"fingers의 {name} {repeated}가 중복입니다 (손가락마다 하나씩)")
        if self.wrist_link in links:
            raise ValueError(f"손가락 끝 링크가 손목 링크 {self.wrist_link}와 같습니다")
        return self


class DexterousArmConfig(Contract):
    """사람 손 하나를 따라가는 팔과 그 끝의 다지 손.

    joints: 기준 링크 → 손목 링크(`dexterous_hand.wrist_link`) 연쇄의 움직이는 관절 (같은
        순서·집합). 손가락 관절은 적지 않는다 (URDF에서 읽는다).
    끝 링크·접근 축·닫힘 축은 없다: 방향 목표를 따로 두지 않고 손목과 손가락 끝의 위치만 맞춘다.
    """

    hand: Hand
    joints: tuple[str, ...] = Field(min_length=1)
    placement: Placement
    dexterous_hand: DexterousHandConfig


class HandIkConfig(Contract):
    """다지 손 IK 설정 (로봇마다, `hand_ik.DlsHandSolver`).

    position_tolerance_m: 손목 위치 오차가 이 이하이고 목표가 있는 손가락 끝이 모두
        `dexterous_hand.fingertip_tolerance_m` 안일 때 그 프레임은 도달 가능이다.
    wrist_weight: 손목 위치 오차의 가중치 (손가락 끝 오차는 1). 손목과 손가락 끝을 함께 맞출 수
        없을 때 어느 쪽 오차를 줄일지 정한다 (1보다 작으면 손가락 끝을 먼저 맞춘다).
    smoothness_weight / damping / max_iterations / restarts / velocity_scale: 팔 IK(`IkConfig`)와
        같은 뜻이다. 손가락 링크는 수 cm라 손가락 관절의 야코비안이 팔 관절보다 한 자릿수 작다.
        damping·smoothness_weight를 팔만 있는 로봇의 값(1e-4, 1e-2)으로 두면 반복이
        max_iterations 안에 덜 수렴해 오차가 남으므로(시험 로봇에서 반복 100회에 최대 0.29 mm) 더
        작게 둔다 (시험 로봇은 1e-6, 1e-4, 반복 200회).
    방향 항목은 없다 (손바닥 방향 목표를 따로 두지 않는다).
    """

    position_tolerance_m: float = Field(gt=0)
    wrist_weight: float = Field(gt=0)
    smoothness_weight: float = Field(ge=0)
    damping: float = Field(gt=0)
    max_iterations: int = Field(ge=1)
    restarts: int = Field(ge=1)
    velocity_scale: float = Field(gt=0, le=1)


class DexterousRobotConfig(Contract):
    """`config/robots/<ID>.yaml` 전체 (다지 손 로봇: `arm.dexterous_hand`가 있는 설정)."""

    robot_id: Identifier
    display_name: str
    model: RobotModel
    base_link: str
    arm: DexterousArmConfig
    ik: HandIkConfig

    def digest(self) -> str:
        """설정 값 해시 (sha256 앞 12자, 리타기팅 버전에 들어간다)."""
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class LoadedRobot:
    """확인을 마친 로봇: 설정, 팔 연쇄, 그리퍼 관절, 모델 파일 해시와 경로."""

    config: RobotConfig
    chain: Chain
    gripper: Joint
    model_sha256: str
    model_path: Path | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        """내보내는 관절 순서: 팔 관절 + 그리퍼 관절."""
        return (*self.chain.names, self.gripper.name)


@dataclass(frozen=True)
class LoadedFinger:
    """확인을 마친 손가락: 대응 설정과 손목 링크 → 끝 링크 연쇄 (그 손가락의 관절)."""

    config: FingerConfig
    chain: Chain


@dataclass(frozen=True)
class LoadedDexterousRobot:
    """확인을 마친 다지 손 로봇.

    chain: 기준 링크 → 손목 링크 팔 연쇄. fingers: 대응 순서의 손가락. tree: 기준 링크 → [손목 링크,
    손가락 끝 링크...] 나무 (관절 값 순서 = 팔 관절 + 손가락 관절). unmapped_joints: 손목 링크
    아래의 움직이는 관절 중 어느 손가락 경로에도 없는 것 (풀지도 내보내지도 않는다).
    """

    config: DexterousRobotConfig
    chain: Chain
    fingers: tuple[LoadedFinger, ...]
    tree: Tree
    unmapped_joints: tuple[str, ...]
    model_sha256: str
    model_path: Path | None = None

    @property
    def joint_names(self) -> tuple[str, ...]:
        """내보내는 관절 순서: 팔 관절 + 손가락 관절 (대응 순서, 손가락마다 손목 쪽부터)."""
        return self.tree.names


# 내보내기·`dlp robot check`가 다루는 로봇 (평행 그리퍼 또는 다지 손)
AnyRobot = LoadedRobot | LoadedDexterousRobot


def sha256_file(path: Path) -> str:
    """파일 sha256 16진."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def robot_dirs(root: Path, env: dict[str, str] | None = None) -> list[Path]:
    """로봇 설정을 찾는 디렉터리: `DLP_ROBOTS_DIR`(있으면) → `<root>/config/robots`."""
    env = dict(os.environ) if env is None else env
    dirs = [Path(env[ROBOTS_DIR_ENV])] if env.get(ROBOTS_DIR_ENV) else []
    return [*dirs, root / "config" / "robots"]


def find_config(robot_id: str, dirs: Sequence[Path]) -> Path:
    """로봇 ID의 설정 파일 (`<ID>.yaml`, 먼저 찾은 디렉터리).

    설정 파일의 실제 위치(심볼릭 링크를 따라간 위치)가 그 디렉터리 바로 안이어야 한다.

    Raises:
        RobotError: ID 형식이 틀렸을 때(경로 구분자·`..`로 디렉터리를 벗어나지 못하게), 어느
            디렉터리에도 없을 때.
    """
    if not _ROBOT_ID_RE.match(robot_id):
        raise RobotError(f"로봇 ID {robot_id!r}: 영숫자로 시작하고 영숫자·_.:-만 씁니다")
    for d in dirs:
        path = d / f"{robot_id}.yaml"
        if path.is_file() and path.resolve().parent == d.resolve():
            return path
    raise RobotError(f"로봇 {robot_id}: 설정 파일이 없습니다 ({', '.join(str(d) for d in dirs)})")


def _arm_section(data: Any) -> dict[str, Any]:
    """설정 사전의 `arm` 절 (사전이 아니면 빈 사전). 로봇 종류를 가르는 데만 쓴다."""
    if not isinstance(data, dict):
        return {}
    arm: Any = cast(dict[str, Any], data).get("arm")
    return cast(dict[str, Any], arm) if isinstance(arm, dict) else {}


def read_any_config(path: Path) -> RobotConfig | DexterousRobotConfig:
    """설정 YAML → 평행 그리퍼(`RobotConfig`) 또는 다지 손(`DexterousRobotConfig`) 설정.

    `arm.dexterous_hand`가 있으면 다지 손 설정으로 읽는다. 파일 이름과 robot_id가 같아야 한다.

    Raises:
        RobotError: `arm.gripper`와 `arm.dexterous_hand`를 함께 적었을 때, ID가 파일 이름과 다를 때.
        pydantic.ValidationError: 값이 틀렸을 때.
    """
    data: Any = yaml.safe_load(path.read_text("utf-8"))
    arm = _arm_section(data)
    cfg: RobotConfig | DexterousRobotConfig
    if "dexterous_hand" in arm:
        if "gripper" in arm:
            raise RobotError(
                f"{path.name}: arm.gripper와 arm.dexterous_hand를 함께 쓸 수 없습니다 "
                "(평행 그리퍼 로봇이면 gripper, 다지 손 로봇이면 dexterous_hand만 적는다)"
            )
        cfg = DexterousRobotConfig.model_validate(data)
    else:
        cfg = RobotConfig.model_validate(data)
    if cfg.robot_id != path.stem:
        raise RobotError(f"{path.name}: robot_id {cfg.robot_id}가 파일 이름과 다릅니다")
    return cfg


def read_config(path: Path) -> RobotConfig:
    """설정 YAML → `RobotConfig` (평행 그리퍼 로봇만, 파일 이름과 robot_id가 같아야 한다).

    Raises:
        RobotError: ID가 파일 이름과 다를 때, 다지 손 설정일 때 (`read_any_config`로 읽는다).
        pydantic.ValidationError: 값이 틀렸을 때.
    """
    cfg = read_any_config(path)
    if isinstance(cfg, DexterousRobotConfig):
        raise RobotError(
            f"{path.name}: 다지 손 로봇 설정입니다 (load_any_robot·load_any_robot_file로 읽는다)"
        )
    return cfg


def _model_joints(
    cfg: RobotConfig | DexterousRobotConfig, path: Path, inside_repo: Path | None
) -> tuple[dict[str, Joint], str, Path]:
    """모델 파일 위치·공개 여부·sha256을 확인하고 관절을 읽는다.

    Returns:
        (관절 이름 → `Joint`, 모델 파일 sha256, 모델 파일 경로).

    Raises:
        RobotError: 저장소 안 설정이 공개 모델이 아닐 때, 비공개 모델 파일이 저장소 안에 있을 때,
            모델 파일이 없거나 sha256이 다를 때.
        UrdfError: URDF가 틀렸을 때.
    """
    in_repo = inside_repo is not None and path.resolve().is_relative_to(inside_repo.resolve())
    if in_repo and not cfg.model.public:
        raise RobotError(
            f"{cfg.robot_id}: 저장소 안에는 공개 시험 모델만 둡니다 (고객 모델은 {ROBOTS_DIR_ENV})"
        )
    model = Path(cfg.model.path)
    model = model if model.is_absolute() else path.parent / model
    if (
        inside_repo is not None
        and not cfg.model.public
        and model.resolve().is_relative_to(inside_repo.resolve())
    ):
        # 설정은 저장소 밖에 있어도 모델 파일이 저장소 작업 트리 안이면 커밋될 수 있다
        raise RobotError(
            f"{cfg.robot_id}: 고객 모델 파일은 저장소 밖({ROBOTS_DIR_ENV})에 둡니다: {model}"
        )
    if not model.is_file():
        raise RobotError(f"{cfg.robot_id}: 모델 파일이 없습니다: {model}")
    digest = sha256_file(model)
    if digest != cfg.model.sha256:
        raise RobotError(
            f"{cfg.robot_id}: 모델 파일 sha256이 다릅니다 ({digest} != {cfg.model.sha256})"
        )
    return load_joints(model), digest, model


def _parallel(cfg: RobotConfig, joints: dict[str, Joint], digest: str, model: Path) -> LoadedRobot:
    """평행 그리퍼 로봇의 팔 연쇄와 그리퍼 관절을 확인한다.

    Raises:
        RobotError: 관절 목록이 연쇄와 다를 때, 그리퍼 관절이 없거나 맞지 않을 때
            (`_check_gripper`). UrdfError: 연쇄를 만들 수 없을 때.
    """
    chain = build_chain(joints, cfg.base_link, cfg.arm.tip_link)
    if chain.names != cfg.arm.joints:
        raise RobotError(
            f"{cfg.robot_id}: arm.joints {list(cfg.arm.joints)}가 {cfg.base_link} → "
            f"{cfg.arm.tip_link} 연쇄의 움직이는 관절 {list(chain.names)}와 다릅니다"
        )
    grip = _check_gripper(cfg, joints, chain)
    return LoadedRobot(cfg, chain, grip, digest, model)


def load_robot_file(path: Path, *, inside_repo: Path | None = None) -> LoadedRobot:
    """평행 그리퍼 로봇의 설정 파일 → 모델 해시 확인 → 연쇄 구성.

    Args:
        path: 설정 YAML.
        inside_repo: 저장소 루트. 설정이 그 안에 있으면 공개 모델(`model.public`)만 허용한다.

    Raises:
        RobotError: 모델 파일이 없거나 sha256이 다를 때, 저장소 안 설정이 공개 모델이 아닐 때,
            비공개 모델 파일이 저장소 안에 있을 때, 관절 목록이 연쇄와 다를 때, 그리퍼 관절이
            없거나 맞지 않을 때(`_check_gripper`), 파일 이름과 robot_id가 다를 때, 다지 손 설정일
            때(`read_config`).
        UrdfError: URDF가 틀렸을 때.
        pydantic.ValidationError: 설정 값이 틀렸을 때.
    """
    cfg = read_config(path)
    joints, digest, model = _model_joints(cfg, path, inside_repo)
    return _parallel(cfg, joints, digest, model)


def load_any_robot_file(path: Path, *, inside_repo: Path | None = None) -> AnyRobot:
    """설정 파일 → 확인된 로봇 (평행 그리퍼 `LoadedRobot` 또는 다지 손 `LoadedDexterousRobot`).

    인자와 모델 파일 확인은 `load_robot_file`과 같다.

    Raises:
        RobotError: `load_robot_file`의 경우에 더해, 다지 손 로봇의 링크가 URDF에 없을 때, 손가락
            끝 링크가 손목 링크 아래에 있지 않을 때, 손가락 경로에 mimic·지원하지 않는 관절이
            있거나 움직이는 관절이 없을 때, 손가락끼리 관절을 함께 쓸 때 (`_dexterous`).
        UrdfError: URDF가 틀렸을 때.
        pydantic.ValidationError: 설정 값이 틀렸을 때.
    """
    cfg = read_any_config(path)
    joints, digest, model = _model_joints(cfg, path, inside_repo)
    if isinstance(cfg, DexterousRobotConfig):
        return _dexterous(cfg, joints, digest, model)
    return _parallel(cfg, joints, digest, model)


def _check_gripper(cfg: RobotConfig, joints: dict[str, Joint], chain: Chain) -> Joint:
    """그리퍼 관절: 있고, 움직이는 독립 관절이고, 팔 연쇄 관절이 아니고, closed·open이 URDF 한계 안.

    도 단위로 적은 값(open: 85)이나 고정 관절, 팔 관절과 같은 이름(내보내는 관절 이름이 겹친다)을
    막는다.

    Raises:
        RobotError: 위 조건을 어길 때.
    """
    g = cfg.arm.gripper
    grip = joints.get(g.joint)
    if grip is None:
        raise RobotError(f"{cfg.robot_id}: 그리퍼 관절 {g.joint}가 없습니다")
    if grip.kind not in MOVING or grip.mimic is not None:
        raise RobotError(f"{cfg.robot_id}: 그리퍼 관절 {g.joint}는 움직이는 독립 관절이어야 합니다")
    if grip.name in chain.names:
        raise RobotError(f"{cfg.robot_id}: 그리퍼 관절 {g.joint}가 팔 관절과 겹칩니다")
    lo, hi = min(g.closed, g.open), max(g.closed, g.open)
    if lo < grip.lower or hi > grip.upper:
        raise RobotError(
            f"{cfg.robot_id}: 그리퍼 closed·open [{g.closed}, {g.open}]이 URDF 한계 "
            f"[{grip.lower}, {grip.upper}] 밖입니다 (단위는 rad 또는 m)"
        )
    return grip


def _dexterous(
    cfg: DexterousRobotConfig, joints: dict[str, Joint], digest: str, model: Path
) -> LoadedDexterousRobot:
    """다지 손 로봇의 팔 연쇄·손가락 연쇄·나무를 만들고 구조를 확인한다.

    Raises:
        RobotError: 설정의 링크가 URDF에 없을 때, arm.joints가 기준 → 손목 연쇄와 다를 때, 손가락
            구조가 맞지 않을 때 (`_finger`, `_check_separate`).
        UrdfError: 팔 연쇄를 만들 수 없을 때 (mimic·지원하지 않는 관절 등).
    """
    hand = cfg.arm.dexterous_hand
    links = {name for j in joints.values() for name in (j.parent, j.child)}
    named = [("base_link", cfg.base_link), ("arm.dexterous_hand.wrist_link", hand.wrist_link)]
    named += [(f"손가락 {f.human_part}의 tip_link", f.tip_link) for f in hand.fingers]
    for field, link in named:
        if link not in links:
            raise RobotError(f"{cfg.robot_id}: {field} {link} 링크가 URDF에 없습니다")
    chain = build_chain(joints, cfg.base_link, hand.wrist_link)
    if chain.names != cfg.arm.joints:
        raise RobotError(
            f"{cfg.robot_id}: arm.joints {list(cfg.arm.joints)}가 {cfg.base_link} → "
            f"{hand.wrist_link} 연쇄의 움직이는 관절 {list(chain.names)}와 다릅니다"
        )
    fingers = tuple(_finger(cfg, joints, f) for f in hand.fingers)
    _check_separate(cfg, fingers)
    tree = build_tree(joints, cfg.base_link, (hand.wrist_link, *(f.tip_link for f in hand.fingers)))
    used = {name for f in fingers for name in f.chain.names}
    unmapped = tuple(n for n in _moving_below(joints, hand.wrist_link) if n not in used)
    return LoadedDexterousRobot(cfg, chain, fingers, tree, unmapped, digest, model)


def _finger(
    cfg: DexterousRobotConfig, joints: dict[str, Joint], finger: FingerConfig
) -> LoadedFinger:
    """손가락 하나의 손목 링크 → 끝 링크 연쇄.

    Raises:
        RobotError: 끝 링크가 손목 링크 아래에 있지 않을 때, 경로에 mimic·지원하지 않는 관절이
            있을 때 (이유는 `build_chain`의 것을 그대로 싣는다), 움직이는 관절이 없을 때.
    """
    wrist = cfg.arm.dexterous_hand.wrist_link
    where = f"{cfg.robot_id}: 손가락 {finger.human_part}(끝 링크 {finger.tip_link})"
    if wrist not in _links_above(joints, finger.tip_link):
        raise RobotError(f"{where}가 손목 링크 {wrist} 아래에 있지 않습니다")
    try:
        chain = build_chain(joints, wrist, finger.tip_link)
    except UrdfError as exc:
        raise RobotError(f"{where}: {exc}") from exc
    if not chain.moving:
        raise RobotError(f"{where}: 손목 링크에서 끝 링크까지 움직이는 관절이 없습니다")
    return LoadedFinger(finger, chain)


def _check_separate(cfg: DexterousRobotConfig, fingers: Sequence[LoadedFinger]) -> None:
    """손가락마다 움직이는 관절이 따로인지 (고정 관절은 함께 써도 된다).

    손가락끼리 관절을 함께 쓰면 손가락 끝 점이 없는 프레임에 그 손가락 관절만 유지할 수 없다.

    Raises:
        RobotError: 두 손가락이 같은 움직이는 관절을 쓸 때.
    """
    owner: dict[str, str] = {}
    for f in fingers:
        for name in f.chain.names:
            if name in owner:
                raise RobotError(
                    f"{cfg.robot_id}: 손가락 {owner[name]}와 {f.config.human_part}가 관절 {name}를 "
                    "함께 씁니다 (손가락마다 관절이 따로인 손만 지원합니다)"
                )
            owner[name] = f.config.human_part


def _links_above(joints: dict[str, Joint], link: str) -> set[str]:
    """그 링크에서 뿌리까지 올라가며 만나는 링크 (자기 자신은 뺀다)."""
    by_child = {j.child: j for j in joints.values()}
    above: set[str] = set()
    while link in by_child and by_child[link].parent not in above:
        link = by_child[link].parent
        above.add(link)
    return above


def _moving_below(joints: dict[str, Joint], link: str) -> list[str]:
    """그 링크 아래(자손 링크까지)에 달린 움직이는 관절 이름 (URDF에 적힌 순서)."""
    below = {link}
    grew = True
    while grew:
        grew = False
        for j in joints.values():
            if j.parent in below and j.child not in below:
                below.add(j.child)
                grew = True
    return [j.name for j in joints.values() if j.parent in below and j.kind in MOVING]


def load_robot(root: Path, robot_id: str, env: dict[str, str] | None = None) -> LoadedRobot:
    """로봇 ID → 확인된 평행 그리퍼 로봇 (`robot_dirs` 순으로 찾는다, 다지 손 설정이면 거부)."""
    return load_robot_file(find_config(robot_id, robot_dirs(root, env)), inside_repo=root)


def load_any_robot(root: Path, robot_id: str, env: dict[str, str] | None = None) -> AnyRobot:
    """로봇 ID → 확인된 로봇 (평행 그리퍼 또는 다지 손, `robot_dirs` 순으로 찾는다)."""
    return load_any_robot_file(find_config(robot_id, robot_dirs(root, env)), inside_repo=root)
