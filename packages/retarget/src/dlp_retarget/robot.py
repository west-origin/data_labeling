"""로봇 등록: `config/robots/<ID>.yaml` 로더, 모델 파일 sha256 확인 (ADR 0047).

로봇 설정 파일 하나가 로봇 하나다. 고객 로봇의 설계 파일(URDF)과 설정은 저장소에 넣지 않는다:
`DLP_ROBOTS_DIR` 환경 변수의 디렉터리(저장소 밖)를 먼저 찾고, 없으면 저장소 `config/robots/`(공개
시험 모델만, `model.public: true`)를 찾는다. 모델 경로는 설정 파일 디렉터리 기준 상대 경로나 절대
경로다. 모델 파일은 읽기 전에 sha256을 확인한다 (다른 파일이면 거부, 내보내기 manifest·버전에 해시가
들어간다).

설정 항목 (`RobotConfig`):
- robot_id, display_name, model(format urdf, path, sha256, license, source, public).
- base_link: 로봇 기준 링크. arm: 사람 손 하나를 따라가는 팔 (손, 끝 링크, 움직일 관절 순서, 끝 링크
  좌표의 접근 축·닫힘 축, 그리퍼, 배치). ik: 허용 오차·가중치·반복 수.

지금은 팔 하나 + 평행 그리퍼만 지원한다. 다지 손(dexterous)은 설정에서 거부한다
(dex-retargeting 미연동, F-COE-17).
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
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_retarget.urdf import MOVING, Chain, Joint, build_chain, load_joints
from dlp_schema.common import Contract, Identifier
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
    PARALLEL = "parallel"  # 평행 그리퍼: 엄지-검지 끝 거리 → 그리퍼 관절 하나
    DEXTEROUS = "dexterous"  # 다지 손 (아직 지원하지 않는다)


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
        """다지 손 거부, 거리 범위 순서 검사. Raises: ValueError."""
        if self.type is GripperType.DEXTEROUS:
            raise ValueError("다지 손(dexterous) 리타기팅은 아직 지원하지 않습니다 (F-COE-17)")
        if self.open_distance_m <= self.closed_distance_m:
            raise ValueError("open_distance_m은 closed_distance_m보다 커야 합니다")
        return self


class PlacementMode(StrEnum):
    # 로봇 기준을 세션 월드 좌표의 고정 자세에 둔다 (base_xyz, base_yaw_rad)
    WORLD_FIXED = "world_fixed"
    # 에피소드 손 잡기 점 평균이 로봇 기준 좌표 workspace_center에 오도록 옮긴다 (요는 월드 X = 로봇
    # X)
    EPISODE_CENTROID = "episode_centroid"


class Placement(Contract):
    """사람 손 → 로봇 기준 좌표 배치.

    position_scale: 위치 배율 (사람 팔 길이 대비 로봇, 1이면 그대로). world_fixed는 로봇 기준 원점,
    episode_centroid는 손 잡기 점 평균을 중심으로 줄인다. 방향(접근·닫힘 축)은 배율과 무관하다.
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

    position_tolerance_m / orientation_tolerance_rad: 이 안이면 그 프레임은 도달 가능.
    orientation_weight: 회전 오차 1 rad를 위치 몇 m로 볼지. smoothness_weight: 반복 걸음을 줄이는
    추가 감쇠 (직전 해에서 시작하므로 가까운 자세 가지로 수렴). damping: 감쇠 최소제곱 λ.
    max_iterations:
    프레임당 반복 수.
    restarts: 첫 프레임·끊긴 뒤 다시 시작할 때 쓰는 시작점 수 (가운데 + 결정적 난수).
    velocity_scale: URDF 속도 한계에 곱하는 값 (보수적으로 낮춘다).
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
    """`config/robots/<ID>.yaml` 전체."""

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

    Raises:
        RobotError: ID 형식이 틀렸을 때(경로 구분자·`..`로 디렉터리를 벗어나지 못하게), 어느
        디렉터리에도
            없을 때.
    """
    if not _ROBOT_ID_RE.match(robot_id):
        raise RobotError(f"로봇 ID {robot_id!r}: 영숫자로 시작하고 영숫자·_.:-만 씁니다")
    for d in dirs:
        path = d / f"{robot_id}.yaml"
        if path.is_file() and path.resolve().parent == d.resolve():
            return path
    raise RobotError(f"로봇 {robot_id}: 설정 파일이 없습니다 ({', '.join(str(d) for d in dirs)})")


def read_config(path: Path) -> RobotConfig:
    """설정 YAML → `RobotConfig` (파일 이름과 robot_id가 같아야 한다).

    Raises:
        RobotError: ID가 파일 이름과 다를 때. pydantic.ValidationError: 값이 틀렸을 때.
    """
    data: Any = yaml.safe_load(path.read_text("utf-8"))
    cfg = RobotConfig.model_validate(data)
    if cfg.robot_id != path.stem:
        raise RobotError(f"{path.name}: robot_id {cfg.robot_id}가 파일 이름과 다릅니다")
    return cfg


def load_robot_file(path: Path, *, inside_repo: Path | None = None) -> LoadedRobot:
    """설정 파일 → 모델 해시 확인 → 연쇄 구성.

    Args:
        path: 설정 YAML.
        inside_repo: 저장소 루트. 설정이 그 안에 있으면 공개 모델(`model.public`)만 허용한다.

    Raises:
        RobotError: 모델 파일이 없거나 sha256이 다를 때, 저장소 안 설정이 공개 모델이 아닐 때,
        비공개
            모델 파일이 저장소 안에 있을 때, 관절 목록이 연쇄와 다를 때, 그리퍼 관절이 없거나 맞지
            않을 때(`_check_gripper`). UrdfError: URDF가 틀렸을 때.
    """
    cfg = read_config(path)
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
    joints = load_joints(model)
    chain = build_chain(joints, cfg.base_link, cfg.arm.tip_link)
    if chain.names != cfg.arm.joints:
        raise RobotError(
            f"{cfg.robot_id}: arm.joints {list(cfg.arm.joints)}가 {cfg.base_link} → "
            f"{cfg.arm.tip_link} 연쇄의 움직이는 관절 {list(chain.names)}와 다릅니다"
        )
    grip = _check_gripper(cfg, joints, chain)
    return LoadedRobot(cfg, chain, grip, digest, model)


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


def load_robot(root: Path, robot_id: str, env: dict[str, str] | None = None) -> LoadedRobot:
    """로봇 ID → 확인된 로봇 (`robot_dirs` 순으로 찾는다)."""
    return load_robot_file(find_config(robot_id, robot_dirs(root, env)), inside_repo=root)
