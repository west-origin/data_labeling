"""URDF 직렬 연쇄 기구학 (정기구학·야코비안, ADR 0047).

URDF(로봇 설명 XML)에서 관절(종류, 부모·자식 링크, origin xyz·rpy, axis, limit)만 읽는다. 메시·관성·
충돌 모양은 읽지 않는다 (IK에 필요 없다, 메시 파일이 없어도 된다). 기준 링크에서 끝 링크까지의 직렬
연쇄(`Chain`)를 만들어 관절 값 q로 끝 링크 자세와 관절 축(기준 좌표)을 계산한다.

지원 관절: revolute, continuous(한계 없음), prismatic, fixed. floating·planar 관절과 `<mimic>`(다른
관절 값을 따라 움직이는 종속 관절)은 연쇄에 있으면 거부한다 (`UrdfError`). mimic 관절을 독립
자유도로 풀면 실제 로봇이 낼 수 없는 관절 조합이 나오기 때문이다. 연쇄 밖 mimic(그리퍼 손가락 등)은
읽기만 한다.

관절 한계: revolute·prismatic은 `<limit lower upper>`가 있어야 한다 (없으면 거부, urdfdom과 같다.
URDF 규격의 기본값 0을 쓰면 관절이 사실상 고정되고, 무한으로 보면 실제보다 넓은 해가 나온다).
continuous만 위치 한계가 없다 (±inf). 속도 한계(velocity)가 없으면 inf다.

좌표 규약 (URDF와 같다):
- 관절 j의 `origin`(xyz, rpy)은 부모 링크 좌표 → 관절 좌표 변환이다 (q = 0일 때). 관절 좌표의 원점이
  관절 위치이고, 자식 링크 좌표는 관절 좌표에 관절 운동(축 둘레 회전 또는 축 방향 이동)을 더한
  것이다.
- `axis`는 관절 좌표에서 본 단위 축이다. revolute·continuous는 오른손 법칙 회전, prismatic은 이동
  방향.
- 정기구학: 기준 링크에서 T = Π_j (origin_j · motion_j(q_j)) 를 차례로 곱하고 끝 링크 안의 추가
  변환(tip_offset)을 곱한다. 결과는 기준 링크 좌표의 끝 링크 자세(4x4)다.

순수 Python + numpy다 (DB·파일 쓰기 없음). 무거운 기구학 라이브러리(Pinocchio, MuJoCo)는 쓰지 않는다
(MuJoCo는 격리 환경의 mink 풀이기에서만, `dlp_retarget.mink`).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Mat4 = NDArray[np.float64]
Vec3 = NDArray[np.float64]
# 지원하는 관절 종류
MOVING = frozenset({"revolute", "continuous", "prismatic"})
SUPPORTED = MOVING | {"fixed"}


class UrdfError(ValueError):
    """URDF를 읽을 수 없거나 연쇄를 만들 수 없을 때."""


def rpy_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """URDF rpy (고정축 x → y → z) → 회전 행렬 R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def transform(rotation: NDArray[np.float64], translation: Vec3) -> Mat4:
    """회전·이동 → 4x4 동차 변환."""
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, translation
    return t


def _floats(text: str | None, default: tuple[float, float, float]) -> Vec3:
    """'x y z' 문자열 → 벡터 (없으면 기본값)."""
    if not text:
        return np.array(default, dtype=np.float64)
    vals = [float(v) for v in text.split()]
    if len(vals) != 3:
        raise UrdfError(f"숫자 3개가 아닙니다: {text!r}")
    return np.array(vals, dtype=np.float64)


@dataclass(frozen=True)
class Joint:
    """URDF 관절 하나.

    origin: 부모 링크 → 관절(자식 링크) 좌표 변환 (q = 0). axis: 관절 좌표 단위 축.
    lower/upper: 위치 한계 (rad 또는 m, continuous는 ±inf, fixed는 의미 없음).
    velocity: 속도 한계 (rad/s 또는 m/s, 없으면 inf).
    mimic: `<mimic joint=...>`가 가리키는 원 관절 이름 (종속 관절이 아니면 None).
    """

    name: str
    kind: str
    parent: str
    child: str
    origin: Mat4
    axis: Vec3
    lower: float
    upper: float
    velocity: float
    mimic: str | None = None


def _limits(name: str, kind: str, lim: ET.Element | None) -> tuple[float, float, float]:
    """(하한, 상한, 속도 한계) — 모듈 설명의 관절 한계 규칙.

    Raises:
        UrdfError: revolute·prismatic에 `<limit>`이나 lower·upper가 없을 때, 하한 > 상한일 때.
    """
    vel = math.inf
    if lim is not None and lim.get("velocity") is not None:
        vel = float(lim.get("velocity") or "inf")
    if kind not in ("revolute", "prismatic"):
        # continuous는 위치 한계가 없고, fixed는 움직이지 않으므로 한계를 쓰지 않는다
        return -math.inf, math.inf, vel
    if lim is None or lim.get("lower") is None or lim.get("upper") is None:
        raise UrdfError(f"{name}: {kind} 관절에 <limit lower upper>가 없습니다")
    lower, upper = float(lim.get("lower") or "nan"), float(lim.get("upper") or "nan")
    if not lower <= upper:
        raise UrdfError(f"{name}: 관절 하한 {lower}이 상한 {upper}보다 큽니다")
    return lower, upper, vel


def _joint(el: ET.Element) -> Joint:
    """<joint> 요소 → `Joint`."""
    name = el.get("name") or ""
    kind = el.get("type") or ""
    parent, child = el.find("parent"), el.find("child")
    if parent is None or child is None:
        raise UrdfError(f"{name}: parent·child가 없습니다")
    o = el.find("origin")
    xyz = _floats(o.get("xyz") if o is not None else None, (0.0, 0.0, 0.0))
    rpy = _floats(o.get("rpy") if o is not None else None, (0.0, 0.0, 0.0))
    a = el.find("axis")
    axis = _floats(a.get("xyz") if a is not None else None, (1.0, 0.0, 0.0))
    if kind in MOVING:
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            raise UrdfError(f"{name}: 관절 축이 0입니다")
        axis = axis / norm
    lower, upper, vel = _limits(name, kind, el.find("limit"))
    mimic = el.find("mimic")
    return Joint(
        name=name,
        kind=kind,
        parent=parent.get("link") or "",
        child=child.get("link") or "",
        origin=transform(rpy_matrix(*rpy), xyz),
        axis=axis,
        lower=lower,
        upper=upper,
        velocity=vel,
        mimic=(mimic.get("joint") or "?") if mimic is not None else None,
    )


def load_joints(path: Path) -> dict[str, Joint]:
    """URDF 파일 → 관절 이름 → `Joint`.

    Raises:
        UrdfError: XML이 아니거나 <robot>이 아니거나 관절 정의가 틀렸을 때.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise UrdfError(f"{path.name}: URDF XML을 읽을 수 없습니다: {exc}") from exc
    if root.tag != "robot":
        raise UrdfError(f"{path.name}: <robot> 요소가 아닙니다")
    return {j.name: j for j in (_joint(el) for el in root.findall("joint"))}


@dataclass(frozen=True)
class Frames:
    """정기구학 결과: 끝 링크 자세 (4x4, 기준 좌표), 움직이는 관절마다 (위치, 축) (기준 좌표)."""

    tip: Mat4
    origins: NDArray[np.float64]  # (dof, 3)
    axes: NDArray[np.float64]  # (dof, 3)


@dataclass(frozen=True)
class Chain:
    """기준 링크 → 끝 링크 직렬 연쇄.

    joints는 기준 쪽부터 (고정 관절 포함). tip_offset: 끝 링크 안의 추가 변환 (도구 중심점, 기본
    단위 행렬).
    """

    base: str
    tip: str
    joints: tuple[Joint, ...]
    tip_offset: Mat4

    @property
    def moving(self) -> tuple[Joint, ...]:
        """움직이는 관절 (q 순서)."""
        return tuple(j for j in self.joints if j.kind in MOVING)

    @property
    def names(self) -> tuple[str, ...]:
        """움직이는 관절 이름 (q 순서)."""
        return tuple(j.name for j in self.moving)

    @property
    def lower(self) -> NDArray[np.float64]:
        """관절 하한 (q 순서)."""
        return np.array([j.lower for j in self.moving])

    @property
    def upper(self) -> NDArray[np.float64]:
        """관절 상한 (q 순서)."""
        return np.array([j.upper for j in self.moving])

    @property
    def velocity(self) -> NDArray[np.float64]:
        """관절 속도 한계 (q 순서)."""
        return np.array([j.velocity for j in self.moving])

    def frames(self, q: NDArray[np.float64]) -> Frames:
        """관절 값 q (움직이는 관절 순)로 정기구학.

        관절을 기준 쪽부터 따라가며 T ← T · origin 을 곱하면 T는 그 관절 좌표(기준 링크 기준)가
        된다. 이때 관절 축을 기준 좌표로 돌려 두고(야코비안에 쓴다), 관절 운동(축 둘레 q 회전 또는
        축 방향 q 이동)을 곱해 자식 링크 좌표로 넘어간다.

        Raises:
            ValueError: q 길이가 움직이는 관절 수와 다를 때.
        """
        if q.shape != (len(self.moving),):
            raise ValueError(f"관절 값 {q.shape}: 움직이는 관절 {len(self.moving)}개와 다릅니다")
        t = np.eye(4)
        origins: list[Vec3] = []
        axes: list[Vec3] = []
        i = 0
        for j in self.joints:
            t = t @ j.origin
            if j.kind in MOVING:
                axis_b = t[:3, :3] @ j.axis
                origins.append(t[:3, 3].copy())
                axes.append(axis_b)
                if j.kind == "prismatic":
                    t = t @ transform(np.eye(3), j.axis * q[i])
                else:
                    t = t @ transform(_axis_angle(j.axis, float(q[i])), np.zeros(3))
                i += 1
        return Frames(t @ self.tip_offset, np.array(origins), np.array(axes))

    def fk(self, q: NDArray[np.float64]) -> Mat4:
        """끝(도구 중심점) 자세 4x4 (기준 링크 좌표)."""
        return self.frames(q).tip

    def jacobian(self, q: NDArray[np.float64]) -> tuple[Mat4, NDArray[np.float64]]:
        """(끝 자세, 기하 야코비안 (6, dof): 위 3행 선속도, 아래 3행 각속도).

        기준 좌표에서 본 끝(도구 중심점) 속도 = J · q̇. 관절 k의 열은
        - 회전 관절: 선속도 = 축 x (끝 위치 - 관절 위치), 각속도 = 축 (축 둘레로 돌면 끝이 원을
          그린다)
        - 직선 관절: 선속도 = 축, 각속도 = 0
        이다. 축·위치는 `frames`가 관절 운동을 곱하기 직전(관절 좌표) 값이라 그 관절의 q와 무관하다.
        """
        f = self.frames(q)
        p = f.tip[:3, 3]
        jac = np.zeros((6, len(self.moving)))
        for k, j in enumerate(self.moving):
            if j.kind == "prismatic":
                jac[:3, k] = f.axes[k]
            else:
                jac[:3, k] = np.cross(f.axes[k], p - f.origins[k])
                jac[3:, k] = f.axes[k]
        return f.tip, jac


def _axis_angle(axis: Vec3, angle: float) -> NDArray[np.float64]:
    """단위 축·각 → 회전 행렬 (로드리게스: R = I + sinθ K + (1 - cosθ) K², K = 축의 외적 행렬)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def build_chain(
    joints: dict[str, Joint], base: str, tip: str, tip_offset: Mat4 | None = None
) -> Chain:
    """기준 링크 → 끝 링크 연쇄를 만든다 (끝에서 부모를 따라 올라간다).

    Raises:
        UrdfError: 끝 링크에서 기준 링크로 이어지지 않을 때, 지원하지 않는 관절(floating·planar)이나
            mimic 관절이 연쇄에 있을 때.
    """
    by_child = {j.child: j for j in joints.values()}
    path: list[Joint] = []
    link = tip
    while link != base:
        j = by_child.get(link)
        if j is None:
            raise UrdfError(f"{tip}에서 {base}로 이어지는 관절이 없습니다 ({link}에서 끊김)")
        if j.kind not in SUPPORTED:
            raise UrdfError(f"{j.name}: 지원하지 않는 관절 종류 {j.kind}")
        if j.mimic is not None:
            # 종속 관절을 독립 자유도로 풀면 결합 조건(값 = 원 관절 x 배율 + 오프셋)을 어긴다
            raise UrdfError(f"{j.name}: mimic 관절({j.mimic} 종속)은 팔 연쇄에 둘 수 없습니다")
        path.append(j)
        link = j.parent
        if len(path) > len(joints):
            raise UrdfError("관절 연쇄에 순환이 있습니다")
    offset = np.eye(4) if tip_offset is None else tip_offset
    return Chain(base, tip, tuple(reversed(path)), offset)
