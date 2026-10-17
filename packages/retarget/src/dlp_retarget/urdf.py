"""URDF 직렬 연쇄 기구학 (정기구학·야코비안, ADR 0047).

URDF(로봇 설명 XML)에서 관절(종류, 부모·자식 링크, origin xyz·rpy, axis, limit)만 읽는다. 메시·관성·
충돌 모양은 읽지 않는다 (IK에 필요 없다, 메시 파일이 없어도 된다). 기준 링크에서 끝 링크까지의 직렬
연쇄(`Chain`)를 만들어 관절 값 q로 끝 링크 자세와 관절 축(기준 좌표)을 계산한다.

지원 관절: revolute, continuous(한계 없음), prismatic, fixed. floating·planar·mimic은 연쇄에 있으면
거부한다 (`UrdfError`).

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
    lower/upper: 위치 한계 (rad 또는 m, continuous는 ±inf).
    velocity: 속도 한계 (rad/s 또는 m/s, 없으면 inf).
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
    lim = el.find("limit")
    lower, upper, vel = -math.inf, math.inf, math.inf
    if lim is not None and kind != "continuous":
        lower = float(lim.get("lower", "-inf"))
        upper = float(lim.get("upper", "inf"))
    if lim is not None and lim.get("velocity") is not None:
        vel = float(lim.get("velocity") or "inf")
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
        """(끝 자세, 기하 야코비안 (6, dof): 위 3행 선속도, 아래 3행 각속도)."""
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
    """단위 축·각 → 회전 행렬 (로드리게스)."""
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def build_chain(
    joints: dict[str, Joint], base: str, tip: str, tip_offset: Mat4 | None = None
) -> Chain:
    """기준 링크 → 끝 링크 연쇄를 만든다 (끝에서 부모를 따라 올라간다).

    Raises:
        UrdfError: 끝 링크에서 기준 링크로 이어지지 않을 때, 지원하지 않는 관절이 연쇄에 있을 때.
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
        path.append(j)
        link = j.parent
        if len(path) > len(joints):
            raise UrdfError("관절 연쇄에 순환이 있습니다")
    offset = np.eye(4) if tip_offset is None else tip_offset
    return Chain(base, tip, tuple(reversed(path)), offset)
