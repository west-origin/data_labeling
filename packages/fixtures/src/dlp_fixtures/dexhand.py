"""직접 쓴 시험용 다지 손 로봇(6관절 팔 + 손바닥 + 다섯 손가락 x 3관절)과 그 설정 (ADR 0068).

실제 로봇이 아니다. 다지 손 리타기팅(`dlp_retarget.hand_ik`)과 내보내기 시험이 정답을 알 수 있게
이 저장소가 치수와 관절을 정해 쓴 URDF다 (남의 모델 파일을 넣지 않는다). 시험은 이 로봇의
정기구학으로 손목·손가락 끝 위치를 만들고, 배치와 손 배율을 거꾸로 써서 사람 손 점으로 바꾼 뒤,
리타기팅 결과가 그 위치로 돌아오는지 본다.

로봇 (길이 m, 각 rad, 영점 자세에서 팔과 손가락이 +z로 곧게 선다):
- 팔 6관절 (`ARM_JOINTS`): 어깨 요(z)·어깨 피치(y)·팔꿈치 피치(y)·아래팔 롤(z)·손목 피치(y)·손목
  요(x). 끝 링크 `hand`(`WRIST_LINK`)의 원점이 손목이다.
- `hand` → 고정 관절 → `palm`. 손가락 다섯(`FINGERS`)이 `palm`에 달린다.
- 손가락마다 3관절: `<손가락>_abduction`(x축, 손바닥 평면 안에서 벌림) → `<손가락>_flexion`(y축,
  손바닥 쪽으로 굽힘) → `<손가락>_curl`(y축, 가운데 마디 굽힘). 끝 링크 `<손가락>_tip`은 고정
  관절로 달린다. 엄지는 같은 구조를 손바닥 옆에 비스듬히 단 것이다.
- 손바닥 좌표: z = 손가락이 뻗는 방향, x = 손바닥이 향하는 쪽(굽히면 손가락 끝이 +x로 간다),
  y = 엄지 쪽.

`__init__`에서 내보내지 않는다 (`from dlp_fixtures.dexhand import ...`). 파일은 `write_robot`이 주는
디렉터리에만 쓴다 (저장소의 `config/robots`에는 넣지 않는다).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# 기준 링크와 사람 손목에 대응하는 링크
BASE_LINK = "base_link"
WRIST_LINK = "hand"
PALM_LINK = "palm"
# 팔 관절: (이름, 부모 링크, 자식 링크, origin xyz, 축, 하한, 상한). 하한·상한의 가운데가 팔꿈치를
# 굽힌 자세라 관절 범위 가운데 둘레의 궤적이 특이 자세(곧게 편 팔)를 지나지 않는다
_ARM = (
    ("shoulder_yaw", BASE_LINK, "shoulder", (0.0, 0.0, 0.12), (0, 0, 1), -2.0, 2.0),
    ("shoulder_pitch", "shoulder", "upper_arm", (0.0, 0.0, 0.04), (0, 1, 0), -0.3, 1.9),
    ("elbow_pitch", "upper_arm", "forearm", (0.0, 0.0, 0.28), (0, 1, 0), 0.1, 2.3),
    ("forearm_roll", "forearm", "wrist_roll_link", (0.0, 0.0, 0.12), (0, 0, 1), -2.0, 2.0),
    ("wrist_pitch", "wrist_roll_link", "wrist_pitch_link", (0.0, 0.0, 0.13), (0, 1, 0), -1.5, 1.5),
    ("wrist_yaw", "wrist_pitch_link", WRIST_LINK, (0.0, 0.0, 0.03), (1, 0, 0), -1.0, 1.0),
)
ARM_JOINTS = tuple(j[0] for j in _ARM)
# 손가락: 이름 → (손바닥 좌표의 뿌리 위치, 뿌리 rpy, 첫 마디 길이, 끝 마디 길이)
_FINGERS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], float, float]] = {
    "thumb": ((0.012, 0.030, 0.020), (-0.9, 0.0, 0.0), 0.040, 0.032),
    "index": ((0.0, 0.030, 0.085), (0.0, 0.0, 0.0), 0.045, 0.035),
    "middle": ((0.0, 0.010, 0.090), (0.0, 0.0, 0.0), 0.050, 0.038),
    "ring": ((0.0, -0.010, 0.085), (0.0, 0.0, 0.0), 0.045, 0.035),
    "pinky": ((0.0, -0.030, 0.075), (0.0, 0.0, 0.0), 0.036, 0.030),
}
FINGERS = tuple(_FINGERS)
# 손가락 관절의 접미사와 (축, 하한, 상한)
_FINGER_JOINTS = (
    ("abduction", (1, 0, 0), -0.4, 0.4),
    ("flexion", (0, 1, 0), -0.2, 1.5),
    ("curl", (0, 1, 0), 0.0, 1.6),
)
# 관절 속도 한계 (rad/s)
ARM_VELOCITY = 4.0
FINGER_VELOCITY = 8.0
# hand → palm 고정 관절의 이동 (m)
_PALM_OFFSET = (0.0, 0.0, 0.02)
# 시험 로봇 손가락 → 사람 손가락 끝의 3D 궤적 부위 (온톨로지 hand_joints)
HUMAN_PART = {
    "thumb": "thumb_tip",
    "index": "index_tip",
    "middle": "middle_tip",
    "ring": "ring_tip",
    "pinky": "pinky_tip",
}


def finger_joints(finger: str) -> tuple[str, ...]:
    """그 손가락의 관절 이름 (손바닥 쪽부터)."""
    return tuple(f"{finger}_{suffix}" for suffix, _, _, _ in _FINGER_JOINTS)


def tip_link(finger: str) -> str:
    """그 손가락의 끝 링크 이름."""
    return f"{finger}_tip"


def _vec(v: Sequence[float]) -> str:
    """URDF 속성에 쓰는 'x y z' 문자열."""
    return " ".join(f"{x:g}" for x in v)


def _joint(
    name: str,
    kind: str,
    parent: str,
    child: str,
    xyz: Sequence[float],
    rpy: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] | None = None,
    limit: tuple[float, float, float] | None = None,
) -> str:
    """<joint> 요소 한 개 (limit: 하한, 상한, 속도)."""
    lines = [
        f'  <joint name="{name}" type="{kind}">',
        f'    <parent link="{parent}"/><child link="{child}"/>',
        f'    <origin xyz="{_vec(xyz)}" rpy="{_vec(rpy)}"/>',
    ]
    if axis is not None:
        lines.append(f'    <axis xyz="{_vec(axis)}"/>')
    if limit is not None:
        lines.append(
            f'    <limit lower="{limit[0]:g}" upper="{limit[1]:g}" velocity="{limit[2]:g}" '
            'effort="10"/>'
        )
    lines.append("  </joint>")
    return "\n".join(lines)


def dexterous_hand_urdf() -> str:
    """시험용 다지 손 로봇의 URDF 문자열 (모듈 설명의 로봇, 같은 호출은 같은 문자열)."""
    links = [BASE_LINK, *(j[2] for j in _ARM), PALM_LINK]
    joints = [
        _joint(name, "revolute", parent, child, xyz, axis=axis, limit=(lo, hi, ARM_VELOCITY))
        for name, parent, child, xyz, axis, lo, hi in _ARM
    ]
    joints.append(_joint("palm_mount", "fixed", WRIST_LINK, PALM_LINK, _PALM_OFFSET))
    for finger, (xyz, rpy, proximal, distal) in _FINGERS.items():
        chain = (f"{finger}_base", f"{finger}_proximal", f"{finger}_distal")
        links += [*chain, tip_link(finger)]
        origins = (xyz, (0.0, 0.0, 0.0), (0.0, 0.0, proximal))
        parents = (PALM_LINK, *chain[:2])
        for k, (suffix, axis, lo, hi) in enumerate(_FINGER_JOINTS):
            joints.append(
                _joint(
                    f"{finger}_{suffix}",
                    "revolute",
                    parents[k],
                    chain[k],
                    origins[k],
                    rpy if k == 0 else (0.0, 0.0, 0.0),
                    axis,
                    (lo, hi, FINGER_VELOCITY),
                )
            )
        joints.append(
            _joint(f"{finger}_tip_mount", "fixed", chain[2], tip_link(finger), (0.0, 0.0, distal))
        )
    body = "\n".join([*(f'  <link name="{name}"/>' for name in links), *joints])
    return (
        '<?xml version="1.0"?>\n'
        "<!-- 시험용 합성 로봇 (dlp_fixtures.dexhand가 쓴다). 실제 로봇이 아니다. -->\n"
        f'<robot name="dlp_dexhand">\n{body}\n</robot>\n'
    )


def dexterous_hand_config(
    sha256: str,
    *,
    robot_id: str = "dexhand",
    model_path: str = "dexhand.urdf",
    fingers: Sequence[str] = FINGERS,
    hand_scale: float = 1.0,
    placement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """시험 로봇의 설정 사전 (`<robot_id>.yaml` 내용, `dlp_retarget.robot.DexterousRobotConfig`).

    Args:
        sha256: URDF 파일 해시.
        robot_id: 로봇 ID (파일 이름과 같아야 한다).
        model_path: 설정 파일 디렉터리 기준 URDF 경로.
        fingers: 사람 손가락에 대응시킬 로봇 손가락 (`FINGERS`의 부분, 순서가 관절 순서가 된다).
        hand_scale: 사람 (손가락 끝 - 손목) 벡터에 곱하는 손 배율.
        placement: 배치 (없으면 world_fixed, 배율 1, 로봇 기준 = 세션 월드 원점).
    """
    place = {"mode": "world_fixed", "position_scale": 1.0} if placement is None else dict(placement)
    return {
        "robot_id": robot_id,
        "display_name": "시험용 6축 팔 + 5지 손 (합성)",
        "model": {
            "format": "urdf",
            "path": model_path,
            "sha256": sha256,
            "license": "이 저장소가 쓴 시험용 합성 모델",
            "source": "dlp_fixtures.dexhand",
            "public": True,
        },
        "base_link": BASE_LINK,
        "arm": {
            "hand": "right",
            "joints": list(ARM_JOINTS),
            "placement": place,
            "dexterous_hand": {
                "wrist_link": WRIST_LINK,
                "hand_scale": hand_scale,
                "fingertip_tolerance_m": 0.005,
                "fingers": [
                    {"human_part": HUMAN_PART[f], "tip_link": tip_link(f)} for f in fingers
                ],
            },
        },
        "ik": {
            "position_tolerance_m": 0.005,
            "wrist_weight": 1.0,
            "smoothness_weight": 0.0001,
            "damping": 0.000001,
            "max_iterations": 200,
            "restarts": 8,
            "velocity_scale": 0.5,
        },
    }


def write_robot(
    directory: Path,
    *,
    robot_id: str = "dexhand",
    urdf: str | None = None,
    **config: Any,
) -> Path:
    """디렉터리에 URDF와 설정 파일을 쓰고 설정 경로를 돌려준다.

    설정은 JSON으로 쓴다 (JSON은 YAML의 부분이라 로더가 그대로 읽는다. 이 패키지는 YAML
    라이브러리에 의존하지 않는다).

    Args:
        directory: 쓸 디렉터리 (시험의 임시 디렉터리. `DLP_ROBOTS_DIR`로 쓸 수 있다).
        robot_id: 로봇 ID. 설정은 `<robot_id>.yaml`, URDF는 `<robot_id>.urdf`다.
        urdf: URDF 문자열 (없으면 `dexterous_hand_urdf()`. 거부 시험이 고친 URDF를 넘긴다).
        **config: `dexterous_hand_config`의 나머지 인자.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / f"{robot_id}.urdf"
    model.write_text(dexterous_hand_urdf() if urdf is None else urdf, "utf-8")
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    data = dexterous_hand_config(sha, robot_id=robot_id, model_path=model.name, **config)
    path = directory / f"{robot_id}.yaml"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return path


def smooth_joint_motion(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    times_ms: NDArray[np.float64],
    *,
    amplitude_rad: float = 0.3,
    seed: int = 0,
) -> NDArray[np.float64]:
    """관절 범위 가운데 둘레의 부드러운 임의 관절 궤적 (n, dof).

    관절마다 주기(1.5~3.5초)와 위상을 seed로 정한 사인이다. 진폭은 `amplitude_rad`와 그 관절 범위
    절반의 0.8배 중 작은 쪽이라 한계 안에 머문다.
    """
    rng = np.random.default_rng(seed)
    mid, half = (lower + upper) / 2, (upper - lower) / 2
    period_s = rng.uniform(1.5, 3.5, lower.size)
    phase = rng.uniform(0.0, 2 * math.pi, lower.size)
    amp = np.minimum(amplitude_rad, 0.8 * half)
    return mid + amp * np.sin(2 * math.pi * times_ms[:, None] / 1000 / period_s + phase)


def human_points_from_robot(
    wrist_b: NDArray[np.float64],
    tips_b: NDArray[np.float64],
    *,
    base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    base_yaw_rad: float = 0.0,
    position_scale: float = 1.0,
    hand_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """로봇 기준 좌표의 손목 (n, 3)·손가락 끝 (n, F, 3) → 세션 월드 좌표의 사람 손 점.

    다지 손 목표 식(`dlp_retarget.targets.finger_targets`, world_fixed 배치)을 거꾸로 쓴다:
    손목 W = base + Rz(yaw) · 손목_b / position_scale, 손가락 끝 = W + Rz(yaw) · (끝_b - 손목_b) /
    hand_scale. 이 점을 넣고 리타기팅하면 목표가 입력한 로봇 위치와 같다.

    Returns:
        (사람 손목 (n, 3), 사람 손가락 끝 (n, F, 3)), m.
    """
    c, s = math.cos(base_yaw_rad), math.sin(base_yaw_rad)
    r_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    wrist = np.asarray(base_xyz, dtype=np.float64) + (wrist_b / position_scale) @ r_wb.T
    tips = wrist[:, None, :] + ((tips_b - wrist_b[:, None, :]) / hand_scale) @ r_wb.T
    return wrist, tips
