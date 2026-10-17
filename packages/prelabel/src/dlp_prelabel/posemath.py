"""회전·자세 계산 도우미 (카메라 자세 추정과 월드 좌표 변환, ADR 0046).

- 회전 벡터 ↔ 회전 행렬 (`exp_so3`, `log_so3`, 로드리게스 식).
- 회전 행렬 ↔ 단위 쿼터니언 (`quat_from_matrix`, `matrix_from_quat`, 해밀턴 규약 (w, x, y, z), w ≥
0).
- 두 회전 사이 구면 선형 보간 (`slerp`).
- 자세 궤적 오차: ATE(정렬 없이 위치 RMS), RPE(구간 상대 이동·회전 오차) (`ate_rmse`, `rpe`).

순수 numpy 함수다 (파일·DB를 건드리지 않는다). 좌표계 규약은 `dlp_prelabel.campose`를 본다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Mat3 = NDArray[np.float64]
Vec3 = NDArray[np.float64]


def skew(v: Vec3) -> Mat3:
    """벡터 → 외적 행렬 [v]x."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(w: Vec3) -> Mat3:
    """회전 벡터(축 x 각, 라디안) → 회전 행렬."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def log_so3(r: Mat3) -> Vec3:
    """회전 행렬 → 회전 벡터 (각 0~π)."""
    cos = min(max((float(np.trace(r)) - 1) / 2, -1.0), 1.0)
    theta = math.acos(cos)
    if theta < 1e-9:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if math.pi - theta < 1e-6:
        # 180도 근처: 대칭 부분에서 축을 얻는다
        m = (r + np.eye(3)) / 2
        axis = m[:, int(np.argmax(np.diag(m)))]
        return axis / np.linalg.norm(axis) * theta
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v * theta / (2 * math.sin(theta))


def rotation_angle(r: Mat3) -> float:
    """회전 행렬의 회전각 (라디안, 0~π)."""
    return float(np.linalg.norm(log_so3(r)))


def orthonormalize(r: Mat3) -> Mat3:
    """적분 누적 오차로 어긋난 회전 행렬을 가장 가까운 회전으로 되돌린다 (SVD)."""
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def quat_from_matrix(r: Mat3) -> tuple[float, float, float, float]:
    """회전 행렬 → 단위 쿼터니언 (w, x, y, z), w ≥ 0."""
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    arr = np.array(q, dtype=np.float64)
    arr /= np.linalg.norm(arr)
    if arr[0] < 0:
        arr = -arr
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def matrix_from_quat(w: float, x: float, y: float, z: float) -> Mat3:
    """단위 쿼터니언 (w, x, y, z) → 회전 행렬 (길이가 1이 아니면 정규화한다)."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(r0: Mat3, r1: Mat3, a: float) -> Mat3:
    """r0(a=0)에서 r1(a=1)까지 구면 선형 보간."""
    return r0 @ exp_so3(log_so3(r0.T @ r1) * a)


def ate_rmse(estimated: Sequence[Vec3], truth: Sequence[Vec3]) -> float:
    """절대 위치 오차 RMS (m). 정렬(회전·배율 맞춤)을 하지 않는다: 같은 월드 좌표 규약이어야
    한다."""
    diff = np.asarray(estimated) - np.asarray(truth)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


def rpe(
    est: Sequence[tuple[Mat3, Vec3]], truth: Sequence[tuple[Mat3, Vec3]], step: int = 1
) -> tuple[float, float]:
    """상대 자세 오차 RMS (이동 m, 회전 라디안): i와 i+step 사이 상대 자세를 정답과 비교한다."""
    trans: list[float] = []
    rot: list[float] = []
    for i in range(len(est) - step):
        (ra, pa), (rb, pb) = est[i], est[i + step]
        (ta, qa), (tb, qb) = truth[i], truth[i + step]
        d_est = ra.T @ (pb - pa)
        d_true = ta.T @ (qb - qa)
        trans.append(float(np.linalg.norm(d_est - d_true)))
        rot.append(rotation_angle((ra.T @ rb).T @ (ta.T @ tb)))
    if not trans:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(np.square(trans)))), float(np.sqrt(np.mean(np.square(rot))))
