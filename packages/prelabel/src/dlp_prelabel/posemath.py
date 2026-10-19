"""회전·자세 계산 도우미 (카메라 자세 추정과 월드 좌표 변환, ADR 0046).

- 회전 벡터 ↔ 회전 행렬 (`exp_so3`, `log_so3`, 로드리게스 식).
- 회전 행렬 ↔ 단위 쿼터니언 (`quat_from_matrix`, `matrix_from_quat`, 해밀턴 규약 (w, x, y, z), w ≥
0).
- 두 회전 사이 구면 선형 보간 (`slerp`).
- 자세 궤적 오차: ATE(정렬 없이 위치 RMS), RPE(구간 상대 이동·회전 오차) (`ate_rmse`, `rpe`).
- 외부 파라미터 검사: 입력 행렬의 회전 부분이 회전인지 (`rotation_problem`, `as_rotation`).

규약 (모두 이 모듈 전체에 쓴다)
- 회전 행렬 R은 "열 벡터에 왼쪽에서 곱한다": 새 좌표 = R · 옛 좌표. R_ab는 b 좌표의 점을 a 좌표로
  옮긴다 (예: campose의 R_wc는 카메라 → 월드). 합성은 R_ac = R_ab · R_bc.
- 회전 벡터 ω는 축 방향 단위 벡터 x 각도(라디안)이고 오른손 법칙이다 (엄지가 축, 손가락이 양의
  방향).
- 쿼터니언은 해밀턴 규약 (w, x, y, z), 단위 길이, w ≥ 0으로 부호를 고정한다 (q와 -q는 같은 회전이라
  저장 값이 하나로 정해지게). 내보내기 형식(LeRobot·JSON)도 이 순서다.
- 길이 단위는 미터, 각도는 라디안이다 (함수 이름에 deg가 있으면 도).

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
    """회전 벡터(축 x 각, 라디안) → 회전 행렬 (로드리게스 식 I + sinθ K + (1 - cosθ) K²)."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        # 아주 작은 각: 1차 근사 I + [ω]x (θ로 나누면 0으로 나누게 된다)
        return np.eye(3) + skew(w)
    k = skew(w / theta)
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


def log_so3(r: Mat3) -> Vec3:
    """회전 행렬 → 회전 벡터 (각 0~π)."""
    # trace(R) = 1 + 2 cosθ. 반올림으로 [-1, 1] 밖이 되면 acos가 실패하므로 자른다
    cos = min(max((float(np.trace(r)) - 1) / 2, -1.0), 1.0)
    theta = math.acos(cos)
    if theta < 1e-9:
        # 아주 작은 각: 반대칭 부분 (R - Rᵀ)/2 = [ω]x 에서 바로 읽는다
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / 2
    if math.pi - theta < 1e-6:
        # 180도 근처: 대칭 부분에서 축을 얻는다
        m = (r + np.eye(3)) / 2
        axis = m[:, int(np.argmax(np.diag(m)))]
        return axis / np.linalg.norm(axis) * theta
    # 일반 경우: 반대칭 부분 = sinθ [k]x 이므로 축 k = v / (2 sinθ), 회전 벡터 = k θ
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return v * theta / (2 * math.sin(theta))


def rotation_angle(r: Mat3) -> float:
    """회전 행렬의 회전각 (라디안, 0~π)."""
    return float(np.linalg.norm(log_so3(r)))


def orthonormalize(r: Mat3) -> Mat3:
    """적분 누적 오차로 어긋난 회전 행렬을 가장 가까운 회전으로 되돌린다 (SVD)."""
    # R = U S Vᵀ에서 특이값 S를 I로 바꾼 U Vᵀ가 프로베니우스 거리로 가장 가까운 직교 행렬이다
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    # 직교여도 행렬식이 -1(반사)이면 회전이 아니다: 가장 작은 특이값 쪽 축을 뒤집는다
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def quat_from_matrix(r: Mat3) -> tuple[float, float, float, float]:
    """회전 행렬 → 단위 쿼터니언 (w, x, y, z), w ≥ 0."""
    # 셰퍼드 방법: w, x, y, z 중 가장 큰 성분을 먼저 구해 나눗셈 분모가 0에 가까워지지 않게 한다.
    # trace > 0이면 w가 크고, 아니면 대각 원소가 가장 큰 축의 성분이 크다
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
    # q와 -q는 같은 회전이다. 저장 값을 하나로 정하려고 w ≥ 0 쪽을 고른다
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
    # r0ᵀ r1 = r0에서 r1로 가는 상대 회전. 그 회전 벡터를 a배 한 만큼만 r0에서 더 돌린다
    return r0 @ exp_so3(log_so3(r0.T @ r1) * a)


def rotation_problem(m: object, tolerance: float) -> str | None:
    """외부 파라미터 행렬의 회전 부분이 쓸 수 있는 회전이 아니면 그 이유, 맞으면 None.

    받는 모양: 3x3 회전 또는 4x4 동차 변환(마지막 행 [0, 0, 0, 1]). 회전 부분 R은 원소별로
    R Rᵀ = I에서 tolerance 안이어야 하고(배율이 섞이지 않음), 행렬식이 양수여야 한다(축 부호를
    하나 틀리게 적은 반사 행렬은 det = -1이라 회전이 아니다, F-PAR-49).
    """
    try:
        a = np.array(m, dtype=np.float64)
    except ValueError:
        return "행 길이가 서로 다릅니다"
    if a.shape not in ((3, 3), (4, 4)):
        return f"3x3 또는 4x4 행렬이어야 합니다 (모양 {a.shape})"
    if a.shape == (4, 4) and not np.allclose(a[3], [0.0, 0.0, 0.0, 1.0], atol=tolerance):
        return "4x4 동차 변환의 마지막 행이 [0, 0, 0, 1]이 아닙니다"
    r = a[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=tolerance):
        return "회전 부분이 직교 행렬이 아닙니다 (배율·전단이 섞였거나 값이 틀림)"
    if np.linalg.det(r) <= 0:
        return "회전 부분의 행렬식이 음수입니다 (축 부호를 틀리게 적은 반사 행렬)"
    return None


def as_rotation(m: object, tolerance: float) -> Mat3 | None:
    """외부 파라미터 행렬 → 회전 부분 3x3 (가장 가까운 회전으로 고친 값). 회전이 아니면 None.

    검사는 `rotation_problem`. 사람이 소수 몇 자리로 적은 값은 직교에서 조금 벗어나므로 SVD로
    가장 가까운 회전으로 되돌려 쓴다 (계산과 저장 쿼터니언이 같은 회전을 가리키게).
    """
    if rotation_problem(m, tolerance) is not None:
        return None
    return orthonormalize(np.array(m, dtype=np.float64)[:3, :3])


def ate_rmse(estimated: Sequence[Vec3], truth: Sequence[Vec3]) -> float:
    """절대 위치 오차 RMS (m). 정렬(회전·배율 맞춤)을 하지 않는다: 같은 월드 좌표 규약이어야
    한다."""
    # 표본마다 위치 차이의 길이 제곱을 평균한 뒤 제곱근 (m)
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
        # 상대 이동을 앞 자세의 카메라 좌표로 본다 (월드 좌표 드리프트와 무관한 국소 오차)
        d_est = ra.T @ (pb - pa)
        d_true = ta.T @ (qb - qa)
        trans.append(float(np.linalg.norm(d_est - d_true)))
        rot.append(rotation_angle((ra.T @ rb).T @ (ta.T @ tb)))
    if not trans:
        return 0.0, 0.0
    return float(np.sqrt(np.mean(np.square(trans)))), float(np.sqrt(np.mean(np.square(rot))))
