"""순수 기하: 핀홀 카메라, 다시점 삼각측량, 닮음 변환(Umeyama)과 RANSAC, 기술자 대응.

DB·저장소·모델 없이 numpy와 OpenCV의 회전 변환(Rodrigues)만 쓴다. 좌표 규약은
`dlp_schema.scene3d` 머리말과 같다 (COLMAP 카메라, `x_cam = R x + t`). 닮음 변환을 거친 카메라 자세
옮기기(`rigid_through_similarity`)와 두 자세 사이 보간(`interpolate_pose`)은 장면 안 바디캠 위치
추정(ADR 0067)이 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_scene3d.scipy_typed import knn
from dlp_schema.scene3d import CameraPose

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Camera:
    """핀홀 카메라 하나. K(3x3), R(3x3), t(3,), 이미지 크기."""

    photo_id: str
    k: F64
    r: F64
    t: F64
    width: int
    height: int

    @classmethod
    def from_pose(cls, pose: CameraPose) -> Camera:
        """계약 `CameraPose`에서 만든다."""
        k = np.array([[pose.fx, 0, pose.cx], [0, pose.fy, pose.cy], [0, 0, 1]], dtype=np.float64)
        r = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)
        t = np.asarray(pose.translation, dtype=np.float64)
        return cls(pose.photo_id, k, r, t, pose.width, pose.height)

    def to_pose(self) -> CameraPose:
        """계약 `CameraPose`로 바꾼다."""
        r = [float(v) for v in self.r.reshape(-1)]
        t = [float(v) for v in self.t]
        return CameraPose(
            photo_id=self.photo_id,
            width=self.width,
            height=self.height,
            fx=float(self.k[0, 0]),
            fy=float(self.k[1, 1]),
            cx=float(self.k[0, 2]),
            cy=float(self.k[1, 2]),
            rotation=(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]),
            translation=(t[0], t[1], t[2]),
        )

    @property
    def projection(self) -> F64:
        """3x4 투영 행렬 K[R|t]."""
        return self.k @ np.hstack([self.r, self.t[:, None]])

    @property
    def center(self) -> F64:
        """세계 좌표의 카메라 중심 -Rᵀt."""
        return -self.r.T @ self.t

    def depth(self, points: F64) -> F64:
        """(N, 3) 세계 점의 카메라 z (앞이 양수)."""
        return (points @ self.r.T + self.t)[:, 2]

    def project(self, points: F64) -> F64:
        """(N, 3) 세계 점 → (N, 2) 픽셀. 카메라 뒤 점은 NaN."""
        cam = points @ self.r.T + self.t
        z = cam[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (cam @ self.k.T)[:, :2] / z
        uv[z[:, 0] <= 0] = np.nan
        return uv


def look_at(center: F64, target: F64, up: F64) -> tuple[F64, F64]:
    """카메라 중심에서 target을 보는 (R, t). up은 세계의 위 방향 (카메라 -y로 간다)."""
    z = target - center
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return r, -r @ center


def triangulate(cameras: list[Camera], pixels: F64) -> F64:
    """다시점 선형 삼각측량 (DLT). pixels는 (V, 2), cameras와 순서가 같다. 결과 (3,).

    Raises:
        ValueError: 시점이 2개 미만일 때.
    """
    if len(cameras) < 2:
        raise ValueError("삼각측량에는 시점이 2개 이상 필요합니다")
    rows: list[F64] = []
    for cam, (u, v) in zip(cameras, pixels, strict=True):
        p = cam.projection
        rows += [u * p[2] - p[0], v * p[2] - p[1]]
    a = np.stack(rows)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(a)
    x = vt[-1]
    return x[:3] / x[3]


@dataclass(frozen=True)
class Similarity:
    """닮음 변환 `y = s R x + t`."""

    s: float
    r: F64
    t: F64

    def apply(self, points: F64) -> F64:
        """(N, 3) 점에 적용한다."""
        return self.s * points @ self.r.T + self.t

    def inverse(self) -> Similarity:
        """역변환."""
        ri = self.r.T
        return Similarity(1.0 / self.s, ri, -(ri @ self.t) / self.s)


def umeyama(src: F64, dst: F64) -> Similarity:
    """src → dst 최소제곱 닮음 변환 (Umeyama 1991). (N, 3), N ≥ 3, 같은 순서의 대응.

    Raises:
        ValueError: 점이 3개 미만이거나 한 점으로 모였을 때.
    """
    if len(src) < 3:
        raise ValueError("닮음 변환에는 대응점이 3개 이상 필요합니다")
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs**2).sum() / len(src))
    if var_s <= 1e-18:
        raise ValueError("대응점이 한 점에 모여 있습니다")
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    r = u @ sign @ vt
    s = float(np.trace(np.diag(d) @ sign) / var_s)
    return Similarity(s, r, mu_d - s * r @ mu_s)


@dataclass(frozen=True)
class RansacResult:
    """`ransac_similarity` 결과. inliers는 대응 번호 배열, rms는 내점 거리 RMS."""

    transform: Similarity | None
    inliers: NDArray[np.int64]
    rms: float | None


def ransac_similarity(
    src: F64, dst: F64, *, threshold: float, iterations: int, seed: int
) -> RansacResult:
    """이상점(움직인 물체의 점, 잘못된 대응)이 섞인 대응에서 닮음 변환을 찾는다.

    3점 표본으로 가설을 세우고 내점(변환 뒤 dst와 거리 ≤ threshold)이 가장 많은 가설을 내점 전체로
    다시 맞춘다 (두 번). 같은 seed면 같은 결과다. 대응이 3개 미만이면 transform=None.
    """
    n = len(src)
    if n < 3:
        return RansacResult(None, np.zeros(0, dtype=np.int64), None)
    rng = np.random.default_rng(seed)
    best = np.zeros(0, dtype=np.int64)
    for _ in range(iterations):
        pick = rng.choice(n, 3, replace=False)
        try:
            guess = umeyama(src[pick], dst[pick])
        except ValueError:
            continue
        inl = np.flatnonzero(np.linalg.norm(guess.apply(src) - dst, axis=1) <= threshold)
        if len(inl) > len(best):
            best = inl
    if len(best) < 3:
        return RansacResult(None, best, None)
    fit = umeyama(src[best], dst[best])
    for _ in range(2):
        inl = np.flatnonzero(np.linalg.norm(fit.apply(src) - dst, axis=1) <= threshold)
        if len(inl) < 3:
            break
        best, fit = inl, umeyama(src[inl], dst[inl])
    err = np.linalg.norm(fit.apply(src[best]) - dst[best], axis=1)
    return RansacResult(fit, best.astype(np.int64), float(np.sqrt((err**2).mean())))


def match_descriptors(a: F64, b: F64, ratio: float) -> NDArray[np.int64]:
    """기술자 상호 최근접 + 비율 검사 대응. (M, 2) [a 번호, b 번호].

    a→b 최근접이 둘째 최근접의 ratio배보다 가깝고, b→a 최근접도 서로일 때만 남긴다.
    """
    if len(a) < 2 or len(b) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    d, i_ab = knn(b, a, 2)
    _, i_ba2 = knn(a, b, 1)
    i_ba = i_ba2[:, 0]
    rows = np.arange(len(a))
    ok = (d[:, 0] < ratio * d[:, 1]) & (i_ba[i_ab[:, 0]] == rows)
    return np.stack([rows[ok], i_ab[ok, 0]], axis=1).astype(np.int64)


# 전수 비교에서 한 번에 거리 행렬로 펴는 질의 행 수 (메모리 상한: 행 수 x 자료 수 x 4바이트)
_MATCH_CHUNK = 256


def _two_nearest(
    query: NDArray[np.float32], data: NDArray[np.float32], data_sq: NDArray[np.float32]
) -> tuple[F64, F64, NDArray[np.int64]]:
    """query 각 행의 data 최근접·둘째 최근접 거리와 최근접 번호 (전수 비교, data는 2행 이상)."""
    n = len(query)
    d1, d2 = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
    i1 = np.empty(n, dtype=np.int64)
    for start in range(0, n, _MATCH_CHUNK):
        q = query[start : start + _MATCH_CHUNK]
        sq = (q * q).sum(axis=1)[:, None] + data_sq[None, :] - 2.0 * (q @ data.T)
        two = np.argpartition(sq, 1, axis=1)[:, :2]
        vals = np.take_along_axis(sq, two, axis=1)
        first = np.argmin(vals, axis=1)
        rows = np.arange(len(q))
        stop = start + len(q)
        d1[start:stop] = np.sqrt(np.maximum(vals[rows, first], 0.0))
        d2[start:stop] = np.sqrt(np.maximum(vals[rows, 1 - first], 0.0))
        i1[start:stop] = two[rows, first]
    return d1, d2, i1


@dataclass(frozen=True)
class DescriptorMatcher:
    """기술자 한 벌(복원 점)에 여러 질의(프레임 특징)를 차례로 대응시킨다.

    판정은 `match_descriptors`와 같다 (상호 최근접 + 비율 검사). 다른 점은 계산뿐이다: 자료 쪽을 한
    번만 준비하고 행렬 곱으로 전수 비교한다. SIFT처럼 차원이 큰(128) 기술자는 k-d 트리가 전수
    비교보다 느려서, 프레임마다 부르는 위치 추정은 이쪽을 쓴다 (ADR 0067). 거리 계산은 float32다.
    """

    data: NDArray[np.float32]
    data_sq: NDArray[np.float32]

    @classmethod
    def build(cls, descriptors: F64) -> DescriptorMatcher:
        """(N, D) 기술자로 만든다."""
        data = np.ascontiguousarray(descriptors, dtype=np.float32)
        return cls(data, (data * data).sum(axis=1))

    def match(self, query: F64, ratio: float) -> NDArray[np.int64]:
        """(M, 2) [질의 번호, 자료 번호]. 질의나 자료가 2개 미만이면 빈 배열."""
        if len(query) < 2 or len(self.data) < 2:
            return np.zeros((0, 2), dtype=np.int64)
        q = np.ascontiguousarray(query, dtype=np.float32)
        d1, d2, nearest = _two_nearest(q, self.data, self.data_sq)
        rows = np.flatnonzero(d1 < ratio * d2)
        if len(rows) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        # 상호 검사는 후보가 된 자료 점만 본다 (그 점에서 가장 가까운 질의가 자기여야 한다)
        cand = np.unique(nearest[rows])
        _, _, back = _two_nearest(self.data[cand], q, (q * q).sum(axis=1))
        mutual = back[np.searchsorted(cand, nearest[rows])] == rows
        return np.stack([rows[mutual], nearest[rows][mutual]], axis=1).astype(np.int64)


def rigid_through_similarity(r_ca: F64, t_ca: F64, a_to_b: Similarity) -> tuple[F64, F64]:
    """a 좌표의 카메라 자세를 닮음 변환 `x_b = s R_ab x_a + t_ab`로 b 좌표의 강체 자세로 옮긴다.

    `x_cam = R_ca x_a + t_ca`에 `x_a = R_abᵀ (x_b - t_ab) / s`를 넣고, 카메라 좌표의 길이 단위도 b로
    맞추려고 양변에 s를 곱하면 `s x_cam = R_ca R_abᵀ x_b + (s t_ca - R_ca R_abᵀ t_ab)`다. 그래서

        R_cb = R_ca R_abᵀ,    t_cb = s t_ca - R_cb t_ab

    회전에 s를 곱하지 않는다 (자세는 강체다). 카메라 중심은 `C_b = s R_ab C_a + t_ab`로 옮겨지고,
    같은 점의 화소 위치는 그대로이며 깊이만 s배가 된다.
    """
    r_cb = r_ca @ a_to_b.r.T
    return r_cb, a_to_b.s * t_ca - r_cb @ a_to_b.t


def interpolate_pose(r0: F64, t0: F64, r1: F64, t1: F64, alpha: float) -> tuple[F64, F64]:
    """두 카메라 자세 사이의 자세 (alpha 0이면 첫째, 1이면 둘째).

    카메라 중심 `C = -Rᵀ t`를 선형으로, 회전을 구면으로(상대 회전의 회전 벡터에 alpha를 곱한다)
    보간한 뒤 `t = -R C`로 되돌린다. t를 바로 선형 보간하면 회전이 다른 두 자세 사이에서 카메라
    중심이 직선 위에 있지 않다.
    """
    c0, c1 = -r0.T @ t0, -r1.T @ t1
    rvec, _ = cv2.Rodrigues(r1 @ r0.T)
    step, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64) * alpha)
    r = np.asarray(step, dtype=np.float64) @ r0
    return r, -r @ ((1.0 - alpha) * c0 + alpha * c1)
