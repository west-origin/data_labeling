"""순수 기하: 핀홀 카메라, 다시점 삼각측량, 닮음 변환(Umeyama)과 RANSAC, 기술자 대응.

DB·저장소·모델 없이 numpy만 쓴다. 좌표 규약은 `dlp_schema.scene3d` 머리말과 같다 (COLMAP 카메라,
`x_cam = R x + t`).
"""

from __future__ import annotations

from dataclasses import dataclass

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
