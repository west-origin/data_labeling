"""표면 평면과 표면 좌표.

표면은 네 꼭짓점(corner_0 원점, corner_1 가로 끝, corner_2, corner_3 세로 끝)의 3D 궤적으로
주어진다. 시각마다 꼭짓점으로 평면(최소제곱)과 표면 좌표축을 만들고, 작용부 점을
(가로 비율, 세로 비율, 평면까지 거리 m)로 바꾼다. 같은 시각의 점끼리만 비교하므로 카메라 좌표계라도
카메라 움직임과 무관하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Vec = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceFrame:
    """한 시각의 표면 좌표계.

    origin: corner_0 (m). u: corner_0 → corner_1 (가로 변 벡터). v: corner_0 → corner_3 (세로 변
    벡터).
    centroid: 네 꼭짓점 평균. normal: 최소제곱 평면의 단위 법선 (부호는 SVD가 정한다).
    """

    origin: Vec
    u: Vec  # corner_0 → corner_1
    v: Vec  # corner_0 → corner_3
    centroid: Vec
    normal: Vec

    @classmethod
    def from_corners(cls, corners: Vec) -> SurfaceFrame:
        """corners: (4, 3).

        네 꼭짓점을 중심화해 SVD한 가장 작은 특이값의 오른쪽 특이벡터를 법선으로 쓴다 (최소제곱
        평면). 꼭짓점이 한 평면에 없어도 된다 (잡음).
        """
        centroid = corners.mean(axis=0)
        # vt의 마지막 행 = 가장 작은 특이값 방향 = 최소제곱 평면의 법선
        vt: Vec = np.linalg.svd(corners - centroid)[2]
        normal = np.asarray(vt[2], dtype=np.float64)
        normal /= float(np.sqrt(normal @ normal))
        return cls(corners[0], corners[1] - corners[0], corners[3] - corners[0], centroid, normal)

    @property
    def size_m(self) -> tuple[float, float]:
        """(가로 길이, 세로 길이) m = (|u|, |v|)."""
        return float(np.linalg.norm(self.u)), float(np.linalg.norm(self.v))

    def local(self, p: Vec) -> tuple[float, float, float]:
        """(가로 비율, 세로 비율, 평면까지 거리 m). 비율은 0~1이면 표면 안이다.

        점을 평면에 수직 투영한 뒤, origin에서 u·v 기저의 계수를 최소제곱으로 푼다 (u, v가 직교가
        아니어도 된다). 거리는 부호 없는 수직 거리다.
        """
        distance = float(abs((p - self.centroid) @ self.normal))
        # 평면 위로 수직 투영한 뒤 origin + a*u + b*v = projected를 최소제곱으로 푼다
        projected = p - ((p - self.centroid) @ self.normal) * self.normal
        basis = np.stack([self.u, self.v], axis=1)
        (a, b), *_ = np.linalg.lstsq(basis, projected - self.origin, rcond=None)
        return float(a), float(b), distance


def interpolate(times: NDArray[np.int64], values: Vec, t: int, max_gap_ms: int) -> Vec | None:
    """시각 t의 값을 선형 보간. 가장 가까운 샘플이 max_gap_ms보다 멀면 None.

    Args:
        times: 오름차순 시각 (ms). values: (N, k) 값.
        t: 찾을 시각. max_gap_ms: 허용 거리.

    범위 밖(첫 샘플 전, 마지막 뒤)이면 끝 샘플이 max_gap_ms 안일 때 그 값을 그대로 쓴다 (외삽하지
    않는다). 사이에서는 가까운 쪽 샘플이 max_gap_ms 안이면 보간한다 (먼 쪽은 얼마나 멀어도 된다).
    """
    i = int(np.searchsorted(times, t))
    if i < len(times) and times[i] == t:
        return values[i]
    if i == 0 or i == len(times):
        # 범위 밖: 가까운 끝 샘플이 max_gap_ms 안이면 그 값을 그대로 쓴다 (외삽하지 않는다)
        j = 0 if i == 0 else len(times) - 1
        return values[j] if abs(int(times[j]) - t) <= max_gap_ms else None
    t0, t1 = int(times[i - 1]), int(times[i])
    # 가까운 쪽 샘플까지 거리로 판단한다
    if min(t - t0, t1 - t) > max_gap_ms:
        return None
    w = (t - t0) / (t1 - t0)
    return values[i - 1] * (1 - w) + values[i] * w
