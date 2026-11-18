"""2D-3D 대응에서 카메라 자세 풀기 (PnP RANSAC과 내점 다듬기, ADR 0067).

장면 안 바디캠 위치 추정(`localize`)이 프레임마다 부른다. 입력은 장면 좌표의 3D 점과 그 점이
프레임에 찍힌 화소 위치(틀린 대응이 섞여 있다), 바디캠 내부 파라미터와 왜곡 계수(OpenCV 방사·접선
모델)다. `cv2.solvePnPRansac`(표본 3점 + 확인 1점, P3P)으로 내점 집합과 처음 자세를 얻고, 내점만으로
다시 풀어(Levenberg-Marquardt) 내점을 다시 고르기를 내점 집합이 그대로일 때까지 되풀이한다 (상한
있음). 결과 자세는 `x_cam = R x_scene + t`다 (`dlp_schema.scene3d` 규약).

풀지 못하면(대응 4개 미만, 퇴화, OpenCV 오류) None을 돌려준다. 받아들일지(내점 수·재투영 오차·
내점이 퍼진 정도)는 호출자가 정책으로 정한다. 같은 입력이면 같은 결과다 (OpenCV의 이 RANSAC은 고정
난수열을 쓴다. 시험이 확인한다).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_scene3d.geometry import F64

# 내점으로 다시 풀고 내점을 다시 고르는 횟수의 상한 (내점 집합이 그대로면 일찍 끝낸다)
_REFINE_ROUNDS = 5
# PnP가 풀리는 최소 대응 수
MIN_POINTS = 4


@dataclass(frozen=True)
class PnpSolution:
    """PnP 결과. r (3, 3), t (3,)는 `x_cam = R x + t`. inliers는 대응 번호(오름차순).

    reproj_px는 내점 재투영 오차의 RMS(화소), matches는 넣은 대응 수.
    """

    r: F64
    t: F64
    inliers: NDArray[np.int64]
    reproj_px: float
    matches: int


def reprojection_errors(
    points: F64, pixels: F64, k: F64, dist: F64, r: F64, t: F64
) -> tuple[F64, NDArray[np.bool_]]:
    """(대응마다 재투영 오차 화소, 그 점이 카메라 앞에 있는가). 왜곡 계수를 적용해 투영한다."""
    rvec, _ = cv2.Rodrigues(r)
    proj, _ = cv2.projectPoints(points, rvec, t.reshape(3, 1), k, dist)
    err = np.linalg.norm(np.asarray(proj, dtype=np.float64).reshape(-1, 2) - pixels, axis=1)
    return err, (points @ r.T + t)[:, 2] > 0


def _refit(
    points: F64, pixels: F64, k: F64, dist: F64, r: F64, t: F64, inliers: NDArray[np.int64]
) -> tuple[F64, F64] | None:
    """지금 자세를 출발점으로 내점만으로 다시 푼다. 실패하면 None."""
    rvec, _ = cv2.Rodrigues(r)
    try:
        ok, rvec2, tvec2 = cv2.solvePnP(
            points[inliers],
            pixels[inliers],
            k,
            dist,
            np.asarray(rvec, dtype=np.float64),
            t.reshape(3, 1).copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec2)
    return np.asarray(rot, dtype=np.float64), np.asarray(tvec2, dtype=np.float64).reshape(3)


def solve_pnp(
    points: F64,
    pixels: F64,
    k: F64,
    dist: F64,
    *,
    threshold_px: float,
    iterations: int,
    confidence: float,
) -> PnpSolution | None:
    """(N, 3) 장면 점과 (N, 2) 화소 대응에서 카메라 자세를 푼다.

    Args:
        points: 장면 좌표 3D 점.
        pixels: 그 점의 프레임 화소 위치 (같은 순서).
        k: 3x3 내부 파라미터 행렬.
        dist: OpenCV 왜곡 계수 (없으면 길이 0 배열).
        threshold_px: 내점으로 볼 재투영 오차 상한 (화소).
        iterations: RANSAC 반복 상한.
        confidence: RANSAC 신뢰 수준 (0~1).

    Returns:
        자세와 내점, 풀지 못했으면 None. 내점은 카메라 앞에 있고 재투영 오차가 상한 이내인 대응이다.
    """
    n = len(points)
    if n < MIN_POINTS:
        return None
    obj = np.ascontiguousarray(points, dtype=np.float64)
    img = np.ascontiguousarray(pixels, dtype=np.float64)
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            obj,
            img,
            k,
            dist,
            iterationsCount=iterations,
            reprojectionError=threshold_px,
            confidence=confidence,
            flags=cv2.SOLVEPNP_P3P,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    r = np.asarray(rot, dtype=np.float64)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    previous = np.zeros(0, dtype=np.int64)
    for _ in range(_REFINE_ROUNDS):
        err, front = reprojection_errors(obj, img, k, dist, r, t)
        inliers = np.flatnonzero((err <= threshold_px) & front).astype(np.int64)
        if len(inliers) < MIN_POINTS:
            return None
        if np.array_equal(inliers, previous):
            break
        fit = _refit(obj, img, k, dist, r, t, inliers)
        if fit is None:
            break
        (r, t), previous = fit, inliers
    err, front = reprojection_errors(obj, img, k, dist, r, t)
    inliers = np.flatnonzero((err <= threshold_px) & front).astype(np.int64)
    if len(inliers) < MIN_POINTS:
        return None
    rms = float(np.sqrt(np.mean(np.square(err[inliers]))))
    return PnpSolution(r, t, inliers, rms, n)
