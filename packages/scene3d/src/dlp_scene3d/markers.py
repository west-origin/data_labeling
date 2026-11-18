"""크기를 아는 ArUco 마커로 복원의 미터 척도 정하기 (ADR 0067).

SfM만으로는 척도를 모른다. `pipeline.reconstruct`가 SfM 뒤, 스플랫 학습 전에 이 모듈을 부른다.

1. `find_markers`: 승인된 블러 사진(`privacy_gate.open_blurred`가 준 것)마다 `cv2.aruco`로
   마커를 찾는다. 자세를 찾지 못한 사진, 정책의 허용 ID 밖 마커는 쓰지 않는다.
2. `estimate_scale`: 마커마다 네 모서리를 SfM 카메라로 삼각측량한다 (복원 단위). 다시 투영한
   모서리가 그 사진 속 마커 크기에 견줘 멀리 떨어진 사진은 하나씩 뺀다 (가려진 모서리, 틀린
   검출). 남은 사진이 `min_photos` 미만이거나 시선 사이 각이 `min_parallax_deg` 미만이면 그
   마커를 버린다. 남은 마커의 네 변 길이를 모아 상대 흩어짐이 `max_side_spread`를 넘으면 척도를
   정하지 않는다. 척도 = `side_m` / 변 길이 중앙값.
3. `scale_sfm`: 카메라 이동과 희소 점에 척도를 곱한다 (회전·내부 파라미터·화소 위치는 그대로).

척도를 정하지 못하면 지어내지 않는다: `ScaleEstimate.factor`가 None이고 `reason`에 이유가 남는다.
모서리 화소 위치는 OpenCV 규약(왼쪽 위 화소의 중심이 (0, 0))이다. 변 길이는 두 모서리의 차라서,
모서리 전체가 같은 쪽으로 반 화소쯤 밀려도(apriltag 다듬기가 그렇다) 거의 달라지지 않는다.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_scene3d.backends import SfmResult
from dlp_scene3d.geometry import F64, Camera, triangulate
from dlp_scene3d.policy import MarkerPolicy
from dlp_scene3d.privacy_gate import BlurredPhoto

# 마커 한 장의 모서리 수 (왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래)
_CORNERS = 4


class MarkerConfigError(ValueError):
    """정책의 마커 사전 이름이 `cv2.aruco`에 없다."""


@dataclass(frozen=True)
class MarkerObservation:
    """사진 한 장에서 찾은 마커 하나. corners는 (4, 2) 화소 (마커 그림의 모서리 순서)."""

    photo_id: str
    marker_id: int
    corners: F64

    @property
    def side_px(self) -> float:
        """그 사진 속 마커 한 변의 평균 길이 (화소)."""
        nxt = np.roll(self.corners, -1, axis=0)
        return float(np.linalg.norm(self.corners - nxt, axis=1).mean())


@dataclass(frozen=True)
class ScaleEstimate:
    """마커가 말하는 척도. factor가 None이면 정하지 못했다 (reason에 이유).

    marker_ids: 쓴 마커 ID (오름차순). photos: 그 마커들을 삼각측량하는 데 쓴 사진 수 (중복 없이).
    side_spread: 복원 단위 변 길이들의 (최대 - 최소) / 중앙값.
    """

    factor: float | None
    marker_ids: tuple[int, ...]
    photos: int
    side_spread: float | None
    reason: str | None


def _dictionary(name: str) -> cv2.aruco.Dictionary:
    """사전 이름(`DICT_4X4_50` 등) → `cv2.aruco` 사전."""
    code = getattr(cv2.aruco, name, None)
    if not isinstance(code, int):
        raise MarkerConfigError(f"cv2.aruco에 없는 마커 사전입니다: {name}")
    return cv2.aruco.getPredefinedDictionary(code)


def _detector(policy: MarkerPolicy) -> cv2.aruco.ArucoDetector:
    """정책의 사전·모서리 다듬기 방식으로 검출기를 만든다."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_APRILTAG
        if policy.corner_refinement == "apriltag"
        else cv2.aruco.CORNER_REFINE_SUBPIX
    )
    return cv2.aruco.ArucoDetector(_dictionary(policy.dictionary), params)


def detect_markers(image: NDArray[np.uint8], policy: MarkerPolicy) -> list[tuple[int, F64]]:
    """(H, W, 3) RGB 사진 한 장의 (마커 ID, 모서리 (4, 2) 화소) 목록. 허용 ID만 낸다.

    같은 ID가 한 사진에 둘 이상 보이면 어느 것이 같은 마커인지 알 수 없어 그 ID는 내지 않는다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _detector(policy).detectMarkers(gray)
    if len(corners) == 0:
        return []
    found: dict[int, list[F64]] = {}
    for quad, marker_id in zip(corners, np.asarray(ids).reshape(-1).tolist(), strict=True):
        if policy.ids and int(marker_id) not in policy.ids:
            continue
        found.setdefault(int(marker_id), []).append(
            np.asarray(quad, dtype=np.float64).reshape(_CORNERS, 2)
        )
    return [(i, quads[0]) for i, quads in sorted(found.items()) if len(quads) == 1]


def find_markers(
    photos: Sequence[BlurredPhoto], cameras: Mapping[str, Camera], policy: MarkerPolicy
) -> list[MarkerObservation]:
    """자세를 찾은 블러 사진에서 마커를 찾는다 (사진 ID, 마커 ID 순)."""
    out: list[MarkerObservation] = []
    for photo in sorted(photos, key=lambda p: p.photo_id):
        if photo.photo_id not in cameras:
            continue
        for marker_id, quad in detect_markers(photo.load(), policy):
            out.append(MarkerObservation(photo.photo_id, marker_id, quad))
    return out


def _corner_errors(
    views: Sequence[MarkerObservation], cameras: Mapping[str, Camera], points: F64
) -> F64:
    """사진마다 모서리 재투영 오차의 최댓값을 그 사진 속 마커 한 변(화소)으로 나눈 값.

    삼각측량한 모서리가 카메라 뒤에 있으면 무한대 (그 사진은 맞지 않는다).
    """
    out = np.empty(len(views), dtype=np.float64)
    for i, v in enumerate(views):
        cam = cameras[v.photo_id]
        if (cam.depth(points) <= 0).any():
            out[i] = np.inf
            continue
        err = np.linalg.norm(cam.project(points) - v.corners, axis=1)
        out[i] = float(err.max()) / max(v.side_px, 1e-9)
    return out


def _triangulate_marker(views: Sequence[MarkerObservation], cameras: Mapping[str, Camera]) -> F64:
    """마커 네 모서리의 3D 위치 (4, 3), 복원 단위."""
    cams = [cameras[v.photo_id] for v in views]
    return np.stack(
        [triangulate(cams, np.stack([v.corners[k] for v in views])) for k in range(_CORNERS)]
    )


def _parallax_deg(
    views: Sequence[MarkerObservation], cameras: Mapping[str, Camera], at: F64
) -> float:
    """점 at을 보는 시선들 사이의 가장 큰 각 (도)."""
    rays = np.stack([at - cameras[v.photo_id].center for v in views])
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    cos = np.clip(rays @ rays.T, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos.min())))


def _marker_sides(
    marker_id: int,
    views: list[MarkerObservation],
    cameras: Mapping[str, Camera],
    policy: MarkerPolicy,
) -> tuple[F64, list[MarkerObservation]] | str:
    """마커 하나의 (복원 단위 네 변 길이, 쓴 사진) 또는 버린 이유.

    재투영이 가장 나쁜 사진부터 하나씩 빼며 다시 삼각측량한다 (모두 상한 안에 들 때까지).
    """
    seen = len(views)
    while len(views) >= policy.min_photos:
        points = _triangulate_marker(views, cameras)
        errors = _corner_errors(views, cameras, points)
        worst = int(np.argmax(errors))
        if errors[worst] > policy.max_corner_error:
            views = views[:worst] + views[worst + 1 :]
            continue
        parallax = _parallax_deg(views, cameras, points.mean(axis=0))
        if parallax < policy.min_parallax_deg:
            return (
                f"마커 {marker_id}: 시선 사이 각 {parallax:.1f}도 < "
                f"최소 {policy.min_parallax_deg}도"
            )
        return np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1), views
    return (
        f"마커 {marker_id}: 모서리가 맞는 사진 {len(views)}장(본 사진 {seen}장) < "
        f"최소 {policy.min_photos}장"
    )


def estimate_scale(
    observations: Sequence[MarkerObservation], cameras: Mapping[str, Camera], policy: MarkerPolicy
) -> ScaleEstimate:
    """마커 관측과 SfM 카메라에서 척도(= 실제 변 길이 / 복원 단위 변 길이)를 구한다.

    Args:
        observations: `find_markers` 결과 (자세를 찾은 사진의 관측만 쓴다).
        cameras: 사진 ID → SfM 카메라 (복원 단위).
        policy: `reconstruction.marker` 절.

    Returns:
        척도와 근거. 마커가 없거나 불안정하면 factor가 None이다 (지어내지 않는다).
    """
    by_marker: dict[int, list[MarkerObservation]] = {}
    for o in observations:
        if o.photo_id in cameras:
            by_marker.setdefault(o.marker_id, []).append(o)
    if not by_marker:
        return ScaleEstimate(None, (), 0, None, "마커를 찾지 못했습니다")
    sides: list[F64] = []
    used: list[int] = []
    photos: set[str] = set()
    reasons: list[str] = []
    for marker_id in sorted(by_marker):
        fit = _marker_sides(marker_id, by_marker[marker_id], cameras, policy)
        if isinstance(fit, str):
            reasons.append(fit)
            continue
        sides.append(fit[0])
        used.append(marker_id)
        photos |= {v.photo_id for v in fit[1]}
    if not sides:
        return ScaleEstimate(None, (), 0, None, "; ".join(reasons))
    lengths = np.concatenate(sides)
    median = float(np.median(lengths))
    spread = float((lengths.max() - lengths.min()) / median)
    if spread > policy.max_side_spread:
        why = f"마커 변 길이 흩어짐 {spread:.3f} > 허용 {policy.max_side_spread}"
        return ScaleEstimate(None, tuple(used), len(photos), spread, why)
    return ScaleEstimate(policy.side_m / median, tuple(used), len(photos), spread, None)


def scale_sfm(result: SfmResult, factor: float) -> SfmResult:
    """SfM 결과의 길이 단위에 factor를 곱한 사본 (metric=True).

    세계 좌표가 factor배가 되면 `x_cam = R x + t`의 카메라 좌표도 factor배가 되어야 화소 위치가
    그대로다. 그래서 점과 카메라 이동 t에 곱하고 회전·내부 파라미터는 그대로 둔다.
    """
    cams = {pid: dataclasses.replace(c, t=c.t * factor) for pid, c in result.cameras.items()}
    return dataclasses.replace(result, cameras=cams, points=result.points * factor, metric=True)
