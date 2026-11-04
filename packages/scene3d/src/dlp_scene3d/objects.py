"""사진별 2D 물체 탐지 → 다시점 3D 물체 (삼각측량과 시점 간 연결).

입력은 한 단계(before 또는 after)의 카메라와 사진별 탐지다. 같은 라벨의 탐지 두 개(다른 사진)로
3D 가설을 세우고, 가설을 다른 사진에 다시 투영해 중심이 상자 대각선의 `center_tolerance` 안에 드는
탐지를 받침으로 모은다. 받침이 가장 많은 가설부터 받침 전체로 다시 삼각측량해 물체로 확정하고, 쓴
탐지는 다른 물체에 쓰지 않는다 (사진 하나에서 한 물체는 탐지 하나). 받침이 `min_views` 미만이면
버린다 (한두 장에서만 보인 오탐을 막는다).

외형 특징은 승인된 블러 사진의 상자 가운데 영역 색 히스토그램(HSV의 H x S)이다. 블러는 얼굴·문서
상자에만 있으므로 물체 색에는 거의 영향이 없다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_scene3d.geometry import F64, Camera, triangulate
from dlp_scene3d.policy import ObjectsPolicy
from dlp_schema.scene3d import ObjectInstance, ScenePhase


@dataclass(frozen=True)
class Detection2D:
    """사진 한 장의 물체 탐지 하나. box는 (x, y, w, h) 픽셀."""

    photo_id: str
    label: str
    box: tuple[float, float, float, float]
    score: float
    appearance: tuple[float, ...]

    @property
    def center(self) -> F64:
        """상자 중심 (u, v)."""
        x, y, w, h = self.box
        return np.array([x + w / 2, y + h / 2], dtype=np.float64)

    @property
    def diagonal(self) -> float:
        """상자 대각선 길이 (픽셀)."""
        return float(np.hypot(self.box[2], self.box[3]))


def appearance(
    image: NDArray[np.uint8],
    box: tuple[float, float, float, float],
    bins: tuple[int, int],
    crop: float,
) -> tuple[float, ...]:
    """상자 가운데(crop 비율) 영역의 HSV H x S 히스토그램 (합 1). 영역이 비면 균등 분포."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    hw, hh = w * crop / 2, h * crop / 2
    height, width = image.shape[:2]
    x1, x2 = int(max(0, np.floor(cx - hw))), int(min(width, np.ceil(cx + hw)))
    y1, y2 = int(max(0, np.floor(cy - hh))), int(min(height, np.ceil(cy + hh)))
    n = bins[0] * bins[1]
    if x2 <= x1 or y2 <= y1:
        return tuple([1.0 / n] * n)
    hsv = cv2.cvtColor(np.ascontiguousarray(image[y1:y2, x1:x2]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256]).reshape(-1)
    total = float(hist.sum())
    return tuple(float(v) / total for v in hist) if total > 0 else tuple([1.0 / n] * n)


def hellinger(a: Sequence[float], b: Sequence[float]) -> float:
    """두 히스토그램(합 1)의 Hellinger 거리 0~1."""
    pa, pb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    bc = float(np.sqrt(pa * pb).sum())
    return float(np.sqrt(max(0.0, 1.0 - bc)))


def _support(
    point: F64, dets: list[Detection2D], cams: Mapping[str, Camera], tol: float, free: set[int]
) -> list[int]:
    """point를 다시 투영해 받쳐 주는 탐지 번호 (사진마다 가장 가까운 하나, free 안에서만)."""
    best: dict[str, tuple[float, int]] = {}
    for i in free:
        d = dets[i]
        cam = cams[d.photo_id]
        if cam.depth(point[None])[0] <= 0:
            continue
        uv = cam.project(point[None])[0]
        err = float(np.linalg.norm(np.subtract(uv, d.center)))
        if err <= tol * d.diagonal and (d.photo_id not in best or err < best[d.photo_id][0]):
            best[d.photo_id] = (err, i)
    return sorted(i for _, i in best.values())


def _refine(
    dets: list[Detection2D], cams: Mapping[str, Camera], seed: F64, tol: float, free: set[int]
) -> tuple[F64, list[int]]:
    """가설점에서 받침을 모으고 받침 전체로 다시 삼각측량한다 (두 번)."""
    point, sup = seed, _support(seed, dets, cams, tol, free)
    for _ in range(2):
        if len(sup) < 2:
            break
        point = triangulate(
            [cams[dets[i].photo_id] for i in sup], np.stack([dets[i].center for i in sup])
        )
        sup = _support(point, dets, cams, tol, free)
    return point, sup


def _hypotheses(dets: list[Detection2D], cams: Mapping[str, Camera]) -> list[F64]:
    """다른 사진 탐지 두 개로 만든 3D 가설 (두 카메라 앞에 있는 것만)."""
    out: list[F64] = []
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            if a.photo_id == b.photo_id:
                continue
            ca, cb = cams[a.photo_id], cams[b.photo_id]
            x = triangulate([ca, cb], np.stack([a.center, b.center]))
            if ca.depth(x[None])[0] > 0 and cb.depth(x[None])[0] > 0:
                out.append(x)
    return out


def _instance(
    phase: ScenePhase,
    label: str,
    point: F64,
    sup: list[int],
    dets: list[Detection2D],
    cams: Mapping[str, Camera],
    policy: ObjectsPolicy,
) -> ObjectInstance:
    """받침 탐지로 물체 하나를 만든다 (크기·외형·재투영 오차·신뢰도)."""
    used = [dets[i] for i in sup]
    sizes = [
        d.diagonal * cams[d.photo_id].depth(point[None])[0] / cams[d.photo_id].k[0, 0] for d in used
    ]
    errs = [
        float(np.linalg.norm(np.subtract(cams[d.photo_id].project(point[None])[0], d.center)))
        for d in used
    ]
    app = np.mean(np.asarray([d.appearance for d in used], dtype=np.float64), axis=0)
    app = app / app.sum()
    view_factor = min(1.0, len(used) / policy.full_views)
    confidence = float(np.mean([d.score for d in used])) * view_factor
    photos = tuple(sorted(d.photo_id for d in used))
    digest = hashlib.sha256(f"{phase}|{label}|{'|'.join(photos)}".encode()).hexdigest()[:12]
    p = [float(v) for v in point]
    return ObjectInstance(
        instance_id=f"{phase}-{label}-{digest}",
        phase=phase,
        label=label,
        position=(p[0], p[1], p[2]),
        size=float(np.median(sizes)),
        appearance=tuple(float(v) for v in app),
        views=len(used),
        photo_ids=photos,
        reproj_px=float(np.sqrt(np.mean(np.square(errs)))),
        confidence=min(1.0, max(0.0, confidence)),
    )


def localize_objects(
    phase: ScenePhase,
    cameras: Mapping[str, Camera],
    detections: Sequence[Detection2D],
    policy: ObjectsPolicy,
) -> list[ObjectInstance]:
    """한 단계의 사진별 탐지에서 3D 물체 목록을 만든다 (라벨, instance_id 순).

    자세를 찾지 못한 사진의 탐지는 쓰지 않는다. 결과는 같은 입력에 대해 결정적이다.
    """
    out: list[ObjectInstance] = []
    for label in sorted({d.label for d in detections}):
        dets = [d for d in detections if d.label == label and d.photo_id in cameras]
        free = set(range(len(dets)))
        scored: list[tuple[int, float, F64, list[int]]] = []
        for k, hyp in enumerate(_hypotheses(dets, cameras)):
            point, sup = _refine(dets, cameras, hyp, policy.center_tolerance, free)
            scored.append((len(sup), -k, point, sup))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        for _, _, _, sup0 in scored:
            if len(sup0) < 2 or not set(sup0) <= free:
                continue
            point, sup = _refine(
                dets,
                cameras,
                triangulate(
                    [cameras[dets[i].photo_id] for i in sup0],
                    np.stack([dets[i].center for i in sup0]),
                ),
                policy.center_tolerance,
                free,
            )
            if len(sup) < policy.min_views:
                continue
            out.append(_instance(phase, label, point, sup, dets, cameras, policy))
            free -= set(sup)
    return sorted(out, key=lambda o: (o.label, o.instance_id))
