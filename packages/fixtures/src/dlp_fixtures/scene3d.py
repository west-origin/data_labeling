"""청소 전후 다각도 사진과 그 방을 지나는 바디캠 영상의 합성 정답 (ADR 0066, 0067).

정답을 아는 방 하나를 만든다: 카메라(내부·외부 파라미터), 고정 구조 점(벽·바닥, 기술자 포함),
물체(라벨·중심·크기·색), 벽에 걸린 얼굴 사진(블러 대상). 청소 뒤(after)에는 일부 물체를 알려진
벡터만큼 옮기고, 하나는 치우고(removed), 하나는 새로 둔다(added). after 장면은 before와 다른 좌표계
(알려진 닮음 변환 `x_b = s R x_a + t`)로 적어 정렬을 시험한다.

사진은 물체 3D 상자의 8 꼭짓점을 투영한 상자를 먼 것부터 칠한 단순 그림이다 (가림은 덧칠로 생긴다).
탐지 정답은 상자가 화면 안에 다 들어오고 절반 이상 보이는 물체만 낸다.

선택 기능 (기본은 모두 꺼져 있고, 끄면 사진·점·카메라가 예전과 한 비트도 다르지 않다):
- 척도 기준 마커 (`SceneMarker`, `floor_marker`): 크기를 아는 ArUco 마커를 바닥에 원근에 맞게
  그린다. 물체보다 먼저 그려 물체가 가린다. 정답은 before 좌표의 3D 꼭짓점 4개와 한 변 길이(m).
- 겉모습이 바뀌는 벽 (`changed_wall_points`): 벽 y=+3의 점은 청소 뒤 기술자가 달라진다 (닦인 벽).
  위치는 그대로라 그 벽만 보는 프레임은 한쪽 복원으로만 풀린다.
- 바디캠 경로 (`generate_ego_path`): 방 안을 걷는 1인칭 카메라의 프레임별 정답 자세(before 좌표)와
  물체·벽이 after 상태로 바뀐 시각. 프레임 그림은 사진과 같은 방식이고 얼굴 사진은 그리지 않는다
  (블러를 거친 영상으로 친다). 벽에 바짝 붙어 특징이 모자란 구간과 닦인 벽만 보는 구간이 있다.

좌표: 미터, before 세계 좌표는 z가 위. 카메라는 COLMAP 규약 (`x_cam = R x + t`, y 아래, z 앞).
`dlp_scene3d`에 의존하지 않는다 (정답은 numpy 배열과 기본 타입만).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.video import write_video

F64 = NDArray[np.float64]
WIDTH, HEIGHT = 640, 480
FOCAL = 520.0
DESC_DIM = 32
# 척도 기준 마커의 기본 사전 (cv2.aruco 이름). 칸이 가장 커서 작은 그림에서도 읽힌다
MARKER_DICTIONARY = "DICT_4X4_50"
# 바디캠 (사진과 다른 내부 파라미터: 위치 추정이 세션 캘리브레이션을 쓰는지 가른다)
EGO_WIDTH, EGO_HEIGHT = 640, 480
EGO_FOCAL = 400.0
# 청소로 겉모습이 바뀌는 벽(y = +3) 점의 key
WALL_KEY = "wall"
_WALL_Y = 3.0
# 바디캠 위치 추정 시험용 장면 (`generate_ego_scene`): 고정 구조 점 수, 닦인 벽에 더하는 점 수
EGO_LANDMARKS = 900
EGO_WALL_POINTS = 120
# 마커를 그릴 때의 초과 표본 배율 (경계를 부드럽게: 모서리 검출이 부화소로 맞게)
_SUPERSAMPLE = 4
_MARKER_CELL_PX = 40


@dataclass(frozen=True)
class SceneCamera:
    """합성 카메라 하나 (그 단계 좌표계)."""

    name: str
    k: F64
    r: F64
    t: F64


@dataclass(frozen=True)
class SceneObject:
    """합성 물체 하나. key는 전후를 잇는 정답 ID, size는 (가로, 세로, 높이) 미터."""

    key: str
    label: str
    center: F64
    size: F64
    color: tuple[int, int, int]


@dataclass(frozen=True)
class SceneMarker:
    """척도 기준 ArUco 마커 하나 (ADR 0067).

    corners는 before 좌표의 3D 꼭짓점 (4, 3), 마커 그림의 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래
    순서다 (`cv2.aruco` 검출이 내는 모서리 순서). side_m은 검은 테두리 바깥 한 변 길이(m)이고 그
    밖에 한 칸 너비의 흰 여백을 그린다.
    """

    dictionary: str
    marker_id: int
    side_m: float
    corners: F64


@dataclass
class PhaseTruth:
    """한 단계의 정답. 위치는 모두 그 단계 좌표계."""

    phase: str
    cameras: list[SceneCamera]
    objects: list[SceneObject]
    landmarks: F64
    descriptors: F64
    images: dict[str, NDArray[np.uint8]] = field(default_factory=dict[str, NDArray[np.uint8]])
    # 사진 이름 → (라벨, (x, y, w, h), 물체 key). 오탐은 key가 "fp-…"
    detections: dict[str, list[tuple[str, tuple[float, float, float, float], str]]] = field(
        default_factory=dict[str, list[tuple[str, tuple[float, float, float, float], str]]]
    )
    faces: dict[str, list[tuple[float, float, float, float]]] = field(
        default_factory=dict[str, list[tuple[float, float, float, float]]]
    )
    # landmarks와 같은 순서: 고정 구조는 "", 물체 표면 점은 물체 key, 바뀌는 벽은 WALL_KEY
    point_keys: list[str] = field(default_factory=list[str])


@dataclass
class ScenePair:
    """전후 정답. moves는 before 좌표의 이동 벡터, after_to_before는 (s, R, t).

    markers는 before 좌표의 마커(없으면 빈 목록), after_world_objects는 청소 뒤 물체를 before 좌표로
    적은 것이다 (after.objects는 after 좌표).
    """

    before: PhaseTruth
    after: PhaseTruth
    after_to_before: tuple[float, F64, F64]
    moves: dict[str, F64]
    removed: set[str]
    added: set[str]
    markers: list[SceneMarker] = field(default_factory=list[SceneMarker])
    after_world_objects: list[SceneObject] = field(default_factory=list[SceneObject])


@dataclass(frozen=True)
class EgoFrame:
    """바디캠 프레임 한 장의 정답 자세 (before 좌표, `x_cam = R x_b + t`)와 PTS 시각(ms)."""

    t_ms: int
    r: F64
    t: F64


@dataclass
class EgoTruth:
    """방을 지나는 바디캠의 정답 (`generate_ego_path`).

    change_ms: 물체 key(또는 `WALL_KEY`) → 그것이 after 상태가 된 시각(ms). 여기 없는 것은 끝까지
    before 상태다. images: 프레임 시각 → 그림 (render를 껐으면 비어 있다).
    """

    k: F64
    width: int
    height: int
    frames: list[EgoFrame]
    change_ms: dict[str, int]
    images: dict[int, NDArray[np.uint8]] = field(default_factory=dict[int, NDArray[np.uint8]])


def _rotation(axis: F64, angle: float) -> F64:
    """축-각 → 회전 행렬 (Rodrigues)."""
    rot, _ = cv2.Rodrigues((axis / np.linalg.norm(axis) * angle).reshape(3, 1))
    return np.asarray(rot, dtype=np.float64)


def _look_at(center: F64, target: F64) -> tuple[F64, F64]:
    """center에서 target을 보는 (R, t). 세계 z가 화면 위."""
    z = (target - center) / np.linalg.norm(target - center)
    x = np.cross(z, np.array([0.0, 0.0, 1.0]))
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])
    return r, -r @ center


def _intrinsics(focal: float, width: int, height: int) -> F64:
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def _cameras(
    n: int, start_deg: float, height: float, rng: np.random.Generator
) -> list[SceneCamera]:
    """방 가운데를 둘러싼 고리 위 카메라 n대 (before 좌표)."""
    k = _intrinsics(FOCAL, WIDTH, HEIGHT)
    out: list[SceneCamera] = []
    for i in range(n):
        a = np.deg2rad(start_deg + 360.0 * i / n)
        radius = 2.4 + rng.uniform(-0.15, 0.15)
        c = np.array([radius * np.cos(a), radius * np.sin(a), height + rng.uniform(-0.1, 0.1)])
        target = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), 0.35])
        r, t = _look_at(c, target)
        out.append(SceneCamera(f"img_{i:02d}", k, r, t))
    return out


def _objects() -> list[SceneObject]:
    """청소 전 물체 (같은 라벨 bucket 두 개는 색으로 구분)."""

    def obj(
        key: str,
        label: str,
        c: tuple[float, float],
        size: tuple[float, float, float],
        color: tuple[int, int, int],
    ) -> SceneObject:
        s = np.array(size)
        return SceneObject(key, label, np.array([c[0], c[1], s[2] / 2]), s, color)

    return [
        obj("bucket_blue", "bucket", (0.9, 0.3), (0.32, 0.32, 0.34), (30, 60, 220)),
        obj("bucket_orange", "bucket", (-0.8, 0.6), (0.32, 0.32, 0.34), (240, 140, 20)),
        obj("trash_bin", "trash_bin", (-0.3, -0.9), (0.35, 0.35, 0.55), (40, 160, 60)),
        obj("pillow", "pillow", (0.2, 0.9), (0.5, 0.35, 0.15), (220, 40, 160)),
        obj("towel", "towel", (0.6, -0.6), (0.4, 0.3, 0.08), (240, 230, 40)),
        obj("cup", "cup", (-0.9, -0.2), (0.12, 0.12, 0.14), (200, 30, 30)),
    ]


def floor_marker(
    marker_id: int,
    center: tuple[float, float],
    side_m: float,
    yaw_deg: float = 0.0,
    dictionary: str = MARKER_DICTIONARY,
) -> SceneMarker:
    """바닥(z = 0)에 놓은 마커. 위에서 내려다볼 때 그림이 뒤집히지 않게 꼭짓점을 시계 방향으로 둔다.

    Args:
        marker_id: 사전 안의 마커 번호.
        center: 마커 중심 (x, y) 미터 (before 좌표).
        side_m: 검은 테두리 바깥 한 변 (m).
        yaw_deg: z축 둘레 회전 (도).
        dictionary: `cv2.aruco` 사전 이름.
    """
    h = side_m / 2
    local = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    rot = _rotation(np.array([0.0, 0.0, 1.0]), np.deg2rad(yaw_deg)) if yaw_deg else np.eye(3)
    return SceneMarker(
        dictionary, marker_id, side_m, local @ rot.T + np.array([center[0], center[1], 0.0])
    )


def _corners(o: SceneObject) -> F64:
    """물체 상자의 8 꼭짓점."""
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64
    )
    return o.center + signs * o.size / 2


def _project(cam: SceneCamera, pts: F64) -> tuple[F64, F64]:
    """(픽셀, 깊이)."""
    c = pts @ cam.r.T + cam.t
    uv = (c @ cam.k.T)[:, :2] / c[:, 2:3]
    return uv, c[:, 2]


def _box(uv: F64) -> tuple[float, float, float, float]:
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))


def _inside(box: tuple[float, float, float, float], width: int, height: int) -> bool:
    x, y, w, h = box
    return x >= 1 and y >= 1 and x + w <= width - 1 and y + h <= height - 1


def _marker_sheet(marker: SceneMarker) -> tuple[NDArray[np.uint8], int]:
    """(흰 여백 한 칸을 두른 마커 그림, 여백 뺀 한 변의 칸 수)."""
    dictionary = cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, marker.dictionary)))
    cells = int(dictionary.markerSize) + 2
    inner = cv2.aruco.generateImageMarker(dictionary, marker.marker_id, cells * _MARKER_CELL_PX)
    sheet = np.full(((cells + 2) * _MARKER_CELL_PX,) * 2, 255, dtype=np.uint8)
    sheet[_MARKER_CELL_PX:-_MARKER_CELL_PX, _MARKER_CELL_PX:-_MARKER_CELL_PX] = inner
    return sheet, cells


def _draw_marker(img: NDArray[np.uint8], cam: SceneCamera, marker: SceneMarker) -> None:
    """마커를 원근에 맞게 img에 그린다 (꼭짓점이 모두 카메라 앞에 있을 때만).

    초과 표본한 큰 화면에 투영 변환으로 그린 뒤 면적 평균으로 줄여, 경계 화소가 덮인 비율만큼
    섞인다. 투영한 화소 좌표는 OpenCV 규약(왼쪽 위 화소의 중심이 (0, 0))으로 본다.
    """
    sheet, cells = _marker_sheet(marker)
    centre = marker.corners.mean(axis=0)
    outer = centre + (marker.corners - centre) * (cells + 2) / cells
    uv, depth = _project(cam, outer)
    if (depth <= 0.05).any():
        return
    height, width = img.shape[:2]
    big = (width * _SUPERSAMPLE, height * _SUPERSAMPLE)
    n = sheet.shape[0]
    # 화소 중심 좌표: 그림의 바깥 모서리는 (-0.5, -0.5)와 (n - 0.5, n - 0.5)
    src = np.array([[-0.5, -0.5], [n - 0.5, -0.5], [n - 0.5, n - 0.5], [-0.5, n - 0.5]])
    dst = (uv + 0.5) * _SUPERSAMPLE - 0.5
    warp = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    drawn = cv2.warpPerspective(sheet, warp, big, flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.full_like(sheet, 255), warp, big, flags=cv2.INTER_LINEAR)
    small = cv2.resize(drawn, (width, height), interpolation=cv2.INTER_AREA).astype(np.float64)
    alpha = cv2.resize(cover, (width, height), interpolation=cv2.INTER_AREA).astype(np.float64)
    alpha = (alpha / 255.0)[:, :, None]
    mixed = img.astype(np.float64) * (1.0 - alpha) + small[:, :, None] * alpha
    img[:] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _draw_faces(
    img: NDArray[np.uint8], cam: SceneCamera, face_quads: Sequence[F64]
) -> list[tuple[float, float, float, float]]:
    """벽의 얼굴 사진을 그리고 그 상자를 돌려준다 (화면 안에 다 들어온 것만)."""
    height, width = img.shape[:2]
    faces: list[tuple[float, float, float, float]] = []
    for quad in face_quads:
        uv, z = _project(cam, quad)
        if (z > 0).all():
            box = _box(uv)
            if _inside(box, width, height):
                x, y, w, h = (round(v) for v in box)
                img[y : y + h, x : x + w] = (225, 180, 150)
                img[y + h // 3 : y + h // 3 + 3, x + w // 4 : x + w // 4 + 3] = 20
                img[y + h // 3 : y + h // 3 + 3, x + 3 * w // 4 - 3 : x + 3 * w // 4] = 20
                faces.append(box)
    return faces


def _draw_view(
    cam: SceneCamera,
    objects: Sequence[SceneObject],
    face_quads: Sequence[F64],
    markers: Sequence[SceneMarker],
    width: int,
    height: int,
) -> tuple[
    NDArray[np.uint8],
    list[tuple[str, tuple[float, float, float, float], str]],
    list[tuple[float, float, float, float]],
]:
    """카메라 한 대의 (그림, 탐지 정답, 얼굴 상자). 바닥·얼굴·마커 뒤에 먼 물체부터 덧칠한다."""
    img = np.full((height, width, 3), 150, dtype=np.uint8)
    img[height // 2 :] = (120, 110, 100)
    ids = np.full((height, width), -1, dtype=np.int64)
    faces = _draw_faces(img, cam, face_quads)
    for marker in markers:
        _draw_marker(img, cam, marker)
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    order = sorted(
        range(len(objects)),
        key=lambda i: -_project(cam, objects[i].center[None])[1][0],
    )
    for i in order:
        uv, z = _project(cam, _corners(objects[i]))
        if (z <= 0).any():
            continue
        box = _box(uv)
        x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
        x2, y2 = (
            min(width, round(box[0] + box[2])),
            min(height, round(box[1] + box[3])),
        )
        if x2 <= x1 or y2 <= y1:
            continue
        img[y1:y2, x1:x2] = objects[i].color
        ids[y1:y2, x1:x2] = i
        boxes.append((i, box))
    dets: list[tuple[str, tuple[float, float, float, float], str]] = []
    for i, box in boxes:
        visible = float((ids == i).sum()) / max(box[2] * box[3], 1.0)
        if _inside(box, width, height) and visible >= 0.5:
            dets.append((objects[i].label, box, objects[i].key))
    return img, dets, faces


def _render(phase: PhaseTruth, face_quads: list[F64], markers: Sequence[SceneMarker]) -> None:
    """사진·탐지 정답·얼굴 정답을 채운다."""
    for cam in phase.cameras:
        img, dets, faces = _draw_view(cam, phase.objects, face_quads, markers, WIDTH, HEIGHT)
        phase.images[cam.name] = img
        phase.detections[cam.name] = dets
        phase.faces[cam.name] = faces


def _landmarks(rng: np.random.Generator, n: int) -> F64:
    """벽(±3 m)·바닥·천장의 고정 구조 점."""
    pts: list[F64] = []
    for _ in range(n):
        side = rng.integers(0, 6)
        u, v = rng.uniform(-3, 3), rng.uniform(0, 2.6)
        p = {
            0: (3.0, u, v),
            1: (-3.0, u, v),
            2: (u, 3.0, v),
            3: (u, -3.0, v),
            4: (u, rng.uniform(-3, 3), 0.0),
            5: (u, rng.uniform(-3, 3), 2.6),
        }[int(side)]
        pts.append(np.asarray(p, dtype=np.float64))
    return np.stack(pts)


def _surface_points(o: SceneObject, rng: np.random.Generator, n: int) -> F64:
    """물체 표면 근처 점 (물체와 함께 움직인다 → 정렬의 이상점)."""
    return o.center + rng.uniform(-0.5, 0.5, (n, 3)) * o.size


def _to_after(points: F64, s: float, r: F64, t: F64) -> F64:
    """before 좌표 → after 좌표 (x_a = Rᵀ (x_b - t) / s)."""
    return (points - t) @ r / s


def generate_scene_pair(
    seed: int = 0,
    n_cameras: int = 14,
    *,
    markers: Sequence[SceneMarker] = (),
    n_landmarks: int = 300,
    changed_wall_points: int = 0,
) -> ScenePair:
    """정답을 아는 청소 전후 장면 한 쌍.

    after 변화 (before 좌표, 미터): pillow (-0.6, +0.3, 0), cup (0, +0.6, +0.4), bucket_orange
    (-0.3, -0.6, 0)만큼 이동, towel 치움, spray_bottle 새로 둠. bucket_blue·trash_bin은 그대로.

    Args:
        seed: 난수 시드 (같으면 같은 결과).
        n_cameras: 단계마다 사진 수.
        markers: 두 단계 사진에 그릴 척도 기준 마커 (before 좌표, 청소 중 그대로 둔다).
        n_landmarks: 고정 구조 점 수.
        changed_wall_points: 0보다 크면 벽 y=+3을 "청소로 겉모습이 바뀐 벽"으로 만든다. 그 벽의
            고정 구조 점과 이 수만큼 더한 점은 두 단계에서 위치가 같고 기술자가 다르다
            (`point_keys`가 `WALL_KEY`).
    """
    rng = np.random.default_rng(seed)
    s, r = 1.25, _rotation(np.array([0.2, 0.3, 0.9]), np.deg2rad(35.0))
    t = np.array([0.4, -0.7, 0.2])
    moves = {
        "pillow": np.array([-0.6, 0.3, 0.0]),
        "cup": np.array([0.0, 0.6, 0.4]),
        "bucket_orange": np.array([-0.3, -0.6, 0.0]),
    }
    before_objs = _objects()
    added = SceneObject(
        "spray_bottle",
        "spray_bottle",
        np.array([0.0, -0.1, 0.13]),
        np.array([0.1, 0.1, 0.26]),
        (130, 60, 200),
    )
    after_world = [
        SceneObject(o.key, o.label, o.center + moves.get(o.key, np.zeros(3)), o.size, o.color)
        for o in before_objs
        if o.key != "towel"
    ] + [added]
    static = _landmarks(rng, n_landmarks)
    static_desc = rng.normal(size=(len(static), DESC_DIM))
    obj_pts = {o.key: _surface_points(o, rng, 12) for o in [*before_objs, added]}
    obj_desc = {k: rng.normal(size=(len(v), DESC_DIM)) for k, v in obj_pts.items()}
    face_quad = np.array([[-3.0 + 1e-3, y, z] for y in (0.2, 0.5) for z in (0.6, 1.0)])
    face_quads = [face_quad, face_quad * np.array([-1, 1, 1]) + np.array([0, -0.6, 0])]

    def phase_points(objs: list[SceneObject], shift: dict[str, F64]) -> tuple[F64, F64, list[str]]:
        pts = [static] + [obj_pts[o.key] + shift.get(o.key, np.zeros(3)) for o in objs]
        desc = [static_desc] + [obj_desc[o.key] for o in objs]
        keys = [""] * len(static) + [o.key for o in objs for _ in range(len(obj_pts[o.key]))]
        return np.concatenate(pts), np.concatenate(desc), keys

    b_pts, b_desc, b_keys = phase_points(before_objs, {})
    a_pts_world, a_desc, a_keys = phase_points(after_world, moves)
    before = PhaseTruth("before", _cameras(n_cameras, 0.0, 1.5, rng), before_objs, b_pts, b_desc)
    after_cams_world = _cameras(n_cameras, 11.0, 1.65, rng)
    marker_list = list(markers)
    _render(before, face_quads, marker_list)
    world_after = PhaseTruth("after", after_cams_world, after_world, a_pts_world, a_desc)
    _render(world_after, face_quads, marker_list)
    after_cams = [SceneCamera(c.name, c.k, c.r @ r, (c.r @ t + c.t) / s) for c in after_cams_world]
    after_objs = [
        SceneObject(o.key, o.label, _to_after(o.center[None], s, r, t)[0], o.size / s, o.color)
        for o in after_world
    ]
    noisy = a_desc + rng.normal(scale=0.02, size=a_desc.shape)
    if changed_wall_points > 0:
        b_pts, b_desc, b_keys, a_pts_world, noisy, a_keys = _change_wall(
            seed, changed_wall_points, (b_pts, b_desc, b_keys), (a_pts_world, noisy, a_keys)
        )
    before.landmarks, before.descriptors, before.point_keys = b_pts, b_desc, b_keys
    after = PhaseTruth(
        "after",
        after_cams,
        after_objs,
        _to_after(a_pts_world, s, r, t),
        noisy,
        world_after.images,
        world_after.detections,
        world_after.faces,
        a_keys,
    )
    return ScenePair(
        before,
        after,
        (s, r, t),
        moves,
        {"towel"},
        {"spray_bottle"},
        marker_list,
        after_world,
    )


def _change_wall(
    seed: int,
    extra: int,
    before: tuple[F64, F64, list[str]],
    after: tuple[F64, F64, list[str]],
) -> tuple[F64, F64, list[str], F64, F64, list[str]]:
    """벽 y=+3의 점을 "닦기 전·후 겉모습이 다른 점"으로 바꾸고 extra개를 더한다.

    두 단계에서 위치는 같고 기술자는 서로 무관하다. 기본 난수열을 건드리지 않게 따로 시드한 난수를
    쓴다. 입력·출력의 after 위치는 before 좌표다 (호출자가 after 좌표로 옮긴다).
    """
    rng = np.random.default_rng([seed, 67])
    b_pts, b_desc, b_keys = before
    a_pts, a_desc, a_keys = after
    more = np.stack(
        [rng.uniform(-3, 3, extra), np.full(extra, _WALL_Y), rng.uniform(0, 2.6, extra)], axis=1
    )
    a_desc = a_desc.copy()
    b_keys, a_keys = list(b_keys), list(a_keys)
    for i, key in enumerate(a_keys):
        if key == "" and abs(float(a_pts[i, 1]) - _WALL_Y) < 1e-9:
            a_desc[i] = rng.normal(size=DESC_DIM)
            a_keys[i] = WALL_KEY
    for i, key in enumerate(b_keys):
        if key == "" and abs(float(b_pts[i, 1]) - _WALL_Y) < 1e-9:
            b_keys[i] = WALL_KEY
    return (
        np.concatenate([b_pts, more]),
        np.concatenate([b_desc, rng.normal(size=(extra, DESC_DIM))]),
        b_keys + [WALL_KEY] * extra,
        np.concatenate([a_pts, more]),
        np.concatenate([a_desc, rng.normal(size=(extra, DESC_DIM))]),
        a_keys + [WALL_KEY] * extra,
    )


# 바디캠 경로의 길잡이 점: (진행 비율 0~1, 카메라 중심, 보는 곳). 사이는 부드럽게 잇는다.
# 0.52~0.58은 벽 x=-3에 바짝 붙어 고정 구조 점이 거의 안 보인다 (특징 부족 구간).
# 0.88 뒤는 벽 y=+3만 가까이서 본다 (닦인 뒤라 after 복원으로만 풀린다).
_EGO_WAYPOINTS: tuple[tuple[float, tuple[float, float, float], tuple[float, float, float]], ...] = (
    (0.00, (1.6, -1.0, 1.35), (-0.5, 0.3, 0.4)),
    (0.15, (1.2, 1.0, 1.35), (-0.6, -0.5, 0.4)),
    (0.30, (-0.6, 1.4, 1.35), (0.4, -0.6, 0.4)),
    (0.45, (-1.6, 0.2, 1.35), (0.8, 0.0, 0.4)),
    (0.52, (-2.7, 0.3, 1.30), (-3.0, 0.3, 1.25)),
    (0.58, (-2.7, 0.3, 1.30), (-3.0, 0.3, 1.25)),
    (0.66, (-1.4, -1.0, 1.35), (0.5, 0.3, 0.4)),
    (0.80, (0.4, -1.4, 1.35), (-0.2, 0.6, 0.4)),
    (0.88, (0.0, 1.2, 1.35), (0.0, 3.0, 1.3)),
    (1.00, (0.3, 1.3, 1.35), (0.3, 3.0, 1.2)),
)
# 물체·벽이 after 상태가 되는 시각 (진행 비율)
_EGO_CHANGES: tuple[tuple[str, float], ...] = (
    ("pillow", 0.25),
    ("cup", 0.38),
    ("bucket_orange", 0.50),
    ("towel", 0.62),
    ("spray_bottle", 0.70),
    (WALL_KEY, 0.82),
)


def _ego_pose(progress: float) -> tuple[F64, F64]:
    """진행 비율의 (카메라 중심, 보는 곳). 길잡이 점 사이를 smoothstep으로 잇는다."""
    for (p0, c0, g0), (p1, c1, g1) in itertools.pairwise(_EGO_WAYPOINTS):
        if progress <= p1:
            u = (progress - p0) / (p1 - p0)
            w = u * u * (3 - 2 * u)
            centre = (1 - w) * np.asarray(c0) + w * np.asarray(c1)
            target = (1 - w) * np.asarray(g0) + w * np.asarray(g1)
            return centre, target
    return np.asarray(_EGO_WAYPOINTS[-1][1]), np.asarray(_EGO_WAYPOINTS[-1][2])


def ego_objects(pair: ScenePair, change_ms: dict[str, int], t_ms: int) -> list[SceneObject]:
    """시각 t_ms에 방에 있는 물체 (before 좌표). 바뀐 시각이 지난 물체는 청소 뒤 상태다."""
    after = {o.key: o for o in pair.after_world_objects}
    out: list[SceneObject] = []
    for o in pair.before.objects:
        changed = o.key in change_ms and change_ms[o.key] <= t_ms
        if not changed:
            out.append(o)
        elif o.key in after:
            out.append(after[o.key])
    out += [
        o
        for o in pair.after_world_objects
        if o.key in pair.added and o.key in change_ms and change_ms[o.key] <= t_ms
    ]
    return out


def generate_ego_path(
    pair: ScenePair,
    seed: int = 0,
    *,
    duration_ms: int = 12_000,
    render: bool = True,
) -> EgoTruth:
    """방을 걸어 지나는 바디캠의 프레임별 정답 자세와 장면 변화 시각.

    프레임 간격은 90~110 ms로 흔들린다 (가변 프레임레이트). 손떨림처럼 중심에 ±1 cm, 보는 곳에
    ±3 cm의 흔들림을 더한다. 그림(render)은 사진과 같은 방식으로 그리되 얼굴 사진은 그리지 않는다.

    Args:
        pair: `generate_scene_pair` 결과 (마커가 있으면 프레임에도 그린다).
        seed: 난수 시드 (장면 난수열과 따로다).
        duration_ms: 영상 길이.
        render: 프레임 그림을 만들지.
    """
    rng = np.random.default_rng([seed, 68])
    k = _intrinsics(EGO_FOCAL, EGO_WIDTH, EGO_HEIGHT)
    change_ms = {key: round(p * duration_ms) for key, p in _EGO_CHANGES}
    truth = EgoTruth(k, EGO_WIDTH, EGO_HEIGHT, [], change_ms)
    t_ms = 0
    while t_ms < duration_ms:
        centre, target = _ego_pose(t_ms / duration_ms)
        centre = centre + rng.uniform(-0.01, 0.01, 3)
        target = target + rng.uniform(-0.03, 0.03, 3)
        r, t = _look_at(centre, target)
        truth.frames.append(EgoFrame(t_ms, r, t))
        if render:
            cam = SceneCamera(f"ego_{t_ms}", k, r, t)
            objects = ego_objects(pair, change_ms, t_ms)
            truth.images[t_ms] = _draw_view(cam, objects, [], pair.markers, EGO_WIDTH, EGO_HEIGHT)[
                0
            ]
        t_ms += int(rng.choice(np.array([90, 100, 100, 110])))
    return truth


def generate_ego_scene(
    seed: int = 0,
    *,
    markers: Sequence[SceneMarker] = (),
    duration_ms: int = 12_000,
    render: bool = True,
) -> tuple[ScenePair, EgoTruth]:
    """바디캠 위치 추정 시험용 장면과 그 방을 지나는 바디캠 경로.

    기본 장면보다 고정 구조 점을 촘촘히 두고(`EGO_LANDMARKS`) 벽 y=+3을 닦인 벽으로 만든다
    (`EGO_WALL_POINTS`): 프레임 하나에 보이는 점이 위치 추정 문턱을 넉넉히 넘고, 끝 구간은 after
    복원으로만 풀린다.
    """
    pair = generate_scene_pair(
        seed, markers=markers, n_landmarks=EGO_LANDMARKS, changed_wall_points=EGO_WALL_POINTS
    )
    return pair, generate_ego_path(pair, seed, duration_ms=duration_ms, render=render)


def _phase_json(phase: PhaseTruth, out: Path) -> dict[str, object]:
    """사진을 PNG로 쓰고 그 단계 정답 JSON을 만든다 (사진은 파일 sha256으로 가리킨다)."""
    d = out / phase.phase
    d.mkdir(parents=True, exist_ok=True)
    photos: list[dict[str, object]] = []
    for cam in phase.cameras:
        path = d / f"{cam.name}.png"
        ok, buf = cv2.imencode(".png", cv2.cvtColor(phase.images[cam.name], cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"PNG 인코딩 실패: {path}")
        path.write_bytes(buf.tobytes())
        photos.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "k": cam.k.tolist(),
                "r": cam.r.tolist(),
                "t": cam.t.tolist(),
                "width": WIDTH,
                "height": HEIGHT,
                "detections": [
                    {"label": lb, "box": list(b), "key": k}
                    for lb, b, k in phase.detections[cam.name]
                ],
                "faces": [list(b) for b in phase.faces[cam.name]],
            }
        )
    return {
        "photos": photos,
        "points": phase.landmarks.tolist(),
        "descriptors": phase.descriptors.tolist(),
        "point_keys": list(phase.point_keys),
        "objects": {o.key: {"label": o.label, "center": o.center.tolist()} for o in phase.objects},
    }


def _ego_json(ego: EgoTruth, out: Path) -> dict[str, object]:
    """바디캠 영상을 MP4(PTS = 정답 시각 ms)로 쓰고 정답 JSON을 만든다."""
    if set(ego.images) != {f.t_ms for f in ego.frames}:
        raise ValueError("바디캠 프레임 그림이 없습니다 (generate_ego_path render=True)")
    path = out / "ego" / "bodycam.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        path,
        ((f.t_ms, ego.images[f.t_ms]) for f in ego.frames),
        width=ego.width,
        height=ego.height,
    )
    return {
        "file": "ego/bodycam.mp4",
        "width": ego.width,
        "height": ego.height,
        "k": ego.k.tolist(),
        "frames": [{"t_ms": f.t_ms, "r": f.r.tolist(), "t": f.t.tolist()} for f in ego.frames],
        "change_ms": dict(ego.change_ms),
    }


def write_scene_pair(
    pair: ScenePair,
    out: Path,
    *,
    ego: EgoTruth | None = None,
    sfm_unit_scale: float = 1.0,
) -> Path:
    """out/before/*.png, out/after/*.png, out/truth.json을 쓰고 truth.json 경로를 돌려준다.

    Args:
        pair: 전후 정답.
        out: 출력 디렉터리.
        ego: 있으면 out/ego/bodycam.mp4와 truth의 `ego` 절(프레임별 정답 자세, 변화 시각)도 쓴다.
        sfm_unit_scale: 1이 아니면 truth에 적어, 정답 stub SfM이 좌표에 이 배율을 곱해 미터가 아닌
            복원을 흉내 내게 한다 (정답 좌표 자체는 미터 그대로 둔다).
    """
    s, r, t = pair.after_to_before
    truth: dict[str, object] = {
        "metric": True,
        "before": _phase_json(pair.before, out),
        "after": _phase_json(pair.after, out),
        "after_to_before": {"s": s, "r": r.tolist(), "t": t.tolist()},
        "moves": {k: v.tolist() for k, v in pair.moves.items()},
        "removed": sorted(pair.removed),
        "added": sorted(pair.added),
        "markers": [
            {
                "dictionary": m.dictionary,
                "id": m.marker_id,
                "side_m": m.side_m,
                "corners": m.corners.tolist(),
            }
            for m in pair.markers
        ],
    }
    if sfm_unit_scale != 1.0:
        truth["sfm_unit_scale"] = sfm_unit_scale
    if ego is not None:
        truth["ego"] = _ego_json(ego, out)
    path = out / "truth.json"
    path.write_text(json.dumps(truth), encoding="utf-8")
    return path
