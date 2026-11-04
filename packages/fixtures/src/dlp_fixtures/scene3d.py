"""청소 전후 다각도 사진의 합성 정답 (ADR 0066).

정답을 아는 방 하나를 만든다: 카메라(내부·외부 파라미터), 고정 구조 점(벽·바닥, 기술자 포함),
물체(라벨·중심·크기·색), 벽에 걸린 얼굴 사진(블러 대상). 청소 뒤(after)에는 일부 물체를 알려진
벡터만큼 옮기고, 하나는 치우고(removed), 하나는 새로 둔다(added). after 장면은 before와 다른 좌표계
(알려진 닮음 변환 `x_b = s R x_a + t`)로 적어 정렬을 시험한다.

사진은 물체 3D 상자의 8 꼭짓점을 투영한 상자를 먼 것부터 칠한 단순 그림이다 (가림은 덧칠로 생긴다).
탐지 정답은 상자가 화면 안에 다 들어오고 절반 이상 보이는 물체만 낸다.

좌표: 미터, before 세계 좌표는 z가 위. 카메라는 COLMAP 규약 (`x_cam = R x + t`, y 아래, z 앞).
`dlp_scene3d`에 의존하지 않는다 (정답은 numpy 배열과 기본 타입만).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
WIDTH, HEIGHT = 640, 480
FOCAL = 520.0
DESC_DIM = 32


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


@dataclass
class ScenePair:
    """전후 정답. moves는 before 좌표의 이동 벡터, after_to_before는 (s, R, t)."""

    before: PhaseTruth
    after: PhaseTruth
    after_to_before: tuple[float, F64, F64]
    moves: dict[str, F64]
    removed: set[str]
    added: set[str]


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


def _cameras(
    n: int, start_deg: float, height: float, rng: np.random.Generator
) -> list[SceneCamera]:
    """방 가운데를 둘러싼 고리 위 카메라 n대 (before 좌표)."""
    k = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]], dtype=np.float64)
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


def _inside(box: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    return x >= 1 and y >= 1 and x + w <= WIDTH - 1 and y + h <= HEIGHT - 1


def _render(phase: PhaseTruth, face_quads: list[F64]) -> None:
    """사진·탐지 정답·얼굴 정답을 채운다 (먼 물체부터 덧칠)."""
    for cam in phase.cameras:
        img = np.full((HEIGHT, WIDTH, 3), 150, dtype=np.uint8)
        img[HEIGHT // 2 :] = (120, 110, 100)
        ids = np.full((HEIGHT, WIDTH), -1, dtype=np.int64)
        faces: list[tuple[float, float, float, float]] = []
        for quad in face_quads:
            uv, z = _project(cam, quad)
            if (z > 0).all():
                box = _box(uv)
                if _inside(box):
                    x, y, w, h = (round(v) for v in box)
                    img[y : y + h, x : x + w] = (225, 180, 150)
                    img[y + h // 3 : y + h // 3 + 3, x + w // 4 : x + w // 4 + 3] = 20
                    img[y + h // 3 : y + h // 3 + 3, x + 3 * w // 4 - 3 : x + 3 * w // 4] = 20
                    faces.append(box)
        boxes: list[tuple[int, tuple[float, float, float, float]]] = []
        order = sorted(
            range(len(phase.objects)),
            key=lambda i: -_project(cam, phase.objects[i].center[None])[1][0],
        )
        for i in order:
            uv, z = _project(cam, _corners(phase.objects[i]))
            if (z <= 0).any():
                continue
            box = _box(uv)
            x1, y1 = max(0, round(box[0])), max(0, round(box[1]))
            x2, y2 = (
                min(WIDTH, round(box[0] + box[2])),
                min(HEIGHT, round(box[1] + box[3])),
            )
            if x2 <= x1 or y2 <= y1:
                continue
            img[y1:y2, x1:x2] = phase.objects[i].color
            ids[y1:y2, x1:x2] = i
            boxes.append((i, box))
        dets: list[tuple[str, tuple[float, float, float, float], str]] = []
        for i, box in boxes:
            visible = float((ids == i).sum()) / max(box[2] * box[3], 1.0)
            if _inside(box) and visible >= 0.5:
                dets.append((phase.objects[i].label, box, phase.objects[i].key))
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


def generate_scene_pair(seed: int = 0, n_cameras: int = 14) -> ScenePair:
    """정답을 아는 청소 전후 장면 한 쌍.

    after 변화 (before 좌표, 미터): pillow (-0.6, +0.3, 0), cup (0, +0.6, +0.4), bucket_orange
    (-0.3, -0.6, 0)만큼 이동, towel 치움, spray_bottle 새로 둠. bucket_blue·trash_bin은 그대로.
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
    static = _landmarks(rng, 300)
    static_desc = rng.normal(size=(len(static), DESC_DIM))
    obj_pts = {o.key: _surface_points(o, rng, 12) for o in [*before_objs, added]}
    obj_desc = {k: rng.normal(size=(len(v), DESC_DIM)) for k, v in obj_pts.items()}
    face_quad = np.array([[-3.0 + 1e-3, y, z] for y in (0.2, 0.5) for z in (0.6, 1.0)])
    face_quads = [face_quad, face_quad * np.array([-1, 1, 1]) + np.array([0, -0.6, 0])]

    def phase_points(objs: list[SceneObject], shift: dict[str, F64]) -> tuple[F64, F64]:
        pts = [static] + [obj_pts[o.key] + shift.get(o.key, np.zeros(3)) for o in objs]
        desc = [static_desc] + [obj_desc[o.key] for o in objs]
        return np.concatenate(pts), np.concatenate(desc)

    b_pts, b_desc = phase_points(before_objs, {})
    a_pts_world, a_desc = phase_points(after_world, moves)
    before = PhaseTruth("before", _cameras(n_cameras, 0.0, 1.5, rng), before_objs, b_pts, b_desc)
    after_cams_world = _cameras(n_cameras, 11.0, 1.65, rng)
    _render(before, face_quads)
    world_after = PhaseTruth("after", after_cams_world, after_world, a_pts_world, a_desc)
    _render(world_after, face_quads)
    after_cams = [SceneCamera(c.name, c.k, c.r @ r, (c.r @ t + c.t) / s) for c in after_cams_world]
    after_objs = [
        SceneObject(o.key, o.label, _to_after(o.center[None], s, r, t)[0], o.size / s, o.color)
        for o in after_world
    ]
    noisy = a_desc + rng.normal(scale=0.02, size=a_desc.shape)
    after = PhaseTruth(
        "after",
        after_cams,
        after_objs,
        _to_after(a_pts_world, s, r, t),
        noisy,
        world_after.images,
        world_after.detections,
        world_after.faces,
    )
    return ScenePair(before, after, (s, r, t), moves, {"towel"}, {"spray_bottle"})


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
        "objects": {o.key: {"label": o.label, "center": o.center.tolist()} for o in phase.objects},
    }


def write_scene_pair(pair: ScenePair, out: Path) -> Path:
    """out/before/*.png, out/after/*.png, out/truth.json을 쓰고 truth.json 경로를 돌려준다."""
    s, r, t = pair.after_to_before
    truth = {
        "metric": True,
        "before": _phase_json(pair.before, out),
        "after": _phase_json(pair.after, out),
        "after_to_before": {"s": s, "r": r.tolist(), "t": t.tolist()},
        "moves": {k: v.tolist() for k, v in pair.moves.items()},
        "removed": sorted(pair.removed),
        "added": sorted(pair.added),
    }
    path = out / "truth.json"
    path.write_text(json.dumps(truth), encoding="utf-8")
    return path
