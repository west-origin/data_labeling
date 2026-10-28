"""학습용 데이터 증강 (ADR 0057). 픽셀과 모든 라벨 종류를 같은 변환으로 함께 바꾼다.

입력 경계 (어기면 `AugmentationInputError`):
- 학습 예제는 데이터셋 버전의 학습·검증 분할(`training.yaml splits`)에서만 온다. 골든·holdout 예제가
  섞이면 거부한다 (`check_examples`).
- 픽셀은 라벨링 버킷의 블러본에서만 온다. 원본 버킷 저장소를 넘기면 거부한다
  (`check_blurred_store`). 증강 결과는 메모리 안의 학습 표본이며 라벨 이력·DB·판매 내보내기에 쓰지
  않는다 (내보내기 경로가 없다).

변환 (정책 `training.yaml augmentation`, 표본마다 `seed`에서 결정적으로 뽑는다):
- 기하 (`dlp_media.geometry`, 테스트 시점 증강과 같은 행렬): 좌우 뒤집기, 작은 회전, 배율·자르기,
  원근 흔들기.
- 광학: 밝기·대비, 가우스 흐림, 잡음, JPEG 압축 손상.
- 1인칭: 움직임 흐림(직선 커널), 젖은 표면·스테인리스 반사광(밝은 타원), 렌즈 방사 왜곡 변화(k1).

라벨 변환 (`augment`):
- 박스·블러 박스: 네 모서리를 옮긴 외접 박스를 화면 안으로 자른다. 남지 않으면 뺀다 (화면 밖).
- 마스크(RLE): 같은 변환으로 최근접 보간해 옮긴다. 좌우 뒤집기면 손 마스크 키
  (left_hand↔right_hand)를 바꾼다.
- 2D 키포인트: 점을 옮기고 화면 밖으로 나간 점은 visibility 0. 좌우 뒤집기면 손 판정(left↔right)을
  바꾸고 COCO 17점은 좌우 관절 번호를 맞바꾼다 (hand21 번호는 해부학 이름이라 그대로).
- 3D 궤적(카메라 좌표): 좌우 뒤집기는 x를 뒤집고 손·팔 개체(left_hand↔right_hand,
  left_arm↔right_arm)를 바꾼다. 회전(화면 가운데 중심)은 같은 각도로 x·y를 돌린다.
  배율·자르기·원근·렌즈 왜곡은 3D를 바꾸지 않는다 (내부 파라미터가 바뀐 것이다). 배율·자르기는
  K를 바꾸는데 `Sample`에 바뀐 K가 없다: 증강본에서 3D를 투영하려면 K' = matrix · K가 필요하다.
  이 근사는 주점이 화면 가운데일 때만 정확하다.
  `Sample.geometry_3d_exact`는 원근·렌즈 왜곡이 들어갔는지만 보고(거짓) 주점 위치는 보지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.geometry import (
    COCO17_FLIP,
    Box,
    Transform,
    clip_box,
    crop_resize,
    hflip,
    identity,
    map_box,
    map_points,
    perspective,
    rotate,
    swap_hand,
    warp_image,
)
from dlp_media.storage import ObjectStore
from dlp_schema.dataset import Split
from dlp_train.extract import Example
from dlp_train.policy import AugmentationPolicy

Image = NDArray[np.uint8]


class AugmentationInputError(ValueError):
    """증강 입력 경계 위반 (골든·holdout 예제, 원본 버킷 픽셀)."""


def check_examples(examples: Iterable[Example], splits: Sequence[Split]) -> None:
    """예제가 모두 학습·검증 분할인지 본다. 아니면 `AugmentationInputError`."""
    bad = sorted({e.session_id for e in examples if e.split not in splits})
    if bad:
        raise AugmentationInputError(f"학습·검증 분할 밖 세션은 증강하지 않습니다: {bad[:5]}")


def check_blurred_store(store: ObjectStore, raw_bucket: str) -> None:
    """픽셀 저장소가 원본 버킷이 아닌지 본다 (블러본만). 원본이면 `AugmentationInputError`."""
    if store.bucket == raw_bucket:
        raise AugmentationInputError("증강 픽셀은 라벨링 버킷의 블러본에서만 받습니다 (원본 버킷)")


def policy_digest(policy: AugmentationPolicy) -> str:
    """증강 설정 해시 12자 (학습 모델 버전·MLflow에 들어간다)."""
    data = json.dumps(policy.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- 표본


@dataclass(frozen=True)
class BoxItem:
    """박스 하나 (객체·도구·블러). kind: box_track | blur_track."""

    key: str
    box: Box
    kind: str = "box_track"


@dataclass(frozen=True)
class KeypointItem:
    """골격 하나. points: (N, 3) [x, y, visibility]."""

    skeleton: str
    hand: str | None
    points: NDArray[np.float64]


@dataclass(frozen=True)
class MaskItem:
    """마스크 하나 (H, W) 불리언."""

    key: str
    mask: NDArray[np.bool_]


@dataclass(frozen=True)
class Point3DItem:
    """카메라 좌표 3D 점 하나 (m)."""

    entity: str
    part: str | None
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Sample:
    """학습 표본: 블러본 프레임 하나와 그 시각의 라벨."""

    image: Image
    boxes: tuple[BoxItem, ...] = ()
    keypoints: tuple[KeypointItem, ...] = ()
    masks: tuple[MaskItem, ...] = ()
    points3d: tuple[Point3DItem, ...] = ()
    # 원근·렌즈 왜곡이 없으면 참. 주점 위치·배율·자르기로 바뀐 K는 보지 않는다 (모듈 docstring)
    geometry_3d_exact: bool = True
    ops: tuple[str, ...] = field(default=())  # 적용한 변환 이름 (기록용)


# ---------------------------------------------------------------- 변환 뽑기


@dataclass(frozen=True)
class Ops:
    """표본 하나에 적용할 변환 (결정적 재현용 값만)."""

    transform: Transform
    brightness: float
    contrast: float
    blur_sigma: float
    noise_std: float
    noise_seed: int
    jpeg_quality: int | None
    motion_px: int
    motion_deg: float
    highlights: tuple[tuple[float, float, float, float], ...]  # (cx, cy, rx, ry) 비율
    k1: float
    # 반사광 가장자리 흐림 폭 (화면 너비 비율, 정책 egocentric.highlight_soft)
    highlight_soft: float = 0.01


def _geometry(rng: np.random.Generator, policy: AugmentationPolicy, w: int, h: int) -> Transform:
    """기하 변환 하나 (뒤집기 → 회전 → 배율·자르기 → 원근 순 합성)."""
    g = policy.geometric
    t = identity(w, h)
    if rng.random() < g.hflip_p:
        t = t.then(hflip(w, h))
    if g.rotate_deg > 0:
        t = t.then(rotate(w, h, float(rng.uniform(-g.rotate_deg, g.rotate_deg))))
    lo, hi = g.scale_range
    s = float(rng.uniform(lo, hi))
    if s != 1.0:
        cw, ch = w / s, h / s
        x0, y0 = (
            float(rng.uniform(min(0.0, w - cw), max(0.0, w - cw))),
            float(rng.uniform(min(0.0, h - ch), max(0.0, h - ch))),
        )
        t = t.then(crop_resize(w, h, (x0, y0, cw, ch)))
    if g.perspective_frac > 0:
        offsets = rng.uniform(-g.perspective_frac, g.perspective_frac, (4, 2)) * [w, h]
        t = t.then(perspective(w, h, offsets))
    return t


def sample_ops(rng: np.random.Generator, policy: AugmentationPolicy, w: int, h: int) -> Ops:
    """정책 범위 안에서 변환 값을 뽑는다 (같은 rng 상태면 같은 결과)."""
    p, e = policy.photometric, policy.egocentric
    jlo, jhi = p.jpeg_quality_range
    n_spec = int(rng.integers(0, e.highlights_max + 1)) if rng.random() < e.highlight_p else 0
    return Ops(
        transform=_geometry(rng, policy, w, h),
        brightness=float(rng.uniform(-p.brightness, p.brightness)),
        contrast=float(rng.uniform(-p.contrast, p.contrast)),
        blur_sigma=float(rng.uniform(0, p.blur_sigma_max)),
        noise_std=float(rng.uniform(0, p.noise_std_max)),
        noise_seed=int(rng.integers(0, 2**31)),
        jpeg_quality=int(rng.integers(jlo, jhi + 1)) if rng.random() < p.jpeg_p else None,
        motion_px=int(rng.integers(0, e.motion_blur_px_max + 1)),
        motion_deg=float(rng.uniform(0, 180)),
        # 반사광 타원 (중심·반지름 모두 화면 너비·높이 비율, 반지름 범위는 정책)
        highlights=tuple(
            (
                float(rng.uniform(0, 1)),
                float(rng.uniform(0, 1)),
                float(rng.uniform(*e.highlight_rx_range)),
                float(rng.uniform(*e.highlight_ry_range)),
            )
            for _ in range(n_spec)
        ),
        k1=float(rng.uniform(-e.distortion_k1_max, e.distortion_k1_max)),
        highlight_soft=e.highlight_soft,
    )


# ---------------------------------------------------------------- 픽셀


def _distort_points(pts: NDArray[np.float64], k1: float, w: int, h: int) -> NDArray[np.float64]:
    """방사 왜곡 x_d = c + (x - c)(1 + k1 r²), r은 화면 중심에서 모서리까지 거리(대각선 절반)로
    정규화한 거리."""
    c = np.array([w / 2, h / 2])
    norm = math.hypot(w / 2, h / 2)
    d = (pts - c) / norm
    r2 = (d**2).sum(axis=1, keepdims=True)
    return c + d * (1 + k1 * r2) * norm


def _distort_image(img: NDArray[np.uint8], k1: float, nearest: bool = False) -> NDArray[np.uint8]:
    """방사 왜곡 이미지: 출력 화소마다 원래 위치를 고정점 반복으로 찾아 다시 표본한다."""
    h, w = img.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    out_pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1)
    src = out_pts.copy()
    for _ in range(5):  # x_u ← x_d - (distort(x_u) - x_u)
        src = out_pts - (_distort_points(src, k1, w, h) - src)
    mx = (src[:, 0] - 0.5).reshape(h, w).astype(np.float32)
    my = (src[:, 1] - 0.5).reshape(h, w).astype(np.float32)
    mode = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    out = cv2.remap(img, mx, my, mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.asarray(out, dtype=np.uint8)


def _motion_blur(img: Image, px: int, deg: float) -> Image:
    """길이 px, 방향 deg 직선 커널 흐림."""
    if px < 2:
        return img
    line = np.zeros((px, px), dtype=np.float32)
    line[px // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((px / 2 - 0.5, px / 2 - 0.5), deg, 1.0)
    kernel = np.asarray(cv2.warpAffine(line, rot, (px, px)), dtype=np.float32)
    kernel /= max(float(kernel.sum()), 1e-6)
    return np.asarray(cv2.filter2D(img, -1, kernel), dtype=np.uint8)


def _highlights(
    img: Image, spots: Sequence[tuple[float, float, float, float]], soft: float
) -> Image:
    """밝은 타원 반사광 (젖은 손·스테인리스). soft: 가장자리 흐림 폭 (화면 너비 비율)."""
    if not spots:
        return img
    h, w = img.shape[:2]
    glow = np.zeros((h, w), dtype=np.float32)
    for cx, cy, rx, ry in spots:
        cv2.ellipse(glow, (int(cx * w), int(cy * h)), (max(1, int(rx * w)), max(1, int(ry * h))),
                    0, 0, 360, 1.0, -1)  # fmt: skip
    # 가장자리를 부드럽게: 흐림 폭 = 화면 너비 x soft (최소 1 px, 해상도와 무관한 비율)
    glow = cv2.GaussianBlur(glow, (0, 0), max(1.0, soft * w))[..., None]
    out = img.astype(np.float32) * (1 - glow) + 255.0 * glow
    return np.clip(out, 0, 255).astype(np.uint8)


def _photometric(img: Image, ops: Ops) -> Image:
    """밝기·대비 → 흐림 → 움직임 흐림 → 반사광 → 잡음 → JPEG."""
    out = img.astype(np.float32)
    out = (out - 128.0) * (1 + ops.contrast) + 128.0 + 255.0 * ops.brightness
    out8 = np.clip(out, 0, 255).astype(np.uint8)
    # sigma 0.3 px 이하 가우스 커널은 거의 항등이라 건너뛴다 (계산만 들고 화소가 거의 그대로)
    if ops.blur_sigma > 0.3:
        out8 = np.asarray(cv2.GaussianBlur(out8, (0, 0), ops.blur_sigma), dtype=np.uint8)
    out8 = _highlights(
        _motion_blur(out8, ops.motion_px, ops.motion_deg), ops.highlights, ops.highlight_soft
    )
    if ops.noise_std > 0:
        noise = np.random.default_rng(ops.noise_seed).normal(0, ops.noise_std, out8.shape)
        out8 = np.clip(out8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if ops.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", out8, [cv2.IMWRITE_JPEG_QUALITY, ops.jpeg_quality])
        if ok:
            out8 = np.asarray(cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), dtype=np.uint8)
    return out8


# ---------------------------------------------------------------- 라벨


def _map(t: Transform, k1: float, pts: NDArray[np.float64]) -> NDArray[np.float64]:
    """점 → 기하 변환 → 렌즈 왜곡."""
    moved = map_points(t, pts)
    return _distort_points(moved, k1, t.out_w, t.out_h) if k1 else moved


def _box(t: Transform, k1: float, box: Box) -> Box | None:
    """박스 → 변환한 둘레 점들의 외접 박스 (화면 안으로 자름, 없으면 None)."""
    x, y, bw, bh = box
    xs, ys = np.linspace(x, x + bw, 5), np.linspace(y, y + bh, 5)
    top = [(u, y) for u in xs]
    bottom = [(u, y + bh) for u in xs]
    sides = [(x, v) for v in ys] + [(x + bw, v) for v in ys]
    ring = np.array([*top, *bottom, *sides])
    p = _map(t, k1, ring) if k1 else None
    if p is None:
        return clip_box(map_box(t, box), t.out_w, t.out_h)
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    return clip_box((float(x0), float(y0), float(x1 - x0), float(y1 - y0)), t.out_w, t.out_h)


def _keypoints(item: KeypointItem, t: Transform, k1: float) -> KeypointItem:
    """골격 점 이동, 화면 밖 visibility 0, 뒤집기면 손 판정·COCO 좌우 관절 교환."""
    pts = item.points.copy()
    moved = _map(t, k1, pts[:, :2])
    out = np.column_stack([moved, pts[:, 2]])
    outside = (
        (moved[:, 0] < 0) | (moved[:, 0] >= t.out_w) | (moved[:, 1] < 0) | (moved[:, 1] >= t.out_h)
    )
    out[outside, 2] = 0
    if t.flipped and item.skeleton == "coco17":
        out = out[list(COCO17_FLIP)]
    hand = swap_hand(item.hand, t) if item.hand else None
    return KeypointItem(item.skeleton, hand, out)


# 좌우가 있는 착용자 개체: 손(3D·마스크 `<손>_hand`)과 팔(팔꿈치 3D `<손>_arm`), ADR 0058·0059
_SIDED_PARTS = ("hand", "arm")


def _swap_side(name: str, t: Transform) -> str:
    """좌우 뒤집기면 `<손>_hand`·`<손>_arm` 개체·마스크 키의 좌우를 바꾼다 (F-ETA-25).

    2D 골격의 손 판정(`swap_hand`)과 같은 표본에서 어긋나지 않게 한다. 다른 이름(객체 클래스
    등)은 그대로 둔다.
    """
    side, sep, part = name.partition("_")
    if not (t.flipped and sep and part in _SIDED_PARTS):
        return name
    return f"{swap_hand(side, t)}_{part}"


def _point3d(item: Point3DItem, t: Transform) -> tuple[Point3DItem, bool]:
    """카메라 좌표 3D 점 (뒤집기 x 반전·손 개체 교환, 회전은 x·y 회전). (점, 정확한가)."""
    a = t.matrix[:2, :2]
    exact = bool(np.allclose(t.matrix[2], [0, 0, 1]))
    x, y, z = item.xyz
    if exact:
        det = abs(float(np.linalg.det(a)))
        r = a / math.sqrt(det) if det > 0 else np.eye(2)
        x, y = (float(v) for v in r @ np.array([x, y]))
    return Point3DItem(_swap_side(item.entity, t), item.part, (x, y, z)), exact


def augment(sample: Sample, ops: Ops) -> Sample:
    """표본 하나에 변환을 적용한다 (픽셀·라벨 함께)."""
    t = ops.transform
    image = warp_image(sample.image, t)
    if ops.k1:
        image = np.asarray(_distort_image(image, ops.k1), dtype=np.uint8)
    image = _photometric(image, ops)
    boxes = tuple(
        replace(b, box=nb) for b in sample.boxes if (nb := _box(t, ops.k1, b.box)) is not None
    )
    masks: list[MaskItem] = []
    for m in sample.masks:
        warped = np.asarray(warp_image(m.mask.astype(np.uint8), t, nearest=True), dtype=np.uint8)
        if ops.k1:
            warped = _distort_image(warped, ops.k1, nearest=True)
        masks.append(MaskItem(_swap_side(m.key, t), warped.astype(bool)))
    points3d: list[Point3DItem] = []
    exact = sample.geometry_3d_exact and not ops.k1
    for p in sample.points3d:
        q, ok = _point3d(p, t)
        points3d.append(q)
        exact = exact and ok
    return Sample(
        image=image,
        boxes=boxes,
        keypoints=tuple(_keypoints(k, t, ops.k1) for k in sample.keypoints),
        masks=tuple(masks),
        points3d=tuple(points3d),
        geometry_3d_exact=exact,
        ops=(*sample.ops, t.name),
    )


def augment_examples(
    samples: Sequence[Sample], policy: AugmentationPolicy, *, offset: int = 0
) -> list[Sample]:
    """표본마다 `copies`개 증강본 (원본 포함하지 않음). 씨앗 = (정책 seed, 표본 번호, 사본 번호).

    같은 정책·같은 표본 순서면 결과가 같다 (CPU 결정적).
    """
    out: list[Sample] = []
    for i, s in enumerate(samples):
        h, w = s.image.shape[:2]
        for c in range(policy.copies):
            rng = np.random.default_rng([policy.seed, offset + i, c])
            out.append(augment(s, sample_ops(rng, policy, w, h)))
    return out
