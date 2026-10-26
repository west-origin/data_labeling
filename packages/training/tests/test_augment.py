"""학습 증강(`dlp_train.augment`)과 RLE(`dlp_media.rle`) 시험 (ADR 0057).

정답: 위치를 아는 합성 프레임 — 검은 화면의 흰 사각형(박스·마스크 정답)과 색 점(키포인트 정답).
변환마다 증강한 이미지에서 다시 찾은 사각형·점이 증강한 라벨과 맞아야 한다 (보간 오차 몇 px 안).
뒤집기는 손 판정·COCO 좌우 관절·3D 손 개체를 바꾸고, 3D 회전은 같은 내부 파라미터로 투영한 위치가
증강 화면의 점과 맞아야 한다. 골든·holdout 예제와 원본 버킷 픽셀은 거부한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlp_media.geometry import Transform, crop_resize, hflip, identity, perspective, rotate
from dlp_media.rle import decode_rle, encode_rle
from dlp_media.storage import LocalStore
from dlp_schema.dataset import Split
from dlp_schema.testing import make_label
from dlp_train.augment import (
    AugmentationInputError,
    BoxItem,
    KeypointItem,
    MaskItem,
    Ops,
    Point3DItem,
    Sample,
    augment,
    augment_examples,
    check_blurred_store,
    check_examples,
    policy_digest,
    sample_ops,
)
from dlp_train.extract import Example
from dlp_train.policy import load_policy

ROOT = Path(__file__).resolve().parents[3]
W, H = 320, 240
BOX = (100.0, 60.0, 40.0, 30.0)
DOT = (200.0, 150.0)
F = 160.0  # 3D 투영 초점 거리 (주점 화면 가운데)


def _sample() -> Sample:
    """흰 사각형(BOX), 빨간 점(DOT), 그 사각형 마스크, 오른손 골격(손목 = DOT), COCO 왼 손목(DOT),
    오른손 손목 3D (DOT로 투영되는 점)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[60:90, 100:140] = 255
    img[148:153, 198:203] = (255, 0, 0)
    mask = np.zeros((H, W), dtype=bool)
    mask[60:90, 100:140] = True
    hand = np.zeros((21, 3))
    hand[:, :2] = DOT
    hand[:, 2] = 2
    coco = np.zeros((17, 3))
    coco[9] = (*DOT, 2)  # 왼 손목
    z = 0.5
    xyz = ((DOT[0] - W / 2) * z / F, (DOT[1] - H / 2) * z / F, z)
    return Sample(
        image=img,
        boxes=(BoxItem("mop", BOX),),
        keypoints=(KeypointItem("hand21", "right", hand), KeypointItem("coco17", None, coco)),
        masks=(MaskItem("mop", mask),),
        points3d=(Point3DItem("right_hand", "wrist", xyz),),
    )


def _ops(t: Transform, k1: float = 0.0) -> Ops:
    """기하 변환만 (광학 효과 없음)."""
    return Ops(t, 0.0, 0.0, 0.0, 0.0, 0, None, 0, 0.0, (), k1)


def _white(img: np.ndarray) -> tuple[float, float, float, float]:
    """흰 화소(세 채널 200 초과)의 외접 박스."""
    ys, xs = np.nonzero((img > 200).all(axis=2))
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    )


def _red(img: np.ndarray) -> tuple[float, float]:
    """빨간 화소 중심 (픽셀 중심 좌표)."""
    ys, xs = np.nonzero((img[:, :, 0] > 120) & (img[:, :, 1] < 60) & (img[:, :, 2] < 60))
    return (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)


TRANSFORMS = {
    "hflip": lambda: hflip(W, H),
    "rotate": lambda: rotate(W, H, 7.0),
    "crop": lambda: crop_resize(W, H, (30.0, 20.0, 260.0, 195.0)),
    "perspective": lambda: perspective(W, H, np.array([[6, 4], [-5, 3], [-4, -6], [5, -3]], float)),
    "distortion": lambda: identity(W, H),
}


@pytest.mark.parametrize("name", list(TRANSFORMS))
def test_labels_follow_pixels(name: str) -> None:
    """변환마다 박스는 흰 사각형을 덮고, 마스크는 흰 화소와 거의 같고, 손목 점은 빨간 점이다."""
    k1 = 0.08 if name == "distortion" else 0.0
    out = augment(_sample(), _ops(TRANSFORMS[name](), k1))
    [box] = out.boxes
    found = _white(out.image)
    assert box.box[0] <= found[0] + 2 and box.box[1] <= found[1] + 2
    assert box.box[0] + box.box[2] >= found[0] + found[2] - 2
    assert box.box[1] + box.box[3] >= found[1] + found[3] - 2
    white = (out.image > 200).all(axis=2)
    [mask] = out.masks
    assert (mask.mask ^ white).sum() <= 0.1 * white.sum()
    hand = out.keypoints[0]
    assert tuple(hand.points[0, :2]) == pytest.approx(_red(out.image), abs=1.5)


def test_flip_swaps_hand_coco_sides_and_3d() -> None:
    """뒤집기: 오른손 → 왼손, COCO 왼 손목(9) → 오른 손목(10), 3D 손 개체 교환·x 반전."""
    s = _sample()
    out = augment(s, _ops(hflip(W, H)))
    hand, coco = out.keypoints
    assert hand.hand == "left"
    assert coco.points[10, 2] == 2 and coco.points[9, 2] == 0
    assert coco.points[10, 0] == pytest.approx(W - DOT[0])
    [p] = out.points3d
    assert p.entity == "left_hand" and p.xyz[0] == pytest.approx(-s.points3d[0].xyz[0])
    assert out.geometry_3d_exact


def test_rotation_moves_3d_consistently() -> None:
    """회전한 3D 점을 같은 내부 파라미터로 투영하면 증강 화면의 빨간 점 위치다."""
    out = augment(_sample(), _ops(rotate(W, H, 9.0)))
    [p] = out.points3d
    u = F * p.xyz[0] / p.xyz[2] + W / 2
    v = F * p.xyz[1] / p.xyz[2] + H / 2
    assert (u, v) == pytest.approx(_red(out.image), abs=1.5)
    offsets = np.array([[6, 4], [-5, 3], [-4, -6], [5, -3]], float)
    persp = augment(_sample(), _ops(perspective(W, H, offsets)))
    assert not persp.geometry_3d_exact  # 원근은 3D를 바꾸지 않고 정확하지 않다고 표시한다


def test_cropped_out_points_and_boxes() -> None:
    """자르기로 화면 밖에 나간 점은 visibility 0, 박스는 빠진다."""
    out = augment(_sample(), _ops(crop_resize(W, H, (0.0, 0.0, 90.0, 60.0))))
    assert out.boxes == ()
    assert (out.keypoints[0].points[:, 2] == 0).all()


def test_photometric_changes_pixels_not_labels() -> None:
    """광학·1인칭 효과(밝기·대비·흐림·잡음·JPEG·움직임 흐림·반사광)는 라벨을 바꾸지 않는다."""
    s = _sample()
    ops = Ops(identity(W, H), 0.1, 0.2, 1.0, 5.0, 3, 60, 7, 30.0, ((0.5, 0.5, 0.05, 0.02),), 0.0)
    out = augment(s, ops)
    assert not np.array_equal(out.image, s.image)
    assert out.boxes == s.boxes
    assert np.array_equal(out.keypoints[0].points, s.keypoints[0].points)


def test_policy_sampling_is_deterministic_and_hashed() -> None:
    """같은 정책·같은 표본 순서면 같은 증강본, 정책 해시는 값이 바뀌면 바뀐다."""
    policy = load_policy(ROOT).augmentation
    a = augment_examples([_sample()], policy)
    b = augment_examples([_sample()], policy)
    assert len(a) == policy.copies
    assert all(
        np.array_equal(x.image, y.image) and x.boxes == y.boxes for x, y in zip(a, b, strict=True)
    )
    other = policy.model_copy(update={"seed": policy.seed + 1})
    assert policy_digest(other) != policy_digest(policy)
    rng = np.random.default_rng(0)
    ops = sample_ops(rng, policy, W, H)
    assert ops.transform.out_w == W and ops.transform.out_h == H
    # 반사광 크기·흐림 폭은 정책 값이다 (코드 고정값 아님)
    e = policy.egocentric
    assert ops.highlight_soft == e.highlight_soft
    for _cx, _cy, rx, ry in ops.highlights:
        assert e.highlight_rx_range[0] <= rx <= e.highlight_rx_range[1]
        assert e.highlight_ry_range[0] <= ry <= e.highlight_ry_range[1]
    soft = policy.model_copy(update={"egocentric": e.model_copy(update={"highlight_soft": 0.02})})
    assert policy_digest(soft) != policy_digest(policy)
    with pytest.raises(ValueError, match="highlight_rx_range"):
        type(e).model_validate({**e.model_dump(), "highlight_rx_range": (0.06, 0.01)})


def _example(split: Split) -> Example:
    """분할만 다른 예제."""
    label = make_label({"kind": "box_track", "entity_id": "m1", "class_id": "mop",
                        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 2, "h": 2}]},
                       stream_id="bodycam")  # fmt: skip
    return Example("s1", split, "added", label, None)


def test_inputs_only_train_val_and_blurred(tmp_path: Path) -> None:
    """골든·holdout 예제와 원본 버킷 저장소를 거부한다."""
    splits = load_policy(ROOT).splits
    check_examples([_example(Split.TRAIN), _example(Split.VAL)], splits)
    for bad in (Split.GOLDEN, Split.HOLDOUT):
        with pytest.raises(AugmentationInputError):
            check_examples([_example(bad)], splits)
    check_blurred_store(LocalStore(tmp_path, "dlp-labeling"), "dlp-raw")
    with pytest.raises(AugmentationInputError):
        check_blurred_store(LocalStore(tmp_path, "dlp-raw"), "dlp-raw")


def test_rle_round_trip() -> None:
    """COCO 압축 RLE 왕복 (빈 마스크, 첫 화소가 참인 마스크, 무작위 마스크)."""
    rng = np.random.default_rng(1)
    for mask in (
        np.zeros((5, 7), dtype=bool),
        np.ones((3, 4), dtype=bool),
        rng.random((37, 53)) > 0.7,
    ):
        assert np.array_equal(decode_rle(encode_rle(mask)), mask)
    # pycocotools 형식 기준값: 3x3 가운데 한 화소 → 열 우선 길이 [4, 1, 4] → "414"
    center = np.zeros((3, 3), dtype=bool)
    center[1, 1] = True
    assert encode_rle(center).counts == "414"
