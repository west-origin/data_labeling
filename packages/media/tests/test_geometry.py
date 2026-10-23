"""기하 변환(`dlp_media.geometry`) 시험: 픽셀과 라벨 좌표가 같이 움직이는가.

정답: 검은 화면에 흰 사각형(박스 정답)과 색 점(점 정답)을 그린 합성 프레임. 변환한 이미지에서 다시
찾은 사각형·점 위치가 `map_box`·`map_points` 결과와 같아야 한다 (보간 오차 2 px 안).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from dlp_media.geometry import (
    COCO17_FLIP,
    Transform,
    clip_box,
    crop_resize,
    hflip,
    identity,
    inside,
    map_box,
    map_points,
    parse_transform,
    perspective,
    rotate,
    scale,
    swap_hand,
    tta_support,
    warp_image,
)

W, H = 320, 240


def _scene() -> tuple[np.ndarray, tuple[float, float, float, float], tuple[float, float]]:
    """흰 사각형(박스 (100, 60, 40, 30))과 빨간 점(200, 150)을 그린 프레임."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[60:90, 100:140] = 255
    img[148:153, 198:203] = (255, 0, 0)
    return img, (100.0, 60.0, 40.0, 30.0), (200.0, 150.0)


def _white_box(img: np.ndarray) -> tuple[float, float, float, float]:
    """흰 화소(세 채널 모두 200 초과)의 외접 박스."""
    ys, xs = np.nonzero((img > 200).all(axis=2))
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() - xs.min() + 1),
        float(ys.max() - ys.min() + 1),
    )


def _red(img: np.ndarray) -> tuple[float, float]:
    """빨간 화소의 중심."""
    ys, xs = np.nonzero((img[:, :, 0] > 150) & (img[:, :, 1] < 80))
    return (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)


@pytest.mark.parametrize(
    "make",
    [
        lambda: hflip(W, H),
        lambda: scale(W, H, 0.75),
        lambda: crop_resize(W, H, (40.0, 30.0, 240.0, 180.0)),
        lambda: rotate(W, H, 8.0),
        lambda: perspective(W, H, np.array([[6, 4], [-5, 3], [-4, -6], [5, -3]], dtype=float)),
    ],
    ids=["hflip", "scale", "crop", "rotate", "perspective"],
)
def test_labels_move_with_pixels(make: Callable[[], Transform]) -> None:
    """변환한 이미지의 사각형·점 위치 = 변환한 라벨 위치 (회전·원근은 외접 박스라 크게 잡힌다)."""
    t = make()
    img, box, pt = _scene()
    out = warp_image(img, t)
    assert out.shape[:2] == (t.out_h, t.out_w)
    expected = map_box(t, box)
    found = _white_box(out)
    # 외접 박스는 실제 사각형을 덮는다 (회전·원근이면 조금 크다)
    assert found[0] >= expected[0] - 2 and found[1] >= expected[1] - 2
    assert found[0] + found[2] <= expected[0] + expected[2] + 2
    assert found[1] + found[3] <= expected[1] + expected[3] + 2
    if t.name in ("hflip", "scale", "crop"):
        assert found == pytest.approx(expected, abs=2.0)
    assert _red(out) == pytest.approx(tuple(map_points(t, np.array([pt]))[0]), abs=1.5)


def test_flip_semantics_and_inverse() -> None:
    """좌우 뒤집기: 손 판정이 바뀌고, 두 번 뒤집거나 역변환하면 제자리. COCO 좌우 짝은 대칭."""
    t = hflip(W, H)
    assert t.flipped and swap_hand("left", t) == "right" and swap_hand("right", t) == "left"
    assert swap_hand("left", scale(W, H, 0.5)) == "left"
    assert not t.then(t).flipped
    pts = np.array([[10.0, 20.0], [300.0, 200.0]])
    assert map_points(t.then(t), pts) == pytest.approx(pts)
    s = scale(W, H, 0.5)
    assert map_points(s.inverse(W, H), map_points(s, pts)) == pytest.approx(pts)
    assert all(COCO17_FLIP[COCO17_FLIP[i]] == i for i in range(17))
    assert COCO17_FLIP[9] == 10  # 왼 손목 ↔ 오른 손목


def test_clip_and_inside() -> None:
    """자르기 변환 뒤 화면 밖으로 나간 점·박스."""
    t = crop_resize(W, H, (0.0, 0.0, 160.0, 120.0))  # 왼쪽 위 1/4 확대
    assert list(inside(t, np.array([[10.0, 10.0], [200.0, 200.0]]))) == [True, False]
    assert clip_box(map_box(t, (150.0, 100.0, 40.0, 40.0)), W, H) == pytest.approx(
        (300.0, 200.0, 20.0, 40.0)
    )
    assert clip_box(map_box(t, (200.0, 200.0, 10.0, 10.0)), W, H) is None
    assert identity(W, H).name == "id"


def test_tta_support_counts_matching_views() -> None:
    """정답: 박스 A는 두 변환 모두에서 다시 찾음(1.0), 박스 B는 뒤집기에서만(0.5)."""
    a, b = (10.0, 10.0, 50.0, 50.0), (200.0, 100.0, 40.0, 40.0)
    f, s = hflip(W, H), scale(W, H, 0.5)
    views = [(f, [map_box(f, a), map_box(f, b)]), (s, [map_box(s, a)])]
    assert tta_support([a, b], views, W, H, 0.5) == [1.0, 0.5]
    assert tta_support([a], [], W, H, 0.5) == [1.0]


def test_parse_transform() -> None:
    """정책 이름 → 변환. 모르는 이름은 거부한다."""
    assert parse_transform("hflip", W, H).flipped
    assert parse_transform("scale:0.75", W, H).out_w == 240
    assert parse_transform("rot:5", W, H).name == "rot5"
    with pytest.raises(ValueError, match="TTA"):
        parse_transform("shear:1", W, H)
