"""표식·시각 표기·확대 조각 테스트 (`dlp_aireview.overlay`, WP17.2).

상자 보간이 outside를 지키는지, 표식이 상자 둘레에만 그려지는지, 작은 박스에 확대 조각이 생기는지,
보낼 이미지가 정렬 단위의 배수 크기이고 같은 입력이면 같은 JPEG인지, 시간 묶음 프레임에는 라벨을
그리지 않는지(시각 표기만) 본다.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_aireview import overlay as overlay_module
from dlp_aireview.overlay import crop, draw_marks, draw_stamp
from dlp_aireview.plan import plan_session
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.sensors import SensorContext
from dlp_aireview.tracks import box_at
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import BoxKeyframe
from dlp_schema.ontology import Ontology

from .conftest import Scene, golden_units, images_for, with_policy

GOLDEN = Path(__file__).parent / "golden"


def test_box_interpolation_respects_outside() -> None:
    """키프레임 사이는 선형 보간, outside 구간·첫 키프레임 전·마지막 뒤는 None."""
    kfs = (
        BoxKeyframe(t_ms=0, x=0, y=0, w=10, h=10),
        BoxKeyframe(t_ms=100, x=10, y=20, w=20, h=10),
        BoxKeyframe(t_ms=200, x=0, y=0, w=0, h=0, outside=True),
        BoxKeyframe(t_ms=300, x=5, y=5, w=5, h=5),
    )
    assert box_at(kfs, 50) == (5, 10, 15, 10)
    assert box_at(kfs, 150) == (10, 20, 20, 10)  # 다음이 outside면 앞 키프레임 그대로
    assert box_at(kfs, 250) is None
    assert box_at(kfs, 300) == (5, 5, 5, 5)
    assert box_at(kfs, -1) is None and box_at(kfs, 301) is None


def test_marks_drawn_only_around_boxes() -> None:
    """표식은 상자 테두리·태그 근처에만 그리고 원본을 바꾸지 않는다. 시각 표기는 왼쪽 위에만."""
    img = np.full((120, 160, 3), 90, dtype=np.uint8)
    marked = draw_marks(img, [("B0", (60.0, 50.0, 30.0, 30.0), None)])
    assert (img == 90).all()
    changed = np.argwhere((marked != img).any(axis=2))
    assert changed.size and changed[:, 0].max() <= 82 and changed[:, 1].min() >= 57
    assert (marked[100:, :] == 90).all() and (marked[:, :40] == 90).all()
    stamped = draw_stamp(img, "F0 +0.0s")
    diff = np.argwhere((stamped != img).any(axis=2))
    assert diff[:, 0].max() < 40 and diff[:, 1].max() < 120


def test_crop_is_square_and_padded() -> None:
    """작은 상자 주변을 한 변 side의 정사각형으로 키운다 (모자라면 검은 여백)."""
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    img[20:30, 30:50] = 200
    piece = crop(img, (30.0, 20.0, 20.0, 10.0), 64)
    assert piece.shape == (64, 64, 3) and piece.max() == 200
    assert crop(img, (500.0, 500.0, 4.0, 4.0), 32).shape == (32, 32, 3)


def test_unit_images_size_marks_crops_and_determinism(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """보낼 이미지: 크기는 정렬 단위 배수, 같은 입력이면 같은 JPEG 해시다.

    공간 단위는 표식과 작은 박스 확대 조각(상한 안), 시간 단위는 시각 표기만 그린다.
    """
    units = plan_session(
        scene.session,
        scene.labels,
        policy,
        ontology,
        media=scene.media,
        sensor=SensorContext.absent(),
    )
    align = policy.image.align_px
    for u in units:
        images = images_for(u, policy)
        frames = [i for i in images if i.crop_of is None]
        crops = [i for i in images if i.crop_of is not None]
        assert [f.t_ms for f in frames] == list(u.frame_ms)
        assert all(f.w % align == 0 and f.h % align == 0 for f in frames)
        assert max(frames[0].w, frames[0].h) <= u.window.max_side_px
        if u.group is AiGroup.SPATIAL:
            assert 0 < len(crops) <= policy.spatial.crops.max
            assert all(
                c.label.startswith("B") and c.w == policy.spatial.crops.side_px for c in crops
            )
        else:
            assert not crops
        assert [i.image.sha256 for i in images] == [i.image.sha256 for i in images_for(u, policy)]
    spatial = next(u for u in units if u.group is AiGroup.SPATIAL)
    no_crops = with_policy(
        policy, spatial={"crops": policy.spatial.crops.model_copy(update={"enabled": False})}
    )
    assert all(i.crop_of is None for i in images_for(spatial, no_crops))
    plain = with_policy(policy, spatial={"marks": False})
    temporal = next(u for u in units if u.group is AiGroup.TEMPORAL)
    assert (
        images_for(temporal, plain)[0].image.sha256 == images_for(temporal, policy)[0].image.sha256
    )
    assert images_for(spatial, plain)[0].image.sha256 != images_for(spatial, policy)[0].image.sha256


# 판정 이미지 그리기가 바뀌었을 때 알리는 절차 (글 골든 `test_prompt.BUMP`와 같은 절차)
OVERLAY_BUMP = (
    "판정 이미지 그리기(dlp_aireview.overlay: 시각 도장 형식, 표식 색·선 두께·태그, 확대 조각 범위·"
    "크기, 줄이기)가 바뀌었다. 단위 키·판정 버전에는 그리기가 들어가지 않으므로 바꿨으면 "
    "dlp_aireview.policy.PROMPT_VERSION과 config/policies/aireview.yaml prompt_version을 함께 "
    "올리고 DLP_UPDATE_GOLDEN=1로 골든을 다시 쓴다 (같은 버전에 다른 그림의 판정이 섞이지 않게). "
    "정책 값(digest에 드는 절)이 바뀐 것이면 판정 버전이 이미 바뀌므로 골든만 다시 쓴다."
)


def test_drawn_images_match_golden_pixels(
    policy: AiReviewPolicy, ontology: Ontology, monkeypatch: pytest.MonkeyPatch
) -> None:
    """골든 단위(시간·공간)에 그린 이미지 픽셀이 골든과 같다 (JPEG 압축 전 RGB 배열의 해시).

    정답 근거: VLM이 보는 이미지 내용(도장 글자, set-of-marks, 확대 조각)은 overlay 코드로만
    정해지고 판정 버전·단위 키에는 들어가지 않는다. 같은 코드 안의 결정성(`test_unit_images_…`)
    만으로는 그리기를 바꿔도 테스트가 통과해, 이미 기록된 단위와 새 그림의 단위가 같은 버전에
    섞인다.
    JPEG 바이트가 아니라 압축 전 배열을 비교해 libjpeg 차이에 흔들리지 않게 한다.
    """
    drawn: list[str] = []
    real = overlay_module.encode_jpeg

    def capture(rgb: Any, quality: int) -> Any:
        arr = np.ascontiguousarray(rgb)
        drawn.append(hashlib.sha256(repr(arr.shape).encode() + arr.tobytes()).hexdigest())
        return real(rgb, quality)

    monkeypatch.setattr(overlay_module, "encode_jpeg", capture)
    units, _ = golden_units(policy, ontology)
    found: dict[str, list[str]] = {}
    for group in (AiGroup.TEMPORAL, AiGroup.SPATIAL):
        unit = next(u for u in units if u.group is group)
        drawn.clear()
        images_for(unit, policy)
        found[group.value] = list(drawn)
    assert found["spatial"] and found["temporal"]
    text = json.dumps(found, indent=2, sort_keys=True) + "\n"
    path = GOLDEN / "images.json"
    if os.environ.get("DLP_UPDATE_GOLDEN") == "1":
        path.write_text(text, "utf-8")
    assert path.read_text("utf-8") == text, f"images.json: {OVERLAY_BUMP}"
