"""결과 미리보기 영상 스크립트(scripts/preview_session.py) 검사.

정답을 아는 합성 블러 시나리오(WP2) 영상 몇 프레임으로 돈다.
- 출력 프레임은 구간 안 원본 프레임 전부이고 PTS 간격이 원본과 같다 (VFR 유지, 프레임 번호 x
  간격 금지).
- 블러 박스는 블러를 적용하지 않고 테두리만 그리고, 문턱 미만 객체 박스는 그리지 않는다.
- 키프레임 사이 박스는 선형 보간하고 마지막 키프레임 뒤에는 유지 시간까지만 보인다.
DB는 쓰지 않는다 (라벨을 직접 넘긴다).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from dlp_fixtures.video import generate_blur_scenario, read_frame_times
from dlp_schema import repo_root
from dlp_schema.labels import BoxKeyframe, LabelRecord, Provenance, Source
from dlp_schema.testing import hand_track_payload, make_label

ROOT = repo_root()


def _load() -> ModuleType:
    """scripts/preview_session.py를 모듈로 읽는다 (scripts는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location(
        "preview_session", ROOT / "scripts/preview_session.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclass가 모듈을 sys.modules에서 찾으므로 실행 전에 등록한다
    sys.modules["preview_session"] = module
    spec.loader.exec_module(module)
    return module


def _box(
    label_id: str, class_id: str, conf: float, version: str, kfs: list[BoxKeyframe]
) -> LabelRecord:
    return make_label(
        {"kind": "box_track", "entity_id": f"{class_id}_01", "class_id": class_id,
         "keyframes": [k.model_dump() for k in kfs]},
        label_id=label_id, t_start_ms=kfs[0].t_ms, t_end_ms=kfs[-1].t_ms, stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version=version), confidence=conf,
    )  # fmt: skip


def test_render_keeps_source_frames_and_pts(tmp_path: Path) -> None:
    """구간 [100, 500] ms의 원본 프레임이 모두, 원본 PTS 간격 그대로 나온다."""
    ps = _load()
    sc = generate_blur_scenario(duration_ms=800, width=320, height=240)
    video = tmp_path / "in.mp4"
    sc.write(video)
    times = sc.frame_times
    hand = make_label(
        hand_track_payload("right_hand", times[2:6]),
        label_id="h1", t_start_ms=times[2], t_end_ms=times[5], stream_id="bodycam",
    )  # fmt: skip
    ov = ps.build_overlays([*sc.labels, hand], stream_id="bodycam", start_ms=0, end_ms=800)
    out = tmp_path / "out.mp4"
    n = ps.render(
        video, out, ov, ps.DrawOptions(width=160), session_id="syn-blur",
        start_ms=100, end_ms=500, rotate=180, expect_size=(320, 240),
    )  # fmt: skip
    window = [t for t in times if 100 <= t <= 500]
    assert n == len(window)
    assert read_frame_times(out) == [t - window[0] for t in window]


def test_draw_outlines_blur_and_hides_low_score_boxes() -> None:
    """블러 대상은 테두리만(안쪽은 원본 그대로), 문턱 미만 객체 박스는 그리지 않는다."""
    ps = _load()
    face = make_label(
        {"kind": "blur_track", "target": "face",
         "keyframes": [{"t_ms": 0, "x": 40, "y": 40, "w": 80, "h": 60}]},
        label_id="b1", t_start_ms=0, t_end_ms=0, stream_id="bodycam",
    )  # fmt: skip
    low = _box("o1", "sponge", 0.1, "owlv2-x", [BoxKeyframe(t_ms=0, x=200, y=150, w=60, h=50)])
    ov = ps.build_overlays([face, low], stream_id="bodycam", start_ms=0, end_ms=10)
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    out = ps.draw_frame(img, 0.0, ov, ps.DrawOptions(width=320, min_score=0.3), session_id="s")
    assert out.shape[1] == 320 and out.shape[0] > 240  # 아래 패널
    edge = out[40, 80].astype(int)  # 위 테두리 (안티에일리어싱이라 근사 비교)
    assert np.abs(edge - np.array(ps.BLUR_COLORS["face"])).max() < 60
    assert tuple(out[70, 80]) == (128, 128, 128)  # 안쪽은 가리지 않는다
    assert tuple(out[150, 230]) == (128, 128, 128)  # 문턱 미만 박스는 없다


def test_box_interpolation_and_hold() -> None:
    """키프레임 사이는 선형 보간, 마지막 키프레임 뒤는 hold_ms까지만."""
    ps = _load()
    kfs = [BoxKeyframe(t_ms=0, x=0, y=0, w=10, h=10), BoxKeyframe(t_ms=100, x=100, y=0, w=10, h=10)]
    ov = ps.build_overlays(
        [_box("o1", "rag", 0.9, "owlv2-x", kfs)], stream_id="bodycam", start_ms=0, end_ms=500
    )
    track = ov.objects[0]
    assert ps.box_at(track, 50, ps.DrawOptions())[0] == 50
    assert ps.box_at(track, 150, ps.DrawOptions(hold_ms=0)) is None
    assert ps.box_at(track, 150, ps.DrawOptions(hold_ms=60))[0] == 100


def test_masks_are_filled_translucent_with_hand_colours() -> None:
    """마스크(ADR 0059)는 반투명 채움: 손은 왼손 초록, 객체는 클래스 색, 마스크 밖은 그대로이고
    가장 가까운 키프레임이 mask_tol_ms 밖이면 그리지 않는다."""
    from dlp_media.rle import encode_rle

    ps = _load()
    hand = np.zeros((240, 320), bool)
    hand[100:140, 50:90] = True
    cup = np.zeros((240, 320), bool)
    cup[20:60, 200:260] = True
    labels = [
        make_label(
            {"kind": "mask_track", "entity_id": e, "class_id": c,
             "keyframes": [{"t_ms": 0, "rle": encode_rle(m).model_dump(), "score": 0.8}]},
            label_id=f"m-{e}", t_start_ms=0, t_end_ms=0, stream_id="bodycam",
        )
        for e, c, m in (("left_hand", "hand", hand), ("cup_01", "cup", cup))
    ]  # fmt: skip
    ov = ps.build_overlays(labels, stream_id="bodycam", start_ms=0, end_ms=10)
    assert {m.hand for m in ov.masks} == {"left", None}
    img = np.full((240, 320, 3), 128, dtype=np.uint8)
    opts = ps.DrawOptions(width=320, mask_tol_ms=100)
    out = ps.draw_frame(img, 0.0, ov, opts, session_id="s")
    green = np.array(ps.HAND_COLORS["left"], dtype=float)
    inside = out[120, 70].astype(float)
    expected = 128 * (1 - opts.mask_alpha) + green * opts.mask_alpha
    assert np.abs(inside - expected).max() <= 1  # 반투명 (원본과 색의 섞임)
    cup_px = out[40, 230].astype(float)
    assert (
        np.abs(
            cup_px
            - (128 * (1 - opts.mask_alpha) + np.array(ps.class_color("cup")) * opts.mask_alpha)
        ).max()
        <= 1
    )
    assert tuple(out[200, 300]) == (128, 128, 128)  # 마스크 밖
    later = ps.draw_frame(img, 500.0, ov, opts, session_id="s")
    assert tuple(later[120, 70]) == (128, 128, 128)  # 키프레임에서 너무 멀다
