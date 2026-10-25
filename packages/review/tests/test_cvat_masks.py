"""CVAT 마스크 왕복 (mask_track, ADR 0059). 서비스 없이 돈다.

정답: 정답 모양 합성 영상(`dlp_fixtures.shapes`)의 마스크로 만든 mask_track. CVAT 마스크 점 형식은
CVAT 데이터셋 변환기(`rle_to_mask`: 외접 박스 안 행 우선, 배경부터 번갈아 가는 연속 길이, 끝에
[왼, 위, 오른, 아래])를 이 시험 안에서 따로 구현해 대조한다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.shapes import generate_shape_scenario
from dlp_media.rle import decode_rle, encode_rle
from dlp_review.cvat import (
    CvatFormatError,
    CvatSchema,
    annotation_tracks,
    cvat_points_to_mask,
    from_cvat_tracks,
    label_spec,
    mask_to_cvat_points,
    quantize,
    to_cvat_shapes,
    to_cvat_tracks,
)
from dlp_schema.labels import LabelRecord, MaskKeyframe, MaskTrackPayload, Provenance, Source
from dlp_schema.testing import make_label


def _schema(names: list[str]) -> CvatSchema:
    """CVAT가 부여하는 것처럼 라벨·속성에 ID를 매긴다."""
    labels: list[dict[str, Any]] = []
    next_id = 100
    for i, spec in enumerate(label_spec(names)):
        attrs = [{**a, "id": next_id + j} for j, a in enumerate(spec["attributes"])]
        next_id += len(attrs)
        labels.append({**spec, "id": i + 1, "attributes": attrs})
    return CvatSchema.from_labels(labels)


def _cvat_rle_to_mask(points: list[float], h: int, w: int) -> NDArray[np.bool_]:
    """CVAT 변환기와 같은 방식의 독립 구현 (외접 박스 안 행 우선, 배경부터)."""
    x0, y0, x1, y1 = (int(v) for v in points[-4:])
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    flat = np.zeros(bw * bh, np.uint8)
    pos, value = 0, 0
    for v in points[:-4]:
        flat[pos : pos + int(v)] = value
        pos += int(v)
        value = 1 - value
    out = np.zeros((h, w), bool)
    out[y0 : y1 + 1, x0 : x1 + 1] = flat.reshape(bh, bw).astype(bool)
    return out


def _mask_labels() -> tuple[list[LabelRecord], list[int]]:
    """정답 컵·손 마스크 트랙 (모델 출처, 점수·derived_from 있음)과 프레임 시각."""
    sc = generate_shape_scenario(3, n_frames=6)
    times = sc.frame_times
    labels = [
        make_label(
            MaskTrackPayload(
                entity_id=e,
                class_id=c,
                derived_from=(f"src-{e}",),
                keyframes=tuple(
                    MaskKeyframe(t_ms=t, rle=encode_rle(sc.masks[e][t]), score=0.9)
                    for t in times[::2]
                ),
            ),
            label_id=f"m-{e}",
            stream_id="bodycam",
            t_start_ms=times[0],
            t_end_ms=times[4],
            provenance=Provenance(source=Source.MODEL, model_version="masks-x"),
            confidence=0.9,
        )
        for e, c in (("cup_01", "cup"), ("left_hand", "hand"))
    ]
    return labels, times


def test_cvat_mask_points_follow_the_cvat_format() -> None:
    """우리 CVAT 마스크 점이 CVAT 변환기 방식으로 같은 마스크가 되고 되돌리기도 정확하다."""
    sc = generate_shape_scenario(0)
    m = sc.masks["left_hand"][0]
    points = mask_to_cvat_points(m)
    assert np.array_equal(_cvat_rle_to_mask(points, sc.height, sc.width), m)
    assert np.array_equal(cvat_points_to_mask(points, (sc.height, sc.width)), m)
    with pytest.raises(ValueError, match="빈 마스크"):
        mask_to_cvat_points(np.zeros((4, 4), bool))
    with pytest.raises(CvatFormatError):
        cvat_points_to_mask([*points[:-5], points[-5] + 1, *points[-4:]], (sc.height, sc.width))


def test_mask_tracks_round_trip_losslessly_through_shape_mode() -> None:
    """마스크는 모양 모드로 가서 원래 라벨 ID로 다시 묶이고 내용이 quantize(원래)와 같다 (점수만
    빠진다). 트랙 목록에는 나가지 않는다."""
    labels, times = _mask_labels()
    schema = _schema(["cup", "hand"])
    assert to_cvat_tracks(labels, times, schema) == []
    shapes = to_cvat_shapes(labels, times, schema)
    assert len(shapes) == 6 and {s["type"] for s in shapes} == {"mask"}
    tracks = annotation_tracks({"tracks": [], "shapes": shapes, "tags": []}, len(times), schema)
    back = from_cvat_tracks(tracks, times, schema, "bodycam", new_box_kind="box_track")
    assert {b.origin_label_id for b in back} == {"m-cup_01", "m-left_hand"}
    for item in back:
        original = next(x for x in labels if x.label_id == item.origin_label_id)
        q = quantize(original)
        assert (item.payload, item.t_start_ms, item.t_end_ms) == (
            q.payload, q.t_start_ms, q.t_end_ms,
        )  # fmt: skip
        assert isinstance(item.payload, MaskTrackPayload)
        assert all(k.score is None for k in item.payload.keyframes)
        assert item.payload.derived_from == original.payload.derived_from  # type: ignore[union-attr]


def test_edited_and_new_masks_come_back_as_changes() -> None:
    """검수자가 한 마스크를 지우고(화소 몇 개) 새 마스크를 그리면: 고친 트랙은 내용이 달라지고,
    새 마스크 모양은 dlp 속성 없이 새 트랙(개체 `<라벨>_h<n>`, 크기는 화면 영상)이 된다."""
    labels, times = _mask_labels()
    schema = _schema(["cup", "hand"])
    shapes = to_cvat_shapes(labels[:1], times, schema)
    m = cvat_points_to_mask(shapes[0]["points"], (240, 320))
    ys, xs = np.nonzero(m)
    m[ys[:5], xs[:5]] = False
    shapes[0] = shapes[0] | {"points": mask_to_cvat_points(m)}
    new = np.zeros((240, 320), bool)
    new[10:20, 30:50] = True
    drawn = {
        "type": "mask", "frame": 1, "label_id": schema.label_ids["cup"],
        "points": mask_to_cvat_points(new), "attributes": [], "occluded": False,
    }  # fmt: skip
    tracks = annotation_tracks(
        {"tracks": [], "shapes": [*shapes, drawn], "tags": []}, len(times), schema
    )
    back = from_cvat_tracks(
        tracks, times, schema, "bodycam", new_box_kind="box_track", frame_size=(240, 320)
    )
    edited = next(b for b in back if b.origin_label_id == "m-cup_01")
    assert edited.payload != quantize(labels[0]).payload
    assert isinstance(edited.payload, MaskTrackPayload)
    first = edited.payload.keyframes[0].rle
    assert first is not None and np.array_equal(decode_rle(first), m)
    fresh = next(b for b in back if b.origin_label_id is None)
    assert isinstance(fresh.payload, MaskTrackPayload) and fresh.payload.class_id == "cup"
    assert fresh.payload.entity_id.startswith("cup_h") and fresh.t_start_ms == times[1]
    rle = fresh.payload.keyframes[0].rle
    assert rle is not None and np.array_equal(decode_rle(rle), new)


def test_privacy_tasks_reject_masks_and_box_tracks_reject_mask_shapes() -> None:
    """블러 검수에서 새 마스크는 받지 않고, 박스 트랙에 마스크 모양이 섞이면 멈춘다."""
    labels, times = _mask_labels()
    schema = _schema(["cup", "hand", "face"])
    drawn: dict[str, Any] = {**to_cvat_shapes(labels[:1], times, schema)[0], "attributes": []}
    tracks = annotation_tracks({"tracks": [], "shapes": [drawn], "tags": []}, len(times), schema)
    with pytest.raises(CvatFormatError, match="블러 검수"):
        from_cvat_tracks(
            tracks, times, schema, "bodycam", new_box_kind="blur_track", frame_size=(240, 320)
        )
    box = {**tracks[0], "attributes": [
        {"spec_id": schema.attr_ids[("cup", "dlp_meta")], "value": '{"kind": "box_track"}'}
    ]}  # fmt: skip
    with pytest.raises(CvatFormatError, match="맞지 않는"):
        from_cvat_tracks([box], times, schema, "bodycam", new_box_kind="box_track")
    with pytest.raises(ValueError, match="원본 해상도"):
        to_cvat_shapes(labels, times, schema, scale=(0.5, 0.5))
