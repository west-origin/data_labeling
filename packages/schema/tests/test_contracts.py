from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.export import ExportedLabel
from dlp_schema.labels import (
    SPATIAL_KINDS,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.session import LifecycleState, Session, StreamKind, can_transition
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session

BOX = {"kind": "box_track", "entity_id": "rag_01", "class_id": "rag",
       "keyframes": [{"t_ms": 0, "x": 1, "y": 2, "w": 3, "h": 4}]}  # fmt: skip
HAND21 = [{"x": 0.0, "y": 0.0, "visibility": 2}] * 21

MASK_FRAMES = [{"t_ms": 0, "rle": {"size": [4, 4], "counts": "04"}}, {"t_ms": 500, "outside": True}]
CAMERA_POSE = {"t_ms": 0, "x": 0, "y": 0, "z": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0}

PAYLOADS: list[dict[str, Any]] = [
    BOX,
    {"kind": "mask_track", "entity_id": "mop_01", "class_id": "mop", "part": "mop_head",
     "keyframes": MASK_FRAMES},
    {"kind": "keypoint_track", "entity_id": "right_hand", "skeleton": "hand21", "hand": "right",
     "keyframes": [{"t_ms": 0, "points": HAND21}]},
    {"kind": "blur_track", "target": "face",
     "keyframes": [{"t_ms": 0, "x": 0, "y": 0, "w": 9, "h": 9}]},
    {"kind": "trajectory3d", "entity_id": "camera", "frame": "world",
     "source_3d": "visual_inertial_slam", "samples": [CAMERA_POSE]},
    {"kind": "hand_state", "hand": "right", "contact_target_kind": "tool", "target_id": "mop_01",
     "grasp_type": "tool_grip", "role": "active"},
    action_payload(),
    {"kind": "segment", "segment_id": "sk01", "level": "skill", "ref_id": "wipe",
     "child_ids": ["a001"]},
    {"kind": "gap", "hand": "left", "gap_type": "idle"},
    {"kind": "object_state", "entity_id": "sink_01", "class_id": "sink",
     "attribute": "cleanliness", "value": "dirty"},
    {"kind": "coverage", "surface_id": "sink_01", "tool_id": "rag_01", "ratio": 0.85},
    {"kind": "event", "event_type": "slip", "severity": 2},
    {"kind": "relation", "subject_id": "rag_01", "subject_part": "cloth_face",
     "predicate": "contact", "object_id": "sink_01", "derived_by": "tool_surface_overlap_v1"},
    {"kind": "description", "segment_id": "sk01", "text": "걸레로 세면대를 닦는다"},
]  # fmt: skip


def _stream(kind: str) -> str | None:
    return "bodycam" if kind in SPATIAL_KINDS else None


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: p["kind"])
def test_every_payload_kind_roundtrips_through_json(payload: dict[str, Any]) -> None:
    label = make_label(payload, stream_id=_stream(payload["kind"]))
    again = LabelRecord.model_validate_json(label.model_dump_json())
    assert again == label
    assert again.kind == payload["kind"]


def test_unknown_fields_and_payload_kinds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_label({**BOX, "color": "red"}, stream_id="bodycam")
    with pytest.raises(ValidationError):
        make_label({"kind": "sticker"})


def test_times_must_be_ordered_and_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="t_start_ms"):
        make_label({"kind": "gap", "gap_type": "idle"}, t_start_ms=10, t_end_ms=5)
    with pytest.raises(ValidationError):
        make_label({"kind": "gap", "gap_type": "idle"}, created_at=datetime(2026, 1, 1))


def test_action_time_rules() -> None:
    with pytest.raises(ValidationError, match="순서"):
        make_label(action_payload(t_contact_start_ms=1_200, t_contact_end_ms=1_300))
    with pytest.raises(ValidationError, match="접촉 시작이 접촉 종료보다"):
        make_label(action_payload(t_contact_start_ms=900, t_contact_end_ms=400))
    with pytest.raises(ValidationError, match="접근 시작~종료"):
        make_label(action_payload(), t_start_ms=100)
    held = action_payload(t_contact_start_ms=None, t_contact_end_ms=None, contact_held=True)
    assert make_label(held).payload.kind == "action"


def test_keyframes_must_lie_inside_label_interval() -> None:
    late = {**BOX, "keyframes": [{"t_ms": 5_000, "x": 0, "y": 0, "w": 1, "h": 1}]}
    with pytest.raises(ValidationError, match="벗어났"):
        make_label(late, stream_id="bodycam")
    dup = {**BOX, "keyframes": [BOX["keyframes"][0], BOX["keyframes"][0]]}
    with pytest.raises(ValidationError, match="오름차순"):
        make_label(dup, stream_id="bodycam")


def test_spatial_labels_need_stream() -> None:
    with pytest.raises(ValidationError, match="stream_id"):
        make_label(BOX)
    # 3D 궤적도 영상 PTS 시각의 공간 라벨이다 (ADR 0019, 0022)
    with pytest.raises(ValidationError, match="stream_id"):
        make_label(PAYLOADS[4])
    assert "trajectory3d" in SPATIAL_KINDS


def test_exported_label_rejects_blur_payload() -> None:
    common: dict[str, Any] = {
        "label_id": "x1", "stream_id": "bodycam", "t_start_ms": 0, "t_end_ms": 0,
        "verification": "human_approved", "source": "human",
    }  # fmt: skip
    assert ExportedLabel.model_validate({**common, "payload": BOX}).payload.kind == "box_track"
    with pytest.raises(ValidationError, match="blur_track"):
        ExportedLabel.model_validate({**common, "payload": PAYLOADS[3]})


def test_keypoint_count_must_match_skeleton() -> None:
    bad = {**PAYLOADS[2], "keyframes": [{"t_ms": 0, "points": HAND21[:20]}]}
    with pytest.raises(ValidationError, match="21"):
        make_label(bad, stream_id="bodycam")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"contact_target_kind": "none"}, "비워야"),
        ({"target_id": None}, "target_id"),
        ({"contact_target_kind": "person", "target_id": None}, "body_part"),
    ],
)
def test_hand_state_rules(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_label({**PAYLOADS[5], **overrides})


def test_provenance_and_verification_rules() -> None:
    with pytest.raises(ValidationError, match="model_version"):
        Provenance(source=Source.MODEL)
    with pytest.raises(ValidationError, match="confidence"):
        make_label(
            BOX, stream_id="bodycam", provenance=Provenance(source=Source.MODEL, model_version="m1")
        )
    with pytest.raises(ValidationError, match="reviewer_id"):
        Verification(state=VerificationState.HUMAN_APPROVED)
    with pytest.raises(ValidationError, match="parent_label_id"):
        make_label(BOX, stream_id="bodycam", retracted=True)
    with pytest.raises(ValidationError, match="자기 자신"):
        make_label(BOX, stream_id="bodycam", parent_label_id="l001")


def test_contracts_are_immutable() -> None:
    label = make_label(BOX, stream_id="bodycam")
    with pytest.raises(ValidationError):
        label.t_end_ms = 5  # type: ignore[misc]


def test_session_requires_exactly_one_reference_bodycam() -> None:
    session = make_session()
    assert session.reference_stream.stream_id == "bodycam"
    assert Session.model_validate_json(session.model_dump_json()) == session
    streams = [s.model_dump() for s in session.streams]
    with pytest.raises(ValidationError, match="바디캠"):
        make_session(streams=streams[1:])
    with pytest.raises(ValidationError, match="중복"):
        make_session(streams=[*streams, streams[1]])
    with pytest.raises(ValidationError, match="reference"):
        make_session(streams=[{**streams[0], "sync_method": "qr_slate"}, streams[1]])
    with pytest.raises(ValidationError, match="reference"):
        make_session(streams=[streams[0], {**streams[1], "sync_method": "reference"}])


@pytest.mark.parametrize(
    "change", [{"offset_ms": 5.0}, {"clock_scale": 1.0001}, {"manual_adjustment_ms": -3.0}]
)
def test_reference_stream_has_identity_clock(change: dict[str, float]) -> None:
    streams = [s.model_dump() for s in make_session().streams]
    with pytest.raises(ValidationError, match="기준 스트림"):
        make_session(streams=[{**streams[0], **change}, streams[1]])


def test_stream_maps_to_master_timeline() -> None:
    stream = (
        make_session()
        .stream("imu")
        .model_copy(update={"offset_ms": 100.0, "clock_scale": 1.001, "manual_adjustment_ms": -5.0})
    )
    assert stream.kind is StreamKind.IMU
    assert stream.to_master_ms(1_000) == pytest.approx(1_096.0)


def test_lifecycle_moves_forward_one_step_or_withdraws() -> None:
    s = LifecycleState
    assert can_transition(s.RAW_INGESTED, s.PRIVACY_APPROVED)
    assert can_transition(s.PRELABELED, s.PRELABELED)
    assert not can_transition(s.RAW_INGESTED, s.PRELABELED)
    assert not can_transition(s.HUMAN_VERIFIED, s.PRELABELED)
    assert can_transition(s.EXPORTED, s.WITHDRAWN)
    assert not can_transition(s.WITHDRAWN, s.RAW_INGESTED)
    assert FIXED_TIME.tzinfo is not None
