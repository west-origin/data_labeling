"""F-PAR-86 회귀: 2D 단계 ID가 128자 안인 긴 세션에서 그 트랙에서 만든 라벨 ID도 128자 안이다.

마스크(`masks.mask_labels`), 착용자 팔(`arm.arm_tracks`), 착용자 사본(`runner.wearer_copy`)은 입력
트랙 ID보다 긴 ID를 만들어 128자를 넘으면 검증 실패(마스크·팔)나 DB 넣기 실패(착용자 사본은
`model_copy`라 검증 없이 지나간다)로 프리라벨 트랜잭션 전체가 롤백됐다. 고친 뒤에는 단계 접두사를
지킨 채 해시로 줄이고, 짧은 ID는 예전 그대로다 (재실행 멱등). DB·서비스 불필요.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dlp_media.rle import encode_rle
from dlp_prelabel.arm import ArmFrame, arm_tracks
from dlp_prelabel.masks import MaskReport, MaskRun, PromptTrack, mask_labels
from dlp_prelabel.runner import WEARER_PREFIX, wearer_copy
from dlp_schema.common import IDENTIFIER_MAX
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    Hand,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    MaskKeyframe,
    Provenance,
    Source,
)
from dlp_schema.predictor import Clip
from dlp_schema.testing import FIXED_TIME, make_label


def _session_id(tail: str) -> str:
    """`<세션><tail>`이 정확히 128자가 되는 세션 ID (2D 단계 ID가 막 통과하는 길이)."""
    return "S" * (IDENTIFIER_MAX - len(tail))


def _masks(session_id: str) -> list[LabelRecord]:
    """오른손 프롬프트 트랙 하나로 만든 마스크 라벨."""
    rec = make_label(
        HandStatePayload(
            hand=Hand.RIGHT, contact_target_kind="object", target_id="x", role="active"
        ),
        label_id=f"{session_id}-hands",
    )
    src = PromptTrack(rec, "right_hand", "hand", True, (0,))
    m = np.zeros((4, 4), bool)
    m[1:3, 1:3] = True
    run = MaskRun(
        {"right_hand": [MaskKeyframe(t_ms=0, rle=encode_rle(m), score=0.9)]}, MaskReport()
    )
    return mask_labels(
        run, [src], session_id=session_id, stream_id="bodycam", ontology_version="1.0.0",
        version="sam2t-x+iabc", now=FIXED_TIME,
    )  # fmt: skip


def test_mask_ids_fit_when_the_hand_track_id_fits() -> None:
    """손 트랙 ID가 128자인 세션의 마스크 ID는 128자 안이고 단계 접두사를 지킨다."""
    sid = _session_id(f"-bodycam-hands-{version_tag('mediapipe-x')}-right")
    [label] = _masks(sid)
    assert len(label.label_id) <= IDENTIFIER_MAX
    assert label.label_id.startswith(f"{sid}-bodycam-masks-")
    [short] = _masks("s1")
    assert short.label_id == f"s1-bodycam-masks-{version_tag('sam2t-x+iabc')}-right_hand"


def _arms(session_id: str) -> list[LabelRecord]:
    """오른팔 키프레임 하나로 만든 착용자 팔 트랙."""
    kp = Keypoint(x=1.0, y=1.0, visibility=2, confidence=0.9)
    frame = ArmFrame(0, Hand.RIGHT, (kp, kp, kp), 0.9, "ok", 1.0)
    return arm_tracks(
        {Hand.RIGHT: [frame]}, Clip(session_id, "bodycam", Path("x.mp4")), prefix="body",
        version="rtm-x", ontology_version="1.0.0", now=FIXED_TIME,
    )  # fmt: skip


def test_arm_ids_fit_when_the_hand_track_id_fits() -> None:
    """손 트랙 ID가 128자인 세션의 팔 트랙 ID는 128자 안이고 전신 단계 접두사를 지킨다."""
    sid = _session_id(f"-bodycam-hands-{version_tag('rtm-x')}-right")
    [label] = _arms(sid)
    assert len(label.label_id) <= IDENTIFIER_MAX
    assert label.label_id.startswith(f"{sid}-bodycam-body-")
    [short] = _arms("s1")
    assert short.label_id == f"s1-bodycam-body-{version_tag('rtm-x')}-arm-right"


def _person(label_id: str) -> LabelRecord:
    """3인칭 coco17 모델 인물 트랙 (키프레임 하나)."""
    pts = tuple(Keypoint(x=1.0, y=1.0, visibility=2) for _ in range(17))
    payload = KeypointTrackPayload(
        entity_id="p", skeleton="coco17", keyframes=(KeypointFrame(t_ms=0, points=pts),)
    )
    return make_label(
        payload, label_id=label_id, stream_id="third", t_start_ms=0, t_end_ms=0,
        provenance=Provenance(source=Source.MODEL, model_version="rtm-x"), confidence=0.9,
    )  # fmt: skip


def test_wearer_copy_ids_fit_and_keep_the_body_stage_prefix() -> None:
    """원래 인물 트랙 ID가 128자 안이면 착용자 사본 ID(`<원래>:wearer`)도 128자 안이고, 전신
    단계 접두사(그 단계가 지울 레코드를 고르는 기준)를 지킨다. 짧은 ID는 그대로다."""
    tag = version_tag("rtm-x")
    for tail in (f"-bodycam-hands-{tag}-right", f"-third-body-{tag}-000"):
        sid = _session_id(tail)
        original = _person(f"{sid}-third-body-{tag}-000")
        copy = wearer_copy(original, 0.8, f"{WEARER_PREFIX}+px", FIXED_TIME)
        assert len(copy.label_id) <= IDENTIFIER_MAX
        assert copy.label_id.startswith(f"{sid}-third-body-")
        assert copy.parent_label_id == original.label_id
        # model_copy는 검증하지 않는다: 계약 검증을 다시 통과하는지 본다 (DB 넣기 전 상태)
        LabelRecord.model_validate(copy.model_dump())
    short = wearer_copy(_person("p-1"), 0.8, f"{WEARER_PREFIX}+px", FIXED_TIME)
    assert short.label_id == "p-1:wearer"
