"""프리라벨 실행기의 DB 없는 단위 테스트 (감사 4차 회귀)."""

from __future__ import annotations

from typing import Any

from dlp_prelabel.contact import ContactInterval
from dlp_prelabel.runner import (
    WEARER_PREFIX,
    _has_live_wearer,  # pyright: ignore[reportPrivateUsage]
    drop_protected_contacts,
    drop_protected_trajectories,
    wearer_copy,
)
from dlp_schema.episode import retractions
from dlp_schema.labels import (
    CoordinateFrame,
    Hand,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
    Verification,
    VerificationState,
)
from dlp_schema.testing import FIXED_TIME, make_label

MODEL: dict[str, Any] = {
    "provenance": Provenance(source=Source.MODEL, model_version="m1"),
    "confidence": 0.5,
}
APPROVED = Verification(
    state=VerificationState.HUMAN_APPROVED, reviewer_id="rev01", reviewed_at=FIXED_TIME
)


def _contact(label_id: str, start: int, end: int, hand: Hand, **kw: Any) -> LabelRecord:
    payload = HandStatePayload(
        hand=hand, contact_target_kind="object", target_id="bucket_01", role="active"
    )
    return make_label(payload, label_id, start, end, **{**MODEL, **kw})


def test_new_contacts_overlapping_reviewed_ones_on_the_same_hand_are_dropped() -> None:
    current = [
        _contact("approved", 1_000, 2_000, Hand.RIGHT, verification=APPROVED),
        _contact(
            "unreviewed", 3_000, 4_000, Hand.RIGHT
        ),  # 검수 전: 지울 대상이지 보호 대상이 아니다
        _contact("human", 5_000, 6_000, Hand.LEFT, provenance=Provenance(source=Source.HUMAN)),
    ]
    new = [
        ContactInterval(1_500, 2_500, "bucket_01", "fused"),  # 승인 구간과 겹친다
        ContactInterval(3_000, 4_000, "bucket_01", "fused"),
        ContactInterval(5_000, 6_000, "bucket_01", "fused"),  # 다른 손의 사람 구간
        ContactInterval(2_000, 2_900, None, "glove"),  # 맞닿기만 한다
    ]
    kept = drop_protected_contacts(new, Hand.RIGHT, current)
    # 원래 순번을 유지한다 (라벨 ID가 바뀌지 않게)
    assert [i for i, _ in kept] == [1, 2, 3]
    assert [i for i, _ in drop_protected_contacts(new, Hand.LEFT, current)] == [0, 1, 3]


def _traj(label_id: str, entity: str, part: str | None, **kw: Any) -> LabelRecord:
    payload = Trajectory3DPayload(
        entity_id=entity,
        part=part,
        frame=CoordinateFrame.CAMERA,
        source_3d=Source3D.MONO_DEPTH,
        samples=(Trajectory3DSample(t_ms=0, x=0.0, y=0.0, z=1.0),),
    )
    return make_label(payload, label_id, 0, 1_000, stream_id="bodycam", **{**MODEL, **kw})


def test_new_trajectories_for_reviewed_entity_parts_are_dropped() -> None:
    current = [
        _traj("old-wrist", "hand_right", "wrist", verification=APPROVED),
        _traj("old-rag", "rag_01", None),  # 검수 전
    ]
    new = [
        _traj("new-wrist", "hand_right", "wrist"),
        _traj("new-tip", "hand_right", "index_tip"),
        _traj("new-rag", "rag_01", None),
    ]
    kept = drop_protected_trajectories(new, current)
    assert [x.label_id for x in kept] == ["new-tip", "new-rag"]


def _person(label_id: str, **kw: Any) -> LabelRecord:
    frame = KeypointFrame(
        t_ms=0, points=tuple(Keypoint(x=1.0, y=1.0, visibility=2) for _ in range(17))
    )
    payload = KeypointTrackPayload(entity_id="person_01", skeleton="coco17", keyframes=(frame,))
    return make_label(payload, label_id, 0, 1_000, stream_id="third", **kw)


def test_wearer_copy_is_a_model_record_even_from_a_human_track() -> None:
    """감사 회귀: 사람이 고친(HUMAN) 트랙을 착용자로 찾아도 사본은 모델 출처·검수 전이다."""
    human = _person("p1:fix", provenance=Provenance(source=Source.HUMAN), verification=APPROVED)
    copy = wearer_copy(human, 0.83, f"{WEARER_PREFIX}+pabc", FIXED_TIME)
    assert copy.provenance == Provenance(source=Source.MODEL, model_version=f"{WEARER_PREFIX}+pabc")
    assert copy.verification.state is VerificationState.UNREVIEWED
    assert copy.parent_label_id == "p1:fix" and copy.label_id == "p1:fix:wearer"
    assert isinstance(copy.payload, KeypointTrackPayload) and copy.payload.entity_id == "wearer"
    assert copy.confidence == 0.83


def test_has_live_wearer_after_human_or_model_retraction() -> None:
    wearer = wearer_copy(_person("p1"), 0.9, f"{WEARER_PREFIX}+pabc", FIXED_TIME)
    assert _has_live_wearer([wearer])
    # 검수자가 착용자 레코드를 지웠다: 다시 찾아 되살리지 않는다 (살아 있는 것으로 본다)
    by_human = wearer.model_copy(
        update={
            "label_id": f"{wearer.label_id}:rev",
            "parent_label_id": wearer.label_id,
            "retracted": True,
            "provenance": Provenance(source=Source.HUMAN),
            "confidence": None,
        }
    )
    assert _has_live_wearer([wearer, by_human])
    # 모델 단계(전신 모델 버전 변경)가 지웠으면 새 트랙으로 다시 찾는다
    assert not _has_live_wearer([wearer, *retractions([wearer], "body-v2", FIXED_TIME)])
    assert not _has_live_wearer([])
