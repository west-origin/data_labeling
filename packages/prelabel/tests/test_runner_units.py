"""프리라벨 실행기의 DB 없는 단위 테스트 (감사 4차 회귀).

검수된 결과 보호(접촉·3D 궤적), 착용자 사본의 출처·검수 상태, 착용자 다시 찾기 조건을 본다. ADR
0026. 입력은 `dlp_schema.testing.make_label`로 만든 작은 레코드다.
"""

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
    """모델 출처 손 상태(대상 bucket_01) 레코드. kw로 출처·검수 상태를 바꾼다."""
    payload = HandStatePayload(
        hand=hand, contact_target_kind="object", target_id="bucket_01", role="active"
    )
    return make_label(payload, label_id, start, end, **{**MODEL, **kw})


def test_new_contacts_overlapping_reviewed_ones_on_the_same_hand_are_dropped() -> None:
    """새 접촉 중 같은 손의 승인·사람 구간과 겹치는 것만 버리고 원래 순번을 유지하는지 본다.

    시나리오: 오른손 승인(1000~2000), 오른손 검수 전(3000~4000, 보호 대상 아님), 왼손
    사람(5000~6000). 오른손 기준 0번(승인과 겹침)만 빠지고, 3번(2000~2900)은 맞닿기만 해서
    남는다. 왼손 기준 2번(사람 구간과 겹침)만 빠진다.
    """
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
    """카메라 좌표 3D 궤적 모델 레코드 (샘플 하나). kw로 검수 상태를 바꾼다."""
    payload = Trajectory3DPayload(
        entity_id=entity,
        part=part,
        frame=CoordinateFrame.CAMERA,
        source_3d=Source3D.MONO_DEPTH,
        samples=(Trajectory3DSample(t_ms=0, x=0.0, y=0.0, z=1.0),),
    )
    return make_label(payload, label_id, 0, 1_000, stream_id="bodycam", **{**MODEL, **kw})


def test_new_trajectories_for_reviewed_entity_parts_are_dropped() -> None:
    """같은 (개체, 부위, 좌표계)의 승인된 궤적이 있는 새 궤적만 버리는지 본다 (손목만 빠지고 검지
    끝, 검수 전 대상인 걸레 궤적은 남는다).
    """
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
    """3인칭 coco17 인물 트랙 레코드 (키프레임 하나)."""
    frame = KeypointFrame(
        t_ms=0, points=tuple(Keypoint(x=1.0, y=1.0, visibility=2) for _ in range(17))
    )
    payload = KeypointTrackPayload(entity_id="person_01", skeleton="coco17", keyframes=(frame,))
    return make_label(payload, label_id, 0, 1_000, stream_id="third", **kw)


def test_wearer_copy_is_a_model_record_even_from_a_human_track() -> None:
    """감사 회귀: 사람이 고친(HUMAN) 트랙을 착용자로 찾아도 사본은 모델 출처·검수 전이다.

    ID `<원래>:wearer`, parent=원래, entity_id="wearer", 신뢰도=상관도 본다.
    """
    human = _person("p1:fix", provenance=Provenance(source=Source.HUMAN), verification=APPROVED)
    copy = wearer_copy(human, 0.83, f"{WEARER_PREFIX}+pabc", FIXED_TIME)
    assert copy.provenance == Provenance(source=Source.MODEL, model_version=f"{WEARER_PREFIX}+pabc")
    assert copy.verification.state is VerificationState.UNREVIEWED
    assert copy.parent_label_id == "p1:fix" and copy.label_id == "p1:fix:wearer"
    assert isinstance(copy.payload, KeypointTrackPayload) and copy.payload.entity_id == "wearer"
    assert copy.confidence == 0.83


def test_has_live_wearer_after_human_or_model_retraction() -> None:
    """착용자 레코드가 살아 있는지 판단: 검수자가 지운 것은 살아 있는 것으로 보고(되살리지 않음),
    모델 단계가 지운 것(전신 모델 버전 변경)은 죽은 것으로 본다(다시 찾음).
    """
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
