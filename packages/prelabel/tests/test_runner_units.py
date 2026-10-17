"""프리라벨 실행기의 DB 없는 단위 테스트 (감사 4차 회귀).

검수된 결과 보호(접촉·3D 궤적), 착용자 사본의 출처·검수 상태, 착용자 다시 찾기 조건을 본다. ADR
0026. 입력은 `dlp_schema.testing.make_label`로 만든 작은 레코드다.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from dlp_prelabel.contact import ContactInterval
from dlp_prelabel.policy import load_policy
from dlp_prelabel.runner import (
    WEARER_PREFIX,
    _has_live_wearer,  # pyright: ignore[reportPrivateUsage]
    contact_label,
    contact_target_kind,
    drop_protected_contacts,
    drop_protected_trajectories,
    pick_hand_tracks,
    pick_lift_tracks,
    pick_world_inputs,
    wearer_copy,
)
from dlp_schema.config import repo_root
from dlp_schema.episode import retractions
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    CoordinateFrame,
    Evidence,
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
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, make_label, make_session

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
    kept = drop_protected_trajectories(new, current, frozenset())
    assert [x.label_id for x in kept] == ["new-tip", "new-rag"]


def test_lapsed_inherited_trajectories_do_not_block_or_feed_new_ones() -> None:
    """ADR 0048: 출처 2D가 바뀌어 물려받은 검수가 끊긴 궤적(lapsed)은 지우지 못하지만, 새 궤적을
    막지 않고(보호 판정에서 뺀다) 월드 좌표 단계 입력으로도 고르지 않는다.

    정답: 끊긴 승인 손목 궤적이 있어도 새 손목 궤적이 남고, 월드 입력은 끊긴 것 대신 새 것이다.
    끊기지 않은 승인 궤적은 그대로 새 것을 막는다.
    """
    current = [
        _traj("old-wrist", "hand_right", "wrist", verification=APPROVED),
        _traj("old-tip", "hand_right", "index_tip", verification=APPROVED),
    ]
    new = [_traj("new-wrist", "hand_right", "wrist"), _traj("new-tip", "hand_right", "index_tip")]
    lapsed = frozenset({"old-wrist"})
    kept = drop_protected_trajectories(new, current, lapsed)
    assert [x.label_id for x in kept] == ["new-wrist"]
    _, cams = pick_world_inputs([*current, kept[0]], "bodycam", lapsed)
    assert sorted(x.label_id for x in cams) == ["new-wrist", "old-tip"]


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


def _hand_track(label_id: str, hand: Hand, end: int, x: float, **kw: Any) -> LabelRecord:
    """바디캠 hand21 트랙 (0~end ms, 모든 관절이 x 좌표). x로 어느 트랙이 골렸는지 구별한다."""
    points = tuple(Keypoint(x=x, y=0.0, visibility=2) for _ in range(21))
    payload = KeypointTrackPayload(
        entity_id=f"{hand.value}_hand",
        skeleton="hand21",
        hand=hand,
        keyframes=(KeypointFrame(t_ms=0, points=points),),
    )
    return make_label(payload, label_id, 0, end, stream_id="bodycam", **kw)


def test_hand_track_choice_is_deterministic() -> None:
    """같은 손 트랙이 여럿이면 사람·검수 → 최신 → 긴 구간 → 라벨 ID 순으로 하나를 고른다.

    감사 회귀: 예전에는 입력 순서의 마지막 트랙을 썼다. 입력 순서를 뒤집어도 결과가 같아야 한다.
    """
    later = {"created_at": FIXED_TIME + timedelta(days=1)}
    old_long = _hand_track("m-old", Hand.LEFT, 5_000, 1.0, **MODEL)
    new_short = _hand_track("m-new", Hand.LEFT, 1_000, 2.0, **MODEL, **later)
    reviewed = _hand_track("m-rev", Hand.LEFT, 500, 3.0, **MODEL, verification=APPROVED)
    right_a = _hand_track("r-a", Hand.RIGHT, 1_000, 4.0, **MODEL)
    right_b = _hand_track("r-b", Hand.RIGHT, 2_000, 5.0, **MODEL)
    for labels in (
        [old_long, new_short, reviewed, right_a, right_b],
        [right_b, right_a, reviewed, new_short, old_long],
    ):
        picked = pick_hand_tracks(labels)
        # 왼손: 검수된 트랙이 짧고 오래됐어도 먼저. 오른손: 같은 시각이면 더 긴 구간
        assert picked[Hand.LEFT].keyframes[0].points[0].x == 3.0
        assert picked[Hand.RIGHT].keyframes[0].points[0].x == 5.0
    # 검수된 것이 없으면 더 새로 만든 트랙
    assert pick_hand_tracks([new_short, old_long])[Hand.LEFT].keyframes[0].points[0].x == 2.0


def test_contact_label_marks_glove_evidence_observed() -> None:
    """접촉 라벨의 증거 표시: 장갑이 관여한 접촉(장갑만·융합)은 OBSERVED, 영상만은 INFERRED.

    AI 1차 검수가 이 표시로 장갑 센서에서 나온 접촉을 판정 대상에서 뺀다 (`dlp_aireview.plan.
    glove_derived`, ADR 0040 결정 5). 장갑만 잡은 접촉은 대상을 몰라 unresolved, 신뢰도는 출처별
    정책 값이다.
    """
    policy = load_policy(repo_root())
    session = make_session("s-contact")
    intervals = {
        source: ContactInterval(1_000, 1_500, None if source == "glove" else "bucket_01", source)
        for source in ("glove", "fused", "video")
    }
    version = "contact-heuristic-1+pX+iY"
    made = {
        source: contact_label(session, version, Hand.LEFT, 3, c, "object", policy, FIXED_TIME)
        for source, c in intervals.items()
    }
    assert {k: v.evidence for k, v in made.items()} == {
        "glove": Evidence.OBSERVED, "fused": Evidence.OBSERVED, "video": Evidence.INFERRED,
    }  # fmt: skip
    glove = made["glove"]
    assert glove.label_id.startswith("s-contact-contact-") and glove.label_id.endswith("-left-0003")
    assert glove.stream_id is None and glove.provenance.source is Source.MODEL
    assert isinstance(glove.payload, HandStatePayload)
    assert glove.payload.target_id == policy.contact.unresolved_target_id
    assert glove.confidence == policy.contact.confidence.glove


def _box_track(label_id: str, entity: str, x: float, **kw: Any) -> LabelRecord:
    """바디캠 박스 트랙 (0 ms 키프레임 하나, 왼쪽 위 x 좌표로 어느 트랙이 골렸는지 구별한다)."""
    payload = BoxTrackPayload(
        entity_id=entity,
        class_id="bucket",
        keyframes=(BoxKeyframe(t_ms=0, x=x, y=0.0, w=10.0, h=10.0),),
    )
    return make_label(payload, label_id, 0, 1_000, stream_id="bodycam", **kw)


def test_lift_inputs_are_one_track_per_entity() -> None:
    """F-PAR-01 회귀: 3D 입력은 바디캠의 (개체, 트랙 종류)마다 트랙 하나다.

    시나리오: 왼손은 승인된 이전 버전(x=3)과 더 새 모델 버전(x=2)이 같은 entity_id로 함께 현재
    라벨이다. 버킷 박스도 사람 트랙(x=7)과 더 새 모델 트랙(x=8)이 같은 entity_id다. 예전에는 둘을
    모두 넘겨 같은 (개체, 부위)에 같은 시각 표본이 두 번 들어가 3D 라벨 검증이 실패했다.

    정답: 손은 `pick_hand_tracks`와 같은 순서(사람·검수 → 최신 → 긴 구간 → ID)로 승인 트랙, 박스는
    사람 트랙. 오른손·걸레처럼 하나뿐인 것은 그대로, 3D로 올리지 않는 coco17 트랙과 다른 스트림
    트랙은 빠진다. 입력 순서를 뒤집어도 같다.
    """
    later = {"created_at": FIXED_TIME + timedelta(days=1)}
    reviewed = _hand_track("left-v1", Hand.LEFT, 1_000, 3.0, **MODEL, verification=APPROVED)
    newer = _hand_track("left-v2", Hand.LEFT, 1_000, 2.0, **MODEL, **later)
    right = _hand_track("right-v2", Hand.RIGHT, 1_000, 4.0, **MODEL)
    human_box = _box_track(
        "bucket-fix", "bucket_01", 7.0, provenance=Provenance(source=Source.HUMAN)
    )
    model_box = _box_track("bucket-v2", "bucket_01", 8.0, **MODEL, **later)
    rag = _box_track("rag-v1", "rag_01", 9.0, **MODEL)
    body = _person("person-v1", **MODEL).model_copy(update={"stream_id": "bodycam"})
    third = newer.model_copy(update={"label_id": "third-left", "stream_id": "third"})
    labels = [reviewed, newer, right, human_box, model_box, rag, body, third]
    for order in (labels, labels[::-1]):
        picked = pick_lift_tracks(order, "bodycam")
        assert [x.label_id for x in picked] == ["bucket-fix", "left-v1", "rag-v1", "right-v2"]
    assert pick_lift_tracks([body, third], "bodycam") == []


def test_person_box_contacts_get_no_contact_kind() -> None:
    """F-PAR-03 회귀: 물리 유형이 person인 박스(COCO person → 온톨로지 person) 접촉은 종류가 없다.

    온톨로지상 사람 접촉(person)은 body_part가 필수인데 subjects.yaml body_parts에 "모름" 값이 없고
    박스로는 부위를 알 수 없다. 예전에는 object로 기록했다. 도구·표면·그 밖의 객체와 모르는 클래스는
    예전과 같다.
    """
    ontology = load_ontology(repo_root() / "config" / "ontology" / "v1")
    assert ontology.objects["person"].physical_type == "person"
    assert "unknown" not in ontology.body_parts  # 부위 "모름" 값이 생기면 이 처리를 다시 본다
    assert contact_target_kind("person", ontology) is None
    assert contact_target_kind("rag", ontology) == "tool"
    assert contact_target_kind("sink", ontology) == "fixed_surface"
    assert contact_target_kind("bucket", ontology) == "object"
    assert contact_target_kind(None, ontology) == "object"  # 장갑만 잡은 접촉 (대상 모름)
    assert contact_target_kind("not_in_ontology", ontology) == "object"
