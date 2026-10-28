# PyAV 타입 정보가 일부 비어 있어 이 모듈에서만 알 수 없는 멤버 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""확인된 미수정 결함의 재현 테스트 (docs/handover/issues.yaml).

각 테스트는 **올바른 동작**을 단언하고 `xfail(strict=True, raises=<결함이 내는 예외>)`로
표시한다.
`raises`가 있어 결함과 무관한 오류(없어진 함수를 패치하다 난 AttributeError 등)는 xfail이 아닌 진짜
실패로 보고된다 — 그런 오류가 xfail로 묻혀 고친 결함을 알아채지 못한 일이 있었다(F-REV-01).
지금은 결함 때문에 실패해야 하며(xfail), 누군가 결함을 고쳐 테스트가 통과하면 strict 규칙에 따라
XPASS가 실패로 보고된다.
그때 이 표시를 지우고 레지스트리 항목의 status를 fixed로 바꾼다 (`tests/test_issue_registry.py`가
레지스트리의 repro 경로와 이 파일의 reason ID가 맞는지 검사한다).

reason은 `<레지스트리 ID>: <한 줄 요약>` 형식이다. 서비스(DB·도구) 없이 돈다.
정답 근거는 각 테스트 docstring에 적었다 (계약·ADR·정책 문구).
"""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from pydantic import ValidationError

from dlp_schema import repo_root
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_schema.validation import OntologyViolationError

ROOT = repo_root()


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-REV-11: 이벤트 속성 입력란을 모두 비우면 원래 값이 남음",
)
def test_ls_event_attributes_cleared_are_cleared() -> None:
    """검수자가 관련 행동 ID 입력란을 비우면(perRegion 결과가 사라지면) 원래 값도 지워져야 한다.

    근거: labelstudio.from_ls_results 주석 "입력란을 비우면 원래 값을 지운다".
    """
    from dlp_review.labelstudio import from_ls_results, to_ls_results
    from dlp_schema.labels import EventPayload

    ev = make_label(
        {"kind": "event", "event_type": "grasp_failure", "related_action_id": "a001"},
        label_id="ev1",
        t_end_ms=500,
    )
    kept = [r for r in to_ls_results([ev]) if r["type"] == "timeserieslabels"]
    [item] = from_ls_results(kept, {"ev1"})
    assert isinstance(item.payload, EventPayload)
    assert item.payload.related_action_id is None


@pytest.mark.xfail(
    strict=True,
    raises=OntologyViolationError,
    reason="F-SCH-06: 기존 위반이 있으면 무관한 새 라벨도 저장 거부",
)
def test_legacy_violation_does_not_block_unrelated_label(ontology: Ontology) -> None:
    """0014 이전에 들어간 잘못된 구간이 있어도 무관한 새 블러 라벨은 저장되어야 한다.

    근거: ADR 0037 "기존 행을 덮어쓰거나 의미를 추측해 고치지 않는다"(기존 위반은 audit로 확인).
    """
    from dlp_schema.label_validation import validate_session_labels

    legacy = make_label(
        {
            "kind": "segment",
            "segment_id": "sk",
            "level": "skill",
            "ref_id": "wipe",
            "child_ids": ["gone"],
        },
        label_id="legacy-seg",
    )
    blur = make_label(
        {
            "kind": "blur_track",
            "target": "face",
            "keyframes": [{"t_ms": 0, "x": 0, "y": 0, "w": 9, "h": 9}],
        },
        label_id="new-blur",
        stream_id="bodycam",
    )
    validate_session_labels(ontology, [legacy], [blur])


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="F-PMS-05: 반올림 ms가 같은 두 프레임이면 블러 탐지 실패",
)
def test_privacy_detect_tolerates_frames_rounding_to_same_ms(tmp_path: Path) -> None:
    """PTS가 달라도 반올림 ms가 같은 프레임(33.0·33.4 ms)이 있으면 같은 시각 키프레임을
    하나만 둬야 한다.

    근거: 프리라벨은 strictly_increasing으로 같은 처리를 한다(ADR 0031 결정 8).
    """
    from dlp_privacy.detection import Detection
    from dlp_privacy.geometry import Box
    from dlp_privacy.pipeline import detect_video
    from dlp_privacy.policy import TargetPolicy, load_policy

    src = tmp_path / "v.mp4"
    with av.open(str(src), "w") as c:
        vs = c.add_stream("libx264", rate=30)
        vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        vs.time_base = Fraction(1, 10000)
        vs.codec_context.time_base = vs.time_base
        for pts in (0, 330, 334, 660, 1000):  # 0, 33.0, 33.4, 66.0, 100 ms
            frame = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = pts, vs.time_base
            c.mux(vs.encode(frame))
        c.mux(vs.encode(None))

    class Always:
        """매 프레임 같은 얼굴 박스를 내는 탐지기."""

        name = "oracle"
        version = "x"

        def detect(self, image: np.ndarray, t_ms: int, threshold: float) -> list[Detection]:
            return [Detection("face", Box(10, 10, 10, 10), 0.9, "oracle")]

    policy = load_policy(ROOT)
    policy = policy.model_copy(
        update={"targets": {"face": TargetPolicy(margin=0.2, detectors=("oracle",))}}
    )
    result = detect_video(
        src,
        session_id="s",
        stream_id="b",
        detectors={"oracle": Always()},  # pyright: ignore[reportArgumentType]
        missing_detectors={},
        policy=policy,
        ontology_version="1.0.0",
        now=FIXED_TIME,
    )
    assert result.labels


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="F-PMS-07: NaN이 섞인 장갑 시각을 받아들임",
)
def test_glove_times_with_nan_rejected() -> None:
    """시각에 NaN이 있으면 단조 증가 검사가 무력해진다. 수집 초기에 거부해야 한다.

    근거: glove.GloveData의 "시각은 엄격히 증가" 검사 의도.
    """
    from dlp_media.glove import GloveData

    with pytest.raises(ValueError):
        GloveData(np.array([0.0, 10.0, np.nan, 5.0]), {"p": np.ones(4)}, "parquet")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-07: 결과와 무관한 VLM timeout이 행동 정책 해시에 들어감",
)
def test_actions_policy_digest_ignores_timeout() -> None:
    """VLM 시간 제한은 결과에 영향이 없다. 바꿔도 버전(정책 해시)이 그대로여야 한다.

    근거: 같은 이유로 backoff를 해시에서 뺐다(actions.policy). 해시가 바뀌면 검수 전 행동을
    모두 지우고
    VLM을 다시 부른다.
    """
    from dlp_actions.policy import load_policy

    p = load_policy(ROOT)
    q = p.model_copy(update={"vlm": p.vlm.model_copy(update={"timeout_s": p.vlm.timeout_s * 2})})
    assert p.digest == q.digest


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-ETA-08: 큐레이션이 공간 마스크를 마스터 시각처럼 변환",
)
def test_curation_keeps_mask_in_stream_time() -> None:
    """mask_track은 공간 라벨이라 시각이 이미 스트림 PTS다. 오프셋을 다시 빼면 안 된다.

    근거: ADR 0019(공간 라벨 키프레임 = 스트림 PTS), dlp_schema.labels.SPATIAL_KINDS.
    """
    from dlp_active.curation import stream_sample
    from dlp_active.policy import load_policy
    from dlp_fixtures.video import write_video
    from dlp_media.pts import build_pts_index
    from dlp_schema.session import Stream, StreamKind, SyncMethod

    policy = load_policy(ROOT)
    times = list(range(0, 1000, 100))
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "v.mp4"
        write_video(
            video, ((t, np.zeros((16, 16, 3), np.uint8)) for t in times), width=16, height=16
        )
        idx = build_pts_index(video)
        third = Stream(
            stream_id="third",
            kind=StreamKind.THIRD_PERSON,
            uri="local://x/t.mp4",
            sync_method=SyncMethod.QR_SLATE,
            offset_ms=400,
        )
        mask = make_label(
            {
                "kind": "mask_track",
                "entity_id": "mop_1",
                "class_id": "mop",
                "keyframes": [
                    {"t_ms": 300, "rle": {"size": [16, 16], "counts": "a"}},
                    {"t_ms": 500, "rle": {"size": [16, 16], "counts": "a"}},
                ],
            },
            label_id="mk",
            stream_id="third",
            t_start_ms=300,
            t_end_ms=500,
        )
        sample = stream_sample("s1", video, idx, (16, 16), third, [mask], policy, {})
    # 프레임 번호는 1부터 (스트림 PTS 300 ms → 4번째, 500 ms → 6번째 프레임). 고치는 방식이
    # 구간을 바로잡든 공간 라벨을 시간 구간에서 빼든 통과한다.
    assert all(x.support == (4, 6) for x in sample.temporal if x.kind == "mask_track")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-AIR-13: 보낸 뒤 지운 라벨의 종류를 LS에서 바꿔 붙이면 다른 종류로 되살아난다",
)
def test_relabel_of_a_retracted_label_does_not_revive_it() -> None:
    """F-AIR-13 재현 (기존 수집 경로, AI와 무관). 작업을 보낸 뒤 다른 단계가 사이 구간 g를 지웠는데
    검수자가 Label Studio에서 그 구간의 라벨을 행동(push)으로 바꿔 붙였다.

    올바른 동작(`packages/review/tests/test_units.py::
    test_correction_of_a_retracted_label_does_not_revive_it`와 같은 규칙): 보낸 뒤 바뀐 라벨에 대한
    결과는 stale로 빠지고, 지운 라벨 자리에 다른 종류 레코드가 운영 라벨로 되살아나지 않는다.
    지금: 종류를 바꿔 붙인 구간은 원 라벨 ID를 잃어(`labelstudio._reviewed_item`) parent 없는 새
    레코드가 되고, `drop_retracted`는 parent가 있는 결과만 빼므로 그 행동이 현재 라벨로 남는다.
    """
    from dlp_review.collect import drop_retracted
    from dlp_review.labelstudio import from_ls_results, to_ls_results
    from dlp_review.reconcile import reconcile
    from dlp_schema.episode import current_labels

    gap = make_label({"kind": "gap", "gap_type": "idle", "hand": "right"}, label_id="g",
                     t_start_ms=1000, t_end_ms=2000)  # fmt: skip
    gone = gap.model_copy(
        update={"label_id": "g:retracted", "parent_label_id": "g", "retracted": True}
    )
    [r] = to_ls_results([gap])
    r["value"]["timeserieslabels"] = ["action.right:push"]
    reviewed = from_ls_results([r], {gap.label_id})
    out = reconcile([gap], reviewed, session_id="s001", ontology_version="1.0.0",
                    reviewer_id="r", now=FIXED_TIME)  # fmt: skip
    assert drop_retracted(out, [gap, gone]) == ["g"]
    assert not current_labels([gap, gone, *out.new_records])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-25: 물려받은 설명을 같은 상태·검수자·시각으로 직접 다시 검수하면 구분되지 않음",
)
def test_same_tuple_direct_review_after_inheritance_is_protected() -> None:
    """F-PAR-25 재현. 승인한 행동 act의 검수(r1, T1)를 모델 설명 d가 물려받았다(기록 있음). 그 뒤
    검수자가 d를 직접 승인했는데 상태·검수자·시각이 물려받은 값과 같다 (r1, T1). 행동 의미가
    다시 바뀐다.

    올바른 동작(CLAUDE.md 규칙, ADR 0043 3절 규칙 6): 기록 뒤 직접 검수한 설명은 보호되어 설명
    따르기가 지우지 않는다. 지금: `record_review`는 행만 갱신하고 흔적을 남기지 않아 지금 검수
    상태가 마지막 물려받기 기록과 같으므로 물려받은 검수만 있는 것으로 보고 지운다
    (`description_follow._active`).
    """
    from datetime import timedelta

    from dlp_schema.db.description_repository import apply_description_follow
    from dlp_schema.db.repository import (
        get_description_inheritance,
        get_labels,
        insert_labels,
        insert_session,
        record_review,
        register_ontology,
    )
    from dlp_schema.description_follow import plan_description_follow
    from dlp_schema.labels import Provenance, Source, Verification, VerificationState
    from dlp_schema.review import InheritanceMechanism
    from dlp_schema.testing import action_payload, make_session, sqlite_db

    t1, t2 = FIXED_TIME + timedelta(hours=1), FIXED_TIME + timedelta(hours=2)
    model = Provenance(source=Source.MODEL, model_version="actions-abc+oracle-vlm-1+i0")
    act = make_label(action_payload(), label_id="act", provenance=model, confidence=0.9)
    desc = make_label({"kind": "description", "segment_id": "a001", "text": "걸레를 잡는다"},
                      label_id="d", provenance=model, confidence=0.9)  # fmt: skip
    fix = act.model_copy(
        update={"label_id": "fix", "parent_label_id": "act", "created_at": t2,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": Verification(state=VerificationState.HUMAN_CORRECTED,
                                             reviewer_id="r1", reviewed_at=t2),
                "payload": act.payload.model_copy(update={"verb": "lift"})}
    )  # fmt: skip
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
        insert_session(conn, make_session("s001"))
        insert_labels(conn, [act, desc])
        record_review(conn, "act", VerificationState.HUMAN_APPROVED, "r1", t1)
        apply_description_follow(conn, "s001", t1, InheritanceMechanism.COLLECT)
        record_review(conn, "d", VerificationState.HUMAN_APPROVED, "r1", t1)  # 직접 검수
        insert_labels(conn, [fix])
        plan = plan_description_follow(
            get_labels(conn, "s001"), get_description_inheritance(conn, "s001")
        )
    engine.dispose()
    assert [x.label_id for x in plan.retract] == []


# ---------------------------------------------------------------------------
# 2026-10-28 전수 검사의 "하" 결함 재현 (서비스 없이, 합성 값·SQLite 메모리 DB)
# ---------------------------------------------------------------------------


def test_update_assignment_rejects_unknown_status() -> None:
    """배정 상태는 `AssignmentStatus` 값만이다. 계약 밖 문자열은 쓰기 전에 거부해야 한다.

    근거: `ReviewAssignment.status`가 `AssignmentStatus`이고 `list_assignments`가 행을 그
    계약으로 읽는다(저장된 "bogus"는 이후 목록 읽기를 ValidationError로 깨뜨린다).
    """
    from dlp_schema.db.repository import insert_session
    from dlp_schema.db.review_repository import insert_assignment, update_assignment
    from dlp_schema.review import ReviewAssignment, ReviewMode
    from dlp_schema.testing import make_session, sqlite_db

    engine = sqlite_db()
    with engine.begin() as conn:
        insert_session(conn, make_session("s001"))
        insert_assignment(
            conn,
            ReviewAssignment(
                assignment_id="a1", session_id="s001", label_kinds=("action",),
                mode=ReviewMode.STANDARD, priority=1.0, created_at=FIXED_TIME,
            ),
        )  # fmt: skip
        with pytest.raises((ValueError, TypeError)):
            update_assignment(conn, "a1", status="bogus")
    engine.dispose()


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "mask_track", "entity_id": "rag_01", "class_id": "rag",
            "keyframes": [{"t_ms": 0, "rle": {"size": [-5, -5], "counts": ""}}],
        },
        {
            "kind": "trajectory3d", "entity_id": "camera", "frame": "world",
            "source_3d": "mono_depth", "samples": [{"t_ms": 0, "x": 0, "y": 0, "z": 0, "qw": 1.0}],
        },
    ],
    ids=["negative_rle_size", "partial_quaternion"],
)  # fmt: skip
def test_contract_rejects_negative_rle_and_partial_quaternion(payload: dict[str, object]) -> None:
    """마스크 크기는 (높이, 너비) 양수이고, 자세 쿼터니언은 네 성분이 다 있거나 다 없어야 한다.

    근거: COCO RLE 정의(size = [h, w], 화소 수), `Trajectory3DSample` 쿼터니언 필드 설명
    (방향은 qw·qx·qy·qz 한 묶음). 일부만 있으면 내보내기·리타기팅이 방향을 해석할 수 없다.
    """
    with pytest.raises(ValidationError):
        make_label(payload, stream_id="bodycam")


def test_capture_check_no_gyro_does_not_claim_motion() -> None:
    """자이로 채널이 없는 기종(`imu_from_gpmf_payloads`가 NaN으로 채움)은 회전을 판단할 수 없다.
    가속도가 정지(중력만)면 "처음 1초가 정지가 아니다"라고 경고하면 안 된다.

    근거: 같은 가속도에 자이로 0이면 경고가 없다(정답 = 정지). NaN은 "모름"이지 움직임이 아니다.
    """
    from dlp_media.capture_check import imu_warnings, load_capture_policy
    from dlp_media.imu import ImuData

    policy = load_capture_policy(ROOT)
    t = np.arange(0, 3000, 5.0)
    acc = np.tile([0.0, 0.0, 9.81], (t.size, 1))
    still = ImuData(t, acc, np.zeros((t.size, 3)), "gpmf")
    assert "no_still_start" not in [w.code for w in imu_warnings(still, policy)]
    no_gyro = ImuData(t, acc, np.full((t.size, 3), np.nan), "gpmf")
    assert "no_still_start" not in [w.code for w in imu_warnings(no_gyro, policy)]


def test_decode_rle_rejects_negative_run() -> None:
    """COCO 압축 RLE "3O2"는 길이 [3, -1, 2]다. 음수 길이는 손상된 입력이라 거부해야 한다
    (지금은 조용히 빈 마스크를 낸다).

    근거: COCO RLE 정의(각 길이는 0 이상, 합 = h·w). 2x2 마스크의 길이 합은 4다.
    """
    from dlp_media.rle import decode_rle
    from dlp_schema.labels import Rle

    with pytest.raises(ValueError):
        decode_rle(Rle(size=(2, 2), counts="3O2"))


def test_short_flicker_hint_for_single_frame_detection(tmp_path: Path) -> None:
    """한 프레임에서만 잡힌 얼굴 블러 트랙(깜박임)에는 short_flicker 힌트가 붙어야 한다.

    근거: privacy.yaml hints.short_ms(300) 규칙의 의도는 짧게 깜박인 탐지다. 탐지 유지(hold
    200 ms)로 늘어난 구간이 아니라 실제 관측 길이로 판단해야 한다(ADR 0055).
    """
    from dlp_fixtures.video import generate_blur_scenario
    from dlp_privacy.detectors.oracle import OracleDetector
    from dlp_privacy.hints import privacy_hints
    from dlp_privacy.pipeline import detect_video
    from dlp_privacy.policy import TargetPolicy, load_policy

    scenario = generate_blur_scenario(5)
    src = tmp_path / "scen.mp4"
    scenario.write(src)
    policy = load_policy(ROOT)
    policy = policy.model_copy(
        update={"targets": {"face": TargetPolicy(margin=0.2, detectors=("oracle",))}}
    )
    ft = scenario.frame_times
    keep = ft[len(ft) // 2]
    spans = [("face", ft[0], keep - 1), ("face", keep + 1, ft[-1])]  # 이 한 프레임 밖은 놓침
    det = OracleDetector("oracle", scenario.labels, targets={"face"}, miss_spans_ms=spans)
    result = detect_video(
        src, session_id="s", stream_id="bodycam", detectors={"oracle": det},
        missing_detectors={}, policy=policy, ontology_version="1.0.0", now=FIXED_TIME,
    )  # fmt: skip
    assert result.labels
    hints = privacy_hints(result.labels, "bodycam", policy.hints)
    assert "short_flicker" in {h.reason for h in hints}


def test_on_hand_hint_ignores_invisible_joints() -> None:
    """visibility 0 관절은 좌표에 의미가 없다. (0, 0)에 놓인 보이지 않는 손끝 하나 때문에 손
    외접 박스가 화면 왼쪽 위까지 넓어져 먼 얼굴(10, 10, 30, 30)을 손 위로 보면 안 된다.

    근거: 같은 손에서 그 점만 빼면 힌트가 없다(정답 = 힌트 없음). `_heads`도 visibility 2만 쓴다.
    """
    from dlp_privacy.hints import privacy_hints
    from dlp_privacy.policy import load_policy
    from dlp_schema.labels import Keypoint

    policy = load_policy(ROOT).hints
    ts = list(range(0, 1000, 100))
    face = make_label(
        {
            "kind": "blur_track", "target": "face",
            "keyframes": [{"t_ms": t, "x": 10, "y": 10, "w": 30, "h": 30} for t in ts],
        },
        label_id="s-face", stream_id="bodycam", t_start_ms=0, t_end_ms=900,
    )  # fmt: skip
    pts = [Keypoint(x=250 + i, y=200 + i, visibility=2).model_dump() for i in range(21)]
    pts[20] = Keypoint(x=0, y=0, visibility=0).model_dump()  # 보이지 않는 점
    hand = make_label(
        {
            "kind": "keypoint_track", "entity_id": "right_hand", "skeleton": "hand21",
            "hand": "right", "keyframes": [{"t_ms": t, "points": pts} for t in ts],
        },
        label_id="s-hand", stream_id="bodycam", t_start_ms=0, t_end_ms=900,
    )  # fmt: skip
    assert privacy_hints([face, hand], "bodycam", policy) == []


def _arm_tracks_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: object
) -> tuple[Any, Path, Any, str]:
    """F-PAR-73용: 착용자 팔 장면의 박스를 탐지·추적 결과로 바로 쓰게 한다."""
    from dlp_fixtures.arm import generate_arm_scenario
    from dlp_fixtures.video import write_video
    from dlp_prelabel.common import BoxTrack
    from dlp_schema.labels import BoxTrackPayload
    from dlp_schema.wearer_body import WearerEvidence

    scene = generate_arm_scenario(2)
    w, h = scene.size
    video = tmp_path / "v.mp4"
    frames = ((t, np.full((h, w, 3), 90, np.uint8)) for t in scene.frame_times[:5])
    write_video(video, frames, width=w, height=h)
    tracks: list[BoxTrack] = []
    for x in scene.boxes:
        p = x.payload
        assert isinstance(p, BoxTrackPayload)
        tracks.append(
            BoxTrack(p.class_id, {k.t_ms: ((k.x, k.y, k.w, k.h), 0.9) for k in p.keyframes})
        )

    def fixed_tracks(*_a: object, **_k: object) -> list[BoxTrack]:
        return list(tracks)

    def keep_all(t: list[BoxTrack], _p: object) -> list[BoxTrack]:
        return list(t)

    monkeypatch.setattr(module, "track_boxes", fixed_tracks)
    monkeypatch.setattr(module, "filter_tracks", keep_all)
    sid = scene.hand_labels[0].stream_id
    assert sid is not None
    evidence = WearerEvidence.from_labels([*scene.hand_labels, scene.other_person], sid)
    return scene, video, evidence, sid


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-73: 객체·OWLv2 어댑터가 wearer_suppression.stages에 자기 단계가 없어도 억제",
)
@pytest.mark.parametrize("which", ["objects", "tools"])
def test_wearer_suppression_respects_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """`prelabel.yaml wearer_suppression.stages`에 objects·tools가 없으면 그 단계는 착용자 몸을
    억제하지 않는다 (RtmPose는 이미 stages를 확인한다).

    근거: ADR 0058 착용자 억제는 stages에 적은 단계에만, 버전 태그(+w)도 그 단계에만 붙는다.
    지금은 태그만 빠지고 억제는 계속돼, 억제 정책을 바꿔도 버전이 그대로라 다시 돌지 않는다.
    """
    from dlp_prelabel.adapters import mediapipe_models as mpm
    from dlp_prelabel.adapters import owl_objects as owl
    from dlp_prelabel.policy import load_policy as load_prelabel_policy
    from dlp_schema.predictor import Clip

    module = mpm if which == "objects" else owl
    scene, video, evidence, sid = _arm_tracks_setup(tmp_path, monkeypatch, module)
    if which == "tools":

        class FakeOwl:
            """탐지는 track_boxes 패치로 대신하므로 아무것도 내지 않는다."""

            def __init__(self, *a: object, **k: object) -> None:
                pass

            def detect(self, img: object, thr: object) -> list[object]:
                return []

        monkeypatch.setattr(owl, "Owlv2", FakeOwl)
    base = load_prelabel_policy(ROOT)
    ws = base.wearer_suppression.model_copy(
        update={"stages": ("body",), "track_share": 0.5, "other_min_area": 0.2}
    )
    policy = base.model_copy(update={"wearer_suppression": ws})
    cls = mpm.MediaPipeObjects if which == "objects" else owl.OwlObjects
    now = FIXED_TIME
    plain = cls(ROOT, policy, ontology_version="1.0.0", now=now)
    expected = sorted(x.label_id for x in plain.run(Clip(scene.session_id, sid, video)))
    adapter = cls(ROOT, policy, ontology_version="1.0.0", now=now)
    adapter.set_wearer(evidence)
    got = sorted(x.label_id for x in adapter.run(Clip(scene.session_id, sid, video)))
    assert "+w" not in adapter.version
    assert got == expected  # 단계 목록에 없으니 억제하지 않은 결과와 같다


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-75: 결과 0개인 예측기 단계가 매 실행 다시 돎",
)
def test_prelabel_empty_predictor_is_idempotent(tmp_path: Path) -> None:
    """같은 입력·같은 버전으로 두 번 돌리면 두 번째는 예측기를 다시 부르지 않는다. 결과가 0개인
    예측기(손이 한 번도 안 보인 영상)도 마찬가지다.

    근거: CLAUDE.md "각 파이프라인 단계는 멱등적인 `dlp <단계>`", 모델 버전으로 재실행을 정한다.
    지금은 '이력에 같은 버전 라벨이 있나'만 봐서 결과 0개면 매번 원본을 받고 다시 돈다.
    """
    import yaml

    from dlp_fixtures.video import write_video
    from dlp_media.ingest import ingest_session, load_manifest
    from dlp_media.storage import LocalStore
    from dlp_prelabel.policy import load_policy as load_prelabel_policy
    from dlp_prelabel.runner import run_prelabel
    from dlp_schema.db.repository import register_ontology
    from dlp_schema.labels import LabelRecord
    from dlp_schema.predictor import Clip
    from dlp_schema.testing import sqlite_db

    times = list(range(0, 1001, 50))
    frames = ((t, np.full((240, 320, 3), 90, np.uint8)) for t in times)
    write_video(tmp_path / "bodycam.mp4", frames, width=320, height=240)
    manifest = {
        "session_id": "s1", "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), "utf-8")
    calls: list[str] = []

    class Empty:
        """손을 하나도 찾지 못하는 손 예측기."""

        name = "hands"
        version = "empty-hands-1"

        def run(self, clip: Clip) -> list[LabelRecord]:
            calls.append(clip.session_id)
            return []

    raw = LocalStore(tmp_path / "store", "dlp-raw")
    ont = load_ontology(ROOT / "config" / "ontology" / "v1")
    policy = load_prelabel_policy(ROOT)
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ont)
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    for _ in range(2):
        with engine.begin() as conn:
            run_prelabel(conn, "s1", raw, [Empty()], policy, ont, FIXED_TIME)  # pyright: ignore[reportArgumentType]
    engine.dispose()
    assert calls == ["s1"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-76: OWLv2 버전에 scene 절 전체 해시가 들어가 VLM timeout만 바꿔도 재실행",
)
def test_owl_version_ignores_scene_timeout() -> None:
    """장면이 같고 도구 질의가 같으면 OWLv2 도구 단계 버전도 같아야 한다. 장면 분류 VLM의 시간
    제한은 결과에 영향이 없다.

    근거: F-PAR-07과 같은 원칙(결과와 무관한 timeout·backoff는 정책 해시에서 뺀다). 버전이
    바뀌면 검수 전 도구 박스를 지우고 다시 만든다.
    """
    from dlp_prelabel.adapters.owl_objects import OwlObjects
    from dlp_prelabel.policy import load_policy as load_prelabel_policy
    from dlp_prelabel.scene import SceneResult

    p = load_prelabel_policy(ROOT)
    q = p.model_copy(
        update={"scene": p.scene.model_copy(update={"timeout_s": p.scene.timeout_s + 1})}
    )
    a = OwlObjects(ROOT, p, ontology_version="1.0.0", now=FIXED_TIME)
    b = OwlObjects(ROOT, q, ontology_version="1.0.0", now=FIXED_TIME)
    for o in (a, b):
        o.set_scene(SceneResult("kitchen", 0.9, "vlm"))
    assert a.version == b.version


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="F-PAR-78: 월드 좌표 변환이 표본 flags를 버림",
)
def test_world_samples_keep_constraint_flags() -> None:
    """카메라 좌표 표본의 물리 제약 표시(`flags`, 예: reach)는 좌표계를 바꿔도 남아야 한다.

    근거: `dlp_prelabel.constraints3d` 표시는 표본의 성질이고(좌표계와 무관), 월드 변환은
    위치·신뢰도만 바꾼다. 자세가 단위 행렬이면 위치도 같다.
    """
    from dlp_prelabel.campose import PoseSample
    from dlp_prelabel.worldframe import to_world_samples
    from dlp_schema.labels import Constraint3DFlag, Lift3DMethod, Trajectory3DSample

    cam = [
        Trajectory3DSample(t_ms=0, x=0.0, y=0.1, z=1.2, method=Lift3DMethod.HAND_FIT,
                           confidence=0.4, flags=(Constraint3DFlag.REACH,)),
        Trajectory3DSample(t_ms=100, x=0.0, y=0.1, z=0.5, method=Lift3DMethod.HAND_FIT,
                           confidence=0.8),
    ]  # fmt: skip
    poses = [PoseSample(t, np.eye(3), np.zeros(3), 1.0) for t in (0, 100)]
    world = to_world_samples(cam, poses, 500)
    assert [w.flags for w in world] == [c.flags for c in cam]


@pytest.mark.xfail(
    strict=True,
    raises=ValidationError,
    reason="F-PAR-79: 긴 세션 ID에서 3D 라벨 ID가 128자를 넘어 ValidationError",
)
def test_lift3d_label_ids_fit_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """손 2D 트랙 ID가 Identifier(128자) 안이면 그 3D 단계 라벨 ID도 128자 안이어야 한다.

    근거: `dlp_schema.common.Identifier` 패턴 `{0,127}`. 2D 단계가 통과한 세션을 3D 단계가 계약
    예외로 실패시키면 안 된다 (ID를 줄이거나 해시로 접어야 한다).
    """
    import dlp_prelabel.lift3d as lift3d
    from dlp_fixtures.hand3d import generate_hand3d_scenario
    from dlp_prelabel.policy import load_policy as load_prelabel_policy
    from dlp_schema.episode import version_tag

    scenario = generate_hand3d_scenario(0, n_frames=10)
    sid = "S" * 88  # Identifier 안의 세션 ID
    hands_id = f"{sid}-bodycam-hands-{version_tag('mediapipe-x')}-right"
    assert len(hands_id) <= 128

    class Lifter(lift3d.DepthLifter):
        """가중치 파일 없이 합성 깊이 모델을 쓰는 3D 단계."""

        def __init__(self) -> None:
            self.policy = load_prelabel_policy(ROOT)
            self.now = FIXED_TIME
            self.version = "depth3d-test+hf5"
            self.model = scenario.depth  # pyright: ignore[reportAttributeAccessIssue]
            self.last_focal = None

    def frames(_path: object) -> Any:
        return scenario.frames()

    monkeypatch.setattr(lift3d, "iter_frames", frames)
    track = scenario.labels[0].model_copy(update={"session_id": sid, "label_id": hands_id})
    out = Lifter().run(
        Path("x.mp4"), session_id=sid, stream_id="bodycam", tracks=[track],
        calib=scenario.intrinsics, ontology_version="1.0.0",
    )  # fmt: skip
    assert out and all(len(x.label_id) <= 128 for x in out)


def test_new_ls_regions_with_same_content_get_distinct_ids() -> None:
    """검수자가 Label Studio에서 같은 시각·같은 이벤트를 두 번 새로 그렸으면(영역 ID aa1·bb2)
    reconcile은 새 레코드 둘에 서로 다른 라벨 ID를 줘야 한다.

    근거: 라벨은 덮어쓰지 않는다(CLAUDE.md 규칙). 같은 ID면 둘째 저장이 첫째와 충돌하거나
    하나를 잃는다. 영역 ID가 다르니 서로 다른 입력이다.
    """
    from dlp_review.labelstudio import from_ls_results
    from dlp_review.reconcile import reconcile

    def region(rid: str) -> list[dict[str, Any]]:
        return [
            {"id": rid, "from_name": "labels", "to_name": "ts", "type": "timeserieslabels",
             "value": {"start": 300, "end": 300, "instant": True,
                       "timeserieslabels": ["event:slip"]}},
            {"id": rid, "from_name": "severity", "to_name": "ts", "type": "choices",
             "value": {"choices": ["2"]}},
        ]  # fmt: skip

    items = from_ls_results([*region("aa1"), *region("bb2")], set())
    out = reconcile(
        [], items, session_id="s001", ontology_version="1.0.0", reviewer_id="rv", now=FIXED_TIME
    )
    ids = [x.label_id for x in out.new_records]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_ai_ok_numbers_must_be_json_integers() -> None:
    """응답 형식의 ok는 정수 번호 목록이다. true·false·"2"·3.0은 형식이 틀린 번호라 위반(또는
    파싱 실패)이고 어느 라벨도 ok로 기록하면 안 된다.

    근거: ADR 0040 3.2 응답 재검사(형식 밖 값은 위반, 빠진 대상은 unsure + response_invalid).
    Python에서 True == 1이고 int("2")가 되는 것에 기대면 모델이 쓰지 않은 판정이 기록된다.
    """
    import json

    from dlp_aireview.policy import load_policy as load_aireview_policy
    from dlp_aireview.validate import RequestView, validate_answer
    from dlp_schema.aireview import AiGroup
    from dlp_schema.labels import Provenance, Source

    model = Provenance(source=Source.MODEL, model_version="m")
    labels = tuple(
        make_label({"kind": "gap", "gap_type": "idle", "hand": "right"}, label_id=f"g{i}",
                   t_start_ms=i * 1000, t_end_ms=i * 1000 + 900, provenance=model, confidence=0.7)
        for i in range(4)
    )  # fmt: skip
    view = RequestView(AiGroup.TEMPORAL, labels, tuple(range(6)), (("cup_01", "cup"),))
    raw = json.dumps({"ok": [True, False, "2", 3.0], "items": [], "missing": []})
    answer = validate_answer(raw, view, load_ontology(ROOT / "config" / "ontology" / "v1"),
                             load_aireview_policy(ROOT))  # fmt: skip
    assert not any(d.verdict.value == "ok" for d in answer.items)


def test_yolox_base_weights_checks_whole_lineage(tmp_path: Path) -> None:
    """출발 가중치 top의 base_model mid가 forbidden 데이터셋으로 학습됐으면 top으로도 학습을
    시작하면 안 된다 (top 자체 항목에는 위반이 없다).

    근거: ADR 0010·0062 출발 가중치는 계보 전체의 라이선스 분류를 물려받는다.
    `dataset_violations`는 "mid: ..."로 계보 위반을 내는데 `base_weights`는 이름 접두사
    (`startswith("top")`)로만 걸러 놓친다. 정답: top도 라이선스 위반으로 거절.
    """
    import shutil

    import yaml

    from dlp_train import yolox
    from dlp_train.policy import load_policy as load_train_policy

    root = tmp_path / "root"
    shutil.copytree(ROOT / "config", root / "config")
    base = load_train_policy(ROOT).yolox.base_model
    models_path, ds_path = root / "config/models.yaml", root / "config/external_datasets.yaml"
    reg: Any = yaml.safe_load(models_path.read_text("utf-8"))
    ds: Any = yaml.safe_load(ds_path.read_text("utf-8"))
    dsname = next(iter(ds["datasets"]))
    ds["datasets"][dsname]["commercial"] = "forbidden"
    spec = reg["models"][base]
    reg["models"]["mid"] = dict(spec, training_datasets=[dsname], base_model=base,
                                commercial="review", path="data/models/mid.pth")  # fmt: skip
    reg["models"]["top"] = dict(spec, training_datasets=[], base_model="mid",
                                commercial="review", path="data/models/top.pth")  # fmt: skip
    models_path.write_text(yaml.safe_dump(reg, allow_unicode=True), "utf-8")
    ds_path.write_text(yaml.safe_dump(ds, allow_unicode=True), "utf-8")
    with pytest.raises(yolox.YoloxTrainingError, match="라이선스"):
        yolox.base_weights(root, "top")


def test_yolox_settings_reject_string_booleans() -> None:
    """`dlp train run --param allow_cpu=False`는 JSON 해석에 실패해 문자열 "False"가 된다.
    문자열을 참으로 보아 GPU 없는 컴퓨터에서 CPU 학습을 허용하면 안 된다 (거절이 정답).
    `fp16=False`도 fp16을 켜면 안 된다.

    근거: CLAUDE.md "GPU가 없으면 `--param allow_cpu=true` 없이는 거절한다"(ADR 0062).
    `dlp_cli.train_cmds`의 `--param` 해석은 JSON(`false`)만 bool로 바꾸고 나머지는 문자열로 둔다.
    """
    from dlp_train import yolox
    from dlp_train.policy import load_policy as load_train_policy

    policy = load_train_policy(ROOT).yolox
    for value in ("False", "no"):
        with pytest.raises(yolox.YoloxTrainingError):
            yolox.settings(policy, {"allow_cpu": value}, gpu=False)
    try:
        fp16 = yolox.settings(policy, {"fp16": "False"}, gpu=True).fp16
    except yolox.YoloxTrainingError:
        return  # 형식이 틀린 값으로 거절해도 맞다
    assert fp16 is False


def test_augment_hflip_swaps_arm_and_mask_hands() -> None:
    """좌우 뒤집기는 왼손을 오른손으로 바꾼다. 3D 손(`left_hand`)·2D 팔(arm3 left)은 바뀌는데
    3D 팔꿈치 개체 `left_arm`과 손 마스크 키 `left_hand`도 함께 바뀌어야 한다.

    근거: ADR 0057 뒤집기의 좌우 교환, ADR 0058 착용자 팔 entity `<손>_arm`, ADR 0059 손 마스크
    entity `<손>_hand`. 한쪽만 바꾸면 같은 표본에서 손과 팔·마스크의 좌우가 어긋난다.
    """
    from dlp_media.geometry import hflip, identity
    from dlp_train.augment import KeypointItem, MaskItem, Ops, Point3DItem, Sample, augment

    w, h = 64, 48
    ops = Ops(
        transform=identity(w, h).then(hflip(w, h)), brightness=0, contrast=0, blur_sigma=0,
        noise_std=0, noise_seed=0, jpeg_quality=None, motion_px=0, motion_deg=0.0,
        highlights=(), k1=0.0,
    )  # fmt: skip
    sample = Sample(
        image=np.zeros((h, w, 3), np.uint8),
        points3d=(Point3DItem("left_hand", "wrist", (0.2, 0.0, 0.5)),
                  Point3DItem("left_arm", "elbow", (0.3, 0.1, 0.6))),
        keypoints=(KeypointItem("arm3", "left",
                                np.array([[10, 10, 2], [20, 20, 2], [30, 30, 2]], float)),),
        masks=(MaskItem("left_hand", np.ones((h, w), bool)),),
    )  # fmt: skip
    out = augment(sample, ops)
    assert [p.entity for p in out.points3d] == ["right_hand", "right_arm"]
    assert [k.hand for k in out.keypoints] == ["right"]
    assert [m.key for m in out.masks] == ["right_hand"]


def _console(tmp_path: Path, bind: str = "127.0.0.1") -> tuple[Any, Any, str]:
    """콘솔 앱 시험 도구: (TestClient, Services, 토큰). 서버 예외는 500 응답으로 받는다.

    DB·S3에는 붙지 않는다 (아래 시험은 연결 전에 끝나는 경로만 쓴다).
    """
    from fastapi.testclient import TestClient

    from dlp_console.app import Services, create_app
    from dlp_console.settings import default_settings

    settings = default_settings(ROOT, tmp_path / "work")
    services = Services(settings)
    app = create_app(settings, services, bind_host=bind)
    return TestClient(app, raise_server_exceptions=False), services, settings.token


def test_console_delete_upload_blocks_running_session_with_trailing_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """세션 s1의 수집 작업이 도는 중이면 `DELETE /api/uploads/s1%20`도 409여야 한다.

    근거: 삭제는 `_ident`로 끝 공백을 지운 s1의 사본을 지우는데, 실행 중 검사는 공백이 남은
    잠금 키(`session:s1 `)로 찾는다. 올리기 API는 strip한 키로 잠근다(ADR 0061 작업 잠금).
    """
    client, services, token = _console(tmp_path)
    (services.settings.work_dir / "uploads" / "s1").mkdir(parents=True)

    def running(lock: str) -> object | None:
        return object() if lock == "session:s1" else None

    monkeypatch.setattr(services.runner, "running", running)
    headers = {"host": "127.0.0.1", "x-console-token": token}
    assert client.delete("/api/uploads/s1", headers=headers).status_code == 409
    assert client.delete("/api/uploads/s1%20", headers=headers).status_code == 409
    assert (services.settings.work_dir / "uploads" / "s1").is_dir()


def test_console_host_check(tmp_path: Path) -> None:
    """콘솔 Host 검사(ADR 0061 4절: 묶은 주소·localhost만)의 네 경우.

    - `--host 0.0.0.0 --allow-remote`: 원격 브라우저의 Host(서버 LAN IP:포트)를 받는다.
    - `--host ::1`: 브라우저 Host `[::1]:8090`을 받는다 (지금은 ":"로 잘라 "[" 만 남는다).
    - 운영(127.0.0.1)에서 Host `testserver`는 묶은 주소도 localhost도 아니라 400이다.
    - 비ASCII 토큰은 401이다 (`secrets.compare_digest`가 str 비ASCII에 TypeError → 500).
    """
    got: dict[str, int] = {}
    c, _, tok = _console(tmp_path / "a", "0.0.0.0")
    got["remote"] = c.get("/api/jobs", headers={"host": "192.168.0.10:8090",
                                                "x-console-token": tok}).status_code  # fmt: skip
    c, _, tok = _console(tmp_path / "b", "::1")
    got["ipv6"] = c.get("/api/jobs", headers={"host": "[::1]:8090",
                                              "x-console-token": tok}).status_code  # fmt: skip
    c, _, tok = _console(tmp_path / "c")
    plain = {"host": "testserver", "x-console-token": tok}
    got["testserver"] = c.get("/api/jobs", headers=plain).status_code
    latin1 = {b"host": b"127.0.0.1", b"x-console-token": "ÿ".encode("latin-1")}
    got["token"] = c.get("/api/jobs", headers=latin1).status_code
    assert got == {"remote": 200, "ipv6": 200, "testserver": 400, "token": 401}


def test_cli_raw_store_purposes_are_service_purposes() -> None:
    """`dlp` 명령이 원본 감사 저장소를 여는 목적(purpose)은 모두 ops.yaml
    `audit.service_purposes`에 있어야 한다. 없으면 서비스 계정의 정상 파이프라인 읽기
    (ops.privacy-audit·privacy.audit-sample·review.hints)가 월간 감사 리포트에 이상으로 표시된다.

    근거: ADR 0020·0021 원본 접근 감사, `dlp_ops.audit.audit_report`의 서비스 목적 판정.
    """
    import re

    from dlp_ops.policy import load_policy as load_ops_policy

    used: set[str] = set()
    for f in (ROOT / "packages/cli/src/dlp_cli").glob("*.py"):
        used |= set(re.findall(r'raw_store\([^)]*?"([a-z_.-]+)"\)', f.read_text("utf-8")))
    assert {"ops.privacy-audit", "privacy.audit-sample", "review.hints"} <= used
    assert sorted(used - set(load_ops_policy(ROOT).audit.service_purposes)) == []


def test_console_upload_does_not_truncate_consent_version() -> None:
    """확인한 동의서 버전은 동의의 근거라 다른 값으로 바꿔 적으면 안 된다. 너무 길면 거부하고
    (ParamError), 받으면 그대로 적어야 한다.

    근거: upload.build_manifest "영상만으로 동의를 추정하지 않습니다"(ADR 0061), 동의 철회·사용
    중지는 동의서 버전으로 찾는다. 지금은 `[:64]`로 잘라 다른 버전 문자열이 남는다.
    """
    from dlp_console.steps import ParamError
    from dlp_console.upload import UploadForm, build_manifest

    long_version = "consent-" + "v" * 80
    form = UploadForm("s1", "w01", "site01", "cleaning", long_version)
    try:
        manifest = build_manifest(form, "bodycam.mp4")
    except ParamError:
        return
    assert manifest.consent_version == long_version


def test_console_global_step_bad_json_is_400(tmp_path: Path) -> None:
    """`POST /api/global/<단계>`의 본문이 JSON이 아니거나 숫자가 범위 밖(1e400 → inf)이면
    입력 오류(400)다. 서버 오류(500)가 아니다.

    근거: 콘솔은 입력 오류를 ParamError → 400으로 알린다(app.param_error, ADR 0061).
    """
    client, _, token = _console(tmp_path)
    headers = {"host": "127.0.0.1", "x-console-token": token,
               "content-type": "application/json"}  # fmt: skip
    codes = [
        client.post("/api/global/ops_weekly", headers=headers, content=body).status_code
        for body in ("{bad", '{"weeks": 1e400}')
    ]
    assert codes == [400, 400]
