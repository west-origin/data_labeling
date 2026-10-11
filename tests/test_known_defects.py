# PyAV 타입 정보가 일부 비어 있어 이 모듈에서만 알 수 없는 멤버 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""확인된 미수정 결함의 재현 테스트 (docs/handover/issues.yaml).

각 테스트는 **올바른 동작**을 단언하고 `xfail(strict=True)`로 표시한다. 지금은 결함 때문에 실패해야
하며(xfail), 누군가 결함을 고쳐 테스트가 통과하면 strict 규칙에 따라 XPASS가 실패로 보고된다.
그때 이 표시를 지우고 레지스트리 항목의 status를 fixed로 바꾼다 (`tests/test_issue_registry.py`가
레지스트리의 repro 경로와 이 파일의 reason ID가 맞는지 검사한다).

reason은 `<레지스트리 ID>: <한 줄 요약>` 형식이다. 서비스(DB·도구) 없이 돈다.
정답 근거는 각 테스트 docstring에 적었다 (계약·ADR·정책 문구).
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import av
import numpy as np
import pytest
import yaml

from dlp_schema import repo_root
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

ROOT = repo_root()


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


@pytest.mark.xfail(strict=True, reason="F-REV-01: dlp review의 ReviewSetup이 정책 파일을 무시")
def test_review_setup_uses_selected_policy_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`DLP_REVIEW_POLICY_FILE`로 고른 정책의 CVAT 계정 연결을 작업 생성도 써야 한다.

    근거: review_cmds 모듈 docstring("CLI에서는 DLP_REVIEW_POLICY_FILE로 정책 파일을 바꿀 수 있다"),
    파일럿 안내의 `dlp review create --stage privacy --assignee pilot-privacy`.
    """
    from dlp_cli import review_cmds
    from dlp_cli.review_config import load_review_policy

    data = yaml.safe_load((ROOT / "config/policies/review.yaml").read_text("utf-8"))
    data["reviewers"]["privacy"] = ["pilot-privacy"]
    data["cvat"]["users"] = {"pilot-privacy": "cvat-privacy"}
    path = tmp_path / "review.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(path))
    # 저장소·도구 클라이언트는 만들지 않는다 (정책 연결만 본다)
    monkeypatch.setattr(review_cmds, "raw_store", Mock(return_value=Mock()))
    monkeypatch.setattr(review_cmds, "S3Store", Mock())
    args = argparse.Namespace(review_command="create", url=None, no_cvat=True, no_ls=True)
    setup = review_cmds._setup(args)  # pyright: ignore[reportPrivateUsage]
    assert setup.cvat_policy().users == load_review_policy(ROOT).cvat.users


@pytest.mark.xfail(
    strict=True, reason="F-REV-03: Label Studio에서 손 상태를 none으로 바꾸면 수집 실패"
)
def test_ls_hand_state_changed_to_none_is_collectable() -> None:
    """접촉(tool) 손 상태를 화면에서 `none`으로 바꾸면 대상·파지 유형이 비워진 채 받아야 한다.

    근거: hand.yaml — none이면 target_id·body_part·grasp_type을 비운다. 장갑 접촉 불일치로 올라온
    라벨의 대표 수정이 "접촉 → none"이다.
    """
    from dlp_review.labelstudio import from_ls_results, to_ls_results
    from dlp_schema.labels import Hand, HandStatePayload

    p = HandStatePayload(
        hand=Hand.RIGHT,
        contact_target_kind="tool",
        target_id="mop_01",
        grasp_type="power",
        role="active",
    )
    [r] = to_ls_results([make_label(p, label_id="hs1")])
    r["value"]["timeserieslabels"] = ["hand.right:none"]
    [item] = from_ls_results([r], {"hs1"})
    assert isinstance(item.payload, HandStatePayload)
    assert item.payload.contact_target_kind == "none"
    assert item.payload.target_id is None and item.payload.grasp_type is None


@pytest.mark.xfail(strict=True, reason="F-REV-11: 이벤트 속성 입력란을 모두 비우면 원래 값이 남음")
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
    strict=True, reason="F-SCH-01: 양손 행동(같은 action_id)과 상위 구간이 함께 저장되지 않음"
)
def test_bimanual_action_with_segment_is_valid(ontology: Ontology) -> None:
    """양손 행동은 두 손 레코드가 같은 action_id를 쓴다. 상위 구간이 그 행동을 가리켜도
    유효해야 한다.

    근거: ADR 0002 §3, labels.ActionPayload docstring(양손 행동 = 같은 action_id).
    """
    from dlp_schema.label_validation import validate_session_labels

    right = make_label(action_payload(action_id="a001", hand="right"), label_id="r")
    left = make_label(action_payload(action_id="a001", hand="left"), label_id="l")
    seg = make_label(
        {
            "kind": "segment",
            "segment_id": "sk",
            "level": "skill",
            "ref_id": "wipe",
            "child_ids": ["a001"],
        },
        label_id="sk",
    )
    validate_session_labels(ontology, [], [right, left, seg])


@pytest.mark.xfail(strict=True, reason="F-SCH-02: 승인 트랙의 모델 자식 삭제로 승인 라벨이 사라짐")
def test_retracting_model_child_of_approved_label_is_rejected(ontology: Ontology) -> None:
    """승인된 원래 트랙의 모델 자식(착용자 사본)을 모델이 지우면 승인 라벨이 운영에서
    사라진다. 막아야 한다.

    근거: CLAUDE.md "검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다"(ADR 0015, 0019).
    """
    from dlp_schema.episode import retractions
    from dlp_schema.label_validation import (
        ProtectedLabelRetractionError,
        validate_session_labels,
    )
    from dlp_schema.labels import Provenance, Source, Verification, VerificationState

    model = Provenance(source=Source.MODEL, model_version="p1")
    box = {
        "kind": "box_track",
        "entity_id": "rag_01",
        "class_id": "rag",
        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}],
    }
    approved = Verification(
        state=VerificationState.HUMAN_APPROVED,
        reviewer_id="r1",
        reviewed_at=FIXED_TIME + timedelta(minutes=1),
    )
    orig = make_label(
        box,
        label_id="orig",
        stream_id="tp1",
        provenance=model,
        confidence=0.9,
        verification=approved,
    )
    copy = make_label(
        box,
        label_id="copy",
        stream_id="tp1",
        provenance=model,
        confidence=0.9,
        parent_label_id="orig",
    )
    with pytest.raises(ProtectedLabelRetractionError):
        validate_session_labels(ontology, [orig, copy], retractions([copy], "p2", FIXED_TIME))


@pytest.mark.xfail(
    strict=True, reason="F-SCH-03: 오류 삽입 사본이 있으면 이관이 운영 라벨까지 재검수로 보냄"
)
def test_migration_ignores_seeded_copies(ontology: Ontology) -> None:
    """오류 삽입 사본(같은 action_id)은 운영 라벨이 아니다. 운영 라벨의 이관을 막으면 안 된다.

    근거: CLAUDE.md "오류 삽입 레코드와 그 후손은 운영 라벨이 아니다",
    dlp_review.ops.seeding(ID만 바꾼 사본).
    """
    from dlp_schema.migration import OntologyMigration, migrate_labels

    target = Ontology.model_validate(ontology.model_dump() | {"version": "1.1.0"})
    action = make_label(action_payload(), label_id="act")
    seg = make_label(
        {
            "kind": "segment",
            "segment_id": "sk",
            "level": "skill",
            "ref_id": "wipe",
            "child_ids": ["a001"],
        },
        label_id="sk",
    )
    seeded = action.model_copy(update={"label_id": "seed-as1-0000", "seeded_error": True})
    m = OntologyMigration(from_version="1.0.0", to_version="1.1.0")
    result = migrate_labels([action, seg, seeded], m, FIXED_TIME, target_ontology=target)
    assert {x.label_id for x in result.migrated} >= {"act", "sk"}


@pytest.mark.xfail(strict=True, reason="F-SCH-06: 기존 위반이 있으면 무관한 새 라벨도 저장 거부")
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
    strict=True, reason="F-PMS-01: 시간대가 다른 같은 recorded_at으로 재수집하면 충돌"
)
def test_reingest_same_instant_other_offset_is_not_conflict() -> None:
    """매니페스트의 +09:00 시각과 DB에서 읽은 UTC 시각은 같은 순간이므로 재수집 충돌이 아니다.

    근거: ingest 모듈의 멱등 재수집 설명, PostgreSQL timestamptz는 오프셋을 보존하지 않는다.
    """
    from dlp_media.ingest import _ingest_fields  # pyright: ignore[reportPrivateUsage]
    from dlp_schema.testing import make_session

    kst = timezone(timedelta(hours=9))
    s = make_session().model_copy(update={"recorded_at": datetime(2026, 10, 9, 9, 0, tzinfo=kst)})
    s = type(s).model_validate(s.model_dump())
    db = type(s).model_validate({**s.model_dump(), "recorded_at": s.recorded_at.astimezone(UTC)})
    assert _ingest_fields(s) == _ingest_fields(db)


def _odd_video(path: Path, w: int, h: int) -> None:
    """홀수 크기 무손실(ffv1) 영상 5프레임."""
    with av.open(str(path), "w") as c:
        vs = c.add_stream("ffv1", rate=10)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv444p"
        vs.time_base = Fraction(1, 1000)
        for i in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((h, w, 3), 40 * i, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i * 100, Fraction(1, 1000)
            c.mux(vs.encode(frame))
        c.mux(vs.encode(None))


@pytest.mark.xfail(
    strict=True, reason="F-PMS-02: 홀수 크기 영상은 수집되지만 블러본을 만들 수 없음"
)
def test_odd_size_video_renders_blurred(tmp_path: Path) -> None:
    """수집이 받은 영상(홀수 크기 포함)은 블러본도 만들 수 있어야 한다 (아니면 수집에서 거부).

    근거: 프록시는 홀수 크기를 짝수로 줄여 처리한다(ADR 0028). 렌더만 libx264 yuv420p에서 실패한다.
    """
    from dlp_privacy.render import render_blurred

    src = tmp_path / "odd.mkv"
    _odd_video(src, 101, 75)
    render_blurred(
        src,
        tmp_path / "b.mp4",
        [],
        mode="mosaic",
        min_block_px=6,
        blocks_per_box=5,
        encoder_rate=30,
        crf=18,
    )


@pytest.mark.xfail(strict=True, reason="F-PMS-05: 반올림 ms가 같은 두 프레임이면 블러 탐지 실패")
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


@pytest.mark.xfail(strict=True, reason="F-PMS-07: NaN이 섞인 장갑 시각을 받아들임")
def test_glove_times_with_nan_rejected() -> None:
    """시각에 NaN이 있으면 단조 증가 검사가 무력해진다. 수집 초기에 거부해야 한다.

    근거: glove.GloveData의 "시각은 엄격히 증가" 검사 의도.
    """
    from dlp_media.glove import GloveData

    with pytest.raises(ValueError):
        GloveData(np.array([0.0, 10.0, np.nan, 5.0]), {"p": np.ones(4)}, "parquet")


@pytest.mark.xfail(
    strict=True, reason="F-PAR-07: 결과와 무관한 VLM timeout이 행동 정책 해시에 들어감"
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


@pytest.mark.xfail(strict=True, reason="F-ETA-06: 액티브 러닝 수정률이 수정 후 클래스로 세어짐")
def test_correction_rate_counts_model_predicted_class() -> None:
    """모델이 cup으로 낸 10개 중 8개를 사람이 mug로 고쳤으면 cup의 수정률이 높아야 한다.

    근거: dlp_active.terms.CorrectionRateTerm은 검수 대기 모델 라벨의 예측 클래스로 수정률을 찾는다.
    """
    from dlp_active.policy import load_policy
    from dlp_active.rates import correction_rates
    from dlp_schema.labels import (
        LabelRecord,
        Provenance,
        Source,
        Verification,
        VerificationState,
    )

    policy = load_policy(ROOT)
    policy = policy.model_copy(
        update={"correction": policy.correction.model_copy(update={"prior_weight": 0})}
    )

    def box(cls: str) -> dict[str, object]:
        return {
            "kind": "box_track",
            "entity_id": f"{cls}_1",
            "class_id": cls,
            "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}],
        }

    corrected = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id="r", reviewed_at=FIXED_TIME
    )
    approved = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r", reviewed_at=FIXED_TIME
    )
    model = Provenance(source=Source.MODEL, model_version="m1")
    history: list[LabelRecord] = []
    for i in range(10):
        history.append(
            make_label(
                box("cup"),
                label_id=f"m{i}",
                session_id="s",
                stream_id="bodycam",
                provenance=model,
                confidence=0.9,
                verification=approved if i >= 8 else Verification(),
            )
        )
        if i < 8:
            history.append(
                make_label(
                    box("mug"),
                    label_id=f"h{i}",
                    session_id="s",
                    stream_id="bodycam",
                    parent_label_id=f"m{i}",
                    verification=corrected,
                )
            )
    rates = correction_rates([history], policy)
    cup = next(v for k, v in rates.classes.items() if "cup" in str(k))
    assert cup.rate > 0.5  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.xfail(strict=True, reason="F-ETA-08: 큐레이션이 공간 마스크를 마스터 시각처럼 변환")
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
