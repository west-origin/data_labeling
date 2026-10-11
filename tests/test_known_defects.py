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
