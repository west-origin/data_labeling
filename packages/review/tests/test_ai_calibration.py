"""AI 판정 표시 보정 시험 (ADR 0062, `dlp_review.ops.ai_calibration`·`dlp_review.ai_calibration`).

정답: 합성 판정·결과 기록 행. 칸마다 사람이 고친 비율을 정해 둔다 (mop 의심은 60건 중 3건 =
쓸모없음, cup 의심은 60건 중 40건 = 쓸모 있음, 틀림은 10건뿐 = 표본 부족). 보정이 쓸모없는
칸만 숨기고, 표본이 부족한 칸은 기본 동작(표시)을 두고, 대조군만 쓰고, ok·증가 프로필을 바꾸지
않는지 본다.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_review.ai_calibration import AiCalibrationError, load_calibration, to_bytes
from dlp_review.ai_display import AiDisplayConfig, AiView, Judged, escalated_ids, exposure_for
from dlp_review.ops.ai_calibration import (
    calibration_effect,
    cross_checked_effect,
    fit_calibration,
)
from dlp_review.ops.ai_quality import AiRow
from dlp_review.ops.policy import AiCalibrationFit, AiReviewDisplayPolicy
from dlp_schema import repo_root
from dlp_schema.aireview import AiArm, AiDecision, AiHumanAction, AiVerdictValue
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTool
from dlp_schema.testing import AI_TARGET_RULE, AI_VERSION, FIXED_TIME, make_ai_verdict, make_label

V = AiVerdictValue
H = AiHumanAction
SINCE, UNTIL = FIXED_TIME - timedelta(days=30), FIXED_TIME
FIT = AiCalibrationFit()  # 기본: 유효 표본 30, 문턱 0.15


def row(
    n: int,
    verdict: AiVerdictValue,
    changed: bool,
    *,
    cls: str = "mop",
    arm: AiArm = AiArm.HOLDOUT,
    task: str | None = None,
    **kw: Any,
) -> AiRow:
    """box_track 결과 행 하나 (기본: 대조군, 표준 작업, 판정 버전 AI_VERSION)."""
    data: dict[str, Any] = {
        "task_key": task or f"cvat:{n % 7}", "label_id": f"{cls}-{verdict.value}-{n}",
        "kind": "box_track", "label_class": cls, "arm": arm, "verdict": verdict,
        "confidence": None, "has_suggestion": False, "shown": arm is AiArm.SHOWN,
        "human": H.CORRECTED if changed else H.APPROVED, "decision": AiDecision.NONE,
        "apply_result": None, "match": "na", "version": AI_VERSION,
    }  # fmt: skip
    return AiRow(**(data | kw))


def rows() -> list[AiRow]:
    """mop 의심 60건(3 고침), cup 의심 60건(40 고침), mop 틀림 10건(9 고침), ok 40건(2 고침)."""
    out = [row(i, V.SUSPECT, i < 3) for i in range(60)]
    out += [row(i, V.SUSPECT, i < 40, cls="cup") for i in range(60)]
    out += [row(i, V.WRONG, i < 9) for i in range(10)]
    out += [row(i, V.OK, i < 2, cls="cup") for i in range(40)]
    return out


def test_fit_hides_only_confidently_useless_cells() -> None:
    """mop 의심(정밀도 5%, 상한 < 15%)만 숨기고 cup 의심은 표시, 틀림은 표본 부족이라 칸이 없다."""
    cal = fit_calibration(rows(), FIT, (AI_VERSION,), SINCE, UNTIL)
    cells = {(c.verdict, c.kind, c.label_class): c for c in cal.cells}
    mop = cells[("suspect", "box_track", "mop")]
    assert not mop.show and mop.precision == pytest.approx(0.05) and mop.upper95 < 0.15
    assert cells[("suspect", "box_track", "cup")].show
    # 판정 칸·종류 칸은 두 클래스를 합친 120건 (정밀도 43/120 → 표시)
    assert cells[("suspect", None, None)].show and cells[("suspect", None, None)].n == 120
    assert not any(c.verdict == "wrong" for c in cal.cells)  # 10건 < 유효 표본 30
    assert cal.prior == {"suspect": pytest.approx(43 / 120, abs=1e-6), "wrong": 0.9}
    assert cal.allows("suspect", "box_track", "mop", None) is False
    assert cal.allows("suspect", "box_track", "cup", None) is True
    assert cal.allows("wrong", "box_track", "mop", None) is True  # 칸 없음 = 기본 동작
    # 클래스 칸이 없는 클래스는 종류 칸으로 (표시)
    assert cal.allows("suspect", "box_track", "sponge", None) is True
    assert cal.weight("suspect", "box_track", "cup") > cal.weight("suspect", "box_track", "mop")


def test_insufficient_samples_keep_default() -> None:
    """모든 칸이 유효 표본 하한 미만이면 칸이 없고 무엇도 숨기지 않는다."""
    few = [row(i, V.SUSPECT, False) for i in range(10)]
    cal = fit_calibration(few, FIT, (AI_VERSION,), SINCE, UNTIL)
    assert cal.cells == ()
    assert cal.allows("suspect", "box_track", "mop", None)
    effect = calibration_effect(few, cal, (AI_VERSION,))
    assert effect.flags_before == effect.flags_after == 10


def test_uses_holdout_only_and_matching_versions() -> None:
    """보인 집단·다른 판정 버전·재검수·이중 작업 행은 칸을 바꾸지 않는다 (편향 방지)."""
    base = fit_calibration(rows(), FIT, (AI_VERSION,), SINCE, UNTIL)
    noise = [row(i, V.SUSPECT, True, arm=AiArm.SHOWN) for i in range(200)]
    noise += [
        row(i, V.SUSPECT, True, version="aireview-" + "f" * 12 + "+vlm-x") for i in range(200)
    ]
    noise += [row(i, V.SUSPECT, True, followup=True) for i in range(200)]
    noise += [row(i, V.SUSPECT, True, mode=ReviewMode.DOUBLE) for i in range(200)]
    other = fit_calibration(rows() + noise, FIT, (AI_VERSION,), SINCE, UNTIL)
    assert other.cells == base.cells
    assert other.source["holdout_rows"] == base.source["holdout_rows"]
    assert other.source["rows_sha256"] == base.source["rows_sha256"]


def test_min_confidence_hides_only_bad_low_band() -> None:
    """확신도 0.3 묶음(40건, 0 고침)은 확실히 쓸모없고 0.9 묶음(40건, 30 고침)은 쓸모 있다.

    → 문턱 0.9.
    """
    data = [row(i, V.SUSPECT, False, confidence=0.3, cls="cup") for i in range(40)]
    data += [row(i, V.SUSPECT, i < 30, confidence=0.9, cls="cup") for i in range(40)]
    cal = fit_calibration(data, FIT, (AI_VERSION,), SINCE, UNTIL)
    cell = cal.cell("suspect", "box_track", "cup")
    assert cell is not None and cell.show and cell.min_confidence == 0.9
    assert not cal.allows("suspect", "box_track", "cup", 0.3)
    assert cal.allows("suspect", "box_track", "cup", 0.95)
    assert cal.allows("suspect", "box_track", "cup", None)  # 확신도 없음은 문턱으로 숨기지 않는다


def test_effect_raises_precision_and_reports_recall() -> None:
    """보정 뒤 지적 정밀도가 오르고 재현율은 숨긴 칸의 고침만큼 내려간다 (같은 자료·2겹 교차)."""
    data = rows()
    cal = fit_calibration(data, FIT, (AI_VERSION,), SINCE, UNTIL)
    e = calibration_effect(data, cal, (AI_VERSION,))
    assert e.flags_before == 130 and e.flags_after == 70  # mop 의심 60건 숨김
    assert e.precision_before.value == pytest.approx(52 / 130)
    assert e.precision_after.value == pytest.approx(49 / 70)
    assert e.recall_before.value == pytest.approx(52 / 54)  # ok 놓침 2건은 지적이 아니다
    assert e.recall_after.value == pytest.approx(49 / 54)
    crossed = cross_checked_effect(data, FIT, (AI_VERSION,), SINCE, UNTIL)
    assert crossed.rows == e.rows and crossed.flags_before == e.flags_before
    assert crossed.flags_after >= e.flags_after  # 반만 보고 학습하면 덜 숨긴다 (보수적)


def test_fit_rejects_ok_and_unsure_verdicts() -> None:
    """ok·unsure는 보정할 수 없다 (정책 검증). 보정은 ok를 늘 허용한다."""
    with pytest.raises(ValidationError):
        AiCalibrationFit.model_validate({"verdicts": ["ok"]})
    with pytest.raises(ValidationError):
        AiCalibrationFit.model_validate({"verdicts": ["unsure"]})
    cal = fit_calibration(rows(), FIT, (AI_VERSION,), SINCE, UNTIL)
    assert cal.allows("ok", "box_track", "mop", None) and cal.allows(
        "unsure", "box_track", "mop", 0
    )


def test_artifact_roundtrip_is_deterministic_and_pinned(tmp_path: Path) -> None:
    """같은 입력이면 같은 바이트·calibration_id. sha256·내용 해시가 다르면 읽지 않는다."""
    a = to_bytes(fit_calibration(rows(), FIT, (AI_VERSION,), SINCE, UNTIL))
    b = to_bytes(fit_calibration(list(reversed(rows())), FIT, (AI_VERSION,), SINCE, UNTIL))
    assert a == b
    path = tmp_path / "cal.json"
    path.write_bytes(a)
    sha = hashlib.sha256(a).hexdigest()
    cal = load_calibration(path, sha)
    assert len(cal.calibration_id) == 12
    with pytest.raises(AiCalibrationError, match="sha256"):
        load_calibration(path, "0" * 64)
    tampered = a.replace(b'"show": false', b'"show": true')
    path.write_bytes(tampered)
    with pytest.raises(AiCalibrationError, match="내용 해시"):
        load_calibration(path, hashlib.sha256(tampered).hexdigest())


# ---------------------------------------------------------------- 검수 화면 표시


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(repo_root() / "config" / "ontology" / "v1")


def _box(lid: str, cls: str) -> LabelRecord:
    """바디캠 모델 박스 트랙 (0~200 ms)."""
    keyframes = [{"t_ms": 0, "x": 0, "y": 0, "w": 10, "h": 10}]
    return make_label(
        {"kind": "box_track", "entity_id": f"e-{lid}", "class_id": cls, "keyframes": keyframes},
        label_id=lid, stream_id="bodycam", t_start_ms=0, t_end_ms=200, confidence=0.9,
        provenance=Provenance(source=Source.MODEL, model_version="det-1"),
    )  # fmt: skip


def _task() -> ReviewTask:
    """CVAT 박스 작업."""
    return ReviewTask.model_validate(
        {"task_key": "cvat:9", "tool": ReviewTool.CVAT, "external_id": "9", "session_id": "s001",
         "stream_id": "bodycam", "stage": ReviewStage.LABELING, "assignee": "labeler01",
         "media_uri": "s3://dlp-labeling/m.mp4", "label_kinds": ("box_track",),
         "created_at": FIXED_TIME, "mode": ReviewMode.STANDARD, "render_hash": "a" * 64}
    )  # fmt: skip


def test_display_hides_calibrated_flags_but_not_escalation(ontology: Ontology) -> None:
    """보정이 mop 의심을 화면에서 숨기고(hidden) cup 의심은 보인다.

    표본 제외(증가 프로필)·ok는 그대로다.
    """
    cal = fit_calibration(rows(), FIT, (AI_VERSION,), SINCE, UNTIL)
    policy = AiReviewDisplayPolicy.model_validate({"mode": "advisory", "holdout_ratio": 0.0})
    labels = [_box("m", "mop"), _box("c", "cup"), _box("o", "mop")]
    uid = "air-" + "6" * 24
    display: dict[str, Judged] = {
        "m": (make_ai_verdict("m", "suspect", kind="box_track", problems=("false_positive",)), uid),
        "c": (make_ai_verdict("c", "suspect", kind="box_track", problems=("false_positive",)), uid),
        "o": (make_ai_verdict("o", "ok", kind="box_track"), uid),
    }
    view = AiView(display=display, escalate=display)
    plain = AiDisplayConfig(policy, (AI_VERSION,), AI_TARGET_RULE)
    calibrated = AiDisplayConfig(policy, (AI_VERSION,), AI_TARGET_RULE, calibration=cal)
    before = exposure_for(_task(), None, view, labels, plain, ontology, limit=10)
    after = exposure_for(_task(), None, view, labels, calibrated, ontology, limit=10)
    assert set(before.shown) == {"m", "c"}
    assert set(after.shown) == {"c"} and set(after.hidden) == {"m", "o"}
    assert after.arm is before.arm is AiArm.SHOWN
    # 증가 프로필은 보정을 보지 않는다: 숨긴 mop 의심도 표본 묶음에서 빠져 전수 검수된다 (I9)
    assert escalated_ids(view.escalate, policy) == {"m", "c"}
