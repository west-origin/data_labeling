"""판정단 불일치 테스트 (`ai_review.disagreement`, `dlp_review.ai_display`, ADR 0063).

허용 판정 버전(판정 모델마다 다르다) 둘이 같은 라벨을 ok와 ok 아님으로 가르면 그 라벨에만
AI_DISAGREEMENT 사유 구간이 붙고 표본 묶음에서 빠진다. 검수 화면에는 보인 판정의 메모(Label
Studio)·이슈(CVAT)에 "AI 판정단 의견이 갈림: <모델> 맞음 / <모델> 틀림"이 덧붙는다 (표시 프로필,
노출 기록·대조군·방식 제외·CVAT 이슈 상한 그대로). 꺼짐·한 버전만 판정·ok 아님끼리·ok끼리는
아무것도 붙지 않는다. AI끼리 ok로 같아도 표본·우선순위·사유가 AI 없을 때와 같다 (사람 검수를 줄이지
않는다, ADR 0040 I9). SQLite 메모리 DB, 판정은 계약 객체로 직접 넣는다 (실행기는
`tests/test_aireview_panel.py`가 본다).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from dlp_review.ai_display import (
    AiDisplayConfig,
    AiView,
    ai_flagged_spans,
    cvat_ai_issues,
    escalated_ids,
    exposure_for,
    judge_model,
    load_ai_view,
    ls_ai_results,
    split_opinions,
    split_text,
)
from dlp_review.labelstudio import AI_NOTE, AI_SUMMARY, LS_KINDS
from dlp_review.ops.policy import AiReviewDisplayPolicy, ReviewOpsPolicy, load_policy
from dlp_review.ops.runner import finish_assignment, plan_session
from dlp_schema import repo_root
from dlp_schema.aireview import AiArm, AiVerdict, AiVerdictValue
from dlp_schema.db.aireview_repository import insert_ai_run, insert_ai_units
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_review_task,
    insert_session,
    mark_review_task_collected,
    record_review,
    register_ontology,
)
from dlp_schema.db.tables import metadata
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import (
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewStage,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.testing import (
    AI_TARGET_RULE,
    AI_VERSION,
    FIXED_TIME,
    action_payload,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
    make_session,
    sqlite_db,
)

ROOT = repo_root()
# 판정단 두 모델의 판정 버전 (같은 digest, 모델 이름만 다르다)
VA = AI_VERSION
VB = "aireview-0123456789ab+vlm-vllm-sr/d3-turbo"
RUN_A = "air-run-" + "0" * 24
RUN_B = "air-run-" + "1" * 24
LABELS = [f"l{i:02d}" for i in range(12)]


def _cfg(
    *, enabled: bool = True, exclude: bool = True, versions: tuple[str, ...] = (VA, VB), **kw: Any
) -> AiDisplayConfig:
    """advisory, 대조군 없음, 허용 버전 둘, 불일치 켬(기본)."""
    data: dict[str, Any] = {
        "mode": "advisory", "holdout_ratio": 0.0,
        "disagreement": {"enabled": enabled, "sampling_exclude": exclude},
    } | kw  # fmt: skip
    return AiDisplayConfig(AiReviewDisplayPolicy.model_validate(data), versions, AI_TARGET_RULE)


def _action(lid: str, start: int, end: int) -> LabelRecord:
    """높은 신뢰도의 검수 전 모델 행동 (표본 묶음 후보)."""
    payload = action_payload(
        action_id=lid, t_approach_ms=start, t_contact_start_ms=start, t_contact_end_ms=end,
        t_end_ms=end,
    )  # fmt: skip
    return make_label(
        payload, label_id=lid, t_start_ms=start, t_end_ms=end, confidence=0.95,
        provenance=Provenance(source=Source.MODEL, model_version="actions-1"),
    )  # fmt: skip


def _engine(a: list[AiVerdict], b: list[AiVerdict] | None) -> sa.Engine:
    """행동 12개(l00~l11) + 모델 A 판정 단위, b가 있으면 모델 B 판정 단위."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, make_session())
        insert_labels(
            conn, [_action(lid, i * 1_000, i * 1_000 + 900) for i, lid in enumerate(LABELS)]
        )
        insert_ai_run(conn, make_ai_run(run_id=RUN_A))
        insert_ai_units(conn, [make_ai_unit("air-" + "a" * 24, a, run_id=RUN_A)])
        if b is not None:
            insert_ai_run(conn, make_ai_run(run_id=RUN_B, version=VB))
            insert_ai_units(conn, [make_ai_unit("air-" + "b" * 24, b, run_id=RUN_B, version=VB)])
    return engine


def _verdicts(values: dict[str, str]) -> list[AiVerdict]:
    """라벨 → 판정 값."""
    return [make_ai_verdict(lid, v) for lid, v in values.items()]


def _copy(engine: sa.Engine) -> sa.Engine:
    """같은 내용의 새 메모리 SQLite (같은 세션을 다른 설정으로 한 번 더 계획한다)."""
    copy = sqlite_db()
    with engine.connect() as src, copy.begin() as dst:
        for table in metadata.sorted_tables:
            rows = [dict(r) for r in src.execute(sa.select(table)).mappings()]
            if rows:
                dst.execute(table.insert(), rows)
    return copy


def _plan(engine: sa.Engine, ops: ReviewOpsPolicy, ai: AiDisplayConfig | None) -> ReviewAssignment:
    """세션 s001을 계획하고 시간 단위 표준 배정을 돌려준다 (커밋한다)."""
    with engine.begin() as conn:
        created = plan_session(
            conn, "s001", ["r1", "r2"], ops, load_ontology(ROOT / "config/ontology/v1"), seed=1,
            now=FIXED_TIME, ai=ai,
        )  # fmt: skip
    return next(a for a in created if a.mode is ReviewMode.STANDARD and a.stream_id is None)


def _view_parts(engine: sa.Engine, cfg: AiDisplayConfig) -> tuple[set[str], set[str], set[str]]:
    """(갈린 라벨, AI_DISAGREEMENT 구간 라벨, 표본 제외 라벨)."""
    with engine.connect() as conn:
        current = list(get_labels(conn, "s001"))
        view = load_ai_view(conn, "s001", current, None, cfg)
    spans = ai_flagged_spans(view.escalate, current, cfg.policy, view.disagree)
    split = {i for s in spans if s.reason is ReviewReason.AI_DISAGREEMENT for i in s.label_ids}
    return set(view.disagree), split, escalated_ids(view.escalate, cfg.policy, view.disagree)


def test_only_split_labels_get_disagreement() -> None:
    """l01(A ok / B wrong)·l02(A ok / B unsure)만 갈림이다. l03(둘 다 ok), l05(suspect / wrong, 둘
    다 ok 아님), l09(B만 판정)는 아니다. 갈린 l02는 판정 값(unsure)이 sampling_exclude 밖이어도
    표본에서 빠진다 (disagreement.sampling_exclude), 끄면 판정 값 규칙만 따른다."""
    a = _verdicts({"l01": "ok", "l02": "ok", "l03": "ok", "l05": "suspect"})
    b = _verdicts({"l01": "wrong", "l02": "unsure", "l03": "ok", "l05": "wrong", "l09": "unsure"})
    engine = _engine(a, b)
    disagree, spans, excluded = _view_parts(engine, _cfg(sampling_exclude=("wrong",)))
    assert disagree == spans == {"l01", "l02"}
    assert excluded == {"l01", "l02", "l05"}
    _, _, kept = _view_parts(engine, _cfg(exclude=False, sampling_exclude=("wrong",)))
    assert kept == {"l01", "l05"}  # 갈린 l02는 우선순위 사유만 남고 표본에는 남는다


def test_no_disagreement_when_off_single_version_or_one_model() -> None:
    """꺼짐, 허용 버전 하나, 한 모델만 판정한 세션은 갈림이 없다 (AI_DISAGREEMENT 구간 없음)."""
    a = _verdicts({"l01": "ok", "l02": "ok"})
    b = _verdicts({"l01": "wrong", "l02": "unsure"})
    both = _engine(a, b)
    assert _view_parts(both, _cfg(enabled=False))[:2] == (set(), set())
    assert _view_parts(both, _cfg(versions=(VA,)))[:2] == (set(), set())
    assert _view_parts(_engine(a, None), _cfg())[:2] == (set(), set())


def test_ok_agreement_never_reduces_sampling_or_priority() -> None:
    """두 모델이 모든 라벨을 ok로 같게 판정해도 표본·보류·사유·우선순위가 AI 없을 때와 같다 (I9)."""
    ops = load_policy(ROOT)
    all_ok = _verdicts(dict.fromkeys(LABELS, "ok"))
    engine = _engine(all_ok, all_ok)
    plain = _plan(_copy(engine), ops, None)
    agreed = _plan(engine, ops, _cfg())
    assert agreed.sample_label_ids == plain.sample_label_ids
    assert agreed.withheld_label_ids == plain.withheld_label_ids
    assert agreed.flagged == plain.flagged and agreed.priority == plain.priority


def test_split_label_goes_first_and_out_of_sample_and_protects_lot() -> None:
    """갈린 l04(A ok / B unsure)는 sampling_exclude가 비어도 표본·보류에 없고 AI_DISAGREEMENT 구간이
    붙어 우선순위가 오른다. 끄면 묶음에 남는다. 마무리에서 그 라벨을 사람이 고쳤으면 표본이 모두
    승인돼도 묶음이 불합격이다 (뺀 라벨의 결함, 결과 `_ai_excluded`가 사유를 되읽는다)."""
    ops = load_policy(ROOT)
    engine = _engine(_verdicts({"l04": "ok"}), _verdicts({"l04": "unsure"}))
    off = _plan(_copy(engine), ops, _cfg(enabled=False, sampling_exclude=()))
    on = _plan(engine, ops, _cfg(sampling_exclude=()))
    assert "l04" in {*off.sample_label_ids, *off.withheld_label_ids}
    assert "l04" not in {*on.sample_label_ids, *on.withheld_label_ids}
    assert (ReviewReason.AI_DISAGREEMENT, ("l04",)) in {(s.reason, s.label_ids) for s in on.flagged}
    assert ReviewReason.AI_DISAGREEMENT not in {s.reason for s in off.flagged}
    assert on.priority > off.priority
    with engine.begin() as conn:
        for lid in on.sample_label_ids:
            record_review(conn, lid, VerificationState.HUMAN_APPROVED, "labeler01", FIXED_TIME)
        fixed = _action("l04-fix", 4_000, 4_900).model_copy(
            update={
                "parent_label_id": "l04", "provenance": Provenance(source=Source.HUMAN),
                "confidence": None,
            }
        )  # fmt: skip
        insert_labels(conn, [fixed])
        task = {
            "task_key": "label_studio:7", "tool": ReviewTool.LABEL_STUDIO, "external_id": "7",
            "session_id": "s001", "stream_id": "bodycam", "stage": ReviewStage.LABELING,
            "assignee": "labeler01", "media_uri": "s3://dlp-labeling/m.mp4",
            "label_kinds": on.label_kinds, "created_at": FIXED_TIME, "mode": ReviewMode.STANDARD,
            "assignment_id": on.assignment_id, "render_hash": "a" * 64,
            "sent_label_ids": on.sample_label_ids,
        }  # fmt: skip
        insert_review_task(conn, ReviewTask.model_validate(task))
        mark_review_task_collected(conn, "label_studio:7", FIXED_TIME)
        result = finish_assignment(conn, on, FIXED_TIME, ops, ai=_cfg(sampling_exclude=()))
    assert result.sample_accepted is False and result.ai_excluded_defects


# ---------------------------------------------------------------- 검수 화면 표시 (ADR 0063)

H = "a" * 64  # make_ai_unit 기본 렌더 해시
QWEN = "Qwen/Qwen3-VL-8B-Instruct"  # AI_VERSION의 모델
D3 = "vllm-sr/d3-turbo"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


def _task(mode: ReviewMode = ReviewMode.STANDARD, **kw: Any) -> ReviewTask:
    """바디캠 Label Studio 작업 (렌더 해시 H)."""
    data: dict[str, Any] = {
        "task_key": "label_studio:7", "tool": ReviewTool.LABEL_STUDIO, "external_id": "7",
        "session_id": "s001", "stream_id": "bodycam", "stage": ReviewStage.LABELING,
        "assignee": "labeler01", "media_uri": "s3://dlp-labeling/m.mp4", "label_kinds": LS_KINDS,
        "created_at": FIXED_TIME, "mode": mode, "render_hash": H,
    }  # fmt: skip
    return ReviewTask.model_validate(data | kw)


def _display_view(engine: sa.Engine, cfg: AiDisplayConfig, rhash: str | None = H) -> AiView:
    """작업 매체 렌더 해시 rhash로 계산한 판정 (`tasks._AiTasks.view`와 같은 호출)."""
    with engine.connect() as conn:
        current = list(get_labels(conn, "s001"))
        hashes = {"bodycam": rhash} if rhash is not None else None
        return load_ai_view(conn, "s001", current, hashes, cfg, ontology=None)


def _split_engine() -> sa.Engine:
    """l01: A ok / B wrong (갈림), l02: A suspect / B wrong (ok 아님끼리), l03: 둘 다 ok."""
    a = _verdicts({"l01": "ok", "l02": "suspect", "l03": "ok"})
    b = _verdicts({"l01": "wrong", "l02": "wrong", "l03": "ok"})
    return _engine(a, b)


def test_judge_model_and_opinion_names() -> None:
    """모델 이름은 판정 버전의 `+vlm-<모델>`이다 (슬래시 포함). 같은 모델의 두 버전(digest만 다름)은
    digest 앞 6자로 가른다. 의견 순서는 허용 버전 순서다."""
    assert judge_model(VA) == QWEN and judge_model(VB) == D3
    assert judge_model("odd-version") == "odd-version"
    va2 = f"aireview-ffffff000000+vlm-{QWEN}"
    by_version = {"x": {
        VA: (make_ai_verdict("x", "ok"), "u1"), va2: (make_ai_verdict("x", "suspect"), "u2"),
    }}  # fmt: skip
    assert split_opinions(by_version, (va2, VA)) == {
        "x": ((f"{QWEN}(ffffff)", AiVerdictValue.SUSPECT), (f"{QWEN}(012345)", AiVerdictValue.OK))
    }
    assert split_text(((QWEN, AiVerdictValue.OK), (D3, AiVerdictValue.SUSPECT))) == (
        f"AI 판정단 의견이 갈림: {QWEN} 맞음 / {D3} 의심"
    )


def test_display_split_follows_render_hash_and_switch() -> None:
    """화면용 갈림은 표시 프로필(렌더 해시 일치)만 본다: 렌더가 다르거나 모르면 비고(증가 프로필의
    `disagree`는 그대로), 꺼지면 둘 다 빈다. ok 아님끼리(l02)·ok끼리(l03)는 갈림이 아니다."""
    engine = _split_engine()
    on = _display_view(engine, _cfg())
    assert on.display_split == {"l01": ((QWEN, AiVerdictValue.OK), (D3, AiVerdictValue.WRONG))}
    stale = _display_view(engine, _cfg(), "c" * 64)
    assert stale.display_split == {} and stale.disagree == {"l01"}
    assert _display_view(engine, _cfg(), None).display_split == {}
    off = _display_view(engine, _cfg(enabled=False))
    assert off.display_split == {} and off.disagree == frozenset()
    assert off.display == on.display  # 표시할 판정(가장 나쁜 것) 자체는 켜고 꺼도 같다


def test_label_studio_note_and_summary_show_split(ontology: Ontology) -> None:
    """보인 갈린 판정(l01)의 라벨 구간 메모 끝에 판정단 의견이 붙고 요약에 그 수가 든다. 갈리지 않은
    지적(l02)의 메모는 그대로다. 노출 기록(shown·hidden)은 켜고 꺼도 같다 (새 결과를 만들지
    않는다)."""
    engine = _split_engine()
    with engine.connect() as conn:
        sent = [x for x in get_labels(conn, "s001") if x.label_id in {"l01", "l02", "l03"}]
    cfg, off_cfg = _cfg(), _cfg(enabled=False)
    view, off_view = _display_view(engine, cfg), _display_view(engine, off_cfg)
    e = exposure_for(_task(), None, view, sent, cfg, ontology)
    off_e = exposure_for(_task(), None, off_view, sent, off_cfg, ontology)
    assert e.arm is AiArm.SHOWN and set(e.shown) == {"l01", "l02"}
    assert (e.shown, e.hidden, e.coverage) == (off_e.shown, off_e.hidden, off_e.coverage)
    results = ls_ai_results(e, view, sent, ontology, cfg.policy, {"apply": "r", "dismiss": "h"})
    off_results = ls_ai_results(off_e, off_view, sent, ontology, off_cfg.policy, None)
    assert len(results) == len(off_results)
    notes = {r["id"]: r["value"]["text"][0] for r in results if r["from_name"] == AI_NOTE}
    assert notes["l01"].startswith("[틀림 · 참고]")
    assert notes["l01"].endswith(f"h 기각 · AI 판정단 의견이 갈림: {QWEN} 맞음 / {D3} 틀림")
    assert "판정단" not in notes["l02"]
    [summary] = [r["value"]["text"][0] for r in results if r["from_name"] == AI_SUMMARY]
    assert summary.endswith("판정률 100% · 판정단 의견 갈림 1")
    texts = {AI_NOTE, AI_SUMMARY}
    off_notes = [r["value"]["text"][0] for r in off_results if r["from_name"] in texts]
    assert not any("판정단" in t for t in off_notes)


@pytest.mark.parametrize(
    ("mode", "holdout", "arm"),
    [
        (ReviewMode.STANDARD, 1.0, AiArm.HOLDOUT),
        (ReviewMode.BLIND, 0.0, AiArm.MODE_EXCLUDED),
        (ReviewMode.SEEDED_ERROR, 0.0, AiArm.MODE_EXCLUDED),
        (ReviewMode.QA, 0.0, AiArm.MODE_EXCLUDED),
    ],
)
def test_split_hidden_outside_shown_arm(
    ontology: Ontology, mode: ReviewMode, holdout: float, arm: AiArm
) -> None:
    """대조군·블라인드·오류 삽입·QA 작업에는 판정단 의견도 보이지 않는다 (요약 "없음" 하나뿐,
    CVAT 이슈 0). 노출 기록은 그 집단으로 남는다."""
    engine = _split_engine()
    cfg = _cfg(holdout_ratio=holdout)
    view = _display_view(engine, cfg)
    assert view.display_split  # 판정단은 갈렸다
    with engine.connect() as conn:
        sent = list(get_labels(conn, "s001"))
    e = exposure_for(_task(mode), None, view, sent, cfg, ontology)
    assert e.arm is arm and e.shown == {}
    results = ls_ai_results(e, view, sent, ontology, cfg.policy)
    assert [r["value"]["text"] for r in results] == [["AI 1차 검수: 없음"]]
    cvat = MagicMock()
    cvat.jobs.return_value = [{"id": 1, "start_frame": 0, "stop_frame": 9}]
    assert cvat_ai_issues(cvat, 9, e, view, sent, list(range(0, 10_000, 1_000)), ontology) == 0
    cvat.add_issue.assert_not_called()


def test_cvat_issue_carries_split_within_limit(ontology: Ontology) -> None:
    """CVAT는 보인 판정의 같은 이슈 끝에 판정단 의견을 붙인다. 이슈 수는 상한(limit) 그대로다."""
    engine = _split_engine()
    cfg = _cfg()
    view = _display_view(engine, cfg)
    with engine.connect() as conn:
        sent = list(get_labels(conn, "s001"))
    task = _task(task_key="cvat:9", tool=ReviewTool.CVAT, external_id="9")
    times = list(range(0, 12_000, 100))
    for limit, expected in ((10, 2), (1, 1)):
        e = exposure_for(task, None, view, sent, cfg, ontology, limit=limit)
        cvat = MagicMock()
        cvat.jobs.return_value = [{"id": 1, "start_frame": 0, "stop_frame": len(times) - 1}]
        assert cvat_ai_issues(cvat, 9, e, view, sent, times, ontology) == expected
        messages = {c.args[3].split("라벨 ")[1].split(" ")[0]: c.args[3]
                    for c in cvat.add_issue.call_args_list}  # fmt: skip
        assert len(messages) == expected
        split = f"AI 판정단 의견이 갈림: {QWEN} 맞음 / {D3} 틀림"
        for lid, message in messages.items():
            assert message.endswith(split) == (lid == "l01")
