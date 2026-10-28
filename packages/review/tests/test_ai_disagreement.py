"""판정단 불일치 테스트 (`ai_review.disagreement`, `dlp_review.ai_display`, ADR 0063).

허용 판정 버전(판정 모델마다 다르다) 둘이 같은 라벨을 ok와 ok 아님으로 가르면 그 라벨에만
AI_DISAGREEMENT 사유 구간이 붙고 표본 묶음에서 빠진다. 꺼짐·한 버전만 판정·ok 아님끼리·ok끼리는
아무것도 붙지 않는다. AI끼리 ok로 같아도 표본·우선순위·사유가 AI 없을 때와 같다 (사람 검수를 줄이지
않는다, ADR 0040 I9). SQLite 메모리 DB, 판정은 계약 객체로 직접 넣는다 (실행기는
`tests/test_aireview_panel.py`가 본다).
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

from dlp_review.ai_display import AiDisplayConfig, ai_flagged_spans, escalated_ids, load_ai_view
from dlp_review.ops.policy import AiReviewDisplayPolicy, ReviewOpsPolicy, load_policy
from dlp_review.ops.runner import finish_assignment, plan_session
from dlp_schema import repo_root
from dlp_schema.aireview import AiVerdict
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
from dlp_schema.ontology import load_ontology
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
