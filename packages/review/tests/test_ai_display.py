"""AI 1차 검수 판정의 검수 화면 표시·노출 집단·우선순위·표본 제외·마무리 보호 (WP17.6, ADR 0040).

서비스 없이 돈다 (순수 함수, 시간대 보존 SQLite `dlp_schema.testing.sqlite_db`). 실제 Label Studio·
CVAT 화면은 `test_ls_ai_feasibility.py`·`test_ai_display_flow.py`(services)가 본다.

정답 근거
- 판정은 `dlp_schema.testing.make_ai_unit`(바디캠 렌더 해시 a…, 버전 `AI_VERSION`)으로 만든다.
- 대조군은 sha256(`<루트 배정 키>|<seed>`) 앞 8바이트 / 2^64 < 비율 (같은 키는 늘 같은 결과).
- 표본 묶음: 신뢰도 0.85 이상 미검수 모델 라벨 N개 → 표본 min(N, max(5, ⌈0.1·N⌉)) (review.yaml).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_review.ai_display import (
    AiDisplayConfig,
    AiView,
    Judged,
    ai_flagged_spans,
    arm_key,
    cvat_ai_issues,
    escalated_ids,
    exposure_for,
    flag_range,
    is_holdout,
    issue_frame,
    issue_position,
    load_ai_view,
    ls_ai_results,
    master_clock,
    note_text,
    required_gate,
    root_assignment_id,
    summary_text,
)
from dlp_review.labelstudio import (
    AI_DECISION,
    AI_FLAGS,
    AI_LABELS,
    AI_NOTE,
    AI_SUMMARY,
    FROM_NAME,
    LS_KINDS,
    from_ls_results,
    label_config,
    ls_ai_decisions,
    to_ls_results,
)
from dlp_review.ops.policy import AiReviewDisplayPolicy, ReviewOpsPolicy, load_policy
from dlp_review.ops.runner import finish_assignment, plan_session
from dlp_review.tasks import ReviewSetup
from dlp_review.ui_theme import UiTheme, load_ui_theme
from dlp_schema import repo_root
from dlp_schema.aireview import AiArm, AiDecision, AiRunEvent, AiRunEventKind, AiSuggestion
from dlp_schema.db.aireview_repository import (
    AiRunState,
    insert_ai_run,
    insert_ai_units,
)
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    insert_review_task,
    insert_session,
    list_assignments,
    mark_review_task_collected,
    record_review,
    register_ontology,
)
from dlp_schema.labels import (
    ActionPayload,
    GapPayload,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewStage,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.session import PrivacyState
from dlp_schema.testing import (
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
H = "a" * 64  # make_ai_unit 기본 렌더 해시
OTHER_H = "c" * 64


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config" / "ontology" / "v1")


@pytest.fixture(scope="module")
def theme() -> UiTheme:
    """저장소 검수 화면 테마."""
    return load_ui_theme(ROOT)


@pytest.fixture(scope="module")
def ops() -> ReviewOpsPolicy:
    """저장소 검수 정책 (ai_review: advisory)."""
    return load_policy(ROOT)


def _cfg(**policy: Any) -> AiDisplayConfig:
    """advisory, 대조군 없음(시험을 결정적으로), 허용 버전 `AI_VERSION`."""
    data: dict[str, Any] = {"mode": "advisory", "holdout_ratio": 0.0} | policy
    return AiDisplayConfig(AiReviewDisplayPolicy.model_validate(data), (AI_VERSION,))


def _action(
    lid: str, start: int = 0, end: int = 1_000, conf: float = 0.95, **kw: Any
) -> LabelRecord:
    """검수 전 모델 출처 오른손 행동 (start~end ms, 접촉 = 구간 전체)."""
    payload = action_payload(
        action_id=lid, t_approach_ms=start, t_contact_start_ms=start, t_contact_end_ms=end,
        t_end_ms=end,
    )  # fmt: skip
    return make_label(
        payload, label_id=lid, t_start_ms=start, t_end_ms=end, confidence=conf,
        provenance=Provenance(source=Source.MODEL, model_version="actions-1"), **kw,
    )  # fmt: skip


def _task(mode: ReviewMode = ReviewMode.STANDARD, aid: str | None = None, **kw: Any) -> ReviewTask:
    """Label Studio 작업 (바디캠, 렌더 해시 H)."""
    data: dict[str, Any] = {
        "task_key": "label_studio:7", "tool": ReviewTool.LABEL_STUDIO, "external_id": "7",
        "session_id": "s001", "stream_id": "bodycam", "stage": ReviewStage.LABELING,
        "assignee": "labeler01", "media_uri": "s3://dlp-labeling/m.mp4", "label_kinds": LS_KINDS,
        "created_at": FIXED_TIME, "mode": mode, "assignment_id": aid, "render_hash": H,
    }  # fmt: skip
    return ReviewTask.model_validate(data | kw)


def _assignment(aid: str, mode: ReviewMode = ReviewMode.STANDARD, **kw: Any) -> ReviewAssignment:
    """시간 단위 배정."""
    return ReviewAssignment.model_validate(
        {"assignment_id": aid, "session_id": "s001", "label_kinds": LS_KINDS, "mode": mode,
         "priority": 1.0, "assignee": "labeler01", "created_at": FIXED_TIME} | kw
    )  # fmt: skip


def _view(display: dict[str, Judged], escalate: dict[str, Judged] | None = None) -> AiView:
    """판정 묶음 (증가 프로필 기본 = 표시 프로필)."""
    return AiView(display=display, escalate=display if escalate is None else escalate)


# ---------------------------------------------------------------- Label Studio 수집 경계·설정


def test_collection_skips_ai_controls_and_reads_decisions(ontology: Ontology) -> None:
    """F-AIR-02: from_name이 있고 labels가 아닌 구간(AI 판정 구간)은 라벨이 되지 않는다. from_name이
    없는 이전 형식 결과는 라벨로 읽는다. 결정은 별칭·화면 값 모두 읽고 모르는 값은 버린다."""
    a, b = _action("a1"), _action("b1", 1_000, 2_000)
    [ra, rb] = to_ls_results([a, b])
    rb.pop("from_name")  # 이전 형식
    flag = {
        "id": "ai-1",
        "from_name": AI_FLAGS,
        "to_name": "ts",
        "type": "timeserieslabels",
        "readonly": True,
        "value": {"start": 0, "end": 500, "timeserieslabels": ["AI:틀림"]},
    }
    extra: list[dict[str, Any]] = [
        flag,
        {"id": "a1", "from_name": AI_NOTE, "type": "textarea", "value": {"text": ["메모"]}},
        {"id": "ai-summary", "from_name": AI_SUMMARY, "type": "textarea",
         "value": {"text": ["요약"]}},
        {"id": "a1", "from_name": AI_DECISION, "type": "choices", "value": {"choices": ["apply"]}},
        {"id": "b1", "from_name": AI_DECISION, "type": "choices",
         "value": {"choices": ["AI 지적 기각"]}},
        {"id": "x9", "from_name": AI_DECISION, "type": "choices", "value": {"choices": ["?"]}},
    ]  # fmt: skip
    items = from_ls_results([ra, rb, *extra], {"a1", "b1"})
    assert [(i.origin_label_id, i.payload) for i in items] == [
        ("a1", a.payload), ("b1", b.payload),
    ]  # fmt: skip
    assert ls_ai_decisions([ra, rb, *extra]) == {"a1": AiDecision.APPLY, "b1": AiDecision.DISMISS}


def test_label_config_always_has_ai_controls(ontology: Ontology, theme: UiTheme) -> None:
    """AI 컨트롤은 테마·모드와 무관하게 늘 있고(설계 C7), 새 데이터 변수가 없다(C6). 테마는 판정
    색과 결정 단축키만 더한다. 기존 라벨 컨트롤은 그대로다."""
    for config in (label_config(ontology), label_config(ontology, theme)):
        root = ET.fromstring(config)
        flags = root.find(f".//TimeSeriesLabels[@name='{AI_FLAGS}']")
        assert flags is not None
        assert [x.get("value") for x in flags.iter("Label")] == list(AI_LABELS.values())
        assert root.find(f".//TextArea[@name='{AI_SUMMARY}']") is not None
        note = root.find(f".//TextArea[@name='{AI_NOTE}']")
        decision = root.find(f".//Choices[@name='{AI_DECISION}']")
        assert note is not None and note.get("perRegion") == "true"
        assert decision is not None and decision.get("perRegion") == "true"
        assert [c.get("alias") for c in decision.iter("Choice")] == ["apply", "dismiss"]
        assert root.find(f".//TimeSeriesLabels[@name='{FROM_NAME}']") is not None
        variables = {w for w in config.split('"') if w.startswith("$")}
        assert variables == {"$video", "$timeseries"}
    themed = ET.fromstring(label_config(ontology, theme))
    flags = themed.find(f".//TimeSeriesLabels[@name='{AI_FLAGS}']")
    assert flags is not None
    colors = theme.ai.colors.model_dump()
    assert [x.get("background") for x in flags.iter("Label")] == [colors[k] for k in AI_LABELS]
    keys = [c.get("hotkey") for c in themed.iter("Choice") if c.get("alias")]
    assert keys == [theme.ai.hotkeys.apply, theme.ai.hotkeys.dismiss]
    # 모드를 바꿔도 같은 설정 (설정 함수에 모드가 들어가지 않는다)
    base = ReviewSetup(raw=MagicMock(), labeling=MagicMock(), labeling_reader=MagicMock(),
                       ontology=None, ui=theme)  # fmt: skip
    on = ReviewSetup(raw=MagicMock(), labeling=MagicMock(), labeling_reader=MagicMock(),
                     ontology=None, ui=theme, ai=_cfg())  # fmt: skip
    assert base.ls_label_config(ontology) == on.ls_label_config(ontology)


def test_ai_hotkeys_must_not_clash(theme: UiTheme) -> None:
    """결정 단축키는 동사 단축키·그 shift 조합과 겹치거나 조합키·같은 키면 로드 오류."""
    data = theme.model_dump()
    for apply, dismiss in (("1", "w"), ("q", "shift+q"), ("q", "q"), ("shift+1", "w")):
        bad = data | {"ai": {**data["ai"], "hotkeys": {"apply": apply, "dismiss": dismiss}}}
        with pytest.raises(ValidationError, match=r"ai\.hotkeys"):
            UiTheme.model_validate(bad)


# ---------------------------------------------------------------- 노출 집단


def test_arm_keys_follow_root_and_pair() -> None:
    """재검수 접미사를 뗀 루트, 이중은 짝 표준 배정의 루트, 직접 작업은 작업 키."""
    assert root_assignment_id("s:all:temporal:standard:g1:resample:ai-flagged") == (
        "s:all:temporal:standard:g1"
    )
    std = "s001:all:temporal:standard:g1"
    task = _task(aid=f"{std}:resample")
    assert arm_key(task, _assignment(f"{std}:resample")) == std
    double = _assignment("s001:all:temporal:double:g1", ReviewMode.DOUBLE, pair_id=std)
    assert arm_key(_task(ReviewMode.DOUBLE, double.assignment_id), double) == std
    assert arm_key(_task(), None) == "label_studio:7"


def test_holdout_is_deterministic_and_matches_ratio() -> None:
    """같은 키는 늘 같은 결과, 1만 개 키에서 비율 0.1 ± 0.01, 비율 0이면 없고 1이면 전부."""
    policy = AiReviewDisplayPolicy(mode="advisory", holdout_ratio=0.1, holdout_seed=3)
    keys = [f"s{i}:all:temporal:standard:g" for i in range(10_000)]
    share = sum(is_holdout(k, policy) for k in keys) / len(keys)
    assert abs(share - 0.1) < 0.01
    k = keys[0]
    digest = hashlib.sha256(f"{k}|3".encode()).digest()
    assert is_holdout(k, policy) == (int.from_bytes(digest[:8], "big") / 2**64 < 0.1)
    assert not any(is_holdout(k, policy.model_copy(update={"holdout_ratio": 0.0})) for k in keys)
    assert all(is_holdout(k, policy.model_copy(update={"holdout_ratio": 1.0})) for k in keys[:50])


def test_exposure_arms_and_shown_hidden_split() -> None:
    """블라인드·오류 삽입·QA·show_modes 밖은 mode_excluded, 판정 없음은 no_verdicts, 대조군은 숨김.
    shown이면 틀림·의심·판단 불가 중 최소 확신도 이상만 보이고 맞음은 hidden이다."""
    labels = [_action(f"l{i}", i * 1_000, i * 1_000 + 900) for i in range(5)]
    display: dict[str, Judged] = {
        "l0": (make_ai_verdict("l0", "wrong", confidence=0.9), "air-" + "1" * 24),
        "l1": (make_ai_verdict("l1", "suspect", confidence=0.2), "air-" + "1" * 24),
        "l2": (make_ai_verdict("l2", "unsure"), "air-" + "1" * 24),
        "l3": (make_ai_verdict("l3", "ok"), "air-" + "1" * 24),
    }
    view = _view(display)
    cfg = _cfg(show_min_confidence={"suspect": 0.5})
    e = exposure_for(_task(), None, view, labels, cfg)
    assert e.arm is AiArm.SHOWN and e.arm_key == "label_studio:7"
    assert list(e.shown) == ["l0", "l2"] and set(e.hidden) == {"l1", "l3"}
    assert e.coverage == pytest.approx(4 / 5) and e.versions == (AI_VERSION,)
    for mode in (ReviewMode.BLIND, ReviewMode.SEEDED_ERROR, ReviewMode.QA):
        ex = exposure_for(_task(mode), None, view, labels, cfg)
        assert ex.arm is AiArm.MODE_EXCLUDED and not ex.shown
    only_std = _cfg(show_modes=["standard"])
    assert exposure_for(_task(ReviewMode.DOUBLE), None, view, labels, only_std).arm is (
        AiArm.MODE_EXCLUDED
    )
    assert exposure_for(_task(), None, _view({}), labels, cfg).arm is AiArm.NO_VERDICTS
    held = exposure_for(_task(), None, view, labels, _cfg(holdout_ratio=1.0))
    assert held.arm is AiArm.HOLDOUT and not held.shown and set(held.hidden) == set(display)


def test_exposure_caps_per_ten_minutes_and_task_limit() -> None:
    """10분당 상한은 나쁜 판정부터 채우고, 작업 상한(CVAT 이슈)은 전체를 자른다."""
    labels = [_action(f"l{i}", i * 1_000, i * 1_000 + 500) for i in range(4)]
    labels.append(_action("late", 700_000, 700_500))  # 다른 10분 묶음
    display: dict[str, Judged] = {
        x.label_id: (make_ai_verdict(x.label_id, v), "air-" + "2" * 24)
        for x, v in zip(labels, ["unsure", "wrong", "suspect", "wrong", "unsure"], strict=True)
    }
    e = exposure_for(_task(), None, _view(display), labels, _cfg(max_flags_per_10min=2))
    assert list(e.shown) == ["l1", "l3", "late"]
    capped = exposure_for(_task(), None, _view(display), labels, _cfg(), limit=2)
    assert list(capped.shown) == ["l1", "l3"]
    assert set(capped.hidden) == {"l0", "l2", "late"}


def _sqlite(*, privacy: PrivacyState = PrivacyState.APPROVED) -> sa.Engine:
    """세션 s001(바디캠 + IMU)과 온톨로지 v1을 넣은 SQLite."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
        insert_session(conn, make_session(privacy_state=privacy))
    return engine


def test_display_profile_is_fail_closed_and_escalate_ignores_render() -> None:
    """표시 프로필은 단위 렌더 해시가 지금과 같아야 하고 렌더를 모르면 비며, 증가 프로필은 렌더를
    보지 않는다. 다른 버전의 단위는 둘 다 쓰지 않는다."""
    engine = _sqlite()
    labels = [_action("l1"), _action("l2", 1_000, 2_000)]
    with engine.begin() as conn:
        insert_labels(conn, labels)
        insert_ai_run(conn, make_ai_run())
        insert_ai_units(conn, [make_ai_unit("air-" + "3" * 24, [make_ai_verdict("l1", "wrong")])])
        other = "aireview-ffffffffffff+vlm-x"
        insert_ai_run(conn, make_ai_run(run_id="air-run-" + "1" * 24, version=other))
        insert_ai_units(conn, [make_ai_unit(
            "air-" + "4" * 24, [make_ai_verdict("l2", "wrong")], run_id="air-run-" + "1" * 24,
            version=other,
        )])  # fmt: skip
        current = list(get_labels(conn, "s001"))
        same = load_ai_view(conn, "s001", current, {"bodycam": H}, _cfg())
        changed = load_ai_view(conn, "s001", current, {"bodycam": OTHER_H}, _cfg())
        unknown = load_ai_view(conn, "s001", current, None, _cfg())
    assert set(same.display) == {"l1"} and set(same.escalate) == {"l1"}
    assert changed.display == {} and set(changed.escalate) == {"l1"}
    assert unknown.display == {} and set(unknown.escalate) == {"l1"}


# ---------------------------------------------------------------- 화면 글·예측·이슈


def test_ls_results_only_add_ai_and_keep_prelabels(ontology: Ontology, theme: UiTheme) -> None:
    """예측 = 프리라벨 결과 그대로 + 요약 하나 + 보인 판정마다 readonly AI 구간과 라벨 구간 메모.
    판정이 없으면 요약 "AI 1차 검수: 없음"만 더한다 (데이터·기존 예측 불변)."""
    short, long_ = _action("s1", 0, 4_000), _action("g1", 10_000, 40_000)
    swap = short.payload.model_copy(update={"verb": "push"})
    display: dict[str, Judged] = {
        "s1": (make_ai_verdict("s1", "wrong", evidence_t_ms=1_500, suggestion=AiSuggestion(
            op="set", fields=("verb",), payload=swap)), "air-" + "5" * 24),
        "g1": (make_ai_verdict("g1", "suspect", evidence_t_ms=25_000), "air-" + "5" * 24),
    }  # fmt: skip
    labels = [short, long_]
    cfg = _cfg()
    e = exposure_for(_task(), None, _view(display), labels, cfg)
    hotkeys = theme.ai.hotkeys.model_dump()
    ai = ls_ai_results(e, _view(display), labels, ontology, cfg.policy, hotkeys)
    prelabels = to_ls_results(labels)
    assert [r for r in prelabels + ai if r["from_name"] == FROM_NAME] == prelabels
    [summary] = [r for r in ai if r["from_name"] == AI_SUMMARY]
    assert summary["value"]["text"] == [
        "AI 1차 검수(참고용·사람 검수 아님): 틀림 1 · 의심 1 · 판단 불가 0 · 맞음 0 · 판정률 100%"
    ]
    flags = {r["value"]["timeserieslabels"][0]: r for r in ai if r["from_name"] == AI_FLAGS}
    assert all(r["readonly"] for r in flags.values())
    assert (flags["AI:틀림"]["value"]["start"], flags["AI:틀림"]["value"]["end"]) == (0, 4_000)
    # 10초보다 긴 라벨은 근거 시각 ±1초만
    assert (flags["AI:의심"]["value"]["start"], flags["AI:의심"]["value"]["end"]) == (
        24_000,
        26_000,
    )
    notes = {r["id"]: r["value"]["text"][0] for r in ai if r["from_name"] == AI_NOTE}
    assert (
        notes["s1"]
        == "[틀림 · 참고] 동사 잡다→밀다 제안 · 동사가 다름 · 근거 00:01.5 · q 적용 / w 기각"
    )
    assert notes["g1"].endswith("근거 00:25.0 · w 기각")  # 제안이 없으면 적용 안내도 없다
    none = exposure_for(_task(), None, _view({}), labels, cfg)
    assert ls_ai_results(none, _view({}), labels, ontology, cfg.policy) == [
        {"id": "ai-summary", "from_name": AI_SUMMARY, "to_name": "ts", "type": "textarea",
         "value": {"text": ["AI 1차 검수: 없음"]}},
    ]  # fmt: skip
    assert summary_text(exposure_for(_task(), None, _view(display), labels, _cfg(
        holdout_ratio=1.0)), _view(display)) == "AI 1차 검수: 없음"  # fmt: skip


def test_note_texts_for_kind_change_delete_and_clock(ontology: Ontology) -> None:
    """종류 변경·삭제 제안 문장과 시각 표기 (mm:ss.s, 한 시간 넘으면 h:mm:ss.s)."""
    action = _action("a1")
    assert isinstance(action.payload, ActionPayload)
    gap = GapPayload(hand=action.payload.hand, gap_type="idle")
    to_gap = make_ai_verdict("a1", "wrong", problems=("should_be_gap",), suggestion=AiSuggestion(
        op="change_kind", fields=("kind", "gap_type"), payload=gap))  # fmt: skip
    assert note_text(action, to_gap, ontology, None).startswith(
        "[틀림 · 참고] 행동→사이 구간(대기) 제안"
    )
    box = make_label(
        {"kind": "box_track", "entity_id": "cup_01", "class_id": "cup",
         "keyframes": [{"t_ms": 0, "x": 0, "y": 0, "w": 10, "h": 10}]},
        label_id="b1", stream_id="bodycam", confidence=0.9,
        provenance=Provenance(source=Source.MODEL, model_version="m"),
    )  # fmt: skip
    delete = make_ai_verdict("b1", "wrong", kind="box_track", problems=("false_positive",),
                             suggestion=AiSuggestion(op="delete", fields=("delete",)))  # fmt: skip
    assert "삭제 제안" in note_text(box, delete, ontology, {"apply": "q", "dismiss": "w"})
    assert master_clock(724_000) == "12:04.0" and master_clock(3_725_050) == "1:02:05.0"


def _box(lid: str, keyframes: list[dict[str, Any]], end: int) -> LabelRecord:
    """바디캠 모델 박스 트랙."""
    return make_label(
        {"kind": "box_track", "entity_id": f"e-{lid}", "class_id": "cup", "keyframes": keyframes},
        label_id=lid, stream_id="bodycam", t_start_ms=0, t_end_ms=end, confidence=0.9,
        provenance=Provenance(source=Source.MODEL, model_version="det-1"),
    )  # fmt: skip


def test_cvat_issues_frame_position_order(ontology: Ontology) -> None:
    """이슈 프레임은 근거 시각(스트림 PTS)에 떠 있는 프레임, 그 프레임의 job, 위치는 그 시각의 박스
    (키프레임 선형 보간), 순서는 틀림 → 의심 → 판단 불가. 화면 밖이면 왼쪽 위 16 화소."""
    times = [0, 33, 80, 121, 160, 207, 240, 290, 320, 371]  # VFR 프레임 PTS (번호 = 인덱스)
    assert (
        issue_frame(times, 95) == 2 and issue_frame(times, -5) == 0 and issue_frame(times, 999) == 9
    )
    a = _box("a", [{"t_ms": 0, "x": 0, "y": 0, "w": 10, "h": 10},
                   {"t_ms": 200, "x": 100, "y": 50, "w": 30, "h": 10}], 200)  # fmt: skip
    gone = {"t_ms": 100, "x": 5, "y": 5, "w": 5, "h": 5, "outside": True}
    b = _box("b", [{"t_ms": 0, "x": 5, "y": 5, "w": 5, "h": 5}, gone], 100)
    assert issue_position(a, 100) == [50.0, 25.0, 70.0, 35.0]
    assert issue_position(b, 150) == [0.0, 0.0, 16.0, 16.0]
    display: dict[str, Judged] = {
        "b": (
            make_ai_verdict("b", "unsure", kind="box_track", evidence_t_ms=150), "air-" + "6" * 24
        ),
        "a": (make_ai_verdict("a", "wrong", kind="box_track", problems=("wrong_class",),
                              evidence_t_ms=100), "air-" + "6" * 24),
    }  # fmt: skip
    task = _task(task_key="cvat:9", tool=ReviewTool.CVAT, external_id="9",
                 label_kinds=("box_track", "keypoint_track"))  # fmt: skip
    e = exposure_for(task, None, _view(display), [a, b], _cfg(), limit=10)
    cvat = MagicMock()
    cvat.jobs.return_value = [{"id": 1, "start_frame": 0, "stop_frame": 4},
                              {"id": 2, "start_frame": 5, "stop_frame": 9}]  # fmt: skip
    assert cvat_ai_issues(cvat, 9, e, _view(display), [a, b], times, ontology) == 2
    calls = [c.args for c in cvat.add_issue.call_args_list]
    assert [(job, frame) for job, frame, _, _ in calls] == [(1, 2), (1, 3)]
    assert calls[0][2] == [50.0, 25.0, 70.0, 35.0]
    assert calls[0][3].startswith("[AI 틀림·참고] 클래스가 다름") and "0.1s" in calls[0][3]
    assert calls[1][3].startswith("[AI 판단 불가·참고]")


# ---------------------------------------------------------------- 우선순위·표본·마무리·게이트


def test_flagged_spans_and_escalated_ids() -> None:
    """증가 프로필 판정마다 사유 구간 (긴 라벨은 span_cap_ms로 자름). ok는 빼지 않고 사유도 없다."""
    labels = [_action("w", 0, 2_000), _action("long", 0, 60_000), _action("ok", 0, 1_000)]
    escalate: dict[str, Judged] = {
        "w": (make_ai_verdict("w", "wrong"), "air-" + "7" * 24),
        "long": (make_ai_verdict("long", "suspect", evidence_t_ms=30_000), "air-" + "7" * 24),
        "ok": (make_ai_verdict("ok", "ok"), "air-" + "7" * 24),
    }
    policy = _cfg().policy
    spans = {s.label_ids[0]: s for s in ai_flagged_spans(escalate, labels, policy)}
    assert set(spans) == {"w", "long"}
    assert spans["w"].reason is ReviewReason.AI_WRONG
    assert (spans["long"].t_start_ms, spans["long"].t_end_ms) == (25_000, 35_000)
    assert escalated_ids(escalate, policy) == {"w", "long"}
    only_wrong = policy.model_copy(update={"sampling_exclude": ("wrong",)})
    assert escalated_ids(escalate, only_wrong) == {"w"}
    assert flag_range(labels[0], escalate["w"][0], 10_000, 2_000) == (0, 2_000)


def _plan(engine: sa.Engine, ops: ReviewOpsPolicy, ai: AiDisplayConfig | None) -> ReviewAssignment:
    """세션 s001을 계획하고 시간 단위 표준 배정을 돌려준다 (계획은 커밋한다)."""
    with engine.begin() as conn:
        created = plan_session(
            conn, "s001", ["r1", "r2"], ops, load_ontology(ROOT / "config/ontology/v1"), seed=1,
            now=FIXED_TIME, ai=ai,
        )  # fmt: skip
    return next(a for a in created if a.mode is ReviewMode.STANDARD and a.stream_id is None)


@pytest.fixture
def twelve() -> Iterator[sa.Engine]:
    """높은 신뢰도 모델 행동 12개(l00~l11, 표본 묶음 하나) + 낡은 렌더 해시로 만든 AI 판정:
    l03 틀림, l07 의심, l01 맞음. 세션은 AUTO_BLURRED (다른 스트림 블러 변경을 흉내, 설계 C10)."""
    engine = _sqlite(privacy=PrivacyState.AUTO_BLURRED)
    with engine.begin() as conn:
        insert_labels(conn, [_action(f"l{i:02d}", i * 1_000, i * 1_000 + 900) for i in range(12)])
        insert_ai_run(conn, make_ai_run())
        verdicts = [make_ai_verdict("l01", "ok"), make_ai_verdict("l03", "wrong"),
                    make_ai_verdict("l07", "suspect")]  # fmt: skip
        insert_ai_units(conn, [make_ai_unit("air-" + "8" * 24, verdicts, render_hash=OTHER_H)])
    yield engine
    engine.dispose()


def test_plan_adds_ai_reasons_and_never_samples_flagged_labels(
    twelve: sa.Engine, ops: ReviewOpsPolicy
) -> None:
    """증가 프로필(렌더 무관): 지적된 l03·l07은 표본·보류 어디에도 없어 반드시 작업에 들어가고,
    AI 사유 구간이 붙어 우선순위가 오른다. 맞음(l01)은 아무것도 줄이지 않는다 (묶음에 그대로)."""
    with twelve.connect() as conn:
        assert get_session(conn, "s001").privacy_state is PrivacyState.AUTO_BLURRED
    plain = _plan(sqlite_copy(twelve), ops, None)
    with_ai = _plan(twelve, ops, _cfg())
    lot = {*with_ai.sample_label_ids, *with_ai.withheld_label_ids}
    assert {"l03", "l07"} & lot == set() and "l01" in lot and len(lot) == 10
    assert len(with_ai.sample_label_ids) == 5 and len(plain.sample_label_ids) == 5
    reasons = {(s.reason, s.label_ids) for s in with_ai.flagged}
    assert (ReviewReason.AI_WRONG, ("l03",)) in reasons
    assert (ReviewReason.AI_SUSPECT, ("l07",)) in reasons
    assert with_ai.priority > plain.priority


def sqlite_copy(engine: sa.Engine) -> sa.Engine:
    """같은 내용의 새 메모리 SQLite (같은 세션을 AI 없이 한 번 더 계획해 보기 위해)."""
    from dlp_schema.db.tables import metadata

    copy = sqlite_db()
    with engine.connect() as src, copy.begin() as dst:
        for table in metadata.sorted_tables:
            rows = [dict(r) for r in src.execute(sa.select(table)).mappings()]
            if rows:
                dst.execute(table.insert(), rows)
    return copy


def _collected_task(conn: sa.Connection, a: ReviewAssignment, sent: tuple[str, ...]) -> None:
    """배정의 Label Studio 작업 하나를 수집된 것으로 둔다."""
    insert_review_task(conn, _task(aid=a.assignment_id, sent_label_ids=sent))
    mark_review_task_collected(conn, "label_studio:7", FIXED_TIME)


def test_finish_protects_late_ai_flags_from_sample_verification(
    twelve: sa.Engine, ops: ReviewOpsPolicy
) -> None:
    """표본 합격 뒤: 늦게 도착한 지적(계획 뒤 단위)이 있는 보류 라벨은 sample_verified가 아니라
    `<배정>:ai-flagged` 재검수 배정으로 간다. 세션이 AUTO_BLURRED이고 단위 렌더 해시가 낡아도
    동작한다 (증가 프로필, I9·C10). 맞음 판정의 보류 라벨은 평소처럼 표본 검증된다."""
    a = _plan(twelve, ops, _cfg())
    late = a.withheld_label_ids[0]
    ok = a.withheld_label_ids[1]
    with twelve.begin() as conn:
        for lid in a.sample_label_ids:
            record_review(conn, lid, VerificationState.HUMAN_APPROVED, "labeler01", FIXED_TIME)
        insert_ai_units(conn, [make_ai_unit(
            "air-" + "9" * 24, [make_ai_verdict(late, "wrong"), make_ai_verdict(ok, "ok")],
            render_hash=OTHER_H, created_at=FIXED_TIME + timedelta(hours=1),
        )])  # fmt: skip
        _collected_task(conn, a, a.sample_label_ids)
        result = finish_assignment(conn, a, FIXED_TIME, ops, ai=_cfg())
        follow = {x.assignment_id: x for x in list_assignments(conn, session_id="s001")}
        states = {x.label_id: x.verification.state for x in get_labels(conn, "s001")}
    assert result.sample_accepted is True and result.ai_flagged == 1
    assert result.ai_flagged_assignment == f"{a.assignment_id}:ai-flagged"
    again = follow[f"{a.assignment_id}:ai-flagged"]
    assert again.only_label_ids == (late,) and again.status is AssignmentStatus.OPEN
    assert again.priority == a.priority + 1 and {s.reason for s in again.flagged} == {
        ReviewReason.AI_WRONG
    }
    assert states[late] is VerificationState.UNREVIEWED
    assert states[ok] is VerificationState.SAMPLE_VERIFIED
    assert result.sample_verified == len(a.withheld_label_ids) - 1


def test_finish_without_ai_verifies_everything(twelve: sa.Engine, ops: ReviewOpsPolicy) -> None:
    """끔(ai=None)이면 지적이 있어도 기존처럼 나머지를 모두 표본 검증한다 (AI 표를 읽지 않는다)."""
    a = _plan(twelve, ops, None)
    with twelve.begin() as conn:
        for lid in a.sample_label_ids:
            record_review(conn, lid, VerificationState.HUMAN_APPROVED, "labeler01", FIXED_TIME)
        _collected_task(conn, a, a.sample_label_ids)
        result = finish_assignment(conn, a, FIXED_TIME, ops)
    assert result.sample_verified == len(a.withheld_label_ids) and result.ai_flagged == 0


def _state(**kw: Any) -> AiRunState:
    """실행 상태 (기본: 마지막 이벤트 started, 완료 버전 없음)."""
    event = AiRunEvent(session_id="s001", event=AiRunEventKind.STARTED, at=FIXED_TIME)
    data: dict[str, Any] = {
        "session_id": "s001", "last_event": event, "last_run": None,
        "completed_versions": frozenset[str](), "waived_at": None,
    } | kw  # fmt: skip
    return AiRunState(**data)


def test_required_gate_branches() -> None:
    """required만 막는다: 허용 버전 완료 → 진행, 건너뜀 기록 → 진행, prelabeled 뒤 max_wait_h 초과
    → 시간 초과 진행(waived:timeout 기록 요청), 그 밖(기록 없음 포함)은 미룬다."""
    req = _cfg(mode="required", max_wait_h=24)
    now = FIXED_TIME + timedelta(hours=30)
    assert required_gate(None, None, now, None).proceed
    assert required_gate(None, None, now, _cfg()).proceed  # advisory
    done = _state(completed_versions=frozenset({AI_VERSION}))
    assert required_gate(done, None, now, req).proceed
    other = _state(completed_versions=frozenset({"aireview-ffffffffffff+vlm-x"}))
    assert not required_gate(other, FIXED_TIME + timedelta(hours=10), now, req).proceed
    assert required_gate(_state(waived_at=FIXED_TIME), None, now, req).proceed
    timeout = required_gate(None, FIXED_TIME, now, req)
    assert timeout.proceed and timeout.waive_timeout
    wait = required_gate(None, FIXED_TIME + timedelta(hours=10), now, req)
    assert not wait.proceed and not wait.waive_timeout
    assert not required_gate(None, None, now, req).proceed


def test_policy_defaults_off_and_rejects_measurement_modes() -> None:
    """절이 없으면 끔, 블라인드·오류 삽입·QA는 show_modes에 넣을 수 없고 ok는 표본 제외에 없다."""
    assert AiReviewDisplayPolicy().mode == "off"
    with pytest.raises(ValidationError):
        AiReviewDisplayPolicy.model_validate({"show_modes": ["blind"]})
    with pytest.raises(ValidationError):
        AiReviewDisplayPolicy.model_validate({"sampling_exclude": ["ok"]})
    with pytest.raises(ValidationError):
        AiReviewDisplayPolicy.model_validate({"versions": ["v1"]})
