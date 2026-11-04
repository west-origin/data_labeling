"""손별 행동 타임라인 공백 검사 (F-REV-37, ADR 0012 "타임라인 공백 0").

정답 근거: 행동 단계는 손마다 [0, 세션 길이]를 action·gap으로 빈틈없이 덮는다. 검수 도구에서
행동을 지우면 그 시간이 비므로, verify는 그 손과 빈 구간을 알리며 완료를 막고, 검수자가 공백으로
채우면 통과한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_review.verify import timeline_gaps, verify_session
from dlp_schema import repo_root
from dlp_schema.db.repository import insert_labels, insert_session, register_ontology, set_lifecycle
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import load_ontology
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

SID = "s001"
LATER = FIXED_TIME + timedelta(days=1)


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


def _action(label_id: str, start: int, end: int, hand: str = "right") -> LabelRecord:
    """사람이 그린 행동 하나 (접근 시작·종료 = 라벨 구간)."""
    payload = action_payload(
        action_id=label_id, hand=hand, t_approach_ms=start, t_contact_start_ms=None,
        t_contact_end_ms=None, t_end_ms=end,
    )  # fmt: skip
    return make_label(payload, label_id=label_id, t_start_ms=start, t_end_ms=end)


def _gap(label_id: str, start: int, end: int, hand: str | None) -> LabelRecord:
    """사람이 그린 공백 (hand None은 양손 공통)."""
    payload: dict[str, Any] = {"kind": "gap", "hand": hand, "gap_type": "idle"}
    return make_label(payload, label_id=label_id, t_start_ms=start, t_end_ms=end)


def test_deleted_action_blocks_verify_until_gap_filled_f_rev_37(engine: sa.Engine) -> None:
    """F-REV-37: 오른손 1000~2000ms 행동이 지워진 타임라인은 verify가 손·구간을 밝혀 막는다.

    정답: 세션 길이 3000ms, 오른손 행동 0~1000·2000~3000 (사람). 이유 하나에 "right"와
    "1000~2000ms". 같은 구간을 사람 공백으로 채우면 human_verified로 옮긴다.
    """
    with engine.begin() as conn:
        session = make_session(SID, privacy_state=PrivacyState.APPROVED, duration_ms=3000)
        insert_session(conn, session)
        set_lifecycle(conn, SID, LifecycleState.PRIVACY_APPROVED)
        set_lifecycle(conn, SID, LifecycleState.PRELABELED)
        insert_labels(conn, [_action("a1", 0, 1000), _action("a2", 2000, 3000)])
    with engine.begin() as conn:
        blocked = verify_session(conn, SID, LATER, "lead")
    assert not blocked.verified
    assert [r for r in blocked.reasons if "타임라인" in r] == [
        "right 손 행동 타임라인 빈 시간 1곳: 1000~2000ms (검수 도구에서 행동이나 공백으로 채운다)"
    ]

    with engine.begin() as conn:
        insert_labels(conn, [_gap("g1", 1000, 2000, "right")])
        done = verify_session(conn, SID, LATER + timedelta(hours=1), "lead")
    assert done.verified, done.reasons


def test_shared_gap_covers_each_hand_and_edges_are_checked_f_rev_37() -> None:
    """F-REV-37: 양손 공통 공백은 두 손 모두를 덮고, 세션 처음·끝의 빈 시간도 찾는다.

    정답: 세션 길이 1000ms. 왼손 행동 0~500 + 공통 공백 500~1000 → 왼손은 빈틈 없음.
    오른손 행동 200~400 + 공통 공백 500~1000 → 0~200, 400~500 빈 시간.
    """
    labels = [
        _action("l1", 0, 500, hand="left"),
        _action("r1", 200, 400),
        _gap("g1", 500, 1000, None),
    ]
    assert timeline_gaps(1000, labels) == [
        "right 손 행동 타임라인 빈 시간 2곳: 0~200ms, 400~500ms "
        "(검수 도구에서 행동이나 공백으로 채운다)"
    ]


def test_hands_without_action_labels_are_not_checked_f_rev_37() -> None:
    """F-REV-37: action·gap 라벨이 없는 손(또는 세션)은 타임라인을 요구하지 않는다.

    정답: 행동 없는 세션 → 빈 목록. 오른손만 빈틈없이 덮으면 왼손은 보지 않아 빈 목록.
    """
    assert timeline_gaps(1000, []) == []
    assert timeline_gaps(1000, [_action("r1", 0, 1000)]) == []
