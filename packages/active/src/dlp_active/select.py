"""세션 점수와 선택 (`dlp active rank`).

1. 수정률: 모든 세션의 라벨 이력에서 클래스별 수정률을 센다 (개별 검수된 것만).
2. 후보: 정책의 생애주기(기본 prelabeled) 세션. 골든셋 세션과 사용 중지 세션은 뺀다.
3. 점수: 켜진 항목의 가중합. 정책 normalize가 per_minute면 영상 1분당으로 나눈다.
4. 높은 점수부터 고른다 (같으면 세션 ID 순 — 같은 입력이면 같은 결과).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlalchemy as sa

from dlp_active.policy import ActivePolicy
from dlp_active.rates import RateTable, correction_rates
from dlp_active.terms import SessionContext, build_terms
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_golden_sets,
    list_session_ids,
    withdrawn_session_ids,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.session import Session


@dataclass
class SessionScore:
    session_id: str
    score: float
    terms: dict[str, float]  # 항목 → 가중치 곱하기 전 점수
    pending: int  # 검수 대기 라벨 수
    top_classes: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])


def pending_labels(history: list[LabelRecord], policy: ActivePolicy) -> list[LabelRecord]:
    return [
        x
        for x in current_labels(history)
        if x.provenance.source is Source.MODEL
        and x.verification.state is VerificationState.UNREVIEWED
        and x.kind not in policy.excluded_kinds
    ]


def score_sessions(
    sessions: list[tuple[Session, list[LabelRecord]]],
    rates: RateTable,
    policy: ActivePolicy,
    top: int = 3,
) -> list[SessionScore]:
    terms = build_terms(policy)
    out: list[SessionScore] = []
    for session, history in sessions:
        ctx = SessionContext(session, pending_labels(history, policy), rates)
        raw = {t.name: t.score(ctx) for t, _ in terms}
        total = sum(w * raw[t.name] for t, w in terms)
        if policy.score.normalize == "per_minute":
            total /= max(session.duration_ms / 60_000, 1e-9)
        contrib: dict[str, float] = {}
        for t, w in terms:
            for k, v in t.explain(ctx).items():
                contrib[k] = contrib.get(k, 0.0) + w * v
        best = sorted(contrib.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        out.append(SessionScore(session.session_id, total, raw, len(ctx.pending), best))
    return sorted(out, key=lambda s: (-s.score, s.session_id))


def rank_sessions(
    conn: sa.Connection, policy: ActivePolicy, limit: int | None = None
) -> tuple[list[SessionScore], RateTable]:
    withdrawn = withdrawn_session_ids(conn)
    golden = (
        {sid for g in list_golden_sets(conn) for sid in g.session_ids}
        if policy.candidates.exclude_golden
        else set[str]()
    )
    histories: list[list[LabelRecord]] = []
    candidates: list[tuple[Session, list[LabelRecord]]] = []
    for sid in list_session_ids(conn):
        history = get_labels(conn, sid)
        histories.append(history)
        session = get_session(conn, sid)
        if (
            session.lifecycle_state in policy.candidates.lifecycle
            and sid not in withdrawn
            and sid not in golden
        ):
            candidates.append((session, history))
    rates = correction_rates(histories, policy)
    ranked = score_sessions(candidates, rates, policy)
    return ranked[: limit or policy.select], rates
