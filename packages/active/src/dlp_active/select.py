"""세션 점수와 선택 (`dlp active rank`).

1. 수정률: 골든셋·사용 중지 세션을 뺀 세션의 라벨 이력에서 클래스별 수정률을 센다
   (개별 검수된 것만).
2. 후보: 정책의 생애주기(기본 prelabeled) 세션. 사용 중지 세션은 늘, 골든셋 세션은
   `candidates.exclude_golden`이면 뺀다.
3. 점수: 켜진 항목의 가중합. 정책 normalize가 per_minute면 영상 1분당으로 나눈다.
4. 높은 점수부터 고른다 (같으면 세션 ID 순 — 같은 입력이면 같은 결과).

DB는 읽기만 한다. 고른 결과는 CLI가 출력하고, 검수 작업 생성은 `dlp review`가 따로 한다.
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
    """세션 하나의 점수."""

    session_id: str
    score: float  # 가중합 (per_minute면 1분당)
    terms: dict[str, float]  # 항목 → 가중치 곱하기 전 점수
    pending: int  # 검수 대기 라벨 수
    # 기여가 큰 클래스 (클래스 키, 가중 기여), 큰 순
    top_classes: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])


def pending_labels(history: list[LabelRecord], policy: ActivePolicy) -> list[LabelRecord]:
    """검수 대기 라벨: 운영 현재 라벨 중 미검수 모델 라벨 (제외 종류 빼고)."""
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
    """세션들에 점수를 매겨 높은 순으로 돌려준다 (순수 함수).

    Args:
        sessions: (세션, 그 세션의 전체 라벨 이력) 목록.
        rates: 클래스별 수정률.
        policy: 정책 (`score`, `excluded_kinds`).
        top: `top_classes`에 남길 클래스 수.

    Raises:
        ValueError: 정책에 등록되지 않은 항목이 있을 때 (`build_terms`).
    """
    terms = build_terms(policy)
    out: list[SessionScore] = []
    for session, history in sessions:
        ctx = SessionContext(session, pending_labels(history, policy), rates)
        raw = {t.name: t.score(ctx) for t, _ in terms}
        total = sum(w * raw[t.name] for t, w in terms)
        if policy.score.normalize == "per_minute":
            # 길이 0 세션의 0 나누기를 막는다
            total /= max(session.duration_ms / 60_000, 1e-9)
        # 클래스별 가중 기여 (모든 항목의 explain을 더한다)
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
    """DB의 모든 세션에서 수정률을 세고, 후보 세션을 점수 순으로 고른다.

    Args:
        conn: DB 연결 (읽기만).
        policy: 정책.
        limit: 고를 수 (None이면 정책 `select`).

    Returns:
        (상위 세션 점수, 수정률 표).

    모든 세션의 라벨 이력을 읽으므로 세션 수에 비례해 느려진다.
    """
    withdrawn = withdrawn_session_ids(conn)
    golden = {sid for g in list_golden_sets(conn) for sid in g.session_ids}
    histories: list[list[LabelRecord]] = []
    candidates: list[tuple[Session, list[LabelRecord]]] = []
    for sid in list_session_ids(conn):
        if sid in withdrawn:
            continue
        history = get_labels(conn, sid)
        # 골든셋 정답은 사람이 처음부터 만든 라벨이라 "추가"로 세면 수정률이 부풀려진다.
        # 수정률은 프리라벨을 검수한 세션에서만 센다.
        if sid not in golden:
            histories.append(history)
        session = get_session(conn, sid)
        if session.lifecycle_state in policy.candidates.lifecycle and not (
            policy.candidates.exclude_golden and sid in golden
        ):
            candidates.append((session, history))
    rates = correction_rates(histories, policy)
    ranked = score_sessions(candidates, rates, policy)
    return ranked[: limit or policy.select], rates
