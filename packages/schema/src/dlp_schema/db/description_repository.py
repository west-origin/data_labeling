"""설명 라벨이 행동 검수 결과를 따르게 DB에 적용한다 (F-PAR-15, ADR 0043 3절).

규칙은 `dlp_schema.description_follow`(순수 계산)에 있다. 이 모듈은 세션을 잠그고 이력을 읽어 그
결과를 쓴다: 지울 설명은 삭제 레코드(출처 `FOLLOW_VERSION`, 추가만), 물려받는 검수는
`record_review`(라벨 행에서 바꿀 수 있는 유일한 경로). 멱등이다.

호출자: 검수 수집(`dlp_review.collect.collect_task`, 표본 판정 뒤), 완료 판정
(`dlp_review.verify.verify_session`, 이 규칙 전에 검수한 세션을 메운다), 번역 단계
(`dlp_actions.translate.run_translate`, 행동 승인 뒤에 만든 번역). 트랜잭션은 호출자가 연다.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa

from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import get_labels, insert_labels, record_review
from dlp_schema.description_follow import (
    FOLLOW_VERSION,
    DescriptionFollow,
    plan_description_follow,
)
from dlp_schema.episode import retractions


def apply_description_follow(
    conn: sa.Connection, session_id: str, now: datetime
) -> DescriptionFollow:
    """세션의 설명이 주인(행동·상위 구간)의 검수 결과를 따르게 한다.

    Args:
        conn: 열린 트랜잭션의 DB 연결.
        session_id: 세션 ID.
        now: 삭제 레코드 생성 시각 (시간대 필수). 물려받는 검수 시각은 규칙이 정한다.

    Returns:
        적용한 결과 (`DescriptionFollow`, 비었으면 바꾼 것이 없다).

    부작용: 세션 잠금(트랜잭션 끝까지), `label_records` 삭제 레코드 추가, 설명 행의 검수 상태 갱신.
    """
    lock_session(conn, session_id)
    plan = plan_description_follow(get_labels(conn, session_id))
    if plan.retract:
        insert_labels(conn, retractions(plan.retract, FOLLOW_VERSION, now))
    for r in plan.approve:
        record_review(conn, r.label_id, r.state, r.reviewer_id, r.reviewed_at)
    return plan
