"""3D 손 궤적이 2D 손 트랙 검수 결과를 따르게 DB에 적용한다 (F-COE-19, ADR 0048).

규칙은 `dlp_schema.trajectory_follow`(순수 계산)에 있다. 이 모듈은 세션을 잠그고 이력·기록을 읽어,
물려받는 검수를 `record_review`(라벨 행에서 바꿀 수 있는 유일한 경로)와 같은 트랜잭션의 물려받기
기록(`trajectory_inheritance`, 추가만)으로 쓴다. 아무것도 지우지 않는다: 출처가 끊긴 물려받은
궤적(`lapsed`)은 결과로 알리기만 한다. 멱등이다.

호출자: 검수 수집(`dlp_review.collect.collect_task`, 표본 판정 뒤), 완료 판정
(`dlp_review.verify.verify_session`), 프리라벨(`dlp_prelabel.runner.run_prelabel`, 3D·월드 좌표
단계 뒤). 트랜잭션은 호출자가 연다.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa

from dlp_schema.db.inheritance_repository import (
    get_trajectory_inheritance,
    record_trajectory_inheritance,
)
from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import get_labels, record_review
from dlp_schema.review import TrajectoryInheritanceMechanism
from dlp_schema.trajectory_follow import TrajectoryFollow, plan_trajectory_follow


def apply_trajectory_follow(
    conn: sa.Connection,
    session_id: str,
    now: datetime,
    mechanism: TrajectoryInheritanceMechanism,
) -> TrajectoryFollow:
    """세션의 3D 손 궤적이 출처 2D 손 트랙의 검수 결과를 따르게 한다.

    Args:
        conn: 열린 트랜잭션의 DB 연결.
        session_id: 세션 ID.
        now: 물려받기 기록 시각 (시간대 필수). 물려받는 검수 시각은 규칙이 정한다.
        mechanism: 물려받기 기록에 남길 경로 (수집·완료 판정·프리라벨).

    Returns:
        적용한 결과 (`TrajectoryFollow`). lapsed는 알림용이다 (쓰지 않는다).

    부작용: 세션 잠금(트랜잭션 끝까지), 궤적 행의 검수 상태 갱신과 같은 트랜잭션의
    `trajectory_inheritance` 기록.
    """
    lock_session(conn, session_id)
    plan = plan_trajectory_follow(
        get_labels(conn, session_id), get_trajectory_inheritance(conn, session_id)
    )
    for r in plan.approve:
        record_review(conn, r.label_id, r.state, r.reviewer_id, r.reviewed_at)
        record_trajectory_inheritance(conn, r.record(session_id, mechanism, now))
    return plan
