"""세션 검수 완료 판정 (`dlp review verify`).

생애주기의 prelabeled → human_verified 전이를 맡는다. 데이터셋 빌드 후보는 프라이버시 승인 세션
전체이지만(ADR 0031), 빌드가 split_assigned로 옮기는 것은 이 전이를 거친 세션뿐이다
(dlp_datasets.build). 운영 지표의 "검증 에피소드"와 원본 보관 만료 알림도 이 상태 이후의 세션만
본다. 전이 시각은 session_lifecycle_events에 남는다 (ADR 0028).

완료 조건 (모두 만족해야 한다):
- 블러 검수가 끝나 privacy_state가 approved다 (승인이 풀린 세션은 완료로 보지 않는다).
- 이 세션의 검수 작업 중 수거되지 않은 것(open)이 없다. 블러 작업 포함.
- 열린 검수 배정(review_assignments.status=open)이 없다.
- 블러를 뺀 현재 운영 라벨 중 모델 라벨은 모두 사람 검수 상태다
  (human_approved, human_corrected, sample_verified). 사람이 만든 라벨은 그 자체로 검수된 것이다.
- 블러를 뺀 현재 운영 라벨이 하나 이상 있다 (빈 세션은 완료가 아니다).

관련: ADR 0028(생애주기 기록), ADR 0029(검수 완료 판정 명령).

공개 이름:
- `VERIFIED_STATES`: 모델 라벨을 "사람이 확인했다"고 보는 검증 상태.
- `VerifyResult`: 판정 결과.
- `verification_gaps`: 완료를 막는 이유 목록 (읽기만).
- `verify_session`: 조건을 보고 생애주기를 옮긴다 (DB 쓰기, 멱등).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_assignments,
    list_review_tasks,
    set_lifecycle,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import Source, VerificationState
from dlp_schema.review import AssignmentStatus, ReviewTaskStatus
from dlp_schema.session import LifecycleState, PrivacyState

# 모델 라벨이 "사람이 확인했다"고 볼 수 있는 검증 상태.
# SAMPLE_VERIFIED(표본 합격으로 일괄 검증)도 포함한다. 오류 삽입 정답 풀(`ops.runner.VERIFIED`)은
# 표본 검증을 정답으로 보지 않아 이 집합과 다르다.
VERIFIED_STATES = (
    VerificationState.HUMAN_APPROVED,
    VerificationState.HUMAN_CORRECTED,
    VerificationState.SAMPLE_VERIFIED,
)


@dataclass(frozen=True)
class VerifyResult:
    """판정 결과. verified면 생애주기를 바꿨거나 이미 완료였다. 아니면 reasons에 남은 일."""

    # 판정한 세션 ID
    session_id: str
    # True: 이번에 human_verified로 옮겼거나 이미 그 뒤 단계다
    verified: bool
    # True: 이미 human_verified 이후 단계라 아무것도 바꾸지 않았다
    already: bool = False
    # verified가 False일 때 완료를 막은 이유 (사람이 읽는 한국어 문장)
    reasons: list[str] = field(default_factory=list[str])


def verification_gaps(conn: sa.Connection, session_id: str) -> list[str]:
    """완료를 막는 이유 목록 (비었으면 완료 조건을 모두 만족).

    인자: conn(DB 연결, 읽기만), session_id(세션 ID).
    반환: 이유 문장 목록. 순서는 블러 승인 → 작업 → 배정 → 라벨.
    예외: 세션이 없으면 `get_session`이 던지는 예외.
    부작용 없음 (`sessions`, `review_tasks`, `review_assignments`, `label_records` 조회).
    """
    session = get_session(conn, session_id)
    reasons: list[str] = []
    if session.privacy_state is not PrivacyState.APPROVED:
        reasons.append(f"블러 승인 전 (privacy_state={session.privacy_state.value})")
    open_tasks = [
        t.task_key for t in list_review_tasks(conn, session_id) if t.status is ReviewTaskStatus.OPEN
    ]
    if open_tasks:
        # 메시지가 길어지지 않게 앞 5개만 보여 준다
        reasons.append(f"수거하지 않은 검수 작업 {len(open_tasks)}개: {', '.join(open_tasks[:5])}")
    open_assignments = list_assignments(conn, session_id, status=AssignmentStatus.OPEN)
    if open_assignments:
        reasons.append(f"열린 검수 배정 {len(open_assignments)}개")
    # 블러 라벨은 블러 승인(privacy_state)으로 따로 판정하므로 라벨 조건에서 뺀다.
    # current_labels는 운영 라벨만 준다 (오류 삽입·측정 레코드 제외).
    labels = [x for x in current_labels(get_labels(conn, session_id)) if x.kind != "blur_track"]
    if not labels:
        reasons.append("블러를 뺀 운영 라벨이 없다")
    pending = [
        x.label_id
        for x in labels
        if x.provenance.source is Source.MODEL and x.verification.state not in VERIFIED_STATES
    ]
    if pending:
        reasons.append(f"검수하지 않은 모델 라벨 {len(pending)}개 (예: {pending[0]})")
    return reasons


def verify_session(conn: sa.Connection, session_id: str, now: datetime, actor: str) -> VerifyResult:
    """조건을 만족하면 prelabeled → human_verified로 옮긴다. 멱등: 이미 그 뒤 단계면 그대로 둔다.

    now: 전이 시각 (시간대 필수, 운영 지표의 검증 주가 된다). actor: 판정을 실행한 사람·서비스.

    반환: `VerifyResult`. prelabeled가 아니면(그 전 단계 또는 withdrawn) verified=False와 그 이유를
    돌려준다 (예외 아님. withdrawn이어도 이유 문장은 "프리라벨 전 단계다"로 나온다).
    부작용: 조건을 모두 만족하면 `set_lifecycle`로 `sessions.lifecycle_state`를 바꾸고
    `session_lifecycle_events`에 전이 기록을 남긴다. 트랜잭션은 호출자가 연다.
    """
    session = get_session(conn, session_id)
    state = session.lifecycle_state
    # human_verified 이후 단계(분할 배정, 내보냄)는 이미 완료로 본다
    if state in (
        LifecycleState.HUMAN_VERIFIED,
        LifecycleState.SPLIT_ASSIGNED,
        LifecycleState.EXPORTED,
    ):
        return VerifyResult(session_id, verified=True, already=True)
    if state is not LifecycleState.PRELABELED:
        return VerifyResult(
            session_id, verified=False, reasons=[f"프리라벨 전 단계다 (lifecycle={state.value})"]
        )
    reasons = verification_gaps(conn, session_id)
    if reasons:
        return VerifyResult(session_id, verified=False, reasons=reasons)
    set_lifecycle(conn, session_id, LifecycleState.HUMAN_VERIFIED, at=now, actor=actor)
    return VerifyResult(session_id, verified=True)
