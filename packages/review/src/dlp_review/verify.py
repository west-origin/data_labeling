"""세션 검수 완료 판정 (`dlp review verify`).

생애주기의 prelabeled → human_verified 전이를 맡는다. 데이터셋 빌드 후보는 프라이버시 승인 세션
전체이지만(ADR 0031), 빌드가 split_assigned로 옮기는 것은 이 전이를 거친 세션뿐이다
(dlp_datasets.build). 운영 지표의 "검증 에피소드"와 원본 보관 만료 알림도 이 상태 이후의 세션만
본다. 전이 시각은 session_lifecycle_events에 남는다 (ADR 0028).

완료 조건 (모두 만족해야 한다):
- 블러 검수가 끝나 privacy_state가 approved다 (승인이 풀린 세션은 완료로 보지 않는다).
- 이 세션의 검수 작업 중 수거되지 않은 것(open)이 없다. 블러 작업 포함.
- 매체 회수(ADR 0039)가 끝나지 않은 작업이 없고, 렌더 해시가 지금 블러본과 다른데 회수 요청이 없는
  작업도 없다 (렌더 해시가 없는 구 작업은 이 판정에서 빼 기존 세션을 갑자기 막지 않는다).
- 열린 검수 배정(review_assignments.status=open)이 없다. 회수돼 다시 만들어야 할 배정은 따로 알린다.
- 배정 없이 만든 직접 작업이 열린 채 회수됐으면(status=revoked) 그 뒤 같은 담당자·스트림·도구로
  다시 만든 작업이 있다. 회수 전에는 열린 작업이 완료를 막았으므로 회수가 그 조건을 조용히 풀지
  않게 한다. 다시 만든 작업은 열린 채 회수되지 않은 것(open 또는 collected)이다. 수집한 재작업은
  그 뒤 블러가 또 바뀌어 회수 요청이 와도 결과가 이미 이력에 있으므로 다시 만든 작업으로 인정한다
  (수집 출력 `collect._note_revocations`와 같은 기준).
- 블러를 뺀 현재 운영 라벨 중 모델 라벨은 모두 사람 검수 상태다
  (human_approved, human_corrected, sample_verified). 사람이 만든 라벨은 그 자체로 검수된 것이다.
  모델 설명 라벨(한국어 원문·영어 번역)은 검수 도구 경로가 없어 행동·상위 구간 검수를 물려받는다
  (F-PAR-15, ADR 0043 3절). `verify_session`은 판정 전에 그 규칙을 한 번 적용해, 규칙 전에 검수한
  세션도 메운다. 행동이 검수 전이면 그 설명도 계속 완료를 막는다. 3D 손 궤적(카메라·월드 좌표)도
  검수 도구 경로가 없어 그것을 계산한 2D 손 트랙의 검수를 물려받는다 (ADR 0048,
  `apply_trajectory_follow`). 2D가 검수 전이거나 출처 2D가 없는 3D(카메라 자세, 박스 중심)는 계속
  완료를 막는다 (F-COE-21).
- 블러를 뺀 현재 운영 라벨이 하나 이상 있다 (빈 세션은 완료가 아니다).

관련: ADR 0028(생애주기 기록), ADR 0029(검수 완료 판정 명령).

공개 이름:
- `VERIFIED_STATES`: 모델 라벨을 "사람이 확인했다"고 보는 검증 상태.
- `VerifyResult`: 판정 결과.
- `reissue_slot`: 회수된 직접 작업을 다시 만드는 단위 (담당자, 스트림, 도구). 수집 출력과 공유한다.
- `verification_gaps`: 완료를 막는 이유 목록 (읽기만).
- `verify_session`: 조건을 보고 생애주기를 옮긴다 (DB 쓰기, 멱등).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import stale_media_tasks
from dlp_schema import repo_root
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_assignments,
    list_review_tasks,
    pending_revocations,
    set_lifecycle,
)
from dlp_schema.db.trajectory_repository import apply_trajectory_follow
from dlp_schema.episode import current_labels
from dlp_schema.labels import Source, VerificationState
from dlp_schema.review import (
    AssignmentStatus,
    InheritanceMechanism,
    ReviewAssignment,
    ReviewTask,
    ReviewTaskStatus,
    TrajectoryInheritanceMechanism,
)
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
    # 판정 전에 행동 검수를 물려받은 설명 수와 지운 설명 수 (F-PAR-15, 메우기)
    descriptions_approved: int = 0
    descriptions_retracted: int = 0
    # 판정 전에 2D 손 트랙 검수를 물려받은 3D 궤적 수와 출처가 끊긴 물려받은 3D 수 (ADR 0048)
    trajectories_approved: int = 0
    trajectories_lapsed: int = 0


def verification_gaps(
    conn: sa.Connection, session_id: str, privacy_policy: PrivacyPolicy | None = None
) -> list[str]:
    """완료를 막는 이유 목록 (비었으면 완료 조건을 모두 만족).

    인자: conn(DB 연결, 읽기만), session_id(세션 ID), privacy_policy(렌더 해시 계산용 프라이버시
    정책, None이면 저장소의 `privacy.yaml`).
    반환: 이유 문장 목록. 순서는 블러 승인 → 작업 → 매체 회수 → 배정 → 라벨.
    예외: 세션이 없으면 `get_session`이 던지는 예외.
    부작용 없음 (`sessions`, `review_tasks`, `review_assignments`, `label_records` 조회).
    """
    session = get_session(conn, session_id)
    reasons: list[str] = []
    if session.privacy_state is not PrivacyState.APPROVED:
        reasons.append(f"블러 승인 전 (privacy_state={session.privacy_state.value})")
    tasks = list_review_tasks(conn, session_id)
    open_tasks = [t.task_key for t in tasks if t.status is ReviewTaskStatus.OPEN]
    if open_tasks:
        # 메시지가 길어지지 않게 앞 5개만 보여 준다
        reasons.append(f"수거하지 않은 검수 작업 {len(open_tasks)}개: {', '.join(open_tasks[:5])}")
    reasons += _revocation_gaps(
        conn, session_id, privacy_policy or load_privacy_policy(repo_root())
    )
    open_assignments = list_assignments(conn, session_id, status=AssignmentStatus.OPEN)
    reasons += _reissue_gaps(session_id, tasks, open_assignments)
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


# 회수 뒤 다시 만들기 안내에 넣는 AI 1차 검수 단계 (WP17.6). 블러가 바뀌면 이전 판정은 렌더 해시가
# 달라 표시되지 않으므로, AI 판정을 쓰는 운영(review.yaml ai_review.mode가 off가 아님)에서는 다시
# 만들기 전에 판정을 다시 돌린다. 완료 판정은 검수 정책을 읽지 않아 조건부 문구로 둔다
AI_RERUN = " → (AI 1차 검수를 쓰면) dlp aireview run {session}"


def reissue_slot(task: ReviewTask) -> tuple[str, str, str]:
    """직접 작업을 다시 만드는 단위 (담당자, 스트림, 도구 값). 담당자가 없으면 "unassigned".

    수집 출력의 `direct_reissue`(`collect._note_revocations`)와 완료 판정(`_direct_reissue`)이
    같은 단위를 쓴다. `dlp review create --assignee --stream --tool` 인자와 대응한다.
    """
    return (task.assignee or "unassigned", task.stream_id, task.tool.value)


def _direct_reissue(tasks: list[ReviewTask]) -> list[tuple[str, str, str]]:
    """다시 만들어야 할 직접 작업의 (담당자, 스트림, 도구), 처음 나온 순.

    기준 (수집 출력의 `direct_reissue`, `collect._note_revocations`와 같다):
    - 다시 만들 대상: 배정 없이 만든 작업이 열린 채 회수됐다 (status=revoked). 수집한 뒤 회수 요청을
      받은 작업은 상태가 collected로 남고 결과가 이미 이력에 있어 대상이 아니다.
    - 다시 만든 작업: 회수 요청 뒤 같은 단계·담당자·스트림·도구로 만든 직접 작업 중 열린 채 회수되지
      않은 것 (open 또는 collected). 수집한 재작업은 그 뒤 블러가 또 바뀌어 회수 요청을 받아도
      (collected + `revoke_requested_at`) 다시 만든 작업으로 인정한다. 회수 요청이 없는 작업만
      인정하면 수집 출력은 다시 만들라고 하지 않는데 완료 판정은 처음 작업을 다시 만들라고 막는다.
    열린 재작업은 "수거하지 않은 검수 작업"으로 따로 완료를 막는다.
    """
    out: list[tuple[str, str, str]] = []
    for t in tasks:
        at = t.revoke_requested_at
        if t.status is not ReviewTaskStatus.REVOKED or t.assignment_id is not None or at is None:
            continue
        renewed = any(
            n.assignment_id is None
            and n.status is not ReviewTaskStatus.REVOKED
            and n.stage is t.stage
            and reissue_slot(n) == reissue_slot(t)
            and n.created_at >= at
            for n in tasks
        )
        if not renewed and reissue_slot(t) not in out:
            out.append(reissue_slot(t))
    return out


def _reissue_gaps(
    session_id: str, tasks: list[ReviewTask], open_assignments: list[ReviewAssignment]
) -> list[str]:
    """회수돼 다시 만들어야 할 일의 이유 문장: 회수한 작업에 딸린 열린 배정, 다시 만들지 않은
    직접 작업 (ADR 0039)."""
    reasons: list[str] = []
    revoked = {t.task_key for t in tasks if t.status is ReviewTaskStatus.REVOKED}
    reissue = [a.assignment_id for a in open_assignments if a.task_key in revoked]
    if reissue:
        reasons.append(
            f"회수돼 다시 만들 배정 {len(reissue)}개 "
            f"(재승인·렌더{AI_RERUN.format(session=session_id)} 뒤 "
            f"dlp review assign {' '.join(reissue[:3])})"
        )
    direct = _direct_reissue(tasks)
    if direct:
        commands = "; ".join(
            f"dlp review create {session_id} --stage labeling --assignee {who} --stream {stream} "
            f"--tool {tool}"
            for who, stream, tool in direct[:2]
        )
        rerun = AI_RERUN.format(session=session_id)
        reasons.append(
            f"회수돼 다시 만들 직접 작업 {len(direct)}개 (재승인·렌더{rerun} 뒤 {commands})"
        )
    return reasons


def _revocation_gaps(conn: sa.Connection, session_id: str, policy: PrivacyPolicy) -> list[str]:
    """매체 회수 미완료와 회수 요청 없는 낡은 작업의 이유 문장 (ADR 0039).

    낡은 작업은 렌더 해시가 있는 것만 센다 (구 작업은 `dlp review revoke --include-legacy`로 정리).
    """
    reasons: list[str] = []
    pending = pending_revocations(conn, session_id)
    if pending:
        reasons.append(f"매체 회수 미완료 {len(pending)}개 (dlp review revoke {session_id})")
    stale = stale_media_tasks(conn, session_id, policy)
    if stale:
        reasons.append(
            f"블러가 바뀐 뒤 회수 요청 없는 작업 {len(stale)}개 (dlp review revoke {session_id})"
        )
    return reasons


def verify_session(
    conn: sa.Connection,
    session_id: str,
    now: datetime,
    actor: str,
    privacy_policy: PrivacyPolicy | None = None,
) -> VerifyResult:
    """조건을 만족하면 prelabeled → human_verified로 옮긴다. 멱등: 이미 그 뒤 단계면 그대로 둔다.

    now: 전이 시각 (시간대 필수, 운영 지표의 검증 주가 된다). actor: 판정을 실행한 사람·서비스.
    privacy_policy: `verification_gaps`에 그대로 넘긴다 (None이면 저장소 정책).

    반환: `VerifyResult`. prelabeled가 아니면(그 전 단계 또는 withdrawn) verified=False와 그 이유를
    돌려준다 (예외 아님. withdrawn이어도 이유 문장은 "프리라벨 전 단계다"로 나온다).
    부작용: prelabeled면 먼저 설명 규칙을 적용한다(`apply_description_follow`: 설명 검수 상태 갱신·
    검수 전 설명 삭제 레코드, 세션 잠금). 조건을 모두 만족하면 `set_lifecycle`로
    `sessions.lifecycle_state`를 바꾸고
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
    # 이 규칙 전에 수집한 세션도 설명이 행동 검수를 따르게 메운다 (멱등, F-PAR-15)
    follow = apply_description_follow(conn, session_id, now, InheritanceMechanism.VERIFY)
    # 3D 손 궤적도 그것을 계산한 2D 손 트랙 검수를 물려받는다 (ADR 0048, 멱등)
    traj = apply_trajectory_follow(conn, session_id, now, TrajectoryInheritanceMechanism.VERIFY)
    reasons = verification_gaps(conn, session_id, privacy_policy)
    if not reasons:
        set_lifecycle(conn, session_id, LifecycleState.HUMAN_VERIFIED, at=now, actor=actor)
    return VerifyResult(
        session_id, verified=not reasons, reasons=reasons,
        descriptions_approved=len(follow.approve), descriptions_retracted=len(follow.retract),
        trajectories_approved=len(traj.approve), trajectories_lapsed=len(traj.lapsed),
    )  # fmt: skip
