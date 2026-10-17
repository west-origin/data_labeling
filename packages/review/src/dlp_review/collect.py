"""검수 결과 수집.

도구에서 결과를 받아 reconcile하고 DB에 이력으로 쓴다. 한 작업은 한 번만 수집한다
(세션과 작업 행을 순서대로 잠근다. 같은 작업의 재수집과 회수한 작업의 수집은 None이며, 다른 작업도
같은 세션의 이력을 직렬로 읽어 같은 부모를 두 번 수정하지 않는다).

- 수집하는 검수자는 작업 담당자여야 한다 (웹훅의 도구 사용자 ID가 아니라 작업의 담당자로 기록).
- Label Studio는 그 검수자 계정(review.yaml label_studio.users)이 만들고 마지막으로 고친 주석만
  결과·검수 시간으로 쓴다. 다른 사용자·서비스 계정의 주석은 받지 않고 경고 로그만 남긴다 (F-REV-06).
- 작업을 보낸 뒤 다른 단계가 지웠거나(retracted) 다른 작업이 먼저 고친(자식 레코드가 생긴) 라벨은
  검수 결과로 되살리거나 두 갈래 이력을 만들지 않는다.
- CVAT는 트랙과 모양(Shape 모드, CVAT 기본) 주석을 모두 받는다. 옮길 수 없는 주석(태그, 다각형 등)이
  있으면 수집하지 않는다 (TaskError).
- Label Studio 결과에 라벨로 옮길 수 없는 구간(새로 그린 객체 상태, 필수 입력이 빠진 손 상태 등)이
  있으면 그 구간들의 영역 ID·시각·이유를 담은 TaskError로 수집하지 않는다 (F-REV-03).
- 운영 블러 라벨이 바뀌면 프라이버시 승인을 풀어 다시 승인·렌더하게 하고, 그 스트림의 이미 배포한
  라벨링 작업 매체에 회수를 요청하고(같은 커밋, ADR 0039), 이전 블러본 렌더 기록을 무효로 둔다
  (ADR 0024). 외부 삭제(S3 사본·도구 작업)는 커밋 뒤 호출자가 `dlp_review.revoke`로 한다.
- 설명 라벨(한국어 원문·영어 번역)은 검수 도구 경로가 없어 그것이 설명하는 행동·상위 구간의 결과를
  따른다 (F-PAR-15, ADR 0043 3절): 그대로 승인·시각만 고침 → 같은 검수자·시각으로 승인, 표본 합격 →
  표본 검증, 지움·의미를 고침 → 검수 전 설명을 지운다. 이미 검수된 설명은 건드리지 않는다.
- AI 1차 검수(WP17.7, ADR 0040): 노출 기록이 있는 작업이면 검수자가 명시적으로 고른 "제안 적용"만
  검수 항목에 반영한다. 제안 내용은 DB의 판정 단위에서 다시 읽어 검사하고, 세션 검증에 걸리면 AI
  적용만 되돌린다 (`ai_collect`). 사람 결과와 판정의 연결은 `ai_review_outcomes`에 작업마다 한 번
  쓴다. 아무것도 고르지 않고 제출하면 결과는 AI가 없을 때와 같다.

WP6·WP12, ADR 0006·0014·0015·0024. 진입점: `dlp review collect <작업 키>`와 웹훅 서버
(`dlp review serve`)가 `collect_task`를 부른다.

공개 이름:
- `OPERATIONAL_MODES`: 결과가 운영 라벨이 되는 배정 방식 (표준, QA).
- `drop_retracted`: 보낸 뒤 바뀐 라벨에 대한 결과를 뺀다.
- `collect_task`: 작업 하나를 수집한다.

부작용 요약 (`collect_task`): `label_records` 삽입, 검수 상태 갱신(`record_review`),
`review_tasks` 수집 표시, `review_work` 기록(Label Studio, 잰 시간이 0보다 클 때만),
`ai_review_outcomes` 기록(AI를 켜고 노출 기록이 있는 작업만), 설명 라벨의 검수 상태 물려받기·
검수 전 설명 삭제(F-PAR-15, `apply_description_follow`, 배정 마무리 뒤), 3D 손 궤적의 2D 트랙
검수 물려받기(ADR 0048, `apply_trajectory_follow`, 그 뒤), 세션
`privacy_state` 되돌림(승인 상태였을 때만), 블러가 바뀐 스트림의 라벨링 작업 회수 요청
(`review_tasks`, 외부 호출 없음), 라벨링 버킷의 렌더 기록 무효화, 배정 마무리
(`review_assignments`). 블러 검수 작업이면 원본 버킷의
프록시를 내려받으므로 원본 접근 기록이 남는다 (setup.raw가 감사 저장소일 때, ADR 0020).
트랜잭션은 호출자가 연다 (잠금이 트랜잭션 끝까지 유지된다).
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from dlp_privacy.runner import invalidate_render, request_media_revocation
from dlp_review import ai_collect
from dlp_review.cvat import (
    UNIT_SCALE,
    CvatFormatError,
    CvatSchema,
    annotation_tracks,
    from_cvat_tracks,
    quantize,
)
from dlp_review.labelstudio import LsFormatError, from_ls_results
from dlp_review.reconcile import ReviewedItem, ReviewOutcome, reconcile
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    current_for_review,
    frame_times,
    label_scale,
    object_key,
    require_label_studio_user,
)
from dlp_review.verify import reissue_slot
from dlp_schema.aireview import AiDecision
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_review_task,
    get_session,
    insert_labels,
    insert_review_work,
    lock_session,
    mark_review_task_collected,
    record_review,
    set_privacy_state,
)
from dlp_schema.db.tables import review_tasks
from dlp_schema.db.trajectory_repository import apply_trajectory_follow
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ops import ReviewWork
from dlp_schema.review import (
    InheritanceMechanism,
    ReviewMode,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
    TrajectoryInheritanceMechanism,
)
from dlp_schema.session import PrivacyState, Session

if TYPE_CHECKING:
    from dlp_review.ops.policy import ReviewOpsPolicy

logger = logging.getLogger(__name__)

# 결과가 운영 라벨에 반영되는 배정 방식. 블라인드·이중은 측정 레코드, 오류 삽입은 seeded_error라서
# 운영 블러가 바뀌지 않으므로 프라이버시 승인을 풀 필요가 없다.
OPERATIONAL_MODES = (ReviewMode.STANDARD, ReviewMode.QA)


def _lock_task(conn: sa.Connection, task_key: str) -> ReviewTaskStatus:
    """작업 행을 잠그고(SELECT … FOR UPDATE) 현재 상태를 돌려준다.

    잠금은 호출자 트랜잭션이 끝날 때까지 유지된다. 작업이 없으면 `scalar_one`이 예외를 던진다.
    열린(open) 작업만 수집한다. 수집한 작업·회수한 작업은 호출자가 아무것도 쓰지 않는다.
    """
    status = conn.execute(
        sa.select(review_tasks.c.status)
        .where(review_tasks.c.task_key == task_key)
        .with_for_update()
    ).scalar_one()
    return ReviewTaskStatus(status)


def _seeded_new_id(assignment_id: str, label_id: str) -> str:
    """오류 삽입 과제에서 검수자가 새로 그린 레코드 ID. 배정의 사본 접두사를 붙여 발견 판정을
    그 배정으로 한정한다 (dlp_review.ops.seeding.detected).

    형식: `seed-<배정>-new-<reconcile이 만든 ID의 sha256 앞 16자>` (결정적이라 재수집해도 같다).
    """
    from dlp_review.ops.seeding import seed_prefix  # 순환 import를 피한다

    digest = hashlib.sha256(label_id.encode()).hexdigest()[:16]
    return f"{seed_prefix(assignment_id)}new-{digest}"


def _after_blur_change(
    conn: sa.Connection, task: ReviewTask, session: Session, setup: ReviewSetup, now: datetime
) -> list[ReviewTask]:
    """블러 검수 결과로 운영 블러가 바뀐 뒤 할 일. 회수를 요청한 작업(요청 전 사본)을 돌려준다.

    1. 승인 상태였으면 AUTO_BLURRED로 되돌린다 (다시 승인해야 새로 렌더한다).
    2. 그 스트림의 이미 배포한 라벨링 작업에 회수를 요청한다. 승인 상태와 무관하게 부른다: 이전 승인
       때 만든 작업이 이미 AUTO_BLURRED인 세션에도 남아 있을 수 있다. 블러가 지금 바뀌었으므로
       렌더 해시 없는 구 작업도 낡은 것으로 본다 (legacy).
    3. 이전 블러본 렌더 기록을 무효로 둔다. 저장소 부작용이라 DB 쓰기 뒤에 둔다.
    외부 삭제는 하지 않는다 (커밋 뒤 `dlp_review.revoke` 실행기 몫).
    """
    if session.privacy_state is PrivacyState.APPROVED:
        set_privacy_state(conn, task.session_id, PrivacyState.AUTO_BLURRED)
    requested = request_media_revocation(
        conn, task.session_id, setup.privacy_policy(), now, reason=f"blur_changed:{task.task_key}",
        streams={task.stream_id}, legacy=True,
    )  # fmt: skip
    # 이전 블러본을 어떤 단계도 현재 것으로 보지 않게 렌더 기록을 무효로 둔다
    with tempfile.TemporaryDirectory() as tmp:
        invalidate_render(
            setup.labeling, task.session_id, task.stream_id, f"collected:{task.task_key}", Path(tmp)
        )
    return requested


def _note_revocations(outcome: ReviewOutcome, requested: list[ReviewTask]) -> None:
    """회수를 요청한 작업과 다시 만들 일을 결과에 적는다 (CLI가 출력한다).

    requested는 요청 전 사본이다. 요청 당시 열려 있던 작업(요청 뒤 revoked)만 다시 만들 대상이다.
    배정이 있으면 배정 ID, 없으면 `verify.reissue_slot`(담당자, 스트림, 도구). 수집한 작업은 결과가
    이미 이력에 있어 다시 만들지 않는다 — 다시 만든 작업을 수집한 뒤 블러가 또 바뀐 경우도 같다.
    완료 판정(`verify._direct_reissue`)도 같은 기준이라, 수집한 재작업은 그 뒤 회수 요청을 받아도
    처음 회수한 작업을 다시 만든 것으로 본다.
    """
    for t in requested:
        outcome.revoked.append(t.task_key)
        if t.status is not ReviewTaskStatus.OPEN:
            continue
        if t.assignment_id is not None:
            if t.assignment_id not in outcome.reissue_assignments:
                outcome.reissue_assignments.append(t.assignment_id)
            continue
        direct = reissue_slot(t)
        if direct not in outcome.direct_reissue:
            outcome.direct_reissue.append(direct)


def _label_studio_reviewed(
    conn: sa.Connection,
    task: ReviewTask,
    setup: ReviewSetup,
    reviewer_id: str,
    session: Session,
    sent: list[LabelRecord],
    now: datetime,
) -> tuple[list[ReviewedItem], list[dict[str, Any]]]:
    """Label Studio 작업에서 검수자(reviewer_id) 계정의 최신 주석만 검수 결과로 읽는다 (F-REV-06).

    검수자 계정은 `review.yaml label_studio.users`로 찾는다. 다른 사용자·서비스 계정이
    만들었거나 마지막으로 고친 주석은 결과로 쓰지 않고 경고 로그(주석·사용자 숫자 ID)로만
    남긴다. 검수 시간도 그 계정 주석의 lead_time만 더한다 (A-10, 0보다 크면 `review_work`).

    sent: 작업에 보낸 라벨. 결과 구간 id가 여기 있으면 원래 라벨로 본다 (CVAT `dlp_label_id`와
    같은 기준. 이중 작업도 원래 ID를 지니지만 비교 기준이 빈 목록이라 reconcile이 새 측정 레코드로
    만든다).

    반환: (검수 항목, 그 주석의 결과 원본). 결과 원본은 AI 결정 읽기(`ai_collect.ls_decisions`)에
    쓴다.

    예외: `TaskError` — 클라이언트 없음, 계정 연결 없음·잘못됨, 검수자 계정의 제출 주석 없음(받지
    않은 다른 주석 수를 알린다), 옮길 수 없는 구간(영역 ID·시각·이유).
    """
    ls = setup.label_studio
    if ls is None:
        raise TaskError("Label Studio 클라이언트가 없습니다")
    user = require_label_studio_user(ls, setup, reviewer_id)
    tid = int(task.external_id)
    ignored = ls.ignored_annotations(tid, user=user)
    if ignored:
        logger.warning(
            "%s: 담당자 %s(Label Studio 사용자 %s)가 아닌 주석은 받지 않음: %s",
            task.task_key, reviewer_id, user,
            ", ".join(f"#{a.annotation_id}(만든 {a.completed_by}, 고친 {a.updated_by})"
                      for a in ignored),
        )  # fmt: skip
    results = ls.latest_results(tid, user=user)
    if results is None:
        # 프리라벨은 예측(prediction)으로 보내므로 사람이 제출한 주석이 없으면 결과가 없다
        extra = f" (다른 사용자 주석 {len(ignored)}개는 받지 않음)" if ignored else ""
        raise TaskError(
            f"{task.task_key}: 담당자 {reviewer_id}(Label Studio 사용자 {user})가 제출한 주석이 "
            f"아직 없습니다{extra}"
        )
    try:
        reviewed = from_ls_results(results, {x.label_id for x in sent})
    except LsFormatError as exc:
        # 옮길 수 없는 구간(영역 ID·시각·이유)을 알려 검수자가 고쳐 다시 제출하게 한다 (F-REV-03)
        raise TaskError(f"{task.task_key}: {exc}") from exc
    # 검수 시간 (운영 지표, WP16). CVAT는 작업 시간을 재지 않아 dlp ops log-work로 기록한다
    seconds = ls.lead_seconds(tid, user=user)
    if seconds > 0:
        insert_review_work(
            conn,
            ReviewWork(
                work_id=f"{task.task_key}:work", task_key=task.task_key,
                session_id=task.session_id, reviewer=reviewer_id, stage=task.stage.value,
                seconds=seconds, video_ms=session.duration_ms, source="label_studio",
                recorded_at=now,
            ),
        )  # fmt: skip
    return reviewed, results


def drop_retracted(outcome: ReviewOutcome, labels: list[LabelRecord]) -> list[str]:
    """보낸 뒤 바뀐 라벨에 대한 검수 결과(승인·수정·삭제)를 뺀다.

    - 다른 단계가 지운 라벨: 그것을 parent로 하는 수정 레코드를 쓰면 지운 라벨이 되살아난다.
    - 다른 작업(같은 라벨을 보낸 다른 검수)이 먼저 고친 라벨: 또 고치면 한 라벨에 자식이 둘인
      갈래 이력이 생겨 두 레코드가 모두 현재 라벨이 된다.
    어느 쪽이든 이 라벨은 이미 자식 레코드가 있다. 뺀 원래 라벨 ID를 돌려준다.

    인자:
    - outcome: `reconcile` 결과. 제자리에서 고친다 (`approved`, `new_records`).
    - labels: 세션의 DB 이력 전체 (이번 결과의 새 레코드는 아직 들어 있지 않아야 한다).

    반환: 뺀 원래 라벨 ID (정렬). 새로 그린 레코드(parent 없음)는 빼지 않는다.
    """
    # 이름은 retracted지만 실제로는 "이미 자식 레코드가 있는 라벨" 집합이다 (삭제·수정 모두 포함)
    retracted = {x.parent_label_id for x in labels if x.parent_label_id}
    dropped = sorted(
        {i for i in outcome.approved if i in retracted}
        | {
            r.parent_label_id
            for r in outcome.new_records
            if r.parent_label_id is not None and r.parent_label_id in retracted
        }
    )
    outcome.approved = [i for i in outcome.approved if i not in retracted]
    outcome.new_records = [
        r
        for r in outcome.new_records
        if r.parent_label_id is None or r.parent_label_id not in retracted
    ]
    return dropped


def _outcome_for(
    task: ReviewTask,
    originals: list[LabelRecord],
    all_labels: list[LabelRecord],
    reviewed: list[ReviewedItem],
    *,
    ontology_version: str,
    reviewer_id: str,
    now: datetime,
    normalize: Callable[[LabelRecord], LabelRecord] | None,
) -> ReviewOutcome:
    """검수 항목 → 수집 결과: reconcile → 보낸 뒤 바뀐 라벨 빼기(stale) → 측정·오류 삽입 표시.

    `collect_task`가 도구 결과로, AI 적용을 되돌릴 때(`ai_collect.settle`)는 다른 항목으로 다시
    부른다 (같은 비교 기준·시각이라 같은 항목이면 같은 결과). DB에 쓰지 않는다.

    인자: originals(비교 기준, 블라인드·이중은 빈 목록), all_labels(세션 이력), reviewed(검수 항목),
    나머지는 `reconcile`과 같다.
    """
    outcome = reconcile(
        originals,
        reviewed,
        session_id=task.session_id,
        ontology_version=ontology_version,
        reviewer_id=reviewer_id,
        now=now,
        normalize=normalize,
    )
    # 보낸 뒤 이미 자식이 생긴 라벨에 대한 결과는 버리고 stale로 보고한다
    outcome.stale = drop_retracted(outcome, all_labels)
    if task.mode is ReviewMode.BLIND or task.mode is ReviewMode.DOUBLE:
        # 측정 레코드: current_labels가 운영 라벨에서 빼고, 품질 리포트만 쓴다
        measurement = "blind" if task.mode is ReviewMode.BLIND else "double"
        outcome.new_records = [
            x.model_copy(update={"measurement": measurement}) for x in outcome.new_records
        ]
    elif task.mode is ReviewMode.SEEDED_ERROR:
        # 오류 삽입 과제에서 나온 모든 레코드는 학습에서 빠진다 (새로 그린 것 포함).
        # 새로 그린 레코드는 배정의 사본 접두사를 붙여 그 배정의 결과로만 센다.
        aid = task.assignment_id
        outcome.new_records = [
            x.model_copy(
                update={
                    "seeded_error": True,
                    **(
                        {"label_id": _seeded_new_id(aid, x.label_id)}
                        if aid is not None and x.parent_label_id is None
                        else {}
                    ),
                }
            )
            for x in outcome.new_records
        ]
    return outcome


def collect_task(
    conn: sa.Connection,
    task_key: str,
    setup: ReviewSetup,
    reviewer_id: str,
    now: datetime,
    ops_policy: ReviewOpsPolicy | None = None,
) -> ReviewOutcome | None:
    """수집했으면 결과를, 이미 수집했거나 회수한(revoked) 작업이면 None을 돌려준다.

    검수 결과는 작업에 실제로 보낸 라벨(sent_label_ids)과 비교한다. 그 사이 다른 단계가 라벨을
    바꿨어도 보내지 않은 라벨을 검수자가 지운 것으로 보지 않는다.
    reviewer_id는 작업 담당자와 같아야 한다 (담당자가 있는 작업).
    ops_policy는 배정 마무리(표본 판정)에 쓴다.

    인자:
    - conn: DB 연결 (호출자 트랜잭션).
    - task_key: 작업 키 (`cvat:<id>`, `label_studio:<id>`).
    - setup: 저장소·도구 클라이언트 (`tasks.ReviewSetup`).
    - reviewer_id: 기록할 검수자 (웹훅이면 `webhook.resolve_reviewer` 결과).
    - now: 수집 시각 (시간대 필수). 새 레코드 `created_at`·검수 시각이 된다.

    예외: `TaskError` — 담당자가 아님, 세션 온톨로지 버전 없음, 도구 클라이언트 없음, 옮길 수 없는
    CVAT 주석·Label Studio 구간, 검수자의 Label Studio 계정 연결 없음·잘못됨, 그 계정이 제출한 주석
    없음. 도구 API 오류는 `clients.ToolError`.
    """
    # 같은 부모를 보낸 서로 다른 작업도 직렬화한다. 작업 행만 잠그면 stale 이력으로 둘 다
    # 수정본을 추가할 수 있다. 세션 → 작업 순으로 잠근 뒤에 이력을 읽는다 (ADR 0032).
    lock_session(conn, get_review_task(conn, task_key).session_id)
    # 수집한 작업은 다시 수집하지 않고, 회수한 작업에 라벨러가 낸 결과는 받지 않는다
    # (그 결과는 회수 실행기의 스냅숏에 남는다, ADR 0039)
    if _lock_task(conn, task_key) is not ReviewTaskStatus.OPEN:
        return None
    task = get_review_task(conn, task_key)
    if task.assignee is not None and reviewer_id != task.assignee:
        raise TaskError(
            f"{task_key}: 담당자({task.assignee})가 아닌 {reviewer_id}의 검수 결과는 받지 않습니다"
        )
    session = get_session(conn, task.session_id)
    if session.ontology_version is None:
        raise TaskError(f"{task.session_id}: 세션에 온톨로지 버전이 없습니다")
    # CVAT 작업은 스트림 하나, Label Studio 작업은 세션 단위(시간 라벨)
    stream_filter = task.stream_id if task.tool is ReviewTool.CVAT else None
    assignment = get_assignment(conn, task.assignment_id) if task.assignment_id else None
    all_labels = get_labels(conn, task.session_id)
    # 비교 기준(originals) 정하기: 보낸 라벨 ID 기록 → 배정 선택 함수 → 현재 운영 라벨 순
    if task.sent_label_ids is not None:
        history = {x.label_id: x for x in all_labels}
        originals = [history[i] for i in task.sent_label_ids if i in history]
    elif assignment is not None:  # sent_label_ids 이전에 만든 작업
        from dlp_review.ops.selection import assignment_selector  # 순환 import를 피한다

        originals = assignment_selector(conn, assignment)(stream_filter, task.label_kinds)
    else:
        originals = current_for_review(conn, task.session_id, stream_filter, task.label_kinds)
    sent = originals  # 작업에 보낸 라벨 (AI 노출·결과 기록의 기준, 측정 작업도 같다)
    if task.mode in (ReviewMode.BLIND, ReviewMode.DOUBLE):
        # 측정용: 검수자가 낸 모든 라벨을 새 측정 레코드로 남기고 운영 라벨은 건드리지 않는다
        originals = []
    # AI 1차 검수 (WP17.7): 끔이거나 노출 기록이 없으면 None이고 아래 AI 호출은 아무것도 하지 않는다
    ai = ai_collect.begin(conn, task, setup.ai, sent, all_labels)

    if task.tool is ReviewTool.CVAT:
        if setup.cvat is None:
            raise TaskError("CVAT 클라이언트가 없습니다")
        privacy = task.stage is ReviewStage.PRIVACY
        # 작업 매체가 원본 버킷(블러 검수 프록시)이면 원본 저장소, 아니면 라벨링 버킷에서 받는다
        store = setup.raw if task.media_uri.startswith(setup.raw.uri("")) else setup.labeling
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            video = work / "media.mp4"
            store.get_file(object_key(store, task.media_uri), video)
            # CVAT 프레임 번호 → 화면 영상 PTS ms (프레임 번호는 저장하지 않는다)
            times = frame_times(video)
            scale = UNIT_SCALE
            if privacy:  # 블러 라벨은 원본 화소 좌표, 화면은 축소 프록시
                stream = next(s for s in session.streams if s.stream_id == task.stream_id)
                scale = label_scale(setup.raw, stream, video, work)
        project = setup.cvat.http.get(f"/api/tasks/{task.external_id}").json()["project_id"]
        schema = CvatSchema.from_labels(setup.cvat.project_labels(int(project)))
        try:
            # 모양(Shape) 모드 직사각형도 트랙으로 받는다 (버리면 놓친 얼굴이 블러 없이 남는다)
            tracks = annotation_tracks(
                setup.cvat.get_annotations(int(task.external_id)), len(times), schema
            )
            reviewed = from_cvat_tracks(
                tracks,
                times,
                schema,
                task.stream_id,
                new_box_kind="blur_track" if privacy else "box_track",
                scale=scale,
            )
        except CvatFormatError as exc:
            raise TaskError(f"{task_key}: {exc}") from exc
        # 고치지 않은 라벨의 비교 기준 = 보냈다가 그대로 받은 값 (좌표 변환·반올림 왕복)
        normalize = partial(quantize, scale=scale)
        decisions: dict[str, AiDecision] = {}  # CVAT 1단계에는 AI 결정 칸이 없다 (F-AIR-06)
    else:
        # 원래 라벨 ID 기준은 보낸 라벨이다 (이중 작업도 표준과 같은 화면·입력란 규칙, AI 결정
        # 대응). 이중·블라인드의 비교 기준(originals)은 빈 목록이라 모두 새 측정 레코드가 된다
        reviewed, results = _label_studio_reviewed(
            conn, task, setup, reviewer_id, session, sent, now
        )
        normalize = None
        decisions = ai_collect.ls_decisions(results, ai)

    replay = partial(
        _outcome_for, task, originals, all_labels, ontology_version=session.ontology_version,
        reviewer_id=reviewer_id, now=now, normalize=normalize,
    )  # fmt: skip
    # 사람이 명시적으로 적용한 AI 제안만 반영하고(I6), 세션 검증에 걸리면 AI 적용만 되돌린다
    applied = ai_collect.apply_decisions(ai, reviewed, decisions, normalize=normalize)
    outcome = ai_collect.settle(ai, replay(applied), replay)
    insert_labels(conn, outcome.new_records)
    for label_id in outcome.approved:
        record_review(conn, label_id, VerificationState.HUMAN_APPROVED, reviewer_id, now)
    mark_review_task_collected(conn, task_key, now)
    if task.stage is ReviewStage.PRIVACY and task.mode in OPERATIONAL_MODES and outcome.new_records:
        # 블러가 바뀌었다: 승인 해제 → 배포 매체 회수 요청(같은 커밋) → 렌더 기록 무효
        _note_revocations(outcome, _after_blur_change(conn, task, session, setup, now))
    # AI 판정과 사람 결과의 연결을 작업마다 한 번 남긴다 (배정 마무리 앞, 같은 트랜잭션)
    ai_collect.record_outcome(conn, ai, outcome, reviewer_id, now)
    if assignment is not None:
        # 함수 안에서 불러온다 (모듈을 읽을 때 ops 계층 전체를 끌어오지 않는다).
        # 모든 작업이 수집됐으면 배정을 끝내고, 표본 배정이면 묶음 합격 판정을 적용한다.
        from dlp_review.ops.runner import finish_assignment

        # AI 판정 사용이 켜져 있으면 표본 합격 뒤 늦게 도착한 지적의 보류 라벨을 재검수로 보낸다
        finish_assignment(conn, assignment, now, ops_policy, ai=setup.ai)
    # 설명 라벨은 검수 도구 경로가 없어 행동·상위 구간 결과를 따른다 (승인·표본 검증은 물려받고,
    # 지웠거나 의미를 고친 행동의 검수 전 설명은 지운다. F-PAR-15). 표본 판정 뒤라 표본 검증도 본다
    follow = apply_description_follow(conn, task.session_id, now, InheritanceMechanism.COLLECT)
    outcome.descriptions_approved = [r.label_id for r in follow.approve]
    outcome.descriptions_retracted = [x.label_id for x in follow.retract]
    # 3D 손 궤적은 검수 도구 경로가 없어 그것을 계산한 2D 손 트랙의 검수를 물려받는다 (ADR 0048)
    traj = apply_trajectory_follow(
        conn, task.session_id, now, TrajectoryInheritanceMechanism.COLLECT
    )
    outcome.trajectories_approved = [r.label_id for r in traj.approve]
    outcome.trajectories_lapsed = [x.label_id for x in traj.lapsed]
    return outcome
