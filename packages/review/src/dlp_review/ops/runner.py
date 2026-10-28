"""검수 운영 실행: 세션 배정 계획, 배정별 검수 작업 만들기, 배정 마무리(표본 판정), 품질 리포트.

- `dlp review plan <세션> --reviewer …`: 배정(review_assignments)을 만든다 (같은 배정 ID는 건너뜀).
- `dlp review assign <배정> [--assignee <검수자>]`: 검수 도구 작업을 만든다. 담당자 없는 배정(선임이
  없어 담당자를 비워 둔 QA 배정 등)은 `--assignee`로 담당자를 정해야 만든다 (F-REV-07).
- 결과 수집(`dlp review collect`)이 끝나면 배정을 마무리한다. 표본 검수 배정이면 묶음 합격 여부에
  따라 나머지를 표본 검증으로 두거나 재검수 배정을 만든다.

WP12, ADR 0014·0015·0023(블러 배정·세대). 이 모듈은 순수 로직(`priority`, `sampling`, `assign`,
`seeding`, `measure`)을 DB·저장소·도구와 엮는다. 트랜잭션은 모두 호출자(CLI)가 연다.

공개 이름:
- `VERIFIED`, `is_verified`: "정답으로 볼 수 있는" 라벨 판정 (사람 출처 또는 사람 승인·수정).
- `known_classes`: 다른 세션에서 사람이 확인한 객체 클래스 (new_object 사유).
- `seed_pool`: 오류 삽입 원천 단위 (정답을 아는 단위).
- `check_privacy_reviewers`: 정책의 원본 접근 권한자로 블러 담당자 검사.
- `plan_session`: 세션 하나의 배정 계획을 세우고 DB에 넣는다.
- `privacy_generation`: 블러 단위 입력 세대 해시.
- `assign_reviewer`, `ReviewerRuleError`: 담당자 없는 배정에 담당자를 정한다 (작업을 만들기 전에만).
  배정 규칙(QA 선임, 짝 배정과 다른 사람) 거부는 `ReviewerRuleError`다.
- `create_assignment_tasks`: 배정의 도구 작업을 만든다 (멱등, 담당자 없는 배정은 거부).
- `FinishResult`, `finish_assignment`: 배정 마무리와 표본 판정.
- `QualityReport`, `quality_report`: 발견율·이중 일치도·프리라벨 편향.
- `shown_prelabels`: 표준 검수자에게 실제로 보낸 모델 프리라벨 (블라인드 편향의 기준).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa

from dlp_privacy.runner import last_detection, operational_blur
from dlp_review import roles
from dlp_review.ai_display import AiDisplayConfig, ai_flagged_spans, escalated_ids, load_ai_view
from dlp_review.ops.assign import Loads, PlannedUnit, plan
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit, flag_unit, merge_spans, unit_priority, units_for
from dlp_review.ops.results import (
    FinishResult as FinishResult,
)
from dlp_review.ops.results import (
    QualityReport as QualityReport,
)
from dlp_review.ops.results import (
    finish_assignment as finish_assignment,
)
from dlp_review.ops.results import (
    quality_report as quality_report,
)
from dlp_review.ops.results import (
    shown_prelabels as shown_prelabels,
)
from dlp_review.ops.sampling import draw_sample, lots
from dlp_review.ops.seeding import seed_labels
from dlp_review.ops.selection import assignment_selector
from dlp_review.roles import AccessError as AccessError  # 이전 위치에서 쓰던 이름 (재수출)
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    create_labeling_tasks,
    create_privacy_tasks,
    lock_assignment,
)
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_session,
    insert_assignment,
    insert_labels,
    latest_privacy_detection,
    list_assignments,
    lock_session,
    update_assignment,
)
from dlp_schema.db.review_queries import (
    existing_assignment_ids,
    open_assignment_counts,
    verified_object_classes,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    LabelRecord,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import (
    AssignmentStatus,
    FlaggedSpan,
    ReviewAssignment,
    ReviewMode,
    ReviewReason,
    ReviewTask,
    ReviewTool,
)
from dlp_schema.session import PrivacyDetectionRun, StreamKind

# 사람이 직접 확인한 검증 상태. 표본 검증(sample_verified)은 사람이 그 라벨을 보지 않았으므로 넣지
# 않는다 (`verify.VERIFIED_STATES`와 다르다: 그쪽은 세션 완료 판정용이라 표본 검증도 인정한다).
VERIFIED = {VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED}


def is_verified(x: LabelRecord) -> bool:
    """사람이 만들었거나 사람이 승인·수정한 라벨인가 (정답으로 쓸 수 있는가)."""
    return x.provenance.source is Source.HUMAN or x.verification.state in VERIFIED


def known_classes(conn: sa.Connection, exclude_session: str) -> set[str]:
    """다른 세션에서 사람이 확인한 객체 클래스.

    DB에서 현재 운영·사람 확인 클래스만 집계한다 (이력과 기하 페이로드 전체를 읽지 않는다).
    """
    return verified_object_classes(conn, exclude_session)


def seed_pool(
    conn: sa.Connection,
    session_ids: Sequence[str],
    policy: ReviewOpsPolicy,
    groups: set[str] | None = None,
) -> list[Unit]:
    """정답을 아는 단위: 운영 라벨이 있고 모두 사람이 만들었거나 승인·수정한 단위.

    groups로 단위 묶음(spatial, temporal, privacy)을 좁힌다.

    인자: session_ids(후보 세션, 보통 골든셋 세션), policy(단위 종류 정의).
    반환: 조건을 만족하는 단위 목록 (세션 순서 → 작업 라벨 단위 → 블러 단위 순).
    """
    pool: list[Unit] = []
    for sid in session_ids:
        session = get_session(conn, sid)
        current = current_labels(get_labels(conn, sid))
        for unit in [*units_for(session, policy), *units_for(session, policy, privacy=True)]:
            labels = unit.select(current)
            if groups is not None and unit.group not in groups:
                continue
            if labels and all(is_verified(x) for x in labels):
                pool.append(unit)
    return pool


def check_privacy_reviewers(reviewers: Sequence[str | None], policy: ReviewOpsPolicy) -> None:
    """블러 담당자가 모두 `policy.reviewers.privacy`(원본 접근 권한자)인지 검사한다.

    예외: `AccessError` (`roles.check_privacy_reviewers`).
    """
    roles.check_privacy_reviewers(reviewers, policy.reviewers.privacy)


def _open_loads(conn: sa.Connection, reviewers: Sequence[str]) -> Loads:
    """모든 세션의 열린 배정 수로 담당자 부하를 초기화한다 (부하가 적은 사람부터 배정)."""
    return Loads(reviewers, Counter(open_assignment_counts(conn, reviewers)))


def _unit_generation(
    labels: Sequence[LabelRecord], unit: Unit, previous: Sequence[ReviewAssignment], session_id: str
) -> str | None:
    """작업 라벨 단위의 입력 세대 (계획하지 않을 단위면 None).

    라벨이 없거나, 열린 운영 배정이 있거나, 끝난 운영 배정 뒤 새 미검수 모델 입력이 없으면 None.
    """
    if not labels:
        return None  # 라벨이 없는 작업 라벨 단위는 검수할 것이 없다
    operational = [
        a
        for a in previous
        if a.session_id == session_id
        and a.stream_id == unit.stream_id
        and a.label_kinds == unit.kinds
        and a.mode in (ReviewMode.STANDARD, ReviewMode.QA)
    ]
    # 열린 운영 배정이 있으면 먼저 끝낸다. 완료 뒤 새 미검수 모델 입력이 생기면 새 세대다.
    # 이미 검수한 결과만 남았을 때는 human 수정본 때문에 끝없이 배정을 만들지 않는다.
    if any(a.status is AssignmentStatus.OPEN for a in operational):
        return None
    pending = any(
        x.provenance.source is Source.MODEL and x.verification.state is VerificationState.UNREVIEWED
        for x in labels
    )
    if operational and not pending:
        return None
    digest = hashlib.sha256("|".join(sorted(x.label_id for x in labels)).encode()).hexdigest()
    return "g" + digest[:8]


def _sample(
    labels: Sequence[LabelRecord], policy: ReviewOpsPolicy, seed: int, exclude: set[str]
) -> tuple[tuple[str, ...], tuple[str, ...], list[FlaggedSpan]]:
    """표본 검수: (표본 ID, 보류 ID, 표본 사유 구간). exclude(AI 지적)는 묶음에 넣지 않는다.

    exclude 라벨은 표본·보류 어디에도 없으므로 작업에 반드시 들어간다. 마무리 판정은 그중 결함이
    하나라도 있으면 묶음을 불합격으로 본다 (`results._ai_excluded`, `sampling.judge`의 forced).
    그래야 AI가 맞게 지적할수록 남은 묶음이 합격하기 쉬워지지 않는다 (AI는 검수를 늘리기만 한다,
    I9).
    """
    by_id = {x.label_id: x for x in labels}
    sample: tuple[str, ...] = ()
    withheld: tuple[str, ...] = ()
    spans: list[FlaggedSpan] = []
    for lot in lots([x for x in labels if x.label_id not in exclude], policy.sampling):
        picked = draw_sample(lot, policy.sampling, seed)
        sample += picked
        withheld += tuple(i for i in lot.label_ids if i not in picked)
        # 표본 라벨 구간도 먼저 볼 구간으로 표시한다 (sample 사유, 가중치 없음)
        spans += [
            FlaggedSpan(
                reason=ReviewReason.SAMPLE, t_start_ms=by_id[i].t_start_ms,
                t_end_ms=by_id[i].t_end_ms, label_ids=(i,),
            )
            for i in picked
        ]  # fmt: skip
    return sample, withheld, spans


def plan_session(
    conn: sa.Connection,
    session_id: str,
    reviewers: Sequence[str],
    policy: ReviewOpsPolicy,
    ontology: Ontology,
    *,
    seed: int,
    now: datetime,
    seed_sessions: Sequence[str] = (),
    seed_groups: set[str] | None = None,
    privacy: bool = False,
    ai: AiDisplayConfig | None = None,
) -> list[ReviewAssignment]:
    """세션의 검수 배정을 계획하고 DB(`review_assignments`)에 넣는다. 새로 넣은 배정을 돌려준다.

    인자:
    - reviewers: 담당자 후보. privacy면 모두 원본 접근 권한자여야 한다.
    - policy: 검수 운영 정책.
    - ontology: 오류 삽입 class_swap 후보.
    - seed: 표본·배정 뽑기 시드 (같은 seed면 같은 계획 → 재실행 멱등).
    - now: 배정·오류 삽입 사본 생성 시각.
    - seed_sessions: 오류 삽입 원천 세션 (골든셋 등). 비면 오류 삽입 배정이 없다.
    - seed_groups: 오류 삽입 원천 단위 묶음을 더 좁힌다.
    - privacy: True면 블러 단위(영상 스트림별)만, False면 작업 라벨 단위만 계획한다.
    - ai: AI 1차 검수 판정 사용 설정 (`ReviewSetup.ai`, WP17.6). 있으면 작업 라벨 계획에서 세션
      잠금 뒤 증가 프로필(렌더 무관)의 유효한 판정으로 우선순위 사유(AI_WRONG·AI_SUSPECT·
      AI_UNSURE, 가중치 `priority.weights`)를 더하고, `ai_review.sampling_exclude` 판정의 라벨을
      표본 묶음에서 뺀다 (반드시 검수). `ai_review.disagreement`를 켜면 판정단이 갈린 라벨에
      AI_DISAGREEMENT 사유를 더하고 표본에서 뺀다 (ADR 0063). 블러 계획에는 쓰지 않는다 (I2).

    반환: 이번에 삽입한 배정 (이미 같은 ID가 있으면 건너뛰므로 다시 부르면 빈 목록).
    예외: privacy이고 권한자가 아닌 후보가 있으면 `AccessError`.
    부작용: `review_assignments` 삽입, 오류 삽입 배정이면 사본 라벨을 `label_records`에 삽입.
    """
    if privacy:
        check_privacy_reviewers(reviewers, policy)
    lock_session(conn, session_id)
    session = get_session(conn, session_id)
    history = get_labels(conn, session_id)
    current = current_labels(history)
    known = known_classes(conn, session_id)
    glove = any(s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams)
    view = load_ai_view(conn, session_id, current, None, ai) if ai and not privacy else None
    exclude = escalated_ids(view.escalate, ai.policy, view.disagree) if view and ai else set[str]()
    planned: list[PlannedUnit] = []
    previous = list_assignments(conn, session_id=session_id)
    for unit in units_for(session, policy, privacy=privacy):
        labels = unit.select(current)
        if privacy:
            # 블러 단위는 탐지가 0개여도 사람이 영상 전체를 봐야 승인할 수 있다 (ADR 0023 3번).
            # 배정 ID에 세대를 붙여 재탐지·승인 취소 뒤 다시 계획하면 새 배정이 생긴다
            run = latest_privacy_detection(conn, session_id, unit.stream_id or "")
            generation = privacy_generation(history, unit.stream_id or "", run)
        else:
            gen = _unit_generation(labels, unit, previous, session_id)
            if gen is None:
                continue
            generation = gen
        flagged = flag_unit(labels, policy, known_classes=known, glove_session=glove)
        if view is not None and ai is not None:
            ai_spans = ai_flagged_spans(view.escalate, labels, ai.policy, view.disagree)
            flagged = merge_spans([*flagged, *ai_spans])
        sample: tuple[str, ...] = ()
        withheld: tuple[str, ...] = ()
        if not privacy:  # 블러는 종료 조건 전까지 전수 검수한다
            sample, withheld, spans = _sample(labels, policy, seed, exclude)
            flagged += spans
        planned.append(
            PlannedUnit(
                unit, tuple(flagged), unit_priority(flagged, policy), sample, withheld, generation
            )
        )
    # 블러 단위(원본 영상)는 블러 계획에서만, 작업 라벨 단위는 작업 라벨 계획에서만
    # 오류 삽입 원천이 된다
    groups = {"privacy"} if privacy else {"spatial", "temporal"}
    if seed_groups is not None:
        groups &= seed_groups
    pool = seed_pool(conn, seed_sessions, policy, groups) if seed_sessions else []
    candidates = plan(
        planned,
        reviewers,
        policy,
        seed=seed,
        now=now,
        seed_pool=pool,
        loads=_open_loads(conn, reviewers),
    )
    # 오류 삽입 배정은 다른 세션의 원천 단위에 속한다. 후보 ID만 일괄 확인해 멱등을 보존한다.
    existing = existing_assignment_ids(conn, [a.assignment_id for a in candidates])
    created: list[ReviewAssignment] = []
    for a in candidates:
        if a.assignment_id in existing:
            continue
        if a.mode is ReviewMode.SEEDED_ERROR:
            # 오류를 넣은 사본을 지금 만들어 DB에 넣고, 넣은 오류를 배정에 기록한다
            a = _prepare_seeded(conn, a, policy, ontology, seed, now)
        insert_assignment(conn, a)
        created.append(a)
    return created


def privacy_generation(
    history: Sequence[LabelRecord], stream_id: str, run: PrivacyDetectionRun | None = None
) -> str:
    """블러 단위 세대: 완료 기록 ID(0건 포함)·탐지 시각·현재 운영 블러 라벨 집합 해시.

    반환: `"g" + sha256 앞 8자`. 재탐지하거나 블러 라벨 집합이 바뀌면 값이 바뀐다.
    """
    since = last_detection(history, stream_id, run.at if run else None)
    ids = sorted(x.label_id for x in operational_blur(history, stream_id))
    key = "|".join([str(run.event_id) if run else "-", since.isoformat() if since else "-", *ids])
    return "g" + hashlib.sha256(key.encode()).hexdigest()[:8]


def _prepare_seeded(
    conn: sa.Connection, a: ReviewAssignment, policy: ReviewOpsPolicy, ontology: Ontology,
    seed: int, now: datetime,
) -> ReviewAssignment:  # fmt: skip
    """오류 삽입 배정의 사본 라벨을 만들어 DB에 넣고, `injected`를 채운 배정을 돌려준다.

    원천 단위는 배정의 (세션, 스트림, 종류)에서 되살린다: 블러 종류면 privacy, 스트림이 있으면
    spatial, 없으면 temporal.
    부작용: `label_records`에 사본(`seeded_error=True`) 삽입.
    """
    group = (
        "privacy" if a.label_kinds == ("blur_track",) else "spatial" if a.stream_id else "temporal"
    )
    unit = Unit(a.session_id, a.stream_id, group, a.label_kinds)
    truth = unit.select(current_labels(get_labels(conn, a.session_id)))
    task = seed_labels(
        truth,
        assignment_id=a.assignment_id,
        ontology=ontology,
        policy=policy.seeding,
        seed=seed,
        now=now,
    )
    insert_labels(conn, task.labels)
    return a.model_copy(update={"injected": tuple(task.injected)})


class ReviewerRuleError(TaskError):
    """정한 담당자가 배정 규칙(QA는 선임, 짝 배정과 다른 사람)에 맞지 않는다 (`assign_reviewer`).

    정책(`reviewers.senior`)을 고치거나 다른 담당자를 고르면 풀린다. 배정 상태 때문인 거부(이미
    담당자·작업이 있음)는 그냥 `TaskError`라 구분한다 (CLI는 이 예외에만 정책 안내를 붙인다).
    """


def assign_reviewer(
    conn: sa.Connection, a: ReviewAssignment, reviewer: str, policy: ReviewOpsPolicy
) -> ReviewAssignment:
    """담당자 없는 배정에 담당자를 정하고, 담당자를 넣은 배정을 돌려준다 (F-REV-07).

    담당자는 작업의 워터마크·라벨링 버킷 경로·검수자 ID가 되므로 작업을 만들기 전에만 정한다.
    계획(`plan_session`·`plan_qa`)이 지키는 배정 규칙을 사람이 정할 때도 지킨다.
    - 블러 배정: 원본 접근 권한자(`reviewers.privacy`)만 (`AccessError`).
    - QA 배정: 선임(`reviewers.senior`)만.
    - 짝이 있는 배정(블라인드·이중·QA): 짝 배정(표준)과 같은 짝의 다른 측정 배정 담당자는 안 된다
      (같은 사람이 자기 검수를 측정·재검수하지 않는다).
    같은 담당자를 다시 정하면 그대로 돌려준다 (멱등).

    세션 → 배정 순으로 잠근다 (`create_assignment_tasks`와 같은 순서).
    부작용: `review_assignments.assignee` 갱신.
    예외: 이미 다른 담당자가 있거나 작업이 있는 배정이면 `TaskError`, 선임이 아닌 QA 담당자·짝
    배정과 같은 담당자면 `ReviewerRuleError`(`TaskError`의 하위 클래스). 블러 배정의 권한자가
    아니면 `AccessError`.
    """
    lock_session(conn, a.session_id)
    live = lock_assignment(conn, a.assignment_id)
    current = get_assignment(conn, a.assignment_id)
    if current.assignee == reviewer:
        return current
    if current.assignee is not None:
        raise TaskError(
            f"{a.assignment_id}: 이미 담당자({current.assignee})가 있는 배정은 바꾸지 않습니다"
        )
    if live is not None:
        raise TaskError(f"{a.assignment_id}: 이미 작업이 있는 배정의 담당자는 정하지 않습니다")
    if current.label_kinds == ("blur_track",):
        check_privacy_reviewers([reviewer], policy)
    if current.mode is ReviewMode.QA and reviewer not in policy.reviewers.senior:
        raise ReviewerRuleError(
            f"{a.assignment_id}: QA 배정은 선임 검수자(검수 정책 reviewers.senior)에게만 줍니다: "
            f"{reviewer}"
        )
    if current.pair_id is not None:
        paired = {
            x.assignee
            for x in list_assignments(conn, session_id=current.session_id)
            if x.assignment_id != current.assignment_id
            and current.pair_id in (x.assignment_id, x.pair_id)
        }
        if reviewer in paired:
            raise ReviewerRuleError(
                f"{a.assignment_id}: 짝 배정({current.pair_id})과 같은 담당자 {reviewer}에게는 "
                "줄 수 없습니다 (측정·재검수는 다른 사람이 한다)"
            )
    update_assignment(conn, current.assignment_id, assignee=reviewer)
    return current.model_copy(update={"assignee": reviewer})


def create_assignment_tasks(
    conn: sa.Connection,
    a: ReviewAssignment,
    setup: ReviewSetup,
    policy: ReviewOpsPolicy,
    now: datetime,
) -> list[ReviewTask]:
    """배정의 검수 도구 작업을 만든다. DB에 연결된 기존 작업이 있으면 그대로 돌려준다.

    세션·배정 행을 잠가(SELECT … FOR UPDATE) 정상 커밋되는 동시 실행을 직렬화한다.
    DB 롤백은 이미 만든 외부 도구 작업을 삭제하지 않는다. 연결 기록 저장 전에 실패하면
    재시도로 외부 작업이 중복 생성될 수 있어 도구 쪽 잔여 작업을 확인해야 한다.

    도구 선택: 블러 배정 → CVAT 프라이버시 작업, 스트림 있는 작업 라벨 배정 → CVAT,
    세션 단위(시간 라벨) 배정 → Label Studio (세션의 기준 스트림 영상으로 연다).
    담당자 없는 배정은 만들지 않는다: 담당자가 워터마크·라벨링 버킷 경로·검수자 ID가 되므로 자리
    표시 문자열로 대신하지 않는다 (F-REV-07, 회귀: 예전에는 "unassigned"를 실제 담당자로 넣었다).
    먼저 `assign_reviewer`(`dlp review assign <배정> --assignee <검수자>`)로 정한다.
    부작용: 작업 생성(`tasks` 함수의 부작용 전부), 배정의 `task_key` 갱신(첫 작업).
    예외: 담당자 없는 배정이면 `TaskError`, 블러 배정의 담당자가 권한자가 아니면 `AccessError`,
    그 밖은 `tasks`의 예외.
    """
    # 수집과 일관된 세션 → 배정 순서. 프라이버시 작업 생성도 세션을 잠근다.
    lock_session(conn, a.session_id)
    existing = lock_assignment(conn, a.assignment_id)
    if existing is not None:
        return existing
    assignee = a.assignee
    if assignee is None:
        raise TaskError(
            f"{a.assignment_id}: 담당자가 없는 배정은 작업을 만들지 않습니다 "
            f"(dlp review assign {a.assignment_id} --assignee <검수자>로 담당자를 정한다)"
        )
    # 수집 때도 같은 선택 함수를 쓴다 (collect가 sent_label_ids가 없는 예전 작업에 쓴다)
    select = assignment_selector(conn, a)
    if a.label_kinds == ("blur_track",):
        check_privacy_reviewers([assignee], policy)  # 블러 검수는 원본 영상을 연다
        tasks = create_privacy_tasks(
            conn, a.session_id, setup, now, assignee=assignee,
            privacy_reviewers=policy.reviewers.privacy, select=select, mode=a.mode,
            assignment_id=a.assignment_id, streams={a.stream_id} if a.stream_id else None,
        )  # fmt: skip
    else:
        tool = ReviewTool.LABEL_STUDIO if a.stream_id is None else ReviewTool.CVAT
        # 시간 라벨 단위는 세션 하나에 작업 하나 (기준 바디캠 영상으로 연다)
        stream = a.stream_id or get_session(conn, a.session_id).reference_stream.stream_id
        tasks = create_labeling_tasks(
            conn, a.session_id, setup, assignee, now, select=select, mode=a.mode,
            assignment_id=a.assignment_id, tools={tool}, streams={stream},
        )  # fmt: skip
    if tasks:
        update_assignment(conn, a.assignment_id, task_key=tasks[0].task_key)
    return tasks
