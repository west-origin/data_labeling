"""검수 운영 실행: 세션 배정 계획, 배정별 검수 작업 만들기, 배정 마무리(표본 판정), 품질 리포트.

- `dlp review plan <세션> --reviewer …`: 배정(review_assignments)을 만든다 (같은 배정 ID는 건너뜀).
- `dlp review assign <배정>`: 검수 도구 작업을 만든다.
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
- `create_assignment_tasks`: 배정의 도구 작업을 만든다 (멱등).
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
from dlp_review.ops.assign import Loads, PlannedUnit, plan
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit, flag_unit, unit_priority, units_for
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
    create_labeling_tasks,
    create_privacy_tasks,
    lock_assignment,
)
from dlp_schema.db.repository import (
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
    planned: list[PlannedUnit] = []
    previous = list_assignments(conn, session_id=session_id)
    for unit in units_for(session, policy, privacy=privacy):
        labels = unit.select(current)
        generation = ""
        if privacy:
            # 블러 단위는 탐지가 0개여도 사람이 영상 전체를 봐야 승인할 수 있다 (ADR 0023 3번).
            # 배정 ID에 세대를 붙여 재탐지·승인 취소 뒤 다시 계획하면 새 배정이 생긴다
            generation = privacy_generation(
                history,
                unit.stream_id or "",
                latest_privacy_detection(conn, session_id, unit.stream_id or ""),
            )
        elif not labels:
            continue  # 라벨이 없는 작업 라벨 단위는 검수할 것이 없다
        else:
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
                continue
            pending = any(
                x.provenance.source is Source.MODEL
                and x.verification.state is VerificationState.UNREVIEWED
                for x in labels
            )
            if operational and not pending:
                continue
            generation = (
                "g"
                + hashlib.sha256("|".join(sorted(x.label_id for x in labels)).encode()).hexdigest()[
                    :8
                ]
            )
        flagged = flag_unit(labels, policy, known_classes=known, glove_session=glove)
        sample: tuple[str, ...] = ()
        withheld: tuple[str, ...] = ()
        if not privacy:  # 블러는 종료 조건 전까지 전수 검수한다
            by_id = {x.label_id: x for x in labels}
            for lot in lots(labels, policy.sampling):
                picked = draw_sample(lot, policy.sampling, seed)
                sample += picked
                withheld += tuple(i for i in lot.label_ids if i not in picked)
                # 표본 라벨 구간도 먼저 볼 구간으로 표시한다 (sample 사유, 가중치 없음)
                flagged += [
                    FlaggedSpan(
                        reason=ReviewReason.SAMPLE, t_start_ms=by_id[i].t_start_ms,
                        t_end_ms=by_id[i].t_end_ms, label_ids=(i,),
                    )
                    for i in picked
                ]  # fmt: skip
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
    부작용: 작업 생성(`tasks` 함수의 부작용 전부), 배정의 `task_key` 갱신(첫 작업).
    예외: 블러 배정의 담당자가 권한자가 아니면 `AccessError`, 그 밖은 `tasks`의 예외.
    """
    # 수집과 일관된 세션 → 배정 순서. 프라이버시 작업 생성도 세션을 잠근다.
    lock_session(conn, a.session_id)
    existing = lock_assignment(conn, a.assignment_id)
    if existing is not None:
        return existing
    # 수집 때도 같은 선택 함수를 쓴다 (collect가 sent_label_ids가 없는 예전 작업에 쓴다)
    select = assignment_selector(conn, a)
    assignee = a.assignee or "unassigned"
    if a.label_kinds == ("blur_track",):
        check_privacy_reviewers([a.assignee], policy)  # 블러 검수는 원본 영상을 연다
        assert a.assignee is not None
        tasks = create_privacy_tasks(
            conn, a.session_id, setup, now, assignee=a.assignee,
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
