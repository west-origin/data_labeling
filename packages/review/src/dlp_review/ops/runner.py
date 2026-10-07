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
from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_privacy.runner import last_detection, operational_blur
from dlp_review import roles
from dlp_review.ops.assign import Loads, PlannedUnit, plan
from dlp_review.ops.measure import Agreement, DetectionRate, agreement, as_items, prelabel_bias
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit, flag_unit, unit_priority, units_for
from dlp_review.ops.sampling import draw_sample, judge, lots
from dlp_review.ops.seeding import detected, seed_labels
from dlp_review.ops.selection import assignment_selector
from dlp_review.roles import AccessError as AccessError  # 이전 위치에서 쓰던 이름 (재수출)
from dlp_review.tasks import (
    ReviewSetup,
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
    list_assignments,
    list_review_tasks,
    list_session_ids,
    record_review,
    update_assignment,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxTrackPayload,
    LabelRecord,
    MaskTrackPayload,
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
    ReviewTaskStatus,
    ReviewTool,
)
from dlp_schema.session import StreamKind

# 사람이 직접 확인한 검증 상태. 표본 검증(sample_verified)은 사람이 그 라벨을 보지 않았으므로 넣지
# 않는다 (`verify.VERIFIED_STATES`와 다르다: 그쪽은 세션 완료 판정용이라 표본 검증도 인정한다).
VERIFIED = {VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED}


def is_verified(x: LabelRecord) -> bool:
    """사람이 만들었거나 사람이 승인·수정한 라벨인가 (정답으로 쓸 수 있는가)."""
    return x.provenance.source is Source.HUMAN or x.verification.state in VERIFIED


def known_classes(conn: sa.Connection, exclude_session: str) -> set[str]:
    """다른 세션에서 사람이 확인한 객체 클래스.

    모든 세션(exclude_session 제외)의 현재 박스·마스크 라벨을 읽는다. 세션 수에 비례해 느려진다.
    """
    out: set[str] = set()
    for sid in list_session_ids(conn):
        if sid == exclude_session:
            continue
        for x in current_labels(get_labels(conn, sid, kinds=["box_track", "mask_track"])):
            if is_verified(x) and isinstance(x.payload, BoxTrackPayload | MaskTrackPayload):
                out.add(x.payload.class_id)
    return out


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
    counts = Counter(
        a.assignee for a in list_assignments(conn, status=AssignmentStatus.OPEN) if a.assignee
    )
    return Loads(reviewers, counts)


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
    session = get_session(conn, session_id)
    history = get_labels(conn, session_id)
    current = current_labels(history)
    known = known_classes(conn, session_id)
    glove = any(s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams)
    planned: list[PlannedUnit] = []
    for unit in units_for(session, policy, privacy=privacy):
        labels = unit.select(current)
        generation = ""
        if privacy:
            # 블러 단위는 탐지가 0개여도 사람이 영상 전체를 봐야 승인할 수 있다 (ADR 0023 3번).
            # 배정 ID에 세대를 붙여 재탐지·승인 취소 뒤 다시 계획하면 새 배정이 생긴다
            generation = privacy_generation(history, unit.stream_id or "")
        elif not labels:
            continue  # 라벨이 없는 작업 라벨 단위는 검수할 것이 없다
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
    # 재실행 멱등: 이미 있는 배정 ID는 건너뛴다 (모든 세션의 배정을 읽는다)
    existing = {a.assignment_id for a in list_assignments(conn)}
    created: list[ReviewAssignment] = []
    for a in plan(
        planned,
        reviewers,
        policy,
        seed=seed,
        now=now,
        seed_pool=pool,
        loads=_open_loads(conn, reviewers),
    ):
        if a.assignment_id in existing:
            continue
        if a.mode is ReviewMode.SEEDED_ERROR:
            # 오류를 넣은 사본을 지금 만들어 DB에 넣고, 넣은 오류를 배정에 기록한다
            a = _prepare_seeded(conn, a, policy, ontology, seed, now)
        insert_assignment(conn, a)
        created.append(a)
    return created


def privacy_generation(history: Sequence[LabelRecord], stream_id: str) -> str:
    """블러 단위 입력 세대: 마지막 자동 탐지 시각 + 현재 운영 블러 라벨 집합의 짧은 해시.

    반환: `"g" + sha256 앞 8자`. 재탐지하거나 블러 라벨 집합이 바뀌면 값이 바뀐다.
    """
    since = last_detection(history, stream_id)
    ids = sorted(x.label_id for x in operational_blur(history, stream_id))
    key = "|".join([since.isoformat() if since else "-", *ids])
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
    """배정의 검수 도구 작업을 만든다. 이미 만들었으면 그 작업을 그대로 돌려준다 (멱등).

    배정 행을 잠가(SELECT … FOR UPDATE) 동시에 실행해도 작업이 두 번 생기지 않는다.

    도구 선택: 블러 배정 → CVAT 프라이버시 작업, 스트림 있는 작업 라벨 배정 → CVAT,
    세션 단위(시간 라벨) 배정 → Label Studio (세션의 기준 스트림 영상으로 연다).
    부작용: 작업 생성(`tasks` 함수의 부작용 전부), 배정의 `task_key` 갱신(첫 작업).
    예외: 블러 배정의 담당자가 권한자가 아니면 `AccessError`, 그 밖은 `tasks`의 예외.
    """
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


@dataclass
class FinishResult:
    """배정 마무리 결과."""

    # 표본 판정 결과 (표본이 없거나 아직 마무리 전이면 None)
    sample_accepted: bool | None = None
    # 표본 합격으로 sample_verified 기록한 라벨 수
    sample_verified: int = 0
    # 표본 불합격으로 만든 재검수 배정 ID
    resample_assignment: str | None = None


def finish_assignment(
    conn: sa.Connection,
    a: ReviewAssignment,
    now: datetime,
    policy: ReviewOpsPolicy | None = None,
) -> FinishResult:
    """배정의 검수 작업이 모두 수집되면 마무리한다 (한 번만).

    표본이 있으면 묶음 합격 판정을 적용한다.

    policy가 없으면 저장소의 정책을 읽는다 (CLI·웹훅 서버는 넘겨 준다).

    부작용: 배정 상태 done 갱신, 합격이면 나머지 라벨에 `sample_verified` 기록(검수자 ID
    `sampling:<배정>`), 불합격이면 재검수 배정(`<배정>:resample`, 보류 라벨만, 우선순위 +1) 삽입.
    배정의 작업이 하나도 없으면 아무것도 하지 않는다 (done이 아니다).
    """
    tasks = [t for t in list_review_tasks(conn, a.session_id) if t.assignment_id == a.assignment_id]
    if not tasks:
        # 회귀: any([])가 거짓이라 작업이 없는(아직 `dlp review assign` 전인) 배정이 사람 검수 없이
        # done이 되고, 표본 배정이면 판정까지 돌았다. 작업을 만들고 수집해야만 마무리한다.
        return FinishResult()
    if any(t.status is not ReviewTaskStatus.COLLECTED for t in tasks):
        return FinishResult()  # 아직 남은 작업이 있다
    # 인자 a는 오래된 사본일 수 있어 DB에서 상태를 다시 읽는다
    if get_assignment(conn, a.assignment_id).status is AssignmentStatus.DONE:
        return FinishResult()  # 이미 마무리했다
    update_assignment(conn, a.assignment_id, status=AssignmentStatus.DONE, completed_at=now)
    result = FinishResult()
    if not a.sample_label_ids:
        return result
    if policy is None:
        from dlp_review.ops.policy import load_policy
        from dlp_schema import repo_root

        policy = load_policy(repo_root())
    labels = get_labels(conn, a.session_id)
    verdict = judge(
        (*a.sample_label_ids, *a.withheld_label_ids), a.sample_label_ids, labels, policy.sampling
    )
    result.sample_accepted = verdict.accepted
    if verdict.accepted:
        for label_id in verdict.to_verify:
            record_review(
                conn,
                label_id,
                VerificationState.SAMPLE_VERIFIED,
                f"sampling:{a.assignment_id}",
                now,
            )
        result.sample_verified = len(verdict.to_verify)
    elif verdict.accepted is False and a.withheld_label_ids:
        # 불합격: 보류했던 라벨만 다시 보는 재검수 배정 (표본 사유 구간은 뺀다)
        follow = a.model_copy(
            update={
                "assignment_id": f"{a.assignment_id}:resample",
                "flagged": tuple(s for s in a.flagged if s.reason is not ReviewReason.SAMPLE),
                "sample_label_ids": (), "withheld_label_ids": (),
                "only_label_ids": a.withheld_label_ids, "task_key": None,
                "status": AssignmentStatus.OPEN, "created_at": now, "completed_at": None,
                "priority": a.priority + 1.0,
            }
        )  # fmt: skip
        insert_assignment(conn, follow)
        result.resample_assignment = follow.assignment_id
    return result


@dataclass
class QualityReport:
    """검수 품질 리포트 (`dlp review quality`)."""

    # 검수자별 오류 삽입 발견율 (검수자 이름순)
    detection: list[DetectionRate] = field(default_factory=list[DetectionRate])
    double: dict[str, Agreement] = field(default_factory=dict[str, Agreement])  # 배정 → 일치도
    blind_bias: dict[str, float] = field(default_factory=dict[str, float])  # 배정 → 프리라벨 편향


def shown_prelabels(
    unit: Sequence[LabelRecord], pair_tasks: Sequence[ReviewTask]
) -> list[LabelRecord]:
    """표준(짝) 배정의 검수자에게 실제로 보낸 모델 프리라벨.

    회귀: 프리라벨 편향을 이력 전체의 모델 레코드(지워진 레코드·이전 모델 버전 포함)로 재서,
    검수자가 본 적 없는 프리라벨까지 기준에 들어가 편향이 틀렸다.

    작업마다:
    - `sent_label_ids`가 있으면 그 ID의 레코드 (작업에 실제로 보낸 라벨).
    - 없으면(예전 작업) 작업 created_at 시점의 운영 현재 라벨 (그 시각까지 만든 레코드로
      `current_labels`: 그 뒤의 삭제·수정 레코드는 아직 없던 것으로 본다).
    둘 다 모델 출처이고 측정·오류 삽입 레코드가 아닌 것만 남긴다.

    Args:
        unit: 측정 단위(종류·스트림)의 전체 이력.
        pair_tasks: 짝 표준 배정의 검수 작업 (없으면 보낸 프리라벨도 없다).

    Returns:
        모델 프리라벨 (중복 없이, label_id 순).
    """
    by_id = {x.label_id: x for x in unit}
    shown: dict[str, LabelRecord] = {}
    for t in pair_tasks:
        if t.sent_label_ids is not None:
            # 단위 밖(다른 종류·스트림) ID는 by_id에 없어 빠진다
            picked = [by_id[i] for i in t.sent_label_ids if i in by_id]
        else:
            picked = current_labels([x for x in unit if x.created_at <= t.created_at])
        for x in picked:
            if x.provenance.source is Source.MODEL and x.measurement is None and not x.seeded_error:
                shown[x.label_id] = x
    return [shown[i] for i in sorted(shown)]


def quality_report(conn: sa.Connection, policy: ReviewOpsPolicy) -> QualityReport:
    """끝난(done) 배정 전체로 검수 품질을 잰다. 읽기만 한다.

    - 오류 삽입 배정: 넣은 오류마다 `seeding.detected`로 발견 여부를 담당자별로 센다.
    - 이중 배정: 짝 표준 배정이 끝났으면, 단위의 현재 운영 라벨과 이 배정의 `double` 측정 레코드의
      일치도(`measure.agreement`).
    - 블라인드 배정: 짝 표준 배정이 끝났으면 프리라벨 편향(`measure.prelabel_bias`).
      표준 결과 = 현재 운영 라벨, 블라인드 결과 = `blind` 측정 레코드, 모델 프리라벨 = 짝 표준
      배정 작업에 실제로 보낸 모델 라벨(`shown_prelabels`).
    정책 값: `measurement.tolerance_ms`·`match_iou`, `seeding.detect_tolerance_ms`·`blur_overlap`.
    """
    mp = policy.measurement
    report = QualityReport()
    done = list_assignments(conn, status=AssignmentStatus.DONE)
    by_id = {a.assignment_id: a for a in done}
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for a in done:
        labels = get_labels(conn, a.session_id)
        if a.mode is ReviewMode.SEEDED_ERROR:
            who = a.assignee or "unassigned"
            for err in a.injected:
                total[who] += 1
                found[who] += detected(
                    err, labels, a.assignee, policy.seeding.detect_tolerance_ms,
                    policy.seeding.blur_overlap,
                )  # fmt: skip
            continue
        # 측정 배정이고 짝(표준 배정)이 끝났을 때만 비교한다
        if a.mode not in (ReviewMode.BLIND, ReviewMode.DOUBLE) or a.pair_id not in by_id:
            continue
        kinds = set(a.label_kinds)
        unit = [
            x
            for x in labels
            if x.kind in kinds and (a.stream_id is None or x.stream_id == a.stream_id)
        ]
        operational = current_labels(unit)
        # 이 배정 방식의 측정 레코드 중 담당자가 남긴 것 (검수자 ID가 없는 레코드도 받는다)
        measured = [
            x for x in unit
            if x.measurement == a.mode.value and x.verification.reviewer_id in (None, a.assignee)
        ]  # fmt: skip
        if a.mode is ReviewMode.DOUBLE:
            report.double[a.assignment_id] = agreement(
                as_items(operational), as_items(measured), mp.tolerance_ms, mp.match_iou
            )
        else:
            # 모델 프리라벨: 짝 표준 배정의 검수자가 실제로 본 것 (이전 버전·지워진 레코드 제외)
            pair_tasks = [
                t for t in list_review_tasks(conn, a.session_id) if t.assignment_id == a.pair_id
            ]
            model = shown_prelabels(unit, pair_tasks)
            report.blind_bias[a.assignment_id] = prelabel_bias(
                as_items(model),
                as_items(operational),
                as_items(measured),
                mp.tolerance_ms,
                mp.match_iou,
            )
    report.detection = [DetectionRate(r, total[r], found[r]) for r in sorted(total)]
    return report
