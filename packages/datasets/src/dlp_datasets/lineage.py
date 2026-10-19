"""사용 중지 전파와 계보 조회.

사용 중지된 세션은 이후 모든 데이터셋 버전과 내보내기에서 빠진다. 이미 만든 버전·학습·내보내기는
지우지 않고(계보 기록), 영향받은 목록을 돌려준다. 영향받은 모델의 재학습·기존 산출물 처리는 운영자가
별도로 수행해야 한다. 이 모듈은 재학습을 예약하거나 실행하지 않는다.

사용 중지 전파 경로:
- 데이터셋 빌드(`build`)와 내보내기(`dlp_export.source.load_source`, `record_export`)는 지금
  `withdrawals` 테이블을 보고 그 세션을 뺀다.
- 이미 나간 내보내기는 계보(`session_lineage().exports`)로 찾을 수 있다. 구매자에게 삭제를 요청하는
  일은 운영 절차이며 이 모듈이 요청을 보내지는 않는다. 결과의 세션 ID는 가명이므로 내부 대응표
  (`dlp_export.runner.id_map_key`)로 가명을 찾는다.
- 내보내기와 동시에 일어난 사용 중지는 `dlp_export.runner`의 세션 행 잠금으로 직렬화된다
  (ADR 0021): 둘 중 먼저 커밋한 쪽이 이긴다.
- 이미 배포한 검수 작업(라벨러 워터마크 사본, Label Studio 서명 URL, CVAT 작업)은 사용 중지
  트랜잭션이 `review_tasks`에 매체 회수를 요청한다 (outbox, ADR 0039, F-REV-17). 외부 삭제는 커밋
  뒤 `dlp_review.revoke` 실행기가 한다 (`dlp dataset withdraw`가 돌린다). 요청은 `dlp_schema` 저장소
  함수(`request_task_revocation`)로만 쓴다 (의존 표: datasets → schema). 블러 검수 작업(원본 접근
  권한자 화면)은 회수 대상이 아니라 남는다. 사용 중지한 세션에는 새 검수 작업을 만들지 않는다
  (`dlp_review.tasks`).

진입점: `dlp dataset withdraw <세션> <이유>`(`withdraw_session`),
`dlp lineage <세션>`(`session_lineage`).
관련: WP7, ADR 0007, 0031(사용 중지 생애주기 기록의 시각·실행자). 호출자가 트랜잭션을 연다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

import sqlalchemy as sa

from dlp_schema.dataset import Split
from dlp_schema.db.calibration_repository import hand_size_records_from_session
from dlp_schema.db.repository import (
    dataset_versions_with_session,
    get_dataset_version,
    get_session,
    insert_export,
    insert_training_run,
    insert_withdrawal,
    list_exports,
    list_golden_sets,
    list_review_tasks,
    list_training_runs,
    request_task_revocation,
    set_lifecycle,
    withdrawn_session_ids,
)
from dlp_schema.labels import VerificationState
from dlp_schema.lineage import ExportRecord, TrainingRun, Withdrawal
from dlp_schema.review import ReviewStage
from dlp_schema.session import LifecycleState


@dataclass(frozen=True)
class SessionLineage:
    """세션 하나의 계보."""

    session_id: str
    lifecycle: LifecycleState  # 지금 생애주기 상태
    dataset_versions: list[str]  # 이 세션이 (어느 분할로든) 들어간 데이터셋 버전 ID
    training_runs: list[TrainingRun]  # 이 세션이 학습·검증 분할이던 버전으로 학습한 실행
    exports: list[ExportRecord]  # 이 세션을 실제로 내보낸 기록 (미완성 내보내기 포함)
    golden_sets: list[str] = field(default_factory=list[str])  # 이 세션이 든 골든셋 버전
    # 이번 사용 중지(`withdraw_session`)가 매체 회수를 요청한 검수 작업 키. 계보 조회는 비어 있다
    revocations: list[str] = field(default_factory=list[str])
    # 이 세션을 추정 출처로 쓴 작업자 손 크기 등록 ID (ADR 0052). 사용 중지되면 그 추정은 손 크기
    # 결정에서 빠지고, 같은 작업자 다른 세션의 검수 전 3D가 다음 프리라벨에서 다시 만들어진다
    hand_sizes: list[str] = field(default_factory=list[str])


# 학습 예제는 학습·검증 분할에서만 뽑는다 (dlp_train.extract).
# 골든·holdout 세션은 학습에 들어가지 않는다.
TRAINING_SPLITS = frozenset({Split.TRAIN, Split.VAL})


def session_lineage(conn: sa.Connection, session_id: str) -> SessionLineage:
    """세션 → 골든셋·데이터셋 버전 → 학습 실행 → 내보내기, 그리고 이 세션에서 추정한 손 크기 등록.

    학습 실행은 그 버전에서 세션이 학습·검증 분할에 있었던 것만 돌려준다.
    내보내기는 그 버전들의 내보내기 중 `session_ids`(실제로 쓴 세션)에 이 세션이 있는 것만.

    Raises:
        sqlalchemy.exc.NoResultFound: 세션이 없을 때 (`get_session`).
    """
    session = get_session(conn, session_id)
    versions = dataset_versions_with_session(conn, session_id)
    exports = (
        [e for e in list_exports(conn, versions) if session_id in e.session_ids] if versions else []
    )
    trained = [
        v
        for v in versions
        if get_dataset_version(conn, v).splits.get(session_id) in TRAINING_SPLITS
    ]
    runs = list_training_runs(conn, trained) if trained else []
    golden = [g.version for g in list_golden_sets(conn) if session_id in g.session_ids]
    return SessionLineage(
        session_id,
        session.lifecycle_state,
        versions,
        runs,
        exports,
        golden,
        hand_sizes=hand_size_records_from_session(conn, session_id),
    )


def withdraw_session(
    conn: sa.Connection, session_id: str, reason: str, now: datetime, *, actor: str = "withdraw"
) -> SessionLineage:
    """세션을 사용 중지하고, 이미 들어간 버전·학습·내보내기 목록을 돌려준다.

    멱등: 이미 `withdrawals`에 있으면 기록을 더하지 않는다 (첫 사유·시각이 남는다). 이미 회수를
    요청한 작업은 다시 요청하지 않는다.
    부작용: 세션 행의 생애주기를 withdrawn으로 갱신하고 `session_lifecycle_events`에 전이를 남긴다
    (이 행 갱신이 내보내기의 FOR SHARE 잠금, 검수 작업 생성의 세션 잠금과 직렬화된다),
    `withdrawals`에 추가, 그 세션의 비프라이버시 검수 작업 중 요청이 없는 것마다 매체 회수 요청
    (`review_tasks`, 사유 `withdrawn:<actor>`, 열린 작업은 revoked. 외부 삭제는 하지 않는다).

    Args:
        conn: 호출자가 연 트랜잭션 연결.
        session_id: 사용 중지할 세션.
        reason: 사유 (예: "동의 철회").
        now: 사용 중지 시각 (시간대 포함). 생애주기 기록의 `at`에도 그대로 쓴다.
        actor: 사용 중지를 실행한 사람·단계. 생애주기 기록의 `actor`가 된다 (CLI는
            `--actor` 또는 `DLP_ACTOR`/OS 사용자, 기본값은 호출 경로를 알리는 "withdraw").
    """
    # 전이 시각·실행자를 함께 남긴다: 동의 철회 대응은 감사 대상이라 "언제·누가"가 필요하다.
    # (예전에는 둘 다 빠져 기록 시각이 DB 시각, 실행자는 비어 있었다.)
    set_lifecycle(conn, session_id, LifecycleState.WITHDRAWN, at=now, actor=actor)
    if session_id not in withdrawn_session_ids(conn):
        insert_withdrawal(conn, Withdrawal(session_id=session_id, reason=reason, withdrawn_at=now))
    revoked = _request_review_revocations(conn, session_id, now, actor)
    return replace(session_lineage(conn, session_id), revocations=revoked)


def _request_review_revocations(
    conn: sa.Connection, session_id: str, now: datetime, actor: str
) -> list[str]:
    """사용 중지한 세션의 배포된 검수 작업마다 매체 회수를 요청한다. 이번에 요청한 작업 키.

    동의를 철회한 영상은 렌더 해시와 무관하게 모두 회수한다 (블러가 맞아도 쓰면 안 된다).
    같은 매체를 쓰는 작업도 같은 세션이라 함께 요청된다. 블러 검수 작업은 저장소 함수가 거른다.
    호출자가 세션 행을 이미 잠갔다 (`set_lifecycle`).
    """
    return [
        t.task_key
        for t in list_review_tasks(conn, session_id)
        if t.stage is not ReviewStage.PRIVACY
        and t.revoke_requested_at is None
        and request_task_revocation(conn, t.task_key, reason=f"withdrawn:{actor}", at=now)
    ]


def register_training_run(conn: sa.Connection, run: TrainingRun) -> None:
    """학습 실행을 계보에 등록한다 (데이터셋 버전이 있어야 한다)."""
    get_dataset_version(conn, run.dataset_version_id)  # 없으면 오류
    insert_training_run(conn, run)


def record_export(
    conn: sa.Connection,
    *,
    export_id: str,
    dataset_version_id: str,
    target: str,
    format: str,
    uri: str,
    splits: tuple[Split, ...] = (Split.TRAIN, Split.VAL),
    label_states: tuple[VerificationState, ...] = (),
    session_ids: tuple[str, ...] | None = None,
    now: datetime,
) -> ExportRecord:
    """내보내기 기록.

    session_ids를 주면(실제로 쓴 세션) 그대로 남기되, 버전 밖이거나 사용 중지된 세션이 있으면
    실패한다. 없으면 버전의 해당 분할에서 지금 사용 중지된 세션을 뺀 것이다.

    Args:
        export_id: 내보내기 ID (`exports` 기본 키).
        dataset_version_id: 원본 데이터셋 버전.
        target: 받는 쪽.
        format: 형식 이름 (coco·intervals·lerobot).
        uri: 결과 폴더 위치 (데이터셋 버킷).
        splits: 내보낸 분할.
        label_states: 적용한 검증 정책 (모델 라벨 검증 상태).
        session_ids: 실제로 쓴 세션 (내부 ID).
        now: 기록 시각.

    Returns:
        DB `exports`에 쓴 `ExportRecord`.

    Raises:
        ValueError: `session_ids`에 버전 밖(또는 다른 분할)·사용 중지 세션이 있을 때.
    """
    version = get_dataset_version(conn, dataset_version_id)
    withdrawn = withdrawn_session_ids(conn)
    allowed = {sid for sid, sp in version.splits.items() if sp in splits and sid not in withdrawn}
    if session_ids is None:
        sessions = tuple(sorted(allowed))
    else:
        bad = sorted(set(session_ids) - allowed)
        if bad:
            raise ValueError(f"내보낼 수 없는 세션 (버전 밖·사용 중지): {bad}")
        sessions = tuple(sorted(session_ids))
    export = ExportRecord(
        export_id=export_id, dataset_version_id=dataset_version_id, target=target, format=format,
        uri=uri, session_ids=sessions, label_states=label_states, created_at=now,
    )  # fmt: skip
    insert_export(conn, export)
    return export
