"""사용 중지 전파와 계보 조회.

사용 중지된 세션은 이후 모든 데이터셋 버전과 내보내기에서 빠진다. 이미 만든 버전·학습·내보내기는
지우지 않고(재현성), 영향받은 목록을 돌려준다. 영향받은 모델은 다음 정기 재학습에서 다시 학습한다.

사용 중지 전파 경로:
- 데이터셋 빌드(`build`)와 내보내기(`dlp_export.source.load_source`, `record_export`)는 지금
  `withdrawals` 테이블을 보고 그 세션을 뺀다.
- 이미 나간 내보내기는 계보(`session_lineage().exports`)로 찾아 구매자에게 삭제를 요청한다. 내보내기
  결과의 세션 ID는 가명이므로 내부 대응표(`dlp_export.runner.id_map_key`)로 가명을 찾는다.
- 내보내기와 동시에 일어난 사용 중지는 `dlp_export.runner`의 세션 행 잠금으로 직렬화된다
  (ADR 0021): 둘 중 먼저 커밋한 쪽이 이긴다.

진입점: `dlp dataset withdraw <세션> <이유>`(`withdraw_session`),
`dlp lineage <세션>`(`session_lineage`).
관련: WP7, ADR 0007. 호출자가 트랜잭션을 연다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import sqlalchemy as sa

from dlp_schema.dataset import Split
from dlp_schema.db.repository import (
    dataset_versions_with_session,
    get_dataset_version,
    get_session,
    insert_export,
    insert_training_run,
    insert_withdrawal,
    list_exports,
    list_golden_sets,
    list_training_runs,
    set_lifecycle,
    withdrawn_session_ids,
)
from dlp_schema.labels import VerificationState
from dlp_schema.lineage import ExportRecord, TrainingRun, Withdrawal
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


# 학습 예제는 학습·검증 분할에서만 뽑는다 (dlp_train.extract).
# 골든·holdout 세션은 학습에 들어가지 않는다.
TRAINING_SPLITS = frozenset({Split.TRAIN, Split.VAL})


def session_lineage(conn: sa.Connection, session_id: str) -> SessionLineage:
    """세션 → 골든셋·데이터셋 버전 → 학습 실행 → 내보내기.

    학습 실행은 그 버전에서 세션이 학습·검증 분할에 있었던 것만 돌려준다.
    내보내기는 그 버전들의 내보내기 중 `session_ids`(실제로 쓴 세션)에 이 세션이 있는 것만.

    Raises:
        KeyError 등: 세션이 없을 때 (`get_session`).
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
    return SessionLineage(session_id, session.lifecycle_state, versions, runs, exports, golden)


def withdraw_session(
    conn: sa.Connection, session_id: str, reason: str, now: datetime
) -> SessionLineage:
    """세션을 사용 중지하고, 이미 들어간 버전·학습·내보내기 목록을 돌려준다.

    멱등: 이미 `withdrawals`에 있으면 기록을 더하지 않는다 (첫 사유·시각이 남는다).
    부작용: 세션 행의 생애주기를 withdrawn으로 갱신(이 행 갱신이 내보내기의 FOR SHARE 잠금과
    직렬화된다), `withdrawals`에 추가.

    Args:
        conn: 호출자가 연 트랜잭션 연결.
        session_id: 사용 중지할 세션.
        reason: 사유 (예: "동의 철회").
        now: 사용 중지 시각 (시간대 포함).
    """
    set_lifecycle(conn, session_id, LifecycleState.WITHDRAWN)
    if session_id not in withdrawn_session_ids(conn):
        insert_withdrawal(conn, Withdrawal(session_id=session_id, reason=reason, withdrawn_at=now))
    return session_lineage(conn, session_id)


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
    실패한다.
    없으면 버전의 해당 분할에서 지금 사용 중지된 세션을 뺀 것이다.

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
