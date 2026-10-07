"""사용 중지 전파와 계보 조회.

사용 중지된 세션은 이후 모든 데이터셋 버전과 내보내기에서 빠진다. 이미 만든 버전·학습·내보내기는
지우지 않고(재현성), 영향받은 목록을 돌려준다. 영향받은 모델은 다음 정기 재학습에서 다시 학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    list_training_runs,
    set_lifecycle,
    withdrawn_session_ids,
)
from dlp_schema.lineage import ExportRecord, TrainingRun, Withdrawal
from dlp_schema.session import LifecycleState


@dataclass(frozen=True)
class SessionLineage:
    session_id: str
    lifecycle: LifecycleState
    dataset_versions: list[str]
    training_runs: list[TrainingRun]
    exports: list[ExportRecord]


def session_lineage(conn: sa.Connection, session_id: str) -> SessionLineage:
    """세션 → 데이터셋 버전 → 학습 실행 → 내보내기."""
    session = get_session(conn, session_id)
    versions = dataset_versions_with_session(conn, session_id)
    exports = (
        [e for e in list_exports(conn, versions) if session_id in e.session_ids] if versions else []
    )
    runs = list_training_runs(conn, versions) if versions else []
    return SessionLineage(session_id, session.lifecycle_state, versions, runs, exports)


def withdraw_session(
    conn: sa.Connection, session_id: str, reason: str, now: datetime
) -> SessionLineage:
    """세션을 사용 중지하고, 이미 들어간 버전·학습·내보내기 목록을 돌려준다."""
    set_lifecycle(conn, session_id, LifecycleState.WITHDRAWN)
    if session_id not in withdrawn_session_ids(conn):
        insert_withdrawal(conn, Withdrawal(session_id=session_id, reason=reason, withdrawn_at=now))
    return session_lineage(conn, session_id)


def register_training_run(conn: sa.Connection, run: TrainingRun) -> None:
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
    now: datetime,
) -> ExportRecord:
    """내보내기 기록. 내보낼 세션은 버전의 해당 분할에서 지금 사용 중지된 세션을 뺀 것이다."""
    version = get_dataset_version(conn, dataset_version_id)
    withdrawn = withdrawn_session_ids(conn)
    sessions = tuple(
        sorted(sid for sid, sp in version.splits.items() if sp in splits and sid not in withdrawn)
    )
    export = ExportRecord(
        export_id=export_id, dataset_version_id=dataset_version_id, target=target, format=format,
        uri=uri, session_ids=sessions, created_at=now,
    )  # fmt: skip
    insert_export(conn, export)
    return export
