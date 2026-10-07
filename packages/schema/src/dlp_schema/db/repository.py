"""계약 타입 ↔ DB 행 변환과 기본 저장·조회. 연결(Connection)과 트랜잭션은 호출자가 관리한다.

역할
    `dlp_schema` 계약 객체를 `db.tables`의 테이블에 넣고 꺼내는 얇은 저장소 계층 (WP1, ADR 0002).
    모든 패키지가 DB를 이 함수들로만 다룬다 (직접 SQL을 쓰지 않는다).

트랜잭션
    모든 함수는 `sa.Connection`을 받아 그 연결에서 실행만 한다. 커밋·롤백은 호출자가 한다
    (보통 `with engine.begin() as conn:`). 여러 함수를 한 트랜잭션에 묶으면 함께 커밋·롤백된다.

절별 함수
    - 온톨로지: `register_ontology` (초안 덧붙이기 허용, ADR 0028).
    - 세션: `insert_session`, `get_session`, `set_lifecycle`(+ 전이 기록), `list_lifecycle_events`,
      `set_privacy_state`, `update_stream_sync`.
    - 라벨: `label_to_row`, `row_to_label`, `insert_labels`, `get_labels`, `record_review`.
    - 검수 작업·배정: `insert_review_task`, `get_review_task`, `list_review_tasks`,
      `mark_review_task_collected`, `insert_assignment`, `get_assignment`, `list_assignments`,
      `update_assignment`.
    - 데이터셋 버전: `insert_dataset_version`, `get_dataset_version`.
    - 세션 목록·계보: `list_session_ids`, 골든셋·학습 실행·모델 버전·내보내기·사용 중지 함수,
      `dataset_versions_with_session`.
    - 운영 기록(추가만): 원본 접근·검수 시간·블러 감사·보관 결정의 insert/list.

주의
    - 라벨은 불변이다. `record_review` 외에 라벨 행을 바꾸는 함수는 없고, DB 트리거도 막는다.
    - 운영 기록 표와 session_lifecycle_events는 추가만 한다
      (DB 트리거가 UPDATE·DELETE·TRUNCATE를 막는다).
    - `get_*` 단건 조회와 `set_lifecycle`은 행이 없으면 SQLAlchemy `NoResultFound`를 던진다.
      그 밖의 갱신 함수(`set_privacy_state`, `update_stream_sync`, `record_review`,
      `update_assignment`, `set_model_status` 등)는 대상이 없으면 `KeyError`를 던진다.
    - datetime 인자는 모두 시간대가 있어야 한다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import sqlalchemy as sa

from dlp_schema.dataset import DatasetVersion
from dlp_schema.db.tables import (
    dataset_split_assignments,
    dataset_versions,
    exports,
    golden_sets,
    label_records,
    model_versions,
    ontology_versions,
    privacy_audits,
    raw_access_log,
    retention_decisions,
    review_assignments,
    review_tasks,
    review_work,
    session_lifecycle_events,
    sessions,
    streams,
    training_runs,
    withdrawals,
)
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.lineage import (
    ExportRecord,
    GoldenSet,
    ModelStatus,
    ModelVersion,
    TrainingRun,
    Withdrawal,
)
from dlp_schema.ontology import Ontology
from dlp_schema.ops import PrivacyAuditRecord, RawAccessEvent, RetentionDecision, ReviewWork
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
)
from dlp_schema.session import (
    LifecycleEvent,
    LifecycleState,
    PrivacyState,
    Session,
    Stream,
    StreamKind,
    SyncMethod,
    can_transition,
)


class TransitionError(ValueError):
    """허용되지 않는 생애주기 전이 (`session.can_transition`이 거부). `set_lifecycle`이 던진다."""


# ---------------------------------------------------------------- 온톨로지


def register_ontology(conn: sa.Connection, ontology: Ontology) -> None:
    """온톨로지 버전을 등록한다.

    같은 버전이 이미 있으면
    - 내용이 같으면 아무것도 하지 않는다.
    - 등록된 버전이 초안(draft)이고 새 내용이 덧붙이기만 한 것이면(기존 키와 값은 그대로, 새 키만
      더함) 내용을 새 것으로 바꾼다 (ADR 0028). 예전 내용으로 기록된 라벨은 새 내용에서도 모두
      유효하다. 상태(draft/frozen) 변경은 덧붙이기가 아니다.
    - 그 밖(확정된 버전, 키 삭제·값 변경)은 오류다. 새 버전을 만들고 이관(migrate_labels)한다.

    Args:
        conn: DB 연결 (트랜잭션은 호출자가 연다).
        ontology: 등록할 온톨로지 (`load_ontology` 결과).

    Raises:
        ValueError: 같은 버전이 다른 내용으로 이미 있고 덧붙이기로 받을 수 없을 때.

    부작용:
        `ontology_versions` 행 추가 또는 content 갱신. 같은 버전 행을 `FOR UPDATE`로 잠가 동시
        등록을 직렬화한다.
    """
    # 비교·저장은 파싱된 값(JSON 직렬화)으로 한다. YAML 주석·서식은 영향을 주지 않는다.
    content = ontology.model_dump(mode="json")
    row = (
        conn.execute(
            sa.select(ontology_versions.c.status, ontology_versions.c.content)
            .where(ontology_versions.c.version == ontology.version)
            .with_for_update()
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        conn.execute(
            ontology_versions.insert().values(
                version=ontology.version, status=ontology.status, content=content
            )
        )
        return
    existing: Any = row["content"]
    if existing == content:
        return
    if row["status"] == "draft" and _is_additive(existing, content):
        conn.execute(
            ontology_versions.update()
            .where(ontology_versions.c.version == ontology.version)
            .values(content=content)
        )
        return
    why = "확정된 버전입니다" if row["status"] != "draft" else "덧붙이기가 아닌 변경입니다"
    raise ValueError(
        f"온톨로지 {ontology.version}이 다른 내용으로 이미 등록되어 있습니다 ({why}). "
        "버전을 올리고 이관하세요"
    )


def _is_additive(old: Any, new: Any) -> bool:
    """new가 old에 키를 더하기만 했는가 (사전은 재귀로, 그 밖의 값은 같아야 한다).

    old의 모든 키가 new에 있고 그 값도 재귀로 덧붙이기여야 한다. 목록·문자열 등 사전이 아닌 값은
    완전히 같아야 한다 (목록에 원소를 더하는 것은 덧붙이기가 아니다).
    """
    if isinstance(old, dict) and isinstance(new, dict):
        old_d: dict[str, Any] = old  # pyright: ignore[reportUnknownVariableType]
        new_d: dict[str, Any] = new  # pyright: ignore[reportUnknownVariableType]
        return all(k in new_d and _is_additive(v, new_d[k]) for k, v in old_d.items())
    return bool(old == new)


# ---------------------------------------------------------------- 세션


def insert_session(
    conn: sa.Connection,
    session: Session,
    *,
    at: datetime | None = None,
    actor: str | None = None,
) -> None:
    """세션과 스트림을 등록하고, 처음 생애주기 상태를 session_lifecycle_events에 남긴다.

    at: 기록 시각 (시간대 필수). 주지 않으면 DB 시각(now())이다.

    Args:
        conn: DB 연결.
        session: 등록할 세션 (스트림 포함).
        at: 처음 생애주기 기록의 시각.
        actor: 등록한 사람·단계 (선택, 예: `ingest`).

    Raises:
        sqlalchemy.exc.IntegrityError: 같은 session_id가 이미 있거나 ontology_version이
            등록되지 않았을 때.
        ValueError: at에 시간대가 없을 때.

    부작용:
        `sessions` 1행, `streams` 스트림 수만큼 (position = 계약의 순서),
        `session_lifecycle_events` 1행.
        멱등이 아니다 (중복 등록은 오류). 멱등 수집은 호출자(`dlp_media`)가 존재 여부로 처리한다.
    """
    data = session.model_dump(mode="json")
    conn.execute(
        sessions.insert().values(
            session_id=session.session_id,
            domain=data["domain"],
            worker_id=session.worker_id,
            site_id=session.site_id,
            consent_version=session.consent_version,
            recorded_at=session.recorded_at,
            duration_ms=session.duration_ms,
            calibration=data["calibration"],
            privacy_state=data["privacy_state"],
            lifecycle_state=data["lifecycle_state"],
            ontology_version=session.ontology_version,
        )
    )
    rows = [
        {"session_id": session.session_id, "position": i, **s}
        for i, s in enumerate(data["streams"])
    ]
    conn.execute(streams.insert(), rows)
    _record_lifecycle(conn, session.session_id, None, session.lifecycle_state, at, actor)


def get_session(conn: sa.Connection, session_id: str) -> Session:
    """세션과 스트림(등록 순서)을 읽어 `Session`으로 만든다.

    Raises:
        sqlalchemy.exc.NoResultFound: 세션이 없을 때.
    """
    row = (
        conn.execute(sa.select(sessions).where(sessions.c.session_id == session_id))
        .mappings()
        .one()
    )
    stream_rows = (
        conn.execute(
            sa.select(streams)
            .where(streams.c.session_id == session_id)
            .order_by(streams.c.position)
        )
        .mappings()
        .all()
    )
    # created_at(DB 등록 시각)과 streams의 session_id·position은 계약에 없는 열이라 뺀다
    data = _without(row, "created_at")
    data["streams"] = [_without(s, "session_id", "position") for s in stream_rows]
    return Session.model_validate(data)


def set_lifecycle(
    conn: sa.Connection,
    session_id: str,
    target: LifecycleState,
    *,
    at: datetime | None = None,
    actor: str | None = None,
) -> None:
    """생애주기 상태를 바꾸고 같은 트랜잭션에서 전이를 session_lifecycle_events에 남긴다.

    같은 상태로의 호출(멱등)은 기록하지 않는다. at: 전이 시각 (시간대 필수). 주지 않으면
    DB 시각(now())이다. actor: 전이를 일으킨 사람·단계 (선택).

    Raises:
        sqlalchemy.exc.NoResultFound: 세션이 없을 때.
        TransitionError: `can_transition`이 거부할 때 (건너뛰기·되돌아가기, withdrawn 이후 이동).
        ValueError: at에 시간대가 없을 때. 세션 행을 읽거나 바꾸기 전에 검사하므로 상태는 그대로다.

    부작용:
        세션 행을 `FOR UPDATE`로 잠그고 `sessions.lifecycle_state` 갱신 +
        `session_lifecycle_events` 1행.
    """
    # 시간대 없는 at은 UPDATE 전에 거부한다. 예전에는 `_record_lifecycle`에서야 검사해서
    # sessions 행이 이미 바뀐 뒤 예외가 났고, 호출자가 롤백하지 않으면 기록 없는 전이가 남았다.
    if at is not None and at.utcoffset() is None:
        raise ValueError("생애주기 기록 시각(at)에는 시간대가 있어야 합니다")
    current = LifecycleState(
        conn.execute(
            sa.select(sessions.c.lifecycle_state)
            .where(sessions.c.session_id == session_id)
            .with_for_update()
        ).scalar_one()
    )
    if not can_transition(current, target):
        raise TransitionError(f"{session_id}: {current} → {target} 전이는 허용되지 않습니다")
    if current is target:
        return
    conn.execute(
        sessions.update()
        .where(sessions.c.session_id == session_id)
        .values(lifecycle_state=target.value)
    )
    _record_lifecycle(conn, session_id, current, target, at, actor)


def _record_lifecycle(
    conn: sa.Connection,
    session_id: str,
    from_state: LifecycleState | None,
    to_state: LifecycleState,
    at: datetime | None,
    actor: str | None,
) -> None:
    """`session_lifecycle_events`에 전이 기록 1행을 추가한다.

    Args:
        from_state: 이전 상태. 세션 등록 기록이면 None.
        to_state: 새 상태.
        at: 기록 시각 (시간대 필수). None이면 DB의 `now()`(트랜잭션 시작 시각)를 쓴다.
        actor: 전이를 일으킨 사람·단계 (선택).

    Raises:
        ValueError: at에 시간대가 없을 때.
    """
    if at is not None and at.utcoffset() is None:
        raise ValueError("생애주기 기록 시각(at)에는 시간대가 있어야 합니다")
    conn.execute(
        session_lifecycle_events.insert().values(
            session_id=session_id,
            from_state=from_state.value if from_state is not None else None,
            to_state=to_state.value,
            at=at if at is not None else sa.func.now(),
            actor=actor,
        )
    )


def list_lifecycle_events(conn: sa.Connection, session_id: str) -> list[LifecycleEvent]:
    """세션의 생애주기 전이 기록 (기록 순서 = event_id 오름차순). 세션이 없으면 빈 목록."""
    rows = conn.execute(
        sa.select(session_lifecycle_events)
        .where(session_lifecycle_events.c.session_id == session_id)
        .order_by(session_lifecycle_events.c.event_id)
    ).mappings()
    return [LifecycleEvent.model_validate(dict(r)) for r in rows]


def set_privacy_state(conn: sa.Connection, session_id: str, state: PrivacyState) -> None:
    """세션의 프라이버시 상태를 바꾼다 (`dlp privacy detect|approve`).

    전이 순서는 검사하지 않는다 (승인을 되돌리는 것도 막지 않는다). 전이 기록도 남기지 않는다.

    Raises:
        KeyError: 세션이 없을 때.
    """
    result = conn.execute(
        sessions.update()
        .where(sessions.c.session_id == session_id)
        .values(privacy_state=state.value)
    )
    if result.rowcount != 1:
        raise KeyError(session_id)


def update_stream_sync(conn: sa.Connection, session_id: str, stream: Stream) -> None:
    """스트림의 동기화 결과(오프셋, 드리프트, 방법, 신뢰도, 사람 조정값)만 갱신한다.

    기준 스트림(바디캠)은 마스터 시계 그 자체라 reference·오프셋 0·배율 1·조정 0만 허용한다.
    다른 스트림을 reference로 바꿀 수도 없다.

    Args:
        conn: DB 연결.
        session_id: 세션 ID.
        stream: 새 동기화 값을 가진 스트림 (stream_id로 행을 찾는다). kind·uri 등 나머지
            필드는 무시된다.

    Raises:
        KeyError: 그런 스트림이 없을 때.
        ValueError: 기준 스트림의 시계를 바꾸려 하거나, 다른 스트림을 reference로 바꾸려 할 때.

    부작용:
        `streams`의 offset_ms, clock_scale, sync_method, sync_confidence, manual_adjustment_ms 갱신.
    """
    # 기준 스트림 여부는 인자의 kind가 아니라 DB에 저장된 kind로 판단한다 (인자를 믿지 않는다)
    stored = conn.execute(
        sa.select(streams.c.kind).where(
            streams.c.session_id == session_id, streams.c.stream_id == stream.stream_id
        )
    ).scalar_one_or_none()
    if stored is None:
        raise KeyError(f"{session_id}/{stream.stream_id}")
    is_reference = StreamKind(stored) is StreamKind.BODYCAM
    if is_reference and not (
        stream.sync_method is SyncMethod.REFERENCE and stream.is_identity_clock
    ):
        raise ValueError(f"{session_id}/{stream.stream_id}: 기준 스트림의 시계는 바꿀 수 없습니다")
    if not is_reference and stream.sync_method is SyncMethod.REFERENCE:
        raise ValueError(f"{session_id}/{stream.stream_id}: 기준 스트림이 아닙니다")
    result = conn.execute(
        streams.update()
        .where(streams.c.session_id == session_id, streams.c.stream_id == stream.stream_id)
        .values(
            offset_ms=stream.offset_ms,
            clock_scale=stream.clock_scale,
            sync_method=stream.sync_method.value,
            sync_confidence=stream.sync_confidence,
            manual_adjustment_ms=stream.manual_adjustment_ms,
        )
    )
    if result.rowcount != 1:
        raise KeyError(f"{session_id}/{stream.stream_id}")


# ---------------------------------------------------------------- 라벨


def label_to_row(label: LabelRecord) -> dict[str, Any]:
    """`LabelRecord` → `label_records` 행 사전.

    출처·검수 정보를 평탄한 열로 풀고, 열거형은 JSON 문자열 값으로, 페이로드는 JSON 사전으로 바꾼다.
    datetime은 객체 그대로 둔다 (DB 드라이버가 시간대 포함 시각으로 쓴다).
    """
    data = label.model_dump(mode="json")
    return {
        "label_id": label.label_id,
        "session_id": label.session_id,
        "stream_id": label.stream_id,
        "kind": label.kind,
        "t_start_ms": label.t_start_ms,
        "t_end_ms": label.t_end_ms,
        "ontology_version": label.ontology_version,
        "source": data["provenance"]["source"],
        "model_version": label.provenance.model_version,
        "sensor_id": label.provenance.sensor_id,
        "evidence": data["evidence"],
        "confidence": label.confidence,
        "verification_state": data["verification"]["state"],
        "reviewer_id": label.verification.reviewer_id,
        "reviewed_at": label.verification.reviewed_at,
        "parent_label_id": label.parent_label_id,
        "retracted": label.retracted,
        "seeded_error": label.seeded_error,
        "measurement": label.measurement,
        "created_at": label.created_at,
        "payload": data["payload"],
    }


def row_to_label(row: Mapping[Any, Any]) -> LabelRecord:
    """`label_records` 행 → `LabelRecord` (`label_to_row`의 역). 계약 검증을 다시 거친다.

    Raises:
        pydantic.ValidationError: 저장된 행이 현재 계약을 어길 때 (예: 계약이 바뀐 뒤 옛 행).
    """
    return LabelRecord.model_validate(
        {
            "label_id": row["label_id"],
            "session_id": row["session_id"],
            "stream_id": row["stream_id"],
            "t_start_ms": row["t_start_ms"],
            "t_end_ms": row["t_end_ms"],
            "ontology_version": row["ontology_version"],
            "provenance": {
                "source": row["source"],
                "model_version": row["model_version"],
                "sensor_id": row["sensor_id"],
            },
            "evidence": row["evidence"],
            "confidence": row["confidence"],
            "verification": {
                "state": row["verification_state"],
                "reviewer_id": row["reviewer_id"],
                "reviewed_at": row["reviewed_at"],
            },
            "parent_label_id": row["parent_label_id"],
            "retracted": row["retracted"],
            "seeded_error": row["seeded_error"],
            "measurement": row["measurement"],
            "created_at": row["created_at"],
            "payload": row["payload"],
        }
    )


def insert_labels(conn: sa.Connection, labels: Iterable[LabelRecord]) -> int:
    """라벨 레코드를 한 번에 추가한다 (executemany).

    부모(`parent_label_id`)가 가리키는 레코드는 이미 DB에 있거나 같은 배치에서 앞서 있어야 한다
    (자기 참조 FK). 같은 label_id가 이미 있으면 IntegrityError다 (덮어쓰지 않는다).
    멱등하게 쓰려면 호출자가 이력(`get_labels`)을 보고 새 레코드만 넘긴다.

    Returns:
        추가한 행 수.
    """
    rows = [label_to_row(x) for x in labels]
    if rows:
        conn.execute(label_records.insert(), rows)
    return len(rows)


def get_labels(
    conn: sa.Connection, session_id: str, kinds: Iterable[str] | None = None
) -> list[LabelRecord]:
    """세션의 라벨 이력 전체 (수정·삭제·측정·오류 삽입 레코드 포함).

    운영 현재 라벨만 필요하면 결과를 `episode.current_labels`에 넘긴다.

    Args:
        conn: DB 연결.
        session_id: 세션 ID.
        kinds: 라벨 종류 필터 (예: `["action", "gap"]`). None이면 모든 종류. 수정·삭제 레코드는
            보통 원래와 같은 종류라 걸러도 `current_labels` 판정이 유지된다.

    Returns:
        (t_start_ms, label_id) 순으로 정렬된 목록.
    """
    query = sa.select(label_records).where(label_records.c.session_id == session_id)
    if kinds is not None:
        query = query.where(label_records.c.kind.in_(list(kinds)))
    query = query.order_by(label_records.c.t_start_ms, label_records.c.label_id)
    return [row_to_label(r) for r in conn.execute(query).mappings()]


def record_review(
    conn: sa.Connection,
    label_id: str,
    state: VerificationState,
    reviewer_id: str,
    reviewed_at: datetime,
) -> None:
    """검수 상태만 갱신한다. 내용 수정은 새 레코드로 한다 (DB 트리거가 강제).

    라벨 행에서 바꿀 수 있는 유일한 경로다 (verification_state, reviewer_id, reviewed_at).

    Args:
        state: 새 검증 상태 (unreviewed는 거부).
        reviewer_id: 검수자 ID.
        reviewed_at: 검수 시각 (시간대 필수).

    Raises:
        ValueError: state가 unreviewed일 때.
        KeyError: 라벨이 없을 때.
    """
    if state is VerificationState.UNREVIEWED:
        raise ValueError("검수 기록에는 미검수 외의 상태가 필요합니다")
    result = conn.execute(
        label_records.update()
        .where(label_records.c.label_id == label_id)
        .values(verification_state=state.value, reviewer_id=reviewer_id, reviewed_at=reviewed_at)
    )
    if result.rowcount != 1:
        raise KeyError(label_id)


# ---------------------------------------------------------------- 검수 작업


def insert_review_task(conn: sa.Connection, task: ReviewTask) -> None:
    """검수 작업 1행을 추가한다 (`review_tasks`). 같은 task_key가 있으면 IntegrityError."""
    data = task.model_dump(mode="json")
    # JSON 직렬화는 시각을 문자열로 바꾸므로 datetime 열은 원래 객체로 되돌린다
    data["created_at"], data["collected_at"] = task.created_at, task.collected_at
    conn.execute(review_tasks.insert().values(**data))


def get_review_task(conn: sa.Connection, task_key: str) -> ReviewTask:
    """작업 키로 검수 작업을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(sa.select(review_tasks).where(review_tasks.c.task_key == task_key))
        .mappings()
        .one()
    )
    return ReviewTask.model_validate(dict(row))


def list_review_tasks(
    conn: sa.Connection, session_id: str, stage: ReviewStage | None = None
) -> list[ReviewTask]:
    """세션의 검수 작업 목록 (만든 순). stage를 주면 그 단계만."""
    query = sa.select(review_tasks).where(review_tasks.c.session_id == session_id)
    if stage is not None:
        query = query.where(review_tasks.c.stage == stage.value)
    rows = conn.execute(query.order_by(review_tasks.c.created_at)).mappings().all()
    return [ReviewTask.model_validate(dict(r)) for r in rows]


def mark_review_task_collected(conn: sa.Connection, task_key: str, at: datetime) -> None:
    """작업을 수집 완료(collected)로 표시하고 collected_at을 기록한다.

    이미 collected여도 다시 덮어쓴다 (시각이 갱신된다). Raises: KeyError (작업이 없을 때).
    """
    result = conn.execute(
        review_tasks.update()
        .where(review_tasks.c.task_key == task_key)
        .values(status=ReviewTaskStatus.COLLECTED.value, collected_at=at)
    )
    if result.rowcount != 1:
        raise KeyError(task_key)


def insert_assignment(conn: sa.Connection, a: ReviewAssignment) -> None:
    """검수 배정 1행을 추가한다 (`review_assignments`). 같은 ID가 있으면 IntegrityError."""
    data = a.model_dump(mode="json")
    # datetime 열은 JSON 문자열이 아니라 원래 객체로 넣는다
    data["created_at"], data["completed_at"] = a.created_at, a.completed_at
    conn.execute(review_assignments.insert().values(**data))


def get_assignment(conn: sa.Connection, assignment_id: str) -> ReviewAssignment:
    """배정 ID로 배정을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(
            sa.select(review_assignments).where(review_assignments.c.assignment_id == assignment_id)
        )
        .mappings()
        .one()
    )
    return ReviewAssignment.model_validate(dict(row))


def list_assignments(
    conn: sa.Connection, session_id: str | None = None, status: AssignmentStatus | None = None
) -> list[ReviewAssignment]:
    """우선순위 높은 순 (같으면 만든 순).

    Args:
        session_id: 주면 그 세션만.
        status: 주면 그 상태만 (예: open = 검수 대기열).

    Returns:
        (priority 내림차순, created_at, assignment_id) 순 목록. 마지막 키는 결정적 순서를 위한 것.
    """
    query = sa.select(review_assignments)
    if session_id is not None:
        query = query.where(review_assignments.c.session_id == session_id)
    if status is not None:
        query = query.where(review_assignments.c.status == status.value)
    query = query.order_by(
        review_assignments.c.priority.desc(),
        review_assignments.c.created_at,
        review_assignments.c.assignment_id,
    )
    return [ReviewAssignment.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def update_assignment(conn: sa.Connection, assignment_id: str, **values: object) -> None:
    """배정 상태·담당자·작업 키만 바꾼다.

    Args:
        assignment_id: 배정 ID.
        **values: assignee, task_key, status(`AssignmentStatus` 또는 문자열), completed_at 중 일부.

    Raises:
        ValueError: 허용되지 않은 필드를 바꾸려 할 때.
        KeyError: 배정이 없을 때.
    """
    allowed = {"assignee", "task_key", "status", "completed_at"}
    if not set(values) <= allowed:
        raise ValueError(f"바꿀 수 없는 필드: {set(values) - allowed}")
    data = {k: (v.value if isinstance(v, AssignmentStatus) else v) for k, v in values.items()}
    result = conn.execute(
        review_assignments.update()
        .where(review_assignments.c.assignment_id == assignment_id)
        .values(**data)
    )
    if result.rowcount != 1:
        raise KeyError(assignment_id)


# ---------------------------------------------------------------- 데이터셋 버전


def insert_dataset_version(conn: sa.Connection, version: DatasetVersion) -> None:
    """데이터셋 버전과 세션별 분할을 추가한다.

    부작용: `dataset_versions` 1행 + `dataset_split_assignments` 분할 수만큼. 분할의 세션은
    `sessions`에 있어야 한다 (FK). 같은 version_id가 있으면 IntegrityError.
    """
    data = version.model_dump(mode="json")
    conn.execute(
        dataset_versions.insert().values(
            version_id=version.version_id,
            parent_version_id=version.parent_version_id,
            ontology_version=version.ontology_version,
            created_at=version.created_at,
            snapshot_uri=version.snapshot_uri,
            golden_set_version=version.golden_set_version,
            excluded_sessions=data["excluded_sessions"],
        )
    )
    rows = [
        {"version_id": version.version_id, "session_id": sid, "split": split}
        for sid, split in data["splits"].items()
    ]
    if rows:
        conn.execute(dataset_split_assignments.insert(), rows)


def get_dataset_version(conn: sa.Connection, version_id: str) -> DatasetVersion:
    """데이터셋 버전과 분할을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(sa.select(dataset_versions).where(dataset_versions.c.version_id == version_id))
        .mappings()
        .one()
    )
    splits = conn.execute(
        sa.select(dataset_split_assignments.c.session_id, dataset_split_assignments.c.split).where(
            dataset_split_assignments.c.version_id == version_id
        )
    ).all()
    data = _without(row)
    data["splits"] = {sid: split for sid, split in splits}
    return DatasetVersion.model_validate(data)


def _without(mapping: Mapping[Any, Any], *keys: str) -> dict[str, Any]:
    """행 매핑을 문자열 키 사전으로 바꾸며 `keys` 열을 뺀다 (계약에 없는 DB 전용 열 제거용)."""
    return {str(k): v for k, v in mapping.items() if k not in keys}


# ---------------------------------------------------------------- 세션 목록·계보


def list_session_ids(conn: sa.Connection, ontology_version: str | None = None) -> list[str]:
    """세션 ID 목록 (ID 순). ontology_version을 주면 그 버전 세션만."""
    query = sa.select(sessions.c.session_id).order_by(sessions.c.session_id)
    if ontology_version is not None:
        query = query.where(sessions.c.ontology_version == ontology_version)
    return [str(r) for r in conn.execute(query).scalars()]


def insert_golden_set(conn: sa.Connection, golden: GoldenSet) -> None:
    """골든셋 버전 1행을 추가한다. 같은 version이 있으면 IntegrityError."""
    data = golden.model_dump(mode="json")
    data["created_at"] = golden.created_at
    conn.execute(golden_sets.insert().values(**data))


def get_golden_set(conn: sa.Connection, version: str) -> GoldenSet:
    """골든셋 버전을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = (
        conn.execute(sa.select(golden_sets).where(golden_sets.c.version == version))
        .mappings()
        .one()
    )
    return GoldenSet.model_validate(dict(row))


def list_golden_sets(conn: sa.Connection, domain: str | None = None) -> list[GoldenSet]:
    """골든셋 버전 목록 (만든 순). domain을 주면 그 도메인만."""
    query = sa.select(golden_sets).order_by(golden_sets.c.created_at)
    if domain is not None:
        query = query.where(golden_sets.c.domain == domain)
    return [GoldenSet.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_training_run(conn: sa.Connection, run: TrainingRun) -> None:
    """학습 실행 1행을 추가한다 (데이터셋 버전이 DB에 있어야 한다, FK)."""
    conn.execute(training_runs.insert().values(**run.model_dump()))


def list_training_runs(conn: sa.Connection, dataset_version_ids: list[str]) -> list[TrainingRun]:
    """주어진 데이터셋 버전들에서 나온 학습 실행 (실행 시각 순). 계보 조회용."""
    query = (
        sa.select(training_runs)
        .where(training_runs.c.dataset_version_id.in_(dataset_version_ids))
        .order_by(training_runs.c.created_at)
    )
    return [TrainingRun.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def get_training_run(conn: sa.Connection, run_id: str) -> TrainingRun:
    """학습 실행을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    row = conn.execute(sa.select(training_runs).where(training_runs.c.run_id == run_id)).mappings()
    return TrainingRun.model_validate(dict(row.one()))


def insert_model_version(conn: sa.Connection, mv: ModelVersion) -> None:
    """모델 버전 1행을 추가한다 (학습 실행이 DB에 있어야 한다, FK)."""
    conn.execute(model_versions.insert().values(**mv.model_dump()))


def get_model_version(conn: sa.Connection, model_version: str) -> ModelVersion:
    """모델 버전을 읽는다. Raises: sqlalchemy.exc.NoResultFound (없을 때)."""
    query = sa.select(model_versions).where(model_versions.c.model_version == model_version)
    return ModelVersion.model_validate(dict(conn.execute(query).mappings().one()))


def list_model_versions(
    conn: sa.Connection, task: str | None = None, status: ModelStatus | None = None
) -> list[ModelVersion]:
    """모델 버전 목록 (등록 순, 같으면 버전 ID 순). task·status로 거를 수 있다."""
    query = sa.select(model_versions).order_by(
        model_versions.c.created_at, model_versions.c.model_version
    )
    if task is not None:
        query = query.where(model_versions.c.task == task)
    if status is not None:
        query = query.where(model_versions.c.status == status.value)
    return [ModelVersion.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def set_model_status(
    conn: sa.Connection,
    model_version: str,
    status: ModelStatus,
    at: datetime,
    report_uri: str | None = None,
) -> None:
    """모델 버전의 상태를 바꾸고 판정 시각(decided_at)을 기록한다.

    Args:
        status: 새 상태. candidate를 뺀 전이 규칙은 검사하지 않는다 (호출자 `dlp_train`이 관리).
        at: 판정·은퇴 시각 (decided_at에 쓴다).
        report_uri: 평가 리포트 위치. None이면 기존 값을 그대로 둔다.

    Raises:
        KeyError: 모델 버전이 없을 때.
        ValueError: status가 candidate일 때. candidate는 등록(`insert_model_version`) 때만 갖는
            초기 상태다. 여기서 candidate로 되돌리면 decided_at이 채워진 candidate가 되어
            `ModelVersion` 검증기("candidate일 때만 decided_at이 None")를 어기고, 이후
            `get_model_version`·`list_model_versions`가 그 행을 읽지 못한다.
    """
    if status is ModelStatus.CANDIDATE:
        raise ValueError(f"{model_version}: 판정된 모델을 candidate로 되돌릴 수 없습니다")
    values: dict[str, Any] = {"status": status.value, "decided_at": at}
    if report_uri is not None:
        values["report_uri"] = report_uri
    result = conn.execute(
        model_versions.update()
        .where(model_versions.c.model_version == model_version)
        .values(**values)
    )
    if result.rowcount != 1:
        raise KeyError(model_version)


def insert_export(conn: sa.Connection, export: ExportRecord) -> None:
    """내보내기 이력 1행을 추가한다 (파일을 올리기 전에 호출자가 따로 커밋한다)."""
    data = export.model_dump(mode="json")
    data["created_at"] = export.created_at
    conn.execute(exports.insert().values(**data))


def list_exports(conn: sa.Connection, dataset_version_ids: list[str]) -> list[ExportRecord]:
    """주어진 데이터셋 버전들에서 만든 내보내기 (만든 순). 계보 조회·사용 중지 전파용."""
    query = (
        sa.select(exports)
        .where(exports.c.dataset_version_id.in_(dataset_version_ids))
        .order_by(exports.c.created_at)
    )
    return [ExportRecord.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_withdrawal(conn: sa.Connection, withdrawal: Withdrawal) -> None:
    """사용 중지 기록을 추가한다. 이미 중지된 세션이면 IntegrityError (session_id가 기본 키)."""
    conn.execute(withdrawals.insert().values(**withdrawal.model_dump()))


def withdrawn_session_ids(conn: sa.Connection) -> set[str]:
    """사용 중지된 모든 세션 ID."""
    return {str(r) for r in conn.execute(sa.select(withdrawals.c.session_id)).scalars()}


def dataset_versions_with_session(conn: sa.Connection, session_id: str) -> list[str]:
    """세션이 분할에 들어간 데이터셋 버전 ID 목록 (ID 순).

    excluded_sessions로만 언급된 버전은 빠진다.
    """
    query = (
        sa.select(dataset_split_assignments.c.version_id)
        .where(dataset_split_assignments.c.session_id == session_id)
        .order_by(dataset_split_assignments.c.version_id)
    )
    return [str(r) for r in conn.execute(query).scalars()]


# ---------------------------------------------------------------- 운영 기록 (추가만 한다)


def _between(column: Any, start: datetime | None, end: datetime | None) -> list[Any]:
    """기간 조건 목록: start 이상(포함), end 미만(제외). None인 쪽은 열어 둔다."""
    out: list[Any] = []
    if start is not None:
        out.append(column >= start)
    if end is not None:
        out.append(column < end)
    return out


def insert_raw_access(conn: sa.Connection, event: RawAccessEvent) -> None:
    """원본 접근 감사 기록 1행을 추가한다. 감사 저장소(`dlp_cli.raw_access`)만 부른다."""
    conn.execute(raw_access_log.insert().values(**event.model_dump()))


def list_raw_access(
    conn: sa.Connection, start: datetime | None = None, end: datetime | None = None
) -> list[RawAccessEvent]:
    """[start, end) 기간의 원본 접근 기록 (시각 순). 월간 감사 리포트용."""
    query = (
        sa.select(raw_access_log)
        .where(*_between(raw_access_log.c.at, start, end))
        .order_by(raw_access_log.c.at, raw_access_log.c.event_id)
    )
    return [RawAccessEvent.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_review_work(conn: sa.Connection, work: ReviewWork) -> None:
    """검수 작업 시간 기록 1행을 추가한다."""
    conn.execute(review_work.insert().values(**work.model_dump()))


def list_review_work(
    conn: sa.Connection, start: datetime | None = None, end: datetime | None = None
) -> list[ReviewWork]:
    """[start, end) 기간(recorded_at 기준)의 검수 시간 기록 (시각 순). 주간 운영 지표용."""
    query = (
        sa.select(review_work)
        .where(*_between(review_work.c.recorded_at, start, end))
        .order_by(review_work.c.recorded_at, review_work.c.work_id)
    )
    return [ReviewWork.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_privacy_audit(conn: sa.Connection, audit: PrivacyAuditRecord) -> None:
    """잔여 블러 누락 감사 결과 1행을 추가한다."""
    conn.execute(privacy_audits.insert().values(**audit.model_dump()))


def list_privacy_audits(
    conn: sa.Connection, start: datetime | None = None, end: datetime | None = None
) -> list[PrivacyAuditRecord]:
    """[start, end) 기간(audited_at 기준)의 블러 감사 결과 (시각 순)."""
    query = (
        sa.select(privacy_audits)
        .where(*_between(privacy_audits.c.audited_at, start, end))
        .order_by(privacy_audits.c.audited_at, privacy_audits.c.audit_id)
    )
    return [PrivacyAuditRecord.model_validate(dict(r)) for r in conn.execute(query).mappings()]


def insert_retention_decision(conn: sa.Connection, decision: RetentionDecision) -> None:
    """원본 보관 결정 1행을 추가한다 (결정을 바꾸려면 새 결정을 추가한다)."""
    conn.execute(retention_decisions.insert().values(**decision.model_dump()))


def list_retention_decisions(
    conn: sa.Connection, session_id: str | None = None
) -> list[RetentionDecision]:
    """보관 결정 목록 (결정 시각 순 = 마지막이 최신). session_id를 주면 그 세션만."""
    query = sa.select(retention_decisions).order_by(
        retention_decisions.c.decided_at, retention_decisions.c.decision_id
    )
    if session_id is not None:
        query = query.where(retention_decisions.c.session_id == session_id)
    return [RetentionDecision.model_validate(dict(r)) for r in conn.execute(query).mappings()]
