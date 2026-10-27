"""세션 상태 읽기 (읽기 전용, ADR 0061 3절).

DB는 기존 저장소 함수(`dlp_schema.db.repository`·`review_repository`·`privacy_repository`)로만 읽고
쓰지 않는다. 화면에 보내는 값에는 원본 위치(스트림 `uri`·`pts_index_uri`, 검수 작업 `media_uri`)를
넣지 않는다. 렌더 여부는 라벨링 버킷을 라벨러 자격 증명(읽기 전용)으로 HEAD 해서 본다.

라벨 수(`label_counts`)는 화면용 집계라 SQL로 센다: 삭제·오류 삽입·측정 레코드와 다른 레코드의
parent인 레코드를 뺀다. `current_labels()`와 달리 오류 삽입 레코드의 후손은 빼지 못한다 (근사).
단계 입력으로 쓰지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa

from dlp_schema.db.privacy_repository import active_privacy_hold, latest_privacy_detection
from dlp_schema.db.repository import get_session, list_session_ids, withdrawn_session_ids
from dlp_schema.db.review_repository import list_review_tasks
from dlp_schema.db.tables import label_records
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import LIFECYCLE_ORDER, LifecycleState, Session, StreamKind

VIDEO_KINDS = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)

# 렌더 확인 함수: (세션 ID, 스트림 ID) → 라벨링 버킷에 블러본·렌더 기록이 있는가
RenderedCheck = Callable[[str, str], bool]


def blurred_key(session_id: str, stream_id: str) -> str:
    """라벨링 버킷의 블러본 키 (`dlp privacy render`가 쓰는 위치, dlp_privacy.rendering)."""
    return f"sessions/{session_id}/blurred/{stream_id}.mp4"


def render_record_key(session_id: str, stream_id: str) -> str:
    """렌더 기록 키 (`<블러본>.render.json`과 같은 이름 규칙)."""
    return f"sessions/{session_id}/blurred/{stream_id}.render.json"


def lifecycle_at_least(state: LifecycleState, target: LifecycleState) -> bool:
    """정상 진행 순서에서 state가 target 이상인가 (withdrawn은 늘 False)."""
    if state is LifecycleState.WITHDRAWN:
        return False
    return LIFECYCLE_ORDER.index(state) >= LIFECYCLE_ORDER.index(target)


def tool_url(task: ReviewTask, env: dict[str, str]) -> str | None:
    """검수 도구에서 작업을 여는 주소 (브라우저용, 이 컴퓨터 기준)."""
    host = env.get("DLP_CONSOLE_TOOL_HOST", "localhost")
    if task.tool is ReviewTool.CVAT:
        return f"http://{host}:{env.get('DLP_CVAT_PORT', '8080')}/tasks/{task.external_id}"
    if task.tool is ReviewTool.LABEL_STUDIO:
        # Label Studio 작업 화면 주소에는 프로젝트 번호가 필요해 프로젝트 목록을 연다 (작업 번호는
        # 화면에 보인다)
        return f"http://{host}:{env.get('DLP_LABEL_STUDIO_PORT', '8081')}/projects/"
    return None


def task_view(task: ReviewTask, env: dict[str, str]) -> dict[str, Any]:
    """검수 작업의 화면용 값 (매체 위치 `media_uri`는 싣지 않는다)."""
    return {
        "task_key": task.task_key,
        "tool": task.tool.value,
        "stage": task.stage.value,
        "stream_id": task.stream_id,
        "assignee": task.assignee,
        "status": task.status.value,
        "created_at": task.created_at.isoformat(),
        "collected_at": task.collected_at.isoformat() if task.collected_at else None,
        "privacy_detection_id": task.privacy_detection_id,
        "url": tool_url(task, env),
    }


def session_view(s: Session) -> dict[str, Any]:
    """세션의 화면용 값 (스트림 위치·PTS 인덱스 위치는 싣지 않는다)."""
    return {
        "session_id": s.session_id,
        "domain": s.domain.value,
        "worker_id": s.worker_id,
        "site_id": s.site_id,
        "recorded_at": s.recorded_at.isoformat(),
        "duration_ms": s.duration_ms,
        "lifecycle_state": s.lifecycle_state.value,
        "privacy_state": s.privacy_state.value,
        "ontology_version": s.ontology_version,
        "camera_id": s.calibration.camera_id,
        "streams": [
            {
                "stream_id": x.stream_id,
                "kind": x.kind.value,
                "sync_method": x.sync_method.value if x.sync_method else None,
            }
            for x in s.streams
        ],
    }


@dataclass
class SessionState:
    """단계 활성 규칙(`dlp_console.steps`)이 보는 세션 상태."""

    session: Session
    video_streams: tuple[str, ...]
    # 영상 스트림 → 마지막 완료 탐지 세대 번호 (없으면 그 스트림은 없음)
    detections: dict[str, int] = field(default_factory=dict[str, int])
    tasks: list[ReviewTask] = field(default_factory=list[ReviewTask])
    rendered: dict[str, bool] = field(default_factory=dict[str, bool])
    held: bool = False
    withdrawn: bool = False

    def stage_tasks(self, stage: ReviewStage) -> list[ReviewTask]:
        return [
            t for t in self.tasks if t.stage is stage and t.status is not ReviewTaskStatus.REVOKED
        ]

    def current_privacy_tasks(self) -> list[ReviewTask]:
        """지금 탐지 세대를 본 블러 검수 작업."""
        return [
            t
            for t in self.stage_tasks(ReviewStage.PRIVACY)
            if self.detections.get(t.stream_id) == t.privacy_detection_id
        ]


def read_state(conn: sa.Connection, session_id: str, rendered: RenderedCheck) -> SessionState:
    """세션 하나의 상태. 예외: 세션이 없으면 sqlalchemy NoResultFound."""
    s = get_session(conn, session_id)
    videos = tuple(x.stream_id for x in s.streams if x.kind in VIDEO_KINDS)
    detections: dict[str, int] = {}
    for sid in videos:
        run = latest_privacy_detection(conn, session_id, sid)
        if run is not None:
            detections[sid] = run.event_id
    return SessionState(
        session=s,
        video_streams=videos,
        detections=detections,
        tasks=list_review_tasks(conn, session_id),
        rendered={sid: rendered(session_id, sid) for sid in videos},
        held=active_privacy_hold(conn, session_id) is not None,
        withdrawn=session_id in withdrawn_session_ids(conn),
    )


def list_sessions(conn: sa.Connection) -> list[dict[str, Any]]:
    """모든 세션의 짧은 화면용 값 (최근 촬영 먼저)."""
    out = [session_view(get_session(conn, sid)) for sid in list_session_ids(conn)]
    return sorted(out, key=lambda x: x["recorded_at"], reverse=True)


def label_counts(conn: sa.Connection, session_id: str) -> list[dict[str, Any]]:
    """종류·출처·검수 상태별 지금 라벨 수 (화면용 근사, 모듈 docstring)."""
    lr = label_records
    child = sa.alias(lr, "child")
    superseded = sa.exists().where(
        child.c.parent_label_id == lr.c.label_id, child.c.session_id == session_id
    )
    query = (
        sa.select(lr.c.kind, lr.c.source, lr.c.verification_state, sa.func.count())
        .where(
            lr.c.session_id == session_id,
            lr.c.retracted.is_(False),
            lr.c.seeded_error.is_(False),
            lr.c.measurement.is_(None),
            ~superseded,
        )
        .group_by(lr.c.kind, lr.c.source, lr.c.verification_state)
        .order_by(lr.c.kind)
    )
    return [
        {"kind": k, "source": src, "verification_state": v, "count": int(n)}
        for k, src, v, n in conn.execute(query)
    ]


def vlm_servers(env: dict[str, str]) -> dict[str, bool]:
    """사내 VLM 서버 설정 여부 (주소는 싣지 않는다).

    행동 구간은 DLP_VLM_URL, AI 1차 검수는 DLP_VLM_URLS를 쓴다.
    """
    return {"actions": bool(env.get("DLP_VLM_URL")), "aireview": bool(env.get("DLP_VLM_URLS"))}
