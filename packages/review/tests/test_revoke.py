# PyAV 타입 정보가 일부 비어 있어 이 모듈에서만 알 수 없는 멤버 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""블러 변경 뒤 이미 배포한 라벨링 작업 매체 회수 (F-REV-02, ADR 0039).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 LocalStore로 돈다.
도구 클라이언트는 호출 순서를 기록하는 가짜다. 영상은 정답을 아는 합성 블러 시나리오다.

정답 근거: 낡은 매체 = 작업의 렌더 해시 ≠ 지금 운영 블러의 렌더 해시(블러가 지금 바뀐 스트림은
렌더 해시 없는 구 작업 포함). 회수 순서는 라벨링 버킷 사본 → CVAT(배정 해제 → 스냅숏 → 삭제)
→ Label Studio(스냅숏 → 삭제). 지우기 전 판정이 하나라도 맞지 않으면 지우지 않는다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

from dlp_fixtures.video import generate_blur_scenario
from dlp_media.storage import LocalStore, ObjectStore, StoredObject, sha256_file
from dlp_privacy.hold import hold_session
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import operational_blur, render_hash, request_media_revocation
from dlp_review import collect as collect_module
from dlp_review import tasks as tasks_module
from dlp_review.clients import TaskOwner, ToolError
from dlp_review.collect import collect_task
from dlp_review.ops.results import finish_assignment
from dlp_review.reconcile import ReviewedItem
from dlp_review.revoke import (
    execute_media_revocations,
    mark_tool_gone,
    snapshot_key,
    timeseries_sessions,
)
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    create_labeling_tasks,
    create_privacy_tasks,
    labeling_media_key,
    lock_assignment,
    proxy_key,
    timeseries_key,
)
from dlp_review.verify import verification_gaps
from dlp_schema import load_ontology, repo_root
from dlp_schema.db import locks, repository, review_repository
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_review_task,
    insert_assignment,
    insert_labels,
    insert_review_task,
    insert_session,
    mark_review_task_collected,
    pending_revocations,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord
from dlp_schema.lineage import Withdrawal
from dlp_schema.review import (
    AssignmentStatus,
    ReviewAssignment,
    ReviewMode,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
)
from dlp_schema.session import LifecycleState, PrivacyState, Session
from dlp_schema.testing import FIXED_TIME, make_label, make_session, sqlite_db

NOW = FIXED_TIME + timedelta(hours=1)
SID = "s001"
REVIEW = f"sessions/{SID}/review"


@pytest.fixture
def policy() -> PrivacyPolicy:
    """저장소의 프라이버시 정책 (렌더 해시 계산에 쓴다)."""
    return load_privacy_policy(repo_root())


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """표와 온톨로지 v1을 만든 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    yield eng
    eng.dispose()


@pytest.fixture
def stores(tmp_path: Path) -> tuple[LocalStore, LocalStore]:
    """(원본, 라벨링) 로컬 저장소. 버킷 이름은 서로 다르다."""
    return LocalStore(tmp_path / "store", "dlp-raw"), LocalStore(tmp_path / "store", "dlp-labeling")


def _video(tmp_path: Path) -> Path:
    """짧은 합성 블러 시나리오 영상 (PTS를 아는 VFR)."""
    path = tmp_path / "scenario.mp4"
    if not path.exists():
        generate_blur_scenario(0, session_id=SID, duration_ms=300).write(path)
    return path


def _session(raw: LocalStore, tmp_path: Path, state: PrivacyState) -> Session:
    """바디캠 + 3인칭 두 영상 스트림 세션. 원본과 검수 프록시를 원본 버킷에 둔다."""
    video = _video(tmp_path)
    streams: list[dict[str, Any]] = []
    for sid, kind, sync in (
        ("bodycam", "bodycam", "reference"),
        ("third", "third_person", "manual"),
    ):
        key = f"sessions/{SID}/raw/{sid}.mp4"
        raw.put_file(key, video, sha256_file(video))
        raw.put_file(proxy_key(SID, sid), video, sha256_file(video))
        streams.append({"stream_id": sid, "kind": kind, "sync_method": sync, "uri": raw.uri(key)})
    return make_session(SID, privacy_state=state, streams=streams)


def _put_media(store: LocalStore, key: str, tmp_path: Path) -> str:
    """라벨링 버킷에 작업 매체 사본을 둔다. URI."""
    video = _video(tmp_path)
    store.put_file(key, video, sha256_file(video))
    return store.uri(key)


def _task(
    key: str, tool: ReviewTool, stream: str, stage: ReviewStage, uri: str, **kw: Any
) -> ReviewTask:
    """시험용 작업 (외부 ID는 키의 숫자)."""
    return ReviewTask(
        task_key=key, tool=tool, external_id=key.split(":")[1], session_id=SID, stream_id=stream,
        stage=stage, media_uri=uri, label_kinds=("blur_track",) if stage is ReviewStage.PRIVACY
        else ("box_track",), created_at=FIXED_TIME, **kw,
    )  # fmt: skip


def _face(label_id: str = "b-new", stream: str = "bodycam") -> LabelRecord:
    """사람이 그린 얼굴 블러 (운영 블러가 바뀐다)."""
    payload = {
        "kind": "blur_track",
        "target": "face",
        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
    }
    return make_label(payload, label_id=label_id, stream_id=stream, t_end_ms=100)


def _face_item() -> ReviewedItem:
    """블러 검수자가 새로 그린 얼굴 (원래 라벨 없음)."""
    payload = BlurTrackPayload(
        target="face", keyframes=(BoxKeyframe(t_ms=0, x=1.0, y=1.0, w=10.0, h=10.0),)
    )
    return ReviewedItem(None, "bodycam", 0, 100, payload)


def _current(engine: sa.Engine, policy: PrivacyPolicy, stream: str) -> str:
    """지금 운영 블러의 렌더 해시."""
    with engine.connect() as conn:
        return render_hash(
            operational_blur(get_labels(conn, SID, kinds=["blur_track"]), stream), policy
        )


def _assignment(
    aid: str, stream: str | None = "bodycam", task_key: str | None = None
) -> ReviewAssignment:
    """열린 표준 배정."""
    return ReviewAssignment(
        assignment_id=aid, session_id=SID, stream_id=stream, label_kinds=("box_track",),
        mode=ReviewMode.STANDARD, priority=1.0, assignee="labeler01", task_key=task_key,
        created_at=FIXED_TIME,
    )  # fmt: skip


@pytest.fixture
def seven(
    engine: sa.Engine,
    stores: tuple[LocalStore, LocalStore],
    tmp_path: Path,
    policy: PrivacyPolicy,
) -> dict[str, ReviewTask]:
    """승인된 세션과 작업 7종. 매체는 라벨링 버킷에 있다.

    블러 CVAT, 바디캠 CVAT 배정 A, 바디캠 LS 직접, 3인칭 CVAT, 바디캠 수집됨, 바디캠 구 작업
    수집됨, 바디캠 LS와 같은 매체를 쓰는 작업.
    """
    raw, labeling = stores
    session = _session(raw, tmp_path, PrivacyState.APPROVED)
    with engine.begin() as conn:
        insert_session(conn, session)
    body, third = _current(engine, policy, "bodycam"), _current(engine, policy, "third")

    def media(who: str, stream: str, rhash: str | None) -> str:
        # 작업 생성과 같은 키 함수 (해시 글자 수를 실행기와 공유한다). 구 작업은 해시 없는 키
        legacy = f"{REVIEW}/{who}/{stream}.mp4"
        key = labeling_media_key(SID, who, stream, rhash) if rhash else legacy
        return _put_media(labeling, key, tmp_path)

    lab, cvat, ls = ReviewStage.LABELING, ReviewTool.CVAT, ReviewTool.LABEL_STUDIO
    shared = media("labeler02", "bodycam", body)
    proxy = raw.uri(proxy_key(SID, "bodycam"))
    tasks = {
        "blur": _task("cvat:1", cvat, "bodycam", ReviewStage.PRIVACY, proxy),
        "body_a": _task("cvat:2", cvat, "bodycam", lab, media("labeler01", "bodycam", body),
                        render_hash=body, assignee="labeler01", assignment_id="A"),
        "body_ls": _task("label_studio:3", ls, "bodycam", lab, shared, render_hash=body,
                         assignee="labeler02"),
        "third": _task("cvat:4", cvat, "third", lab, media("labeler01", "third", third),
                       render_hash=third, assignee="labeler01"),
        "collected": _task("cvat:5", cvat, "bodycam", lab, media("labeler03", "bodycam", body),
                           render_hash=body, assignee="labeler03"),
        "legacy": _task("cvat:6", cvat, "bodycam", lab, media("labeler04", "bodycam", None),
                        assignee="labeler04"),
        "shared": _task("cvat:7", cvat, "bodycam", lab, shared, render_hash=body,
                        assignee="labeler02"),
    }  # fmt: skip
    with engine.begin() as conn:
        for t in tasks.values():
            insert_review_task(conn, t)
        insert_assignment(conn, _assignment("A", task_key="cvat:2"))
        mark_review_task_collected(conn, "cvat:5", FIXED_TIME)
        mark_review_task_collected(conn, "cvat:6", FIXED_TIME)
    return tasks


def _status(engine: sa.Engine, key: str) -> ReviewTask:
    """DB의 작업."""
    with engine.connect() as conn:
        return get_review_task(conn, key)


# ── 선택과 요청 ──


def test_blur_change_requests_every_bodycam_labeling_task(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """바디캠 블러가 바뀌면 바디캠 비프라이버시 작업이 모두 요청된다 (상태·방식 무관, 구 작업 포함).

    열린 작업은 revoked, 수집한 작업은 상태를 두고 요청 시각만 남는다.
    블러 검수·3인칭 작업은 그대로다.
    배정 A는 열린 채로 두되 `lock_assignment`가 "작업 없음"(None)으로 본다. 다시 불러도 그대로다.
    """
    with engine.begin() as conn:
        insert_labels(conn, [_face()])
        got = request_media_revocation(
            conn, SID, policy, NOW, reason="blur_changed:cvat:1", streams={"bodycam"}, legacy=True
        )
    assert sorted(t.task_key for t in got) == [
        "cvat:2",
        "cvat:5",
        "cvat:6",
        "cvat:7",
        "label_studio:3",
    ]
    # 돌려준 사본은 요청 전 상태다 (다시 만들 작업을 고르는 데 쓴다)
    assert {t.task_key: t.status for t in got}["cvat:2"] is ReviewTaskStatus.OPEN
    for key in ("cvat:2", "label_studio:3", "cvat:7"):
        assert _status(engine, key).status is ReviewTaskStatus.REVOKED
    for key in ("cvat:5", "cvat:6"):
        task = _status(engine, key)
        assert task.status is ReviewTaskStatus.COLLECTED and task.revoke_requested_at == NOW
    assert _status(engine, "cvat:1").revoke_requested_at is None
    assert _status(engine, "cvat:4").status is ReviewTaskStatus.OPEN
    with engine.begin() as conn:
        assert get_assignment(conn, "A").status is AssignmentStatus.OPEN
        assert lock_assignment(conn, "A") is None
        again = request_media_revocation(
            conn, SID, policy, NOW, reason="again", streams={"bodycam"}, legacy=True
        )
    assert again == []
    assert _status(engine, "cvat:2").revoke_reason == "blur_changed:cvat:1"


def test_selection_rules_hash_legacy_force_and_shared_media(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """해시가 같으면(0건 재탐지) 요청하지 않는다. sweep(legacy=False)은 구 작업을 건너뛴다.

    force(사고 대응)는 해시와 무관하게 그 스트림 작업을 모두 고른다. 같은 매체를 쓰는 작업은
    `:shared-media` 사유로 함께 요청된다.
    """
    with engine.begin() as conn:
        assert request_media_revocation(conn, SID, policy, NOW, reason="sweep") == []
        insert_labels(conn, [_face()])
        # 렌더 해시가 있는 작업만 낡은 것으로 고른다. 구 작업(cvat:6)은 건너뛴다
        swept = request_media_revocation(conn, SID, policy, NOW, reason="sweep")
        assert sorted(t.task_key for t in swept) == ["cvat:2", "cvat:5", "cvat:7", "label_studio:3"]
        forced = request_media_revocation(
            conn, SID, policy, NOW, reason="manual:ops:incident", streams={"third"}, force=True
        )
        assert [t.task_key for t in forced] == ["cvat:4"]
    assert _status(engine, "cvat:6").revoke_requested_at is None
    assert _status(engine, "cvat:4").revoke_reason == "manual:ops:incident"


def test_shared_media_is_requested_with_its_reason(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """지정한 작업과 같은 매체를 쓰는 작업은 `<사유>:shared-media`로 함께 요청된다."""
    with engine.begin() as conn:
        got = request_media_revocation(
            conn, SID, policy, NOW, reason="manual:ops:x", task_keys=("label_studio:3",)
        )
    assert sorted(t.task_key for t in got) == ["cvat:7", "label_studio:3"]
    assert _status(engine, "cvat:7").revoke_reason == "manual:ops:x:shared-media"
    assert _status(engine, "label_studio:3").revoke_reason == "manual:ops:x"


def test_named_tasks_are_requested_alone_even_when_others_are_stale(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """수동 선택(task_keys)은 지정한 작업(과 같은 매체 작업)만 고른다. 낡은 작업을 섞지 않는다.

    낡은 작업을 함께 고르면 그 사유가 사고 대응 수동 요청(`manual:…`)으로 잘못 남는다. 낡은 작업은
    따로 sweep으로 요청해 사유가 `sweep`으로 남는다 (`dlp review revoke`가 그렇게 부른다).
    """
    with engine.begin() as conn:
        insert_labels(conn, [_face()])  # 바디캠 작업이 낡았다
        got = request_media_revocation(
            conn, SID, policy, NOW, reason="manual:ops:x", task_keys=("cvat:4",)
        )
        assert [t.task_key for t in got] == ["cvat:4"]
        swept = request_media_revocation(conn, SID, policy, NOW, reason="sweep")
    assert sorted(t.task_key for t in swept) == ["cvat:2", "cvat:5", "cvat:7", "label_studio:3"]
    assert _status(engine, "cvat:4").revoke_reason == "manual:ops:x"
    assert _status(engine, "cvat:2").revoke_reason == "sweep"


# ── 실행기 ──


class Calls(list[tuple[str, str, str]]):
    """(채널, 동작, 대상) 호출 기록."""


class RecordingStore:
    """LocalStore를 감싸 삭제를 기록한다 (DeletableStore)."""

    def __init__(self, inner: LocalStore, calls: Calls) -> None:
        self.inner, self.calls, self.bucket = inner, calls, inner.bucket

    def uri(self, key: str) -> str:
        return self.inner.uri(key)

    def head(self, key: str) -> StoredObject | None:
        return self.inner.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        self.inner.put_file(key, path, sha256)

    def get_file(self, key: str, dest: Path) -> None:
        self.inner.get_file(key, dest)

    def delete(self, key: str) -> None:
        self.calls.append(("s3", "delete", key))
        self.inner.delete(key)


class NoDeleteStore:
    """삭제가 없는 저장소 (ObjectStore만)."""

    def __init__(self, inner: LocalStore) -> None:
        self.inner, self.bucket = inner, inner.bucket

    def uri(self, key: str) -> str:
        return self.inner.uri(key)

    def head(self, key: str) -> StoredObject | None:
        return self.inner.head(key)

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        self.inner.put_file(key, path, sha256)

    def get_file(self, key: str, dest: Path) -> None:
        self.inner.get_file(key, dest)


# seven 픽스처 작업의 도구 쪽 주인 (CVAT는 작업 이름 `<세션>/<스트림>/<담당자>`, LS는 스트림 없음)
SEVEN_OWNERS: dict[str, dict[int, TaskOwner]] = {
    "cvat": {
        i: TaskOwner(SID, stream, who, f"{SID}/{stream}/{who}")
        for i, stream, who in (
            (1, "bodycam", "privacy"),
            (2, "bodycam", "labeler01"),
            (4, "third", "labeler01"),
            (5, "bodycam", "labeler03"),
            (6, "bodycam", "labeler04"),
            (7, "bodycam", "labeler02"),
        )
    },
    "ls": {3: TaskOwner(SID, None, "labeler02", "s001/labeler02")},
}


class FakeTool:
    """호출을 기록하는 가짜 도구. tasks: 살아 있는 작업 ID. fail: 동작 → 남은 실패 횟수.

    owners: 작업 ID → 도구에 저장된 주인 (기본은 seven 픽스처와 같다).
    """

    def __init__(self, name: str, calls: Calls, tasks: set[int]) -> None:
        self.name, self.calls, self.tasks = name, calls, tasks
        self.fail: dict[str, int] = {}
        self.undeletable: set[int] = set()
        self.owners = dict(SEVEN_OWNERS[name])

    def _maybe_fail(self, op: str) -> None:
        if self.fail.get(op, 0) > 0:
            self.fail[op] -= 1
            raise ToolError(f"{self.name} {op} 실패")

    def task_exists(self, task_id: int) -> bool:
        return task_id in self.tasks

    def task_owner(self, task_id: int) -> TaskOwner:
        self.calls.append((self.name, "owner", str(task_id)))
        return self.owners.get(task_id, TaskOwner(None, None, None, "?"))

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        self.calls.append((self.name, "snapshot", str(task_id)))
        self._maybe_fail("snapshot")
        return {"annotations": [{"id": task_id}]}

    def delete_task(self, task_id: int) -> bool:
        self.calls.append((self.name, "delete", str(task_id)))
        self._maybe_fail("delete")
        if task_id in self.undeletable:
            return True
        found = task_id in self.tasks
        self.tasks.discard(task_id)
        return found

    def unassign(self, task_id: int) -> None:
        self.calls.append((self.name, "unassign", str(task_id)))
        self._maybe_fail("unassign")


def _requested(engine: sa.Engine, policy: PrivacyPolicy, *, add_blur: bool = True) -> None:
    """바디캠 블러를 바꾸고(add_blur) 회수를 요청한다 (바디캠 작업 다섯)."""
    with engine.begin() as conn:
        if add_blur:
            insert_labels(conn, [_face()])
        request_media_revocation(
            conn, SID, policy, NOW, reason="blur_changed:cvat:1", streams={"bodycam"}, legacy=True
        )


def _run(
    engine: sa.Engine, labeling: ObjectStore, cvat: FakeTool | None, ls: FakeTool | None
) -> Any:
    """실행기를 돌린다 (원본 버킷 이름은 dlp-raw)."""
    return execute_media_revocations(
        engine, SID, labeling=labeling, raw_bucket="dlp-raw", cvat=cvat, label_studio=ls,
        clock=lambda: NOW,
    )  # fmt: skip


def test_executor_deletes_media_before_tools_in_order(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """S3 사본 삭제가 모든 도구 호출보다 먼저다.

    CVAT는 배정 해제 → 스냅숏 → 삭제, LS는 스냅숏 → 삭제.

    같은 매체를 쓰는 작업 둘(label_studio:3, cvat:7)의 사본은 한 번만 지운다. 끝나면 대기가 없고,
    블러본·3인칭 매체는 남는다. 스냅숏에는 주석이 있다.
    """
    _, labeling = stores
    _requested(engine, policy)
    calls = Calls()
    cvat = FakeTool("cvat", calls, {2, 5, 6, 7, 4, 1})
    ls = FakeTool("ls", calls, {3})
    report = _run(engine, RecordingStore(labeling, calls), cvat, ls)
    assert report.complete and report.errors == [] and report.warnings == []
    deletes = [c for c in calls if c[0] == "s3"]
    assert len(deletes) == 4  # 서로 다른 매체 URI 넷 (공유 매체는 한 번)
    first_tool = next(i for i, c in enumerate(calls) if c[0] != "s3")
    assert all(c[0] == "s3" for c in calls[:first_tool]) and len(calls[:first_tool]) == 4
    for tid in ("2", "5", "6", "7"):
        mine = [c[1] for c in calls if c[0] == "cvat" and c[2] == tid]
        assert mine == ["owner", "unassign", "snapshot", "delete"]
    assert [c[1] for c in calls if c[0] == "ls"] == ["owner", "snapshot", "delete"]
    assert cvat.tasks == {1, 4} and ls.tasks == set()
    for t in (seven["body_a"], seven["body_ls"], seven["legacy"]):
        assert labeling.head(t.media_uri.removeprefix(labeling.uri(""))) is None
        assert _status(engine, t.task_key).media_revoked_at == NOW
    assert labeling.head(seven["third"].media_uri.removeprefix(labeling.uri(""))) is not None
    snap = labeling.root / snapshot_key(seven["body_ls"])
    assert json.loads(snap.read_text())["tool_task"]["annotations"] == [{"id": 3}]
    with engine.connect() as conn:
        assert pending_revocations(conn, SID) == []


def test_partial_failure_leaves_only_failed_channel_pending_then_retry_finishes(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """CVAT 삭제가 실패하면 매체·LS는 끝나고 그 CVAT 작업만 대기로 남는다. 다시 돌리면 끝난다.

    재시도에서 스냅숏은 덮어쓰지 않는다 (이전 실행의 사본을 지킨다).
    """
    _, labeling = stores
    _requested(engine, policy)
    calls = Calls()
    cvat = FakeTool("cvat", calls, {2, 5, 6, 7})
    cvat.fail["delete"] = 1
    ls = FakeTool("ls", calls, {3})
    store = RecordingStore(labeling, calls)
    report = _run(engine, store, cvat, ls)
    assert report.pending == ["cvat:2"] and len(report.errors) == 1
    assert _status(engine, "cvat:2").media_revoked_at == NOW
    assert _status(engine, "label_studio:3").tool_revoked_at == NOW
    before = (labeling.root / snapshot_key(seven["body_a"])).read_text()
    calls.clear()
    again = _run(engine, store, cvat, ls)
    assert again.complete and again.snapshots == []
    assert (labeling.root / snapshot_key(seven["body_a"])).read_text() == before
    assert [c for c in calls if c[0] == "s3"] == []  # 매체는 이미 끝났다


def test_snapshot_failure_still_deletes_and_lingering_task_stays_pending(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """스냅숏이 실패해도 삭제한다 (노출 차단 우선). 삭제 뒤에도 작업이 보이면 대기로 남는다."""
    _, labeling = stores
    _requested(engine, policy)
    calls = Calls()
    cvat = FakeTool("cvat", calls, {2, 5, 6, 7})
    cvat.fail["snapshot"] = 4
    cvat.undeletable.add(7)
    ls = FakeTool("ls", calls, set())  # 이미 없는 작업 (404) = 성공
    report = _run(engine, RecordingStore(labeling, calls), cvat, ls)
    assert len(report.warnings) == 4
    assert cvat.tasks == {7} and report.pending == ["cvat:7"]
    # 없는 작업은 주인 확인·스냅숏·삭제를 부르지 않는다
    assert [c[1] for c in calls if c[0] == "ls"] == []


def test_withdrawn_session_revocation_also_deletes_timeseries_csv(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
    tmp_path: Path,
) -> None:
    """사용 중지 회수는 세션의 검수 시계열 CSV도 지운다. 블러 변경 회수는 CSV를 둔다.

    정답 근거: CSV는 동의를 철회한 작업자의 장갑·IMU 값이고 이미 발급한 Label Studio 서명 URL이
    가리킨다 (ADR 0039: 회수는 매체 객체 삭제로 끊는다). 블러 변경은 얼굴이 없는 CSV를 회수하지
    않는다. 사용 중지 전에 블러 변경으로 바디캠 작업이 이미 회수됐어도, 사용 중지 뒤 실행기가 CSV를
    지운다. CSV를 지우지 못하면 작업 매체 표시와 별도로 실행이 미완료(`timeseries_left`, 대기)로
    끝나고 다시 돌리면 끝난다 (CSV에는 작업 행 표시가 없어 실행마다 확인한다).
    """
    _, labeling = stores
    csv = timeseries_key(SID)
    _put_media(labeling, csv, tmp_path)
    _requested(engine, policy)  # 블러 변경: 바디캠 작업만
    calls = Calls()
    cvat, ls = FakeTool("cvat", calls, {2, 4, 5, 6, 7}), FakeTool("ls", calls, {3})
    assert _run(engine, labeling, cvat, ls).complete
    assert labeling.head(csv) is not None  # 블러 변경은 CSV를 두었다

    class FailingCsv(RecordingStore):
        """시계열 CSV 삭제를 한 번 실패시킨다."""

        failed = False

        def delete(self, key: str) -> None:
            if key == csv and not self.failed:
                self.failed = True
                raise OSError("디스크 오류")
            super().delete(key)

    with engine.begin() as conn:
        repository.set_lifecycle(conn, SID, LifecycleState.WITHDRAWN, at=NOW, actor="dpo01")
        assert request_task_revocation(conn, "cvat:4", reason="withdrawn:dpo01", at=NOW)
    store = FailingCsv(labeling, calls)
    first = _run(engine, store, cvat, ls)
    assert not first.complete and first.pending == [] and first.timeseries_left == [csv]
    assert any(e.startswith("[매체 노출 남음]") and csv in e for e in first.errors)
    assert labeling.head(csv) is not None
    assert _status(engine, "cvat:4").media_revoked_at == NOW  # 작업 매체는 지웠다
    again = _run(engine, store, cvat, ls)
    assert again.complete and again.media_done == [csv]
    assert labeling.head(csv) is None
    assert labeling.head(seven["third"].media_uri.removeprefix(labeling.uri(""))) is None


def test_withdrawn_session_csv_is_deleted_even_when_no_revocation_is_pending(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
    tmp_path: Path,
) -> None:
    """사용 중지 전에 모든 비프라이버시 작업의 회수가 끝났어도 사용 중지 뒤 실행기가 CSV를 지운다.

    정답 근거: 바디캠·3인칭 블러를 모두 바꿔 비프라이버시 작업 6개가 블러 변경으로 회수됐다(CSV는
    둔다). 그 뒤 사용 중지는 새로 요청할 작업이 없어 회수 대기가 비지만, CSV는 동의를 철회한
    작업자의 센서 값이라 지워야 한다. 대기 유무와 무관하게 사용 중지 세션의 CSV를 확인해 지운다:
    모든 세션(`--pending`, 사용 중지 기록으로 찾는다)과 세션 지정 모두. 다시 돌리면 할 일이 없다.
    """
    _, labeling = stores
    csv = timeseries_key(SID)
    _put_media(labeling, csv, tmp_path)
    with engine.begin() as conn:
        insert_labels(conn, [_face(), _face("b-new3", "third")])
        request_media_revocation(
            conn, SID, policy, NOW, reason="blur_changed:cvat:1", streams={"bodycam", "third"},
            legacy=True,
        )  # fmt: skip
    calls = Calls()
    cvat, ls = FakeTool("cvat", calls, {2, 4, 5, 6, 7}), FakeTool("ls", calls, {3})
    assert _run(engine, labeling, cvat, ls).complete
    assert labeling.head(csv) is not None
    with engine.begin() as conn:
        assert all(
            t.revoke_requested_at is not None
            for t in repository.list_review_tasks(conn, SID)
            if t.stage is not ReviewStage.PRIVACY
        )
        repository.set_lifecycle(conn, SID, LifecycleState.WITHDRAWN, at=NOW, actor="dpo01")
        repository.insert_withdrawal(
            conn, Withdrawal(session_id=SID, reason="동의 철회", withdrawn_at=NOW)
        )
        assert pending_revocations(conn, None) == []
        assert timeseries_sessions(conn, None) == [SID] == timeseries_sessions(conn, SID)
    report = execute_media_revocations(
        engine, None, labeling=labeling, raw_bucket="dlp-raw", cvat=cvat, label_studio=ls,
        clock=lambda: NOW,
    )  # fmt: skip
    assert report.complete and report.media_done == [csv] and report.errors == []
    assert labeling.head(csv) is None
    _put_media(labeling, csv, tmp_path)  # 같은 세션 지정 실행도 지운다
    assert _run(engine, labeling, cvat, ls).media_done == [csv]
    again = _run(engine, labeling, cvat, ls)
    assert again.complete and again.media_done == [] and labeling.head(csv) is None


def test_executor_refuses_to_delete_when_any_precheck_fails(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
    tmp_path: Path,
) -> None:
    """지우기 전 판정이 실패하면 지우지 않고 대기로 남긴다 (삭제 0회).

    삭제 없는 저장소, 라벨링 = 원본 버킷, 다른 버킷 URI, 키 모양 밖(시계열 CSV·블러본),
    회수 요청 없는 작업이 같은 키를 쓰는 경우. 도구 클라이언트가 없으면 도구 작업도 대기다.
    """
    raw, labeling = stores
    with engine.begin() as conn:
        insert_labels(conn, [_face()])
        for key in ("cvat:2", "label_studio:3", "cvat:7", "cvat:5", "cvat:6"):
            request_task_revocation(conn, key, reason="x", at=NOW)
    calls = Calls()
    store = RecordingStore(labeling, calls)
    # 삭제 없는 저장소
    report = _run(engine, NoDeleteStore(labeling), None, None)
    assert len(report.pending) == 5 and all(
        "삭제를 지원하지 않" in e or "클라이언트 없음" in e for e in report.errors
    )
    # 라벨링 저장소가 원본 버킷
    same = RecordingStore(LocalStore(tmp_path / "store", "dlp-raw"), calls)
    assert "설정 오류" in _run(engine, same, None, None).errors[0]
    assert calls == []
    # 다른 버킷 URI·키 모양 밖·같은 키를 쓰는 미요청 작업
    weird = {
        "cvat:2": raw.uri(proxy_key(SID, "bodycam")),
        "cvat:5": labeling.uri(f"{REVIEW}/timeseries.csv"),
        "cvat:6": labeling.uri(f"sessions/{SID}/blurred/bodycam.mp4"),
        "label_studio:3": seven["third"].media_uri,
    }
    with engine.begin() as conn:
        for key, uri in weird.items():
            conn.execute(
                sa.text("UPDATE review_tasks SET media_uri=:u WHERE task_key=:k"),
                {"u": uri, "k": key},
            )
        conn.execute(sa.text("UPDATE review_tasks SET media_uri=:u WHERE task_key='cvat:7'"),
                     {"u": seven["third"].media_uri})  # fmt: skip
    report = _run(engine, store, None, None)
    assert [c for c in calls if c[0] == "s3"] == []
    assert len([e for e in report.errors if e.startswith("[매체 노출 남음]")]) == 4
    assert len(report.pending) == 5


def test_executor_never_deletes_a_tool_task_owned_by_another_row(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """도구 작업의 주인이 작업 행과 다르면 배정 해제·스냅숏·삭제를 하지 않고 대기로 남긴다.

    시나리오: CVAT·LS 인스턴스를 새로 만들고 DB는 남겼다. 같은 번호 2·3이 다른 세션·라벨러의 진행
    중 작업이다 (CVAT 이름 `s999/bodycam/labeler09`, LS data의 담당자 다름). 매체 사본은 지운다
    (라벨링 버킷은 그대로라 그 판정은 따로다). 운영자가 원래 작업이 없어진 것을 확인해
    `mark_tool_gone`으로 표시하면 대기가 끝난다. 주인이 같은 작업(5·6·7)은 평소처럼 지운다.
    """
    _, labeling = stores
    _requested(engine, policy)
    calls = Calls()
    cvat = FakeTool("cvat", calls, {2, 5, 6, 7})
    cvat.owners[2] = TaskOwner("s999", "bodycam", "labeler09", "s999/bodycam/labeler09")
    ls = FakeTool("ls", calls, {3})
    ls.owners[3] = TaskOwner(SID, None, "labeler77", "s001/labeler77")
    report = _run(engine, RecordingStore(labeling, calls), cvat, ls)
    assert [c[1] for c in calls if c[2] == "2" and c[0] == "cvat"] == ["owner"]
    assert [c[1] for c in calls if c[0] == "ls"] == ["owner"]
    assert cvat.tasks == {2} and ls.tasks == {3}
    assert sorted(report.pending) == ["cvat:2", "label_studio:3"]
    assert any("s999/bodycam/labeler09" in e and "--mark-tool-gone" in e for e in report.errors)
    assert _status(engine, "cvat:2").media_revoked_at == NOW  # 매체는 끊었다
    assert _status(engine, "cvat:2").tool_revoked_at is None
    assert labeling.head(snapshot_key(seven["body_a"])) is None  # 남의 작업은 스냅숏도 없다
    with engine.begin() as conn:
        for key in ("cvat:2", "label_studio:3"):
            assert mark_tool_gone(conn, key, note="CVAT·LS 재설치", actor="ops", now=NOW)
    assert _run(engine, labeling, cvat, ls).complete
    assert cvat.tasks == {2} and ls.tasks == {3}  # 남의 작업은 끝까지 그대로


def test_mark_tool_gone_needs_note_and_records_actor(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """도구 인스턴스 폐기 등은 사유와 행위자를 남겨 수동으로 도구 완료를 표시한다."""
    _requested(engine, policy)
    with engine.begin() as conn:
        with pytest.raises(ValueError, match="사유"):
            mark_tool_gone(conn, "cvat:2", note=" ", actor="ops", now=NOW)
        assert mark_tool_gone(conn, "cvat:2", note="CVAT 폐기", actor="ops", now=NOW)
        assert mark_tool_gone(conn, "cvat:2", note="again", actor="ops", now=NOW) is False
        assert mark_tool_gone(conn, "cvat:4", note="요청 없음", actor="ops", now=NOW) is False
    task = _status(engine, "cvat:2")
    assert task.tool_revoked_at == NOW and task.revoke_note == "ops: CVAT 폐기"


# ── 수집 연결 ──


def _blur_setup(engine: sa.Engine, stores: tuple[LocalStore, LocalStore], tmp_path: Path,
                policy: PrivacyPolicy, state: PrivacyState, monkeypatch: pytest.MonkeyPatch,
                ) -> tuple[ReviewSetup, str]:  # fmt: skip
    """블러 검수 CVAT 작업(cvat:1)과 CVAT 해석부 대체. (설정, 라벨링 작업 매체 URI)."""
    raw, labeling = stores
    blur = _task(
        "cvat:1", ReviewTool.CVAT, "bodycam", ReviewStage.PRIVACY,
        raw.uri(proxy_key(SID, "bodycam")), sent_label_ids=(),
    )  # fmt: skip
    with engine.begin() as conn:
        insert_session(conn, _session(raw, tmp_path, state))
        insert_review_task(conn, blur)
    monkeypatch.setattr(collect_module, "CvatSchema", MagicMock())
    monkeypatch.setattr(collect_module, "annotation_tracks", MagicMock(return_value=[]))
    monkeypatch.setattr(collect_module, "from_cvat_tracks", MagicMock(return_value=[_face_item()]))
    cvat = MagicMock()
    cvat.http.get.return_value.json.return_value = {"project_id": 1}
    setup = ReviewSetup(
        raw=raw, labeling=labeling, labeling_reader=MagicMock(), ontology=None, cvat=cvat,
        label_studio=None, privacy=policy,
    )  # fmt: skip
    uri = _put_media(labeling, f"{REVIEW}/labeler01/bodycam.mp4", tmp_path)
    return setup, uri


def test_blur_change_after_approval_revokes_labeling_tasks(
    engine: sa.Engine,
    stores: tuple[LocalStore, LocalStore],
    tmp_path: Path,
    policy: PrivacyPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """승인 뒤 블러 재검수에서 얼굴을 새로 찾으면 이미 만든 바디캠 라벨링 작업이 열려 있지 않다.

    근거: CLAUDE.md "원본(블러 안 된) 내용을 일반 라벨러 경로에 노출하지 않는다", ADR 0024
    "모든 소비자가 확인을 거친다". 이미 배포한 Label Studio 작업은 새 얼굴이 블러되지 않은 영상의
    서명 URL을 담고 있으므로 수집 결과와 같은 커밋에서 회수 요청(REVOKED)이 남아야 한다.
    """
    setup, ls_uri = _blur_setup(
        engine, stores, tmp_path, policy, PrivacyState.APPROVED, monkeypatch
    )
    # 구 형식 키(내용 주소 아님)로 만든 라벨링 작업. 객체가 라벨링 버킷에 있다.
    ls = _task(
        "label_studio:7", ReviewTool.LABEL_STUDIO, "bodycam", ReviewStage.LABELING, ls_uri,
        assignee="labeler01", sent_label_ids=(),
    )  # fmt: skip
    with engine.begin() as conn:
        insert_review_task(conn, ls)
    with engine.begin() as conn:
        outcome = collect_task(conn, "cvat:1", setup, "privacy01", NOW)
    assert outcome is not None and outcome.added == 1
    with engine.connect() as conn:
        assert get_review_task(conn, "label_studio:7").status is not ReviewTaskStatus.OPEN


def test_collect_fills_reissue_and_never_collects_revoked_tasks(
    engine: sa.Engine,
    stores: tuple[LocalStore, LocalStore],
    tmp_path: Path,
    policy: PrivacyPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이미 AUTO_BLURRED인 세션에서도 블러가 바뀌면 요청한다. 다시 만들 일을 결과에 적는다.

    배정 있는 열린 작업 → 배정 ID, 배정 없는 열린 작업 → (담당자, 스트림, 도구). 회수한 작업은
    수집해도 None이고 아무것도 쓰지 않는다 (라벨 이력·작업 상태 그대로).
    """
    setup, uri = _blur_setup(
        engine, stores, tmp_path, policy, PrivacyState.AUTO_BLURRED, monkeypatch
    )
    with engine.begin() as conn:
        insert_review_task(
            conn,
            _task(
                "cvat:2",
                ReviewTool.CVAT,
                "bodycam",
                ReviewStage.LABELING,
                uri,
                assignee="labeler01",
                assignment_id="A",
            ),
        )
        insert_assignment(conn, _assignment("A", task_key="cvat:2"))
        insert_review_task(
            conn,
            _task(
                "label_studio:3",
                ReviewTool.LABEL_STUDIO,
                "bodycam",
                ReviewStage.LABELING,
                uri,
                assignee="labeler01",
            ),
        )
    with engine.begin() as conn:
        outcome = collect_task(conn, "cvat:1", setup, "privacy01", NOW)
    assert outcome is not None
    assert sorted(outcome.revoked) == ["cvat:2", "label_studio:3"]
    assert outcome.reissue_assignments == ["A"]
    assert outcome.direct_reissue == [("labeler01", "bodycam", "label_studio")]
    with engine.connect() as conn:
        count = len(get_labels(conn, SID))
    with engine.begin() as conn:
        assert collect_task(conn, "cvat:2", setup, "labeler01", NOW) is None
    with engine.connect() as conn:
        assert len(get_labels(conn, SID)) == count
        assert get_review_task(conn, "cvat:2").status is ReviewTaskStatus.REVOKED


# ── 작업 생성 게이트·배정 마무리·검수 완료 판정 ──


def test_create_labeling_tasks_refuses_pending_revocation_before_tools(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """회수 대기가 있는 스트림에는 도구·매체 작업 전에 TaskError. 세션을 먼저 잠근다.

    회수 요청이 없어도 렌더 해시가 지금과 다른 작업이 있으면 같은 오류다 (백스톱).
    """
    raw, labeling = stores
    locks: list[str] = []
    real_lock = repository.lock_session

    def spy(conn: sa.Connection, session_id: str) -> None:
        locks.append(session_id)
        real_lock(conn, session_id)

    monkeypatch.setattr(tasks_module, "lock_session", spy)
    cvat, ls = MagicMock(), MagicMock()
    setup = ReviewSetup(
        raw=raw, labeling=labeling, labeling_reader=MagicMock(), ontology=None, cvat=cvat,
        label_studio=ls, privacy=policy,
    )  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [_face()])  # 바디캠 작업이 낡았다 (요청은 아직 없음)
    with pytest.raises(TaskError, match="dlp review revoke"), engine.begin() as conn:
        create_labeling_tasks(conn, SID, setup, "labeler09", NOW, streams={"bodycam"})
    _requested(engine, policy, add_blur=False)
    with pytest.raises(TaskError, match="매체 회수가 끝나지 않은"), engine.begin() as conn:
        create_labeling_tasks(conn, SID, setup, "labeler09", NOW)
    assert locks == [SID, SID]
    assert cvat.mock_calls == [] and ls.mock_calls == []


def test_withdrawn_session_gets_no_new_review_tasks(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """F-REV-17: 사용 중지한 세션에는 블러 검수·라벨링 작업을 새로 만들지 않는다 (도구 호출 전).

    사용 중지가 배포된 작업을 회수한 뒤 다시 만들면(회수 뒤 다시 만들기 안내, 배정 재생성) 동의를
    철회한 영상이 다시 배포된다.
    """
    raw, labeling = stores
    cvat, ls = MagicMock(), MagicMock()
    setup = ReviewSetup(
        raw=raw, labeling=labeling, labeling_reader=MagicMock(), ontology=None, cvat=cvat,
        label_studio=ls, privacy=policy,
    )  # fmt: skip
    with engine.begin() as conn:
        repository.set_lifecycle(conn, SID, LifecycleState.WITHDRAWN, at=NOW, actor="dpo01")
    with pytest.raises(TaskError, match="사용 중지"), engine.begin() as conn:
        create_labeling_tasks(conn, SID, setup, "labeler09", NOW)
    with pytest.raises(TaskError, match="사용 중지"), engine.begin() as conn:
        create_privacy_tasks(conn, SID, setup, NOW, assignee="rev01", privacy_reviewers=("rev01",))
    assert cvat.mock_calls == [] and ls.mock_calls == []


def test_held_session_refuses_labeling_tasks_but_allows_blur_review(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
) -> None:
    """F-REV-18: 승인 보류 중에는 라벨링 작업을 도구 호출 전에 거부하고, 배포된 작업은 회수
    요청된다 (해시가 지금과 같아도). 블러 검수 작업은 보류 중에도 만들 수 있다 (보류 중에 블러를
    고친다)."""
    raw, labeling = stores
    cvat, ls = MagicMock(), MagicMock()
    setup = ReviewSetup(
        raw=raw, labeling=labeling, labeling_reader=MagicMock(), ontology=None, cvat=cvat,
        label_studio=ls, privacy=policy,
    )  # fmt: skip
    with engine.begin() as conn:
        result = hold_session(
            conn, SID, policy, NOW, reason="labeler report", actor="ops01", labeling=labeling
        )
    assert sorted(t.task_key for t in result.revocations) == [
        "cvat:2", "cvat:4", "cvat:5", "cvat:6", "cvat:7", "label_studio:3",
    ]  # fmt: skip
    with pytest.raises(TaskError, match="승인 보류 중"), engine.begin() as conn:
        create_labeling_tasks(conn, SID, setup, "labeler09", NOW)
    assert cvat.mock_calls == [] and ls.mock_calls == []
    # 블러 검수 작업 생성은 보류로 막지 않는다 (계정 연결이 없어 그 다음 단계에서 멈춘다)
    with pytest.raises(TaskError, match="CVAT 계정"), engine.begin() as conn:
        create_privacy_tasks(conn, SID, setup, NOW, assignee="rev01", privacy_reviewers=("rev01",))


def test_finish_assignment_ignores_revoked_tasks(
    engine: sa.Engine, stores: tuple[LocalStore, LocalStore], tmp_path: Path
) -> None:
    """회수한 작업만 남은 배정은 마무리하지 않고, 다시 만든 작업을 수집하면 마무리한다."""
    raw, labeling = stores
    uri = _put_media(labeling, f"{REVIEW}/labeler01/bodycam.mp4", tmp_path)
    with engine.begin() as conn:
        insert_session(conn, _session(raw, tmp_path, PrivacyState.APPROVED))
        insert_review_task(conn, _task("cvat:2", ReviewTool.CVAT, "bodycam", ReviewStage.LABELING,
                                       uri, assignment_id="A"))  # fmt: skip
        insert_assignment(conn, _assignment("A", task_key="cvat:2"))
        request_task_revocation(conn, "cvat:2", reason="x", at=NOW)
        a = get_assignment(conn, "A")
        finish_assignment(conn, a, NOW)
        assert get_assignment(conn, "A").status is AssignmentStatus.OPEN
        insert_review_task(conn, _task("cvat:8", ReviewTool.CVAT, "bodycam", ReviewStage.LABELING,
                                       uri.replace(".mp4", "-0123456789abcdef.mp4"),
                                       assignment_id="A"))  # fmt: skip
        mark_review_task_collected(conn, "cvat:8", NOW)
        finish_assignment(conn, a, NOW)
        assert get_assignment(conn, "A").status is AssignmentStatus.DONE


def test_verification_gaps_report_revocation_work(
    engine: sa.Engine, seven: dict[str, ReviewTask], policy: PrivacyPolicy
) -> None:
    """검수 완료 판정이 회수 대기, 요청 없는 낡은 작업, 다시 만들 배정을 이유로 보인다.

    렌더 해시 없는 구 작업만 있으면 새 이유가 없다 (기존 세션을 갑자기 막지 않는다).
    """

    def reasons() -> list[str]:
        with engine.connect() as conn:
            return verification_gaps(conn, SID, policy)

    assert not any("회수" in r for r in reasons())
    with engine.begin() as conn:
        insert_labels(conn, [_face()])
    assert any(r.startswith("블러가 바뀐 뒤 회수 요청 없는 작업 4개") for r in reasons())
    _requested(engine, policy, add_blur=False)
    got = reasons()
    assert any(r.startswith("매체 회수 미완료 5개") for r in got)
    assert any(
        r.startswith("회수돼 다시 만들 배정 1개") and "dlp review assign A" in r for r in got
    )
    # AI 판정을 쓰면 렌더 뒤·다시 만들기 전에 판정을 다시 돌린다 (이전 판정은 렌더 해시가 달라
    # 보이지 않는다, ADR 0040 I10, WP17.6)
    [reissue] = [r for r in got if r.startswith("회수돼 다시 만들 배정")]
    assert f"렌더 → (AI 1차 검수를 쓰면) dlp aireview run {SID} 뒤 dlp review assign A" in reissue
    assert not any(r.startswith("블러가 바뀐 뒤") for r in got)


def test_verification_waits_for_reissue_of_revoked_direct_tasks(
    engine: sa.Engine,
    seven: dict[str, ReviewTask],
    stores: tuple[LocalStore, LocalStore],
    tmp_path: Path,
    policy: PrivacyPolicy,
) -> None:
    """배정 없이 만든 열린 작업이 회수되면, 같은 담당자·스트림·도구로 다시 만들 때까지 완료가
    아니다.

    회수 전에는 그 열린 작업이 완료를 막았다 (수거하지 않은 작업). 회수가 이 조건을 조용히 풀면
    재검수 없이 검수 완료로 넘어간다. seven에서 열린 직접 작업은 label_studio:3·cvat:7
    (labeler02)이다. 회수 요청 전에 만든 작업은 다시 만든 것으로 보지 않는다.
    """
    _, labeling = stores
    _requested(engine, policy)

    def direct() -> list[str]:
        with engine.connect() as conn:
            return [
                r
                for r in verification_gaps(conn, SID, policy)
                if r.startswith("회수돼 다시 만들 직접 작업")
            ]

    [reason] = direct()
    assert reason.startswith("회수돼 다시 만들 직접 작업 2개")
    assert f"렌더 → (AI 1차 검수를 쓰면) dlp aireview run {SID} 뒤 dlp review create" in reason
    for tool in ("label_studio", "cvat"):
        assert (
            f"dlp review create {SID} --stage labeling --assignee labeler02 --stream bodycam "
            f"--tool {tool}"
        ) in reason
    current = _current(engine, policy, "bodycam")
    uri = _put_media(labeling, labeling_media_key(SID, "labeler02", "bodycam", current), tmp_path)

    def renewed(key: str, tool: ReviewTool, at: Any) -> ReviewTask:
        made = _task(key, tool, "bodycam", ReviewStage.LABELING, uri, render_hash=current,
                     assignee="labeler02")  # fmt: skip
        return made.model_copy(update={"created_at": at})

    with engine.begin() as conn:
        # 회수 요청(NOW)보다 먼저 만든 작업은 다시 만든 작업이 아니다
        insert_review_task(conn, renewed("cvat:8", ReviewTool.CVAT, FIXED_TIME))
    assert direct()[0].startswith("회수돼 다시 만들 직접 작업 2개")
    with engine.begin() as conn:
        insert_review_task(conn, renewed("label_studio:9", ReviewTool.LABEL_STUDIO, NOW))
    assert direct()[0].startswith("회수돼 다시 만들 직접 작업 1개")
    with engine.begin() as conn:
        insert_review_task(conn, renewed("cvat:9", ReviewTool.CVAT, NOW))
    assert direct() == []


def test_collected_reissue_counts_even_after_a_later_revocation(
    engine: sa.Engine,
    stores: tuple[LocalStore, LocalStore],
    tmp_path: Path,
    policy: PrivacyPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """다시 만든 직접 작업을 수집한 뒤 블러가 한 번 더 바뀌어도 그 작업은 다시 만든 작업이다.

    1. 열린 직접 작업 label_studio:3(labeler01)이 블러 재검수 수집(cvat:1)으로 회수된다 → 수집
       출력이 다시 만들라고 한다.
    2. 같은 담당자·스트림·도구로 다시 만든 label_studio:9를 수집한다.
    3. 두 번째 블러 재검수(cvat:10)가 다른 얼굴을 더한다 → 수집한 label_studio:9에도 회수 요청이
       남지만 상태는 collected다 (결과는 이미 이력에 있다). 수집 출력은 다시 만들라고 하지 않는다.
    완료 판정도 수집 출력과 같은 기준이어야 한다: label_studio:3을 다시 만들라고 하면 거짓 차단이다.
    """
    raw, labeling = stores
    setup, uri = _blur_setup(engine, stores, tmp_path, policy, PrivacyState.APPROVED, monkeypatch)
    with engine.begin() as conn:
        direct = _task("label_studio:3", ReviewTool.LABEL_STUDIO, "bodycam", ReviewStage.LABELING,
                       uri, assignee="labeler01")  # fmt: skip
        insert_review_task(conn, direct)
    with engine.begin() as conn:
        first = collect_task(conn, "cvat:1", setup, "privacy01", NOW)
    assert first is not None
    assert first.direct_reissue == [("labeler01", "bodycam", "label_studio")]
    current = _current(engine, policy, "bodycam")
    key = labeling_media_key(SID, "labeler01", "bodycam", current)
    renewed = _task("label_studio:9", ReviewTool.LABEL_STUDIO, "bodycam", ReviewStage.LABELING,
                    _put_media(labeling, key, tmp_path), render_hash=current,
                    assignee="labeler01").model_copy(update={"created_at": NOW})  # fmt: skip
    blur_again = _task("cvat:10", ReviewTool.CVAT, "bodycam", ReviewStage.PRIVACY,
                       raw.uri(proxy_key(SID, "bodycam")), sent_label_ids=())  # fmt: skip
    later = NOW + timedelta(minutes=5)
    with engine.begin() as conn:
        insert_review_task(conn, renewed)
        mark_review_task_collected(conn, "label_studio:9", later)
        insert_review_task(conn, blur_again)
    other = BlurTrackPayload(
        target="face", keyframes=(BoxKeyframe(t_ms=0, x=40.0, y=40.0, w=10.0, h=10.0),)
    )
    monkeypatch.setattr(
        collect_module,
        "from_cvat_tracks",
        MagicMock(return_value=[ReviewedItem(None, "bodycam", 0, 100, other)]),
    )
    with engine.begin() as conn:
        second = collect_task(conn, "cvat:10", setup, "privacy01", later)
    assert second is not None and "label_studio:9" in second.revoked
    assert second.direct_reissue == []
    task = _status(engine, "label_studio:9")
    assert task.status is ReviewTaskStatus.COLLECTED and task.revoke_requested_at == later
    with engine.connect() as conn:
        gaps = verification_gaps(conn, SID, policy)
    assert not any(r.startswith("회수돼 다시 만들 직접 작업") for r in gaps), gaps


def test_verification_ignores_legacy_tasks_without_hash(
    engine: sa.Engine, stores: tuple[LocalStore, LocalStore], tmp_path: Path, policy: PrivacyPolicy
) -> None:
    """렌더 해시 없는 구 작업은 블러가 바뀌어도 판정 이유에 넣지 않는다 (수동 정리 대상)."""
    raw, labeling = stores
    uri = _put_media(labeling, f"{REVIEW}/labeler01/bodycam.mp4", tmp_path)
    with engine.begin() as conn:
        insert_session(conn, _session(raw, tmp_path, PrivacyState.APPROVED))
        insert_review_task(
            conn, _task("cvat:2", ReviewTool.CVAT, "bodycam", ReviewStage.LABELING, uri)
        )
        mark_review_task_collected(conn, "cvat:2", NOW)
        insert_labels(conn, [_face()])
        assert not any("회수" in r for r in verification_gaps(conn, SID, policy))


@pytest.mark.parametrize("b_first", [False, True])
def test_concurrent_runner_never_deletes_media_of_a_recreated_task(
    tmp_path: Path,
    stores: tuple[LocalStore, LocalStore],
    policy: PrivacyPolicy,
    monkeypatch: pytest.MonkeyPatch,
    b_first: bool,
) -> None:
    """F-REV-29: 회수 실행기 둘이 겹쳐도 다시 만든 작업의 매체(같은 키)를 지우지 않는다.

    흉내 내는 순서: 실행기 A가 대기를 읽는다 → (그 사이) 실행기 B가 같은 회수를 끝내고 생성
    게이트가 열려 같은 렌더 해시·매체 키로 새 작업이 만들어진다(B) → A가 이어서 지운다.
    작업 생성은 세션 잠금을 쥐고 올린다. 그래서 A가 세션 잠금을 쥔 동안에는 B가 끼어들 수 없고
    (잠금이 풀린 뒤로 미룬다), 잠금 전이면 바로 끼어든다(b_first). 행 잠금이 없는 SQLite라
    잠금 시점은 `lock_session` 대역으로 본다.

    정답: 어느 순서든 새 작업의 사본은 남고 새 작업에 매체 회수 표시가 없다.
    """
    from dlp_review import revoke as revoke_module

    raw, labeling = stores
    engine = sqlite_db(f"sqlite:///{tmp_path / 'race.db'}")
    current = render_hash([], policy)
    key = labeling_media_key(SID, "labeler01", "bodycam", current)
    uri = _put_media(labeling, key, tmp_path)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, _session(raw, tmp_path, PrivacyState.APPROVED))
        insert_review_task(
            conn, _task("cvat:2", ReviewTool.CVAT, "bodycam", ReviewStage.LABELING, uri,
                        render_hash=current, assignee="labeler01"),
        )  # fmt: skip
        request_task_revocation(conn, "cvat:2", reason="hold:ops01:x", at=NOW)
    new_bytes = tmp_path / "new.bin"
    new_bytes.write_bytes(b"recreated task media")
    state = {"held": False, "done": False}

    def runner_b_and_recreate() -> None:
        """B: 같은 회수를 끝내고(사본 삭제·표시), 같은 키로 새 작업과 새 사본을 만든다."""
        state["done"] = True
        labeling.delete(key)
        with engine.begin() as conn:
            repository.mark_task_media_revoked(conn, SID, uri, NOW)
            insert_review_task(
                conn, _task("cvat:9", ReviewTool.CVAT, "bodycam", ReviewStage.LABELING, uri,
                            render_hash=current, assignee="labeler01"),
            )  # fmt: skip
        labeling.put_file(key, new_bytes, sha256_file(new_bytes))

    real_list = review_repository.list_review_tasks
    real_lock = locks.lock_session if hasattr(revoke_module, "lock_session") else None

    def list_then_maybe_b(conn: sa.Connection, session_id: str) -> list[ReviewTask]:
        out = real_list(conn, session_id)
        if not state["held"] and not state["done"]:
            runner_b_and_recreate()  # 잠금 밖에서 확인한 직후 B가 끼어든다
        return out

    def lock_after_b(conn: sa.Connection, session_id: str) -> None:
        if b_first and not state["done"]:
            runner_b_and_recreate()  # B가 먼저 잠금을 쥐고 끝냈다
        state["held"] = True
        assert real_lock is not None
        real_lock(conn, session_id)

    monkeypatch.setattr(revoke_module, "list_review_tasks", list_then_maybe_b)
    if real_lock is not None:
        monkeypatch.setattr(revoke_module, "lock_session", lock_after_b)
    report = execute_media_revocations(
        engine, SID, labeling=labeling, raw_bucket="dlp-raw", cvat=None, label_studio=None,
        clock=lambda: NOW,
    )  # fmt: skip
    if not state["done"]:
        runner_b_and_recreate()  # A의 잠금이 풀린 뒤 B
    assert labeling.head(key) is not None, "다시 만든 작업의 사본이 지워졌다"
    assert (labeling.root / key).read_bytes() == b"recreated task media"
    assert not any("[매체 노출 남음]" in e for e in report.errors), report.errors
    with engine.connect() as conn:
        assert get_review_task(conn, "cvat:9").media_revoked_at is None
        assert get_review_task(conn, "cvat:2").media_revoked_at == NOW
    engine.dispose()
