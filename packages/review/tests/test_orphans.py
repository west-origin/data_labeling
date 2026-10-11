"""DB 기록 없이 남은 검수 도구 작업·라벨링 버킷 사본(고아) 찾기·지우기 (F-REV-15, R4).

서비스 없이 시간대를 보존하는 SQLite(`dlp_schema.testing.sqlite_db`)와 LocalStore로 돈다. 도구는
작업 목록과 호출을 기록하는 가짜다.

정답 근거: 이 플랫폼 이름 규칙(프로젝트 `dlp-…`·`-v<버전>-<해시 12자>`, CVAT 작업
`<세션>/<스트림>/<담당자>`, LS data의 세션·담당자, 사본
`sessions/<세션>/review/<담당자>/<스트림>…mp4`)에 맞고, DB에 같은 키의 행이 없거나(no_db_row)
다른 작업의 행이거나(other_owner) 회수 완료 표시가 있는데 남은(revoked) 것만 고아다. 살아 있는
행·회수 대기(실행기 몫)·다른 시스템 이름·이 DB에 없는 세션·나이 하한보다 어린 것은 고아가 아니다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_media.storage import LocalStore, sha256_file
from dlp_review import orphans as orphans_module
from dlp_review.clients import TaskOwner, ToolTask, cvat_owner, ls_owner
from dlp_review.ontology_projects import is_ontology_project
from dlp_review.orphans import (
    OrphanReport,
    delete_orphans,
    find_orphans,
    media_reason,
    orphan_snapshot_key,
    tool_reason,
)
from dlp_review.tasks import labeling_media_key, parse_labeling_media_key
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    insert_review_task,
    insert_session,
    mark_task_media_revoked,
    mark_task_tool_revoked,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

SID = "s001"
# 로컬 파일 mtime(지금)이 나이 하한(1시간)을 넘도록 기준 시각을 미래로 둔다
NOW = datetime.now(UTC) + timedelta(hours=2)
MIN_AGE = timedelta(hours=1)
OLD = NOW - timedelta(days=1)
SPATIAL = "dlp-labeling-spatial-v1.0.0-0123456789ab"
PRIVACY = "dlp-privacy-v1.0.0-0123456789ab"
TEMPORAL = "dlp-labeling-temporal-v1.0.0-0123456789ab"
HASH = "f" * 64


class Calls(list[tuple[str, str, str]]):
    """(채널, 동작, 대상) 호출 기록."""


class FakeTool:
    """작업 목록과 호출을 기록하는 가짜 도구 (CVAT·LS 공통, CVAT는 unassign도 쓴다)."""

    def __init__(self, name: str, calls: Calls, tasks: list[ToolTask]) -> None:
        self.name, self.calls, self.listed = name, calls, tasks
        self.alive = {t.task_id for t in tasks}

    def list_tasks(self, project: Callable[[str], bool]) -> list[ToolTask]:
        return [t for t in self.listed if project(t.project) and t.task_id in self.alive]

    def task_exists(self, task_id: int) -> bool:
        return task_id in self.alive

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        self.calls.append((self.name, "snapshot", str(task_id)))
        return {"annotations": [{"id": task_id}]}

    def delete_task(self, task_id: int) -> bool:
        self.calls.append((self.name, "delete", str(task_id)))
        found = task_id in self.alive
        self.alive.discard(task_id)
        return found

    def unassign(self, task_id: int) -> None:
        self.calls.append((self.name, "unassign", str(task_id)))


class RecordingStore(LocalStore):
    """삭제를 기록하는 라벨링 버킷 (LocalStore: 나열·삭제 지원)."""

    def __init__(self, root: Path, bucket: str, calls: Calls) -> None:
        super().__init__(root, bucket)
        self.calls = calls

    def delete(self, key: str) -> None:
        self.calls.append(("s3", "delete", key))
        super().delete(key)


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """온톨로지 v1과 세션 s001(승인됨)을 넣은 메모리 SQLite."""
    eng = sqlite_db()
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session(SID))
    yield eng
    eng.dispose()


def _cvat(tid: int, name: str, project: str = SPATIAL, at: datetime = OLD) -> ToolTask:
    """CVAT 작업 목록 한 줄 (주인은 실제 클라이언트와 같은 규칙으로 이름에서 읽는다)."""
    return ToolTask(tid, cvat_owner(name), at, project)


def _ls(tid: int, who: str | None, project: str = TEMPORAL, sid: str = SID) -> ToolTask:
    """Label Studio 작업 목록 한 줄 (data의 세션·담당자)."""
    data: dict[str, Any] = {"session_id": sid, "video": "https://signed"}
    if who is not None:
        data["assignee"] = who
    return ToolTask(tid, ls_owner(data), OLD, project)


def _row(key: str, who: str, uri: str, **kw: Any) -> ReviewTask:
    """바디캠 라벨링 작업 행 (외부 ID는 키의 숫자)."""
    tool = ReviewTool.CVAT if key.startswith("cvat:") else ReviewTool.LABEL_STUDIO
    return ReviewTask(
        task_key=key, tool=tool, external_id=key.split(":")[1], session_id=SID,
        stream_id="bodycam", stage=ReviewStage.LABELING, assignee=who, media_uri=uri,
        label_kinds=("box_track",), created_at=FIXED_TIME, render_hash=HASH, **kw,
    )  # fmt: skip


def _put(store: LocalStore, key: str, tmp_path: Path) -> str:
    """라벨링 버킷에 객체를 둔다. URI."""
    src = tmp_path / "obj.bin"
    src.write_bytes(key.encode())
    store.put_file(key, src, sha256_file(src))
    return store.uri(key)


@pytest.fixture
def world(engine: sa.Engine, tmp_path: Path) -> dict[str, Any]:
    """DB 행·도구 작업·라벨링 버킷 객체의 정답을 아는 구성.

    고아(정답): 매체 labeler02(행 없음)·labeler03(회수 완료인데 남음), CVAT 11(회수 완료인데
    남음)·13(행은 다른 담당자의 것)·20(행 없음), LS 31(행 없음).
    고아 아님: CVAT 10(살아 있음)·12(회수 대기)·14(블러 검수, 살아 있음)·21(데모 프로젝트)·
    22(이 DB에 없는 세션)·23(다른 이름 규칙)·24(어림), LS 30(살아 있음)·32(데모)·33(담당자 없음),
    매체 labeler01(살아 있음)·labeler04(회수 대기), 시계열 CSV·블러본·회수 스냅숏.
    """
    calls = Calls()
    labeling = RecordingStore(tmp_path / "store", "dlp-labeling", calls)

    def media(who: str) -> str:
        return _put(labeling, labeling_media_key(SID, who, "bodycam", HASH), tmp_path)

    live, revoked, pending = media("labeler01"), media("labeler03"), media("labeler04")
    media("labeler02")  # 행 없음
    for key in (
        f"sessions/{SID}/review/timeseries.csv",
        f"sessions/{SID}/blurred/bodycam.mp4",
        f"sessions/{SID}/revoked/cvat-11.json",
    ):
        _put(labeling, key, tmp_path)
    proxy = f"s3://dlp-raw/sessions/{SID}/derived/bodycam.proxy.mp4"
    rows = [
        _row("cvat:10", "labeler01", live),
        _row("label_studio:30", "labeler01", live),
        _row("cvat:11", "labeler03", revoked),
        _row("cvat:12", "labeler04", pending),
        _row("cvat:13", "labeler09", live),
        _row("cvat:14", "rev01", proxy).model_copy(
            update={
                "stage": ReviewStage.PRIVACY,
                "label_kinds": ("blur_track",),
                "render_hash": None,
            }
        ),
    ]
    with engine.begin() as conn:
        for r in rows:
            insert_review_task(conn, r)
        for key in ("cvat:11", "cvat:12"):
            request_task_revocation(conn, key, reason="sweep", at=FIXED_TIME)
        mark_task_media_revoked(conn, SID, revoked, FIXED_TIME)
        mark_task_tool_revoked(conn, "cvat:11", FIXED_TIME)
    cvat = FakeTool(
        "cvat",
        calls,
        [
            _cvat(10, f"{SID}/bodycam/labeler01"),
            _cvat(11, f"{SID}/bodycam/labeler03"),
            _cvat(12, f"{SID}/bodycam/labeler04"),
            _cvat(13, f"{SID}/bodycam/labeler05"),
            _cvat(14, f"{SID}/bodycam/privacy", PRIVACY),
            _cvat(20, f"{SID}/bodycam/labeler02"),
            _cvat(21, f"{SID}/bodycam/labeler02", "dlp-labeling-spatial-demo"),
            _cvat(22, "zzz/bodycam/labeler02"),
            _cvat(23, "demo-ui"),
            _cvat(24, f"{SID}/bodycam/labeler06", at=NOW - timedelta(minutes=5)),
        ],
    )
    ls = FakeTool(
        "label_studio",
        calls,
        [_ls(30, "labeler01"), _ls(31, "labeler02"), _ls(32, "labeler02", "데모"), _ls(33, None)],
    )
    return {"labeling": labeling, "cvat": cvat, "ls": ls, "calls": calls}


def _find(engine: sa.Engine, w: dict[str, Any], **kw: Any) -> OrphanReport:
    """정책 나이 하한 1시간으로 찾는다."""
    return find_orphans(
        engine, labeling=w["labeling"], cvat=w["cvat"], label_studio=w["ls"], min_age=MIN_AGE,
        now=NOW, **kw,
    )  # fmt: skip


def _media_key(who: str) -> str:
    """바디캠 워터마크 사본 키."""
    return labeling_media_key(SID, who, "bodycam", HASH)


EXPECTED = {
    _media_key("labeler02"): "no_db_row",
    _media_key("labeler03"): "revoked",
    "cvat:11": "revoked",
    "cvat:13": "other_owner",
    "cvat:20": "no_db_row",
    "label_studio:31": "no_db_row",
}


def test_find_reports_only_platform_orphans_with_reasons(
    engine: sa.Engine, world: dict[str, Any]
) -> None:
    """정답 고아만 이유와 함께 보고한다 (매체 → CVAT → LS 순). 읽기만 한다.

    다른 배포(이 DB에 없는 세션)의 작업은 수만 경고에 남고, 어린 작업은 young이다.
    """
    report = _find(engine, world)
    assert {o.ref: o.reason for o in report.found} == EXPECTED
    assert [o.channel for o in report.found] == sorted(
        (o.channel for o in report.found), key=["media", "cvat", "label_studio"].index
    )
    assert [o.ref for o in report.young] == ["cvat:24"]
    assert any("cvat: 이 DB에 없는 세션의 작업 1개" in w for w in report.warnings)
    assert not report.unscanned and not report.clean
    assert world["calls"] == []  # 찾기는 지우지도 스냅숏을 쓰지도 않는다
    # 세션을 고르면 그 세션만 본다 (다른 DB 세션의 작업은 다른 배포 경고에 넣지 않는다)
    assert {o.ref for o in _find(engine, world, session_id="s404").found} == set()


def test_apply_deletes_orphans_in_order_and_keeps_everything_else(
    engine: sa.Engine, world: dict[str, Any]
) -> None:
    """지우기: 매체 삭제가 도구 호출보다 먼저, CVAT는 배정 해제 → 스냅숏 → 삭제다.

    살아 있는 작업·회수 대기·블러 검수·다른 시스템·어린 작업, 시계열 CSV·블러본·회수 스냅숏은
    그대로다. 고아 도구 작업마다 스냅숏이 남는다.
    """
    labeling, cvat, ls, calls = world["labeling"], world["cvat"], world["ls"], world["calls"]
    report = _find(engine, world)
    delete_orphans(
        engine, report, labeling=labeling, raw_bucket="dlp-raw", cvat=cvat, label_studio=ls,
        now=NOW,
    )  # fmt: skip
    assert sorted(report.deleted) == sorted(EXPECTED)
    assert report.clean and not report.errors
    first_tool = next(i for i, c in enumerate(calls) if c[0] != "s3")
    assert all(c[0] == "s3" for c in calls[:first_tool])
    assert [c for c in calls if c[0] == "cvat" and c[2] == "20"] == [
        ("cvat", "unassign", "20"),
        ("cvat", "snapshot", "20"),
        ("cvat", "delete", "20"),
    ]
    assert cvat.alive == {10, 12, 14, 21, 22, 23, 24}
    assert ls.alive == {30, 32, 33}
    for who in ("labeler02", "labeler03"):
        assert labeling.head(_media_key(who)) is None
    for key in (
        _media_key("labeler01"),
        _media_key("labeler04"),
        f"sessions/{SID}/review/timeseries.csv",
        f"sessions/{SID}/blurred/bodycam.mp4",
        f"sessions/{SID}/revoked/cvat-11.json",
    ):
        assert labeling.head(key) is not None, key
    for channel, tid in (("cvat", 20), ("cvat", 13), ("label_studio", 31)):
        assert labeling.head(orphan_snapshot_key(SID, channel, tid)) is not None
    # 다시 돌리면 남은 고아가 없다 (멱등)
    assert _find(engine, world).found == []


def test_apply_skips_busy_sessions_and_rechecks_db_under_session_lock(
    engine: sa.Engine, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """세션을 잠그지 못하면(작업 생성 중) 그 세션은 건너뛴다. 잠근 뒤 다시 읽으니 행이 생긴 것
    (나열 뒤 생성이 커밋됨)은 지우지 않는다."""
    labeling, cvat, ls = world["labeling"], world["cvat"], world["ls"]
    busy = _find(engine, world)
    real_lock = orphans_module.try_lock_session

    def never(conn: sa.Connection, session_id: str) -> bool:
        return False

    monkeypatch.setattr(orphans_module, "try_lock_session", never)
    delete_orphans(
        engine, busy, labeling=labeling, raw_bucket="dlp-raw", cvat=cvat, label_studio=ls, now=NOW
    )
    assert busy.deleted == [] and sorted(busy.busy) == sorted(EXPECTED)
    assert world["calls"] == [] and not busy.clean
    monkeypatch.setattr(orphans_module, "try_lock_session", real_lock)

    report = _find(engine, world)
    with engine.begin() as conn:  # 나열한 뒤 커밋된 작업 생성
        insert_review_task(
            conn, _row("cvat:20", "labeler02", labeling.uri(_media_key("labeler02")))
        )
    delete_orphans(
        engine, report, labeling=labeling, raw_bucket="dlp-raw", cvat=cvat, label_studio=ls,
        now=NOW,
    )  # fmt: skip
    assert "cvat:20" not in report.deleted and 20 in cvat.alive
    assert labeling.head(_media_key("labeler02")) is not None
    assert {o.ref for o in report.found} == set(EXPECTED) - {"cvat:20", _media_key("labeler02")}
    assert sorted(report.deleted) == sorted(o.ref for o in report.found)
    assert sum("다시 읽으니 DB 기록이 있어" in w for w in report.warnings) == 2
    assert report.clean


def test_apply_refuses_raw_bucket_and_reports_unscanned_channels(
    engine: sa.Engine, world: dict[str, Any], tmp_path: Path
) -> None:
    """라벨링 저장소가 원본 버킷이면 아무것도 지우지 않는다. 클라이언트가 없는 채널은 unscanned."""
    report = _find(engine, world)
    delete_orphans(
        engine, report, labeling=world["labeling"], raw_bucket="dlp-labeling", cvat=world["cvat"],
        label_studio=world["ls"], now=NOW,
    )  # fmt: skip
    assert report.deleted == [] and "원본 버킷" in report.errors[0]
    assert world["calls"] == []
    partial = find_orphans(
        engine, labeling=world["labeling"], cvat=None, label_studio=world["ls"], min_age=MIN_AGE,
        now=NOW,
    )  # fmt: skip
    assert partial.unscanned == ["cvat: 클라이언트 없음"] and not partial.clean
    with pytest.raises(ValueError, match="시간대"):
        find_orphans(
            engine, labeling=world["labeling"], cvat=None, label_studio=None, min_age=MIN_AGE,
            now=NOW.replace(tzinfo=None),
        )  # fmt: skip


def test_reason_rules_and_platform_names() -> None:
    """판정 규칙과 이 플랫폼 이름 규칙 (다른 시스템의 이름은 고르지 않는다)."""
    row = _row("cvat:1", "labeler01", "s3://dlp-labeling/x")
    mine = TaskOwner(SID, "bodycam", "labeler01", "")
    assert tool_reason(None, mine) == "no_db_row"
    assert tool_reason(row, mine) is None
    assert tool_reason(row, TaskOwner(SID, "third", "labeler01", "")) == "other_owner"
    assert tool_reason(row, TaskOwner(SID, None, "labeler01", "")) is None  # LS는 스트림 없음
    requested = row.model_copy(update={"revoke_requested_at": FIXED_TIME})
    assert tool_reason(requested, mine) is None  # 실행기 몫
    done = requested.model_copy(update={"tool_revoked_at": FIXED_TIME})
    assert tool_reason(done, mine) == "revoked"
    assert media_reason([]) == "no_db_row"
    assert media_reason([done]) is None  # 매체 완료 표시는 없다
    gone = done.model_copy(update={"media_revoked_at": FIXED_TIME})
    assert media_reason([gone]) == "revoked" and media_reason([gone, row]) is None

    assert is_ontology_project("dlp-labeling-spatial", "dlp-labeling-spatial")
    assert is_ontology_project(SPATIAL, "dlp-labeling-spatial")
    for name in ("dlp-labeling-spatial-demo", "x-dlp-labeling-spatial", "dlp-privacy-v1-abc"):
        assert not is_ontology_project(name, "dlp-labeling-spatial")
    assert not is_ontology_project(SPATIAL, "dlp-privacy")

    assert parse_labeling_media_key(_media_key("labeler01")) == (SID, "labeler01", "bodycam")
    assert parse_labeling_media_key(f"sessions/{SID}/review/a/third.mp4") == (SID, "a", "third")
    for key in (
        f"sessions/{SID}/review/timeseries.csv",
        f"sessions/{SID}/blurred/bodycam.mp4",
        f"sessions/{SID}/review/a/b/bodycam.mp4",
    ):
        assert parse_labeling_media_key(key) is None
