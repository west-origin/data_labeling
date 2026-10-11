"""검수·프라이버시 CLI의 매체 회수 연결 (ADR 0039).

시간대 보존 SQLite 파일 DB와 LocalStore로 돈다 (서비스 없음).

정답 근거: 회수 실행은 수집·탐지 커밋 뒤에 돌고, 대기가 남으면 종료 코드 4다. `--dry-run`은 쓰지
않는다. `--pending`은 모든 세션의 대기를 실행한다. 도구 설정이 실패해도(tolerant) 매체 삭제는 한다.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sqlalchemy

from dlp_cli import privacy_cmds, review_cmds, review_config
from dlp_cli.review_config import REVOCATION_PENDING_EXIT
from dlp_media.storage import LocalStore, sha256_file
from dlp_privacy.pipeline import StreamResult
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import render_hash
from dlp_review.revoke import RevocationReport
from dlp_review.tasks import ReviewSetup
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_review_task,
    insert_labels,
    insert_review_task,
    insert_session,
    mark_review_task_collected,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_label, make_session, sqlite_db

NOW = FIXED_TIME + timedelta(hours=1)


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """세션 둘(s001, s002, 승인됨)을 넣은 SQLite 파일 DB URL. CLI 엔진도 시간대를 보존한다."""
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(db)
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    video = tmp_path / "v.bin"
    video.write_bytes(b"synthetic")
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for sid in ("s001", "s002"):
            key = f"sessions/{sid}/raw/bodycam.mp4"
            raw.put_file(key, video, sha256_file(video))
            stream = {"stream_id": "bodycam", "kind": "bodycam", "sync_method": "reference",
                      "uri": raw.uri(key)}  # fmt: skip
            insert_session(
                conn, make_session(sid, privacy_state=PrivacyState.APPROVED, streams=[stream])
            )
    engine.dispose()
    for module in (review_cmds, privacy_cmds):
        monkeypatch.setattr(module, "sa", _AwareSqlalchemy())
    return db


def _engine(url: str) -> sqlalchemy.Engine:
    """시험 쪽에서 DB를 읽고 쓸 엔진."""
    return sqlite_db(url, create=False)


def _labeling(tmp_path: Path) -> LocalStore:
    """라벨링 버킷 (로컬)."""
    return LocalStore(tmp_path / "store", "dlp-labeling")


def _media(tmp_path: Path, sid: str, name: str) -> str:
    """라벨링 버킷에 워터마크 사본을 둔다. URI."""
    store = _labeling(tmp_path)
    src = tmp_path / "m.bin"
    src.write_bytes(b"marked")
    key = f"sessions/{sid}/review/labeler01/{name}.mp4"
    store.put_file(key, src, sha256_file(src))
    return store.uri(key)


def _task(key: str, sid: str, uri: str, rhash: str | None) -> ReviewTask:
    """바디캠 라벨링 CVAT 작업."""
    return ReviewTask(
        task_key=key, tool=ReviewTool.CVAT, external_id=key.split(":")[1], session_id=sid,
        stream_id="bodycam", stage=ReviewStage.LABELING, media_uri=uri,
        label_kinds=("box_track",), created_at=FIXED_TIME, render_hash=rhash, assignee="labeler01",
    )  # fmt: skip


def _face(sid: str = "s001", label_id: str = "b-new") -> Any:
    """바디캠 얼굴 블러 (사람, 작업보다 늦게 만든 것)."""
    payload = {"kind": "blur_track", "target": "face",
               "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}]}  # fmt: skip
    return make_label(
        payload, label_id=label_id, session_id=sid, stream_id="bodycam", t_end_ms=100,
        created_at=NOW,
    )  # fmt: skip


def _local_setup(tmp_path: Path) -> ReviewSetup:
    """도구 없는 로컬 검수 설정 (CVAT 로그인 실패를 견딘 것과 같다)."""
    return ReviewSetup(
        raw=LocalStore(tmp_path / "store", "dlp-raw"), labeling=_labeling(tmp_path),
        labeling_reader=Mock(), ontology=None, cvat=None, label_studio=None,
        privacy=load_privacy_policy(repo_root()),
    )  # fmt: skip


def _tolerant(tmp_path: Path) -> Any:
    """`review_setup` 대역: tolerant 설정만 받고 CVAT 접속 실패 경고를 돌려준다."""

    def fake(purpose: str, url: str | None, **kw: Any) -> tuple[ReviewSetup, list[str]]:
        assert kw.get("tolerant") is True
        return _local_setup(tmp_path), ["CVAT에 접속하지 못했습니다 (ConnectError)"]

    return fake


def _revoke_args(url: str, **kw: Any) -> argparse.Namespace:
    """`dlp review revoke` 인자 (기본값)."""
    values: dict[str, Any] = dict(
        session_id=None, stream=[], task=[], all=False, include_legacy=False, dry_run=False,
        reason=None, pending=False, mark_tool_gone=None, note=None, url=url,
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_collect_runs_revocations_after_commit_and_exits_4_when_pending(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """이미 수집한 작업(결과 None)이어도 커밋 뒤 그 세션의 회수를 실행하고, 대기가 남으면 4."""
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "bodycam"), None))
        mark_review_task_collected(conn, "cvat:2", NOW)
    setup = _local_setup(tmp_path)
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=setup))
    run = Mock(return_value=RevocationReport(pending=["label_studio:9"]))
    monkeypatch.setattr(review_cmds, "run_revocations", run)
    args = argparse.Namespace(task_key="cvat:2", reviewer="labeler01", url=url)
    assert review_cmds.cmd_collect(args) == REVOCATION_PENDING_EXIT
    assert run.call_args.args[1:] == ("s001", setup)


def test_revoke_dry_run_writes_nothing_and_estimates_legacy(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--dry-run`은 대상과 이유만 보이고 쓰지 않는다. 구 작업에는 생성 시점 추정을 붙인다."""
    empty = render_hash([], load_privacy_policy(repo_root()))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "a"), empty))
        insert_review_task(conn, _task("cvat:3", "s001", _media(tmp_path, "s001", "b"), None))
        insert_labels(conn, [_face()])
    args = _revoke_args(url, session_id="s001", dry_run=True, include_legacy=True)
    assert review_cmds.cmd_revoke(args) == 0
    out = capsys.readouterr().out
    assert "cvat:2" in out and "hash" in out and "추정: 생성 시점 블러가 지금과 다름" in out
    with _engine(url).connect() as conn:
        assert get_review_task(conn, "cvat:2").revoke_requested_at is None
        assert get_review_task(conn, "cvat:3").status is ReviewTaskStatus.OPEN


def test_revoke_requires_reason_for_task_and_all(url: str) -> None:
    """`--task`·`--all`은 사유 없이 쓰지 않는다 (사고 대응 기록)."""
    with pytest.raises(SystemExit, match="--reason"):
        review_cmds.cmd_revoke(_revoke_args(url, session_id="s001", all=True))


def test_revoke_pending_runs_every_session_and_tolerates_missing_tools(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--pending`은 대기가 있는 모든 세션의 매체를 지운다. 도구가 없으면 도구 작업은 대기다."""
    keys = {"s001": "cvat:2", "s002": "cvat:3"}
    with _engine(url).begin() as conn:
        for sid, key in keys.items():
            insert_review_task(conn, _task(key, sid, _media(tmp_path, sid, "bodycam"), None))
            request_task_revocation(conn, key, reason="sweep", at=NOW)
    monkeypatch.setattr(review_config, "review_setup", _tolerant(tmp_path))
    assert review_cmds.cmd_revoke(_revoke_args(url, pending=True)) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "CVAT에 접속하지 못했습니다" in out and "회수 대기 2개" in out
    with _engine(url).connect() as conn:
        for key in keys.values():
            task = get_review_task(conn, key)
            assert task.media_revoked_at is not None and task.tool_revoked_at is None
    for sid in keys:
        assert _labeling(tmp_path).head(f"sessions/{sid}/review/labeler01/bodycam.mp4") is None


def test_detect_requests_and_runs_revocation_even_without_tools(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """탐지로 블러가 생기면 같은 커밋에 회수를 요청하고 커밋 뒤 실행한다.

    도구가 없어도 매체는 지운다.
    """
    empty = render_hash([], load_privacy_policy(repo_root()))
    uri = _media(tmp_path, "s001", f"bodycam-{empty[:16]}")
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, empty))
    monkeypatch.setattr(review_config, "review_setup", _tolerant(tmp_path))
    model = Provenance(source=Source.MODEL, model_version="m1")
    found = [_face().model_copy(update={"provenance": model, "confidence": 0.9})]
    args = argparse.Namespace(session_id="s001", store="unused", url=url)
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    with (
        patch("dlp_cli.privacy_cmds.raw_store", return_value=raw),
        patch("dlp_cli.privacy_cmds.build_detectors", return_value=({}, {})),
        patch("dlp_cli.privacy_cmds.store_from_spec", return_value=_labeling(tmp_path)),
        patch(
            "dlp_cli.privacy_cmds.deployed_predictors",
            return_value=SimpleNamespace(notes=[], predictors=[]),
        ),
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult(found, [], {}, 3)),
    ):
        assert privacy_cmds.cmd_detect(args) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "라벨링 작업 1개 회수: cvat:2" in out and "회수 대기 1개" in out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "privacy_detect"
    assert task.media_revoked_at is not None


def _create_args(url: str, **kw: Any) -> argparse.Namespace:
    """`dlp review create --stage labeling` 인자."""
    values: dict[str, Any] = dict(
        session_id="s001", stage="labeling", assignee="labeler01", stream=[], tool=None,
        no_cvat=False, no_ls=False, url=url, review_command="create",
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_create_passes_stream_and_tool_and_stops_on_pending(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--stream`·`--tool`을 작업 생성에 넘기고 먼저 sweep 회수를 돈다. 대기가 남으면 4로 멈춘다."""
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=_local_setup(tmp_path)))
    run = Mock(return_value=RevocationReport())
    create = Mock(return_value=[])
    monkeypatch.setattr(review_cmds, "run_revocations", run)
    monkeypatch.setattr(review_cmds, "create_labeling_tasks", create)
    args = _create_args(url, stream=["bodycam"], tool="label_studio")
    assert review_cmds.cmd_create(args) == 0
    assert create.call_args.kwargs == {"streams": {"bodycam"}, "tools": {ReviewTool.LABEL_STUDIO}}
    assert run.call_args.kwargs == {"sweep": True}
    run.return_value = RevocationReport(pending=["cvat:2"])
    create.reset_mock()
    assert review_cmds.cmd_create(_create_args(url)) == REVOCATION_PENDING_EXIT
    create.assert_not_called()


def test_render_backstop_requests_tasks_made_under_other_render_policy(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """렌더 정책(블러 강도)을 바꾼 뒤 렌더하면 이전 정책의 블러본으로 만든 작업을 회수 요청한다.

    같은 정책으로 다시 렌더하면 해시가 같아 요청하지 않는다 (렌더 백스톱, ADR 0039).
    """
    from dlp_fixtures.video import write_video

    raw = LocalStore(tmp_path / "store", "dlp-raw")
    video = tmp_path / "tiny.mp4"
    frames = [(t, np.zeros((48, 64, 3), dtype=np.uint8)) for t in (0, 100)]
    write_video(video, frames, width=64, height=48)
    raw.put_file("sessions/s001/raw/bodycam.mp4", video, sha256_file(video))
    policy = load_privacy_policy(repo_root())
    stronger = policy.model_copy(
        update={"render": policy.render.model_copy(update={"crf": policy.render.crf + 1})}
    )
    old = render_hash([], policy)
    uri = _media(tmp_path, "s001", f"bodycam-{old[:16]}")
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, old))
    monkeypatch.setattr(review_config, "review_setup", _tolerant(tmp_path))
    monkeypatch.setattr(privacy_cmds, "raw_store", Mock(return_value=raw))
    monkeypatch.setattr(privacy_cmds, "store_from_spec", Mock(return_value=_labeling(tmp_path)))
    args = argparse.Namespace(session_id="s001", store="unused", url=url)
    monkeypatch.setattr(privacy_cmds, "load_policy", Mock(return_value=policy))
    assert privacy_cmds.cmd_render(args) == 0
    with _engine(url).connect() as conn:
        assert get_review_task(conn, "cvat:2").revoke_requested_at is None
    monkeypatch.setattr(privacy_cmds, "load_policy", Mock(return_value=stronger))
    assert privacy_cmds.cmd_render(args) == REVOCATION_PENDING_EXIT
    assert "렌더 해시가 바뀐 라벨링 작업 1개 회수: cvat:2" in capsys.readouterr().out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "render"
    assert task.media_revoked_at is not None
