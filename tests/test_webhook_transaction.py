"""CLI 콜백이 수집 예외를 삼키면 부분 변경을 커밋하므로 실제 트랜잭션으로 회귀를 검사한다.

짝 시험: 수집이 커밋된 뒤 매체 회수가 덜 끝나 예외(503)를 내도 수집·회수 요청은 남는다 (ADR 0039).
열린 작업이 아니면 검수자 확인 전에 무시한다 (회수 실행기의 CVAT 배정 해제 웹훅). 웹훅 수집도
`collect`와 같이 회수한 작업과 다시 만들 명령을 서버 로그에 남긴다.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa

from dlp_cli import review_cmds
from dlp_review.reconcile import ReviewOutcome
from dlp_review.revoke import RevocationIncompleteError, RevocationReport
from dlp_review.tasks import TaskError
from dlp_review.webhook import CollectRequest
from dlp_schema.review import ReviewTaskStatus


def test_collection_failure_rolls_back_and_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = f"sqlite:///{tmp_path / 'callback.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE changes (value INTEGER)"))
    setup = Mock()
    setup.label_studio = None
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=setup))
    monkeypatch.setattr(review_cmds, "_secret", lambda: "secret")
    monkeypatch.setattr(
        review_cmds, "get_review_task", Mock(return_value=Mock(status=ReviewTaskStatus.OPEN))
    )
    monkeypatch.setattr(review_cmds, "resolve_reviewer", Mock(return_value="reviewer"))

    def failing_collect(conn: sa.Connection, *args: Any) -> None:
        conn.execute(sa.text("INSERT INTO changes VALUES (1)"))
        raise TaskError("도구 응답 실패")

    monkeypatch.setattr(review_cmds, "collect_task", failing_collect)
    closed: list[bool] = []

    class FakeServer:
        def __init__(self, callback: Callable[[CollectRequest], None]) -> None:
            self.callback = callback

        def serve_forever(self) -> None:
            self.callback(CollectRequest("label_studio:3", None))

        def server_close(self) -> None:
            closed.append(True)

    def fake_serve(
        port: int,
        secret: str,
        callback: Callable[[CollectRequest], None],
        **kwargs: Any,
    ) -> FakeServer:
        return FakeServer(callback)

    monkeypatch.setattr(review_cmds, "serve", fake_serve)
    with pytest.raises(TaskError, match="도구 응답 실패"):
        review_cmds.cmd_serve(argparse.Namespace(port=0, url=url))
    with engine.connect() as conn:
        assert conn.scalar(sa.text("SELECT count(*) FROM changes")) == 0
    engine.dispose()
    assert closed == [True]


class _FakeServer:
    """serve_forever가 웹훅 한 건을 콜백에 넘기는 가짜 서버."""

    def __init__(self, callback: Callable[[CollectRequest], None]) -> None:
        self.callback = callback

    def serve_forever(self) -> None:
        self.callback(CollectRequest("cvat:3", "someone"))

    def server_close(self) -> None:
        pass


def _fake_serve(
    port: int, secret: str, callback: Callable[[CollectRequest], None], **_: Any
) -> Any:
    return _FakeServer(callback)


def _serve_env(monkeypatch: pytest.MonkeyPatch, status: ReviewTaskStatus) -> Mock:
    """웹훅 서버 의존을 가짜로 바꾼다. resolve_reviewer Mock을 돌려준다."""
    setup = Mock()
    setup.label_studio = None
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=setup))
    monkeypatch.setattr(review_cmds, "_secret", lambda: "secret")
    monkeypatch.setattr(review_cmds, "serve", _fake_serve)
    monkeypatch.setattr(
        review_cmds,
        "get_review_task",
        Mock(return_value=Mock(status=status, session_id="s001", task_key="cvat:3")),
    )
    resolve = Mock(return_value="reviewer")
    monkeypatch.setattr(review_cmds, "resolve_reviewer", resolve)
    return resolve


def test_pending_revocation_after_commit_raises_but_keeps_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """수집 커밋 뒤 회수가 덜 끝나면 RevocationIncompleteError(→ 503)지만 DB 변경은 남는다."""
    url = f"sqlite:///{tmp_path / 'callback.db'}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE changes (value INTEGER)"))
    _serve_env(monkeypatch, ReviewTaskStatus.OPEN)

    def collect(conn: sa.Connection, *args: Any) -> None:
        conn.execute(sa.text("INSERT INTO changes VALUES (1)"))

    monkeypatch.setattr(review_cmds, "collect_task", collect)
    left = RevocationReport(pending=["label_studio:7"])
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=left))
    with pytest.raises(RevocationIncompleteError, match="회수 대기 1개"):
        review_cmds.cmd_serve(argparse.Namespace(port=0, url=url))
    with engine.connect() as conn:
        assert conn.scalar(sa.text("SELECT count(*) FROM changes")) == 1
    engine.dispose()


@pytest.mark.parametrize("status", [ReviewTaskStatus.COLLECTED, ReviewTaskStatus.REVOKED])
def test_webhook_for_closed_task_skips_reviewer_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: ReviewTaskStatus
) -> None:
    """수집했거나 회수한 작업의 웹훅은 담당자 확인(403) 없이 무시한다 (배정 해제 웹훅 포함)."""
    url = f"sqlite:///{tmp_path / 'callback.db'}"
    resolve = _serve_env(monkeypatch, status)
    collect = Mock()
    monkeypatch.setattr(review_cmds, "collect_task", collect)
    assert review_cmds.cmd_serve(argparse.Namespace(port=0, url=url)) == 0
    resolve.assert_not_called()
    collect.assert_not_called()


def test_webhook_collection_logs_revocations_and_reissue_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """웹훅으로 수집한 블러 변경이 작업을 회수하면 다시 만들 배정·직접 작업 명령을 출력한다.

    운영의 주 경로가 웹훅이라, 여기서 출력하지 않으면 다시 만들라는 신호가 DB 행 말고는 없다.
    """
    url = f"sqlite:///{tmp_path / 'callback.db'}"
    _serve_env(monkeypatch, ReviewTaskStatus.OPEN)
    outcome = ReviewOutcome(
        revoked=["cvat:7", "label_studio:8"],
        reissue_assignments=["A"],
        direct_reissue=[("labeler02", "bodycam", "label_studio")],
    )
    monkeypatch.setattr(review_cmds, "collect_task", Mock(return_value=outcome))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    assert review_cmds.cmd_serve(argparse.Namespace(port=0, url=url)) == 0
    out = capsys.readouterr().out
    assert "cvat:3 ← reviewer" in out
    assert "라벨링 작업 2개 회수: cvat:7, label_studio:8" in out
    assert "dlp review assign A" in out
    assert (
        "dlp review create s001 --stage labeling --assignee labeler02 --stream bodycam "
        "--tool label_studio"
    ) in out
