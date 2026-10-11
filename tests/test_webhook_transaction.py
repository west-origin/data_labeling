"""CLI 콜백이 수집 예외를 삼키면 부분 변경을 커밋하므로 실제 트랜잭션으로 회귀를 검사한다."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa

from dlp_cli import review_cmds
from dlp_review.tasks import TaskError
from dlp_review.webhook import CollectRequest


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
    monkeypatch.setattr(review_cmds, "get_review_task", Mock(return_value=Mock()))
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
