"""콘솔 시험 공용 도구: 가짜 `dlp` 명령, 임시 작업 디렉터리, SQLite DB를 쓰는 앱."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from dlp_console.app import Services, create_app
from dlp_console.settings import ConsoleSettings, load_console_config
from dlp_schema import repo_root
from dlp_schema.testing import sqlite_db

# 가짜 `dlp`: 인자를 찍고, 첫 인자가 fail이면 오류 줄을 찍고 3으로, sleep이면 잠깐 쉬고 끝난다.
FAKE_DLP = """
import sys, time
args = sys.argv[1:]
print("args:", " ".join(args), flush=True)
if args and args[0] == "fail":
    print("ConnectError: localhost:8080 Connection refused", flush=True)
    print("s3://dlp-raw/sessions/x/raw/bodycam.mp4 비밀 dlp-dev-password", flush=True)
    sys.exit(3)
if args and args[0] == "sleep":
    time.sleep(float(args[1]))
"""


@pytest.fixture
def fake_dlp(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_dlp.py"
    script.write_text(FAKE_DLP, "utf-8")
    return [sys.executable, str(script)]


@pytest.fixture
def settings(tmp_path: Path, fake_dlp: list[str]) -> ConsoleSettings:
    root = repo_root()
    work = tmp_path / "work"
    work.mkdir()
    return ConsoleSettings(
        root=root,
        work_dir=work,
        config=load_console_config(root),
        dlp_argv=fake_dlp,
        env={"DLP_POSTGRES_PASSWORD": "dlp-dev-password", "PATH": "/usr/bin:/bin"},
        token="test-token-123",
    )


class FakeServices(Services):
    """DB는 SQLite 파일, 라벨링 버킷은 가짜 (렌더 여부·서명)."""

    def __init__(self, settings: ConsoleSettings, engine: sa.Engine) -> None:
        super().__init__(settings)
        self.engine.dispose()
        self.engine = engine
        self.rendered_map: dict[tuple[str, str], bool] = {}
        self.reviewers: tuple[str, ...] = ("priv01",)

    def rendered(self, session_id: str, stream_id: str) -> bool:
        return self.rendered_map.get((session_id, stream_id), False)

    def sign(self, key: str) -> str:
        return f"https://labeling.example/{key}?sig=labeler"

    def privacy_reviewers(self) -> tuple[str, ...]:
        return self.reviewers


@pytest.fixture
def services(settings: ConsoleSettings, tmp_path: Path) -> Iterator[FakeServices]:
    engine = sqlite_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    svc = FakeServices(settings, engine)
    yield svc
    engine.dispose()


@dataclass
class Resp:
    """응답 (TestClient 응답 중 시험이 쓰는 것만, 타입 검사용)."""

    status_code: int
    text: str
    body: Any

    def json(self) -> Any:
        return self.body


class Api:
    """토큰 머리글을 붙이는 TestClient 감싸개 (httpx 응답 타입을 시험 밖으로 내지 않는다)."""

    def __init__(self, client: Any, token: str) -> None:
        self._client = client
        self._token = token

    def _wrap(self, res: Any) -> Resp:
        try:
            body = res.json()
        except ValueError:
            body = None
        return Resp(int(res.status_code), str(res.text), body)

    def _headers(self, headers: dict[str, str] | None) -> dict[str, str]:
        return {"X-Console-Token": self._token, **(headers or {})}

    def get(self, path: str, headers: dict[str, str] | None = None) -> Resp:
        return self._wrap(self._client.get(path, headers=self._headers(headers)))

    def delete(self, path: str) -> Resp:
        return self._wrap(self._client.delete(path, headers=self._headers(None)))

    def post(self, path: str, headers: dict[str, str] | None = None, **kw: Any) -> Resp:
        return self._wrap(self._client.post(path, headers=self._headers(headers), **kw))


@pytest.fixture
def client(settings: ConsoleSettings, services: FakeServices) -> Iterator[Api]:
    app = create_app(settings, services)
    test_client: Any = TestClient(app)
    with test_client as c:
        yield Api(c, settings.token)
