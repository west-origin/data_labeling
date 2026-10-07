from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator

import pytest

from dlp_cli.health import ServiceCheck, default_checks, run_check
from dlp_cli.main import main


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/ok" else 503)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def http_port() -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_check_ok_and_error_status(http_port: int) -> None:
    ok = run_check(ServiceCheck("svc", "http", f"http://127.0.0.1:{http_port}/ok"))
    bad = run_check(ServiceCheck("svc", "http", f"http://127.0.0.1:{http_port}/down"))
    assert ok.ok
    assert not bad.ok
    assert "503" in bad.detail


def test_tcp_check(http_port: int) -> None:
    assert run_check(ServiceCheck("svc", "tcp", f"127.0.0.1:{http_port}")).ok
    assert not run_check(ServiceCheck("svc", "tcp", f"127.0.0.1:{_closed_port()}"), 0.5).ok


def test_default_checks_respect_env_ports() -> None:
    env = {"DLP_HOST": "example", "DLP_LABEL_STUDIO_PORT": "9999"}
    checks = {c.name: c for c in default_checks(env)}
    assert checks["label-studio"].target == "http://example:9999/health"
    assert checks["postgres"].target == "example:5432"
    assert "cvat" not in checks
    assert "cvat" in {c.name for c in default_checks(env, include_cvat=True)}


def test_services_check_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    port = str(_closed_port())
    for key in (
        "DLP_POSTGRES_PORT",
        "DLP_S3_PORT",
        "DLP_LABEL_STUDIO_PORT",
        "DLP_PREFECT_PORT",
        "DLP_MLFLOW_PORT",
    ):
        monkeypatch.setenv(key, port)
    monkeypatch.setenv("DLP_HOST", "127.0.0.1")
    assert main(["services", "check", "--timeout", "0.5"]) == 1


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "dlp 0.1.0" in capsys.readouterr().out
