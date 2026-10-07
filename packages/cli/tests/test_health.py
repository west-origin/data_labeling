"""`dlp_cli.health`와 `dlp` 진입점의 단위 테스트 (WP0).

실제 docker 서비스 없이, 테스트 안에서 띄운 작은 HTTP 서버와 닫힌 포트로 정상/실패를 만든다.
- HTTP 점검: `/ok`는 200, 그 밖의 경로는 503을 돌려주는 서버로 2xx 판정을 확인한다.
- TCP 점검: 열린 포트(HTTP 서버)와 방금 닫은 포트로 연결 성공/실패를 확인한다.
- 기본 점검 목록이 환경 변수로 호스트·포트를 바꾸는지, CVAT가 선택인지 확인한다.
- `dlp services check` 종료 코드와 `dlp --version` 출력을 확인한다.
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator

import pytest

from dlp_cli.health import ServiceCheck, default_checks, run_check
from dlp_cli.main import main


class _Handler(http.server.BaseHTTPRequestHandler):
    """테스트용 HTTP 처리기: `/ok`에는 200, 나머지는 503으로 답한다."""

    def do_GET(self) -> None:
        """GET 요청에 상태 코드만 보내고 본문은 없다."""
        self.send_response(200 if self.path == "/ok" else 503)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        """요청 로그를 표준 오류에 찍지 않는다 (테스트 출력 정리)."""
        pass


@pytest.fixture
def http_port() -> Iterator[int]:
    """임의 포트(127.0.0.1:0)에 `_Handler` 서버를 데몬 스레드로 띄우고 그 포트를 준다. 테스트 뒤
    닫는다."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _closed_port() -> int:
    """잠깐 바인드했다가 닫은 포트 번호. 테스트 동안 아무도 듣지 않을 것으로 기대한다(드물게
    재사용될 수 있다)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_check_ok_and_error_status(http_port: int) -> None:
    """HTTP 점검은 2xx만 정상이고, 5xx는 실패이며 상태 코드(503)가 `detail`에 남는다."""
    ok = run_check(ServiceCheck("svc", "http", f"http://127.0.0.1:{http_port}/ok"))
    bad = run_check(ServiceCheck("svc", "http", f"http://127.0.0.1:{http_port}/down"))
    assert ok.ok
    assert not bad.ok
    assert "503" in bad.detail


def test_tcp_check(http_port: int) -> None:
    """TCP 점검은 듣는 포트면 성공, 닫힌 포트면 실패한다 (제한 시간 0.5초)."""
    assert run_check(ServiceCheck("svc", "tcp", f"127.0.0.1:{http_port}")).ok
    assert not run_check(ServiceCheck("svc", "tcp", f"127.0.0.1:{_closed_port()}"), 0.5).ok


def test_default_checks_respect_env_ports() -> None:
    """`DLP_HOST`·`DLP_LABEL_STUDIO_PORT`가 대상 URL에 반영되고, 지정하지 않은 postgres는 기본
    5432를 쓴다. CVAT는 `include_cvat=True`일 때만 들어간다.
    """
    env = {"DLP_HOST": "example", "DLP_LABEL_STUDIO_PORT": "9999"}
    checks = {c.name: c for c in default_checks(env)}
    assert checks["label-studio"].target == "http://example:9999/health"
    assert checks["postgres"].target == "example:5432"
    assert "cvat" not in checks
    assert "cvat" in {c.name for c in default_checks(env, include_cvat=True)}


def test_services_check_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """모든 서비스 포트를 닫힌 포트로 돌리면 `dlp services check`가 종료 코드 1을 낸다."""
    port = str(_closed_port())
    for key in (
        "DLP_POSTGRES_PORT",
        "DLP_S3_PORT",
        "DLP_LABEL_STUDIO_PORT",
        "DLP_PREFECT_PORT",
        "DLP_MLFLOW_PORT",
        "DLP_LAKEFS_PORT",
    ):
        monkeypatch.setenv(key, port)
    monkeypatch.setenv("DLP_HOST", "127.0.0.1")
    assert main(["services", "check", "--timeout", "0.5"]) == 1


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`dlp --version`이 `SystemExit(0)`으로 끝나고 패키지 버전(`0.1.0`)을 출력한다."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "dlp 0.1.0" in capsys.readouterr().out
