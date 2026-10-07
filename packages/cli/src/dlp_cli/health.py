"""개발 서비스(docker compose) 헬스체크 (WP0, `dlp services check`, `make health`).

`services/docker-compose.yml`로 띄운 PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow,
lakeFS와 (선택) CVAT가 응답하는지 TCP 연결 또는 HTTP GET으로 확인한다. 포트는 `.env`와 같은 이름의
환경 변수(`DLP_*_PORT`)로 바꿀 수 있고, 호스트는 `DLP_HOST`(기본 localhost)다.

공개:
- `ServiceCheck` / `CheckResult` — 점검 대상과 결과.
- `default_checks(env, include_cvat)` — 기본 점검 목록.
- `run_check(check, timeout)` / `run_checks(checks, timeout)` — 점검 실행.

외부 의존성 없이 표준 라이브러리만 쓴다 (서비스가 죽어 있어도 CLI가 떠야 하므로).
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

# 점검 방식: HTTP GET 상태 코드로 보거나(http), TCP 연결만 본다(tcp, PostgreSQL처럼 HTTP가 없을 때)
CheckKind = Literal["http", "tcp"]


@dataclass(frozen=True)
class ServiceCheck:
    """서비스 하나의 헬스체크 대상.

    필드:
        name: 출력용 서비스 이름.
        kind: `http`(GET, 2xx면 정상) 또는 `tcp`(연결만 되면 정상).
        target: http는 URL, tcp는 `"host:port"`.
    """

    name: str
    kind: CheckKind
    target: str  # http면 URL, tcp면 "host:port"


@dataclass(frozen=True)
class CheckResult:
    """점검 결과. `ok`가 거짓이면 `detail`에 실패 이유(HTTP 상태나 예외 메시지)가 있다."""

    name: str
    ok: bool
    detail: str


def default_checks(
    env: Mapping[str, str] | None = None, *, include_cvat: bool = False
) -> list[ServiceCheck]:
    """환경 변수(.env와 같은 이름)로 포트를 덮어쓸 수 있는 기본 점검 목록.

    인자:
        env: 환경 변수 사전. None이면 `os.environ` (테스트는 사전을 직접 넘긴다).
        include_cvat: 참이면 CVAT(`/api/server/about`)도 넣는다. CVAT는 `make cvat-up`으로 따로
            띄운다.

    기본 포트: postgres 5432(tcp), seaweedfs-s3 8333(`/healthz`), label-studio 8081(`/health`),
    prefect 4200(`/api/health`), mlflow 5000(`/health`), lakefs 8000(`/_health`), cvat 8080.
    이 값은 `services/docker-compose.yml`·`.env.example`의 기본값과 맞춰야 한다.
    """
    env = os.environ if env is None else env
    host = env.get("DLP_HOST", "localhost")

    def port(key: str, default: int) -> str:
        """환경 변수 `key`가 있으면 그 값, 없으면 기본 포트(문자열)."""
        return env.get(key, str(default))

    checks = [
        ServiceCheck("postgres", "tcp", f"{host}:{port('DLP_POSTGRES_PORT', 5432)}"),
        ServiceCheck("seaweedfs-s3", "http", f"http://{host}:{port('DLP_S3_PORT', 8333)}/healthz"),
        ServiceCheck(
            "label-studio", "http", f"http://{host}:{port('DLP_LABEL_STUDIO_PORT', 8081)}/health"
        ),
        ServiceCheck(
            "prefect", "http", f"http://{host}:{port('DLP_PREFECT_PORT', 4200)}/api/health"
        ),
        ServiceCheck("mlflow", "http", f"http://{host}:{port('DLP_MLFLOW_PORT', 5000)}/health"),
        ServiceCheck("lakefs", "http", f"http://{host}:{port('DLP_LAKEFS_PORT', 8000)}/_health"),
    ]
    if include_cvat:
        checks.append(
            ServiceCheck(
                "cvat", "http", f"http://{host}:{port('DLP_CVAT_PORT', 8080)}/api/server/about"
            )
        )
    return checks


def run_check(check: ServiceCheck, timeout: float = 3.0) -> CheckResult:
    """점검 하나를 실행한다. 예외를 던지지 않고 항상 `CheckResult`를 돌려준다.

    - tcp: `host:port`로 소켓 연결을 시도한다 (`rpartition`이라 IPv6 주소의 마지막 `:`를 포트로
      본다).
    - http: GET 응답 상태가 200~299면 정상. `HTTPError`(4xx·5xx)도 상태 코드로 보고 실패로
      판정한다. 연결 실패·시간 초과는 실패.

    인자:
        timeout: 연결·응답 제한 시간(초).
    """
    if check.kind == "tcp":
        host, _, port = check.target.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return CheckResult(check.name, True, f"tcp {check.target} 연결됨")
        except OSError as exc:
            return CheckResult(check.name, False, f"tcp {check.target} 실패: {exc}")

    try:
        with urllib.request.urlopen(check.target, timeout=timeout) as resp:
            status: int = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, OSError) as exc:
        return CheckResult(check.name, False, f"GET {check.target} 실패: {exc}")
    ok = 200 <= status < 300
    return CheckResult(check.name, ok, f"GET {check.target} -> {status}")


def run_checks(checks: list[ServiceCheck], timeout: float = 3.0) -> list[CheckResult]:
    """점검 목록을 순서대로 실행한다 (병렬 아님: 서비스 수 * `timeout`초까지 걸릴 수 있다)."""
    return [run_check(c, timeout) for c in checks]
