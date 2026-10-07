"""개발 서비스(docker compose) 헬스체크."""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

CheckKind = Literal["http", "tcp"]


@dataclass(frozen=True)
class ServiceCheck:
    """서비스 하나의 헬스체크 대상."""

    name: str
    kind: CheckKind
    target: str  # http: URL, tcp: "host:port"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def default_checks(
    env: Mapping[str, str] | None = None, *, include_cvat: bool = False
) -> list[ServiceCheck]:
    """환경 변수(.env와 같은 이름)로 포트를 덮어쓸 수 있는 기본 점검 목록."""
    env = os.environ if env is None else env
    host = env.get("DLP_HOST", "localhost")

    def port(key: str, default: int) -> str:
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
    ]
    if include_cvat:
        checks.append(
            ServiceCheck(
                "cvat", "http", f"http://{host}:{port('DLP_CVAT_PORT', 8080)}/api/server/about"
            )
        )
    return checks


def run_check(check: ServiceCheck, timeout: float = 3.0) -> CheckResult:
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
    return [run_check(c, timeout) for c in checks]
