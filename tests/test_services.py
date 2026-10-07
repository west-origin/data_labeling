"""실행 중인 개발 서비스에 대한 통합 테스트. `make up` 후 `make test-services`로 실행한다.

WP0 완료 기준(서비스 기동·헬스체크)과 서비스 사이 연결을 확인한다.
- 모든 서비스 헬스체크 통과 (`dlp services check`와 같은 목록).
- MLflow 산출물이 SeaweedFS S3를 거쳐 왕복된다.
- Prefect 서버가 자기 DB(PostgreSQL)에 닿는다.
- 모든 버킷에 실제로 쓸 수 있다 (버킷별 볼륨 회귀 방지).
호스트·포트는 `DLP_HOST`, `DLP_MLFLOW_PORT`, `DLP_PREFECT_PORT` 환경 변수로 바꾼다.
"""

from __future__ import annotations

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import pytest

from dlp_cli.health import default_checks, run_checks
from dlp_media.storage import S3Store, put_immutable

pytestmark = pytest.mark.services

HOST = os.environ.get("DLP_HOST", "localhost")
MLFLOW = f"http://{HOST}:{os.environ.get('DLP_MLFLOW_PORT', '5000')}"
PREFECT = f"http://{HOST}:{os.environ.get('DLP_PREFECT_PORT', '4200')}"


def _request(method: str, url: str, body: bytes | None = None, json_body: Any = None) -> bytes:
    """표준 라이브러리 HTTP 요청 도우미. `json_body`가 있으면 JSON으로 보낸다. 2xx가 아니면
    `HTTPError`."""
    headers: dict[str, str] = {}
    if json_body is not None:
        body = json.dumps(json_body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read()


def test_all_services_healthy() -> None:
    """기본 헬스체크 목록(CVAT 제외)이 제한 시간 5초 안에 모두 정상이다."""
    failed = [r for r in run_checks(default_checks(), timeout=5) if not r.ok]
    assert not failed, failed


def test_mlflow_artifact_roundtrip_through_s3() -> None:
    """MLflow 실험을 만들고, 산출물 프록시 API로 쓴 바이트를 그대로 다시 읽는다 (MLflow → S3 설정
    확인)."""
    name = f"wp0-smoke-{uuid.uuid4().hex[:8]}"
    created = json.loads(
        _request("POST", f"{MLFLOW}/api/2.0/mlflow/experiments/create", json_body={"name": name})
    )
    assert created["experiment_id"]

    path = f"{MLFLOW}/api/2.0/mlflow-artifacts/artifacts/smoke/{name}.txt"
    _request("PUT", path, body=b"hello")
    assert _request("GET", path) == b"hello"


def test_prefect_database_reachable() -> None:
    """Prefect 흐름 목록 API가 목록을 돌려준다 (Prefect가 DB에 연결됨)."""
    flows = json.loads(_request("POST", f"{PREFECT}/api/flows/filter", json_body={"limit": 1}))
    assert isinstance(flows, list)


@pytest.mark.parametrize("bucket", ["dlp-raw", "dlp-labeling", "dlp-datasets", "dlp-mlflow"])
def test_every_bucket_accepts_writes(bucket: str, tmp_path: Path) -> None:
    """버킷마다 볼륨이 따로 잡히므로 모든 버킷에 실제로 써 본다 (볼륨 부족 회귀 방지).

    `_smoke/<임의>.txt`를 불변 쓰기로 올리고 존재를 확인한 뒤 지운다. 원본 버킷에도 직접 쓰지만
    테스트 전용 키라 감사 대상 경로가 아니다.
    """
    store = S3Store.from_env(bucket)
    src = tmp_path / "probe.txt"
    src.write_text(uuid.uuid4().hex)
    key = f"_smoke/{src.read_text()}.txt"
    assert put_immutable(store, key, src)
    assert store.head(key) is not None
    store.client.delete_object(Bucket=bucket, Key=key)
