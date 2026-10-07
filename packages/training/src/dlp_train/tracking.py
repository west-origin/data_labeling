"""실험 기록 (MLflow).

MLflow 서버의 REST API로 실행·파라미터·지표·산출물을 남기고, 배포된 모델은 MLflow 모델
레지스트리에도 같은 이름으로 올린다 (별칭 deployed). 배포 여부의 기준은 DB 레지스트리
(model_versions)이고 MLflow는 사본이다. 테스트는 MemoryTracker를 쓴다.

WP13, ADR 0016. 공개 이름: `Tracker`(프로토콜), `MlflowTracker`(REST 2.0, 서버가 --serve-artifacts로
산출물을 대신 받아야 한다), `MemoryTracker`·`MemoryRun`(테스트용), `TrackingError`, `RunStatus`.
서버 주소는 `DLP_MLFLOW_URL`, 없으면 `http://localhost:${DLP_MLFLOW_PORT:-5000}` (`make up`).
NaN·무한 지표는 MLflow가 받지 않으므로 기록하지 않는다 (`_finite`).
"""

# 외부 JSON 응답을 다루므로 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

RunStatus = Literal["FINISHED", "FAILED"]  # MLflow 실행 종료 상태


class Tracker(Protocol):
    """실험 기록기 인터페이스 (재학습 루프가 쓴다)."""

    def start(self, experiment: str, run_name: str, tags: dict[str, str]) -> str:
        """실행을 만들고 실행 ID를 돌려준다."""
        ...

    def log_params(self, run_id: str, params: dict[str, str]) -> None: ...
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...
    def log_artifact(self, run_id: str, path: Path, artifact_path: str) -> str:
        """산출물을 올리고 그 위치(URI)를 돌려준다."""
        ...

    def finish(self, run_id: str, status: RunStatus) -> None: ...
    def register(self, name: str, run_id: str, source: str, alias: str) -> str:
        """모델 레지스트리에 버전을 만들고 별칭을 옮긴다. 레지스트리 버전 번호를 돌려준다."""
        ...


class TrackingError(RuntimeError):
    """MLflow 호출 실패 (상태 코드와 응답 앞부분을 메시지에 담는다)."""


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    """NaN·무한 값을 뺀 지표 (MLflow는 유한 값만 받는다. 정의되지 않은 지표는 기록하지 않는다)."""
    return {k: v for k, v in metrics.items() if math.isfinite(v)}


class MlflowTracker:
    """MLflow REST API 2.0 기록기.

    부작용: MLflow 서버에 실험·실행·파라미터·지표·산출물·등록 모델을 만든다 (외부 서비스 호출).
    HTTP 클라이언트는 닫지 않는다 (CLI 프로세스 수명 동안 쓴다).
    """

    def __init__(self, url: str, *, timeout: float = 60) -> None:
        """url: MLflow 서버 기본 주소. timeout: 요청 제한 시간 (초)."""
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        # 실행 ID → 그 실행의 artifact_uri (log_artifact가 업로드 경로를 만든다)
        self._artifact_roots: dict[str, str] = {}

    @classmethod
    def from_env(cls) -> MlflowTracker:
        """환경 변수 `DLP_MLFLOW_URL`(없으면 localhost:`DLP_MLFLOW_PORT`, 기본 5000)로 만든다."""
        url = os.environ.get("DLP_MLFLOW_URL") or (
            f"http://localhost:{os.environ.get('DLP_MLFLOW_PORT', '5000')}"
        )
        return cls(url)

    def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """`/api/2.0/mlflow/<path>` 호출. 200이 아니면 TrackingError. 빈 응답은 {}."""
        resp = self.http.request(method, f"/api/2.0/mlflow/{path}", **kwargs)
        if resp.status_code != 200:
            raise TrackingError(f"MLflow {path}: {resp.status_code} {resp.text[:300]}")
        data: dict[str, Any] = resp.json() if resp.content else {}
        return data

    def _experiment(self, name: str) -> str:
        """실험 ID. 이름으로 찾고, 없으면(200이 아니면) 만든다."""
        resp = self.http.get(
            "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name}
        )
        if resp.status_code == 200:
            return str(resp.json()["experiment"]["experiment_id"])
        return str(self._call("POST", "experiments/create", json={"name": name})["experiment_id"])

    def start(self, experiment: str, run_name: str, tags: dict[str, str]) -> str:
        """실행을 만든다 (시작 시각 = 지금, ms). 실행 ID를 돌려준다."""
        info = self._call(
            "POST",
            "runs/create",
            json={
                "experiment_id": self._experiment(experiment),
                "run_name": run_name,
                "start_time": int(time.time() * 1000),
                "tags": [{"key": k, "value": v} for k, v in tags.items()],
            },
        )["run"]["info"]
        run_id = str(info["run_id"])
        self._artifact_roots[run_id] = str(info["artifact_uri"])
        return run_id

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        """파라미터를 한 번에 남긴다 (MLflow 파라미터는 다시 쓰면 같은 값이어야 한다)."""
        self._call(
            "POST",
            "runs/log-batch",
            json={"run_id": run_id, "params": [{"key": k, "value": v} for k, v in params.items()]},
        )

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """유한한 지표만 step 0으로 남긴다 (1000개씩 나눠 보낸다)."""
        now = int(time.time() * 1000)
        items = [
            {"key": k, "value": v, "timestamp": now, "step": 0} for k, v in _finite(metrics).items()
        ]
        for i in range(0, len(items), 1000):  # log-batch 한 번에 지표 1000개까지
            self._call(
                "POST", "runs/log-batch", json={"run_id": run_id, "metrics": items[i : i + 1000]}
            )

    def log_artifact(self, run_id: str, path: Path, artifact_path: str) -> str:
        """파일 하나를 MLflow 산출물 프록시(`mlflow-artifacts:`)로 올린다.

        Returns:
            `<artifact_uri>/<artifact_path>/<파일 이름>`.

        Raises:
            TrackingError: 서버가 산출물을 대신 받지 않거나(--serve-artifacts 없음) 업로드가 실패할
            때.
        """
        root = self._artifact_roots[run_id]
        if not root.startswith("mlflow-artifacts:"):
            raise TrackingError(f"서버가 산출물을 대신 받지 않습니다 (--serve-artifacts): {root}")
        rel = f"{root.removeprefix('mlflow-artifacts:').strip('/')}/{artifact_path}/{path.name}"
        resp = self.http.put(
            f"/api/2.0/mlflow-artifacts/artifacts/{rel}", content=path.read_bytes()
        )
        if resp.status_code != 200:
            raise TrackingError(f"MLflow 산출물 업로드 실패: {resp.status_code} {resp.text[:300]}")
        return f"{root}/{artifact_path}/{path.name}"

    def finish(self, run_id: str, status: RunStatus) -> None:
        """실행을 끝낸다 (상태와 종료 시각)."""
        self._call(
            "POST",
            "runs/update",
            json={"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)},
        )

    def register(self, name: str, run_id: str, source: str, alias: str) -> str:
        """등록 모델(없으면 만든다)에 새 버전을 만들고 별칭을 그 버전으로 옮긴다.

        멱등이 아니다: 부를 때마다 새 레지스트리 버전이 생긴다. 버전 번호(문자열)를 돌려준다.
        """
        resp = self.http.post("/api/2.0/mlflow/registered-models/create", json={"name": name})
        if resp.status_code != 200 and "RESOURCE_ALREADY_EXISTS" not in resp.text:
            raise TrackingError(f"MLflow 등록 모델 생성 실패: {resp.status_code} {resp.text[:300]}")
        version = str(
            self._call(
                "POST",
                "model-versions/create",
                json={"name": name, "source": source, "run_id": run_id},
            )["model_version"]["version"]
        )
        self._call(
            "POST",
            "registered-models/alias",
            json={"name": name, "alias": alias, "version": version},
        )
        return version


@dataclass
class MemoryRun:
    """메모리 기록기의 실행 하나."""

    experiment: str
    name: str
    tags: dict[str, str]
    params: dict[str, str] = field(default_factory=dict[str, str])
    metrics: dict[str, float] = field(default_factory=dict[str, float])
    artifacts: dict[str, bytes] = field(default_factory=dict[str, bytes])  # "경로/파일" → 내용
    status: RunStatus | None = None  # 끝나기 전에는 None


@dataclass
class MemoryTracker:
    """테스트용. 기록을 메모리에 둔다."""

    runs: dict[str, MemoryRun] = field(default_factory=dict[str, MemoryRun])
    registry: dict[str, list[str]] = field(default_factory=dict[str, list[str]])  # 이름 → 실행
    # (등록 모델 이름, 별칭) → 레지스트리 버전 번호
    aliases: dict[tuple[str, str], str] = field(default_factory=dict[tuple[str, str], str])

    def start(self, experiment: str, run_name: str, tags: dict[str, str]) -> str:
        """실행 ID "mem-<순번>"을 만든다."""
        run_id = f"mem-{len(self.runs) + 1}"
        self.runs[run_id] = MemoryRun(experiment, run_name, dict(tags))
        return run_id

    def log_params(self, run_id: str, params: dict[str, str]) -> None:
        """파라미터를 덮어쓴다."""
        self.runs[run_id].params.update(params)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """유한한 지표만 남긴다 (MlflowTracker와 같은 규칙)."""
        self.runs[run_id].metrics.update(_finite(metrics))

    def log_artifact(self, run_id: str, path: Path, artifact_path: str) -> str:
        """파일 내용을 메모리에 담고 `memory://<실행>/<경로>/<파일>`을 돌려준다."""
        key = f"{artifact_path}/{path.name}"
        self.runs[run_id].artifacts[key] = path.read_bytes()
        return f"memory://{run_id}/{key}"

    def finish(self, run_id: str, status: RunStatus) -> None:
        """실행 상태를 기록한다."""
        self.runs[run_id].status = status

    def register(self, name: str, run_id: str, source: str, alias: str) -> str:
        """등록할 때마다 버전 번호를 1씩 늘리고 별칭을 옮긴다 (source는 쓰지 않는다)."""
        versions = self.registry.setdefault(name, [])
        versions.append(run_id)
        self.aliases[(name, alias)] = str(len(versions))
        return str(len(versions))
