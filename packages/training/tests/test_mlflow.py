"""MLflow 서버(make up)에 실행·지표·산출물·등록 모델을 남기고 다시 읽어 확인한다.

WP13, ADR 0016. `@pytest.mark.services` (실행 중인 MLflow 필요). 실험 이름에 난수를 붙여 다른
실행과 겹치지 않게 한다.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from dlp_train.tracking import MlflowTracker

pytestmark = pytest.mark.services


def test_mlflow_tracker_round_trip(tmp_path: Path) -> None:
    """기록한 것을 REST로 다시 읽어 같은지 본다: NaN 지표는 빠지고, 등록마다 버전이 늘고 별칭이
    옮겨진다."""
    tracker = MlflowTracker.from_env()
    name = f"dlp-test-{uuid.uuid4().hex[:8]}"
    run = tracker.start(name, "run-1", {"dlp.task": "objects"})
    tracker.log_params(run, {"jitter_px": "0.0"})
    tracker.log_metrics(run, {"golden/hota": 0.9, "golden/ece": float("nan")})
    artifact = tmp_path / "model.json"
    artifact.write_text('{"ok": true}')
    uri = tracker.log_artifact(run, artifact, "model")
    tracker.finish(run, "FINISHED")
    version = tracker.register(name, run, f"runs:/{run}/model", "deployed")
    assert version == "1"
    assert tracker.register(name, run, f"runs:/{run}/model", "deployed") == "2"

    http = tracker.http
    info = http.get("/api/2.0/mlflow/runs/get", params={"run_id": run}).json()["run"]
    assert info["info"]["status"] == "FINISHED"
    assert {p["key"]: p["value"] for p in info["data"]["params"]} == {"jitter_px": "0.0"}
    metrics = {m["key"]: m["value"] for m in info["data"]["metrics"]}
    assert metrics == {"golden/hota": 0.9}  # NaN은 남기지 않는다
    rel = uri.removeprefix("mlflow-artifacts:").strip("/")
    got = http.get(f"/api/2.0/mlflow-artifacts/artifacts/{rel}")
    assert got.status_code == 200 and got.content == b'{"ok": true}'
    alias = http.get(
        "/api/2.0/mlflow/registered-models/alias", params={"name": name, "alias": "deployed"}
    ).json()
    assert alias["model_version"]["version"] == "2"
