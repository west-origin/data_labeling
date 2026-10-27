"""재학습·성과 화면 자료 (읽기 전용, ADR 0061 5절).

- 회차: DB `model_versions`(`dlp train models`와 같은 레지스트리) 순서대로. 각 회차의 골든셋
  지표·게이트 결과는 평가 리포트(`report_uri`, MLflow 산출물 버킷의 golden.json)에서 읽어 작업
  디렉터리에 캐시한다 (리포트는 모델 버전마다 한 번 쓰고 바뀌지 않는다). 원본 버킷은 읽지 않는다.
- 학습기 이름은 회차의 `trainer`와 `training.yaml` 과제 템플릿에서 읽는다. `oracle-stub`처럼 시험용
  학습기면 화면에 "시험용 학습기" 배지를 단다 (정답 좌표를 흔드는 CI용이라 지표가 실제 성능이
  아니다).
- 수정률 추이: `dlp ops weekly --out`이 쓴 JSON(작업 디렉터리 reports/weekly-*.json) 중 가장 최근
  것.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml

from dlp_schema.db.repository import list_model_versions
from dlp_schema.lineage import ModelVersion

# 시험용 학습기 이름 접두 (dlp_train.trainers의 CI용 학습기)
TEST_TRAINER_PREFIXES = ("oracle",)

# report_uri → 로컬 파일로 받는 함수 (S3 키 해석은 호출자)
ReportFetcher = Callable[[str, Path], None]


def is_test_trainer(name: str | None) -> bool:
    return bool(name) and str(name).startswith(TEST_TRAINER_PREFIXES)


def policy_trainers(root: Path) -> dict[str, str]:
    """`training.yaml` 과제 → 기본 학습기 이름 (표시용, 값 검증은 dlp train이 한다)."""
    data: Any = yaml.safe_load((root / "config" / "policies" / "training.yaml").read_text("utf-8"))
    tasks: dict[str, Any] = data.get("tasks", {})
    return {str(k): str(v.get("trainer", "")) for k, v in tasks.items()}


def _finite(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def summarize_report(data: dict[str, Any], task: str) -> dict[str, Any]:
    """golden.json → 그 과제의 지표(유한 값만)·게이트 통과·이유."""
    overall: dict[str, Any] = data.get("overall", {}).get(task, {})
    metrics = {k: v for k, v in overall.get("metrics", {}).items() if _finite(v) is not None}
    gate: dict[str, Any] = data.get("gate", {})
    task_gate: dict[str, Any] = gate.get("tasks", {}).get(task, {})
    return {
        "golden_version": data.get("golden_version"),
        "metrics": metrics,
        "sessions": overall.get("sessions"),
        "gate_passed": gate.get("passed"),
        "gate_reasons": list(task_gate.get("reasons", [])),
        "lower_is_better": list(data.get("lower_is_better", [])),
    }


class ReportCache:
    """평가 리포트 캐시 (작업 디렉터리 cache/reports/<모델 버전>.json)."""

    def __init__(self, work_dir: Path, fetch: ReportFetcher) -> None:
        self.dir = work_dir / "cache" / "reports"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fetch = fetch

    def load(self, mv: ModelVersion) -> dict[str, Any] | None:
        """그 회차의 리포트. 없거나 받지 못하면 None (화면은 '리포트 없음')."""
        if mv.report_uri is None:
            return None
        path = self.dir / f"{mv.model_version}.json"
        if not path.is_file():
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "golden.json"
                try:
                    self.fetch(mv.report_uri, local)
                except Exception:  # 저장소 장애·없는 키: 화면에 '받지 못함'만 보인다
                    return None
                path.write_bytes(local.read_bytes())
        try:
            data: dict[str, Any] = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            return None
        return data


def rounds(
    conn: sa.Connection, cache: ReportCache, trainers: dict[str, str]
) -> list[dict[str, Any]]:
    """과제별 회차 목록 (등록 순)."""
    out: list[dict[str, Any]] = []
    for mv in list_model_versions(conn):
        report = cache.load(mv)
        out.append(
            {
                "model_version": mv.model_version,
                "task": mv.task,
                "trainer": mv.trainer,
                "test_trainer": is_test_trainer(mv.trainer),
                "policy_trainer": trainers.get(mv.task),
                "status": mv.status.value,
                "train_examples": mv.train_examples,
                "created_at": mv.created_at.isoformat(),
                "report": summarize_report(report, mv.task) if report else None,
            }
        )
    return out


def latest_weekly(reports_dir: Path) -> dict[str, Any] | None:
    """가장 최근 `dlp ops weekly` JSON의 주별 수정률·자동 승인율·검수 시간."""
    files = sorted(reports_dir.glob("weekly-*.json"))
    if not files:
        return None
    try:
        data: dict[str, Any] = json.loads(files[-1].read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    keys = ("correction_rate", "auto_approval_rate", "verified_episodes", "review_hours")
    weeks = [{"week": w.get("week"), **{k: w.get(k) for k in keys}} for w in data.get("weeks", [])]
    return {"file": files[-1].name, "weeks": weeks, "alerts": list(data.get("alerts", []))}
