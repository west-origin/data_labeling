"""계약 타입의 JSON Schema 생성. 다른 언어·도구(검수 UI 등)가 같은 계약을 쓰게 한다."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from dlp_schema.config import PlatformConfig
from dlp_schema.dataset import DatasetVersion
from dlp_schema.episode import EpisodeGraph
from dlp_schema.labels import LabelRecord
from dlp_schema.lineage import ExportRecord, GoldenSet, TrainingRun, Withdrawal
from dlp_schema.migration import OntologyMigration
from dlp_schema.ontology import Ontology
from dlp_schema.review import ReviewAssignment, ReviewTask
from dlp_schema.session import Session

CONTRACTS: dict[str, type[BaseModel]] = {
    "session": Session,
    "label_record": LabelRecord,
    "episode_graph": EpisodeGraph,
    "dataset_version": DatasetVersion,
    "ontology": Ontology,
    "ontology_migration": OntologyMigration,
    "platform_config": PlatformConfig,
    "review_task": ReviewTask,
    "review_assignment": ReviewAssignment,
    "golden_set": GoldenSet,
    "training_run": TrainingRun,
    "export_record": ExportRecord,
    "withdrawal": Withdrawal,
}


def render_schemas() -> dict[str, str]:
    """파일 이름 → JSON 문자열."""
    return {
        f"{name}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for name, model in CONTRACTS.items()
    }


def write_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in render_schemas().items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def stale_schemas(out_dir: Path) -> list[str]:
    """저장소의 스키마 파일이 코드와 다르면 그 파일 이름 목록."""
    return [
        name
        for name, text in render_schemas().items()
        if not (out_dir / name).is_file() or (out_dir / name).read_text(encoding="utf-8") != text
    ]
