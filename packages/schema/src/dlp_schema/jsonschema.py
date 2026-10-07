"""계약 타입의 JSON Schema 생성. 다른 언어·도구(검수 UI 등)가 같은 계약을 쓰게 한다.

위치
    `dlp schema export`(= `make schemas`)가 `write_schemas`로 저장소의 `schemas/*.schema.json`을
    다시 쓰고, `dlp schema export --check`(`make contracts`, `make check`에 포함)와 계약 테스트가
    `stale_schemas`로 커밋된 파일이 코드와 같은지 검사한다 (WP1).

주요 이름
    - `CONTRACTS`: 파일 이름 접두사 → 최상위 계약 타입. 여기 없는 타입은 이 타입들에 중첩된 경우에만
      스키마(`$defs`)에 나타난다.
    - `render_schemas`: 파일 이름 → JSON 문자열 (메모리에서만).
    - `write_schemas`: 디렉터리에 파일로 쓴다.
    - `stale_schemas`: 디렉터리의 파일 중 코드와 다른(또는 없는) 것의 이름.

주의
    - Pydantic은 클래스 docstring과 `Field(description=...)`을 스키마의 description으로 넣는다.
      계약 타입의 docstring·description을 바꾸면 `schemas/`를 다시 생성해야 한다.
    - 출력은 `sort_keys=True`, 들여쓰기 2, 끝 줄바꿈 하나로 고정해 diff가 안정적이다.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from dlp_schema.config import PlatformConfig
from dlp_schema.dataset import DatasetVersion
from dlp_schema.episode import EpisodeGraph
from dlp_schema.export import IntervalFile
from dlp_schema.labels import LabelRecord
from dlp_schema.lineage import ExportRecord, GoldenSet, ModelVersion, TrainingRun, Withdrawal
from dlp_schema.migration import OntologyMigration
from dlp_schema.ontology import Ontology
from dlp_schema.ops import PrivacyAuditRecord, RawAccessEvent, RetentionDecision, ReviewWork
from dlp_schema.review import ReviewAssignment, ReviewTask
from dlp_schema.session import LifecycleEvent, Session

# 생성할 스키마 목록. 키가 파일 이름이 된다 (`<키>.schema.json`).
# 새 최상위 계약을 추가하면 여기에 등록하고 `make schemas`로 파일을 만든다.
CONTRACTS: dict[str, type[BaseModel]] = {
    "session": Session,
    "lifecycle_event": LifecycleEvent,
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
    "model_version": ModelVersion,
    "export_record": ExportRecord,
    "withdrawal": Withdrawal,
    "export_intervals": IntervalFile,
    "raw_access_event": RawAccessEvent,
    "review_work": ReviewWork,
    "privacy_audit": PrivacyAuditRecord,
    "retention_decision": RetentionDecision,
}


def render_schemas() -> dict[str, str]:
    """파일 이름 → JSON 문자열.

    Returns:
        `{"<이름>.schema.json": "<JSON 텍스트>\\n"}`. 한글 description이 그대로 보이도록
        `ensure_ascii=False`로 직렬화한다. 파일을 쓰지 않는다.
    """
    return {
        f"{name}.schema.json": json.dumps(
            model.model_json_schema(), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
        for name, model in CONTRACTS.items()
    }


def write_schemas(out_dir: Path) -> list[Path]:
    """모든 스키마를 `out_dir`에 UTF-8로 쓴다 (덮어쓴다).

    Args:
        out_dir: 출력 디렉터리 (보통 저장소의 `schemas/`). 없으면 만든다.

    Returns:
        쓴 파일 경로 목록 (`CONTRACTS` 순서).

    부작용:
        파일 시스템 쓰기. `CONTRACTS`에서 빠진 옛 스키마 파일은 지우지 않는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, text in render_schemas().items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def stale_schemas(out_dir: Path) -> list[str]:
    """저장소의 스키마 파일이 코드와 다르면 그 파일 이름 목록.

    파일이 없거나 내용이 한 글자라도 다르면 포함한다. 빈 목록이면 최신이다.
    `CONTRACTS`에 없는 여분 파일은 검사하지 않는다.
    """
    return [
        name
        for name, text in render_schemas().items()
        if not (out_dir / name).is_file() or (out_dir / name).read_text(encoding="utf-8") != text
    ]
