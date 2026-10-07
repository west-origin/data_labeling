"""데이터셋 정책 (config/policies/dataset.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class LakeFSPolicy(Contract):
    repository: str
    branch: str
    storage_namespace: str


class DatasetPolicy(Contract):
    version: int
    val_ratio: float = Field(gt=0, lt=1)
    golden_sessions_per_domain: int = Field(gt=0)
    eligible_privacy_state: str
    include_label_history: bool
    lakefs: LakeFSPolicy


def load_policy(root: Path) -> DatasetPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "dataset.yaml").read_text("utf-8"))
    return DatasetPolicy.model_validate(data)
