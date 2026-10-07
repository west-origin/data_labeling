"""운영 정책 (config/policies/ops.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class CostPolicy(Contract):
    hourly_cost: float | None = Field(default=None, gt=0)
    currency: str


class AlertPolicy(Contract):
    auto_approval_rise: float = Field(ge=0)
    detection_drop: float = Field(ge=0)
    stagnation_weeks: int = Field(ge=2)


class AuditPolicy(Contract):
    service_accounts: tuple[str, ...]
    service_purposes: tuple[str, ...]  # 서비스 계정이 원본에 접근해도 되는 용도 (파이프라인 명령)
    raw_viewers: tuple[str, ...]
    timezone: str
    off_hours: tuple[int, int]


class RetentionPolicy(Contract):
    alert_days_before: int = Field(ge=0)


class OpsPolicy(Contract):
    version: int
    cost: CostPolicy
    alerts: AlertPolicy
    audit: AuditPolicy
    retention: RetentionPolicy


def load_policy(root: Path) -> OpsPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "ops.yaml").read_text("utf-8"))
    return OpsPolicy.model_validate(data)
