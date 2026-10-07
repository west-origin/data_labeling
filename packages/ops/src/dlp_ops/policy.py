"""운영 정책 (config/policies/ops.yaml) 로더와 타입 (WP16).

`dlp ops weekly|audit-report|retention` 이 이 정책을 읽는다. 값의 의미·단위는
`config/policies/ops.yaml`의 주석에 자세히 적었다.

공개:
- `OpsPolicy` — 정책 전체 (`version`, `cost`, `alerts`, `audit`, `retention`).
- `load_policy(root)` — `<root>/config/policies/ops.yaml`을 읽어 검증한다.

주의: `Contract` 기반이라 알 수 없는 키가 있으면 검증 오류이고, 만든 뒤에는 바꿀 수 없다(frozen).
이 정책은 모델 버전 해시에 들어가지 않으므로(운영 리포트 전용) 값을 바꿔도 다른 단계가 다시 돌지
않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class CostPolicy(Contract):
    """에피소드당 생산원가 계산 설정 (`ops.yaml cost`)."""

    # 검수자 시간당 인건비 (`currency` 단위). None이면 원가 지표를 내지 않는다 (Gate 0에서 정함).
    hourly_cost: float | None = Field(default=None, gt=0)
    # 통화 코드 (예: KRW). 리포트 표시에만 쓴다.
    currency: str


class AlertPolicy(Contract):
    """주간 지표 경고 규칙 (`ops.yaml alerts`). 비율 값은 0~1 사이 소수(퍼센트포인트/100)다."""

    # 자동 승인율이 직전 주 대비 이만큼 이상 올랐고…
    auto_approval_rise: float = Field(ge=0)
    # …오류 삽입 발견율이 이만큼 이상 떨어졌으면 "검수 품질 저하" 경고
    detection_drop: float = Field(ge=0)
    # 검수 시간과 수정률이 이 주 수 동안 줄지 않으면 정체 경고 (비교하려면 2주 이상)
    stagnation_weeks: int = Field(ge=2)


class AuditPolicy(Contract):
    """원본 접근 감사 규칙 (`ops.yaml audit`, ADR 0020·0021)."""

    # 사람이 아닌 파이프라인 계정 이름 (`DLP_ACTOR` 값). 이들의 접근은 용도로 검사한다.
    service_accounts: tuple[str, ...]
    service_purposes: tuple[str, ...]  # 서비스 계정이 원본에 접근해도 되는 용도 (파이프라인 명령)
    # `review.yaml reviewers.privacy` 외에 원본을 볼 수 있는 사람 (감사인 등)
    raw_viewers: tuple[str, ...]
    # IANA 시간대 이름. 업무 시간 판정과 월간 리포트의 달 경계에 쓴다.
    timezone: str
    # 업무 시간 밖 (시작 시, 끝 시). 시작 > 끝이면 자정을 넘는 구간
    # (예: [22, 6] = 22시부터 다음날 6시 전까지).
    off_hours: tuple[int, int]


class RetentionPolicy(Contract):
    """원본 보관 만료 알림 (`ops.yaml retention`). 보관 기간 자체는 `defaults.yaml`에 있다."""

    # 만료일 이 일수 전부터 `due_soon`으로 알린다 (0이면 만료 당일부터).
    alert_days_before: int = Field(ge=0)


class OpsPolicy(Contract):
    """`config/policies/ops.yaml` 전체."""

    # 정책 파일 형식 버전 (현재 1). 형식을 바꿀 때 올린다.
    version: int
    cost: CostPolicy
    alerts: AlertPolicy
    audit: AuditPolicy
    retention: RetentionPolicy


def load_policy(root: Path) -> OpsPolicy:
    """`<root>/config/policies/ops.yaml`을 읽어 `OpsPolicy`로 검증한다.

    인자:
        root: 저장소 루트 (보통 `dlp_schema.repo_root()`).
    예외: 파일이 없으면 `FileNotFoundError`, 값이 틀리면 `pydantic.ValidationError`.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "ops.yaml").read_text("utf-8"))
    return OpsPolicy.model_validate(data)
