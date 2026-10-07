"""데이터셋 정책 (config/policies/dataset.yaml) 로더 (WP7).

`load_policy(root)`가 YAML을 읽어 `DatasetPolicy`로 검증한다. 분할 비율·골든셋 크기·프라이버시 조건·
스냅샷 이력 포함 여부·lakeFS 위치를 담는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class LakeFSPolicy(Contract):
    """dataset.yaml `lakefs` 절: 스냅샷을 커밋할 lakeFS 저장소."""

    repository: str  # lakeFS 저장소 이름 (없으면 LakeFSSnapshotStore가 만든다)
    branch: str  # 커밋할 브랜치 (저장소 기본 브랜치)
    storage_namespace: str  # 저장소 생성 때 쓰는 실제 객체 저장 위치 (s3://…)


class DatasetPolicy(Contract):
    """dataset.yaml 전체."""

    version: int  # 정책 파일 형식 버전
    # 학습·검증 후보(골든·holdout 제외) 중 검증 세션 비율 목표 (0~1 개구간)
    val_ratio: float = Field(gt=0, lt=1)
    golden_sessions_per_domain: int = Field(gt=0)  # 골든셋 제안 때 도메인별 목표 세션 수
    # 데이터셋 후보가 될 수 있는 세션의 privacy_state 값 (`PrivacyState` 값 문자열, 보통 approved)
    eligible_privacy_state: str
    # 참이면 스냅샷 labels.jsonl에 수정 이력 전체(자동 원본과 수정본)를, 거짓이면 현재 라벨만
    include_label_history: bool
    lakefs: LakeFSPolicy


def load_policy(root: Path) -> DatasetPolicy:
    """`<root>/config/policies/dataset.yaml`을 읽어 검증한다.

    Raises:
        pydantic.ValidationError: 키가 빠졌거나 범위를 어길 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "dataset.yaml").read_text("utf-8"))
    return DatasetPolicy.model_validate(data)
