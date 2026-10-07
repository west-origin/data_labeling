"""액티브 러닝 정책 (config/policies/active.yaml) 로더 (WP14).

`load_policy(root)`가 YAML을 읽어 `ActivePolicy`로 검증한다. 점수 항목 이름(`score.terms`의 키)이
등록돼 있는지는 여기서가 아니라 점수를 매길 때(`dlp_active.terms.build_terms`) 확인한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract
from dlp_schema.session import LifecycleState


class CorrectionPolicy(Contract):
    """active.yaml `correction` 절: 클래스별 수정률 계산."""

    # 사람이 놓쳐서 새로 추가한 라벨을 "수정"으로 셀지 (분모·분자 모두에 들어간다)
    count_added: bool
    # 베이즈 평활 강도: 전체 수정률을 이만큼의 가상 검수 수로 섞는다 (0이면 평활 없음)
    prior_weight: float = Field(ge=0)


class CandidatePolicy(Contract):
    """active.yaml `candidates` 절: 점수를 매길 후보 세션."""

    lifecycle: tuple[LifecycleState, ...] = Field(min_length=1)  # 후보가 되는 생애주기 상태
    exclude_golden: bool  # 참이면 골든셋 세션을 후보에서 뺀다 (사람이 처음부터 라벨링)


class ScorePolicy(Contract):
    """active.yaml `score` 절: 세션 점수 = Σ 가중치 * 항목 점수."""

    # total: 세션 전체 합, per_minute: 영상 1분당 (긴 세션이 유리해지는 것을 막을 때)
    normalize: Literal["total", "per_minute"]
    terms: dict[str, float] = Field(min_length=1)  # 항목 이름(`register_term`) → 가중치


class FiftyOnePolicy(Contract):
    """active.yaml `fiftyone` 절."""

    dataset_prefix: str  # FiftyOne 데이터셋 이름 접두어 (CLI 기본 이름: <접두어>top<N>)
    # 라벨 키프레임 시각과 블러본 프레임 시각의 허용 차이 (ms, 반올림 오차만)
    frame_tolerance_ms: int = Field(ge=0)


class ActivePolicy(Contract):
    """active.yaml 전체."""

    version: int  # 정책 파일 형식 버전
    correction: CorrectionPolicy
    candidates: CandidatePolicy
    excluded_kinds: tuple[str, ...]  # 수정률·점수·FiftyOne에서 빼는 라벨 종류 (blur_track)
    score: ScorePolicy
    select: int = Field(ge=1)  # 한 번에 고르는 세션 수 (`rank_sessions`의 limit 기본값)
    fiftyone: FiftyOnePolicy


def load_policy(root: Path) -> ActivePolicy:
    """`<root>/config/policies/active.yaml`을 읽어 검증한다.

    Raises:
        pydantic.ValidationError: 키가 빠졌거나 범위를 어길 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "active.yaml").read_text("utf-8"))
    return ActivePolicy.model_validate(data)
