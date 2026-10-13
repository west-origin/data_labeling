"""설명 라벨 영어 번역 정책 (config/policies/translate.yaml, ADR 0043).

- `TranslatePolicy`: 최상위. `digest`(server 절을 뺀 파싱값의 해시, 12자)가 번역 모델 버전에
  들어간다
  (`translate.translation_version`).
- `TranslateBackend`: 번역 서버 요청 설정 (해시 포함).
- `TranslateServer`: 사내 주소 대역·시간 초과·백오프 (해시 제외, 결과에 영향 없음).
- `load_policy`: 저장소 루트 → `TranslatePolicy`.

`Contract`라 알 수 없는 키는 적재 오류다. 프롬프트 틀 버전은 코드
(`dlp_actions.translators.PROMPT_VERSION`)와 같아야 한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract

# 번역 프롬프트 틀 버전. 틀 문장(`dlp_actions.translators`)을 바꾸면 올리고 정책도 함께 올린다
PROMPT_VERSION = 1


class TranslateBackend(Contract):
    """`translate.yaml backend`: 사내 OpenAI 호환 서버에 보낼 번역 요청 설정."""

    # 지금 허용하는 유일한 값 (외부 API 금지)
    kind: Literal["openai_compatible"]
    # 모델 이름 (라이선스 검사 `config/models.yaml external`, 버전 문자열에 들어간다)
    model: str = Field(min_length=1)
    # 서버에 등록된 이름이 다르면 그 이름 (None이면 model)
    served_name: str | None = None
    # 응답 생성 토큰 상한
    max_tokens: int = Field(ge=16)
    # 번역문 글자 수 상한
    max_chars: int = Field(ge=1)
    # 응답 위반 시 다시 묻는 횟수
    max_retries: int = Field(ge=0)
    # 번역문에 한글이 남으면 위반
    reject_hangul: bool = True
    # 용어집: matched(원문에 나온 이름만) | all | off
    glossary: Literal["matched", "all", "off"]


class TranslateServer(Contract):
    """`translate.yaml server`: 접속 규칙 (해시 제외)."""

    # 사내 대역 CIDR (밖이면 보내지 않는다)
    internal_hosts: tuple[str, ...] = Field(min_length=1)
    # 요청 하나의 시간 초과 초
    timeout_s: float = Field(gt=0)
    # 서버 장애 뒤 대기 초 목록 (길이 = 재시도 횟수)
    backoff_s: tuple[float, ...] = ()


class TranslatePolicy(Contract):
    """`config/policies/translate.yaml` 전체."""

    # 정책 형식 버전
    version: int
    # 번역 프롬프트 틀 버전 (코드 PROMPT_VERSION과 같아야 한다)
    prompt_version: int
    # 원문·번역 언어 코드
    source_language: str = Field(min_length=2)
    target_language: str = Field(min_length=2)
    # 원문 신뢰도가 없을 때 번역 레코드 신뢰도
    default_confidence: float = Field(ge=0, le=1)
    backend: TranslateBackend
    server: TranslateServer

    @model_validator(mode="after")
    def _check(self) -> TranslatePolicy:
        """프롬프트 틀 버전이 코드와 같고, 원문·번역 언어가 다른지 본다."""
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"prompt_version {self.prompt_version}이 코드 PROMPT_VERSION "
                f"{PROMPT_VERSION}과 다릅니다"
            )
        if self.source_language == self.target_language:
            raise ValueError("source_language와 target_language가 같습니다")
        return self

    @property
    def digest(self) -> str:
        """정책 해시 (sha256 앞 12자). server 절은 결과를 바꾸지 않아 뺀다."""
        data = self.model_dump(mode="json", exclude={"server"})
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_policy(root: Path) -> TranslatePolicy:
    """`<root>/config/policies/translate.yaml`을 읽어 검증한다.

    Args:
        root: 저장소 루트 (`dlp_schema.repo_root()`).

    Raises:
        pydantic.ValidationError: 키가 없거나, 모르는 키가 있거나, 값이 범위를 벗어날 때.
    """
    path = root / "config" / "policies" / "translate.yaml"
    data: Any = yaml.safe_load(path.read_text("utf-8"))
    return TranslatePolicy.model_validate(data)
