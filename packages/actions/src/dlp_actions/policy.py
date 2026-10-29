"""행동 구간 정책 (config/policies/actions.yaml, WP10, ADR 0012).

- `ActionsPolicy`: 최상위. `stamp`(파싱된 값의 해시 `digest` 12자 + 틀 버전 표시)가 모델
  버전에 들어간다 (`runner.run_actions`: `actions-<stamp>+<VLM 버전>+i<입력 해시>`). YAML
  주석·서식은 해시에 영향이 없고, 값이 바뀌면 버전이 바뀌어 다음 실행에서 다시 만든다.
- `BoundaryPolicy`: 1단 경계 후보 (`boundaries.py`)
- `ToleranceMs`: 평가 허용 오차 (테스트·평가 하네스용, 실행 결과에는 영향 없음)
- `VlmPolicy`: 2단 VLM 분류
- `load_policy`: 저장소 루트 → `ActionsPolicy`

`Contract`라 알 수 없는 키는 적재 오류다. 프롬프트 틀 버전 `prompt_version`은 코드
`PROMPT_VERSION`과 같아야 하고, 해시 대신 `stamp`에 따로 붙는다 (F-PAR-84).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract

# VLM 프롬프트 틀 버전: 행동 분류(`vlm.build_prompt`·`vlm.classify` 재시도 문구)와 설명 재생성
# (`describe.describe_prompt`·`describe.ask_description` 재시도 문구). 문구는 결과를 바꾸는데 정책
# 해시에 들어가지 않으므로, 문구를 바꾸면 이 값과 actions.yaml prompt_version을 함께 올린다.
# 올리면 `stamp`가 바뀌어 다음 `dlp actions run|describe`가 검수 전 결과를 새로 만든다.
PROMPT_VERSION = 1
# 첫 틀 버전. 이 값이면 버전 문자열에 표시를 붙이지 않는다 (지금 결과의 버전 문자열이 그대로다)
BASE_PROMPT_VERSION = 1


class BoundaryPolicy(Contract):
    """`actions.yaml boundaries`: 손목 속도·접촉 기반 경계 후보.

    속도 단위는 손바닥 길이(손목 → 가운뎃손가락 뿌리, hand21 0 → 9)/초, 시간은 ms.
    """

    # 속도 이동 평균 창 ms
    smooth_ms: int = Field(ge=0)
    # 이보다 느리면 정지
    still_speed: float = Field(gt=0)
    # 정지가 이만큼(ms) 이어지면 시작·끝을 후보로
    min_still_ms: int = Field(ge=0)
    # 접촉 밖 속도 골짜기의 상한
    valley_speed: float = Field(gt=0)
    # 골짜기 ≤ 앞뒤 창 안 최고 속도의 이 비율
    valley_ratio: float = Field(gt=0, le=1)
    # 골짜기 앞뒤 비교 창 ms
    valley_window_ms: int = Field(ge=0)
    # 골짜기 후보 사이 최소 간격 ms
    min_valley_separation_ms: int = Field(ge=0)
    # 이 이하로 가까운 후보는 하나로 (접촉 > 정지 > 골짜기 순 우선)
    merge_ms: int = Field(ge=0)
    # 이보다 짧은 조각은 앞 구간에 붙인다
    min_segment_ms: int = Field(ge=0)


class ToleranceMs(Contract):
    """`actions.yaml tolerance_ms`: 경계 평가 허용 오차 ms (기준 문서 경계 규칙).

    approach(접근 시작), contact_glove(장갑 접촉), contact_video(영상 접촉), end(행동 끝).
    """

    approach: int
    contact_glove: int
    contact_video: int
    end: int


class VlmPolicy(Contract):
    """`actions.yaml vlm`: 2단 VLM 분류 설정."""

    # 응답 위반(스키마·온톨로지 밖) 시 다시 묻는 횟수
    max_retries: int = Field(ge=0)
    # 구간마다 VLM에 보낼 프레임 수
    frames_per_segment: int = Field(ge=1)
    # 보낼 프레임의 긴 변 상한 픽셀
    max_side_px: int = Field(ge=64)
    # 응답에 신뢰도가 없을 때 쓸 값
    default_confidence: float = Field(ge=0, le=1)
    # VLM 모델 이름 (서버에 보낼 model, 라이선스는 config/models.yaml external)
    model: str
    # 요청 하나의 시간 초과 초
    timeout_s: float = Field(gt=0)
    # 서버 오류·시간 초과 뒤 다시 묻기 전 대기(초). 길이가 재시도 횟수다. 결과에 영향이 없어
    # 정책 해시(모델 버전)에 넣지 않는다
    unavailable_backoff_s: tuple[float, ...] = ()
    # 사내 대역 CIDR (서버 주소가 밖이면 보내지 않는다, F-PAR-82). 결과에 영향이 없어 정책 해시에
    # 넣지 않는다
    internal_hosts: tuple[str, ...] = Field(min_length=1)


class ActionsPolicy(Contract):
    """`config/policies/actions.yaml` 전체."""

    # 정책 형식 버전
    version: int
    # 프롬프트 틀 버전 (코드 PROMPT_VERSION과 같아야 한다). 해시 밖, `stamp`에 붙는다
    prompt_version: int
    # 대상을 모르는 접촉의 target_id (prelabel.yaml contact.unresolved_target_id).
    # 대상 후보에서 뺀다
    unresolved_entity_ids: tuple[str, ...] = ()
    boundaries: BoundaryPolicy
    tolerance_ms: ToleranceMs
    vlm: VlmPolicy

    @model_validator(mode="after")
    def _check_prompt_version(self) -> ActionsPolicy:
        """정책의 프롬프트 틀 버전이 코드와 다르면 ValueError (문구와 버전이 따로 놀지 않게)."""
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"prompt_version {self.prompt_version}이 코드 PROMPT_VERSION "
                f"{PROMPT_VERSION}과 다릅니다 (dlp_actions.policy)"
            )
        return self

    @property
    def digest(self) -> str:
        """정책 해시 (sha256 앞 12자).

        파싱된 값(`model_dump(mode="json")`)을 키 정렬 JSON으로 만들어 해시하므로 YAML
        주석·순서·서식은 영향이 없다. `vlm.unavailable_backoff_s`·`vlm.internal_hosts`는 결과를
        바꾸지 않아 빼고, `prompt_version`은 `stamp`에 따로 붙이므로 뺀다 (해시에 넣으면 첫 틀
        버전에서도 해시가 바뀌어 모든 세션이 다시 돈다).
        """
        data = self.model_dump(
            mode="json",
            exclude={
                "prompt_version": True,
                "vlm": {"unavailable_backoff_s", "internal_hosts"},
            },
        )
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def stamp(self) -> str:
        """모델 버전에 넣는 정책 표시: `digest`, 틀 버전이 `BASE_PROMPT_VERSION`이 아니면
        `+pv<틀 버전>`을 덧붙인다 (`runner`·`describe`의 버전 문자열)."""
        if self.prompt_version == BASE_PROMPT_VERSION:
            return self.digest
        return f"{self.digest}+pv{self.prompt_version}"


def load_policy(root: Path) -> ActionsPolicy:
    """`<root>/config/policies/actions.yaml`을 읽어 검증한다.

    Args:
        root: 저장소 루트 (`dlp_schema.repo_root()`).

    Raises:
        pydantic.ValidationError: 키가 없거나, 모르는 키가 있거나, 값이 범위를 벗어날 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "actions.yaml").read_text("utf-8"))
    return ActionsPolicy.model_validate(data)
