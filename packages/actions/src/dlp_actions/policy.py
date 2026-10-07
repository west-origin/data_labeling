"""행동 구간 정책 (config/policies/actions.yaml, WP10, ADR 0012).

- `ActionsPolicy`: 최상위. `digest`(파싱된 값의 해시, 12자)가 모델 버전에 들어간다
  (`runner.run_actions`: `actions-<digest>+<VLM 버전>+i<입력 해시>`). YAML 주석·서식은 해시에
  영향이 없고, 값이 바뀌면 버전이 바뀌어 다음 실행에서 다시 만든다.
- `BoundaryPolicy`: 1단 경계 후보 (`boundaries.py`)
- `ToleranceMs`: 평가 허용 오차 (테스트·평가 하네스용, 실행 결과에는 영향 없음)
- `VlmPolicy`: 2단 VLM 분류
- `load_policy`: 저장소 루트 → `ActionsPolicy`

`Contract`라 알 수 없는 키는 적재 오류다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract


class BoundaryPolicy(Contract):
    # 속도 이동 평균 창 ms
    """`actions.yaml boundaries`: 손목 속도·접촉 기반 경계 후보.

    속도 단위는 손바닥 길이(손목 → 가운뎃손가락 뿌리, hand21 0 → 9)/초, 시간은 ms.
    """

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
    # 이보다 가까운 후보는 하나로 (접촉 > 정지 > 골짜기 순 우선)
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
    # 응답 위반(스키마·온톨로지 밖) 시 다시 묻는 횟수
    """`actions.yaml vlm`: 2단 VLM 분류 설정."""

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


class ActionsPolicy(Contract):
    # 정책 형식 버전
    """`config/policies/actions.yaml` 전체."""

    version: int
    # 대상을 모르는 접촉의 target_id (prelabel.yaml contact.unresolved_target_id).
    # 대상 후보에서 뺀다
    unresolved_entity_ids: tuple[str, ...] = ()
    boundaries: BoundaryPolicy
    tolerance_ms: ToleranceMs
    vlm: VlmPolicy

    @property
    def digest(self) -> str:
        """정책 해시 (sha256 앞 12자). 모델 버전에 들어간다.

        파싱된 값(`model_dump(mode="json")`)을 키 정렬 JSON으로 만들어 해시하므로 YAML
        주석·순서·서식은 영향이 없다. `vlm.unavailable_backoff_s`는 결과를 바꾸지 않아 뺀다.
        """
        data = self.model_dump(mode="json", exclude={"vlm": {"unavailable_backoff_s"}})
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


def load_policy(root: Path) -> ActionsPolicy:
    """`<root>/config/policies/actions.yaml`을 읽어 검증한다.

    Args:
        root: 저장소 루트 (`dlp_schema.repo_root()`).

    Raises:
        pydantic.ValidationError: 키가 없거나, 모르는 키가 있거나, 값이 범위를 벗어날 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "actions.yaml").read_text("utf-8"))
    return ActionsPolicy.model_validate(data)
