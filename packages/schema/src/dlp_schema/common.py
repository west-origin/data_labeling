"""공통 타입: 시각(ms)·신뢰도·식별자·온톨로지 ID·버전 문자열과 모든 계약 타입의 기반 클래스.

역할
    `dlp_schema`의 다른 모듈(라벨·세션·계보·운영 기록 등)과 모든 패키지가 같은 제약을 쓰도록
    `Annotated` 별칭으로 한곳에 모은다 (WP1, ADR 0002).

시간 단위 (ADR 0019)
    모든 시각은 정수 ms다. 다만 기준이 둘이다.
    - 시간 구간 라벨(행동·공백·상위 구간·손 상태·객체 상태·이벤트·관계·커버리지·설명)과 세션 길이는
      마스터 타임라인(= 기준 스트림인 바디캠 시계) ms.
    - 공간 라벨(박스·마스크·키포인트·블러 트랙, 3D 궤적)의 키프레임과 그 라벨의 구간은 그 스트림
      영상의 PTS 시각 ms. 바디캠은 마스터 시계 그 자체라 두 시각이 같다.
    `Ms`의 Field description은 "마스터 타임라인 기준"이라고 쓰여 있지만, 위 규약에 따라 공간
    라벨에서는 스트림 PTS 시각을 뜻한다 (description은
    JSON Schema에 들어가므로 여기서 바꾸지 않는다).

주요 공개 이름
    - `Ms`: 0 이상 정수 ms.
    - `Confidence`: 0~1 실수 (모델 신뢰도, 동기화 신뢰도 등).
    - `Identifier`: 세션·라벨·스트림 등 일반 ID (영숫자로 시작, 최대 128자).
    - `OntologyId`: 온톨로지 사전 키 (소문자 snake_case, 최대 64자).
    - `SemVer`: `X.Y.Z` 형식 버전 (온톨로지 버전 등).
    - `IDENTIFIER_MAX`: `Identifier` 최대 길이 (128). DB의 VARCHAR(128) 열과 맞춘다.
    - `derived_id`: 기존 ID에서 결정적으로 파생 ID를 만든다 (이관·삭제 레코드 ID).
    - `Contract`: 모든 계약 타입의 Pydantic 기반 클래스.
"""

from __future__ import annotations

import hashlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# 정수 ms 시각. 음수는 허용하지 않는다 (세션 시작 = 0).
# 시간 구간 라벨은 마스터 타임라인, 공간 라벨은 스트림 PTS 기준이다 (모듈 docstring 참고, ADR 0019).
Ms = Annotated[int, Field(ge=0, description="마스터 타임라인 기준 시각(ms)")]
# 확률 형태의 신뢰도. 0.0(전혀 믿지 않음) ~ 1.0(확실).
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
# 일반 식별자. 첫 글자는 영숫자, 이후 영숫자와 `_ . : -`, 전체 1~128자.
# `:`를 허용하는 이유: 파생 ID(`<base>:<suffix>`)와 검수 작업 키(`cvat:42`)를 담기 위해서다.
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
# 온톨로지 사전 키 (config/ontology/<버전>/*.yaml의 키). 소문자로 시작하는 snake_case, 1~64자.
OntologyId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
# 의미 있는 버전 문자열 `주.부.수`. 접두사(v)나 접미사(-rc1)는 허용하지 않는다.
SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]

# `Identifier` 최대 길이(자). DB의 ID 열이 대부분 VARCHAR(128)이라 이 값과 같아야 한다.
IDENTIFIER_MAX = 128


def derived_id(base: str, suffix: str) -> str:
    """기존 ID에 접미사를 붙인 파생 ID (`<base>:<suffix>`). Identifier 길이(128자)를 넘으면
    base 앞부분 + base의 해시로 줄인다: `<base 앞부분>:h<sha256 16자>:<suffix>`.

    결정적이라 같은 입력은 같은 ID가 된다 (멱등). 줄인 ID도 해시로 원래 ID와 1:1이다.

    쓰임: 온톨로지 이관 레코드(`<원래>:v<새 버전>`, `migration.migrate_labels`),
    모델 버전 교체 때의 삭제 레코드(`<원래>:retracted`, `episode.retractions`).

    Args:
        base: 원래 ID (보통 라벨 ID, 최대 128자).
        suffix: 붙일 접미사 (예: `"retracted"`, `"v1.1.0"`).

    Returns:
        128자 이하의 파생 ID. 길이 안이면 `f"{base}:{suffix}"` 그대로다.

    Raises:
        ValueError: 접미사가 너무 길어 base를 한 글자도 남길 수 없을 때
            (`":h" + 16자 해시 + ":" + suffix`가 128자 이상, 즉 suffix가 109자 이상).
    """
    full = f"{base}:{suffix}"
    if len(full) <= IDENTIFIER_MAX:
        return full
    # base 전체의 해시를 넣어, 앞부분만 같은 서로 다른 base가 같은 ID로 겹치지 않게 한다.
    digest = hashlib.sha256(base.encode()).hexdigest()[:16]
    tail = f":h{digest}:{suffix}"
    keep = IDENTIFIER_MAX - len(tail)
    if keep < 1:
        raise ValueError(f"접미사가 너무 깁니다: {suffix}")
    return f"{base[:keep]}{tail}"


class Contract(BaseModel):
    """모든 계약 타입의 기반. 알 수 없는 필드를 거부하고 생성 후 변경을 막는다."""

    # extra="forbid": 오타·알 수 없는 필드를 검증 오류로 낸다 (계약 표류 방지).
    # frozen=True: 생성 후 속성 대입을 막는다. 값을 바꾼 사본은 `model_copy(update=...)`로 만든다.
    #   주의: `model_copy(update=...)`는 검증을 다시 하지 않는다. 검증이 필요하면 `model_validate`.
    # use_enum_values=False: 열거형 필드는 문자열이 아니라 열거형 멤버로 보관한다 (`is` 비교 가능).
    # 하위 클래스의 클래스 docstring은 생성 JSON Schema의 description이 되므로 함부로 바꾸지 않는다.
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)
