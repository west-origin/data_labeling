"""공통 타입. 모든 시간 값은 마스터 타임라인 기준 정수 ms다."""

from __future__ import annotations

import hashlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Ms = Annotated[int, Field(ge=0, description="마스터 타임라인 기준 시각(ms)")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
OntologyId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]

IDENTIFIER_MAX = 128


def derived_id(base: str, suffix: str) -> str:
    """기존 ID에 접미사를 붙인 파생 ID (`<base>:<suffix>`). Identifier 길이(128자)를 넘으면
    base 앞부분 + base의 해시로 줄인다: `<base 앞부분>:h<sha256 16자>:<suffix>`.

    결정적이라 같은 입력은 같은 ID가 된다 (멱등). 줄인 ID도 해시로 원래 ID와 1:1이다.
    """
    full = f"{base}:{suffix}"
    if len(full) <= IDENTIFIER_MAX:
        return full
    digest = hashlib.sha256(base.encode()).hexdigest()[:16]
    tail = f":h{digest}:{suffix}"
    keep = IDENTIFIER_MAX - len(tail)
    if keep < 1:
        raise ValueError(f"접미사가 너무 깁니다: {suffix}")
    return f"{base[:keep]}{tail}"


class Contract(BaseModel):
    """모든 계약 타입의 기반. 알 수 없는 필드를 거부하고 생성 후 변경을 막는다."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)
