"""공통 타입. 모든 시간 값은 마스터 타임라인 기준 정수 ms다."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Ms = Annotated[int, Field(ge=0, description="마스터 타임라인 기준 시각(ms)")]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
OntologyId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
SemVer = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]


class Contract(BaseModel):
    """모든 계약 타입의 기반. 알 수 없는 필드를 거부하고 생성 후 변경을 막는다."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)
