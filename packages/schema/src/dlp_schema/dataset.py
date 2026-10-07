"""데이터셋 버전과 분할."""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier, SemVer


class Split(StrEnum):
    GOLDEN = "golden"
    TRAIN = "train"
    VAL = "val"


class DatasetVersion(Contract):
    version_id: Identifier
    parent_version_id: Identifier | None = None
    ontology_version: SemVer
    created_at: AwareDatetime
    snapshot_uri: str = Field(description="라벨 스냅샷 위치 (lakeFS 커밋 또는 DVC 해시)")
    splits: dict[Identifier, Split] = Field(description="세션 ID → 분할")
    excluded_sessions: tuple[Identifier, ...] = Field(
        default=(), description="사용 중지 등으로 제외된 세션"
    )
    golden_set_version: str | None = None

    @model_validator(mode="after")
    def _check(self) -> DatasetVersion:
        overlap = set(self.splits) & set(self.excluded_sessions)
        if overlap:
            raise ValueError(f"제외된 세션이 분할에 들어 있습니다: {sorted(overlap)}")
        return self
