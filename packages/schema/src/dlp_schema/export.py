"""내보내기 파일 계약 (구매자에게 주는 형식).

구간 JSON은 세션 하나에 파일 하나다. 라벨마다 검증 상태를 그대로 표시한다.
검수자 ID 같은 내부 정보와
원본 저장소 위치는 넣지 않는다. 작업자·장소 ID는 가명이다.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from dlp_schema.common import Confidence, Contract, Identifier, Ms, SemVer
from dlp_schema.dataset import Split
from dlp_schema.labels import LabelPayload, Source, VerificationState
from dlp_schema.session import Domain, StreamKind

INTERVAL_FORMAT_VERSION = "1.0"


class ExportedStream(Contract):
    stream_id: Identifier
    kind: StreamKind
    video: str | None = Field(default=None, description="내보내기 안의 블러본 상대 경로 (있으면)")


class ExportedLabel(Contract):
    label_id: Identifier
    stream_id: Identifier | None = None
    t_start_ms: Ms
    t_end_ms: Ms
    verification: VerificationState
    source: Source
    model_version: str | None = None
    confidence: Confidence | None = None
    payload: LabelPayload


class IntervalFile(Contract):
    """세션 하나의 시간 구간 라벨.

    행동, 상위 구간, 공백, 손 상태, 객체 상태, 이벤트, 관계, 커버리지, 설명.
    """

    format: Literal["dlp-intervals"] = "dlp-intervals"
    format_version: str = INTERVAL_FORMAT_VERSION
    export_id: Identifier
    dataset_version_id: Identifier
    ontology_version: SemVer
    session_id: Identifier
    split: Split
    domain: Domain
    worker_id: Identifier = Field(description="가명")
    site_id: Identifier = Field(description="가명")
    duration_ms: Ms
    streams: tuple[ExportedStream, ...]
    labels: tuple[ExportedLabel, ...]
    exported_at: AwareDatetime
