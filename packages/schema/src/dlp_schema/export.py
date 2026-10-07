"""내보내기 파일 계약 (구매자에게 주는 형식).

구간 JSON은 세션 하나에 파일 하나다. 라벨마다 검증 상태를 그대로 표시한다.
검수자 ID 같은 내부 정보와 원본 저장소 위치는 넣지 않는다.
작업자·장소·세션·라벨 ID는 내보내기마다 다른 가명이다 (내보내기 ID로 유도한 키의 HMAC,
`config/policies/export.yaml ids.pseudonymize`). 같은 원래 ID는 한 내보내기 안에서 같은 가명이라
파일 사이 참조는 유지되지만, 다른 내보내기와는 이어 붙일 수 없다.

위치
    WP15(내보내기), ADR 0018·0021·0027. `dlp export intervals`가 데이터셋 버전 스냅샷에서
    `IntervalFile`을 만들어 쓴다 (`dlp_export`). COCO·LeRobot
    형식은 각 표준을 따르므로 여기 계약이 없다.

주요 이름
    - `INTERVAL_FORMAT_VERSION`: 구간 JSON 형식 버전 문자열.
    - `ExportedStream`: 내보낸 스트림 하나.
    - `ExportedLabel`: 내보낸 라벨 하나 (블러 라벨 금지).
    - `IntervalFile`: 세션 하나의 구간 JSON 파일 전체.

주의
    - 블러 라벨(`blur_track`)·원본 위치(`dlp-raw` URI)·검수자
      ID는 어떤 형식에도 넣지 않는다 (CLAUDE.md).
      블러 라벨은 `ExportedLabel` 검증기가 막는다. 원본 URI·검수자 ID는 필드 자체가 없다.
    - 시각 규약은 내부와 같다: 시간 구간 라벨은 마스터 타임라인 ms, 공간 라벨은 그 스트림 PTS ms
      (ADR 0019).
    - 기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만 넣는다. 미검수는 명시 옵션으로만.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, field_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, SemVer
from dlp_schema.dataset import Split
from dlp_schema.labels import BlurTrackPayload, LabelPayload, Source, VerificationState
from dlp_schema.session import Domain, StreamKind

# 구간 JSON 형식 버전. 필드를 바꾸면(하위 호환이 깨지면) 올리고 구매자 문서를 함께 갱신한다.
INTERVAL_FORMAT_VERSION = "1.0"


# 내보낸 스트림 하나.
#   stream_id: 스트림 ID (가명 아님. 세션 안에서만 의미가 있는 이름, 예: bodycam).
#   kind: 스트림 종류.
#   video: 내보내기 묶음 안의 블러본 영상 상대 경로. 영상을 함께 내보내지 않으면 None.
class ExportedStream(Contract):
    stream_id: Identifier
    kind: StreamKind
    video: str | None = Field(default=None, description="내보내기 안의 블러본 상대 경로 (있으면)")


# 내보낸 라벨 하나. 내부 `LabelRecord`에서 검수자 ID·부모 ID·측정 표시 등 내부 정보를 뺀 형태다.
#   label_id: 가명 라벨 ID (내보내기마다 다름).
#   stream_id: 공간 라벨이 그려진 스트림. 시간 구간 라벨이면 None.
#   t_start_ms / t_end_ms: 구간 (시간 구간 라벨은 마스터 ms, 공간 라벨은 스트림 PTS ms).
#   verification: 검증 상태 (구매자가 품질 수준을 거를 수 있게 그대로 둔다).
#   source: 출처 (human / model / sensor).
#   model_version: 모델 출처일 때 모델 버전. 아니면 None.
#   confidence: 모델 신뢰도 (0~1). 사람 라벨이면 보통 None.
#   payload: 종류별 내용. blur_track은 검증기가 거부한다.
class ExportedLabel(Contract):
    label_id: Identifier = Field(description="가명 (내보내기마다 다른 HMAC)")
    stream_id: Identifier | None = None
    t_start_ms: Ms
    t_end_ms: Ms
    verification: VerificationState
    source: Source
    model_version: str | None = None
    confidence: Confidence | None = None
    payload: LabelPayload = Field(
        description="블러 라벨(blur_track)은 어떤 내보내기에도 넣지 않는다"
    )

    @field_validator("payload")
    @classmethod
    def _no_blur(cls, value: LabelPayload) -> LabelPayload:
        """블러 라벨을 거부한다 (블러 대상 위치 자체가 개인정보 단서이기 때문).

        Raises:
            ValueError: 페이로드가 `BlurTrackPayload`일 때.
        """
        if isinstance(value, BlurTrackPayload):
            raise ValueError("블러 라벨(blur_track)은 내보낼 수 없습니다")
        return value


class IntervalFile(Contract):
    """세션 하나의 시간 구간 라벨.

    행동, 상위 구간, 공백, 손 상태, 객체 상태, 이벤트, 관계, 커버리지, 설명.
    """

    # 필드 (위 클래스 docstring은 JSON Schema description이라 여기 주석으로 보강한다):
    #   format / format_version: 파일 형식 식별자와 버전 (구매자 측 파서가 확인).
    #   export_id: 내보내기 ID (DB exports.export_id). 가명 키를 유도하는 재료이기도 하다.
    #   dataset_version_id: 출처 데이터셋 버전.
    #   ontology_version: 라벨이 따르는 온톨로지 버전.
    #   session_id / worker_id / site_id: 가명 (내보내기마다 다른 HMAC).
    #   split: 이 세션이 데이터셋 버전에서 속한 분할.
    #   domain: 세션 도메인 (청소·요양돌봄·간호).
    #   duration_ms: 세션 길이 (마스터 타임라인 ms).
    #   streams: 내보낸 스트림 목록.
    #   labels: 내보낸 라벨 목록 (블러 라벨 없음).
    #   exported_at: 내보낸 시각 (시간대 필수).
    format: Literal["dlp-intervals"] = "dlp-intervals"
    format_version: str = INTERVAL_FORMAT_VERSION
    export_id: Identifier
    dataset_version_id: Identifier
    ontology_version: SemVer
    session_id: Identifier = Field(description="가명 (내보내기마다 다른 HMAC)")
    split: Split
    domain: Domain
    worker_id: Identifier = Field(description="가명 (내보내기마다 다른 HMAC)")
    site_id: Identifier = Field(description="가명 (내보내기마다 다른 HMAC)")
    duration_ms: Ms
    streams: tuple[ExportedStream, ...]
    labels: tuple[ExportedLabel, ...]
    exported_at: AwareDatetime
