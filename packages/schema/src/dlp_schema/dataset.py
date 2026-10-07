"""데이터셋 버전과 분할.

역할
    데이터셋 버전(라벨 스냅샷 하나 + 세션별 분할 배정)의 계약 (WP7, ADR 0007).
    `dlp dataset build`가 만들고 DB `dataset_versions`·`dataset_split_assignments`에 저장한다
    (`db.repository.insert_dataset_version`). 학습 예제 추출(`dlp_train.extract`)과
    내보내기(`dlp_export`)는 반드시 데이터셋 버전에서 출발한다.

주요 이름
    - `Split`: 세션 분할 (golden / train / val / holdout).
    - `DatasetVersion`: 데이터셋 버전 하나.

주의
    - 분할은 `dlp_datasets.splitter`로만 만든다. 골든·학습·검증 사이에 작업자·장소가 겹치면 안 된다
      (`config/defaults.yaml golden_set.split_unit`).
    - 사용 중지(withdraw)된 세션은 `excluded_sessions`에만
      있고 `splits`에는 없어야 한다 (검증기가 강제).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier, SemVer


# 세션이 데이터셋 버전에서 어디에 쓰이는지. 값은 DB dataset_split_assignments.split에
# 그대로 저장된다.
class Split(StrEnum):
    GOLDEN = "golden"  # 골든셋: 평가 전용, 학습에 절대 쓰지 않는다
    TRAIN = "train"  # 학습
    VAL = "val"  # 검증 (학습 중 조기 종료·선택용)
    # 골든셋과 작업자·장소를 공유하거나, 학습·검증 경계에 걸쳐 어느 쪽에도 넣을 수 없는 세션.
    # 정보 누수를 막기 위해 학습·검증·평가 어디에도 쓰지 않는다.
    HOLDOUT = "holdout"


# 데이터셋 버전 하나. 생성 후 바뀌지 않는다 (새 버전은 parent_version_id로 이전 버전을 가리킨다).
# 필드:
#   version_id: 데이터셋 버전 ID (예: ds-0001).
#   parent_version_id: 이전 버전 ID. 첫 버전이면 None.
#   ontology_version: 이 버전의 라벨이 따르는 온톨로지 버전 (DB FK → ontology_versions).
#   created_at: 만든 시각 (시간대 필수).
#   snapshot_uri: 라벨 스냅샷 위치 (lakeFS 커밋 URI 등).
#   splits: 세션 ID → 분할.
#   excluded_sessions: 사용 중지 등으로 뺀 세션 (splits와 겹치면 검증 오류).
#   golden_set_version: 이 버전과 함께 쓰는 골든셋 버전 (`lineage.GoldenSet.version`). 없으면 None.
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
        """제외된 세션이 분할에 들어 있지 않은지 확인한다.

        Raises:
            ValueError: `splits`와 `excluded_sessions`가 겹칠 때
                (Pydantic이 ValidationError로 감싼다).
        """
        overlap = set(self.splits) & set(self.excluded_sessions)
        if overlap:
            raise ValueError(f"제외된 세션이 분할에 들어 있습니다: {sorted(overlap)}")
        return self
