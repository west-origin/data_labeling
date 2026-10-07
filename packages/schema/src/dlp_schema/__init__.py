"""세션·스트림·라벨·에피소드 그래프 계약 타입과 온톨로지.

`dlp_schema`는 플랫폼의 모든 패키지가 공유하는 계약(데이터 형태)을 정의한다 (WP1, ADR 0002).
모듈 사이 데이터는 이 패키지의 타입으로만 주고받는다. 계약을 바꿀 때는 ADR, Alembic 마이그레이션,
계약 테스트, `make schemas`(JSON Schema 재생성)를 함께 한다.

하위 모듈 지도
    - `common`: 공통 타입 별칭(`Ms`, `Identifier` 등)과 `Contract` 기반 클래스, `derived_id`.
    - `session`: 세션·스트림·캘리브레이션, 생애주기 상태와 전이 규칙.
    - `labels`: 라벨 레코드(`LabelRecord`)와 종류별 페이로드 14종, 출처·검증 메타데이터.
    - `episode`: 에피소드 그래프, `current_labels`(운영 라벨 고르기), `retractions`, `version_tag`.
    - `history`: 라벨 이력에서 검수 결과(인정·수정·추가·삭제) 읽기.
    - `ontology`: 온톨로지 사전 타입과 YAML 로더.
    - `validation`: 라벨을 온톨로지 사전에 대조.
    - `migration`: 온톨로지 버전 간 라벨 이관.
    - `dataset`, `lineage`: 데이터셋 버전·분할, 골든셋·학습·모델·내보내기·사용 중지 기록.
    - `review`: 검수 작업·배정 계약.
    - `ops`: 운영·감사 기록 (원본 접근, 검수 시간, 블러 감사, 보관 결정).
    - `export`: 구매자에게 주는 구간 JSON 형식.
    - `config`: `config/defaults.yaml` 로더.
    - `jsonschema`: 계약 → `schemas/*.schema.json` 생성.
    - `predictor`: 자동 모델 공통 인터페이스.
    - `testing`: 다른 패키지 테스트에서도 쓰는 계약 객체 생성 도우미.
    - `db`: SQLAlchemy 테이블, 저장소 함수, Alembic 마이그레이션.

여기서 다시 내보내는 이름(`__all__`)은 가장 자주 쓰는
진입점이다. 나머지는 하위 모듈에서 직접 import한다.
"""

from dlp_schema.config import PlatformConfig, load_config, repo_root
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.episode import Entity, EntityKind, EpisodeGraph, current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.migration import OntologyMigration, migrate_labels
from dlp_schema.ontology import Ontology, find_ontology_dir, load_ontology
from dlp_schema.session import LifecycleState, Session, Stream, StreamKind, can_transition
from dlp_schema.validation import OntologyViolationError, check_label, validate_label

__all__ = [
    "DatasetVersion",
    "Entity",
    "EntityKind",
    "EpisodeGraph",
    "LabelRecord",
    "LifecycleState",
    "Ontology",
    "OntologyMigration",
    "OntologyViolationError",
    "PlatformConfig",
    "Provenance",
    "Session",
    "Source",
    "Split",
    "Stream",
    "StreamKind",
    "Verification",
    "VerificationState",
    "can_transition",
    "check_label",
    "current_labels",
    "find_ontology_dir",
    "load_config",
    "load_ontology",
    "migrate_labels",
    "repo_root",
    "validate_label",
]
