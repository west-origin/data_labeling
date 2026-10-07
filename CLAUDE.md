# CLAUDE.md

1인칭 돌봄 영상 라벨링 플랫폼. 기준 문서는 `docs/labeling-platform-plan.md`(무엇을, 왜),
구현 계획은 `docs/ai-implementation-plan.md`(작업 패키지 WP0~WP16, 순서, 완료 기준)다.
작업 전에 해당 WP 절을 읽는다.

## 명령어

- `make install` — 의존성 설치 (uv, 잠금 파일 기준)
- `make check` — 린트(ruff) + 포맷 검사 + 타입 검사(pyright strict) + 단위 테스트. PR 전에 반드시 통과
- `make fmt` — 자동 포맷·린트 수정
- `make contracts` — 온톨로지 검증 + JSON Schema가 코드와 일치하는지 검사 (`make check`에 포함)
- `make schemas` — 계약 타입을 바꾼 뒤 `schemas/*.schema.json` 재생성
- `make models` — `config/models.yaml`의 모델 가중치를 `data/models/`에 받고 sha256 확인 (저장소에는 넣지 않음)
- `make export-models` — 공개 ONNX가 없는 모델(메트릭 깊이)을 공식 가중치에서 변환 (일회용 PyTorch 환경)
- `make todo-models` — 실제 모델을 아직 연동하지 못한 곳(`TODO(real-model):` 표시) 목록
- `make licenses` — 정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 표 (위반이 있으면 실패, `make check`에 포함)
- `make fixtures` — 합성 픽스처를 `data/fixtures/`에 생성 (저장소에는 넣지 않음)
- `make db-upgrade` — 개발 DB에 Alembic 마이그레이션 적용 + 온톨로지 v1 등록
- `make up` / `make down` / `make clean` — 개발 서비스 기동 / 중지 / 볼륨까지 삭제
- `make health` — `dlp services check`로 서비스 헬스체크
- `make test-services` — 실행 중인 서비스 대상 통합 테스트 (`@pytest.mark.services`)
- `make cvat-up` / `make cvat-down` / `make cvat-superuser` — CVAT (공식 compose, 고정 버전)

## 구조

- `packages/<이름>/` — uv 워크스페이스 멤버. `src/<모듈>/`과 `tests/`를 둔다.
  새 패키지는 루트 `pyproject.toml`의 `[tool.uv.sources]`와 `dependencies`에 등록한다.
- `packages/schema/` (`dlp_schema`) — 계약 타입, 온톨로지 로더·검증, 이관, 설정 로더, DB 테이블·마이그레이션·저장소.
  테스트용 객체는 `dlp_schema.testing`을 쓴다.
- `packages/fixtures/` (`dlp_fixtures`) — 정답을 아는 합성 데이터: 가짜 세션, 오프셋·드리프트를 아는 동기화 신호와
  QR 슬레이트 MP4, 위치를 아는 블러 대상 VFR 영상, 경계를 아는 행동 시퀀스(손 키포인트·장갑 압력·정답 라벨).
  알고리즘 모듈의 테스트는 이 생성기로 작성하고, 생성기 출력의 정답을 기준으로 판정한다.
- `packages/media/` (`dlp_media`) — 세션 수집: 원본 저장소(불변, 멱등), PTS 인덱스, 프록시, IMU 추출기(GPMF·사이드카),
  장갑 Parquet·HDF5 정규화. `dlp ingest <매니페스트>`로 실행한다.
- `packages/sync/` (`dlp_sync`) — 멀티스트림 동기화: QR 슬레이트, 두드림, 오디오·운동 상호상관, 드리프트 보정.
  `dlp sync run <세션>`, `dlp sync adjust <세션> <스트림> <ms>`. 정책은 `config/policies/sync.yaml`.
- `packages/privacy/` (`dlp_privacy`) — 프라이버시 게이트: 탐지기(YuNet, QR·바코드, OWLv2 오픈 보캐뷸러리, 반사면, 오라클 stub), 추적·보간·유지,
  모자이크 렌더, 검수 우선 구간, 잔여 누락 감사. `dlp privacy detect|approve|render`. 정책은 `config/policies/privacy.yaml`.
- `packages/review/` (`dlp_review`) — 검수 도구 연동: CVAT·Label Studio 변환기(무손실 왕복), 작업 생성, reconcile(검수 결과 →
  라벨 이력), 웹훅 수집, 라벨러 워터마크, 역할 경계. `dlp review create|collect|serve|register-webhooks`.
- `packages/datasets/` (`dlp_datasets`) — 데이터셋 버전(lakeFS 커밋), 작업자·장소 단위 분할(holdout 포함), 골든셋 제안,
  사용 중지 전파, 계보 조회. `dlp dataset golden|build|withdraw`, `dlp lineage <세션>`. 정책은 `config/policies/dataset.yaml`.
- `packages/models/` (`dlp_models`) — 모델 레지스트리(`config/models.yaml`, 해시 확인)와 공용 ONNX 런타임(OWLv2, 메트릭 깊이).
- `packages/prelabel/` (`dlp_prelabel`) — 자동 프리라벨: MediaPipe 손·COCO 객체, RTMPose 전신, OWLv2 도구, 깊이 기반 3D 궤적,
  Oracle stub, 장갑·영상 접촉, 3인칭 착용자 매칭. `dlp prelabel run <세션>`. 정책은 `config/policies/prelabel.yaml`.
- `packages/relations/` (`dlp_relations`) — 관계 도출(YAML 규칙 엔진, `derived_by`=규칙 ID), 도구-표면 접촉(작용부-평면 거리),
  표면 커버리지. `dlp relations run <세션>` (멱등, 규칙 변경 시 차이만 반영). 정책은 `config/policies/relations.yaml`.
- `packages/actions/` (`dlp_actions`) — 행동 구간 2단 구조: 손목 속도·접촉으로 경계 후보 → VLM이 온톨로지 안에서 분류·설명
  (JSON Schema 강제, 재시도 후 미상) → 병합·채우기(공백 0). `dlp actions run <세션> --vlm-url`. CPU·CI는 `OracleVlm` stub.
  정책은 `config/policies/actions.yaml`.
- `config/` — 온톨로지·정책·기본값 YAML. 코드에 값을 하드코딩하지 않는다.
- `schemas/` — 생성된 JSON Schema. 직접 편집하지 않는다 (`make schemas`).
- `services/` — 개발용 docker compose (PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow, lakeFS), CVAT 실행 스크립트.
- `tests/` — 패키지를 가로지르는 통합 테스트.
- `docs/adr/` — 아키텍처 결정 기록.

## 규칙

- 모든 시간 값은 마스터 타임라인 기준 정수 ms. 프레임 번호를 저장하지 않는다.
- 라벨은 덮어쓰지 않는다. 수정은 새 레코드 + `parent_label_id`.
- 모듈 사이 데이터는 `dlp_schema` 타입으로만 주고받는다. 계약 변경은 ADR + Alembic 마이그레이션 + 계약 테스트 + `make schemas`를 함께 한다.
- DB 스키마는 `dlp_schema/db/tables.py`와 새 Alembic 리비전을 함께 바꾼다. 테스트가 둘의 일치를 검사한다.
- 모든 datetime은 시간대 정보가 있어야 한다 (UTC 권장).
- 영상 시각은 PTS 인덱스로만 계산한다. 프레임 번호에 프레임 간격을 곱해 계산하지 않는다.
- 원본 버킷(`dlp-raw`) URI를 일반 라벨러 경로(블러본, 검수 작업, 내보내기)에 노출하지 않는다.
  `dlp_review.roles.check_stage_uris`로 검사하고, 라벨러용 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
- 정책 값(비율, 허용 오차, 임계값)은 `config/`에서 읽는다.
- 골든·학습·검증 사이에 작업자나 장소가 겹치면 안 된다. 분할은 `dlp_datasets.splitter`로만 만든다.
- 새 모델은 공통 `Predictor` 어댑터와 CPU용 stub 구현을 함께 추가한다. CI는 stub으로 돈다.
- 실제 모델을 연동하지 못하고 stub만 둔 곳에는 주석 `TODO(real-model): <무엇이 필요한지>`를 단다 (코드와 정책 YAML 모두).
  연동하면 표시를 지운다.
- 라벨링한 데이터를 판매하므로 모델의 상업 사용 여부를 따진다 (ADR 0010). 새 모델은 `config/models.yaml`에 가중치
  라이선스, 직접 학습 데이터와 그 라이선스, `commercial` 분류를 적는다. 가중치가 비상업(`forbidden`)이면 쓰지 않는다.
- 각 파이프라인 단계는 멱등적인 `dlp <단계>` 하위 명령으로 만든다.
- 테스트는 정답을 아는 합성 픽스처(WP2)로 작성한다. 실제 영상·개인정보를 저장소에 넣지 않는다.
- 코드 주석과 문서는 한국어, 식별자는 영어.
