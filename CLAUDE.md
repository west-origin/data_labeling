# CLAUDE.md

1인칭 돌봄 영상 라벨링 플랫폼. 기준 문서는 `docs/labeling-platform-plan.md`(무엇을, 왜),
구현 계획은 `docs/ai-implementation-plan.md`(작업 패키지 WP0~WP17, 순서, 완료 기준)다.
작업 전에 해당 WP 절을 읽는다. 처음 맡는 개발자는 `docs/handover/README.md`(전체 지도, 기능 상태,
환경 재현, 변경·복구 안내, 남은 과제)부터 본다. 계획 문서의 목표와 현재 구현 상태를 구분한다.
알려진 문제의 단일 목록은 `docs/handover/issues.yaml`, 외주 검수·리팩토링의 인수 기준은 `docs/handover/acceptance.md`다.

## 작업 방식 우선순위 (2026-10-28 소유자 결정)

이 저장소는 pstack 플러그인(`pstack@cursor-plugins`, `.claude/settings.json`에서 켬)을 쓴다. 작업 방식(계획 세우기,
구현·검증 순서, 테스트 작성 방식, 커밋·PR 흐름, 검토 방식, 글쓰기 문체)은 pstack 스킬을 이 문서보다 우선한다.
예외: 아래 "규칙" 절은 데이터·개인정보·라이선스·라벨 이력을 지키는 약속이라 pstack 스킬과 부딪쳐도 이 문서를 따른다
(원본 버킷·블러·사람 승인, 라벨 덮어쓰기 금지, 상업 사용 분류, 분할·내보내기 규칙, 주석·문서 한국어 포함).
`make check` 통과도 그대로 필수다.

## 명령어

- `make install` — 의존성 설치 (uv, 잠금 파일 기준)
- 전체 타입 검사·큐레이션 검사 환경은 `make install-curation`으로 설치한다. `make install` 재실행 시 선택 그룹이 제거된다.
- `make check` — 린트(ruff) + 포맷 검사 + 타입 검사(pyright strict) + 단위 테스트. PR 전에 반드시 통과
- `make fmt` — 자동 포맷·린트 수정
- `make contracts` — 온톨로지 검증 + JSON Schema가 코드와 일치하는지 검사 (`make check`에 포함)
- `make schemas` — 계약 타입을 바꾼 뒤 `schemas/*.schema.json` 재생성
- `make models` — `config/models.yaml`의 모델 가중치를 `data/models/`에 받고 sha256 확인 (저장소에는 넣지 않음)
- `make export-models` — 공개 ONNX가 없는 모델(메트릭 깊이)을 공식 가중치에서 변환 (일회용 PyTorch 환경)
- `make todo-models` — 실제 모델을 아직 연동하지 못한 곳(`TODO(real-model):` 표시) 목록
- `make licenses` — 정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 표 (위반이 있으면 실패, `make check`에 포함)
- `make fixtures` — 합성 픽스처를 `data/fixtures/`에 생성 (저장소에는 넣지 않음)
- `make db-migrate` — Alembic 마이그레이션만 적용 (운영 배포 단계, 온톨로지 등록 없음)
- `make db-upgrade` — 개발 DB에 Alembic 마이그레이션 적용 + 온톨로지 v1 등록 (온톨로지 이름을 덧붙인 뒤에도 다시 돌린다.
  `en` 없는 예전 등록은 읽을 때 `LegacyOntologyContentError`, ADR 0043). 등록된 초안의 물리 사전값이 바뀌면 승인 없이
  종료 코드 3으로 아무것도 쓰지 않는다. 승인은 `uv run dlp ontology register 1.0.0 --accept-physical-priors` (ADR 0053).
  사전값 없는 객체를 덧붙이는 것은 사전값 변경이 아니다 (승인 불필요, ADR 0059)
- `make up` / `make down` / `make clean` — 개발 서비스 기동 / 중지 / 볼륨까지 삭제
- `make health` — `dlp services check`로 서비스 헬스체크
- `make test-services` — 실행 중인 서비스 대상 통합 테스트 (`@pytest.mark.services`, `isolated_env` 표시 테스트는 뺀다)
- `make test-isolated` — 격리된 일회용 환경(LeRobot·PyTorch CPU판)을 받는 테스트 (`@pytest.mark.isolated_env`).
  내보내기 종단 테스트는 DB도 쓰므로 `make up` 뒤에 돌린다. 환경은 `scripts/lerobot-env/uv.lock`에 고정한다
  (바꾸면 그 디렉터리에서 `uv lock`). YOLOX CPU 스모크도 포함한다 (`scripts/yolox-train-env/uv.lock`, ADR 0062)
- `uv run dlp console serve` — 브라우저 운영 콘솔 (기본 127.0.0.1:8090, 토큰 주소 출력, 작업 기록은 `DLP_CONSOLE_DIR`,
  기본 ~/.dlp-console). 새 컴퓨터 설치는 `scripts/install/install.sh` (Mac·Linux·WSL2, ADR 0061)
- `make install-curation` — FiftyOne(데이터 큐레이션, 약 1 GB) 포함 설치. `make install`을 다시 하면 빠진다
- `make coverage` — 기본 테스트의 줄 커버리지 (reports/coverage, 하한 없음)
- `make review-scope` — 전수 검수 범위표 (추적 파일별 위험 등급·관련 미해결 문제 수, reports/review-scope.csv)
- `make cvat-up` / `make cvat-down` / `make cvat-superuser` — CVAT (공식 compose, 고정 버전, UI는 검수 화면 테마를 얹은 이미지)
- `make ui-theme` — 검수 화면 테마(`config/review_ui.yaml`)를 바꾼 뒤 CVAT 테마 색 파일 재생성 (`make contracts`가 일치 검사, ADR 0038)

## 구조

패키지마다 역할·진입 명령·정책 파일·꼭 알아야 할 경계 하나만 적는다. 상세(관련 ADR·문제 ID, 모듈별 설명,
단계별 재실행 판정)는 `docs/handover/architecture.md` "패키지 상세", 용어는 같은 문서 "용어" 절.

- `packages/<이름>/` — uv 워크스페이스 멤버. `src/<모듈>/`과 `tests/`를 둔다.
  새 패키지는 루트 `pyproject.toml`의 `[tool.uv.sources]`와 `dependencies`에 등록한다.
- `packages/cli/` (`dlp_cli`) — `dlp` 진입점. `<단계>_cmds.py`가 정책·저장소·DB를 엮어 각 패키지 실행 함수를 부른다.
  신규 도메인 로직은 각 패키지에 둔다.
- `packages/schema/` (`dlp_schema`) — 계약 타입, 온톨로지 로더·검증·이관, 설정 로더, DB 테이블·마이그레이션·저장소.
  테스트용 객체는 `dlp_schema.testing`. 물려받기·따르기 규칙(`description_follow`, `trajectory_follow`)도 여기 있다.
- `packages/fixtures/` (`dlp_fixtures`) — 정답을 아는 합성 데이터(세션, 동기화 신호, 블러 대상 영상, 행동 시퀀스).
  알고리즘 테스트는 이 생성기의 정답으로 판정한다.
- `packages/media/` (`dlp_media`) — 수집(원본 불변·멱등 저장, PTS 인덱스, 프록시, IMU·장갑 정규화), 카메라 내부
  파라미터·작업자 손 크기 등록. `dlp ingest`, `dlp media prepare`, `dlp calib ...`. 정책 `media.yaml`, `calibration.yaml`.
- `packages/sync/` (`dlp_sync`) — 멀티스트림 동기화(QR 슬레이트, 두드림, 상호상관, 드리프트). `dlp sync run|adjust`.
  정책 `sync.yaml`.
- `packages/privacy/` (`dlp_privacy`) — 프라이버시 게이트: 탐지·추적, 모자이크 렌더, 회수 요청, 승인 보류.
  `dlp privacy detect|approve|render|audit-sample|hold`. 정책 `privacy.yaml`. 탐지는 거르기·힌트로 지우지 않는다.
- `packages/review/` (`dlp_review`) — CVAT·Label Studio 변환(무손실 왕복)·작업 생성·수집·완료 판정, 운영(`ops`: 큐·표본·배정·품질),
  AI 판정 표시·적용. `dlp review create|collect|verify|revoke|plan|assign|quality|hints ...`. 정책 `review.yaml`, `review_ui.yaml`.
- `packages/datasets/` (`dlp_datasets`) — 데이터셋 버전(lakeFS), 작업자·장소 단위 분할(holdout 포함), 골든셋, 사용 중지 전파.
  `dlp dataset golden|build|withdraw`, `dlp lineage`. 정책 `dataset.yaml`.
- `packages/models/` (`dlp_models`) — 모델 레지스트리(`config/models.yaml`, 해시 확인)와 공용 ONNX 런타임·사내 VLM 전송.
- `packages/prelabel/` (`dlp_prelabel`) — 자동 프리라벨(손·객체·전신·도구, 3D·자세·월드 좌표, 접촉, 팔 골격, 마스크).
  `dlp prelabel run <세션>`. 정책 `prelabel.yaml`. 재실행 판정은 단계마다 다르다(상세 문서의 프리라벨 절).
- `packages/retarget/` (`dlp_retarget`) — 고객 로봇 리타기팅(URDF 기구학·IK, 닿지 못한 프레임은 표시).
  `dlp robot check`, `dlp export lerobot --robot`. 정책 `retarget.yaml`. 고객 모델 파일은 저장소 밖 `DLP_ROBOTS_DIR`.
- `packages/relations/` (`dlp_relations`) — 관계 도출(YAML 규칙)·도구-표면 접촉·표면 커버리지. `dlp relations run`.
  정책 `relations.yaml`.
- `packages/actions/` (`dlp_actions`) — 행동 구간(경계 후보 → VLM 분류·설명 → 병합), 설명 재생성, 설명 영어 번역.
  `dlp actions run|describe`, `dlp translate run|pending`. 정책 `actions.yaml`, `translate.yaml`. 설명은 행동 검수를 따른다.
- `packages/aireview/` (`dlp_aireview`) — AI 1차 검수 판정(사내 vLLM, 참고용, 추가 전용 `ai_review_*` 표만).
  `dlp aireview run|status|show|eval ...`. 정책 `aireview.yaml`. 판정 입력은 승인된 블러본만.
- `packages/evaluation/` (`dlp_eval`) — 지표 라이브러리, 골든셋 평가, 배포 게이트, 정답 없는 대리 지표.
  `dlp eval golden`, `dlp ops quality-proxy`. 정책 `evaluation.yaml`.
- `packages/training/` (`dlp_train`) — 재학습 루프(예제 추출·MLflow·레지스트리·게이트 후 배포), 학습 증강, objects용
  실제 학습기 `yolox`. `dlp train run|models|approve`. 정책 `training.yaml`. 기본 학습기는 `oracle-stub`.
- `packages/console/` (`dlp_console`) — 브라우저 운영 콘솔. `dlp` 명령을 작업으로 돌리고 DB는 읽기만 한다.
  `dlp console serve`. 설정 `config/console.yaml`. 도메인 로직을 두지 않는다.
- `packages/active/` (`dlp_active`) — 액티브 러닝 세션 점수, FiftyOne 연동(블러본만, 선택 설치).
  `dlp active rank|fiftyone`. 정책 `active.yaml`.
- `packages/export/` (`dlp_export`) — COCO·구간 JSON·LeRobot 내보내기. 데이터셋 버전에서, 블러본만, 내보내기마다 다른 가명.
  `dlp export coco|intervals|lerobot`. 정책 `export.yaml`.
- `packages/ops/` (`dlp_ops`) — 주간 운영 지표·경고, 원본 접근 감사 리포트, 원본 보관 만료.
  `dlp ops weekly|audit-report|retention ...`. 정책 `ops.yaml`.
- `config/` — 온톨로지·정책·기본값 YAML. 코드에 값을 하드코딩하지 않는다. 온톨로지 항목은 `ko`·`en` 이름 필수.
- `schemas/` — 생성된 JSON Schema. 직접 편집하지 않는다 (`make schemas`).
- `services/` — 개발용 docker compose (PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow, lakeFS), CVAT 실행 스크립트.
- `tests/` — 패키지를 가로지르는 통합 테스트.
- `docs/adr/` — 아키텍처 결정 기록.

## 규칙

- 계약 라벨의 시각은 정수 ms로 저장하고 프레임 번호로 대체하지 않는다. 동기화 오프셋·드리프트 계산과
  일부 시계열은 실수 값을 사용한다. 시간 구간 라벨은 마스터 타임라인 시각, 공간 라벨
  (박스·마스크·키포인트·블러·3D 궤적)의 키프레임은 그 스트림 영상의 PTS 시각이다 (기준 스트림은 같다, ADR 0019).
- 라벨은 덮어쓰지 않는다. 수정은 새 레코드 + `parent_label_id`. 번역(영어 설명)은 수정이 아니라 별도 운영 라벨이라
  parent가 아니라 `DescriptionPayload.source_label_id`로 원문을 가리킨다 (ADR 0043).
- 다른 단계의 입력은 `current_labels()`(운영 라벨)로 고른다. 오류 삽입 레코드와 그 후손, 블라인드·이중 측정
  레코드(`measurement`)는 운영 라벨이 아니며 학습·내보내기에 들어가면 안 된다.
- 모듈 사이 계약 데이터(세션·라벨·검수·데이터셋 등)는 `dlp_schema` 타입으로 주고받는다. 저장소·렌더 게이트 같은
  인프라 API를 다른 패키지에서 쓰는 것은 `docs/handover/architecture.md` 의존 표 안에서만 한다
  (`tests/test_import_boundaries.py`가 표와 실제 import를 대조한다). 다른 모듈의 `_` 이름은 import하지 않는다.
  계약 변경은 ADR + Alembic 마이그레이션 + 계약 테스트 + `make schemas`를 함께 한다.
- DB 스키마는 `dlp_schema/db/tables.py`와 새 Alembic 리비전을 함께 바꾼다. 테스트가 둘의 일치를 검사한다.
- 모든 datetime은 시간대 정보가 있어야 한다 (UTC 권장).
- 계약의 실수 필드는 `dlp_schema.common.FiniteFloat`로 NaN·±Inf를 거부한다 (F-SCH-27).
- 영상 시각은 PTS 인덱스로만 계산한다. 프레임 번호에 프레임 간격을 곱해 계산하지 않는다.
- 원본 버킷(`dlp-raw`) URI를 일반 라벨러 경로(블러본, 검수 작업, 내보내기)에 노출하지 않는다.
  `dlp_review.roles.check_stage_uris`로 검사하고, 라벨러용 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
- 정책 값(비율, 허용 오차, 임계값)은 `config/`에서 읽는다.
- 골든·학습·검증 사이에 작업자나 장소가 겹치면 안 된다. 분할은 `dlp_datasets.splitter`로만 만든다.
- 새 모델은 공통 `Predictor` 어댑터와 CPU용 stub 구현을 함께 추가한다. CI는 stub과 준비된 실제 모델 경로를 검사한다.
- 실제 모델을 연동하지 못하고 stub만 둔 곳에는 주석 `TODO(real-model): <무엇이 필요한지>`를 단다 (코드와 정책 YAML 모두).
  연동하면 표시를 지운다.
- 라벨링한 데이터를 판매하므로 모델의 상업 사용 여부를 따진다 (ADR 0010). 새 모델은 `config/models.yaml`에 가중치
  라이선스, 직접 학습 데이터와 그 라이선스, `commercial` 분류를 적는다. 가중치가 비상업(`forbidden`)이면 쓰지 않는다.
  데이터는 당분간 판매하지 않고 사내 모델 개발에 쓰지만(2026-10-24), 라이선스 분류·forbidden 금지·개인정보·내보내기
  규칙은 그대로 적용한다 (ADR 0058).
- 학습 예제는 데이터셋 버전의 학습·검증 분할에서만 뽑는다 (`dlp_train.extract`). 후보 모델의 골든셋 예측은
  DB에 쓰지 않는다. 배포는 게이트를 통과한 모델만, 프라이버시는 사람 승인 후에만 한다.
- 각 파이프라인 단계는 멱등적인 `dlp <단계>` 하위 명령으로 만든다. 모델 버전에는 그 단계가 쓰는 정책 절의 해시를 넣는다. 모델 출처 라벨 ID에는 `version_tag(모델 버전)`을
  넣고, 일반 단계는 전체 이력(`get_labels`)으로 재실행을 정한다.
  프라이버시는 최신 완료 세대의 활성 버전 집합으로 정하며, 과거 정책 복귀도 새 세대다 (ADR 0033). 버전이 바뀌면 검수 전인 이전 버전만
  `retractions()`로 지운다. 검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 (ADR 0015, 0019).
  좁은 예외 둘. (1) 행동 검수를 물려받기만 한 모델 설명은 물려준 행동이 다음 회차에 의미가 다시 바뀌거나 지워지면
  설명 따르기(`review-follow-1`)만 지울 수 있다 (ADR 0043 3절 규칙 6). (2) 2D 트랙(손 키포인트·객체 박스) 검수를
  물려받기만 한 모델 3D 궤적은 출처(`derived_from`의 2D 트랙, 월드 좌표면 카메라 좌표 궤적·카메라 자세)가 바뀌거나
  지워지면, 또는 같은 출처 2D에서 3D 단계가 다른 판(내부 파라미터·작업자 손 크기·depth 정책·추정기 판·자세 입력 변경)으로
  새로 계산한 검수 전 3D가 같은 (스트림, 개체, 부위, 좌표계)에 생기면(새 3D가 같은 2D 검수를 물려받는다) 궤적
  따르기(`trajectory-follow-1`)만 지울 수 있다 (ADR 0048·0050·0051). "물려받기만 함"은 추정하지 않고 명시
  기록(설명 `description_inheritance`, 3D `trajectory_inheritance`, 추가만, 물려받는 `record_review`와 같은
  트랜잭션)으로만 정한다: 그 라벨의 지금 검수 상태가 마지막 물려받기 기록과 같을 때뿐이다. 저장 검증
  (`label_validation`)은 같은 이력으로 다시 계산한 따르기 계획이 지우는 것만 허용한다. 기록 없는 검수(검수자·시각이
  우연히 같아도)와 기록 뒤 직접 검수, 사람이 쓰거나 고친 라벨은 보호한다.
- 내보내기는 데이터셋 버전에서만 만들고(`dlp_export`), 기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만 넣는다.
  미검수는 명시적 옵션으로만. 블러 라벨·원본 위치·검수자 ID는 어떤 형식에도 넣지 않는다.
- 원본 버킷 저장소는 `dlp_cli.raw_access.raw_store`(감사 저장소)로만 만든다 (DB 없는 로컬 개발은
  `raw_store_offline`, 기록은 로컬 JSON Lines). 모든 원본 접근은 `raw_access_log`에 남고, 운영·감사 기록은 추가만
  한다 (ADR 0020, 0021). 정적 검사 테스트가 우회(`buckets.raw`, `raw_bucket()`)를 막는다.
- 블러(원본 영상) 검수는 `review.yaml reviewers.privacy`의 원본 접근 권한자에게만 배정한다.
- AI 1차 검수 판정은 라벨도 검수 상태도 아니다 (ADR 0040). 판정은 추가 전용 `ai_review_*` 표에만 쓰고 `label_records`·
  `record_review`에 쓰지 않는다. 판정 입력 픽셀은 `dlp_aireview.media.open_blurred`(지금 승인된 블러본)로만 받는다.
  AI 제안은 검수자가 명시적으로 적용할 때만 사람 라벨이 되며, 제안 내용은 DB에서 다시 읽어 검증한다 (`dlp_review.ai_collect`).
  AI의 ok는 사람 검수·표본을 줄이지 않는다. 판정 서버는 사내 서버만(외부 API 금지). Label Studio 설정의 AI 컨트롤은
  모드와 무관하게 늘 둔다.
- 테스트는 정답을 아는 합성 픽스처(WP2)로 작성한다. 실제 영상·개인정보를 저장소에 넣지 않는다.
- 함수 복잡도 상한은 15다 (ruff C901). 상한을 넘는 기존 함수의 `# noqa: C901`은 분해 대상 기준선이며 새로 달지 않는다.
- 새로 찾은 문제는 `docs/handover/issues.yaml`에 넣는다. 확정 결함은 `tests/test_known_defects.py`에
  `xfail(strict=True)` 재현을 두고, 고치면 표시를 지우고 status를 fixed로 바꾼다 (`tests/test_issue_registry.py`).
- 1인칭 바디캠의 MediaPipe 손 판정은 뒤집지 않는다 (`hands.swap_handedness: false`, F-PAR-61).
- 프라이버시 탐지는 시간 거르기·오탐 힌트로 지우거나 바꾸지 않는다. 오탐 의심은 검수 화면 표시로만 쓴다 (ADR 0054·0055).
- 보간해 채운 손 키프레임(`interpolated`)은 관측이 아니다. 3D로 올리지 않는다 (ADR 0054).
- 손 크기로 추정한 내부 파라미터는 `estimated_hand_size`로 밝히고 `calibrated`로 쓰거나 실제 캘리브레이션을 덮지
  않는다. 메트릭 깊이 모델의 화각 사전값을 되돌려 주므로 기본 끈다 (`depth.focal_estimate.enabled: false`, F-PAR-65).
- 학습 증강의 입력은 데이터셋 버전의 학습·검증 분할 블러본만이다. 증강본은 라벨 이력·판매 내보내기에 넣지 않는다 (ADR 0057).
- 1인칭 바디캠에서 착용자 자신의 팔·몸통 박스·자세는 객체·사람 라벨로 만들지 않고 단계 요약에만 센다. 프라이버시
  탐지·블러는 이 억제와 무관하다 (ADR 0058).
- 팔꿈치가 화면 밖이거나 판정 검사를 통과하지 못하면 visibility 0으로 두고 위치를 지어내지 않는다. 팔 손목은 같은 손
  hand21 손목이다 (ADR 0058).
- 모양 마스크는 프라이버시 블러와 무관하다 (블러는 박스 그대로). 착용자 팔·몸통은 마스크 프롬프트로 쓰지 않는다 (ADR 0058·0059).
- 코드 주석과 문서는 한국어, 식별자는 영어.
