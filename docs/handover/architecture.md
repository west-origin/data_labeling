# 구조 지도

[인수인계 첫 화면](README.md) · [변경 및 복구 안내](change-guide.md)

## 실행 구조와 코드 읽는 순서

`dlp`는 CLI 진입점이다. 운영자가 단계별로 실행하며, 검수 도구와 DB·S3·lakeFS·MLflow를 연결한다.
Prefect 서비스가 기동된 것만으로 모든 작업이 영속 큐에서 자동 재개되는 구조는 아니다.
먼저 CLI → 단계 실행 함수 → DB 저장소/외부 클라이언트 순서로 읽는다.

| 책임 | 진입 파일 또는 디렉터리 | 설정 |
| --- | --- | --- |
| 명령 등록·실행 인자 | [CLI main](../../packages/cli/src/dlp_cli/main.py), 같은 디렉터리의 `*_cmds.py` | `.env.example` |
| 입력 정규화·원본 수집·PTS | [media](../../packages/media/src/dlp_media/)의 `prepare.py`, `ingest.py`, `pts.py` | `config/policies/media.yaml` |
| 멀티스트림 시각 보정 | [sync pipeline](../../packages/sync/src/dlp_sync/pipeline.py) | `config/policies/sync.yaml` |
| 탐지·승인·렌더 | [privacy](../../packages/privacy/src/dlp_privacy/)의 `runner.py`, `generation.py`, `rendering.py`, `revocation.py`(블러 변경 뒤 배포 매체 회수 요청), `hold.py`(사고 대응 승인 보류, ADR 0042) | `config/policies/privacy.yaml` |
| 자동 라벨 초안 | [prelabel runner](../../packages/prelabel/src/dlp_prelabel/runner.py), `adapters/` | `config/policies/prelabel.yaml`, `config/models.yaml` |
| 관계·접촉·행동 | [relations](../../packages/relations/src/dlp_relations/), [actions](../../packages/actions/src/dlp_actions/) | 각 단계의 `config/policies/*.yaml` |
| 검수 도구·이력 수집 | [review](../../packages/review/src/dlp_review/)의 `tasks.py`, `collect.py`, `reconcile.py`, `clients.py`, `webhook.py`, `revoke.py`(회수 실행기), `orphans.py`(DB 기록 없이 남은 도구 작업·사본 정리) | `config/policies/review.yaml` |
| 배정·QA·품질 | [review ops](../../packages/review/src/dlp_review/ops/) | 위 검수 정책, `config/defaults.yaml` |
| 계약·온톨로지·DB | [schema](../../packages/schema/src/dlp_schema/), `db/` | `config/ontology/v1`, `schemas/`는 생성물 |
| 데이터셋·학습·평가 | [datasets](../../packages/datasets/src/dlp_datasets/), [training](../../packages/training/src/dlp_train/), [evaluation](../../packages/evaluation/src/dlp_eval/) | `dataset.yaml`, `training.yaml`, `evaluation.yaml` |
| 내보내기·게시 | [export runner](../../packages/export/src/dlp_export/runner.py), 같은 디렉터리의 형식별 모듈 | `config/policies/export.yaml` |
| 모델·합성 정답·운영·선별 | [models](../../packages/models/src/dlp_models/), [fixtures](../../packages/fixtures/src/dlp_fixtures/), [ops](../../packages/ops/src/dlp_ops/), [active](../../packages/active/src/dlp_active/) | 모델 레지스트리, 각 정책 |

## 실제 의존 관계

패키지는 역할별로 나뉘지만 단계가 완전히 독립적이지는 않다. 아래는 소스의 패키지 간 import를
기준으로 한 지도다(테스트·외부 라이브러리 제외). `dlp_` 접두사는 표에서 생략했다.

| 패키지 | 직접 사용하는 내부 패키지 |
| --- | --- |
| schema | 없음 |
| fixtures, media, models, datasets, relations, eval | schema |
| sync | media, schema |
| privacy | media, models, schema |
| prelabel | media, models, schema, sync |
| actions, active | media, privacy, schema |
| review | eval, media, privacy, schema, sync |
| train | datasets, eval, media, schema |
| export | datasets, media, privacy, schema |
| ops | privacy, review, schema |
| cli | 나머지 16개 패키지 |

`schema`에는 계약뿐 아니라 DB 구현도 함께 있다. CLI에는 연결 코드 외에 모델 선택·온톨로지 선택도
남아 있다. 단순한 계층 구조로 가정해 파일을 옮기기 전에 [후속 과제](known-issues.md)를 확인한다.

## 상태와 시간의 경계

- 생애주기: `raw_ingested → privacy_approved → prelabeled → human_verified → split_assigned → exported`.
  어디서든 `withdrawn`으로 갈 수 있다. 이력은 추가하며 사용 중지를 되돌리지 않는다.
- 프라이버시 상태: `pending → auto_blurred → approved`. 재탐지·블러 수정은 승인과 렌더의 유효성을
  다시 판단하게 한다. 생애주기가 앞 단계로 돌아가는 것과는 별개다. 사고 대응 보류(`dlp privacy
  hold`, `privacy_holds` 추가 전용 기록)는 승인을 auto_blurred로 되돌리고 해제 기록 전까지 재승인을
  막는다 (ADR 0042).
- 최신 **완료 탐지 세대**와 그 세대를 본 검수 작업이 맞아야 승인할 수 있다. 정책을 옛 값으로
  되돌려도 새 세대다. [ADR 0033](../adr/0033-privacy-generation-and-review-query.md)을 따른다.
- 라벨 저장은 `label_records`의 추가 이력이다. 현재 라벨 조회는 철회와 검수 측정용 사본을 제외한다.
  검수자가 승인·수정·표본 검증한 라벨을 재추론으로 덮어쓰지 않는다.
- 영상 시각은 PTS가 기준이다. 공간 라벨은 스트림 시각, 행동·접촉 등은 마스터 타임라인을 따른다.
  경계 변환은 `Stream.to_master_ms` 등으로 한다. 프레임 번호 × FPS로 대체하지 않는다.
  라벨 시각은 정수 ms지만 동기화 오프셋·드리프트 계산 및 일부 시계열에는 실수가 쓰인다.

## 저장·접근·트랜잭션의 경계

| 저장 위치 | 내용·접근 |
| --- | --- |
| PostgreSQL | 세션·라벨 이력·탐지 세대·검수 작업·배포·내보내기 이력·감사 |
| 원본 버킷 | 영상·프록시·PTS·센서 파생물. CLI는 [raw_store](../../packages/cli/src/dlp_cli/raw_access.py) 감사 경로 사용 |
| 라벨링 버킷 | 승인된 블러본·렌더 기록·워터마크 검수 영상·시계열 CSV |
| 데이터셋 버킷/lakeFS | 버전 스냅샷·내보내기·내부 가명 대응표(구매자 공개 금지) |
| MLflow | 학습 기록·산출물 |
| CVAT/Label Studio | 검수 작업과 도구가 보관하는 사본. DB 롤백으로 함께 지워지지 않음 (그렇게 남은 고아는 `dlp review revoke --orphans`가 이름 규칙·DB 대조로 찾아 지운다, F-REV-15). 블러가 바뀐 라벨링 작업은 `review_tasks`의 회수 요청(outbox)으로 커밋 뒤 지운다 (ADR 0039) |

원본 검수자는 `reviewers.privacy`와 CVAT 사용자 연결로 제한한다. 검수 결과는 작업 담당자의 도구
계정(CVAT job 담당자, Label Studio `label_studio.users` 계정이 만들고 고친 주석)의 것만 받는다.
일반 검수자·행동 VLM·큐레이션·내보내기는 승인된 블러본을 사용한다. 프라이버시 탐지·서버 내 프리라벨 추론은 감사되는 원본을
사용한다. `DLP_ACTOR`는 기록용 자기 신고 값이며 인증 체계는 아니다.

탐지·렌더·검수 생성/수집은 세션 행 잠금을 사용한다. 일부 함수는 영상 처리와 외부 API 호출까지
같은 DB 트랜잭션 안에서 수행하므로 다른 작업을 오래 기다리게 할 수 있다. 원본 접근 감사는
별도 DB 연결로 남긴다. S3·CVAT·lakeFS·MLflow와 DB 사이에는 분산 트랜잭션이 없다.
재시도·잠금 축소·파일 게시 순서를 바꿀 때는 [복구 안내](change-guide.md)를 먼저 읽는다.
