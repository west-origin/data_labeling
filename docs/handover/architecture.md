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
| 착용자 팔 골격·착용자 몸 억제 (ADR 0058) | [prelabel arm](../../packages/prelabel/src/dlp_prelabel/arm.py)(팔 사슬 고르기·팔꿈치 판정·arm3 트랙), [wearer_suppress](../../packages/prelabel/src/dlp_prelabel/wearer_suppress.py)(객체·도구 박스 억제, 어댑터 공통), [adapters/rtmpose.py](../../packages/prelabel/src/dlp_prelabel/adapters/rtmpose.py)(`body_pass`: 착용자 자세 분리·합성 착용자 박스), 공용 판정 [wearer_body](../../packages/schema/src/dlp_schema/wearer_body.py)(검수 힌트 `dlp_review.object_hints`도 씀), 러너 `wearer_evidence` | `prelabel.yaml wearer_suppression`·`arm`·`depth.arm_points` |
| 바디캠 자세·세션 월드 좌표 3D (ADR 0046) | [prelabel campose](../../packages/prelabel/src/dlp_prelabel/campose.py)(추정기 어댑터·내장 IMU 시각 오도메트리·자세 라벨), `worldframe.py`(월드 좌표 궤적), 러너 `_PoseStage` | `prelabel.yaml camera_pose`·`world_frame` |
| 로봇 리타기팅 (ADR 0047) | [retarget](../../packages/retarget/src/dlp_retarget/)(로봇 등록·URDF 기구학·IK·도달 보고), [export robot](../../packages/export/src/dlp_export/robot.py)(LeRobot 로봇 모드), `scripts/retarget_mink.py`(격리 환경 mink) | `config/robots/<ID>.yaml`(고객 것은 `DLP_ROBOTS_DIR`), `config/policies/retarget.yaml`, 절차 [robot-retargeting.md](robot-retargeting.md) |
| 3D 궤적 검수 물려받기 (ADR 0048·0050·0051) | [trajectory_follow](../../packages/schema/src/dlp_schema/trajectory_follow.py)(순수 규칙: 손·박스 중심 물려받기, 끊긴·대체된 물려받은 3D 삭제 계획, `is_camera_pose`), [reviewer_changes](../../packages/schema/src/dlp_schema/reviewer_changes.py)(물려받기만 한 라벨을 뺀 검수 변화: 주간 지표·원본 보관 기산점·액티브 러닝 수정률), 완료 판정의 지금 2D 3D 확인 `verify.trajectory_gaps`, 프리라벨 다시 돌리기 판정 `runner.stage_version`, [trajectory_repository](../../packages/schema/src/dlp_schema/db/trajectory_repository.py)(적용, 기록 `trajectory_inheritance`, 삭제 레코드), [label_validation](../../packages/schema/src/dlp_schema/label_validation.py)(`_TrajectoryExemption`), 호출: 수집 `collect.py`·완료 판정 `verify.py`(자동 라벨 제외)·프리라벨 `runner.py`, 내보내기 `source.py`(끊긴 3D 제외)·`runner.py`(`trajectory_manifest`)·`robot.py`(meta `camera_pose`) | `review.yaml verify.automatic`·`verify.require_current_3d` (규칙 버전 `trajectory-follow-1`은 코드 상수) |
| 카메라 내부 파라미터·작업자 손 크기 (ADR 0045, 손 크기 출처·무효·세션 기기 연결 0052) | [media calibration](../../packages/media/src/dlp_media/calibration.py)(체커보드·파일), [prelabel palm](../../packages/prelabel/src/dlp_prelabel/palm.py)(손 크기 추정), [등록 저장소](../../packages/schema/src/dlp_schema/db/calibration_repository.py), [CLI](../../packages/cli/src/dlp_cli/calib_cmds.py) | `config/policies/calibration.yaml`, `prelabel.yaml palm_estimate`, 절차 [camera-calibration.md](camera-calibration.md) |
| 관계·접촉·행동 | [relations](../../packages/relations/src/dlp_relations/), [actions](../../packages/actions/src/dlp_actions/) | 각 단계의 `config/policies/*.yaml` |
| 설명 영어 번역 (ADR 0043) | [actions](../../packages/actions/src/dlp_actions/)의 `translate.py`(계획·실행), `translators.py`(stub·사내 서버 어댑터, 공통 전송 `dlp_models.vlm`), `translate_policy.py`, [CLI](../../packages/cli/src/dlp_cli/translate_cmds.py)(`dlp actions run` 끝의 자동 번역 포함) | `config/policies/translate.yaml`, `config/models.yaml` external |
| 고친 행동의 설명 재생성 (ADR 0043 3절) | [actions](../../packages/actions/src/dlp_actions/)의 `describe.py`(대상 계획·설명 전용 VLM 요청·검수 물려받기), [CLI](../../packages/cli/src/dlp_cli/actions_cmds.py)(`dlp actions describe`, `dlp actions run` 끝의 자동 실행) | `config/policies/actions.yaml` (`vlm` 절) |
| AI 1차 검수 (참고용 판정, WP17 진행 중) | [aireview](../../packages/aireview/src/dlp_aireview/)의 `plan.py`, `prompt.py`, `validate.py`, `judge.py`, 실행기 `runner.py`·블러본 경계 `media.py`·엔드포인트 묶음 `backend.py`·진행 상태 `status.py`([실행 CLI](../../packages/cli/src/dlp_cli/aireview_run_cmds.py)), 점검·벤치·추정 `probe.py`·`bench.py`·`estimate.py`([CLI](../../packages/cli/src/dlp_cli/aireview_cmds.py)), 골든 평가 `evaluate.py`([CLI](../../packages/cli/src/dlp_cli/aireview_eval_cmds.py), DB에 쓰지 않는다), 공통 VLM 전송 [models vlm](../../packages/models/src/dlp_models/vlm.py), AI 표 저장소 [aireview_repository](../../packages/schema/src/dlp_schema/db/aireview_repository.py) | `config/policies/aireview.yaml`(`gate` 절), `config/models.yaml` external |
| 검수 도구·이력 수집 | [review](../../packages/review/src/dlp_review/)의 `tasks.py`, `collect.py`, `reconcile.py`, `clients.py`, `webhook.py`, `revoke.py`(회수 실행기), `orphans.py`(DB 기록 없이 남은 도구 작업·사본 정리), `ai_display.py`(AI 1차 검수 판정 표시·노출 기록·대조군·우선순위·표본 제외·required 게이트, WP17.6. 판정은 `dlp_schema.aireview`·`aireview_repository`로 읽고 허용 버전은 CLI가 넘긴다), `ai_collect.py`(검수자가 고른 AI 제안 적용·세션 검증 실패 시 되돌림·결과 기록, WP17.7) | `config/policies/review.yaml`(`ai_review` 절), `config/review_ui.yaml`(`ai` 절) |
| 배정·QA·품질 | [review ops](../../packages/review/src/dlp_review/ops/)(AI 판정 품질 `ai_quality.py`, WP17.8) | 위 검수 정책, `config/defaults.yaml` |
| 계약·온톨로지·DB | [schema](../../packages/schema/src/dlp_schema/), `db/` | `config/ontology/v1`, `schemas/`는 생성물 |
| 데이터셋·학습·평가 | [datasets](../../packages/datasets/src/dlp_datasets/), [training](../../packages/training/src/dlp_train/), [evaluation](../../packages/evaluation/src/dlp_eval/) | `dataset.yaml`, `training.yaml`, `evaluation.yaml` |
| 내보내기·게시 | [export runner](../../packages/export/src/dlp_export/runner.py), 같은 디렉터리의 형식별 모듈 | `config/policies/export.yaml` |
| 운영 콘솔 (ADR 0061) | [console](../../packages/console/src/dlp_console/)의 `app.py`(API·토큰), `jobs.py`(`dlp` 명령 작업 실행·JSON Lines 기록), `steps.py`·`global_steps.py`(단계 활성 규칙·명령), `status.py`(DB 읽기 전용), `upload.py`(매니페스트), `preview.py`(블러본 서명 URL), `training.py`(회차 그래프 자료), `pretrain.py`·`imports.py`(외부 학습 가중치 가져오기), 화면 `static/`, [CLI](../../packages/cli/src/dlp_cli/console_cmds.py) | `config/console.yaml`, `config/review_ui.yaml`(테마), 안내 [operator-guide.md](operator-guide.md) |
| 모델·합성 정답·운영·선별 | [models](../../packages/models/src/dlp_models/), [fixtures](../../packages/fixtures/src/dlp_fixtures/), [ops](../../packages/ops/src/dlp_ops/), [active](../../packages/active/src/dlp_active/) | 모델 레지스트리, 각 정책 |

## 실제 의존 관계

패키지는 역할별로 나뉘지만 단계가 완전히 독립적이지는 않다. 아래는 소스의 패키지 간 import를
기준으로 한 지도다(테스트·외부 라이브러리 제외). `dlp_` 접두사는 표에서 생략했다.

| 패키지 | 직접 사용하는 내부 패키지 |
| --- | --- |
| schema | 없음 |
| fixtures, media, models, datasets, relations, retarget, eval | schema |
| sync | media, schema |
| privacy | media, models, schema |
| prelabel | media, models, schema, sync |
| actions | media, models, privacy, schema |
| active | media, privacy, schema |
| aireview | eval, media, models, privacy, schema |
| review | eval, media, privacy, schema, sync |
| train | datasets, eval, media, models, prelabel, schema |
| export | datasets, media, privacy, retarget, schema |
| ops | privacy, review, schema |
| console | media, models, review, schema |
| cli | 나머지 19개 패키지 |

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
| PostgreSQL | 세션·라벨 이력·탐지 세대·검수 작업·배포·내보내기 이력·감사, AI 1차 검수 기록(`ai_review_*` 5개 표, 추가 전용, 라벨이 아니다, ADR 0040), 기기 내부 파라미터·작업자 손 크기 등록(추가 전용, ADR 0045) |
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

## 용어

처음 보는 사람이 자주 막히는 말이다. 정의는 코드 기준이며, 괄호 안이 기준 코드다.

- **운영 라벨** (`dlp_schema.episode.current_labels`): 라벨 이력에서 지금 유효한 레코드. 고쳐져 다른 레코드의
  parent가 된 레코드, 삭제 레코드와 그것이 가리키는 레코드, 오류 삽입(`seeded_error`)·측정용(`measurement`,
  블라인드·이중) 레코드와 그 후손을 뺀다. 다른 단계·학습·내보내기의 입력은 이것만 쓰고, 다시 돌릴지는 운영
  라벨이 아니라 전체 이력(`get_labels`)으로 정한다.
- **표본 검증** (`VerificationState.SAMPLE_VERIFIED`, `dlp_review.ops.sampling`): 높은 신뢰도 라벨 묶음에서
  일부만 뽑아 사람이 보고, 고치거나 지운 비율이 `review.yaml sampling.max_defect_ratio` 이하이면 묶음의 나머지
  미검수 라벨을 검증된 것으로 두는 상태. 사람 승인·수정처럼 어떤 단계도 지우지 않는다.
- **물려받기**: 검수 도구로 직접 보지 않는 라벨이 출처 라벨의 검수 상태·검수자·시각을 이어받는 것. 설명(ko·en)은
  `segment_id`가 가리키는 행동의 승인·표본 검증을(`dlp_schema.description_follow`, 기록 `description_inheritance`),
  3D 궤적은 `derived_from`의 2D 손·박스 트랙 검수를(`dlp_schema.trajectory_follow`, 기록
  `trajectory_inheritance`) 이어받는다. "물려받기만 함"은 추정하지 않고, 그 라벨의 지금 검수 상태가 마지막
  물려받기 기록과 같을 때만 참이다. 마스크의 `derived_from`은 출처 기록일 뿐 물려받지 않는다.
- **궤적 따르기** (`trajectory-follow-1`, 규칙 `dlp_schema.trajectory_follow`, 적용 `trajectory_repository`):
  확인된 2D에서 만든 새 3D에 물려받기를 쓰고, 물려받기만 한 3D 중 출처 2D가 바뀌거나 지워진 것(끊김)과 같은
  2D에서 다른 판으로 새로 계산한 3D가 생긴 것(대체됨)을 지우는 규칙. 검수된 라벨을 지울 수 있는 좁은 예외
  둘 중 하나다(다른 하나는 설명 따르기 `review-follow-1`). 프리라벨 3D·월드 단계 뒤, 수집, `dlp review verify`에서
  돈다.
- **대조군** (holdout, AI 1차 검수, `dlp_review.ai_display`): AI 판정을 일부러 보이지 않는 검수 작업. 루트
  배정마다 sha256(키|seed) < `review.yaml ai_review.holdout_ratio`로 정하고 노출 기록에만 남긴다. 판정을 본
  검수자의 결과는 판정 쪽으로 기울므로 AI 품질 지표(`dlp review quality --ai`)와 표시 보정(`dlp aireview
  calibrate`)은 대조군만으로 계산한다. 데이터셋 분할의 `holdout`(작업자·장소가 학습·검증에 걸쳐 어디에도 쓰지
  않는 세션, `dlp_datasets.splitter`)과는 이름만 같고 다른 것이다.
- **원본 버킷 / 블러본**: 원본 버킷(`config/defaults.yaml buckets.raw`, 기본 `dlp-raw`)은 블러 전 영상·센서
  파일과 그 파생물(프록시·PTS 인덱스 등)이다. 원본 권한자와 서버 내 추론만 감사 저장소
  (`dlp_cli.raw_access.raw_store`)로 읽는다. 블러본은 승인된 블러 라벨로 렌더한 영상으로 라벨링 버킷
  (`buckets.labeling`)에 있고, 일반 검수자·행동 VLM·AI 1차 검수·큐레이션·학습·내보내기는 이것만 쓴다.
  "지금 승인된 블러본"은 렌더 해시가 지금 승인과 맞는 것이다.
- **`+r<n>`** (`dlp_prelabel.runner.stage_version`): 3D·자세·월드·마스크 단계 버전 끝의 재실행 세대 표시.
  입력이 되돌아와 이전에 낸 적 있는 버전을 다시 만들 때 새 라벨 ID가 이전 것과 겹치지 않게 붙인다. n은 표시를
  뗀 같은 버전으로 이미 쓴(돈) 세대 수다. 다른 단계에는 없다(아래 프리라벨 절의 재실행 판정).

## 패키지 상세

[CLAUDE.md](../../CLAUDE.md) "구조" 절의 요약 뒤에 있는 상세다. 패키지마다 역할·명령·정책 파일·경계와 관련
ADR·문제 ID를 적는다. 패키지를 바꾸면 이 절과 CLAUDE.md 요약을 함께 고친다.

### `packages/<이름>/`

uv 워크스페이스 멤버. `src/<모듈>/`과 `tests/`를 둔다.
새 패키지는 루트 `pyproject.toml`의 `[tool.uv.sources]`와 `dependencies`에 등록한다.

### `packages/cli/` (`dlp_cli`)

`dlp` 명령줄 진입점. 하위 명령마다 `<단계>_cmds.py`가 정책·저장소·DB를 엮어 각 패키지의
실행 함수를 부른다. 모델 선택·온톨로지 선택 로직도 일부 남아 있다. 신규 도메인 로직은 각 패키지에 둔다.

### `packages/schema/` (`dlp_schema`)

계약 타입, 온톨로지 로더·검증, 이관, 설정 로더, DB 테이블·마이그레이션·저장소.
테스트용 객체는 `dlp_schema.testing`을 쓴다. 온톨로지 이관은 라벨 참조(`labels.LABEL_REF_FIELDS`)를 같은 배치의 새 ID로
바꾼다 (F-SCH-29). 추가 전용 기록이 있으면 오래된 리비전(0005·0009·0011·0012)의 downgrade도 거부한다.
`record_review`는 라벨의 세션 행을 잠근다(여러 세션이면 호출자가 세션 ID 순서로 먼저 잠근다, F-SCH-31).

### `packages/fixtures/` (`dlp_fixtures`)

정답을 아는 합성 데이터: 가짜 세션, 오프셋·드리프트를 아는 동기화 신호와
QR 슬레이트 MP4, 위치를 아는 블러 대상 VFR 영상, 경계를 아는 행동 시퀀스(손 키포인트·장갑 압력·정답 라벨).
알고리즘 모듈의 테스트는 이 생성기로 작성하고, 생성기 출력의 정답을 기준으로 판정한다.

### `packages/media/` (`dlp_media`)

세션 수집: 원본 저장소(불변, 멱등), PTS 인덱스, 프록시, IMU 추출기(GPMF·사이드카),
장갑 Parquet·HDF5 정규화. `dlp ingest <매니페스트>`로 실행한다. 수집은 업로드 전에 스트림끼리 저장소 키가 겹치는지 보고(`planned_keys`),
원본 불변 업로드는 키가 없을 때만 원자적으로 쓴다(LocalStore 배타 링크, S3 `If-None-Match: *`, F-PMS-40).
`dlp calib intrinsics`는 표시 회전 태그 영상을 거부한다 (`dlp media prepare` 안내). PTS 인덱스는 컨테이너가 버림 표시한
패킷(편집 목록 밖)을 넣지 않고, 그래도 첫 PTS가 음수면 수집이 업로드 전에 거부한다 (F-PMS-42).
카메라 내부 파라미터 등록(`dlp_media.calibration`, ADR 0045): 세션 값 → 기기(`calibration.camera_id`) 등록값
(`camera_calibrations`, 추가만) → 화각 근사 순으로 고른다. 작업자 손 크기 사전값은 `worker_hand_sizes`(추가만).
수집 뒤 세션 → 기기 연결은 `session_camera_assignments`(추가만, `dlp calib assign`, 매니페스트 camera_id가 앞선다).
작업자 손 크기는 유효한 등록 중 최신 → 정책 순이다. 무효 기록 `worker_hand_size_invalidations`(`dlp calib palm-revoke`,
동의 철회·삭제 요구)와, 추정 출처 세션(`worker_hand_size_sources`)이 사용 중지된 추정은 결정에서 빠진다 (ADR 0052).
`palm-estimate`는 잰 값을 `--override-measured` 없이 덮지 않고, 화면비가 다른 캘리브레이션 세션은 이유를 알리고 건너뛴다 (F-PAR-77). 캘리브레이션은 보드 기울기·타당 범위 검사
(`calibration.yaml`)를 통과해야 등록한다. 화면비가 다른 캘리브레이션은 쓰지 않고 화각 근사로 떨어진다.
`dlp calib intrinsics|import|assign|show|palm|palm-estimate|palm-revoke`. 정책은 `config/policies/calibration.yaml`.
촬영 점검 경고 `dlp_media.capture_check`(prepare·ingest, 기준 `media.yaml capture`, 가이드 `docs/handover/capture-guide.md`),
TTA·학습 증강 공용 기하 변환 `dlp_media.geometry`, COCO RLE `dlp_media.rle`. 방향 태그가 틀린 영상은
`dlp media prepare --display-rotation <반시계 각도>`.

### `packages/sync/` (`dlp_sync`)

멀티스트림 동기화: QR 슬레이트, 두드림, 오디오·운동 상호상관, 드리프트 보정.
`dlp sync run <세션>`, `dlp sync adjust <세션> <스트림> <ms>`. 정책은 `config/policies/sync.yaml`.

### `packages/privacy/` (`dlp_privacy`)

프라이버시 게이트: 탐지기(YuNet, QR·바코드, OWLv2 오픈 보캐뷸러리, 반사면, 오라클 stub), 추적·보간·유지,
모자이크 렌더, 검수 우선 구간, 잔여 누락 감사, 블러 변경 뒤 배포 매체 회수 요청(`dlp_privacy.revocation`, ADR 0039),
사고 대응 승인 보류(`dlp_privacy.hold`, ADR 0042: DB 단계 커밋 뒤에 저장소 클라이언트를 만들고, 렌더 무효화는
별도 트랜잭션의 세션 잠금 아래 그 보류가 아직 유효할 때만 쓴다).
`dlp privacy detect|approve|render|audit-sample|hold`. 정책은 `config/policies/privacy.yaml`.
블러 오탐 의심 힌트 `dlp_privacy.hints` (ADR 0055).
OWLv2 간격 캐시 프레임은 관측이 아니다(`Detection.cached`). 트래커는 실측 사이를 외접 박스·유지로 덮고 검수 우선
구간(track_gap)에 넣는다 (iou-tracker-2, ADR 0009 개정, F-PMS-45).

### `packages/review/` (`dlp_review`)

검수 도구 연동: CVAT·Label Studio 변환기(무손실 왕복), 작업 생성, reconcile(검수 결과 →
라벨 이력), 웹훅 수집, 라벨러 워터마크, 역할 경계, 배포 매체 회수 실행기(`dlp_review.revoke`).
`dlp review create|collect|verify|serve|register-webhooks|revoke|refresh-urls` (verify: 검수가 끝난 세션을 human_verified로. 설명·3D 궤적 따르기를 먼저 적용하고, `review.yaml verify.automatic`(기본 `camera_pose`·`relation`·`coverage`, 모델 출처만, ADR 0011 개정)의 자동 라벨은 검수 전이어도 완료를 막지 않는다(검수 상태는 그대로), ADR 0050, 출처 2D가 바뀌어 지운 3D를 지금 2D로 다시 만들기 전에는 완료를 막고 `dlp prelabel run`을 안내한다(`verify.require_current_3d`, ADR 0051),
revoke: 블러가 바뀐 뒤 이미 배포한 라벨링 매체 회수, 대기가 남으면 종료 코드 4, ADR 0039. `revoke --orphans`: DB 기록 없이
남은 도구 작업·사본 정리(기본 보고만, `--apply`일 때 삭제). refresh-urls: 열린 Label Studio 작업의 서명 URL 다시 서명).
Label Studio 결과는 담당자 계정(`review.yaml label_studio.users`)이 낸 주석만 받는다.
운영 로직(`dlp_review.ops`): 우선순위 큐, 표본 검증, 블라인드·오류 삽입·이중·QA 배정, 품질 측정.
`dlp review plan|assign|queue|qa|quality`. 정책은 `config/policies/review.yaml`.
검수 화면 테마(`dlp_review.ui_theme`, `config/review_ui.yaml`, `services/review-theme/`): 색·로고·단축키·다크/화이트만
도구 밖에서 덧씌우고 라벨 형식은 바꾸지 않는다 (ADR 0038).
AI 1차 검수 표시·수집(ADR 0040): `dlp_review.ai_display`(판정 → 검수 화면 표시·우선순위·표본 제외, 노출 기록·대조군),
`dlp_review.ai_collect`(검수자가 고른 제안 적용·세션 검증·결과 기록 `ai_review_outcomes`),
`dlp_review.ops.ai_quality`(대조군만으로, 표본 가중 지적 정밀도·재현율·ok 놓침률 95% 상한·ECE·κ). `dlp review quality --ai`.
판정 표시 보정 `dlp aireview calibrate`(대조군만, `dlp_review.ops.ai_calibration`, ADR 0062): 버전 붙은 산출물을
`review.yaml ai_review.calibration.artifact`(경로·sha256)로 고정해야 쓰인다. 의심·틀림 표시만 거르고 ok와 검수 대기열은
바꾸지 않는다.
설정은 `review.yaml ai_review`(코드 기본 off).
판정단 불일치 `ai_review.disagreement`(기본 끔, ADR 0063): 허용 버전 둘 이상이 같은 라벨을 ok와 ok 아님으로 가르면
`ai_disagreement` 사유 구간·표본 제외(`dlp_schema.aireview.valid_verdicts_by_version`·`split_verdict_ids`). 줄이는 쪽으로는 쓰지 않는다.
켜면 보인 판정의 메모(LS)·이슈(CVAT) 끝에 `AI 판정단 의견이 갈림: <모델> 맞음 / <모델> 틀림`을 덧붙인다(표시 프로필,
`dlp_review.ai_display.split_text`, 새 결과·이슈가 아니라 노출 기록·이슈 상한·대조군 규칙 그대로).
오탐 의심 힌트 `dlp review hints <세션> [--tta] [--ai]`(블러: 원본 버킷 hints 파일 → 블러 검수 CVAT 이슈, 객체
`dlp_review.object_hints`: 라벨링 CVAT 이슈, 잘라 낸 박스 AI 확인 `dlp_aireview.crops`). 라벨·블러는 바꾸지 않는다
(ADR 0055·0056).
손이 있는 골격(hand21·arm3)의 entity_id는 수집 때 손에서 정하고(`<손>_hand`·`<손>_arm`), 모든 점이 visibility 0인
키포인트 키프레임은 CVAT outside로 오간다 (F-REV-32·33).
마스크(`mask_track`)는 CVAT 모양 모드 마스크로 무손실 왕복한다 (`dlp_review.cvat.to_cvat_shapes`, 원본 해상도 작업만,
블러 검수에서는 받지 않음, ADR 0059).

### `packages/datasets/` (`dlp_datasets`)

데이터셋 버전(lakeFS 커밋), 작업자·장소 단위 분할(holdout 포함), 골든셋 제안,
사용 중지 전파, 계보 조회. `dlp dataset golden|build|withdraw`, `dlp lineage <세션>`. 정책은 `config/policies/dataset.yaml`.
빌드는 지정한 골든셋뿐 아니라 등록된 모든 골든셋 세션을 학습·검증에서 뺀다(지정하지 않은 골든셋은 holdout, F-ETA-19).

### `packages/models/` (`dlp_models`)

모델 레지스트리(`config/models.yaml`, 해시 확인)와 공용 ONNX 런타임(OWLv2, 메트릭 깊이).

### `packages/prelabel/` (`dlp_prelabel`)

자동 프리라벨: MediaPipe 손·COCO 객체, RTMPose 전신, OWLv2 도구, 깊이 기반 3D 궤적,
Oracle stub, 장갑·영상 접촉, 3인칭 착용자 매칭. `dlp prelabel run <세션>`. 정책은 `config/policies/prelabel.yaml`.
손 키프레임에는 MediaPipe 손 중심 3D 점(`KeypointFrame.world`, 선택, 검수 도구로 보내지 않는다. 사람이 고친 트랙에는
2D가 그대로인 키프레임에만 parent의 world가 남는다, ADR 0044 1절 개정)을 싣고, 손목·다섯 손가락 끝 3D(카메라 좌표)는
그 손 모양 + 2D 투영 + 손바닥 깊이·손 크기 사전값으로 맞춘다 (`dlp_prelabel.handfit`, ADR 0044. 손 모델 자세는 손 중심
회전으로 다듬고 문턱은 해상도와 무관한 상대 재투영 오차다. world가 없거나 맞춤이 나쁘면 손바닥 관절만 픽셀 깊이로
올리고 손가락 끝은 내지 않는다, `depth.fingertip_fallback`). 표본마다 방식·신뢰도, 궤적마다 내부
파라미터 출처를 남기고 작업자 손 크기(`dlp_prelabel.palm`)를 우선한다 (ADR 0045).
바디캠 6자유도 자세(IMU 회전 + 깊이 기반 이동, `dlp_prelabel.campose`)와 월드 좌표 3D 궤적(`dlp_prelabel.worldframe`,
ADR 0046). 3D 궤적(손 관절·박스 중심, 카메라·월드 좌표)은 `derived_from`의 2D 트랙 검수를 물려받고
(`dlp_schema.trajectory_follow`, 기록 `trajectory_inheritance`, ADR 0048·0050), 물려받기만 한 3D의 출처가 바뀌거나
같은 2D에서 입력이 바뀌어 새로 계산한 3D가 생기면 궤적 따르기가 지운다(새 3D가 같은 검수를 물려받는다, ADR 0051).
궤적 따르기는 3D·월드 단계 뒤와 수집·`dlp review verify`에서 적용한다(세션 잠금, 멱등). 카메라 자세
(entity `camera`)는 사람 검수 대상이 아닌 자동 라벨이다.

다시 돌릴지(재실행 판정)는 단계마다 다르다.

- 3D(`3d`)·자세(`campose`)·월드(`world3d`)·마스크(`masks`) 단계: 그 단계가 마지막으로 낸 버전으로 정한다
  (`dlp_prelabel.runner.stage_version`). 가장 늦은 기록에 지금 버전이 있으면 건너뛰고, 입력이 A → B → A로
  되돌아오면 A를 다시 만든다(라벨 ID가 겹치지 않게 버전 끝에 `+r<n>`, ADR 0051 3절, F-PAR-54).
- 1단계 예측기 어댑터(`runner._predictor_done`)·접촉 단계: 이력이나 실행 기록에 같은 버전이 어디든 있으면
  건너뛴다. 되돌린 버전은 다시 만들지 않는다(다시 만들려면 버전을 바꾼다). 사람이 검수하는 결과라 되돌림이
  검수에서 보이기 때문이다 (F-PAR-54).
- 행동 단계(`dlp_actions.runner`): 그 손에 같은 버전 라벨이 이력에 있으면 건너뛴다(실행 기록은 쓰지 않는다).
  되돌린 버전도 다시 만들지 않는다.
- 착용자 매칭: 버전이 아니라 살아 있는 착용자 레코드가 있으면 다시 하지 않는다.

프리라벨 단계는 결과 0개인 실행도 추가 전용 실행 기록 `prelabel_stage_runs`(Alembic 0025)를 남기고 다시 돌릴지는
라벨 이력과 함께 본다 (F-PAR-75, ADR 0015 개정).

시간 일관성 `dlp_prelabel.temporal`(박스 트랙 거르기·손 One Euro 평활·짧은 끊김 보간, ADR 0054), 손 3D 물리 제약
`dlp_prelabel.constraints3d`(표본 `flags`), 장면 분류 `dlp_prelabel.scene`과 `dlp prelabel run <세션> --scene <장면>`
(도구 질의를 kitchen·bathroom·bedroom·living·unknown으로 고름, ADR 0055), 손 크기 초점 거리 추정 `dlp_prelabel.fov`
(출처 `estimated_hand_size`, 기본 끔, ADR 0056).
3D·월드 라벨 ID는 128자를 넘으면 단계 접두사를 지키고 해시로 줄인다 (`dlp_prelabel.common.fit_label_id`, F-PAR-79).
착용자 팔 골격 arm3(손목·팔꿈치·어깨, 손별 entity `<손>_arm`, 손목은 hand21 손목 기준, `dlp_prelabel.arm`, 전신 단계가
착용자 자세·합성 착용자 박스 자세에서 고름)와 팔꿈치 3D(`<손>_arm`, 아래팔 길이 표시 `forearm_length`), 바디캠 착용자
몸 억제(`dlp_prelabel.wearer_suppress`, 판정 규칙 `dlp_schema.wearer_body`, 객체·도구·전신 단계, 정책
`prelabel.yaml wearer_suppression`·`arm`, ADR 0058).
객체·손 모양 마스크 `dlp_prelabel.masks`(SAM 2.1 tiny ONNX `dlp_models.sam2`, review 등급, 이미지 모드
`masks.frame_stride_ms` 간격, 프롬프트 = 현재 박스 트랙마다 하나(박스 번호 재사용 무관, 검수 마스크 보호는 `derived_from`·조상, F-PAR-74)·hand21 손 트랙, 착용자 몸 박스·화면 아래 사람 박스 제외, 손은
class `hand`·entity `<손>_hand`, `derived_from`은 출처 기록이고 검수를 물려받지 않음, 키프레임 `score`, 보간 없음,
`dlp prelabel run --skip masks`, CPU stub `BoxEllipseSegmenter`, ADR 0059).

### `packages/retarget/` (`dlp_retarget`)

고객 로봇 리타기팅(ADR 0047): 로봇 등록(`config/robots/<ID>.yaml`, 모델 sha256,
고객 모델 파일은 저장소 밖 `DLP_ROBOTS_DIR`), URDF 기구학, 손 → 끝단 목표·평행 그리퍼, 관절·속도 한계 IK(내장, mink는
격리 환경 `scripts/retarget-env`), 닿지 못한 프레임은 자르지 않고 표시. `orientation_mode`는 관절 수가 아니라 구조로
고른다(접근 축 둘레 롤 관절이 있으면 full, SO-101은 full). 재시작 프레임·그리퍼에도 속도 한계, 손 목표가 없는 에피소드는
drop, 설정·URDF 검증(축 단위·직교, 그리퍼, mimic, robot_id 경로) (ADR 0053). `dlp robot check <ID>`,
`dlp export lerobot --robot <ID>`. 정책은 `config/policies/retarget.yaml`.

### `packages/relations/` (`dlp_relations`)

관계 도출(YAML 규칙 엔진, `derived_by`=규칙 ID), 도구-표면 접촉(작용부-평면 거리),
표면 커버리지. `dlp relations run <세션>` (멱등, 규칙 변경 시 차이만 반영). 정책은 `config/policies/relations.yaml`.

### `packages/actions/` (`dlp_actions`)

행동 구간 2단 구조: 손목 속도·접촉으로 경계 후보 → VLM이 온톨로지 안에서 분류·설명
(JSON Schema 강제, 재시도 후 미상) → 병합·채우기(공백 0). `dlp actions run <세션> --vlm-url`. 사람이 의미를 고친 행동의 설명은
`dlp actions describe <세션>|--pending`이 VLM에 다시 물어 만든다(검수를 물려받음, ADR 0043). CPU·CI는 `OracleVlm` stub.
정책은 `config/policies/actions.yaml`.
설명 영어 번역(ADR 0043, `dlp_actions.translate`): 현재 한국어 설명마다 영어 형제 레코드(`language=en`,
`source_label_id`=원문 ID, parent 아님)를 사내 서버(`dlp_models.vlm`, 기본 Qwen3-VL-8B)로 만든다. 온톨로지 ko→en
이름을 용어집으로 넣는다. 멱등, 원문이 바뀌면 검수 전 번역만 교체, 서버 장애는 대기로 남긴다(예외 없음).
설명(ko·en)은 검수 도구 경로가 없어 `segment_id`가 가리키는 행동·상위 구간의 검수를 따른다
(`dlp_schema.description_follow`, ADR 0043 3절): 승인·표본 검증은 물려받고, 의미를 고쳤거나 지운 행동의
검수 전 설명은 지운다. 물려받은 뒤 행동이 다시 바뀌면 물려받은 설명도 지운다(규칙 6, 물려받음은
`description_inheritance` 기록으로 판정, Alembic 0021). 수집·`dlp review verify`·번역·재생성 단계가 적용한다.
재생성 요청은 손마다 고친 의미 필드 전부(동사·대상·부위·도구·방향·전후 상태)를 묶고 손 단위로 검증한다.
재생성·번역은 쓰기 직전 세션 잠금 아래 사용 중지·승인 보류·(프레임을 보냈으면) 렌더 해시를 다시 보고, 막히면
아무것도 쓰지 않는다 (`dlp_actions.write_check`).
`dlp translate run <세션>|--pending`, `dlp translate pending`. `dlp actions run`이 커밋 뒤 자동으로 부른다
(실패해도 행동 결과·종료 코드 불변). CPU·CI는 `GlossaryStubTranslator`. 정책은 `config/policies/translate.yaml`.

### `packages/aireview/` (`dlp_aireview`)

AI 1차 검수 판정(ADR 0040): 사내 vLLM(OpenAI 호환, 기본 Qwen3-VL-8B)이 블러본
프레임과 라벨 맥락으로 라벨마다 ok·suspect·wrong·unsure와 이유·제안을 낸다. 참고용 판정만 추가 전용 `ai_review_*` 표에
쓴다(멱등·재개, 세션 잠금). CPU·CI는 `OracleJudge`·`FakeOpenAIServer`. 골든셋 평가(`evaluate`: 교란·이력, DB에 쓰지 않음,
게이트 실패는 기본 경고만). `dlp aireview run|status|show|waive|estimate|bench|probe|eval`.
응답 파싱은 pydantic strict다(번호는 JSON 정수만, 확신도는 JSON 수만, F-AIR-36).
판정단(ADR 0063): `backend.panel`의 모델을 `dlp aireview run|status|show|estimate|calibrate|bench|probe|eval --model <이름>`으로 골라
모델마다 따로 판정한다(병렬, 서로의 판정을 보지 않음, 버전 `+vlm-<모델>`, 판정단 목록은 digest 밖, 판정단 모델은
로드 때 models.yaml 라이선스 검사). 기본 판정단은 비어 있다.
정책은 `config/policies/aireview.yaml`.

### `packages/evaluation/` (`dlp_eval`)

지표 라이브러리(mAP·HOTA·IDF1·PCK·접촉·구간 F1·temporal mAP·ECE 등, 참조 구현과 일치 테스트),
골든셋 평가 하네스, 하위 집단 리포트, 배포 게이트, 정답 없는 대리 지표 `dlp_eval.proxy`(`dlp ops quality-proxy <세션>`,
기준 `evaluation.yaml proxy`). `dlp eval golden <골든셋> --model <과제>=<버전>`. 정책은 `config/policies/evaluation.yaml`.

### `packages/training/` (`dlp_train`)

재학습 루프: 데이터셋 버전에서 과제별 학습 예제 추출(자동 원본과 수정본 차이),
학습 작업 템플릿, MLflow 기록, 모델 레지스트리(DB `model_versions`), 골든셋 평가·게이트 후 배포.
`dlp train run|models|approve`. CI·CPU는 `oracle-stub` 학습기. 정책은 `config/policies/training.yaml`.
학습 증강 `dlp_train.augment` (ADR 0057).
replaces 과제의 첫 배포는 골든셋에 예측을 낸 기본 어댑터마다 비교 버전이 있어야 한다(어댑터는 라벨 ID 접두사로 판별,
F-ETA-20). 비교 예측 확인과 학습 예제 추출은 같은 과제 대상 판정 `dlp_eval.runner.is_task_target`을 쓴다 (F-ETA-21).
학습 모델 로더는 출발 가중치 계보의 지금 분류를 다시 본다 (F-ETA-22).
실제 학습기 `yolox`(objects, `dlp_train.yolox`·`yolox_data`, ADR 0062): 지금 승인된 블러본 키프레임 프레임 → COCO(증강
포함) → 격리 환경 `scripts/yolox-train-env`(uv.lock, `--extra cpu|cu128`)에서 학습 → `yolox-model.zip`(ONNX
decode_in_inference false + manifest). 로더는 `YoloxTools`에 끼운다(`trained-objects`). GPU가 없으면
`--param allow_cpu=true` 없이는 거절한다. 출발 가중치는 `training.yaml yolox.base_model`(models.yaml .pth)이고 그 분류를
물려받는다. 기본 학습기는 아직 `oracle-stub`이다.

### `packages/console/` (`dlp_console`)

운영 콘솔 (ADR 0061): 기존 `dlp` 명령을 작업으로 돌리고(셸 없음, 작업 디렉터리
JSON Lines 추가만) DB는 읽기만 한다. 응답·로그에 원본 위치를 싣지 않고, 미리보기는 승인·렌더된 블러본의 라벨러 서명
URL만 쓴다. Git이 관리하는 설정 파일은 고치지 않고 패치 미리보기만 낸다(외부 학습 가중치 가져오기: 구조 이름은 식별자 모양만, 가중치 경로는 models_dir 안인지
confirm에서 다시 본다). 자식 `dlp`는 새 프로세스 그룹으로 돌고(pid·pgid 기록) 콘솔 종료 때 멈추며, 재시작 때 살아 있는
그룹은 잠금을 유지한다 (`console.yaml jobs`). Host는 묶은 주소·localhost만 받고(모든 주소에 묶으면 IP 리터럴만 더,
시험은 `create_app(extra_hosts=…)`), 단계 API 본문 오류는 400이다 (F-COE-81·84). 설정
`config/console.yaml`. 도메인 로직을 두지 않는다 (단계 조건·권한 검사는 `dlp` 명령이 하고, 콘솔 활성 규칙은 안내용).

### `packages/active/` (`dlp_active`)

액티브 러닝: 클래스별 수정률 기반 세션 점수(점수 항목 플러그인 `register_term`),
FiftyOne 연동(블러본만). `dlp active rank|fiftyone`. 정책은 `config/policies/active.yaml`.
FiftyOne은 선택 설치(`make install-curation`).

### `packages/export/` (`dlp_export`)

내보내기: COCO, 구간 JSON(`dlp_schema.export`), LeRobot v3.0 에피소드.
데이터셋 버전 스냅샷에서, 검증 정책 적용, 블러본만. 작업자·장소 ID는 내보내기마다 다른 가명(HMAC). 내보내기 이력은
올리기 전에 따로 커밋한다. 세션·라벨 ID도 내보내기마다 다른 가명이고(이력에는 내부 ID, 세션 대응표는 내부 경로),
manifest에 파일 목록과 sha256을 넣는다 (ADR 0027). `dlp export coco|intervals|lerobot`. 정책은 `config/policies/export.yaml`.
온톨로지 한·영 이름을 싣는다(COCO 범주 `name_ko`·`name_en`, 구간 JSON `term_names` 형식 1.2, LeRobot 어휘 `term_names`).
설명은 언어마다 한 레코드(ko·en)이고 번역의 `source_label_id`도 가명 라벨 ID다 (ADR 0043).
LeRobot은 격리된 일회용 환경(`scripts/lerobot_write.py`, `scripts/lerobot_check.py`)에서 공식 API로 쓰고 읽는다.
객체 물리 사전값(재질·촉감·질량·마찰, ADR 0049)은 세 형식 모두 `measured: false`로 싣는다 (측정값 아님).
manifest `verification_policy.trajectory3d`는 3D 검수 근거(2D에서 물려받음, `camera_pose: automatic_unreviewed`,
`pose_derived`, 근거별 수)를 밝히고, 끊긴 물려받은 3D는 옵션과 관계없이 넣지 않는다 (ADR 0048·0050).
3D는 `lerobot.interp_max_gap_3d_ms`로 보간하고 궤적 구간 채움 비율이 낮으면 경고한다. LeRobot에 3D 방식·내부 파라미터
출처·접촉 객체 클래스 특징을 싣는다. 페이로드의 라벨 ID 필드(`pseudonym.LABEL_REF_FIELDS`, 예: `derived_from`)도 라벨
가명이 된다. 손 부위 설정은 내보내기 시작 때 hand_joints와 대조한다 (ADR 0053).
COCO `wearer_arm` 범주, LeRobot `annotation.wearer_arm`(arm3 2D + 팔꿈치 3D, ADR 0058).
리타기팅이 버린 에피소드 보고에는 세션 가명을 싣지 않는다 (`session_id: null`, F-COE-76).
COCO에 마스크 `segmentation`(압축 RLE)·`mask_score`·`hand`, 객체 hand 범주가 손 키포인트 범주를 겸한다 (ADR 0059).

### `packages/ops/` (`dlp_ops`)

운영 대시보드와 보안: 주간 운영 지표(검수 시간·수정률·자동 승인율·편향·발견율·
잔여 블러 누락·원가, AI 1차 검수 절: 백로그·fallback·캐시 적중·장비 시간·ok 놓침 상한)와 경고, 원본 접근 월간 감사
리포트, 원본 보관 만료 알림·결정.
`dlp ops weekly|audit-report|retention|retention-decide|log-work|privacy-audit`. 정책은 `config/policies/ops.yaml`.

### `config/`

온톨로지·정책·기본값 YAML. 코드에 값을 하드코딩하지 않는다.
온톨로지 항목은 한국어 `ko`와 영어 `en` 표시 이름이 모두 필수다 (ID는 영어 snake_case, 라벨에는 ID만, ADR 0043).
객체 물리 사전값(`physical_priors`: 재질·촉감·질량·마찰 범위, 출처는 사전값만)은 `config/ontology/v1/physics.yaml` 어휘를 쓴다 (ADR 0049).
by_state는 상태 속성 하나만, 젖음 기본 범위 = 건조, prior_literature는 note 필수, 사전값 표에 `content_sha256` (ADR 0053).

### `schemas/`

생성된 JSON Schema. 직접 편집하지 않는다 (`make schemas`).

### `services/`

개발용 docker compose (PostgreSQL, SeaweedFS S3, Label Studio, Prefect, MLflow, lakeFS), CVAT 실행 스크립트.

### `tests/`

패키지를 가로지르는 통합 테스트.

### `docs/adr/`

아키텍처 결정 기록.
