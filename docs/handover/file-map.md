# 파일 지도

> 생성 파일이다. 직접 고치지 않는다. 각 파일 맨 위 설명의 첫 문단(파이썬은 모듈 docstring, 설정과
> 스크립트는 맨 위 주석)을 고친 뒤 `make file-map`으로 다시 만든다. `make contracts`가 이 파일이
> 코드와 같은지 검사한다.

"이 일은 어느 파일인가"를 찾는 표다. 파일마다 그 파일이 스스로 적은 역할을 한 칸에 옮겼다. 자세한
설명(공개 함수, 지키는 규칙, 호출 위치)은 각 파일 맨 위에 이어서 적혀 있다. 패키지 사이의 큰 그림과
의존 표는 [구조 지도](architecture.md), 바꿀 때 함께 볼 파일과 시험은
[변경 및 복구 안내](change-guide.md)에 있다.

파일 683개.

## 목차

- [packages/actions](#packagesactions)
- [packages/active](#packagesactive)
- [packages/aireview](#packagesaireview)
- [packages/cli](#packagescli)
- [packages/console](#packagesconsole)
- [packages/datasets](#packagesdatasets)
- [packages/evaluation](#packagesevaluation)
- [packages/export](#packagesexport)
- [packages/fixtures](#packagesfixtures)
- [packages/media](#packagesmedia)
- [packages/models](#packagesmodels)
- [packages/ops](#packagesops)
- [packages/prelabel](#packagesprelabel)
- [packages/privacy](#packagesprivacy)
- [packages/relations](#packagesrelations)
- [packages/retarget](#packagesretarget)
- [packages/review](#packagesreview)
- [packages/scene3d](#packagesscene3d)
- [packages/schema](#packagesschema)
- [packages/sync](#packagessync)
- [packages/training](#packagestraining)
- [config](#config)
- [scripts](#scripts)
- [services](#services)
- [tests](#tests)
- [저장소 루트](#저장소-루트)

## packages/actions

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/actions/src/dlp_actions/__init__.py) | 행동 구간 2단 구조: 운동·접촉 신호로 경계 후보 → VLM이 후보 구간 분류·설명 (WP10). |
| [`assemble.py`](../../packages/actions/src/dlp_actions/assemble.py) | 분류된 후보 구간을 행동·사이 구간 라벨로 만든다 (WP10). |
| [`boundaries.py`](../../packages/actions/src/dlp_actions/boundaries.py) | 1단: 손목 속도와 접촉 신호로 행동 경계 후보를 만든다 (WP10, ADR 0012). |
| [`clients.py`](../../packages/actions/src/dlp_actions/clients.py) | VLM 클라이언트: CPU용 stub(정답 기반)과 OpenAI 호환 서버(vLLM 등) 어댑터 (WP10). |
| [`describe.py`](../../packages/actions/src/dlp_actions/describe.py) | 사람이 의미를 고친 행동의 한국어 설명을 행동 VLM으로 다시 만든다 (`dlp actions describe`). 멱등. |
| [`pipeline.py`](../../packages/actions/src/dlp_actions/pipeline.py) | 한 손의 행동 구간: 경계 후보 → 후보 구간 VLM 분류 → 병합·채우기 → 라벨 (WP10, ADR 0012). |
| [`policy.py`](../../packages/actions/src/dlp_actions/policy.py) | 행동 구간 정책 (config/policies/actions.yaml, WP10, ADR 0012). |
| [`runner.py`](../../packages/actions/src/dlp_actions/runner.py) | 세션 행동 구간 실행 (`dlp actions run`). 멱등이다 (WP10, ADR 0012·0015·0024·0026). |
| [`translate.py`](../../packages/actions/src/dlp_actions/translate.py) | 한국어 설명 라벨의 영어 번역 단계 (`dlp translate run`, ADR 0043). 멱등이다. |
| [`translate_policy.py`](../../packages/actions/src/dlp_actions/translate_policy.py) | 설명 라벨 영어 번역 정책 (config/policies/translate.yaml, ADR 0043). |
| [`translators.py`](../../packages/actions/src/dlp_actions/translators.py) | 설명 번역기: CPU용 용어집 stub과 사내 OpenAI 호환 서버 어댑터 (ADR 0043). |
| [`vlm.py`](../../packages/actions/src/dlp_actions/vlm.py) | 2단: VLM이 후보 구간을 온톨로지 목록 안에서 분류하고 한 문장 설명을 쓴다 (WP10, ADR 0012·0026). |
| [`write_check.py`](../../packages/actions/src/dlp_actions/write_check.py) | 행동 단계·설명 재생성·번역의 쓰기 직전 확인 (F-PAR-23·80, ADR 0043 3절). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_actions.py`](../../packages/actions/tests/test_actions.py) | 행동 구간 2단 구조 단위 테스트 (WP10 완료 기준, ADR 0012·0026). |
| [`test_describe.py`](../../packages/actions/tests/test_describe.py) | 사람이 의미를 고친 행동의 설명 재생성 (`dlp_actions.describe`, ADR 0043 3절). |
| [`test_runner.py`](../../packages/actions/tests/test_runner.py) | `run_actions` DB 통합 테스트 (WP10, `@pytest.mark.services`, ADR 0015·0024·0026). |
| [`test_runner_write_check.py`](../../packages/actions/tests/test_runner_write_check.py) | `run_actions`의 쓰기 직전 확인과 재계획 (F-PAR-80·81). 시간대 보존 SQLite와 `OracleVlm`으로 돈다. |
| [`test_translate.py`](../../packages/actions/tests/test_translate.py) | 한국어 설명 → 영어 번역 단계 테스트 (ADR 0043). 서비스 없이 시간대 보존 SQLite로 돈다. |
| [`test_vlm_body.py`](../../packages/actions/tests/test_vlm_body.py) | 행동 VLM 요청 본문 고정 테스트 (WP17.1, ADR 0040). |

## packages/active

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/active/src/dlp_active/__init__.py) | 액티브 러닝 (WP14): 다음에 검수·라벨링할 세션을 고른다. |
| [`curation.py`](../../packages/active/src/dlp_active/curation.py) | FiftyOne 연동 (`dlp active fiftyone`): 고른 세션을 FiftyOne 동영상 데이터셋으로 만들어 분석한다. |
| [`policy.py`](../../packages/active/src/dlp_active/policy.py) | 액티브 러닝 정책 (config/policies/active.yaml) 로더 (WP14). |
| [`rates.py`](../../packages/active/src/dlp_active/rates.py) | 클래스별 검수자 수정률: 라벨 이력에서 "라벨 종류/클래스"마다 사람이 모델 라벨을 고치거나 지운 비율을 낸다. |
| [`select.py`](../../packages/active/src/dlp_active/select.py) | 세션 점수와 선택 (`dlp active rank`). |
| [`terms.py`](../../packages/active/src/dlp_active/terms.py) | 세션 점수 항목 (플러그인): 액티브 러닝 점수를 이루는 항목을 이름으로 등록하고 정책의 가중치로 켠다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_curation.py`](../../packages/active/tests/test_curation.py) | FiftyOne 샘플 변환과 실제 MongoDB 드라이버 왕복 검사 (WP14). |
| [`test_db.py`](../../packages/active/tests/test_db.py) | DB(make up)의 세션에서 순위를 매기고, 고른 세션의 블러본으로 FiftyOne 샘플을 만든다. |
| [`test_select.py`](../../packages/active/tests/test_select.py) | 수정률·세션 점수·선택 단위 테스트 (DB 없이, WP14). |

## packages/aireview

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/aireview/src/dlp_aireview/__init__.py) | AI 1차 검수(로컬 VLM): 승인된 블러본 프레임으로 검수 전 자동 라벨을 판정한다 (WP17, ADR 0040). |
| [`backend.py`](../../packages/aireview/src/dlp_aireview/backend.py) | 판정 백엔드 경계 (WP17.2, ADR 0040). 백엔드를 바꿔 끼우는 곳은 여기 하나다. |
| [`bench.py`](../../packages/aireview/src/dlp_aireview/bench.py) | 처리량 벤치 `dlp aireview bench` (WP17.3, ADR 0040 5.6·8.3). DB에 쓰지 않는다. |
| [`crops.py`](../../packages/aireview/src/dlp_aireview/crops.py) | 잘라 낸 박스의 AI 확인: 블러·도구 박스 오탐 의심 힌트 (ADR 0056 3절, ADR 0040). |
| [`estimate.py`](../../packages/aireview/src/dlp_aireview/estimate.py) | 작업량·장비 대수 추정 `dlp aireview estimate` (WP17.3, ADR 0040 5.8·8.3). 순수 함수. |
| [`evaluate.py`](../../packages/aireview/src/dlp_aireview/evaluate.py) | 골든셋 판정 정확도 (`dlp aireview eval --golden <버전>`, WP17.8, ADR 0040 7.4). DB에 쓰지 않는다. |
| [`judge.py`](../../packages/aireview/src/dlp_aireview/judge.py) | 단위 하나를 판정한다: 재시도, 부분 수용, 미응답 대체, 단위 안 문맥 초과 분할 (WP17.2). |
| [`media.py`](../../packages/aireview/src/dlp_aireview/media.py) | 블러본 경계: AI 판정이 픽셀을 얻는 유일한 길 (WP17.5, ADR 0040 2.4·I1). |
| [`oracle.py`](../../packages/aireview/src/dlp_aireview/oracle.py) | 정답 라벨로 답하는 판정 stub (CPU·CI용 `JudgeBackend`, WP17.2, ADR 0040). |
| [`overlay.py`](../../packages/aireview/src/dlp_aireview/overlay.py) | 판정 프레임에 표식·시각을 그리고 확대 조각을 만든다 (WP17.2, ADR 0040 2.3). |
| [`perturb.py`](../../packages/aireview/src/dlp_aireview/perturb.py) | 시드가 정해진 라벨 교란 9종 (통제된 오류, WP17.2, ADR 0040 7.4). |
| [`plan.py`](../../packages/aireview/src/dlp_aireview/plan.py) | 판정 요청 단위 계획 (순수·결정적, WP17.2, ADR 0040 2.1·2.2·4.3). |
| [`policy.py`](../../packages/aireview/src/dlp_aireview/policy.py) | AI 1차 검수 정책 (`config/policies/aireview.yaml`, WP17.2, ADR 0040). |
| [`probe.py`](../../packages/aireview/src/dlp_aireview/probe.py) | VLM 서버 점검 `dlp aireview probe` (WP17.3, ADR 0040 5.7·I7). DB에 쓰지 않는다. |
| [`prompt.py`](../../packages/aireview/src/dlp_aireview/prompt.py) | 백엔드 중립 판정 프롬프트 (WP17.2, ADR 0040 2.6). |
| [`runner.py`](../../packages/aireview/src/dlp_aireview/runner.py) | 판정 실행기: 계획 → 생산자·작업자·기록자 → 마감 (WP17.5, ADR 0040 5.1·5.3·5.4). |
| [`schema.py`](../../packages/aireview/src/dlp_aireview/schema.py) | 판정 응답 JSON Schema와 라벨 종류별 문제 유형·제안 필드 (WP17.2, ADR 0040 3.1). |
| [`sensors.py`](../../packages/aireview/src/dlp_aireview/sensors.py) | 장갑 압력 접촉 구간을 판정 프롬프트 문장으로 바꾼다 (WP17.2, ADR 0040 2.5). |
| [`status.py`](../../packages/aireview/src/dlp_aireview/status.py) | 판정 진행 상태와 `--pending` 세션 고르기 (WP17.5, ADR 0040 5.2). DB를 읽기만 한다. |
| [`suggest.py`](../../packages/aireview/src/dlp_aireview/suggest.py) | 검증된 제안 패치를 완성 페이로드로 만들고 온톨로지로 검사한다 (WP17.2, ADR 0040 3.2 5단계). |
| [`testing.py`](../../packages/aireview/src/dlp_aireview/testing.py) | 가짜 OpenAI 호환 VLM 서버 (테스트·CI용, WP17.3). 실제 모델을 부르지 않는다. |
| [`tracks.py`](../../packages/aireview/src/dlp_aireview/tracks.py) | 공간 트랙의 시각별 모양 (키프레임 선형 보간, 순수 함수, WP17.2, ADR 0040 2.2). |
| [`validate.py`](../../packages/aireview/src/dlp_aireview/validate.py) | 판정 응답 재검사 (WP17.2, ADR 0040 3.2). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`blurred.py`](../../packages/aireview/tests/blurred.py) | DB·블러본이 있는 합성 세션 (실행기·블러본 경계 테스트 공용, WP17.5). |
| [`conftest.py`](../../packages/aireview/tests/conftest.py) | AI 1차 검수 테스트 공용 픽스처 (WP17.2). |
| [`test_bench.py`](../../packages/aireview/tests/test_bench.py) | 처리량 벤치(`dlp_aireview.bench`) 테스트 (WP17.3, ADR 0040 5.6). |
| [`test_blurred_only.py`](../../packages/aireview/tests/test_blurred_only.py) | 블러본 경계 테스트 (`media.open_blurred`·`frames_for`, WP17.5, ADR 0040 2.4·I1). |
| [`test_crops.py`](../../packages/aireview/tests/test_crops.py) | 잘라 낸 박스 AI 확인(`dlp_aireview.crops`) 시험 (ADR 0056). |
| [`test_estimate.py`](../../packages/aireview/tests/test_estimate.py) | 작업량·장비 대수 추정(`dlp_aireview.estimate`) 테스트 (WP17.3, ADR 0040 5.8·8.3). |
| [`test_evaluate.py`](../../packages/aireview/tests/test_evaluate.py) | 골든셋 판정 정확도 (`evaluate`, WP17.8 완료 기준: 손 계산 일치, DB에 쓰지 않음). |
| [`test_fake_server.py`](../../packages/aireview/tests/test_fake_server.py) | 가짜 OpenAI 호환 VLM 서버(`dlp_aireview.testing.FakeOpenAIServer`) 테스트 (WP17.3). |
| [`test_judge.py`](../../packages/aireview/tests/test_judge.py) | 단위 판정 경로 테스트 (`dlp_aireview.judge`, 정해 둔 응답을 내는 백엔드). |
| [`test_oracle_catches.py`](../../packages/aireview/tests/test_oracle_catches.py) | 정답을 아는 합성 오류 탐지와 판정 경로 테스트 (`perturb`·`oracle`·`judge`, WP17.2 완료 기준). |
| [`test_overlay.py`](../../packages/aireview/tests/test_overlay.py) | 표식·시각 표기·확대 조각 테스트 (`dlp_aireview.overlay`, WP17.2). |
| [`test_plan.py`](../../packages/aireview/tests/test_plan.py) | 판정 단위 계획 테스트 (`dlp_aireview.plan`, WP17.2, ADR 0040 2.1·2.2·4.3). |
| [`test_policy.py`](../../packages/aireview/tests/test_policy.py) | AI 1차 검수 정책 테스트 (`dlp_aireview.policy`, WP17.2). |
| [`test_pool.py`](../../packages/aireview/tests/test_pool.py) | 여러 엔드포인트 묶음·차단기 테스트 (`backend.EndpointPool`, WP17.5, ADR 0040 5.3). |
| [`test_probe.py`](../../packages/aireview/tests/test_probe.py) | VLM 서버 점검(`dlp_aireview.probe`) 테스트 (WP17.3, ADR 0040 5.7·I7). |
| [`test_prompt.py`](../../packages/aireview/tests/test_prompt.py) | 프롬프트·스키마·백엔드 요청 테스트 (`dlp_aireview.prompt`·`schema`·`backend`, WP17.2). |
| [`test_runner.py`](../../packages/aireview/tests/test_runner.py) | 판정 실행기 테스트 (`runner.run_session`, `MemorySink`, WP17.5, ADR 0040 5.1·5.3·9장). |
| [`test_runner_db.py`](../../packages/aireview/tests/test_runner_db.py) | DB 경로 실행기 테스트 (`runner.run_from_db`·`load_input`·`DbSink`, SQLite, WP17.5). |
| [`test_runner_pg.py`](../../packages/aireview/tests/test_runner_pg.py) | 실행기의 PostgreSQL 성질 (services, WP17.5, ADR 0040 5.1·I5·I8). |
| [`test_status.py`](../../packages/aireview/tests/test_status.py) | 판정 진행 상태와 `--pending` 선택 테스트 (`status`, SQLite, WP17.5, ADR 0040 5.2 C12). |
| [`test_validate.py`](../../packages/aireview/tests/test_validate.py) | 응답 재검사·제안 완성 테스트 (`dlp_aireview.validate`·`suggest`, WP17.2, ADR 0040 3.2). |

## packages/cli

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/cli/src/dlp_cli/__init__.py) | `dlp` 명령줄 도구 패키지 (`dlp_cli`). |
| [`actions_cmds.py`](../../packages/cli/src/dlp_cli/actions_cmds.py) | 행동 구간 하위 명령 (WP10, ADR 0012·0026·0043). |
| [`active_cmds.py`](../../packages/cli/src/dlp_cli/active_cmds.py) | 액티브 러닝 하위 명령 (WP14, ADR 0017). |
| [`ai_quality_print.py`](../../packages/cli/src/dlp_cli/ai_quality_print.py) | AI 1차 검수 품질 보고 출력 (`dlp review quality --ai`, WP17.8, ADR 0040 7.3). |
| [`aireview_cmds.py`](../../packages/cli/src/dlp_cli/aireview_cmds.py) | AI 1차 검수 하위 명령 (WP17, ADR 0040): 서버 점검·처리량 벤치·작업량 추정 (WP17.3). |
| [`aireview_common.py`](../../packages/cli/src/dlp_cli/aireview_common.py) | AI 1차 검수 명령의 공용 도우미 (`aireview_cmds`·`aireview_run_cmds`·`aireview_eval_cmds`, WP17). |
| [`aireview_eval_cmds.py`](../../packages/cli/src/dlp_cli/aireview_eval_cmds.py) | `dlp aireview eval`: 골든셋 판정 정확도와 게이트 (WP17.8, ADR 0040 7.4). DB에 쓰지 않는다. |
| [`aireview_run_cmds.py`](../../packages/cli/src/dlp_cli/aireview_run_cmds.py) | AI 1차 검수 실행·상태 하위 명령 (WP17.5, ADR 0040 5.2~5.5). `aireview_cmds.add_commands`가 등록한다. |
| [`calib_cmds.py`](../../packages/cli/src/dlp_cli/calib_cmds.py) | 카메라 내부 파라미터·작업자 손 크기 등록 하위 명령 (ADR 0045, 0052). |
| [`console_cmds.py`](../../packages/cli/src/dlp_cli/console_cmds.py) | 운영 콘솔 하위 명령 (ADR 0061). |
| [`dataset_cmds.py`](../../packages/cli/src/dlp_cli/dataset_cmds.py) | 데이터셋·계보 하위 명령 (WP7, ADR 0007). |
| [`eval_cmds.py`](../../packages/cli/src/dlp_cli/eval_cmds.py) | 골든셋 평가 하위 명령 (WP11, ADR 0013·0025). |
| [`export_cmds.py`](../../packages/cli/src/dlp_cli/export_cmds.py) | 내보내기 하위 명령 (WP15, ADR 0018·0021·0027·0029). |
| [`fixtures_cmds.py`](../../packages/cli/src/dlp_cli/fixtures_cmds.py) | 합성 픽스처 하위 명령 (WP2, `make fixtures`). |
| [`health.py`](../../packages/cli/src/dlp_cli/health.py) | 개발 서비스(docker compose) 헬스체크 (WP0, `dlp services check`, `make health`). |
| [`hints_cmds.py`](../../packages/cli/src/dlp_cli/hints_cmds.py) | 오탐 의심 힌트 하위 명령 `dlp review hints <세션>` (ADR 0055·0056). |
| [`main.py`](../../packages/cli/src/dlp_cli/main.py) | `dlp` 명령줄 진입점 (WP0): 최상위 파서를 만들고 `<단계>_cmds.py`의 하위 명령을 등록해 실행 함수로 넘긴다. |
| [`media_cmds.py`](../../packages/cli/src/dlp_cli/media_cmds.py) | 미디어·수집 하위 명령 (WP3, ADR 0003). |
| [`models_cmds.py`](../../packages/cli/src/dlp_cli/models_cmds.py) | 모델 가중치 내려받기·변환과 라이선스 검사 (`config/models.yaml`, ADR 0009·0010). |
| [`ontology_access.py`](../../packages/cli/src/dlp_cli/ontology_access.py) | CLI에서 세션·데이터셋 버전에 맞는 등록 사전을 읽는다. 연결은 항상 닫는다. |
| [`ops_cmds.py`](../../packages/cli/src/dlp_cli/ops_cmds.py) | 운영 대시보드와 보안 하위 명령 (WP16, ADR 0020·0021). |
| [`pilot_cmds.py`](../../packages/cli/src/dlp_cli/pilot_cmds.py) | 실제 영상 파일럿의 실행 전 준비 검사 (`dlp pilot check`). |
| [`prelabel_cmds.py`](../../packages/cli/src/dlp_cli/prelabel_cmds.py) | 자동 프리라벨 하위 명령 (WP8, ADR 0008·0009·0026). |
| [`privacy_cmds.py`](../../packages/cli/src/dlp_cli/privacy_cmds.py) | 프라이버시 게이트 하위 명령 (WP5, ADR 0005·0023·0024). |
| [`raw_access.py`](../../packages/cli/src/dlp_cli/raw_access.py) | 원본 버킷 저장소는 이 모듈로만 만든다 (모든 접근이 감사 기록으로 남는다, WP16, ADR 0020·0021). |
| [`relations_cmds.py`](../../packages/cli/src/dlp_cli/relations_cmds.py) | 관계 도출·커버리지 하위 명령 (WP9, ADR 0011·0026). |
| [`review_cmds.py`](../../packages/cli/src/dlp_cli/review_cmds.py) | 검수 도구 연동·검수 운영 하위 명령 (WP6·WP12, ADR 0006·0014·0023·0024·0029). |
| [`review_config.py`](../../packages/cli/src/dlp_cli/review_config.py) | CLI의 검수 정책·설정 선택과 매체 회수 실행 (ADR 0039). |
| [`robot_cmds.py`](../../packages/cli/src/dlp_cli/robot_cmds.py) | 로봇 등록 확인 하위 명령 (ADR 0047·0068). |
| [`scene3d_cmds.py`](../../packages/cli/src/dlp_cli/scene3d_cmds.py) | 청소 전후 다각도 사진의 3D 복원·물체 이동 비교와 장면 안 바디캠 위치 추정 하위 명령 (ADR 0066, 0067). |
| [`schema_cmds.py`](../../packages/cli/src/dlp_cli/schema_cmds.py) | 스키마·온톨로지·DB 하위 명령 (WP1, ADR 0002). |
| [`sync_cmds.py`](../../packages/cli/src/dlp_cli/sync_cmds.py) | 동기화 하위 명령 (WP4, ADR 0004·0022·0028). |
| [`train_cmds.py`](../../packages/cli/src/dlp_cli/train_cmds.py) | 재학습 하위 명령 (WP13, ADR 0016·0025). |
| [`translate_cmds.py`](../../packages/cli/src/dlp_cli/translate_cmds.py) | 설명 라벨 영어 번역 하위 명령 (ADR 0043). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_actions_describe_cmd.py`](../../packages/cli/tests/test_actions_describe_cmd.py) | `dlp actions describe` 명령 (ADR 0043 3절). 시간대 보존 SQLite 파일 DB와 `OracleVlm`으로 돈다. |
| [`test_actions_run_cmd.py`](../../packages/cli/tests/test_actions_run_cmd.py) | `dlp actions run\|describe`의 VLM 주소 검사(F-PAR-82)와 쓰기 직전 확인 종료 코드(F-PAR-80). |
| [`test_aireview_calibrate_cmd.py`](../../packages/cli/tests/test_aireview_calibrate_cmd.py) | `dlp aireview calibrate`와 검수 CLI의 보정 고정 읽기 (ADR 0062). |
| [`test_aireview_cmds.py`](../../packages/cli/tests/test_aireview_cmds.py) | `dlp aireview probe\|bench\|estimate` CLI 종단 테스트 (WP17.3, ADR 0040 5.6~5.8). |
| [`test_aireview_eval_cmds.py`](../../packages/cli/tests/test_aireview_eval_cmds.py) | `dlp aireview eval` CLI 종단 (WP17.8 완료 기준: eval이 DB에 쓰지 않음, 게이트는 경고만). |
| [`test_aireview_run_cmds.py`](../../packages/cli/tests/test_aireview_run_cmds.py) | `dlp aireview run\|status\|show\|waive` CLI 종단 테스트 (WP17.5, ADR 0040 5.2·5.3). |
| [`test_calib_cmds.py`](../../packages/cli/tests/test_calib_cmds.py) | `dlp calib intrinsics\|import\|assign\|palm\|show` 명령 테스트 (ADR 0045·0052, SQLite 파일 DB). |
| [`test_calib_palm_estimate.py`](../../packages/cli/tests/test_calib_palm_estimate.py) | `dlp calib palm-estimate\|palm-revoke` 명령 경로 (ADR 0045·0052, F-COE-71, SQLite 파일 DB). |
| [`test_dataset_build_cmd.py`](../../packages/cli/tests/test_dataset_build_cmd.py) | `dlp dataset build`가 빌드 거절을 traceback 없이 알린다 (F-ETA-31). |
| [`test_export_cmds.py`](../../packages/cli/tests/test_export_cmds.py) | `dlp export`의 촉각장갑용 원본 감사 저장소 선택과 요약 출력 (ADR 0069·0070, DB·저장소 없이). |
| [`test_golden_cmds.py`](../../packages/cli/tests/test_golden_cmds.py) | 골든셋 CLI의 정답 고정 (`dlp dataset golden --create\|--from`, `dlp dataset build --golden`, `dlp eval golden`, ADR 0041). |
| [`test_health.py`](../../packages/cli/tests/test_health.py) | `dlp_cli.health`와 `dlp` 진입점의 단위 테스트 (WP0). |
| [`test_main.py`](../../packages/cli/tests/test_main.py) | `dlp` 진입점의 인자 해석 (`dlp_cli.main.parse_cli_args`). |
| [`test_ops_weekly_args.py`](../../packages/cli/tests/test_ops_weekly_args.py) | `dlp ops weekly --weeks`는 1~52만 받는다 (F-COE-88). |
| [`test_pilot.py`](../../packages/cli/tests/test_pilot.py) | 준비 검사·모델 명령·검수 정책은 오타/부분 준비를 성공으로 숨기지 않는다. |
| [`test_prelabel_scene_hosts.py`](../../packages/cli/tests/test_prelabel_scene_hosts.py) | 장면 분류 VLM 주소가 사내 대역 밖이면 원본을 받기 전에 멈춘다 (F-PAR-89). |
| [`test_raw_access_offline.py`](../../packages/cli/tests/test_raw_access_offline.py) | DB 없이(--no-db, 로컬 저장소) 수집해도 원본 접근이 감사 기록(JSON Lines)에 남는다 (WP16, ADR 0021). |
| [`test_review_ai_cmds.py`](../../packages/cli/tests/test_review_ai_cmds.py) | `dlp review`의 AI 1차 검수 연결: 판정 사용 설정, required 배정 게이트, 회수 뒤 안내, 큐 표시 (WP17.6, ADR 0040). |
| [`test_review_policy.py`](../../packages/cli/tests/test_review_policy.py) | `dlp review` 명령은 고른 검수 정책(`DLP_REVIEW_POLICY_FILE`)을 한 출처로 쓴다 (F-REV-01). |
| [`test_revoke_cmds.py`](../../packages/cli/tests/test_revoke_cmds.py) | 검수·프라이버시 CLI의 매체 회수 연결 (ADR 0039). |
| [`test_robot_cmds.py`](../../packages/cli/tests/test_robot_cmds.py) | `dlp robot check` (ADR 0047): 공개 시험 로봇은 통과하고 없는 로봇·틀린 해시는 종료 코드 1. |
| [`test_robot_hand_cmds.py`](../../packages/cli/tests/test_robot_hand_cmds.py) | `dlp robot check`의 다지 손 로봇 출력과 내보내기 요약의 손가락 줄 (ADR 0068). |
| [`test_scene3d_cmds.py`](../../packages/cli/tests/test_scene3d_cmds.py) | `dlp scene3d …` 오프라인 종단 시험 (로컬 저장소, SQLite 파일 DB, 정답 stub, ADR 0066·0067). |
| [`test_sync_cmds.py`](../../packages/cli/tests/test_sync_cmds.py) | `dlp sync adjust` 명령줄 인자 검사 (F-SCH-27). |
| [`test_translate_cmds.py`](../../packages/cli/tests/test_translate_cmds.py) | `dlp translate run\|pending`와 `dlp actions run` 끝의 자동 번역 (ADR 0043). |

## packages/console

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/console/src/dlp_console/__init__.py) | 운영 콘솔 (ADR 0061): 기존 `dlp` 명령을 브라우저에서 작업으로 돌리는 화면. |
| [`app.py`](../../packages/console/src/dlp_console/app.py) | 운영 콘솔 웹 앱 (FastAPI, ADR 0061). |
| [`global_steps.py`](../../packages/console/src/dlp_console/global_steps.py) | 세션 밖 단계: 데이터셋 버전·재학습·평가·운영 지표 (ADR 0061 3절). |
| [`hints.py`](../../packages/console/src/dlp_console/hints.py) | 실패한 작업 로그 → 마지막 오류 줄과 다음 할 일 안내 (한국어, ADR 0061 3절). |
| [`imports.py`](../../packages/console/src/dlp_console/imports.py) | 가중치 가져오기 진행 기록 (작업 디렉터리, ADR 0061 6절). |
| [`jobs.py`](../../packages/console/src/dlp_console/jobs.py) | 작업(job) 실행기: 기존 `dlp <단계>` 명령을 자식 프로세스로 돌리고 상태·로그를 남긴다. |
| [`pretrain.py`](../../packages/console/src/dlp_console/pretrain.py) | 외부에서 학습한 가중치 가져오기 (ADR 0060 1단계 산출물, ADR 0061 6절). |
| [`preview.py`](../../packages/console/src/dlp_console/preview.py) | 결과 보기: 승인·렌더된 블러본 + 라벨 겹쳐 그리기 자료 (ADR 0061 3절). |
| [`redact.py`](../../packages/console/src/dlp_console/redact.py) | 화면에 보내는 글에서 원본 위치와 비밀값을 가린다 (ADR 0061 4절). |
| [`server.py`](../../packages/console/src/dlp_console/server.py) | `dlp console serve` 실행 (uvicorn, ADR 0061 4절). |
| [`settings.py`](../../packages/console/src/dlp_console/settings.py) | 콘솔 설정·실행 환경 (ADR 0061). |
| [`status.py`](../../packages/console/src/dlp_console/status.py) | 세션 상태 읽기 (읽기 전용, ADR 0061 3절). |
| [`steps.py`](../../packages/console/src/dlp_console/steps.py) | 세션 파이프라인 단계와 활성 규칙 (ADR 0061 3절). |
| [`training.py`](../../packages/console/src/dlp_console/training.py) | 재학습·성과 화면 자료 (읽기 전용, ADR 0061 5절). |
| [`upload.py`](../../packages/console/src/dlp_console/upload.py) | 영상 올리기 → 매니페스트 → `dlp media prepare` + `dlp ingest` 작업 (ADR 0061 3절). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/console/tests/conftest.py) | 콘솔 시험 공용 도구: 가짜 `dlp` 명령, 임시 작업 디렉터리, SQLite DB를 쓰는 앱. |
| [`test_app.py`](../../packages/console/tests/test_app.py) | API: 토큰·Host 검사, 업로드 → 매니페스트 → 작업, 세션 상태·단계, 결과 보기, 원본 위치 비노출. |
| [`test_body_limit.py`](../../packages/console/tests/test_body_limit.py) | F-COE-90: 올리기 경로는 본문을 읽기 전에 Content-Length로 전체 크기를 막는다. |
| [`test_jobs.py`](../../packages/console/tests/test_jobs.py) | 작업 실행기: 상태 전이·로그·실패 처리·동시 실행 방지·중단 복구·가리기. |
| [`test_pretrain.py`](../../packages/console/tests/test_pretrain.py) | 외부 학습 가중치 가져오기: manifest 검사, models.yaml 항목·분류, 확인 기록, 비교 버튼 조건. |
| [`test_server.py`](../../packages/console/tests/test_server.py) | `dlp console serve`가 찍는 콘솔 주소의 호스트 표기. |
| [`test_steps.py`](../../packages/console/tests/test_steps.py) | 단계 활성 규칙과 명령 (앞 단계가 끝나지 않으면 막고 이유를 보인다). |

## packages/datasets

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/datasets/src/dlp_datasets/__init__.py) | 데이터셋 버전과 계보 (WP7, ADR 0007). |
| [`build.py`](../../packages/datasets/src/dlp_datasets/build.py) | 데이터셋 버전 빌드 (`dlp dataset build`): 후보 세션을 고르고 작업자·장소 단위로 분할한 뒤 라벨 스냅샷을 만든다. |
| [`lineage.py`](../../packages/datasets/src/dlp_datasets/lineage.py) | 사용 중지 전파와 계보 조회 (`dlp dataset withdraw`, `dlp lineage`): 사용 중지한 세션과 영향받은 버전·학습·내보내기. |
| [`policy.py`](../../packages/datasets/src/dlp_datasets/policy.py) | 데이터셋 정책 (config/policies/dataset.yaml) 로더 (WP7). |
| [`snapshot.py`](../../packages/datasets/src/dlp_datasets/snapshot.py) | 데이터셋 스냅샷 저장소. 한 번 커밋한 스냅샷은 바뀌지 않으며 URI로 다시 읽을 수 있다. |
| [`splitter.py`](../../packages/datasets/src/dlp_datasets/splitter.py) | 작업자·장소 단위 분할: 같은 작업자나 장소가 학습과 평가에 함께 들지 않게 세션을 golden·train·val·holdout으로 나눈다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_build_duplicate.py`](../../packages/datasets/tests/test_build_duplicate.py) | 같은 버전 ID로 다시 빌드하면 스냅샷을 커밋하기 전에 거절한다 (F-ETA-31). |
| [`test_lineage.py`](../../packages/datasets/tests/test_lineage.py) | 데이터셋 빌드·사용 중지·계보 통합 테스트 (PostgreSQL·lakeFS, `make up`, WP7). |
| [`test_splits.py`](../../packages/datasets/tests/test_splits.py) | 작업자·장소 단위 분할과 로컬 스냅샷 단위 테스트 (DB 없이, WP7). |
| [`test_withdraw_revocation.py`](../../packages/datasets/tests/test_withdraw_revocation.py) | 사용 중지한 세션의 배포된 검수 작업에 매체 회수를 요청한다 (F-REV-17, ADR 0039). |

## packages/evaluation

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/evaluation/src/dlp_eval/__init__.py) | 지표와 평가 하네스 (WP11): 지표 라이브러리, 골든셋 평가, 하위 집단 리포트, 배포 게이트. |
| [`gate.py`](../../packages/evaluation/src/dlp_eval/gate.py) | 배포 게이트: 후보 모델의 골든셋 리포트를 기존 모델과 비교해 배포 여부를 정한다. |
| [`harness.py`](../../packages/evaluation/src/dlp_eval/harness.py) | 라벨 레코드 → 과제별 지표. 골든셋 세션들의 정답(사람)과 예측(모델 버전)을 비교한다. |
| [`metrics/__init__.py`](../../packages/evaluation/src/dlp_eval/metrics/__init__.py) | 지표 함수. 입력은 단순한 배열·튜플이고 라벨 계약과 무관하다 (하네스가 변환한다). |
| [`metrics/assign.py`](../../packages/evaluation/src/dlp_eval/metrics/assign.py) | 헝가리안 할당 (scipy)의 타입 래퍼. scipy에는 타입 스텁이 없다. |
| [`metrics/classification.py`](../../packages/evaluation/src/dlp_eval/metrics/classification.py) | 분류 지표: macro F1, Cohen 카파, 기대 보정 오차(ECE), 클래스별 표본 수. |
| [`metrics/detection.py`](../../packages/evaluation/src/dlp_eval/metrics/detection.py) | 객체 검출 AP (COCO 방식, pycocotools와 같은 결과). |
| [`metrics/keypoints.py`](../../packages/evaluation/src/dlp_eval/metrics/keypoints.py) | 키포인트 PCK: 정답 관절과 예측 관절 거리가 alpha * 기준 길이 이하인 비율. |
| [`metrics/proportion.py`](../../packages/evaluation/src/dlp_eval/metrics/proportion.py) | 비율의 신뢰 상한 (Clopper-Pearson 정확 구간). scipy에는 타입 스텁이 없다. |
| [`metrics/states.py`](../../packages/evaluation/src/dlp_eval/metrics/states.py) | 상태 전이 정확도: 개체·속성별 상태 값이 바뀌는 시점(이전 값 → 다음 값)을 정답과 맞춘다. |
| [`metrics/temporal.py`](../../packages/evaluation/src/dlp_eval/metrics/temporal.py) | 시간 구간 지표: 구간 IoU, 시점 사건 매칭(접촉 시작·종료 오차·F1), 구간 F1@IoU, temporal mAP, 경계 일치율. |
| [`metrics/tracking.py`](../../packages/evaluation/src/dlp_eval/metrics/tracking.py) | 다중 객체 추적 지표: HOTA, IDF1, CLEAR(MOTA). TrackEval과 같은 계산. |
| [`policy.py`](../../packages/evaluation/src/dlp_eval/policy.py) | 평가 정책 (config/policies/evaluation.yaml). |
| [`proxy.py`](../../packages/evaluation/src/dlp_eval/proxy.py) | 정답 없는 품질 대리 지표 (proxy metrics, `dlp ops quality-proxy`). |
| [`runner.py`](../../packages/evaluation/src/dlp_eval/runner.py) | 골든셋 평가 실행 (`dlp eval golden`). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_baseline_targets.py`](../../packages/evaluation/tests/test_baseline_targets.py) | 비교 버전 '예측 없음' 검사가 과제의 평가 대상만 세는지 (F-ETA-21). |
| [`test_harness.py`](../../packages/evaluation/tests/test_harness.py) | 평가 하네스(`dlp_eval.harness`)와 배포 게이트(`dlp_eval.gate`)의 단위 테스트 (WP11, ADR 0013·0025). |
| [`test_metrics_hand.py`](../../packages/evaluation/tests/test_metrics_hand.py) | 손으로 계산한 작은 예제와의 대조. |
| [`test_metrics_reference.py`](../../packages/evaluation/tests/test_metrics_reference.py) | 참조 구현과의 일치 테스트. |
| [`test_metrics_temporal_reference.py`](../../packages/evaluation/tests/test_metrics_temporal_reference.py) | 구간 지표를 공개 참조 구현의 계산 방식과 대조한다. |
| [`test_prediction_source.py`](../../packages/evaluation/tests/test_prediction_source.py) | 골든 평가의 예측은 모델 출처 레코드만이다. 센서 출처는 정답도 예측도 아니다 (F-ETA-32). |
| [`test_proxy.py`](../../packages/evaluation/tests/test_proxy.py) | 정답 없는 품질 대리 지표(`dlp_eval.proxy`) 시험. |
| [`test_runner.py`](../../packages/evaluation/tests/test_runner.py) | 골든셋 평가 실행기(`dlp_eval.runner`)의 DB 통합 테스트 (WP11, ADR 0013·0015·0025). |
| [`test_truth_lag.py`](../../packages/evaluation/tests/test_truth_lag.py) | 고정한 정답과 지금 정답의 차이 경고 (`dlp_eval.runner.truth_lag`, ADR 0041). |

## packages/export

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/export/src/dlp_export/__init__.py) | 내보내기 (WP15): 데이터셋 버전 → COCO, 구간 JSON, LeRobot 에피소드. |
| [`coco.py`](../../packages/export/src/dlp_export/coco.py) | COCO (객체 박스·마스크·키포인트): 정답 키프레임 시각의 블러본 프레임을 이미지로 낸다. |
| [`frames.py`](../../packages/export/src/dlp_export/frames.py) | 블러본에서 시각에 맞는 프레임을 고른다 (PTS 인덱스로만). |
| [`gr00t.py`](../../packages/export/src/dlp_export/gr00t.py) | NVIDIA Isaac GR00T N1.7 미세조정 형식 (ADR 0065). |
| [`gr00t_check.py`](../../packages/export/src/dlp_export/gr00t_check.py) | GR00T LeRobot 데이터셋 확인 (`dlp export gr00t-check <디렉터리>`, ADR 0065). |
| [`intervals.py`](../../packages/export/src/dlp_export/intervals.py) | 구간 JSON (`dlp-intervals`): 세션마다 파일 하나 (dlp_schema.export.IntervalFile). |
| [`lerobot.py`](../../packages/export/src/dlp_export/lerobot.py) | LeRobot 에피소드 (형식 v3.0, 공식 쓰기·읽기 API). |
| [`policy.py`](../../packages/export/src/dlp_export/policy.py) | 내보내기 정책 (config/policies/export.yaml) 로더와 검증 (WP15, ADR 0018·0021·0027). |
| [`pseudonym.py`](../../packages/export/src/dlp_export/pseudonym.py) | 내보내기마다 다른 가명: 작업자·장소·세션·라벨 ID (export.yaml ids, ADR 0021·0027·0031). |
| [`robot.py`](../../packages/export/src/dlp_export/robot.py) | LeRobot 로봇 리타기팅 에피소드 (`dlp export lerobot --robot <ID>`, ADR 0047·0068). |
| [`runner.py`](../../packages/export/src/dlp_export/runner.py) | 내보내기 실행 (`dlp export <형식> <데이터셋 버전>`). |
| [`scene.py`](../../packages/export/src/dlp_export/scene.py) | 장면 좌표의 바디캠 자세와 전후 3D 장면 파일을 싣는다 (`dlp export lerobot\|gr00t`, ADR 0070). |
| [`source.py`](../../packages/export/src/dlp_export/source.py) | 내보낼 세션과 라벨을 고른다 (모든 형식 공용, WP15, ADR 0018·0021·0024). |
| [`tactile.py`](../../packages/export/src/dlp_export/tactile.py) | 촉각장갑 채널을 에피소드 프레임 시각에 맞춰 싣는다 (`dlp export lerobot\|gr00t`, ADR 0069). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/export/tests/__init__.py) | dlp_export 테스트 패키지 (conftest의 합성 시나리오를 상대 import로 공유한다). |
| [`conftest.py`](../../packages/export/tests/conftest.py) | 정답을 아는 내보내기 시나리오: 블러본(VFR) 하나와 검증 상태가 섞인 라벨. |
| [`test_formats.py`](../../packages/export/tests/test_formats.py) | 내보내기 형식 단위 테스트 (DB 없이). |
| [`test_gr00t.py`](../../packages/export/tests/test_gr00t.py) | GR00T LeRobot v2.1 내보내기 (ADR 0065, DB 없이). |
| [`test_lerobot_official.py`](../../packages/export/tests/test_lerobot_official.py) | LeRobot 공식 쓰기 API로 쓰고 공식 로더(LeRobotDataset)로 다시 읽는다. |
| [`test_robot.py`](../../packages/export/tests/test_robot.py) | LeRobot 로봇 리타기팅 에피소드 (ADR 0047). 서비스·격리 환경 없이 돈다 (내장 IK 풀이기). |
| [`test_robot_hand.py`](../../packages/export/tests/test_robot_hand.py) | LeRobot 다지 손 로봇 리타기팅 에피소드 (ADR 0068). 서비스·격리 환경 없이 돈다 (내장 풀이기). |
| [`test_runner.py`](../../packages/export/tests/test_runner.py) | 데이터셋 버전 → 내보내기 → 이력 (PostgreSQL, make up). |
| [`test_scene.py`](../../packages/export/tests/test_scene.py) | 장면 좌표 바디캠 자세 내보내기 (ADR 0070, DB 없이). |
| [`test_source_inheritance.py`](../../packages/export/tests/test_source_inheritance.py) | 물려받은 3D의 기본 내보내기 경로를 DB·스냅샷을 거쳐 검사한다 (ADR 0048·0050, F-COE-55). |
| [`test_tactile.py`](../../packages/export/tests/test_tactile.py) | 촉각장갑 채널 내보내기 (ADR 0069, DB 없이). |

## packages/fixtures

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/fixtures/src/dlp_fixtures/__init__.py) | 정답을 아는 합성 데이터 생성기. 모든 생성기는 같은 seed에 같은 결과를 낸다 (WP2). |
| [`actions.py`](../../packages/fixtures/src/dlp_fixtures/actions.py) | 정답 경계를 아는 행동 시퀀스: 손 키포인트 궤적, 장갑 압력, 정답 라벨 (WP2 → WP10). |
| [`aireview.py`](../../packages/fixtures/src/dlp_fixtures/aireview.py) | AI 1차 검수 벤치용 합성 장면 (WP17.3, ADR 0040). 실제 영상·사람·글자가 없다. |
| [`arm.py`](../../packages/fixtures/src/dlp_fixtures/arm.py) | 정답을 아는 1인칭 착용자 팔·몸 장면 (ADR 0058). |
| [`calibration.py`](../../packages/fixtures/src/dlp_fixtures/calibration.py) | 정답 내부 파라미터를 아는 체커보드 캘리브레이션 영상 (ADR 0045, `dlp calib intrinsics` 시험). |
| [`camera_motion.py`](../../packages/fixtures/src/dlp_fixtures/camera_motion.py) | 정답 자세를 아는 머리 카메라 움직임: 렌더 프레임, 해석적 깊이, IMU 신호, 월드 좌표 손 (ADR 0046). |
| [`dexhand.py`](../../packages/fixtures/src/dlp_fixtures/dexhand.py) | 직접 쓴 시험용 다지 손 로봇(6관절 팔 + 손바닥 + 다섯 손가락 x 3관절)과 그 설정 (ADR 0068). |
| [`hand3d.py`](../../packages/fixtures/src/dlp_fixtures/hand3d.py) | 정답 3D를 아는 손 시퀀스: 카메라 좌표 21관절, 그 투영 2D, 손 중심 3D 점, 합성 깊이 (ADR 0044). |
| [`io.py`](../../packages/fixtures/src/dlp_fixtures/io.py) | 픽스처 파일 입출력: WAV, Parquet, JSONL, JSON (WP2). |
| [`retarget.py`](../../packages/fixtures/src/dlp_fixtures/retarget.py) | 정답 끝단 궤적을 아는 사람 손 점 (로봇 리타기팅 시험, ADR 0047). |
| [`scene3d.py`](../../packages/fixtures/src/dlp_fixtures/scene3d.py) | 청소 전후 다각도 사진과 그 방을 지나는 바디캠 영상의 합성 정답 (ADR 0066, 0067). |
| [`sessions.py`](../../packages/fixtures/src/dlp_fixtures/sessions.py) | 작업자·장소·도메인 메타데이터를 가진 가짜 세션 (WP2, WP7 분할 시험용). |
| [`shapes.py`](../../packages/fixtures/src/dlp_fixtures/shapes.py) | 정답 모양을 아는 합성 영상: 객체·손 마스크 단계(ADR 0059) 시험용 (WP2). |
| [`sync.py`](../../packages/fixtures/src/dlp_fixtures/sync.py) | 오프셋·드리프트를 아는 다중 스트림 동기화 시나리오 (WP2 → WP4 동기화 테스트). |
| [`tactile.py`](../../packages/fixtures/src/dlp_fixtures/tactile.py) | 시계와 신호를 아는 촉각장갑 기록 (내보내기 촉각 채널 시험, ADR 0069). |
| [`video.py`](../../packages/fixtures/src/dlp_fixtures/video.py) | 합성 영상: 가변 프레임레이트(VFR) MP4 쓰기, QR 슬레이트, 정답을 아는 블러 대상 영상 (WP2). |
| [`wiping.py`](../../packages/fixtures/src/dlp_fixtures/wiping.py) | 정답 접촉 구간·커버리지를 아는 걸레질 시나리오 (WP2 → WP9 관계·커버리지 테스트). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/fixtures/tests/conftest.py) | 픽스처 생성기 테스트 공용 픽스처: 저장소의 온톨로지 v1 (정답 라벨 검증용). |
| [`test_actions.py`](../../packages/fixtures/tests/test_actions.py) | 행동 시퀀스 생성기(`generate_action_scenario`) 자체 테스트 (WP2). |
| [`test_aireview_scene.py`](../../packages/fixtures/tests/test_aireview_scene.py) | AI 1차 검수 벤치용 합성 장면 테스트 (WP17.3). |
| [`test_hand3d.py`](../../packages/fixtures/tests/test_hand3d.py) | `dlp_fixtures.hand3d` 검사: 결정성, 2D가 정답 3D의 투영인지, 합성 깊이가 손바닥에서만 손 깊이인지. |
| [`test_sessions_and_determinism.py`](../../packages/fixtures/tests/test_sessions_and_determinism.py) | 가짜 세션 구조, 생성기 결정성, `generate_all` 출력 테스트 (WP2). |
| [`test_sync.py`](../../packages/fixtures/tests/test_sync.py) | 동기화 시나리오 생성기(`generate_sync_scenario`) 자체 테스트 (WP2). |
| [`test_video.py`](../../packages/fixtures/tests/test_video.py) | 블러 대상 영상 생성기(`generate_blur_scenario`)와 VFR 쓰기 테스트 (WP2, WP5 전제). |
| [`test_wiping.py`](../../packages/fixtures/tests/test_wiping.py) | 걸레질 시나리오 생성기(`generate_wiping_scenario`) 자체 테스트 (WP2, WP9 전제). |

## packages/media

### 소스

| 파일 | 역할 |
| --- | --- |
| [`audit.py`](../../packages/media/src/dlp_media/audit.py) | 원본 버킷 접근 감사 (WP16): 원본 저장소를 감싸 내려받기·서명 URL·올리기 전에 접근 기록을 남긴다. |
| [`calibration.py`](../../packages/media/src/dlp_media/calibration.py) | 카메라 내부 파라미터 캘리브레이션: 체커보드 영상 → OpenCV calibrateCamera, 또는 파일 가져오기 (ADR 0045·0052, `dlp calib intrinsics\|import`). |
| [`capture_check.py`](../../packages/media/src/dlp_media/capture_check.py) | 촬영 조건 점검: 준비(`dlp media prepare`)·수집(`dlp ingest`) 때 내는 경고. |
| [`encoding.py`](../../packages/media/src/dlp_media/encoding.py) | libx264 yuv420p 재인코딩 공용 도우미 (블러본·워터마크 사본). |
| [`frames.py`](../../packages/media/src/dlp_media/frames.py) | PTS 인덱스로 프레임을 고르고 한 번의 순차 디코드로 꺼낸다 (WP17.2, ADR 0040). |
| [`geometry.py`](../../packages/media/src/dlp_media/geometry.py) | 영상 프레임 기하 변환과 라벨 좌표 변환 (테스트 시점 증강 TTA·학습 증강 공용, ADR 0055·0057). |
| [`glove.py`](../../packages/media/src/dlp_media/glove.py) | 촉각장갑 데이터 수집. Parquet·HDF5를 읽어 정규화된 Parquet(t_ms + 채널 열)으로 바꾼다. |
| [`imu.py`](../../packages/media/src/dlp_media/imu.py) | IMU 추출. 카메라 기종별 추출기를 등록해 쓴다. |
| [`ingest.py`](../../packages/media/src/dlp_media/ingest.py) | 세션 수집 (`dlp ingest`): 매니페스트의 스트림 파일을 검증해 원본 버킷에 올리고 PTS 인덱스·프록시·센서 Parquet을 만든다. |
| [`prepare.py`](../../packages/media/src/dlp_media/prepare.py) | 로컬 영상의 픽셀 방향을 정규화한 H.264 MP4 사본. 원본/기존 출력을 덮어쓰지 않는다. |
| [`probe.py`](../../packages/media/src/dlp_media/probe.py) | 컨테이너 정보 조회 (PyAV, ffprobe와 같은 libav 사용). |
| [`proxy.py`](../../packages/media/src/dlp_media/proxy.py) | 검수 화면용 프록시 영상. 원본 PTS·time_base를 그대로 유지한다. |
| [`pts.py`](../../packages/media/src/dlp_media/pts.py) | PTS 인덱스: 프레임 번호 ↔ 표시 시각(ms). |
| [`rle.py`](../../packages/media/src/dlp_media/rle.py) | COCO 압축 RLE 마스크 부호화·복호화 (`dlp_schema.labels.Rle`, ADR 0059). |
| [`storage.py`](../../packages/media/src/dlp_media/storage.py) | 객체 저장소. 원본(raw)과 라벨링(labeling) 버킷을 별도 인스턴스로 다룬다. |
| [`storage_checks.py`](../../packages/media/src/dlp_media/storage_checks.py) | 라벨링 버킷 노출 경로의 운영 점검 (F-REV-19, ADR 0039 한계). |
| [`tables.py`](../../packages/media/src/dlp_media/tables.py) | Parquet 입출력. pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/media/tests/conftest.py) | 미디어 수집 테스트 공용 픽스처 (WP2·WP3). |
| [`test_audit.py`](../../packages/media/tests/test_audit.py) | 원본 버킷 접근 감사 저장소 단위 테스트 (dlp_media.audit, WP16, ADR 0020). |
| [`test_calibration.py`](../../packages/media/tests/test_calibration.py) | 카메라 내부 파라미터 캘리브레이션 단위 테스트 (ADR 0045, `dlp_media.calibration`). |
| [`test_capture_check.py`](../../packages/media/tests/test_capture_check.py) | 촬영 점검 경고(`dlp_media.capture_check`) 시험. |
| [`test_data_tracks.py`](../../packages/media/tests/test_data_tracks.py) | 데이터 트랙이 있는 영상의 probe 회귀 테스트. |
| [`test_frames.py`](../../packages/media/tests/test_frames.py) | PTS 기준 프레임 고르기·순차 디코드 테스트 (`dlp_media.frames`, WP17.2). |
| [`test_geometry.py`](../../packages/media/tests/test_geometry.py) | 기하 변환(`dlp_media.geometry`) 시험: 픽셀과 라벨 좌표가 같이 움직이는가. |
| [`test_gpmf.py`](../../packages/media/tests/test_gpmf.py) | GoPro GPMF 해석 단위 테스트 (dlp_media.imu, WP3). |
| [`test_ingest.py`](../../packages/media/tests/test_ingest.py) | 세션 수집 테스트 (dlp_media.ingest, WP3, ADR 0003). |
| [`test_orientation.py`](../../packages/media/tests/test_orientation.py) | 스마트폰의 표시 회전 태그를 무시하면 원본·검수 화면의 라벨 좌표가 어긋난다. |
| [`test_prepare.py`](../../packages/media/tests/test_prepare.py) | 정규화 사본은 VFR PTS·표시 방향을 보존하고 실패 시 공개하지 않아야 한다. |
| [`test_pts_proxy.py`](../../packages/media/tests/test_pts_proxy.py) | PTS 인덱스·컨테이너 조회·프록시 테스트 (dlp_media.pts·probe·proxy, WP3). |
| [`test_sensors_storage.py`](../../packages/media/tests/test_sensors_storage.py) | 센서(IMU 사이드카·장갑) 정규화와 로컬 저장소 불변성 테스트 (dlp_media.imu·glove·storage, WP3). |
| [`test_storage_checks.py`](../../packages/media/tests/test_storage_checks.py) | 라벨링 버킷 노출 경로 운영 점검 (F-REV-19, `dlp_media.storage_checks`). |

## packages/models

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/models/src/dlp_models/__init__.py) | 모델 가중치 레지스트리와 공용 런타임. |
| [`datasets.py`](../../packages/models/src/dlp_models/datasets.py) | 공개 데이터셋 목록과 1차 학습 설정 검사 (config/external_datasets.yaml, ADR 0060). |
| [`depth.py`](../../packages/models/src/dlp_models/depth.py) | 단안 메트릭 깊이 (Depth Anything V2 Metric Indoor Small, ONNX Runtime, CPU)와 2D→3D 변환. |
| [`onnx.py`](../../packages/models/src/dlp_models/onnx.py) | ONNX Runtime 세션의 얇은 타입 래퍼 (onnxruntime에는 타입 스텁이 없다). CPU 실행. |
| [`owlv2.py`](../../packages/models/src/dlp_models/owlv2.py) | OWLv2 오픈 보캐뷸러리 탐지 (ONNX Runtime, CPU). 텍스트 질의로 임의의 대상을 찾는다. |
| [`registry.py`](../../packages/models/src/dlp_models/registry.py) | 모델 가중치 레지스트리 (config/models.yaml). |
| [`sam2.py`](../../packages/models/src/dlp_models/sam2.py) | SAM 2.1 박스·점 프롬프트 분할 (ONNX, CPU, ADR 0059). |
| [`vlm.py`](../../packages/models/src/dlp_models/vlm.py) | OpenAI 호환 VLM 서버 공통 전송 (WP17.1, ADR 0040). |
| [`yolox.py`](../../packages/models/src/dlp_models/yolox.py) | YOLOX 고정 클래스 탐지 (ONNX Runtime, CPU). 공개 데이터로 미세조정한 도구·가사 객체 탐지기 (ADR 0060). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_datasets.py`](../../packages/models/tests/test_datasets.py) | 공개 데이터셋 목록·1차 학습 설정 검사 (`dlp_models.datasets`, ADR 0060). |
| [`test_models.py`](../../packages/models/tests/test_models.py) | dlp_models 단위 테스트: 레지스트리(목록·라이선스·해시), OWLv2·메트릭 깊이 전후처리. |
| [`test_vlm.py`](../../packages/models/tests/test_vlm.py) | OpenAI 호환 VLM 공통 전송(`dlp_models.vlm`) 테스트 (WP17.1). |
| [`test_yolox.py`](../../packages/models/tests/test_yolox.py) | YOLOX 전·후처리 단위 테스트 (`dlp_models.yolox`, ADR 0060). 가중치 없이 합성 배열로 정답을 계산한다. |

## packages/ops

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/ops/src/dlp_ops/__init__.py) | 운영 대시보드와 보안 (WP16): 주간 운영 지표, 원본 접근 감사 리포트, 원본 보관 만료 알림. |
| [`audit.py`](../../packages/ops/src/dlp_ops/audit.py) | 원본 접근 감사 월간 리포트 (`dlp ops audit-report`, WP16, ADR 0020·0021). |
| [`metrics.py`](../../packages/ops/src/dlp_ops/metrics.py) | 주간 운영 지표 (`dlp ops weekly`, WP16, ADR 0014·0020·0027·0028·0029). |
| [`policy.py`](../../packages/ops/src/dlp_ops/policy.py) | 운영 정책 (config/policies/ops.yaml) 로더와 타입 (WP16). |
| [`retention.py`](../../packages/ops/src/dlp_ops/retention.py) | 원본 보관 기간 만료 알림 (`dlp ops retention`, WP16, ADR 0020). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_ai_weekly.py`](../../packages/ops/tests/test_ai_weekly.py) | 주간 운영 지표의 AI 1차 검수 절 (`dlp_ops.metrics.ai_weekly`, WP17.8 완료 기준). |
| [`test_ops.py`](../../packages/ops/tests/test_ops.py) | 완료 기준 (WP16): 합성 운영 이벤트로 지표 값이 기대와 일치한다. |

## packages/prelabel

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/prelabel/src/dlp_prelabel/__init__.py) | 자동 프리라벨 (WP8): `dlp prelabel run <세션>`. |
| [`adapters/__init__.py`](../../packages/prelabel/src/dlp_prelabel/adapters/__init__.py) | Predictor 어댑터: 실제 모델(MediaPipe, RTMPose, OWLv2, YOLOX)과 CPU용 stub이 같은 인터페이스를 따른다. |
| [`adapters/mediapipe_models.py`](../../packages/prelabel/src/dlp_prelabel/adapters/mediapipe_models.py) | MediaPipe 실제 모델 어댑터 (CPU): 손 21관절, COCO 객체 탐지. |
| [`adapters/owl_objects.py`](../../packages/prelabel/src/dlp_prelabel/adapters/owl_objects.py) | OWLv2 오픈 보캐뷸러리 객체 탐지 (CPU): COCO에 없는 청소·돌봄 도구 박스 트랙. |
| [`adapters/rtmpose.py`](../../packages/prelabel/src/dlp_prelabel/adapters/rtmpose.py) | RTMPose 전신 포즈 (COCO 17점, CPU): YOLOX-m 사람 탐지(COCO) → RTMPose-m (Body7). |
| [`adapters/stubs.py`](../../packages/prelabel/src/dlp_prelabel/adapters/stubs.py) | CPU용 stub Predictor: 정답을 그대로 돌려주는 Oracle*, 미연동 기능의 빈 결과 Unavailable*, 정해 둔 탐지를 내는 도구 탐지기. |
| [`adapters/yolox_tools.py`](../../packages/prelabel/src/dlp_prelabel/adapters/yolox_tools.py) | 고정 클래스 도구·가사 객체 탐지 (YOLOX, 공개 데이터 미세조정, ADR 0060): 도구 박스 트랙. |
| [`arm.py`](../../packages/prelabel/src/dlp_prelabel/arm.py) | 착용자 팔 골격 arm3 (손목 → 아래팔 → 팔꿈치 → 어깨, ADR 0058). |
| [`campose.py`](../../packages/prelabel/src/dlp_prelabel/campose.py) | 바디캠 6자유도 자세 추정 (ADR 0046, F-PAR-26). |
| [`common.py`](../../packages/prelabel/src/dlp_prelabel/common.py) | 어댑터 공통: 프레임 순회, 박스 IoU 추적, 모델 출처 라벨 만들기. |
| [`constraints3d.py`](../../packages/prelabel/src/dlp_prelabel/constraints3d.py) | 손 3D 물리 제약: 팔 길이 밖 거리, 속도·가속도 튐, 세션 손끝 거리와 다른 표본 (ADR 0054). |
| [`contact.py`](../../packages/prelabel/src/dlp_prelabel/contact.py) | 접촉 구간 추정 (모델이 아니라 알고리즘). |
| [`fov.py`](../../packages/prelabel/src/dlp_prelabel/fov.py) | 손 크기로 세션 초점 거리(화각) 추정 (ADR 0056, `depth.focal_estimate`). |
| [`handfit.py`](../../packages/prelabel/src/dlp_prelabel/handfit.py) | 손 중심 3D 점·2D 픽셀·메트릭 깊이로 손 21관절의 카메라 좌표 3D를 정한다 (ADR 0044). |
| [`lift3d.py`](../../packages/prelabel/src/dlp_prelabel/lift3d.py) | 단안 메트릭 깊이로 바디캠의 손 관절·객체를 카메라 좌표 3D 궤적으로 올린다. |
| [`masks.py`](../../packages/prelabel/src/dlp_prelabel/masks.py) | 객체·손 모양 마스크 단계 (ADR 0059). |
| [`palm.py`](../../packages/prelabel/src/dlp_prelabel/palm.py) | 작업자 손 크기(손목 → 중지 MCP 길이) 추정 (ADR 0045·0052, `dlp calib palm-estimate`). |
| [`policy.py`](../../packages/prelabel/src/dlp_prelabel/policy.py) | 프리라벨 정책 (config/policies/prelabel.yaml). |
| [`posemath.py`](../../packages/prelabel/src/dlp_prelabel/posemath.py) | 회전·자세 계산 도우미 (카메라 자세 추정과 월드 좌표 변환, ADR 0046). |
| [`runner.py`](../../packages/prelabel/src/dlp_prelabel/runner.py) | 세션 하나의 프리라벨 실행 (`dlp prelabel run <세션>`). |
| [`scene.py`](../../packages/prelabel/src/dlp_prelabel/scene.py) | 장면 분류와 장면별 OWLv2 질의 목록 (ADR 0055). |
| [`temporal.py`](../../packages/prelabel/src/dlp_prelabel/temporal.py) | 시간 일관성: 박스 트랙 거르기(히스테리시스·최소 길이), 손 키포인트 평활(One Euro)·짧은 끊김 채우기. |
| [`wearer.py`](../../packages/prelabel/src/dlp_prelabel/wearer.py) | 3인칭 영상 속 바디캠 착용자 찾기 (알고리즘). |
| [`wearer_suppress.py`](../../packages/prelabel/src/dlp_prelabel/wearer_suppress.py) | 착용자 자신의 몸 탐지를 라벨로 만들지 않는다 (ADR 0058). |
| [`worldframe.py`](../../packages/prelabel/src/dlp_prelabel/worldframe.py) | 카메라 좌표 3D 궤적 → 세션 월드 좌표 궤적 (ADR 0046, F-PAR-26). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_adapters.py`](../../packages/prelabel/tests/test_adapters.py) | 프리라벨 어댑터 테스트: 손 판정 뒤집기, 손별 최고 점수, 정책·레지스트리·온톨로지 일치, stub, 가중치 없음 보고, 실제 모델 CPU 실행(가중치·libEGL이 있을 때만). |
| [`test_algorithms.py`](../../packages/prelabel/tests/test_algorithms.py) | 접촉·착용자 알고리즘 단위 테스트 (DB 없음). |
| [`test_arm.py`](../../packages/prelabel/tests/test_arm.py) | 착용자 팔 골격 arm3와 착용자 몸 억제 (ADR 0058). |
| [`test_campose.py`](../../packages/prelabel/tests/test_campose.py) | 바디캠 자세 추정과 월드 좌표 변환 (ADR 0046, F-PAR-26). 서비스 없이 돈다. |
| [`test_fov.py`](../../packages/prelabel/tests/test_fov.py) | 손 크기 초점 거리 추정(`dlp_prelabel.fov`)과 3D 단계 연동 시험 (ADR 0056). |
| [`test_handfit.py`](../../packages/prelabel/tests/test_handfit.py) | 손가락 끝 3D 추정(`dlp_prelabel.handfit`, `lift3d.lift_tracks`의 손 맞춤) 단위 테스트 (ADR 0044). |
| [`test_handfit_quality.py`](../../packages/prelabel/tests/test_handfit_quality.py) | 3D 표본 방식·신뢰도, 작업자 손 크기 사전값과 그 추정 단위 테스트 (ADR 0045). |
| [`test_lift3d.py`](../../packages/prelabel/tests/test_lift3d.py) | `dlp_prelabel.lift3d` 단위 테스트: 깊이 역투영, 추론 간격, 내부 파라미터 비례 조정, 같은 개체 트랙이 여럿일 때의 입력 고르기(F-PAR-01). |
| [`test_long_label_ids.py`](../../packages/prelabel/tests/test_long_label_ids.py) | F-PAR-86 회귀: 2D 단계 ID가 128자 안인 긴 세션에서 그 트랙에서 만든 라벨 ID도 128자 안이다. |
| [`test_mask_prompt_rank.py`](../../packages/prelabel/tests/test_mask_prompt_rank.py) | 마스크 프롬프트 순위: 신뢰도 0 트랙을 1.0으로 올려 세우지 않는다 (F-PAR-91). |
| [`test_masks.py`](../../packages/prelabel/tests/test_masks.py) | 객체·손 모양 마스크 단계 (ADR 0059): 정답 모양 합성 영상(`dlp_fixtures.shapes`)으로 본다. |
| [`test_runner.py`](../../packages/prelabel/tests/test_runner.py) | 프리라벨 실행기 통합 테스트 (`@pytest.mark.services`: PostgreSQL·SeaweedFS S3 필요, `make up`). |
| [`test_runner_lift_lineage.py`](../../packages/prelabel/tests/test_runner_lift_lineage.py) | 3D 보호는 출처 2D 계보로 정한다 (F-PAR-92, F-PAR-87의 남은 부분). |
| [`test_runner_rerun.py`](../../packages/prelabel/tests/test_runner_rerun.py) | 프리라벨 러너의 다시 돌리기 규칙 (SQLite + 로컬 저장소, 서비스 없이 돈다). |
| [`test_runner_units.py`](../../packages/prelabel/tests/test_runner_units.py) | 프리라벨 실행기의 DB 없는 단위 테스트 (감사 4차 회귀). |
| [`test_scene.py`](../../packages/prelabel/tests/test_scene.py) | 장면 분류와 장면별 질의(`dlp_prelabel.scene`) 시험 (ADR 0055). |
| [`test_temporal.py`](../../packages/prelabel/tests/test_temporal.py) | 시간 일관성(`dlp_prelabel.temporal`)과 손 3D 제약(`dlp_prelabel.constraints3d`) 시험 (ADR 0054). |
| [`test_yolox_tools.py`](../../packages/prelabel/tests/test_yolox_tools.py) | 고정 클래스 도구 단계(`adapters.yolox_tools`, ADR 0060) 테스트. 가중치 없이 CPU stub 탐지기로 돈다. |

## packages/privacy

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/privacy/src/dlp_privacy/__init__.py) | 프라이버시 게이트 (WP5, ADR 0005·0019·0023·0024). |
| [`audit.py`](../../packages/privacy/src/dlp_privacy/audit.py) | 잔여 누락 감사와 전수 검수 종료 판정. |
| [`detection.py`](../../packages/privacy/src/dlp_privacy/detection.py) | 프레임 단위 탐지 결과와 탐지기 인터페이스. |
| [`detectors/__init__.py`](../../packages/privacy/src/dlp_privacy/detectors/__init__.py) | 탐지기 구현과 정책에서 탐지기를 만드는 공장. |
| [`detectors/codes.py`](../../packages/privacy/src/dlp_privacy/detectors/codes.py) | QR 코드·바코드 탐지 (OpenCV, 가중치 없음). 송장·라벨이 있다는 근거로 shipping_label을 낸다. |
| [`detectors/open_vocab.py`](../../packages/privacy/src/dlp_privacy/detectors/open_vocab.py) | 오픈 보캐뷸러리 탐지 (OWLv2): 문서·화면·사진물·문패·송장과 반사면 영역. |
| [`detectors/oracle.py`](../../packages/privacy/src/dlp_privacy/detectors/oracle.py) | 정답 블러 트랙을 그대로 돌려주는 stub 탐지기 (테스트·CI용). |
| [`detectors/reflection.py`](../../packages/privacy/src/dlp_privacy/detectors/reflection.py) | 반사면 속 얼굴: 반사면 영역(거울, 유리, 꺼진 화면)을 찾고 그 안에서 얼굴 탐지 문턱을 낮춘다. |
| [`detectors/yunet.py`](../../packages/privacy/src/dlp_privacy/detectors/yunet.py) | YuNet 얼굴 탐지 (OpenCV FaceDetectorYN, CPU). |
| [`errors.py`](../../packages/privacy/src/dlp_privacy/errors.py) | 프라이버시 게이트·렌더 검증의 공용 예외. |
| [`generation.py`](../../packages/privacy/src/dlp_privacy/generation.py) | 프라이버시 운영 이력과 검수 세대 판정 (WP5·WP12, ADR 0033). |
| [`geometry.py`](../../packages/privacy/src/dlp_privacy/geometry.py) | 박스 연산. 박스는 (x, y, w, h) 픽셀. |
| [`hints.py`](../../packages/privacy/src/dlp_privacy/hints.py) | 블러 검수 힌트: 오탐 의심 표시 (ADR 0055). 블러 트랙을 지우거나 바꾸지 않는다. |
| [`hold.py`](../../packages/privacy/src/dlp_privacy/hold.py) | 사고 대응용 프라이버시 승인 보류·해제 (WP5, ADR 0039·0042, F-REV-18). |
| [`pipeline.py`](../../packages/privacy/src/dlp_privacy/pipeline.py) | 한 영상 스트림의 프라이버시 프리라벨: 프레임 순회 → 탐지·유지 → 트랙 → 라벨 + 검수 우선 구간. |
| [`policy.py`](../../packages/privacy/src/dlp_privacy/policy.py) | 프라이버시 정책 (config/policies/privacy.yaml + config/defaults.yaml privacy). |
| [`render.py`](../../packages/privacy/src/dlp_privacy/render.py) | 블러 렌더. 원본 PTS를 그대로 두고 오디오는 넣지 않는다. |
| [`rendering.py`](../../packages/privacy/src/dlp_privacy/rendering.py) | 승인된 블러본 렌더와 소비자 검증 (렌더 게이트, WP5, ADR 0024·0036). |
| [`review.py`](../../packages/privacy/src/dlp_privacy/review.py) | 검수 우선 구간. 블러 검수 화면은 누락 찾기에 맞춰 이 구간을 먼저 보여준다. |
| [`revocation.py`](../../packages/privacy/src/dlp_privacy/revocation.py) | 블러가 바뀐 뒤 낡은 라벨링 작업 매체의 회수 요청 (WP5·WP6, ADR 0039). |
| [`runner.py`](../../packages/privacy/src/dlp_privacy/runner.py) | DB에 등록된 세션의 프라이버시 게이트 실행: 탐지 → (사람 검수) → 승인 → 블러본 렌더. |
| [`tracker.py`](../../packages/privacy/src/dlp_privacy/tracker.py) | 탐지를 트랙으로 묶고, 끊김을 보간하고, 소실 후에도 블러를 유지한다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/privacy/tests/conftest.py) | 프라이버시 패키지 테스트 공용 픽스처 (WP2·WP5). |
| [`test_audit.py`](../../packages/privacy/tests/test_audit.py) | 잔여 누락 감사·전수 검수 종료 판정 단위 테스트 (dlp_privacy.audit, WP5·WP16). |
| [`test_gate.py`](../../packages/privacy/tests/test_gate.py) | 프라이버시 탐지·트래커·검수 구간·렌더 단위 테스트 (DB·저장소 없이, WP5). |
| [`test_hold.py`](../../packages/privacy/tests/test_hold.py) | 사고 대응용 프라이버시 승인 보류·해제 (F-REV-18, ADR 0042). |
| [`test_render_recovery.py`](../../packages/privacy/tests/test_render_recovery.py) | 손상된 블러본/렌더 기록은 소비자가 거부하고 재렌더로 복구할 수 있어야 한다. |
| [`test_revocation.py`](../../packages/privacy/tests/test_revocation.py) | 블러가 바뀐 스트림의 배포 매체 회수 요청 (ADR 0039). |
| [`test_runner.py`](../../packages/privacy/tests/test_runner.py) | 프라이버시 게이트 통합 테스트: 수집 → 탐지 → 검수 → 승인 → 렌더 → 렌더 확인 (WP5). |
| [`test_tracker_cover.py`](../../packages/privacy/tests/test_tracker_cover.py) | 트래커 덮기 불변식: 블러 박스가 매 프레임의 모든 탐지(관측·간격 캐시)를 덮는다 (F-PMS-49). |

## packages/relations

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/relations/src/dlp_relations/__init__.py) | 관계 도출(YAML 규칙 엔진)과 도구-표면 접촉·표면 커버리지 (WP9). |
| [`contact.py`](../../packages/relations/src/dlp_relations/contact.py) | 도구 작용부-표면 접촉 구간 (평면까지 거리, 히스테리시스). |
| [`coverage.py`](../../packages/relations/src/dlp_relations/coverage.py) | 표면 커버리지: 접촉 중 작용부 경로를 반지름 footprint 원으로 쓸어 덮은 면적 비율. |
| [`derive.py`](../../packages/relations/src/dlp_relations/derive.py) | 세션 라벨에서 관계·커버리지를 도출한다 (DB와 무관한 순수 계산). |
| [`geometry.py`](../../packages/relations/src/dlp_relations/geometry.py) | 표면 평면과 표면 좌표 (순수 수치 계산, DB·정책 없음). |
| [`policy.py`](../../packages/relations/src/dlp_relations/policy.py) | 관계·커버리지 정책 (config/policies/relations.yaml). |
| [`rules.py`](../../packages/relations/src/dlp_relations/rules.py) | 규칙 엔진: 입력 구간(손 상태, 도구-표면 접촉)에 YAML 규칙을 적용해 관계 구간을 만든다. |
| [`runner.py`](../../packages/relations/src/dlp_relations/runner.py) | 세션 관계·커버리지 도출 실행 (`dlp relations run`). 멱등이다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_relations.py`](../../packages/relations/tests/test_relations.py) | 관계·커버리지 순수 계산 테스트 (DB 없음). |
| [`test_runner.py`](../../packages/relations/tests/test_runner.py) | 관계 실행기 통합 테스트 (`@pytest.mark.services`: PostgreSQL 필요, `make up`). |

## packages/retarget

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/retarget/src/dlp_retarget/__init__.py) | 사람 손 3D → 로봇 관절 궤적 (리타기팅, ADR 0047·0068). |
| [`hand_ik.py`](../../packages/retarget/src/dlp_retarget/hand_ik.py) | 다지 손 역기구학: 손목·손가락 끝 위치를 팔 관절과 손가락 관절로 함께 푼다 (ADR 0068). |
| [`ik.py`](../../packages/retarget/src/dlp_retarget/ik.py) | 역기구학 풀이기 인터페이스와 내장 감쇠 최소제곱 풀이기 (ADR 0047·0053). |
| [`mink.py`](../../packages/retarget/src/dlp_retarget/mink.py) | mink(MuJoCo 기반 미분 IK, Apache-2.0) 풀이기 어댑터 (선택, 격리 환경, ADR 0047). |
| [`policy.py`](../../packages/retarget/src/dlp_retarget/policy.py) | 리타기팅 정책 (`config/policies/retarget.yaml`, ADR 0047). |
| [`retarget.py`](../../packages/retarget/src/dlp_retarget/retarget.py) | 에피소드 하나를 로봇 관절 궤적으로 옮기고 도달 가능성 보고를 만든다 (ADR 0047·0068). |
| [`robot.py`](../../packages/retarget/src/dlp_retarget/robot.py) | 로봇 등록: `config/robots/<ID>.yaml` 로더, 모델 파일 sha256 확인 (ADR 0047·0068). |
| [`targets.py`](../../packages/retarget/src/dlp_retarget/targets.py) | 사람 손 점 → 로봇 목표: 끝단·그리퍼 열림(평행 그리퍼), 손목·손가락 끝(다지 손) (ADR 0047·0068). |
| [`urdf.py`](../../packages/retarget/src/dlp_retarget/urdf.py) | URDF 직렬 연쇄·나무 구조 기구학 (정기구학·야코비안, ADR 0047·0068). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_hand.py`](../../packages/retarget/tests/test_hand.py) | 다지 손 리타기팅 (ADR 0068): 나무 구조 기구학, 다지 손 로봇 등록 검사, 손목·손가락 끝 IK 정확도, 닿지 못한 손가락과 끝 점이 없는 손가락의 표시, 속도 한계. |
| [`test_mink.py`](../../packages/retarget/tests/test_mink.py) | mink(MuJoCo) 격리 환경 풀이기 (ADR 0047, `@pytest.mark.isolated_env`, `make test-isolated`). |
| [`test_retarget.py`](../../packages/retarget/tests/test_retarget.py) | 로봇 리타기팅 (ADR 0047): URDF 기구학, 로봇 등록 검사, IK 정확도, 도달하지 못한 구간 표시. |

## packages/review

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/review/src/dlp_review/__init__.py) | 검수 도구 연동 (WP6 검수 도구, WP12 검수 운영). |
| [`ai_calibration.py`](../../packages/review/src/dlp_review/ai_calibration.py) | AI 1차 검수 판정의 표시 보정 산출물 (ADR 0062, ADR 0040 6절 보강). |
| [`ai_collect.py`](../../packages/review/src/dlp_review/ai_collect.py) | AI 1차 검수 제안의 사람 적용과 수집 결과 기록 (WP17.7, ADR 0040 6.6, I6). |
| [`ai_display.py`](../../packages/review/src/dlp_review/ai_display.py) | AI 1차 검수 판정을 사람 검수에 얹는다: 노출 집단·표시·우선순위·표본 제외·필수 게이트 (WP17.6). |
| [`clients.py`](../../packages/review/src/dlp_review/clients.py) | CVAT·Label Studio REST 클라이언트 (필요한 호출만). |
| [`collect.py`](../../packages/review/src/dlp_review/collect.py) | 검수 결과 수집 (검수 생애주기의 수집 단계, `dlp_review` 패키지 docstring). |
| [`cvat.py`](../../packages/review/src/dlp_review/cvat.py) | LabelRecord ↔ CVAT 비디오 트랙. |
| [`labelstudio.py`](../../packages/review/src/dlp_review/labelstudio.py) | LabelRecord ↔ Label Studio 시계열 구간 라벨, 그리고 프로젝트 설정 XML. |
| [`object_hints.py`](../../packages/review/src/dlp_review/object_hints.py) | 객체·도구 오탐 의심 힌트 (ADR 0055). 라벨을 바꾸지 않고 검수 화면에 표시만 한다. |
| [`ontology_projects.py`](../../packages/review/src/dlp_review/ontology_projects.py) | 검수 화면 사전을 온톨로지 내용별로 격리한다. 기존 프로젝트를 덮어쓰지 않는다. |
| [`ops/__init__.py`](../../packages/review/src/dlp_review/ops/__init__.py) | 검수 운영 로직 (WP12): 우선순위 큐, 표본 검수, 블라인드·오류 삽입·이중·QA 배정, 측정. |
| [`ops/ai_calibration.py`](../../packages/review/src/dlp_review/ops/ai_calibration.py) | AI 판정 표시 보정 학습과 효과 측정 (`dlp aireview calibrate`, ADR 0062). |
| [`ops/ai_quality.py`](../../packages/review/src/dlp_review/ops/ai_quality.py) | AI 1차 검수 판정의 품질 측정 (WP17.8, ADR 0040 7.1~7.3). |
| [`ops/assign.py`](../../packages/review/src/dlp_review/ops/assign.py) | 배정 계획: 검수 단위마다 표준 배정을 두고 정책 비율대로 블라인드·이중·오류 삽입·QA 배정을 더한다 (DB를 쓰지 않는다). |
| [`ops/measure.py`](../../packages/review/src/dlp_review/ops/measure.py) | 검수 품질 측정: 이중 라벨링 일치도, 블라인드 과제 대비 프리라벨 편향, 오류 삽입 발견율. |
| [`ops/policy.py`](../../packages/review/src/dlp_review/ops/policy.py) | 검수 운영 정책 (config/policies/review.yaml + config/defaults.yaml review 비율). |
| [`ops/priority.py`](../../packages/review/src/dlp_review/ops/priority.py) | 우선순위 큐: 검수 단위 안에서 먼저 볼 구간을 찾고 단위 우선순위를 매긴다. |
| [`ops/results.py`](../../packages/review/src/dlp_review/ops/results.py) | 수집 후 배정 마무리·표본 판정·검수 품질 보고 (WP12, ADR 0014). |
| [`ops/runner.py`](../../packages/review/src/dlp_review/ops/runner.py) | 검수 운영 실행: 세션 배정 계획, 배정별 검수 작업 만들기, 배정 마무리(표본 판정), 품질 리포트. |
| [`ops/sampling.py`](../../packages/review/src/dlp_review/ops/sampling.py) | 높은 신뢰도 라벨의 표본 검수 (합격 판정 샘플링). |
| [`ops/seeding.py`](../../packages/review/src/dlp_review/ops/seeding.py) | 오류 삽입 과제: 정답을 아는 단위의 라벨 사본에 오류를 넣고, 검수 뒤 발견 여부를 판정한다. |
| [`ops/selection.py`](../../packages/review/src/dlp_review/ops/selection.py) | 배정 방식에 따라 검수 작업에 넣을 라벨을 고른다 (WP12, ADR 0014). |
| [`orphans.py`](../../packages/review/src/dlp_review/orphans.py) | DB 기록 없이 남은 검수 도구 작업·라벨링 버킷 사본(고아)을 찾고 지운다 (F-REV-15, R4, ADR 0039). |
| [`reconcile.py`](../../packages/review/src/dlp_review/reconcile.py) | 검수 결과를 라벨 이력으로 바꾼다 (검수 생애주기의 대조 단계, `dlp_review` 패키지 docstring). |
| [`refresh.py`](../../packages/review/src/dlp_review/refresh.py) | 열린 Label Studio 작업의 서명 URL을 다시 서명한다 (F-REV-08). |
| [`revoke.py`](../../packages/review/src/dlp_review/revoke.py) | 회수 요청된 라벨링 작업 매체를 실제로 끊는다: 라벨링 버킷 사본 삭제 → 도구 작업 삭제 (ADR 0039). |
| [`roles.py`](../../packages/review/src/dlp_review/roles.py) | 검수 단계별 접근 경계: 원본은 프라이버시 검수자만 보고, 일반 라벨러 경로에 원본 버킷 URI가 섞이면 오류를 낸다. |
| [`tasks.py`](../../packages/review/src/dlp_review/tasks.py) | 세션에서 검수 작업을 만든다 (검수 생애주기의 만들기 단계, `dlp_review` 패키지 docstring). |
| [`timeseries.py`](../../packages/review/src/dlp_review/timeseries.py) | 검수 화면용 시계열 CSV (WP6, ADR 0006). |
| [`ui_theme.py`](../../packages/review/src/dlp_review/ui_theme.py) | 검수 화면 테마 (`config/review_ui.yaml`). CVAT·Label Studio 위에 덧씌우는 모양만 정한다. |
| [`verify.py`](../../packages/review/src/dlp_review/verify.py) | 세션 검수 완료 판정 (`dlp review verify`, 검수 생애주기의 마지막 단계). |
| [`watermark.py`](../../packages/review/src/dlp_review/watermark.py) | 라벨러 ID 워터마크. 화면 캡처가 유출되면 누가 본 영상인지 알 수 있게 한다. |
| [`webhook.py`](../../packages/review/src/dlp_review/webhook.py) | 검수 도구 웹훅. 작업이 끝나면 수집을 일으킨다. |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/review/tests/conftest.py) | 검수 패키지 테스트 공용 픽스처·도우미. |
| [`test_ai_calibration.py`](../../packages/review/tests/test_ai_calibration.py) | AI 판정 표시 보정 시험 (ADR 0062, `dlp_review.ops.ai_calibration`·`dlp_review.ai_calibration`). |
| [`test_ai_collect.py`](../../packages/review/tests/test_ai_collect.py) | AI 1차 검수 제안의 사람 적용과 수집 결과 기록 (`ai_collect`, WP17.7, ADR 0040 6.6). |
| [`test_ai_disagreement.py`](../../packages/review/tests/test_ai_disagreement.py) | 판정단 불일치 테스트 (`ai_review.disagreement`, `dlp_review.ai_display`, ADR 0063). |
| [`test_ai_display.py`](../../packages/review/tests/test_ai_display.py) | AI 1차 검수 판정의 검수 화면 표시·노출 집단·우선순위·표본 제외·마무리 보호 (WP17.6, ADR 0040). |
| [`test_ai_display_flow.py`](../../packages/review/tests/test_ai_display_flow.py) | AI 1차 검수 표시 종단 테스트 (`@pytest.mark.services`, WP17.6, ADR 0040). |
| [`test_ai_quality.py`](../../packages/review/tests/test_ai_quality.py) | AI 1차 검수 판정 품질 지표 (`dlp_review.ops.ai_quality`, WP17.8 완료 기준: 손 계산 일치). |
| [`test_clients.py`](../../packages/review/tests/test_clients.py) | 검수 도구 클라이언트의 회수용 메서드 (ADR 0039). 실제 도구 대신 httpx.MockTransport로 답한다. |
| [`test_cvat_masks.py`](../../packages/review/tests/test_cvat_masks.py) | CVAT 마스크 왕복 (mask_track, ADR 0059). 서비스 없이 돈다. |
| [`test_description_follow.py`](../../packages/review/tests/test_description_follow.py) | 설명 라벨이 행동 검수 결과를 따른다 (F-PAR-15, ADR 0043 3절 "행동 승인할 때 같이"). |
| [`test_flow.py`](../../packages/review/tests/test_flow.py) | 검수 흐름 통합 테스트 (`@pytest.mark.services`, WP6, ADR 0020·0023·0024). |
| [`test_long_ids.py`](../../packages/review/tests/test_long_ids.py) | 긴 세션 ID에서도 파생 배정·라벨 ID가 128자 안에 든다 (F-REV-36). |
| [`test_ls_accounts.py`](../../packages/review/tests/test_ls_accounts.py) | Label Studio 결과는 작업 담당자의 Label Studio 계정이 낸 주석만 받는다 (F-REV-06, A-10). |
| [`test_ls_ai_feasibility.py`](../../packages/review/tests/test_ls_ai_feasibility.py) | Label Studio 1.23.2 AI 1차 검수 표시 실현 가능성 (`@pytest.mark.services`, WP17.6, ADR 0040). |
| [`test_ops.py`](../../packages/review/tests/test_ops.py) | 검수 운영 로직 (WP12) 단위 테스트. |
| [`test_ops_flow.py`](../../packages/review/tests/test_ops_flow.py) | 검수 운영 흐름 (Label Studio): 계획 → 블라인드·오류 삽입 작업 → 검수 → 수집 → 품질 리포트. |
| [`test_orphans.py`](../../packages/review/tests/test_orphans.py) | DB 기록 없이 남은 검수 도구 작업·라벨링 버킷 사본(고아) 찾기·지우기 (F-REV-15, R4). |
| [`test_refresh.py`](../../packages/review/tests/test_refresh.py) | 열린 Label Studio 작업의 서명 URL 갱신 (F-REV-08). |
| [`test_revoke.py`](../../packages/review/tests/test_revoke.py) | 블러 변경 뒤 이미 배포한 라벨링 작업 매체 회수 (F-REV-02, ADR 0039). |
| [`test_timeline_gaps.py`](../../packages/review/tests/test_timeline_gaps.py) | 손별 행동 타임라인 공백 검사 (F-REV-37, ADR 0012 "타임라인 공백 0"). |
| [`test_trajectory_follow_flow.py`](../../packages/review/tests/test_trajectory_follow_flow.py) | 3D 궤적이 수집·완료 판정에서 2D 트랙 검수를 따른다 (F-COE-19·21·22, ADR 0048·0050). |
| [`test_ui_theme.py`](../../packages/review/tests/test_ui_theme.py) | 검수 화면 테마 단위 테스트 (ADR 0038). 서비스 없이 돈다. |
| [`test_units.py`](../../packages/review/tests/test_units.py) | 검수 연동 단위 테스트 (WP6, ADR 0006·0024). 서비스(DB·도구) 없이 돈다. |
| [`test_verify.py`](../../packages/review/tests/test_verify.py) | 검수 완료 판정(dlp review verify) 시험: 남은 일이 있으면 거부하고, 끝나면 전이·시각을 남긴다. |
| [`test_webhook_http.py`](../../packages/review/tests/test_webhook_http.py) | 실제 HTTP 경계: 잘못된 입력·느린 연결·과부하·수집 실패가 성공으로 보이지 않아야 한다. |

## packages/scene3d

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/scene3d/src/dlp_scene3d/__init__.py) | 청소 전후 다각도 사진의 3D 복원·물체 이동 비교와 장면 안 바디캠 위치 추정 (ADR 0066, 0067). |
| [`alignment.py`](../../packages/scene3d/src/dlp_scene3d/alignment.py) | after 장면 → before 장면 정렬 (고정 구조 점의 닮음 변환, 척도 포함). |
| [`backends.py`](../../packages/scene3d/src/dlp_scene3d/backends.py) | 복원·탐지·프레임 특징 백엔드 어댑터와 CPU stub (ADR 0066, 0067). |
| [`change.py`](../../packages/scene3d/src/dlp_scene3d/change.py) | 전후 물체 대응(헝가리안)과 변화 분류. |
| [`ego_video.py`](../../packages/scene3d/src/dlp_scene3d/ego_video.py) | 바디캠 블러본 경계: 장면 안 위치 추정이 영상 프레임을 얻는 유일한 길 (ADR 0067). |
| [`geometry.py`](../../packages/scene3d/src/dlp_scene3d/geometry.py) | 순수 기하: 핀홀 카메라, 다시점 삼각측량, 닮음 변환(Umeyama)과 RANSAC, 기술자 대응. |
| [`layout.py`](../../packages/scene3d/src/dlp_scene3d/layout.py) | 저장소 키 배치와 JSON 읽고 쓰기 (ADR 0066, 0067). |
| [`localize.py`](../../packages/scene3d/src/dlp_scene3d/localize.py) | 장면 안 바디캠 위치 추정과 위치 기록 읽기 (`dlp scene3d localize`, ADR 0067). |
| [`markers.py`](../../packages/scene3d/src/dlp_scene3d/markers.py) | 크기를 아는 ArUco 마커로 복원의 미터 척도 정하기 (ADR 0067). |
| [`objects.py`](../../packages/scene3d/src/dlp_scene3d/objects.py) | 사진별 2D 물체 탐지 → 다시점 3D 물체 (삼각측량과 시점 간 연결). |
| [`oracle.py`](../../packages/scene3d/src/dlp_scene3d/oracle.py) | 합성 픽스처 정답 파일(truth.json, `dlp_fixtures.scene3d.write_scene_pair`)에서 stub을 만든다. |
| [`photos.py`](../../packages/scene3d/src/dlp_scene3d/photos.py) | 사진 수집: 원본은 불변으로 원본 버킷에 두고, EXIF 방향은 읽을 때 적용한다 (ADR 0066). |
| [`pipeline.py`](../../packages/scene3d/src/dlp_scene3d/pipeline.py) | 복원과 전후 비교 실행 (`dlp scene3d reconstruct\|compare\|report`, ADR 0066·0067). |
| [`pnp.py`](../../packages/scene3d/src/dlp_scene3d/pnp.py) | 2D-3D 대응에서 카메라 자세 풀기 (PnP RANSAC과 내점 다듬기, ADR 0067). |
| [`policy.py`](../../packages/scene3d/src/dlp_scene3d/policy.py) | 청소 전후 3D 비교와 장면 안 바디캠 위치 추정 정책 (config/policies/scene3d.yaml). |
| [`privacy_gate.py`](../../packages/scene3d/src/dlp_scene3d/privacy_gate.py) | 사진 프라이버시 게이트: 탐지 → 사람 승인 → 블러 렌더 → 하류의 블러본 확인 (ADR 0066). |
| [`scipy_typed.py`](../../packages/scene3d/src/dlp_scene3d/scipy_typed.py) | scipy 호출(헝가리안 할당, k-최근접)의 타입 래퍼. scipy에는 타입 스텁이 없다. |
| [`status.py`](../../packages/scene3d/src/dlp_scene3d/status.py) | 세션의 전후 3D 공간이 다 갖춰졌는지 보기 (`dlp scene3d status`, ADR 0067). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_ego_geometry.py`](../../packages/scene3d/tests/test_ego_geometry.py) | 위치 추정의 순수 계산 시험: 기술자 대응기, 닮음 변환을 거친 자세, PnP, 내부 파라미터 맞춤, 보간. |
| [`test_geometry.py`](../../packages/scene3d/tests/test_geometry.py) | 순수 기하·대응·분류 시험 (합성 정답, ADR 0066). |
| [`test_localize.py`](../../packages/scene3d/tests/test_localize.py) | 장면 안 바디캠 위치 추정 종단 시험 (로컬 저장소, SQLite, CPU stub, ADR 0067). |
| [`test_marker_scale.py`](../../packages/scene3d/tests/test_marker_scale.py) | 마커로 미터 척도 정하기 시험 (합성 방 사진에 원근에 맞게 그린 ArUco 마커, ADR 0067). |
| [`test_pipeline.py`](../../packages/scene3d/tests/test_pipeline.py) | 수집 → 블러 승인 → 복원 → 전후 비교 종단 시험 (로컬 저장소, CPU stub, ADR 0066). |
| [`test_policy_records.py`](../../packages/scene3d/tests/test_policy_records.py) | 정책·라이선스·기록 표·격리 환경 어댑터 시험 (ADR 0066, 0067). |
| [`test_status.py`](../../packages/scene3d/tests/test_status.py) | `dlp scene3d status`의 상태 읽기 시험: 전후가 다 갖춰졌는지, 읽기 전용인지 (ADR 0067). |
| [`world.py`](../../packages/scene3d/tests/world.py) | scene3d 시험 공용 준비: 로컬 저장소의 전후 사진 게이트·복원과, DB에 넣은 바디캠 세션·블러본. |

## packages/schema

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/schema/src/dlp_schema/__init__.py) | 세션·스트림·라벨·에피소드 그래프 계약 타입과 온톨로지. |
| [`aireview.py`](../../packages/schema/src/dlp_schema/aireview.py) | AI 1차 검수(로컬 VLM) 계약: 판정·제안·요청 단위·실행·노출·결과 (WP17, ADR 0040). |
| [`calibration.py`](../../packages/schema/src/dlp_schema/calibration.py) | 카메라 내부 파라미터 등록과 작업자 손 크기 사전값 (ADR 0045, 0052). |
| [`common.py`](../../packages/schema/src/dlp_schema/common.py) | 공통 타입: 시각(ms)·신뢰도·식별자·온톨로지 ID·버전 문자열과 모든 계약 타입의 기반 클래스. |
| [`config.py`](../../packages/schema/src/dlp_schema/config.py) | config/defaults.yaml 로더. 코드는 정책 값을 하드코딩하지 않고 이 설정을 읽는다. |
| [`dataset.py`](../../packages/schema/src/dlp_schema/dataset.py) | 데이터셋 버전과 분할의 계약 타입 (라벨 스냅샷 하나 + 세션별 분할 배정). |
| [`db/__init__.py`](../../packages/schema/src/dlp_schema/db/__init__.py) | DB 계층: Core 테이블(`tables`), 계약 ↔ 행 저장소(`repository`), Alembic 실행(`migrate`). |
| [`db/aireview_repository.py`](../../packages/schema/src/dlp_schema/db/aireview_repository.py) | AI 1차 검수 기록 저장소 (WP17.4, ADR 0040). 연결·트랜잭션·커밋은 호출자가 관리한다. |
| [`db/calibration_repository.py`](../../packages/schema/src/dlp_schema/db/calibration_repository.py) | 카메라 내부 파라미터·작업자 손 크기 등록 저장·조회 (ADR 0045, 0052). |
| [`db/description_repository.py`](../../packages/schema/src/dlp_schema/db/description_repository.py) | 설명 라벨이 행동 검수 결과를 따르게 DB에 적용한다 (F-PAR-15, ADR 0043 3절). |
| [`db/inheritance_repository.py`](../../packages/schema/src/dlp_schema/db/inheritance_repository.py) | 검수 물려받기 기록 저장·조회 (추가만). |
| [`db/locks.py`](../../packages/schema/src/dlp_schema/db/locks.py) | 세션·모델 배포·AI 판정 실행 잠금. 연결과 커밋은 호출자가 관리한다. |
| [`db/migrate.py`](../../packages/schema/src/dlp_schema/db/migrate.py) | Alembic 마이그레이션 실행. alembic.ini 없이 코드에서 설정한다. |
| [`db/ontology_repository.py`](../../packages/schema/src/dlp_schema/db/ontology_repository.py) | DB에 등록된 사전을 세션·데이터셋 버전에 맞춰 읽는 공통 경계. |
| [`db/prelabel_repository.py`](../../packages/schema/src/dlp_schema/db/prelabel_repository.py) | 프리라벨 단계 실행 기록(결과 0개 포함, 추가 전용) 저장·조회 (ADR 0015 개정, F-PAR-75). |
| [`db/privacy_repository.py`](../../packages/schema/src/dlp_schema/db/privacy_repository.py) | 추가 전용 탐지 완료 이력(ADR 0032·0033)과 승인 보류 기록(ADR 0042) 저장·조회. |
| [`db/repository.py`](../../packages/schema/src/dlp_schema/db/repository.py) | 계약 타입 ↔ DB 행 변환과 기본 저장·조회. 연결(Connection)과 트랜잭션은 호출자가 관리한다. |
| [`db/review_queries.py`](../../packages/schema/src/dlp_schema/db/review_queries.py) | 검수 계획에 필요한 집계. 라벨 이력·기하 페이로드 전체를 파이썬으로 가져오지 않는다. |
| [`db/review_repository.py`](../../packages/schema/src/dlp_schema/db/review_repository.py) | 검수 도구 작업·운영 배정 저장소. 커밋과 잠금 순서는 호출자가 관리한다. |
| [`db/scene3d_repository.py`](../../packages/schema/src/dlp_schema/db/scene3d_repository.py) | 청소 전후 3D 비교 산출물 기록(추가 전용 `scene3d_records`) 저장·조회 (ADR 0066). |
| [`db/tables.py`](../../packages/schema/src/dlp_schema/db/tables.py) | PostgreSQL 테이블 정의 (SQLAlchemy Core). 스키마 변경은 반드시 Alembic 마이그레이션으로 한다. |
| [`db/trajectory_repository.py`](../../packages/schema/src/dlp_schema/db/trajectory_repository.py) | 3D 궤적이 2D 트랙 검수 결과를 따르게 DB에 적용한다 (F-COE-19·21·22, ADR 0048·0050). |
| [`description_follow.py`](../../packages/schema/src/dlp_schema/description_follow.py) | 설명 라벨이 그것이 설명하는 행동·상위 구간의 검수 결과를 따른다 (F-PAR-15, ADR 0043 3절). |
| [`episode.py`](../../packages/schema/src/dlp_schema/episode.py) | 에피소드 그래프: 개체·관계·이벤트·상태 전이 네 층으로 라벨을 묶는다. |
| [`export.py`](../../packages/schema/src/dlp_schema/export.py) | 내보내기 파일 계약 (구매자에게 주는 형식). |
| [`history.py`](../../packages/schema/src/dlp_schema/history.py) | 라벨 이력에서 검수가 모델 라벨을 어떻게 바꿨는지 읽는다. |
| [`jsonschema.py`](../../packages/schema/src/dlp_schema/jsonschema.py) | 계약 타입의 JSON Schema 생성. 다른 언어·도구(검수 UI 등)가 같은 계약을 쓰게 한다. |
| [`label_graph.py`](../../packages/schema/src/dlp_schema/label_graph.py) | 현재 라벨의 상위 구간 연결 검사. DB 저장과 에피소드 계약이 같은 규칙을 사용한다. |
| [`label_validation.py`](../../packages/schema/src/dlp_schema/label_validation.py) | 라벨 배치를 저장하기 전 세션 단위 순수 검증. 잠금·조회·INSERT는 저장소가 맡는다. |
| [`labels.py`](../../packages/schema/src/dlp_schema/labels.py) | 라벨 레코드(`LabelRecord`)와 종류별 페이로드: 사람·모델·센서 라벨을 모두 담는 계약 타입 (덮어쓰지 않고 추가만). |
| [`lineage.py`](../../packages/schema/src/dlp_schema/lineage.py) | 계보: 골든셋 버전, 학습 실행, 내보내기, 사용 중지 기록. |
| [`migration.py`](../../packages/schema/src/dlp_schema/migration.py) | 온톨로지 버전 간 라벨 이관: 매핑 표로 ID를 바꾼 새 레코드를 만들고 목적지 사전으로 의미를 검사한다. |
| [`ontology.py`](../../packages/schema/src/dlp_schema/ontology.py) | 온톨로지 사전과 로더. 온톨로지는 config/ontology/<버전>/*.yaml에 둔다. |
| [`ops.py`](../../packages/schema/src/dlp_schema/ops.py) | 운영 기록: 원본 접근 감사, 검수 작업 시간, 잔여 블러 감사, 원본 보관 결정 (WP16). |
| [`predictor.py`](../../packages/schema/src/dlp_schema/predictor.py) | 자동 모델의 공통 인터페이스 (`Predictor`): 클립 하나를 받아 라벨 레코드를 내는 모델 어댑터의 규약. |
| [`review.py`](../../packages/schema/src/dlp_schema/review.py) | 검수 작업: 외부 검수 도구(CVAT, Label Studio)의 작업과 세션·스트림의 대응. |
| [`reviewer_changes.py`](../../packages/schema/src/dlp_schema/reviewer_changes.py) | 검수자가 직접 낸 검수 변화 (물려받기만 한 라벨 제외, F-COE-23·31·32). |
| [`scene3d.py`](../../packages/schema/src/dlp_schema/scene3d.py) | 청소 전후 3D 비교와 장면 안 바디캠 위치의 계약 타입 (ADR 0066, 0067). |
| [`session.py`](../../packages/schema/src/dlp_schema/session.py) | 세션과 스트림의 계약 타입: 녹화 세션 하나와 그 안의 스트림, 캘리브레이션, 생애주기·프라이버시 상태. |
| [`testing.py`](../../packages/schema/src/dlp_schema/testing.py) | 테스트용 계약 객체 생성 도우미. 다른 패키지의 테스트에서도 쓴다. |
| [`trajectory_follow.py`](../../packages/schema/src/dlp_schema/trajectory_follow.py) | 3D 궤적이 그것을 계산한 2D 트랙의 검수 결과를 따른다 (F-COE-19·21·22, ADR 0048·0050·0051·0058). |
| [`validation.py`](../../packages/schema/src/dlp_schema/validation.py) | 라벨을 온톨로지에 대조해 검증한다. 스키마 검증(타입)과 별개로, 값이 사전 안에 있는지 본다. |
| [`wearer_body.py`](../../packages/schema/src/dlp_schema/wearer_body.py) | 1인칭 영상에서 착용자 자신의 몸(팔·몸통)을 알아보는 순수 규칙 (ADR 0055·0058). |

### DB 마이그레이션

| 파일 | 역할 |
| --- | --- |
| [`db/migrations/__init__.py`](../../packages/schema/src/dlp_schema/db/migrations/__init__.py) | Alembic 마이그레이션 스크립트 디렉터리 (`env.py`, `script.py.mako`, `versions/NNNN_*.py`). |
| [`db/migrations/env.py`](../../packages/schema/src/dlp_schema/db/migrations/env.py) | Alembic 실행 환경 (Alembic이 마이그레이션마다 이 파일을 실행한다). |
| [`db/migrations/versions/0001_initial_schema.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0001_initial_schema.py) | initial schema |
| [`db/migrations/versions/0002_stream_position.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0002_stream_position.py) | keep stream order within a session |
| [`db/migrations/versions/0003_review_tasks.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0003_review_tasks.py) | review tasks in external annotation tools |
| [`db/migrations/versions/0004_lineage_tables.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0004_lineage_tables.py) | lineage tables |
| [`db/migrations/versions/0005_review_operations.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0005_review_operations.py) | review operations: measurement labels, review modes, assignments |
| [`db/migrations/versions/0006_review_task_sent_labels.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0006_review_task_sent_labels.py) | review tasks remember the labels they sent |
| [`db/migrations/versions/0007_model_versions.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0007_model_versions.py) | retrained model registry |
| [`db/migrations/versions/0008_export_label_states.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0008_export_label_states.py) | exports record their verification policy |
| [`db/migrations/versions/0009_ops_records.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0009_ops_records.py) | operations records: raw access audit log, review work, privacy audits, retention decisions |
| [`db/migrations/versions/0010_audit_3_columns_truncate.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0010_audit_3_columns_truncate.py) | audit 3: wider model_version / golden_set_version, block TRUNCATE on append-only tables |
| [`db/migrations/versions/0011_session_lifecycle_events.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0011_session_lifecycle_events.py) | audit 4: append-only session lifecycle events (backfilled with each session's current state) |
| [`db/migrations/versions/0012_privacy_detection_and_deployment.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0012_privacy_detection_and_deployment.py) | 탐지 완료 이력(0건 포함)과 과제별 배포 유일성 (ADR 0032). |
| [`db/migrations/versions/0013_privacy_review_generation.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0013_privacy_review_generation.py) | 프라이버시 검수 작업을 완료 탐지 세대에 연결 (ADR 0033). |
| [`db/migrations/versions/0014_label_ontology_version.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0014_label_ontology_version.py) | 새 라벨의 세션 온톨로지 버전 일치 강제 (ADR 0037). |
| [`db/migrations/versions/0015_review_task_revocation.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0015_review_task_revocation.py) | 검수 작업 매체 회수 outbox 열 (ADR 0039). |
| [`db/migrations/versions/0017_privacy_holds.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0017_privacy_holds.py) | 사고 대응 프라이버시 승인 보류·해제 기록 (ADR 0042, F-REV-18) |
| [`db/migrations/versions/0018_golden_truth_label_ids.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0018_golden_truth_label_ids.py) | 골든셋 버전의 정답 라벨 ID 고정 열 (ADR 0041). |
| [`db/migrations/versions/0019_ai_review.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0019_ai_review.py) | AI 1차 검수 기록 표 다섯 개 (추가 전용, ADR 0040). |
| [`db/migrations/versions/0020_ai_unit_ontology_digest.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0020_ai_unit_ontology_digest.py) | AI 1차 검수 판정 단위의 온톨로지 내용 해시 열 (F-AIR-32, ADR 0040). |
| [`db/migrations/versions/0021_description_inheritance.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0021_description_inheritance.py) | 설명의 검수 물려받기 기록 (F-PAR-20, ADR 0043 3절 2026-10-15 개정) |
| [`db/migrations/versions/0022_calibration_registry.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0022_calibration_registry.py) | 카메라 내부 파라미터·작업자 손 크기 등록 (ADR 0045) |
| [`db/migrations/versions/0023_trajectory_inheritance.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0023_trajectory_inheritance.py) | 3D 궤적의 검수 물려받기 기록 (F-COE-19, ADR 0048) |
| [`db/migrations/versions/0024_hand_size_lifecycle_and_camera_assignment.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0024_hand_size_lifecycle_and_camera_assignment.py) | 작업자 손 크기 출처·무효 기록과 수집 뒤 세션 기기 연결 (ADR 0052) |
| [`db/migrations/versions/0025_prelabel_stage_runs.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0025_prelabel_stage_runs.py) | 프리라벨 단계 실행 기록 (결과 0개 포함, ADR 0015 개정, F-PAR-75) |
| [`db/migrations/versions/0026_training_run_inputs.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0026_training_run_inputs.py) | 학습 실행의 입력 지문·해시와 학습 세션 (ADR 0016 2026-11-05 보강, T2·T3) |
| [`db/migrations/versions/0027_scene3d_records.py`](../../packages/schema/src/dlp_schema/db/migrations/versions/0027_scene3d_records.py) | 청소 전후 3D 비교 산출물 기록 (ADR 0066) |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/schema/tests/conftest.py) | dlp_schema 테스트 공용 픽스처: 저장소 루트와 온톨로지 v1. |
| [`test_aireview_contracts.py`](../../packages/schema/tests/test_aireview_contracts.py) | AI 1차 검수 계약 테스트 (`dlp_schema.aireview`, WP17.2, ADR 0040). |
| [`test_aireview_repository.py`](../../packages/schema/tests/test_aireview_repository.py) | AI 1차 검수 기록 저장소 테스트 (SQLite, 서비스 없음. WP17.4, ADR 0040). |
| [`test_aireview_tables.py`](../../packages/schema/tests/test_aireview_tables.py) | AI 1차 검수 표의 PostgreSQL 성질 (services, WP17.4, ADR 0040 I5·I8). |
| [`test_calibration_registry.py`](../../packages/schema/tests/test_calibration_registry.py) | 카메라 내부 파라미터·작업자 손 크기 등록과 결정 규칙 (ADR 0045, `dlp_schema.calibration`). |
| [`test_contracts.py`](../../packages/schema/tests/test_contracts.py) | 계약 타입(Pydantic) 규칙 테스트. |
| [`test_db.py`](../../packages/schema/tests/test_db.py) | DB 계층 테스트: 마이그레이션과 테이블 정의 일치(SQLite), 저장소 함수와 트리거(PostgreSQL). |
| [`test_description_follow.py`](../../packages/schema/tests/test_description_follow.py) | 설명 라벨이 행동 검수 결과를 따르는 규칙 (`dlp_schema.description_follow`, F-PAR-15). |
| [`test_description_inheritance_migration.py`](../../packages/schema/tests/test_description_inheritance_migration.py) | Alembic 0021의 물려받기 기록 백필 (F-PAR-20, ADR 0043 3절 2026-10-15 개정). |
| [`test_episode_migration_config.py`](../../packages/schema/tests/test_episode_migration_config.py) | 에피소드 그래프·현재 라벨·온톨로지 이관·데이터셋 버전·설정 로더·JSON Schema·파생 ID 테스트. |
| [`test_golden_truth.py`](../../packages/schema/tests/test_golden_truth.py) | 골든셋 정답 고정 계약·저장소·마이그레이션 0018 (ADR 0041). |
| [`test_history.py`](../../packages/schema/tests/test_history.py) | 검수 결과 읽기 (dlp_schema.history.review_changes). |
| [`test_label_graph.py`](../../packages/schema/tests/test_label_graph.py) | 상위 구간 연결 검사의 행동 노드 식별 (F-SCH-01). |
| [`test_migration_downgrade_guards.py`](../../packages/schema/tests/test_migration_downgrade_guards.py) | 추가 전용 기록이 있으면 오래된 리비전 downgrade를 거부하는지 (F-SCH-28). |
| [`test_migration_sets.py`](../../packages/schema/tests/test_migration_sets.py) | 온톨로지 이관의 집합 검사 묶음 (F-SCH-03). |
| [`test_ontology.py`](../../packages/schema/tests/test_ontology.py) | 온톨로지 로더·교차 참조 검증과 라벨의 온톨로지 대조(`validation.check_label`) 테스트. |
| [`test_ontology_boundaries.py`](../../packages/schema/tests/test_ontology_boundaries.py) | 온톨로지 저장·계층·이관 경계의 회귀 검사. 실영상 대신 합성 계약을 사용한다. |
| [`test_privacy_detection.py`](../../packages/schema/tests/test_privacy_detection.py) | 탐지 완료 계약의 DB 왕복·스트림 참조 무결성 (SQLite, ADR 0032). |
| [`test_protected_retraction.py`](../../packages/schema/tests/test_protected_retraction.py) | 자동 단계가 사람이 확인한 라벨을 지우지 못하는지 (저장 시점 검사). |
| [`test_review_queries.py`](../../packages/schema/tests/test_review_queries.py) | 검수 계획 집계의 운영 라벨·부하 규약 회귀 (실제 PostgreSQL). |
| [`test_review_revocation.py`](../../packages/schema/tests/test_review_revocation.py) | 검수 작업 매체 회수 계약·저장소·마이그레이션 0015 (ADR 0039). 서비스 없이 SQLite로 돈다. |
| [`test_scene3d_contracts.py`](../../packages/schema/tests/test_scene3d_contracts.py) | 청소 전후 3D·바디캠 위치 계약의 규칙 테스트 (ADR 0066·0067). |
| [`test_trajectory_follow.py`](../../packages/schema/tests/test_trajectory_follow.py) | 3D 궤적이 그것을 계산한 2D 트랙의 검수를 따른다 (F-COE-19·21·22, ADR 0048·0050). |
| [`test_wearer_body.py`](../../packages/schema/tests/test_wearer_body.py) | 착용자 몸 판정 규칙 (`dlp_schema.wearer_body`, ADR 0055·0058). |

## packages/sync

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/sync/src/dlp_sync/__init__.py) | 멀티스트림 동기화 (WP4, ADR 0004·0028). |
| [`anchors.py`](../../packages/sync/src/dlp_sync/anchors.py) | 앵커와 시계 맞춤 (WP4, ADR 0004·0028). |
| [`loader.py`](../../packages/sync/src/dlp_sync/loader.py) | 세션 스트림의 URI에서 동기화 입력(`StreamMedia`)을 불러온다 (WP4). |
| [`pipeline.py`](../../packages/sync/src/dlp_sync/pipeline.py) | 세션 동기화 (WP4, ADR 0004·0028). `dlp sync run`의 계산 본체. |
| [`policy.py`](../../packages/sync/src/dlp_sync/policy.py) | 동기화 정책 (config/policies/sync.yaml, WP4, ADR 0004·0028). |
| [`runner.py`](../../packages/sync/src/dlp_sync/runner.py) | DB에 등록된 세션을 동기화하고 결과를 저장한다 (`dlp sync run`, `dlp sync adjust`, WP4). |
| [`signals.py`](../../packages/sync/src/dlp_sync/signals.py) | 동기화에 쓰는 신호 적재. 모든 시각은 해당 스트림 시계의 ms다 (WP4). |
| [`slate.py`](../../packages/sync/src/dlp_sync/slate.py) | QR 슬레이트 검출 (`qr_slate` 방법, WP4, ADR 0004·0028). |
| [`taps.py`](../../packages/sync/src/dlp_sync/taps.py) | 두 번 두드림 검출과 스트림 사이 대응 (`tap_event` 방법, WP4, ADR 0004·0028). |
| [`xcorr.py`](../../packages/sync/src/dlp_sync/xcorr.py) | 상호상관으로 오프셋 찾기 (`audio_xcorr`, `motion_xcorr` 방법, WP4, ADR 0004·0028). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`conftest.py`](../../packages/sync/tests/conftest.py) | 동기화 테스트 공용 픽스처 (WP4). |
| [`test_long_recording.py`](../../packages/sync/tests/test_long_recording.py) | 긴 녹화(20분)·큰 드리프트(약 80 ppm)에서의 동기화 (ADR 0028). |
| [`test_pipeline.py`](../../packages/sync/tests/test_pipeline.py) | `synchronize`의 방법 선택·대체 순서·정확도 테스트 (WP4 완료 기준, ADR 0004). |
| [`test_runner.py`](../../packages/sync/tests/test_runner.py) | `dlp sync run`/`adjust`의 DB·저장소 통합 테스트 (WP4, `@pytest.mark.services`). |
| [`test_units.py`](../../packages/sync/tests/test_units.py) | 동기화 구성 요소 단위 테스트 (WP4). |

## packages/training

### 소스

| 파일 | 역할 |
| --- | --- |
| [`__init__.py`](../../packages/training/src/dlp_train/__init__.py) | 재학습 루프 (WP13): 검수 승인 → 데이터셋 버전 → (누적 시) 학습 → 골든셋 평가 → 게이트 통과 시 배포. |
| [`augment.py`](../../packages/training/src/dlp_train/augment.py) | 학습용 데이터 증강 (ADR 0057). 픽셀과 모든 라벨 종류를 같은 변환으로 함께 바꾼다. |
| [`base.py`](../../packages/training/src/dlp_train/base.py) | 학습기·로더 인터페이스 (`Trainer`, `ModelLoader`, `TrainOutput`, `LoadContext`, `TrainingError`). |
| [`deployed.py`](../../packages/training/src/dlp_train/deployed.py) | 배포된 재학습 모델을 단계(프리라벨·프라이버시)에 붙인다. |
| [`extract.py`](../../packages/training/src/dlp_train/extract.py) | 데이터셋 버전에서 과제별 학습 예제를 뽑는다. |
| [`loop.py`](../../packages/training/src/dlp_train/loop.py) | 재학습 루프 한 번 (`dlp train run`). |
| [`policy.py`](../../packages/training/src/dlp_train/policy.py) | 재학습 정책 (config/policies/training.yaml). |
| [`tracking.py`](../../packages/training/src/dlp_train/tracking.py) | 실험 기록 (MLflow): 실행·파라미터·지표·산출물을 남기고 배포 모델을 MLflow 레지스트리에도 올린다 (기준은 DB). |
| [`trainers.py`](../../packages/training/src/dlp_train/trainers.py) | 학습기와 로더: 과제별 학습기(`oracle-stub`, `yolox`)와, 배포된 산출물을 단계가 쓰는 부품으로 읽는 로더의 등록부. |
| [`yolox.py`](../../packages/training/src/dlp_train/yolox.py) | YOLOX 고정 클래스 물건 탐지기 실제 학습기·로더 (학습기 이름 `yolox`, ADR 0062). |
| [`yolox_data.py`](../../packages/training/src/dlp_train/yolox_data.py) | 학습 예제 → YOLOX COCO 데이터 디렉터리 (ADR 0062, 학습기 `dlp_train.yolox`). |

### 시험

| 파일 | 역할 |
| --- | --- |
| [`test_augment.py`](../../packages/training/tests/test_augment.py) | 학습 증강(`dlp_train.augment`)과 RLE(`dlp_media.rle`) 시험 (ADR 0057). |
| [`test_extract.py`](../../packages/training/tests/test_extract.py) | 학습 예제 추출(`dlp_train.extract`), 학습 정책 검증, oracle-stub 학습기·로더의 단위 테스트. |
| [`test_loop.py`](../../packages/training/tests/test_loop.py) | stub 모델로 재학습 루프 전체: 누적 → 학습 → 골든셋 평가 → 게이트 실패 시 미배포 / 통과 시 배포. |
| [`test_mlflow.py`](../../packages/training/tests/test_mlflow.py) | MLflow 서버(make up)에 실행·지표·산출물·등록 모델을 남기고 다시 읽어 확인한다. |
| [`test_policy.py`](../../packages/training/tests/test_policy.py) | 학습 정책과 다른 단계 정책·코드의 대응 (서비스 없이). |
| [`test_yolox.py`](../../packages/training/tests/test_yolox.py) | YOLOX 실제 학습기·로더 시험 (ADR 0062). PyTorch 없이 도는 부분만 (실제 학습은 isolated_env 시험). |
| [`test_yolox_isolated.py`](../../packages/training/tests/test_yolox_isolated.py) | YOLOX 실제 학습 스모크 (격리 환경, CPU 1~2 반복, ADR 0062). `make test-isolated`에서만 돈다. |

## config

| 파일 | 역할 |
| --- | --- |
| [`console.yaml`](../../config/console.yaml) | 운영 콘솔 설정 (ADR 0061). dlp_console.settings.load_console_config가 읽는다 (모르는 키는 거부). 콘솔은 기존 `dlp <단계>` 명령을 작업으로 돌리는 화면이라 단계의 정책 값은 여기 두지 않는다 (각 단계의 config/policies/*.yaml이 그대로 쓰인다). 여기는 화면·작업 실행·업로드 사본 다루기만 정한다. |
| [`defaults.yaml`](../../config/defaults.yaml) | 미결정 사항의 기본값. 사람이 결정하면 이 값만 바꾼다 (기준 문서 "미결정 사항" 참조). 코드는 이 값을 하드코딩하지 않고 설정 로더(WP1)를 통해 읽는다. |
| [`external_datasets.yaml`](../../config/external_datasets.yaml) | 공개 데이터셋 목록 (ADR 0060). 파이프라인 밖 1차 학습에만 쓴다: 라벨 이력·데이터셋 버전·골든셋·내보내기에 넣지 않는다. 읽는 곳: dlp_models.datasets (make licenses가 검사). 필드: version(판), homepage, download(공식 주소), annotation_license, media_license(원문 표기), terms(약관 요지), access: open \| request \| signed, commercial: allowed \| review \| forbidden (ADR 0010과 같은 뜻, 이 데이터로 학습한 가중치의 분류. forbidden은 학습에 쓰지 않는다), tasks, verified(원문 확인일, 없으면 미확인), sources(확인 근거 주소. verified가 있으면 필수). 원문을 확인하기 전 항목은 verified를 비워 두고 분류를 보수적으로(review) 둔다. 원문 확인 결과는 docs/handover/public-datasets.md (2026-10-26). 학술 기관 한정(academic only)·사용자 자격 제한 약관은 법무 판단 전까지 review로 두지만 회사가 쓸 자격 자체가 쟁점이다 (같은 문서 법무 질문 2). 원문을 열지 못해 넣지 않은 것: EgoHands, 100DOH, EGTEA Gaze+, EgoObjects(동의서 PDF), CrowdPose. |
| [`models.yaml`](../../config/models.yaml) | 모델 가중치 목록. `make models`가 받아 sha256을 확인한다 (data/models/, 저장소에는 넣지 않는다). 정책 파일(privacy.yaml, prelabel.yaml, actions.yaml vlm.model)은 여기 이름으로 모델을 가리킨다. url: 직접 받는 파일. archive_member가 있으면 zip을 받아 그 이름으로 끝나는 파일을 꺼낸다 (sha256은 zip의 해시, artifact_sha256은 추출 파일의 해시). export: 공개 ONNX가 없어 공식 가중치를 직접 변환한다 (스크립트 실행, make export-models). sha256은 내보낸 환경의 값이며 PyTorch 버전에 따라 달라질 수 있다. |
| [`ontology/v1/domains_tasks.yaml`](../../config/ontology/v1/domains_tasks.yaml) | 온톨로지 v1 사전: 도메인(domains)과 작업(tasks). |
| [`ontology/v1/events.yaml`](../../config/ontology/v1/events.yaml) | 온톨로지 v1 사전: 실패·안전·감염 관리 이벤트(events). |
| [`ontology/v1/hand.yaml`](../../config/ontology/v1/hand.yaml) | 온톨로지 v1 사전: 손 상태 어휘 (접촉 대상 종류, 파지, 손 역할, 손·팔 관절 이름). |
| [`ontology/v1/manifest.yaml`](../../config/ontology/v1/manifest.yaml) | 온톨로지 v1 (초안). 기준 문서 "온톨로지 v1" 절을 옮긴 것이다. 식별자는 영어 snake_case, 사람용 이름은 ko(한국어)·en(영어) 필드에 둔다 (en은 ADR 0043에서 덧붙임). 객체 클래스와 하위 단계는 파일럿 후 확정한다 (provisional). version: SemVer 문자열 (따옴표 필수가 아니지만 문자열로 둔다). 라벨의 ontology_version과 대조된다. status: draft(초안, 덧붙이기 허용) \| frozen(확정, 같은 버전으로 내용 변경 금지). provisional: 파일럿 후 확정할 범주 (설명용, 동작에 영향 없음). |
| [`ontology/v1/objects.yaml`](../../config/ontology/v1/objects.yaml) | 온톨로지 v1 사전: 객체 클래스(objects)와 물리 유형, 상태 속성, 표면 부분. |
| [`ontology/v1/physics.yaml`](../../config/ontology/v1/physics.yaml) | 온톨로지 v1 사전: 물리 사전값 어휘 (재질 materials, 촉감 tactile_properties). |
| [`ontology/v1/privacy.yaml`](../../config/ontology/v1/privacy.yaml) | 온톨로지 v1 사전: 프라이버시(블러) 대상 (privacy_targets). |
| [`ontology/v1/subjects.yaml`](../../config/ontology/v1/subjects.yaml) | 온톨로지 v1 사전: 사람 대상 어휘 (신체 부위 body_parts, 자세 postures). |
| [`ontology/v1/verbs.yaml`](../../config/ontology/v1/verbs.yaml) | 온톨로지 v1 사전: 동사(verbs)와 공백 유형(gap_types). |
| [`policies/actions.yaml`](../../config/policies/actions.yaml) | 행동 구간 2단 구조 정책 (WP10). 1단: 손목 속도와 접촉 신호로 경계 후보를 만든다 (VLM은 경계를 정하지 않는다). 2단: VLM이 후보 구간마다 온톨로지 목록 안에서 행동·사이 구간 종류를 고르고 한 문장 설명을 쓴다. 같은 분류의 인접 구간은 병합하고, 행동이 아닌 구간은 대기·미상·범위 외로 채워 공백을 없앤다. |
| [`policies/active.yaml`](../../config/policies/active.yaml) | 액티브 러닝 정책 (WP14). 다음에 검수·라벨링할 세션을 고른다. 세션 안의 검수 순서는 config/policies/review.yaml priority(우선순위 큐)이고, 이 점수와 별개다. 읽는 곳: dlp_active.policy.load_policy (`dlp active rank\|fiftyone`). 관련: ADR 0017. 이 파일은 파싱한 값으로만 쓴다 (주석을 바꿔도 결과가 같다). |
| [`policies/aireview.yaml`](../../config/policies/aireview.yaml) | AI 1차 검수(로컬 VLM) 정책 (WP17, ADR 0040). 읽는 곳: dlp_aireview.policy.load_policy (Pydantic AiReviewPolicy, 알 수 없는 키는 오류). 쓰는 명령: dlp aireview probe\|bench\|estimate (WP17.3), run\|status\|show\|waive (WP17.5), eval (WP17.8). |
| [`policies/calibration.yaml`](../../config/policies/calibration.yaml) | 카메라 내부 파라미터 캘리브레이션 (ADR 0045·0052, dlp_media.calibration, `dlp calib intrinsics\|import`). 체커보드(평평한 판에 붙인, 칸이 정사각형인 인쇄물. 칸 크기는 내부 파라미터에 영향이 없다)를 그 카메라로 여러 위치·기울기에서 찍은 영상에서 OpenCV calibrateCamera로 fx·fy·cx·cy·왜곡 계수를 구한다. 결과는 DB camera_calibrations에 기기 ID로 등록한다. 절차는 docs/handover/camera-calibration.md. |
| [`policies/dataset.yaml`](../../config/policies/dataset.yaml) | 데이터셋 버전 정책 (WP7). 분할기는 작업자·장소를 누수 방지 단위로 고정해 쓴다. config/defaults.yaml golden_set.split_unit 값을 바꿔도 현재 분할 동작은 바뀌지 않는다. 읽는 곳: dlp_datasets.policy.load_policy (`dlp dataset golden\|build`). 관련: ADR 0007. 이 파일은 파싱한 값으로만 쓴다 (주석을 바꿔도 결과가 같다). |
| [`policies/evaluation.yaml`](../../config/policies/evaluation.yaml) | 지표와 평가 하네스 정책 (WP11). 골든셋으로 모델을 평가하고 배포 여부를 판정한다. 로더: dlp_eval.policy.load_policy → EvaluationPolicy (모르는 키는 거부한다). 사용처: dlp eval golden(dlp_cli.eval_cmds), 재학습 루프(dlp_train.loop). 관련 ADR 0013·0025. 이 파일은 모델 버전 해시에 들어가지 않는다 (값을 바꿔도 프리라벨 등이 다시 돌지 않는다). 값을 바꾸면 같은 모델이라도 지표·게이트 판정이 달라지므로, 비교하는 두 리포트는 같은 정책으로 만든다. |
| [`policies/export.yaml`](../../config/policies/export.yaml) | 내보내기 정책 (WP15). 미검수 포함 여부의 기본값은 config/defaults.yaml export.include_unreviewed. 읽는 곳: dlp_export.policy.load_policy (`dlp export coco\|intervals\|lerobot\|gr00t`). 관련: ADR 0018(내보내기), 0021(이력 선커밋·가명·LeRobot 격리 환경), 0027(세션·라벨 ID 가명), 0065(GR00T 형식), 0069(촉각장갑 채널), 0070(장면 좌표 바디캠 자세). 이 파일은 파싱한 값으로만 쓴다 (텍스트 해시를 쓰지 않으므로 주석을 바꿔도 결과가 같다). |
| [`policies/media.yaml`](../../config/policies/media.yaml) | 로컬 영상 준비. 원본은 보존하고 PTS/오디오/데이터 트랙을 검증한 사본만 출력한다. |
| [`policies/ops.yaml`](../../config/policies/ops.yaml) | 운영 대시보드와 보안 정책 (WP16). 원본 보관 기간은 config/defaults.yaml retention.raw_retention_days. 읽는 곳: dlp_ops.policy.load_policy (`dlp ops weekly\|audit-report\|retention`). 이 정책은 모델 버전 해시에 들어가지 않으므로 값을 바꿔도 파이프라인 단계가 다시 돌지 않는다 (리포트·경고에만 영향). 알 수 없는 키를 넣으면 로드가 실패한다 (Contract: extra=forbid). 정책 파일 형식 버전. 키 구조를 바꿀 때만 올린다. |
| [`policies/pilot.yaml`](../../config/policies/pilot.yaml) | 실제 영상 첫 시험의 범위. 임계값은 각 단계의 기존 정책을 그대로 사용한다. |
| [`policies/prelabel.yaml`](../../config/policies/prelabel.yaml) | 자동 프리라벨 정책 (WP8). 읽는 곳: dlp_prelabel.policy (`dlp prelabel run <세션>`). 실제 모델을 아직 연동하지 못한 곳은 "TODO(real-model)"로 표시하고 필요한 것을 적는다. 저장소 전체에서 찾기: make todo-models |
| [`policies/privacy.yaml`](../../config/policies/privacy.yaml) | 프라이버시 게이트 정책 (WP5). 유지 시간·렌더 방식·오디오 제거는 config/defaults.yaml privacy를 따른다. |
| [`policies/relations.yaml`](../../config/policies/relations.yaml) | 관계 도출과 표면 커버리지 정책 (WP9). 관계는 사람이 그리지 않고 이 규칙으로 만든다. 규칙을 바꾸면 `dlp relations run`이 관계를 다시 만든다: 같은 내용은 그대로 두고, 더 이상 나오지 않는 관계는 삭제 레코드로 표시하고, 새 관계만 넣는다. 검수자가 고치거나 지운 관계는 다시 넣지 않는다. 읽는 곳: dlp_relations.policy. 정책 해시(RelationsPolicy.digest, 관계·커버리지 레코드의 model_version "relations-<해시>")는 파싱된 값으로 계산하므로 주석은 해시에 영향이 없다. 검수자가 승인·표본 검증한 관계는 규칙에서 빠져도 지우지 않는다 (ADR 0015). |
| [`policies/retarget.yaml`](../../config/policies/retarget.yaml) | 로봇 리타기팅 정책 (ADR 0047·0068, dlp_retarget.policy). 읽는 곳: `dlp export lerobot --robot <ID>` (dlp_export.robot). 로봇마다 다른 값(허용 오차, 배치, 그리퍼 또는 다지 손, IK 가중치)은 로봇 설정 config/robots/<ID>.yaml(또는 DLP_ROBOTS_DIR)에 둔다. 이 파일과 로봇 설정, 모델 파일 sha256은 리타기팅 판(retarget-...+r..+m..+p..)에 들어가 내보내기 manifest에 남는다. |
| [`policies/review.yaml`](../../config/policies/review.yaml) | 검수 운영 정책 (WP12). 배정 비율(QA·이중·블라인드·오류 삽입)은 config/defaults.yaml review를 따른다. |
| [`policies/scene3d.yaml`](../../config/policies/scene3d.yaml) | 청소 전후 다각도 사진의 3D 복원·물체 이동 비교와 장면 안 바디캠 위치 추정 정책 (dlp scene3d, ADR 0066·0067). 읽는 곳: dlp_scene3d.policy.load_policy. 각 절의 해시가 그 단계 결과의 모델 버전에 들어간다 (절을 바꾸면 그 단계부터 다시 계산한다. 이전 결과는 지우지 않고 새 버전 키로 쓴다). 맨 아래 interpolation 절만 예외다: 위치 기록을 읽을 때만 쓰므로 바꿔도 아무것도 다시 계산하지 않는다. |
| [`policies/sync.yaml`](../../config/policies/sync.yaml) | 동기화 정책 (WP4, ADR 0004·0028). 스트림 종류별로 방법을 순서대로 시도하고, 신뢰도가 min_confidence 이상인 첫 결과를 쓴다. |
| [`policies/training.yaml`](../../config/policies/training.yaml) | 재학습 루프 정책 (WP13). 검수 승인 → 데이터셋 버전 → 누적 확인 → 학습 → 골든셋 평가 → 게이트(config/policies/evaluation.yaml) → 배포. 로더: dlp_train.policy.load_policy → TrainingPolicy (모르는 키는 거부, 아래 규칙은 검증기가 막는다). 사용처: dlp train run\|approve(dlp_cli.train_cmds), 배포 모델 붙이기(dlp_train.deployed). 관련 ADR 0016·0025. 이 파일은 모델 버전 해시에 들어가지 않는다 (모델 버전 태그는 과제·데이터셋 버전·학습기·파라미터·시각). |
| [`policies/translate.yaml`](../../config/policies/translate.yaml) | 설명 라벨 영어 번역 정책 (ADR 0043). 한국어 설명(description, language=ko) 라벨마다 영어 번역 레코드(language=en)를 따로 만든다. 번역은 원문의 수정이 아니라 별도 운영 라벨이다: parent_label_id를 쓰지 않고 페이로드 source_label_id로 원문 레코드를 가리킨다. 원문이 바뀌면(새 레코드) 검수 전 번역만 지우고 다시 만든다. |
| [`pretrain/objects.yaml`](../../config/pretrain/objects.yaml) | 공개 데이터셋 1차 학습: 가사·청소·돌봄 객체 고정 클래스 탐지기 (ADR 0060). 읽는 곳: scripts/pretrain/build_objects.py(학습 데이터 만들기), dlp_models.datasets(검사). 학습은 파이프라인 밖(Kaggle·Colab 등)에서 공개 데이터만으로 한다. 우리 데이터는 넣지 않는다. |
| [`review_ui.yaml`](../../config/review_ui.yaml) | 검수 화면 테마 (CVAT·Label Studio 위에 덧씌우는 모양). dlp_review.ui_theme.load_ui_theme가 읽는다. 화면 모양(색·로고·단축키·영상 높이)만 정한다. 라벨 값, 검수 흐름, 저장 형식은 바꾸지 않는다. 값을 바꾼 뒤: - CVAT: `make ui-theme`으로 services/review-theme/cvat/dlp-tokens.css를 다시 만들고 `make cvat-up` (테마 이미지를 다시 빌드한다). 테스트가 생성 파일과 이 파일의 일치를 검사한다. - Label Studio: 다음 `dlp review create` 때 프로젝트 설정(label_config)이 바뀐 값으로 갱신된다. - CVAT 라벨 색: 다음 `dlp review create` 때 프로젝트 라벨 색이 맞춰진다. 색은 모두 #rrggbb (소문자 또는 대문자 16진수 6자리). |
| [`robots/so101.yaml`](../../config/robots/so101.yaml) | 공개 시험 로봇: SO-101 6자유도 팔(5관절 + 평행 그리퍼) (ADR 0047, dlp_retarget.robot). 리타기팅 경로·시험에만 쓰는 공개 모델이다. 고객 로봇 설정·설계 파일은 저장소에 넣지 않고 DLP_ROBOTS_DIR (저장소 밖 디렉터리)에 둔다 (docs/handover/robot-retargeting.md). 저장소 안 설정은 model.public: true여야 로더가 받는다. 이 파일은 평행 그리퍼 로봇의 형식이다. 손가락 관절이 있는 다지 손 로봇은 arm.gripper· tip_link·접근·닫힘 축 대신 arm.dexterous_hand를 적는다 (형식과 예시는 같은 문서 2절, ADR 0068). |

## scripts

| 파일 | 역할 |
| --- | --- |
| [`doc_coverage.py`](../../scripts/doc_coverage.py) | 패키지별 설명(docstring)·주석 양을 센다 (`make doc-coverage`). |
| [`export_depth_onnx.py`](../../scripts/export_depth_onnx.py) | Depth Anything V2 Metric Indoor Small(공식 Hugging Face 가중치)을 ONNX로 내보낸다. |
| [`file_map.py`](../../scripts/file_map.py) | 파일 지도 (`make file-map`): 파일마다 역할 한 문단을 모아 docs/handover/file-map.md를 만든다. |
| [`gsplat-env/extract_frame_features.py`](../../scripts/gsplat-env/extract_frame_features.py) | 바디캠 블러본 프레임 → pycolmap SIFT 특징 → frame_features.npz (격리 환경, ADR 0067). |
| [`gsplat-env/run_sfm.py`](../../scripts/gsplat-env/run_sfm.py) | 블러 사진 → pycolmap SfM → cameras.json, points.npz (격리 환경, ADR 0066). |
| [`gsplat-env/train_splat.py`](../../scripts/gsplat-env/train_splat.py) | SfM 결과 + 블러 사진 → gsplat 3D 가우시안 학습 → 3DGS PLY (격리 환경, CUDA, ADR 0066). |
| [`install/install.sh`](../../scripts/install/install.sh) | 플랫폼 설치 스크립트 (Mac·Linux·Windows WSL2, ADR 0061 7절). |
| [`lerobot_check.py`](../../scripts/lerobot_check.py) | LeRobot 공식 로더로 내보내기를 다시 읽어 요약을 JSON으로 출력한다. |
| [`lerobot_write.py`](../../scripts/lerobot_write.py) | LeRobot 공식 API로 에피소드를 쓴다 (dlp export lerobot이 격리 환경에서 실행한다). |
| [`pretrain/build_objects.py`](../../scripts/pretrain/build_objects.py) | 공개 데이터셋으로 가사·청소·돌봄 객체 탐지 학습 데이터를 만든다 (ADR 0060 1단계). |
| [`pretrain/train_yolox.py`](../../scripts/pretrain/train_yolox.py) | YOLOX(Apache-2.0)를 공개 데이터 객체 탐지 데이터로 미세조정하고 ONNX로 내보낸다 (ADR 0060 1단계). |
| [`pretrain/yolox_ext.py`](../../scripts/pretrain/yolox_ext.py) | YOLOX 0.3.0 확장: 이미지에서 확인 안 된 클래스의 배경 손실을 빼는 데이터·손실 (ADR 0060). |
| [`preview_session.py`](../../scripts/preview_session.py) | 세션 결과 미리보기 영상 (검수 도구 없이 프리라벨 결과를 눈으로 보기). |
| [`profile_review.py`](../../scripts/profile_review.py) | 합성 PostgreSQL 검수 계획의 조회 비용 측정. 일회용 DB를 만들고 반드시 제거한다. |
| [`retarget_mink.py`](../../scripts/retarget_mink.py) | mink(MuJoCo 미분 IK)로 프레임별 팔 관절 값을 푼다 (retarget.yaml solver: mink, ADR 0047). |
| [`review_scope.py`](../../scripts/review_scope.py) | 전수 검수 범위표 (`make review-scope`). |
| [`yolox-train-env/finetune.py`](../../scripts/yolox-train-env/finetune.py) | YOLOX 0.3.0 미세조정 + ONNX 내보내기 (격리 환경 전용, ADR 0062). |

## services

| 파일 | 역할 |
| --- | --- |
| [`cvat/cvat.sh`](../../services/cvat/cvat.sh) | CVAT 공식 docker compose를 고정 버전으로 받아 띄운다. 사용법: services/cvat/cvat.sh up\|down\|status\|superuser |
| [`cvat/ports.override.yml`](../../services/cvat/ports.override.yml) | CVAT 공식 compose 덮어쓰기 (services/cvat/cvat.sh가 함께 쓴다). traefik의 호스트 포트를 DLP_CVAT_PORT(기본 8080)로 바꾼다. !override는 목록을 합치지 않고 바꾼다. |
| [`docker-compose.yml`](../../services/docker-compose.yml) | 개발용 서비스 묶음. CVAT는 서비스 구성이 커서 공식 compose로 따로 띄운다 (services/cvat/cvat.sh). 실행: `make up`(= docker compose --env-file .env -f services/docker-compose.yml up -d --wait 후 Label Studio 토큰 설정), `make down`, `make clean`(볼륨까지 삭제). 값은 .env(.env.example 복사본)에서 온다. 서비스 관계: postgres  ← label-studio, prefect, mlflow, dlp CLI(dlp DB) 모두의 DB 서버 (DB는 init 스크립트가 만든다) seaweedfs ← bucket-init(버킷 생성), mlflow(산출물), lakefs(데이터셋 객체), dlp CLI(원본·블러본·내보내기) 헬스체크가 통과해야 `up --wait`가 끝나고, `make health`(dlp services check)가 같은 주소를 다시 점검한다. 프로젝트 이름 = 컨테이너·볼륨 이름 접두 (.env COMPOSE_PROJECT_NAME) |
| [`label-studio/bootstrap.py`](../../services/label-studio/bootstrap.py) | 개발용 Label Studio 조직에서 레거시 API 토큰을 켠다. |
| [`mlflow/Dockerfile`](../../services/mlflow/Dockerfile) | Python 기반 이미지에 MLflow와 PostgreSQL·S3 드라이버를 함께 설치해 빌드한다. TLS를 가로채는 사내 프록시 뒤에서는 CA 묶음을 빌드 시크릿으로 넘긴다: docker build --secret id=extra_ca,src=<ca-bundle.crt> -t dlp-mlflow:3.16.1 services/mlflow services/docker-compose.yml의 mlflow 서비스가 `build: ./mlflow`, 이미지 이름 dlp-mlflow:3.16.1로 빌드한다. psycopg2-binary: PostgreSQL 백엔드 저장소, boto3: S3(SeaweedFS) 산출물 저장소. |
| [`postgres/init-databases.sh`](../../services/postgres/init-databases.sh) | 서비스별 데이터베이스를 만든다. 컨테이너 최초 기동 시 한 번만 실행된다. 공식 postgres 이미지가 데이터 디렉터리가 비어 있을 때만 /docker-entrypoint-initdb.d/*.sh를 실행한다 (services/docker-compose.yml이 이 파일을 마운트). 볼륨을 지우지 않으면(`make clean`) 다시 돌지 않는다. DB: dlp(메타데이터, `dlp db upgrade`), labelstudio, prefect, mlflow — 모두 POSTGRES_USER 소유. POSTGRES_DB=dlp라 dlp는 이미 있으므로 존재 확인 뒤 없을 때만 만든다 (멱등). compose 헬스체크는 마지막 mlflow DB로 접속해 이 스크립트가 끝났는지 본다. |
| [`review-theme/cvat/Dockerfile`](../../services/review-theme/cvat/Dockerfile) | aniatman 검수 화면 테마를 얹은 CVAT UI 이미지 (ADR 0038). 공식 cvat/ui 이미지의 정적 파일은 그대로 두고, /dlp/ 아래에 테마 파일을 넣은 뒤 index.html의 </head> 앞에 CSS·JS 연결을 덧붙인다. 화면 소스(React)는 고치지 않는다. 빌드: services/cvat/cvat.sh up이 services/review-theme을 컨텍스트로 빌드한다. |
| [`review-theme/cvat/compose.override.yml`](../../services/review-theme/cvat/compose.override.yml) | CVAT 공식 compose에 덧붙여 UI 컨테이너를 테마 이미지로 바꾼다 (services/cvat/cvat.sh가 쓴다, ADR 0038). 이미지는 cvat.sh up이 먼저 빌드한다 (dlp-cvat-ui:<CVAT 버전>). |
| [`seaweedfs/entrypoint.sh`](../../services/seaweedfs/entrypoint.sh) | 환경 변수로 S3 자격 증명 파일을 만든 뒤 (관리자, 라벨러: 라벨링 버킷 읽기 전용) 단일 노드 SeaweedFS(마스터+볼륨+파일러+S3)를 띄운다. 입력 환경 변수 (services/docker-compose.yml seaweedfs.environment): S3_ACCESS_KEY / S3_SECRET_KEY                — 서비스 계정 (모든 버킷 Admin·Read·List·Tagging·Write) S3_LABELER_ACCESS_KEY / S3_LABELER_SECRET_KEY — 일반 라벨러 (LABELING_BUCKET 읽기·목록만) LABELING_BUCKET                              — 라벨러가 읽을 수 있는 유일한 버킷 (dlp-labeling) VOLUME_SIZE_LIMIT_MB, VOLUME_MAX             — 선택. 볼륨 크기(MB, 기본 1024)·최대 개수(기본 200) 라벨러 키로는 원본 버킷(dlp-raw)을 읽을 수 없다 (WP6 역할 경계, 서명 URL도 이 키로 만든다). 자격 증명 파일은 컨테이너의 /tmp에만 두고 기동할 때마다 다시 만든다. |
| [`vlm/docker-compose.yml`](../../services/vlm/docker-compose.yml) | NVIDIA DGX Spark(GB10, Arm64 + CUDA, 통합 메모리 128 GB)용 VLM 서버 (WP17, ADR 0040). 개발 서비스(services/docker-compose.yml)와 따로, Spark 장비에서만 띄운다. 안내: docs/handover/vlm-server.md |
| [`vlm/vlm.env.example`](../../services/vlm/vlm.env.example) | DGX Spark VLM 서버 설정 (services/vlm/docker-compose.yml). `cp vlm.env.example vlm.env` 뒤 채운다. vlm.env는 저장소에 넣지 않는다 (.gitignore). 안내: docs/handover/vlm-server.md |

## tests

| 파일 | 역할 |
| --- | --- |
| [`test_aireview_boundaries.py`](../../tests/test_aireview_boundaries.py) | AI 1차 검수 패키지의 경계 정적 검사 (WP17, ADR 0040 I1·I3·I4). |
| [`test_aireview_isolation.py`](../../tests/test_aireview_isolation.py) | AI 판정은 라벨·검수·학습·내보내기·스냅샷에 섞이지 않는다 (격리 회귀, WP17.5, ADR 0040 I3·I4·I9). |
| [`test_aireview_panel.py`](../../tests/test_aireview_panel.py) | 판정단 종단 회귀: 두 판정 모델 실행 → 검수 운영의 불일치 사유 (ADR 0063, 서비스 불필요). |
| [`test_doc_test_refs.py`](../../tests/test_doc_test_refs.py) | 인수인계 문서가 인용하는 시험 이름이 실제로 있는지 (F-COE-35). |
| [`test_end_to_end.py`](../../tests/test_end_to_end.py) | 종단 검증 (계획서 "종단 검증"). |
| [`test_fiftyone_server_guard.py`](../../tests/test_fiftyone_server_guard.py) | 우리 코드는 FiftyOne 웹 서버(GraphQL)를 띄우지 않는다 (S3 완화, ADR 0036). |
| [`test_file_map.py`](../../tests/test_file_map.py) | 파일 지도 생성기(scripts/file_map.py) 검사. |
| [`test_fp_hints.py`](../../tests/test_fp_hints.py) | 오탐 의심 힌트(`dlp_privacy.hints`, `dlp_review.object_hints`)와 TTA(`geometry.tta_tracks`). |
| [`test_import_boundaries.py`](../../tests/test_import_boundaries.py) | 패키지 import 경계 검사 (외주 리팩토링의 경계 고정). |
| [`test_inheritance_required.py`](../../tests/test_inheritance_required.py) | 물려받기 계획기는 물려받기 기록을 빠뜨릴 수 없다 (F-PAR-24, ADR 0043 3절, ADR 0048). |
| [`test_issue_registry.py`](../../tests/test_issue_registry.py) | 알려진 문제 레지스트리(docs/handover/issues.yaml)의 일관성 검사. |
| [`test_known_defects.py`](../../tests/test_known_defects.py) | 확인된 미수정 결함의 재현 테스트 (docs/handover/issues.yaml). |
| [`test_migration_seeded_copies.py`](../../tests/test_migration_seeded_copies.py) | 온톨로지 이관이 실제 오류 삽입 사본을 배정별 묶음으로 검사하는지 (F-SCH-03, 패키지 횡단). |
| [`test_model_licenses.py`](../../tests/test_model_licenses.py) | 정책이 쓰는 모델의 상업 사용 분류 검사 (ADR 0010). |
| [`test_policy_consistency.py`](../../tests/test_policy_consistency.py) | 정책 파일 사이의 값이 서로 맞는지 검사한다. |
| [`test_preview_session.py`](../../tests/test_preview_session.py) | 결과 미리보기 영상 스크립트(scripts/preview_session.py) 검사. |
| [`test_raw_access_audited.py`](../../tests/test_raw_access_audited.py) | 원본 버킷 저장소는 감사 저장소로만 만든다 (WP16, ADR 0020·0021). |
| [`test_raw_access_db.py`](../../tests/test_raw_access_db.py) | 완료 기준 (WP16): 원본 열람 이벤트가 빠짐없이 감사 로그에 남는다 (PostgreSQL, S3, make up). |
| [`test_review_regressions.py`](../../tests/test_review_regressions.py) | 전수 검사 후 회귀: 저장소 오류 경계, 0건 검수와 실제 PostgreSQL 동시성 (ADR 0032). |
| [`test_review_scope.py`](../../tests/test_review_scope.py) | 검수 범위표 생성기(scripts/review_scope.py) 검사. |
| [`test_services.py`](../../tests/test_services.py) | 실행 중인 개발 서비스에 대한 통합 테스트. `make up` 후 `make test-services`로 실행한다. |
| [`test_webhook_transaction.py`](../../tests/test_webhook_transaction.py) | CLI 콜백이 수집 예외를 삼키면 부분 변경을 커밋하므로 실제 트랜잭션으로 회귀를 검사한다. |

## 저장소 루트

| 파일 | 역할 |
| --- | --- |
| [`.env.example`](../../.env.example) | 개발용 기본값. `make up`이 .env가 없으면 이 파일을 복사한다. 운영 환경에서는 반드시 바꾼다. |
| [`Makefile`](../../Makefile) | 모든 패키지에 같은 명령으로 검증한다. 에이전트와 CI는 `make check`를 기준으로 삼는다. 개발 명령 모음 (CLAUDE.md "명령어" 절과 같은 목록). |
| [`conftest.py`](../../conftest.py) | 저장소 전체 테스트 공용 설정: 개발자 셸의 AWS 프로필·설정 파일이 시험의 S3 클라이언트에 섞이지 않게 한다. |
