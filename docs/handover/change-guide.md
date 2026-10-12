# 변경 영향과 장애 복구

[인수인계 첫 화면](README.md) · [남은 과제](known-issues.md)

## 변경할 때 함께 보는 곳

아래 테스트 위치는 해당 변경의 검토 시작점이다. 공통 완료 검사는 `make check`이며, 실제
DB·외부 도구·내보내기 동작을 바꾸면 그 경로의 서비스/격리 환경 검사도 실행한다.
변경 전에 [구현 계획](../ai-implementation-plan.md)의 해당 WP와 [작업 규칙](../../CLAUDE.md)을 읽는다.

| 변경 대상 | 코드·설정 | 영향·확인할 테스트 |
| --- | --- | --- |
| 계약/DB/온톨로지 | `schema`, `db/migrations/versions`, `config/ontology` | 모든 소비 패키지. ADR·이관·계약 테스트·`make schemas`/`make contracts` |
| 영상 회전·PTS·프록시 | `media/prepare.py`, `probe.py`, `proxy.py`, `pts.py` | 공간 좌표·시간·검수 재생. `packages/media/tests`, `tests/test_end_to_end.py` |
| 모델 파일·어댑터 | `models/registry.py`, `prelabel/adapters`, `config/models.yaml` | 해시·라이선스·라벨 버전·검수 보존. `packages/models/tests`, `packages/prelabel/tests`, `packages/cli/tests/test_pilot.py` |
| 프라이버시 정책/렌더 | `privacy/generation.py`, `runner.py`, `rendering.py`, `privacy.yaml` | 재검수·재승인·블러 유효성. `packages/privacy/tests`, `packages/schema/tests/test_privacy_detection.py`, `tests/test_review_regressions.py` |
| 검수 작업/수집/권한 | `review/tasks.py`, `collect.py`, `revoke.py`, `ops/`, `cli/review_config.py`, `privacy/revocation.py` | 세대·담당자·이력·완료 판정·배포 매체 회수(ADR 0039). `packages/review/tests`(`test_revoke.py`), `tests/test_webhook_transaction.py`, `packages/schema/tests/test_review_queries.py`, `tests/test_review_regressions.py` |
| 동기화/관계/행동 | `sync`, `relations`, `actions`, 각 정책 | 마스터/스트림 시간과 후속 라벨 재생성. 각 패키지 테스트, 종단 테스트 |
| 학습/평가/배포 | `training/loop.py`, `trainers.py`, `evaluation/gate.py`, 정책 | 골든셋 누수·거절 시 미배포·활성 모델 유일성. training/evaluation 테스트와 DB 서비스 검사 |
| 데이터셋/내보내기 | `datasets`, `export/runner.py`, 형식별 모듈 | 분할·사용 중지·프라이버시·가명·게시 순서. datasets/export 테스트, `make test-isolated` |
| 검수 화면 테마 | `config/review_ui.yaml`, `review/ui_theme.py`, `services/review-theme/`, `cvat.sh` | 색·로고·단축키·모드 전환만. 라벨 이름·왕복 불변. `packages/review/tests/test_ui_theme.py`, `make contracts`(생성 파일 일치). CVAT·Label Studio 버전을 올리면 두 모드 화면을 열어 확인 (ADR 0038) |
| 원본 접근/보관 | `cli/raw_access.py`, `ops`, 버킷 정책 | 감사 우회·CVAT 사본 보관. `tests/test_raw_access_audited.py`, `tests/test_raw_access_db.py`, ops 테스트 |
| AI 1차 검수 실행 | `aireview/runner.py`, `backend.py`(`EndpointPool`), `media.py`, `status.py`, `cli/aireview_run_cmds.py`, `aireview.yaml` | 블러본 경계(I1)·판정 버전·재개·종료 코드. `packages/aireview/tests`(`test_runner*.py`, `test_pool.py`, `test_status.py`, `test_blurred_only.py`), `packages/cli/tests/test_aireview_run_cmds.py`, `tests/test_aireview_boundaries.py`, `tests/test_aireview_isolation.py`, services `test_runner_pg.py`·`test_aireview_tables.py`. 판정 버전 해시에서 빠지는 절(`selection`·`kinds`·`pending`·`server`·`gate`·`estimate`) 밖을 바꾸면 모든 단위를 다시 판정한다 (ADR 0040 4.3) |
| AI 1차 검수 표시 | `review/ai_display.py`, `review/labelstudio.py`(AI 컨트롤), `review/tasks.py`(노출 기록), `review/ops/runner.py`·`results.py`(우선순위·표본 제외·마무리 보호), `cli/review_config.py`·`review_cmds.py`(설정·required 게이트), `review.yaml ai_review`, `review_ui.yaml ai` | 표시는 지금 렌더의 판정만(I10), 증가(표본 제외·보호·우선순위)는 렌더 무관(I9), AI 컨트롤 상시, 판정 없으면 예측·데이터 불변. `packages/review/tests/test_ai_display.py`, `packages/cli/tests/test_review_ai_cmds.py`, services `test_ls_ai_feasibility.py`·`test_ai_display_flow.py`. Label Studio·CVAT 버전을 올리면 아래 "AI 1차 검수 표시" 확인 목록 |
| 의존성·환경 | `pyproject.toml`, `uv.lock`, `services`, CI | 모듈 import 외에 실제 서버/DB 실행·의존성 감사. `make check`, 관련 서비스 검사 |

계약·DB 변경에는 ADR과 필요한 이관을 함께 남긴다. 생성된 `schemas/`는 직접 편집하지 않는다.
문서·주석만 바꿀 때에도 소스 파일의 docstring이 JSON Schema 설명으로 들어가는지 확인한다.

## 상태를 보존하는 리팩토링 순서

1. 관련 테스트와 현재 입출력·부작용을 확인한다. 모의 모델 결과와 실영상 품질을 구분한다.
2. 함수 분리는 입력·결과·외부 호출·DB 쓰기의 경계를 기준으로 한다. 파일 이동만으로 완성으로 보지 않는다.
3. 저장 시 온톨로지 검증과 버전 선택은 [ADR 0037](../adr/0037-ontology-validation-boundaries.md)에
   따라 반영했다. [용어·확장 안내](../ontology-guide.md)를 먼저 보고 다음 작은 변경을 진행한다.
   미해결 데이터의 호환 정책과 과제별 완료 기준은 [남은 과제](known-issues.md)를 따른다.
4. 잠금·긴 트랜잭션은 작업 세대/버전과 재개 가능한 상태를 설계한 뒤 줄인다. 무거운 처리를 밖으로
   옮겨도 최종 승인 상태·사용 중지·입력 세대를 짧은 트랜잭션에서 재검사해야 한다.
5. 외부 API에는 영속 작업 상태·중복 방지 키·결과 대조·보상/정리 절차를 추가한다. DB 롤백이
   CVAT·S3·lakeFS·MLflow 부작용까지 취소한다고 가정하지 않는다.

여러 세션 잠금은 ID 순서를 지킨다. 잠금 제거는 최적화의 완료 기준이 아니며, 병행 실행에서
검수 완료 라벨 보존·잘못된 승인 거부·사용 중지 데이터 게시 금지를 확인해야 한다.

## 실패 시 무엇이 남는가

| 실패 지점 | 남을 수 있는 상태 | 재시도·복구 기준 |
| --- | --- | --- |
| 입력 정규화 | 원본, 실패 변환의 임시 산출물 | `media prepare` 오류와 PTS/방향 확인. 검수한 세션의 파일 교체 금지 |
| 탐지/블러 검수 | 이전 완료 세대 또는 생성된 외부 작업 | 현재 완료 세대와 작업의 `privacy_detection_id` 확인. 승인 우회 금지 |
| 승인 보류(사고 대응) | `privacy_holds`의 hold 기록, auto_blurred 상태, 회수 대기 | `dlp privacy hold <세션> --reason`은 다시 실행해도 기록을 더하지 않고 회수 요청·렌더 무효만 다시 한다. 해제는 `--release --reason` 뒤 `approve` → `render`. 보류 기록을 DB에서 지우거나 privacy_state를 직접 고치지 않는다 (ADR 0042) |
| 세대 변경 | 이전 작업 완료 기록 | 새 세대 작업 생성 → 실제 전수 검수 → collect → approve → render. 옛 작업 재수집으로 승인하지 않음 |
| 블러 렌더 | 승인 상태, 누락/손상된 블러 파일·렌더 기록 | `privacy render` 재실행은 파일 해시/기록을 대조해 손상 산출물을 복구. 접근·네트워크 오류는 원인 해결 |
| CVAT/Label Studio 생성 | DB 롤백과 별개로 도구 작업/라벨링 버킷 사본 (고아) | `dlp review revoke --orphans [<세션>]`로 이름 규칙·DB 대조 결과를 보고, 확인 뒤 `--apply`로 지운다 (나이 하한 `review.yaml orphans.min_age_s`, 작업 생성 중인 세션은 건너뜀, F-REV-15). 확인 없이 create 반복 금지 |
| 웹훅 수집 | 미수집 작업, 또는 일부 이벤트 전달 실패 | 담당자 확인 후 `review collect <작업 키> --reviewer <ID>`. 도구 재시도만으로 전달 보장되지 않음 |
| 배포 매체 회수 | 커밋된 회수 요청과 남은 S3 사본·도구 작업 (회수 대기). 블러 변경 외에 사용 중지(`dlp dataset withdraw`)도 요청한다 | `dlp review revoke <세션>` 또는 `--pending` (미리보기는 `--dry-run`). 대기 중에는 그 세션의 verify가, CLI `dlp review create --stage labeling`·`assign`은 세션에 대기가 하나라도 있으면 대상 스트림과 무관하게 막힌다 (함수 `create_labeling_tasks`는 대상 스트림의 대기만 본다). 도구 인스턴스 폐기, 또는 실행기가 "도구 작업의 주인이 다르다"고 남긴 대기는 원래 작업이 없어진 것을 확인한 뒤 `--mark-tool-gone <작업> --note <사유>`. 대기 행을 DB에서 직접 지우지 않는다 |
| lakeFS 빌드 | 미커밋 객체 또는 DB에 없는 커밋 | 브랜치 차이·커밋과 DB 버전 대조. 실패 잔여 객체가 다음 스냅샷에 섞이지 않도록 정리 |
| 내보내기 | 선커밋된 이력·일부 파일·내부 가명 대응표 | manifest가 최종 전달 기준. 없는 결과는 미완료로 취급하고 현재 승인·사용 중지를 다시 확인 |
| 학습/배포 | MLflow 실행·산출물, 미승인/거절 모델 | DB 모델 상태·골든 평가·게이트 확인. 산출물 존재만으로 배포하지 않음 |
| 디스크 부족 | 미완료 산출물·살아 있는 서비스 볼륨 | 공간과 작업 상태 확인. `make clean`/볼륨 삭제로 복구하지 않음 |
| AI 1차 검수 실행 | 커밋한 판정 단위(추가 전용), 실행 기록과 started 이벤트(마감 없이 끝났으면 `status`가 "중단 추정"), 실행 임시 디렉터리의 블러본(정상 종료면 지움) | 같은 명령을 다시 돌린다: 기록된 단위는 묻지 않고 남은 단위만 묻는다. `--pending`은 판정 없는 대상이 남았거나 지금 버전 최근 실행이 completed가 아닌 세션을 고르고, 최근에 실패한 세션은 뒤로 미룬다. AI 표 행을 지우거나 고치지 않는다 (아래 "AI 1차 검수 실행기") |

렌더 재실행은 실제 파일을 읽어 검사하므로 정상 파일이어도 I/O 비용이 든다.
내보내기는 이력 선커밋 → 파일 업로드 → 최종 상태 재검사 → manifest 게시 순서를 보존한다.
분산 트랜잭션이 없으므로 모든 외부 실패가 자동 복구된다고 약속할 수 없다.

## AI 1차 검수 실행기 (WP17.5, ADR 0040)

- 재개: 단위 키는 판정 버전·대상·맥락·프레임 시각·렌더 해시·센서 해시의 해시라, 같은 입력이면 다시
  묻지 않는다. 판정은 라벨이 아니라 `retractions()`를 쓰지 않는다. 같은 세션에는 실행기 하나만
  돈다(세션 수준 advisory lock, 연결이 끊기면 풀린다). 상태 조회는 잠금을 잡지 않고 `pg_locks`만 본다.
- 차단기: 엔드포인트마다 연속 장애 `breaker_failures`번이면 열고, `breaker_cooldown_s` 뒤
  `/v1/models` 확인과 시험 요청 하나로 닫는다. 모두 `abort_after_s`보다 오래 막히면 묶음이 멈춘다
  (`EndpointPool.dead`, 종료 3, CLI가 남은 세션을 시작하지 않는다). 요청 하나만 재시도 끝에도
  장애면 그 단위만 남기고 다음 세션을 계속한다.
- 멈춤: `--max-minutes`·Ctrl-C·설정 오류·렌더 변경·사용 중지·승인 보류·버그는 새 요청을 멈춘다.
  묶음에서 보낼 곳을 기다리거나 백오프하던 요청도 멈춤 신호를 본다. 이미 보낸 요청은 받아 기록한다.
  `--parallel-sessions`의 Ctrl-C는 공유 신호로 모든 세션에 알리고 세션마다 interrupted:signal을 쓴
  뒤 끝난다. 작업자 스레드의 예외는 aborted:error로 멈춘다 (스레드가 조용히 죽지 않는다).
- 종료 코드(가장 심각한 것): 5 설정 > 3 장애 > 130 Ctrl-C > 1 버그 > 6 세션 전제·블러 변경·사용
  중지 > 7 처리 중 > 8 시간. `--pending`은 6·7을 0으로 세고, `--limit`은 실제로 실행한 세션만 센다.
  4는 회수 대기의 코드라 쓰지 않는다.
- 원본 접근은 장갑 Parquet 하나(감사 저장소, 용도 `aireview.run`)이고 블러본 게이트를 모두 통과한
  뒤에만 읽는다. `--dry-run`도 손 상태 대상이 있으면 읽는다 (단위 키에 센서 해시가 든다).
- DB 연결: 세션 하나가 최대 2개(실행 잠금 + 기록자)라 `--parallel-sessions K`면 연결 풀을
  `2K+1`(+넘침 10)로 잡는다. K는 PostgreSQL `max_connections`의 절반보다 작게 둔다.

## AI 1차 검수 표시 (WP17.6, ADR 0040)

- 설정: `review.yaml ai_review.mode`가 off면 AI 표를 읽지 않는다(`ReviewSetup.ai`가 None). 허용 판정 버전은
  `ai_review.versions`, 비면 aireview.yaml 지금 버전이고 CLI(`review_config.ai_display_config`)가 계산해
  넘긴다. 정책·모델을 바꿔 재판정하는 동안 이전 버전도 보이려면 두 버전을 함께 적는다. 웹훅 서버
  (`dlp review serve`)는 띄울 때 이 설정을 읽으므로 두 정책 파일을 바꾸면 다시 띄운다.
- Label Studio 설정의 AI 컨트롤(`ai_flags`·`ai_summary`·`ai_note`·`ai_decision`)은 모드와 무관하게 **늘**
  둔다. AI 영역이 든 주석이 있는 프로젝트에서 컨트롤을 빼면 설정 갱신이 400으로 거부되고 그 프로젝트의
  시간 작업 생성이 모두 멈춘다 (`test_ls_ai_feasibility.py`가 재현한다). 데이터 변수(`$…`)를 새로 쓰지
  않는다. 컨트롤 이름·라벨 값을 바꾸려면 새 프로젝트(온톨로지 프로젝트 이름)로 한다.
- 수집은 라벨 컨트롤(`labels`) 결과만 라벨로 읽고(F-AIR-02), 결정은 `ls_ai_decisions`로만 읽는다 (적용은
  WP17.7). 검수자가 AI 결과를 지우거나 AI 라벨로 구간을 그려도 수집 결과는 그대로다.
- 노출 기록(`ai_review_exposures`)은 작업 행과 같은 트랜잭션이라 생성이 롤백되면 함께 없어진다. 도구에
  남은 예측·이슈는 고아 작업 정리(`dlp review revoke --orphans`) 범위다. 이미 만든 작업에는 나중에 판정을
  붙이지 않는다 (노출 기록과 어긋나므로). 블러가 바뀌면 회수 뒤 `dlp aireview run <세션>`을 먼저 돌린다.
- required 게이트: `dlp review plan`이 종료 6으로 미루면 `dlp aireview run <세션>`을 돌리거나 서버 장애면
  `dlp aireview waive <세션> --reason …`. prelabeled 뒤 `max_wait_h`가 지나면 계획이 waived:timeout을 남기고
  진행한다. 블러 계획(`--privacy`)은 막지 않는다.
- Label Studio(지금 1.23.2)를 올릴 때 확인할 것: 같은 TimeSeries에 라벨 컨트롤 두 개, 예측 결과의
  `readonly`(구간 목록에 잠금 표시, Backspace로 지워지지 않음), 구간별 결정 Choice의 단축키(q·w, 동사
  단축키 1~9·shift와 겹치지 않음), 작업 단위 TextArea 예측, 테마 CSS의 선택자(`data-testid=textarea-form`·
  `htx-textbox-actions`, `.lsf-label`). Label Studio는 단축키가 없는 라벨에 남는 글자 키를 스스로 붙이는데
  AI 결정 키(q·w)는 그 후보에서 빠진다 (동사가 아닌 라벨의 자동 키가 WP17.6 전과 달라질 수 있다).
  services 테스트 두 파일과 실화면(구간 고르기 → q·w → 메모·요약 가독성, 두 모드)을 확인한다.
- CVAT(지금 v2.78.0)를 올릴 때: 이슈 API(`/api/issues`, 메시지는 `/api/comments`)와 테마 CSS의 이슈
  선택자(`cvat-objects-sidebar-issue-item`, `cvat-issue-dialog-*`, `cvat-hidden-issue-label`). CVAT 테마
  CSS를 바꾸면 `make cvat-up`으로 테마 이미지를 다시 빌드해야 화면에 반영된다.

## 웹훅과 승인 경계

웹훅은 성공/무시 204, 인증 실패 401, 담당자 불일치 403, 형식 오류 400/411, 본문 초과 413,
읽기 제한 408, 수집 실패/동시 연결 초과 503을 반환한다. 503에는 `Retry-After`가 붙는다.
영속 사건 큐는 미구현이다. 운영 HTTP 진입점은 TLS·속도 제한·헤더 수신 제한이 있는 게이트웨이와
방화벽으로 제한하며, 로컬 전용이면 `webhook.bind_host`를 `127.0.0.1`로 설정한다.

0013 이관은 옛 작업의 탐지 세대를 추측해 채우지 않는다. 기존 APPROVED 세션은 이관 시 유지하지만
재탐지 때 새 세대 검수가 필요하다. 복구를 위해 DB 상태를 직접 승인 값으로 바꾸지 않는다.
