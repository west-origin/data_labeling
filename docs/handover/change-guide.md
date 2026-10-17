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
| 바디캠 자세·월드 좌표 3D | `prelabel/campose.py`, `worldframe.py`, `runner.py`(`_PoseStage`), `prelabel.yaml camera_pose`·`world_frame` | 자세·월드 버전(정책 절 해시 + 입력 해시), 검수 보존, IMU 동기화 오프셋. `packages/prelabel/tests/test_campose.py`(합성 정답 ATE·RPE), services `test_runner.py::test_camera_pose_and_world_trajectories_are_idempotent_and_keep_reviewed`. 추정기 판(`imuvo1`)을 바꾸면 모든 세션의 검수 전 자세·월드 궤적이 다시 만들어진다 (ADR 0046) |
| 로봇 리타기팅 | `retarget`(robot·urdf·targets·ik·mink·retarget), `export/robot.py`, `scripts/retarget_mink.py`·`retarget-env`, `config/robots`, `retarget.yaml` | 모델 sha256·도달 표시(지어내지 않음)·manifest robot. `packages/retarget/tests`, `packages/export/tests/test_robot.py`, `make test-isolated`(mink, LeRobot 로봇 모드). 로봇 설정·정책·모델 해시가 리타기팅 판에 들어간다 (ADR 0047) |
| 3D 궤적 검수 물려받기 (ADR 0048·0050) | `schema/trajectory_follow.py`, `schema/db/trajectory_repository.py`, `schema/label_validation.py`(`_TrajectoryExemption`), `prelabel/lift3d.py`(`derived_from`), `prelabel/runner.py`(`_follow`·끊김 제외), `review/collect.py`·`verify.py`(자동 라벨), `review/ops/policy.py`·`review.yaml verify`, `export/source.py`·`runner.py`·`robot.py` | 명시 기록으로만 판정, 출처 2D가 현재·확인 상태일 때만, 지우는 것은 물려받기만 한 끊긴 3D뿐(`trajectory-follow-1`, 저장 검증이 같은 계획을 다시 계산), 기록 인자 기본값 없음, 카메라 자세는 검수 상태를 바꾸지 않고 완료 판정에서만 뺀다. `packages/schema/tests/test_trajectory_follow.py`, `packages/review/tests/test_trajectory_follow_flow.py`, prelabel `test_runner.py`(서비스), export `test_robot.py` |
| 검수 화면 테마 | `config/review_ui.yaml`, `review/ui_theme.py`, `services/review-theme/`, `cvat.sh` | 색·로고·단축키·모드 전환만. 라벨 이름·왕복 불변. `packages/review/tests/test_ui_theme.py`, `make contracts`(생성 파일 일치). CVAT·Label Studio 버전을 올리면 두 모드 화면을 열어 확인 (ADR 0038) |
| 원본 접근/보관 | `cli/raw_access.py`, `ops`, 버킷 정책 | 감사 우회·CVAT 사본 보관. `tests/test_raw_access_audited.py`, `tests/test_raw_access_db.py`, ops 테스트 |
| AI 1차 검수 실행 | `aireview/runner.py`, `backend.py`(`EndpointPool`), `media.py`, `status.py`, `cli/aireview_run_cmds.py`, `aireview.yaml` | 블러본 경계(I1)·판정 버전·재개·종료 코드. `packages/aireview/tests`(`test_runner*.py`, `test_pool.py`, `test_status.py`, `test_blurred_only.py`), `packages/cli/tests/test_aireview_run_cmds.py`, `tests/test_aireview_boundaries.py`, `tests/test_aireview_isolation.py`, services `test_runner_pg.py`·`test_aireview_tables.py`. 판정 버전 해시에서 빠지는 절(`selection`·`kinds`·`pending`·`server`·`gate`·`estimate`) 밖을 바꾸면 모든 단위를 다시 판정한다 (ADR 0040 4.3) |
| AI 1차 검수 표시 | `review/ai_display.py`, `review/labelstudio.py`(AI 컨트롤), `review/tasks.py`(노출 기록), `review/ops/runner.py`·`results.py`(우선순위·표본 제외·마무리 보호), `cli/review_config.py`·`review_cmds.py`(설정·required 게이트), `review.yaml ai_review`, `review_ui.yaml ai` | 표시는 지금 렌더의 판정만(I10), 증가(표본 제외·보호·우선순위)는 렌더 무관(I9), AI 컨트롤 상시, 판정 없으면 예측·데이터 불변. `packages/review/tests/test_ai_display.py`, `packages/cli/tests/test_review_ai_cmds.py`, services `test_ls_ai_feasibility.py`·`test_ai_display_flow.py`. Label Studio·CVAT 버전을 올리면 아래 "AI 1차 검수 표시" 확인 목록 |
| AI 1차 검수 수집·품질 | `review/ai_collect.py`, `review/collect.py`(`_outcome_for`), `review/reconcile.py`(`same_as_sent`), `review/ops/ai_quality.py`, `ops/metrics.py`(`ai_weekly`), `aireview/evaluate.py`, `cli/aireview_eval_cmds.py`·`ai_quality_print.py`, `ops.yaml alerts.ai_*`, `aireview.yaml gate` | 제안은 사람이 적용할 때만·DB에서 다시 검증(I6), 적용 실패는 수집을 막지 않음, 편향 없는 지표는 대조군만, eval은 DB에 쓰지 않음. `packages/review/tests/test_ai_collect.py`·`test_ai_quality.py`, `packages/ops/tests/test_ai_weekly.py`, `packages/aireview/tests/test_evaluate.py`, `packages/cli/tests/test_aireview_eval_cmds.py`·`test_review_ai_cmds.py`, services `test_ai_display_flow.py` |
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
| AI 제안 적용 수집 | 없음 (수집 한 트랜잭션: 라벨·검수 상태·결과 기록이 함께 커밋되거나 함께 롤백된다) | 수집을 다시 한다. 세션 검증에 걸린 적용은 되돌려 수집이 계속되고, 검수자 자신의 결과가 걸린 경우만 실패한다 (그 구간을 고쳐 다시 제출) |
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
- 수집은 라벨 컨트롤(`labels`) 결과만 라벨로 읽고(F-AIR-02), 결정은 `ai_decision` 선택만 읽는다 (적용은
  아래 "AI 1차 검수 수집·적용"). 검수자가 AI 결과를 지우거나 AI 라벨로 구간을 그려도 라벨 결과는
  그대로다 (그린 AI 구간은 경고 수로만 남는다).
- 구간을 고르면 보이는 칸(AI 메모·결정, 이벤트 속성, 손 상태 입력란)은 라벨 컨트롤 구간을 골랐을 때만
  보인다 (`whenTagName="labels"`). AI 구간을 고르면 안내 한 줄만 보인다. 결정 단축키는 칸이 숨어도
  눌리므로, AI 구간을 고른 채 누른 결정은 AI 구간 id(`ai-<해시 8자>`)에 붙는다. 라벨 ID별 결정은
  `ai_display.ls_label_decisions(results, exposure, sent_ids)`로 얻는다: 노출 기록의 보인 판정으로 AI 구간
  id를 원 라벨로 되돌리고, 작업 라벨이 아닌 구간의 결정과 한 라벨의 서로 다른 결정은 경고로 남기고
  버린다. WP17.7 적용은 이 함수를 쓴다 (`ls_ai_decisions`의 id를 라벨 ID로 쓰지 않는다).
- 판정률(`coverage`, 요약의 "판정률")의 분모는 판정 실행과 같은 대상 조건
  (`dlp_schema.aireview.is_judge_target`, 정책 값은 aireview.yaml에서 CLI가 `AiDisplayConfig.targets`로
  넘긴다)이다. 장갑 근거 손 상태·센서 출처 라벨·긴 상위 구간·원본 접근 이벤트는 세지 않는다. 대상 조건을
  바꿀 때는 이 함수 하나를 고친다.
- 노출 기록(`ai_review_exposures`)은 작업 행과 같은 트랜잭션이라 생성이 롤백되면 함께 없어진다. 도구에
  남은 예측·이슈는 고아 작업 정리(`dlp review revoke --orphans`) 범위다. 이미 만든 작업에는 나중에 판정을
  붙이지 않는다 (노출 기록과 어긋나므로). 블러가 바뀌면 회수 뒤 `dlp aireview run <세션>`을 먼저 돌린다.
- required 게이트: `dlp review plan`이 종료 6으로 미루면 `dlp aireview run <세션>`을 돌리거나 서버 장애면
  `dlp aireview waive <세션> --reason …`. prelabeled 뒤 `max_wait_h`가 지나면 계획이 waived:timeout을 남기고
  진행한다. 블러 계획(`--privacy`)은 막지 않는다.
- 단축키: Label Studio는 단축키 없는 라벨(review_ui.yaml `hotkeys`에 없는 원시 동작 pull·rotate, 사이
  구간, 손 상태, skill 앞쪽)에 자동 키 후보 `1234567890qwetasdfgzxcvbyiopjklnm`에서 남는 키를 설정
  순서대로 붙인다 (`ui_theme.LS_AUTO_HOTKEYS`). 명시 단축키를 이 후보 안의 키로 새로 주면 그 뒤 라벨의
  자동 키가 한 칸씩 밀려, 검수자가 누르던 키가 조용히 다른 라벨(손이 바뀐 라벨 포함)을 붙인다. WP17.6
  처음 값(q·w)이 그랬다 (22개 라벨, 오른손 pull q→e 등). 그래서 AI 결정 키는 후보 밖의 r(적용)·h(기각)이고
  테마 로더가 후보·내장 한 글자 단축키(u, [, ])를 거부한다. 지금 자동 키는 WP17.6 전과 같다 (오른손 pull q·
  rotate w, 왼손 pull e·rotate t, gap.right:idle a …). 동사 단축키 표(`hotkeys`)에 키를 더하거나 빼면 같은
  이유로 자동 키가 밀리므로 검수자에게 알린다. 자동 키 규칙은
  `test_ai_display.py::test_ai_controls_do_not_shift_label_studio_auto_hotkeys`가 시뮬레이터로 지키며, 이
  시뮬레이터는 실화면 범례와 같다 (아래 확인 기록). 범례 순서는 설정을 만든 온톨로지의 사전 순서이고, 작업
  생성은 DB에 등록한 온톨로지(PostgreSQL JSONB 키 순서: 짧은 이름 먼저)를 쓰므로 파일에서 읽은 온톨로지와
  자동 키 배치가 다를 수 있다 (같은 DB 온톨로지면 늘 같다).
- Label Studio(지금 1.23.2)를 올릴 때 확인할 것: 같은 TimeSeries에 라벨 컨트롤 두 개, 예측 결과의
  `readonly`(구간 목록에 잠금 표시, Backspace로 지워지지 않음), 구간별 결정 Choice의 단축키(r·h), 자동 키
  후보(`makeComb`)와 내장 단축키 표가 `ui_theme.LS_AUTO_HOTKEYS`·`LS_RESERVED_KEYS`와 같은지, `View
  whenTagName`(고른 구간의 라벨 컨트롤 이름과 비교), 작업 단위 TextArea 예측, 테마 CSS의 선택자
  (`data-testid=textarea-form`·`htx-textbox-actions`, `.lsf-label`). services 테스트 두 파일과 아래 실화면
  확인을 다시 한다.
- CVAT(지금 v2.78.0)를 올릴 때: 이슈 API(`/api/issues`, 메시지는 `/api/comments`)와 테마 CSS의 이슈
  선택자(`cvat-objects-sidebar-issue-item`, `cvat-objects-sidebar-issues-list`, `cvat-issue-dialog-*`,
  `cvat-hidden-issue-label`). CVAT 테마 CSS를 바꾸면 `make cvat-up`으로 테마 이미지를 다시 빌드해야 화면에
  반영된다.

### 검수 화면 AI 표시 실화면 확인

Playwright(전역 node playwright, `PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers`)로 개발 서비스의 실화면을
띄워 확인한다. 화면 데이터는 합성 세션(`test_ai_display_flow.py`와 같은 절차: 합성 블러 영상·장갑 세션
승인·렌더 → 모델 출처 행동·사이 구간 → 판정 단위 → `create_labeling_tasks`)이고 실제 사용자·개인정보는
쓰지 않는다. 스크린샷 파일은 저장소에 넣지 않는다 (확인한 사람이 따로 보관한다).

| 확인 항목 | 기대 | 2026-10-12 결과 (LS 1.23.2, CVAT v2.78.0) |
| --- | --- | --- |
| 라벨 구간(지적 있음)을 고름 | AI 메모·결정 칸, 이벤트·손 상태 입력란이 보인다 | 그대로 |
| 그 구간에서 r | "제안 적용" 체크, 결정이 라벨 구간 id에 붙는다 | 그대로 |
| AI 구간(AI:의심)을 고름 | 메모·결정·입력란이 숨고 안내 한 줄만 보인다 | 그대로 |
| AI 구간에서 h | 결정이 AI 구간 id에 붙고(Info 칸에 Choices 표시) `ls_label_decisions`가 원 라벨로 되돌린다 | 제출 뒤 API 결과로 확인, 경고 0 |
| 구간을 고르지 않고 r·h | 아무것도 바뀌지 않는다 | 라벨 결과 그대로 |
| 범례 단축키 | AI 컨트롤이 없을 때와 같다 (r·h는 자동 키 후보 밖) | 108개 라벨이 시뮬레이터 결과와 같고, 파일 순서 온톨로지의 WP17.6 전 실화면 범례와도 같다 |
| 다크·화이트 모드 | 두 칸의 글자가 바탕에 묻히지 않는다 | 그대로 |
| CVAT Issues 탭 | 카드와 아래 목록 칸이 두 모드 패널 색이다 | 저장소 CSS를 얹어 확인 (실행 중 이미지는 `make cvat-up` 뒤 반영) |

## AI 1차 검수 수집·적용 (WP17.7, ADR 0040)

- 흐름(`collect_task`): `ai_collect.begin`(노출 기록·판정 단위 읽기, AI 끔이거나 노출 기록이 없으면 None) →
  도구 결과 → `ls_decisions`(라벨 ID별 결정) → `apply_decisions` → `_outcome_for`(reconcile → stale →
  측정·오류 삽입 표시) → `settle` → `insert_labels`·`record_review` → `record_outcome`(배정 마무리 앞). 모든
  AI 호출은 `ai`가 None이면 아무것도 하지 않는다 (분기 없이 부른다). 수집은 한 트랜잭션이라 실패하면
  결과 기록도 남지 않는다.
- 적용 표(검수자 행동 → 결과 레코드 → `AiOutcomeItem`)는 `dlp_review.ai_collect` 모듈 docstring이 기준이고
  `test_ai_collect.py`가 행마다 지킨다. 적용 규칙을 바꾸면 그 표와 테스트를 함께 고친다.
- 적용이 세션 검증에 걸리면 라벨 ID 순으로 하나씩 더해 보며 통과하는 것만 남긴다 (적용 수만큼 세션 검증을
  다시 한다). 검수자 자신의 결과가 걸리면 지금처럼 `OntologyViolationError`로 수집이 실패한다.
- 제안 일치(`match`)는 제안 페이로드가 원 라벨과 다른 필드만 본다 (exact: 모두 같음, partial: 일부).
  "replaced"는 적용한 종류 변경, 또는 원 라벨이 지워지고 같은 손·같은 구간에 다른 종류의 새 사람 레코드가
  있을 때다 (Label Studio에서 구간 라벨만 바꾼 경우. 스트림은 둘 다 있을 때만 비교한다: 새로 그린 시간
  구간은 스트림이 없다). 검수자가 제안대로 직접 종류를 바꿔 붙였으면 적용 결과도 applied다.
- 측정이 아닌 작업(표준·QA·오류 삽입)에서 보낸 뒤 다른 단계가 바꾼 라벨(이력에 자식이 있음,
  `collect.drop_retracted`와 같은 기준)에는 적용 결정이 있어도 항목을 바꾸지 않는다 (invalid_now). 종류 변경
  제안을 그대로 적용하면 원 라벨 없는 새 레코드가 drop_retracted를 지나 지운 라벨이 되살아나기 때문이다.
  측정 작업(블라인드·이중)은 이 검사를 하지 않는다: 비교 기준이 빈 목록이라 drop_retracted가 빼는 결과가
  없고 결과가 parent 없는 측정 레코드라 운영 라벨을 되살리지 않는다. 막으면 짝 표준 작업이 먼저 고친 라벨
  (AI가 지적해 고친 라벨이 대부분이다)에서 이중 검수자의 적용이 버려져 보인 집단의 이중 일치 κ가
  체계적으로 낮아진다. AI 없이 검수자가 stale 라벨의 종류를 직접 바꿔 붙인 경우의 같은 문제는
  F-AIR-13(기존 수집 경로)이다.
- 종류 변경을 적용하면서 검수자가 같은 구간에 같은 내용을 직접 그렸으면 새 항목을 더하지 않는다 (같은
  label_id 레코드 둘 방지). 세션 검증(`settle`)은 새 레코드 ID 중복도 실패로 보고 AI 적용만 되돌린다.
- Label Studio 결과의 원래 라벨 ID 기준은 작업에 보낸 라벨이다 (이중 작업 포함). 이중·블라인드의 비교
  기준은 빈 목록이라 결과는 모두 새 측정 레코드다.

## AI 1차 검수 품질·평가 (WP17.8, ADR 0040)

- `dlp review quality --ai`: 기간은 결과 기록의 수집 시각과 노출 기록의 작업 생성 시각 기준
  [`--since`, `--until`)이다. 대조군(holdout) 줄이 편향 없는 값이고 보인 집단(shown) 줄은 자동화 편향
  감시용이다 (같은 줄에 섞지 않는다). 분모 밖 행 수(판정 없음·재검수·stale·측정)를 함께 낸다.
  `--ai-thresholds P`는 조안만 내고 정책을 바꾸지 않는다 (사람이 review.yaml을 고친다). 같은 확신도의
  판정은 함께 보이거나 함께 숨으므로 조안은 관측된 확신도 값 중에서 고른다. `--json`·`--ai-thresholds`·
  `--since`·`--until`은 `--ai`가 필요하다.
- 판정 지표는 표본 가중이다 (`AiRow.weight`: 표본 라벨 = 묶음 모집단/표본, 전수 = 1). 계획 단계가 AI
  지적 라벨을 표본에서 빼 전수 검수하므로(집단과 무관) 행을 그대로 세면 대조군 재현율·κ가 부풀려진다.
  가중치가 섞인 비율은 `가중 num/den, 유효 n`으로 보이고 상한은 Korn-Graubard 근사다. 혼동 행렬·행 수와
  제안 적용 비율은 개수 그대로다. 표본 정책(`sampling`)이나 AI 표본 제외(`ai_review.sampling_exclude`)를
  바꾸면 가중치 계산(`ops/ai_quality._stratum`)이 그 설계와 맞는지 본다.
- 주간 AI 절은 `review.yaml ai_review.mode`가 off면 없다. 백로그는 주 끝 시점의 생애주기·실행 기록으로 세고
  프라이버시 승인 상태만 지금 값이다. 완료는 required 게이트와 같이 전체 완료(`is_full_completion`)만 센다
  (좁힌 실행의 `partial:` 완료만 있으면 백로그다). 장비 시간은 끝난 실행의 걸린 시간 합이라 세션을 함께 돌리면 크게
  나온다 (상한).
- `dlp aireview eval`: 골든셋 세션마다 블러본 게이트(`open_blurred`)를 거치므로 승인·렌더된 세션만
  판정한다 (아니면 건너뛴 사유를 보고서에 남긴다). 판정 버전은 지금 aireview.yaml 값이고 실행 기록·단위는
  메모리에만 남는다. 모델·프롬프트·프레임 설정을 바꾸기 전후로 같은 `--seed`·`--inject`로 돌려 비교한다.
  게이트 차단은 `gate.block`으로만 켠다 (지금 false, F-AIR-10). ok 놓침률은 품질 보고와 같은
  P(틀린 라벨 | AI ok)이고(교란 비율에 따라 달라진다), P(AI ok | 틀린 라벨)은 `wrong_called_ok_rate`로
  보고서에만 있다. history 방식의 정답은 대상 후손 사슬의 지금 상태(QA 재수정·삭제 반영)이고 정답으로 쓴
  레코드(후손, 같은 손·구간에 다른 종류로 다시 그린 레코드)는 맥락에서 뺀다. `gate.inject`의 모르는 교란
  종류는 정책 로드 오류다.

## 웹훅과 승인 경계

웹훅은 성공/무시 204, 인증 실패 401, 담당자 불일치 403, 형식 오류 400/411, 본문 초과 413,
읽기 제한 408, 수집 실패/동시 연결 초과 503을 반환한다. 503에는 `Retry-After`가 붙는다.
영속 사건 큐는 미구현이다. 운영 HTTP 진입점은 TLS·속도 제한·헤더 수신 제한이 있는 게이트웨이와
방화벽으로 제한하며, 로컬 전용이면 `webhook.bind_host`를 `127.0.0.1`로 설정한다.

0013 이관은 옛 작업의 탐지 세대를 추측해 채우지 않는다. 기존 APPROVED 세션은 이관 시 유지하지만
재탐지 때 새 세대 검수가 필요하다. 복구를 위해 DB 상태를 직접 승인 값으로 바꾸지 않는다.
