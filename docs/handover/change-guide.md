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
| 세대 변경 | 이전 작업 완료 기록 | 새 세대 작업 생성 → 실제 전수 검수 → collect → approve → render. 옛 작업 재수집으로 승인하지 않음 |
| 블러 렌더 | 승인 상태, 누락/손상된 블러 파일·렌더 기록 | `privacy render` 재실행은 파일 해시/기록을 대조해 손상 산출물을 복구. 접근·네트워크 오류는 원인 해결 |
| CVAT/Label Studio 생성 | DB 롤백과 별개로 도구 작업/라벨링 버킷 사본 (고아) | `dlp review revoke --orphans [<세션>]`로 이름 규칙·DB 대조 결과를 보고, 확인 뒤 `--apply`로 지운다 (나이 하한 `review.yaml orphans.min_age_s`, 작업 생성 중인 세션은 건너뜀, F-REV-15). 확인 없이 create 반복 금지 |
| 웹훅 수집 | 미수집 작업, 또는 일부 이벤트 전달 실패 | 담당자 확인 후 `review collect <작업 키> --reviewer <ID>`. 도구 재시도만으로 전달 보장되지 않음 |
| 배포 매체 회수 | 커밋된 회수 요청과 남은 S3 사본·도구 작업 (회수 대기). 블러 변경 외에 사용 중지(`dlp dataset withdraw`)도 요청한다 | `dlp review revoke <세션>` 또는 `--pending` (미리보기는 `--dry-run`). 대기 중에는 그 세션의 verify가, CLI `dlp review create --stage labeling`·`assign`은 세션에 대기가 하나라도 있으면 대상 스트림과 무관하게 막힌다 (함수 `create_labeling_tasks`는 대상 스트림의 대기만 본다). 도구 인스턴스 폐기, 또는 실행기가 "도구 작업의 주인이 다르다"고 남긴 대기는 원래 작업이 없어진 것을 확인한 뒤 `--mark-tool-gone <작업> --note <사유>`. 대기 행을 DB에서 직접 지우지 않는다 |
| lakeFS 빌드 | 미커밋 객체 또는 DB에 없는 커밋 | 브랜치 차이·커밋과 DB 버전 대조. 실패 잔여 객체가 다음 스냅샷에 섞이지 않도록 정리 |
| 내보내기 | 선커밋된 이력·일부 파일·내부 가명 대응표 | manifest가 최종 전달 기준. 없는 결과는 미완료로 취급하고 현재 승인·사용 중지를 다시 확인 |
| 학습/배포 | MLflow 실행·산출물, 미승인/거절 모델 | DB 모델 상태·골든 평가·게이트 확인. 산출물 존재만으로 배포하지 않음 |
| 디스크 부족 | 미완료 산출물·살아 있는 서비스 볼륨 | 공간과 작업 상태 확인. `make clean`/볼륨 삭제로 복구하지 않음 |

렌더 재실행은 실제 파일을 읽어 검사하므로 정상 파일이어도 I/O 비용이 든다.
내보내기는 이력 선커밋 → 파일 업로드 → 최종 상태 재검사 → manifest 게시 순서를 보존한다.
분산 트랜잭션이 없으므로 모든 외부 실패가 자동 복구된다고 약속할 수 없다.

## 웹훅과 승인 경계

웹훅은 성공/무시 204, 인증 실패 401, 담당자 불일치 403, 형식 오류 400/411, 본문 초과 413,
읽기 제한 408, 수집 실패/동시 연결 초과 503을 반환한다. 503에는 `Retry-After`가 붙는다.
영속 사건 큐는 미구현이다. 운영 HTTP 진입점은 TLS·속도 제한·헤더 수신 제한이 있는 게이트웨이와
방화벽으로 제한하며, 로컬 전용이면 `webhook.bind_host`를 `127.0.0.1`로 설정한다.

0013 이관은 옛 작업의 탐지 세대를 추측해 채우지 않는다. 기존 APPROVED 세션은 이관 시 유지하지만
재탐지 때 새 세대 검수가 필요하다. 복구를 위해 DB 상태를 직접 승인 값으로 바꾸지 않는다.
