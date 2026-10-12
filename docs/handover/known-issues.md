# 현재 검증 결과와 남은 과제

[인수인계 첫 화면](README.md) · [변경 및 복구 안내](change-guide.md)

기준일: **2026-10-12**. 아래는 코드와 실행 검사로 확인한 상태다. 개발 환경에서의 통과는
실영상 정확도·다중 고객 격리·상용 출시 승인에 대한 증거가 아니다.

알려진 문제 전체의 단일 목록은 [레지스트리](issues.yaml)다 (S·R·T·O·A·F 항목, 상태·심각도·관련 파일·
재현 테스트). 아래 표의 ID도 모두 레지스트리에 있고 `tests/test_issue_registry.py`가 대조한다.

## 2026-10-12 AI 1차 검수(WP17) 검증 결과

[ADR 0040](../adr/0040-ai-first-pass-review.md)의 WP17.0~17.8(판정 실행·표시·수집·품질·골든 평가)을 구현했다.
판정은 참고용이고 라벨·검수 상태가 아니다. 검수자가 고른 제안만 사람 레코드가 된다(I6). 아래 검사는 개발
가상 환경으로 `make check`·`make contracts`·`make test-services`와 같은 명령을 직접 돌린 결과다. 검사 대상은 합성
장면과 가짜 OpenAI 호환 서버다. 실제 DGX Spark·vLLM의 처리량과 판정 정확도는 파일럿(WP17.9)에서 잰다.

| 검사 | 결과 | 범위·한계 |
| --- | --- | --- |
| 기본 테스트 | **1085 passed, 6 skipped, 162 deselected, 6 xfailed**, 439초 | 6 skipped는 모델 가중치 없음. xfailed는 알려진 결함 재현. 서비스/격리 환경 제외. WP17.7·17.8 검증 지적 반영(stale 적용 차단, 표본 가중 품질 지표, 골든 평가 정답·ok 놓침 정의) 뒤 값 |
| 서비스 테스트 | **159 passed, 1 skipped**, 437초 | 실제 PostgreSQL·Label Studio 1.23.2·CVAT·저장소. AI 종단(LS 작업 생성 → 결정이 든 주석 → 수집 → 사람 레코드·`ai_review_outcomes`) 포함. 생략 1건은 외부 큐레이션 MongoDB URI 없음 |
| ruff·포맷·pyright strict | 오류 0 | `noqa: C901` 기준선은 처음 13개에서 지금 **11**개다 (이번 묶음에서 `collect_task` 분해로 하나 줄었다) |
| 계약 검사 | 온톨로지·JSON Schema·테마 색·모델 라이선스 통과 | 이번 묶음은 계약 타입을 바꾸지 않았다. DB는 Alembic 0019(`ai_review_*` 추가 전용 표) |
| 격리 환경(LeRobot) | 이번 미실행 | AI 1차 검수와 무관 |

남은 과제(레지스트리 등록): 행동 VLM 직렬 호출(F-AIR-01), 오류 삽입 작업의 AI 표시·사본 판정(F-AIR-03),
AI 충돌 기반 표적 QA(F-AIR-04), AI 표 파티션·보관 ADR(F-AIR-05), CVAT 2단계 표시(F-AIR-06), 경계 이동 제안
적용(F-AIR-07), 누락 발견 표시(F-AIR-08), 장갑+영상 융합 접촉 판정(F-AIR-09), 골든 게이트 차단 여부(F-AIR-10,
지금은 경고만), 층화 ok 놓침 상한의 보수적 근사(F-AIR-11), AI 끔 수집에서 결정 무시(F-AIR-12), stale
라벨의 종류를 Label Studio에서 직접 바꿔 붙이면 원 라벨 없는 새 레코드가 남는 기존 수집 결함(F-AIR-13, AI 적용
경로는 막았다. 재현은 `tests/test_known_defects.py::test_relabel_of_a_retracted_label_does_not_revive_it`), 골든 평가 history 맥락에
남는 사람 추가 레코드(F-AIR-14).
Label Studio 수집의 `from_name` 문제(F-AIR-02)는 고쳤다.

판정 품질 지표는 표본 가중이다: 계획 단계가 AI 지적 라벨을 집단과 무관하게 표본에서 빼 전수 검수하므로
대조군도 행을 그대로 세면 재현율·κ가 부풀려진다. 골든 게이트의 ok 놓침률은 품질 보고와 같은
P(틀린 라벨 | AI ok)다.

## 2026-10-11 전수 검수와 외주 준비

- 6개 영역 전 파일을 전수 검수했다. 새 발견 71건(상 4·중 17·하 50)은 [전수 검수 결과](full-review-2026-10-11.md)와
  레지스트리 `F-*`에 있다. 이번에는 고치지 않았고, 14건은 `tests/test_known_defects.py`에 `xfail(strict=True)`로 재현했다.
- 외주 검수·리팩토링을 위해 [인수 기준](acceptance.md), [불변 조건과 강제 수단](invariants.md), PR 양식,
  패키지 의존 검사(`tests/test_import_boundaries.py`), 함수 복잡도 상한(ruff C901, 기존 13개는 `noqa` 기준선),
  `make coverage`, `make review-scope`를 추가했다. CI의 uv 버전을 고정했다.

| 검사 | 결과 | 범위·한계 |
| --- | --- | --- |
| `make check` | **526 passed, 122 deselected, 14 xfailed**, 231초 | 14 xfailed는 알려진 결함 재현(의도된 실패). 서비스/격리 환경 제외 |
| 검수·테마 서비스 검사 | `packages/review`·`tests` 서비스 표시 **46 passed** | 실제 CVAT·Label Studio·PostgreSQL. 전체 `make test-services`는 이번 미실행 |

## 2026-10-11 온톨로지 개선 검증

[ADR 0037](../adr/0037-ontology-validation-boundaries.md)과 [용어·확장 안내](../ontology-guide.md)에
따라 저장·버전 선택·구간·이관·검수 화면 분리를 보강했다. 기존 모델 미준비 거부 검사도 유지했다.

| 검사 | 결과 | 범위·한계 |
| --- | --- | --- |
| `make check` | **501 passed, 1 skipped, 122 deselected**, 190.63초. ruff·포맷·pyright·계약 검사 통과 | 실제 깊이 모델 1건 생략; 서비스/격리 환경은 별도 |
| `make test-services` | **120 passed**, 146.70초 | 실제 PostgreSQL·CVAT·Label Studio·저장소 등. LeRobot 격리 환경 이번 미실행 |
| 온톨로지 경계 회귀 | **19 passed**, 2.57초 | 위 기본/서비스 검사와 일부 중복. 마지막 이관 배치 검증을 포함해 모두 실행 |
| 개발 DB 적용 | Alembic **0014**, v1 등록과 INSERT 버전 트리거 확인 | 현재 개발 DB의 감사 대상 세션은 0개. 실제 기존 고객 데이터의 무결성 확인을 뜻하지 않음 |

`make schemas`를 실행했고 JSON 스키마의 저장 형태는 바뀌지 않았다. 기존 위반은 자동 수정하지
않는다. 실제 청소 하위 단계·용어 확정·실영상 예시·전체 이관 실행은 [용어 안내](../ontology-guide.md)의
현장 검수 절차에 따라 추가한다. 이번 기본 검사와 감사에서는 `DLP_REVIEW_POLICY_FILE`을 해제해
기본 정책을 사용했다. 환경별 로그는 저장소 밖 `.cloud-onboarding/data-labeling`에 보관했다.

## 2026-10-10 검증 결과와 한계

이번 인수인계 재검수에서 `make check`·`make test-services`·청소 파일럿 준비 검사를 다시 실행했다.
웹훅 HTTP 14건은 이번 기본 검사에도 포함됐다. 입력 영상·전 소스 주석 검토·의존성 감사는 같은 날의 앞선 기록을
참조한다. 새 개발 호스트의 설치·복원 성공은 아직 별도 확인 대상이다.

| 검사 | 확인 결과 | 검사 밖에 남은 것 |
| --- | --- | --- |
| `make check` | 489 passed, 1 skipped, 115 deselected; 린트·타입·계약 포함 통과 | 깊이 실제 모델 1건 생략; 서비스/격리 환경 제외 |
| `make test-services` | 113 passed, 0 skipped, 492 deselected | LeRobot 격리 환경 2건 이번 미실행 |
| 웹훅 실제 HTTP | 14건 통과; 과부하 슬롯 해제와 재시도 검사 | 영속 사건 전달·운영 게이트웨이·장기 부하 |
| 청소 파일럿 준비 | 실제 CPU 어댑터 합성 추론, DB 0013, 7개 서비스, 분리된 CVAT 일반 계정, Label Studio 서비스 인증 | 실제 현장 라벨 정답·누락률·검수 시간·학습 향상 |
| 입력 진단 | 제공 영상의 정규화 사본 571프레임 PTS/방향 검사 | 전체 현장 영상 처리·사람의 개인정보 검수 |
| 소스·주석 재검수 | 전 소스 구문/주석 수집과 주요 위험 경로 수동 검토 | 모든 실행 경로에 대한 형식 검증이나 무결함 보장 아님 |
| 의존성 감사 | 결과 176항목 중 158개 감사, 18개 제외; 취약점 2건 잔존 | 로컬 패키지 17개와 Git TrackEval 1개는 PyPI 감사 제외 |

환경별 상세 로그는 저장소 밖, 보고서는 Git에서 제외된 `reports/`에 있다. 인수에 필요한 결과는
이 표에 남긴다. 새 검사는 명령·환경·생략 사유와 함께 갱신한다.

## 인수인계 문서 재검수

인수인계 문서와 연결된 README·CLAUDE·Makefile·구현 계획·파일럿 절차를 코드와 대조해 다음을 정정했다.

- 테스트가 끝난 뒤 검수 정책 연결이 사라지던 명령을 서브셸로 분리했다. 첫 검사 실패 코드도 보존한다.
- FiftyOne 데이터셋 쓰기·읽기가 기본 테스트에 포함되므로 사용 가능한 Mongo가 필요함을 명시하고 ping 절차를 추가했다.
- 정수 ms인 계약 라벨 시각과 실수인 동기화 계산·일부 시계열을 구분했다.
- CA 예제의 꺾쇠 괄호가 셸 리다이렉션으로 해석되던 문제를 실행 인자 비교로 확인하고 경로 예제로 바꿨다.
- CVAT를 기본 서비스와 별도로 기동하는 실제 절차에 맞춰 구현 계획을 정정했다.
- 공통 `--session` 옵션 예시를 실제 CLI 형식으로 바꾸고, Prefect 연결을 향후 구현 사항으로 표시했다.

문서 12개에서 파일 링크 133개·제목 링크 3개·셸 예제 15개를 확인했다. CLI 예제 15개는 실제 파서로
인자 해석을 검사했고, 구조 표는 17개 패키지의 소스 import와 대조했다. 새 호스트 설치 재현과
실제 현장 정확도는 이 문서 검수 범위에 포함하지 않는다.

## 외부 서비스 제공 전 해결할 과제

| ID | 확인된 문제·관련 코드 | 완료 기준 |
| --- | --- | --- |
| S1 | 인증/테넌트 경계 미구현. [raw_access](../../packages/cli/src/dlp_cli/raw_access.py)의 `DLP_ACTOR`는 자기 신고 | 인증 주체와 감사 연결; 고객별 DB·객체·검수 도구 권한 경계와 다른 고객 접근 거부 검사 |
| S2 | CVAT 원본 사본 자동 삭제 없음. [review tasks](../../packages/review/src/dlp_review/tasks.py), [ops](../../packages/ops/src/dlp_ops/). 삭제 수단은 ADR 0039의 `CvatClient.delete_task`·`task_snapshot`을 다시 쓸 수 있다 | 보관 만료·사용 중지 시 도구 사본까지 삭제, 실패 재시도·증거 기록. 파일럿은 권한자가 수동 확인 |
| S3 | FiftyOne GraphQL 의존성 2건 미패치(아래 상세) | 호환 패치 후 실제 서버 import·기동·관련 보안 회귀·감사 재실행; 해결 전 해당 GUI 외부 제공 보류 |
| S4 | `commercial: review` 모델 사용 허용은 법적 검토 완료가 아님. [모델 목록](../../config/models.yaml) | 모델별 가중치/학습 데이터 조건에 대한 상업 이용 결론을 기록하고 실제 사용할 정책에 반영 |
| S5 | 영속 작업 큐·백업 복구·장기 부하 검증 미완료. [webhook](../../packages/review/src/dlp_review/webhook.py), 개발 Compose | 중단 재개·사건 재전달·복원·부하 목표와 실제 검증 기록. 파일럿 수동 운영과 분리 |

FiftyOne 1.22.1이 요구하는 Strawberry GraphQL 0.316.0에는
[CVE-2026-107727 / GHSA-m952-2w3f-6r8h](https://github.com/advisories/GHSA-m952-2w3f-6r8h)과
[CVE-2026-107728 / GHSA-pfvf-fwfp-25mp](https://github.com/advisories/GHSA-pfvf-fwfp-25mp)가 남아 있다.
첫 건은 legacy `graphql-ws`의 완료 구독 상태가 남아 설정된 구독 한도를 소진하는 문제이고,
둘째는 동기 권한 함수가 awaitable을 반환하는 조건에서 발생하는 권한 검사 우회다.
모든 FiftyOne 경로에서 악용 가능하다는 판정은 아니다. 현재 파일럿 절차는 그 웹 서버를 시작하지 않는다.

패치 0.327.2로 강제 변경해 본 결과 FiftyOne 서버의 `strawberry.extensions.Extension` import가
실패했다. 임시 override는 철회했으며 잠금 파일은 0.316.0이다. 설치 성공이나 CLI import만으로
호환 패치가 됐다고 판정하지 않는다. [ADR 0036](../adr/0036-model-cache-and-render-recovery.md) 참고.

## 다음 리팩토링의 우선순위

| ID | 현재 문제·근거 | 권장 변경과 완료 기준 |
| --- | --- | --- |
| R1 / 반영 | [insert_labels](../../packages/schema/src/dlp_schema/db/repository.py)의 배치 전체 의미·버전·구간 검증, Alembic 0014 버전 트리거 추가 | QA 오류·삭제 이력 구분, 부분 소속 검사, 회귀 검사. 기존 데이터는 `dlp ontology audit`로 확인; 직접 SQL의 사전 의미 검증은 지원하지 않음 |
| R2 / 반영 | [공통 DB 로더](../../packages/schema/src/dlp_schema/db/ontology_repository.py)와 [CLI 연결](../../packages/cli/src/dlp_cli/ontology_access.py) 추가 | 프리라벨·행동·관계·검수·내보내기에 세션/데이터셋 버전 적용. 검수 프로젝트는 사전 버전·내용별 분리. 오프라인 합성 파일럿은 명시적으로 v1 사용 |
| R3 | 탐지·렌더·작업 생성에 긴 세션 잠금. [privacy](../../packages/privacy/src/dlp_privacy/), [tasks](../../packages/review/src/dlp_review/tasks.py) | 입력 세대 확인 → 트랜잭션 밖 처리 → 최종 짧은 재검사·게시. 병행 검수 보존·철회/재탐지 충돌 검사 |
| R4 | DB 롤백과 외부 도구/저장소 부작용 불일치. 회수 실행기(ADR 0039)도 DB에 요청된 작업만 지운다 (고아 정리는 F-REV-15) | 영속 작업 상태·중복 방지·외부 결과 대조·고아 정리. 각 호출 전후 장애 주입 후 재실행해 중복/잘못된 게시 없음 |
| R5 | [검수 클라이언트](../../packages/review/src/dlp_review/clients.py), [lakeFS](../../packages/datasets/src/dlp_datasets/snapshot.py)가 localhost와 포트를 조합 | 공통 URL 설정과 기존 기본값 호환. 원격 테스트 서버로 실제 요청·인증 확인. `DLP_HOST`는 현재 헬스체크 호스트만 변경 |
| R6 | [주간 지표](../../packages/ops/src/dlp_ops/metrics.py), [세션 순위](../../packages/active/src/dlp_active/select.py), [감사 후보](../../packages/privacy/src/dlp_privacy/audit.py)의 전체 이력 조회 | 실제 규모 측정 후 집계/증분 조회. 기존 결과 일치와 데이터 증가 시 쿼리·시간·메모리 비교 |
| R7 | CLI 정책 선택, `schema`의 계약/DB 혼합, export/training/review의 큰 실행 함수 | 공개 인터페이스·입출력·부작용 단위 분리. import 경계와 회귀 검사 보존; R3/R4의 게시 규칙 유지 |

R1/R2는 2026-10-11에 반영했다. 다음에는 R5부터 작은 변경으로 시작하고, R3/R4는 함께 설계한다. 일괄 재작성보다 기존 계약·
이력·회귀 검사를 유지하는 단계별 리팩토링을 권장한다. 검수 조회는 이미 일부 DB 집계로 바뀌었고
[profile_review.py](../../scripts/profile_review.py)로 측정할 수 있으나 전체 파이프라인 성능 보장은 아니다.

## 제품 기능과 정확도 검증

- [온톨로지 구조 검사](ontology-structure-review.md)에서 사전 밖 동사 저장·세션과 다른 버전의
  라벨 저장·없는 구간 자식 참조·잘못된 이관 목적지·개체 클래스와 맞지 않는 부분을 재현했다.
  당시 기존 테스트 67건은 통과했지만 해당 경계는 비어 있었다. 2026-10-11에는 저장·버전·구간·
  이관 검증과 알려진 개체의 부분 소속 검사를 보강했다. 명시적 개체 없는 부분 소속 확인·전체
  이관 실행·청소 하위 단계 확정은 남아 있다. 청소 작업 5종의 하위 단계는 아직 0개이며,
  자동 행동 분류는 기술(`wipe` 등)이 아닌 원시 동작을 출력한다.
- [학습 루프 구조 검사](training-loop-review.md)에서 기존 학습·평가·동시 배포 테스트 97건을
  다시 실행하고 문제 동작 5건을 재현했다. 같은 예제 수의 정답 수정은 재학습에서 누락되고,
  학습 도중 철회된 세션의 모델도 배포될 수 있다(T2·T3, 미수정, 실제 학습기 연결과 함께 해결).
  골든 정답 고정(T5, ADR 0041)과 직접 `deploy()` 호출의 게이트 검사(T4)는 2026-10-12에 고쳤다.
  골든 평가 입력의 미디어·동기화 값 고정은 T6으로 남는다.
- 실제 학습기는 [TRAINERS/LOADERS](../../packages/training/src/dlp_train/trainers.py)에
  `oracle-stub`만 있다. 고객 영상 정답·독립 골든셋·실제 학습기·변경 전후 평가가 있어야
  “사람이 검수할수록 모델이 좋아진다”를 확인할 수 있다.
- 실제 행동 VLM 서버, 배포된 행동 모델 소비, OCR, 도구 작용부 마스크, 바디캠 6자유도와 실제
  표면 3D 입력은 추가 구현/연동 과제다. 준비된 어댑터를 완성 기능으로 표기하지 않는다.
- 손 좌우·장갑·가림·모션 블러, 개인정보 누락, 광각 카메라 3D 오차를 실제 현장 정답으로 측정한다.
  confidence·검출 프레임 비율은 정확도가 아니다. 처리시간과 사람 수정시간도 함께 기록한다.
- 현재 로컬 파일럿의 분리된 CVAT 계정과 접근 거부는 확인했지만 임의 조직/여러 고객 권한 설계가
  끝난 것은 아니다. 가명 ID가 영상 내용에 의한 재연결까지 막지는 못한다.

기존 상세 검수에서 나온 나머지 후보는 [이전 검토 메모](audit-notes.md)에 보존했다.
해당 목록은 이번 재검수에서 전부 재현하지 않은 가설이며 확정 결함 건수로 합산하지 않는다.
