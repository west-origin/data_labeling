# 외주 검수·리팩토링 인수 기준

[인수인계 첫 화면](README.md) · [불변 조건과 강제 수단](invariants.md) · [알려진 문제](issues.yaml) ·
[변경 및 복구 안내](change-guide.md)

외부 전문가에게 전수 검수나 리팩토링을 맡길 때의 범위, 결과물 형식, 받아 들이는 기준이다.
계약서나 작업 지시서에 이 문서를 그대로 붙여도 되게 썼다.

## 시작 기준선

- 기준 커밋(또는 태그)을 작업 지시서에 적는다. 그 커밋에서 `make check`가 통과하고, CI(`.github/workflows/ci.yml`)의
  `check`·`services` 작업이 성공한 상태여야 한다.
- 알려진 문제는 [issues.yaml](issues.yaml)이 전부다. 이 목록에 있는 문제를 다시 보고하면 "기존 ID 확인"으로,
  없는 문제는 "신규"로 센다.
- 확정·미수정 결함 중 재현이 있는 것은 [tests/test_known_defects.py](../../tests/test_known_defects.py)에
  `xfail(strict=True)`로 있다. 외주자가 따로 재현할 필요가 없다.

## 검수 결과물 형식

발견마다 아래를 낸다. 형식이 맞지 않으면 받지 않는다.

| 항목 | 내용 |
| --- | --- |
| ID | 기존이면 레지스트리 ID, 신규면 `F-<영역>-nn` 다음 번호 제안 |
| 위치 | `파일:줄` (기준 커밋 기준) |
| 심각도 | high(데이터 손상·개인정보 노출·파이프라인 전체 중단) / medium(특정 입력에서 틀린 결과·작업 중단) / low(주석·테스트·작은 경계) |
| 종류 | code / comment / test / ops / product |
| 문제 | 한두 문장 |
| 재현 | 입력 → 실제 결과. 가능하면 `tests/test_known_defects.py`에 추가할 xfail 테스트 코드 |
| 고치는 방법 | 한 줄 제안 |

결과는 레지스트리에 항목을 추가하는 PR로 받는다 (`tests/test_issue_registry.py`가 형식을 검사한다).

## 리팩토링 완료의 정의

1. 기존 테스트를 고치지 않고 통과한다. 테스트를 바꿔야 한다면 PR에 이유를 적고, 그 테스트가 지키던
   [불변 조건](invariants.md)을 다른 테스트가 계속 지키는지 보인다.
2. `make check` 통과. DB·외부 도구·내보내기 경로를 바꾸면 `make test-services`, LeRobot 경로면 `make test-isolated`도.
   통과 수(passed/skipped/deselected)를 PR에 적는다. 로컬에서 생략된 테스트가 있으면 이유를 적는다.
3. 패키지 의존이 바뀌면 `docs/handover/architecture.md` 표를 함께 고친다 (`tests/test_import_boundaries.py`).
4. 계약 타입·DB를 바꾸면 ADR + Alembic 리비전 + `make schemas`를 같은 PR에 둔다.
5. 고친 결함은 xfail 표시를 지우고 레지스트리 status를 `fixed`로 바꾼다 (XPASS가 실패로 떠서 강제된다).
6. 동작을 바꾸지 않는 리팩토링이면 PR 제목에 "동작 불변"을 적고, 바뀌는 공개 이름 목록을 PR에 적는다.
7. 한 PR은 한 목적만. 파일 이동과 동작 변경을 섞지 않는다.

## 받아 들일 때 확인하는 것 (검수자 체크리스트)

- [ ] CI `check`·`services` 성공 (실행 링크)
- [ ] PR 양식(`.github/pull_request_template.md`)의 불변 조건 체크가 모두 채워짐
- [ ] 새 의존·새 비공개 import 없음, 또는 architecture.md·허용 목록 갱신과 이유
- [ ] 레지스트리 갱신 (신규 발견 추가, 고친 항목 fixed, xfail 제거)
- [ ] 개인정보 경계(원본 URI·블러 라벨·검수자 ID) 관련 경로를 건드렸으면 사람 검수자가 따로 읽음
- [ ] 문서의 수치·경로가 바뀐 코드와 맞음 (README·handover·CLAUDE.md)

## 저장소 설정 (소유자가 할 일)

코드로 만들 수 없는 것들이다. 외주 시작 전에 저장소 소유자가 설정한다.

- 기본 브랜치(`main`)를 두고 기준선 태그(예: `handover-YYYY-MM-DD`)를 붙인다. 외주 PR은 이 브랜치를 대상으로 한다.
- 브랜치 보호: CI `check`·`services`를 필수 검사로, 직접 푸시 금지, 리뷰 1명 이상.
- `.github/CODEOWNERS`에 사람 승인자를 지정한다. 최소 범위: `packages/privacy/`, `packages/export/`,
  `packages/cli/src/dlp_cli/raw_access.py`, `packages/schema/src/dlp_schema/db/migrations/`,
  `config/policies/privacy.yaml`, `config/models.yaml`.
- 저장소 공개 범위를 확인한다 (외주 계약 범위에 맞게).
