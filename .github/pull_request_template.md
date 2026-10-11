## 목적

<!-- 한 PR은 한 목적만. 동작을 바꾸지 않는 리팩토링이면 제목에 "동작 불변"을 적는다. -->

- 관련 레지스트리 ID (docs/handover/issues.yaml):
- 종류: 결함 수정 / 리팩토링(동작 불변) / 기능 / 문서·주석 / 검수 결과 추가

## 바뀐 것

<!-- 바뀐 공개 이름(다른 패키지가 import하는 이름), 정책 키, DB·계약 변경을 적는다. -->

## 실행한 검사

| 명령 | passed / skipped / deselected / xfailed | 생략 이유 |
| --- | --- | --- |
| `make check` | | |
| `make test-services` (DB·도구·내보내기 경로를 바꿨으면) | | |
| `make test-isolated` (LeRobot 경로를 바꿨으면) | | |

## 불변 조건 확인 (docs/handover/invariants.md)

- [ ] 라벨을 덮어쓰지 않는다 (수정은 새 레코드 + `parent_label_id`)
- [ ] 검수자가 승인·표본 검증한 라벨을 지우는 경로가 없다
- [ ] 다른 단계 입력은 `current_labels()`로 고른다 (오류 삽입·측정 레코드 제외)
- [ ] 원본 버킷 URI·블러 라벨·검수자 ID가 라벨러 경로·내보내기에 나가지 않는다
- [ ] 원본 접근은 감사 저장소(`raw_store`)로만 한다
- [ ] 영상 시각은 PTS 인덱스로만 계산한다
- [ ] 정책 값은 `config/`에서 읽는다 (새 하드코딩 없음)
- [ ] 분할은 `dlp_datasets.splitter`로만 만든다
- [ ] 모든 datetime에 시간대가 있다
- [ ] 해당 없음 항목은 지우지 말고 "해당 없음"이라고 적는다

## 함께 고친 것

- [ ] 계약·DB 변경: ADR + Alembic 리비전 + `make schemas`
- [ ] 패키지 의존 변경: `docs/handover/architecture.md` 표 (`tests/test_import_boundaries.py`)
- [ ] 고친 결함: `tests/test_known_defects.py`의 xfail 제거 + 레지스트리 status `fixed`
- [ ] 새 발견: 레지스트리 항목 추가 (형식은 docs/handover/acceptance.md)
- [ ] 문서의 수치·경로·명령 (README, docs/handover, CLAUDE.md)
