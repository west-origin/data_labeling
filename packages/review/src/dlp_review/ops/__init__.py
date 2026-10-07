"""검수 운영 로직 (WP12): 우선순위 큐, 표본 검수, 블라인드·오류 삽입·이중·QA 배정, 측정.

`dlp review plan|assign|queue|qa|quality` (`dlp_cli.review_cmds`)가 쓴다. 정책은
`config/policies/review.yaml`과 `config/defaults.yaml`의 `review` 절(배정 비율)이다.

모듈 구성:
- `policy`: 정책 YAML 로더와 Pydantic 모델 (`ReviewOpsPolicy`).
- `priority`: 검수 단위(`Unit`) 정의, 먼저 볼 구간(`FlaggedSpan`) 찾기, 단위 우선순위.
- `sampling`: 높은 신뢰도 모델 라벨 묶음(lot)의 표본 뽑기와 합격 판정.
- `assign`: 배정 계획 (표준 + 블라인드·이중·오류 삽입·QA), 담당자 부하 균형.
- `seeding`: 오류 삽입 과제 사본 만들기와 발견 판정.
- `measure`: 이중 라벨 일치도, 프리라벨 편향, 오류 발견율.
- `selection`: 배정 방식에 따라 작업에 넣을 라벨 고르기 (작업 생성과 수집이 같은 함수를 쓴다).
- `runner`: 위 부품을 DB와 엮는 실행 함수 (`plan_session`, `create_assignment_tasks`,
  `finish_assignment`, `quality_report`).

주의: 블라인드·이중 결과는 `measurement` 레코드, 오류 삽입 과제의 모든 레코드는 `seeded_error`라서
운영 라벨(`current_labels`)에 들어가지 않고 학습·내보내기에서 빠진다.
"""
