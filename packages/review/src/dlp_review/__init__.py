"""검수 도구 연동 (WP6 검수 도구, WP12 검수 운영).

관련 ADR: 0006(검수 도구 연동), 0014(검수 운영), 0015·0023·0024(감사 정정), 0020(원본 접근 감사),
0028·0029(생애주기 기록, 검수 완료 판정).

파이프라인 위치: 자동 블러·프리라벨이 끝난 세션의 라벨을 사람이 검수하는 단계다.
`dlp review create|collect|serve|register-webhooks|plan|assign|queue|qa|quality|verify`
(`packages/cli/src/dlp_cli/review_cmds.py`)가 이 패키지의 함수를 부른다.

입력 → 출력:
- 입력: DB의 세션·라벨 이력(`label_records`), 원본 버킷(`dlp-raw`)의 프록시·센서 파일,
  라벨링 버킷(`dlp-labeling`)의 블러본, 정책 `config/policies/review.yaml`.
- 출력: CVAT·Label Studio 검수 작업, DB의 `review_tasks`·`review_assignments`·`review_work`,
  검수 결과로 생긴 새 라벨 레코드(덮어쓰지 않고 `parent_label_id`로 잇는다), 검수 상태 갱신,
  세션 생애주기 전이(prelabeled → human_verified).

모듈 구성:
- `tasks`: 세션에서 검수 작업을 만든다 (프라이버시 검수·작업 라벨 검수).
- `collect`: 도구에서 결과를 받아 `reconcile`로 이력을 만들고 DB에 쓴다.
- `reconcile`: 보낸 라벨과 돌아온 항목을 비교해 승인·수정·삭제·추가 레코드로 바꾼다.
- `cvat`·`labelstudio`: LabelRecord ↔ 도구 형식 무손실 변환기.
- `clients`: CVAT·Label Studio REST 클라이언트.
- `webhook`: 도구 웹훅 서명 확인과 수집 요청 파싱, 웹훅 서버.
- `roles`: 단계별 원본 접근 경계 검사.
- `watermark`·`timeseries`: 검수 화면용 매체 (라벨러 ID 워터마크 영상, 센서 시계열 CSV).
- `verify`: 세션 검수 완료 판정 (`dlp review verify`).
- `ops`: 검수 운영 (우선순위·표본·배정·오류 삽입·품질 측정).

주의점:
- 시간: 시간 구간 라벨은 마스터 타임라인 정수 ms, 공간 라벨 키프레임은 그 스트림 영상 PTS ms다
  (ADR 0019). CVAT가 쓰는 프레임 번호는 PTS 인덱스로 맞바꾸고 우리 쪽에는 남기지 않는다.
- 원본 버킷 URI는 일반 라벨러 경로(블러본·작업 라벨 검수)에 나오면 안 된다
  (`roles.check_stage_uris`). 원본 접근은 감사 저장소를 거쳐 기록된다 (ADR 0020).
"""
