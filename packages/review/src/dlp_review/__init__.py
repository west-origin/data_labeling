"""검수 도구 연동 (WP6 검수 도구, WP12 검수 운영).

관련 ADR: 0006(검수 도구 연동), 0014(검수 운영), 0015·0023·0024(감사 정정), 0020(원본 접근 감사),
0028·0029(생애주기 기록, 검수 완료 판정), 0039(블러 변경 뒤 배포 매체 회수), 0040(AI 1차 검수).

파이프라인 위치: 자동 블러·프리라벨이 끝난 세션의 라벨을 사람이 검수하는 단계다.
`dlp review create|collect|serve|revoke|refresh-urls|register-webhooks|plan|assign|queue|qa|quality|
verify`
(`packages/cli/src/dlp_cli/review_cmds.py`)가 이 패키지의 함수를 부른다.

입력 → 출력:
- 입력: DB의 세션·라벨 이력(`label_records`), 원본 버킷(`dlp-raw`)의 프록시·센서 파일,
  라벨링 버킷(`dlp-labeling`)의 블러본, 정책 `config/policies/review.yaml`.
- 출력: CVAT·Label Studio 검수 작업, DB의 `review_tasks`·`review_assignments`·`review_work`,
  검수 결과로 생긴 새 라벨 레코드(덮어쓰지 않고 `parent_label_id`로 잇는다), 검수 상태 갱신,
  세션 생애주기 전이(prelabeled → human_verified).

검수 작업 하나의 생애주기 (만들기 → 수집 → 대조 → 완료 판정):
1. 만들기 (`tasks`). `dlp review create <세션>`이 바로 부르거나, `dlp review plan`이 배정
   (`review_assignments`)을 세운 뒤 `dlp review assign <배정>`이
   `ops.runner.create_assignment_tasks`를 거쳐 부른다.
   - 블러 검수 (`tasks.create_privacy_tasks`): 원본 버킷의 검수 프록시에 지금 블러 트랙을 얹은 CVAT
     작업을 스트림마다 만든다. 담당자는 원본 접근 권한자(`review.yaml reviewers.privacy`)여야 하고,
     권한 검사와 원본 접근 감사 기록(grant)이 원본을 도구에 올리기 전에 일어난다.
   - 작업 라벨 검수 (`tasks.create_labeling_tasks`): 지금 승인된 블러본에 담당자 워터마크를 입혀
     라벨링 버킷에 올리고 CVAT(공간 트랙)·Label Studio(시간 구간) 작업을 만든다.
   두 경우 모두 `review_tasks` 행(상태 open)에 작업에 보낸 라벨 ID(`sent_label_ids`)를 남긴다.
   수집은 이 목록을 비교 기준으로 쓴다.
2. 수집 (`collect.collect_task`). `dlp review collect <작업 키>` 또는 웹훅 서버
   (`dlp review serve`)가 부른다. 세션 행 → 작업 행 순으로 잠그고 open인 작업만 받는다
   (collected·revoked면 None). 도구 결과를 변환기(`cvat`·`labelstudio`)로 `reconcile.ReviewedItem`
   목록으로 바꾼다.
3. 대조 (`reconcile.reconcile`). 돌아온 항목을 보낸 라벨과 비교해 네 결과로 나눈다.
   - 그대로: 원래 라벨의 검수 상태만 human_approved로 바꾼다 (`record_review`).
   - 고침: 원래 라벨을 parent로 하는 새 사람 레코드 (human_corrected).
   - 없어짐: 원래 라벨을 parent로 하는 삭제(retracted) 레코드.
   - 새로 그림: parent 없는 새 사람 레코드.
   보낸 뒤 이미 자식 레코드가 생긴 라벨의 결과는 버리고 stale로 알린다 (`collect.drop_retracted`).
   블라인드·이중 작업의 결과는 측정 레코드(`measurement`), 오류 삽입 작업의 결과는
   `seeded_error` 레코드라서 운영 라벨(`current_labels`)에 들지 않는다. 수집은 이어서 작업을
   collected로 표시하고, 배정이 있으면 배정을 마무리한다 (`ops.results.finish_assignment`: 표본
   합격이면 보류한 묶음을 sample_verified로, 불합격이면 재검수 배정). 블러 검수가 운영 블러를 바꾸면
   프라이버시 승인을 풀고 그 스트림의 배포 매체에 회수를 요청한다 (`revoke`가 커밋 뒤 끊는다).
4. 완료 판정 (`verify.verify_session`, `dlp review verify <세션>`). 블러 승인, 열린 작업·배정 없음,
   회수 완료, 모델 라벨 모두 사람 검수 상태, 손별 행동 타임라인 빈 시간 없음 등을 모두 만족하면
   세션 생애주기를 prelabeled → human_verified로 옮긴다.

이 패키지가 지키는 약속:
- 라벨을 덮어쓰지 않는다. 내용이 바뀌면 새 레코드 + `parent_label_id`이고, 기존 행에서 바꾸는
  것은 검수 상태(`dlp_schema.db.repository.record_review`)뿐이다 (DB 트리거가 강제한다).
- 검수를 줄이는 것은 사람뿐이다. 이 패키지에서 모델 라벨을 검수된 상태로 바꾸는 경로는 사람의
  승인(`collect`)과 사람이 본 표본의 합격(`ops.results.finish_assignment`의 sample_verified) 둘이다.
  AI 1차 검수 판정(`ai_display`, `ai_collect`)은 검수를 늘리는 쪽(우선순위, 표본 제외, 재검수)으로만
  쓰고, AI의 ok는 어떤 라벨도 검수된 것으로 만들지 않는다 (ADR 0040). 설명 라벨과 3D 궤적은 사람이
  검수한 행동·2D 트랙의 결과를 물려받기만 한다 (`apply_description_follow`,
  `apply_trajectory_follow`).
- 원본 버킷 URI는 일반 라벨러 경로(작업 라벨 검수의 매체 URI·서명 URL)에 나오지 않는다.
  `roles.check_stage_uris`가 작업 생성(`tasks`), URL 다시 서명(`refresh`), 회수(`revoke`)에서
  검사한다. 검수자에게 원본 영상을 보이는 것은 블러 검수뿐이다. 이 패키지가 원본 버킷을 읽는 곳(블러
  검수 프록시와 원본 해상도, 시계열 CSV를 만들 장갑·IMU Parquet)은 `ReviewSetup.raw`(운영에서는
  감사 저장소)를 거치므로 원본 접근 기록이 남는다 (ADR 0020).

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
- `revoke`: 블러가 바뀐 뒤 회수 요청된 라벨링 작업 매체(라벨링 버킷 사본, CVAT·Label Studio 작업)를
  커밋 뒤 끊는 실행기 (`dlp review revoke`, ADR 0039). 요청은 `dlp_privacy.revocation`이 남긴다.
- `refresh`: 열린 Label Studio 작업의 서명 URL을 다시 서명한다 (`dlp review refresh-urls`,
  F-REV-08).
- `ops`: 검수 운영 (우선순위·표본·배정·오류 삽입·품질 측정).
- `ai_display`: AI 1차 검수 판정(참고용, 라벨 아님)을 검수 화면에 표시하고(Label Studio 예측·CVAT
  이슈) 노출 기록·대조군을 정하며, 사람 검수를 늘리는 쪽(우선순위·표본 제외·마무리 보호·required
  게이트)으로만 쓴다 (WP17.6, ADR 0040, `review.yaml ai_review`).

주의점:
- 시간: 시간 구간 라벨은 마스터 타임라인 정수 ms, 공간 라벨 키프레임은 그 스트림 영상 PTS ms다
  (ADR 0019). CVAT가 쓰는 프레임 번호는 PTS 인덱스로 맞바꾸고 우리 쪽에는 남기지 않는다.
- 트랜잭션은 CLI가 연다. 작업 생성은 세션 행을 잠근 채 도구 작업을 만들고 사본을 올리므로, DB가
  롤백돼도 도구 작업과 라벨링 버킷 사본은 남는다 (고아는 `orphans`가 찾아 지운다).
"""
