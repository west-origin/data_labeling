# ADR 0027: 내보내기 ID 가명·파일 목록, COCO 이미지, 검증 완료 시각 (4차 검수)

- 상태: 채택
- 날짜: 2026-10-07
- 관련: ADR 0018(내보내기), ADR 0020·0021(운영 지표·내보내기 정정). 계약(dlp_schema)·DB 변경 없음.
  정책 YAML만 바뀜 (`export.yaml ids.dev_secrets·env_var·dev_envs`, `export.yaml lerobot.aspect_tolerance`)

## 결정

### 내보내기 (ADR 0018·0021 보완)

1. **세션·라벨 ID도 내보내기마다 다른 가명이다.** ADR 0021은 작업자·장소만 가명으로 바꿔, 같은 세션을 담은
   두 내보내기를 세션 ID·라벨 ID로 이어 붙일 수 있었다. 이제 같은 내보내기 키(HMAC(비밀값, 내보내기 ID))로
   - 세션 ID → `session-<16자>`: 구간 JSON(`session_id`, 파일 이름), COCO(이미지 `session_id`, 파일 이름),
     LeRobot(`meta/dlp_episodes.json`), manifest(`sessions`, `details.labels_per_session`의 키).
   - 라벨 ID → `label-<16자>`: 구간 JSON `label_id`, COCO 주석 `label_id`.
   - 세션 ID가 들어간 다른 ID(예: 행동 ID `<세션>-right-…-a120`, 그 설명의 `segment_id`, COCO `track_id`)는
     그 부분만 세션 가명으로 바꾼다 (페이로드에서 이름이 `_id`로 끝나는 문자열 값).
   한 내보내기 안에서는 일관된다. manifest `pseudonymized_ids`에 바꾼 ID 종류를 적는다.
2. **내부 대응.** 내보내기 이력(`exports.session_ids`)과 사용 중지 계보는 내부 세션 ID를 그대로 쓴다.
   사용 중지 때 구매자에게 지울 세션을 가명으로 알려야 하므로, 세션 대응표(내부 ID → 가명)를 데이터셋 버킷의
   `internal/export-id-maps/<내보내기 ID>.json`(내보내기 폴더 밖, 전달하지 않음)에 올린다. 비밀값이 없어
   실행마다 임의 키를 쓴 내보내기도 이 표로 되짚을 수 있다.
3. **개발용 비밀값 거부.** `.env.example`의 `DLP_EXPORT_ID_SECRET` 값은 공개돼 있어 그것으로 만든 가명은 누구나
   다시 계산할 수 있다. `DLP_ENV`(export.yaml `ids.env_var`)가 `dev`(`ids.dev_envs`)가 아니면 그 값
   (`ids.dev_secrets`)으로 내보내지 않는다 (`dlp export`가 오류로 끝난다).
4. **manifest 파일 목록.** manifest에 `files: {경로: sha256}`(manifest 자신 제외)을 넣는다. manifest를
   쓰기 전에 계산하고, 올릴 때도 같은 값을 쓴다. 받는 쪽이 빠지거나 바뀐 파일을 확인할 수 있다.
5. **COCO 이미지는 남은 주석이 있는 프레임만.** 전에는 키프레임 프레임마다 이미지를 먼저 만들고 주석을 나중에
   걸러, 주석이 모두 버려진 세션(`unknown_class`, `no_visible_keypoints`, `unsupported_skeleton`)의 이미지가
   manifest `sessions`·이력 `session_ids`에 없이 나갔다 (사용 중지 전파에서 빠짐). 이제 주석을 먼저 만들고
   걸러낸 뒤 남은 주석이 있는 프레임만 이미지로 쓴다.
6. LeRobot 화면비 허용 차이를 정책 `lerobot.aspect_tolerance`로 옮긴다.

### 운영 지표 (ADR 0021 8항 대체)

7. **검증 완료 시각(`verified_at`)** 은 운영 라벨 "항목"(수정 이력 사슬)마다 처음 사람 손을 거친 시각(사슬에서
   가장 이른 검수: 모델 레코드는 승인·수정·표본 검증의 `reviewed_at`, 사람 레코드는 작성 시각) 중 가장 늦은
   것이다. 현재 운영 라벨과 사람이 지운 항목을 본다 (블러·오류 삽입·측정 제외). 그 시각 이전에 있던 미검수
   현재 모델 라벨이 있으면 완료가 아니다.
   - 단계별 파이프라인(객체를 먼저 검수, 2주 뒤 행동 구간 검수)은 마지막 단계의 검수 주에 센다. 전의 정의
     ("T까지 있던 라벨이 모두 검수된 가장 이른 T")는 첫 단계의 검수 주에 셌다.
   - 이미 검수한 항목의 QA 수정·삭제는 사슬의 처음 검수 시각을 바꾸지 않는다.
   - 완료 시각 뒤에 생긴 미검수 모델 라벨(검증 뒤의 새 모델 버전)은 보지 않는다.

## 한계와 후속 (스키마 담당)

- 세션 수명 주기 전이 시각이 DB에 없다. `sessions.lifecycle_state`는 현재 값만 있어, 주간 지표는 "지금
  검증 상태인 세션"을 고른 뒤 라벨 이력으로 시각을 추정한다. 그래서 (a) 수명 주기를 마지막 검수보다 늦게
  `human_verified`로 바꾸면 지난 주의 수가 나중에 늘고, (b) 검증 뒤 생긴 라벨을 나중에 검수하면 완료 시각이
  늦춰진다. 정확한 값에는 추가 전용 표가 필요하다:
  `session_lifecycle_events(session_id text FK sessions, from_state varchar(32), to_state varchar(32),
  at timestamptz not null, actor text)`, `set_lifecycle`이 같은 트랜잭션에서 한 행을 넣고(UPDATE·DELETE 금지
  트리거), 기존 세션에는 현재 상태로 한 행을 채운다. 그러면 `verified_at` = 첫 `to_state = human_verified`
  행의 `at`이고 수명 주기 조건이 필요 없다.
- 가명은 ID로 잇는 것을 막을 뿐이다. 같은 세션을 담은 두 내보내기는 내용(시각·좌표·값)이 같아 내용으로 맞춰 볼
  수 있다. 같은 구매자에게 같은 세션을 다시 줄 때는 이를 전제로 한다.
- 라벨 가명 대응표는 남기지 않는다 (사용 중지는 세션 단위). 비밀값을 아는 내부만 다시 계산할 수 있다.
