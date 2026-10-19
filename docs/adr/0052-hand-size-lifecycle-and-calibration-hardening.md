# ADR 0052: 작업자 손 크기의 출처·무효 기록, 수집 뒤 세션 기기 연결, 캘리브레이션 값 검사

- 상태: 채택
- 날짜: 2026-10-19
- 관련: ADR 0045(카메라 내부 파라미터 등록, 작업자 손 크기 사전값, 이 ADR이 3절 결정 규칙과 "선택하지
  않은 대안"을 보완한다), ADR 0020·0021(원본 접근 감사, 운영 기록은 추가만), ADR 0015·0026(검수 결과
  보호). 전수 검수 2026-10-17 묶음 B(F-PMS-20~30, F-COE-70~73).
  계약 변경: `WorkerHandSize`에 `source_session_ids`·`estimator`, 새 계약 `HandSizeInvalidation`·
  `SessionCameraAssignment`(`dlp_schema.calibration`, `schemas/*.schema.json`). DB: Alembic 0024
  (`worker_hand_sizes.estimator`, `worker_hand_size_sources`, `worker_hand_size_invalidations`,
  `session_camera_assignments`, 셋 다 추가 전용 트리거). 정책: `calibration.yaml intrinsics`에
  퇴화·타당 범위 기준.

## 배경

ADR 0045 뒤 전수 검수가 찾은 문제 (`docs/handover/full-review-2026-10-17.md` 묶음 B).

- 손 크기 수명: `worker_hand_sizes`는 추가 전용이고 무효를 뜻하는 기록이 없어, 작업자 동의 철회·삭제
  요구 때 그 작업자를 정책 기본값으로 되돌릴 길이 DBA 개입(트리거 해제)뿐이었다 (F-PMS-30). 추정값은
  어느 세션에서 나왔는지 남기지 않아, 표본을 낸 세션을 사용 중지해도 그 영상에서 잰 값이 같은 작업자의
  다른 세션 3D·내보내기에 계속 쓰였다 (F-PMS-29). `palm-estimate --apply`는 잰 값을 경고 없이 추정값으로
  덮었다 (F-PMS-28).
- 기기 연결: 기기 캘리브레이션은 매니페스트 `calibration.camera_id`로만 세션에 이어졌고, 이미 수집한
  세션에 camera_id를 더해 다시 수집하면 `SessionConflictError`가 났다. ADR 0045 이전 세션은 등록이 있어도
  영원히 화각 근사였다 (F-PMS-24).
- 캘리브레이션 값: 정면만 찍은 체커보드 영상은 재투영 RMS가 작아도 초점 거리가 정해지지 않아(퇴화)
  10배 넘게 틀린 값이 "캘리브레이션됨"으로 등록됐다 (F-PMS-20). 파일 가져오기는 ROS 미보정(K=0)·음수·
  nan·화면 밖 주점·어안 모델을 그대로 받았다 (F-PMS-21·23).

사용자 결정 (2026-10-19): 작업자 손 크기 삭제 요구는 **추가 전용 무효 기록**으로 처리한다 (행을 수정·
삭제하지 않는다). 무효가 된 작업자는 정책 기본값으로 돌아가고, 3D 입력 해시가 바뀌어 다음 실행이 검수
전 3D를 다시 만든다.

## 결정

### 1. 손 크기 무효 기록 (`worker_hand_size_invalidations`, `dlp calib palm-revoke`)

- 한 행 = 무효 기록 하나: 작업자, 대상 등록(`record_id`, NULL이면 그 작업자의 기록 시각 이전 모든 등록),
  근거(필수), 기록 시각. 추가만 한다 (0009·0010 트리거 함수).
- 결정 규칙(`resolve_palm`): 그 작업자의 **유효한** 등록 중 가장 늦은 것 → 정책 `reference_palm_m`.
  유효하지 않은 등록 = 무효 기록이 가리키는 것, 또는 추정 출처 세션 중 하나라도 사용 중지된 것(2절).
  무효 뒤 새로 등록한 값(동의를 다시 받고 잰 값)은 다시 쓰인다.
- 멱등: 새로 무효가 될 유효한 등록이 없으면 쓰지 않는다. 같은 값을 다시 등록할 때의 멱등 비교도 지금
  쓰이는 (유효한) 등록과 한다 (무효가 된 값과 같은 값은 새 행으로 다시 쓴다).
- 3D: 결정이 바뀌면 `ResolvedPalm.digest_input`(출처 글자·값)이 바뀌어 다음 `dlp prelabel run`이 검수 전
  3D만 다시 만든다 (`+hpw` → `+hpp`). 검수·표본 검증한 3D는 기존 보존 규칙대로 남는다.
- 행은 물리적으로 남는다(가명 작업자 ID + 손바닥 길이). 완전 삭제가 법무상 필요하면 감사 기록과 함께
  트리거 예외를 두는 전용 절차가 따로 필요하다 (남은 결정).

### 2. 추정 출처 (`worker_hand_size_sources`, `worker_hand_sizes.estimator`)

- `palm-estimate --apply`는 표본을 실제로 낸 세션 ID를 등록과 같은 트랜잭션에 `worker_hand_size_sources`
  (등록 ID, 세션 ID)로 쓰고, 추정기 판 `<깊이 3D 판>+pe<palm_estimate 절 해시>`를 `estimator`에 남긴다.
- 사용 중지는 따로 기록을 더하지 않아도 결정에 반영된다: `session_palm`이 사용 중지 세션 목록과 함께
  판정한다. 사용 중지 처리(`dlp dataset withdraw`)와 계보(`dlp lineage`)는 그 세션을 출처로 쓴 손 크기
  등록 ID를 보여 준다 (`SessionLineage.hand_sizes`).
- 추정은 긴 트랜잭션이라, 쓰기 직전에 출처 세션 행을 FOR SHARE로 잠그고 사용 중지를 다시 읽는다
  (`share_lock_sessions`, 내보내기 게시와 같은 방식). 그 사이 사용 중지가 있으면 쓰지 않는다 (종료 코드 5).
- 0024 이전 추정에는 출처가 없다 (백필 불가). 그런 등록이 걱정되면 `palm-revoke --record`로 무효로 하고
  다시 추정한다.

### 3. 잰 값 보호, 작업자 ID 확인, 불필요한 원본 접근 방지

- 지금 쓰이는 값이 잰 값(measured)이면 `palm-estimate --apply`는 거부한다 (종료 코드 4). 바꾸려면
  `--override-measured`. 결정 규칙(유효한 최신 등록)은 바꾸지 않는다.
- `dlp calib palm`은 세션이 없는 작업자 ID를 오타로 보고 거부한다 (종료 코드 2, `--allow-unknown-worker`).
  `palm-estimate`도 세션이 없으면 따로 알린다.
- `palm-estimate`는 원본을 받기 전에 캘리브레이션, 손 3D 점(world)이 있는 손 키프레임, 원본 존재(head,
  감사 대상 아님)를 보고, 원본 삭제 결정(`retention_decisions` 최신이 delete)·사용 중지 세션은 처음부터
  뺀다. `--reviewed-only`는 승인·표본 검증한 손 트랙만 쓴다 (돌봄 대상자 손이 섞이는 위험 완화, F-PMS-27).

### 4. 수집 뒤 세션 기기 연결 (`session_camera_assignments`, `dlp calib assign`)

- 한 행 = 세션 하나를 기기 하나에 잇는 기록(근거 필수, 추가만). 결정: 매니페스트 `camera_id` → 그 세션의
  가장 늦은 연결 기록 → 없음. 매니페스트에 다른 기기가 있으면 연결을 거부한다 (매니페스트가 앞선다).
- ADR 0045가 뺀 대안("세션 JSON에 써 넣기")과 같은 충돌을 피하려고 세션 행 밖에 둔다. 연결도 3D 입력
  해시(`ResolvedIntrinsics.digest_input`)를 바꾼다.

### 5. 캘리브레이션 값 검사 (`dlp_media.calibration.check_intrinsics`, `calibration.yaml`)

- 체커보드: 장면별 보드 기울기(보드 법선과 광축 사이 각도, `calibrateCamera` rvecs) 최대값이
  `min_max_view_tilt_deg`(10도)보다 작으면 퇴화로 거부한다. 합성 실험에서 퇴화(진폭 0~3도)는 최대
  2.2~7.5도, 정상(10·30도)은 12.4~32.9도였다. `calibrateCameraExtended`의 fx 표준편차는 퇴화 해에서
  오히려 작아 기준으로 쓰지 않는다.
- 공통(영상·파일): 유한값, fx·fy > 0, 주점이 영상 안, 왜곡 모델 opencv와 OpenCV 계수 길이, 그리고 정책
  타당 범위(가로 화각 20~170도, 주점 중앙 이탈 15%, fx/fy 비 1.2, 왜곡 계수 절댓값 20).
- 파일: 파싱 오류·모르는 키·FileStorage가 아닌 XML·BOM을 `CalibrationError`(종료 코드 2)로 감싼다. ROS
  camera_info의 `distortion_model`은 plumb_bob·rational_polynomial만 opencv로 받고 equidistant 등은
  거부한다. K가 모두 0이면 "미보정"으로 알린다.
- 잘못 등록한 값 바로잡기: 유효 시작 없는 등록은 유효 시작 있는 등록을 이기지 못한다 (ADR 0045 규칙
  그대로). `--supersede <등록 ID>`가 그 등록의 유효 시작을 복사하고, 새 등록이 가려지면 CLI가 알린다.

## 결과

- 동의 철회 작업자의 손 크기는 행 수정 없이 결정에서 빠지고, `dlp ops retention`이 세션이 모두 사용
  중지된 작업자의 쓰이는 손 크기를 알린다. 사용 중지된 영상에서 나온 추정은 자동으로 쓰이지 않는다.
- 이미 수집한 세션도 기기 캘리브레이션을 쓸 수 있다.
- 비용: DB 표 3개·열 1개, 손 크기 결정이 사용 중지 목록을 읽는다(작은 표). 정책 절이 바뀌어도 3D 단계
  버전은 바뀌지 않는다 (`calibration.yaml`은 등록 명령만 쓴다).
- 남은 결정: 손 크기 행의 물리 삭제(법무 검토), 카메라-IMU 외부 파라미터(`camera_imu_extrinsics`)도 같은
  세션 밖 연결이 필요한지(F-PAR-30), 착용자 손 판정 기준(정책).
