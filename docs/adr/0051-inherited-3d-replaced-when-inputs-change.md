# ADR 0051: 물려받기만 한 3D는 입력이 바뀌면 새 계산으로 바꾸고, 지금 2D의 3D가 없으면 완료하지 않는다

- 상태: 채택
- 날짜: 2026-10-19
- 관련: ADR 0048(3D 궤적이 2D 트랙 검수를 물려받음, 1절의 "입력 해시를 따로 비교할 필요가 없다"와
  4절 "완료 판정은 막지 않는다"를 이 ADR이 바꾼다), ADR 0050(끊긴 물려받은 3D 삭제, 이 ADR이 궤적
  따르기 규칙과 저장 검증 예외를 넓힌다), ADR 0045(캘리브레이션·작업자 손 크기가 3D 입력), ADR 0046
  (자세·월드 좌표), ADR 0015·0026(검수 결과 보호), 전수 검수 2026-10-17 묶음 A
  (`docs/handover/full-review-2026-10-17.md`). 문제 레지스트리 F-PAR-41·54·56, F-COE-30(고침).
  계약 변경: 없음 (라벨·`TrajectoryInheritance` 계약, `trajectory_inheritance` 표 그대로, Alembic 없음).
  바뀌는 것은 궤적 따르기 규칙(`dlp_schema.trajectory_follow` 규칙 5), 저장 검증 예외가 허용하는 삭제
  범위(`dlp_schema.label_validation._TrajectoryExemption`이 다시 계산하는 계획), 프리라벨 단계의 다시
  돌리기 판정(`dlp_prelabel.runner.stage_version`), 완료 판정(`dlp_review.verify.trajectory_gaps`,
  `review.yaml verify.require_current_3d`), CLAUDE.md "좁은 예외 둘"의 (2)다.

## 배경

ADR 0048 뒤 2D 손·박스 트랙이 승인된 세션의 3D는 사실상 모두 "물려받기만 한" human_approved다. 사람은
3D 값을 본 적이 없다. 그런데 프리라벨 보호 판정(`drop_protected_trajectories`)은 검수 상태가 검수 전이
아니면 같은 대상의 새 3D를 버렸다. ADR 0048 1절은 "derived_from의 2D가 현재면 3D는 정확히 그 2D 내용으로
계산된 것이라 입력 해시를 비교할 필요가 없다"고 했지만, 3D 단계의 입력에는 2D 말고도 내부 파라미터
(세션·등록 캘리브레이션, ADR 0045), 작업자 손 크기 사전값, depth 정책 절, 추정기 판(+hf), 자세 입력(IMU
동기화·외부 파라미터)이 있다. 그 결과 (F-PAR-41):

- 검수 끝난(판매 대상) 세션에는 등록한 캘리브레이션·손 크기·정책 변경이 영영 반영되지 않았다.
- 새 버전이 아무것도 쓰지 못해 이력에 남지 않으므로, `dlp prelabel run`마다 원본 바디캠을 감사
  저장소로 다시 받고(raw_access_log) 깊이 추론을 전 구간 다시 돌린 뒤 전부 버렸다.

함께 본 두 결함.

- F-PAR-54: 단계의 멱등 판정이 "같은 버전이 이력 어디든 있으면 건너뜀"이라, 입력이 A → B → A로
  되돌아오면(IMU 오프셋 조정 뒤 원복, 손 크기 재등록) B 입력으로 만든 결과가 남았다.
- F-COE-30: 물려받은 뒤 2D를 고치면 궤적 따르기가 3D를 지우고, 고친 2D로 다시 만들기 전에도
  `dlp review verify`가 세션을 완료로 옮겼다 (그 손·박스의 3D 없이 판매). 물려받기 전에 고치면 반대로
  낡은 3D가 검수 전으로 남아 막혔다 (같은 수정이 시점에 따라 다른 판정).

사용자 결정 (2026-10-19)

1. 물려받기만 한 3D는 입력이 바뀌면 바꾼다: 같은 출처 2D에서 3D 단계가 다른 판을 계산하면 새 3D를 쓰고,
   이전 것은 궤적 따르기(`trajectory-follow-1`)가 지우며, 새 3D가 같은 2D 검수를 물려받는다. 직접 검수·
   물려받은 뒤 다시 검수·사람 출처 3D는 그대로 보호한다. 쓸 것이 없으면 원본을 매번 받지 않는다.
2. 단계의 건너뛰기 판정이 되돌아온 입력을 낡은 결과로 두지 않는다.
3. 승인한 2D 트랙의 지금 3D가 없으면(물려받은 3D가 지워짐) 완료하지 않는다 (물려받기 전후 같은 판정).

## 결정

### 1. 궤적 따르기 규칙 5: 대체됨 (`plan_trajectory_follow`)

- 대상: 현재 모델 3D 궤적 x 중 물려받은 검수만 있는 것(지금 검수 상태 = 마지막 `trajectory_inheritance`
  기록, `inheritance_only_trajectory`).
- 조건: 같은 (스트림, 개체, 부위, 좌표계)에 검수 전 모델 3D 궤적 y가 있고, y의 출처(`trajectory_basis`)
  가 x의 기록에 남은 출처 2D와 같은 레코드이며, y의 모델 버전이 x와 다르고, y가 x보다 늦게(같은 시각
  포함) 만들어졌다.
- 결과: x는 `retract`(궤적 따르기 삭제 레코드), y는 같은 계획의 규칙 1로 같은 2D 검수를 물려받는다
  (`approve`, 새 기록). 같은 계획에서 지우는 카메라 좌표 궤적을 경유한 물려받기만 한 월드 좌표 궤적도
  지우고, 그것을 경유하는 검수 전 월드 좌표 궤적에는 물려주지 않는다 (다음 월드 좌표 단계가 새 카메라
  좌표 궤적으로 다시 만든다, 한 번 적용으로 멱등).
- 보호 유지: 기록 없는 검수(직접 검수), 기록 뒤 상태가 바뀐 검수, 사람 출처 3D. 다른 2D 트랙에서 계산한
  새 궤적은 대신하지 않는다 (같은 개체에 두 트랙이 확인된 경우 중복을 막는다).

### 2. 저장 검증 예외 (`label_validation._TrajectoryExemption`)

예외 조건은 그대로다(삭제 레코드 출처 `trajectory-follow-1`, 지우는 레코드가 물려받은 검수만 있는 3D,
저장 전 이력으로 다시 계산한 계획이 지운다). 계획이 규칙 5를 포함하므로 허용 범위가 **정확히** 대체된
궤적(과 그 카메라 좌표 궤적을 경유한 물려받기만 한 월드 좌표 궤적)까지 넓어진다. 새 3D가 이력에 아직
없으면 계획이 대체를 보지 않으므로 거부한다.

### 3. 프리라벨 단계 (`dlp_prelabel.runner`)

- 보호 판정은 지울 궤적(`lapsed_trajectories`)을 새로 계산한 궤적을 이력에 더해 구한다 (`_lapsed(...,
  candidates)`). 그러면 규칙 5로 대체될 물려받기만 한 궤적이 새 궤적을 막지 않는다. 같은 실행의
  `_follow`가 이전 것을 지우고 새 것이 물려받는다 (3D·자세·월드 좌표 단계 모두).
- 헛계산 막기: 고른 트랙이 낼 대상 키((개체, 관절, camera), 박스는 (개체, None, camera))가 모두 바꿀 수
  없는 보호 3D에 막히면 원본을 받거나 깊이를 돌리지 않는다 (`blocked_lift_keys`, 요약 `lift_blocked`,
  CLI `[3D 보호]`). 검수 전 이전 결과가 있으면 그 삭제만 쓴다.
- 다시 돌리기 판정(`stage_version`): 단계 라벨 ID 접두사의 모델 레코드와 그 단계가 쓴 삭제 레코드 중
  가장 늦은 created_at 묶음에 지금 버전이 있으면 건너뛴다 (그 결과를 검수자가 고치거나 지웠어도
  되살리지 않는다). 아니면 다시 만들고, 그 버전을 이전에 쓴 적이 있으면 라벨 ID가 겹치지 않게 버전 끝에
  재실행 세대 `+r<n>`을 붙인다. 궤적 따르기·배포 모델 대체·사람의 삭제 레코드는 단계 실행으로 세지
  않는다. 같은 created_at으로 여러 번 돌린 경우(시험)는 예전처럼 "이력에 있으면 건너뜀"과 같다.
- 자세 단계: 버전을 `campose-<추정기>+p<camera_pose 절>+s<센서 입력>+d<깊이 입력>`으로 나눈다. 센서 입력
  (IMU 동기화·IMU → 카메라 회전·내부 파라미터)이 바뀌었는데 깊이 모델이 없어 다시 추정할 수 없으면 그
  입력으로 만든 검수 전 자세를 지운다 (`[자세 낡음]`, F-PAR-56). IMU·외부 파라미터가 없어지거나 잘못되면
  검수 전 자세를 지운다. 새 버전 추정이 실패하거나 표본이 0개여도 이전 자세를 지운다 (F-PAR-50). 이어서
  월드 좌표 단계가 검수 전 월드 3D를, 궤적 따르기가 물려받기만 한 월드 3D를 정리한다.

### 4. 완료 판정 (`dlp_review.verify.trajectory_gaps`)

`review.yaml verify.require_current_3d`(기본 참)이면 다음을 완료를 막는 이유로 넣고 `dlp prelabel run
<세션>`을 안내한다.

- 궤적 따르기가 지운 3D의 대상(스트림, 개체, 좌표계)에 지금 입력에서 나온 3D(derived_from 첫 항목이 지금
  현재 라벨)가 없는데, 3D 단계가 올릴 현재 2D 트랙(같은 개체의 hand21 손·박스)이 있다.
- 검수 전인 채 출처가 바뀐 낡은 3D가 있다 (물려받기 전에 2D를 고친 경우).

깊이 모델이 없어 3D를 다시 만들 수 없는 운영은 이 값을 끈다 (그러면 그 손·박스 3D 없이 완료된다).
ADR 0048 4절 "완료 판정은 막지 않는다"는 지울 수 없던 레코드 때문이었고, ADR 0050으로 지울 수 있게 된
뒤로는 근거가 없다.

## 결과

- 검수 끝난 세션도 캘리브레이션·작업자 손 크기 등록, depth 정책·추정기 판 변경 뒤 `dlp prelabel run` 한
  번으로 새 값의 3D를 받고, 그 3D는 같은 2D 검수를 물려받는다 (사람이 다시 볼 것이 없다).
- 입력이 이전 값으로 되돌아오면 결과도 되돌아온다. 같은 입력의 재실행은 원본을 받지 않는다.
- 승인한 2D의 3D가 없는 세션은 완료되지 않는다. 운영자는 안내대로 `dlp prelabel run`을 돈다.
- CLAUDE.md "좁은 예외 둘"의 (2)가 "출처가 바뀌거나 지워짐"에서 "출처가 바뀌거나 지워짐, 또는 같은
  출처 2D에서 다시 계산한 새 3D가 대신함"으로 넓어진다.

## 한계

- 대체는 새 판이 그 대상에 표본을 낼 때만 일어난다. 새 입력에서 그 대상 표본이 하나도 나오지 않으면
  (예: 새 캘리브레이션에서 모두 깊이 범위 밖) 이전 3D가 그대로 남는다.
- 물려받은 뒤 검수자가 3D를 직접 다시 검수했으면 새 입력이 반영되지 않는다 (보호, `[3D 보호]`로 알림).
- 완료 판정의 "지금 3D"는 대상(스트림, 개체, 좌표계) 단위다: 다시 만든 3D가 일부 관절만 내면(가림)
  나머지 관절은 막지 않는다.
- 단계 실행 시각 순서는 레코드 created_at이다. 같은 시각에 다른 입력으로 두 번 돈 경우는 구분하지
  못한다 (운영 CLI는 실행마다 지금 시각이다).

## 검증

- 순수 규칙·저장 검증(SQLite): `packages/schema/tests/test_trajectory_follow.py`
  (`test_inherited_3d_is_superseded_by_a_new_version_from_the_same_2d`,
  `test_supersede_needs_inheritance_only_same_source_and_another_version`,
  `test_storage_allows_retracting_a_superseded_inherited_3d_only_after_the_new_one`,
  `test_storage_exemption_is_narrow`).
- 프리라벨 끝까지(SQLite + 로컬 저장소): `packages/prelabel/tests/test_runner_rerun.py`
  (`test_inherited_3d_is_recomputed_when_calibration_and_palm_are_registered`: 등록 뒤 새 3D가 calibrated·
  +hpw·human_approved로 정답과 1 mm 안, 이전 것은 trajectory-follow-1로 지움, 다시 돌리면 원본·깊이 0회.
  `test_directly_reviewed_3d_still_blocks_and_blocked_runs_skip_the_raw_video`,
  `test_reverting_an_input_regenerates_the_earlier_result`,
  `test_resync_without_a_depth_model_retracts_the_stale_pose`,
  `test_failed_estimate_with_a_new_version_retracts_the_previous_pose`).
- 완료 판정: `packages/review/tests/test_trajectory_follow_flow.py`
  (`test_verify_retracts_lapsed_inherited_3d_and_keeps_direct_reviews`,
  `test_correction_before_and_after_inheritance_block_alike`,
  `test_regeneration_requirement_can_be_turned_off`).

## 배포

1. 코드 배포 (DB 마이그레이션 없음). 깊이 3D 단계 판이 `+hf4`, 자세 추정기 판이 `imuvo2`, 자세 버전 형식이
   바뀌어 모든 세션의 3D·자세·월드 3D가 다음 `dlp prelabel run`에서 한 번 다시 만들어진다. 물려받기만 한
   3D는 새 3D로 바뀌고 같은 2D 검수를 물려받는다. 직접 검수 3D는 남는다.
2. 깊이 모델이 있는 환경에서 프리라벨 단계 이후 세션마다 `dlp prelabel run <세션>`, 이어서
   `dlp review verify <세션>`(prelabeled 세션)으로 완료 판정을 다시 본다.
3. 데이터셋 버전을 다시 만든 뒤 내보낸다.
