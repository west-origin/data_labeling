# ADR 0050: 끊긴 물려받은 3D는 궤적 따르기가 지우고, 박스 중심 3D도 물려받으며, 카메라 자세는 자동 라벨이다

- 상태: 채택
- 날짜: 2026-10-17
- 관련: ADR 0048(3D 궤적이 2D 손 트랙 검수를 물려받음, 이 ADR이 4절과 "남은 결정"을 바꾼다),
  ADR 0043 3절 규칙 6(설명 따르기의 좁은 삭제 예외, 이 ADR의 본보기), ADR 0046(카메라 자세·월드 좌표),
  ADR 0015·0026(검수 결과 보호). 문제 레지스트리 F-COE-21·F-COE-22(고침).
  계약 변경: 없음. 라벨·물려받기 기록 계약(`TrajectoryInheritance`)과 DB 표(`trajectory_inheritance`,
  Alembic 0023)는 그대로다. 바뀌는 것은 규칙(`dlp_schema.trajectory_follow`), 저장 검증의 예외
  (`dlp_schema.label_validation`), 검수 정책 절 `review.yaml verify`, 내보내기 manifest 절이다.

## 배경

ADR 0048 뒤 남은 두 문제.

- F-COE-22: 물려받은 뒤 출처 2D가 고쳐진 3D("끊김")를 지울 수 없어(CLAUDE.md의 좁은 예외가 설명만 다룸)
  이력에 검수 상태로 남았다. 프리라벨 보호 판정·월드 입력과 내보내기만 그것을 뺐고, 현재 라벨을 직접
  고르는 다른 소비자는 끊긴 3D를 검수된 것으로 볼 수 있었다.
- F-COE-21: 객체 박스 중심 3D와 카메라 자세는 물려받을 2D 손 트랙이 없어 검수 전으로 남아, 깊이·자세를
  쓰는 세션을 `dlp review verify`가 완료로 옮기지 못했다.

사용자 결정 (2026-10-17): 두 변경을 승인한다.

## 결정

### 1. 끊긴 물려받은 3D는 궤적 따르기가 지운다 (F-COE-22, 설명 따르기 규칙 6과 같은 틀)

- 대상: 모델 출처 3D 궤적 중 **지금 검수 상태가 그 궤적의 마지막 `trajectory_inheritance` 기록과
  같은 것**(물려받은 검수만 있음, `inheritance_only_trajectory`)의 출처가 더는 성립하지 않을 때.
  출처가 성립하지 않음 = derived_from의 2D 트랙이 현재가 아님(고쳐짐·지워짐), 또는 월드 좌표 궤적의
  카메라 좌표 궤적·카메라 자세가 바뀜 (`plan_trajectory_follow`의 `retract`). 월드 좌표 궤적은 카메라
  좌표 궤적의 2D가 바뀌면 같은 계획에서 함께 끊긴다.
- 지우는 쪽: 궤적 따르기(`trajectory-follow-1`, 모델 출처 삭제 레코드 `<궤적>:retracted`).
- 보호 유지: 기록 없는 검수(직접 검수), 기록 뒤 상태가 바뀐 검수(물려받은 뒤 다시 검수), 사람이
  쓰거나 고친 3D. "물려받기만 함"은 추정하지 않고 명시 기록으로만 정한다.
- 저장 검증(`label_validation._TrajectoryExemption`): 삭제 레코드가 `trajectory-follow-1` 출처이고,
  지우는 레코드가 3D 궤적이며 물려받은 검수만 있고, 저장 전 이력(세션 잠금 아래)으로 다시 계산한 궤적
  따르기 계획이 그것을 지울 때만 보호 라벨 삭제를 허용한다. `insert_labels`는 그런 삭제가 있을 때만
  세션의 `trajectory_inheritance`를 읽는다. 그 밖의 보호 라벨 삭제는 그대로 막는다.
- 적용 지점(ADR 0048 3절과 같다, 세션 잠금, 멱등): 검수 수집(`collect_task`), 완료 판정
  (`dlp review verify`), 프리라벨(`run_prelabel`, 깊이 3D 단계 뒤와 월드 좌표 단계 뒤). 수집이 2D 수정을
  받으면 그 수집 끝에 끊긴 3D가 지워지고, 다음 `dlp prelabel run`이 고친 2D로 새 3D를 만들어 물려받는다.
- 이중 방어로 남긴 것: `lapsed_trajectories`(같은 끊김 집합)로 프리라벨 보호 판정·월드 좌표 입력과
  내보내기 선택이 끊긴 3D를 계속 뺀다. 지금 DB에서는 적용 뒤 보통 비지만, (1) 데이터셋 스냅샷은 지우기
  전 이력일 수 있고, (2) 따르기를 거치지 않은 경로로 2D가 바뀌어도 프리라벨이 새 3D를 막히지 않고 만든
  뒤 같은 실행의 따르기가 끊긴 것을 지운다.

### 2. 박스 중심 3D는 그 박스 트랙의 검수를 물려받는다 (F-COE-21)

- 출처(`trajectory_basis`): 모델 출처, frame=camera, part 없음, `derived_from` = (박스 트랙 하나),
  그 박스 트랙이 지금 운영 현재이고 개체·스트림이 같다. 깊이 3D 단계는 ADR 0048 판(`+hf3`)부터 이미
  이 derived_from을 남긴다 (재생성 불필요). 월드 좌표 박스 중심 궤적도 같은 경유 규칙으로 물려받는다.
- 손 3D와 같은 규칙: 명시 기록, 낡은(출처가 현재가 아닌) 3D는 물려받지 않음, 끊기면 1절로 지움.

### 3. 카메라 자세는 자동 라벨이다 (F-COE-21)

- 카메라 자세(모델 출처, entity `camera`, `is_camera_pose`)는 검수 도구 경로도 물려받을 2D도 없는 자동
  추정이라 사람 검수 대상이 아니다. 검수 상태는 바꾸지 않는다 (검수 전 그대로).
- 완료 판정: 검수 정책 `review.yaml verify.automatic`(기본·저장소 값 `[camera_pose]`)에 있으면 검수 전
  자세가 `dlp review verify`를 막지 않는다. 결과·CLI 출력에 그 수(`automatic_unreviewed`)를 알린다.
  절을 빼도 코드 기본값이 `[camera_pose]`다. 목록을 비우면 자세가 다시 완료를 막는다.
- 내보내기: 자세 자체는 검수 전이라 기본 내보내기(검수 라벨만)에 들어가지 않는다. 자세에 기대는 결과를
  밝힌다. 모든 형식의 manifest `verification_policy.trajectory3d`에 `camera_pose: automatic_unreviewed`,
  `pose_derived`(결과에 든 월드 좌표 궤적 수와 자세 궤적 수), `object_review:
  inherited_from_2d_box_track`, `review_basis`(근거별 수, 검수 전 자세는 `automatic`)를 싣는다. 로봇
  모드 `meta/dlp_retarget.json`(manifest `details.retarget`)에도 `camera_pose: automatic_unreviewed`를
  싣는다 (목표가 월드 좌표 손 3D다).

## 결과

- 깊이·자세 단계를 거친 세션도 2D 손·박스 트랙만 검수하면 완료된다.
- 2D를 고치면 그 2D에서 물려받기만 한 3D가 이력에서 사라져(삭제 레코드) 현재 라벨을 직접 고르는 모든
  소비자가 같은 결과를 본다.
- CLAUDE.md "좁은 예외"가 둘이 된다 (설명 따르기, 궤적 따르기).

## 한계

- 같은 상태·검수자·시각으로 직접 다시 검수하면 물려받음과 구분되지 않는 한계는 설명과 같다 (F-PAR-25).
- 자세 품질 지표(추적 상실·IMU 잔차)로 자세를 자동 승인하는 길은 택하지 않았다 (검수 상태를 꾸미지
  않는다). 실영상 자세 정확도는 F-PAR-29.

## 검증

- 순수 규칙·SQLite: `packages/schema/tests/test_trajectory_follow.py`
  (`test_inherited_3d_is_retracted_when_its_source_changes`, `test_box_center_3d_inherits_the_box_track_and_lapses_with_it`,
  `test_apply_retracts_lapsed_inherited_3d_only_and_is_idempotent`, `test_storage_exemption_is_narrow`).
- 수집·완료 판정: `packages/review/tests/test_trajectory_follow_flow.py`
  (`test_session_with_depth_and_pose_is_verified_after_2d_approvals`,
  `test_verify_retracts_lapsed_inherited_3d_and_keeps_direct_reviews`).
- 프리라벨(PostgreSQL·S3): `packages/prelabel/tests/test_runner.py::test_3d_inherits_the_approved_2d_track_and_regenerates_after_correction`.
- 내보내기: `packages/export/tests/test_robot.py::test_world_hand_inherited_from_approved_2d_is_used_by_default`
  (`trajectory_manifest`), `::test_robot_episode_follows_reviewed_world_hand`(retarget meta).

## 배포

1. 코드 배포 (DB 마이그레이션 없음).
2. 깊이·자세를 쓰는 세션마다 `dlp review verify <세션>`: 박스 중심 3D 물려받기, 끊긴 물려받은 3D 삭제를
   메우고 자세 때문에 막혔던 세션을 완료로 옮긴다. 끊긴 3D를 지운 세션은 `dlp prelabel run <세션>`으로
   고친 2D에서 3D를 다시 만든다.
3. 데이터셋 버전을 다시 만든 뒤 내보낸다 (스냅샷의 라벨이 새 상태여야 한다).
