# ADR 0048: 3D 손 궤적은 그것을 계산한 2D 손 트랙의 검수를 물려받는다

- 상태: 채택 (4절 "끊김"·박스 중심 3D·카메라 자세는 ADR 0050이, 1절 입력 해시 전제와 4절 완료 판정은
  ADR 0051이 바꿨다)
- 날짜: 2026-10-17
- 관련: ADR 0043 3절(설명 따르기, 이 ADR의 본보기), ADR 0044·0045(손 3D), ADR 0046(카메라 자세·월드 좌표),
  ADR 0047(로봇 리타기팅 내보내기), ADR 0015·0026(검수 결과 보호), ADR 0050(후속). 문제 레지스트리
  F-COE-19(고침), F-COE-21·22(ADR 0050으로 고침).
  계약 변경: 새 계약 `TrajectoryInheritance`·`TrajectoryInheritanceMechanism`(`dlp_schema.review`,
  `schemas/trajectory_inheritance.schema.json`), 새 추가 전용 표 `trajectory_inheritance`(Alembic 0023).
  라벨 계약은 그대로다 (`Trajectory3DPayload.derived_from`은 ADR 0046에 이미 있다).

## 배경

검수 도구(CVAT·Label Studio)는 3D 궤적(`trajectory3d`)을 보이지 않고, 표본 검증 묶음도 3D를 다루지 않는다.
그래서 3D 궤적은 언제나 검수 전이었고, 기본 내보내기(사람이 만들거나 확인한 라벨만)에서 로봇 목표
(`dlp export lerobot --robot`)와 사람 모드 손 3D 점(`hand_points_3d`)이 비었다. `--include-unreviewed`를
켜야만 목표가 생겼다 (F-COE-19). `dlp review verify`도 검수 전 3D 때문에 세션을 완료로 옮기지 못했다.

사용자 결정 (2026-10-17): **3D 궤적은 그것을 계산한 2D 손 키포인트 트랙의 검수 상태를 물려받는다.**

## 결정

### 1. 출처 (`dlp_schema.trajectory_follow.trajectory_basis`)

- 카메라 좌표 손 궤적: 깊이 3D 단계(`dlp_prelabel.lift3d.DepthLifter`)가 궤적마다 올린 입력 트랙 ID를
  `derived_from = (2D 트랙,)`으로 남긴다. 단계 판을 `+hf3`으로 올려, 다음 `dlp prelabel run`이 검수 전
  예전 3D를 이 형식으로 다시 만든다. 라벨은 덮어쓰지 않으므로(고치면 새 ID) derived_from의 ID가 지금
  운영 현재 트랙이면 3D는 정확히 그 2D 내용으로 계산된 것이다. 입력 해시를 따로 비교할 필요가 없다.
  (ADR 0051 정정: 3D 단계 입력에는 2D 말고도 내부 파라미터·작업자 손 크기·depth 정책·추정기 판이 있다.
  같은 2D에서 다른 판으로 다시 계산한 3D가 물려받기만 한 3D를 대신한다.)
- 월드 좌표 손 궤적: `derived_from = (카메라 좌표 궤적, 카메라 자세)` (ADR 0046). 둘 다 현재여야 하고,
  카메라 좌표 궤적이 위 조건을 만족해야 한다 (같은 개체·부위). 출처는 그 2D 트랙이다.
- 출처가 없는 것: derived_from이 없는 예전 3D, 박스 중심 3D, 카메라 자세, 사람 출처 3D. 물려받지 않는다.
  (ADR 0050: 박스 중심 3D는 그 박스 트랙에서 같은 방식으로 물려받고, 카메라 자세는 자동 라벨로 완료
  판정에서 뺀다.)

### 2. 물려받기 (`plan_trajectory_follow`, 설명 따르기 규칙 5와 같은 틀)

- 검수 전 모델 3D의 출처 2D가 사람이 확인한 상태(`description_follow.human_confirmed`: 승인·표본 검증·
  사람 수정, 또는 검수자가 있는 사람 출처)이면 물려받는다. 상태는 2D가 표본 검증이면 sample_verified,
  아니면 human_approved. 검수자는 2D의 검수자, 시각은 max(2D 검수 시각, 3D 생성 시각).
- 2D가 검수 전이면 기다린다. 2D를 고치면(새 레코드) 예전 2D로 만든 3D는 출처가 현재가 아니므로 물려받지
  않는다. 다음 `dlp prelabel run`이 고친 2D로 새 3D를 만들고(입력 해시가 바뀌어 다시 돈다), 고친 2D는
  사람이 확인한 상태라 새 3D가 같은 실행에서 물려받는다.
- 물려받음은 명시 기록으로만 판정한다. `trajectory_inheritance`(추가 전용, 수정·삭제·TRUNCATE 금지
  트리거)에 물려받는 `record_review`와 같은 트랜잭션으로 한 행(궤적, 출처 2D, 경유 입력, 상태·검수자·
  시각, 경로, 규칙 버전 `trajectory-follow-1`)을 쓴다. "물려받은 검수만 있음"은 그 궤적의 마지막 기록과
  지금 검수 상태가 같을 때뿐이다. 기록이 없는 검수는 직접 검수다 (추정하지 않는다).

### 3. 적용 지점 (모두 세션 잠금 아래, 멱등)

`dlp_schema.db.trajectory_repository.apply_trajectory_follow`를 설명 따르기와 같은 곳에서 부른다.

- 검수 수집(`dlp_review.collect.collect_task`, 설명 따르기 뒤, 경로 collect). CVAT로 2D를 승인·수정하면
  그 수집 끝에 3D가 따른다.
- 완료 판정(`dlp review verify`, 경로 verify). 이 규칙 전에 검수한 세션도 메운다.
- 프리라벨(`dlp_prelabel.runner.run_prelabel`, 경로 prelabel): 깊이 3D 단계 뒤와 월드 좌표 단계 뒤.
  이미 확인된 2D에서 새로 만든 3D가 바로 물려받는다.

### 4. 출처가 바뀐 물려받은 3D ("끊김", lapsed)

> ADR 0050이 이 절을 바꿨다: 끊긴 물려받은 3D는 궤적 따르기(`trajectory-follow-1`)가 지우고(저장 검증의
> 두 번째 좁은 예외), 아래의 빼기는 이중 방어로 남는다. 아래는 채택 당시 결정이다.

물려받은 뒤 출처 2D가 고쳐지거나 지워지면(또는 월드 좌표의 카메라 좌표 궤적·자세가 바뀌면) 그 3D는
낡았다. 설명 따르기 규칙 6은 이런 설명을 지우지만, CLAUDE.md의 좁은 예외는 "행동 검수를 물려받기만 한
모델 설명"만 다룬다. 3D를 지우려면 예외를 넓혀야 하므로 **지우지 않는다** (사용자 결정 필요, F-COE-22).
대신 기록을 남긴 채 다음처럼 다룬다.

- 프리라벨: 끊긴 3D는 보호 판정(`drop_protected_trajectories`)에서 빼, 고친 2D로 만든 새 3D가 같은
  (개체, 부위, 좌표계)에 들어가게 한다. 월드 좌표 단계 입력(`pick_world_inputs`)으로도 고르지 않는다.
- 내보내기: 끊긴 3D는 옵션(`--include-unreviewed`)과 관계없이 넣지 않는다 (`dlp_export.source.select_labels`,
  지금 DB의 기록으로 판정). 새 3D가 아직 없으면 그 손의 3D가 빠진다 → `dlp prelabel run`.
- 수집·완료 판정·프리라벨 출력이 끊긴 수를 알린다. 완료 판정은 막지 않는다 (지울 수 없는 레코드라
  막으면 세션이 영원히 완료되지 못한다). ADR 0051이 바꿨다: 지운 3D를 지금 2D로 다시 만들기 전에는
  완료를 막고 `dlp prelabel run`을 안내한다.
- 직접 검수 3D(기록 없음, 또는 기록 뒤 상태가 바뀐 것)와 사람 출처 3D는 끊김으로 보지 않는다.

### 5. 내보내기 정직성

- 기본 내보내기(검수 라벨만)에 물려받은 3D가 들어간다: 로봇 모드 손 목표, 사람 모드 `hand_points_3d`.
  `--include-unreviewed`는 그대로 동작한다.
- manifest `verification_policy.trajectory3d`: `hand_review: inherited_from_2d_hand_track`,
  `camera_pose: automatic_unreviewed`, `lapsed_inheritance: excluded`, `review_basis`(결과에 들어간
  3D의 근거별 수: inherited_from_2d·direct·human·unreviewed). 사람이 3D 값 자체를 본 것이 아니고 월드
  좌표는 자동 자세에도 기대며, 손 모양(MediaPipe world 점)·깊이도 사람이 보지 않았다는 것을 받는 쪽이
  알 수 있다. 라벨별 검증 등급(`annotation.verification`, `retarget.verification`)은 물려받은 상태를 쓴다.
- 스냅샷에는 기록 표가 없어 내보내기가 지금 DB의 `trajectory_inheritance`를 읽는다 (추가 전용이라 스냅샷
  때의 기록은 그대로 있다. 사용 중지·렌더 확인도 이미 지금 DB를 본다).

## 결과

- 2D 손 트랙을 승인하거나 표본 검증하면 3D 손 궤적(카메라·월드 좌표)이 같이 검수 상태가 되어 기본
  내보내기의 로봇 목표가 빈 채로 나가지 않는다. `dlp review verify`도 손 3D 때문에 막히지 않는다.
- 카메라 자세·박스 중심 3D는 여전히 검수 경로가 없어 완료 판정을 막는다 (F-COE-21, ADR 0050으로 고침:
  박스 중심 3D는 박스 트랙을 물려받고 카메라 자세는 자동 라벨).
- 주간 운영 지표(자동 승인율·수정률)는 물려받기만 한 라벨(3D 궤적, 그리고 같은 문제가 있던 설명)을
  검수자의 "그대로 승인"으로 세지 않는다 (`dlp_ops.metrics._reviewer_changes`, F-COE-23).
- `select_labels`·`plan_trajectory_follow`·`lapsed_trajectories`·`review_basis`·
  `active_trajectory_inheritance`의 기록 인자는 기본값이 없다 (`tests/test_inheritance_required.py`).

## 남은 결정·한계

- **끊긴 물려받은 3D 지우기 (F-COE-22)**: ADR 0050으로 결정·구현했다 (아래는 당시 기록). 당시에는 남겨 두고 빼는 방식이었다. 지우려면 CLAUDE.md의 좁은 예외에
  "2D 손 트랙 검수를 물려받기만 한 3D 궤적은 출처 2D가 바뀌면 궤적 따르기(`trajectory-follow-1`)만 지울 수
  있다(명시 기록 `trajectory_inheritance`로만 판정)"를 더하고, 저장 검증 예외(`label_validation`)를 넓혀야 한다.
- 같은 상태·검수자·시각으로 직접 다시 검수하면 물려받음과 구분되지 않는 한계는 설명과 같다 (F-PAR-25).
- 3D 표본 검증(리타기팅 재생 영상을 보고 판정)은 두지 않았다.

## 검증

- 순수 규칙·SQLite 적용: `packages/schema/tests/test_trajectory_follow.py` (승인·표본 검증 물려받기, 고친 2D의
  낡은 3D 미물려받음, 다시 만든 3D 물려받음, 끊김(지우지 않음), 직접 검수 보호, 자세 변경, 멱등, 계약).
- 수집·완료 판정: `packages/review/tests/test_trajectory_follow_flow.py`.
- 프리라벨(PostgreSQL·S3): `packages/prelabel/tests/test_runner.py::test_3d_inherits_the_approved_2d_track_and_regenerates_after_correction`,
  단위 `packages/prelabel/tests/test_runner_units.py::test_lapsed_inherited_trajectories_do_not_block_or_feed_new_ones`,
  `packages/prelabel/tests/test_lift3d.py`(derived_from).
- 내보내기: `packages/export/tests/test_robot.py::test_world_hand_inherited_from_approved_2d_is_used_by_default`.
- DB: `packages/schema/tests/test_db.py::test_trajectory_inheritance_is_append_only_and_blocks_downgrade`.

## 배포

1. `make db-upgrade` (Alembic 0023, 백필 없음).
2. 3D가 있는 세션마다 `dlp prelabel run <세션>`: 판 `+hf3`으로 검수 전 3D를 derived_from이 있는 3D로 다시
   만들고, 이미 확인된 2D의 검수를 물려받는다 (월드 좌표도 입력이 바뀌어 다시 만든다).
3. 이미 완료 판정 전인 세션은 `dlp review verify <세션>`이 남은 물려받기를 메운다.
4. 데이터셋 버전을 다시 만든 뒤 내보낸다 (스냅샷의 라벨 검수 상태가 물려받은 상태여야 한다).
