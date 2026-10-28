# ADR 0046: 바디캠 6자유도 자세(IMU 회전 + 메트릭 깊이 이동)로 손 3D를 세션 월드 좌표에 둔다

- 상태: 채택 (2026-10-19 개정: 2절 정지 판정·IMU 늦은 시작·빈틈, 3절 외부 파라미터 검사·이전 자세 처리,
  4절 버전 형식·되돌림, 5절 관계 단계, 전수 검수 2026-10-17 묶음 A F-PAR-44·46~57, ADR 0051)
- 날짜: 2026-10-17
- 관련: ADR 0044·0045(카메라 좌표 손가락 끝 3D, 내부 파라미터 등록), ADR 0010(모델 상업 사용),
  ADR 0015·0026(검수된 라벨 보존, 3D 중복 방지), ADR 0019(공간 라벨 시각, 동기화), ADR 0003(IMU 추출),
  ADR 0047(로봇 리타기팅이 이 월드 좌표를 쓴다). 문제 레지스트리 F-PAR-26(이 ADR로 고침), F-PAR-29·30·31(남은 일).
  계약 변경: `Trajectory3DPayload.derived_from`(선택, None이면 JSON에 없음). `make schemas`로
  label_record·episode_graph·export_intervals·ai_review_unit 스키마 재생성. DB 변경 없음 (페이로드는
  JSON 열, 카메라 자세는 기존 `trajectory3d` 라벨 종류).
  정책: `prelabel.yaml camera_pose`·`world_frame`(새 절).

## 배경

ADR 0044·0045의 손목·손가락 끝 3D는 그 시각 카메라 좌표뿐이다. 머리에 단 바디캠은 계속 움직이므로
카메라 좌표 궤적은 손이 가만히 있어도 움직인다. 로봇 학습 데이터(ADR 0047)와 도구-표면 관계는 세션 동안
고정된 월드 좌표가 필요하다 (F-PAR-26). 바디캠 IMU는 수집 단계가 이미 꺼낸다 (`dlp_media.imu`, GPMF·
사이드카), IMU와 바디캠의 시계 관계는 동기화 단계가 정한다 (`Stream.to_master_ms`).

## 라이선스 조사 (ADR 0010: 판매용 라벨이므로 상업 사용 가능해야 한다)

| 후보 | 코드 | 가중치·학습 데이터 | 실행 | 판단 |
| --- | --- | --- | --- | --- |
| Basalt (시각-관성 오도메트리·매핑) | BSD-3-Clause (gitlab.com/VladyslavUsenko/basalt) | 학습 가중치 없음 | CPU, C++ 빌드, 카메라-IMU 외부 파라미터·IMU 잡음 캘리브레이션 필요 | **허용**. 다음 단계의 외부 엔진 후보 (`TODO(real-model)`) |
| Kimera-VIO | BSD-2-Clause (MIT-SPARK, GTSAM·OpenGV 의존, 모두 BSD 계열) | 없음 | CPU, C++ 빌드 | 허용 (Basalt와 같은 등급, 빌드 의존성이 더 무겁다) |
| DROID-SLAM | BSD-3-Clause | `droid.pth`(TartanAir 합성 데이터로 학습, 가중치에 별도 라이선스 표시 없음) | CUDA GPU 11 GB 이상 | **검토(review)**: 가중치·학습 데이터 조건을 법무가 확인하기 전에는 쓰지 않는다. CPU 경로 없음 |
| ORB-SLAM3, OpenVINS | **GPL-3.0** | 없음 | CPU | **쓰지 않는다**. GPL은 실행 결과(자세)에는 미치지 않고, 내부 서버에서 별도 프로세스로만 돌리면 배포 의무도 생기지 않는다. 그러나 이 플랫폼을 고객 설비에 설치·배포하거나 코드를 고쳐 묶는 순간 전체 소스 공개 의무가 생기고, 그 경계를 운영에서 지키는 비용이 크다. 상업 선택지를 깨끗하게 두기 위해 제외한다 |
| MASt3R, DUSt3R (및 MASt3R-SLAM) | **CC BY-NC-SA 4.0** | 같음 | GPU | **금지(forbidden)**: 비상업 |
| Open3D RGB-D 오도메트리 | MIT | 없음 | CPU | 허용. IMU 없는 세션의 영상 전용 대안 후보 |
| 내장 구현 (이 ADR) | OpenCV(Apache-2.0)·numpy(BSD) | 없음 (깊이는 기존 Depth Anything V2 Metric, ADR 0010 분류 그대로) | CPU | **채택** |

결론: 상업 사용이 확실하고 CI·CPU에서 바로 도는 **내장 IMU 시각 오도메트리**를 먼저 둔다. 루프 닫힘·묶음
조정이 있는 외부 엔진은 BSD 계열(Basalt 우선)만 같은 어댑터(`CameraPoseEstimator`)로 붙인다. 새 가중치가
없으므로 `config/models.yaml`은 바뀌지 않는다.

## 결정

### 1. 좌표계와 자세 표현

- 세션 월드 좌표: Z 위(중력 반대), 원점 = 첫 자세의 카메라 중심, X = 첫 자세 광축의 수평 성분, Y = Z x X.
  미터. 세션마다 다르다 (세션 사이 정렬은 하지 않는다).
- 자세는 새 라벨 종류가 아니라 기존 `trajectory3d`로 둔다 (계약에 이미 있던 자리: entity_id="camera",
  `EntityKind.CAMERA`, 쿼터니언 qw..qz): frame=world, source_3d=visual_inertial_slam, 표본 x·y·z = 카메라
  중심, qw..qz = 카메라 → 월드 회전(해밀턴 단위 쿼터니언, w ≥ 0), 표본 신뢰도. 라벨 ID
  `<세션>-<바디캠>-campose-<버전 태그>`.
- 월드 좌표 손·물체 궤적은 카메라 좌표 궤적을 바꾼 **새** trajectory3d다 (frame=world, source_3d는 원래 값,
  `derived_from` = (카메라 좌표 궤적 ID, 자세 ID)). 원래 카메라 좌표 궤적은 그대로 남는다. 표본 방식은 원래
  값, 신뢰도 = 원래 표본 신뢰도 x 자세 신뢰도(보간 두 표본 중 낮은 쪽). 라벨 ID
  `<세션>-<바디캠>-world3d-<버전 태그>-<개체>-<부위|center>`.
- `derived_from`은 선택 필드다. None이면 JSON에 싣지 않아 예전 라벨의 JSON·내용 해시가 그대로다
  (`test_trajectory_derived_from_is_optional_and_world_pose_round_trips`).

### 2. 추정기 어댑터와 내장 구현 (`dlp_prelabel.campose`)

`CameraPoseEstimator`(name, version, `estimate(frames, imu, imu_to_camera, intrinsics, depth)`)에
- `ImuVisualOdometry` (내장, CPU):
  - 회전: IMU 자이로 적분 + 가속도 기울기 보정(상보 필터, 월드 Z 쪽 수평축 회전이라 요는 건드리지 않음).
    IMU가 덮는 첫 프레임부터 `init_window_ms`의 평균 가속도로 중력을 정하고, 그 구간이 정지면 자이로 평균을
    편향으로 뺀다. 카메라 회전 = IMU 회전 x (IMU → 카메라 외부 파라미터)ᵀ.
    2026-10-19 개정 (판 `imuvo2`): 정지는 자이로·가속도 크기의 표준편차뿐 아니라 자이로 평균 크기
    (`static_gyro_mean_max`)와 가속도 크기 평균(중력 ± `accel_tolerance`)도 본다 (F-PAR-46: 일정한 속도로
    돌며 시작하면 크기 표준편차가 0에 가까워 정지로 오판하고 실제 회전을 편향으로 빼, 2.9초 뒤 33도
    틀렸다). 문턱보다 느린 회전은 여전히 정지로 보일 수 있다 (오차 ≤ 문턱 x 세션 길이). IMU 첫 표본보다
    앞선 프레임은 건너뛴다 (F-PAR-47: 사이드카 동기화로 IMU가 몇 ms만 늦어도 자세가 0개였다). IMU 표본
    간격이 `max_gap_ms`를 넘으면 그 앞에서 멈추고 `lost_at_ms`를 남긴다 (F-PAR-48: 빈틈을 건너 적분해 1초
    빈틈에 회전 28도·위치 0.95 m를 지어냈다). 초기화 구간 안의 빈틈은 거부한다. LK 종료 기준도 정책 값이다
    (`features.lk_max_iterations`·`lk_epsilon`, F-PAR-55). 화면비가 다른 캘리브레이션은 쓰지 않고 화각
    근사로 떨어진다 (F-PAR-44, `lift3d.calibration_fits`).
  - 이동: 직전 자세 프레임의 특징점(Shi-Tomasi, 피라미드 LK, 앞뒤 왕복 검사)을 메트릭 깊이로 월드 3D에
    올리고, 지금 프레임 2D 투영식을 회전 고정 상태에서 카메라 중심 3개 미지수의 선형식으로 푼다 (두 점
    RANSAC → 깊이 역수 가중 재풀이). 배율은 깊이 모델에서 온다.
  - 인라이어가 `min_inliers`보다 적으면 그 프레임은 자세를 내지 않고, `max_lost_ms` 넘게 못 잡으면 그 뒤는
    멈춘다 (같은 월드 좌표로 이을 수 없으므로 지어내지 않는다, `lost_at_ms`).
  - 신뢰도: base x 인라이어 수 항 x 재투영 항, 정지 초기화가 없으면 `no_static_init`, 캘리브레이션 없으면
    `hfov_approx`를 곱한다.
- `OraclePoseEstimator` (CPU stub): 정답 자세를 그대로 낸다 (러너 시험).
- `TODO(real-model)`: 외부 SLAM 엔진(Basalt). `adapters/stubs.SLAM_ENGINE`과 `prelabel.yaml camera_pose`에
  표시. 예전 `stubs.CAMERA_POSE`(빈 결과 stub)는 지웠다.

### 3. 시간 동기화와 입력 조건

- IMU 시각은 `Stream.to_master_ms`(offset + 사람 조정 + 배율)로 바디캠 PTS ms로 바꾼다. IMU 스트림은
  sync_method가 unsynced가 아닌 첫 IMU 스트림이다 (내장 IMU는 shared_clock).
- IMU → 카메라 회전: `Session.calibration.camera_imu_extrinsics`(수집 매니페스트, 4x4)의 회전 부분 →
  정책 `camera_pose.imu.default_imu_to_camera`(기본 null) → 없으면 자세를 만들지 않고 이유를 남긴다.
  세션 값도 정책 값과 같은 기준으로 검사한다 (`posemath.rotation_problem`: 3x3·4x4, 직교, 행렬식 양수,
  허용 오차 `imu.rotation_tolerance`. 안이면 가장 가까운 회전으로 고친다). 반사·배율·모양이 틀리면 예외
  없이 이유를 남기고 자세를 만들지 않는다 (F-PAR-49: 예전에는 반사 행렬로 자세를 최고 신뢰도로 냈고,
  모양이 틀리면 IMU 없는 세션의 프리라벨까지 실패했다).
- 깊이: 3D 단계의 깊이 모델(`DepthLifter`)을 같이 쓴다. 없으면 자세 단계를 건너뛴다 (배율을 정할 수 없다).
- 조건이 안 되면 `PrelabelSummary.pose_note`에 이유가 남고 CLI가 `[자세 없음]`으로 보인다.
- 이전 자세 처리 (2026-10-19, F-PAR-50·56): 자세는 사람 검수 대상이 아닌 자동 라벨이라, 입력이 더는
  성립하지 않으면(IMU 없음·동기화 해제, 외부 파라미터 없음·잘못됨) 검수 전 이전 자세를 지운다 (이유에
  지운 수를 붙인다). 깊이 모델이 없어 다시 추정할 수 없으면 센서 입력(`+s`)이 지금 세션과 다른 자세만
  지우고 `[자세 낡음]`으로 알린다 (재동기화 뒤 프리라벨을 깊이 없이 돌려도 낡은 자세가 남지 않게). 새
  버전 추정이 실패하거나 표본이 0개여도 이전 자세를 지운다. 이어서 월드 좌표 단계가 검수 전 월드 3D를,
  궤적 따르기가 물려받기만 한 월드 3D를 정리한다. 자세 라벨에 내부 파라미터 출처(`intrinsics`)를 남긴다
  (F-PAR-51).

### 4. 단계·버전·멱등 (`dlp prelabel run`, `runner._PoseStage`)

- 자세: `campose-<추정기 판>+p<camera_pose 절 해시>+s<센서 입력 해시>+d<깊이 입력 해시>` (2026-10-19,
  예전 `+i<입력 해시>`). 센서 입력 = IMU 스트림(위치·offset·배율·사람 조정·방법), IMU → 카메라 회전, 내부
  파라미터(출처·값, ADR 0045). 깊이 입력 = 깊이 가중치 판, 화각 근사 값, 화면비 허용 오차 (예전에는 depth
  절 전체 해시가 든 깊이 단계 버전이라 손 맞춤 정책만 바꿔도 자세를 다시 추정했다, F-PAR-53).
- 월드 좌표: `world3d-1+p<world_frame 절 해시>+i<입력 해시>`. 입력 = 고른 자세 라벨 하나와 바디캠의 현재
  카메라 좌표 궤적((개체, 부위)마다 `track_rank`로 하나)의 라벨 ID.
- 단계가 마지막으로 낸 버전이 지금 버전이면 건너뛰고(2026-10-19 전에는 "이력에 있으면", F-PAR-54:
  입력이 A → B → A로 되돌아오면 B 결과가 남았다. 지금은 A를 `+r<n>` 세대로 다시 만든다, ADR 0051), 버전이
  바뀌면 검수 전 이전 결과만 `retractions()`로 지운다. 같은 (개체, 부위, 좌표계)의 검수된·사람 궤적이
  있으면 새 궤적은 버린다 (ADR 0026, `drop_protected_trajectories`). 2D 검수를 물려받기만 한 월드 좌표
  궤적은 막지 않는다 (ADR 0051).
  회귀: `packages/prelabel/tests/test_runner.py::test_camera_pose_and_world_trajectories_are_idempotent_and_keep_reviewed`.

### 5. 바뀌지 않는 것

- 카메라 좌표 3D 단계(ADR 0044·0045)와 그 버전.
- LeRobot 사람 모드 observation.state는 카메라 좌표 궤적만 읽는다 (월드 좌표 궤적을 같은 (손, 부위) 열에
  섞지 않게 `frame=camera`로 거른다, `packages/export/tests/test_robot.py::test_world_trajectories_do_not_leak_into_human_state`).
  월드 좌표는 로봇 모드(ADR 0047)만 쓴다. COCO·구간 JSON은 3D를 싣지 않는다.
- 관계 단계(도구-표면 접촉)는 좌표계마다 따로 계산한다. 다만 월드 좌표 궤적은 카메라 좌표 궤적의
  사본이라, 그 카메라 좌표 원본이 아직 현재면 관계·커버리지에 쓰지 않는다 (같은 측정을 두 번 써 접촉이
  합쳐지고 커버리지가 반올림 차로 흔들렸다). 작용부 샘플 사이가 `tool_surface.max_sample_gap_ms`보다
  비면(자세 공백) 접촉을 잇지 않는다 (2026-10-19, F-PAR-57).

## 정확도 (합성 정답, `dlp_fixtures.camera_motion`, `packages/prelabel/tests/test_campose.py`)

장면: 무늬 있는 탁자·벽, 320x240, fx 160, 4초(처음 0.6초 정지), 경로 약 1.5 m, 고개 돌림 ±0.3 rad, 숙임
-0.6±0.12 rad, IMU 200 Hz, 자세 표본 100 ms, 정답 깊이(배율 오차 지정 가능). 손 6점은 66 ms 표본이라 자세
사이 보간 오차가 손 오차에 들어간다.

| 조건 | 자세 ATE | RPE(100 ms) 이동·회전 | 최대 회전 오차 | 월드 손 평균·최대 오차 | 신뢰도 |
| --- | --- | --- | --- | --- | --- |
| 잡음 없음 | 2.1 mm | 0.7 mm·0.02도 | 0.09도 | 2.8·12 mm | 0.75~0.8 |
| 자이로 편향 0.01 rad/s·잡음 0.01, 가속도 잡음 0.1 | 3.7 mm | 0.9 mm·0.03도 | 0.47도 | 3.9·12 mm | 0.75~0.8 |
| 깊이 +10% 편향 | 23 mm (경로 1.67 m로 10% 길게) | 4.4 mm | 0.09도 | 17·43 mm | 0.75~0.8 |
| 처음부터 움직임 (정지 초기화 없음) | 23 mm | 0.9 mm | 7.1도 (중력 방향 초기 오차) | 74·90 mm | 0.53~0.56 |
| IMU 시계 오프셋 120 ms, 오프셋 적용 (IMU가 영상 전체를 덮음) | 2.0 mm | 0.8 mm | 0.09도 | 3.0·16 mm | — |
| 같은 경우 오프셋을 잊음 | 359 mm | — | 10.1도 | — | — |

시험 상한: 잡음 없음 ATE 1 cm·최대 회전 0.2도, 잡음·편향 ATE 1.5 cm·0.5도, 깊이 편향은 경로 길이 비
1.05~1.15, 오프셋 적용/미적용 회전 오차 비 3배 이상, 월드 손(내장 자세) 평균 1.5 cm·최대 3 cm.
**실영상 정확도는 측정하지 않았다** (F-PAR-29). 실제로는 특징점이 적은 장면(흰 벽), 빠른 고개 돌림의
흐림, 단안 깊이 배율 오차(장면마다 수~수십 %), 롤링 셔터, 렌즈 왜곡(F-PAR-28)이 더해진다.

## 결과

- 장점: 새 DB 표·라벨 종류 없이 세션 월드 좌표 손 궤적이 생긴다. 상업 사용이 확실한 코드만 쓰고 CPU·CI에서
  돈다. 외부 SLAM은 같은 어댑터로 붙일 수 있다.
- 비용·한계: 루프 닫힘·묶음 조정이 없어 거리에 비례해 드리프트가 쌓인다. 배율은 깊이 모델 배율 오차를
  그대로 따른다. 요는 자이로 편향만큼 천천히 돈다 (가속도는 요를 고치지 못한다). 처음에 가만히 있지 않으면
  중력 방향이 수 도 틀린다 (신뢰도로만 드러난다). 자세 표본마다 깊이 추론 1회 비용.
- 열린 결정: 카메라-IMU 외부 파라미터·시간 오프셋을 데이터에서 추정할지(자이로와 영상 회전 정렬,
  Kalibr(BSD-3) 등, F-PAR-30), 외부 엔진(Basalt) 연동 시점과 IMU 없는 세션의 영상 전용 경로(Open3D,
  F-PAR-31).

## 개정 (2026-10-28)

- 월드 좌표 표본은 카메라 좌표 표본의 물리 제약 표시(`flags`, `dlp_prelabel.constraints3d`)를 그대로 물려받는다
  (F-PAR-78). 결과가 바뀌므로 단계 버전 접두사를 `world3d-2`로 올렸다 (검수 전 `world3d-1` 결과는 다음 실행에서
  지우고 다시 만든다).
- 3D·월드 라벨 ID가 128자를 넘으면 단계 접두사(`<세션>-<스트림>-3d-`·`-world3d-`)는 그대로 두고 그 뒤를
  `h<전체 ID sha256 16자(자리가 모자라면 8자까지)>`로 줄인다 (`dlp_prelabel.common.fit_label_id`, F-PAR-79).
  128자 안의 ID는 바뀌지 않는다.
