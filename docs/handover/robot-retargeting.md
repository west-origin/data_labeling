# 고객 로봇 등록과 리타기팅 내보내기 절차 (운영자용)

기준: ADR 0046(바디캠 자세·세션 월드 좌표 손 3D), ADR 0047(로봇 리타기팅 내보내기), ADR 0068(다지 손
로봇). 고객이 로봇 팔 설계를 주면 사람 1인칭 손 동작을 그 로봇 관절 궤적으로 옮긴 LeRobot 에피소드를
만든다. 영상은 바꾸지 않는다. 로봇 끝이 평행 그리퍼면 엄지-검지 거리를 그리퍼 관절 하나로, 손가락
관절이 있는 손(다지 손)이면 손목과 손가락 끝 위치를 팔 관절 + 손가락 관절로 옮긴다.

[인수인계 첫 화면](README.md) · [카메라 캘리브레이션](camera-calibration.md)

## 0. 고객에게 받을 것

| 항목 | 내용 | 없으면 |
| --- | --- | --- |
| 로봇 설계 파일 | URDF (관절 이름·origin·axis·limit, 메시는 없어도 된다). MJCF만 있으면 URDF로 변환해 받는다 | 진행 불가 |
| 끝단(도구 중심점) 링크 | 그리퍼가 물체를 잡는 점에 있는 링크 이름. 없으면 고정 관절로 링크를 하나 더해 달라고 한다 | 손목 링크로 잡으면 수 cm 어긋난다 |
| 끝단 좌표의 축 | 접근 축(손목 → 집게 끝 방향)과 닫힘 축(집게가 여닫히는 방향)이 끝 링크 좌표에서 무엇인지 | 영점 자세 그림으로 운영자가 정한다 (아래 3) |
| 팔 관절 목록과 순서 | 기준 링크 → 끝 링크의 움직이는 관절 | URDF에서 읽어 확인받는다 |
| 그리퍼 (평행 그리퍼 로봇) | 관절 이름, 닫힘·열림 관절 값 | 진행 불가 |
| 다지 손 (손가락 관절이 있는 로봇, ADR 0068) | 팔과 손이 한 URDF에 있어야 한다. 사람 손목에 대응하는 손목 링크, 손가락마다 끝 링크 이름과 사람 손가락(엄지·검지·중지·약지·새끼) 가운데 어느 것에 대응하는지 (1~5개), 로봇 손이 왼손·오른손 어느 쪽 모양인지. 위 "끝단 링크"·"끝단 좌표의 축" 행은 필요 없다 | 손가락 끝 링크가 없으면 고정 관절로 더해 달라고 한다. 연동(mimic) 손가락 관절이 있거나 손가락끼리 관절을 함께 쓰는 손은 지금 안 된다 (F-COE-93) |
| 손 크기 배율 (다지 손) | 사람 손 대비 로봇 손 크기. 운영자가 정한다 (아래 3의 5) | 1.0으로 먼저 보여 주고 손가락별 `frames_out_of_reach`로 고친다 |
| 작업 공간·배치 | 로봇 기준을 어디에 둘지, 사람 동작을 몇 배로 줄일지, 작업 중심 | 기본 `episode_centroid` + 배율 0.5로 먼저 보여 준다 |
| 관절 속도 한계 | URDF limit velocity가 실제와 다르면 실제 값 | URDF 값 x velocity_scale |
| 관절 값 단위·영점 | 받는 쪽 컨트롤러의 단위(rad·도·0~100 등)와 영점 정의 | 내보내기는 URDF 기준 rad다. 변환은 받는 쪽에서 |
| 라이선스·비밀 유지 | 파일 사용 범위 (학습 데이터 생성에 써도 되는지) | 진행 불가 |

## 1. 세션 쪽 준비 (월드 좌표 손 3D)

1. 바디캠 내부 파라미터 등록 ([camera-calibration.md](camera-calibration.md)).
2. 수집 매니페스트에 IMU 스트림과 카메라-IMU 외부 파라미터(IMU → 카메라 4x4, 회전만 쓴다)를 적는다.
   기종마다 IMU 축이 다르므로 기본값을 쓰지 않는다 (`prelabel.yaml camera_pose.imu.default_imu_to_camera`는
   null). 외부 파라미터 추정 도구는 아직 없다 (F-PAR-30): 기종 설명서의 IMU 축 그림이나 Kalibr(BSD-3)
   결과를 쓴다.

   ```yaml
   calibration:
     camera_id: gopro-c3471325-linear-4k
     camera_imu_extrinsics: [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
   ```
3. 촬영 시작 때 1초쯤 머리를 가만히 둔다 (중력 방향·자이로 편향 초기화. 안 하면 중력이 수 도 틀리고
   신뢰도가 낮게 기록된다. 몸을 일정하게 돌리며 시작해도 정지가 아니다: 평균 각속도가 정책 문턱
   `camera_pose.imu.static_gyro_mean_max`보다 크면 편향을 빼지 않는다. 그보다 느린 회전은 정지로 보일 수
   있다).
4. `dlp sync run <세션>` (IMU가 사이드카면 동기화가 오프셋을 정한다. 내장 IMU는 shared_clock).
5. `dlp prelabel run <세션>`: 손 2D → 카메라 좌표 3D → 카메라 자세 → 월드 좌표 3D까지 돈다. 출력의
   `카메라 자세 표본 N개, 월드 좌표 3D 궤적 M개`를 확인한다. `[자세 없음] <이유>`가 나오면 이유(동기화된
   IMU 없음, 외부 파라미터 없음·회전이 아님(반사·배율), 깊이 모델 없음, IMU 초기화 실패, 자세 표본 0개
   (IMU가 영상 구간을 덮지 않음))를 고친다. 이유에 "이전 자세 N개를 지움"이 붙으면 낡은 자세와 그 월드
   좌표 3D를 지운 것이다. `[자세 낡음]`은 재동기화·캘리브레이션 뒤 깊이 모델 없이 돌려 낡은 자세를 지웠다는
   뜻이다 (깊이 모델로 다시 돌린다). `... ms부터 추적을 잃음`이면 그 뒤 구간(추적 실패 또는 IMU 기록
   끊김)에는 월드 좌표가 없다.
6. 3D 궤적에는 검수 화면이 없다. 대신 3D 손 궤적(카메라·월드 좌표)은 그것을 계산한 2D 손 트랙의 검수를
   물려받는다 (ADR 0048). 2D 손 트랙을 CVAT에서 승인·수정하거나 표본 검증하면 수집·`dlp review verify`·
   `dlp prelabel run`이 3D에 검수 상태를 기록하고, 기본 내보내기(검수 라벨만)에 손 목표가 생긴다.
   2D를 고치면 그 2D에서 물려받기만 한 예전 3D는 수집·완료 판정이 지우고(ADR 0050), `dlp prelabel run
   <세션>`이 고친 2D에서 3D를 다시 만든다. 다시 만들기 전에는 `dlp review verify`가 완료를 막고 이 명령을
   안내한다 (ADR 0051, `review.yaml verify.require_current_3d`). 캘리브레이션·손 크기를 등록하거나 정책을
   바꾼 뒤 `dlp prelabel run`을 돌리면 물려받기만 한 3D가 새 값으로 바뀌고 같은 2D 검수를 물려받는다. 카메라 자세는 사람 검수 대상이 아닌 자동 라벨이라 완료
   판정을 막지 않는다 (`review.yaml verify.automatic`). manifest `verification_policy.trajectory3d`에 근거
   (`inherited_from_2d_hand_track`, 자세 `automatic_unreviewed`, `pose_derived`, 근거별 수)가, 로봇 모드
   `meta/dlp_retarget.json`에 `camera_pose: automatic_unreviewed`가 남는다.
   `--include-unreviewed`는 그대로 쓸 수 있다.

## 2. 로봇 설정 만들기 (저장소 밖)

고객 파일은 저장소에 넣지 않는다. 운영 서버의 보호된 디렉터리를 `DLP_ROBOTS_DIR`로 둔다.

```bash
export DLP_ROBOTS_DIR=/srv/dlp/robots          # 저장소 밖, 접근 통제
mkdir -p $DLP_ROBOTS_DIR/models/acme_arm
cp acme_arm.urdf $DLP_ROBOTS_DIR/models/acme_arm/
sha256sum $DLP_ROBOTS_DIR/models/acme_arm/acme_arm.urdf
```

`$DLP_ROBOTS_DIR/acme_arm.yaml` (파일 이름 = robot_id). 형식은 공개 예시 `config/robots/so101.yaml`과 같다.

```yaml
robot_id: acme_arm
display_name: ACME 6축 팔 (계약 C-2026-031)
model:
  format: urdf
  path: models/acme_arm/acme_arm.urdf     # 이 YAML 기준 상대 경로 또는 절대 경로
  sha256: <위에서 잰 값>
  license: 고객 제공, 학습 데이터 생성 한정 (계약 C-2026-031)
  source: 고객 전달 2026-10-20, 담당 홍길동
  public: false
base_link: base_link
arm:
  hand: right                            # 이 팔이 따라갈 작업자 손
  tip_link: tcp_link
  joints: [j1, j2, j3, j4, j5, j6]       # 기준 → 끝 연쇄의 움직이는 관절 (순서까지 같아야 한다)
  approach_axis: [0.0, 0.0, 1.0]
  closing_axis: [0.0, 1.0, 0.0]
  gripper: {type: parallel, joint: finger_joint, closed: 0.0, open: 0.8, closed_distance_m: 0.01, open_distance_m: 0.09}
  placement: {mode: episode_centroid, position_scale: 0.7, workspace_center: [0.45, 0.0, 0.15]}
ik:
  position_tolerance_m: 0.01
  orientation_mode: full                 # 접근 축 둘레 롤 관절(손목 끝 roll)이 있으면 full (아래 3의 4)
  orientation_tolerance_rad: 0.35
  orientation_weight: 0.1
  smoothness_weight: 0.01
  damping: 0.0001
  max_iterations: 100
  restarts: 8
  velocity_scale: 0.5
```

확인 (로더가 sha256·관절 목록·그리퍼 관절을 검사하고, 영점 자세의 끝 링크 축·접근·닫힘 축·최대 팔
길이를 보인다. 실패하면 종료 코드 1과 이유). 로더가 거부하는 것:

- robot_id가 영숫자로 시작하지 않거나 `/`가 있음 (설정 디렉터리를 벗어나지 못하게).
- `public: false` 설정의 모델 파일이 저장소 작업 트리 안에 있음 (고객 파일은 `DLP_ROBOTS_DIR`).
- 접근·닫힘 축이 단위 벡터가 아니거나 서로 직교하지 않음 (예: 대각 축은 [0.7071068, 0.7071068, 0]으로
  적는다. 조용히 정규화하지 않는다).
- 그리퍼 관절이 고정·mimic 관절이거나 팔 관절과 같음, closed·open이 URDF 한계 밖 (단위는 rad 또는 m,
  도로 적으면 거부).
- 팔 연쇄에 floating·planar·mimic 관절, lower/upper가 없는 revolute·prismatic 관절.

```bash
uv run dlp robot check acme_arm
```

### 다지 손 로봇 (ADR 0068)

손가락 관절이 있는 손이면 `arm`에 `gripper`·`tip_link`·접근·닫힘 축 대신 `dexterous_hand`를 적는다
(둘을 함께 적으면 거부). `model`·`base_link`·`placement`는 위와 같다. `$DLP_ROBOTS_DIR/acme_hand.yaml`:

```yaml
robot_id: acme_hand
display_name: ACME 6축 팔 + 4지 손 (계약 C-2026-044)
model:
  format: urdf
  path: models/acme_hand/acme_hand.urdf   # 팔과 손이 함께 있는 URDF
  sha256: <sha256sum으로 잰 값>
  license: 고객 제공, 학습 데이터 생성 한정 (계약 C-2026-044)
  source: 고객 전달 2026-11-20, 담당 홍길동
  public: false
base_link: base_link
arm:
  hand: right                             # 따라갈 작업자 손 (로봇 손과 같은 쪽)
  joints: [j1, j2, j3, j4, j5, j6]        # 기준 → 손목 링크 연쇄의 움직이는 관절 (손가락 관절은 적지 않는다)
  placement: {mode: episode_centroid, position_scale: 0.7, workspace_center: [0.45, 0.0, 0.20]}
  dexterous_hand:
    wrist_link: palm_link                 # 사람 손목에 대응하는 링크 (원점이 손목 목표를 따라간다)
    hand_scale: 1.0                       # 사람 (손가락 끝 - 손목) 벡터에 곱하는 배율 (아래 3의 5)
    fingertip_tolerance_m: 0.01           # 손가락 끝 위치 오차가 이 안이면 그 손가락은 도달
    fingers:                              # 사람 손가락 끝 부위 → 로봇 손가락 끝 링크 (1~5개, 이 순서로 내보낸다)
      - {human_part: thumb_tip, tip_link: thumb_tip_link}
      - {human_part: index_tip, tip_link: index_tip_link}
      - {human_part: middle_tip, tip_link: middle_tip_link}
      - {human_part: ring_tip, tip_link: ring_tip_link}
ik:
  position_tolerance_m: 0.01              # 손목 위치 오차가 이 안이고 목표가 있는 손가락이 모두 도달이면 프레임 도달
  wrist_weight: 1.0                       # 손목 오차 가중치 (손가락 끝은 1). 1보다 작으면 손가락 끝을 먼저 맞춘다
  smoothness_weight: 0.0001               # 팔만 있는 로봇(0.01)보다 작게 (아래 설명)
  damping: 0.000001                       # 팔만 있는 로봇(0.0001)보다 작게
  max_iterations: 200
  restarts: 8
  velocity_scale: 0.5
```

- `human_part`는 손가락 끝 3D 궤적의 부위 이름이다 (온톨로지 hand_joints: `thumb_tip`, `index_tip`,
  `middle_tip`, `ring_tip`, `pinky_tip`). 손목 부위는 `retarget.yaml hand_parts.wrist`에서 읽는다.
  이름이 온톨로지에 없으면 내보내기 시작 때 거부한다.
- 손가락 관절은 적지 않는다. 로더가 URDF에서 손목 링크 → 끝 링크 경로의 움직이는 관절을 읽는다.
  내보내는 관절 순서는 `arm.joints` + 손가락마다(위 순서) 손목 쪽 관절부터다.
- 대응하지 않은 손가락 등 손목 링크 아래의 나머지 움직이는 관절은 풀지도 내보내지도 않는다.
  `dlp robot check`가 "풀지 않는 손 관절"로 보이고 meta `dexterous_hand.unmapped_joints`에 남는다.
  받는 쪽이 그 관절을 어떻게 둘지 정해야 한다.
- 방향(접근·닫힘 축, `orientation_*`)은 적지 않는다. 손목과 손가락 끝의 위치만 맞춘다. 손바닥 방향과
  손가락 마디 각은 사람 손과 같다고 볼 수 없다 (F-COE-95): 손가락 끝이 같은 자리에 오는 자세 가운데
  직전 자세에서 가장 적게 움직인 것이 나온다.
- `damping`·`smoothness_weight`: 손가락 링크는 수 cm라 팔만 있는 로봇의 값을 그대로 쓰면 반복이 덜
  수렴해 오차가 남는다 (시험 로봇에서 0.29 mm). 위 값에서 시작한다.
- `retarget.yaml solver`는 `dls`여야 한다. mink 풀이기는 다지 손을 풀지 않는다 (F-COE-92).

로더가 더 거부하는 것 (이유를 출력한다):

- `wrist_link`·`tip_link`·`base_link`가 URDF에 없음, 끝 링크가 손목 링크 아래에 있지 않음, 손목 링크와
  같은 끝 링크, 손목 링크에서 끝 링크까지 움직이는 관절이 없음.
- 손가락 경로의 mimic 관절·floating·planar 관절 (연동 손가락을 독립 관절로 풀면 실제 손이 낼 수 없는
  관절 조합이 나온다).
- 두 손가락이 같은 움직이는 관절을 씀 (끝 점이 없는 손가락만 따로 유지할 수 없다).
- 같은 `human_part`나 같은 `tip_link`를 두 번 씀, 손가락 0개·6개 이상, 0 이하 `hand_scale`·허용 오차.

```bash
uv run dlp robot check acme_hand
```

출력의 손가락마다 `영점 자세 손목 → 끝 A m, 손목 → 끝 거리 상한 B m`가 손 배율을 정하는 값이다.

## 3. 축·배치 정하기

1. `dlp robot check <ID>`가 보이는 영점(q = 0) 자세의 끝 링크 축(기준 좌표)을 본다. 팔을 편 자세에서
   집게 끝이 향하는 축이 접근 축, 집게가 여닫히는 축이 닫힘 축이다.
2. 배치: `episode_centroid`는 에피소드마다 손 잡기 점 평균을 `workspace_center`에 둔다 (작업 자리가
   세션마다 달라도 로봇 작업 공간 가운데에 온다). 사람과 로봇의 위치 관계가 중요하면 `world_fixed`로
   세션 월드 좌표(Z 위, 원점 = 첫 자세 카메라, X = 첫 시선의 수평 방향)에 로봇 기준을 고정한다.
3. `position_scale`은 사람 팔 길이 대비 로봇 팔 길이로 시작한다. 배율을 바꾸면 리타기팅 판이 바뀐다.
4. `ik.orientation_mode`는 관절 수가 아니라 구조로 고른다 (ADR 0053).
   - `full`: 접근 축 둘레 롤을 바꾸는 관절이 있으면 (영점에서 어떤 관절 축이 접근 축과 평행, 보통 손목 끝
     wrist roll). 그 관절이 사람 손의 롤(엄지 → 검지)을 따라간다. SO-101처럼 pan + pitch 연쇄 끝에 roll이
     있는 5축 팔도 full이다 (접근 축의 요는 따로 맞추지 못하고 오차로 남는다).
   - `approach`: 롤을 바꿀 관절이 없을 때만. 롤 관절이 있는데 approach면 그 관절 값이 사람 손과 무관한
     IK 부산물로 팔린다.
   - `none`: 방향을 맞출 수 없는 팔 (위치만). 회전 오차는 -1(재지 않음)로 나간다.
   - 다지 손 로봇에는 이 항목이 없다 (1의 축 정하기도 필요 없다). 배치(2·3)는 같되 옮기는 점이 잡기 점이
     아니라 손목이다 (`episode_centroid`는 손목 평균을 `workspace_center`에 둔다).
5. 다지 손의 `hand_scale` (ADR 0068). 손가락 끝 목표 = 손목 목표 + `hand_scale` x (사람 손가락 끝 - 사람
   손목)이다. 팔 배율 `position_scale`과 따로다 (팔 궤적을 줄여도 손가락 벌림은 로봇 손 크기에 맞춘다).
   - 시작값: `dlp robot check`의 손가락별 "손목 → 끝 거리 상한" ÷ 그 작업자가 손을 폈을 때 손목 → 그
     손가락 끝 길이. 손가락마다 다르면 가장 작은 값에서 시작한다 (한 배율을 모든 손가락에 쓴다).
   - 한 번 내보내 보고의 손가락별 `frames_out_of_reach`를 본다. 0이 아니면 그 프레임은 사람 손가락이
     로봇 손가락 길이보다 길게 옮겨진 것이니 배율을 줄인다. 배율이 너무 작으면 손가락을 굽힌 자세에서
     끝 점이 손목에 너무 가까워져 관절 한계에 걸린다 (`frames_out_of_reach`는 0인데 그 손가락의
     `reachable_fraction`이 낮다).
   - 배율을 바꾸면 리타기팅 판이 바뀐다. 작업자 손 크기에서 자동으로 정하지 않는다 (F-COE-94).

## 4. 내보내기

```bash
uv run dlp export lerobot <데이터셋 버전> --target acme --robot acme_arm [--include-unreviewed]
```

출력에 에피소드마다 `목표 a%, 도달 b%, 최대 위치 오차 c mm, keep|drop`이 나온다.

- 결과: `observation.state`·`action` = 로봇 관절(rad, URDF 이름), `observation.human_state` = 사람
  특징, `retarget.*` = 프레임별 목표 유무·도달·오차·속도 한계(팔·그리퍼, 재시작 프레임 포함)·검증 등급.
  `meta/dlp_retarget.json`과 manifest(`robot`, `details.retarget`)에 로봇 ID, 모델 sha256, 리타기팅 판,
  에피소드 보고, 도달 판정 기준(`ik`, `hand_parts`, `reachable_definition`, `orientation_error_definition`)이
  있다.
- 손 목표가 하나도 없는 에피소드는 정책과 관계없이 빠진다 (보고에 `drop`과 이유). 목표가 생기기 전 프레임은
  시작 자세(관절 범위 가운데)·닫힌 그리퍼이고 `target_present=0`이다.
- 손이 끊겼다 다시 잡히는 프레임은 직전 자세에 가까운 해로 이어 가고, 속도 한계 때문에 몇 프레임 늦게
  도달할 수 있다 (그 프레임은 `velocity_limited=1`, `reachable=0`).
- 출력 끝의 `경고:`는 궤적 구간 안 손 3D가 기준(`export.yaml lerobot.min_hand_points_3d_coverage`)보다
  덜 채워진 에피소드다. 3D 표본 간격(`prelabel.yaml depth.frame_stride_ms`)과 `lerobot.interp_max_gap_3d_ms`를
  확인한다.
- 닿지 못한 프레임은 값이 아니라 `retarget.reachable=0`으로 표시된다 (가장 가까운 해). 고객 학습에서 그
  프레임을 빼거나 가중치를 낮추게 안내한다.
- 도달 비율이 낮은 에피소드를 아예 빼려면 `config/policies/retarget.yaml`에서 `unreachable: drop_episode`와
  `max_unreachable_fraction`·`min_target_fraction`을 정한다 (정책 해시가 판에 들어간다).
- mink 풀이기(`solver: mink`)는 격리 환경을 받는다 (처음 수십 MB). 결과 판정 규칙은 내장과 같다.
  다지 손 로봇은 mink로 내보낼 수 없다 (거부한다).

다지 손 로봇(ADR 0068)이면 같은 명령이고 결과가 아래처럼 다르다. 출력은 에피소드 줄(손목 기준) 아래에
손가락마다 `손가락 <부위>: 목표 a%, 도달 b%, 최대 위치 오차 c mm, 손 길이 밖 N프레임`이 더 나온다.

- `observation.state`·`action` = 팔 관절 + 손가락 관절 (`arm.joints`, 그 뒤 `fingers` 순서로 손가락마다
  손목 쪽 관절부터).
- `retarget.target_present`는 손목 목표 유무, `retarget.reachable`은 손목과 목표가 있는 손가락 끝이 모두
  허용 오차 안, `retarget.position_error_m`은 손목 오차, `retarget.orientation_error_rad`는 늘 -1이다.
- 손가락별 `retarget.finger_target_present`·`retarget.finger_reachable`·
  `retarget.finger_position_error_m` (열 순서·이름 = `fingers`의 `human_part`). 손가락 끝 점이 없던
  프레임은 그 손가락의 `finger_target_present=0`이고 그 관절은 직전 값 그대로다 (한 번도 없었으면 시작
  자세). 손목 점이 없던 프레임은 `target_present=0`이고 모든 관절이 직전 값이다. 고객 학습에서 이
  표시로 프레임·손가락을 가리게 안내한다.
- `meta/dlp_retarget.json`에 `dexterous_hand`(손목 링크, 팔 관절, 손 배율, 손가락 끝 허용 오차, 손가락마다
  사람 부위 → 끝 링크와 관절, 싣지 않은 손 관절)와 에피소드 보고의 손가락별 통계(`fingers`)가 있다.
- `unreachable: drop_episode`는 손가락마다도 `min_target_fraction`·`max_unreachable_fraction`을 본다
  (이유에 손가락이 적힌다).
- 한 손가락이 닿지 못하는 프레임은 오차를 손목·다른 손가락과 나눠 져 다른 손가락도 허용 오차를 넘을 수
  있다. `wrist_weight`를 낮추면 손목보다 손가락 끝을 먼저 맞춘다.

## 5. 품질 확인 (납품 전)

1. 에피소드 보고의 `reachable_fraction`, `frames_out_of_reach`, `velocity_limited_frames`,
   `longest_unreachable_ms`(이어진 미도달 구간의 첫·마지막 프레임 시각 차, 한 프레임이면 0)를 본다. 팔 길이 밖이 많으면 배치·배율을, 속도 한계가 많으면 velocity_scale
   또는 고객의 실제 속도 한계를 다시 정한다.
2. 고객 시뮬레이터(MuJoCo·Isaac 등)에서 관절 궤적을 재생해 충돌·작업 성공을 본다. 이 플랫폼은 충돌·
   동역학을 검사하지 않는다 (F-COE-20). 다지 손은 손가락끼리·손가락과 손바닥의 충돌도 보지 않는다.
3. 영상 속 손은 사람 손 그대로다 (F-COE-18). 고객 정책 학습이 영상을 쓰면 이 차이를 알린다.
4. 다지 손 로봇: 보고의 손가락별 `target_fraction`(끝 점이 있던 비율), `reachable_fraction`,
   `frames_out_of_reach`(손 배율이 큰지)를 본다. 손바닥 방향·손가락 마디 각은 사람 손과 다를 수 있으니
   (F-COE-95) 고객 시뮬레이터에서 손 모양을 눈으로 확인한다. 다지 손은 합성 정답으로만 검사했고 실제
   손 로봇에서 돌려 보지 않았다.

## 문제가 생기면

| 증상 | 원인·조치 |
| --- | --- |
| `모델 파일 sha256이 다릅니다` | 고객 파일이 바뀌었다. 새 버전이면 sha256을 새로 적는다 (판이 바뀐다) |
| `arm.joints ...가 ... 연쇄의 움직이는 관절 ...와 다릅니다` | joints 순서·이름을 연쇄 출력대로 고친다 |
| `저장소 안에는 공개 시험 모델만 둡니다` | 고객 설정을 저장소에 두었다. DLP_ROBOTS_DIR로 옮긴다 |
| `목표 0%, ..., drop (손 목표가 있는 프레임이 없습니다)` | 월드 좌표 손 3D가 없다 (1의 5 확인) 또는 검수 전이라 기본 정책이 뺐다 (1의 6). 그 에피소드는 내보내지 않는다 |
| `손 부위 [...]가 온톨로지 ... hand_joints에 없습니다` | `export.yaml lerobot.hand_points_3d`나 `retarget.yaml hand_parts`의 이름 오타 |
| `로봇 ID ...: 영숫자로 시작하고` | `--robot`에 경로를 적었다. 파일 이름(robot_id)만 적는다 |
| `리타기팅 정책이 모든 에피소드를 버렸습니다` | drop_episode 기준을 어긴 에피소드뿐이다. 보고의 reason을 본다 |
| `다지 손(dexterous)은 arm.gripper가 아니라 arm.dexterous_hand 절에 적습니다` | `gripper.type: dexterous`로 적었다. 2절의 다지 손 형식으로 고친다 |
| `arm.gripper와 arm.dexterous_hand를 함께 쓸 수 없습니다` | 로봇 하나에 끝단 하나다. 한쪽만 남긴다 |
| `손가락 ...(끝 링크 ...)가 손목 링크 ... 아래에 있지 않습니다` | `tip_link`가 `wrist_link`의 자손 링크가 아니다. 링크 이름을 URDF에서 확인한다 |
| `손가락 ...: ...: mimic 관절(... 종속)은 연쇄에 둘 수 없습니다` | 연동 손가락 관절이 있는 손이다. 지금 지원하지 않는다 (F-COE-93). 그 손가락을 대응에서 빼면 나머지는 풀 수 있다 (뺀 손가락의 관절은 내보내지 않는다) |
| `손가락 ...와 ...가 관절 ...를 함께 씁니다` | 두 끝 링크가 같은 손가락 위에 있거나 손바닥에 함께 쓰는 관절이 있다. 손가락마다 관절이 따로인 대응만 된다 |
| `다지 손 로봇은 내장 풀이기(solver: dls)만 풉니다` | `retarget.yaml solver`가 mink다. dls로 내보낸다 (F-COE-92) |
| 손가락 줄의 `목표`가 에피소드 줄보다 낮거나 0% | 그 프레임에 손목 3D는 있는데 손가락 끝 3D가 없다. 손 맞춤이 실패한 프레임은 손목만 3D로 나오고 손가락 끝은 빠진다 (`prelabel.yaml depth.fingertip_fallback: drop`, ADR 0044). 그 프레임의 손가락 관절은 직전 값(한 번도 없었으면 시작 자세) 그대로 나가고 `retarget.finger_target_present=0`이다 |
| 손가락마다 `손 길이 밖`이 많음 | `hand_scale`이 로봇 손보다 크다 (3의 5) |
