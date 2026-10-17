# 고객 로봇 등록과 리타기팅 내보내기 절차 (운영자용)

기준: ADR 0046(바디캠 자세·세션 월드 좌표 손 3D), ADR 0047(로봇 리타기팅 내보내기). 고객이 로봇 팔 설계를
주면 사람 1인칭 손 동작을 그 로봇 관절 궤적으로 옮긴 LeRobot 에피소드를 만든다. 영상은 바꾸지 않는다.

[인수인계 첫 화면](README.md) · [카메라 캘리브레이션](camera-calibration.md)

## 0. 고객에게 받을 것

| 항목 | 내용 | 없으면 |
| --- | --- | --- |
| 로봇 설계 파일 | URDF (관절 이름·origin·axis·limit, 메시는 없어도 된다). MJCF만 있으면 URDF로 변환해 받는다 | 진행 불가 |
| 끝단(도구 중심점) 링크 | 그리퍼가 물체를 잡는 점에 있는 링크 이름. 없으면 고정 관절로 링크를 하나 더해 달라고 한다 | 손목 링크로 잡으면 수 cm 어긋난다 |
| 끝단 좌표의 축 | 접근 축(손목 → 집게 끝 방향)과 닫힘 축(집게가 여닫히는 방향)이 끝 링크 좌표에서 무엇인지 | 영점 자세 그림으로 운영자가 정한다 (아래 3) |
| 팔 관절 목록과 순서 | 기준 링크 → 끝 링크의 움직이는 관절 | URDF에서 읽어 확인받는다 |
| 그리퍼 | 종류(평행 그리퍼만 지원), 관절 이름, 닫힘·열림 관절 값 | 다지 손은 아직 안 된다 (F-COE-17) |
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
   신뢰도가 낮게 기록된다).
4. `dlp sync run <세션>` (IMU가 사이드카면 동기화가 오프셋을 정한다. 내장 IMU는 shared_clock).
5. `dlp prelabel run <세션>`: 손 2D → 카메라 좌표 3D → 카메라 자세 → 월드 좌표 3D까지 돈다. 출력의
   `카메라 자세 표본 N개, 월드 좌표 3D 궤적 M개`를 확인한다. `[자세 없음] <이유>`가 나오면 이유(동기화된
   IMU 없음, 외부 파라미터 없음, 깊이 모델 없음, IMU 초기화 실패)를 고친다. `... ms부터 추적을 잃음`이면
   그 뒤 구간에는 월드 좌표가 없다.
6. 3D 궤적에는 검수 화면이 없다. 대신 3D 손 궤적(카메라·월드 좌표)은 그것을 계산한 2D 손 트랙의 검수를
   물려받는다 (ADR 0048). 2D 손 트랙을 CVAT에서 승인·수정하거나 표본 검증하면 수집·`dlp review verify`·
   `dlp prelabel run`이 3D에 검수 상태를 기록하고, 기본 내보내기(검수 라벨만)에 손 목표가 생긴다.
   2D를 고치면 그 2D에서 물려받기만 한 예전 3D는 수집·완료 판정이 지우고(ADR 0050), `dlp prelabel run
   <세션>`이 고친 2D에서 3D를 다시 만든다. 카메라 자세는 사람 검수 대상이 아닌 자동 라벨이라 완료
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
  orientation_mode: full                 # 6축 이상이면 full, 5축이면 approach
  orientation_tolerance_rad: 0.35
  orientation_weight: 0.1
  smoothness_weight: 0.01
  damping: 0.0001
  max_iterations: 100
  restarts: 8
  velocity_scale: 0.5
```

확인 (로더가 sha256·관절 목록·그리퍼 관절을 검사하고, 영점 자세의 끝 링크 축·접근·닫힘 축·최대 팔
길이를 보인다. 실패하면 종료 코드 1과 이유):

```bash
uv run dlp robot check acme_arm
```

## 3. 축·배치 정하기

1. `dlp robot check <ID>`가 보이는 영점(q = 0) 자세의 끝 링크 축(기준 좌표)을 본다. 팔을 편 자세에서
   집게 끝이 향하는 축이 접근 축, 집게가 여닫히는 축이 닫힘 축이다.
2. 배치: `episode_centroid`는 에피소드마다 손 잡기 점 평균을 `workspace_center`에 둔다 (작업 자리가
   세션마다 달라도 로봇 작업 공간 가운데에 온다). 사람과 로봇의 위치 관계가 중요하면 `world_fixed`로
   세션 월드 좌표(Z 위, 원점 = 첫 자세 카메라, X = 첫 시선의 수평 방향)에 로봇 기준을 고정한다.
3. `position_scale`은 사람 팔 길이 대비 로봇 팔 길이로 시작한다. 배율을 바꾸면 리타기팅 판이 바뀐다.

## 4. 내보내기

```bash
uv run dlp export lerobot <데이터셋 버전> --target acme --robot acme_arm [--include-unreviewed]
```

출력에 에피소드마다 `목표 a%, 도달 b%, 최대 위치 오차 c mm, keep|drop`이 나온다.

- 결과: `observation.state`·`action` = 로봇 관절(rad, URDF 이름), `observation.human_state` = 사람
  특징, `retarget.*` = 프레임별 목표 유무·도달·오차·속도 한계·검증 등급. `meta/dlp_retarget.json`과
  manifest(`robot`, `details.retarget`)에 로봇 ID, 모델 sha256, 리타기팅 판, 에피소드 보고가 있다.
- 닿지 못한 프레임은 값이 아니라 `retarget.reachable=0`으로 표시된다 (가장 가까운 해). 고객 학습에서 그
  프레임을 빼거나 가중치를 낮추게 안내한다.
- 도달 비율이 낮은 에피소드를 아예 빼려면 `config/policies/retarget.yaml`에서 `unreachable: drop_episode`와
  `max_unreachable_fraction`·`min_target_fraction`을 정한다 (정책 해시가 판에 들어간다).
- mink 풀이기(`solver: mink`)는 격리 환경을 받는다 (처음 수십 MB). 결과 판정 규칙은 내장과 같다.

## 5. 품질 확인 (납품 전)

1. 에피소드 보고의 `reachable_fraction`, `frames_out_of_reach`, `velocity_limited_frames`,
   `longest_unreachable_ms`를 본다. 팔 길이 밖이 많으면 배치·배율을, 속도 한계가 많으면 velocity_scale
   또는 고객의 실제 속도 한계를 다시 정한다.
2. 고객 시뮬레이터(MuJoCo·Isaac 등)에서 관절 궤적을 재생해 충돌·작업 성공을 본다. 이 플랫폼은 충돌·
   동역학을 검사하지 않는다 (F-COE-20).
3. 영상 속 손은 사람 손 그대로다 (F-COE-18). 고객 정책 학습이 영상을 쓰면 이 차이를 알린다.

## 문제가 생기면

| 증상 | 원인·조치 |
| --- | --- |
| `모델 파일 sha256이 다릅니다` | 고객 파일이 바뀌었다. 새 버전이면 sha256을 새로 적는다 (판이 바뀐다) |
| `arm.joints ...가 ... 연쇄의 움직이는 관절 ...와 다릅니다` | joints 순서·이름을 연쇄 출력대로 고친다 |
| `저장소 안에는 공개 시험 모델만 둡니다` | 고객 설정을 저장소에 두었다. DLP_ROBOTS_DIR로 옮긴다 |
| 모든 프레임 `목표 0%` | 월드 좌표 손 3D가 없다 (1의 5 확인) 또는 검수 전이라 기본 정책이 뺐다 (1의 6) |
| `리타기팅 정책이 모든 에피소드를 버렸습니다` | drop_episode 기준을 어긴 에피소드뿐이다. 보고의 reason을 본다 |
