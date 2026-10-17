# ADR 0047: 사람 1인칭 손 동작을 고객 로봇 관절 궤적으로 옮겨 LeRobot 에피소드로 낸다

- 상태: 채택
- 날짜: 2026-10-17
- 관련: ADR 0046(세션 월드 좌표 손 3D, 이 ADR의 입력), ADR 0018·0021·0027(내보내기·격리 환경·가명),
  ADR 0010(상업 사용), ADR 0045(LeRobot 손 3D 특징). 문제 레지스트리 F-COE-17~20(남은 일).
  계약 변경 없음 (라벨·DB 그대로). 새 패키지 `packages/retarget` (`dlp_retarget`, schema에만 의존),
  `dlp_export`가 이를 쓴다 (architecture.md 의존 표 갱신). 설정: `config/robots/<ID>.yaml`(로봇),
  `config/policies/retarget.yaml`(정책). 격리 환경 `scripts/retarget-env`(mink, uv.lock 고정).

## 배경

제품 목표: 고객이 로봇 팔 설계(URDF/MJCF)를 주면, 사람 1인칭 돌봄 영상의 손 동작을 그 로봇 관절 궤적으로
옮겨 LeRobot 에피소드(영상은 그대로)로 납품한다. ADR 0046으로 세션 월드 좌표 손목·손가락 끝이 생겼다.

## 결정

### 1. 로봇 등록 (`dlp_retarget.robot`)

- 로봇 하나 = 설정 YAML 하나 (`<ID>.yaml`): 모델(형식 urdf, 경로, **sha256**, 라이선스, 출처, public),
  기준 링크, 팔(따라갈 손, 끝 링크, 관절 순서, 끝 링크 좌표의 접근·닫힘 축, 그리퍼, 배치), IK 설정.
- 고객 설정·설계 파일은 **저장소에 넣지 않는다**: `DLP_ROBOTS_DIR`(저장소 밖)을 먼저 찾고, 저장소
  `config/robots/`에는 공개 시험 모델만 둔다. 저장소 안 설정이 `public: false`면 로더가 거부한다
  (`test_every_repo_robot_is_a_public_model_with_matching_hash`).
- 모델 파일은 읽기 전에 sha256을 확인한다. 다르면 거부한다. 해시는 리타기팅 판과 내보내기 manifest에 남는다.
- 형식은 URDF만 읽는다 (관절만, 메시·관성 없이). MJCF는 고객에게 URDF를 받거나 변환해 받는다 (MuJoCo의
  URDF 읽기 경로를 mink 풀이기가 이미 쓴다).
- 공개 시험 모델: **SO-101** (TheRobotStudio/SO-ARM100 `Simulation/SO101/so101_new_calib.urdf`,
  **Apache-2.0**, 2026-10-17 받음, sha256 `3a65d2d3…022c`). URDF 원본 그대로와 LICENSE 사본을
  `config/robots/models/so101/`에 두었다 (메시 STL은 넣지 않았다). MuJoCo Menagerie의 SO-ARM100도
  후보였지만 같은 설계의 원 저장소 파일을 택했다.

### 2. 사람 손 → 끝단 목표 (`dlp_retarget.targets`)

- 손 점 세 개(월드 좌표): 손목 W, 엄지 끝 T, 검지 끝 I (`retarget.yaml hand_parts`).
- 끝단(도구 중심점) 위치 = 잡기 점 (T + I) / 2, 접근 축 = (잡기 점 - W) 방향, 닫힘 축 = (I - T)의 접근 축
  수직 성분, 평행 그리퍼 열림 = |I - T|를 [closed_distance_m, open_distance_m] → [closed, open] 관절 값으로
  선형 대응.
- 배치 (`placement`): `world_fixed`(로봇 기준을 세션 월드의 고정 자세에) 또는 `episode_centroid`(그
  에피소드 잡기 점 평균을 로봇 기준 `workspace_center`에, 월드 X·Z = 로봇 X·Z). 둘 다 `position_scale`로
  사람 동작 크기를 로봇 크기로 줄인다 (SO-101 기본 0.5). 배율·배치는 고객과 정하는 값이고 결과에 기록된다.
  어깨·몸통(RTMPose 전신) 기준 배치는 1인칭 바디캠에서 작업자 몸이 거의 안 보여 이번에는 두지 않았다.
- 세 점 중 하나라도 없는 프레임은 목표가 없다 (지어내지 않는다).

### 3. IK 풀이기 (`IkSolver` 어댑터)

| 풀이기 | 라이선스 | 실행 | 쓰임 |
| --- | --- | --- | --- |
| `DlsIkSolver` (내장) | 이 저장소 코드, numpy(BSD) | 작업공간, CPU | 기본 (`solver: dls`), CI |
| `MinkIkSolver` | mink Apache-2.0, MuJoCo Apache-2.0, daqp MIT, qpsolvers **LGPL-3.0**(수정 없이 별도 격리 환경에서 동적 import만 하므로 결과물·우리 코드에 의무가 생기지 않는다), scipy BSD | 격리 환경 `scripts/retarget-env` (uv.lock 고정, `uv run --locked --isolated`) | 선택 (`solver: mink`), `make test-isolated` |
| pink(Pinocchio) | Apache-2.0 / BSD-2 | — | 선택하지 않음: mink로 충분하고 Pinocchio 빌드가 더 무겁다 |

- 내장 풀이기: 프레임마다 직전 해에서 시작하는 감쇠 최소제곱 반복 `(JᵀJ + (λ+μ)I)Δq = JᵀE`, 관절 위치
  한계는 반복마다 자르고, 속도 한계(URDF velocity x `velocity_scale` x 프레임 간격)는 직전 해 기준으로
  자른 뒤 `velocity_limited`로 표시한다. 회전 오차는 `orientation_mode`(none, approach: 접근 축만 — 5자유도
  팔, full: 접근·닫힘 축). 첫 목표·끊긴 뒤는 여러 시작점(관절 범위 가운데 + 결정적 난수)에서 고른다.
  μ(smoothness_weight)는 걸음 감쇠라 해 자체를 끌어당기지 않는다 (끌어당기면 0.5 mm 넘는 위치 오차가
  남아 이렇게 정했다).
- mink 풀이기: URDF에서 메시·충돌·전달 장치를 빼고(고정 링크 보존) MuJoCo로 읽어 FrameTask(접근 축 둘레
  회전 비용 0)와 ConfigurationLimit로 프레임마다 수렴할 때까지 미분 IK. 오차·도달·속도 한계 판정은 내장과
  같은 코드(`ik.frame_errors`)로 한다.

### 4. 도달 가능성과 정직성 규칙 (`dlp_retarget.retarget`)

- 프레임마다 `target_present`(손 목표 있음), `reachable`(위치 오차 ≤ `position_tolerance_m` 그리고
  회전 오차 ≤ `orientation_tolerance_rad`), 위치·회전 오차, `velocity_limited`.
- 닿지 못한 프레임도 **가장 가까운 해와 실제 오차**를 낸다. 목표를 작업 공간 안으로 잘라 "도달"로
  만들지 않는다. 목표가 없는 프레임은 직전 해를 두고 `target_present=0`으로 표시한다.
- 에피소드 보고 `FeasibilityReport`: 프레임 수, 목표 비율, 도달 비율, 위치·회전 오차 평균·최대, 팔 길이
  밖 프레임 수(첫 관절에서 링크 길이 합보다 먼 목표), 속도 한계 프레임 수, 가장 긴 미도달 구간(ms), 결정.
- 정책 `retarget.yaml unreachable`: `mark`(기본, 표시만 하고 낸다) 또는 `drop_episode`(도달 못 한 비율이
  `max_unreachable_fraction` 초과 또는 목표 비율이 `min_target_fraction` 미만이면 그 에피소드를 내지 않고
  보고에만 남긴다). 프레임을 빼지 않는다 (영상과 프레임이 어긋나므로).

### 5. 내보내기 (`dlp export lerobot <버전> --robot <ID>`, `dlp_export.robot`)

- 데이터셋 버전 스냅샷에서만, 기존 검증 정책 그대로 (기본은 사람이 만들거나 승인·수정·표본 검증한 라벨만).
  월드 좌표 손 3D(ADR 0046)도 이 규칙으로 고른다. 3D 궤적에는 아직 검수 화면이 없어 검수 전 3D는
  `--include-unreviewed`로만 들어간다 (F-COE-19, 결정 필요).
- 특징: `observation.state`·`action` = 로봇 관절(팔 + 그리퍼, rad, URDF 관절 이름, action = 다음 프레임),
  `observation.human_state` = 사람 모드 state, `annotation.*` 그대로, `retarget.target_present`·`reachable`·
  `position_error_m`·`orientation_error_rad`·`velocity_limited`·`verification`(그 프레임 월드 손 라벨의 최저
  검증 등급). LeRobot `robot_type` = 로봇 ID. 영상은 그대로 (사람 손이 보인다, F-COE-18).
- `meta/dlp_retarget.json`과 manifest `details.retarget`: 로봇 ID, 모델 sha256·라이선스, 관절 이름, 배치,
  리타기팅 판, 에피소드별 보고. manifest 최상위 `robot`에 요약 (로봇 ID, 모델 sha256, 판).
- 리타기팅 판: `retarget-<풀이기>-<판>+r<로봇 설정 해시>+m<모델 sha256 앞 12자>+p<정책 해시>`.
- 쓰기·읽기 스크립트는 특징 → npz 배열 대응(`frame_arrays`)으로 일반화했다 (사람 모드 결과는 같다).
- 작업자·장소 ID는 어디에도 넣지 않는다 (세션 ID는 기존처럼 내보내기마다 다른 가명).

## 정확도 (합성 정답, SO-101, 3초 30 fps, `packages/retarget/tests`, `packages/export/tests/test_robot.py`)

정답 끝단 궤적은 관절 범위 안 부드러운 관절 궤적의 정기구학이고, 손 점은 그 역규약으로 만든다
(`dlp_fixtures.retarget`).

| 조건 | 풀이기 | 도달 비율 | 끝단 위치 오차 (정답 대비) | 접근 축 오차 | 비고 |
| --- | --- | --- | --- | --- | --- |
| 도달 가능 궤적 (world_fixed, 배율 0.5) | 내장 | 100% | 최대 0.07 mm | 최대 0.5도 | 시험 상한 2 mm·1도, 약 50 ms/프레임 |
| 같음 | mink | 100% | 0.01 mm 미만 | 0.01도 미만 | 시험 상한 5 mm, 약 25 ms/프레임 (격리 환경) |
| 1.0~1.5초 손을 위로 2 m (로봇 기준 1 m) | 내장 | 74% | 도달 구간 0.05 mm | — | 그 15프레임 모두 미도달·팔 길이 밖, 보고 위치 오차 최대 1.07 m, 돌아올 때 속도 한계 23프레임 |
| 같음 | mink | 59% | 도달 구간 0.01 mm 미만 | — | 같은 15프레임 미도달, 속도 한계 35프레임 |
| 에피소드 중심 배치 | 내장 | 100% | 최대 2 mm 안 (시험) | — | |
| 내보내기 경로 (고정 fps가 손 표본 사이를 보간) | 내장 | 100% | 5 mm 안 (시험) | — | 승인 라벨, 검증 등급 2 |

실제 사람 동작·실제 고객 로봇에서의 품질(작업 성공, 충돌, 동역학)은 측정하지 않았다 (F-COE-20).

## 결과

- 장점: 고객 URDF와 sha256만 받으면 기존 데이터셋 버전에서 로봇 관절 에피소드를 다시 만들 수 있다.
  닿지 못한 구간이 프레임 단위로 드러나고, 어떤 로봇·모델·판·정책으로 만들었는지 manifest에 남는다.
- 비용·한계: 팔 하나 + 평행 그리퍼만 (다지 손은 dex-retargeting(MIT) 후보, F-COE-17). 자기 충돌·환경 충돌·
  토크·동역학을 보지 않는다 (F-COE-20). 영상 속 손과 로봇 상태의 도메인 차이(손 인페인팅·로봇 렌더 합성은
  연구 과제, F-COE-18). 5자유도 팔은 접근 축 둘레 회전을 맞추지 않는다. 배율·배치는 운영 결정이다.
  SO-101 그리퍼 열림 방향(closed 0, open 1.5 rad)은 공개 모델 문서에 정의가 없어 근사다.
- 운영 절차: `docs/handover/robot-retargeting.md`.
