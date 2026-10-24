# ADR 0058: 착용자 팔 골격(손목 → 아래팔 → 팔꿈치)과 착용자 자신의 몸 억제

- 상태: 채택
- 날짜: 2026-10-24
- 관련: ADR 0010(상업 사용), ADR 0015·0019(재실행·시각), ADR 0028·0043(온톨로지 초안 덧붙이기·한영 이름),
  ADR 0044·0045(손 3D), ADR 0048·0050·0051(3D가 2D 검수를 물려받음), ADR 0054(손 3D 물리 제약),
  ADR 0055(오탐 의심 힌트 `wearer_body`), 실영상 파일럿 [pilot-real-2026-10-22.md](../handover/pilot-real-2026-10-22.md)
  11절, 문제 레지스트리 F-PAR-66·67.
- 계약 변경: 있음 (DB 스키마·Alembic 변경 없음, 라벨 JSON 안의 값만 늘어난다).
  - `KeypointTrackPayload.skeleton`에 `arm3`(손목·팔꿈치·어깨, `ARM3_JOINTS`), arm3는 hand 필수
    (`HANDED_SKELETONS`).
  - `Constraint3DFlag.forearm_length`.
  - 온톨로지 `arm_joints`(elbow·shoulder, ko·en, v1 초안 덧붙이기 → `make db-upgrade`로 재등록).
  - 정책: `prelabel.yaml wearer_suppression`·`arm`·`depth.arm_points`·`depth.constraints.forearm_*`,
    `export.yaml lerobot.arm_points_3d`, `evaluation.yaml proxy.forearm_m·wearer_margin·wearer_share`,
    `review_ui.yaml` 색. 내보내기: COCO 범주 `wearer_arm`, LeRobot 특징 `annotation.wearer_arm`.
  - `make schemas` (label_record·ontology 등).

## 배경

1인칭(가슴 장착) 돌봄·가사 영상에서 손 21관절(hand21)만으로는 팔의 자세(아래팔 방향, 팔꿈치 위치)를
알 수 없다. 사내 모델 개발에 손 → 손목 → 아래팔 → 팔꿈치로 이어지는 골격이 필요하다. 동시에 파일럿
(pilot-real2)에서 COCO 사람 박스 72개 중 상당수와 YOLOX → RTMPose 자세 176개 트랙의 대부분이 착용자
자신의 팔·몸통이었다 (MediaPipe 손목과 RTMPose 손목 일치 0.976, ADR 0055의 `wearer_body` 힌트는
표시만 한다). 착용자 몸은 객체도, 다른 사람도 아니다.

## 결정

### 1. 표현: 새 골격 arm3 (손마다 하나)

- `KeypointTrackPayload(skeleton="arm3", hand=<손>, entity_id="<손>_arm")`, 점 순서 손목·팔꿈치·어깨.
  COCO17의 일부를 재사용하지 않는다: coco17 트랙은 사람 한 명(좌우 두 팔·머리)이라 hand 필드가 없고,
  14점이 늘 비어 검수자에게 의미 없는 점이 보이며, 1인칭 RTMPose 좌우는 자주 바뀐다(파일럿 0.74).
- 손과의 연결: 같은 스트림·같은 손(hand). arm3 손목 점은 같은 시각 hand21 손목(0번)을 그대로 쓴다
  (손목 기준은 손 모델). 키프레임 시각도 손 키프레임 시각이다. 별도 참조 필드는 두지 않는다 (라벨은
  덮어쓰지 않아 손 트랙 ID가 검수로 바뀌므로, 손 판정으로 잇는 쪽이 검수 뒤에도 성립한다).
- 보이지 않는 점은 visibility 0, 좌표는 손목 자리(의미 없음). 어깨는 가슴 장착 화면에 거의 없어
  프리라벨은 늘 0이고(`arm.emit_shoulder: false`), 보이면 검수자가 찍는다.
- 3D: 팔꿈치(·어깨)는 trajectory3d `entity_id="<손>_arm"`, part = 온톨로지 `arm_joints`. 손목 3D는
  손 궤적(`<손>_hand`, wrist)이 맡는다 (중복 없음).

### 2. 추정: 기존 RTMPose(Body7, COCO17)를 착용자 박스에 다시 쓴다 (새 가중치 없음)

전신 단계(`RtmPose`, 바디캠이고 같은 스트림 손 트랙이 있을 때)가 프레임마다:

1. YOLOX 사람 박스 → RTMPose. 손목이 착용자 손목(손 크기 x `wrist_match_palms` 안)과 맞고 아래
   가장자리 띠(`bottom_band`)에 닿는 자세는 착용자 자세다 (`dlp_schema.wearer_body.wearer_pose_frame`).
   1인칭 RTMPose는 착용자 팔에 머리를 지어내 "보임"으로 내므로(파일럿 7,083개 중 5,203개) 자세에는
   머리 조건을 쓰지 않는다.
2. 팔 시각(`arm.frame_stride_ms`, 100 ms)에는 합성 착용자 박스(보이는 손 박스를 옆 3·위 2 손 크기만큼
   넓히고 아래는 화면 끝)로 RTMPose를 한 번 더 돌린다. 착용자 몸은 화면에서 잘려 있어 YOLOX가 놓치는
   시각에도 팔 사슬 후보가 생긴다.
3. 후보 자세의 두 팔 사슬(COCO 9-7-5, 10-8-6) 중 손목이 hand21 손목에 가까운 것부터 손마다 하나를
   고른다 (`dlp_prelabel.arm`). 좌우는 hand21 손 판정을 따르고 사슬의 COCO 좌우는 보지 않는다.
4. 팔꿈치는 점수(`elbow_min_score`)·화면 안(가장자리 `edge_margin_px` 여유)·화면 아래팔 길이/손 크기
   (`forearm_palms`)·손가락 반대 방향(`max_hand_cos`)을 모두 통과해야 보임이다. 하나라도 어기면 보이지
   않음이다 (화면 밖 팔꿈치를 지어내지 않는다). 두 팔꿈치가 모두 보이는데 좌우가 엇갈리면 둘 다 보이지
   않음(`side_conflict`).
5. 사슬 손목과 hand21 손목의 일치 정도(1 - 거리/허용 거리)를 arm3 손목 점 신뢰도에 싣는다 (사슬이
   없으면 신뢰도 없음). 팔 모델과 손 모델의 손목 불일치 대리 지표다.

다른 후보와 라이선스 (2026-10-24 공개 문서 기준, `config/models.yaml`에 새 항목 없음):
- 1인칭 몸 자세 전용 공개 모델은 학습 데이터(Ego-Exo4D·EgoBody 등)가 비상업 조건이거나 가중치가
  비상업(Sapiens: CC BY-NC 4.0)이라 쓰지 않는다 (ADR 0010, `forbidden`).
- 전신 133점(RTMW·DWPose)은 학습 데이터에 연구용 데이터가 섞여 지금 RTMPose와 같은 `review` 등급이고,
  팔꿈치·어깨는 COCO17과 같은 점이라 이점이 작아 더하지 않았다.
- 손목에서 아래팔 방향으로 팔꿈치를 찾는 순수 기하 방법은 영상 증거 없이 길이를 가정하게 되어(지어냄)
  쓰지 않았다. 손 방향은 판정(손가락 반대 방향)에만 쓴다.

### 3. 3D와 아래팔 제약

깊이 단계(`lift3d`, 판 `+ar1`)가 arm3 트랙의 보이는 팔꿈치를 그 픽셀 깊이(주변 패치 중앙값)로 올려
`<손>_arm`/elbow 궤적을 만든다 (`derived_from` = arm3 트랙, 방식 point_depth). 같은 시각 손목 3D와의
거리가 `forearm_m` 밖이거나 세션 중앙값에서 `forearm_tolerance` 넘게 다르면 `forearm_length` 표시
(기본 downweight, ADR 0054와 같은 틀). 팔 3D도 손 3D처럼 arm3 2D 검수를 물려받고 출처가 바뀌면 궤적
따르기가 지운다 (`trajectory_follow.LIFTED_SKELETONS`, ADR 0048·0050·0051).

### 4. 착용자 자신의 몸 억제 (라벨로 만들지 않음)

- 객체(COCO 사람 박스)·도구(OWLv2)·전신(YOLOX → RTMPose) 단계가 바디캠 스트림에서 착용자 몸을
  라벨로 만들지 않는다 (`dlp_prelabel.wearer_suppress`, 정책 `wearer_suppression`, 단계 버전에 `+w<해시>`).
- 판정 규칙은 ADR 0055 힌트와 같은 공용 모듈 `dlp_schema.wearer_body`로 옮겼다: 박스(넓힘 `margin`)
  안에 착용자 손목이 있고, 다른 사람의 보이는 머리 관절이 없고, 아래 변이 화면 아래 띠 안. 박스
  트랙은 그 키프레임 비율이 `track_share` 이상이면 트랙째 뺀다. 사람 아닌 클래스(도구)는 박스가
  화면의 `other_min_area` 이상일 때만 (손에 든 도구는 남긴다).
- 남기는 것: 머리가 보이거나 손목이 착용자 손목과 다른 사람(대상자). 3인칭 스트림·손 트랙이 없는
  스트림은 억제하지 않는다.
- 억제한 것은 라벨이 아니라 단계 요약(`PrelabelSummary.wearer_suppressed`, CLI `[착용자 억제]`)에만
  센다. 억제한 착용자 자세는 팔 단계 후보로만 쓴다.
- 프라이버시 탐지·블러는 이 결정과 무관하다 (블러를 약하게 하지 않는다).
- 증거는 러너가 같은 실행에서 먼저 낸 손·전신 트랙(없으면 지금 운영 라벨)으로 만든다
  (`runner.wearer_evidence`).

### 5. 검수·내보내기·대리 지표

- CVAT: 손별 라벨 `kp_arm3_left`·`kp_arm3_right` (점 세 개, 점별 가시성 속성). 기존 프로젝트에는 작업을
  만들 때 빠진 라벨을 덧붙인다. 수집·reconcile은 다른 키포인트 트랙과 같다. Label Studio는 이 플랫폼에서
  시간 구간 라벨만 다루므로 키포인트 경로가 없다 (손 골격과 같음).
- COCO: 키포인트 범주 `wearer_arm` (wrist·elbow·shoulder, 뼈대 1-2·2-3), 주석 `hand`로 같은 손 hand
  주석과 잇는다. LeRobot: `annotation.wearer_arm` = 손마다 arm3 2D(x, y, v) + `arm_points_3d`
  3D(x, y, z, present). observation.state는 바꾸지 않는다 (열 순서 호환).
- `dlp ops quality-proxy`: 손마다 팔꿈치 보임 비율, 사슬 손목 일치 비율·정도, 아래팔 3D 길이 분포·
  변동계수·범위 밖 비율, 사람 박스 중 착용자처럼 보이는 트랙 수.

### 6. 데이터 용도 (문서만)

2026-10-24 사용자 결정: 라벨링 데이터는 당분간 판매하지 않고 사내 모델 개발에 쓴다. 이 ADR은 라이선스·
개인정보·내보내기 규칙(ADR 0010·0020·0027 등)을 완화하지 않는다. 비상업 가중치는 상업 제품 개발에도
일반적으로 허용되지 않으므로 `forbidden` 규칙은 그대로이고, 개인정보 규칙도 그대로 적용한다.

## 결과

- 바디캠 객체·전신 라벨에서 착용자 몸 탐지가 사라지고, 팔 골격이 손 골격에 이어 붙는다.
- 정답 표본 없이 정한 문턱(`arm`·`wearer_suppression`)이라 실영상 정확도는 모른다 (F-PAR-66). 손목이
  착용자 손목과 맞는 다른 사람 자세(손을 맞잡음)를 착용자로 볼 수 있고, 머리가 화면 밖인 대상자 박스가
  착용자 손목을 담고 아래에 닿으면 억제될 수 있다 (`track_share`로 일부만 막는다).
- 팔·억제는 전신·객체·도구 단계 안에서 돌아 그 단계 버전에 손 트랙 입력 해시가 없다. 손 트랙을 검수로
  고쳐도 팔·억제를 다시 판정하지 않는다 (F-PAR-67, 정책·가중치가 바뀌면 다시 돈다).
- 전신 단계 CPU 시간이 팔 시각마다 RTMPose 한 번만큼 는다.
