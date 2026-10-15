# ADR 0044: 단일 카메라 영상에서 손가락 끝 3D를 손 모델 3D 점으로 맞춘다

- 상태: 채택
- 날짜: 2026-10-15
- 관련: ADR 0002(계약·라벨 저장), ADR 0006(검수 도구 무손실 왕복), ADR 0008·0009(프리라벨 어댑터·모델
  레지스트리), ADR 0010(모델 상업 사용), ADR 0015·0026(검수된 라벨 보존, 3D 중복 방지), ADR 0019(공간 라벨
  시각), ADR 0028(초안 온톨로지 덧붙이기).
  계약 변경: `dlp_schema.labels.KeypointFrame.world`(선택)·`WorldPoint`·`without_world`.
  `schemas/label_record.schema.json`, `episode_graph`·`export_intervals`·`ai_review_unit` 스키마 재생성.
  DB 변경 없음 (페이로드는 JSON 열). 정책: `prelabel.yaml depth.hand_points`·`depth.hand_fit`,
  `export.yaml lerobot.hand_points_3d`.

## 배경

요청: 보통 단일 카메라 영상에서 손가락 끝의 3D 위치를 기록한다 (1단계: 손 모델 3D 점 저장, 2단계:
카메라 좌표 손가락 끝 3D. 3단계 월드 좌표는 후속 F-PAR-26).

지금까지 3D는 손목·엄지 끝·검지 끝 세 점을 메트릭 깊이(Depth Anything V2 Metric Indoor Small)의 그 픽셀
값으로 역투영했다 (`lift3d`). 손가락 끝은 몇 픽셀 폭이라 깊이 맵에서 배경이나 잡은 물체의 깊이가 섞이고
(깊이 모델은 가장자리를 흐린다), 다섯 손가락 끝 사이의 상대 위치는 깊이 잡음에 그대로 묻힌다.

MediaPipe Hand Landmarker(이미 쓰는 Apache-2.0 모델)는 2D 21점과 함께 `hand_world_landmarks`(21점,
미터, 원점은 손의 대략적인 기하 중심, 카메라와 같은 축)를 낸다. 손 모양은 주지만 카메라에서의 위치는
주지 않는다.

## 결정

### 1. 손 모델 3D 점은 2D 키프레임의 선택 필드로 둔다

- `KeypointFrame.world: tuple[WorldPoint, ...] | None` (기본 None). points와 같은 순서·개수, hand21
  골격에만 허용한다 (`KeypointTrackPayload` 검증). None이면 JSON에 싣지 않는다 (Pydantic
  `exclude_if`): 예전 라벨의 JSON·라벨 ID 해시(검수 수집의 내용 해시 등)가 그대로다.
- 별도 페이로드(예: `hand_world_track`)는 두지 않는다. 같은 모델·같은 시각의 출력이 둘로 갈라지면
  어느 2D와 짝인지 다시 맞춰야 하고, 검수·내보내기·AI 검수·온톨로지 검증에 새 종류를 모두 더해야 한다.
- world는 손 모양 보조 정보이고 검수 대상이 아니다. 검수 도구(CVAT)는 2D만 보이므로:
  - 도구로 보낸 형태(`dlp_review.cvat.quantize`)에서 world를 뺀다 (신뢰도처럼). 고치지 않은 트랙은
    원래 레코드를 승인하므로 world가 남는다 (2D를 승인한 것이고 world는 그 2D와 짝인 모델 출력이다).
  - 고치거나 새로 그린 사람 레코드에는 world가 없다. `reconcile`도 사람 레코드를 만들 때
    `without_world`를 적용한다 (다른 경로 방어). **고친 2D와 모델의 예전 world가 함께 남지 않는다.**
    그 트랙의 3D는 관절별 깊이 방식으로 돌아간다 (아래 4절).
  - Label Studio는 키포인트 트랙을 다루지 않는다.
- MediaPipe 어댑터 버전에 `+w1`을 붙여 다시 돌리면 검수 전 손 트랙이 world를 가진 새 버전으로
  바뀐다. 검수된 예전 손 트랙은 지우지 않으므로 world가 없고, 그 트랙을 고르면 관절별 깊이로 올린다.

### 2. 손목과 다섯 손가락 끝을 올린다

`prelabel.yaml depth.hand_points: [0, 4, 8, 12, 16, 20]` (wrist, thumb_tip, index_tip, middle_tip,
ring_tip, pinky_tip). 온톨로지 `hand_joints`에 다섯 손가락 끝이 이미 있다 (변경 없음). 정책 절 해시가
바뀌므로 다시 돌리면 검수 전 3D만 교체되고, 검수된 3D와 같은 (개체, 부위, 좌표계)의 새 궤적은 만들지
않는다 (기존 규칙, ADR 0026).

### 3. 추정기: 손 모양 + 2D 투영 + 크기 융합 (`dlp_prelabel.handfit`)

1. **이동량 맞춤**: 손 모양 W_i를 그대로 두고 카메라 좌표 Q_i = W_i + T가 2D 픽셀로 투영되도록 T를
   선형 최소제곱으로 푼다 (핀홀 식이 T에 선형). 관절 깊이로 한 번 다시 가중해 잔차를 정규화 재투영
   오차로 맞춘다. 내부 파라미터는 세션 캘리브레이션(`Session.calibration.intrinsics`, 해상도 비례 조정),
   없으면 `depth.default_hfov_deg` 근사다 (기존 `intrinsics_for`).
2. **크기(거리)**: 단안 투영은 Q 전체를 카메라 원점 기준으로 k배 해도 같은 2D다. k를 두 근거로 정한다.
   - 깊이 k_depth: 손바닥 관절(`hand_fit.anchor_points`: 손목·MCP)의 깊이 맵 값 / Q_jz의 중앙값.
     얇은 손가락 끝은 쓰지 않는다.
   - 손 크기 사전값 k_size = `reference_palm_m`(0.09 m, 손목 → 중지 MCP) / |W_0 - W_9|. null이면 1
     (MediaPipe 크기를 믿는다).
   - k = exp(w log k_depth + (1 - w) log k_size), w = `depth_weight`(0.5). 깊이가 없거나 두 배율 비가
     `max_scale_ratio`(1.5)를 넘으면(배경·가림 깊이) k = k_size.
3. P_i = k Q_i. 2D 투영은 맞춘 그대로, 손 안의 상대 위치는 손 모양을 k배 한 것이다.

보이는 관절이 `min_points`(8)보다 적거나, 풀 수 없거나, 관절이 카메라 뒤이거나, 재투영 RMS가
`max_reprojection_px`(8 px)를 넘으면 그 시각은 관절별 깊이 방식으로 올린다.

선택하지 않은 대안: 관절마다 깊이를 읽고 world로 평활만 하는 방식(손가락 끝 깊이 오염이 남는다), 크기도
최소제곱 미지수로 두는 방식(단안 투영에서 크기는 관측되지 않아 퇴화한다), MANO 손 메시 모델(5절).

### 4. 바뀌지 않는 것

- 결과는 기존과 같은 trajectory3d (frame=camera, source_3d=mono_depth, part=관절 이름, 표본 시각은 바디캠
  PTS ms, `depth.confidence` 0.5). 한 궤적 안에 맞춘 시각과 관절별 깊이 시각이 섞일 수 있다. 어느 방식인지
  표본마다 기록하지 않는다 (열린 결정).
- 모델 버전 `depth3d-<가중치>+hf1+p<depth 절 해시>+i<입력 해시>`. 정책 값이나 추정기 판(hf1)이 바뀌면
  다시 돌고, 검수 전 이전 3D만 `retractions()`로 지운다. 검수·표본 검증한 3D는 지우지 않는다.
- 카메라 좌표뿐이다. 월드 좌표(세션 고정)는 카메라 6자유도(SLAM)가 생긴 뒤에 만든다 (F-PAR-26,
  `stubs.CAMERA_POSE`의 `TODO(real-model)` 그대로).

### 5. 상업 사용 (ADR 0010)

HaMeR·WiLoR 등 손 메시 회귀 모델은 MANO 손 모델(비상업 연구 라이선스)과 그 파라미터 학습에 묶여 있어
판매용 라벨에 쓰지 않는다 (F-PAR-27). MediaPipe Hand Landmarker(Apache-2.0, `commercial: allowed`)의
world landmarks만 쓰므로 `config/models.yaml`에 새 모델은 없다.

### 6. 내보내기

- LeRobot: `export.yaml lerobot.hand_points_3d`를 손목 + 다섯 손가락 끝으로 늘렸다 (observation.state 열
  = 2 x 63 + 2 x 4 x 6 + 2 x 4). 카메라 좌표(m)다.
- COCO는 2D 키포인트만, 구간 JSON은 시간 구간 라벨만 싣는다 (3D 궤적·world 없음). world(손 중심 3D 점)는
  어떤 형식으로도 내보내지 않는다 (내부 중간값).

## 정확도 (합성 정답, `packages/prelabel/tests/test_handfit.py`)

정답 3D를 아는 손 시퀀스(`dlp_fixtures.hand3d`: 카메라 앞 0.4~0.55 m, 320x240, fx 160, 손바닥만 손
깊이이고 손가락 끝은 배경 1.2 m인 합성 깊이):

| 조건 | 손가락 끝 평균 오차 (맞춤) | 관절별 깊이 |
| --- | --- | --- |
| 흔들림 없음 | < 1 mm | 약 700 mm (배경 깊이) |
| 2D 1 px + world 3 mm | 약 11 mm (시험 상한 20 mm) | 약 700 mm |
| 손 모델 크기 +10% | < 1 mm (손 크기 사전값이 바로잡음) | — |
| 깊이 +10% 편향 | 약 25 mm (상한 35 mm), 손목 기준 상대 위치 < 10 mm | 손목 48 mm |

실영상 정확도는 측정하지 않았다. 단안이라 절대 거리 오차(손 크기 개인차 약 ±10%, 깊이 모델 오차, 렌즈
왜곡)가 남는다. MediaPipe world landmarks 자체의 손 모양 오차도 실제 영상에서는 수 mm~1 cm 이상일 수 있다.

## 결과

- 장점: 손가락 끝 3D가 배경 깊이에 오염되지 않고, 다섯 손가락 끝의 상대 위치가 손 모양으로 묶인다. 새
  모델·DB 변경 없이 기존 버전·검수 보존 규칙을 그대로 쓴다.
- 비용: 손 트랙 JSON이 키프레임마다 63개 수만큼 커진다 (0.1 mm 반올림). 사람이 고친 손 트랙의 3D는 관절별
  깊이로 돌아간다.
- 열린 결정: (1) 고친 트랙에서 고치지 않은 프레임의 world를 되살릴지, (2) 표본별 추정 방식 기록,
  (3) 실제 세션 캘리브레이션 수집 절차(지금은 없으면 화각 근사), (4) 손 크기 사전값을 작업자별로 둘지.
