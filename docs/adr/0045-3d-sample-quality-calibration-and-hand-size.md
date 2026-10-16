# ADR 0045: 3D 표본의 방식·신뢰도, 카메라 내부 파라미터 등록, 작업자 손 크기 사전값

- 상태: 채택
- 날짜: 2026-10-16
- 관련: ADR 0044(손가락 끝 3D, 이 ADR은 그 열린 결정 2~4를 닫는다), ADR 0002(계약), ADR 0015·0026
  (검수된 라벨 보존, 3D 중복 방지), ADR 0019(공간 라벨 시각), ADR 0020(원본 접근 감사), ADR 0027
  (내보내기 가명).
  계약 변경: `Trajectory3DSample.method`·`confidence`(선택), `Trajectory3DPayload.intrinsics`(선택),
  `Lift3DMethod`, `IntrinsicsSource`, `Calibration.camera_id`(선택), 새 모듈 `dlp_schema.calibration`
  (`CameraCalibration`, `WorkerHandSize`, 결정 규칙). `make schemas`로 label_record·session·
  episode_graph·export_intervals·ai_review_unit 스키마 재생성.
  DB: 추가 전용 테이블 `camera_calibrations`, `worker_hand_sizes` (Alembic 0022(down 0021)).
  정책: `prelabel.yaml depth.sample_confidence`(새 절, `depth.confidence` 삭제), `palm_estimate`(새 절),
  `config/policies/calibration.yaml`(새 파일), `ops.yaml audit.service_purposes`에 `calib.palm_estimate`.

## 배경

ADR 0044의 손가락 끝 3D는 세 가지가 비어 있었다.

1. 한 궤적 안에 손 맞춤 표본과 관절별 깊이 표본이 섞이는데 어느 쪽인지 기록하지 않았고, 신뢰도는
   `depth.confidence` 0.5 고정이었다. 검수·학습·내보내기에서 좋은 표본과 나쁜 표본을 가를 수 없다.
2. 내부 파라미터는 세션 매니페스트에 직접 적을 수만 있었고, 실제로는 거의 비어 있어 가로 화각 90도
   근사를 썼다. 기기(바디캠)마다 한 번 재 두고 세션이 가져다 쓰는 길이 없었다.
3. 손 크기 사전값(손목 → 중지 MCP 0.09 m)이 모든 작업자에게 같아, 손이 작거나 큰 작업자는 그 비율만큼
   (약 ±10%) 멀거나 가깝게 놓였다.

## 결정

### 1. 표본마다 방식과 신뢰도 (계약, 선택 필드)

- `Trajectory3DSample.method: Lift3DMethod | None` — `hand_fit`(손 중심 3D 점 맞춤) 또는
  `point_depth`(그 픽셀 깊이 역투영: 관절별 깊이 대체, 박스 중심).
- `Trajectory3DSample.confidence: Confidence | None` — 그 표본의 신뢰도.
- `Trajectory3DPayload.intrinsics: IntrinsicsSource | None` — `calibrated`(세션 값 또는 등록된 기기
  값) 또는 `hfov_approx`. 내부 파라미터는 한 번의 3D 실행(궤적 하나) 안에서 늘 같으므로 표본마다
  두지 않고 페이로드에 한 번 둔다.
- 셋 다 None이면 JSON에 싣지 않는다 (`exclude_if`). 예전 trajectory3d의 JSON이 글자 그대로 같아 라벨
  내용 해시(검수 수집의 레코드 ID 해시 등)가 바뀌지 않는다 (`test_trajectory_sample_method_and_confidence_are_optional`).
  사람이 만든 3D·예전 3D는 None이다.
- 신뢰도 식 (`dlp_prelabel.handfit.fit_confidence`, 값은 `prelabel.yaml depth.sample_confidence`):
  - 맞춤: `hand_fit`(0.8) x 재투영 항(RMS 0 → 1, `hand_fit.max_reprojection_px` → `reprojection_floor`)
    x 관절 수 항(21 → 1, `hand_fit.min_points` → `points_floor`) x 일치 항(깊이·사전값 배율 비 1 → 1,
    `hand_fit.max_scale_ratio` → `agreement_floor`, 깊이를 못 쓰거나 버렸으면 `no_depth`).
  - 관절별 깊이·박스 중심: `point_depth`(0.4).
  - 캘리브레이션이 없으면 둘 다 `hfov_approx`(0.6)를 곱한다.
  - 라벨 신뢰도(`LabelRecord.confidence`)는 표본 신뢰도의 평균이다 (예전 고정 0.5 대신).
  값은 합성 정답에서의 상대 순서(오차가 큰 조건일수록 낮게)만 검사했다. 실영상 오차와의 보정
  (calibration curve)은 하지 않았다 (열린 결정).
- 3D 단계 버전 판을 `+hf2`로 올렸다. 정책 절 해시도 바뀌므로 다시 돌리면 검수 전 3D만 교체된다.
- LeRobot: 새 특징 `annotation.hand_points_3d_confidence` (float32, 2 x len(hand_points_3d), 손(왼,
  오른) x 점 순서, 값이나 신뢰도가 없으면 -1). 두 표본 사이는 선형 보간이다. `observation.state`의
  `present` 열은 그대로 0/1이다 (뜻을 바꾸지 않는다). 방식(method)은 내보내지 않는다 (신뢰도가 이미
  방식을 반영한다). COCO·구간 JSON은 3D를 싣지 않으므로 바뀌지 않는다.

### 2. 카메라 내부 파라미터 등록 (`dlp calib intrinsics|import`)

- 있던 것: `Session.calibration.intrinsics`(수집 매니페스트에 직접 적음, 세션 JSON에 고정). 그대로
  두고 가장 앞선다.
- 더한 것: `Session.calibration.camera_id`(선택, 매니페스트에 기기 ID만 적는다)와 DB
  `camera_calibrations`(추가 전용). 3D 단계는 세션 값 → 그 기기의 등록 중 유효 시작(`valid_from`)이
  없거나 녹화 시각 이하인 것 중 가장 늦은 것(같으면 늦게 등록한 것) → 화각 근사 순으로 고른다
  (`dlp_schema.calibration.resolve_intrinsics`). 렌즈를 바꾸거나 다시 재면 `--valid-from`으로 새로
  등록한다 (예전 세션은 예전 값을 계속 쓴다).
- 3D 단계 입력 해시에 고른 값과 출처가 들어간다. 등록하면 `dlp prelabel run`이 검수 전 3D만 새
  값으로 다시 만든다 (검수·표본 검증한 3D는 지우지 않는다, ADR 0015·0026).
- 체커보드 캘리브레이션 (`dlp_media.calibration.calibrate_video`, OpenCV 4.x, 이미 작업공간 의존성):
  PTS 간격으로 프레임을 검사해 `findChessboardCornersSB`로 코너를 찾고, 거의 같은 장면은 건너뛴 뒤
  `calibrateCamera`(k3 고정 기본)로 푼다. 장면 수·재투영 RMS 기준(`calibration.yaml`)을 못 맞추면
  등록하지 않는다 (종료 코드 2). 입력은 그 기기로 보드만 찍은 로컬 영상이고 세션 영상(원본 버킷)이
  아니므로 원본 접근 감사 대상이 아니다. 운영 절차는 `docs/handover/camera-calibration.md`.
- 파일 가져오기 (`load_intrinsics_file`): 이 저장소 형식(YAML·JSON, `CameraIntrinsics` 필드), OpenCV
  FileStorage(YAML·XML), ROS camera_info YAML.
- 멱등: 같은 기기·같은 유효 시작의 최신 등록과 값이 같으면 쓰지 않는다.
- 왜곡 계수는 기록만 한다. 3D 단계는 아직 핀홀(fx·fy·cx·cy)만 쓴다 (F-PAR-28).
- 선택하지 않은 대안: 세션 JSON에 캘리브레이션을 써 넣는 방식 — 세션 행이 바뀌어 수집 재실행의 충돌
  검사(`SessionConflictError`)와 부딪히고, 같은 기기의 여러 세션을 한 번에 고칠 수 없다. ChArUco
  보드 — 부분 가림에 강하지만 보드 사전·인쇄 사양을 하나 더 관리해야 해 1단계는 체커보드만 둔다.

### 3. 작업자 손 크기 사전값 (`dlp calib palm|palm-estimate`)

- DB `worker_hand_sizes`(추가 전용): 작업자 ID, 손바닥 길이(m, 0.04~0.16), 출처 `measured`·`estimated`,
  표본 수. 그 작업자의 가장 늦은 등록이 쓰인다. 결정 순서: 작업자 값 → 정책
  `hand_fit.reference_palm_m` (`resolve_palm`).
- 3D 단계 버전: `<lifter 버전>+i<입력 해시>+hp<w|p>`. 입력 해시에 사전값 출처와 값이 들어가 값이
  바뀌면 다시 돈다. 버전 문자열에는 출처 글자만 두고 작업자 ID·값은 넣지 않는다 (모델 버전은 COCO·구간
  JSON 내보내기에 실린다). 작업자 ID는 이 테이블에만 있고 어떤 내보내기에도 넣지 않는다.
- 추정 (`dlp_prelabel.palm`, `dlp calib palm-estimate <작업자> [--apply]`): 그 작업자의 세션 중 내부
  파라미터가 캘리브레이션된 것만, 손 맞춤 품질이 좋은 시각(`palm_estimate`: 재투영 RMS 3 px, 보이는
  관절 18개 이상)의 깊이 배율만으로 잰 손바닥 길이(사전값과 무관)를 모아 중앙값을 낸다. 표본이
  `min_samples`보다 적으면 등록하지 않는다. 기본은 보고만, `--apply`일 때 `estimated`로 등록한다.
  원본 바디캠을 읽으므로 감사 저장소(`calib.palm_estimate`)로만 연다. 깊이 모델의 배율 오차는 그대로
  들어간다 (깊이 10% 편향이면 10%). 잰 값이 있으면 그쪽이 낫다.

## 정확도 (합성 정답)

| 조건 | 결과 | 시험 |
| --- | --- | --- |
| 체커보드 640x480, 12자세, 정답 fx 520·fy 515 | fx·fy 0.3% 안팎, 주점 2 px 안팎, RMS 0.09 px (상한 1%, 4 px) | `packages/media/tests/test_calibration.py` |
| 손 20% 작음(손바닥 0.072 m), 정책 0.09 | 40 mm 넘게 틀림 → 작업자 0.072면 1 mm 안 | `test_handfit_quality.py` |
| 손바닥 추정, 정답 깊이 | 0.072 m와 0.5 mm 안 | `test_handfit_quality.py` |
| 초점 200 px 기기, 화각 근사 160 | 등록 뒤 다시 돌려 1 mm 안 (DB·원본 저장소 경로) | `test_runner.py::test_registered_calibration_and_worker_palm_relift_only_unreviewed_3d` |

## 결과

- 장점: 표본마다 왜 믿을 만한지(또는 아닌지)가 남고, 기기를 한 번 재면 그 기기의 모든 세션이 쓴다.
  손 크기 개인차를 작업자 단위로 바로잡는다. 모두 기존 버전·검수 보존 규칙 위에서 돈다.
- 비용: 표본 JSON이 두 키만큼 커진다. DB 테이블 2개. 정책 해시가 바뀌어 다음 실행에서 검수 전 3D가
  모두 다시 만들어진다.
- 열린 결정: 신뢰도 값의 실영상 보정, 왜곡 계수 적용(F-PAR-28), ChArUco·어안 렌즈, 카메라-IMU 외부
  파라미터 캘리브레이션(SLAM, F-PAR-26).
