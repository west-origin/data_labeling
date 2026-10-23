# ADR 0056: 손 크기로 세션 초점 거리 추정, 잘라 낸 박스의 AI 확인 힌트

- 상태: 채택
- 날짜: 2026-10-23
- 관련: ADR 0040(AI 1차 검수는 라벨·검수 상태가 아님, 블러본만), ADR 0044·0045(손 맞춤 3D·내부 파라미터
  출처), ADR 0052(작업자 손 크기), ADR 0055(오탐 의심 힌트), 실영상 파일럿 10.4절, F-PAR-59.
- 계약 변경:
  - `IntrinsicsSource.ESTIMATED_HAND_SIZE = "estimated_hand_size"` (`make schemas`). LeRobot 어휘
    `hand_point_3d_intrinsics`에 2번으로 실린다 (0 calibrated, 1 hfov_approx). DB 스키마 변경 없음.
  - 정책: `prelabel.yaml depth.focal_estimate`, `depth.sample_confidence.estimated_hand_size`.
    3D 단계 판 `+fe1`. `review.yaml ai_crops`.
  - 힌트 이유 `ai_crop_no` (블러·객체 힌트, ADR 0055의 저장 위치 그대로). `dlp review hints --ai`.

## 배경

첫 실영상은 0.5x 초광각(1920x1080에서 초점 약 720 px)인데 캘리브레이션이 없어 기본 화각 90도(가로 1920 px에
걸면 초점 960 px)로 근사했고, 손목 3D 거리 중앙값이 0.7 m를 넘었다. 같은 기기의 체커보드 영상이 가장 좋은
해결이지만(`dlp calib intrinsics`) 늘 있지 않다. 손 크기 역산(약 700 px)이 공칭값과 맞았으므로 이를 정직한
출처 표시와 함께 쓸 수 있다.

## 결정

1. **손 크기 초점 거리 추정 (`dlp_prelabel.fov`).** 세션·기기 캘리브레이션이 모두 없을 때만 3D 단계 앞에서
   `focal_estimate.frame_stride_ms` 간격으로 손 맞춤을 풀어, 기본 화각 근사 초점 f0로 잰 깊이 손바닥 길이
   p0와 손 크기 사전값 L(작업자 등록값, 없으면 `hand_fit.reference_palm_m`)의 비 중앙값으로 f = f0·p0/L을
   정한다. 표본 수(`min_samples`)·퍼짐(`max_spread`)·화각 범위(`hfov_range_deg`)를 못 지키면 버리고 화각
   근사로 남는다. 주점은 화면 가운데, fx = fy, 왜곡 없음.
2. **정직한 출처.** 추정으로 올린 3D 궤적의 `intrinsics`는 `estimated_hand_size`이고 표본 신뢰도에
   `estimated_hand_size`(0.75)를 곱한다. 캘리브레이션이 아니므로 `calibrated`로 쓰지 않고, 세션 값·기기
   등록값을 덮지 않으며, 기기 캘리브레이션 표에 등록하지 않는다. 카메라 자세 단계는 이 추정을 쓰지 않는다.
   손 크기 사전값·깊이 모델 오차가 그대로 남는다 (성인 평균 사전값이면 약 ±10%).
3. **잘라 낸 박스의 AI 확인 (`dlp_aireview.crops`).** 지금 승인된 블러본(`open_blurred`·`frames_for`)에서
   블러·도구 박스를 넉넉한 여유로 잘라 사내 VLM(aireview.yaml backend·server, `check_hosts`로 사내 주소만)에
   "그 대상이 맞나"를 JSON Schema로 묻는다. 블러 박스 가운데는 모자이크이므로 주변 맥락으로 그럴듯한지만
   묻는다. no 답(신뢰도 `min_confidence` 이상)을 오탐 의심 힌트 `ai_crop_no`로만 싣는다. 판정은 라벨도
   검수 상태도 아니고 `label_records`·`record_review`·`ai_review_*` 표에 쓰지 않는다 (힌트 파일뿐). CPU·CI는
   `OracleCropVerifier`(늘 unsure, 힌트 없음)와 `FakeOpenAIServer`로 시험한다.

## 결과

- 캘리브레이션 없는 세션의 3D 거리 배율이 손 크기 수준으로 맞춰진다. 그래도 측정이 아니므로 판매 데이터는
  출처로 구분된다 (`estimated_hand_size`). 체커보드 캘리브레이션이 생기면 3D 단계가 다시 돌며 바뀐다.
- 3D 단계 앞에 깊이 추론이 1초 간격으로 더 든다 (200 ms 간격 3D의 약 1/5).
- AI 조각 확인은 승인된 블러본이 있어야 하므로 블러 승인 뒤 라벨링 검수 전에 돌린다. 블러 승인 전 블러 검수
  화면의 힌트에는 규칙·TTA 힌트만 있다.

## 개정 (2026-10-23, 실영상 결과): 초점 거리 추정은 기본 끔

pilot-real2(iPhone 13 0.5x 초광각, 1920x1080, 공칭 초점 약 720 px)에서 추정이 **fx 1767 px(가로 57도)**,
표본 140, 퍼짐 0.42로 문턱을 모두 통과했다. 공칭값의 약 2.4배라 손 3D 거리가 오히려 멀어진다.

원인: 결정 1의 식은 깊이 z가 카메라 화각과 무관하게 맞다고 전제한다. 그런데 메트릭 깊이 모델(Depth Anything V2
Metric Indoor)은 화각을 입력받지 않고 학습 영상의 화각(대략 50~60도)을 전제로 거리를 낸다. 초광각에서는 물체가
작게 보이는 만큼 멀다고 답하므로 z·l/f0가 커지고, p0/L은 카메라가 아니라 **깊이 모델의 화각 사전값**을
되돌려 준다 (4절의 "깊이로 잰 손바닥이 맞는 화각 약 60도" 빠른 측정도 같은 현상이었다). 합성 시험은 정답 깊이를
쓰므로 이 편향을 드러내지 못했다. 손 크기만으로는 f/z만 정해지고, f를 따로 정하려면 화각과 무관한 거리 근거가
필요하다.

바꾼 것: `prelabel.yaml depth.focal_estimate.enabled: false`(기본). 코드·계약 값(`estimated_hand_size`)은
그대로 두고, 화각을 함께 추정하거나 내부 파라미터를 입력받는 깊이 모델로 바꾼 뒤 실영상(체커보드가 있는 세션)에서
캘리브레이션과 비교해 맞을 때만 켠다. 그때까지 캘리브레이션 없는 세션은 화각 근사(`hfov_approx`)로 남고, 바로
잡는 길은 체커보드 캘리브레이션(`dlp calib intrinsics`)이다. F-PAR-65.
