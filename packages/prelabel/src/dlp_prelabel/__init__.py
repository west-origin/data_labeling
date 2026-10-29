"""자동 프리라벨 (WP8): `dlp prelabel run <세션>`.

실제 모델을 아직 연동하지 못한 곳은 주석에 "TODO(real-model)"로 표시한다 (make todo-models).

흐름: 원본 영상 스트림 → 모델 어댑터(Predictor) → 모델 출처 라벨(손 21관절, 전신 17관절·착용자 팔
arm3, 객체·도구 박스, 모양 마스크) → 3D 궤적(메트릭 깊이) → 접촉 구간(장갑·영상 융합,
hand_state) → 3인칭 착용자 매칭 → 카메라 자세·월드 좌표 3D → 궤적 따르기 (순서와 조건은
`runner` 설명).
DB 쓰기는 추가만 한다: `label_records`에 새 레코드(덮어쓰지 않고 이전 버전은 retracted 레코드로
지운다), 단계 실행 기록 `prelabel_stage_runs`(결과 0개 실행 포함), 3D 궤적의 검수 물려받기 기록
`trajectory_inheritance`. 그 기록과 같은 트랜잭션에서 물려받은 3D의 검수 열만 `record_review`로
바꾼다.

모듈:
- `runner`: 세션 단위 실행 (`run_prelabel`). 멱등성·이전 버전 지우기·검수 결과 보호를 맡는다.
- `adapters/`: 모델 어댑터 (MediaPipe 손·객체, RTMPose 전신, OWLv2 도구, YOLOX 고정 클래스 도구)와
  CPU stub.
- `common`: 프레임 순회(PTS ms)와 같은 ms 프레임 건너뛰기, 박스 IoU 추적, 입력 해시, 모델 출처 라벨
  생성, 라벨 ID 길이 맞춤(`fit_label_id`).
- `policy`: `config/policies/prelabel.yaml` 로더와 정책 절 해시(`digest`).
- 2D: `arm`(착용자 팔 골격 arm3, ADR 0058), `wearer_suppress`(착용자 몸 탐지 억제, ADR 0058),
  `masks`(객체·손 모양 마스크, SAM 2.1, ADR 0059), `temporal`(박스 트랙 거르기·손 평활·짧은 끊김
  보간, ADR 0054), `scene`(장면 분류와 장면별 도구 질의, ADR 0055).
- 3D: `lift3d`(손 관절·객체 중심을 카메라 좌표 3D 궤적으로), `handfit`(손 21관절 3D 맞춤, ADR 0044),
  `palm`(작업자 손 크기 추정, ADR 0045·0052), `fov`(손 크기로 초점 거리 추정, 기본 끔, ADR 0056),
  `constraints3d`(손 3D 물리 제약 표시, ADR 0054), `campose`(바디캠 6자유도 자세, ADR 0046),
  `worldframe`(월드 좌표 궤적, ADR 0046), `posemath`(회전·자세 계산 도우미).
- `contact`: 장갑 압력·손가락 끝-박스 거리로 접촉 구간 추정과 융합 (모델이 아닌 알고리즘).
- `wearer`: 3인칭 영상 속 바디캠 착용자 찾기 (운동 신호 상관).

관련 ADR: 0008(프리라벨 어댑터), 0009(실제 모델 레지스트리), 0010(상업 사용), 0015(감사 정정:
버전·지우기), 0019(시간 규약), 0026(감사 4차: 검수 결과 보호·착용자 사본).
"""
