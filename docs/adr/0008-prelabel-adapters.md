# ADR 0008: 자동 프리라벨 어댑터

- 상태: 채택 (모델 목록과 미연동 목록은 ADR 0009가 갱신)
- 날짜: 2026-10-07
- 관련: WP8

> **갱신:** 이 문서의 "Hugging Face 접근 불가"와 미연동 목록은 작성 당시 기준이다. 지금 상태는 ADR 0009(실제 모델 연동),
> ADR 0010(상업 사용 분류), ADR 0015(감사 뒤 정정)를 따른다.

## 결정

1. **모든 프리라벨 모델은 공통 `Predictor`(`run(clip) -> list[LabelRecord]`)를 구현하고 CPU용 stub을 함께 둔다.** stub은 정답 라벨을 모델 출처로 돌려주는 `OraclePredictor`다. CI와 다른 모듈 개발은 stub으로 진행한다.
2. **이 환경(CPU, Hugging Face 접근 불가)에서 실제로 연동한 모델은 MediaPipe다.**
   - 손 21관절 (Hand Landmarker). MediaPipe는 거울상 입력을 가정하므로, 거울상이 아닌 바디캠에서는 왼손·오른손을 뒤집는다 (`hands.input_is_mirrored`).
   - 전신 포즈 (Pose Landmarker lite, 33점 → COCO 17점)
   - COCO 객체 탐지 (EfficientDet-Lite0). 온톨로지에 대응하는 클래스(컵, 변기, 세면대, 침대, 병→분무기, 사람)만 남긴다.

   모델 파일은 `make models`로 받고 sha256을 확인한다. MediaPipe는 EGL·GLES 시스템 라이브러리가 필요하다.
3. **MediaPipe와 같은 cv2 모듈 충돌을 피하려고 작업공간 전체를 `opencv-contrib-python-headless`로 통일한다.** MediaPipe가 요구하는 GUI판 OpenCV는 uv override로 지운다.
4. **모델이 아닌 알고리즘으로 풀 수 있는 것은 실제로 구현한다.**
   - 장갑 접촉 구간: 히스테리시스 문턱. 합성 데이터에서 정답과 17 ms 이내다.
   - 영상 접촉 추정: 손가락 끝-객체 박스 거리
   - 융합: 장갑 시각과 영상의 대상
   - 3인칭 착용자 매칭: 바디캠 IMU와 인물 손목 속도의 상관. 찾은 인물 트랙은 entity_id="wearer"인 새 레코드로 남긴다.
5. **실제 모델을 아직 연동하지 못한 곳은 코드·정책 파일 주석에 `TODO(real-model):`로 표시하고, 필요한 것(GPU, 가중치 접근, 빌드)을 적는다.** `make todo-models`로 전부 볼 수 있다. 실행 시에도 CLI가 미연동 기능을 출력한다.

## 미연동 목록 (`make todo-models`)

| 기능 | 필요한 것 |
| --- | --- |
| 오픈 보캐뷸러리 객체·도구 탐지, 도구 작용부 마스크 | Grounding DINO / OWLv2 + SAM 2, GPU, Hugging Face |
| 블러용 문서·화면·사진물·문패·반사면 영역 탐지 (WP5) | 같은 오픈 보캐뷸러리 탐지기 |
| 카메라 6자유도 궤적 | 시각-관성 SLAM (Basalt, ORB-SLAM3) C++ 빌드, GPU |
| 단안 메트릭 깊이 | Depth Anything V2 Small 가중치 (Hugging Face) |
| 영상 기반 학습 접촉 분류기 | 장갑 세션 정답 누적 후 WP13에서 학습 |

## 확인이 남은 것

- 실제 바디캠 영상에서 MediaPipe 손·전신·객체 품질 (사람 확인, 골든셋 평가).
- 장갑 접촉 문턱(on 0.3, off 0.15)은 합성 압력 기준이다. 장갑 기종이 정해지면 실제 압력 단위로 다시 정해야 한다.
- 영상 접촉 휴리스틱은 움직이는 객체에서 객체 추적이 정확해야 대상을 맞힌다. 합성 데이터에서는 고정 객체만 검증했다.
