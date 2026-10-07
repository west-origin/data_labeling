"""모델 가중치 레지스트리와 공용 런타임.

여러 단계(프리라벨 `dlp prelabel run`, 프라이버시 `dlp privacy detect`, 모델 받기 `make models`)가
함께 쓰는 실제 모델 계층이다. 관련: WP8(프리라벨), ADR 0009(실제 모델 레지스트리), ADR 0010(상업
사용).

모듈:
- `registry`: `config/models.yaml` 로더, 가중치 경로·해시 확인(`resolve`), 내려받기(`fetch`),
  상업 사용 분류 검사(`ModelRegistry.violations`, `make licenses`).
- `onnx`: ONNX Runtime CPU 세션의 얇은 타입 래퍼(`OnnxModel`).
- `owlv2`: OWLv2 오픈 보캐뷸러리 탐지(텍스트 질의 → 박스). 전처리·후처리(NMS) 포함.
- `depth`: Depth Anything V2 메트릭 깊이(미터)와 픽셀 → 카메라 좌표 역투영(`Intrinsics`).

주의:
- 가중치는 저장소에 넣지 않는다 (`data/models/`, `make models`·`make export-models`).
- 파일이 없거나 해시가 다르면 `dlp_schema.predictor.ModelUnavailableError`를 낸다. 호출하는 어댑터는
  이를 받아 stub이나 전수 검수로 넘긴다.
- 이 패키지는 라벨·DB·원본 버킷을 다루지 않는다. 입력은 메모리의 RGB 이미지(uint8)다.
"""
