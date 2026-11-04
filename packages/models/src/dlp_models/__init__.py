"""모델 가중치 레지스트리와 공용 런타임.

여러 단계(프리라벨 `dlp prelabel run`, 프라이버시 `dlp privacy detect`, 행동 `dlp actions run`,
AI 1차 검수 `dlp aireview`, 모델 받기 `make models`)가
함께 쓰는 실제 모델 계층이다. 관련: WP8(프리라벨), ADR 0009(실제 모델 레지스트리), ADR 0010(상업
사용).

모듈:
- `registry`: `config/models.yaml` 로더, 가중치 경로·해시 확인(`resolve`), 내려받기(`fetch`),
  상업 사용 분류 검사(`ModelRegistry.violations`, `make licenses`).
- `onnx`: ONNX Runtime CPU 세션의 얇은 타입 래퍼(`OnnxModel`).
- `owlv2`: OWLv2 오픈 보캐뷸러리 탐지(텍스트 질의 → 박스). 전처리·후처리(NMS) 포함.
- `yolox`: YOLOX 고정 클래스 탐지(공개 데이터 미세조정 도구 탐지기, ADR 0060). letterbox·격자
  디코드·클래스별 NMS 포함.
- `depth`: Depth Anything V2 메트릭 깊이(미터)와 픽셀 → 카메라 좌표 역투영(`Intrinsics`).
- `vlm`: OpenAI 호환 VLM 서버 공통 전송(요청 본문, 오류 분류, 백오프·재시도). 행동 분류
  (`dlp_actions`)와 AI 1차 검수(`dlp_aireview`)가 함께 쓴다 (WP17.1, ADR 0040).

버전 문자열 (라벨 `provenance.model_version`과 라벨 ID의 `version_tag`에 들어간다)
- 로컬 가중치: `registry.resolve`가 `<이름>-<실제 파일 sha256 앞 12자>`를 돌려준다. 가중치 파일이
  바뀌면 버전이 바뀌어 그 모델을 쓰는 단계가 다시 돌고, 검수 전인 이전 결과만 지운다 (ADR 0015).
  `config/models.yaml`의 등록 해시가 아니라 실제로 읽은 파일의 해시다.
- 외부 서버 모델(`models.yaml external`, VLM): 가중치가 이 저장소 밖에 있어 해시를 볼 수 없다.
  버전은 호출자가 정책의 모델 이름으로 만든다 (행동 `vlm-<모델>`, AI 1차 검수
  `aireview-<digest>+vlm-<모델>`). 엔진·양자화·revision 차이는 AI 1차 검수 정책의 `fingerprint`가
  digest에 넣어 구별한다.

주의:
- 가중치는 저장소에 넣지 않는다 (`data/models/`, `make models`·`make export-models`).
- 파일이 없거나 해시가 다르면 `dlp_schema.predictor.ModelUnavailableError`를 낸다. 호출 경로가
  중단·명시적 부분 실행·전수 검수를 결정한다. 기본 프리라벨은 선택 모델 미준비 시 중단하며,
  실제 모델의 결과를 stub으로 조용히 대신하지 않는다.
- `vlm`은 주소가 사내 대역인지 검사하지 않는다. 그 검사는 호출자가 보내기 전에 한다
  (`dlp_aireview.backend.check_hosts`). 이 모듈은 프록시 환경 변수를 따르지 않아, 검사한 주소로만
  연결한다.
- 이 패키지는 라벨·DB·원본 버킷을 다루지 않는다. 입력은 메모리의 RGB 이미지(uint8)다
  (`vlm`은 호출자가 고른 블러본 프레임의 JPEG 바이트).
"""
