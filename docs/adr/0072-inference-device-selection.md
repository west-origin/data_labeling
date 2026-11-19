# ADR 0072: 추론 장치를 환경 변수로 고르고, GPU를 골랐으면 조용히 CPU로 내려가지 않는다

- 상태: 채택 (구현됨. GPU가 없는 환경에서 값 해석·CPU 세션·CPU 대체 차단을 검사했고, 실제 onnxruntime-gpu 1.30.0
  판을 GPU 없는 기계에 깔아 CUDA를 받지 못한 세션이 멈추는 것을 확인했다. GPU에서 실제로 빨라지는지는 재지 않았다)
- 날짜: 2026-11-19
- 관련: ADR 0009(실제 모델 레지스트리, 보충 2026-11-14), F-PAR-98
- 소유자 결정: 2026-11-19 소유자가 이 문제의 결정을 위임했다 ("전부 네 추천대로"). 아래 3의 "장치는 모델 버전에
  넣지 않는다"가 F-PAR-98이 소유자 결정으로 남겨 둔 부분이다.
- 계약 변경: 없음 (라벨·DB·JSON Schema 그대로)

## 배경

자동 라벨과 블러 탐지가 CPU 4코어에서 90초 영상에 34분(영상 길이의 약 23배) 걸렸다 (2026-11-14 실영상).
ONNX Runtime 세션(`dlp_models.onnx.OnnxModel`: OWLv2, 메트릭 깊이, SAM 2.1, YOLOX)이 실행 제공자를
`CPUExecutionProvider` 하나로 고정했고, 전신 자세(rtmlib RTMPose·YOLOX)는 `device="cpu"`였다. 그래서 ADR 0009의
"GPU 서버에서는 onnxruntime-gpu로 바꾸면 된다"는 패키지를 바꿔도 맞지 않았다.

조건 둘이 더 있다. ONNX Runtime은 CUDA 실행 제공자를 요청받았는데 만들지 못하면(판이 CPU판이거나 CUDA·cuDNN
라이브러리를 찾지 못함) 경고만 내고 CPU 세션을 만든다. 또 onnxruntime과 onnxruntime-gpu는 같은 `onnxruntime`
모듈을 깔아 한 환경에 함께 둘 수 없고, rtmlib이 CPU판 `onnxruntime`을 의존성으로 요구한다.

## 결정

1. **장치는 환경 변수 `DLP_INFERENCE_DEVICE`로 고른다** (`dlp_models.device`). 값은 `cpu`(기본), `auto`, `cuda`다.
   `auto`는 이 환경의 onnxruntime에 CUDA 실행 제공자가 있으면(GPU판) GPU, 없으면 CPU다. `cuda`인데 CUDA 실행
   제공자가 없으면 멈춘다. 그 밖의 값도 멈춘다. 정책 YAML이 아니라 환경 변수인 것은 장치가 기계의 성질이고,
   정책 절은 모델 버전 해시에 들어가기 때문이다 (3).
2. **GPU를 골랐으면 조용히 CPU로 내려가지 않는다.** 세션을 만든 뒤 실제 실행 제공자(`get_providers()`)에 CUDA가
   없으면 `DeviceError`로 멈춘다 (`check_session`, `cuda`와 `auto` 모두). GPU판을 깐 기계는 GPU를 쓰려는
   기계이므로, 라이브러리가 맞지 않으면 34분 걸리는 CPU 실행보다 바로 알리는 쪽이 낫다. GPU 세션의 제공자 목록은
   CUDA 다음에 CPU다 (CUDA가 지원하지 않는 연산만 CPU로 간다). 첫 GPU 세션 전에 `onnxruntime.preload_dlls()`로
   pip로 깐 CUDA·cuDNN 라이브러리를 불러온다 (CPU판에서는 아무것도 하지 않는다).
3. **장치는 모델 버전에 넣지 않는다.** 같은 모델 파일·정책이면 같은 판으로 본다. 넣으면 장치를 바꿀 때 모든 세션이
   다시 돌고, 프라이버시는 새 탐지 세대가 되어 사람 승인이 풀린다(ADR 0033). CPU와 GPU의 차이는 부동소수점 연산
   순서에서 오는 작은 값 차이이고, 모든 결과는 어차피 사람 검수를 거친다. 어느 장치로 돌렸는지는 무거운 단계
   (`dlp prelabel run`, `dlp privacy detect`)가 맨 앞에 출력하는 한 줄(`추론 장치: cuda (DLP_INFERENCE_DEVICE=auto)`)로
   구분한다. 운영 콘솔로 돌린 작업은 그 출력이 작업 로그(`DLP_CONSOLE_DIR`의 `logs/<작업 ID>.log`)에 남는다. DB에는
   남기지 않는다.
4. **바꾸는 곳은 ONNX Runtime 둘이다.** `OnnxModel`과 rtmlib 전신 자세(`device`)다. MediaPipe 손·COCO 객체는
   `Delegate.CPU`, YuNet 얼굴은 OpenCV 기본 백엔드 그대로다. MediaPipe GPU 위임은 리눅스에서 EGL과 GPU 드라이버를
   따로 요구해 여기서 확인할 수 없고, OpenCV PyPI 판에는 CUDA 백엔드가 없다. 재학습 YOLOX 학습기는 이미 격리
   환경에서 `nvidia-smi`로 GPU를 고른다 (ADR 0062).
5. **GPU판 설치는 잠금 파일 밖의 교체다** (`make install-gpu`). `make install` 뒤 같은 판의
   `onnxruntime-gpu[cuda,cudnn]`(CUDA 13·cuDNN 9 라이브러리를 pip로 함께 받는다)로 바꾼다. 잠금 파일에 선택 설치로
   넣지 않은 것은 rtmlib이 CPU판을 요구해 둘이 함께 잠기기 때문이다. 교체 뒤에는 `uv run`이 잠금 파일대로 CPU판을
   다시 깔지 않도록 `UV_NO_SYNC=1`을 둔다. `make install`을 다시 하면 CPU판으로 돌아간다. 설치 확인은
   `dlp models device`다 (아주 작은 Identity 모델로 세션을 열어 실제 실행 제공자를 보인다).

## 결과

- 기본은 지금까지와 같은 CPU다. CI와 개발 환경은 바뀌지 않는다.
- GPU 서버는 `make install-gpu`, `export UV_NO_SYNC=1 DLP_INFERENCE_DEVICE=cuda`, `uv run dlp models device`로
  준비한다 (`docs/handover/development.md` "GPU 서버"). 운영 콘솔은 `.env`의 `DLP_INFERENCE_DEVICE`를 자식 명령에
  넘기고, 명령을 `uv run` 없이 같은 파이썬으로 돌리므로 다시 깔지 않는다.
- 남은 것 (F-PAR-98): 실제 GPU에서 OWLv2(양자화 가중치)·깊이·SAM 2.1·전신 자세가 얼마나 빨라지는지, 블러 탐지·자동
  라벨 전체 시간이 얼마인지 재지 않았다. MediaPipe와 YuNet은 CPU 그대로다. aarch64(Blackwell) GPU판 휠은 PyPI에
  있지만(1.30.0, manylinux_2_34) 그 GPU를 지원하는지는 확인하지 않았다.

## 검증

- `packages/models/tests/test_device.py`: 값 해석(기본 cpu, 대소문자·공백, auto, 잘못된 값), 제공자 목록, 세션 확인,
  출력 한 줄, 기본 CPU 세션으로 Identity 모델 실행, CPU판에서 `cuda`가 세션 전에 멈춤, GPU판이 깔렸지만 CUDA를 못
  받은 경우(제공자 목록에는 CUDA가 있고 세션은 CPU)에 `cuda`·`auto` 모두 멈춤.
- `packages/cli/tests/test_device_cmds.py`: `dlp models device` 성공·실패, 설정이 틀리면 `dlp prelabel run`과
  `dlp privacy detect`가 DB·버킷에 닿기 전에 1로 끝남.
- 수동 확인 (2026-11-19, GPU 없는 기계): 따로 만든 환경에 실제 onnxruntime-gpu 1.30.0을 깔았다. 제공자 목록은
  TensorRT·CUDA·CPU였고, `cpu`는 CPU 세션, `auto`·`cuda`는 ONNX Runtime이 CUDA 라이브러리(libcublasLt.so.13)를
  찾지 못해 CPU 세션을 만들었고 `check_session`이 `DeviceError`로 멈췄다.
