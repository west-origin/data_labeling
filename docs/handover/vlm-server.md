# VLM 서버 (NVIDIA DGX Spark, vLLM)

[인수인계 첫 화면](README.md) · [개발 환경](development.md) · [ADR 0040](../adr/0040-ai-first-pass-review.md) ·
[구현 계획 WP17](../ai-implementation-plan.md#wp17-ai-1차-검수로컬-vlm)

AI 1차 검수(`dlp aireview`, WP17)와 행동 분류(`dlp actions run`, WP10)는 사내 OpenAI 호환 VLM 서버를
쓴다. 장비는 **NVIDIA DGX Spark**(GB10, Arm64 + CUDA, 통합 메모리 128 GB), 서빙은 **vLLM**으로 정했다
(ADR 0040 결정 3). 외부 API(Claude 등)는 쓰지 않는다. 이 문서는 Spark 1대를 받아 서버를 띄우고, 처리량을
재서 대수를 정하기까지의 절차다.

> 상태: 장비 도착 전. `services/vlm/`의 compose는 실제 Spark에서 아직 띄워 보지 않았다
> (`TODO(real-model)`). 이미지 태그와 vLLM 플래그 이름은 버전마다 달라서 도착 뒤 확인해 고정한다.

## 구성

| 서버 | 포트 | 모델 | 쓰는 명령 | 정책 |
| --- | --- | --- | --- | --- |
| `vlm-aireview` | 8000 | `Qwen/Qwen3-VL-8B-Instruct` (판정, 기본 후보) | `dlp aireview run`·`bench`·`probe` (WP17.3~) | `config/policies/aireview.yaml backend` |
| `vlm-actions` (`--profile actions`) | 8001 | `Qwen/Qwen2.5-VL-7B-Instruct` (행동 분류) | `dlp actions run --vlm-url` | `config/policies/actions.yaml vlm` |

- 판정 모델은 행동 분류 모델과 **다르다** (같은 모델이 자기 결과를 검수하면 같은 방향으로 틀린다).
  `tests/test_policy_consistency.py`가 검사한다.
- 두 모델은 한 Spark에 함께 들어간다 (7~8B BF16 가중치 약 16~17 GB씩 + KV 캐시). `gpu-memory-utilization`
  합을 0.7 안팎으로 두고 나머지를 프리라벨·영상 디코드·OS에 남긴다.
- 서비스 호스트(PostgreSQL, S3, Label Studio, CVAT, lakeFS)는 지금처럼 x86 호스트에 둔다. Spark는 추론
  작업자로만 쓴다 (CVAT 공식 이미지·테마 이미지의 arm64 지원을 확인하지 않았다). 개발 호스트의 lakeFS가
  8000 포트를 쓰므로 VLM 서버를 개발 호스트에 띄우지 않는다.
- 블러본 전용 서버다. 요청 이미지는 승인된 블러본에서만 나온다 (ADR 0040 I1). 원본 프레임에 VLM을
  쓰려면 별도 서버 인스턴스와 ADR이 필요하다.

## 띄우기

전제: DGX OS(Spark 기본 OS), NVIDIA Container Toolkit, Docker Compose v2. 사내망 고정 주소
(예: `10.0.0.11`).

```bash
cd services/vlm
cp vlm.env.example vlm.env          # 저장소에 넣지 않는다 (.gitignore)
# vlm.env: VLM_IMAGE(NGC vLLM 컨테이너 고정 태그), DLP_VLM_API_KEY, 모델 revision을 채운다
docker compose --env-file vlm.env up -d vlm-aireview
docker compose --env-file vlm.env --profile actions up -d     # 행동 분류 서버도 띄울 때
docker compose --env-file vlm.env logs -f vlm-aireview        # 가중치 내려받기·로드 확인
```

확인 (키는 셸 기록에 남지 않게 환경 변수로 둔다):

```bash
curl -s -H "Authorization: Bearer $DLP_VLM_API_KEY" http://10.0.0.11:8000/v1/models
```

응답의 `id`가 `aireview.yaml backend.model`과 같아야 한다 (compose는 `--served-model-name`을 모델 저장소
이름과 같게 둔다). 서버 이름을 따로 두려면(`--served-model-name` 별칭, llama.cpp 등) 그 이름을
`backend.served_name`에 적는다. 그러면 요청 본문의 `model`과 응답 대조 모두 그 이름을 쓴다 (vLLM은
`--served-model-name`에 준 이름이 아닌 요청 `model`을 404로 거부한다). 다르면 판정 실행이 설정
오류(종료 5)로 멈춘다 (I7).

### 권장 설정과 보안

- `--limit-mm-per-prompt '{"image":16}'`: 요청당 이미지 상한 (temporal 최대 12장, spatial 4장 + 확대
  조각 6장). `--max-model-len 16384`, `--enable-prefix-caching`(고정 접두사 = 시스템 + 온톨로지 블록).
- `--max-num-seqs 32~64`(연속 배칭), `--gpu-memory-utilization 0.4`(판정) / `0.3`(행동).
- 이미지 크기: 클라이언트가 긴 변(`max_side_px`)과 32px 배수(`image.align_px`, Qwen3-VL patch 16 ×
  merge 2)로 맞춰 보낸다. 서버가 다시 줄이지 않게 `--mm-processor-kwargs`의 최대 픽셀을 그보다 크게 둔다.
- **요청 본문 로그를 켜지 않는다** (`--enable-log-requests` 금지, 구 버전은 `--disable-log-requests`).
  이미지 base64가 로그에 남는다.
- `--api-key`(compose는 `VLLM_API_KEY` 환경 변수)와 방화벽으로 파이프라인 호스트만 허용한다. 파이프라인
  쪽은 `DLP_VLM_API_KEY`로 보낸다 (Authorization 헤더로만, 출력·기록하지 않는다).
- 가중치 revision을 고정한다 (`--revision`). 바꾸면 `aireview.yaml backend.fingerprint`를 바꿔 판정
  버전을 새로 만든다. FP8 등 양자화도 결과가 달라지므로 다른 fingerprint다. 엔진·양자화가 다른 서버를 한
  실행에 섞지 않는다.

## 파이프라인 호스트 설정

| 변수 | 쓰는 곳 | 예 |
| --- | --- | --- |
| `DLP_VLM_URL` | `dlp actions run`만 (행동 분류 서버) | `http://10.0.0.11:8001` |
| `DLP_VLM_URLS` | `dlp aireview run`·`bench`·`probe`의 판정 서버 (하나여도 이 변수, WP17.5) | `spark-01=http://10.0.0.11:8000,spark-02=http://10.0.0.12:8000` |
| `DLP_VLM_API_KEY` | 두 명령의 서버 API 키 | (출력·기록 금지) |

- `dlp aireview`는 `DLP_VLM_URL`로 대신하지 않는다. 판정 모델은 행동 분류 모델과 달라(위 구성) 행동 분류
  서버로 보내면 늘 서버 모델 불일치(I7, 종료 5)가 난다. `DLP_VLM_URLS`도 `--vlm-url`도 없으면 종료 2다.
- 주소는 `aireview.yaml server.internal_hosts`(사내 대역) 안이어야 한다 (WP17.5에서 검사).

## 처리량 재기와 대수 정하기 (WP17.3)

먼저 1대를 사서 잰다. 공식과 가정은 [구현 계획 WP17](../ai-implementation-plan.md#wp17-ai-1차-검수로컬-vlm)이다.

1. `dlp aireview probe --vlm-url …`: 모델 이름, 이미지 수 한도, `json_schema` 강제 여부, `full` 스키마
   프로필 지원, `cached_tokens` 제공, `/tokenize` 유무. 강제하지 않으면 `schema_enforcement: prompt`.
2. `dlp aireview bench --vlm-url … --concurrency 1,8,16,32,64 --frames 8,12 --side 336,448,672`:
   후보 모델 2~3개(`config/models.yaml external`의 AI 1차 검수 후보)의 프리필 tok/s와 생성 tok/s를
   **따로**, p50·p95 지연, 캐시 적중률, 스키마 적합률. `--decode`로 블러본 디코드 fps도 잰다.
3. `dlp aireview estimate --include-actions`: 실제 파일럿 영상의 aireview 단위 수와 행동 VLM 후보 밀도.
   행동 VLM은 후보 구간마다 직렬로 물어 대수를 정할 수 있다 (F-AIR-01).
4. 같은 Spark에서 VLM 밖 단계(OWLv2·깊이·블러 렌더) 처리 시간도 잰다.
5. 공식으로 대수를 정하고 이 문서 아래에 기록한다 (장비, 엔진 버전, 모델·revision, 수치, 결정).

## 판정 모델 바꾸기

1. 새 후보의 Hugging Face 모델 카드(`https://huggingface.co/api/models/<저장소>`의 `cardData.license`)로
   가중치 라이선스를 확인하고 `config/models.yaml external`에 license·training_data·commercial을 적는다
   (학습 데이터 미공개면 `review`, 비상업이면 `forbidden`이라 쓸 수 없다). `make licenses`가 검사한다.
2. `aireview.yaml backend.model`·`fingerprint`(필요하면 `served_name`, `image.align_px`)를 바꾼다. 판정
   버전이 바뀌어 모든 단위를 다시 판정한다. 예전 값으로 되돌리면 그 버전의 기존 판정을 다시 쓴다.
3. `dlp aireview eval --golden <버전> --gate`(WP17.8)로 골든 판정 정확도를 본다. 지금은 게이트가 실패해도
   경고만 한다 (차단은 파일럿 뒤 결정).

## 운영 모드

`review.yaml ai_review.mode`(WP17.6): 코드 기본값 off, 장비 도착 전 저장소 값 advisory. 파일럿(WP17.9)
뒤 required로 바꾸면 판정이 없는 세션의 검수 배정을 미루고, prelabeled 뒤 `max_wait_h`(24) 시간이 지나면
`waived:timeout`으로 진행한다. 서버가 멈춰도 검수가 영구히 멈추지 않는다 (ADR 0040 결정 7).

## Arm64(Spark) 재현 점검표

Spark에서 파이프라인 일부(프리라벨·프라이버시 탐지·VLM 클라이언트)를 돌리려면 저장소를 Arm64에서
재현해야 한다. 도착하면 아래를 차례로 확인하고 결과를 이 표에 적는다 (지금은 모두 미확인).

| 항목 | 확인 방법 | 막히면 |
| --- | --- | --- |
| uv·Python 3.12 (aarch64) | `uv python install`, `make install` | — |
| PyAV, OpenCV(contrib headless), pyarrow, h5py 휠 | `make install` 성공 | 소스 빌드 |
| onnxruntime CPU (aarch64) | `make check`의 모델 테스트 | — |
| onnxruntime-gpu (aarch64 CUDA, Blackwell) | `make models` 뒤 OWLv2·깊이 실행 | PyPI 휠이 없을 수 있다: NGC 컨테이너 또는 CPU 경로 |
| MediaPipe Linux aarch64 휠, rtmlib | 프리라벨 손·전신 테스트 | CPU 경로, 컨테이너 |
| FiftyOne 내장 MongoDB (arm64) | `make install-curation`, 큐레이션 테스트 | 외부 MongoDB |
| `scripts/lerobot-env/uv.lock` | `make test-isolated` | 그 디렉터리에서 `uv lock`으로 arm64 재잠금 |
| `make check` 전체 | 통과 | 실패 항목을 issues.yaml에 등록 |
