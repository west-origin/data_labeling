# VLM 서버 (NVIDIA DGX Spark, vLLM)

[인수인계 첫 화면](README.md) · [개발 환경](development.md) · [ADR 0040](../adr/0040-ai-first-pass-review.md) ·
[구현 계획 WP17](../ai-implementation-plan.md#wp17-ai-1차-검수로컬-vlm)

AI 1차 검수(`dlp aireview`, WP17)와 행동 분류(`dlp actions run`, WP10)는 사내 OpenAI 호환 VLM 서버를
쓴다. 장비는 **NVIDIA DGX Spark**(GB10, Arm64 + CUDA, 통합 메모리 128 GB), 서빙은 **vLLM**으로 정했다
(ADR 0040 결정 3). 외부 API(Claude 등)는 쓰지 않는다. 이 문서는 Spark 1대를 받아 서버를 띄우고, 처리량을
재서 대수를 정하기까지의 절차다.

> 상태: 장비 도착 전. `services/vlm/`의 compose는 실제 Spark에서 아직 띄워 보지 않았다
> (`TODO(real-model)`). 이미지 태그와 vLLM 플래그 이름은 버전마다 달라서 도착 뒤 확인해 고정한다.
> 점검·벤치·추정 명령(`dlp aireview probe|bench|estimate`, WP17.3)은 구현했고 가짜 서버로만
> 검사했다. 실측 기록은 아래 "실측 기록" 표에 채운다.

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
- `--enable-prompt-tokens-details`: 응답 usage에 `prompt_tokens_details.cached_tokens`를 넣는다 (vLLM
  기본은 꺼짐). 없으면 벤치 캐시 적중, 추정의 고정 접두사 캐시 판단, probe `cached_tokens` 점검이
  "모름"이 된다 (추정은 그때 고정 접두사도 매번 계산한다고 보수적으로 센다).
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
| `DLP_VLM_URLS` | `dlp aireview probe`·`bench`(WP17.3)·`run`(WP17.5)의 판정 서버 (하나여도 이 변수) | `spark-01=http://10.0.0.11:8000,spark-02=http://10.0.0.12:8000` |
| `DLP_VLM_API_KEY` | 두 명령의 서버 API 키 | (출력·기록 금지) |

- `dlp aireview`는 `DLP_VLM_URL`로 대신하지 않는다. 판정 모델은 행동 분류 모델과 달라(위 구성) 행동 분류
  서버로 보내면 늘 서버 모델 불일치(I7, 종료 5)가 난다. `DLP_VLM_URLS`도 `--vlm-url`도 없으면 종료 2다.
- 주소는 `aireview.yaml server.internal_hosts`(사내 대역) 안이어야 한다 (probe·bench·run이 보내기
  전에 검사한다). 호스트 이름이면 DNS로 풀어 모든 주소를 본다.
- VLM 요청(`dlp aireview`, `dlp actions run`)은 `HTTP_PROXY`·`HTTPS_PROXY`·`ALL_PROXY` 환경 변수를
  따르지 않고 그 주소로 직접 연결한다 (프록시를 따르면 사내 주소 검사를 통과한 요청과 API 키가
  프록시로 나간다). 사내 CA 인증서는 `SSL_CERT_FILE`·`SSL_CERT_DIR`로 계속 쓸 수 있다.

## 처리량 재기와 대수 정하기 (WP17.3)

먼저 1대를 사서 잰다. 명령 세 개는 모두 DB에 쓰지 않는다. CI는 가짜 서버(`dlp_aireview.testing.
FakeOpenAIServer`)로 같은 명령의 보고서 형식만 검사한다 (`packages/cli/tests/test_aireview_cmds.py`).
실측 값은 Spark에서만 나온다.

| 명령 | 하는 일 | 종료 코드 |
| --- | --- | --- |
| `dlp aireview probe` | 서버 점검: 사내 주소, 모델 목록, 엔진 버전, `/tokenize`, 이미지 입력·응답 모델 대조, 요청당 이미지 수 한도, `json_schema` 강제 여부(어기기 쉬운 지시로 시험), `full` 프로필 키워드, `cached_tokens`, 문맥 길이. 정책 조정안(권고)을 출력한다 | 0 통과 / 2 주소 없음 / 3 서버 장애 / 5 설정 불일치 |
| `dlp aireview bench` | 장비 1대 처리량: 합성 장면을 실제 계획기·프롬프트·스키마·재검사로 판정한다. 묶음(temporal·temporal_long·spatial) × 프레임 수(temporal) × 긴 변 설정(정책 값은 늘 포함)마다 동시성 단계별 단위/s, p50·p95, 관측 프리필·생성 tok/s, 캐시 적중, 스키마 적합률, 재시도·fallback율, 문맥 초과 수, 첫 토큰 시간. 무릎 동시성에서 프리필·생성 용량을 따로 잰다(`--no-capacity`로 끔). `--cache-probe`, `--decode`(CPU 순차 디코드 fps) | 0 / 2 / 3 서버 장애·서버 멈춤 / 5 |
| `dlp aireview estimate` | DB의 실제 세션으로 영상 1시간당 단위·대상·프레임·이미지·글·예상 출력 토큰을 센다. `--include-actions`면 행동 VLM 요청(후보 구간) 수도. `--bench 보고서`를 주면 공식으로 장비 대수를 낸다 | 0 |

벤치가 재는 방식 (`dlp_aireview.bench`)

- 입력은 기본이 합성 장면(`dlp_fixtures.aireview`, 사람·글자 없음)이다. 프레임 크기는 `aireview.yaml
  estimate.frame_px`(기본 1920x1080, `--frame-size`로 바꾼다). 실제 세션으로 재려면 `--source
  session:<id> --url <DB> --store s3`: 그 세션의 승인된 블러본(블러본 경계 `open_blurred`, 장갑 문장 없음)을
  임시 디렉터리에 받아 쓴다. DB·라벨링 버킷은 읽기만 하고, 승인 전·보류·사용 중지면 종료 6이다.
- 요청마다 고정 블록(온톨로지) 끝에 측정 표식 한 줄을 넣어, system만 접두사 캐시에 맞고 온톨로지 뒤
  (이미지·가변 글)는 매번 새로 계산하게 한다.
- 요청마다 새 이미지: 합성 장면의 계획 단위는 `--requests`(200)보다 적어 단위를 돌려 쓴다. 같은 이미지를
  다시 보내면 vLLM의 멀티모달 전처리 캐시(`--mm-processor-cache-gb`, 기본 켬)와 이미지 해시 기반 인코더
  출력 재사용이 비전 인코딩을 건너뛰어 처리량이 부풀려진다. 그래서 벤치는 측정 요청마다 이미지 모서리에
  표식별 잡음 조각을 넣어 다시 인코딩한다 (시간 측정 전에). 보고서의 점마다 `distinct_image_sets`가
  요청 수와 같아야 하고, 적으면 진행 줄에 경고가 나온다. 서버 캐시는 끄지 않는다 (운영도 켜 두고,
  운영에서 이 캐시는 같은 창 분할 요청에만 맞는다).
- 그래서 벤치의 요청당 계산량은 운영보다 조금 크다 (운영은 온톨로지 블록까지 캐시에 맞고 같은 창
  분할 요청이 이미지를 공유한다). 이 점에서 벤치 처리량은 보수적이다.
- 서버 장애(과부하 503 포함)는 다시 묻지 않고 그 점의 오류로 센다. 무릎 지점 = 장애 0이고 p95가
  `server.timeout_s`보다 작은 점 중 단위/s가 가장 큰 점(같으면 낮은 동시성). 그 동시성이
  `server.concurrency_per_endpoint` 후보다.
- 점의 프리필·생성 tok/s는 그 혼합 실행의 관측 속도이지 용량이 아니다. 무릎 동시성에서 용량을 따로
  잰다: 프리필은 생성 1토큰 요청(요청마다 새 이미지), 생성은 프롬프트가 캐시에 맞는 요청에 구조화 출력
  없이 `max_tokens = min_tokens = --decode-tokens`(256)와 `ignore_eos`(vLLM 확장)로 생성 길이를 고정한다.
  서버가 확장을 따르지 않으면(`honored=false`) 또는 거부하면(`error`) 추정이 그 용량을 쓰지 않는다
  (구조화 출력 문법의 생성 비용은 들어가지 않는다).
- 보고서: `reports/aireview-bench/<장비>-<UTC 시각>.json` (Git 제외). 서버 모델 목록, 엔진 버전, 정책
  판정 버전·모델·fingerprint·스키마 방식·셈법 값(`align_px`·`text_bytes_per_token`), 격자, 설정별
  측정값(무릎, 근사 토큰, 프리필·생성 용량), 디코드 fps를 담는다.

### Spark 실측 절차

1. 서버를 띄우고(위 "띄우기") 파이프라인 호스트에서 점검한다. 실패 항목은 권고대로 서버 플래그나
   `aireview.yaml backend`를 고치고 다시 돌린다.

   ```bash
   export DLP_VLM_URLS="spark-01=http://10.0.0.11:8000" DLP_VLM_API_KEY=...   # 키는 셸 기록에 남기지 않는다
   dlp aireview probe --out reports/aireview-probe/spark-01.json
   ```

2. 후보 모델마다(`config/models.yaml external`의 AI 1차 검수 후보 2~3개, 기본 Qwen3-VL-8B) 서버를
   그 모델로 다시 띄우고 벤치를 돈다. 모델을 바꾸면 `aireview.yaml backend.model`·`fingerprint`·
   `image.align_px`도 그 모델에 맞춘다 (판정 버전이 바뀐다).

   ```bash
   dlp aireview bench --concurrency 1,8,16,32,64 --frames 12 --side 336,448,672 \
       --requests 200 --warmup 10 --cache-probe --decode --device spark-01
   ```

   `--frames`·`--side`는 비교할 후보이고, 지금 정책 값(`aireview.yaml`의 묶음별 프레임 수·긴 변)은 늘
   잰다 (주지 않으면 정책 값만). 추정은 정책 값과 같은 설정의 측정만 쓴다. 정책 값을 바꾸면 벤치를 다시
   돈다.
3. 실제 파일럿 세션(프리라벨·관계·행동까지 끝난 것)으로 작업량을 센다. 행동 VLM 처리량은 벤치가
   재지 않는다. 같은 Spark에서 파일럿 세션 하나에 `dlp actions run`을 돌린 시간과 출력의
   `segments` 합으로 요청/s를 구해 `--actions-req-s`로 준다 (손마다 직렬이라 세션 여러 개를 동시에
   돌린 값이 장비 처리량이다).

   ```bash
   dlp aireview estimate --prelabeled --include-actions \
       --bench reports/aireview-bench/spark-01-<시각>.json --actions-req-s <실측>
   ```

4. 같은 Spark에서 VLM 밖 단계(OWLv2·깊이 `dlp prelabel run`, 블러 렌더 `dlp privacy render`) 처리
   시간도 파일럿 세션으로 잰다 (영상 1시간당 장비 시간). 대수 공식의 D에 더해 함께 판단한다.
5. 아래 기록 양식을 채워 이 문서에 남기고, 대수와 다음 구매를 결정한다. 지적 비율(`estimate.flag_ratio`)
   등 가정 값은 파일럿 판정 결과(WP17.9)로 바꾼 뒤 다시 계산한다.

공식 (`dlp_aireview.estimate`, 가정 값은 `aireview.yaml estimate`)

```
T_s  = 무릎 단위/s × t(벤치) ÷ t(작업량)                         (묶음 s)
t(x) = P_x ÷ C_p + O_x ÷ C_d                                     (단위 하나의 장비 시간 모형)
       P = 단위당 새 프롬프트 근사 토큰 (이미지·확대 조각 + 가변 글 + 캐시에 맞지 않은 고정 접두사)
       O = 단위당 출력 토큰 (벤치는 무릎 실측, 작업량은 예상 출력)
       C_p, C_d = 벤치가 무릎 동시성에서 따로 잰 프리필·생성 용량
D    = Σ_s (영상 1시간당 단위_s ÷ T_s) + 행동 VLM 영상 1시간당 요청 ÷ actions_req_s
대수  = ceil(H × (1+ρ) × D / (86,400 × u))        H 164, ρ 0.2, u 0.7
디코드 = H × video_fps × 영상 스트림 수 ÷ (디코드 fps × decode_workers) ≤ 24시간
```

- 작업량 구성(단위당 토큰)이 벤치와 같으면 T_s = 무릎 단위/s다. 무릎이 프리필에 묶였는데 실제 출력이
  더 길면 생성 몫만큼만 T가 준다 (관측 생성 tok/s를 "디코드 한계"로 쓰지 않는다).
- 벤치와 추정은 같은 근사 셈법(이미지 ⌈w/align⌉·⌈h/align⌉, 글 UTF-8 바이트 ÷ `text_bytes_per_token`)을
  써서 서버 토크나이저 차이가 비율에서 지워진다. 보고서의 셈법 값(`align_px`·`text_bytes_per_token`)이
  지금 정책과 다르면(다른 후보 모델의 보고서) 그 보고서용 작업량을 보고서 셈법으로 다시 센다. 모델·
  fingerprint가 다르면 경고만 한다 (후보 비교용).
- 서버가 `cached_tokens`를 주지 않거나 0이면 고정 접두사(시스템 + 온톨로지 블록)도 양쪽에서 매번
  계산한다고 본다.
- 용량 측정이 없는 보고서(`--no-capacity`, 이전 형식, 생성 길이 고정 실패)는 무릎 단위/s를 보정 없이
  쓰고 경고한다. 벤치 합성 장면과 실제 세션의 단위당 토큰이 다르면 틀리므로 용량을 재는 것을 기본으로
  한다.
- 근사가 남기는 오차: 추정은 프레임 수를 원하는 시각 수(상한)로 세고, 장갑 문장을 넣지 않는다. 확대
  조각은 실행과 같은 규칙(`overlay.crop_plan`)으로 센다.

### 실측 기록 (장비 도착 뒤 채운다)

| 항목 | 값 |
| --- | --- |
| 날짜·장비·엔진 버전(probe) | |
| 모델·revision·fingerprint | |
| probe 결과 (실패·권고) | |
| 묶음별 무릎 지점 (동시성, 단위/s, p95, 캐시 적중) | |
| 묶음별 프리필·생성 용량 (무릎 동시성, 생성 길이 고정 `honored`) | |
| 디코드 fps (CPU) | |
| 파일럿 세션 수·영상 시간, 영상 1시간당 단위(묶음별)·행동 요청 | |
| 행동 VLM 요청/s (실측) | |
| VLM 밖 단계 장비 시간 (OWLv2·깊이·렌더) | |
| D, 필요 대수, 결정 | |

## 판정 실행 (WP17.5)

| 명령 | 하는 일 | 종료 코드 |
| --- | --- | --- |
| `dlp aireview run <세션>… \| --pending [--limit N] [--parallel-sessions K]` | 판정 → AI 표 기록 (멱등·재개). `--groups`, `--mode all\|flagged`, `--max-minutes`, `--no-sensor`, `--dry-run`(계획만, 서버 불필요. 손 상태 대상이 있으면 장갑 Parquet은 읽는다), `--debug-dir`(형식이 맞지 않던 응답 원문만 로컬 파일), `--json-progress` | 0 / 1 세션 하나의 예상하지 못한 오류(다음 세션은 계속) / 2 주소 없음 / 3 서버 장애 / 5 설정 / 6 세션 전제·블러 변경·사용 중지 / 7 처리 중 / 8 시간 / 130 Ctrl-C |
| `dlp aireview status [<세션>… \| --pending \| --all] [--json]` | 판정률(판정된 대상/계획 대상), 단위·fallback, 판정 분포, 마지막 이벤트(마감 없이 끝났으면 "중단 추정"), 최근 처리량 기반 남은 시간, 백로그. 실행 중인지는 잠금을 잡지 않고 본다 | 0 / 1 없는 세션 |
| `dlp aireview show <unit_id> \| <세션>:<라벨>` | 감사 기록 (프레임 PTS·sha, 렌더 해시, 프롬프트 다이제스트, 판정, 응답 sha, 서버 모델, 엔드포인트 이름, 실행자) | 0 / 1 없음 |
| `dlp aireview waive <세션> --reason` | required 모드에서 그 세션의 AI 단계를 건너뛴다 (`waived` 이벤트) | 0 / 1 없음 |

- 운영 cron은 행동 단계 뒤에 잇는다: `dlp actions run S && dlp aireview run S`. 따라잡기는
  `dlp aireview run --pending --max-minutes 50`을 매시간 (prelabeled·승인·보류 아님이고, 마지막 라벨 기록
  뒤 `pending.quiet_minutes`(30분)가 지났고, 판정 없는 대상이 남았거나 지금 버전 최근 실행이 completed가
  아닌 세션만 고른다. 최근 시도가 장애·멈춤으로 끝난 세션은 뒤로 미룬다). 한 세션의 단위 몇 개만
  장애이거나 예상하지 못한 오류가 나도 다음 세션을 계속하고, `--limit N`은 실제로 실행한 세션만 센다
  (블러본 게이트에서 거부되는 세션, 실행 기록 전 입력 단계 오류(`[입력 오류]`, 손상된 장갑 Parquet·
  블러본 등, 종료 코드는 1)로 끝난 세션이 자리를 차지하지 않는다).
- `--parallel-sessions K`: 세션마다 DB 연결을 최대 2개 쥐므로 연결 풀을 K에 맞춰 키운다. K는
  PostgreSQL `max_connections`의 절반보다 작게 둔다. Ctrl-C는 함께 돌던 세션 모두를 진행 중 요청만
  끝내고 interrupted:signal로 닫는다.
- 작업자 수 = 엔드포인트 수 × `server.concurrency_per_endpoint`. 같은 창의 분할 요청은 같은
  엔드포인트로, 그곳이 상한까지 바쁘면 다른 곳으로 간다. 진행 줄(표준 오류)에 엔드포인트별 상태
  (ok·open·half-open)가 나온다.
- 서버가 멈추면: 차단기가 그 엔드포인트를 빼고, 모두 `abort_after_s`(15분)보다 오래 막히면 완료한 단위를
  남기고 종료 3이다. 서버가 돌아오면 같은 명령을 다시 돌린다 (남은 단위만 묻는다).
- 시작 전에 엔드포인트마다 `/v1/models`에 정책 모델(`served_name`)이 있는지 본다. 없으면 종료 5이고
  아무것도 쓰지 않는다.
- 장갑 접촉 문장은 장갑 Parquet을 감사 원본 저장소(용도 `aireview.run`)로 읽어 프리라벨과 같은 함수로
  계산한다. 선언된 장갑의 Parquet이 없으면 종료 6이고, `--no-sensor`로 센서 없이 판정할 수 있다.

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
