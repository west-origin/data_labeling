# ADR 0040: AI 1차 검수(로컬 VLM)는 참고용 판정만 남기고 사람 검수를 줄이지 않는다

- 상태: 채택 (단계 구현 중. WP17.0~17.5 구현(17.3은 Spark 실측 대기), WP17.6~17.9는 [구현 계획 WP17](../ai-implementation-plan.md#wp17-ai-1차-검수로컬-vlm))
- 날짜: 2026-10-11
- 관련: WP10(행동 VLM), WP6·WP12(검수 연동·운영), ADR 0010(상업 사용), ADR 0012(행동 2단 구조),
  ADR 0015·0019(검수된 라벨 보호, 시각 규약), ADR 0024(블러본만), ADR 0033(세대·과거 정책 복귀),
  ADR 0039(블러 변경 시 매체 회수).
  계약 추가: `dlp_schema.aireview`(판정·단위·실행·노출·결과), `ReviewReason`의 `ai_*` 4개,
  `schemas/ai_review_unit|ai_review_run|ai_exposure|ai_outcome.schema.json`,
  `schemas/review_assignment.schema.json`(사유 열거형). DB 표 5개는 Alembic 0019(down 0017, WP17.4),
  저장소는 `dlp_schema.db.aireview_repository`. 0018(골든 정답 고정, fix/m-trn)도 0017을 이으므로,
  둘 중 나중에 메인에 병합하는 쪽이 0019의 `down_revision`을 0018로 바꾸고 이 줄과 구현 계획의 "down
  0017"을 고친다 (head가 둘이면 `upgrade`가 실패한다. `test_migrations_have_single_head`가 잡는다).

## 배경

자동 라벨(프리라벨·관계·행동)은 사람이 모두 검수한다. 검수자가 먼저 볼 곳을 알면 검수가 빨라지고
놓치는 오류가 준다. 사내 GPU 장비에서 VLM이 **승인된 블러본**만 보고 검수 전 자동 라벨마다
"맞음/의심/틀림/판단 불가"와 이유, 온톨로지 안의 수정 제안을 남기면 그 신호가 된다. 그러나 AI 판정을
라벨이나 검수 상태로 다루면 판매 데이터·학습 데이터·검증 판정에 AI 결과가 섞이고, 자동화 편향으로
사람 검수가 사실상 줄어든다.

## 결정

### 1. 판정은 라벨도 검수도 아니다 (I3, I4)

- 판정은 `label_records`가 아니라 추가 전용 표(`ai_review_runs`, `ai_review_run_events`,
  `ai_review_units`, `ai_review_exposures`, `ai_review_outcomes`, WP17.4)에만 쓴다.
  - 버린 안: 제안을 원 라벨의 측정용 자식 레코드로 저장. `current_labels`는 superseded를 비운영 레코드
    까지 포함한 전체 이력에서 계산하므로 원 라벨이 운영 라벨에서 사라진다. `drop_retracted`는 자식이
    있는 라벨의 사람 결과를 버린다. 부모 없는 형제로 넣으면 운영 라벨이 둘이 되고, `measurement`는
    `Literal["blind","double"]`이라 계약도 바꿔야 한다. "맞음"에는 페이로드도 없다.
- 제안이 라벨이 되는 길은 검수 도구 안의 **사람의 명시적 적용 한 번**뿐이다 (WP17.7). 결과는 HUMAN 출처,
  human_corrected, 담당 검수자 레코드다. AI가 제안하고 사람이 적용했다는 사실은 라벨 계약에 넣지 않고
  `ai_review_outcomes`에만 남긴다 (결정: 내보내기·학습에서 구별하는 계약 변경은 하지 않는다).
  - 버린 안: 예측 미리 채우기·대체 예측으로 제안을 보이기. 지나치기만 해도 수락되어 자동화 편향이 된다.
- `dlp_aireview`는 라벨·검수·생애주기 쓰기 함수를 부르지 않는다 (정적 검사
  `tests/test_aireview_boundaries.py`). 내보내기·학습·데이터셋·액티브 러닝은 AI 계약을 import하지 않는다.

### 2. 입력은 승인된 블러본뿐이다 (I1, I2)

- 프레임은 라벨링 버킷의 지금 승인된 블러본에서만 얻는다. 원본 버킷 이름의 저장소(감사 원본 저장소
  포함)를 거부하고, 렌더 게이트(`expected_render_hash`·`assert_render_current`)와 받은 뒤 렌더 해시
  재확인을 거친다 (`dlp_aireview.media.open_blurred`, WP17.5). 이미지는 data URI로만 보내고 서명 URL을
  쓰지 않는다. 프레임과 블러본은 실행 임시 디렉터리에만 둔다.
- 블러 트랙은 판정하지 않는다 (정책 검증기·계약 검증기). 원본 접근 이벤트(불편 반응)는 대상·맥락에서
  뺀다.
- 장갑 접촉 문장은 CLI가 프리라벨과 같은 함수(`glove_contact_intervals`)에 같은 입력을 넣어 계산한다.
  장갑 Parquet은 감사 원본 저장소로 읽으며 용도 `aireview.run`을 `ops.yaml audit.service_purposes`에
  등록했다 (월간 감사가 이상 접근으로 표시하지 않게).
- 앞으로 원본 프레임에 VLM을 쓰려면 블러본 서버와 **다른 서버 인스턴스**와 별도 ADR이 필요하다.

### 3. 하드웨어·서빙·백엔드 (사용자 결정)

- 장비는 **NVIDIA DGX Spark**(GB10, Arm64 + CUDA, 통합 메모리 128 GB)로 정했다. 서빙 기준은 Spark의
  **vLLM**(OpenAI 호환 API, 구조화 출력 `response_format: json_schema`, 접두사 캐시)이다. 먼저 1대를
  사서 `dlp aireview bench`(WP17.3)로 프리필·디코드를 실측한 뒤 대수를 정한다
  (공식은 [구현 계획 WP17](../ai-implementation-plan.md#wp17-ai-1차-검수로컬-vlm)). 실행 안내는
  `services/vlm/`와 [VLM 서버 안내](../handover/vlm-server.md)다. Mac 서빙은 다루지 않는다.
- **외부 API(Claude 등)는 쓰지 않는다.** `server.allow_external`은 `false`만 로드된다. 백엔드는
  `dlp_aireview.backend.JudgeBackend` 프로토콜 하나로 바꿔 끼울 수 있게 두고, 지금 구현은 사내 OpenAI
  호환 서버(`OpenAICompatibleJudge`)와 CPU·CI용 stub(`OracleJudge`)뿐이다. 외부 백엔드를 더하려면 이
  ADR을 대체하는 결정과 코드 변경(검증기 해제, 사내 주소 검사 예외)이 필요하다.
- 판정 모델 기본값은 행동 분류(`actions.yaml vlm.model`, Qwen2.5-VL-7B)와 **다른 모델**이다
  (자기 검수 상관 오류 방지). 기본 후보는 `Qwen/Qwen3-VL-8B-Instruct`이며, 등록 전에 Hugging Face
  모델 카드(`api/models/<저장소>`의 `cardData.license = apache-2.0`, revision 0c351dd)로 라이선스를
  확인했다. 학습 데이터가 공개되지 않아 `commercial: review`다. 같은 방법으로 Qwen3-VL-30B-A3B,
  Qwen2.5-VL-32B, Mistral-Small-3.2-24B(모두 apache-2.0, review)를 후보로 등록했고, Qwen2.5-VL-3B는
  `qwen-research`(비상업)라 `forbidden`으로 남겼다. **최종 모델은 WP17.3 bench와 WP17.8 골든 평가로
  확정한다.** 크기 정렬 단위는 Qwen3-VL의 patch 16 × merge 2 = 32px이다.
- `config/models.yaml`과 `make licenses`가 판정 모델도 검사한다 (`used_models`의
  `aireview.backend.model`). 서버에 등록한 이름이 다르면 `backend.served_name`에 적고, 요청 본문
  `model`과 응답 대조 모두 그 이름을 쓴다 (`backend.model`은 라이선스·판정 버전 이름).
- 엔드포인트는 `DLP_VLM_URLS` 또는 `--vlm-url`이다. 행동 분류 서버 주소(`DLP_VLM_URL`)로 대신하지
  않는다 (판정 모델이 달라 늘 서버 모델 불일치가 난다). 429의 Retry-After는 정책 대기(`backoff_s`)의
  최댓값까지만 따른다.

### 4. 판정 버전·단위 키·멱등 (I8)

- 판정 버전 = `aireview-<digest>+vlm-<model>` (`AiReviewPolicy.verdict_version`). digest는
  `selection`·`kinds`·`pending`·`server`·`gate`·`estimate`를 뺀 파싱값의 sha256 앞 12자다. 그 절들은
  "어느 대상을 묻나·어떻게 돌리나"만 바꾸고 대상 집합은 단위 키가 담는다 (all↔flagged 전환이 재판정을
  부르지 않게). 그래서 그 절들은 **요청 본문을 바꾸지 않는다**: 프롬프트 고정 블록의 문제 유형 줄,
  응답 스키마의 문제 유형 열거, 맥락은 `kinds`가 아니라 판정할 수 있는 종류 전체(`TEMPORAL_KINDS`,
  `SPATIAL_KINDS`) 기준이다. 결과(미응답 대상)를 바꾸는 문맥 초과 분할 횟수는
  `response.max_context_splits`로 해시 절에 둔다.
- 프롬프트 틀(고정 문장·스키마 모양·가변 글 틀·재시도 글)은 정책 값이 아니라 `PROMPT_VERSION`
  (`aireview.yaml prompt_version`과 같아야 로드)으로 버전에 들어간다. 틀 바이트는
  `packages/aireview/tests/golden/`의 골든 파일이 고정하고, 틀이 바뀌면 테스트가 버전을 올리라고
  알린다 (같은 버전에 다른 프롬프트의 판정이 섞이지 않게).
- 단위 키 = sha256([버전, 세션, 묶음, 스트림, 증거 구간, 분할 순번, 대상 ID(순서), 맥락 ID(정렬), 프레임
  ms, 렌더 해시, 센서 해시, 온톨로지 버전])의 앞 24자. 라벨을 고치면 그 창만, 블러가 바뀌면 그 스트림만
  다시 판정한다. 문맥 초과 분할은 단위 안 일이라 키에 넣지 않는다 (계획 단위 하나 = 행 하나).
- 같은 버전·입력이면 재실행이 서버를 부르지 않는다 (WP17.5 `done_unit_ids`). 예전 정책 값으로 되돌리면
  그 버전의 기존 단위를 다시 쓴다 (ADR 0033과 같은 원칙). 판정은 라벨이 아니라 `retractions()`가 필요
  없고, 이전 버전 행은 남아 표시 허용 버전 목록으로 걸러진다.
- 재현은 다시 계산해서가 아니라 저장한 판정 JSON과 응답 sha로 보장한다. 응답 원문은 DB에 남기지
  않는다 (블러본은 글자 OCR 블러가 없어 원문에 명찰·호실 번호가 옮겨 적힐 수 있다). 디버그 원문은
  로컬 `--debug-dir`에만 쓴다.

### 5. 요청과 응답

- 묶음: temporal(바디캠 블러본, 마스터 ms 10초 격자, 짧은 대상은 중점 창 하나), temporal_long(상위
  구간 전부·긴 라벨·긴 객체 상태, 라벨 하나 = 한 단위), spatial(영상 스트림마다 스냅샷 창, 그 스트림
  PTS). spatial 프레임은 대상 트랙의 키프레임 시각에서 골라(가장 가까운 프레임) 대상마다 적어도 한
  프레임에 표식이 그려지게 하고, 정한 장수로 다 덮지 못하면 같은 구간의 다음 창으로 넘긴다.
- 대상은 검수 전 **모델 출처** 운영 라벨이다. **장갑 센서에서 나온 라벨은 판정하지 않는다**(결정,
  `dlp_aireview.plan.glove_derived`): 센서 출처 라벨, 그리고 프리라벨 접촉 단계의 장갑만 잡은 접촉
  (대상 미상)과 장갑 시각 + 영상 대상 융합 접촉이다. 접촉 단계는 모델 출처로 쓰므로 출처로는 가를 수
  없고, 장갑이 관여한 접촉을 증거 OBSERVED, 영상만 잡은 접촉을 INFERRED로 쓰는 규칙
  (`dlp_prelabel.runner.contact_label`, 양쪽 테스트가 고정)으로 가른다. 융합 접촉도 있고 없음과 경계가
  장갑 압력에서 나와 함께 뺀다 (그 영상 대상의 판정은 후속 결정 F-AIR-09). 빠진 라벨은 맥락으로는
  남는다. 손 상태를 만드는 다른 자동 단계가 생기면 영상 근거 라벨을 INFERRED로 써야 판정 대상이 된다.
- 넣지 않는 정보: 모델 버전, 신뢰도, 검수자, 우선순위 사유 (자동 라벨의 자신감에 끌리지 않게).
- 프롬프트: system(고정) → user 하나 [고정 온톨로지 블록(캐시 경계) → 이미지 → 가변 글] → 재시도는
  끝에만 덧붙인다. 고정 부분은 세션과 무관하게 바이트까지 같다. 응답 스키마도 세션·창과 무관하다
  (개체·라벨을 창 지역 번호로 가리킨다). portable 프로필은 서버마다 다른 키워드(범위·길이·minItems)를
  쓰지 않고 재검사로 맡긴다.
- 재검사: 번호(빠짐·중복·목록 밖), 항목(값·문제 유형·한글 이유·근거 프레임), 제안(종류별 키, 개체표,
  온톨로지, 헛제안, 타임라인 종류 삭제 금지). 서버가 구조화 출력을 강제해도 다시 검사하며 어떤 응답
  값(목록·객체·실수 등)도 예외를 내지 않는다. 문맥 초과로 프레임을 줄인 요청도 프레임 번호는 단위
  번호 그대로다(이미지 도장·프레임 지도·응답 f가 같은 번호). 누락 후보(기본 끔)는 값(원본 접근이 아닌
  이벤트·객체 클래스)·프레임·정규화 상자 범위를 검사하고, 같은 종류 라벨과 겹치면 버리며, 이유도
  안전 검사를 거친다. 빠진 대상과 위반 항목은 `unsure + response_invalid,
  answered=false`(시스템 오류를 모델 판단처럼 쓰지 않는다). 파싱 실패나 위반 비율이 높을 때만 다시
  묻고, 그 밖에는 부분 수용한다. 끝까지 형식이 맞지 않으면 fallback(모두 미응답)이지만 응답을 받은
  결과라 완료로 기록한다. 서버 장애·설정 오류는 기록하지 않는다 (장애를 unsure로 쓰면 같은 버전
  재실행이 건너뛰어 영구히 남는다).
- 이유 안전 검사: URL·@·숫자 4자리 이상·프레임 참조·금지어(외모·신원)·글자 수 초과면 문제 유형 기반
  고정 문장으로 바꾼다. 단위 기록의 오류 문장(추가 전용)에는 번호와 위반 종류만 남기고 모델이 쓴
  값·서버 오류 본문은 넣지 않는다 (재시도 프롬프트에만 상세 문장을 쓴다). 단위 기록의 `frames`와
  `prompt_digest`는 실제로 보낸 요청(문맥 초과로 나눈 요청들, 재시도 대화 포함)의 것이다.
- 제안은 완성 페이로드로 만들어 `check_label`을 통과해야 한다. 경계(시각)는 제안하지 않는다 (기준 문서:
  VLM은 경계를 정하지 않는다). gap→action의 새 action_id는 Label Studio 새 그리기와 같은
  `h-<손>-<시작>`이다 (AI 기원 표시가 판매 데이터에 섞이지 않게).

### 6. 사람 검수 연동 원칙 (WP17.6~17.7)

1. 채우는 내용(LS 예측, CVAT 트랙)은 원 자동 라벨 그대로다. AI는 **표시만** 한다. 아무것도 누르지 않고
   제출하면 결과는 AI가 없을 때와 같다.
2. 제안 적용은 명시 행동 한 번이다. 적용이 세션 단위 검증(상위 구간 자식 연결 등)에 걸리면 AI 적용만
   되돌리고(`invalid_now`) 수집은 계속한다 (AI는 흐름을 막지 않는다).
3. 노출 집단: 대조군(holdout) 비율 **0.1** (결정), 블라인드·오류 삽입·QA 작업은 표시하지 않고, 이중
   작업은 짝 표준 배정의 집단을 따른다. 편향 없는 품질 추정은 holdout으로만 낸다.
4. LS 설정의 AI 컨트롤은 모드와 무관하게 **늘 둔다** (빼면 기존 주석이 있는 프로젝트에서 LS가 설정
   갱신을 거부한다).
5. 유효한 판정은 두 프로필로 계산한다 (`dlp_schema.aireview.valid_verdicts`). 표시 프로필은 렌더
   해시까지 같아야 한다 (I10, fail-closed). 증가 프로필(표본 제외, 마무리 보호, 우선순위)은 렌더와
   무관하다: 다른 스트림의 블러 변경으로 세션이 AUTO_BLURRED가 돼도 보호가 꺼지지 않는다 (I9).
   `strict_context` 기본값은 **false**다 (결정): 맥락 변화로 낡을 수 있는 판정도 참고 표시와 검수
   증가에만 쓰여 해가 작고, true면 사람 검수가 진행될 때마다 재판정만 는다.

### 7. 운영 모드와 절차 (사용자 결정)

- `review.yaml ai_review.mode`(WP17.6)의 **코드 기본값은 off**, 장비가 오기 전 **저장소 설정값은
  advisory**다. off는 AI 표를 읽지 않고, advisory는 그 시점의 유효한 판정만 얹는다 (판정이 없으면 화면이
  지금과 같다).
- 파일럿(WP17.9: 실세션 10개 이상, holdout 포함) 뒤 **required**로 바꾸는 절차:
  1. 파일럿 지표(지적 정밀도, holdout 기준 층별 ok 놓침률 상한, 검수 시간 비교)를 이 ADR 후속 절과
     인수인계 상태표에 기록한다.
  2. `review.yaml ai_review.mode: required`, `max_wait_h: 24`로 바꾼다. 그러면 `dlp review plan`이 지금
     버전 completed 실행이 없는 세션의 배정을 미룬다.
  3. prelabeled 뒤 24시간이 지나면 `waived:timeout`을 남기고 진행한다. 운영자는
     `dlp aireview waive <세션> --reason`으로 즉시 건너뛸 수 있다. 서버가 없거나 고장이어도 검수가
     영구히 멈추지 않는다.
- 판정 대상 선택 기본값은 **전량(all)** 이다 (결정). 선별(`selection.mode: flagged`, 지적 단위 +
  감사 비율)은 정책으로 고를 수 있다.
- 골든 게이트(`dlp aireview eval --gate`, WP17.8)는 실패해도 **경고만** 한다. 차단 여부는 파일럿 뒤
  정한다.

### 8. AI 판정은 검증 상태를 바꾸지 않는다 (I9)

AI가 모두 ok여도 사람 검수 없이는 `dlp review verify`가 실패한다. AI의 ok는 아무것도 줄이지 않는다.
사람 검수 생략은 별도 ADR로 정한다. 근거는 holdout 기준 층별 ok 놓침률 상한, 연속 주 수, 표본 감사다.
코드에는 그런 경로를 두지 않는다.

## 불변 조건

| ID | 불변 조건 | 강제 수단 |
| --- | --- | --- |
| I1 | AI 입력 픽셀은 지금 승인된 블러본에서만 나온다 | `open_blurred`(원본 버킷 이름 거부, 렌더 게이트, 받은 뒤 렌더 재확인), 실행기 프레임 공급은 `BlurredFrame`만(`frames_for`), 픽셀 생성 위치·게이트 호출 정적 검사 |
| I2 | 블러 검수 단계에 AI를 쓰지 않는다 | 정책·계약 검증기(blur_track 금지), privacy 작업에 노출 없음 |
| I3 | AI 판정은 라벨도 검수 상태도 아니다 | 정적 검사(쓰기 함수·라벨 표 이름), `record_review`의 `aireview:` 검수자 거부(WP17.4), verify 회귀 |
| I4 | 판정·제안은 운영 라벨·학습·내보내기·스냅샷에 섞이지 않는다 | 별도 표, import 정적 검사, 격리 회귀 테스트(`tests/test_aireview_isolation.py`) |
| I5 | AI 표는 추가만 한다 | 0019 트리거(추가 전용·TRUNCATE 금지, WP17.4) |
| I6 | 제안은 사람이 적용할 때만 라벨이 되고, 내용은 DB에서 다시 읽어 검증한다 | `ai_collect.apply_decisions`·`settle`(WP17.7) |
| I7 | 사내 서버만, 서버 모델 = 정책 모델 | `allow_external: false`, 사내 주소 검사(probe·bench WP17.3, run WP17.5, 밖이면 요청 0건), 전송은 프록시 환경 변수를 따르지 않고 그 주소로 직접 연결(`OpenAICompatibleChat`, `trust_env=False`), 응답 `model` 대조(종료 5), `make licenses` |
| I8 | 재실행은 멱등이다 | 입력 전체를 담은 단위 키, 단위 안 문맥 분할, `done_unit_ids`, 세션 잠금, `ON CONFLICT DO NOTHING` |
| I9 | AI는 사람 검수를 줄이지 않는다 | 증가 프로필은 렌더와 무관, 생략 경로 없음 |
| I10 | 블러가 바뀌면 이전 블러본으로 만든 이유는 보이지 않는다 | 단위 키·표시 프로필의 렌더 해시, 실행 중·마감 전 렌더 재확인(aborted:render_changed), 회수 실행기 |

## 행동 단계와 공통 전송 (WP17.1)

- `dlp_models.vlm`이 OpenAI 호환 요청 본문·오류 분류·백오프를 맡는다. 행동 단계(`dlp_actions`)도 이것을
  쓰되 **요청 본문 바이트와 버전 문자열(`vlm-<model>`)은 그대로**다 (리팩토링 전 본문을 고정한
  `packages/actions/tests/test_vlm_body.py`). 행동 버전에는 요청 형식이 들어가지 않아 본문이 바뀌면
  버전이 그대로인 채 결과만 달라지기 때문이다.
- 4xx(설정 오류)는 이제 재시도하지 않고 `VlmRequestError`로 올라가며 `dlp actions run`은 종료 5로
  끝난다 (A-08의 4xx 부분). 트랜잭션이 되돌려져 결과는 그대로다.
- 하지 않은 것: 행동 단계의 메시지 순서 변경, 공용 프레임 추출기 적용, F-PAR-07. 셋 다 행동 버전을
  바꿔 검수 전 결과를 다시 만들게 하므로 `actions.yaml vlm.prompt_version` 같은 별도 결정이 필요하다.

## 점검·벤치·추정 (WP17.3)

- `dlp aireview probe`는 서버 성질(모델, 이미지 수 한도, `json_schema` 강제, 범위·길이 키워드,
  `cached_tokens`, 문맥 길이)을 실제 요청으로 확인하고 정책 조정안을 낸다. 사내 대역 밖 주소에는 요청을
  하나도 보내지 않는다.
- `dlp aireview bench`는 합성 장면(사람·글자 없음)을 실제 계획기·프롬프트·재검사로 판정해 장비 1대의
  무릎 지점(장애 0, p95 < 시간 초과인 최대 처리량)을 잰다. 격자에는 지금 정책 값(묶음별 프레임 수·긴
  변)이 늘 들어간다. 요청마다 고정 블록 끝에 측정 표식을 넣어 system만 접두사 캐시에 맞게 하고, 계획
  단위를 돌려 쓰더라도 이미지에 표식별 잡음 조각을 넣어 요청마다 새 이미지를 보낸다 (vLLM의 이미지
  해시 기반 전처리·인코더 캐시가 비전 인코딩을 건너뛰어 처리량을 부풀리지 않게). 운영보다 요청당
  계산이 조금 많으므로(온톨로지 블록 캐시, 같은 창 분할 요청의 이미지 공유를 빼므로) 보수적이다. 무릎
  동시성에서 프리필 용량(생성 1토큰)과 생성 용량(길이 고정 `min_tokens`·`ignore_eos`)을 따로 잰다.
  서버 장애는 다시 묻지 않고 오류로 센다. 실제 세션 블러본 입력은 `--source session:<id>`(WP17.5,
  `open_blurred` 게이트, 장갑 문장 없음, DB·라벨링 버킷 읽기만)다.
- `dlp aireview estimate`는 DB의 실제 세션으로 영상 1시간당 작업량을 세고, 벤치 보고서로
  `대수 = ceil(H × (1+ρ) × D / (86,400 × u))`를 낸다. T_s = 무릎 단위/s × t(벤치)/t(작업량),
  t = 새 프롬프트 토큰/프리필 용량 + 출력 토큰/생성 용량이다 (설계 초안의 `min(프리필 한계, 디코드
  한계)`는 혼합 실행의 관측 tok/s를 용량으로 잘못 써서, 묶이지 않은 자원이 T를 깎았다). 벤치와 추정은
  같은 근사 셈법(이미지 ⌈w/align⌉·⌈h/align⌉ + 확대 조각, 글 바이트/`text_bytes_per_token`)을 쓰고,
  보고서의 셈법 값이 다르면 작업량을 보고서 셈법으로 다시 센다. 고정 접두사는 서버가 캐시 적중을
  보고하지 않거나 0이면 양쪽에서 매번 계산한다고 본다. 용량 측정이 없으면 무릎 처리량을 보정 없이 쓰고
  경고한다. 행동 VLM 처리량은 벤치가 재지 않고 실측 값(`--actions-req-s`)을 받는다. 가정 값은
  `aireview.yaml estimate`(digest 제외)에 둔다.

## 실행 (WP17.5)

- `dlp aireview run <세션>… | --pending`(`dlp_aireview.runner`): 세션 잠금(세션 수준 advisory lock,
  종료 7) → 세션 상태 확인(사용 중지·승인 보류·승인 전이면 종료 6) → 판정할 창이 있는 스트림만
  `open_blurred` → 계획 → 기록된 단위 제외 → 생산자(스트림마다 순차 디코드)·작업자(엔드포인트 수 ×
  동시성)·기록자(묶어 쓰기, 렌더 재확인) → 마감 이벤트. VLM 호출 중에는 DB 트랜잭션을 열지 않는다.
- 장애: 요청 하나는 `backoff_s`대로 다시 묻고, 엔드포인트가 연속 장애면 차단기를 연다. 쉼 뒤
  `/v1/models` 확인을 통과하면 시험 요청 하나로 닫을지 정한다(모델 목록만 응답하고 채팅이 실패하는
  서버도 닫히지 않는다). 모든 엔드포인트가 `abort_after_s`보다 오래 막히면 새 요청을 멈추고
  interrupted:unavailable(종료 3). 재시도 끝에도 장애인 단위는 기록하지 않고 넘어간다(다음 실행이 다시
  묻는다). 설정 오류(4xx, 모델 불일치) aborted:config(종료 5), 실행 중 블러 변경
  aborted:render_changed(종료 6), `--max-minutes` interrupted:max_minutes(종료 8).
- 같은 계획이 이미 completed면 아무것도 쓰지 않는다. 단위는 모두 있는데 마감 이벤트가 없으면(강제
  종료) 서버를 부르지 않고 완료 기록만 남긴다.
- `--pending`은 prelabeled·승인·보류 아님이고, 마지막 라벨 기록 뒤 `pending.quiet_minutes`가 지났고,
  지금 버전·렌더의 단위에 든 적 없는 대상이 남은 세션을 prelabeled 전이 순으로 고른다(맥락 변화는
  보지 않는다). 운영 cron은 `dlp actions run S && dlp aireview run S`를 잇고 `--pending`은 따라잡기용이다.
- 원본 접근은 장갑 Parquet뿐이고 감사 원본 저장소 용도 `aireview.run`이다(손 상태 대상이 있고 센서를
  켰을 때만, 정적 검사). `dlp aireview status|show|waive`는 진행 상태, 단위 감사 기록, required 모드의
  건너뛰기 결정(`waived`, 사유 `manual: …`)이다.

## 범위 밖 (후속, `docs/handover/issues.yaml` F-AIR)

오류 삽입 사본 판정, 표적 QA 비율, AI 표 파티션·보관, 행동 VLM 처리량 개선(병렬·창 단위), CVAT 2단계
(트랙 속성 결정), 경계 이동 제안 적용, 누락 발견 표시, 장갑+영상 융합 접촉의 영상 대상 판정, 사람
생략 ADR.

## 결과

- 좋은 점: 사람 검수를 줄이지 않고 먼저 볼 곳과 수정 후보를 준다. 판정 품질을 holdout으로 편향 없이 잰다.
  판매·학습 데이터와 검증 판정에 AI 결과가 섞이지 않는다.
- 비용: 장비(Spark)와 서빙 운영, 판정 표 저장 공간, 검수 화면 표시 유지보수(LS·CVAT 업그레이드 때 확인).
- 위험: 행동 VLM이 후보 구간마다 직렬로 묻는 구조라 장비 대수를 정할 수 있다 (F-AIR-01, WP17.3에서 실측).
  자동화 편향은 holdout·표시 상한·"참고용" 문구로 감시하지만 없애지는 못한다.
