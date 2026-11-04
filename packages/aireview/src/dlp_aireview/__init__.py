"""AI 1차 검수(로컬 VLM): 승인된 블러본 프레임으로 검수 전 자동 라벨을 판정한다 (WP17, ADR 0040).

사내 OpenAI 호환 서버(DGX Spark의 vLLM)의 VLM이 블러본만 보고, 검수 전 자동 라벨마다
맞음(ok)·의심(suspect)·틀림(wrong)·판단 불가(unsure)와 한국어 이유, 온톨로지 안의 수정 제안을
남긴다. 사람은 여전히 모든 라벨을 검수한다.

판정은 참고용이다 (ADR 0040 "불변 조건" 표의 I1~I10)
- 판정은 라벨도 검수 상태도 아니다 (I3). 이 패키지는 라벨 이력 표와 검수 기록에 쓰지 않고, 추가 전용
  `ai_review_*` 표만 쓴다. 이 패키지가 쓰는 표는 `ai_review_runs`·`ai_review_run_events`·
  `ai_review_units`이고 `runner`만 쓴다. `ai_review_exposures`·`ai_review_outcomes`(검수자에게
  보인 판정과 그 처리)는 검수 패키지(`dlp_review`)가 쓴다. 정적 검사
  `tests/test_aireview_boundaries.py`가 쓰기 함수·라벨 표 이름을 소스(주석 포함)에서 막는다.
- 판정·제안은 운영 라벨·학습·내보내기에 섞이지 않는다 (I4). 내보내기·학습·데이터셋·액티브 러닝
  패키지는 이 패키지를 import하지 않는다 (같은 정적 검사).
- 입력 픽셀은 `media.open_blurred`가 받은 지금 승인된 블러본에서만 나온다 (I1). 다른 모듈은 저장소
  파일을 받거나 영상을 디코드하지 않는다. 블러 트랙은 판정하지 않는다 (I2, 정책 검증기).
- 판정 서버는 사내 서버만이다 (I7). `server.allow_external`은 false만 허용하고, 보내기 전에
  `backend.check_hosts`가 주소를 사내 대역(`server.internal_hosts`)과 대조한다. 응답 `model`이
  정책 모델과 다르면 설정 오류로 멈춘다.
- AI의 ok는 사람 검수·표본을 줄이지 않는다 (I9). AI는 사람 검수를 늘리는 방향(우선순위, 표본
  제외, 늦은 지적의 재검수)으로만 쓴다. 제안은 검수자가 검수 도구에서 명시적으로 적용할 때만 사람
  레코드가 되고, 그때 검수 패키지가 제안 내용을 DB에서 다시 읽어 검증한다 (I6,
  `dlp_review.ai_collect`).

판정 버전과 재실행
- 판정 버전은 `aireview-<digest>+vlm-<model>`이다 (`policy.AiReviewPolicy.verdict_version`).
  `digest`는 요청 본문·결과를 바꾸는 정책 값의 해시이고, `PROMPT_VERSION`(프롬프트 틀·이미지
  그리기)과 `backend`의 model·served_name·fingerprint가 들어간다. 대상 고르기·서버 연결 절은
  빠진다 (`policy` 모듈 docstring).
- 단위 키는 판정 입력(대상·맥락 ID·프레임·렌더 해시·센서·온톨로지)의 해시다. 같은 버전·입력의
  단위는 다시 묻지 않는다 (I8). 블러가 바뀌면 렌더 해시가 달라져 새 단위가 된다 (I10).

판정단 (ADR 0063)
- 판정 모델은 기본 `backend.model` 하나다. `backend.panel`에 모델을 더하면
  `dlp aireview run --model <이름>`으로 고른다 (`policy.select_model`). 모델마다 판정 버전이 달라
  `ai_review_*`에 따로 쌓이고, 판정단 목록 자체는 digest에 들지 않는다.
- 판정단 모델은 같은 블러본·같은 요청 본문으로 각자 판정하고 서로의 판정을 보지 않는다 (병렬, 직렬
  아님). 판정 사이 불일치(ok와 ok 아님)는 검수 패키지가 우선순위 사유 `ai_disagreement`로 쓴다
  (`review.yaml ai_review.disagreement`, 기본 끔).
- 판정단 모델도 `config/models.yaml` 라이선스 검사를 받는다 (`policy.load_policy`). 지금 후보 둘은
  선택지 확률 API라 OpenAI 호환 백엔드로 부를 수 없다 (F-AIR-37).

운영 순서: `dlp prelabel run` → `dlp relations run` → `dlp actions run` → `dlp aireview run` →
`dlp review plan` → 사람 검수.

모듈 (WP17.2 순수 핵심)
- `policy`: `config/policies/aireview.yaml`, 판정 버전, 판정단 고르기.
- `plan`: 대상·맥락 고르기, 요청 창(temporal·temporal_long·spatial), 프레임 고르기, 단위 키.
- `tracks`: 공간 트랙의 시각별 모양 (키프레임 선형 보간, `plan`과 `overlay`가 함께 쓴다).
- `prompt`: 백엔드 중립 프롬프트 (고정 접두사 = 캐시 경계, 가변 글).
- `schema`: 응답 JSON Schema(portable·full), 종류별 문제 유형·제안 키.
- `validate`: 응답 재검사·부분 수용·이유 안전 검사.
- `suggest`: 제안 → 완성 페이로드 + 온톨로지 검사.
- `judge`: 단위 판정 (재시도, 단위 안 문맥 초과 분할, 미응답 대체).
- `overlay`: 표식(set-of-marks)·시각 표기·확대 조각, 보낼 이미지 만들기.
- `sensors`: 장갑 접촉 문장 (계산은 CLI가 prelabel 함수로).
- `backend`: `JudgeBackend` 프로토콜, 사내 OpenAI 호환 서버(`OpenAICompatibleJudge`), 사내 주소
  검사(`check_hosts`).
- `oracle`: CPU·CI용 정답 stub(`OracleJudge`). `perturb`: 통제된 오류 9종.

모듈 (WP17.3 점검·벤치·추정, DB에 쓰지 않는다)
- `probe`: 서버 점검 (`dlp aireview probe`). `bench`: 장비 1대 처리량 (`dlp aireview bench`).
- `estimate`: 작업량·장비 대수 (`dlp aireview estimate`, 순수 함수).
- `testing`: 가짜 OpenAI 호환 서버(`FakeOpenAIServer`, 127.0.0.1, 테스트·CI용).

모듈 (WP17.5 실행)
- `media`: 블러본 경계 (`open_blurred`, `frames_for`, `BlurredFrame`). 이 패키지에서 픽셀을 얻는
  유일한 길이다.
- `backend.EndpointPool`: 여러 엔드포인트 묶음·백오프·차단기 (모두 막히면 `AllEndpointsDownError`).
- `runner`: 생산자·작업자·기록자 실행기, 재개·멱등, 렌더 재확인, 세션 잠금 (`run_session`,
  `run_from_db`, `DbSink`, `MemorySink`).
- `status`: 판정 범위·진행 상태와 `--pending` 세션 고르기 (DB 읽기만).

그 밖의 모듈
- `evaluate`: 골든셋 판정 정확도 (`dlp aireview eval`, DB에 쓰지 않는다).
- `crops`: 잘라 낸 블러·도구 박스의 오탐 의심 힌트 (`dlp review hints --ai`, ADR 0056). 힌트
  파일에만 실린다.
"""
