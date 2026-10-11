"""AI 1차 검수(로컬 VLM): 승인된 블러본 프레임으로 검수 전 자동 라벨을 판정한다 (WP17, ADR 0040).

사내 OpenAI 호환 서버(DGX Spark의 vLLM)의 VLM이 블러본만 보고, 검수 전 자동 라벨마다
맞음(ok)·의심(suspect)·틀림(wrong)·판단 불가(unsure)와 한국어 이유, 온톨로지 안의 수정 제안을
남긴다. 판정은 라벨도 검수 상태도 아니다. 라벨 이력 표 밖의 추가 전용 표에만 쓰고(WP17.4),
사람은 여전히 모든 라벨을 검수한다. 제안은 사람이 검수 도구에서 명시적으로 적용할 때만 사람
레코드가 된다. AI는 사람 검수를 늘리는 방향(우선순위, 표본 제외, 늦은 지적의 재검수)으로만 쓴다.

운영 순서: `dlp prelabel run` → `dlp relations run` → `dlp actions run` → `dlp aireview run` →
`dlp review plan` → 사람 검수.

모듈 (WP17.2 순수 핵심)
- `policy`: `config/policies/aireview.yaml` (판정 버전 `aireview-<digest>+vlm-<model>`).
- `plan`: 대상·맥락 고르기, 요청 창(temporal·temporal_long·spatial), 프레임 고르기, 단위 키.
- `prompt`: 백엔드 중립 프롬프트 (고정 접두사 = 캐시 경계, 가변 글).
- `schema`: 응답 JSON Schema(portable·full), 종류별 문제 유형·제안 키.
- `validate`: 응답 재검사·부분 수용·이유 안전 검사.
- `suggest`: 제안 → 완성 페이로드 + 온톨로지 검사.
- `judge`: 단위 판정 (재시도, 단위 안 문맥 초과 분할, 미응답 대체).
- `overlay`: 표식(set-of-marks)·시각 표기·확대 조각, 보낼 이미지 만들기.
- `sensors`: 장갑 접촉 문장 (계산은 CLI가 prelabel 함수로).
- `backend`: `JudgeBackend` 프로토콜, 사내 OpenAI 호환 서버(`OpenAICompatibleJudge`).
- `oracle`: CPU·CI용 정답 stub(`OracleJudge`). `perturb`: 통제된 오류 9종.

뒤 단계(WP17.3~): 블러본 경계(`media.open_blurred`), 실행기·재개·차단기(`runner`), 벤치·추정·점검,
골든 평가. 이 패키지는 라벨·검수 기록을 쓰지 않는다 (정적 검사, I3).
"""
