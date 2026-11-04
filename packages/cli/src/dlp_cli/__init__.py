"""`dlp` 명령줄 도구 패키지 (`dlp_cli`).

플랫폼의 모든 파이프라인 단계를 하나의 `dlp` 명령으로 묶는 얇은 접착층이다 (WP0, 각 WP의 CLI
산출물). 각 하위 명령은 `<단계>_cmds.py` 모듈에 있고, 정책
YAML(`config/policies/*.yaml`)·저장소·DB 엔진을 엮어 해당 패키지(`dlp_media`, `dlp_sync`,
`dlp_privacy` …)의 실행 함수를 부를 뿐 로직은 두지 않는다.

주요 모듈:
- `main` — `argparse` 파서 조립과 `dlp` 진입점(`main`), `dlp services check`.
- `raw_access` — 원본 버킷 감사 저장소를 만드는 유일한 곳 (ADR 0020, 0021).
- `health` — 개발 서비스(docker compose) 헬스체크.
- `<단계>_cmds` — 단계별 하위 명령 등록(`add_commands`)과 실행 함수(`cmd_*`). 각 모듈 docstring에
  하위 명령마다 인자, 종료 코드와 뜻, 부르는 패키지 함수, 읽는 정책 파일, 원본 버킷 접근 여부를
  적는다.
- `review_config` — 검수 정책 선택(`DLP_REVIEW_POLICY_FILE`), 검수 도구 설정 묶음, 매체 회수 실행과
  회수 대기 종료 코드(`REVOCATION_PENDING_EXIT` 4).
- `ontology_access` — 세션·데이터셋 버전의 DB 등록 온톨로지 사전 읽기.
- `aireview_common`, `ai_quality_print` — AI 1차 검수 명령의 공용 도우미와 품질 보고 출력.

종료 코드 관례: 0은 성공이다. argparse 사용 오류는 2, `SystemExit("메시지")`와 잡지 않은 예외는
1이다. 그 밖의 값(3, 4, 5, 6 …)은 명령마다 뜻이 다르므로 각 모듈 docstring을 본다. 예외적으로 몇
명령은 입력 거부에 2를 쓴다(`dlp calib …`, `dlp export`, `dlp console serve`, VLM 주소 없음).

주의: 원본 버킷 저장소를 이 패키지의 다른 모듈에서 직접 만들면 정적 검사
(`tests/test_raw_access_audited.py`)가 실패한다. 반드시 `raw_access.raw_store`를 쓴다.
"""

# `dlp --version`이 출력하는 버전 문자열. 패키지 `pyproject.toml`의 version과 맞춰 둔다.
__version__ = "0.1.0"
