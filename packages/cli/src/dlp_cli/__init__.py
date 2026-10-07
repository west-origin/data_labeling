"""`dlp` 명령줄 도구 패키지 (`dlp_cli`).

플랫폼의 모든 파이프라인 단계를 하나의 `dlp` 명령으로 묶는 얇은 접착층이다 (WP0, 각 WP의 CLI
산출물). 각 하위 명령은 `<단계>_cmds.py` 모듈에 있고, 정책
YAML(`config/policies/*.yaml`)·저장소·DB 엔진을 엮어 해당 패키지(`dlp_media`, `dlp_sync`,
`dlp_privacy` …)의 실행 함수를 부를 뿐 로직은 두지 않는다.

주요 모듈:
- `main` — `argparse` 파서 조립과 `dlp` 진입점(`main`), `dlp services check`.
- `raw_access` — 원본 버킷 감사 저장소를 만드는 유일한 곳 (ADR 0020, 0021).
- `health` — 개발 서비스(docker compose) 헬스체크.
- `<단계>_cmds` — 단계별 하위 명령 등록(`add_commands`)과 실행 함수(`cmd_*`).

주의: 원본 버킷 저장소를 이 패키지의 다른 모듈에서 직접 만들면 정적 검사
(`tests/test_raw_access_audited.py`)가 실패한다. 반드시 `raw_access.raw_store`를 쓴다.
"""

# `dlp --version`이 출력하는 버전 문자열. 패키지 `pyproject.toml`의 version과 맞춰 둔다.
__version__ = "0.1.0"
