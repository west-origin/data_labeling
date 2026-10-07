"""운영 대시보드와 보안 (WP16): 주간 운영 지표, 원본 접근 감사 리포트, 원본 보관 만료 알림.

`dlp ops …` 명령(`dlp_cli.ops_cmds`)이 이 패키지의 함수를 부른다. 모두 DB를 읽기만 하며,
운영 기록(`review_work`, `privacy_audits`, `retention_decisions`)을 쓰는 일은 CLI가 한다(추가만).

모듈:
- `metrics` — 주간 운영 지표(`weekly_metrics`)와 경고(`alerts`), Markdown 표 (`dlp ops weekly`).
- `audit` — 원본 접근(`raw_access_log`) 월간 감사 리포트 (`dlp ops audit-report`, ADR 0020·0021).
- `retention` — 원본 보관 기간 만료 상태 계산 (`dlp ops retention`).
- `policy` — `config/policies/ops.yaml` 로더.

관련 ADR: 0020(운영 대시보드와 보안), 0021(내보내기·운영 감사 수정), 0029(검증 주 = 생애주기 기록).
"""
