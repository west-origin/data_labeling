"""원본 접근 감사 월간 리포트 (`dlp ops audit-report`, WP16, ADR 0020·0021).

한 달의 원본 접근 기록을 사람·용도·동작별로 세고, 다음을 표시한다.
- 원본 열람 권한이 없는 사람의 열람·서명 URL
  (권한자 = review.yaml reviewers.privacy + ops.yaml raw_viewers, 서비스 계정은 파이프라인)
- 원본 접근 권한이 없는 사람에게 보여 준 것 (grant), 담당자 없이 올린 원본 검수 작업
- 업무 시간 밖(ops.yaml off_hours) 사람의 원본 접근
- 서비스 계정을 파이프라인 용도(ops.yaml audit.service_purposes) 밖에서 쓴 접근 (사람이 서비스
  계정 이름으로 실행했을 수 있다)

달(month)의 경계는 정책 시간대(ops.yaml audit.timezone)의 그 달 1일 0시다
(업무 시간 판정과 같은 시간대).

입력: DB `raw_access_log` (`dlp_cli.raw_access`의 감사 저장소가 접근 **전에** 남긴 기록).
`action` 값: `read`(내용 읽기), `presign`(서명 URL 발급), `write`(쓰기), `grant`(검수 도구에 원본을
보여 줌 — 이때 `actor`는 보게 될 담당자, 없으면 `unassigned`).

공개 함수:
- `month_range(month, tz)` — 달의 [시작, 끝) UTC 경계.
- `blur_reviewers(history, stream_id)` — 스트림의 현재 블러 트랙을 사람 검수한 사람들
  (`dlp ops privacy-audit`의 독립성 검사에 쓴다).
- `audit_report(events, …)` — 순수 함수. 이벤트 목록 → `AuditReport` (테스트하기 쉬운 핵심).
- `monthly_audit(conn, …)` — DB에서 그 달 이벤트를 읽어 `audit_report`를 부른다.
- `markdown(report)` — 사람용 Markdown.

한계: 실행자(actor)는 DLP_ACTOR(없으면 OS 사용자)로 스스로 밝힌 값이다. 서비스 계정 이름을
사람이 쓰거나 다른 사람 이름을 쓰는 것을 이 기록만으로 막지 못한다. 원본 버킷 자격 증명을
서비스 계정·권한자에게만 나눠 주는 것(저장소 접근 제어)이 1차 통제이고, 이 리포트는 사후
확인이다 (ADR 0021).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from zoneinfo import ZoneInfo

import sqlalchemy as sa

from dlp_ops.policy import OpsPolicy
from dlp_schema.db.repository import list_raw_access
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.ops import RawAccessEvent


def month_range(month: str, tz: tzinfo = UTC) -> tuple[datetime, datetime]:
    """'2026-10' → (그 달 1일 0시, 다음 달 1일 0시), tz 시간대 기준 (UTC로 바꿔 돌려준다).

    반환 구간은 [시작, 끝) 반열림이다. 12월이면 다음 해 1월 1일이 끝이다.
    예외: `YYYY-MM` 형식이 아니면 `ValueError`.
    """
    y, m = (int(x) for x in month.split("-"))
    start = datetime(y, m, 1, tzinfo=tz)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=tz)
    return start.astimezone(UTC), end.astimezone(UTC)


def blur_reviewers(history: list[LabelRecord], stream_id: str) -> list[str]:
    """스트림의 운영 현재 블러 트랙을 사람 검수한 사람들 (트랙 수가 많은 순, 같으면 이름순).

    잔여 블러 누락 감사자는 이 모두와 달라야 한다 (감사 기록에는 첫 사람을 남긴다).

    인자:
        history: 세션의 전체 라벨 이력 (`get_labels`). 여기서 `current_labels`로 운영 현재
            라벨만 고른다 (수정 전 레코드·오류 삽입·측정 레코드 제외).
        stream_id: 대상 영상 스트림.

    반환: 승인(`human_approved`)·수정(`human_corrected`) 상태 블러 트랙의 `reviewer_id` 목록
    (표본 검증·미검수는 세지 않는다). 없으면 빈 목록.
    """
    reviewed = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)
    who = Counter(
        x.verification.reviewer_id
        for x in current_labels(history)
        if x.kind == "blur_track"
        and x.stream_id == stream_id
        and x.verification.state in reviewed
        and x.verification.reviewer_id
    )
    return [r for r, _ in sorted(who.items(), key=lambda kv: (-kv[1], kv[0])) if r]


@dataclass
class AuditReport:
    """월간 원본 접근 감사 결과. `dataclasses.asdict`로 JSON 리포트가 된다."""

    month: str
    events: int
    by_actor: dict[str, int] = field(default_factory=dict[str, int])
    by_purpose: dict[str, int] = field(default_factory=dict[str, int])
    by_action: dict[str, int] = field(default_factory=dict[str, int])
    sessions: int = 0
    flags: list[str] = field(default_factory=list[str])


def audit_report(
    events: list[RawAccessEvent], month: str, policy: OpsPolicy, privacy_reviewers: tuple[str, ...]
) -> AuditReport:
    """원본 접근 이벤트 목록에서 월간 감사 리포트를 만든다 (DB 없이 쓰는 순수 함수).

    인자:
        events: 그 달의 `RawAccessEvent` 목록 (`at`은 시간대 있는 datetime).
        month: 리포트에 적을 달 `YYYY-MM` (필터링은 호출자가 이미 했다).
        policy: `ops.yaml` (`audit.raw_viewers`, `service_accounts`, `service_purposes`, `timezone`,
            `off_hours`).
        privacy_reviewers: `review.yaml reviewers.privacy` (원본 접근 권한자).

    판정 규칙:
    - `grant`: 담당자가 `unassigned`이거나 권한자가 아니면 표시. 다른 규칙은 보지 않는다.
    - 사람(서비스 계정이 아닌 actor)의 `read`·`presign`은 권한자가 아니면 표시.
    - 서비스 계정은 용도(`purpose`, 예: `privacy.detect`)가 `service_purposes`에 없으면 표시.
    - 사람의 접근이 정책 시간대에서 업무 시간 밖이면 표시. `off_hours = (시작, 끝)` 시(hour)이며,
      시작 > 끝이면 자정을 넘는 구간(예: 22~6시)이다.

    반환: 집계(사람·용도·동작별 건수, 세션 수)와 `flags`(사람이 확인할 문장 목록)를 담은
    `AuditReport`.
    """
    viewers = set(privacy_reviewers) | set(policy.audit.raw_viewers)
    services = set(policy.audit.service_accounts)
    service_purposes = set(policy.audit.service_purposes)
    tz = ZoneInfo(policy.audit.timezone)
    off_start, off_end = policy.audit.off_hours
    report = AuditReport(month, len(events))
    report.by_actor = dict(sorted(Counter(e.actor for e in events).items()))
    report.by_purpose = dict(sorted(Counter(e.purpose for e in events).items()))
    report.by_action = dict(sorted(Counter(e.action for e in events).items()))
    report.sessions = len({e.session_id for e in events if e.session_id})
    for e in events:
        # 서비스 계정 목록에 없는 실행자는 사람으로 본다
        # (실행자 이름은 자기 신고라는 한계는 모듈 docstring 참고)
        human = e.actor not in services
        where = f"{e.at.isoformat()} {e.actor} {e.action} {e.key} ({e.purpose})"
        if e.action == "grant":
            if e.actor == "unassigned":
                report.flags.append(f"담당자 없이 원본 검수 작업을 올림: {where}")
            elif e.actor not in viewers:
                report.flags.append(f"원본 권한 없는 사람에게 보여 줌: {where}")
            continue
        if human and e.action in ("read", "presign") and e.actor not in viewers:
            report.flags.append(f"원본 권한 없는 사람의 열람: {where}")
        if not human and e.purpose not in service_purposes:
            report.flags.append(f"서비스 계정을 파이프라인 밖 용도로 씀: {where}")
        # 정책 시간대의 시(hour)로 판정한다. 시작 > 끝(예: 22, 6)이면 자정을 넘는 구간이라
        # "시작 이후 또는 끝 이전", 아니면 [시작, 끝) 안이다
        hour = e.at.astimezone(tz).hour
        off = (
            hour >= off_start or hour < off_end
            if off_start > off_end
            else off_start <= hour < off_end
        )
        if human and off:
            report.flags.append(f"업무 시간 밖 원본 접근: {where}")
    return report


def monthly_audit(
    conn: sa.Connection, month: str, policy: OpsPolicy, privacy_reviewers: tuple[str, ...]
) -> AuditReport:
    """DB `raw_access_log`에서 그 달(정책 시간대 기준) 이벤트를 읽어 `audit_report`를 만든다. 읽기
    전용."""
    start, end = month_range(month, ZoneInfo(policy.audit.timezone))
    return audit_report(list_raw_access(conn, start, end), month, policy, privacy_reviewers)


def markdown(r: AuditReport) -> str:
    """`AuditReport` → 사람용 Markdown (집계 3절 + "확인할 것"). 확인할 것이 없으면 "- 없음"."""
    lines = [f"# 원본 접근 감사 {r.month}", "", f"접근 {r.events}건, 세션 {r.sessions}개", ""]
    for title, counts in (("사람·계정", r.by_actor), ("용도", r.by_purpose), ("동작", r.by_action)):
        lines += [f"## {title}", ""] + [f"- {k}: {v}" for k, v in counts.items()] + [""]
    lines += ["## 확인할 것", ""] + ([f"- {f}" for f in r.flags] or ["- 없음"])
    return "\n".join(lines) + "\n"
