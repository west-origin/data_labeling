"""운영 대시보드와 보안 하위 명령 (WP16, ADR 0020·0021).

등록하는 명령 (`dlp ops …`):
- `weekly [--week] [--weeks] [--out]` — 주간 운영 지표(검수 시간, 수정률, 자동 승인율, 프리라벨
  편향, 오류 삽입 발견율, 잔여 블러 누락률, 에피소드당 원가)와 경고. 최근 N주를 한 표로 낸다.
- `audit-report <YYYY-MM> [--fail-on-flags]` — 원본 접근(`raw_access_log`) 월간 감사 리포트.
  권한 없는 접근·업무 시간 밖 접근 같은 확인할 것(flags)을 표시한다.
- `retention [--today] [--all]` — 원본 보관 기간 만료 알림 (기간은 `defaults.yaml`
  `retention.raw_retention_days`, 아직 미정이면 알림만 없음).
- `retention-decide <세션> extend|delete [--until] --reason` — 보관 연장·삭제 결정을 기록한다.
  실제 삭제는 이 명령이 하지 않는다 (`RETENTION_NOTE` 참고).
- `log-work <세션> --reviewer --stage --minutes` — 도구가 재지 않는 검수 작업 시간을 손으로
  기록한다.
- `privacy-audit <세션> <스트림> --misses` — 잔여 블러 누락 감사 결과를 기록한다 (길이는 그
  스트림의 길이). 감사자는 그 스트림의 블러 검수자와 달라야 한다. 감사 표본은
  `dlp privacy audit-sample`로 뽑는다.

주기: `weekly`는 매주, `audit-report`는 매월, `retention`은 매주 정도 (저장소에 예약 실행 설정은
없다 — 운영자가 cron 등으로 건다).
운영·감사 기록 테이블(`review_work`, `privacy_audits`, `retention_decisions`)은 추가만 한다
(ADR 0020, 0021). 지표 계산 로직은 `dlp_ops` 패키지에 있다.

정책 출처: `config/policies/ops.yaml` (경고 임계값, 원가, 감사 규칙, 보관 알림 일수),
`config/policies/review.yaml`(`reviewers.privacy`: 원본 접근 권한자).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.audit import current_actor
from dlp_ops import audit as audit_mod
from dlp_ops import metrics as metrics_mod
from dlp_ops.policy import load_policy
from dlp_ops.retention import RETENTION_NOTE, retention_status
from dlp_privacy import audit as audit_privacy
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_privacy_audit,
    insert_retention_decision,
    insert_review_work,
)
from dlp_schema.ops import PrivacyAuditRecord, RetentionDecision, ReviewWork
from dlp_schema.session import StreamKind


def _engine(args: argparse.Namespace) -> sa.Engine:
    """`--url`(없으면 `DLP_DATABASE_URL`/기본값)로 엔진을 만든다. 호출자가 `dispose`한다."""
    return sa.create_engine(database_url(args.url))


def _write(out: str | None, text: str, data: object) -> None:
    """리포트를 파일로 쓰거나 표준 출력에 낸다.

    인자:
        out: 리포트 경로(보통 `.md`). None/빈 문자열이면 `text`만 표준 출력에 찍는다.
        text: 사람용 Markdown 본문.
        data: 기계용 데이터. `out`이 있으면 확장자를 `.json`으로 바꾼 옆 파일에 쓴다
            (`datetime` 등은 `default=str`로 문자열화).

    부작용: 상위 디렉터리를 만들고 두 파일을 덮어쓴다.
    """
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        path.with_suffix(".json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"리포트: {path}")
    else:
        print(text)


def cmd_weekly(args: argparse.Namespace) -> int:
    """`dlp ops weekly`: 최근 `--weeks`주(기본 8)의 운영 지표와 경고를 낸다. 읽기 전용.

    인자:
        args.week: 마지막 주 (ISO 주 `YYYY-Www`). None이면 지금(UTC)이 속한 주.
        args.weeks: 몇 주를 거슬러 볼지.
        args.out: 리포트 파일 경로 (없으면 표준 출력).

    경고(`alerts`)는 여러 주 추세를 보고 `ops.yaml`의 임계값으로 정한다.
    반환: 항상 0 (경고가 있어도 실패로 보지 않는다).
    """
    root = repo_root()
    policy, review = load_policy(root), load_review_policy(root)
    last = args.week or metrics_mod.week_of(datetime.now(UTC))
    start, _ = metrics_mod.week_range(last)
    weeks = [metrics_mod.week_of(start - timedelta(weeks=i)) for i in reversed(range(args.weeks))]
    engine = _engine(args)
    with engine.connect() as conn:
        history = [metrics_mod.weekly_metrics(conn, w, review, policy) for w in weeks]
    engine.dispose()
    warnings = metrics_mod.alerts(history, policy)
    _write(
        args.out,
        metrics_mod.markdown(history, warnings, policy.cost.currency),
        {"weeks": [m.as_dict() for m in history], "alerts": warnings},
    )
    return 0


def cmd_audit_report(args: argparse.Namespace) -> int:
    """`dlp ops audit-report <YYYY-MM>`: 원본 접근 감사 월간 리포트. 읽기 전용.

    반환: `--fail-on-flags`이고 확인할 것이 있으면 1, 그 밖에는 0 (CI·cron에서 알림용).
    """
    root = repo_root()
    policy, review = load_policy(root), load_review_policy(root)
    engine = _engine(args)
    with engine.connect() as conn:
        report = audit_mod.monthly_audit(conn, args.month, policy, review.reviewers.privacy)
    engine.dispose()
    _write(args.out, audit_mod.markdown(report), asdict(report))
    return 1 if report.flags and args.fail_on_flags else 0


def cmd_retention(args: argparse.Namespace) -> int:
    """`dlp ops retention`: 원본 보관 만료 상태를 세션별로 계산해 알린다. 읽기 전용.

    인자:
        args.today: 기준일 `YYYY-MM-DD`. 없으면 오늘(UTC 날짜).
        args.all: 참이면 알림이 없는 세션도 출력한다.

    보관 기간(`raw_retention_days`)이 None이면 안내를 출력하고, 만료 계산은 `retention_status`에
    맡긴다. 상태 집계와, 알림 대상 세션(`[상태] 세션 만료일 메모`)을 출력한다. 만료·사용 중지·삭제
    결정이 있으면 실제 삭제 절차 안내(`RETENTION_NOTE`)를 덧붙인다.
    """
    root = repo_root()
    days = load_config(root / "config/defaults.yaml").retention.raw_retention_days
    if days is None:
        print("원본 보관 기간이 정해지지 않았습니다 (defaults.yaml retention.raw_retention_days)")
    today = date.fromisoformat(args.today) if args.today else datetime.now(UTC).date()
    engine = _engine(args)
    with engine.connect() as conn:
        items = retention_status(conn, today, days, load_policy(root).retention.alert_days_before)
    engine.dispose()
    print(f"상태: {dict(Counter(i.status for i in items))}")
    for i in items:
        if i.alert or args.all:
            print(f"  [{i.status}] {i.session_id} 만료 {i.expires_on or '-'} {i.note}")
    if any(i.status in ("expired", "withdrawn", "delete_decided") for i in items):
        print(f"참고: {RETENTION_NOTE}")
    return 0


def cmd_retention_decide(args: argparse.Namespace) -> int:
    """`dlp ops retention-decide <세션> extend|delete`: 원본 보관 결정을 기록한다 (추가만).

    인자:
        args.decision: `extend`(연장, `--until` 필수) 또는 `delete`(삭제 결정, `--until` 없어야 함).
            조합이 틀리면 `RetentionDecision` 검증에서 `ValidationError`.
        args.until: 연장 기한 `YYYY-MM-DD`.
        args.reason: 결정 이유 (필수).

    결정자는 `current_actor()`(`DLP_ACTOR` 또는 OS 사용자)다.
    부작용: 세션이 있는지 확인한 뒤 `retention_decisions` INSERT. 원본은 지우지 않는다.
    """
    d = RetentionDecision(
        decision_id=uuid.uuid4().hex,
        session_id=args.session_id,
        decision=args.decision,
        until=date.fromisoformat(args.until) if args.until else None,
        reason=args.reason,
        decided_by=current_actor(),
        decided_at=datetime.now(UTC),
    )
    engine = _engine(args)
    with engine.begin() as conn:
        get_session(conn, d.session_id)  # 없으면 오류
        insert_retention_decision(conn, d)
    engine.dispose()
    print(f"{d.session_id}: {d.decision} {d.until or ''} ({d.decided_by})")
    return 0


def _aware(text: str) -> datetime:
    """ISO 시각. 시간대가 없으면 UTC로 본다.

    예: `2026-10-05T09:00` → `2026-10-05T09:00+00:00`. 형식이 틀리면 `ValueError`.
    """
    t = datetime.fromisoformat(text)
    return t if t.tzinfo is not None else t.replace(tzinfo=UTC)


def cmd_log_work(args: argparse.Namespace) -> int:
    # 검수한 시각(--at)의 주에 센다. 나중에 몰아서 기록해도 그 주의 지표에 들어간다
    """`dlp ops log-work <세션>`: 검수 작업 시간을 `review_work`에 기록한다 (추가만).

    인자:
        args.reviewer: 검수자 ID.
        args.stage: `privacy` / `labeling` / `qa`.
        args.minutes: 작업 시간(분, 실수). 초로 바꿔 `seconds`에 저장한다.
        args.video_ms: 검수한 영상 길이(ms). 없거나 0이면 세션 전체 길이.
        args.task_key: 검수 작업 키 (`cvat:<id>` / `label_studio:<id>`). 출처는 키 접두사의 도구,
            키가 없으면 `manual`이다 (`metrics.work_source`). 알 수 없는 접두사면 `SystemExit`.
        args.at: 검수한 시각 (ISO 8601). 그 시각이 속한 주의 지표에 들어간다. 미래면 `SystemExit`.

    부작용: 한 트랜잭션에서 `review_work` INSERT. 세션이 없으면 DB 조회 오류.
    """
    at = _aware(args.at) if args.at else datetime.now(UTC)
    if at > datetime.now(UTC):
        raise SystemExit(f"--at이 미래입니다: {at.isoformat()}")
    try:
        # 작업 키 접두사로 도구를 정한다 (Label Studio 작업을 CVAT로 세지 않는다)
        source = metrics_mod.work_source(args.task_key)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    engine = _engine(args)
    with engine.begin() as conn:
        session = get_session(conn, args.session_id)
        insert_review_work(
            conn,
            ReviewWork(
                work_id=uuid.uuid4().hex,
                task_key=args.task_key,
                session_id=session.session_id,
                reviewer=args.reviewer,
                stage=args.stage,
                seconds=args.minutes * 60,
                video_ms=args.video_ms or session.duration_ms,
                source=source,
                recorded_at=at,
            ),
        )
    engine.dispose()
    return 0


def cmd_privacy_audit(args: argparse.Namespace) -> int:
    """`dlp ops privacy-audit <세션> <스트림> --misses <N>`: 잔여 블러 누락 감사 결과를 기록한다.

    검사:
    - 그 스트림의 운영 현재 블러 트랙에 사람 검수자가 없으면 `SystemExit` (감사할 블러본이 없다).
    - 감사자(`--auditor`, 없으면 `current_actor()`)가 그 블러 검수자 중 하나면 `SystemExit`
      (독립 감사, `PrivacyAuditRecord`도 같은 규칙을 검증한다).

    기록: `blur_reviewer`에는 검수자 목록의 첫 사람만, `duration_ms`에는 감사한 그 스트림의 길이
    (`dlp_privacy.audit.stream_duration_ms`: 바디캠은 세션 길이, 3인칭 등은 그 스트림의 PTS
    인덱스)를 넣는다. 누락률은 시간당(누락 수 / 길이) 계산에 쓴다. 세션에 그 영상 스트림이 없으면
    `SystemExit`.
    부작용: 한 트랜잭션에서 `privacy_audits` INSERT (추가만). 바디캠이 아니면 원본 버킷에서 PTS
    인덱스를 읽는다 (`args.store`, 감사 저장소 → 읽기 기록).
    """
    engine = _engine(args)
    with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
        session = get_session(conn, args.session_id)
        stream = next((s for s in session.streams if s.stream_id == args.stream_id), None)
        if stream is None or stream.kind not in audit_privacy.VIDEO_KINDS:
            raise SystemExit(f"{args.session_id}/{args.stream_id}: 세션의 영상 스트림이 아닙니다")
        # 운영 현재 블러 트랙의 검수자 모두 (수정 이력 전체가 아니라 지금 렌더에 쓰인 트랙)
        reviewers = audit_mod.blur_reviewers(get_labels(conn, args.session_id), args.stream_id)
        if not reviewers:
            raise SystemExit(f"{args.session_id}/{args.stream_id}: 사람이 검수한 블러가 없습니다")
        auditor = args.auditor or current_actor()
        if auditor in reviewers:
            raise SystemExit(
                f"감사자 {auditor}는 이 스트림의 블러 검수자입니다 ({', '.join(reviewers)}). "
                "다른 사람이 감사합니다"
            )
        reviewer = reviewers[0]
        insert_privacy_audit(
            conn,
            PrivacyAuditRecord(
                audit_id=uuid.uuid4().hex,
                session_id=session.session_id,
                stream_id=args.stream_id,
                # 회귀: 세션(바디캠) 길이를 써서 길이가 다른 3인칭 스트림의 누락률이 틀렸다.
                # 원본 저장소는 바디캠이 아닐 때만 만든다 (바디캠은 세션 길이와 같다).
                duration_ms=audit_privacy.stream_duration_ms(
                    session,
                    stream,
                    None
                    if stream.kind is StreamKind.BODYCAM
                    else raw_store(args.store, args.url, "ops.privacy-audit"),
                    Path(tmp),
                ),
                misses=args.misses,
                auditor=auditor,
                blur_reviewer=reviewer,
                audited_at=datetime.now(UTC),
            ),
        )
    engine.dispose()
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`ops weekly|audit-report|retention|retention-decide|log-work|privacy-audit`을 등록한다."""
    ops = sub.add_parser("ops", help="운영 지표·원본 접근 감사·보관 기간")
    osub = ops.add_subparsers(dest="ops_command", required=True)
    w = osub.add_parser("weekly", help="주간 운영 지표 (최근 여러 주와 경고)")
    w.add_argument("--week", help="마지막 주 (예: 2026-W41, 기본: 이번 주)")
    w.add_argument("--weeks", type=int, default=8)
    w.add_argument("--out", help="리포트 파일 (.md, 옆에 .json)")
    w.add_argument("--url")
    w.set_defaults(func=cmd_weekly)
    a = osub.add_parser("audit-report", help="원본 접근 감사 월간 리포트")
    a.add_argument("month", help="예: 2026-10")
    a.add_argument("--out")
    a.add_argument("--fail-on-flags", action="store_true", help="확인할 것이 있으면 종료 코드 1")
    a.add_argument("--url")
    a.set_defaults(func=cmd_audit_report)
    r = osub.add_parser("retention", help="원본 보관 기간 만료 알림")
    r.add_argument("--today", help="기준일 (기본: 오늘, UTC)")
    r.add_argument("--all", action="store_true", help="알림 없는 세션도 보인다")
    r.add_argument("--url")
    r.set_defaults(func=cmd_retention)
    d = osub.add_parser("retention-decide", help="원본 보관 연장·삭제 결정 기록")
    d.add_argument("session_id")
    d.add_argument("decision", choices=["extend", "delete"])
    d.add_argument("--until", help="연장 기한 (YYYY-MM-DD, extend에 필수)")
    d.add_argument("--reason", required=True)
    d.add_argument("--url")
    d.set_defaults(func=cmd_retention_decide)
    lw = osub.add_parser("log-work", help="검수 작업 시간 기록 (CVAT 등 도구가 재지 않을 때)")
    lw.add_argument("session_id")
    lw.add_argument("--reviewer", required=True)
    lw.add_argument("--stage", choices=["privacy", "labeling", "qa"], required=True)
    lw.add_argument("--minutes", type=float, required=True)
    lw.add_argument("--video-ms", type=int, help="검수한 영상 길이 (기본: 세션 길이)")
    lw.add_argument("--task-key")
    lw.add_argument(
        "--at", help="검수한 시각 (ISO 8601, 시간대 없으면 UTC; 기본: 지금). 그 주의 지표에 센다"
    )
    lw.add_argument("--url")
    lw.set_defaults(func=cmd_log_work)
    pa = osub.add_parser("privacy-audit", help="잔여 블러 누락 감사 결과 기록")
    pa.add_argument("session_id")
    pa.add_argument("stream_id")
    pa.add_argument("--misses", type=int, required=True)
    pa.add_argument("--auditor", help="기본: DLP_ACTOR 또는 OS 사용자")
    pa.add_argument("--url")
    # 바디캠이 아닌 스트림은 그 스트림의 PTS 인덱스(원본 버킷)로 길이를 정한다
    pa.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    pa.set_defaults(func=cmd_privacy_audit)
