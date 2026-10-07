"""운영 대시보드와 보안 하위 명령 (WP16)."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.audit import current_actor
from dlp_ops import audit as audit_mod
from dlp_ops import metrics as metrics_mod
from dlp_ops.policy import load_policy
from dlp_ops.retention import retention_status
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_privacy_audit,
    insert_retention_decision,
    insert_review_work,
)
from dlp_schema.labels import VerificationState
from dlp_schema.ops import PrivacyAuditRecord, RetentionDecision, ReviewWork


def _engine(args: argparse.Namespace) -> sa.Engine:
    return sa.create_engine(database_url(args.url))


def _write(out: str | None, text: str, data: object) -> None:
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
    root = repo_root()
    policy, review = load_policy(root), load_review_policy(root)
    engine = _engine(args)
    with engine.connect() as conn:
        report = audit_mod.monthly_audit(conn, args.month, policy, review.reviewers.privacy)
    engine.dispose()
    _write(args.out, audit_mod.markdown(report), asdict(report))
    return 1 if report.flags and args.fail_on_flags else 0


def cmd_retention(args: argparse.Namespace) -> int:
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
    return 0


def cmd_retention_decide(args: argparse.Namespace) -> int:
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


def cmd_log_work(args: argparse.Namespace) -> int:
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
                source="manual" if args.task_key is None else "cvat",
                recorded_at=datetime.now(UTC),
            ),
        )
    engine.dispose()
    return 0


def cmd_privacy_audit(args: argparse.Namespace) -> int:
    engine = _engine(args)
    with engine.begin() as conn:
        session = get_session(conn, args.session_id)
        blur = [
            x
            for x in get_labels(conn, args.session_id, kinds=["blur_track"])
            if x.stream_id == args.stream_id
            and x.verification.state
            in (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)
            and x.verification.reviewer_id
        ]
        if not blur:
            raise SystemExit(f"{args.session_id}/{args.stream_id}: 사람이 검수한 블러가 없습니다")
        reviewer = Counter(x.verification.reviewer_id for x in blur).most_common(1)[0][0]
        assert reviewer is not None
        insert_privacy_audit(
            conn,
            PrivacyAuditRecord(
                audit_id=uuid.uuid4().hex,
                session_id=session.session_id,
                stream_id=args.stream_id,
                duration_ms=session.duration_ms,
                misses=args.misses,
                auditor=args.auditor or current_actor(),
                blur_reviewer=reviewer,
                audited_at=datetime.now(UTC),
            ),
        )
    engine.dispose()
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
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
    lw.add_argument("--url")
    lw.set_defaults(func=cmd_log_work)
    pa = osub.add_parser("privacy-audit", help="잔여 블러 누락 감사 결과 기록")
    pa.add_argument("session_id")
    pa.add_argument("stream_id")
    pa.add_argument("--misses", type=int, required=True)
    pa.add_argument("--auditor", help="기본: DLP_ACTOR 또는 OS 사용자")
    pa.add_argument("--url")
    pa.set_defaults(func=cmd_privacy_audit)
