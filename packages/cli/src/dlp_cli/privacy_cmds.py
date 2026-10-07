"""프라이버시 게이트 하위 명령."""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_privacy.audit import (
    AuditResult,
    audit_candidates,
    iso_week,
    iso_week_bounds,
    previous_weeks,
    review_mode,
    select_audit_sample,
    weekly_miss_rates,
)
from dlp_privacy.detectors import build_detectors
from dlp_privacy.policy import load_policy
from dlp_privacy.runner import approve_session, detect_session, render_session
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import get_session, list_privacy_audits
from dlp_train.deployed import deployed_predictors
from dlp_train.policy import load_policy as load_training_policy
from dlp_train.trainers import LoadContext


def _engine(args: argparse.Namespace) -> sa.Engine:
    return sa.create_engine(database_url(args.url))


def cmd_detect(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    raw = raw_store(args.store, args.url, "privacy.detect")
    detectors, missing = build_detectors(policy, root)
    for name, reason in missing.items():
        print(f"[탐지기 없음] {name}: {reason}")
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    now = datetime.now(UTC)
    engine = _engine(args)
    with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
        # 게이트를 통과하고 사람이 승인한 재학습 블러 모델이 있으면 탐지기 결과와 합집합으로 쓴다
        ontology_version = get_session(conn, args.session_id).ontology_version or ""
        deployed = deployed_predictors(
            conn,
            store_from_spec(args.store, buckets.mlflow),
            load_training_policy(root),
            "privacy",
            LoadContext(now=now, ontology_version=ontology_version),
            Path(tmp),
            strict=True,
        )
        for note in deployed.notes:
            print(f"[재학습 모델] {note}")
        s = detect_session(
            conn, args.session_id, raw, detectors, missing, policy, now, extra=deployed.predictors
        )
    engine.dispose()
    for stream, n in s.detected.items():
        print(f"{stream}: 블러 트랙 {n}개")
    for stream in s.skipped:
        print(f"{stream}: 같은 모델 버전 결과가 있어 건너뜀")
    for target, reason in s.missing.items():
        print(f"[전수 검수 필요] {target}: 쓸 수 있는 탐지기가 없음 ({reason})")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    engine = _engine(args)
    with engine.begin() as conn:
        session = approve_session(conn, args.session_id)
    engine.dispose()
    print(f"{session.session_id}: 프라이버시 승인 ({session.lifecycle_state.value})")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    raw = raw_store(args.store, args.url, "privacy.render")
    labeling = store_from_spec(args.store, buckets.labeling)
    engine = _engine(args)
    with engine.begin() as conn:
        uris = render_session(conn, args.session_id, raw, labeling, load_policy(root))
    engine.dispose()
    for stream, uri in uris.items():
        print(f"{stream}: {uri}")
    return 0


def cmd_audit_sample(args: argparse.Namespace) -> int:
    """그 주 승인된 블러본에서 잔여 누락 감사 표본을 뽑고, 블러 검수 방식(전수/표본)을 판정한다."""
    root = repo_root()
    config = load_config(root / "config" / "defaults.yaml")
    exit_policy = config.privacy.full_review_exit
    week = args.week or iso_week(datetime.now(UTC))
    weeks = previous_weeks(week, exit_policy.weeks_below_target)
    engine = _engine(args)
    with engine.connect() as conn:
        candidates = audit_candidates(conn, week)
        start, _ = iso_week_bounds(weeks[0])
        _, end = iso_week_bounds(week)
        audits = [
            (a.audited_at, AuditResult(a.session_id, a.stream_id, a.duration_ms, a.misses,
                                       a.auditor, a.blur_reviewer))
            for a in list_privacy_audits(conn, start, end)
        ]  # fmt: skip
    engine.dispose()
    sample = select_audit_sample(candidates, exit_policy.audit_sample_ratio, week)
    print(f"{week}: 승인된 블러본 {len(candidates)}개 중 감사 표본 {len(sample)}개")
    for c in sample:
        print(f"  {c.session_id}/{c.stream_id} (원 검수자 {c.blur_reviewer}, 감사자는 다른 사람)")
    rates = weekly_miss_rates(audits, weeks)
    for w, r in zip(weeks, rates, strict=True):
        print(f"  {w}: " + ("감사 없음 (통과로 보지 않음)" if r is None else f"{r:.3f}/시간"))
    target = config.success_criteria.residual_blur_miss_per_hour_max
    mode = review_mode(rates, target, exit_policy.weeks_below_target)
    print(f"블러 검수 방식: {'표본' if mode == 'sampled' else '전수'} (목표 {target})")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    privacy = sub.add_parser("privacy", help="프라이버시 게이트")
    psub = privacy.add_subparsers(dest="privacy_command", required=True)
    for name, func, help_text in (
        ("detect", cmd_detect, "블러 대상 자동 탐지 → 블러 트랙 라벨 + 검수 우선 구간"),
        ("approve", cmd_approve, "모든 블러 라벨이 사람 검수를 거쳤으면 프라이버시 승인"),
        ("render", cmd_render, "승인된 세션의 블러본을 라벨링 버킷에 렌더"),
    ):
        p = psub.add_parser(name, help=help_text)
        p.add_argument("session_id")
        p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
        if name != "approve":
            p.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
        p.set_defaults(func=func)
    audit = psub.add_parser(
        "audit-sample", help="그 주 승인된 블러본의 잔여 누락 감사 표본과 블러 검수 방식(전수/표본)"
    )
    audit.add_argument("--week", help="ISO 주 (예: 2026-W41, 기본: 이번 주)")
    audit.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    audit.set_defaults(func=cmd_audit_sample)
