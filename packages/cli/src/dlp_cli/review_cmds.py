"""검수 도구 연동 하위 명령."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import S3Store
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.ops.assign import plan_qa
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.ops.runner import create_assignment_tasks, plan_session, quality_report
from dlp_review.tasks import (
    PRIVACY_PROJECT,
    SPATIAL_PROJECT,
    TEMPORAL_PROJECT,
    ReviewSetup,
    TaskError,
    create_labeling_tasks,
    create_privacy_tasks,
)
from dlp_review.webhook import CollectRequest, ReviewerMismatchError, resolve_reviewer, serve
from dlp_schema import load_config, load_ontology, repo_root
from dlp_schema.db.repository import (
    get_assignment,
    get_golden_set,
    get_review_task,
    insert_assignment,
    list_assignments,
)
from dlp_schema.review import AssignmentStatus


def _setup(args: argparse.Namespace) -> ReviewSetup:
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    return ReviewSetup(
        raw=raw_store("s3", getattr(args, "url", None), f"review.{args.review_command}"),
        labeling=S3Store.from_env(buckets.labeling),
        labeling_reader=S3Store.labeler_from_env(buckets.labeling),
        ontology=load_ontology(root / "config" / "ontology" / "v1"),
        cvat=None if getattr(args, "no_cvat", False) else CvatClient.from_env(),
        label_studio=None if getattr(args, "no_ls", False) else LabelStudioClient.from_env(),
    )


def _secret() -> str:
    secret = os.environ.get("DLP_WEBHOOK_SECRET")
    if not secret:
        raise SystemExit("DLP_WEBHOOK_SECRET를 설정하세요")
    return secret


def cmd_create(args: argparse.Namespace) -> int:
    setup = _setup(args)
    engine = sa.create_engine(database_url(args.url))
    now = datetime.now(UTC)
    if not args.assignee:
        raise SystemExit("검수 작업에는 --assignee가 필요합니다 (블러 검수는 원본 접근 권한자)")
    with engine.begin() as conn:
        if args.stage == "privacy":
            # 담당자가 review.yaml reviewers.privacy에 없으면 원본을 올리기 전에 막는다
            tasks = create_privacy_tasks(
                conn,
                args.session_id,
                setup,
                now,
                assignee=args.assignee,
                privacy_reviewers=load_ops_policy(repo_root()).reviewers.privacy,
            )
        else:
            tasks = create_labeling_tasks(conn, args.session_id, setup, args.assignee, now)
    engine.dispose()
    for t in tasks:
        print(f"{t.task_key:<20} {t.stage.value:<9} {t.stream_id:<14} {t.media_uri}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        outcome = collect_task(
            conn, args.task_key, _setup(args), args.reviewer, datetime.now(UTC),
            load_ops_policy(repo_root()),
        )  # fmt: skip
    engine.dispose()
    if outcome is None:
        print(f"{args.task_key}: 이미 수집함")
    else:
        print(
            f"{args.task_key}: 승인 {len(outcome.approved)}, 수정 {outcome.corrected}, "
            f"삭제 {outcome.retracted}, 추가 {outcome.added}"
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    setup = _setup(args)
    engine = sa.create_engine(database_url(args.url))

    service_user = setup.label_studio.service_user_id() if setup.label_studio else None
    ops_policy = load_ops_policy(repo_root())

    def on_collect(req: CollectRequest) -> None:
        with engine.begin() as conn:
            task = get_review_task(conn, req.task_key)
            try:
                # 검수자는 웹훅의 도구 사용자 ID가 아니라 작업 담당자다
                reviewer = resolve_reviewer(req, task, service_user, ops_policy.cvat.users)
            except ReviewerMismatchError as exc:
                print(f"{req.task_key}: 수집하지 않음 ({exc})")
                return
            if reviewer is None:
                print(f"{req.task_key}: 서비스 계정의 주석이라 수집하지 않음")
                return
            try:
                outcome = collect_task(
                    conn, req.task_key, setup, reviewer, datetime.now(UTC), ops_policy
                )
            except TaskError as exc:
                print(f"{req.task_key}: 수집하지 않음 ({exc})")
                return
        print(f"{req.task_key} ← {reviewer}: {'이미 수집함' if outcome is None else '수집함'}")

    server = serve(args.port, _secret(), on_collect)
    print(f"웹훅 대기: http://0.0.0.0:{args.port}/webhooks/{{cvat,label_studio}}")
    server.serve_forever()
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    setup = _setup(args)
    secret = _secret()
    if setup.cvat is not None:
        for name in (PRIVACY_PROJECT, SPATIAL_PROJECT):
            project = setup.cvat.find_project(name)
            if project:
                setup.cvat.add_webhook(int(project["id"]), f"{args.url_base}/webhooks/cvat", secret)
                print(f"CVAT {name}: 웹훅 등록")
    if setup.label_studio is not None:
        project = setup.label_studio.find_project(TEMPORAL_PROJECT)
        if project:
            setup.label_studio.add_webhook(
                int(project["id"]), f"{args.url_base}/webhooks/label_studio", secret
            )
            print(f"Label Studio {TEMPORAL_PROJECT}: 웹훅 등록")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_ops_policy(root)
    ontology = load_ontology(root / "config" / "ontology" / "v1")
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        pool: list[str] = []
        for version in args.seed_golden:
            pool += list(get_golden_set(conn, version).session_ids)
        created = plan_session(
            conn, args.session_id, args.reviewer, policy, ontology, seed=args.seed,
            now=datetime.now(UTC), seed_sessions=pool, privacy=args.privacy,
        )  # fmt: skip
    engine.dispose()
    for a in created:
        reasons = sorted({f.reason.value for f in a.flagged})
        who = a.assignee or "-"
        print(f"{a.assignment_id} {a.mode.value} → {who} (우선순위 {a.priority:.2f} {reasons})")
    if not created:
        print("새 배정이 없습니다 (이미 계획됨)")
    return 0


def cmd_assign(args: argparse.Namespace) -> int:
    engine = sa.create_engine(database_url(args.url))
    setup = _setup(args)
    with engine.begin() as conn:
        a = get_assignment(conn, args.assignment_id)
        policy = load_ops_policy(repo_root())
        tasks = create_assignment_tasks(conn, a, setup, policy, datetime.now(UTC))
    engine.dispose()
    for t in tasks:
        print(f"{t.task_key} ({t.mode.value}, {t.assignee}) {t.media_uri}")
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    policy = load_ops_policy(repo_root())
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        done = list_assignments(conn, status=AssignmentStatus.DONE)
        existing = {a.assignment_id for a in list_assignments(conn)}
        created = [
            a for a in plan_qa(done, policy, seed=args.seed, now=datetime.now(UTC))
            if a.assignment_id not in existing
        ]  # fmt: skip
        for a in created:
            insert_assignment(conn, a)
    engine.dispose()
    print(f"QA 배정 {len(created)}개")
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        queue = list_assignments(conn, status=AssignmentStatus.OPEN)
    engine.dispose()
    for a in queue[: args.limit]:
        print(f"{a.priority:8.2f}  {a.assignment_id}  {a.mode.value}  {a.assignee or '-'}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    policy = load_ops_policy(repo_root())
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        report = quality_report(conn, policy)
    engine.dispose()
    for d in report.detection:
        print(f"오류 삽입 발견율 {d.reviewer}: {d.detected}/{d.injected} ({d.rate:.0%})")
    for key, agr in report.double.items():
        print(
            f"이중 라벨 일치 {key}: 카파 {agr.kappa:.3f}, "
            f"구간 F1 {agr.segment_f1:.3f}, 경계 F1 {agr.boundary_f1:.3f}"
        )
    for key, bias in report.blind_bias.items():
        print(f"프리라벨 편향 {key}: {bias:+.3f} (양수면 검수자가 프리라벨에 끌려감)")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    review = sub.add_parser("review", help="검수 도구 연동")
    rsub = review.add_subparsers(dest="review_command", required=True)

    create = rsub.add_parser("create", help="세션의 검수 작업 생성")
    create.add_argument("session_id")
    create.add_argument("--stage", choices=["privacy", "labeling"], required=True)
    create.add_argument(
        "--assignee",
        required=True,
        help="검수 담당자 (블러 검수는 review.yaml reviewers.privacy, 작업 라벨은 워터마크)",
    )
    create.add_argument("--no-cvat", action="store_true")
    create.add_argument("--no-ls", action="store_true")
    create.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    create.set_defaults(func=cmd_create)

    collect = rsub.add_parser("collect", help="검수 결과 수집 (예: cvat:42)")
    collect.add_argument("task_key")
    collect.add_argument("--reviewer", required=True)
    collect.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    collect.set_defaults(func=cmd_collect)

    srv = rsub.add_parser("serve", help="웹훅을 받아 끝난 작업을 수집 (DLP_WEBHOOK_SECRET 필요)")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    srv.set_defaults(func=cmd_serve)

    reg = rsub.add_parser("register-webhooks", help="검수 프로젝트에 웹훅 등록")
    reg.add_argument("url_base", help="도구에서 닿는 웹훅 서버 주소 (예: http://host:8765)")
    reg.set_defaults(func=cmd_register)

    pl = rsub.add_parser("plan", help="세션 검수 배정 계획 (우선순위·표본·블라인드·이중·오류 삽입)")
    pl.add_argument("session_id")
    pl.add_argument("--reviewer", action="append", required=True, help="검수자 (여러 번)")
    pl.add_argument("--seed-golden", action="append", default=[], help="오류 삽입 과제 원천 골든셋")
    pl.add_argument("--privacy", action="store_true", help="블러 검수 단위로 계획 (전수 검수)")
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    pl.set_defaults(func=cmd_plan)

    asg = rsub.add_parser("assign", help="배정의 검수 도구 작업 생성")
    asg.add_argument("assignment_id")
    asg.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    asg.set_defaults(func=cmd_assign)

    qa = rsub.add_parser("qa", help="끝난 표준 배정에서 QA 재검수 배정을 뽑는다")
    qa.add_argument("--seed", type=int, default=0)
    qa.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    qa.set_defaults(func=cmd_qa)

    q = rsub.add_parser("queue", help="열린 배정을 우선순위 순으로")
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    q.set_defaults(func=cmd_queue)

    ql = rsub.add_parser("quality", help="오류 삽입 발견율, 이중 라벨 일치도, 프리라벨 편향")
    ql.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    ql.set_defaults(func=cmd_quality)
