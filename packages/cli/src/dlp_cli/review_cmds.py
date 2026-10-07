"""검수 도구 연동·검수 운영 하위 명령 (WP6·WP12, ADR 0006·0014·0023·0024·0029).

등록하는 명령 (`dlp review …`):
- 도구 연동 (WP6, `dlp_review.tasks`·`collect`·`webhook`·`verify`)
  - `create <세션> --stage privacy|labeling --assignee` — 세션의 검수 작업을 CVAT(공간 라벨·블러)와
    Label Studio(시간 구간)에 만든다. 블러 검수 담당자는 `review.yaml reviewers.privacy`의 원본 접근
    권한자여야 한다.
  - `collect <작업 키> --reviewer` — 끝난 작업의 결과를 라벨 이력으로 들여온다 (reconcile:
    승인·수정·삭제·추가 → 새 `LabelRecord` + `parent_label_id`). 이미 수집했으면 아무것도 하지
    않는다(멱등).
  - `serve [--port]` — CVAT·Label Studio 웹훅을 받아 끝난 작업을 자동 수집하는 HTTP 서버.
  - `register-webhooks <url_base>` — 검수 프로젝트에 웹훅을 등록한다.
  - `verify <세션>` — 검수가 끝난 세션을 `prelabeled → human_verified`로 옮긴다 (ADR 0029).
- 운영 로직 (WP12, `dlp_review.ops`)
  - `plan <세션> --reviewer …` — 우선순위·표본 검증·블라인드·이중·오류 삽입 배정을 계획해 DB에 쓴다.
  - `assign <배정 ID>` — 배정 하나의 검수 도구 작업을 만든다 (멱등).
  - `qa` — 끝난 표준 배정에서 QA 재검수 배정을 뽑는다.
  - `queue [--limit]` — 열린 배정을 우선순위 높은 순으로 보여 준다.
  - `quality` — 오류 삽입 발견율, 이중 라벨 일치도(카파·구간 F1·경계 F1), 프리라벨 편향.

정상 순서: (프리라벨·관계·행동 뒤) `plan` → `assign`(또는 `create`) → 검수 → `collect`/`serve`
→ `qa` → `verify` → `dlp dataset build`. 블러 검수는 `dlp privacy detect` 뒤 `create --stage
privacy` (또는 `plan --privacy`) → `collect` → `dlp privacy approve`.

정책 출처: `config/policies/review.yaml` (우선순위·표본·측정 비율, `reviewers.privacy`,
`cvat.users`, 미디어).

주의:
- `_setup`이 원본 저장소를 `raw_store("s3", …, "review.<명령>")` 감사 저장소로 만든다 (블러 검수
  작업은 원본 영상을 CVAT에 올린다, ADR 0020). 일반 라벨링 작업은 블러본만 쓰고, 라벨러용 서명 URL은
  라벨러 자격 증명(`S3Store.labeler_from_env`, 라벨링 버킷 읽기 전용)으로 만든다.
- 블라인드·이중 측정 레코드(`measurement`)와 오류 삽입 레코드는 운영 라벨이 아니다
  (`current_labels()`).
- 검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 (ADR 0015, 0019).
"""

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
from dlp_review.verify import verify_session
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
    """명령 인자로 검수 도구 연동에 필요한 저장소·온톨로지·도구 클라이언트를 묶는다.

    인자:
        args.review_command: 감사 기록의 목적 문자열(`review.<명령>`)에 쓴다.
        args.url: (있으면) 감사 기록용 DB URL. 없는 명령은 `DLP_DATABASE_URL`/기본값.
        args.no_cvat / args.no_ls: 참이면 해당 도구 클라이언트를 만들지 않는다 (None).

    반환: `ReviewSetup`.
    - `raw`: 원본 버킷 감사 저장소 (항상 S3, 목적 `review.<명령>`).
    - `labeling`: 라벨링 버킷 (서비스 자격 증명, 블러본·작업 미디어 쓰기).
    - `labeling_reader`: 라벨러 자격 증명(읽기 전용) — 검수 화면 서명 URL용.
    - `ontology`: `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보: 세션의 온톨로지 버전을
      따라야 함).
    - `cvat`/`label_studio`: 환경 변수(`DLP_CVAT_*`, `DLP_LABEL_STUDIO_*`)로 만든 클라이언트.

    부작용: 없음 (클라이언트·엔진 생성만). 환경 변수가 빠지면 클라이언트 생성에서 예외가 날 수 있다.
    """
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
    """웹훅 서명 검증·등록에 쓰는 공유 비밀(`DLP_WEBHOOK_SECRET`)을 읽는다. 없으면 `SystemExit`."""
    secret = os.environ.get("DLP_WEBHOOK_SECRET")
    if not secret:
        raise SystemExit("DLP_WEBHOOK_SECRET를 설정하세요")
    return secret


def cmd_create(args: argparse.Namespace) -> int:
    """`dlp review create <세션> --stage privacy|labeling --assignee <사람>`: 세션의 검수 작업을
    만든다.

    - `privacy`: 영상 스트림마다 CVAT 블러 검수 작업(원본 영상 + 현재 블러 트랙). 담당자가
      `reviewers.privacy`에 없으면 원본을 올리기 전에 거부한다.
    - `labeling`: 블러본 기반 공간 라벨 작업(CVAT)과 시간 구간 작업(Label Studio). 세션이 프라이버시
      승인 상태여야 한다(아니면 `TaskError`). 담당자 ID는 라벨러 워터마크에도 쓴다.

    반환: 0. 출력: 작업 키, 단계, 스트림, 미디어 URI.
    부작용: 한 트랜잭션에서 `review_tasks` 등 DB 쓰기, 검수 도구 API 호출, 라벨링 버킷 쓰기, 블러
    검수면 원본 읽기(감사 기록). 실패 시 DB는 롤백되지만 이미 만든 도구 쪽 작업은 남을 수 있다.
    """
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
    """`dlp review collect <작업 키> --reviewer <사람>`: 끝난 검수 작업의 결과를 라벨 이력으로
    들여온다.

    인자:
        args.task_key: `cvat:<CVAT task id>` 또는 `label_studio:<LS task id>` 형식의 작업 키.
        args.reviewer: 검수자 ID. 작업에 담당자가 있으면 같아야 한다 (다르면 `TaskError`).

    반환: 0. 이미 수집한 작업이면 "이미 수집함"만 출력(멱등, 작업 행을 잠가 동시 수집을 막는다).
    부작용: 한 트랜잭션에서 `label_records`에 검수 결과 레코드 추가(덮어쓰지 않음), 작업·배정 상태
    갱신, 배정 마무리(표본 판정). 검수 도구 API에서 주석을 읽는다.
    """
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


def cmd_verify(args: argparse.Namespace) -> int:
    """검수가 끝난 세션을 human_verified로 옮긴다. 남은 일이 있으면 이유를 출력하고 1을 돌려준다.

    인자:
        args.actor: 판정한 사람. 없으면 `DLP_ACTOR`, 그것도 없으면 `"unknown"` (다른 명령의
            `current_actor()`는 OS 사용자로 대체하는 것과 다르다).

    반환: 이미 완료 단계이거나 이번에 완료했으면 0, 아직 조건을 못 맞췄으면 1.
    부작용: 조건을 만족하면 한 트랜잭션에서 세션 생애주기 전이와 생애주기 기록을 쓴다 (ADR 0029).
    """
    actor = args.actor or os.environ.get("DLP_ACTOR") or "unknown"
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        result = verify_session(conn, args.session_id, datetime.now(UTC), actor)
    engine.dispose()
    if result.already:
        print(f"{args.session_id}: 이미 검수 완료 단계")
    elif result.verified:
        print(f"{args.session_id}: 검수 완료 (human_verified)")
    else:
        print(f"{args.session_id}: 아직 완료가 아니다")
        for reason in result.reasons:
            print(f"  - {reason}")
        return 1
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """`dlp review serve [--port]`: 웹훅 서버를 띄워 끝난 작업을 자동 수집한다 (블로킹, Ctrl-C로
    끝냄).

    웹훅마다 새 트랜잭션을 열어:
    1. 작업을 조회하고 검수자를 정한다 — 웹훅의 도구 사용자 ID가 아니라 작업 담당자다
       (`resolve_reviewer`). 담당자와 CVAT job 담당자가 맞지 않으면 수집하지 않는다.
    2. 서비스 계정(Label Studio 서비스 사용자)이 만든 주석이면 수집하지 않는다.
    3. `collect_task`로 수집한다. `TaskError`는 출력만 하고 넘어간다 (서버는 계속 돈다).

    필요 환경 변수: `DLP_WEBHOOK_SECRET`(서명 검증), DB·S3·도구 접속 정보.
    반환: 서버가 끝나면 0.
    """
    setup = _setup(args)
    engine = sa.create_engine(database_url(args.url))

    service_user = setup.label_studio.service_user_id() if setup.label_studio else None
    ops_policy = load_ops_policy(repo_root())

    def on_collect(req: CollectRequest) -> None:
        """웹훅 한 건 처리 콜백. 수집 여부를 출력하고 예외를 서버로 올리지 않는다 (거부 사유는
        출력만)."""
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
    """`dlp review register-webhooks <url_base>`: 검수 프로젝트에 웹훅을 등록한다.

    CVAT의 블러(`PRIVACY_PROJECT`)·공간(`SPATIAL_PROJECT`) 프로젝트와 Label Studio의 시간 구간
    (`TEMPORAL_PROJECT`) 프로젝트가 있으면 각각 `<url_base>/webhooks/cvat|label_studio`를 등록한다.
    프로젝트가 아직 없으면 조용히 건너뛴다. 클라이언트는 기존 웹훅을 확인하지 않고 매번 POST하므로
    다시 돌리면 같은 웹훅이 중복 등록된다 (멱등 아님).
    """
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
    """`dlp review plan <세션> --reviewer A --reviewer B …`: 세션의 검수 배정을 계획한다 (WP12).

    인자:
        args.reviewer: 배정에 쓸 검수자 목록 (`--privacy`면 모두 `reviewers.privacy`에 있어야 한다).
        args.seed_golden: 오류 삽입 과제의 원천이 될 골든셋 버전들 (그 세션들의 라벨에 오류를
            심는다).
        args.privacy: 참이면 블러 검수 단위(스트림 전체, 전수)로 계획한다.
        args.seed: 표본 추출·배정 난수 시드 (같은 시드면 같은 계획, 결정적).

    반환: 0. 이미 계획된 단위는 새로 만들지 않는다(멱등) — 이때 "새 배정이 없습니다".
    부작용: 한 트랜잭션에서 `review_assignments`에 새 배정 INSERT.
    온톨로지는 `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보).
    """
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
    """`dlp review assign <배정 ID>`: 배정의 검수 도구 작업을 만든다.

    이미 만든 배정이면 기존 작업을 그대로 돌려준다 (배정 행을 `FOR UPDATE`로 잠가 동시 실행에도 한
    번만).
    부작용: `review_tasks` 쓰기, 검수 도구 API 호출, 블러 배정이면 원본 읽기(감사 기록).
    """
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
    """`dlp review qa [--seed]`: 끝난(DONE) 표준 배정 중 정책 비율만큼 QA 재검수 배정을 뽑는다.

    이미 있는 배정 ID는 다시 넣지 않는다 (`plan_qa`가 결정적 ID를 만들므로 재실행해도 중복 없음).
    부작용: 한 트랜잭션에서 `review_assignments` INSERT.
    """
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
    """`dlp review queue [--limit]`: 열린(OPEN) 배정을 우선순위 높은 순(같으면 만든 순)으로
    출력한다. 읽기 전용."""
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        queue = list_assignments(conn, status=AssignmentStatus.OPEN)
    engine.dispose()
    for a in queue[: args.limit]:
        print(f"{a.priority:8.2f}  {a.assignment_id}  {a.mode.value}  {a.assignee or '-'}")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    """`dlp review quality`: 검수 품질 측정 리포트를 출력한다. 읽기 전용.

    - 오류 삽입 발견율: 검수자별 (발견한 삽입 오류 수 / 삽입한 수).
    - 이중 라벨 일치도: 키별 Cohen 카파, 구간 F1, 경계 F1.
    - 프리라벨 편향: 블라인드 과제와 일반 과제 결과 차이 (양수면 검수자가 프리라벨에 끌려감).
    """
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
    """`review` 하위 명령 묶음을 등록한다 (도구 연동 5개 + 운영 5개)."""
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

    vf = rsub.add_parser("verify", help="검수가 끝난 세션을 검수 완료(human_verified)로 표시")
    vf.add_argument("session_id")
    vf.add_argument("--actor", help="판정한 사람 (기본: DLP_ACTOR)")
    vf.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    vf.set_defaults(func=cmd_verify)

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
