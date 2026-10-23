"""검수 도구 연동·검수 운영 하위 명령 (WP6·WP12, ADR 0006·0014·0023·0024·0029).

등록하는 명령 (`dlp review …`):
- 도구 연동 (WP6, `dlp_review.tasks`·`collect`·`webhook`·`verify`)
  - `create <세션> --stage privacy|labeling --assignee [--stream S…] [--tool T]` — 세션의 검수
    작업을 CVAT(공간 라벨·블러)와 Label Studio(시간 구간)에 만든다. 블러 검수 담당자는
    `review.yaml reviewers.privacy`의 원본 접근 권한자여야 한다. 라벨링은 먼저 회수 대기를
    실행하고, 세션에 대기가 하나라도 남으면(대상이 아닌 스트림 포함) 만들지 않는다 (종료 코드 4).
  - `collect <작업 키> --reviewer` — 끝난 작업의 결과를 라벨 이력으로 들여온다 (reconcile:
    승인·수정·삭제·추가 → 새 `LabelRecord` + `parent_label_id`). 이미 수집했거나 회수한 작업이면
    아무것도 하지 않는다(멱등). 블러가 바뀌면 커밋 뒤 배포 매체를 회수하고 다시 만들 명령을 보인다.
  - `serve [--port]` — CVAT·Label Studio 웹훅을 받아 끝난 작업을 자동 수집하는 HTTP 서버.
    수집 결과(회수한 작업·다시 만들 명령 포함)는 `collect`와 같이 서버 로그에 남는다. 담당자의 도구
    계정(`cvat.users`, `label_studio.users`)이 아닌 사람의 제출은 받지 않는다 (F-REV-06).
  - `revoke [<세션>] [--stream] [--task --reason] [--all --reason] [--include-legacy] [--dry-run]
    [--pending] [--mark-tool-gone KEY --note]` — 블러가 바뀐 뒤 이미 배포한 라벨링 매체를
    회수한다 (ADR 0039). 세션 경로는 늘 sweep을 하고 `--task`·`--all`은 그 위에 수동 요청을 더한다.
    대기가 남으면 종료 코드 4 (cron `--pending`의 경보).
    `revoke --orphans [<세션>] [--apply]` — DB 롤백으로 기록 없이 남은 도구 작업·라벨링 버킷 사본
    (고아)을 이름 규칙과 DB로 찾는다 (`dlp_review.orphans`, F-REV-15). 기본은 보고만 하고
    `--apply`일 때만 지운다. 나이 하한은 `review.yaml orphans.min_age_s`. 고아가 남거나 확인하지
    못한 채널이 있으면 종료 코드 4.
  - `register-webhooks <url_base>` — 검수 프로젝트에 웹훅을 등록한다.
  - `refresh-urls [--session S…]` — 열린 Label Studio 작업의 영상·시계열 서명 URL을 정책 유효 시간
    (`media.signed_url_ttl_s`)으로 다시 서명한다 (F-REV-08, cron으로 유효 시간보다 짧은 주기).
    수집·회수한 작업과 매체가 낡은 작업은 갱신하지 않는다. 갱신하지 못한 작업이 있으면 종료 코드 1.
  - `verify <세션>` — 검수가 끝난 세션을 `prelabeled → human_verified`로 옮긴다 (ADR 0029).
  - `ui-theme [--check]` — `config/review_ui.yaml`로 CVAT 테마 색 파일을 다시 만든다 (ADR 0038,
    `make ui-theme`). `--check`는 파일을 쓰지 않고 다르면 1로 끝난다.
- 운영 로직 (WP12, `dlp_review.ops`)
  - `plan <세션> --reviewer …` — 우선순위·표본 검증·블라인드·이중·오류 삽입 배정을 계획해 DB에 쓴다.
    AI 1차 검수(`review.yaml ai_review`, WP17.6)가 켜져 있으면 판정 지적을 우선순위 사유로 더하고
    표본 묶음에서 뺀다. required면 판정이 끝나지 않은 세션의 배정을 미룬다 (`AI_PENDING_EXIT` 6).
  - `assign <배정 ID> [--assignee]` — 배정 하나의 검수 도구 작업을 만든다 (멱등). 담당자 없는
    배정은 `--assignee`로 담당자를 정해야 만든다 (없으면 종료 코드 1, F-REV-07).
  - `qa` — 끝난 표준 배정에서 QA 재검수 배정을 뽑는다. 선임(`reviewers.senior`)이 없거나 원래
    담당자뿐이면 담당자를 비워 둔다. 그 배정은 `reviewers.senior`에 원래 담당자가 아닌 선임을 넣은
    뒤 `assign --assignee <그 선임>`으로 만든다 (QA 담당자는 선임이고 짝 배정 담당자가 아니어야
    한다).
  - `queue [--limit]` — 열린 배정을 우선순위 높은 순으로 보여 준다 (AI 지적 사유 수 포함).
  - `quality` — 오류 삽입 발견율, 이중 라벨 일치도(카파·구간 F1·경계 F1), 프리라벨 편향.
    `--ai`면 AI 1차 검수 판정 품질(대조군 기준 지적 정밀도·재현율·층화 ok 놓침률과 상한, 제안
    적용, 자동화 편향·오류 삽입 누출, WP17.8), `--ai-thresholds`면 표시 최소 확신도 조안.

정상 순서: (프리라벨·관계·행동·`dlp aireview run` 뒤) `plan` → `assign`(또는 `create`) → 검수 →
`collect`/`serve` → `qa` → `verify` → `dlp dataset build`. 블러 검수는 `dlp privacy detect` 뒤
`create --stage privacy` (또는 `plan --privacy`) → `collect` → `dlp privacy approve`.

정책 출처: 기본은 `config/policies/review.yaml` (우선순위·표본·측정 비율, `reviewers.privacy`,
`cvat.users`, `label_studio.users`, 미디어).
CLI에서는 `DLP_REVIEW_POLICY_FILE`로 정책 파일을 바꿀 수 있다. 검수 정책을
읽는 하위 명령은 모두 `review_config.load_review_policy` 하나로 읽는다. 저장소·도구 설정(`_setup`)을
쓰는 명령(create·collect·serve·assign·register-webhooks·refresh-urls)은 정책을 한 번 읽어
`_setup`과 자기 몫(원본 검수자, 수집·배정 마무리, 웹훅 검수자 확인, 서명 URL 유효 시간 출력)에 같은
객체를 넘긴다. 그래서 `ReviewSetup`의 CVAT·Label Studio 계정 연결(`cvat_policy`,
`label_studio_policy`)·매체(`media_policy`)도 고른 파일의 값이다 (라이브러리 지연 로더의 Git
기본값이 아니다, F-REV-01). `plan`·`qa`·`quality`는 같은 로더로 정책만 읽고, `queue`·`verify`는
검수 정책을 쓰지 않는다. 매체 회수만 하는 `revoke`도 검수 정책을 읽지 않는다 (회수 실행기는 정책
값을 쓰지 않는다. 정책 파일이 깨져도 낡은 라벨링 매체를 지운다). 예외는 `revoke --orphans`로, 나이
하한(`orphans.min_age_s`)을 고른 정책 파일에서 엄격히 읽는다. 정책을 읽는 명령에서는
`orphans.min_age_s`·`media.signed_url_ttl_s`가 모두 필수 키다 (이전에 복사한 정책 파일에 없으면 그
명령들이 `Field required`로 멈춘다. 기본 review.yaml 값을 옮겨 적는다).

주의:
- `_setup`(`review_config.review_setup`)이 원본 저장소를 `raw_store("s3", …, "review.<명령>")` 감사
  저장소로 만든다 (블러 검수 작업은 원본 영상을 CVAT에 올린다, ADR 0020). 일반 라벨링 작업은
  블러본만 쓰고, 라벨러용 서명 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 만든다.
- 매체 회수(ADR 0039): 블러를 바꾼 트랜잭션이 회수 요청을 같은 커밋에 남기고, 명령은 커밋 뒤
  `review_config.run_revocations`로 라벨링 버킷 사본 → CVAT → Label Studio 순으로 끊는다.
- 블라인드·이중 측정 레코드(`measurement`)와 오류 삽입 레코드는 운영 라벨이 아니다
  (`current_labels()`).
- 검수자가 승인·표본 검증한 라벨은 어떤 단계도 지우지 않는다 (ADR 0015, 0019).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.exc import NoResultFound

from dlp_aireview.status import prelabeled_at
from dlp_cli import hints_cmds
from dlp_cli.ai_quality_print import (
    ai_quality_json,
    print_ai_quality,
    print_thresholds,
    thresholds_json,
)
from dlp_cli.review_config import (
    REVOCATION_PENDING_EXIT,
    ai_display_config,
    print_revocations,
    review_setup,
    run_revocations,
)
from dlp_cli.review_config import load_review_policy as load_ops_policy
from dlp_cli.schema_cmds import database_url
from dlp_media.audit import current_actor
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import (
    operational_blur,
    render_hash,
    request_media_revocation,
    stale_media_tasks,
)
from dlp_review.ai_collect import summary_counts
from dlp_review.ai_display import REASONS, AiDisplayConfig, GateDecision, required_gate
from dlp_review.collect import collect_task
from dlp_review.ontology_projects import ontology_project
from dlp_review.ops.assign import plan_qa
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.runner import (
    ReviewerRuleError,
    assign_reviewer,
    create_assignment_tasks,
    plan_session,
    quality_report,
)
from dlp_review.orphans import OrphanReport, delete_orphans, find_orphans
from dlp_review.reconcile import ReviewOutcome
from dlp_review.refresh import refresh_session_urls, sessions_with_open_ls_tasks
from dlp_review.revoke import RevocationIncompleteError, mark_tool_gone
from dlp_review.roles import AccessError
from dlp_review.tasks import (
    PRIVACY_PROJECT,
    SPATIAL_PROJECT,
    TEMPORAL_PROJECT,
    ReviewSetup,
    TaskError,
    create_labeling_tasks,
    create_privacy_tasks,
    label_studio_user_id,
)
from dlp_review.ui_theme import TOKENS_FILE, cvat_tokens_css, load_ui_theme
from dlp_review.verify import verify_session
from dlp_review.webhook import CollectRequest, resolve_reviewer, serve
from dlp_schema import repo_root
from dlp_schema.aireview import AiRunEvent, AiRunEventKind
from dlp_schema.db.aireview_repository import insert_ai_run_event, latest_run_state
from dlp_schema.db.repository import (
    get_assignment,
    get_golden_set,
    get_labels,
    get_ontology,
    get_review_task,
    get_session_ontology,
    insert_assignment,
    list_assignments,
    lock_session,
    pending_revocations,
)
from dlp_schema.db.tables import ontology_versions
from dlp_schema.labels import LabelRecord
from dlp_schema.review import (
    AssignmentStatus,
    ReviewMode,
    ReviewReason,
    ReviewStage,
    ReviewTask,
    ReviewTaskStatus,
    ReviewTool,
)

# `dlp review assign`: 담당자 없는 배정에 `--assignee`를 주지 않아 작업을 만들지 않았다 (F-REV-07)
NO_ASSIGNEE_EXIT = 1
# `dlp review plan`: review.yaml ai_review.mode=required인데 AI 1차 검수가 끝나지 않아 배정을 미뤘다
# (WP17.6). 회수 대기(4)·담당자 없음(1)과 구별한다 (cron이 `dlp aireview run`을 먼저 돌리게)
AI_PENDING_EXIT = 6


def _setup(args: argparse.Namespace, policy: ReviewOpsPolicy | None = None) -> ReviewSetup:
    """명령 인자로 검수 도구 연동에 필요한 저장소·온톨로지·도구 클라이언트·정책을 묶는다.

    인자:
        args.review_command: 감사 기록의 목적 문자열(`review.<명령>`)에 쓴다.
        args.url: (있으면) 감사 기록용 DB URL. 없는 명령은 `DLP_DATABASE_URL`/기본값.
        args.no_cvat / args.no_ls: 참이면 해당 도구 클라이언트를 만들지 않는다 (None).
        policy: 명령이 이미 읽은 검수 정책 (`load_ops_policy`). None이면 여기서 고른 파일을 읽는다.
            명령이 정책을 따로 쓰면 반드시 같은 객체를 넘긴다 (한 출처).

    반환: `ReviewSetup`.
    - `raw`: 원본 버킷 감사 저장소 (항상 S3, 목적 `review.<명령>`).
    - `labeling`: 라벨링 버킷 (서비스 자격 증명, 블러본·작업 미디어 쓰기).
    - `labeling_reader`: 라벨러 자격 증명(읽기 전용) — 검수 화면 서명 URL용.
    - `ontology`: None. 작업 생성 시 DB에서 각 세션 버전의 사전을 읽는다.
    - `cvat`/`label_studio`: 환경 변수(`DLP_CVAT_*`, `DLP_LABEL_STUDIO_*`)로 만든 클라이언트.
    - `cvat_config`/`media`: 고른 검수 정책(`DLP_REVIEW_POLICY_FILE`)의 `cvat`·`media`.

    부작용: 설정 파일을 읽고 저장소·HTTP 클라이언트를 만든다. CVAT 클라이언트는 생성 중
    로그인 API를 호출해 토큰을 받는다 (`--no-cvat`이면 생략). 환경 변수 누락, 파일·설정 오류,
    로그인 실패·통신 오류는 호출자에게 전파된다. 본문은 `review_config.review_setup`이다.
    """
    setup, _ = review_setup(
        f"review.{args.review_command}",
        getattr(args, "url", None),
        cvat=not getattr(args, "no_cvat", False),
        label_studio=not getattr(args, "no_ls", False),
        policy=policy,
    )
    return setup


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
      `--stream`(반복)·`--tool`로 만들 작업을 좁힌다 (회수한 직접 작업만 다시 만들 때).
      만들기 전에 회수 대기를 실행한다 (렌더 해시가 지금과 다른 작업은 먼저 요청한다, sweep).
      세션에 대기가 하나라도 남으면 `--stream`으로 고른 스트림과 무관하게 만들지 않고
      `REVOCATION_PENDING_EXIT`(4)로 끝낸다 (함수 `create_labeling_tasks`는 대상 스트림만 본다).

    반환: 0 (또는 4). 출력: 작업 키, 단계, 스트림, 미디어 URI.
    부작용: 한 트랜잭션에서 `review_tasks` 등 DB 쓰기, 검수 도구 API 호출, 라벨링 버킷 쓰기, 블러
    검수면 원본 읽기(감사 기록). 실패 시 DB는 롤백되지만 이미 만든 도구 쪽 작업은 남을 수 있다.
    """
    policy = load_ops_policy(repo_root())
    setup = _setup(args, policy)
    engine = sa.create_engine(database_url(args.url))
    now = datetime.now(UTC)
    if not args.assignee:
        raise SystemExit("검수 작업에는 --assignee가 필요합니다 (블러 검수는 원본 접근 권한자)")
    if args.stage != "privacy":
        report = run_revocations(engine, args.session_id, setup, sweep=True)
        if print_revocations(report, args.session_id):
            engine.dispose()
            print(f"{args.session_id}: 매체 회수가 끝나지 않아 라벨링 작업을 만들지 않습니다")
            return REVOCATION_PENDING_EXIT
    with engine.begin() as conn:
        if args.stage == "privacy":
            # 담당자가 review.yaml reviewers.privacy에 없으면 원본을 올리기 전에 막는다
            tasks = create_privacy_tasks(
                conn,
                args.session_id,
                setup,
                now,
                assignee=args.assignee,
                privacy_reviewers=policy.reviewers.privacy,
            )
        else:
            tasks = create_labeling_tasks(
                conn, args.session_id, setup, args.assignee, now,
                streams=set(args.stream) if args.stream else None,
                tools={ReviewTool(args.tool)} if args.tool else None,
            )  # fmt: skip
    engine.dispose()
    for t in tasks:
        print(f"{t.task_key:<20} {t.stage.value:<9} {t.stream_id:<14} {t.media_uri}")
    return 0


def _print_trajectories(approved: int, retracted: int) -> None:
    """3D 궤적의 2D 트랙 검수 물려받기 결과 (ADR 0048·0050). 지운 것은 다시 만들라고 안내한다."""
    if approved:
        print(f"  3D 궤적: 2D 트랙 검수 물려받음 {approved} (ADR 0048)")
    if retracted:
        print(
            f"  3D 궤적 {retracted}개: 출처 2D가 바뀌어 물려받은 검수가 끊겨 지움 (ADR 0050. "
            "dlp prelabel run <세션>으로 고친 2D에서 다시 만든다)"
        )


def _print_ai_outcome(outcome: ReviewOutcome) -> None:
    """AI 1차 검수 결과 기록이 있으면 적용·기각 수와 경고를 출력한다 (WP17.7)."""
    if outcome.ai is None:
        return
    counts = summary_counts(outcome.ai)
    others = ", ".join(
        f"{k} {v}" for k, v in sorted(counts.items()) if k not in ("applied", "dismissed")
    )
    print(
        f"  AI 제안 적용 {counts['applied']}, 기각 {counts['dismissed']}"
        + (f", 적용 안 함({others})" if others else "")
    )
    for w in outcome.ai.warnings:
        print(f"  [AI 경고] {w}")


def _print_outcome(task: ReviewTask, outcome: ReviewOutcome | None, *, ai: bool = False) -> None:
    """수집 결과와 회수 요청·다시 만들 명령을 출력한다.

    ai: AI 1차 검수 판정을 쓰는가 (`ReviewSetup.ai`). 블러가 바뀌면 이전 판정이 표시 프로필에서
    빠지므로 다시 만들기 전에 `dlp aireview run <세션>`을 안내한다 (WP17.6, ADR 0040 I10).
    """
    if outcome is None:
        if task.status is ReviewTaskStatus.REVOKED:
            print(f"{task.task_key}: 회수한 작업이라 수집하지 않음 (결과는 회수 스냅숏에 남는다)")
        else:
            print(f"{task.task_key}: 이미 수집함")
        return
    print(
        f"{task.task_key}: 승인 {len(outcome.approved)}, 수정 {outcome.corrected}, "
        f"삭제 {outcome.retracted}, 추가 {outcome.added}"
    )
    if outcome.descriptions_approved or outcome.descriptions_retracted:
        # 설명 라벨은 행동·상위 구간 결과를 따른다 (ADR 0043 3절)
        print(
            f"  설명: 행동 검수 물려받음 {len(outcome.descriptions_approved)}, "
            f"검수 전 설명 삭제 {len(outcome.descriptions_retracted)}"
        )
    _print_trajectories(len(outcome.trajectories_approved), len(outcome.trajectories_retracted))
    _print_ai_outcome(outcome)
    if not outcome.revoked:
        return
    print(
        f"  블러가 바뀌어 라벨링 작업 {len(outcome.revoked)}개 회수: {', '.join(outcome.revoked)}"
    )
    rerun = f" → dlp aireview run {task.session_id}" if ai else ""
    print(f"  다시 만들기 (남은 블러 검수 → dlp privacy approve → dlp privacy render{rerun} 뒤):")
    for aid in outcome.reissue_assignments:
        print(f"    dlp review assign {aid}")
    for assignee, stream, tool in outcome.direct_reissue:
        print(
            f"    dlp review create {task.session_id} --stage labeling --assignee {assignee} "
            f"--stream {stream} --tool {tool}"
        )


def cmd_collect(args: argparse.Namespace) -> int:
    """`dlp review collect <작업 키> --reviewer <사람>`: 끝난 검수 작업의 결과를 라벨 이력으로
    들여온다.

    인자:
        args.task_key: `cvat:<CVAT task id>` 또는 `label_studio:<LS task id>` 형식의 작업 키.
        args.reviewer: 검수자 ID. 작업에 담당자가 있으면 같아야 한다 (다르면 `TaskError`).

    반환: 0. 이미 수집한 작업이면 "이미 수집함"만 출력(멱등, 작업 행을 잠가 동시 수집을 막는다).
    회수한 작업이면 수집하지 않는다. 커밋 뒤 그 세션의 회수 대기를 실행하고(수집 결과와 무관하게),
    대기가 남으면 `REVOCATION_PENDING_EXIT`(4).
    부작용: 한 트랜잭션에서 `label_records`에 검수 결과 레코드 추가(덮어쓰지 않음), 작업·배정 상태
    갱신, 배정 마무리(표본 판정), 블러가 바뀌면 라벨링 작업 회수 요청. 검수 도구 API에서 주석을
    읽는다. 커밋 뒤 라벨링 버킷 사본·도구 작업 삭제 (`review_config.run_revocations`).
    """
    policy = load_ops_policy(repo_root())
    setup = _setup(args, policy)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        outcome = collect_task(conn, args.task_key, setup, args.reviewer, datetime.now(UTC), policy)
        task = get_review_task(conn, args.task_key)
    # 수집·회수 요청이 커밋된 뒤에 외부 채널을 끊는다 (도구·S3 장애가 수집을 되돌리지 않는다)
    report = run_revocations(engine, task.session_id, setup)
    engine.dispose()
    _print_outcome(task, outcome, ai=setup.ai is not None)
    return print_revocations(report, task.session_id)


def cmd_verify(args: argparse.Namespace) -> int:
    """검수가 끝난 세션을 human_verified로 옮긴다. 남은 일이 있으면 이유를 출력하고 1을 돌려준다.

    인자:
        args.actor: 판정한 사람. 없으면 다른 명령과 같이 `current_actor()` (`DLP_ACTOR`, 그것도
            없으면 OS 사용자). 생애주기 기록에 누가 판정했는지 남기려는 것이라 "unknown"으로
            두지 않는다.

    반환: 이미 완료 단계이거나 이번에 완료했으면 0, 아직 조건을 못 맞췄으면 1.
    부작용: 조건을 만족하면 한 트랜잭션에서 세션 생애주기 전이와 생애주기 기록을 쓴다 (ADR 0029).
    """
    # 회귀: 예전에는 DLP_ACTOR가 없으면 "unknown"으로 기록해 판정자를 알 수 없었다
    actor = args.actor or current_actor()
    # 자동 라벨 분류(카메라 자세 등, ADR 0050)는 DLP_REVIEW_POLICY_FILE을 따르는 정책에서 읽는다
    automatic = load_ops_policy(repo_root()).verify.automatic
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        result = verify_session(
            conn, args.session_id, datetime.now(UTC), actor, automatic=automatic
        )
    engine.dispose()
    if result.descriptions_approved or result.descriptions_retracted:
        print(
            f"{args.session_id}: 설명이 행동 검수를 따름 (물려받음 {result.descriptions_approved}, "
            f"검수 전 설명 삭제 {result.descriptions_retracted}, ADR 0043 3절)"
        )
    _print_trajectories(result.trajectories_approved, result.trajectories_retracted)
    if result.automatic_unreviewed:
        print(
            f"  자동 라벨(카메라 자세) {result.automatic_unreviewed}개: 사람 검수 대상이 아니라 "
            "완료를 막지 않음 (검수 전 그대로, 내보내기 manifest에 automatic_unreviewed, ADR 0050)"
        )
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


def _ls_users(setup: ReviewSetup, task: ReviewTask) -> dict[str, int]:
    """Label Studio 작업 담당자의 Label Studio 사용자 ID (`resolve_reviewer`의 ls_users).

    다른 도구의 작업·담당자 없는 작업·연결 없는 담당자는 빈 사전이다 (그러면 웹훅이 받지 않는다).
    연결은 웹훅마다 다시 찾는다 (서버를 띄운 뒤 만든 계정도 받는다). 연결한 계정이 Label Studio에
    없거나 서비스 계정이면 `TaskError` (설정 오류, 서버는 503).
    """
    if task.tool is not ReviewTool.LABEL_STUDIO or task.assignee is None:
        return {}
    if setup.label_studio is None:
        return {}
    users = setup.label_studio_policy().users
    uid = label_studio_user_id(setup.label_studio, users, task.assignee)
    return {} if uid is None else {task.assignee: uid}


def cmd_serve(args: argparse.Namespace) -> int:
    """`dlp review serve [--port]`: 웹훅 서버를 띄워 끝난 작업을 자동 수집한다 (블로킹, Ctrl-C로
    끝냄).

    웹훅마다 새 트랜잭션을 열어:
    1. 작업을 조회한다. 열린 작업이 아니면(수집함·회수함) 검수자 확인 전에 무시한다 (204). 회수
       실행기의 CVAT 배정 해제가 일으킨 `update:job` 웹훅을 담당자 불일치(403)로 만들지 않는다.
    2. 검수자를 정한다 — 웹훅의 도구 사용자 ID가 아니라 작업 담당자다
       (`resolve_reviewer`). 담당자와 CVAT job 담당자가 맞지 않으면 수집하지 않는다. Label Studio
       작업은 주석을 만든·고친 사용자가 담당자의 Label Studio 계정(`label_studio.users`)이 아니면
       수집하지 않는다 (403, 경고 로그. 수집도 그 계정 주석만 받는다, F-REV-06).
    3. 서비스 계정(Label Studio 서비스 사용자)이 만든 주석이면 수집하지 않는다.
    4. `collect_task`로 수집한다. 실패는 트랜잭션을 롤백하고 HTTP 경계에 전달한다. 결과는
       `collect`와 같이 출력한다 (블러가 바뀌어 회수한 작업과 다시 만들 명령 포함, 서버 로그).
    5. 커밋 뒤 그 세션의 회수 대기를 실행한다. 대기가 남으면 `RevocationIncompleteError`를 던져
       503으로 답한다 (수집·회수 요청은 이미 커밋됐다. 도구가 재전송하지 않을 수 있어 cron
       `dlp review revoke --pending`이 보완한다).

    필요 환경 변수: `DLP_WEBHOOK_SECRET`(서명 검증), DB·S3·도구 접속 정보.
    반환: 서버가 끝나면 0.
    """
    ops_policy = load_ops_policy(repo_root())
    setup = _setup(args, ops_policy)
    engine = sa.create_engine(database_url(args.url))

    service_user = setup.label_studio.service_user_id() if setup.label_studio else None

    def on_collect(req: CollectRequest) -> None:
        """웹훅 한 건을 한 트랜잭션으로 수집한다. 실패를 성공 응답으로 바꾸지 않는다."""
        with engine.begin() as conn:
            task = get_review_task(conn, req.task_key)
            if task.status is not ReviewTaskStatus.OPEN:
                # 수집했거나 회수한 작업 (회수 실행기의 배정 해제 웹훅 포함)
                print(f"{req.task_key}: {task.status.value} 작업이라 수집하지 않음")
                return
            # 검수자는 웹훅의 도구 사용자 ID가 아니라 작업 담당자다
            reviewer = resolve_reviewer(
                req, task, service_user, ops_policy.cvat.users, ls_users=_ls_users(setup, task)
            )
            if reviewer is None:
                print(f"{req.task_key}: 서비스 계정의 주석이라 수집하지 않음")
                return
            outcome = collect_task(
                conn, req.task_key, setup, reviewer, datetime.now(UTC), ops_policy
            )
        # 서버 로그에도 수집 결과와 블러 변경으로 회수한 작업·다시 만들 명령을 남긴다
        print(f"{req.task_key} ← {reviewer}")
        _print_outcome(task, outcome, ai=setup.ai is not None)
        report = run_revocations(engine, task.session_id, setup)
        if print_revocations(report, task.session_id):
            left = len(report.remaining)
            raise RevocationIncompleteError(f"{task.session_id}: 회수 대기 {left}개")

    server = serve(args.port, _secret(), on_collect, policy=ops_policy.webhook)
    print(
        f"웹훅 대기: http://{ops_policy.webhook.bind_host}:{args.port}/webhooks/{{cvat,label_studio}}"
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.dispose()
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
    # 구 프로젝트의 웹훅도 유지하고, 등록된 사전별 새 화면의 프로젝트를 함께 등록한다.
    engine = sa.create_engine(database_url(getattr(args, "url", None)))
    try:
        with engine.connect() as conn:
            ontologies = [
                get_ontology(conn, v)
                for v in conn.execute(sa.select(ontology_versions.c.version)).scalars()
            ]
    finally:
        engine.dispose()
    cvat_names = [PRIVACY_PROJECT, SPATIAL_PROJECT]
    ls_names = [TEMPORAL_PROJECT]
    for ontology in ontologies:
        cvat_names += [
            ontology_project(base, ontology) for base in (PRIVACY_PROJECT, SPATIAL_PROJECT)
        ]
        ls_names.append(ontology_project(TEMPORAL_PROJECT, ontology))
    if setup.cvat is not None:
        for name in cvat_names:
            project = setup.cvat.find_project(name)
            if project:
                setup.cvat.add_webhook(int(project["id"]), f"{args.url_base}/webhooks/cvat", secret)
                print(f"CVAT {name}: 웹훅 등록")
    if setup.label_studio is not None:
        for name in ls_names:
            project = setup.label_studio.find_project(name)
            if project:
                setup.label_studio.add_webhook(
                    int(project["id"]), f"{args.url_base}/webhooks/label_studio", secret
                )
                print(f"Label Studio {name}: 웹훅 등록")
    return 0


def _ai_gate(conn: sa.Connection, session_id: str, ai: AiDisplayConfig | None) -> GateDecision:
    """required 모드의 배정 게이트 (`ai_display.required_gate`). 시간 초과면 waived 이벤트를 남긴다.

    required가 아니면(끔·advisory, 블러 계획) AI 표와 생애주기 기록을 읽지 않고 바로 진행한다
    (review.yaml `ai_review.mode` 설명: off는 AI 표를 읽지 않는다).
    required면 실행 상태를 읽기 전에 세션 행을 잠근다(`lock_session`, 계획과 같은 순서). waived
    INSERT가 세션 FK로 FOR KEY SHARE를 먼저 잡고 계획이 FOR UPDATE로 올리면, 같은 세션의 두 계획이
    서로를 기다려 교착이 난다. 먼저 잠그면 뒤 계획은 앞 계획이 커밋한 waived를 읽어 다시 쓰지
    않는다.
    부작용: 시간 초과로 진행할 때 `ai_review_run_events`에 waived 1행 (사유 `timeout: …`, 계획과
    같은 트랜잭션).
    """
    if ai is None or ai.policy.mode != "required":
        return required_gate(None, None, datetime.now(UTC), ai)
    lock_session(conn, session_id)
    states = latest_run_state(conn, [session_id])
    now = datetime.now(UTC)
    decision = required_gate(states.get(session_id), prelabeled_at(conn, session_id), now, ai)
    if decision.waive_timeout:
        insert_ai_run_event(
            conn,
            AiRunEvent(
                session_id=session_id, event=AiRunEventKind.WAIVED,
                reason=f"timeout: prelabeled 뒤 max_wait_h={ai.policy.max_wait_h:g}시간",
                actor=current_actor(), at=now,
            ),
        )  # fmt: skip
    return decision


def cmd_plan(args: argparse.Namespace) -> int:
    """`dlp review plan <세션> --reviewer A --reviewer B …`: 세션의 검수 배정을 계획한다 (WP12).

    인자:
        args.reviewer: 배정에 쓸 검수자 목록 (`--privacy`면 모두 `reviewers.privacy`에 있어야 한다).
        args.seed_golden: 오류 삽입 과제의 원천이 될 골든셋 버전들 (그 세션들의 라벨에 오류를
            심는다).
        args.privacy: 참이면 블러 검수 단위(스트림 전체, 전수)로 계획한다.
        args.seed: 표본 추출·배정 난수 시드 (같은 시드면 같은 계획, 결정적).

    AI 1차 검수 (WP17.6, `review.yaml ai_review`): 작업 라벨 계획에서 끔이 아니면 증가 프로필의
    유효한 판정으로 우선순위 사유(AI_*)를 더하고 지적된 라벨을 표본 묶음에서 뺀다. required면 계획
    전에 게이트를 본다: 허용 버전의 완료된 판정 실행(`dlp aireview run`)이 없고 건너뜀 기록도 없고
    prelabeled 뒤 `max_wait_h`도 지나지 않았으면 배정하지 않고 `AI_PENDING_EXIT`(6)로 끝낸다. 시간이
    지났으면 waived:timeout을 남기고 계획한다. 블러 계획(`--privacy`)에는 쓰지 않는다.

    반환: 0. 이미 계획된 단위는 새로 만들지 않는다(멱등) — 이때 "새 배정이 없습니다". required
    게이트로 미루면 6.
    부작용: 한 트랜잭션에서 `review_assignments`에 새 배정 INSERT (required 시간 초과면 waived
    이벤트도). 온톨로지는 세션 버전의 DB 등록 사전을 읽는다.
    """
    root = repo_root()
    policy = load_ops_policy(root)
    ai = None if args.privacy else ai_display_config(policy)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        gate = _ai_gate(conn, args.session_id, ai)
        if not gate.proceed:
            engine.dispose()
            print(
                f"{args.session_id}: AI 1차 검수(required)를 기다리며 배정을 미룹니다 — "
                f"{gate.reason}. dlp aireview run {args.session_id} 또는 dlp aireview waive "
                f"{args.session_id} --reason <사유> (prelabeled 뒤 {policy.ai_review.max_wait_h:g}"
                "시간이 지나면 자동으로 진행)"
            )
            return AI_PENDING_EXIT
        if gate.waive_timeout:
            print(f"{args.session_id}: {gate.reason} → AI 단계 건너뜀(waived:timeout) 기록")
        ontology = get_session_ontology(conn, args.session_id)
        pool: list[str] = []
        for version in args.seed_golden:
            pool += list(get_golden_set(conn, version).session_ids)
        created = plan_session(
            conn, args.session_id, args.reviewer, policy, ontology, seed=args.seed,
            now=datetime.now(UTC), seed_sessions=pool, privacy=args.privacy, ai=ai,
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
    """`dlp review assign <배정 ID> [--assignee <검수자>]`: 배정의 검수 도구 작업을 만든다.

    이미 만든 배정이면 기존 작업을 그대로 돌려준다 (배정 행을 `FOR UPDATE`로 잠가 동시 실행에도 한
    번만). 작업이 모두 회수된 배정이면 지금 블러본으로 다시 만든다 (ADR 0039).
    담당자 없는 배정(선임이 없어 담당자를 비워 둔 QA 배정 등)은 `--assignee`로 담당자를 정해야
    만든다 (`ops.runner.assign_reviewer`: 블러는 원본 접근 권한자, QA는 선임, 짝 배정과 다른 사람).
    없으면 아무것도 하지 않고 `NO_ASSIGNEE_EXIT`(1)로 끝낸다. 정한 담당자가 배정 규칙에 맞지
    않으면(선임이 아닌 QA 담당자, 짝 배정과 같은 담당자, 권한 없는 블러 담당자) 담당자를 바꾸지도
    작업을 만들지도 않고 이유와 고칠 곳을 출력해 같은 종료 코드로 끝낸다 (traceback 없음). QA
    배정의 규칙 거부(`ReviewerRuleError`)에만 `reviewers.senior`를 고치라는 안내를 붙인다: 이미
    담당자·작업이 있어 거부한 것은 정책을 고쳐도 풀리지 않는다.
    담당자가 이미 있는 배정의 담당자는 바꾸지 않는다 (F-REV-07, 회귀: 예전에는 "unassigned"를
    실제 담당자로 넣어 워터마크·라벨링 버킷 경로·검수자 ID가 모두 그 문자열이 됐다).
    작업 라벨 배정은 만들기 전에 그 세션의 회수 대기를 실행하고(sweep 포함), 세션에 대기가 하나라도
    남으면(배정 스트림이 아니어도) 만들지 않고 `REVOCATION_PENDING_EXIT`(4)로 끝낸다. 블러 배정은
    원본 프록시를 쓰므로 막지 않는다.
    부작용: `review_tasks` 쓰기, 검수 도구 API 호출, 블러 배정이면 원본 읽기(감사 기록),
    `--assignee`면 배정 담당자 갱신(작업 생성과 같은 트랜잭션).
    """
    policy = load_ops_policy(repo_root())
    setup = _setup(args, policy)
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        planned = get_assignment(conn, args.assignment_id)
    if planned.assignee is None and not args.assignee:
        engine.dispose()
        print(
            f"{args.assignment_id}: 담당자가 없는 배정입니다. "
            "--assignee <검수자>로 담당자를 정해야 작업을 만듭니다"
        )
        return NO_ASSIGNEE_EXIT
    if planned.label_kinds != ("blur_track",):
        report = run_revocations(engine, planned.session_id, setup, sweep=True)
        if print_revocations(report, planned.session_id):
            engine.dispose()
            print(f"{args.assignment_id}: 매체 회수가 끝나지 않아 작업을 만들지 않습니다")
            return REVOCATION_PENDING_EXIT
    refused: str | None = None
    hint = ""
    tasks: list[ReviewTask] = []
    with engine.begin() as conn:
        a = get_assignment(conn, args.assignment_id)
        try:
            if args.assignee:
                a = assign_reviewer(conn, a, args.assignee, policy)
        except (TaskError, AccessError) as exc:
            refused = str(exc)  # 쓰기 전에 거부됐다 (담당자 갱신 없음)
            # 정책 안내는 QA 배정 규칙 거부(선임 아님·짝과 같음)에만: 이미 담당자·작업이 있어 거부한
            # 것은 정책을 고쳐도 풀리지 않는다
            if isinstance(exc, ReviewerRuleError) and a.mode is ReviewMode.QA:
                hint = (
                    " — review.yaml reviewers.senior에 원래 담당자가 아닌 선임을 넣은 뒤 그 "
                    "선임으로 다시 실행하세요"
                )
        else:
            tasks = create_assignment_tasks(conn, a, setup, policy, datetime.now(UTC))
    engine.dispose()
    if refused is not None:
        print(f"{args.assignment_id}: 담당자를 정하지 못했습니다: {refused}{hint}")
        return NO_ASSIGNEE_EXIT
    for t in tasks:
        print(f"{t.task_key} ({t.mode.value}, {t.assignee}) {t.media_uri}")
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    """`dlp review qa [--seed]`: 끝난(DONE) 표준 배정 중 정책 비율만큼 QA 재검수 배정을 뽑는다.

    이미 있는 배정 ID는 다시 넣지 않는다 (`plan_qa`가 결정적 ID를 만들므로 재실행해도 중복 없음).
    선임(`reviewers.senior`)이 없거나 원래 담당자뿐이면 담당자 없이 넣고, 그 배정들은
    `reviewers.senior`에 원래 담당자가 아닌 선임을 넣은 뒤 `dlp review assign <배정> --assignee
    <그 선임>`으로 만들라고 출력한다 (F-REV-07, 같은 정책으로는 assign이 거부된다).
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
    for a in created:
        if a.assignee is None:
            print(f"  {a.assignment_id}: 담당자 없음 → review.yaml reviewers.senior에 원래 "
                  "담당자가 아닌 선임을 넣은 뒤 dlp review assign "
                  f"{a.assignment_id} --assignee <그 선임>")  # fmt: skip
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    """`dlp review queue [--limit]`: 열린(OPEN) 배정을 우선순위 높은 순(같으면 만든 순)으로
    출력한다. AI 1차 검수 사유(AI_*) 구간이 있으면 그 수를 덧붙인다 (WP17.6). 읽기 전용."""
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        queue = list_assignments(conn, status=AssignmentStatus.OPEN)
    engine.dispose()
    ai_reasons = {r.value for r in REASONS.values()} | {ReviewReason.AI_MISSING.value}
    for a in queue[: args.limit]:
        ai = sum(1 for f in a.flagged if f.reason.value in ai_reasons)
        note = f"  AI 지적 {ai}" if ai else ""
        print(f"{a.priority:8.2f}  {a.assignment_id}  {a.mode.value}  {a.assignee or '-'}{note}")
    return 0


def _iso_time(text: str) -> datetime:
    """ISO 시각 (argparse 형식 함수, 시간대 없는 값은 UTC). 잘못된 값은 사용법 오류다."""
    try:
        at = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ISO 시각이 아닙니다: {text}") from exc
    return at if at.tzinfo is not None else at.replace(tzinfo=UTC)


def _precision(text: str) -> float:
    """목표 정밀도 0~1 (argparse 형식 함수)."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"0~1 사이 수가 아닙니다: {text}") from exc
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"0~1 사이 수가 아닙니다: {text}")
    return value


def _ai_window(args: argparse.Namespace) -> tuple[datetime, datetime] | None:
    """`--ai`의 기간 [since, until) (없으면 처음부터 지금까지). AI 전용 인자를 --ai 없이 주면
    조용히 무시하지 않고 `SystemExit`로 알린다 (`--json`은 AI 절만 내므로 --ai가 필요하다)."""
    if not args.ai:
        extra = [
            name for name, on in (
                ("--json", args.json), ("--ai-thresholds", args.ai_thresholds is not None),
                ("--since", args.since is not None), ("--until", args.until is not None),
            ) if on
        ]  # fmt: skip
        if extra:
            raise SystemExit(f"{', '.join(extra)}는 --ai와 함께 씁니다")
        return None
    since = args.since or datetime(1970, 1, 1, tzinfo=UTC)
    until = args.until or datetime.now(UTC) + timedelta(seconds=1)
    if since >= until:
        raise SystemExit(
            f"--since({since.isoformat()})가 --until({until.isoformat()})보다 앞이어야 합니다"
        )
    return since, until


def cmd_quality(args: argparse.Namespace) -> int:
    """`dlp review quality`: 검수 품질 측정 리포트를 출력한다. 읽기 전용.

    - 오류 삽입 발견율: 검수자별 (발견한 삽입 오류 수 / 삽입한 수).
    - 이중 라벨 일치도: 키별 Cohen 카파, 구간 F1, 경계 F1.
    - 프리라벨 편향: 블라인드 과제와 일반 과제 결과 차이 (양수면 검수자가 프리라벨에 끌려감).
    - `--ai [--since ISO] [--until ISO]`: AI 1차 검수 판정 품질 (WP17.8,
      `dlp_review.ops.ai_quality`). 편향 없는 값은 대조군(holdout)이다. `--ai-thresholds P`는 목표
      정밀도 P를 만족하는 판정별 최소 확신도를 `review.yaml ai_review.show_min_confidence` 조안으로
      낸다. `--json`은 AI 절을 JSON으로 낸다 (`--ai-thresholds`를 함께 주면 조안을 `thresholds`
      키에 넣는다). `--json`·`--ai-thresholds`·`--since`·`--until`은
      `--ai`가 필요하다 (없으면 종료 1과 안내, 잘못된 시각·정밀도는 사용법 오류 종료 2).
    """
    policy = load_ops_policy(repo_root())
    window = _ai_window(args)
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        report = quality_report(conn, policy, ai_window=window)
    engine.dispose()
    if args.json:
        data = ai_quality_json(report.ai) if report.ai else {}
        if report.ai is not None and args.ai_thresholds is not None:
            data["thresholds"] = thresholds_json(report.ai, args.ai_thresholds)
        print(json.dumps(data, ensure_ascii=False))
        return 0
    for d in report.detection:
        print(f"오류 삽입 발견율 {d.reviewer}: {d.detected}/{d.injected} ({d.rate:.0%})")
    for key, agr in report.double.items():
        print(
            f"이중 라벨 일치 {key}: 카파 {agr.kappa:.3f}, "
            f"구간 F1 {agr.segment_f1:.3f}, 경계 F1 {agr.boundary_f1:.3f}"
        )
    for key, bias in report.blind_bias.items():
        print(f"프리라벨 편향 {key}: {bias:+.3f} (양수면 검수자가 프리라벨에 끌려감)")
    if report.ai is not None:
        print_ai_quality(report.ai, policy.sampling.max_defect_ratio)
        if args.ai_thresholds is not None:
            print_thresholds(report.ai, args.ai_thresholds)
    return 0


def _check_orphan_args(args: argparse.Namespace, selection: list[str]) -> None:
    """`revoke --orphans [--apply]` 인자 조합. 맞지 않으면 `SystemExit`."""
    if not args.orphans:
        raise SystemExit("--apply는 --orphans와만 씁니다")
    if args.apply and args.dry_run:
        raise SystemExit("--apply와 --dry-run을 함께 쓸 수 없습니다 (--orphans는 기본이 보고만)")
    others = {
        "--pending": args.pending,
        "--mark-tool-gone": bool(args.mark_tool_gone),
        "--note": bool(args.note),
    }
    extra = selection + [name for name, on in others.items() if on]
    if extra:
        raise SystemExit(f"--orphans는 {', '.join(extra)}와 함께 쓸 수 없습니다")


def _check_revoke_args(args: argparse.Namespace) -> None:
    """`revoke` 인자 조합을 확인한다. 일부 인자가 조용히 무시되는 조합이면 `SystemExit`.

    특히 `--dry-run`은 미리보기가 있는 경로(세션, `--pending`, `--orphans`)에서만 받는다.
    `--mark-tool-gone`은 표시만 하는 쓰기 명령이라 `--dry-run`과 함께 쓰면 실제로 표시되는 것을
    막는다. `--orphans`는 기본이 보고만이고 `--apply`일 때만 지운다 (`--apply`와 `--dry-run`을 함께
    쓰면 거부한다). 세션은 고를 수 있고 다른 선택 인자와는 쓰지 않는다.
    """
    chosen = {
        "--task": bool(args.task), "--all": args.all, "--stream": bool(args.stream),
        "--include-legacy": args.include_legacy, "--reason": bool(args.reason),
    }  # fmt: skip
    selection = [name for name, on in chosen.items() if on]
    if args.apply or args.orphans:
        _check_orphan_args(args, selection)
    elif args.mark_tool_gone:
        others = {
            "<세션>": bool(args.session_id),
            "--pending": args.pending,
            "--dry-run": args.dry_run,
        }
        extra = selection + [name for name, on in others.items() if on]
        if extra:
            raise SystemExit(f"--mark-tool-gone은 {', '.join(extra)}와 함께 쓸 수 없습니다")
        if not args.note:
            raise SystemExit("--mark-tool-gone에는 --note(사유)가 필요합니다")
    elif args.pending:
        extra = selection + (["<세션>"] if args.session_id else [])
        if extra:
            raise SystemExit(f"--pending은 모든 세션의 대기를 실행합니다 ({', '.join(extra)} 불가)")
    elif not args.session_id:
        raise SystemExit("세션을 주거나 --pending을 쓰세요")
    elif bool(args.task or args.all) != bool(args.reason):
        raise SystemExit("--task·--all에는 --reason(사유)이 필요하고, --reason은 둘과만 씁니다")


def _check_task_keys(conn: sa.Connection, session_id: str, keys: list[str]) -> None:
    """`--task` 키마다 이 세션의 회수할 수 있는 작업인지 확인한다. 하나라도 아니면 `SystemExit`.

    없는 키, 다른 세션의 작업, 블러 검수 작업(원본 권한자 화면, 회수 대상 아님)은 요청이 조용히
    빠지므로 쓰기 전에 거부한다. 이미 회수 요청된 작업은 그렇다고 알리고 계속한다.
    """
    problems: list[str] = []
    for key in keys:
        try:
            task = get_review_task(conn, key)
        except NoResultFound:
            problems.append(f"{key}: 없는 작업")
            continue
        if task.session_id != session_id:
            problems.append(
                f"{key}: 세션 {task.session_id}의 작업 (dlp review revoke {task.session_id} --task)"
            )
        elif task.stage is ReviewStage.PRIVACY:
            problems.append(f"{key}: 블러 검수 작업이라 회수 대상이 아님 (원본 권한자 화면)")
        elif task.revoke_requested_at is not None:
            print(f"{key}: 이미 회수 요청됨 ({task.revoke_reason}, {task.revoke_requested_at})")
    if problems:
        raise SystemExit("--task 회수를 요청하지 않았습니다:\n  " + "\n  ".join(problems))


def _legacy_note(task: ReviewTask, history: list[LabelRecord], policy: PrivacyPolicy) -> str:
    """구 작업(렌더 해시 없음)의 참고용 추정: 작업을 만든 시각까지의 이력으로 복원한 운영 블러의
    렌더 해시가 지금과 같은가. 시계·경합 때문에 틀릴 수 있어 결정은 운영자가 한다."""
    then = [x for x in history if x.created_at <= task.created_at]
    same = render_hash(operational_blur(then, task.stream_id), policy) == render_hash(
        operational_blur(history, task.stream_id), policy
    )
    return f" (추정: 생성 시점 블러가 지금과 {'같음' if same else '다름'})"


def _revoke_dry_run(engine: sa.Engine, args: argparse.Namespace) -> int:
    """`revoke <세션> --dry-run`: 쓰지 않고 고를 작업과 이유, 지금 대기를 출력한다. 0.

    실제 실행과 같이 수동 선택(`--task`·`--all`)과 sweep(해시가 다른 작업, `--include-legacy`면
    구 작업 포함)을 따로 보인다. 구 작업에는 `_legacy_note` 추정을 붙인다.
    """
    sid = args.session_id
    policy = load_privacy_policy(repo_root())
    streams = set(args.stream) if args.stream else None
    with engine.connect() as conn:
        _check_task_keys(conn, sid, args.task)
        manual = (
            stale_media_tasks(
                conn, sid, policy, streams=streams, force=args.all, task_keys=tuple(args.task)
            )
            if args.task or args.all
            else []
        )
        picked = {m.task.task_key for m in manual}
        sweep = stale_media_tasks(conn, sid, policy, streams=streams, legacy=args.include_legacy)
        swept = [m for m in sweep if m.task.task_key not in picked]
        history = get_labels(conn, sid, kinds=["blur_track"])
        pending = pending_revocations(conn, sid)
    for kind, item in [*(("수동", m) for m in manual), *(("sweep", m) for m in swept)]:
        t = item.task
        note = _legacy_note(t, history, policy) if item.cause == "legacy" else ""
        head = f"{t.task_key:<20} {t.stream_id:<12} {t.status.value:<9} {kind:<5} {item.cause:<12}"
        print(f"{head} {t.media_uri}{note}")
    total = len(manual) + len(swept)
    print(f"{sid}: 회수 요청 대상 {total}개, 지금 회수 대기 {len(pending)}개 (쓰지 않음)")
    return 0


def _revoke_mark_gone(engine: sa.Engine, args: argparse.Namespace) -> int:
    """`revoke --mark-tool-gone KEY --note TEXT`: 도구 작업 수동 완료 표시. 대기가 남으면 4."""
    with engine.begin() as conn:
        task = get_review_task(conn, args.mark_tool_gone)
        done = mark_tool_gone(
            conn, task.task_key, note=args.note, actor=current_actor(), now=datetime.now(UTC)
        )
        left = [t.task_key for t in pending_revocations(conn, task.session_id)]
    print(f"{task.task_key}: {'도구 작업 완료로 표시' if done else '표시할 회수 대기가 없음'}")
    if left:
        keys = ", ".join(left[:5])
        print(f"  회수 대기 {len(left)}개: {keys} (dlp review revoke {task.session_id})")
        return REVOCATION_PENDING_EXIT
    return 0


def _revoke_pending(engine: sa.Engine, args: argparse.Namespace) -> int:
    """`revoke --pending [--dry-run]`: 모든 세션의 회수 대기를 실행한다 (미리보기는 쓰지 않는다).

    미리보기 반환: 0. 실행 반환: 대기가 남으면 `REVOCATION_PENDING_EXIT`(4).
    """
    if not args.dry_run:
        return print_revocations(run_revocations(engine, None, url=args.url), None)
    with engine.connect() as conn:
        pending = pending_revocations(conn, None)
    for t in pending:
        channels = (("매체", t.media_revoked_at), ("도구", t.tool_revoked_at))
        left = [name for name, done in channels if done is None]
        print(
            f"{t.task_key:<20} {t.session_id:<12} {t.stream_id:<12} 남은 채널 "
            f"{'·'.join(left):<6} {t.revoke_reason}"
        )
    print(f"모든 세션: 회수 대기 {len(pending)}개 (쓰지 않음)")
    return 0


def _print_orphans(report: OrphanReport, *, applied: bool, min_age_s: int) -> int:
    """고아 찾기·지우기 결과를 출력한다. 깨끗하면 0, 아니면 `REVOCATION_PENDING_EXIT`(4).

    실패와 확인하지 못한 채널이 먼저 나온다.
    """
    for line in report.errors:
        print(f"  ! {line}")
    for line in report.unscanned:
        print(f"  ! 확인하지 못함: {line}")
    done = set(report.deleted)
    for o in report.found:
        mark = "지움" if o.ref in done else "남음"
        when = o.created_at.isoformat() if o.created_at else "?"
        print(f"{mark} {o.channel:<12} {o.ref} ({o.session_id}, {o.reason}, {when}) {o.label}")
    for line in report.warnings:
        print(f"  경고: {line}")
    print(
        f"고아 {len(report.found)}개, 지움 {len(report.deleted)}개, 스냅숏 "
        f"{len(report.snapshots)}개, 나이 하한({min_age_s}초)보다 어려 보지 않음 "
        f"{len(report.young)}개"
    )
    if report.busy:
        print(f"  작업 생성 중이라 건너뜀 {len(report.busy)}개 (다시 실행하세요)")
    if report.remaining and not applied:
        print("  지우지 않았습니다 (보고만). 지우려면 dlp review revoke --orphans --apply")
    return 0 if report.clean else REVOCATION_PENDING_EXIT


def _revoke_orphans(engine: sa.Engine, args: argparse.Namespace) -> int:
    """`revoke --orphans [<세션>] [--apply]`: 고아 도구 작업·라벨링 버킷 사본을 찾고(지운다).

    나이 하한은 고른 검수 정책(`DLP_REVIEW_POLICY_FILE` 또는 review.yaml)의 `orphans.min_age_s`다.
    도구 클라이언트는 회수 전용 설정으로 만든다 (CVAT 로그인 실패는 경고, 그 채널은 확인하지 못함).
    """
    policy = load_ops_policy(repo_root())
    setup, warnings = review_setup("review.revoke", args.url, revocation_only=True, policy=policy)
    now = datetime.now(UTC)
    report = find_orphans(
        engine, labeling=setup.labeling, cvat=setup.cvat, label_studio=setup.label_studio,
        min_age=timedelta(seconds=policy.orphans.min_age_s), now=now, session_id=args.session_id,
    )  # fmt: skip
    report.warnings[:0] = warnings
    if args.apply:
        delete_orphans(
            engine, report, labeling=setup.labeling, raw_bucket=setup.raw.bucket, cvat=setup.cvat,
            label_studio=setup.label_studio, now=now,
        )  # fmt: skip
    return _print_orphans(report, applied=args.apply, min_age_s=policy.orphans.min_age_s)


def _revoke_session(engine: sa.Engine, args: argparse.Namespace) -> int:
    """`revoke <세션>`: 수동 요청(`--task`·`--all`) → sweep 요청을 한 트랜잭션에 남기고 실행한다.

    수동 요청의 사유는 `manual:<행위자>:<사유>`, sweep(해시가 지금과 다른 작업, `--include-legacy`면
    구 작업 포함)의 사유는 `sweep`이다. 둘을 따로 불러 낡은 작업이 수동 사유로 기록되지 않게 한다.
    """
    if args.dry_run:
        return _revoke_dry_run(engine, args)
    sid = args.session_id
    policy = load_privacy_policy(repo_root())
    streams = set(args.stream) if args.stream else None
    now = datetime.now(UTC)
    with engine.begin() as conn:
        _check_task_keys(conn, sid, args.task)
        manual: list[ReviewTask] = []
        if args.task or args.all:
            manual = request_media_revocation(
                conn, sid, policy, now, reason=f"manual:{current_actor()}:{args.reason}",
                streams=streams, force=args.all, task_keys=tuple(args.task),
            )  # fmt: skip
        swept = request_media_revocation(
            conn, sid, policy, now, reason="sweep", streams=streams, legacy=args.include_legacy
        )
    for kind, requested in (("수동", manual), ("sweep", swept)):
        for t in requested:
            print(f"회수 요청 {t.task_key} ({t.stream_id}, {t.status.value}, {kind}) {t.media_uri}")
    report = run_revocations(engine, sid, url=args.url)
    return print_revocations(report, sid)


def cmd_revoke(args: argparse.Namespace) -> int:
    """`dlp review revoke [<세션>] …`: 블러가 바뀐 뒤 이미 배포한 라벨링 매체를 회수한다 (ADR 0039).

    - `<세션>`: 세션 잠금 트랜잭션에서 회수를 요청하고 커밋 뒤 대기를 실행한다. 늘 sweep을 한다:
      렌더 해시가 지금과 다른 작업(사유 `sweep`). `--stream`으로 스트림을, `--include-legacy`로 렌더
      해시 없는 구 작업을 sweep에 더한다. 그 위에 수동 요청을 더할 수 있다: `--task KEY`(반복)는 그
      작업(과 같은 매체를 쓰는 작업)만, `--all`은 해시와 무관하게 그 스트림의 모든 비프라이버시
      작업을 요청한다 (사고 대응). 수동 요청에는 `--reason`이 필요하고 사유는
      `manual:<행위자>:<사유>`다. `--task` 키가 이 세션의 회수할 수 있는 작업이 아니면(없음, 다른
      세션, 블러 검수 작업) 아무것도 쓰지 않고 거부한다.
    - `--dry-run`: 쓰지 않고 대상과 이유를 출력한다 (`<세션>` 또는 `--pending`과 함께).
    - `--pending`(세션 없이): 대기가 있는 모든 세션을 실행한다 (cron, 10분마다 권장).
    - `--mark-tool-gone KEY --note TEXT`: 도구 인스턴스 폐기 등으로 확인할 수 없는 도구 작업을
      사유와 함께 완료로 표시한다 (행위자 = `current_actor()`). 다른 선택 인자·`--dry-run`과 함께
      쓰지 않는다.
    - `--orphans [<세션>] [--apply]`: DB 기록 없이 남은 도구 작업·라벨링 버킷 사본(고아)을 보고하고,
      `--apply`면 세션을 잠근 뒤 다시 확인해 지운다 (`dlp_review.orphans`, F-REV-15). 고아가 남거나
      확인하지 못한 채널·작업 생성 중이라 건너뛴 세션이 있으면 4.

    반환: 대기가 남으면 `REVOCATION_PENDING_EXIT`(4), 아니면 0. `<세션>`·`--pending`의
    `--dry-run`은 늘 0이다. `--orphans`(기본 보고만, `--dry-run`을 붙여도 같다)는 고아가 남으면
    보고만이어도 4다 (cron 경보용). 인자 조합이 맞지 않거나 `--task` 키를 요청할 수 없으면
    `SystemExit`.
    부작용: `review_tasks` 회수 요청·완료 표시, 라벨링 버킷 사본 삭제·스냅숏, CVAT·Label Studio
    작업 배정 해제·삭제 (되돌릴 수 없다).
    """
    _check_revoke_args(args)
    engine = sa.create_engine(database_url(args.url))
    try:
        if args.orphans:
            return _revoke_orphans(engine, args)
        if args.mark_tool_gone:
            return _revoke_mark_gone(engine, args)
        if args.pending:
            return _revoke_pending(engine, args)
        return _revoke_session(engine, args)
    finally:
        engine.dispose()


def cmd_refresh_urls(args: argparse.Namespace) -> int:
    """`dlp review refresh-urls [--session S …]`: 열린 Label Studio 작업의 서명 URL을 다시 서명한다.

    Label Studio 작업 data의 영상·시계열 URL은 `review.yaml media.signed_url_ttl_s`(기본 7일) 뒤
    만료된다. 이 명령을 그보다 짧은 주기(cron, 하루 한 번 권장)로 돌려 열린 작업만 라벨러 자격
    증명으로 다시 서명한다 (F-REV-08, `dlp_review.refresh`). 수집·회수한 작업, 매체가 지금 블러와
    다른 작업, 렌더 해시 없는 구 작업, 사용 중지·승인 보류·승인 아님 세션의 작업은 갱신하지 않는다
    (건너뜀에 `dlp review revoke <세션> …`을 안내한다). CVAT에는 접속하지 않는다.

    인자: args.session — 이 세션만 (반복). 없으면 열린 Label Studio 작업이 있는 모든 세션.
    반환: 0, 갱신하지 못한 작업이 있으면 1 (cron 경보).
    부작용: 세션마다 짧은 트랜잭션에서 세션 행을 잠그고 Label Studio 작업 data를 바꾼다 (DB 쓰기
    없음, 원본 버킷 읽기 없음).
    """
    policy = load_ops_policy(repo_root())
    setup = _setup(args, policy)
    engine = sa.create_engine(database_url(args.url))
    failed = 0
    try:
        with engine.connect() as conn:
            sessions = list(args.session) or sessions_with_open_ls_tasks(conn)
        for sid in sessions:
            with engine.begin() as conn:
                report = refresh_session_urls(conn, sid, setup)
            for key in report.refreshed:
                print(f"{key}: 서명 URL 갱신 ({policy.media.signed_url_ttl_s}초)")
            for line in report.skipped:
                print(f"  건너뜀 {line}")
            for line in report.errors:
                print(f"  ! {line}")
            failed += len(report.errors)
    finally:
        engine.dispose()
    print(f"세션 {len(sessions)}개, 갱신하지 못한 작업 {failed}개")
    return 1 if failed else 0


def cmd_ui_theme(args: argparse.Namespace) -> int:
    """CVAT 테마 색 생성 파일(`ui_theme.TOKENS_FILE`)을 `config/review_ui.yaml`과 맞춘다.

    인자: args.check — 참이면 쓰지 않고 비교만 한다.
    반환: 0 (썼거나 이미 같음), 1 (`--check`에서 다름).
    부작용: `--check`가 아니면 생성 파일을 덮어쓴다 (CVAT에 반영하려면 `make cvat-up`으로 테마
    이미지를 다시 빌드한다).
    """
    root = repo_root()
    path = root / TOKENS_FILE
    text = cvat_tokens_css(load_ui_theme(root))
    current = path.read_text("utf-8") if path.exists() else None
    if args.check:
        if current != text:
            print(f"{TOKENS_FILE}가 config/review_ui.yaml과 다릅니다 (make ui-theme)")
            return 1
        print(f"{TOKENS_FILE}: 최신")
        return 0
    path.write_text(text, "utf-8")
    print(f"{TOKENS_FILE} 갱신 (CVAT 반영: make cvat-up)")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`review` 하위 명령 묶음을 등록한다 (도구 연동 7개 + 운영 5개 + 화면 테마 1개)."""
    review = sub.add_parser("review", help="검수 도구 연동")
    rsub = review.add_subparsers(dest="review_command", required=True)
    hints_cmds.add_review_commands(rsub)  # dlp review hints (ADR 0055)

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
    create.add_argument(
        "--stream", action="append", default=[], help="이 영상 스트림만 (반복, 라벨링)"
    )
    create.add_argument(
        "--tool",
        choices=[t.value for t in ReviewTool],
        help="이 도구만 (라벨링, 회수 뒤 다시 만들기)",
    )
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

    rv = rsub.add_parser(
        "revoke", help="블러가 바뀐 뒤 이미 배포한 라벨링 매체 회수 (대기가 남으면 종료 코드 4)"
    )
    rv.add_argument("session_id", nargs="?")
    rv.add_argument("--stream", action="append", default=[], help="이 영상 스트림만 (반복)")
    rv.add_argument(
        "--task", action="append", default=[], help="이 세션의 이 작업도 요청 (반복, --reason)"
    )
    rv.add_argument("--all", action="store_true", help="해시와 무관하게 모두 요청 (사고 대응)")
    rv.add_argument("--include-legacy", action="store_true", help="렌더 해시 없는 구 작업 포함")
    rv.add_argument(
        "--dry-run", action="store_true", help="쓰지 않고 대상·대기만 출력 (세션 또는 --pending)"
    )
    rv.add_argument("--reason", help="수동 요청 사유 (--task·--all에 필수)")
    rv.add_argument("--pending", action="store_true", help="대기가 있는 모든 세션 실행 (cron)")
    rv.add_argument(
        "--mark-tool-gone", metavar="KEY", help="도구 작업 수동 완료 표시 (--note, 단독으로)"
    )
    rv.add_argument("--note", help="--mark-tool-gone 사유")
    rv.add_argument(
        "--orphans", action="store_true",
        help="DB 기록 없이 남은 도구 작업·라벨링 버킷 사본 보고 (기본 보고만, --apply면 지움)",
    )  # fmt: skip
    rv.add_argument("--apply", action="store_true", help="--orphans가 찾은 고아를 지운다")
    rv.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    rv.set_defaults(func=cmd_revoke)

    ru = rsub.add_parser(
        "refresh-urls", help="열린 Label Studio 작업의 서명 URL 다시 서명 (cron, 하루 한 번)"
    )
    ru.add_argument("--session", action="append", default=[], help="이 세션만 (반복)")
    ru.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    ru.set_defaults(func=cmd_refresh_urls, no_cvat=True)

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
    asg.add_argument(
        "--assignee",
        help="담당자 없는 배정에 정할 담당자 (블러: 원본 접근 권한자, QA: 선임, 짝 배정과 다른 "
        "사람)",
    )
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
    ql.add_argument("--ai", action="store_true", help="AI 1차 검수 판정 품질도 낸다 (WP17.8)")
    ql.add_argument(
        "--since",
        type=_iso_time,
        help="--ai 기간 시작 (ISO 시각, 시간대 없으면 UTC, 기본: 처음부터)",
    )
    ql.add_argument("--until", type=_iso_time, help="--ai 기간 끝 (ISO 시각, 기본: 지금)")
    ql.add_argument(
        "--ai-thresholds", type=_precision, metavar="PRECISION",
        help="목표 정밀도(0~1)를 만족하는 판정별 최소 확신도 조안 (--ai와 함께)",
    )  # fmt: skip
    ql.add_argument("--json", action="store_true", help="AI 절을 JSON으로 (--ai와 함께)")
    ql.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    ql.set_defaults(func=cmd_quality)

    ut = rsub.add_parser(
        "ui-theme", help="검수 화면 테마(config/review_ui.yaml)로 CVAT 색 파일 생성"
    )
    ut.add_argument("--check", action="store_true", help="쓰지 않고 다르면 1로 끝낸다")
    ut.set_defaults(func=cmd_ui_theme)
