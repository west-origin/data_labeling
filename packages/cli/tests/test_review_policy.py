"""`dlp review` 명령은 고른 검수 정책(`DLP_REVIEW_POLICY_FILE`)을 한 출처로 쓴다 (F-REV-01).

저장소·도구 클라이언트는 만들지 않는다 (`review_config`의 생성자를 Mock으로 바꾼다). DB가 필요한
명령은 메모리 SQLite 엔진을 열기만 하고, 도구·DB를 쓰는 라이브러리 함수는 호출 인자만 받아 둔다.

정답 근거: 시험이 정책 파일에 직접 적은 CVAT 계정 연결(`cvat.users`), 원본 검수자
(`reviewers.privacy`), 매체 화질(`media.watermark_crf`), 웹훅 바인드 주소(`webhook.bind_host`)가
작업 생성·수집·웹훅 서버가 보는 값과 같아야 한다. Git의 기본 `review.yaml`은 계정 연결과 원본
검수자가 비어 있고 바인드 주소도 다르다. 한 명령 안에서는 정책을 한 번만 읽어 `ReviewSetup`과
명령이 같은 객체를 본다 (`is`로 확인한다). 매체 회수는 검수 정책 값을 쓰지 않으므로 정책 파일을
읽지 않는다: 파일이 없거나 깨져도 회수 대기를 실행한다.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import sqlalchemy as sa
import yaml

from dlp_cli import review_cmds, review_config
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.policy import load_policy as load_default_policy
from dlp_review.refresh import RefreshReport
from dlp_review.revoke import RevocationReport
from dlp_review.tasks import ReviewSetup, TaskError
from dlp_review.webhook import CollectRequest, ReviewerMismatchError
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    insert_review_task,
    insert_session,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

# 시험이 고른 정책 파일에 적는 값 (Git 기본 정책과 다르다)
USERS = {"pilot-privacy": "cvat-privacy", "pilot-labeler": "cvat-labeler"}
LS_USERS = {"pilot-labeler": "pilot-labeler@example.invalid"}
PRIVACY_REVIEWERS = ("pilot-privacy",)
WATERMARK_CRF = 31
BIND_HOST = "10.20.30.40"
# 깨진 검수 정책 파일: 없음(FileNotFoundError), 필수 절 없음(ValidationError), YAML 문법 오류
BROKEN_POLICIES = {"missing": None, "invalid": "version: 1\n", "syntax": "media: [\n"}


@pytest.fixture
def selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewOpsPolicy:
    """기본 review.yaml을 고쳐 별도 파일로 두고 `DLP_REVIEW_POLICY_FILE`로 고른다. 기대 정책."""
    root = repo_root()
    data = yaml.safe_load((root / "config/policies/review.yaml").read_text("utf-8"))
    data["reviewers"]["privacy"] = list(PRIVACY_REVIEWERS)
    data["cvat"]["users"] = dict(USERS)
    data["label_studio"]["users"] = dict(LS_USERS)
    data["media"]["watermark_crf"] = WATERMARK_CRF
    data["webhook"]["bind_host"] = BIND_HOST
    path = tmp_path / "review.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(path))
    default = load_default_policy(root)
    # 전제: 기본 정책(라이브러리 로더)은 이 값들을 갖지 않는다
    assert default.cvat.users == {} and default.media.watermark_crf != WATERMARK_CRF
    assert default.label_studio.users == {}
    assert default.webhook.bind_host != BIND_HOST
    return review_config.load_review_policy(root)


@pytest.fixture(params=sorted(BROKEN_POLICIES))
def broken(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """`DLP_REVIEW_POLICY_FILE`이 깨진 파일을 고른다 (종류 이름을 돌려준다)."""
    kind: str = request.param
    path = tmp_path / f"{kind}.yaml"
    text = BROKEN_POLICIES[kind]
    if text is not None:
        path.write_text(text, "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(path))
    return kind


@pytest.fixture(autouse=True)
def no_clients(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    """저장소·도구 클라이언트를 만들지 않는다 (네트워크·DB 없이 정책 연결만 본다). 이름 → 대역."""
    fakes = {name: Mock() for name in ("raw_store", "S3Store", "CvatClient", "LabelStudioClient")}
    for name, fake in fakes.items():
        monkeypatch.setattr(review_config, name, fake)
    return fakes


def _args(command: str, **kw: Any) -> argparse.Namespace:
    """`dlp review <command>` 인자. DB는 메모리 SQLite (엔진을 열기만 한다)."""
    return argparse.Namespace(review_command=command, url="sqlite://", **kw)


def _same_source(setup: ReviewSetup, policy: ReviewOpsPolicy, expected: ReviewOpsPolicy) -> None:
    """설정과 명령이 한 번 읽은 같은 정책 객체를 보고, 그 값이 고른 파일의 값이다."""
    assert policy == expected
    assert setup.cvat_policy() is policy.cvat
    assert setup.label_studio_policy() is policy.label_studio
    assert setup.media_policy() is policy.media


def test_setup_common_path_carries_selected_policy(selected: ReviewOpsPolicy) -> None:
    """`_setup` 공통 경로: 설정의 CVAT 계정 연결·매체 정책이 고른 파일의 값이다 (기본 로더 아님).

    create·collect·serve·assign·register-webhooks가 이 함수를 부르고, 명령 이름은 원본 감사 기록의
    목적 문자열에만 쓰여 명령마다 갈리지 않는다. 명령별로 정책을 함께 쓰는 연결은 아래 명령 시험이
    본다. 회귀: 예전에는 `ReviewSetup`에 정책을 넘기지 않아 처음 쓸 때 Git의 review.yaml을 읽었다.
    그래서 파일럿의 `dlp review create --stage privacy --assignee pilot-privacy`가 "CVAT 계정이
    없습니다"로 실패했다.
    """
    setup = review_cmds._setup(_args("create"))  # pyright: ignore[reportPrivateUsage]
    assert setup.cvat_policy() == selected.cvat
    assert setup.cvat_policy().users == USERS
    assert setup.label_studio_policy().users == LS_USERS
    assert setup.media_policy().watermark_crf == WATERMARK_CRF


def test_revocation_setup_does_not_read_review_policy(
    broken: str, no_clients: dict[str, Mock]
) -> None:
    """매체 회수가 스스로 만드는 설정(`revocation_only`)은 검수 정책 파일을 읽지 않는다.

    회수 실행기는 정책 값을 쓰지 않는다. 깨진 파일 때문에 회수 설정이 실패하면 낡은 라벨링 매체가
    지워지지 않는다. 정책 필드는 비어 있다 (작업 생성·수집에 쓰는 설정이 아니다).
    """
    setup, _ = review_config.review_setup("privacy.detect", None, revocation_only=True)
    assert setup.cvat_config is None and setup.media is None
    assert setup.labeling is no_clients["S3Store"].from_env.return_value


def _pending_db(tmp_path: Path) -> sa.Engine:
    """회수 요청이 남은 라벨링 작업(s001, cvat:2)이 있는 시간대 보존 SQLite 파일 DB."""
    engine = sqlite_db(f"sqlite:///{tmp_path / 'revoke.db'}")
    task = ReviewTask(
        task_key="cvat:2", tool=ReviewTool.CVAT, external_id="2", session_id="s001",
        stream_id="bodycam", stage=ReviewStage.LABELING,
        media_uri="s3://dlp-labeling/sessions/s001/review/labeler01/bodycam.mp4",
        label_kinds=("box_track",), created_at=FIXED_TIME, assignee="labeler01",
    )  # fmt: skip
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session("s001"))
        insert_review_task(conn, task)
        request_task_revocation(conn, "cvat:2", reason="privacy_detect", at=FIXED_TIME)
    return engine


@pytest.mark.parametrize("session_id", ["s001", None])
def test_revocations_run_with_broken_review_policy(
    session_id: str | None,
    broken: str,
    no_clients: dict[str, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """검수 정책 파일이 깨져도 회수 대기를 실행한다 (`dlp privacy detect|render` 뒤 회수,
    `dlp review revoke <세션>`은 세션, cron `revoke --pending`은 None).

    회귀: 회수 설정이 고른 정책을 엄격히 읽어 FileNotFoundError·ValidationError로 끝났고, 라벨링
    버킷 사본이 지워지지 않았다 (고치기 전에는 정책을 읽지 않던 경로였다).
    """
    execute = Mock(return_value=RevocationReport())
    monkeypatch.setattr(review_config, "execute_media_revocations", execute)
    engine = _pending_db(tmp_path)
    review_config.run_revocations(engine, session_id, purpose="privacy.detect")
    execute.assert_called_once()
    assert execute.call_args.args[1] == session_id
    assert execute.call_args.kwargs["labeling"] is no_clients["S3Store"].from_env.return_value


def test_broken_selected_policy_fails_before_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """고른 파일이 없으면 기본 정책으로 되돌리지 않고, 도구에 접속하기 전에 실패한다."""
    cvat = Mock()
    monkeypatch.setattr(review_config, "CvatClient", cvat)
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        review_cmds._setup(_args("create"))  # pyright: ignore[reportPrivateUsage]
    cvat.from_env.assert_not_called()


def _capture(store: dict[str, Any], result: Any = None) -> Callable[..., Any]:
    """라이브러리 함수 대역: 받은 위치·키워드 인자를 store에 남기고 result를 돌려준다."""

    def fake(*args: Any, **kw: Any) -> Any:
        store["args"], store["kw"] = args, kw
        return result

    return fake


def test_create_privacy_uses_selected_accounts_and_reviewers(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """블러 검수 작업 생성이 고른 정책의 원본 검수자와 CVAT 계정 연결을 함께 쓴다."""
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "create_privacy_tasks", _capture(got, []))
    args = _args(
        "create", session_id="s001", stage="privacy", assignee="pilot-privacy",
        no_cvat=False, no_ls=False, stream=[], tool=None,
    )  # fmt: skip
    assert review_cmds.cmd_create(args) == 0
    setup: ReviewSetup = got["args"][2]
    assert got["kw"]["privacy_reviewers"] == PRIVACY_REVIEWERS
    assert setup.cvat_policy().users == USERS


def test_assign_passes_one_policy_to_setup_and_runner(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dlp review assign`(ops runner)이 작업 생성에 넘기는 정책과 설정의 정책이 같은 출처다."""
    planned = SimpleNamespace(label_kinds=("blur_track",), session_id="s001", assignee="p1")
    monkeypatch.setattr(review_cmds, "get_assignment", Mock(return_value=planned))
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "create_assignment_tasks", _capture(got, []))
    assert review_cmds.cmd_assign(_args("assign", assignment_id="a1", assignee=None)) == 0
    _, _, setup, policy, _ = got["args"]
    _same_source(setup, policy, selected)


def test_assign_without_assignee_needs_explicit_reviewer(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-REV-07: 담당자 없는 배정은 `--assignee` 없이는 작업을 만들지 않고 종료 코드 1로 알린다.

    `--assignee`를 주면 같은 정책으로 담당자를 정한 배정(`assign_reviewer` 결과)으로 작업을 만든다.
    회귀: 예전에는 "unassigned"를 실제 담당자로 넣어 워터마크·경로·검수자 ID가 그 문자열이 됐다.
    """
    planned = SimpleNamespace(label_kinds=("action",), session_id="s001", assignee=None)
    monkeypatch.setattr(review_cmds, "get_assignment", Mock(return_value=planned))
    revocations = Mock(return_value=RevocationReport())
    monkeypatch.setattr(review_cmds, "run_revocations", revocations)
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "create_assignment_tasks", _capture(got, []))
    args = _args("assign", assignment_id="a1:qa", assignee=None)
    assert review_cmds.cmd_assign(args) == review_cmds.NO_ASSIGNEE_EXIT == 1
    assert "--assignee" in capsys.readouterr().out
    assert not got and not revocations.called  # 회수·작업 생성 전에 멈춘다

    assigned = SimpleNamespace(**{**vars(planned), "assignee": "s1"})
    picked: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "assign_reviewer", _capture(picked, assigned))
    args = _args("assign", assignment_id="a1:qa", assignee="s1")
    assert review_cmds.cmd_assign(args) == 0
    _, a, reviewer, policy = picked["args"]
    assert (a, reviewer) == (planned, "s1") and policy == selected
    assert got["args"][1] is assigned


def test_assign_refused_reviewer_reports_fix_without_traceback(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """QA 담당자가 선임이 아니거나 짝 배정과 같으면 traceback 대신 이유·고칠 곳과 종료 코드 1.

    정답 근거: `plan_qa`가 담당자를 비우는 경우(선임 없음, 선임이 원래 담당자뿐)는 같은 정책으로는
    어떤 `--assignee`도 받아들여지지 않으므로, 안내는 `reviewers.senior`를 먼저 고치라고 해야 한다.
    """
    planned = SimpleNamespace(
        label_kinds=("action",), session_id="s001", assignee=None, mode=ReviewMode.QA
    )
    monkeypatch.setattr(review_cmds, "get_assignment", Mock(return_value=planned))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    created = Mock(return_value=[])
    monkeypatch.setattr(review_cmds, "create_assignment_tasks", created)
    refused = Mock(side_effect=TaskError("a1:qa: QA 배정은 선임 검수자에게만 줍니다: r1"))
    monkeypatch.setattr(review_cmds, "assign_reviewer", refused)
    args = _args("assign", assignment_id="a1:qa", assignee="r1")
    assert review_cmds.cmd_assign(args) == review_cmds.NO_ASSIGNEE_EXIT
    out = capsys.readouterr().out
    assert "QA 배정은 선임" in out and "reviewers.senior에 원래 담당자가 아닌 선임" in out
    assert not created.called


def test_collect_passes_one_policy_to_setup_and_collector(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dlp review collect`의 수집(배정 마무리)과 설정이 같은 정책을 본다."""
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "collect_task", _capture(got, None))
    task = SimpleNamespace(session_id="s001", task_key="cvat:1", status=ReviewTaskStatus.COLLECTED)
    monkeypatch.setattr(review_cmds, "get_review_task", Mock(return_value=task))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    args = _args("collect", task_key="cvat:1", reviewer="pilot-labeler")
    assert review_cmds.cmd_collect(args) == 0
    _, _, setup, _, _, policy = got["args"]
    _same_source(setup, policy, selected)


def test_serve_resolves_reviewers_and_collects_with_one_policy(
    selected: ReviewOpsPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """웹훅 서버의 검수자 확인(CVAT 계정 연결)·수집·작업 생성 설정이 같은 정책을 본다.

    담당자 확인이 고른 연결을 보는데 설정이 기본 연결을 보면, 서버가 만든 작업과 웹훅 판정이
    서로 다른 계정 표를 쓴다.
    """
    monkeypatch.setenv("DLP_WEBHOOK_SECRET", "s3cret")
    server = Mock()
    serve = Mock(return_value=server)
    monkeypatch.setattr(review_cmds, "serve", serve)
    task = SimpleNamespace(
        session_id="s001", task_key="cvat:1", status=ReviewTaskStatus.OPEN,
        tool=ReviewTool.CVAT, assignee="pilot-labeler",
    )  # fmt: skip
    monkeypatch.setattr(review_cmds, "get_review_task", Mock(return_value=task))
    users: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "resolve_reviewer", _capture(users, "pilot-labeler"))
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "collect_task", _capture(got, None))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    assert review_cmds.cmd_serve(_args("serve", port=0)) == 0
    on_collect = serve.call_args.args[2]
    assert serve.call_args.kwargs["policy"] == selected.webhook
    assert serve.call_args.kwargs["policy"].bind_host == BIND_HOST
    on_collect(CollectRequest(task_key="cvat:1", reviewer="cvat-labeler"))
    _, _, setup, _, _, policy = got["args"]
    _same_source(setup, policy, selected)
    assert users["args"][3] is policy.cvat.users


def test_serve_checks_label_studio_author_against_selected_accounts(
    selected: ReviewOpsPolicy, no_clients: dict[str, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Label Studio 웹훅은 고른 정책의 `label_studio.users`로 찾은 담당자 계정 ID와 대조한다.

    회귀(F-REV-06): 예전에는 서비스 계정만 아니면 누가 낸 주석이든 담당자의 검수로 수집했다.
    담당자 계정(이메일 → 사용자 21)이 아닌 사용자(22)의 주석이면 수집하지 않고
    `ReviewerMismatchError`(서버는 403)를 낸다. 담당자 계정의 주석이면 담당자로 수집한다.
    """
    monkeypatch.setenv("DLP_WEBHOOK_SECRET", "s3cret")
    ls: Mock = no_clients["LabelStudioClient"].from_env.return_value
    ls.service_user_id.return_value = 1
    accounts = {"pilot-labeler@example.invalid": 21}
    ls.find_user_id.side_effect = accounts.get
    serve = Mock(return_value=Mock())
    monkeypatch.setattr(review_cmds, "serve", serve)
    task = SimpleNamespace(
        session_id="s001", task_key="label_studio:4", status=ReviewTaskStatus.OPEN,
        tool=ReviewTool.LABEL_STUDIO, assignee="pilot-labeler",
    )  # fmt: skip
    monkeypatch.setattr(review_cmds, "get_review_task", Mock(return_value=task))
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "collect_task", _capture(got, None))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    assert review_cmds.cmd_serve(_args("serve", port=0)) == 0
    on_collect = serve.call_args.args[2]
    with pytest.raises(ReviewerMismatchError):
        on_collect(CollectRequest("label_studio:4", None, 22, 22))
    assert "args" not in got  # 수집하지 않았다
    on_collect(CollectRequest("label_studio:4", None, 21, 21))
    assert got["args"][3] == "pilot-labeler"


def test_refresh_urls_uses_selected_media_policy_without_cvat(
    selected: ReviewOpsPolicy, no_clients: dict[str, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dlp review refresh-urls`(F-REV-08)는 고른 정책의 매체 절(서명 URL 유효 시간)로 설정을
    만들고 CVAT에는 접속하지 않는다. 세션을 주지 않으면 열린 Label Studio 작업이 있는 모든 세션을
    세션마다 갱신하고, 갱신하지 못한 작업이 있으면 1로 끝난다 (cron 경보)."""
    monkeypatch.setattr(
        review_cmds, "sessions_with_open_ls_tasks", Mock(return_value=["s001", "s002"])
    )
    seen: list[tuple[str, ReviewSetup]] = []

    def refresh(conn: sa.Connection, sid: str, setup: ReviewSetup) -> RefreshReport:
        seen.append((sid, setup))
        errors = ["label_studio:9: 도구 작업의 주인이 다릅니다"] if sid == "s002" else []
        return RefreshReport(refreshed=[f"label_studio:{sid}"], errors=errors)

    monkeypatch.setattr(review_cmds, "refresh_session_urls", refresh)
    args = _args("refresh-urls", session=[], no_cvat=True)
    assert review_cmds.cmd_refresh_urls(args) == 1
    assert [sid for sid, _ in seen] == ["s001", "s002"]
    setup = seen[0][1]
    assert setup.media_policy() == selected.media and setup.cvat is None
    no_clients["CvatClient"].from_env.assert_not_called()
    seen.clear()
    assert review_cmds.cmd_refresh_urls(_args("refresh-urls", session=["s001"], no_cvat=True)) == 0
    assert [sid for sid, _ in seen] == ["s001"]


def test_register_webhooks_reads_selected_policy_before_tools(
    no_clients: dict[str, Mock], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`dlp review register-webhooks`도 `_setup`으로 고른 파일을 엄격히 읽는다.

    정책 값은 쓰지 않지만, 깨진 파일이면 도구에 접속하기 전에 실패한다 (웹훅을 받을 `serve`가 같은
    파일을 읽으므로 설정 단계에서 알린다). 파일이 맞으면 찾은 프로젝트마다 웹훅을 등록한다.
    """
    monkeypatch.setenv("DLP_WEBHOOK_SECRET", "s3cret")
    url = f"sqlite:///{tmp_path / 'register.db'}"
    sqlite_db(url).dispose()  # 표만 만든다 (등록된 사전 없음: 기본 프로젝트만 찾는다)
    cvat_cls, ls_cls = no_clients["CvatClient"], no_clients["LabelStudioClient"]
    cvat: Mock = cvat_cls.from_env.return_value
    cvat.find_project.return_value = {"id": 7}
    ls_cls.from_env.return_value.find_project.return_value = None
    args = _args("register-webhooks", url_base="http://gw:8000")
    args.url = url
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        review_cmds.cmd_register(args)
    cvat_cls.from_env.assert_not_called()
    monkeypatch.delenv("DLP_REVIEW_POLICY_FILE")
    assert review_cmds.cmd_register(args) == 0
    urls: set[str] = {c.args[1] for c in cvat.add_webhook.call_args_list}
    assert urls == {"http://gw:8000/webhooks/cvat"}
