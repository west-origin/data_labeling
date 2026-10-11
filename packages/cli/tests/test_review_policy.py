"""`dlp review` 명령은 고른 검수 정책(`DLP_REVIEW_POLICY_FILE`)을 한 출처로 쓴다 (F-REV-01).

저장소·도구 클라이언트는 만들지 않는다 (`review_config`의 생성자를 Mock으로 바꾼다). DB가 필요한
명령은 메모리 SQLite 엔진을 열기만 하고, 도구·DB를 쓰는 라이브러리 함수는 호출 인자만 받아 둔다.

정답 근거: 시험이 정책 파일에 직접 적은 CVAT 계정 연결(`cvat.users`), 원본 검수자
(`reviewers.privacy`), 매체 화질(`media.watermark_crf`)이 작업 생성·수집·웹훅 서버·매체 회수가 보는
값과 같아야 한다. Git의 기본 `review.yaml`은 계정 연결과 원본 검수자가 비어 있다. 한 명령 안에서는
정책을 한 번만 읽어 `ReviewSetup`과 명령이 같은 객체를 본다 (`is`로 확인한다).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from dlp_cli import review_cmds, review_config
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.policy import load_policy as load_default_policy
from dlp_review.revoke import RevocationReport
from dlp_review.tasks import ReviewSetup
from dlp_review.webhook import CollectRequest
from dlp_schema import repo_root
from dlp_schema.review import ReviewTaskStatus

# 시험이 고른 정책 파일에 적는 값 (Git 기본 정책과 다르다)
USERS = {"pilot-privacy": "cvat-privacy", "pilot-labeler": "cvat-labeler"}
PRIVACY_REVIEWERS = ("pilot-privacy",)
WATERMARK_CRF = 31


@pytest.fixture
def selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReviewOpsPolicy:
    """기본 review.yaml을 고쳐 별도 파일로 두고 `DLP_REVIEW_POLICY_FILE`로 고른다. 기대 정책."""
    root = repo_root()
    data = yaml.safe_load((root / "config/policies/review.yaml").read_text("utf-8"))
    data["reviewers"]["privacy"] = list(PRIVACY_REVIEWERS)
    data["cvat"]["users"] = dict(USERS)
    data["media"]["watermark_crf"] = WATERMARK_CRF
    path = tmp_path / "review.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(path))
    default = load_default_policy(root)
    # 전제: 기본 정책(라이브러리 로더)은 이 값들을 갖지 않는다
    assert default.cvat.users == {} and default.media.watermark_crf != WATERMARK_CRF
    return review_config.load_review_policy(root)


@pytest.fixture(autouse=True)
def no_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    """저장소·도구 클라이언트를 만들지 않는다 (네트워크·DB 없이 정책 연결만 본다)."""
    for name in ("raw_store", "S3Store", "CvatClient", "LabelStudioClient"):
        monkeypatch.setattr(review_config, name, Mock())


def _args(command: str, **kw: Any) -> argparse.Namespace:
    """`dlp review <command>` 인자. DB는 메모리 SQLite (엔진을 열기만 한다)."""
    return argparse.Namespace(review_command=command, url="sqlite://", **kw)


def _same_source(setup: ReviewSetup, policy: ReviewOpsPolicy, expected: ReviewOpsPolicy) -> None:
    """설정과 명령이 한 번 읽은 같은 정책 객체를 보고, 그 값이 고른 파일의 값이다."""
    assert policy == expected
    assert setup.cvat_policy() is policy.cvat
    assert setup.media_policy() is policy.media


@pytest.mark.parametrize(
    "command", ["create", "collect", "serve", "assign", "register-webhooks", "revoke"]
)
def test_setup_carries_selected_policy(command: str, selected: ReviewOpsPolicy) -> None:
    """`_setup`이 만든 설정의 CVAT 계정 연결·매체 정책이 고른 파일의 값이다 (지연 기본 로더 아님).

    회귀: 예전에는 `ReviewSetup`에 정책을 넘기지 않아 처음 쓸 때 Git의 review.yaml을 읽었다.
    그래서 파일럿의 `dlp review create --stage privacy --assignee pilot-privacy`가 "CVAT 계정이
    없습니다"로 실패했다.
    """
    setup = review_cmds._setup(_args(command))  # pyright: ignore[reportPrivateUsage]
    assert setup.cvat_policy() == selected.cvat
    assert setup.cvat_policy().users == USERS
    assert setup.media_policy().watermark_crf == WATERMARK_CRF


def test_revocation_setup_uses_selected_policy(selected: ReviewOpsPolicy) -> None:
    """매체 회수가 스스로 만드는 설정(`tolerant`, `dlp privacy detect|render`·`revoke --pending`)도
    고른 정책을 싣는다."""
    setup, _ = review_config.review_setup("privacy.detect", None, tolerant=True)
    assert setup.cvat_policy() == selected.cvat
    assert setup.media_policy() == selected.media


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
    planned = SimpleNamespace(label_kinds=("blur_track",), session_id="s001")
    monkeypatch.setattr(review_cmds, "get_assignment", Mock(return_value=planned))
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "create_assignment_tasks", _capture(got, []))
    assert review_cmds.cmd_assign(_args("assign", assignment_id="a1")) == 0
    _, _, setup, policy, _ = got["args"]
    _same_source(setup, policy, selected)


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
    task = SimpleNamespace(session_id="s001", task_key="cvat:1", status=ReviewTaskStatus.OPEN)
    monkeypatch.setattr(review_cmds, "get_review_task", Mock(return_value=task))
    users: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "resolve_reviewer", _capture(users, "pilot-labeler"))
    got: dict[str, Any] = {}
    monkeypatch.setattr(review_cmds, "collect_task", _capture(got, None))
    monkeypatch.setattr(review_cmds, "run_revocations", Mock(return_value=RevocationReport()))
    assert review_cmds.cmd_serve(_args("serve", port=0)) == 0
    on_collect = serve.call_args.args[2]
    assert serve.call_args.kwargs["policy"] == selected.webhook
    on_collect(CollectRequest(task_key="cvat:1", reviewer="cvat-labeler"))
    _, _, setup, _, _, policy = got["args"]
    _same_source(setup, policy, selected)
    assert users["args"][3] is policy.cvat.users
