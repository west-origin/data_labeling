"""`dlp actions run|describe`의 VLM 주소 검사(F-PAR-82)와 쓰기 직전 확인 종료 코드(F-PAR-80).

정답 근거: 사내 대역은 `actions.yaml vlm.internal_hosts`(사설 대역·루프백)다. TEST-NET-3
(203.0.113.0/24)은 그 밖이라 보내지 않고 종료 코드 5, 10.x는 사내라 클라이언트를 만든다 (요청은
보내지 않는다). 쓰기 직전 확인이 막은 실행은 종료 코드 3이고 설명 재생성·번역을 돌리지 않는다.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from dlp_actions.policy import load_policy
from dlp_actions.runner import ActionsSummary
from dlp_cli import actions_cmds
from dlp_schema import repo_root

# 사내 대역 밖 주소 (TEST-NET-3, 연결하지 않는다)
OUTSIDE = "http://203.0.113.10:8000/v1"


def _args(**kw: Any) -> argparse.Namespace:
    values: dict[str, Any] = dict(
        session_id="s001", pending=False, vlm_url=OUTSIDE, store="s3", url="sqlite://",
        translate_url=None, no_translate=True,
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_client_refuses_outside_internal_hosts(capsys: pytest.CaptureFixture[str]) -> None:
    """F-PAR-82: 사내 대역 밖 VLM 주소에는 클라이언트를 만들지 않고 5 (번역·판정과 같은 안내)."""
    client, code = actions_cmds._client(OUTSIDE, load_policy(repo_root()))  # pyright: ignore[reportPrivateUsage]
    assert client is None and code == actions_cmds.CONFIG
    assert "[설정 오류] 사내 대역 밖 주소에는 보내지 않습니다" in capsys.readouterr().out


def test_client_accepts_internal_address() -> None:
    """사내 주소면 클라이언트를 만든다 (요청은 보내지 않는다)."""
    client, code = actions_cmds._client("http://10.0.0.11:8000", load_policy(repo_root()))  # pyright: ignore[reportPrivateUsage]
    assert client is not None and code == actions_cmds.OK


def test_run_and_describe_exit_5_for_outside_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """run·describe 모두 DB·저장소에 닿기 전에 5로 끝난다."""

    def boom(*_: object) -> None:
        raise AssertionError("주소 검사 전에 DB·저장소에 닿으면 안 된다")

    monkeypatch.setattr(actions_cmds, "session_ontology", boom)
    monkeypatch.setattr(actions_cmds, "store_from_spec", boom)
    assert actions_cmds.cmd_run(_args()) == actions_cmds.CONFIG
    assert actions_cmds.cmd_describe(_args()) == actions_cmds.CONFIG


def test_run_blocked_by_write_check_exits_3(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-PAR-80: 쓰기 직전 확인이 막으면 3, 설명 재생성·번역은 돌리지 않는다."""

    def blocked(*_: object) -> ActionsSummary:
        return ActionsSummary("v", blocked="s001: 승인 보류 중입니다 (ops01: report)")

    def never(*_: object) -> None:
        raise AssertionError("막힌 실행 뒤 설명 재생성·번역을 돌리면 안 된다")

    def nothing(*_: object) -> None:
        return None

    monkeypatch.setattr(actions_cmds, "session_ontology", nothing)
    monkeypatch.setattr(actions_cmds, "store_from_spec", nothing)
    monkeypatch.setattr(actions_cmds, "run_actions", blocked)
    monkeypatch.setattr(actions_cmds, "_describe_after_run", never)
    monkeypatch.setattr(actions_cmds, "translate_after_actions", never)
    code = actions_cmds.cmd_run(_args(vlm_url="http://10.0.0.11:8000", no_translate=False))
    assert code == actions_cmds.UNAVAILABLE
    assert (
        "[쓰지 않음] s001: 쓰기 직전 확인이 막았습니다 (s001: 승인 보류" in capsys.readouterr().out
    )
