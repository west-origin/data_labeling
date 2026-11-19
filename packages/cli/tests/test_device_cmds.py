"""추론 장치 설정(`DLP_INFERENCE_DEVICE`, ADR 0072)을 명령이 확인하고 출력에 남긴다.

`dlp models device`는 아주 작은 Identity 모델로 세션을 열어 실제 실행 제공자를 보인다. 무거운 단계
(`dlp prelabel run`, `dlp privacy detect`)는 설정이 틀리면 DB·버킷에 닿기 전에 1로 끝난다.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from dlp_cli import prelabel_cmds, privacy_cmds
from dlp_cli.models_cmds import cmd_device
from dlp_models.device import ENV


def test_models_device_opens_a_cpu_session_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    assert cmd_device(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert f"추론 장치: cpu ({ENV} 없음, 기본값)" in out
    assert "세션 확인: 실제 실행 제공자 CPUExecutionProvider" in out


def test_models_device_fails_for_an_unknown_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV, "gpu")
    assert cmd_device(argparse.Namespace()) == 1
    assert "[실패]" in capsys.readouterr().out


def test_heavy_stages_stop_before_touching_anything_on_a_bad_device(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV, "gpu")
    args = argparse.Namespace(session_id="s001", store="s3", url="postgresql://unused")
    with (
        patch("dlp_cli.prelabel_cmds.session_ontology") as ontology,
        patch("dlp_cli.privacy_cmds.raw_store") as raw,
    ):
        assert prelabel_cmds.cmd_run(args) == 1
        assert privacy_cmds.cmd_detect(args) == 1
    ontology.assert_not_called()
    raw.assert_not_called()
    assert capsys.readouterr().out.count("[추론 장치]") == 2
