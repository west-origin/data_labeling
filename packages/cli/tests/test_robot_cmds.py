"""`dlp robot check` (ADR 0047): 공개 시험 로봇은 통과하고 없는 로봇·틀린 해시는 종료 코드 1."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest
import yaml
from dlp_retarget.robot import ROBOTS_DIR_ENV

from dlp_cli.robot_cmds import cmd_check

ROOT = Path(__file__).resolve().parents[3]


def test_robot_check_reports_chain_and_rejects_bad_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """so101: 0과 관절 연쇄·모델 해시 출력. 없는 ID와 sha256이 틀린 고객 설정: 1과 이유."""
    monkeypatch.delenv(ROBOTS_DIR_ENV, raising=False)
    assert cmd_check(argparse.Namespace(robot_id="so101")) == 0
    out = capsys.readouterr().out
    assert "shoulder_pan" in out and "3a65d2d35e68" in out and "최대 팔 길이" in out
    assert cmd_check(argparse.Namespace(robot_id="nope")) == 1
    cfg = yaml.safe_load((ROOT / "config/robots/so101.yaml").read_text("utf-8"))
    shutil.copy(ROOT / "config/robots/models/so101/so101_new_calib.urdf", tmp_path / "arm.urdf")
    cfg["robot_id"] = "acme"
    cfg["model"].update(path="arm.urdf", sha256="1" * 64, public=False)
    (tmp_path / "acme.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), "utf-8")
    monkeypatch.setenv(ROBOTS_DIR_ENV, str(tmp_path))
    assert cmd_check(argparse.Namespace(robot_id="acme")) == 1
    assert "sha256" in capsys.readouterr().out
