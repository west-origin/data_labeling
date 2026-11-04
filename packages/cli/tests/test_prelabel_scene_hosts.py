"""장면 분류 VLM 주소가 사내 대역 밖이면 원본을 받기 전에 멈춘다 (F-PAR-89)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from dlp_cli.prelabel_cmds import _scene  # pyright: ignore[reportPrivateUsage]
from dlp_prelabel.policy import load_policy
from dlp_schema import repo_root


class _NoRaw:
    """원본을 받으면 실패하는 저장소 (주소 검사가 받기보다 먼저여야 한다)."""

    def get_file(self, *_: Any) -> None:
        raise AssertionError("사내 대역 밖 주소인데 원본을 받았다")


def test_scene_vlm_outside_internal_hosts_is_refused_before_raw_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy = load_policy(repo_root())
    vlm_policy = policy.model_copy(
        update={"scene": policy.scene.model_copy(update={"classifier": "vlm"})}
    )
    monkeypatch.setenv("DLP_VLM_URL", "http://8.8.8.8:8000/v1")
    args = argparse.Namespace(scene=None, session_id="s1")
    with pytest.raises(SystemExit, match="사내 대역 밖"):
        _scene(args, None, _NoRaw(), vlm_policy, tmp_path)  # pyright: ignore[reportArgumentType]
