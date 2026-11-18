"""`dlp export`의 촉각장갑용 원본 감사 저장소 선택과 요약 출력 (ADR 0069·0070, DB·저장소 없이).

lerobot·gr00t 형식이고 정책이 켜져 있을 때만 원본 감사 저장소를 연다. 용도는 형식마다 다른 상수
(`export.lerobot`, `export.gr00t`)이고 `ops.yaml audit.service_purposes`에 등록돼 있어야 월간 감사
리포트가 정상 파이프라인 읽기로 본다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from dlp_cli import export_cmds
from dlp_cli.main import build_parser
from dlp_export.policy import load_policy
from dlp_export.scene import SceneData
from dlp_export.tactile import TactileData
from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.storage import LocalStore
from dlp_ops.policy import load_policy as load_ops_policy
from dlp_schema import repo_root
from dlp_schema.labels import Hand

ROOT = repo_root()


def test_tactile_store_is_opened_only_for_episode_formats_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """coco·intervals는 원본 저장소를 열지 않고, lerobot·gr00t는 형식별 용도로 감사 저장소를 연다.

    정책 `tactile.enabled`가 꺼져 있으면 어느 형식도 열지 않는다. 두 용도는 ops.yaml에 등록돼 있다.
    """
    opened: list[tuple[str, str | None, str]] = []

    def fake_raw_store(spec: str, url: str | None, purpose: str) -> AuditedStore:
        opened.append((spec, url, purpose))
        return AuditedStore(LocalStore(tmp_path, "raw"), MemorySink(), actor="t", purpose=purpose)

    monkeypatch.setattr(export_cmds, "raw_store", fake_raw_store)
    policy = load_policy(ROOT)
    assert policy.tactile.enabled  # 저장소 기본값: 장갑이 있고 동기화됐으면 싣는다

    def args(fmt: str) -> argparse.Namespace:
        return argparse.Namespace(format=fmt, store="local:/x", url="postgresql://db")

    assert export_cmds.tactile_store(args("coco"), policy) is None
    assert export_cmds.tactile_store(args("intervals"), policy) is None
    assert opened == []
    lerobot = export_cmds.tactile_store(args("lerobot"), policy)
    gr00t = export_cmds.tactile_store(args("gr00t"), policy)
    assert lerobot is not None and gr00t is not None
    assert opened == [
        ("local:/x", "postgresql://db", "export.lerobot"),
        ("local:/x", "postgresql://db", "export.gr00t"),
    ]
    assert {"export.lerobot", "export.gr00t"} <= set(load_ops_policy(ROOT).audit.service_purposes)
    off = policy.model_copy(
        update={"tactile": policy.tactile.model_copy(update={"enabled": False})}
    )
    assert export_cmds.tactile_store(args("gr00t"), off) is None
    assert export_cmds.tactile_store(args("lerobot"), off) is None
    assert len(opened) == 2


def test_export_parser_gives_the_format_and_store_the_tactile_store_reads() -> None:
    """`dlp export gr00t …`가 `tactile_store`가 읽는 값(format·store·url)을 채운다."""
    args = build_parser().parse_args(["export", "gr00t", "dv1", "--target", "internal"])
    assert (args.format, args.store, args.url) == ("gr00t", "s3", None)


def test_tactile_summary_line_counts_hands_by_status(capsys: pytest.CaptureFixture[str]) -> None:
    """요약 한 줄: 실은 손의 채널 수와 세션·손 상태별 수 (`TactileData.meta`의 키를 읽는다)."""
    sessions = [
        {
            "session_id": "session-a",
            "glove_model": "syn-glove-a",
            "hands": {"left": {"status": "no_glove"}, "right": {"status": "exported"}},
        },
        {
            "session_id": "session-b",
            "glove_model": None,
            "hands": {"left": {"status": "no_glove"}, "right": {"status": "not_synced"}},
        },
    ]
    data = TactileData(
        {Hand.RIGHT: ("pressure_0", "pressure_1")}, [{}, {}], [3, 3], sessions,
        load_policy(ROOT).tactile,
    )  # fmt: skip
    export_cmds._print_tactile(data.meta("gr00t"))  # pyright: ignore[reportPrivateUsage]
    export_cmds._print_tactile(None)  # pyright: ignore[reportPrivateUsage]
    assert capsys.readouterr().out == (
        "촉각장갑: right 2채널 (세션·손 상태: exported 1, no_glove 2, not_synced 1)\n"
    )


def test_scene_summary_line_counts_sessions_and_posed_frames(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """장면 좌표 자세 요약 한 줄: 세션 상태별 수와 실은 세션의 자세가 있는 프레임 비율."""
    sessions = [
        {"session_id": "session-a", "status": "exported", "frames": 60, "frames_valid": 45},
        {"session_id": "session-b", "status": "stale"},
        {"session_id": "session-c", "status": "exported", "frames": 40, "frames_valid": 35},
    ]
    data = SceneData([None, None, None], [60, 0, 40], sessions, 500)
    export_cmds._print_scene(data.meta("lerobot"))  # pyright: ignore[reportPrivateUsage]
    export_cmds._print_scene(None)  # pyright: ignore[reportPrivateUsage]
    assert capsys.readouterr().out == (
        "장면 좌표 자세: 세션 상태 exported 2, stale 1, 자세가 있는 프레임 80%\n"
    )
