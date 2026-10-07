"""정답을 아는 합성 데이터 생성기. 모든 생성기는 같은 seed에 같은 결과를 낸다."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dlp_fixtures.actions import ActionScenario, generate_action_scenario
from dlp_fixtures.io import write_jsonl
from dlp_fixtures.sessions import generate_sessions
from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_fixtures.video import BlurScenario, generate_blur_scenario

__all__ = [
    "ActionScenario",
    "BlurScenario",
    "SyncScenario",
    "generate_action_scenario",
    "generate_all",
    "generate_blur_scenario",
    "generate_sessions",
    "generate_sync_scenario",
]

RECORDED_AT = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)


def generate_all(out_dir: Path, seed: int = 0, n_sessions: int = 300) -> dict[str, Path]:
    """모든 픽스처를 out_dir 아래에 쓴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sessions": out_dir / "sessions.jsonl",
        "sync": out_dir / "sync",
        "blur": out_dir / "blur",
        "actions": out_dir / "actions",
    }
    write_jsonl(paths["sessions"], generate_sessions(n_sessions, seed))
    generate_sync_scenario(seed, recorded_at=RECORDED_AT).write(paths["sync"])
    blur = generate_blur_scenario(seed)
    paths["blur"].mkdir(parents=True, exist_ok=True)
    blur.write(paths["blur"] / "bodycam.mp4")
    write_jsonl(paths["blur"] / "labels.jsonl", blur.labels)
    generate_action_scenario(seed).write(paths["actions"])
    return paths
