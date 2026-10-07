"""LeRobot 공식 쓰기 API로 쓰고 공식 로더(LeRobotDataset)로 다시 읽는다.

격리된 일회용 환경(PyTorch CPU판, 약 1.5 GB)을 받으므로 기본 테스트에서 빼고
`make test-isolated`로 돈다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlp_export.lerobot import Vocab, build_episode, run_script, write_package
from dlp_export.policy import ExportPolicy
from dlp_export.source import label_states, select_labels
from dlp_media.pts import build_pts_index
from dlp_schema.ontology import Ontology

from .conftest import ROOT, Scenario

pytestmark = pytest.mark.isolated_env


def test_official_lerobot_loader_reads_export(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    lp = policy.lerobot
    video = tmp_path / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    labels = select_labels(scenario.labels, policy, label_states(policy, False))
    ep = build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48), labels,
        Vocab.from_ontology(ontology), lp,
    )  # fmt: skip
    pkg = write_package(
        [(scenario.session, video, ep), (scenario.session, video, ep)],
        policy,
        (64, 48),
        tmp_path / "pkg",
    )
    dest = tmp_path / "lerobot"
    run_script(ROOT, lp, "lerobot_write.py", str(pkg), str(dest))
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == lp.fps
    check = json.loads(run_script(ROOT, lp, "lerobot_check.py", str(dest), lp.repo_id))
    n = len(ep.times_ms)
    assert check["episodes"] == 2 and check["frames"] == 2 * n
    assert {"observation.state", "action", "annotation.hand_state", "annotation.verb",
            "annotation.verification", "annotation.tool_surface_contact",
            "observation.images.bodycam"} <= set(check["features"])  # fmt: skip
    assert check["tasks"] == ["cleaning", "floor_sweep_mop"]
    for s in check["samples"]:
        k = s["index"] % n  # 두 에피소드가 같다
        assert s["image_shape"] == [3, 48, 64]
        assert s["timestamp"] == pytest.approx(k / lp.fps, abs=1e-4)
        assert np.allclose(s["state"], ep.state[k], atol=1e-6)
        assert s["hand_state"] == ep.hand_state[k].tolist()
        assert s["verb"] == ep.verb[k].tolist()
        assert s["verification"] == ep.verification[k].tolist()
        assert s["task"] == ep.tasks[k]
