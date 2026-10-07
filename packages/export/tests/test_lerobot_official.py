"""LeRobot 공식 쓰기 API로 쓰고 공식 로더(LeRobotDataset)로 다시 읽는다.

격리된 일회용 환경(PyTorch CPU판, 약 1.5 GB)을 받으므로 기본 테스트에서 빼고
`make test-isolated`로 돈다.

프레임 내용: 블러본의 i번째 프레임을 i로 정해지는 회색 단색으로 만들어, 로더가 돌려준 영상
프레임이 PTS로 고른 그 프레임(frame_index)인지 밝기로 확인한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlp_export.lerobot import Vocab, build_episode, run_script, write_package
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.source import label_states, select_labels
from dlp_fixtures.video import write_video
from dlp_media.pts import build_pts_index
from dlp_schema.ontology import Ontology

from .conftest import ROOT, Scenario


def gray(i: int) -> int:
    """i번째 프레임의 밝기. 이웃 프레임끼리 37씩 달라 한 프레임만 어긋나도 드러난다."""
    return 28 + (i * 37) % 200


pytestmark = pytest.mark.isolated_env


def test_official_lerobot_loader_reads_export(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """격리 환경의 공식 쓰기 API로 에피소드 2개를 쓰고 공식 로더로 읽은 값이 우리 특징과 같은지.

    시나리오: 시나리오 프레임 시각으로 회색 단색 영상을 만들고, 같은 에피소드를 두 번 넣는다.
    정답 근거: `build_episode`가 만든 특징(state·hand_state·verb·verification·task)이 그대로 읽혀야
    하고, 각 프레임 밝기가 `gray(frame_index[k])`여야 한다 (PTS로 고른 프레임이 들어갔다는 뜻).
    작업 목록은 시나리오의 작업 구간(floor_sweep_mop)과 구간 밖 도메인(cleaning).
    """
    lp = policy.lerobot
    video = tmp_path / "v.mp4"
    write_video(
        video,
        ((t, np.full((48, 64, 3), gray(i), np.uint8)) for i, t in enumerate(scenario.times)),
        width=64,
        height=48,
    )
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
        Pseudonymizer(None),
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
        # PTS로 고른 블러본 프레임이 그대로 들어갔다 (손실 압축 허용 오차 안)
        assert s["image_mean"] * 255 == pytest.approx(gray(int(ep.frame_index[k])), abs=6)
