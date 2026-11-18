"""`dlp robot check`의 다지 손 로봇 출력과 내보내기 요약의 손가락 줄 (ADR 0068).

정답: 직접 쓴 시험 로봇(`dlp_fixtures.dexhand`)을 `DLP_ROBOTS_DIR`에 두고 확인 명령을 돌린다. 손가락
대응·관절·손목 → 끝 거리(손 배율을 정하는 값)·풀지 않는 관절이 출력되고, 지원하지 않는 손 구조와
mink 풀이기 정책은 종료 코드 1과 이유로 끝난다. DB·서비스 없이 돈다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from dlp_retarget.policy import RetargetPolicy, SolverName, load_policy
from dlp_retarget.robot import ROBOTS_DIR_ENV

from dlp_cli import robot_cmds
from dlp_cli.export_cmds import _print_retarget  # pyright: ignore[reportPrivateUsage]
from dlp_cli.robot_cmds import cmd_check
from dlp_fixtures.dexhand import write_robot

ROOT = Path(__file__).resolve().parents[3]


def test_robot_check_shows_finger_mapping_and_hand_scale_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """손가락 셋만 대응한 다지 손 로봇: 0과 팔 연쇄, 손가락마다 대응·관절, 영점 자세의 손목 → 끝
    거리와 그 상한, 손 배율, 풀지 않는 손 관절(대응하지 않은 약지·새끼)을 출력한다. 평행 그리퍼
    항목(그리퍼·접근 축)은 나오지 않는다."""
    write_robot(tmp_path, fingers=("thumb", "index", "middle"), hand_scale=0.9)
    monkeypatch.setenv(ROBOTS_DIR_ENV, str(tmp_path))
    assert cmd_check(argparse.Namespace(robot_id="dexhand")) == 0
    out = capsys.readouterr().out
    assert "팔 연쇄 base_link → hand: shoulder_yaw" in out
    assert "손 배율 0.9" in out and "손가락 끝 허용 오차 0.005 m" in out
    assert "손가락 index_tip → index_tip: index_abduction, index_flexion, index_curl" in out
    # 검지: 손바닥 0.02 + 뿌리 높이 0.085 + 마디 0.045 + 0.035 = 0.185, 옆으로 0.03 → 0.187 m
    assert "영점 자세 손목 → 끝 0.187 m, 손목 → 끝 거리 상한 0.189 m" in out
    assert "풀지 않는 손 관절" in out and "ring_abduction" in out and "pinky_curl" in out
    assert "최대 팔 길이 0.600 m" in out and "리타기팅 정책 풀이기 dls" in out
    assert "그리퍼" not in out and "접근 축" not in out


def test_robot_check_fails_with_a_reason_for_unsupported_hands_and_mink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """손가락끼리 관절을 함께 쓰는 대응은 1과 이유. 설정이 맞아도 정책 풀이기가 mink면 다지 손을
    풀 수 없어 1과 이유 (내보내기가 거부할 조합을 먼저 알린다)."""
    path = write_robot(tmp_path)
    monkeypatch.setenv(ROBOTS_DIR_ENV, str(tmp_path))

    def mink_policy(root: Path) -> RetargetPolicy:
        return load_policy(root).model_copy(update={"solver": SolverName.MINK})

    with monkeypatch.context() as patched:
        patched.setattr(robot_cmds, "load_policy", mink_policy)
        assert cmd_check(argparse.Namespace(robot_id="dexhand")) == 1
    assert "mink" in capsys.readouterr().out
    data: Any = json.loads(path.read_text("utf-8"))
    data["arm"]["dexterous_hand"]["fingers"][2]["tip_link"] = "index_proximal"
    path.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    assert cmd_check(argparse.Namespace(robot_id="dexhand")) == 1
    assert "함께 씁니다" in capsys.readouterr().out


def test_export_summary_prints_one_line_per_finger(capsys: pytest.CaptureFixture[str]) -> None:
    """내보내기 요약: 다지 손 에피소드는 에피소드 줄(손목 오차) 아래에 손가락마다 한 줄(목표·도달
    비율, 최대 오차, 손 길이 밖 프레임)을 낸다. 평행 그리퍼 에피소드(fingers 없음)는 한 줄이다."""
    episode = {
        "session_id": "p1",
        "target_fraction": 1.0,
        "reachable_fraction": 0.5,
        "position_error_max_m": 0.012,
        "decision": "keep",
        "reason": None,
    }
    finger = {
        "human_part": "thumb_tip",
        "target_fraction": 0.75,
        "reachable_fraction": 0.4,
        "position_error_max_m": None,
        "frames_out_of_reach": 7,
    }
    meta = {
        "robot_id": "dexhand",
        "model_sha256": "a" * 64,
        "retarget_version": "retarget-dls-dlshand1+r0+m0+p0",
        "episodes": [{**episode, "fingers": [finger]}, {**episode, "session_id": "p2"}],
    }
    _print_retarget(meta)
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "  p1: 목표 100%, 도달 50%, 최대 위치 오차 12 mm, keep"
    assert lines[2] == (
        "    손가락 thumb_tip: 목표 75%, 도달 40%, 최대 위치 오차 -, 손 길이 밖 7프레임"
    )
    assert lines[3] == "  p2: 목표 100%, 도달 50%, 최대 위치 오차 12 mm, keep" and len(lines) == 4
