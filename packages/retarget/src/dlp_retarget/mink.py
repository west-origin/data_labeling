"""mink(MuJoCo 기반 미분 IK, Apache-2.0) 풀이기 어댑터 (선택, 격리 환경, ADR 0047).

mink·MuJoCo는 작업공간 의존성에 넣지 않는다 (`make install`을 가볍게). `scripts/retarget-env`
(uv.lock 고정)에서 `uv run --project <project> --locked --isolated`로 `scripts/retarget_mink.py`를
돌린다. 경계는 JSON 파일이다: 이 모듈이 문제(URDF 경로, 연쇄, 목표, 설정)를 쓰고 스크립트가 목표가
있는 프레임의 관절 값을 쓴다. 오차·도달 여부·속도 한계는 내장 풀이기와 같은 규칙으로 여기서
계산한다 (`ik.frame_errors`).

스크립트 쪽: URDF에서 메시·충돌·전달 장치를 빼고 MuJoCo 모델로 읽는다. 끝 링크에 도구 중심점 보조
site(z = 접근 축, x = 닫힘 축)를 달아 그 site의 FrameTask(approach 모드면 site z 둘레 회전 비용 0,
접근 축이 끝 링크 좌표축이 아니어도 롤이 정확히 자유)와 관절 한계(ConfigurationLimit)로 프레임마다
수렴할 때까지 미분 IK를 반복한다 (QP 풀이기 daqp). 첫 목표는 관절 범위 가운데에서, 끊긴 뒤는 마지막
해에서 시작한다. 속도 한계는 여기서 내장 풀이기와 같은 규칙(`ik.limit_step`, 첫 프레임 뺀 모든
프레임)으로 자른다.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from dlp_retarget.ik import Goal, IkProblem, IkSolution, frame_errors, limit_step, seeds
from dlp_retarget.policy import MinkEnv
from dlp_retarget.robot import OrientationMode


def env_command(root: Path, env: MinkEnv) -> list[str]:
    """격리 환경 실행 명령 (뒤에 스크립트와 인자를 붙인다)."""
    return [
        "uv", "run", "--project", str(root / env.project), "--locked", "--isolated",
        "--python", env.python, "python",
    ]  # fmt: skip


def problem_json(problem: IkProblem) -> dict[str, Any]:
    """스크립트 입력 JSON (목표가 없는 프레임은 null)."""
    tg, cfg, chain = problem.targets, problem.config, problem.chain
    if problem.model_path is None:
        raise ValueError("mink 풀이기에는 로봇 모델 파일 경로가 필요합니다 (IkProblem.model_path)")
    frames: list[dict[str, Any] | None] = []
    for k in range(tg.present.size):
        if not tg.present[k]:
            frames.append(None)
            continue
        frames.append(
            {
                "position": tg.position[k].tolist(),
                "approach": tg.approach[k].tolist(),
                "closing": tg.closing[k].tolist(),
            }
        )
    return {
        "urdf": str(problem.model_path),
        "base_link": chain.base,
        "tip_link": chain.tip,
        "joints": list(chain.names),
        "approach_axis": problem.approach_axis.tolist(),
        "closing_axis": problem.closing_axis.tolist(),
        "orientation_mode": cfg.orientation_mode.value,
        "orientation_weight": cfg.orientation_weight,
        "max_iterations": cfg.max_iterations,
        "damping": cfg.damping,
        "frames": frames,
    }


class MinkIkSolver:
    """mink 격리 환경 풀이기 (`IkSolver`). version: `mink2-<격리 환경 uv.lock sha256 앞 12자>`.

    mink2: 도구 중심점 site로 접근 축 비용을 바로잡고, 끊긴 뒤 마지막 해에서 시작하며, 다시 시작하는
    프레임에도 속도 한계를 적용한다 (ADR 0053).
    """

    name = "mink"

    def __init__(self, root: Path, env: MinkEnv) -> None:
        """root: 저장소 루트. env: `retarget.yaml mink` (격리 환경 위치·Python)."""
        self.root, self.env = root, env
        lock = root / env.project / "uv.lock"
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()[:12] if lock.is_file() else "nolock"
        self.version = f"mink2-{digest}"

    def run_script(self, data: dict[str, Any]) -> list[list[float] | None]:
        """격리 환경에서 스크립트를 돌려 프레임별 관절 값(목표 없는 프레임 None)을 받는다.

        Raises:
            subprocess.CalledProcessError: 스크립트 실패 (stderr 포함).
        """
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "problem.json", Path(tmp) / "solution.json"
            src.write_text(json.dumps(data), "utf-8")
            script = self.root / "scripts" / "retarget_mink.py"
            subprocess.run(
                [*env_command(self.root, self.env), str(script), str(src), str(dst)],
                cwd=self.root, env=dict(os.environ), check=True, capture_output=True, text=True,
            )  # fmt: skip
            out: Any = json.loads(dst.read_text("utf-8"))
        return list(out["q"])

    def solve(self, problem: IkProblem) -> IkSolution:
        """mink로 풀고 내장 규칙으로 오차·도달·속도 한계를 정한다 (`IkSolver.solve`)."""
        raw = self.run_script(problem_json(problem))
        chain, cfg, tg = problem.chain, problem.config, problem.targets
        n, dof = tg.present.size, len(chain.moving)
        out = IkSolution(
            q=np.zeros((n, dof)),
            position_error_m=np.full(n, np.nan),
            orientation_error_rad=np.full(n, np.nan),
            reachable=np.zeros(n, dtype=np.bool_),
            velocity_limited=np.zeros(n, dtype=np.bool_),
        )
        vmax = chain.velocity * cfg.velocity_scale
        q_prev = seeds(chain, 1)[0]  # 첫 목표 전 시작 자세 (관절 범위 가운데, 내장과 같다)
        for k in range(n):
            row = raw[k]
            if row is None or not tg.present[k]:
                out.q[k] = q_prev
                continue
            # MuJoCo 관절 범위와 URDF 한계가 다를 수 있어 URDF 한계로 한 번 더 자른다
            q = np.clip(np.array(row, dtype=np.float64), chain.lower, chain.upper)
            if k > 0:
                q, out.velocity_limited[k] = limit_step(q, q_prev, vmax, problem.times_ms, k)
            goal = Goal(tg.position[k], tg.approach[k], tg.closing[k])
            pos, rot = frame_errors(chain, q, goal, problem)
            out.q[k], out.position_error_m[k], out.orientation_error_rad[k] = q, pos, rot
            out.reachable[k] = pos <= cfg.position_tolerance_m and (
                cfg.orientation_mode is OrientationMode.NONE or rot <= cfg.orientation_tolerance_rad
            )
            q_prev = q
        return out
