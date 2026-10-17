"""mink(MuJoCo 미분 IK)로 프레임별 팔 관절 값을 푼다 (retarget.yaml solver: mink, ADR 0047).

    python scripts/retarget_mink.py <problem.json> <solution.json>

`dlp_retarget.mink.MinkIkSolver`가 격리 환경(`scripts/retarget-env`, uv.lock 고정)에서 부른다.
작업공간 밖이라 `dlp_*` 패키지를 쓰지 않는다 (ruff만 검사, pyright 범위 밖).

입력 JSON: `urdf`(경로), `base_link`, `tip_link`, `joints`(팔 관절 순서), `approach_axis`·
`closing_axis`(끝 링크 좌표), `orientation_mode`(none|approach|full), `orientation_weight`,
`max_iterations`, `damping`, `frames`(프레임마다 null 또는 {position, approach, closing}, 로봇 기준
좌표).
출력 JSON: `{"q": [프레임마다 null 또는 팔 관절 값 목록]}`. 오차·도달·속도 한계는 호출자가 계산한다.

URDF의 메시·충돌·전달 장치는 빼고(관절만 쓴다) 고정 링크를 합치지 않게(fusestatic false) 읽는다.
끝 링크 FrameTask의 회전 비용은 접근 축 모드면 접근 축 둘레를 0으로 둔다 (그 회전은 자유).
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mink
import mujoco
import numpy as np

# 프레임마다 미분 IK 적분 간격 (s, 가상 시간)과 수렴 판정 (관절 변화 rad)
STEP_DT = 0.05
EPS = 1e-6


def load_model(urdf: Path) -> mujoco.MjModel:
    """URDF → MuJoCo 모델 (메시·충돌·전달 장치 제거, 고정 링크 보존)."""
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for el in root.findall("transmission"):
        root.remove(el)
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", {"fusestatic": "false", "discardvisual": "true"})
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def target_rotation(approach_axis: np.ndarray, closing_axis: np.ndarray, frame: dict) -> np.ndarray:
    """끝 링크 축(approach, closing)을 목표 방향으로 보내는 회전 행렬 (기준 좌표)."""
    a_t = np.asarray(frame["approach"], dtype=float)
    c_t = np.asarray(frame["closing"], dtype=float)
    local = np.stack([approach_axis, closing_axis, np.cross(approach_axis, closing_axis)], axis=1)
    world = np.stack([a_t, c_t, np.cross(a_t, c_t)], axis=1)
    u, _, vt = np.linalg.svd(world @ local.T)
    return u @ vt


def main() -> None:
    """문제 JSON을 읽어 프레임마다 풀고 해 JSON을 쓴다."""
    spec = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    model = load_model(Path(spec["urdf"]))
    cfg = mink.Configuration(model)
    adr = [int(model.jnt_qposadr[model.joint(name).id]) for name in spec["joints"]]
    lo = np.array([model.jnt_range[model.joint(n).id][0] for n in spec["joints"]])
    hi = np.array([model.jnt_range[model.joint(n).id][1] for n in spec["joints"]])
    a_ax = np.asarray(spec["approach_axis"], dtype=float)
    c_ax = np.asarray(spec["closing_axis"], dtype=float)
    mode = spec["orientation_mode"]
    w = float(spec["orientation_weight"])
    rot_cost = np.zeros(3) if mode == "none" else np.full(3, w)
    if mode == "approach":
        # FrameTask의 회전 오차는 끝 링크 좌표: 접근 축 성분 비용을 0으로 (그 둘레 회전 자유)
        rot_cost = w * (1.0 - np.abs(a_ax))
    task = mink.FrameTask(
        frame_name=spec["tip_link"], frame_type="body", position_cost=1.0,
        orientation_cost=rot_cost, lm_damping=float(spec["damping"]),
    )  # fmt: skip
    limits = [mink.ConfigurationLimit(model)]
    out: list[list[float] | None] = []
    had = False
    for frame in spec["frames"]:
        if frame is None:
            out.append(None)
            had = False
            continue
        if not had:
            q = cfg.q.copy()
            q[adr] = (lo + hi) / 2
            cfg.update(q)
        rot = target_rotation(a_ax, c_ax, frame)
        se3 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(rot), np.asarray(frame["position"], dtype=float)
        )
        task.set_target(se3)
        for _ in range(int(spec["max_iterations"])):
            vel = mink.solve_ik(cfg, [task], STEP_DT, solver="daqp", damping=1e-6, limits=limits)
            cfg.integrate_inplace(vel, STEP_DT)
            if np.linalg.norm(vel[adr]) * STEP_DT < EPS:
                break
        out.append([float(v) for v in cfg.q[adr]])
        had = True
    Path(sys.argv[2]).write_text(json.dumps({"q": out}), "utf-8")


if __name__ == "__main__":
    main()
