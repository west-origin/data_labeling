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

좌표·비용:
- 끝 링크에 보조 site `dlp_tcp`를 단다. site 좌표의 z = approach_axis, x = closing_axis,
  y = z x x (끝 링크 좌표의 축을 그대로 쓰므로 위치는 끝 링크 원점, 회전만 다르다).
- mink FrameTask의 회전 오차는 site 자기 좌표(local)의 회전 벡터이고 비용은 성분별 대각 가중치다.
  그래서 approach 모드는 비용 [w, w, 0]으로 site z(= 접근 축) 둘레 회전만 정확히 자유로 둔다. 끝
  링크 좌표에서 바로 비용을 주면 접근 축이 좌표축(±x, ±y, ±z)이 아닐 때 롤에도 비용이 붙는다. full은
  [w, w, w], none은 0이다.
- 목표 회전: site 축(x = 닫힘, z = 접근)을 목표 닫힘·접근 축으로 보내는 회전 (기준 좌표).

시작점: 첫 목표 프레임은 관절 범위 가운데, 목표가 끊긴 뒤는 마지막 해에서 시작한다 (가까운 자세
가지).
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


# 끝 링크에 다는 도구 중심점 site 이름
SITE = "dlp_tcp"


def tcp_frame(approach_axis: np.ndarray, closing_axis: np.ndarray) -> np.ndarray:
    """끝 링크 좌표에서 본 site 축 행렬 (열 = site x, y, z = 닫힘, 접근 x 닫힘, 접근)."""
    return np.stack([closing_axis, np.cross(approach_axis, closing_axis), approach_axis], axis=1)


def load_model(
    urdf: Path, tip_link: str, approach_axis: np.ndarray, closing_axis: np.ndarray
) -> mujoco.MjModel:
    """URDF → MuJoCo 모델 (메시·충돌·전달 장치 제거, 고정 링크 보존, 끝 링크에 `dlp_tcp` site)."""
    root = ET.parse(urdf).getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    for el in root.findall("transmission"):
        root.remove(el)
    ext = ET.SubElement(root, "mujoco")
    ET.SubElement(ext, "compiler", {"fusestatic": "false", "discardvisual": "true"})
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, tcp_frame(approach_axis, closing_axis).flatten())
    spec.body(tip_link).add_site(name=SITE, quat=quat.tolist())
    return spec.compile()


def target_rotation(frame: dict) -> np.ndarray:
    """site 목표 자세의 회전 행렬 (기준 좌표, 열 = 목표 닫힘, 접근 x 닫힘, 접근).

    목표 닫힘 축은 접근 축에 수직이다 (`dlp_retarget.targets`). 수치 오차는 SVD로 가장 가까운 회전에
    맞춘다.
    """
    a_t = np.asarray(frame["approach"], dtype=float)
    c_t = np.asarray(frame["closing"], dtype=float)
    u, _, vt = np.linalg.svd(tcp_frame(a_t, c_t))
    return u @ vt


def orientation_cost(mode: str, weight: float) -> np.ndarray:
    """site 좌표 회전 오차 성분별 비용 (모듈 설명: approach는 site z 둘레 0)."""
    if mode == "none":
        return np.zeros(3)
    if mode == "approach":
        return np.array([weight, weight, 0.0])
    return np.full(3, weight)


def main() -> None:
    """문제 JSON을 읽어 프레임마다 풀고 해 JSON을 쓴다."""
    spec = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    a_ax = np.asarray(spec["approach_axis"], dtype=float)
    c_ax = np.asarray(spec["closing_axis"], dtype=float)
    model = load_model(Path(spec["urdf"]), spec["tip_link"], a_ax, c_ax)
    cfg = mink.Configuration(model)
    adr = [int(model.jnt_qposadr[model.joint(name).id]) for name in spec["joints"]]
    lo = np.array([model.jnt_range[model.joint(n).id][0] for n in spec["joints"]])
    hi = np.array([model.jnt_range[model.joint(n).id][1] for n in spec["joints"]])
    rot_cost = orientation_cost(spec["orientation_mode"], float(spec["orientation_weight"]))
    task = mink.FrameTask(
        frame_name=SITE, frame_type="site", position_cost=1.0,
        orientation_cost=rot_cost, lm_damping=float(spec["damping"]),
    )  # fmt: skip
    limits = [mink.ConfigurationLimit(model)]
    out: list[list[float] | None] = []
    # 첫 목표 프레임만 관절 범위 가운데에서 시작한다. 끊긴 뒤는 cfg에 남은 마지막 해에서 이어 간다
    q = cfg.q.copy()
    q[adr] = (lo + hi) / 2
    cfg.update(q)
    for frame in spec["frames"]:
        if frame is None:
            out.append(None)
            continue
        rot = target_rotation(frame)
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
    Path(sys.argv[2]).write_text(json.dumps({"q": out}), "utf-8")


if __name__ == "__main__":
    main()
