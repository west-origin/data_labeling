"""LeRobot 공식 로더로 내보내기를 다시 읽어 요약을 JSON으로 출력한다.

python scripts/lerobot_check.py <루트> <repo_id>
"""

from __future__ import annotations

import json
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    ds = LeRobotDataset(sys.argv[2], root=sys.argv[1])
    out = {
        "frames": len(ds),
        "episodes": ds.num_episodes,
        "fps": ds.fps,
        "features": sorted(ds.features),
        "tasks": sorted(ds.meta.tasks.index.tolist()),
        "samples": [],
    }
    for i in sorted({0, len(ds) // 2, len(ds) - 1}):
        x = ds[i]
        out["samples"].append({
            "index": int(x["index"]), "episode_index": int(x["episode_index"]),
            "timestamp": float(x["timestamp"]), "task": x["task"],
            "image_shape": list(x["observation.images.bodycam"].shape),
            "state": [float(v) for v in x["observation.state"]],
            "hand_state": [int(v) for v in x["annotation.hand_state"]],
            "tool_surface_contact": [int(v) for v in x["annotation.tool_surface_contact"]],
            "verb": [int(v) for v in x["annotation.verb"]],
            "verification": [int(v) for v in x["annotation.verification"]],
        })  # fmt: skip
    print(json.dumps(out))


if __name__ == "__main__":
    main()
