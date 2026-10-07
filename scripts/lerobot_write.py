"""LeRobot 공식 API로 에피소드를 쓴다 (dlp export lerobot이 격리 환경에서 실행한다).

    python scripts/lerobot_write.py <package.json> <출력 루트>

package.json은 dlp_export.lerobot.write_package가 만든다: 에피소드마다 블러본 경로,
프레임별 특징(npz),
그 시각에 보이던 블러본 프레임 번호(PTS로 고른 것)와 task 문자열.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def frames(video: str, wanted: np.ndarray, size: tuple[int, int]):
    """wanted[k]번째(표시 순서) 프레임을 차례로 낸다.

    같은 번호가 이어지면 같은 프레임을 다시 낸다.
    """
    with av.open(video) as c:
        decoded = c.decode(c.streams.video[0])
        i, frame = -1, None
        cached_i, cached = -1, None
        for target in wanted:
            while i < target:
                frame = next(decoded)
                i += 1
            if cached_i != i:
                rgb = frame.to_ndarray(format="rgb24")
                if (rgb.shape[1], rgb.shape[0]) != size:
                    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                cached_i, cached = i, rgb
            yield cached


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    root = Path(sys.argv[2])
    feats = {k: {**v, "shape": tuple(v["shape"])} for k, v in spec["features"].items()}
    ds = LeRobotDataset.create(
        spec["repo_id"], fps=spec["fps"], features=feats, root=root,
        robot_type=spec["robot_type"], use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec=spec["vcodec"]),
    )  # fmt: skip
    size = (spec["width"], spec["height"])
    for ep in spec["episodes"]:
        data = np.load(ep["npz"])
        for k, img in enumerate(frames(ep["video"], data["frame_index"], size)):
            ds.add_frame({
                "observation.images.bodycam": img,
                "observation.state": data["state"][k],
                "action": data["action"][k],
                "annotation.hand_state": data["hand_state"][k],
                "annotation.tool_surface_contact": data["tool_surface_contact"][k],
                "annotation.verb": data["verb"][k],
                "annotation.verification": data["verification"][k],
                "task": ep["tasks"][k],
            })  # fmt: skip
        ds.save_episode()
    ds.finalize()
    print(json.dumps({"episodes": len(spec["episodes"])}))


if __name__ == "__main__":
    main()
