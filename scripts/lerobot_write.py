"""LeRobot 공식 API로 에피소드를 쓴다 (dlp export lerobot이 격리 환경에서 실행한다).

    python scripts/lerobot_write.py <package.json> <출력 루트>

package.json은 dlp_export.lerobot.write_package가 만든다: 에피소드마다 블러본 경로,
프레임별 특징(npz), 그 시각에 보이던 블러본 프레임 번호(PTS로 고른 것)와 task 문자열.

실행 환경: `scripts/lerobot-env`(uv.lock 고정, PyTorch CPU판)에서 `dlp_export.lerobot.run_script`가
부른다. 작업공간 밖이라 `dlp_*` 패키지를 쓰지 않는다 (ruff만 검사, pyright 범위 밖).
시간 처리: 고정 fps 에피소드의 k번째 프레임에 들어갈 블러본 프레임은 호출자가 PTS 인덱스로 미리
골라 `frame_index`로 넘긴다. 이 스크립트는 프레임 번호로 시간을 계산하지 않는다.
원본 버킷에는 접근하지 않는다 (블러본 로컬 사본만 읽는다).

package.json 키: `repo_id`, `fps`, `robot_type`, `vcodec`, `width`, `height`, `video_key`,
`features`(이름 → LeRobot 특징 정의), `episodes`(각각 `video`, `npz`, `tasks`).
npz 배열: `frame_index`, `state`, `action`, `hand_state`, `tool_surface_contact`, `verb`,
`verification` (첫 축 = 에피소드 프레임).
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

    인자:
        video: 블러본 영상 경로.
        wanted: 내보낼 프레임마다의 원본 프레임 번호(디코드 순서 = 표시 순서로 가정). 감소하지
            않아야 한다 (앞으로만 디코드하므로, 줄어들면 이전 프레임을 다시 내지 못하고 현재
            프레임을 낸다).
        size: 출력 (너비, 높이). 다르면 `INTER_AREA`로 줄인다.

    반환: RGB `uint8` 배열(높이, 너비, 3)을 내는 생성기. 영상이 `wanted`보다 짧으면
    `StopIteration`이 생성기 안에서 `RuntimeError`로 바뀌어 실패한다.
    """
    with av.open(video) as c:
        decoded = c.decode(c.streams.video[0])
        # i: 지금까지 디코드한 마지막 프레임 번호, cached: 그 프레임의 RGB(크기 조정 후)
        i, frame = -1, None
        cached_i, cached = -1, None
        for target in wanted:
            # 원하는 번호까지 앞으로만 디코드한다 (VFR 블러본이라 건너뛰거나 반복될 수 있다)
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
    """명령줄 인자 `<package.json> <출력 루트>`로 LeRobot v3.0 데이터셋을 만든다.

    에피소드마다 프레임을 넣고 `save_episode`, 마지막에 `finalize`(메타데이터·통계 기록)한다.
    표준 출력에 `{"episodes": N}`을 낸다. 출력 루트가 이미 있으면 `LeRobotDataset.create`가
    실패한다.
    """
    spec = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    root = Path(sys.argv[2])
    # JSON에는 튜플이 없어 shape가 리스트로 온다. LeRobot 특징 정의는 튜플을 기대한다
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
                spec["video_key"]: img,
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
