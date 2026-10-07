"""LeRobot 공식 로더로 내보내기를 다시 읽어 요약을 JSON으로 출력한다.

python scripts/lerobot_check.py <루트> <repo_id>

`dlp export lerobot`(WP15, ADR 0018)이 `lerobot_write.py`로 쓴 직후, 같은 격리 환경
(`scripts/lerobot-env`, 오프라인: `HF_HUB_OFFLINE=1`)에서 `dlp_export.lerobot.run_script`가
실행한다. 작업공간 밖에서 도므로 `dlp_*` 패키지를 가져오지 않는다 (ruff만 검사, pyright 범위 밖).

출력(표준 출력 JSON 한 줄): 프레임 수, 에피소드 수, fps, 특징 이름, task 목록, 그리고
처음·가운데·끝 프레임 표본(인덱스, 에피소드, 타임스탬프(초), 영상 크기·평균 밝기, 상태·주석 값).
호출자가 이 값을 기대값(에피소드 수, 주석 값)과 비교해 공식 로더로 읽힘을 확인한다.
"""

from __future__ import annotations

import json
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    """명령줄 인자 `<루트> <repo_id>`로 데이터셋을 열고 요약 JSON을 출력한다.

    영상 특징은 `observation.images.`로 시작하는 첫 키를 쓴다. 표본 프레임은 중복 없이 최대 3개다.
    `annotation.*` 특징 이름은 `dlp_export.lerobot`이 정하는 이름과 같아야 한다.
    """
    ds = LeRobotDataset(sys.argv[2], root=sys.argv[1])
    out = {
        "frames": len(ds),
        "episodes": ds.num_episodes,
        "fps": ds.fps,
        "features": sorted(ds.features),
        "tasks": sorted(ds.meta.tasks.index.tolist()),
        "samples": [],
    }
    video_key = next(k for k in ds.features if k.startswith("observation.images."))
    for i in sorted({0, len(ds) // 2, len(ds) - 1}):
        x = ds[i]
        out["samples"].append({
            "index": int(x["index"]), "episode_index": int(x["episode_index"]),
            "timestamp": float(x["timestamp"]), "task": x["task"],
            "image_shape": list(x[video_key].shape),
            "image_mean": float(x[video_key].float().mean()),  # 0~1, 프레임 내용 확인용
            "state": [float(v) for v in x["observation.state"]],
            "hand_state": [int(v) for v in x["annotation.hand_state"]],
            "tool_surface_contact": [int(v) for v in x["annotation.tool_surface_contact"]],
            "verb": [int(v) for v in x["annotation.verb"]],
            "verification": [int(v) for v in x["annotation.verification"]],
        })  # fmt: skip
    print(json.dumps(out))


if __name__ == "__main__":
    main()
