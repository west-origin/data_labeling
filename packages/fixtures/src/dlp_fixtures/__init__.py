"""정답을 아는 합성 데이터 생성기. 모든 생성기는 같은 seed에 같은 결과를 낸다 (WP2).

알고리즘 모듈의 테스트는 실제 영상·개인정보 대신 이 생성기의 출력으로 작성하고, 생성기가 함께
내는 정답(오프셋·드리프트, 대상 박스, 경계, 커버리지 등)으로 판정한다. 저장소에는 결과 파일을
넣지 않는다 (`make fixtures` → `data/fixtures/`).

생성기와 정답
- `generate_sessions` (`sessions.py`): 작업자·장소·도메인 메타데이터를 가진 가짜 세션 수백 개.
  정답 = 작업자·장소 구조 자체 (분할 교집합 시험용, WP7).
- `generate_sync_scenario` (`sync.py`): 오프셋·드리프트를 아는 다중 스트림 (오디오 두드림, QR
  슬레이트, 장갑 압력, IMU). 정답 = 스트림별 `ClockTruth`, 두드림·슬레이트 마스터 시각 (WP4).
- `generate_blur_scenario` (`video.py`): 위치를 아는 블러 대상이 움직이는 VFR 영상.
  정답 = 프레임마다의 대상 박스(`BlurTrackPayload` 라벨) (WP5).
- `generate_action_scenario` (`actions.py`): 경계를 아는 행동 시퀀스 (손 키포인트, 장갑 압력,
  정답 행동·사이 구간·손 상태 라벨) (WP10).
- `generate_wiping_scenario` (`wiping.py`): 정답 도구-표면 접촉 구간과 커버리지를 아는 걸레질 (WP9).
- `generate_all`: 위 모두를 디렉터리에 쓴다 (`dlp fixtures generate`).

시간 규약: 라벨 시각은 정수 ms. 시간 구간 라벨은 마스터 타임라인, 공간 라벨 키프레임은 영상 PTS
(합성 데이터에서는 바디캠 프레임 시각과 같다, ADR 0019). 정답 라벨의 출처는 `Source.HUMAN`이다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dlp_fixtures.actions import ActionScenario, generate_action_scenario
from dlp_fixtures.io import write_jsonl
from dlp_fixtures.sessions import generate_sessions
from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_fixtures.video import BlurScenario, generate_blur_scenario
from dlp_fixtures.wiping import WipingScenario, generate_wiping_scenario

__all__ = [
    "ActionScenario",
    "BlurScenario",
    "SyncScenario",
    "WipingScenario",
    "generate_action_scenario",
    "generate_all",
    "generate_blur_scenario",
    "generate_sessions",
    "generate_sync_scenario",
    "generate_wiping_scenario",
]

# 동기화 시나리오의 녹화 시작 시각 (슬레이트 QR의 절대 시각 계산에 쓴다)
RECORDED_AT = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)


def generate_all(out_dir: Path, seed: int = 0, n_sessions: int = 300) -> dict[str, Path]:
    """모든 픽스처를 out_dir 아래에 쓴다.

    Args:
        out_dir: 출력 디렉터리 (없으면 만든다. 같은 이름 파일은 덮어쓴다).
        seed: 모든 생성기에 같은 seed를 쓴다.
        n_sessions: 가짜 세션 수.

    Returns:
        이름 → 경로. `sessions`(JSONL 파일), `sync`·`blur`·`actions`·`wiping`(디렉터리).
        - sync/: bodycam·third_person의 WAV·MP4, glove_right.parquet, imu.parquet, truth.json
        - blur/: bodycam.mp4, labels.jsonl (정답 블러 트랙)
        - actions/: labels.jsonl, entities.jsonl, glove_right.parquet
        - wiping/: labels.jsonl (입력 라벨), truth_relations.jsonl (정답 관계)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sessions": out_dir / "sessions.jsonl",
        "sync": out_dir / "sync",
        "blur": out_dir / "blur",
        "actions": out_dir / "actions",
        "wiping": out_dir / "wiping",
    }
    write_jsonl(paths["sessions"], generate_sessions(n_sessions, seed))
    generate_sync_scenario(seed, recorded_at=RECORDED_AT).write(paths["sync"])
    blur = generate_blur_scenario(seed)
    paths["blur"].mkdir(parents=True, exist_ok=True)
    blur.write(paths["blur"] / "bodycam.mp4")
    write_jsonl(paths["blur"] / "labels.jsonl", blur.labels)
    generate_action_scenario(seed).write(paths["actions"])
    wiping = generate_wiping_scenario(seed)
    paths["wiping"].mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["wiping"] / "labels.jsonl", wiping.labels)
    write_jsonl(paths["wiping"] / "truth_relations.jsonl", wiping.truth_relations)
    return paths
