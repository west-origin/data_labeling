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
- `generate_hand3d_scenario` (`hand3d.py`): 정답 카메라 좌표 3D를 아는 손 시퀀스 (2D 투영, 손 중심
  3D 점, 손바닥만 손 깊이인 합성 깊이). 정답 = 프레임마다 21관절 카메라 좌표 (손가락 끝 3D,
  ADR 0044). 메모리에서만 쓴다 (`generate_all`이 파일로 쓰지 않는다).
- `generate_checkerboard_video` (`calibration.py`): 정답 내부 파라미터로 렌더한 체커보드 영상.
  정답 = 렌더에 쓴 fx·fy·cx·cy (`dlp calib intrinsics` 시험, ADR 0045). `generate_all`은
  쓰지 않는다.
- `generate_arm_scenario` (`arm.py`): 정답 팔꿈치 위치(일부 화면 밖)를 아는 1인칭 착용자 팔과
  가짜 자세 모델 출력(좌우 뒤바뀐 팔 사슬), 착용자 몸·다른 사람·손에 든 도구 박스. 정답 = 팔꿈치
  위치·화면 안 여부, 억제할 박스 (ADR 0058). 메모리에서만 쓴다.
- `generate_all`: 위 모두를 디렉터리에 쓴다 (`dlp fixtures generate`, `make fixtures`).
  hand3d·체커보드·팔은 쓰지 않는다.

`__init__`에서 내보내지 않고 모듈에서 바로 가져다 쓰는 생성기 (모두 메모리에서만 쓴다)
- `generate_camera_motion` (`camera_motion.py`): 정답 카메라 자세·IMU 신호·월드 좌표 손을 아는 머리
  카메라 움직임 (ADR 0046, 카메라 자세·월드 단계 시험).
- `hand_points_for_grasp`, `world_hand_labels` 등 (`retarget.py`): 정답 끝단 궤적에서 거꾸로 만든
  손 점과 월드 좌표 3D 라벨 (ADR 0047, 로봇 리타기팅·내보내기 시험).
- `generate_shape_scenario` (`shapes.py`): 정답 마스크를 아는 객체·손 영상 (ADR 0059, 마스크 단계
  시험).
- `aireview_scene` (`aireview.py`): AI 1차 검수 벤치 입력 (ADR 0040, `dlp aireview bench`).

시간 규약 (ADR 0019)
- 라벨 시각은 정수 ms다. 프레임 번호로 대신하지 않는다.
- 시간 구간 라벨(행동, 사이 구간, 손 상태, 관계)은 마스터 타임라인 ms다. 공간 라벨(박스, 키포인트,
  블러, 3D 궤적)의 키프레임은 그 스트림 영상의 PTS ms다. 합성 데이터는 거의 모두 바디캠(기준
  스트림) 하나이고 기준 스트림은 오프셋 0·배율 1이라 두 시각이 같다.
- 동기화 시나리오만 스트림마다 다른 시계를 쓴다. 신호 시각은 그 스트림 자기 시계 ms이고, 정답 시계
  `ClockTruth`가 master = offset_ms + stream_ms * clock_scale로 마스터 시각과 잇는다.
- 생성기는 고정 fps 합성 영상의 프레임 PTS를 `round(i * 1000 / fps)`로 정해 쓴다. 블러 영상
  (`video.py`)만 25~50 ms로 흔들리는 VFR 간격(`vfr_times`)이고, 팔 장면은 `step_ms` 간격이다.
  영상을 읽는 코드는 이렇게 계산하지 않고 PTS 인덱스를 쓴다. 생성기는 정답을 만드는 쪽이라 예외다.

정답 라벨의 출처는 `Source.HUMAN`이다 (검수된 정답처럼 다룬다). 프롬프트·입력 흉내 라벨(팔, 모양,
AI 검수 장면, 리타기팅 3D)은 `Source.MODEL`이다. 실제 영상·개인정보는 쓰지 않는다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from dlp_fixtures.actions import ActionScenario, generate_action_scenario
from dlp_fixtures.calibration import CheckerboardScenario, generate_checkerboard_video
from dlp_fixtures.hand3d import Hand3DScenario, generate_hand3d_scenario
from dlp_fixtures.io import write_jsonl
from dlp_fixtures.sessions import generate_sessions
from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_fixtures.video import BlurScenario, generate_blur_scenario
from dlp_fixtures.wiping import WipingScenario, generate_wiping_scenario

__all__ = [
    "ActionScenario",
    "BlurScenario",
    "CheckerboardScenario",
    "Hand3DScenario",
    "SyncScenario",
    "WipingScenario",
    "generate_action_scenario",
    "generate_all",
    "generate_blur_scenario",
    "generate_checkerboard_video",
    "generate_hand3d_scenario",
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
