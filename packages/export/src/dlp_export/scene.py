"""장면 좌표의 바디캠 자세를 에피소드 프레임마다 싣는다 (`dlp export lerobot|gr00t`, ADR 0070).

세션의 위치 기록(`dlp scene3d localize`, ADR 0067: 바디캠 프레임별 카메라 자세, 그 세션 before
복원의 장면 좌표)에서 에피소드 프레임 시각(`Episode.times_ms`, 마스터)의 자세를 읽는다. 프레임
시각은 위치 기록 스트림의 영상 시각으로 옮기고(`frames.stream_ms`), 그 시각의 자세는
`dlp_scene3d.localize.PoseTrack.at`이 이웃한 두 자세 사이에서 보간한다 (중심 선형, 회전 구면,
간격 한도 `scene3d.yaml interpolation.max_gap_ms`). 같은 행에 영상 프레임·손 3D(카메라 좌표)·
촉각장갑 값(ADR 0069)이 있으므로 이 자세로 그 프레임을 전후 사진으로 만든 공간 안에 놓는다.

규칙
- 위치 기록이 지금 전후 복원으로 만든 것(`current_localization`)이고, 그 기록을 만든 블러본이 이
  내보내기가 쓰는 블러본(`ExportSession.render_hashes`)과 같을 때만 싣는다. 아니면 싣지 않고 세션
  기록(`stale`, `other_video`)과 경고로 남긴다.
- 자세를 지어내지 않는다. 위치 기록 밖(외삽 없음), 이웃한 두 자세의 간격이 한도를 넘는 곳, 쓸 위치
  기록이 없는 세션의 프레임은 "없음"이다: 값 0과 표시 0 (손 3D 점·촉각과 같은 표현).
- 장면 좌표계는 세션마다 그 세션의 before 복원이다 (세션끼리 공통 좌표가 아니다). 길이 단위는 그
  복원이 미터일 때(`metric`)만 미터다.
- 위치 기록은 라벨링 버킷에 있는 모델 출력이다 (승인된 블러본에서 만들고 사람이 검수하지 않는다).
  원본 버킷을 읽지 않는다.
- 결과에 복원·위치 기록 태그, 렌더 해시, 경고 원문(내부 세션 ID가 든 예외 문장)을 넣지 않는다.
  세션마다 고정된 값이라 내보내기마다 다른 가명(ADR 0027)을 넘어 같은 세션을 잇게 된다.

싣는 곳 (자세를 실은 세션이 하나라도 있을 때만 생긴다)
- LeRobot: 특징 `observation.scene_camera` (float32[8]: x, y, z, qw, qx, qy, qz, present).
- GR00T: `observation.state` 끝(촉각 구간 뒤)의 `scene_camera` 구간(7)과
  `observation.scene_camera_valid`(float32[1]). action에는 넣지 않는다.
- 값: x·y·z = 카메라 중심, q = 카메라 → 장면 회전 (w ≥ 0). 카메라 좌표의 점 p(OpenCV 축, 손 3D와
  같은 좌표)는 장면 좌표 R(q) p + (x, y, z)다. 위치 기록의 `x_cam = R x + t`에서 R(q) = Rᵀ,
  중심 = -Rᵀ t.
- 두 형식 모두 `meta/dlp_scene.json`(`SceneData.meta`)과 manifest `details.scene`.

흐름: `runner.build_episodes`가 세션마다 `SceneCollector.add`를 부르고, 끝에 `result()`의
`SceneData`를 `lerobot.write_package`·`gr00t.write_gr00t`에 넘긴다. 위치 기록이 있는 세션이
하나도 없으면 `result()`는 None이고 결과물은 이 기능이 없던 때와 같다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_export.frames import stream_ms
from dlp_export.source import ExportSession
from dlp_media.storage import ObjectStore
from dlp_scene3d.localize import PoseTrack, current_localization, stored_localization
from dlp_scene3d.pipeline import Scene3dError
from dlp_schema.scene3d import EgoLocalization
from dlp_schema.session import Stream

# meta/dlp_scene.json 형식 이름과 판 (판 규칙은 dlp-tactile과 같다: 필드를 더하면 부 버전)
FORMAT = "dlp-scene"
FORMAT_VERSION = "1.0"
# LeRobot 특징·npz 배열·GR00T state 키·유효 표시 열과 그 안 이름
FEATURE = "observation.scene_camera"
NPZ_KEY = "scene_camera"
STATE_KEY = "scene_camera"
VALID_COLUMN = "observation.scene_camera_valid"
VALID_NAME = "scene_camera"
# 값 순서: 카메라 중심 x·y·z, 카메라 → 장면 회전의 단위 사원수 w·x·y·z
FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz")
NAMES = tuple(f"{STATE_KEY}.{f}" for f in FIELDS)
PRESENT_NAME = f"{STATE_KEY}.present"

# 위치 기록 찾기 결과: 쓸 수 있는 기록, 기록 없음, 지금 복원으로 만든 기록이 아님
Found = EgoLocalization | Literal["none", "stale"]

# meta/dlp_scene.json 설명 (구매자용 영어, 계약의 의미라 정책 값이 아니다)
_POSE = (
    "Bodycam camera pose at the frame time, in that session's scene frame: x, y, z = camera "
    "centre; qw, qx, qy, qz = unit quaternion (w >= 0) of the camera-to-scene rotation. A point "
    "p_cam in the camera frame (OpenCV axes: x right, y down, z forward; the frame of the hand 3D "
    "points in observation.state) is p_scene = R(q) p_cam + (x, y, z)."
)
_FRAME = (
    "Each session has its own scene frame: the 3D reconstruction from that session's photos "
    "taken before the work. Poses of different sessions are not in a common frame. The gravity "
    "direction is unknown."
)
_UNITS = (
    "Metres when the session's reconstruction is metric (sessions[].metric: scale from a marker "
    "of known size). Otherwise arbitrary reconstruction units; then hand 3D points (metres) must "
    "not be moved into the scene frame."
)
_TIME = (
    "Frame times are the episode's master-timeline times mapped to the localized stream's video "
    "time. The pose is interpolated between the two neighbouring localized bodycam frames (centre "
    "linear, rotation spherical)."
)
_MISSING = (
    "All values 0 with the present/valid flag 0 when the frame is outside the localized range "
    "(no extrapolation), when the two neighbouring localized frames are more than max_gap_ms "
    "apart, or when the session has no usable localization (see sessions[].status)."
)
_SOURCE = (
    "Model output (bodycam feature matching and PnP against the 3D reconstruction, computed on "
    "the approved blurred video), not reviewed by a person."
)


def quaternion_wxyz(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """회전 행렬 (3, 3) → 단위 사원수 (w, x, y, z), w ≥ 0.

    셰퍼드 방법: 가장 큰 성분부터 구해 나눗셈 분모가 0에 가까워지지 않게 한다 (180도 회전 포함).
    `dlp_prelabel.posemath.quat_from_matrix`와 같은 식이다 (패키지 의존을 늘리지 않으려고 둔다).
    """
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    out = np.asarray(q, dtype=np.float64)
    out /= np.linalg.norm(out)
    # q와 -q는 같은 회전이다. 값을 하나로 정하려고 w ≥ 0 쪽을 고른다
    return -out if out[0] < 0 else out


class SceneSource(Protocol):
    """세션의 위치 기록을 찾는 곳 (`StoredScenes`, 시험은 대역)."""

    @property
    def max_gap_ms(self) -> int:
        """보간을 허용하는 이웃한 두 자세의 간격 상한 (ms)."""
        ...

    def find(self, session_id: str) -> Found:
        """지금 쓸 수 있는 위치 기록. 없으면 "none", 지금 복원으로 만든 것이 아니면 "stale"."""
        ...


@dataclass(frozen=True)
class StoredScenes:
    """라벨링 버킷의 위치 기록 (`dlp_scene3d.localize`).

    Attributes:
        labeling: 라벨링 버킷 저장소 (위치 기록·복원 기록을 읽는다).
        max_gap_ms: `scene3d.yaml interpolation.max_gap_ms`.
    """

    labeling: ObjectStore
    max_gap_ms: int

    def find(self, session_id: str) -> Found:
        """`SceneSource.find`. 복원이 바뀌었거나 낡아 기록과 맞지 않으면 "stale"이다."""
        if stored_localization(self.labeling, session_id) is None:
            return "none"
        try:
            return current_localization(self.labeling, session_id)
        except Scene3dError:
            return "stale"


@dataclass(frozen=True)
class ScenePoses:
    """에피소드 하나의 프레임별 자세 (없음은 값 0)."""

    values: NDArray[np.float32]  # (프레임 수, 7): x, y, z, qw, qx, qy, qz
    valid: NDArray[np.bool_]  # (프레임 수,)


def frame_poses(
    record: EgoLocalization, stream_of: Stream, frame_ms: NDArray[np.float64], max_gap_ms: int
) -> ScenePoses:
    """프레임 시각(마스터 ms)마다 장면 좌표의 카메라 중심과 카메라 → 장면 사원수.

    Args:
        record: 위치 기록 (자세 `x_cam = R x + t`, 시각은 그 스트림의 영상 시각).
        stream_of: 위치 기록 스트림 (마스터 시각 → 영상 시각에 쓴다).
        frame_ms: 에피소드 프레임 시각 (마스터 ms).
        max_gap_ms: 보간을 허용하는 이웃 간격 상한.
    """
    n = frame_ms.size
    values = np.zeros((n, len(FIELDS)), np.float32)
    valid = np.zeros(n, bool)
    if not record.poses:
        return ScenePoses(values, valid)
    track = PoseTrack.of(record)
    for k, t in enumerate(frame_ms):
        pose = track.at(stream_ms(stream_of, float(t)), max_gap_ms)
        if pose is None:
            continue
        values[k, :3] = pose.center
        values[k, 3:] = quaternion_wxyz(pose.r.T)
        valid[k] = True
    return ScenePoses(values, valid)


@dataclass(frozen=True)
class SceneData:
    """한 내보내기의 장면 좌표 자세 전체 (`SceneCollector.result`).

    Attributes:
        episodes: 에피소드마다(`BuiltEpisodes.items`와 같은 순서) 자세. 쓸 위치 기록이 없던
            세션은 None (전부 없음으로 채운다).
        frames: 에피소드마다 프레임 수.
        sessions: 세션 기록 (가명 세션 ID, 상태, 실었으면 미터 여부·채움 수).
        max_gap_ms: 적용한 보간 간격 상한.
    """

    episodes: list[ScenePoses | None]
    frames: list[int]
    sessions: list[dict[str, Any]]
    max_gap_ms: int

    @property
    def carried(self) -> bool:
        """자세를 실은 세션이 있는가. 없으면 특징·열을 더하지 않고 세션 기록만 남긴다."""
        return any(e is not None for e in self.episodes)

    def _poses(self, i: int) -> ScenePoses:
        """에피소드 i의 자세. 실은 기록이 없으면 전부 없음(값 0, 표시 0)이다."""
        got = self.episodes[i]
        if got is not None:
            return got
        n = self.frames[i]
        return ScenePoses(np.zeros((n, len(FIELDS)), np.float32), np.zeros(n, bool))

    # LeRobot ---------------------------------------------------------------------------------

    def lerobot_features(self) -> dict[str, Any]:
        """LeRobot `features`에 더할 정의: float32 [x, y, z, qw, qx, qy, qz, present]."""
        names = [*NAMES, PRESENT_NAME]
        return {FEATURE: {"dtype": "float32", "shape": [len(names)], "names": names}}

    def lerobot_frame_arrays(self) -> dict[str, str]:
        """LeRobot 특징 → npz 배열 키 (`lerobot.FRAME_ARRAYS`에 더한다)."""
        return {FEATURE: NPZ_KEY}

    def lerobot_arrays(self, i: int) -> dict[str, NDArray[np.float32]]:
        """에피소드 i의 npz 배열 (프레임 수, 8), 마지막 열이 present."""
        p = self._poses(i)
        return {NPZ_KEY: np.hstack([p.values, p.valid[:, None].astype(np.float32)])}

    # GR00T -----------------------------------------------------------------------------------

    def gr00t_modality(self, start: int) -> dict[str, dict[str, int]]:
        """modality.json state에 더할 구간 (`start`부터 7칸)."""
        return {STATE_KEY: {"start": start, "end": start + len(FIELDS)}}

    def gr00t_arrays(self, i: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """에피소드 i의 (state에 붙일 값 (프레임 수, 7), 유효 표시 (프레임 수, 1))."""
        p = self._poses(i)
        return np.ascontiguousarray(p.values), p.valid[:, None].astype(np.float32)

    # 기록 ------------------------------------------------------------------------------------

    def meta(self, fmt: Literal["lerobot", "gr00t"]) -> dict[str, Any]:
        """meta/dlp_scene.json·manifest `details.scene` 내용 (태그·렌더 해시·내부 ID 없음).

        Args:
            fmt: 형식. 값이 실린 위치를 그 형식의 이름으로 적는다 (LeRobot `feature`·`present`,
                GR00T `state_key`·`valid_column`·`valid_name`).
        """
        where = (
            {"feature": FEATURE, "present": PRESENT_NAME}
            if fmt == "lerobot"
            else {"state_key": STATE_KEY, "valid_column": VALID_COLUMN, "valid_name": VALID_NAME}
        )
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "source": _SOURCE,
            "pose": _POSE,
            "frame": _FRAME,
            "units": _UNITS,
            "time_alignment": _TIME,
            "missing": _MISSING,
            "interp_max_gap_ms": self.max_gap_ms,
            "names": list(NAMES),
            **where,
            "sessions": self.sessions,
        }


@dataclass
class SceneCollector:
    """세션마다 위치 기록을 찾아 프레임 자세를 모은다 (`runner.build_episodes`).

    Attributes:
        source: 위치 기록을 찾는 곳 (`StoredScenes`).
        warnings: 내보내기 경고 (manifest details.warnings, 세션은 가명으로 적는다).
    """

    source: SceneSource
    warnings: list[str] = field(default_factory=list[str])
    _episodes: list[ScenePoses | None] = field(default_factory=list[ScenePoses | None])
    _frames: list[int] = field(default_factory=list[int])
    _sessions: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    _found: int = 0  # 위치 기록이 있는 세션 수 (쓸 수 없는 기록 포함)

    def add(self, es: ExportSession, sid: str, frame_ms: NDArray[np.float64]) -> None:
        """세션 하나(에피소드 하나)의 자세를 더한다. 내보낼 순서대로 부른다.

        세션 기록의 상태(`status`)는 넷 중 하나다: exported(실음), no_localization(위치 기록
        없음), stale(지금 전후 복원으로 만든 기록이 아님), other_video(이 내보내기와 다른
        블러본으로 만든 기록).

        Args:
            es: 내보낼 세션 (스트림, 이 내보내기가 확인한 블러본 렌더 해시).
            sid: 결과물에 적을 세션 ID (가명).
            frame_ms: 에피소드 프레임 시각 (`Episode.times_ms`, 마스터 ms).
        """
        found = self.source.find(es.session.session_id)
        poses: ScenePoses | None = None
        if found == "none":
            report: dict[str, Any] = {"status": "no_localization"}
        else:
            self._found += 1
            poses, report = self._use(es, sid, found, frame_ms)
        self._episodes.append(poses)
        self._frames.append(int(frame_ms.size))
        self._sessions.append({"session_id": sid, **report})

    def result(self) -> SceneData | None:
        """모은 값. 위치 기록이 있는 세션이 하나도 없으면 None (결과물이 예전과 같다)."""
        if self._found == 0:
            return None
        return SceneData(self._episodes, self._frames, self._sessions, self.source.max_gap_ms)

    def _use(
        self,
        es: ExportSession,
        sid: str,
        found: EgoLocalization | Literal["stale"],
        frame_ms: NDArray[np.float64],
    ) -> tuple[ScenePoses | None, dict[str, Any]]:
        """기록을 쓸 수 있으면 (프레임 자세, 기록), 아니면 (None, 상태와 경고)."""
        redo = "dlp scene3d localize를 다시 돌리고 내보낸다"
        if found == "stale":
            self.warnings.append(
                f"{sid}: 위치 기록이 지금 전후 복원으로 만든 것이 아니라 장면 좌표 자세를 싣지 "
                f"않음 ({redo})"
            )
            return None, {"status": "stale"}
        stream = next((s for s in es.session.streams if s.stream_id == found.stream_id), None)
        if stream is None or es.render_hashes.get(found.stream_id) != found.render_hash:
            self.warnings.append(
                f"{sid}: 위치 기록을 만든 블러본이 이 내보내기의 블러본과 달라 장면 좌표 자세를 "
                f"싣지 않음 ({redo})"
            )
            return None, {"status": "other_video"}
        poses = frame_poses(found, stream, frame_ms, self.source.max_gap_ms)
        valid = int(poses.valid.sum())
        if valid == 0:
            self.warnings.append(
                f"{sid}: 에피소드 프레임 중 장면 좌표 자세가 있는 프레임이 없음 (위치 기록이 "
                "에피소드 구간과 겹치지 않거나 자세 사이 간격이 interpolation.max_gap_ms보다 김)"
            )
        return poses, {
            "status": "exported",
            "metric": found.metric,
            "model_version": found.model_version,
            "frames": int(frame_ms.size),
            "frames_valid": valid,
            "localized_frames": found.localized,
            "localization_frames_tried": found.frame_count,
        }
