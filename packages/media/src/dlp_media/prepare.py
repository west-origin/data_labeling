"""로컬 영상의 픽셀 방향을 정규화한 H.264 MP4 사본. 원본/기존 출력을 덮어쓰지 않는다.

FFmpeg 자동 회전을 사용하고 프레임 수·정확한 PTS·오디오/데이터 트랙·촬영 시각을 비교한다.
변환 실패나 검증 실패 시 임시 파일을 삭제하며 최종 경로에는 파일을 남기지 않는다.
진단 명령(media probe와 동일한 로컬 경계)이며 원본 저장소의 감사 접근을 대신하지 않는다.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_media.probe import MediaInfo, probe, require_upright
from dlp_media.pts import build_pts_index
from dlp_schema.common import Contract


class NormalizePolicy(Contract):
    crf: int = Field(ge=0, le=51)
    preset: Literal["fast", "veryfast", "medium"]
    threads: int = Field(gt=0)


class MediaPolicy(Contract):
    version: int
    normalize: NormalizePolicy


def load_policy(root: Path) -> NormalizePolicy:
    data: Any = yaml.safe_load((root / "config/policies/media.yaml").read_text("utf-8"))
    return MediaPolicy.model_validate(data).normalize


def prepare_video(src: Path, dst: Path, settings: NormalizePolicy) -> MediaInfo:
    """프레임 좌표와 시간이 일치하는 사본을 만들고 probe 결과를 반환한다.

    오디오/데이터 트랙은 복사한다. 지원하지 않는 코덱/태그는 삭제하지 않고 실패한다.
    이 사본으로 새 세션을 수집할 수 있다. 이미 라벨링한 세션의 영상을 교체하면 안 된다.
    """
    if dst.exists():
        raise FileExistsError(f"기존 파일을 덮어쓸 수 없습니다: {dst}")
    if dst.suffix.lower() != ".mp4":
        raise ValueError("출력은 .mp4 파일이어야 합니다")
    source = probe(src)
    video = source.video
    if video is None:
        raise ValueError("비디오 트랙이 없습니다")
    if video.width % 2 or video.height % 2:
        raise ValueError("홀수 크기 영상은 좌표 보존을 위해 자동으로 잘라내지 않습니다")
    if video.rotation_degrees % 90:
        raise ValueError("90도 배수가 아닌 표시 회전은 좌표 보존을 위해 거부합니다")
    before = build_pts_index(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        candidate = Path(tmp) / "prepared.mp4"
        tb = video.time_base
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-copyts",
                "-i",
                str(src),
                "-map",
                "0",
                "-map_metadata",
                "0",
                "-c",
                "copy",
                "-c:v:0",
                "libx264",
                "-pix_fmt:v:0",
                "yuv420p",
                "-threads:v:0",
                str(settings.threads),
                "-preset:v:0",
                settings.preset,
                "-crf:v:0",
                str(settings.crf),
                "-fps_mode:v:0",
                "passthrough",
                "-enc_time_base:v:0",
                f"{tb.numerator}:{tb.denominator}",
                "-video_track_timescale",
                str(tb.denominator),
                "-metadata:s:v:0",
                "rotate=0",
                "-movflags",
                "+faststart",
                str(candidate),
            ],
            check=True,
            capture_output=True,
        )
        prepared = probe(candidate)
        require_upright(prepared)
        after = build_pts_index(candidate)
        if len(before) != len(after) or any(
            before.frame_ms(i) != after.frame_ms(i) for i in range(len(before))
        ):
            raise ValueError("정규화 사본의 프레임 수 또는 PTS가 원본과 다릅니다")
        expected_size = (
            (video.height, video.width)
            if video.rotation_degrees % 180
            else (video.width, video.height)
        )
        if prepared.video is None or (prepared.video.width, prepared.video.height) != expected_size:
            raise ValueError("정규화 사본의 해상도가 표시 방향과 맞지 않습니다")
        if (
            source.audio != prepared.audio
            or source.data_streams != prepared.data_streams
            or source.creation_time != prepared.creation_time
        ):
            raise ValueError("정규화 사본의 오디오/데이터 트랙 또는 촬영 시각이 원본과 다릅니다")
        # 같은 파일 시스템의 하드 링크로 원자적으로 공개한다.
        # 다른 실행이 만든 출력도 덮어쓰지 않는다.
        dst.hardlink_to(candidate)
    return probe(dst)
