"""데이터 트랙이 있는 영상의 probe 회귀 테스트.

PyAV 19의 `DataStream`에는 `codec_tag`가 없어, 예전 `probe`는 데이터 트랙(GoPro `gpmd`, 타임코드
`tmcd`)이 있는 파일에서 `AttributeError`로 죽었다. 그러면 GoPro 바디캠 영상의 probe·prepare·ingest가
모두 실패하고 내장 IMU 추출(`GpmfExtractor`)에 닿지 못한다. 합성 타임코드 트랙으로 이를 재현한다.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dlp_media import probe as probe_mod
from dlp_media.probe import probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 없음")


def _timecode_video(path: Path) -> None:
    """1초짜리 H.264 MOV에 타임코드(`tmcd`) 데이터 트랙을 붙인다."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10",
         "-t", "1", "-timecode", "00:00:00:00", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(path)],
        check=True,
    )  # fmt: skip


def test_probe_reads_data_track_tag(tmp_path: Path) -> None:
    """데이터 트랙이 있어도 probe가 죽지 않고, ffprobe로 태그 `tmcd`를 읽는다."""
    src = tmp_path / "tc.mov"
    _timecode_video(src)
    info = probe(src)
    assert [d.codec_tag for d in info.data_streams] == ["tmcd"]
    assert info.video is not None


def test_probe_falls_back_to_handler_name_without_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ffprobe가 없거나 실패하면(빈 결과) 트랙 handler_name(TimeCodeHandler)으로 태그를 추정한다."""
    src = tmp_path / "tc.mov"
    _timecode_video(src)
    monkeypatch.setattr(probe_mod, "data_stream_tags", lambda _path: {})  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    assert [d.codec_tag for d in probe(src).data_streams] == ["tmcd"]


def test_gopro_handler_names_map_to_tags() -> None:
    """GoPro가 쓰는 handler_name(앞에 탭이 붙기도 한다)을 GPMF·타임코드 태그로 바꾼다."""
    assert probe_mod._handler_tag("\tGoPro MET") == "gpmd"  # pyright: ignore[reportPrivateUsage]
    assert probe_mod._handler_tag("GoPro TCD") == "tmcd"  # pyright: ignore[reportPrivateUsage]
    assert probe_mod._handler_tag("Unknown") == ""  # pyright: ignore[reportPrivateUsage]
