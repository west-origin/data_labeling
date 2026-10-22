"""정규화 사본은 VFR PTS·표시 방향을 보존하고 실패 시 공개하지 않아야 한다."""

# pyright: reportUnknownMemberType=false
from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pytest

from dlp_media.prepare import NormalizePolicy, prepare_video
from dlp_media.pts import build_pts_index
from dlp_prelabel.common import iter_frames

SETTINGS = NormalizePolicy(crf=0, preset="fast", threads=1)


def source_video(path: Path, rotation: int) -> None:
    with av.open(str(path), "w") as container:
        container.metadata["creation_time"] = "2026-10-09T00:00:00Z"
        stream = container.add_stream("libx264", rate=30)
        assert isinstance(stream, av.VideoStream)
        stream.width, stream.height, stream.pix_fmt = 106, 64, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = stream.time_base
        stream.set_display_rotation(rotation)
        image = np.zeros((64, 106, 3), np.uint8)
        image[:32, :53] = (240, 20, 20)
        for time in (0, 33, 100):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = time, stream.time_base
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def test_decoder_returns_unpadded_rgb_rows(tmp_path: Path) -> None:
    """폭 106은 RGB 행 정렬에 맞지 않는다. 패딩은 MediaPipe에서 검출 누락을 일으켰다."""
    src = tmp_path / "source.mp4"
    source_video(src, 0)
    frames = list(iter_frames(src))
    assert [t for t, _ in frames] == [0, 33, 100]
    for _, image in frames:
        assert image.flags.c_contiguous and image.strides == (106 * 3, 3, 1)
        assert image[8, 8, 0] > 200 and image[8, 8, 1] < 40
        assert image[-8, -8].max() < 10


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg 필요")
@pytest.mark.parametrize("rotation", [0, -90, 90, 180])
def test_prepare_preserves_pts_and_rotates_pixels(tmp_path: Path, rotation: int) -> None:
    src, dst = tmp_path / "source.mp4", tmp_path / "ready.mp4"
    source_video(src, rotation)
    original = src.read_bytes()
    result = prepare_video(src, dst, SETTINGS)
    assert src.read_bytes() == original
    assert result.video is not None and result.video.rotation_degrees == 0
    assert result.creation_time is not None
    before, after = build_pts_index(src), build_pts_index(dst)
    assert len(before) == len(after) == 3
    assert [before.frame_ms(i) for i in range(3)] == [after.frame_ms(i) for i in range(3)]
    _, pixels = next(iter_frames(dst))
    # 표시 행렬 각도는 반시계 방향이다. 색 영역의 위치까지 검사해 단순 태그 삭제를 막는다.
    expected = np.rot90(next(iter_frames(src))[1], k=rotation // 90)
    assert pixels.shape == expected.shape
    assert np.abs(pixels.astype(float) - expected.astype(float)).mean() < 3
    with pytest.raises(FileExistsError):
        prepare_video(src, dst, SETTINGS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg 필요")
@pytest.mark.parametrize(("tag", "override"), [(-90, 180), (90, 0), (0, -90)])
def test_prepare_display_rotation_overrides_wrong_tag(
    tmp_path: Path, tag: int, override: int
) -> None:
    """표시 태그가 장착 방향과 다르면 운영자가 준 각도를 태그 대신 쓴다 (실영상 파일럿).

    태그(-90)를 따르면 옆으로 눕는 영상을 180도로 바로 세우는 경우 등. 결과 픽셀은 저장 픽셀을
    지정 각도만큼 반시계로 돌린 것과 같고, 태그 각도와는 다르다.
    """
    src, dst = tmp_path / "source.mp4", tmp_path / "ready.mp4"
    source_video(src, tag)
    result = prepare_video(src, dst, SETTINGS, display_rotation=override)
    assert result.video is not None and result.video.rotation_degrees == 0
    before, after = build_pts_index(src), build_pts_index(dst)
    assert [before.frame_ms(i) for i in range(3)] == [after.frame_ms(i) for i in range(3)]
    stored = next(iter_frames(src))[1]  # iter_frames는 태그를 적용하지 않은 저장 픽셀이다
    _, pixels = next(iter_frames(dst))
    expected = np.rot90(stored, k=override // 90)
    assert pixels.shape == expected.shape
    assert np.abs(pixels.astype(float) - expected.astype(float)).mean() < 3
    by_tag = np.rot90(stored, k=tag // 90)
    assert by_tag.shape != pixels.shape or np.abs(by_tag.astype(float) - pixels).mean() > 20


def test_prepare_rejects_non_right_angle_override(tmp_path: Path) -> None:
    src, dst = tmp_path / "source.mp4", tmp_path / "ready.mp4"
    source_video(src, 0)
    with pytest.raises(ValueError, match="90의 배수"):
        prepare_video(src, dst, SETTINGS, display_rotation=45)
    assert not dst.exists()


def test_failed_prepare_does_not_publish_output(tmp_path: Path) -> None:
    src, dst = tmp_path / "source.mp4", tmp_path / "ready.mp4"
    source_video(src, 0)
    with (
        patch("dlp_media.prepare.subprocess.run", side_effect=RuntimeError("encoder failed")),
        pytest.raises(RuntimeError, match="encoder failed"),
    ):
        prepare_video(src, dst, SETTINGS)
    assert not dst.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.mp4"]
