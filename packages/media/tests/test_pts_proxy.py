# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""PTS 인덱스·컨테이너 조회·프록시 테스트 (dlp_media.pts·probe·proxy, WP3).

정답 근거: 블러 픽스처는 프레임마다 정해 둔 정수 ms PTS(VFR)로 영상을 쓴다. 동기화 픽스처의
바디캠은 h264 영상 + 16 kHz 오디오이며 creation_time 태그가 없다.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from dlp_fixtures.sync import SyncScenario
from dlp_fixtures.video import BlurScenario
from dlp_media.probe import probe
from dlp_media.proxy import make_proxy
from dlp_media.pts import PtsIndex, build_pts_index
from dlp_schema.config import ProxyConfig


def test_pts_index_matches_vfr_frame_times_exactly(blur: tuple[BlurScenario, Path]) -> None:
    """VFR 영상의 PTS 인덱스가 픽스처 프레임 시각과 정확히(분수·실수 모두) 같다."""
    scenario, path = blur
    index = build_pts_index(path)
    assert index.is_vfr
    assert len(index) == len(scenario.frame_times)
    assert [index.frame_ms(i) for i in range(len(index))] == scenario.frame_times
    assert index.ms.tolist() == [float(t) for t in scenario.frame_times]


def test_frame_lookup_at_exact_boundaries(blur: tuple[BlurScenario, Path]) -> None:
    """frame_at·nearest의 경계 동작.

    프레임 시각 정각이면 그 프레임, 1 µs 앞이면 앞 프레임, 첫 프레임 앞은 0, 끝 뒤는 마지막.
    두 프레임 중간 ±0.1 ms는 가까운 쪽.
    """
    scenario, path = blur
    index = build_pts_index(path)
    times = scenario.frame_times
    for i in range(1, len(times)):
        assert index.frame_at(times[i]) == i
        assert index.frame_at(times[i] - 0.001) == i - 1
    assert index.frame_at(-5) == 0
    assert index.frame_at(10**9) == len(times) - 1
    mid = (times[3] + times[4]) / 2
    assert index.nearest(mid - 0.1) == 3 and index.nearest(mid + 0.1) == 4


def test_pts_index_roundtrip_and_validation(
    blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """Parquet 쓰기·읽기 왕복이 같고, 중복 PTS는 ValueError."""
    index = build_pts_index(blur[1])
    index.write(tmp_path / "pts.parquet")
    again = PtsIndex.read(tmp_path / "pts.parquet")
    assert again.time_base == index.time_base
    assert np.array_equal(again.pts, index.pts) and np.array_equal(again.keyframe, index.keyframe)
    with pytest.raises(ValueError, match="중복"):
        PtsIndex(Fraction(1, 1000), np.array([0, 33, 33]), np.zeros(3, dtype=bool))


def test_probe_reports_streams(sync: tuple[SyncScenario, Path]) -> None:
    """probe가 h264 비디오, 16 kHz 오디오, 데이터 트랙 없음, creation_time 없음을 알려 준다."""
    info = probe(sync[1] / "bodycam.mp4")
    assert info.video is not None and info.video.codec == "h264"
    assert info.audio is not None and info.audio.sample_rate == 16_000
    assert info.data_streams == ()
    assert info.creation_time is None


def test_proxy_keeps_pts_drops_audio_and_adds_keyframes(
    sync: tuple[SyncScenario, Path], blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """프록시는 원본 PTS를 그대로 두고, 오디오를 빼고, 키프레임을 일정 간격으로 넣는다.

    높이 120으로 줄고(320x240 → 160x120), 첫 프레임이 키프레임이며 키프레임 간격은
    500 ms + 한 프레임 여유(100 ms) 이하다.
    """
    for src in (blur[1], sync[1] / "bodycam.mp4"):
        dst = tmp_path / f"{src.parent.name}-proxy.mp4"
        make_proxy(src, dst, ProxyConfig(max_height=120, crf=28, keyframe_ms=500))
        source, proxy = build_pts_index(src), build_pts_index(dst)
        assert proxy.ms.tolist() == source.ms.tolist()
        info = probe(dst)
        assert info.video is not None and info.video.height == 120
        assert info.audio is None
        key_ms = proxy.ms[proxy.keyframe]
        assert key_ms[0] == proxy.ms[0]
        assert np.diff(key_ms).max() <= 500 + 100

    with av.open(str(tmp_path / f"{blur[1].parent.name}-proxy.mp4")) as c:
        assert c.streams.video[0].codec_context.width == 160


def _odd_source(path: Path, width: int, height: int) -> None:
    """홀수 크기 원본 (무손실 FFV1, yuv444p는 홀수 크기를 받는다)."""
    with av.open(str(path), "w") as c:
        vs = c.add_stream("ffv1", rate=10)
        assert isinstance(vs, av.VideoStream)
        vs.width, vs.height, vs.pix_fmt = width, height, "yuv444p"
        vs.time_base = Fraction(1, 1000)
        for i in range(5):
            img = np.full((height, width, 3), 40 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = i * 100, Fraction(1, 1000)
            c.mux(vs.encode(frame))
        c.mux(vs.encode(None))


@pytest.mark.parametrize(("size", "expected"), [((101, 75), (100, 74)), ((333, 241), (166, 120))])
def test_proxy_handles_odd_source_size(
    size: tuple[int, int], expected: tuple[int, int], tmp_path: Path
) -> None:
    """홀수 높이·너비 원본도 짝수 크기로 인코딩한다 (libx264 yuv420p)."""
    src, dst = tmp_path / "odd.mkv", tmp_path / "odd-proxy.mp4"
    _odd_source(src, *size)
    make_proxy(src, dst, ProxyConfig(max_height=120, crf=28, keyframe_ms=500))
    with av.open(str(dst)) as c:
        ctx = c.streams.video[0].codec_context
        assert (ctx.width, ctx.height) == expected
    assert build_pts_index(dst).ms.tolist() == [0, 100, 200, 300, 400]


def test_proxy_max_height_must_be_even() -> None:
    """프록시 높이 상한은 짝수여야 한다 (설정 검증, ProxyConfig)."""
    with pytest.raises(ValueError, match="multiple"):
        ProxyConfig(max_height=121, crf=28, keyframe_ms=500)
