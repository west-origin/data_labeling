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


def test_pts_index_matches_vfr_frame_times_exactly(blur: tuple[BlurScenario, Path]) -> None:
    scenario, path = blur
    index = build_pts_index(path)
    assert index.is_vfr
    assert len(index) == len(scenario.frame_times)
    assert [index.frame_ms(i) for i in range(len(index))] == scenario.frame_times
    assert index.ms.tolist() == [float(t) for t in scenario.frame_times]


def test_frame_lookup_at_exact_boundaries(blur: tuple[BlurScenario, Path]) -> None:
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
    index = build_pts_index(blur[1])
    index.write(tmp_path / "pts.parquet")
    again = PtsIndex.read(tmp_path / "pts.parquet")
    assert again.time_base == index.time_base
    assert np.array_equal(again.pts, index.pts) and np.array_equal(again.keyframe, index.keyframe)
    with pytest.raises(ValueError, match="중복"):
        PtsIndex(Fraction(1, 1000), np.array([0, 33, 33]), np.zeros(3, dtype=bool))


def test_probe_reports_streams(sync: tuple[SyncScenario, Path]) -> None:
    info = probe(sync[1] / "bodycam.mp4")
    assert info.video is not None and info.video.codec == "h264"
    assert info.audio is not None and info.audio.sample_rate == 16_000
    assert info.data_streams == ()
    assert info.creation_time is None


def test_proxy_keeps_pts_drops_audio_and_adds_keyframes(
    sync: tuple[SyncScenario, Path], blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    for src in (blur[1], sync[1] / "bodycam.mp4"):
        dst = tmp_path / f"{src.parent.name}-proxy.mp4"
        make_proxy(src, dst, max_height=120, keyframe_ms=500)
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
