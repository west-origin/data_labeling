"""PTS 기준 프레임 고르기·순차 디코드 테스트 (`dlp_media.frames`, WP17.2).

정답을 아는 합성 VFR 영상(`generate_blur_scenario`, 프레임 시각을 안다)과, 키프레임 간격을
고정한 작은 VFR 영상(키프레임 10장마다)을 쓴다. 고른 프레임 시각이 `PtsIndex.frame_at`과
같고(프레임 번호에 평균 간격을 곱한 값과 다른 경우 포함), seek을 섞은 디코드가 전체 순차
디코드와 픽셀까지 같은지 본다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from dlp_fixtures.video import BlurScenario
from dlp_media.frames import decode_frames, fit_size, pick_frames, plan_seeks
from dlp_media.pts import PtsIndex, build_pts_index


@pytest.fixture(scope="module")
def gop_video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int]]:
    """키프레임이 10장마다 있는 70장 VFR 영상 (64와 48)과 프레임 시각 ms."""
    path = tmp_path_factory.mktemp("gop") / "v.mp4"
    rng = np.random.default_rng(0)
    times: list[int] = []
    with av.open(str(path), "w") as c:
        vs = c.add_stream("libx264", rate=30)
        assert isinstance(vs, av.VideoStream)
        vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
        vs.time_base = vs.codec_context.time_base = Fraction(1, 1000)
        vs.options = {
            "crf": "18", "preset": "veryfast", "x264-params": "keyint=10:min-keyint=10:scenecut=0"
        }  # fmt: skip
        t = 0
        for i in range(70):
            img = np.full((48, 64, 3), (i * 3) % 255, dtype=np.uint8)
            img[:, : i % 64] = 200
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = t, Fraction(1, 1000)
            c.mux(vs.encode(frame))
            times.append(t)
            t += int(rng.choice([25, 33, 40]))
        c.mux(vs.encode(None))
    return path, times


def _all_frames(path: Path) -> list[np.ndarray]:
    """전체를 처음부터 순차 디코드한 RGB 프레임 (기준)."""
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def test_pick_frames_follow_pts_not_frame_count(blur: tuple[BlurScenario, Path]) -> None:
    """VFR에서 고른 프레임은 그 시각에 화면에 떠 있는 프레임(`frame_at`)이다. 같은 프레임을 가리키는
    시각은 하나로 합치고 정렬한다. 프레임 번호를 평균 간격으로 계산한 값과 다른 경우가 있다."""
    scenario, path = blur
    pts = build_pts_index(path)
    times = scenario.frame_times
    wanted = [times[5] + 1, times[5] + 2, 1_000.0, 2_222.5, times[-1] + 500, -10.0]
    picked = pick_frames(pts, wanted)
    expected = sorted({max(i for i, t in enumerate(times) if t <= max(w, 0)) for w in wanted})
    assert picked == expected == sorted({pts.frame_at(w) for w in wanted})
    # 시각을 평균 간격으로 나눠 프레임 번호를 구하면 VFR에서 다른 프레임을 가리키는 시각이 있다
    mean_gap = (times[-1] - times[0]) / (len(times) - 1)
    grid = range(0, times[-1], 50)
    assert any(int(ms // mean_gap) != pts.frame_at(ms) for ms in grid)


def test_decode_frames_match_scenario_and_are_deterministic(
    blur: tuple[BlurScenario, Path],
) -> None:
    """디코드한 프레임의 ms는 정답 프레임 시각이고, 픽셀은 그 정답 프레임에 가장 가깝다.
    같은 입력이면 같은 바이트가 나온다."""
    scenario, path = blur
    pts = build_pts_index(path)
    wanted = [3, 17, 40, len(scenario.frame_times) - 1]
    out = list(decode_frames(path, pts, [40, 3, 17, 3, wanted[-1]], max_side=1000))
    assert [i for i, _, _ in out] == wanted
    assert [ms for _, ms, _ in out] == [scenario.frame_times[i] for i in wanted]
    for i, _, img in out:
        assert img.shape == (scenario.height, scenario.width, 3) and img.dtype == np.uint8
        diffs = [np.abs(img.astype(int) - f.astype(int)).mean() for f in scenario.frames]
        assert int(np.argmin(diffs)) == i
    again = list(decode_frames(path, pts, wanted, max_side=1000))
    assert all(a[2].tobytes() == b[2].tobytes() for a, b in zip(out, again, strict=True))


def test_seek_only_when_next_keyframe_is_ahead(gop_video: tuple[Path, list[int]]) -> None:
    """목표 직전 키프레임이 디코드 위치보다 뒤일 때만 seek한다.

    seek을 섞은 결과는 전체 순차 디코드와 픽셀까지 같다.
    """
    path, times = gop_video
    pts = build_pts_index(path)
    assert list(np.flatnonzero(pts.keyframe)) == [0, 10, 20, 30, 40, 50, 60]
    targets = [3, 5, 12, 25, 27, 61, 69]
    assert plan_seeks(pts, targets) == [None, None, 10, 20, None, 60, None]
    # 디코드 위치가 이미 키프레임(10)에 닿아 있으면 seek하지 않고 이어서 디코드한다
    assert plan_seeks(pts, [9, 12]) == [None, None]
    assert plan_seeks(pts, [11]) == [10]
    assert plan_seeks(pts, []) == []
    with pytest.raises(ValueError):
        plan_seeks(pts, [5, 3])
    with pytest.raises(ValueError):
        plan_seeks(pts, [70])
    full = _all_frames(path)
    got = list(decode_frames(path, pts, targets, max_side=64))
    assert [ms for _, ms, _ in got] == [times[i] for i in targets]
    for i, _, img in got:
        assert img.tobytes() == full[i].tobytes()


def test_fit_size_and_downscale(gop_video: tuple[Path, list[int]]) -> None:
    """긴 변 상한으로 줄이고 정렬 단위의 배수로 내린다 (키우지 않는다)."""
    assert fit_size(1920, 1080, 448) == (448, 252)
    assert fit_size(1920, 1080, 448, 28) == (448, 252)
    assert fit_size(1920, 1080, 448, 32) == (448, 224)
    assert fit_size(320, 240, 1000, 32) == (320, 224)
    assert fit_size(20, 10, 1000, 32) == (32, 32)
    path, _ = gop_video
    pts = build_pts_index(path)
    [(_, _, img)] = list(decode_frames(path, pts, [7], max_side=32, align=8))
    assert img.shape == (24, 32, 3)


def test_index_mismatch_is_an_error(gop_video: tuple[Path, list[int]]) -> None:
    """인덱스에 있는 PTS가 영상에 없으면(다른 영상의 인덱스) 조용히 다른 프레임을 내지 않는다."""
    path, _ = gop_video
    real = build_pts_index(path)
    shifted = PtsIndex(real.time_base, real.pts + 1, real.keyframe)
    with pytest.raises(ValueError, match="PTS"):
        list(decode_frames(path, shifted, [4], max_side=64))
