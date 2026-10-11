"""스마트폰의 표시 회전 태그를 무시하면 원본·검수 화면의 라벨 좌표가 어긋난다."""

# pyright: reportUnknownMemberType=false
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from dlp_media.ingest import SessionManifest, ingest_session
from dlp_media.probe import probe
from dlp_media.proxy import make_proxy
from dlp_media.storage import LocalStore
from dlp_schema.config import ProxyConfig
from dlp_schema.testing import FIXED_TIME


def video(path: Path, rotation: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        assert isinstance(stream, av.VideoStream)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = stream.time_base
        stream.set_display_rotation(rotation)
        for time in (0, 33, 100):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 96, 3), np.uint8), format="rgb24")
            frame.pts, frame.time_base = time, Fraction(1, 1000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


@pytest.mark.parametrize("rotation", [-90, 90, 180])
def test_rotation_detected_and_proxy_refused_before_output(tmp_path: Path, rotation: int) -> None:
    src, dst = tmp_path / "phone.mp4", tmp_path / "proxy.mp4"
    video(src, rotation)
    info = probe(src)
    assert info.video is not None
    assert info.video.rotation_degrees % 360 == rotation % 360
    assert (info.video.width, info.video.height) == (96, 64)
    with pytest.raises(ValueError, match="표시 회전"):
        make_proxy(src, dst, ProxyConfig(max_height=64, crf=28, keyframe_ms=500))
    assert not dst.exists()


def test_all_video_inputs_checked_before_any_upload(tmp_path: Path) -> None:
    body, third = tmp_path / "body.mp4", tmp_path / "third.mp4"
    video(body, 0)
    video(third, -90)
    store = LocalStore(tmp_path / "store", "dlp-raw")
    manifest = SessionManifest.model_validate(
        {
            "session_id": "s001",
            "domain": "cleaning",
            "worker_id": "w01",
            "site_id": "site01",
            "consent_version": "c1",
            "recorded_at": FIXED_TIME,
            "streams": [
                {"stream_id": "body", "kind": "bodycam", "path": "body.mp4"},
                {"stream_id": "third", "kind": "third_person", "path": "third.mp4"},
            ],
        }
    )
    with pytest.raises(ValueError, match="표시 회전"):
        ingest_session(manifest, tmp_path, store)
    assert not list(store.root.rglob("*.mp4"))
