"""합성 영상: 가변 프레임레이트(VFR) MP4 쓰기, QR 슬레이트, 정답을 아는 블러 대상 영상.

PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
"""

# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import segno
from numpy.typing import NDArray

from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord, Provenance, Source
from dlp_schema.testing import FIXED_TIME

Image = NDArray[np.uint8]
MS = Fraction(1, 1000)


# ---------------------------------------------------------------- 쓰기


def write_video(
    path: Path,
    frames: Iterable[tuple[int, Image]],
    *,
    width: int,
    height: int,
    audio: NDArray[np.float32] | None = None,
    audio_rate: int = 16_000,
    codec: str = "libx264",
) -> None:
    """프레임마다 PTS(ms)를 그대로 기록한다. 오디오가 있으면 AAC로 함께 넣는다."""
    with av.open(str(path), "w") as container:
        # 모든 스트림을 첫 mux 전에 추가해야 한다 (헤더가 그때 쓰인다).
        vs = container.add_stream(codec, rate=30)
        assert isinstance(vs, av.VideoStream)
        vs.width, vs.height, vs.pix_fmt = width, height, "yuv420p"
        vs.time_base = MS
        vs.codec_context.time_base = MS
        vs.options = {"crf": "18", "preset": "veryfast"} if codec == "libx264" else {}
        aus = None
        if audio is not None:
            aus = container.add_stream("aac", rate=audio_rate, layout="mono")
            assert isinstance(aus, av.AudioStream)
            aus.time_base = Fraction(1, audio_rate)

        for pts, img in frames:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts, frame.time_base = pts, MS
            container.mux(vs.encode(frame))
        container.mux(vs.encode(None))

        if aus is not None and audio is not None:
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            chunk = 1024
            for start in range(0, pcm.size, chunk):
                block = np.ascontiguousarray(pcm[np.newaxis, start : start + chunk])
                af = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
                af.sample_rate = audio_rate
                af.pts, af.time_base = start, Fraction(1, audio_rate)
                container.mux(aus.encode(af))
            container.mux(aus.encode(None))


def read_frame_times(path: Path) -> list[int]:
    """디코딩한 비디오 프레임의 PTS(ms) 목록."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [round(f.time * 1000) for f in container.decode(stream)]


def vfr_times(rng: np.random.Generator, duration_ms: int) -> list[int]:
    """25~50 ms 사이로 흔들리는 프레임 간격 (평균 약 30 fps)."""
    times = [0]
    choices = np.array([25, 33, 33, 33, 34, 40, 50])
    while True:
        nxt = times[-1] + int(rng.choice(choices))
        if nxt >= duration_ms:
            return times
        times.append(nxt)


# ---------------------------------------------------------------- QR 슬레이트


def render_qr(payload: str, size: int) -> Image:
    """흰 바탕에 검은 QR. 한 변이 size 픽셀인 정사각형."""
    matrix = np.array(segno.make(payload, error="m").matrix, dtype=bool)
    quiet = 4
    padded = np.pad(matrix, quiet, constant_values=False)
    scale = max(1, size // padded.shape[0])
    big = np.kron(padded, np.ones((scale, scale), dtype=bool))
    canvas = np.full((size, size), 255, dtype=np.uint8)
    off = (size - big.shape[0]) // 2
    canvas[off : off + big.shape[0], off : off + big.shape[1]][big] = 0
    return np.repeat(canvas[:, :, np.newaxis], 3, axis=2)


def slate_frame(payload: str, width: int, height: int) -> Image:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    size = min(width, height) - 8
    qr = render_qr(payload, size)
    y, x = (height - size) // 2, (width - size) // 2
    frame[y : y + size, x : x + size] = qr
    return frame


# ---------------------------------------------------------------- 블러 대상 영상


@dataclass(frozen=True)
class MovingTarget:
    """한 프라이버시 대상. 위치는 start + velocity * t (픽셀, 픽셀/초)."""

    target: str
    w: int
    h: int
    start: tuple[float, float]
    velocity: tuple[float, float]
    inside: tuple[int, int, int, int] | None = None  # 반사면 등 이 영역 안에서만 보임 (x, y, w, h)

    def box_at(self, t_ms: int) -> tuple[float, float]:
        s = t_ms / 1000
        return self.start[0] + self.velocity[0] * s, self.start[1] + self.velocity[1] * s


@dataclass
class BlurScenario:
    width: int
    height: int
    frame_times: list[int]
    targets: list[MovingTarget]
    labels: list[LabelRecord]
    frames: list[Image] = field(repr=False)
    mirror: tuple[int, int, int, int]

    def write(self, path: Path) -> None:
        write_video(
            path,
            zip(self.frame_times, self.frames, strict=True),
            width=self.width,
            height=self.height,
        )


# 대상별 대표 색 (테스트에서 블러 영역 확인용)
TARGET_COLORS: dict[str, tuple[int, int, int]] = {
    "face": (224, 172, 140),
    "reflection": (200, 150, 120),
    "document": (250, 250, 250),
    "screen": (40, 90, 230),
    "photo": (150, 60, 160),
    "shipping_label": (240, 230, 120),
}


def generate_blur_scenario(
    seed: int = 0,
    *,
    session_id: str = "syn-blur",
    duration_ms: int = 3_000,
    width: int = 320,
    height: int = 240,
    ontology_version: str = "1.0.0",
) -> BlurScenario:
    rng = np.random.default_rng(seed)
    times = vfr_times(rng, duration_ms)
    mirror = (200, 20, 100, 110)
    mx, my, mw, mh = mirror
    # 그리는 순서 = 목록 순서. 거울 속 반사가 가장 뒤(먼저)에 그려진다.
    targets = [
        MovingTarget("reflection", 26, 32, (mx + 5.0, my + 10.0), (20.0, 15.0), inside=mirror),
        MovingTarget("face", 36, 44, (-30.0, 120.0), (90.0, -10.0)),  # 왼쪽에서 들어온다
        MovingTarget("document", 60, 40, (40.0, 180.0), (30.0, 0.0)),
        MovingTarget("screen", 50, 34, (120.0, 30.0), (0.0, 0.0)),
        MovingTarget("photo", 30, 38, (20.0, 20.0), (0.0, 15.0)),
        MovingTarget("shipping_label", 40, 26, (260.0, 190.0), (60.0, 0.0)),  # 오른쪽으로 나간다
    ]

    yy, xx = np.mgrid[0:height, 0:width]
    background = np.stack(
        [60 + xx * 60 // width, 80 + yy * 60 // height, np.full_like(xx, 70)], axis=2
    ).astype(np.int16)
    background += rng.integers(-6, 7, size=background.shape, dtype=np.int16)

    frames: list[Image] = []
    keyframes: dict[str, list[BoxKeyframe]] = {t.target: [] for t in targets}
    for t_ms in times:
        img = background.copy()
        img[my : my + mh, mx : mx + mw] = (170, 175, 180)  # 거울
        for target in targets:
            box = _clip_box(target, t_ms, width, height)
            if box is None:
                keyframes[target.target].append(
                    BoxKeyframe(t_ms=t_ms, x=0, y=0, w=0, h=0, outside=True)
                )
                continue
            x, y, w, h = box
            img[y : y + h, x : x + w] = TARGET_COLORS[target.target]
            if target.target in ("face", "reflection"):
                _draw_eyes(img, x, y, w, h)
            if target.target == "document":
                img[y + 4 : y + h - 2 : 6, x + 4 : x + w - 4] = (30, 30, 30)
            keyframes[target.target].append(BoxKeyframe(t_ms=t_ms, x=x, y=y, w=w, h=h))
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    labels = [
        LabelRecord(
            label_id=f"{session_id}-blur-{name}",
            session_id=session_id,
            stream_id="bodycam",
            t_start_ms=times[0],
            t_end_ms=times[-1],
            ontology_version=ontology_version,
            provenance=Provenance(source=Source.HUMAN),
            created_at=FIXED_TIME,
            payload=BlurTrackPayload(target=name, keyframes=tuple(kfs)),
        )
        for name, kfs in keyframes.items()
    ]
    return BlurScenario(width, height, times, targets, labels, frames, mirror)


def _clip_box(
    target: MovingTarget, t_ms: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    x0, y0 = target.box_at(t_ms)
    bx, by, bw, bh = (0, 0, width, height) if target.inside is None else target.inside
    if target.inside is not None:
        # 반사면 안에서 벽에 부딪히듯 되돌아오게 해 계속 보이도록 한다
        x0 = bx + _bounce(x0 - bx, bw - target.w)
        y0 = by + _bounce(y0 - by, bh - target.h)
    x1, y1 = max(round(x0), bx), max(round(y0), by)
    x2, y2 = min(round(x0) + target.w, bx + bw), min(round(y0) + target.h, by + bh)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def _bounce(pos: float, span: int) -> float:
    period = 2 * span
    p = pos % period
    return p if p <= span else period - p


def _draw_eyes(img: NDArray[np.int16], x: int, y: int, w: int, h: int) -> None:
    ey = y + h // 3
    for ex in (x + w // 4, x + 3 * w // 4 - 3):
        img[ey : ey + 3, ex : ex + 3] = (20, 20, 20)


def target_boxes_at(labels: Sequence[LabelRecord], t_ms: int) -> dict[str, BoxKeyframe]:
    """시각 t_ms의 대상별 정답 박스 (보이는 것만)."""
    out: dict[str, BoxKeyframe] = {}
    for label in labels:
        p = label.payload
        if isinstance(p, BlurTrackPayload):
            for k in p.keyframes:
                if k.t_ms == t_ms and not k.outside:
                    out[p.target] = k
    return out
