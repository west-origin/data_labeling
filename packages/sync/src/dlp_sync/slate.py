"""QR 슬레이트 검출.

영상 앞뒤 search_window_ms 구간만 디코딩한다. frame_stride 간격으로 QR을 찾고, 새 QR을 찾으면
그 사이에 건너뛴 프레임을 되짚어 그 QR이 처음 보인 프레임을 정한다. 결과는 (스트림 시각, 내용)이다.
"""

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.probe import to_fraction
from dlp_sync.policy import SlatePolicy


@dataclass(frozen=True)
class SlateSighting:
    stream_ms: float
    payload: str


def _decode_qr(detector: cv2.QRCodeDetector, gray: NDArray[np.uint8], prefix: str) -> str | None:
    text, _, _ = detector.detectAndDecode(gray)
    return text if text and text.startswith(prefix) else None


def detect_slates(path: Path, policy: SlatePolicy) -> list[SlateSighting]:
    detector = cv2.QRCodeDetector()
    found: dict[str, float] = {}
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        duration_ms = float(c.duration / 1000) if c.duration is not None else None
        windows = [(0.0, policy.search_window_ms)]
        if duration_ms is not None and duration_ms > 2 * policy.search_window_ms:
            windows.append((duration_ms - policy.search_window_ms, duration_ms))
        else:
            windows = [(0.0, float("inf"))]

        for start_ms, end_ms in windows:
            if start_ms > 0:
                c.seek(int(start_ms / 1000 / tb), stream=stream, backward=True)
            recent: deque[tuple[float, NDArray[np.uint8]]] = deque(maxlen=policy.frame_stride)
            for i, frame in enumerate(c.decode(stream)):
                if frame.pts is None:
                    continue
                t_ms = float(frame.pts * tb * 1000)
                if t_ms < start_ms:
                    continue
                if t_ms > end_ms:
                    break
                gray = np.asarray(frame.to_ndarray(format="gray"), dtype=np.uint8)
                recent.append((t_ms, gray))
                if i % policy.frame_stride:
                    continue
                payload = _decode_qr(detector, gray, policy.payload_prefix)
                if payload is None or payload in found:
                    continue
                first = t_ms
                for prev_ms, prev in list(recent)[:-1][::-1]:
                    if _decode_qr(detector, prev, policy.payload_prefix) != payload:
                        break
                    first = prev_ms
                found[payload] = first
    return sorted((SlateSighting(t, p) for p, t in found.items()), key=lambda s: s.stream_ms)
