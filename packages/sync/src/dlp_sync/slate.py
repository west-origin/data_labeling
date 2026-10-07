"""QR 슬레이트 검출.

영상 앞뒤 search_window_ms 구간만 디코딩한다. frame_stride 간격으로 QR을 찾고, 새 QR을 찾으면
그 사이에 건너뛴 프레임을 되짚어 그 QR이 처음 보인 프레임을 정한다.
결과는 (스트림 시각, 내용, 양자화 간격)이다. 양자화 간격(gap_ms)은 처음 보인 프레임과 그 앞
프레임 사이 간격으로, 슬레이트가 실제로 뜬 시각은 [stream_ms - gap_ms, stream_ms] 안에 있다.
스트림의 첫 프레임부터 보인 슬레이트는 녹화 전에 떴을 수 있어(시작 시각을 모른다) 버린다.
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
    gap_ms: float = 0.0  # 처음 보인 프레임과 그 앞 프레임 사이 간격 (양자화 오차 상한)


@dataclass(frozen=True)
class SlateScan:
    sightings: list[SlateSighting]
    duration_ms: float | None  # 컨테이너 길이 (모르면 None)


def _decode_qr(detector: cv2.QRCodeDetector, gray: NDArray[np.uint8], prefix: str) -> str | None:
    text, _, _ = detector.detectAndDecode(gray)
    return text if text and text.startswith(prefix) else None


def detect_slates(path: Path, policy: SlatePolicy) -> list[SlateSighting]:
    return scan_slates(path, policy).sightings


def scan_slates(path: Path, policy: SlatePolicy) -> SlateScan:
    detector = cv2.QRCodeDetector()
    found: dict[str, SlateSighting] = {}
    unknown_onset: set[str] = set()
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
            # (시각, 영상, 바로 앞 프레임 시각). 앞 프레임이 없으면(스트림 첫 프레임) None
            recent: deque[tuple[float, NDArray[np.uint8], float | None]] = deque(
                maxlen=policy.frame_stride
            )
            prev_ms: float | None = None
            for i, frame in enumerate(c.decode(stream)):
                if frame.pts is None:
                    continue
                t_ms = float(frame.pts * tb * 1000)
                before, prev_ms = prev_ms, t_ms
                if t_ms < start_ms:
                    continue
                if t_ms > end_ms:
                    break
                gray = np.asarray(frame.to_ndarray(format="gray"), dtype=np.uint8)
                recent.append((t_ms, gray, before))
                if i % policy.frame_stride:
                    continue
                payload = _decode_qr(detector, gray, policy.payload_prefix)
                if payload is None or payload in found or payload in unknown_onset:
                    continue
                first, first_before = t_ms, before
                for prev_t, prev, prev_before in list(recent)[:-1][::-1]:
                    if _decode_qr(detector, prev, policy.payload_prefix) != payload:
                        break
                    first, first_before = prev_t, prev_before
                if first_before is None:
                    unknown_onset.add(payload)  # 첫 프레임부터 보였다: 슬레이트가 뜬 시각을 모른다
                    continue
                found[payload] = SlateSighting(first, payload, first - first_before)
    sightings = sorted(found.values(), key=lambda s: s.stream_ms)
    return SlateScan(sightings, duration_ms)
