"""QR 슬레이트 검출 (`qr_slate` 방법, WP4, ADR 0004·0028).

촬영 프로토콜: 녹화 시작·끝에 슬레이트 앱이 `DLP-SLATE|<세션>|<Unix ms>` QR을 띄우고, 바디캠과
3인칭 카메라가 함께 찍는다. 같은 payload의 첫 등장 시각 쌍이 앵커가 된다 (`pipeline._slate`).

영상 앞뒤 search_window_ms 구간만 디코딩한다. frame_stride 간격으로 QR을 찾고, 새 QR을 찾으면
그 사이에 건너뛴 프레임을 되짚어 그 QR이 처음 보인 프레임을 정한다.
결과는 (스트림 시각, 내용, 양자화 간격)이다. 양자화 간격(gap_ms)은 처음 보인 프레임과 그 앞
프레임 사이 간격으로, 슬레이트가 실제로 뜬 시각은 [stream_ms - gap_ms, stream_ms] 안에 있다.
스트림의 첫 프레임부터 보인 슬레이트는 녹화 전에 떴을 수 있어(시작 시각을 모른다) 버린다.

시각은 프레임 PTS * time_base로만 계산한다 (프레임 번호 * 간격을 쓰지 않는다, VFR 대응).
정책: `sync.yaml slate` 절 (`payload_prefix`, `search_window_ms`, `frame_stride`).
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
    """슬레이트 하나의 첫 등장.

    Attributes:
        stream_ms: 처음 보인 프레임의 PTS 시각 ms (그 스트림 시계).
        payload: QR 내용 (`payload_prefix`로 시작). 두 영상의 같은 슬레이트를 이걸로 짝짓는다.
        gap_ms: 처음 보인 프레임과 그 앞 프레임 사이 간격 ms. 실제로 뜬 시각의 불확실성(양자화)
            상한.
    """

    stream_ms: float
    payload: str
    gap_ms: float = 0.0  # 처음 보인 프레임과 그 앞 프레임 사이 간격 (양자화 오차 상한)


@dataclass(frozen=True)
class SlateScan:
    """영상 하나의 슬레이트 검출 결과.

    Attributes:
        sightings: 스트림 시각 순 첫 등장 목록.
        duration_ms: 컨테이너 길이 ms (모르면 None). 슬레이트 신뢰도 계산에서 앵커 밖 외삽 구간을
            재는 데 쓴다.
    """

    sightings: list[SlateSighting]
    duration_ms: float | None  # 컨테이너 길이 (모르면 None)


def _decode_qr(detector: cv2.QRCodeDetector, gray: NDArray[np.uint8], prefix: str) -> str | None:
    """회색조 프레임에서 QR을 읽어 `prefix`로 시작하면 내용을, 아니면(없음·다른 QR) None을 준다."""
    text, _, _ = detector.detectAndDecode(gray)
    return text if text and text.startswith(prefix) else None


def detect_slates(path: Path, policy: SlatePolicy) -> list[SlateSighting]:
    """`scan_slates`의 첫 등장 목록만 돌려주는 편의 함수."""
    return scan_slates(path, policy).sightings


def scan_slates(path: Path, policy: SlatePolicy) -> SlateScan:
    """영상 파일에서 슬레이트 첫 등장을 찾는다.

    Args:
        path: 영상 파일 (로컬). 첫 영상 스트림만 본다.
        policy: `sync.yaml slate`.

    Returns:
        `SlateScan`. payload마다 첫 등장 하나. 스트림 첫 프레임부터 보인 슬레이트는 넣지 않는다.

    탐색 구간: 길이가 2·`search_window_ms`보다 길면 [0, w]와 [길이 - w, 길이] 두 구간,
    아니면(또는 길이를 모르면) 전체. 두 번째 구간은 seek로 건너뛴다.
    """
    detector = cv2.QRCodeDetector()
    found: dict[str, SlateSighting] = {}
    # 첫 프레임부터 보여 뜬 시각을 모르는 payload. 다음 프레임에서 다시 잡지 않도록 기억한다
    unknown_onset: set[str] = set()
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        # 컨테이너 길이는 마이크로초 단위(av.time_base)라 1000으로 나눠 ms
        duration_ms = float(c.duration / 1000) if c.duration is not None else None
        windows = [(0.0, policy.search_window_ms)]
        if duration_ms is not None and duration_ms > 2 * policy.search_window_ms:
            windows.append((duration_ms - policy.search_window_ms, duration_ms))
        else:
            windows = [(0.0, float("inf"))]

        for start_ms, end_ms in windows:
            if start_ms > 0:
                # 끝 구간: 그 앞 키프레임으로 seek. 디코딩은 start_ms 이전 프레임부터 나올 수 있다
                c.seek(int(start_ms / 1000 / tb), stream=stream, backward=True)
            # (시각, 영상, 바로 앞 프레임 시각). 앞 프레임이 없으면(스트림 첫 프레임) None
            recent: deque[tuple[float, NDArray[np.uint8], float | None]] = deque(
                maxlen=policy.frame_stride
            )
            # 직전에 디코딩한 프레임 시각. seek 뒤 첫 프레임도 None이지만, 그 프레임은 보통
            # start_ms 앞이라 건너뛰고 다음 프레임부터는 앞 시각이 채워진다
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
                # frame_stride 간격 프레임만 QR 디코딩 (비용 절감). 사이 프레임은 recent에 보관
                if i % policy.frame_stride:
                    continue
                payload = _decode_qr(detector, gray, policy.payload_prefix)
                if payload is None or payload in found or payload in unknown_onset:
                    continue
                # 새 슬레이트: 건너뛴 프레임을 최신부터 거꾸로 되짚어 처음 보인 프레임을 찾는다
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
