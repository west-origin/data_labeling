"""검수 화면용 시계열 CSV.

장갑 압력 합(좌·우)과 IMU 가속도 크기를 마스터 타임라인 ms 격자로 옮긴다.

스트림 시각은 세션의 동기화 결과(Stream.to_master_ms)로 마스터 시각으로 바꾼다.
없는 스트림의 열은 0이다.
센서 값만 담으므로 개인정보가 없고, 라벨링 버킷에 둔다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dlp_review.labelstudio import CHANNELS
from dlp_schema.session import Session, StreamKind
from dlp_sync.signals import Series

KIND_TO_CHANNEL = {
    StreamKind.GLOVE_LEFT: "glove_left",
    StreamKind.GLOVE_RIGHT: "glove_right",
    StreamKind.IMU: "imu_acc",
}


def write_timeseries_csv(
    session: Session, series: dict[str, Series], out: Path, *, rate_hz: float
) -> int:
    """series: 스트림 ID → 그 스트림 시계의 시계열. 쓴 행 수를 돌려준다.

    rate_hz: 격자 간격 (config/policies/review.yaml media.timeseries_rate_hz).
    """
    grid = np.arange(0, session.duration_ms, 1000 / rate_hz)
    columns = {c: np.zeros(grid.size) for c in CHANNELS}
    for stream in session.streams:
        channel = KIND_TO_CHANNEL.get(stream.kind)
        if channel is None or stream.stream_id not in series:
            continue
        s = series[stream.stream_id]
        master = np.array([stream.to_master_ms(float(t)) for t in s.t_ms])
        columns[channel] = np.interp(grid, master, s.values, left=0.0, right=0.0)
    with out.open("w", encoding="utf-8") as f:
        f.write("time_ms," + ",".join(CHANNELS) + "\n")
        for i, t in enumerate(grid):
            f.write(f"{t:.1f}," + ",".join(f"{columns[c][i]:.4f}" for c in CHANNELS) + "\n")
    return int(grid.size)
