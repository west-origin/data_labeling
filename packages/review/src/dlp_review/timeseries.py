"""검수 화면용 시계열 CSV (WP6, ADR 0006).

장갑 압력 합(좌·우)과 IMU 가속도 크기를 마스터 타임라인 ms 격자로 옮긴다.

스트림 시각은 세션의 동기화 결과(Stream.to_master_ms)로 마스터 시각으로 바꾼다.
없는 스트림의 열은 0이다.
센서 값만 담으므로 개인정보가 없고, 라벨링 버킷에 둔다.

파이프라인 위치: `dlp_review.tasks.create_labeling_tasks`가 Label Studio 작업을 만들 때 부른다
(`dlp review create`, `dlp review assign`). 입력은 원본 버킷에서 읽은 장갑·IMU Parquet의 시계열,
출력은 `sessions/<세션>/review/timeseries.csv`(라벨링 버킷)로 올라갈 로컬 CSV 파일이다.

공개 이름:
- `KIND_TO_CHANNEL`: 스트림 종류 → CSV 열 이름.
- `write_timeseries_csv`: 시계열을 격자로 보간해 CSV로 쓴다.

주의점:
- CSV의 `time_ms` 열은 마스터 타임라인 ms(ADR 0019)라서 Label Studio 시간 구간 라벨의 시각과
  같은 축이다. 격자 간격은 `config/policies/review.yaml` `media.timeseries_rate_hz`에서 온다.
- `time_ms`는 소수 한 자리 문자열로 쓴다. 격자 간격(1000/rate_hz)이 정수가 아니면 소수가 생긴다
  (현재 기본 50 Hz → 20 ms 간격이라 정수).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dlp_review.labelstudio import CHANNELS
from dlp_schema.session import Session, StreamKind
from dlp_sync.signals import Series

# 스트림 종류 → CSV 열 이름. 열 순서와 이름 목록은 `dlp_review.labelstudio.CHANNELS`가 정한다
# (Label Studio 프로젝트 설정 XML의 Channel column과 같아야 화면에 그려진다).
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

    인자:
    - session: 세션. `duration_ms`(마스터 타임라인 길이)와 스트림별 동기화 결과를 쓴다.
    - series: 스트림 ID → `Series(t_ms, values)`. t_ms는 그 스트림 자체 시계의 ms다.
      장갑은 압력 채널 합, IMU는 가속도 크기 (`dlp_sync.signals`가 만든다).
    - out: 쓸 CSV 경로 (덮어쓴다).
    - rate_hz: 격자 주파수(Hz). 격자 간격 = 1000 / rate_hz ms.

    반환: 데이터 행 수(머리글 제외) = 격자 점 개수.

    동작: 격자 [0, duration_ms) 위로 선형 보간하고, 스트림 구간 밖은 0으로 채운다.
    `KIND_TO_CHANNEL`에 없는 스트림(영상 등)이나 series에 없는 스트림은 건너뛰어 열이 0으로 남는다.
    부작용: out 파일 쓰기만 한다 (저장소 업로드는 호출자가 한다).
    """
    # 마스터 타임라인 격자 (ms, float). 끝점 duration_ms는 넣지 않는다.
    grid = np.arange(0, session.duration_ms, 1000 / rate_hz)
    columns = {c: np.zeros(grid.size) for c in CHANNELS}
    for stream in session.streams:
        channel = KIND_TO_CHANNEL.get(stream.kind)
        if channel is None or stream.stream_id not in series:
            continue
        s = series[stream.stream_id]
        # 스트림 시각 → 마스터 시각 (오프셋·드리프트 보정, dlp_sync 결과)
        master = np.array([stream.to_master_ms(float(t)) for t in s.t_ms])
        # 녹화 전후(스트림이 없는 구간)는 0으로 둔다 (마지막 값을 끌어오지 않는다)
        columns[channel] = np.interp(grid, master, s.values, left=0.0, right=0.0)
    with out.open("w", encoding="utf-8") as f:
        f.write("time_ms," + ",".join(CHANNELS) + "\n")
        for i, t in enumerate(grid):
            f.write(f"{t:.1f}," + ",".join(f"{columns[c][i]:.4f}" for c in CHANNELS) + "\n")
    return int(grid.size)
