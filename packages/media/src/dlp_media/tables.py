"""Parquet 입출력. pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.

WP3. PTS 인덱스(`pts.PtsIndex`), IMU(`imu.ImuData`), 장갑(`glove.GloveData`)의 정규화 파일과
사이드카 읽기에 쓴다. 파일 단위 메타데이터(time_base, source, sample_rate_hz 등)는 Parquet 스키마
메타데이터(문자열 → 문자열)에 둔다.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray


def write_parquet(
    path: Path, columns: Mapping[str, NDArray[Any]], metadata: Mapping[str, str] | None = None
) -> None:
    """열 이름 → 1차원 배열을 Parquet 파일로 쓴다 (덮어쓴다).

    Args:
        path: 쓸 파일.
        columns: 열 이름 → 배열 (모두 같은 길이, 입력 순서가 열 순서).
        metadata: 스키마 메타데이터 (없으면 넣지 않는다).
    """
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    if metadata:
        table = table.replace_schema_metadata(dict(metadata))
    pq.write_table(table, path)


def read_parquet(path: Path) -> tuple[dict[str, NDArray[Any]], dict[str, str]]:
    """(열 이름 → 배열, 스키마 메타데이터).

    메타데이터는 UTF-8로 디코딩한다. pyarrow가 넣는 "pandas" 등 다른 키도 그대로 들어올 수 있다.
    """
    table = pq.read_table(path)
    raw = table.schema.metadata or {}
    metadata = {k.decode(): v.decode() for k, v in raw.items()}
    return {name: table.column(name).to_numpy() for name in table.column_names}, metadata
