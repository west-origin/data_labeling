"""Parquet 입출력. pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다."""

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
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    if metadata:
        table = table.replace_schema_metadata(dict(metadata))
    pq.write_table(table, path)


def read_parquet(path: Path) -> tuple[dict[str, NDArray[Any]], dict[str, str]]:
    """(열 이름 → 배열, 스키마 메타데이터)."""
    table = pq.read_table(path)
    raw = table.schema.metadata or {}
    metadata = {k.decode(): v.decode() for k, v in raw.items()}
    return {name: table.column(name).to_numpy() for name in table.column_names}, metadata
