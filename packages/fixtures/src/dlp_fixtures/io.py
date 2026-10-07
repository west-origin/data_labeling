"""픽스처 파일 입출력: WAV, Parquet, JSONL.

pyarrow에는 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import wave
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel


def write_wav(path: Path, samples: NDArray[np.float32], sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32767).astype(np.float32), rate


def write_parquet(
    path: Path, columns: Mapping[str, NDArray[Any]], metadata: Mapping[str, str] | None = None
) -> None:
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    if metadata:
        table = table.replace_schema_metadata(dict(metadata))
    pq.write_table(table, path)


def read_parquet(path: Path) -> dict[str, NDArray[Any]]:
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy() for name in table.column_names}


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
