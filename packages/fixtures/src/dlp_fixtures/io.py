"""픽스처 파일 입출력: WAV, Parquet, JSONL, JSON (WP2).

생성기(`sync.py`, `actions.py`, `__init__.generate_all`)와 일부 테스트가 쓴다. 실제 수집 경로의
정규화 형식(`dlp_media`)과 같은 열 이름(`t_ms` 등)을 쓰도록 생성기가 맞춘다.

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
    """모노 16비트 PCM WAV로 쓴다.

    Args:
        path: 출력 파일.
        samples: -1~1 범위의 float 샘플 (밖은 잘라낸다).
        sample_rate: 샘플레이트 Hz.
    """
    # -1~1 → int16 (리틀 엔디언). 32767을 곱해 +1이 넘치지 않게 한다
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    """`write_wav`로 쓴 모노 16비트 WAV → (float32 샘플 -1~1, 샘플레이트 Hz)."""
    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / 32767).astype(np.float32), rate


def write_parquet(
    path: Path, columns: Mapping[str, NDArray[Any]], metadata: Mapping[str, str] | None = None
) -> None:
    """열 dict를 Parquet 한 파일로 쓴다.

    Args:
        path: 출력 파일.
        columns: 열 이름 → 같은 길이의 1차원 배열 (시각 열은 `t_ms`, ms).
        metadata: 스키마 메타데이터 (예: `{"clock": "bodycam"}` = 어느 시계의 시각인지). None·빈
            dict면 넣지 않는다.
    """
    table = pa.table({name: pa.array(values) for name, values in columns.items()})
    if metadata:
        table = table.replace_schema_metadata(dict(metadata))
    pq.write_table(table, path)


def read_parquet(path: Path) -> dict[str, NDArray[Any]]:
    """Parquet 파일 → 열 이름 → numpy 배열 (메타데이터는 버린다)."""
    table = pq.read_table(path)
    return {name: table.column(name).to_numpy() for name in table.column_names}


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Pydantic 레코드를 한 줄에 하나씩 JSON으로 쓴다 (`model_dump_json`).

    Returns:
        쓴 레코드 수.
    """
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def write_json(path: Path, data: Any) -> None:
    """JSON 파일로 쓴다 (UTF-8, 한글 그대로, 들여쓰기 2, 끝 줄바꿈)."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
