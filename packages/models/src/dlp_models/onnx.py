"""ONNX Runtime 세션의 얇은 타입 래퍼 (onnxruntime에는 타입 스텁이 없다). CPU 실행."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class OnnxModel:
    def __init__(self, model: Path, threads: int = 0) -> None:
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )

    def run(
        self, outputs: list[str], feeds: dict[str, NDArray[np.generic]]
    ) -> list[NDArray[np.float32]]:
        return [np.asarray(x, dtype=np.float32) for x in self._session.run(outputs, feeds)]
