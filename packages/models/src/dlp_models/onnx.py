"""ONNX Runtime 세션의 얇은 타입 래퍼 (onnxruntime에는 타입 스텁이 없다). CPU 실행.

`owlv2.Owlv2`와 `depth.MetricDepth`가 쓴다. CI·개발 환경은 GPU가 없다고 보고 실행 제공자를
`CPUExecutionProvider` 하나로 고정한다. GPU 서버에서 쓰려면 이 래퍼에 제공자 선택을 더해야 한다.
관련: WP8, ADR 0009.

- `OnnxModel`: 모델 파일을 열어 두고 `run(출력 이름, 입력)`으로 추론한다. 출력은 모두 float32 배열로
  바꾼다.
"""

# onnxruntime은 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray


class OnnxModel:
    """CPU용 ONNX Runtime 추론 세션.

    생성할 때 모델 파일을 읽어 그래프를 준비하므로(수백 MB면 수 초) 프레임마다 만들지 말고 한 번
    만들어 재사용한다.
    """

    def __init__(self, model: Path, threads: int = 0) -> None:
        """모델 파일로 세션을 연다.

        Args:
            model: `.onnx` 파일 경로 (보통 `registry.resolve`가 돌려준 경로).
            threads: 연산 내부 스레드 수(`intra_op_num_threads`). 0이면 onnxruntime 기본값
                (물리 코어 수)을 쓴다.

        Raises:
            onnxruntime 예외: 파일이 없거나 ONNX 형식이 아닐 때 (호출자는 `resolve`로 먼저
            확인한다).
        """
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"]
        )

    def run(
        self, outputs: list[str], feeds: dict[str, NDArray[np.generic]]
    ) -> list[NDArray[np.float32]]:
        """추론한다.

        Args:
            outputs: 받을 출력 텐서 이름 (모델 그래프의 이름과 같아야 한다).
            feeds: 입력 이름 → 배열. dtype·shape는 모델이 요구하는 그대로 넘긴다 (변환하지 않는다).

        Returns:
            `outputs` 순서대로 float32 배열. 정수 출력이 있는 모델에는 맞지 않는다 (지금 쓰는 모델은
            모두 실수 출력이다).
        """
        return [np.asarray(x, dtype=np.float32) for x in self._session.run(outputs, feeds)]
