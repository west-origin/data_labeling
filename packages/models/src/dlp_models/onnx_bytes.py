"""아주 작은 ONNX 모델 바이트를 onnx 패키지 없이 만든다 (protobuf 형식을 직접 쓴다).

본 가상 환경에는 onnx 패키지가 없다 (onnxruntime만 있다). `dlp models device`가 Identity 모델로
ONNX Runtime 세션을 열어 고른 장치가 실제로 잡히는지 보고(ADR 0072), 세션을 실제로 여는 시험(장치
선택, 재학습 YOLOX 로더)이 대역 모델로 쓴다. 쓰는 필드: ModelProto 1 ir_version·7 graph·8
opset_import, GraphProto 1 node·2 name·5 initializer·11 input·12 output, NodeProto 1 input·2
output·4 op_type, TensorProto 1 dims·2 data_type·8 name·9 raw_data. 학습한 모델이 아니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# TensorProto.DataType FLOAT
_FLOAT = 1
# ONNX IR 8, 기본 연산 집합 13 (Identity가 있다)
IR_VERSION = 8
OPSET = 13


def _varint(n: int) -> bytes:
    """protobuf varint (음수가 아닌 정수)."""
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | 0x80 if n else b)
        if not n:
            return bytes(out)


def _field(no: int, data: bytes) -> bytes:
    """길이 구분 필드 (wire type 2)."""
    return _varint(no << 3 | 2) + _varint(len(data)) + data


def _int(no: int, v: int) -> bytes:
    """varint 필드 (wire type 0)."""
    return _varint(no << 3) + _varint(v)


def _value_info(name: str, shape: tuple[int, ...]) -> bytes:
    """ValueInfoProto (float 텐서, 고정 shape)."""
    dims = b"".join(_field(1, _int(1, d)) for d in shape)
    return _field(1, name.encode()) + _field(2, _field(1, _int(1, _FLOAT) + _field(2, dims)))


def _model(graph: bytes) -> bytes:
    """ModelProto (기본 도메인 연산 집합 하나)."""
    return _int(1, IR_VERSION) + _field(7, graph) + _field(8, _field(1, b"") + _int(2, OPSET))


def identity_onnx(shape: tuple[int, ...]) -> bytes:
    """입력 `x`(float, shape)를 그대로 `y`로 내는 ONNX (Identity 하나)."""
    node = _field(1, b"x") + _field(2, b"y") + _field(4, b"Identity")
    graph = _field(1, node) + _field(2, b"identity")
    graph += _field(11, _value_info("x", shape)) + _field(12, _value_info("y", shape))
    return _model(graph)


def constant_onnx(output: np.ndarray[Any, Any], size: int) -> bytes:
    """입력 images (1, 3, size, size)를 무시하고 상수 output을 내는 ONNX (Identity 하나)."""
    raw = np.ascontiguousarray(output, dtype="<f4")
    init = b"".join(_int(1, d) for d in raw.shape) + _int(2, _FLOAT) + _field(8, b"const")
    init += _field(9, raw.tobytes())
    node = _field(1, b"const") + _field(2, b"output") + _field(4, b"Identity")
    graph = _field(1, node) + _field(2, b"tiny") + _field(5, init)
    graph += _field(11, _value_info("images", (1, 3, size, size)))
    graph += _field(12, _value_info("output", raw.shape))
    return _model(graph)
