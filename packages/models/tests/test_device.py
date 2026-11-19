"""추론 장치 선택 `DLP_INFERENCE_DEVICE` (ADR 0072, F-PAR-98).

GPU가 없는 환경에서 볼 수 있는 것만 본다: 값 해석, 실행 제공자 목록, 기본 CPU 세션, GPU를 골랐는데
CUDA를 받지 못한 세션이 조용히 CPU로 돌지 않고 멈추는 것. 실제 GPU에서 빨라지는지는 GPU 서버에서
`dlp models device`와 실영상으로 잰다 (F-PAR-98에 남김).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlp_models import device as device_mod
from dlp_models.device import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ENV,
    DeviceError,
    check_session,
    describe,
    inference_device,
    onnx_providers,
)
from dlp_models.onnx import OnnxModel
from dlp_models.onnx_bytes import identity_onnx

GPU_BUILD = [CUDA_PROVIDER, CPU_PROVIDER]
CPU_BUILD = ["AzureExecutionProvider", CPU_PROVIDER]


def _never() -> list[str]:
    """cpu는 실행 제공자를 묻지 않는다는 확인용."""
    raise AssertionError("cpu를 고를 때 실행 제공자를 묻지 않는다")


def test_cpu_is_the_default_and_values_ignore_case_and_spaces() -> None:
    assert inference_device({}, _never) == "cpu"
    assert inference_device({ENV: ""}, _never) == "cpu"
    assert inference_device({ENV: " CPU "}, _never) == "cpu"
    assert inference_device({ENV: "Cuda"}, lambda: GPU_BUILD) == "cuda"


def test_auto_picks_the_gpu_only_when_the_cuda_provider_is_there() -> None:
    assert inference_device({ENV: "auto"}, lambda: GPU_BUILD) == "cuda"
    assert inference_device({ENV: "auto"}, lambda: CPU_BUILD) == "cpu"


def test_cuda_without_the_provider_and_unknown_values_stop() -> None:
    with pytest.raises(DeviceError, match="onnxruntime-gpu"):
        inference_device({ENV: "cuda"}, lambda: CPU_BUILD)
    with pytest.raises(DeviceError, match="cpu, auto, cuda"):
        inference_device({ENV: "gpu"}, lambda: GPU_BUILD)


def test_gpu_sessions_keep_the_cpu_after_the_cuda() -> None:
    assert onnx_providers("cpu") == [CPU_PROVIDER]
    assert onnx_providers("cuda") == [CUDA_PROVIDER, CPU_PROVIDER]


def test_session_check_stops_only_a_gpu_request_that_got_no_cuda() -> None:
    check_session("cpu", [CPU_PROVIDER], "m.onnx")
    check_session("cuda", GPU_BUILD, "m.onnx")
    with pytest.raises(DeviceError, match=r"m\.onnx: .*CPUExecutionProvider"):
        check_session("cuda", [CPU_PROVIDER], "m.onnx")


def test_describe_names_the_device_and_why() -> None:
    assert describe({}) == f"추론 장치: cpu ({ENV} 없음, 기본값)"
    assert describe({ENV: "auto"}, lambda: GPU_BUILD) == f"추론 장치: cuda ({ENV}=auto)"
    assert describe({ENV: "auto"}, lambda: CPU_BUILD) == f"추론 장치: cpu ({ENV}=auto)"


def _identity(tmp_path: Path) -> Path:
    path = tmp_path / "identity.onnx"
    path.write_bytes(identity_onnx((2, 3)))
    return path


def test_onnx_model_runs_on_the_cpu_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV, raising=False)
    model = OnnxModel(_identity(tmp_path))
    assert model.providers == [CPU_PROVIDER]
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(model.run(["y"], {"x": x})[0], x)


@pytest.mark.filterwarnings("ignore:Specified provider 'CUDAExecutionProvider'")
def test_requested_gpu_never_falls_back_to_the_cpu_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ONNX Runtime은 CUDA를 못 쓰면 경고만 내고 CPU 세션을 만든다. 우리는 거기서 멈춘다."""
    if CUDA_PROVIDER in device_mod.available_providers():
        pytest.skip("CUDA를 실제로 쓸 수 있는 환경 (CPU 대체를 재현할 수 없다)")
    path = _identity(tmp_path)
    monkeypatch.setenv(ENV, "cuda")
    # CPU판 onnxruntime: 세션을 열기 전에 멈춘다 (파일을 읽지 않는다)
    with pytest.raises(DeviceError, match="onnxruntime-gpu"):
        OnnxModel(tmp_path / "없는-파일.onnx")
    # GPU판이 깔렸지만 CUDA·cuDNN을 찾지 못한 것처럼: 제공자 목록에는 CUDA가 있고, 세션은 실제
    # onnxruntime이 경고만 낸 뒤 CPU로 열린다
    monkeypatch.setattr(device_mod, "available_providers", lambda: GPU_BUILD)
    with pytest.raises(DeviceError, match=r"identity\.onnx: .*쓰지 못합니다"):
        OnnxModel(path)
    monkeypatch.setenv(ENV, "auto")
    with pytest.raises(DeviceError, match="쓰지 못합니다"):
        OnnxModel(path)
