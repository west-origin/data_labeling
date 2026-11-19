"""추론 장치 선택: 환경 변수 `DLP_INFERENCE_DEVICE` (cpu | auto | cuda, 기본 cpu, ADR 0072).

모델 파일과 정책은 그대로이고 실행 장치만 고른다. CI·개발 환경은 GPU가 없어 기본은 cpu다 (지금까지와
같다). GPU 서버는 onnxruntime 대신 같은 판의 onnxruntime-gpu를 깔고(`docs/handover/development.md`
"GPU 서버") `cuda`(CUDA 실행 제공자가 없으면 실패) 또는 `auto`(있으면 GPU, 없으면 CPU)로 둔다.

쓰는 곳: `onnx.OnnxModel`(OWLv2, 메트릭 깊이, SAM2, YOLOX 탐지의 ONNX 세션), 프리라벨 전신 자세
(rtmlib RTMPose·YOLOX의 `device`). MediaPipe 손 판정과 YuNet 얼굴 탐지는 CPU 그대로다: MediaPipe GPU
위임은 리눅스에서 EGL과 GPU 드라이버가 따로 필요하고, OpenCV pip 판에는 CUDA 백엔드가 없다.

GPU를 골랐는데 세션이 CUDA를 받지 못하면 멈춘다 (`check_session`). ONNX Runtime은 CUDA·cuDNN
라이브러리를 찾지 못하면 경고만 내고 CPU로 돌기 때문이다.

장치는 모델 버전에 넣지 않는다 (ADR 0072). 같은 모델·정책이면 같은 판으로 보고, CPU와 GPU의
부동소수점 차이는 다시 계산할 사유로 보지 않는다. 어느 장치로 돌렸는지는 명령 출력의 `describe` 한
줄(운영 콘솔 작업 기록에 남는다)로 구분한다. 관련 문제: F-PAR-98.
"""

# onnxruntime은 타입 스텁이 없어 이 모듈에서만 알 수 없는 타입 경고를 끈다 (onnx.py와 같다).
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Literal

import onnxruntime as ort

ENV = "DLP_INFERENCE_DEVICE"
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
Device = Literal["cpu", "cuda"]


class DeviceError(RuntimeError):
    """추론 장치 설정이 틀렸거나, 요청한 GPU를 이 환경의 onnxruntime이 쓸 수 없다."""


def available_providers() -> list[str]:
    """이 환경의 onnxruntime이 쓸 수 있는 실행 제공자 (onnxruntime-gpu면 CUDA가 들어 있다)."""
    return [str(p) for p in ort.get_available_providers()]


def inference_device(
    env: Mapping[str, str] | None = None,
    available: Callable[[], list[str]] | None = None,
) -> Device:
    """`DLP_INFERENCE_DEVICE`가 고른 장치.

    Args:
        env: 환경 변수 (None이면 `os.environ`). 값은 대소문자·앞뒤 공백을 무시한다.
        available: 쓸 수 있는 실행 제공자 목록을 주는 함수 (None이면 `available_providers`).

    Returns:
        `cpu` 또는 `cuda`. 값이 없거나 `cpu`면 cpu, `auto`면 CUDA 실행 제공자가 있을 때 cuda.

    Raises:
        DeviceError: 값이 cpu·auto·cuda가 아닐 때, `cuda`인데 CUDA 실행 제공자가 없을 때
            (onnxruntime-gpu가 아니거나 CUDA·cuDNN 라이브러리를 찾지 못함).
    """
    value = (env if env is not None else os.environ).get(ENV, "").strip().lower() or "cpu"
    if value == "cpu":
        return "cpu"
    if value not in ("auto", "cuda"):
        raise DeviceError(f"{ENV}={value!r}: cpu, auto, cuda 중 하나여야 합니다")
    has_cuda = CUDA_PROVIDER in (available or available_providers)()
    if value == "auto":
        return "cuda" if has_cuda else "cpu"
    if not has_cuda:
        raise DeviceError(
            f"{ENV}=cuda인데 onnxruntime에 {CUDA_PROVIDER}가 없습니다. onnxruntime 대신 같은 판의 "
            "onnxruntime-gpu를 깔고 CUDA·cuDNN이 있는지 확인하세요 (docs/handover/development.md "
            "GPU 서버). GPU가 없을 때 CPU로 돌리려면 auto로 둡니다"
        )
    return "cuda"


@cache
def _preload_cuda() -> None:
    """pip로 깐 CUDA·cuDNN 라이브러리(onnxruntime-gpu[cuda,cudnn])를 첫 GPU 세션보다 먼저 불러온다.

    CPU판 onnxruntime에서는 아무것도 하지 않는다. 시스템에 깐 CUDA(LD_LIBRARY_PATH)도 그대로 찾는다.
    """
    ort.preload_dlls()


def prepare(device: Device) -> None:
    """세션을 만들기 전에 부른다. GPU면 CUDA 라이브러리를 한 번 불러온다."""
    if device == "cuda":
        _preload_cuda()


def onnx_providers(device: Device) -> list[str]:
    """ONNX Runtime 세션의 실행 제공자 목록 (GPU면 CUDA 다음에 CPU를 대비로 둔다)."""
    return [CUDA_PROVIDER, CPU_PROVIDER] if device == "cuda" else [CPU_PROVIDER]


def check_session(device: Device, providers: Sequence[str], what: str) -> None:
    """GPU를 골랐는데 세션이 CUDA 실행 제공자를 받지 못했으면 멈춘다.

    Args:
        device: `inference_device`가 고른 장치.
        providers: 만든 세션이 실제로 쓰는 실행 제공자 (`InferenceSession.get_providers()`).
        what: 오류 메시지에 쓸 모델 이름.

    Raises:
        DeviceError: device가 cuda인데 providers에 CUDA 실행 제공자가 없을 때.
    """
    if device == "cuda" and CUDA_PROVIDER not in providers:
        raise DeviceError(
            f"{what}: {ENV}가 GPU를 골랐는데 세션이 {CUDA_PROVIDER}를 쓰지 못합니다 (실제 "
            f"{', '.join(providers) or '없음'}). CUDA·cuDNN 라이브러리가 onnxruntime-gpu 판과 "
            "맞는지 확인하세요 (docs/handover/development.md GPU 서버)"
        )


def describe(
    env: Mapping[str, str] | None = None,
    available: Callable[[], list[str]] | None = None,
) -> str:
    """명령 출력에 남기는 한 줄: 고른 장치와 근거 (`dlp prelabel run`, `dlp privacy detect`).

    Raises:
        DeviceError: `inference_device`와 같다.
    """
    source = env if env is not None else os.environ
    device = inference_device(source, available)
    value = source.get(ENV, "").strip()
    reason = f"{ENV}={value}" if value else f"{ENV} 없음, 기본값"
    return f"추론 장치: {device} ({reason})"
