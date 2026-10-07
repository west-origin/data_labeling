"""Depth Anything V2 Metric Indoor Small(공식 Hugging Face 가중치)을 ONNX로 내보낸다.

PyTorch CPU판과 transformers가 필요하다. 저장소 의존성에는 넣지 않고 일회성으로 실행한다:

    uv run --no-project --python 3.12 \\
      --with "torch==2.9.1" --with "transformers==4.57.1" \\
      --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
      python scripts/export_depth_onnx.py

출력: data/models/depth_anything_v2_metric_indoor_small.onnx (입력 pixel_values (1,3,H,W),
H·W는 14의 배수, 출력 predicted_depth (1,H,W) 미터).

보통은 `make export-models`(`dlp models export`)가 `config/models.yaml`의 `export` 항목을 보고 위
명령으로 이 스크립트를 부른다. 프리라벨의 깊이 기반 3D 궤적(`dlp_prelabel.lift3d.DepthLifter`)이
이 ONNX를 쓴다. 상업 사용 분류는 `config/models.yaml`과 ADR 0010을 따른다.
마지막에 출력 경로와 sha256을 찍는다 (`config/models.yaml`에 적을 해시 확인용).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForDepthEstimation

# Hugging Face 모델 저장소와 고정 커밋 (같은 리비전이면 같은 가중치, 재현성)
REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
# 출력 경로 (<저장소>/data/models, 저장소에 넣지 않는다). config/models.yaml의 path와 같아야 한다
OUT = Path(__file__).resolve().parents[1] / "data/models/depth_anything_v2_metric_indoor_small.onnx"


class Wrapper(torch.nn.Module):
    """ONNX 내보내기용 얇은 래퍼: 출력 객체에서 `predicted_depth` 텐서만 돌려준다."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) 정규화된 입력 → (1, H, W) 깊이(미터)."""
        return self.model(pixel_values=pixel_values).predicted_depth


def main() -> int:
    """고정 리비전의 가중치를 받아 ONNX(opset 17, 높이·너비 동적 축)로 내보낸다. 반환: 종료 코드 0.

    더미 입력은 518x518(14의 배수, 모델 기본 크기)이다. `dynamo=False`로 기존 TorchScript 내보내기
    경로를 쓴다. Hugging Face 허브에 접속하므로 네트워크가 필요하다.
    """
    model = AutoModelForDepthEstimation.from_pretrained(REPO, revision=REVISION).eval()
    dummy = torch.zeros(1, 3, 518, 518)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Wrapper(model),
        (dummy,),
        str(OUT),
        input_names=["pixel_values"],
        output_names=["predicted_depth"],
        dynamic_axes={
            "pixel_values": {2: "height", 3: "width"},
            "predicted_depth": {1: "height", 2: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(OUT, hashlib.sha256(OUT.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
