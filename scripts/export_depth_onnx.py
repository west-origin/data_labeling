"""Depth Anything V2 Metric Indoor Small(공식 Hugging Face 가중치)을 ONNX로 내보낸다.

PyTorch CPU판과 transformers가 필요하다. 저장소 의존성에는 넣지 않고 일회성으로 실행한다:

    uv run --no-project --python 3.12 \\
      --with "torch==2.9.1" --with "transformers==4.57.1" \\
      --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \\
      python scripts/export_depth_onnx.py

출력: data/models/depth_anything_v2_metric_indoor_small.onnx (입력 pixel_values (1,3,H,W),
H·W는 14의 배수, 출력 predicted_depth (1,H,W) 미터).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
from transformers import AutoModelForDepthEstimation

REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
OUT = Path(__file__).resolve().parents[1] / "data/models/depth_anything_v2_metric_indoor_small.onnx"


class Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth


def main() -> int:
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
