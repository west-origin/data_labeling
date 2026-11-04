"""SfM 결과 + 블러 사진 → gsplat 3D 가우시안 학습 → 3DGS PLY (격리 환경, CUDA, ADR 0066).

TODO(real-model): 실행해 보지 않았다. 밀도 조절 없는 최소 학습 루프다 (README.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

C0 = 0.28209479177387814


def main() -> None:
    """SfM 점에서 시작해 L1 손실로 위치·크기·회전·불투명도·색을 학습한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--sfm", type=Path, required=True)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    cams = json.loads((a.sfm / "cameras.json").read_text("utf-8"))
    npz = np.load(a.sfm / "points.npz")
    n = len(npz["points"])
    means = torch.tensor(npz["points"], dtype=torch.float32, device=dev, requires_grad=True)
    scales = torch.full((n, 3), float(np.log(0.01)), device=dev, requires_grad=True)
    quats = torch.tensor([[1.0, 0, 0, 0]] * n, device=dev, requires_grad=True)
    opac = torch.full((n,), 2.197, device=dev, requires_grad=True)
    sh0 = torch.tensor(
        (npz["colors"] / 255.0 - 0.5) / C0, dtype=torch.float32, device=dev, requires_grad=True
    )
    opt = torch.optim.Adam([means, scales, quats, opac, sh0], lr=1e-3)
    views = []
    for c in cams:
        img = (
            np.asarray(
                Image.open(a.images / f"{c['photo_id']}.png").convert("RGB"), dtype=np.float32
            )
            / 255.0
        )
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3] = np.asarray(c["rotation"]).reshape(3, 3)
        vm[:3, 3] = c["translation"]
        k = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float32)
        views.append(
            (
                torch.tensor(img, device=dev),
                torch.tensor(vm, device=dev),
                torch.tensor(k, device=dev),
                c["width"],
                c["height"],
            )
        )
    rng = np.random.default_rng(0)
    for _ in range(a.iterations):
        img, vm, k, w, h = views[int(rng.integers(len(views)))]
        colors = (sh0 * C0 + 0.5).clamp(0, 1)
        out, _, _ = rasterization(
            means,
            quats / quats.norm(dim=1, keepdim=True),
            scales.exp(),
            opac.sigmoid(),
            colors,
            vm[None],
            k[None],
            w,
            h,
        )
        loss = (out[0] - img).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    data = torch.cat(
        [
            means,
            torch.zeros_like(means),
            sh0,
            opac[:, None],
            scales,
            quats / quats.norm(dim=1, keepdim=True),
        ],
        dim=1,
    )
    names = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
    a.out.write_bytes(header.encode() + data.detach().cpu().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    main()
