#!/usr/bin/env python3
"""YOLOX 0.3.0 미세조정 + ONNX 내보내기 (격리 환경 전용, ADR 0062).

`dlp_train.yolox.YoloxTrainer`가 `uv run --project scripts/yolox-train-env --locked --isolated
--extra <cpu|cu128> python scripts/yolox-train-env/finetune.py --spec <spec.json>`으로 부른다.
이 스크립트는 DB·버킷에 접근하지 않는다. 입력은 spec이 가리키는 COCO 디렉터리(블러본 프레임, 학습
임시 디렉터리)와 출발 가중치(.pth, 호출자가 sha256 확인)뿐이다.

순서
1. YOLOX를 고정 커밋으로 clone(spec yolox_src, 이미 있으면 커밋만 확인)하고 경로로 쓴다
   (`scripts/pretrain/train_yolox.py ensure_yolox`, 설치하지 않는다).
2. Exp 파일을 만든다: 클래스 수·depth·width·입력 크기·데이터. YOLOX 내부
증강(모자이크·mixup·HSV·뒤집기·
   회전)은 모두 끈다. 증강은 학습 데이터에 이미 들어 있다 (ADR 0057, `dlp_train.augment`).
3. 학습: YOLOX Trainer는 CUDA를 요구하므로 쓰지 않고 같은 구성 요소(COCODataset·TrainTransform·
   get_model·get_optimizer)로 짧은 학습 반복을 직접 돈다. 장치는 spec device(cuda면 CUDA가 실제로
   있어야 한다). 출발 가중치에서 모양이 맞는 층만 읽는다 (`yolox.utils.load_ckpt`, 클래스 수가 다른
   머리는 새로 시작).
4. 내보내기: `train_yolox.export_onnx` (decode_in_inference false, onnxsim 없음)와 `check_onnx`
   (PyTorch 출력과 ONNX Runtime 출력 비교).
5. `<out_dir>/model.onnx`, `<out_dir>/report.json`(학습 반복·손실·시간·환경·ONNX 확인)을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

EXP_TEMPLATE = """# 자동 생성 파일 (scripts/yolox-train-env/finetune.py). 손으로 고치지 않는다.
from yolox.exp import Exp as BaseExp

SETTINGS = {settings!r}


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        s = SETTINGS
        self.num_classes = len(s["classes"])
        self.depth = s["depth"]
        self.width = s["width"]
        self.data_dir = s["data_dir"]
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.input_size = (s["input_size"], s["input_size"])
        self.test_size = (s["input_size"], s["input_size"])
        self.max_epoch = s["epochs"]
        self.data_num_workers = s["workers"]
        self.output_dir = s["out_dir"]
        self.exp_name = "dlp_finetune"
        self.seed = s["seed"]
        # 증강은 학습 데이터에 이미 있다 (ADR 0057): YOLOX 내부 무작위 변환을 끈다
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.shear = 0.0
        self.enable_mixup = False
"""


def load_spec(path: Path) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("format") != 1:
        raise SystemExit(f"모르는 spec 형식: {spec.get('format')}")
    if spec.get("decode_in_inference") is not False:
        raise SystemExit("decode_in_inference는 false여야 한다 (프리라벨 어댑터가 격자 디코드)")
    return spec


def import_pretrain_tools(directory: Path) -> Any:
    """scripts/pretrain/train_yolox.py (clone·내보내기·확인·환경 기록 공용 함수)."""
    sys.path.insert(0, str(directory))
    import train_yolox  # type: ignore[import-not-found]

    return train_yolox


def build_loader(spec: dict[str, Any]) -> Any:
    """COCO 학습 분할 → (이미지, 패딩된 라벨 (max_labels, 5)) 배치. 내부 무작위 변환 없음."""
    import torch
    from yolox.data import COCODataset, TrainTransform

    size = (spec["input_size"], spec["input_size"])
    dataset = COCODataset(
        data_dir=spec["data_dir"],
        json_file="train.json",
        name="train2017",
        img_size=size,
        preproc=TrainTransform(max_labels=120, flip_prob=0.0, hsv_prob=0.0),
    )
    gen = torch.Generator()
    gen.manual_seed(int(spec["seed"]))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(spec["batch_size"]),
        shuffle=True,
        num_workers=int(spec["workers"]),
        drop_last=False,
        generator=gen,
    ), len(dataset)


def load_start(model: Any, weights: str | None) -> dict[str, Any]:
    """출발 가중치에서 모양이 맞는 층만 읽는다 (클래스 수가 다른 머리는 건너뛴다)."""
    import torch
    from yolox.utils import load_ckpt

    if not weights:
        return {"start": "random"}
    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception:  # 공식 릴리스 체크포인트(호출자가 sha256 확인)는 pickle 객체를 담을 수 있다
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    before = {k: v.clone() for k, v in model.state_dict().items()}
    load_ckpt(model, state)
    after = model.state_dict()
    loaded = sum(1 for k in after if not torch.equal(before[k], after[k]))
    return {"start": "checkpoint", "tensors_loaded": loaded, "tensors_total": len(after)}


def train(spec: dict[str, Any], exp_file: Path) -> tuple[Any, dict[str, Any]]:
    """짧은 학습 반복 (선형 warmup 뒤 고정 학습률). 반환: (모델, 학습 기록)."""
    import numpy as np
    import torch
    from yolox.exp import get_exp

    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(spec["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("spec device가 cuda인데 이 환경에서 CUDA를 쓸 수 없습니다")
    exp = get_exp(str(exp_file), None)
    model = exp.get_model()
    start = load_start(model, spec.get("base_weights"))
    model.to(device).train()
    loader, n_images = build_loader(spec)
    per_epoch = max(1, math.ceil(n_images / int(spec["batch_size"])))
    total = int(spec["epochs"]) * per_epoch
    if spec.get("max_iters") is not None:
        total = min(total, int(spec["max_iters"]))
    lr = float(spec["lr_per_img"]) * int(spec["batch_size"])
    optimizer = exp.get_optimizer(int(spec["batch_size"]))
    use_amp = bool(spec["fp16"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup = int(spec["warmup_iters"])
    losses: list[float] = []
    started = time.monotonic()
    it = 0
    while it < total:
        for inps, targets, _info, _ids in loader:
            if it >= total:
                break
            for group in optimizer.param_groups:
                group["lr"] = lr * min(1.0, (it + 1) / warmup) if warmup else lr
            inps = inps.to(device).float()
            targets = targets.to(device).float()
            targets.requires_grad = False
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(inps, targets)
            loss = out["total_loss"]
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            it += 1
            print(f"반복 {it}/{total} 손실 {losses[-1]:.4f}", flush=True)
    model.eval()
    record = {
        "iterations": it,
        "iters_per_epoch": per_epoch,
        "images": n_images,
        "lr": lr,
        "device": str(device),
        "amp": use_amp,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "seconds": round(time.monotonic() - started, 3),
        **start,
    }
    return model, record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    out = Path(spec["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tools = import_pretrain_tools(Path(spec["pretrain_tools"]))
    src = tools.ensure_yolox(Path(spec["yolox_src"]).parent, spec["yolox_commit"])
    if Path(src) != Path(spec["yolox_src"]):
        # ensure_yolox는 <work>/YOLOX에 clone한다. spec 경로와 같게 둔다 (training.yaml source_dir)
        raise SystemExit(f"YOLOX 위치가 다릅니다: {src} (source_dir 끝은 YOLOX여야 한다)")
    sys.path.insert(0, str(src))
    import torch

    exp_file = out / "exp_finetune.py"
    exp_file.write_text(EXP_TEMPLATE.format(settings=spec), encoding="utf-8")
    model, record = train(spec, exp_file)
    ckpt = out / "finetuned.pth"
    torch.save({"model": model.state_dict()}, ckpt)
    onnx_path = out / "model.onnx"
    tools.export_onnx(src, exp_file, ckpt, onnx_path, int(spec["opset"]))
    check = tools.check_onnx(exp_file, ckpt, onnx_path, int(spec["input_size"]))
    report = {
        "training": record,
        "onnx_check": check,
        "environment": tools.environment(out),
        "yolox_commit": spec["yolox_commit"],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    ckpt.unlink()  # 산출물은 ONNX만 (체크포인트는 임시 디렉터리와 함께 지운다)
    print(f"완료: {onnx_path} ({check})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
