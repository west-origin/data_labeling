#!/usr/bin/env python3
"""YOLOX(Apache-2.0)를 공개 데이터 객체 탐지 데이터로 미세조정하고 ONNX로 내보낸다 (ADR 0060 1단계).

순서:
1. YOLOX를 objects.yaml `train.yolox_commit`으로 clone하고 커밋을 확인한다
   (pip 설치하지 않고 경로로 쓴다).
2. 출발 가중치(공식 릴리스 0.1.1rc0/<model>.pth)를 받아 sha256을 확인한다.
3. Exp 파일을 만든다 (클래스 수 = objects.yaml classes 수, 데이터 = build_objects.py 출력).
   확인 안 된 클래스의 배경 objectness 손실 무시는 yolox_ext.py (한계는 README).
4. YOLOX tools/train.py로 학습한다. 체크포인트는 --ckpt-dir 아래 (Colab이면 Drive 경로).
   --resume이면 마지막 체크포인트(latest_ckpt.pth)에서 이어서 한다.
5. YOLOX tools/export_onnx.py로 ONNX를 내보낸다. decode_in_inference는 끈다
   (공식 기본값, 출력은 격자
   상대값 [1, N, 5 + C]이고 프리라벨 어댑터가 strides 8·16·32로 디코드한다).
6. ONNX Runtime 출력과 PyTorch 출력을 비교하고 model_manifest.json을 쓴다.

GPU 없는 곳에서는 --cpu-smoke로 Exp 생성·데이터 로더·손실 1회 계산·역전파·ONNX 내보내기만 확인한다
(YOLOX Trainer는 CUDA를 요구한다).

실행 예:
    python train_yolox.py --config config/pretrain/objects.yaml --data /kaggle/working/objects \\
        --ckpt-dir /kaggle/working/ckpt --resume
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
import runpy
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
YOLOX_REPO = "https://github.com/Megvii-BaseDetection/YOLOX.git"
WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{model}.pth"
# 공식 릴리스 파일의 sha256 (2026-10-26 직접 받아 계산). 없는 모델은 받은 값을 기록만 한다.
WEIGHTS_SHA256 = {"yolox_s": "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"}
# YOLOX exps/default/<model>.py의 depth·width
MODEL_SIZES = {
    "yolox_s": (0.33, 0.50),
    "yolox_m": (0.67, 0.75),
    "yolox_l": (1.0, 1.0),
    "yolox_x": (1.33, 1.25),
}
EXP_NAME = "objects_pretrain"
ONNX_INPUT = "images"
ONNX_OUTPUT = "output"
DECODE_IN_INFERENCE = False  # 프리라벨 어댑터 정책 decoded_output: false와 맞춘다 (바꾸지 않는다)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    """objects.yaml + 명령줄 덮어쓰기 → 학습 설정 (manifest에 그대로 남긴다)."""
    raw = args.config.read_bytes()
    cfg = yaml.safe_load(raw)
    train = cfg["train"]
    model = str(train["model"])
    if model not in MODEL_SIZES:
        raise SystemExit(f"지원하지 않는 모델 {model} (지원: {sorted(MODEL_SIZES)})")
    epochs = int(args.epochs or train["epochs"])
    return {
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "classes": list(cfg["classes"]),
        "yolox_commit": str(train["yolox_commit"]),
        "model": model,
        "depth": MODEL_SIZES[model][0],
        "width": MODEL_SIZES[model][1],
        "input_size": int(train["input_size"]),
        "epochs": epochs,
        "batch_size": int(args.batch_size or train["batch_size"]),
        "fp16": bool(train["fp16"]) if args.fp16 is None else args.fp16,
        # YOLOX 공식 사용자 데이터 예시(exps/example/yolox_voc/yolox_voc_s.py)처럼 warmup 1
        "warmup_epochs": args.warmup_epochs,
        "no_aug_epochs": min(args.no_aug_epochs, max(0, epochs - 1)),
        "eval_interval": args.eval_interval,
        "workers": args.workers,
        "federated": not args.no_federated,
        "ignore_score": args.ignore_score,
        "data_dir": str(args.data.resolve()),
        "output_dir": str(args.ckpt_dir.resolve()),
        "exp_name": EXP_NAME,
        "fast_cocoeval": False,
    }


def preflight(need_cuda: bool) -> None:
    """필요한 모듈과 GPU를 먼저 확인한다 (몇 시간 뒤 첫 평가에서 멈추지 않게)."""
    import importlib.util

    # tensorboard·torchvision·cv2는 Kaggle·Colab 기본 환경에 있다 (requirements.txt에 넣지 않음)
    mods = [
        "torch",
        "torchvision",
        "cv2",
        "tensorboard",
        "loguru",
        "tabulate",
        "thop",
        "pycocotools",
        "onnx",
        "onnxruntime",
    ]
    missing = [m for m in mods if importlib.util.find_spec(m) is None]
    if missing:
        raise SystemExit(f"설치되지 않은 모듈: {missing} (pip install -r requirements.txt, README)")
    if need_cuda:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "GPU(CUDA)가 없습니다. YOLOX Trainer는 CUDA가 필요합니다 (--cpu-smoke 참고)"
            )


def ensure_yolox(work: Path, commit: str) -> Path:
    """YOLOX를 고정 커밋으로 clone하고 HEAD가 그 커밋인지 확인한다."""
    src = work / "YOLOX"
    if not (src / ".git").exists():
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", YOLOX_REPO, str(src)], check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != commit:
        subprocess.run(["git", "fetch", "--quiet", "origin", commit], cwd=src, check=False)
        subprocess.run(["git", "checkout", "--quiet", commit], cwd=src, check=True)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True
        ).stdout.strip()
    if head != commit:
        raise SystemExit(f"YOLOX 커밋이 다릅니다: {head} != {commit}")
    return src


def ensure_weights(ckpt_dir: Path, model: str) -> tuple[Path, str]:
    """출발 가중치를 받아 sha256을 확인한다."""
    path = ckpt_dir / "weights" / f"{model}.pth"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(WEIGHTS_URL.format(model=model), tmp)
        tmp.replace(path)
    digest = sha256_file(path)
    expected = WEIGHTS_SHA256.get(model)
    if expected is not None and digest != expected:
        raise SystemExit(f"{path}: sha256 {digest} != 공식 {expected}")
    return path, digest


EXP_TEMPLATE = """# 자동 생성 파일 (scripts/pretrain/train_yolox.py). 손으로 고치지 않는다.
import sys

sys.path.insert(0, {ext_dir!r})

from yolox.exp import Exp as BaseExp  # noqa: E402

import yolox_ext  # noqa: E402

SETTINGS = {settings!r}

if not SETTINGS["fast_cocoeval"]:
    yolox_ext.use_pycocotools_eval()


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
        self.warmup_epochs = s["warmup_epochs"]
        self.no_aug_epochs = s["no_aug_epochs"]
        self.eval_interval = s["eval_interval"]
        self.data_num_workers = s["workers"]
        self.output_dir = s["output_dir"]
        self.exp_name = s["exp_name"]
        self.save_history_ckpt = False  # latest·best·last_epoch만 (외부 저장소 용량)

    def get_model(self):
        s = SETTINGS
        return yolox_ext.build_model(self, s["federated"], s["ignore_score"])

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        return yolox_ext.build_train_loader(
            self, batch_size, is_distributed, no_aug, cache_img, SETTINGS["federated"]
        )

    def preprocess(self, inputs, targets, tsize):
        return yolox_ext.scale_targets(self, inputs, targets, tsize)
"""


def write_exp(settings: dict[str, Any], ckpt_dir: Path) -> Path:
    path = ckpt_dir / f"{EXP_NAME}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXP_TEMPLATE.format(ext_dir=str(HERE), settings=settings), encoding="utf-8")
    return path


def yolox_env(src: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src), str(HERE), env.get("PYTHONPATH", "")])
    return env


def run_training(
    src: Path, exp_file: Path, settings: dict[str, Any], weights: Path, resume: bool
) -> str:
    """YOLOX tools/train.py를 실행한다. 이어 하기면 latest_ckpt.pth, 아니면 출발 가중치에서."""
    latest = Path(settings["output_dir"]) / EXP_NAME / "latest_ckpt.pth"
    cmd = [
        sys.executable,
        "tools/train.py",
        "-f",
        str(exp_file),
        "-d",
        "1",
        "-b",
        str(settings["batch_size"]),
        "-expn",
        EXP_NAME,
    ]
    if settings["fp16"]:
        cmd.append("--fp16")
    if resume and latest.exists():
        cmd += ["--resume", "-c", str(latest)]
        mode = f"resume:{latest}"
    else:
        if resume:
            print(f"이어 할 체크포인트가 없어 처음부터 시작합니다 ({latest})")
        cmd += ["-c", str(weights)]
        mode = "finetune"
    print("실행:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=src, env=yolox_env(src), check=True)
    return mode


def pick_checkpoint(settings: dict[str, Any]) -> Path:
    """내보낼 체크포인트: 검증 AP가 가장 좋은 best_ckpt.pth, 없으면 latest_ckpt.pth."""
    out = Path(settings["output_dir"]) / EXP_NAME
    for name in ("best_ckpt.pth", "latest_ckpt.pth"):
        if (out / name).exists():
            return out / name
    raise SystemExit(f"{out}에 체크포인트가 없습니다")


def export_onnx(src: Path, exp_file: Path, ckpt: Path, onnx_path: Path, opset: int) -> None:
    """YOLOX tools/export_onnx.py를 이 프로세스에서 실행한다.

    YOLOX 0.3.0은 torch.onnx._export를 부르는데 최근 torch(2.9 등)에는 없다. 없으면 TorchScript 기반
    내보내기(dynamo=False, 0.3.0 당시와 같은 경로)로 이어 준다.
    onnxsim은 쓰지 않는다 (--no-onnxsim).
    """
    import torch

    if not hasattr(torch.onnx, "_export"):
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            torch.onnx._export = functools.partial(torch.onnx.export, dynamo=False)  # type: ignore[attr-defined]
        else:
            torch.onnx._export = torch.onnx.export  # type: ignore[attr-defined]
    argv = [
        "export_onnx.py",
        "--output-name",
        str(onnx_path),
        "-f",
        str(exp_file),
        "-c",
        str(ckpt),
        "--opset",
        str(opset),
        "--no-onnxsim",
    ]
    if DECODE_IN_INFERENCE:
        argv.append("--decode_in_inference")
    if onnx_path.exists():
        onnx_path.unlink()
    old = sys.argv
    sys.argv = argv
    try:
        runpy.run_path(str(src / "tools" / "export_onnx.py"), run_name="__main__")
    finally:
        sys.argv = old
    if not onnx_path.exists():  # export_onnx.main은 @logger.catch라 예외를 삼킨다
        raise SystemExit("ONNX 내보내기 실패 (위 로그 참고)")


def check_onnx(exp_file: Path, ckpt: Path, onnx_path: Path, size: int) -> dict[str, Any]:
    """ONNX 검사 + 같은 입력에서 PyTorch와 ONNX Runtime 출력 차이."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from yolox.exp import get_exp

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    exp = get_exp(str(exp_file), None)
    model = exp.get_model()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.eval()
    model.head.decode_in_inference = DECODE_IN_INFERENCE
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 255, size=(1, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {ONNX_INPUT: x})[0]
    return {
        "onnx_checker": "ok",
        "opset": max(o.version for o in model_proto.opset_import if o.domain in ("", "ai.onnx")),
        "output_shape": list(got.shape),
        "max_abs_diff_vs_torch": float(np.abs(ref - got).max()),
    }


def environment(out: Path) -> dict[str, Any]:
    """pip freeze(파일로 저장 + 해시), torch·CUDA·GPU 정보."""
    import torch

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout
    method = "pip freeze"
    if not freeze.strip():  # pip이 없는 환경 (uv venv 등)
        from importlib import metadata

        names = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()}
        freeze = "\n".join(sorted(names, key=str.lower)) + "\n"
        method = "importlib.metadata"
    (out / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": gpu,
        "pip_freeze_file": "pip_freeze.txt",
        "pip_freeze_method": method,
        "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "platform": detect_platform(),
    }


def detect_platform() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        return "colab"
    return "other"


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cpu_smoke(exp_file: Path, src: Path, weights: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """GPU 없이: 데이터 로더 1배치 → 손실 → 역전파 → 최적화 1회, 체크포인트 저장."""
    import torch
    from yolox.exp import get_exp
    from yolox.utils import load_ckpt

    exp = get_exp(str(exp_file), None)
    exp.data_num_workers = 0
    model = exp.get_model()
    model = load_ckpt(model, torch.load(weights, map_location="cpu")["model"])
    model.train()
    loader = exp.get_data_loader(batch_size=settings["batch_size"], is_distributed=False)
    inps, targets, _, _ = next(iter(loader))
    inps, targets = inps.float(), targets.float()
    inps, targets = exp.preprocess(inps, targets, exp.input_size)
    optimizer = exp.get_optimizer(settings["batch_size"])
    for group in optimizer.param_groups:
        group["lr"] = 1e-4
    out = model(inps, targets)
    out["total_loss"].backward()
    optimizer.step()
    ckpt_dir = Path(settings["output_dir"]) / EXP_NAME
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "start_epoch": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_ap": 0.0,
        },
        ckpt_dir / "latest_ckpt.pth",
    )
    head = model.head
    return {
        "batch_shape": list(inps.shape),
        "targets_shape": list(targets.shape),
        "losses": {
            k: float(v.detach()) if hasattr(v, "detach") else float(v) for k, v in out.items()
        },
        "ignored_anchor_fraction": getattr(head, "last_ignored_fraction", None),
    }


def write_manifest(
    args: argparse.Namespace, settings: dict[str, Any], extra: dict[str, Any], onnx_path: Path
) -> Path:
    data_manifest = args.data / "data_manifest.json"
    manifest = {
        "kind": "dlp-pretrain-objects-model",
        "created_at": datetime.now(UTC).isoformat(),
        "task": "objects",
        "data_manifest_sha256": sha256_file(data_manifest) if data_manifest.exists() else None,
        "train_settings": settings,
        "yolox": {"repo": YOLOX_REPO, "commit": settings["yolox_commit"]},
        "code": {
            "git_commit": git_commit(),
            "train_yolox_sha256": sha256_file(Path(__file__)),
            "yolox_ext_sha256": sha256_file(HERE / "yolox_ext.py"),
        },
        "onnx": {
            "file": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "decode_in_inference": DECODE_IN_INFERENCE,
            "input_name": ONNX_INPUT,
            "output_name": ONNX_OUTPUT,
            "input": (
                f"1x3x{settings['input_size']}x{settings['input_size']} float32, BGR 0~255, "
                "letterbox(비율 유지 축소, 왼쪽 위 배치, 나머지 114), 정규화 없음"
            ),
            "output": "1xNx(5+C): cx, cy, w, h (격자 상대값, strides 8·16·32로 디코드), "
            "objectness 확률, 클래스 확률 C개 (sigmoid 적용됨)",
            "classes": settings["classes"],
        },
        **extra,
    }
    path = args.out / "model_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True, help="config/pretrain/objects.yaml")
    p.add_argument("--data", type=Path, required=True, help="build_objects.py --out 디렉터리")
    p.add_argument("--ckpt-dir", type=Path, required=True, help="체크포인트 (Colab이면 Drive 경로)")
    p.add_argument(
        "--out", type=Path, default=None, help="ONNX·manifest 위치 (기본 ckpt-dir/export)"
    )
    p.add_argument("--work", type=Path, default=Path("work"), help="YOLOX clone 위치")
    p.add_argument("--resume", action="store_true", help="latest_ckpt.pth에서 이어서 학습")
    p.add_argument("--epochs", type=int, default=None, help="train.epochs 덮어쓰기")
    p.add_argument("--batch-size", type=int, default=None, help="train.batch_size 덮어쓰기")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--no-aug-epochs", type=int, default=5, help="마지막 N epoch 모자이크·mixup 끔")
    p.add_argument("--eval-interval", type=int, default=5, help="검증·best 저장 간격 (epoch)")
    p.add_argument("--workers", type=int, default=2, help="DataLoader 작업자 수")
    p.add_argument("--no-federated", action="store_true", help="확인 안 된 클래스 손실 무시를 끔")
    p.add_argument("--ignore-score", type=float, default=0.3, help="무시 기준 점수 (README)")
    p.add_argument("--opset", type=int, default=11)
    p.add_argument("--skip-train", action="store_true", help="학습 없이 내보내기만")
    p.add_argument("--cpu-smoke", action="store_true", help="GPU 없이 1회 확인 (README)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # tools/train.py는 YOLOX 디렉터리에서 돌므로 경로는 모두 절대 경로로 쓴다
    args.config, args.data = args.config.resolve(), args.data.resolve()
    args.ckpt_dir, args.work = args.ckpt_dir.resolve(), args.work.resolve()
    if args.out is not None:
        args.out = args.out.resolve()
    settings = load_settings(args)
    preflight(need_cuda=not (args.cpu_smoke or args.skip_train))
    args.out = args.out or (args.ckpt_dir / "export")
    args.out.mkdir(parents=True, exist_ok=True)
    src = ensure_yolox(args.work, settings["yolox_commit"])
    for path in (str(HERE), str(src)):  # 이 프로세스의 내보내기·확인에서 yolox·yolox_ext를 쓴다
        if path not in sys.path:
            sys.path.insert(0, path)
    weights, weights_sha = ensure_weights(args.ckpt_dir, settings["model"])
    exp_file = write_exp(settings, args.ckpt_dir)
    extra: dict[str, Any] = {
        "start_weights": {
            "url": WEIGHTS_URL.format(model=settings["model"]),
            "sha256": weights_sha,
        },
        "exp_file_sha256": sha256_file(exp_file),
    }
    if args.cpu_smoke:
        extra["cpu_smoke"] = cpu_smoke(exp_file, src, weights, settings)
        extra["training"] = {"mode": "cpu_smoke (학습 아님, 배포 금지)"}
    elif not args.skip_train:
        extra["training"] = {"mode": run_training(src, exp_file, settings, weights, args.resume)}
    ckpt = pick_checkpoint(settings)
    extra["exported_checkpoint"] = {"file": ckpt.name, "sha256": sha256_file(ckpt)}
    onnx_path = args.out / f"{EXP_NAME}_{settings['model']}.onnx"
    export_onnx(src, exp_file, ckpt, onnx_path, args.opset)
    extra["onnx_check"] = check_onnx(exp_file, ckpt, onnx_path, settings["input_size"])
    extra["environment"] = environment(args.out)
    path = write_manifest(args, settings, extra, onnx_path)
    print(f"완료: {onnx_path} ({extra['onnx_check']}) → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
