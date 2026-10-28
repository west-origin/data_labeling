"""YOLOX 고정 클래스 물건 탐지기 실제 학습기·로더 (학습기 이름 `yolox`, ADR 0062).

학습 (`YoloxTrainer.train`, 재학습 루프 `loop.run_training_job`의 3단계):
1. 설정: 정책 `training.yaml yolox` 위에 과제 params(`PARAM_KEYS`)를 덮는다. 클래스 순서·입력 크기는
   `prelabel.yaml closed_vocab_tools`(로더가 끼우는 어댑터의 정책)다.
2. 출발 가중치: `config/models.yaml`의 `base_model`(.pth)을 `registry.resolve`로 해시 확인한다. 상업
   사용 분류가 허용 목록(`accept`) 밖이거나 공개 데이터셋 분류 위반(`dlp_models.datasets`)이면
   거절한다. 학습 모델은 그 분류를 물려받는다 (우리 데이터셋 버전은 더 엄하게 만들지 않는다, ADR
   0058).
3. 장치: `nvidia-smi`로 GPU가 보이면 CUDA판 환경, 아니면 CPU. CPU는 정책 `cpu.allowed` 또는
   `--param allow_cpu=true`가 있어야 하고, 정책 상한(epoch·반복·배치)으로 자른다 (매우 느림 경고).
4. 데이터: `yolox_data.build_coco` (블러본 키프레임 프레임, ADR 0057 증강).
5. 학습·내보내기: 격리 환경(`env.project`, uv.lock, `--locked --isolated`)에서 `env.script`를
   subprocess로 돌린다. 본 가상 환경에는 PyTorch가 없다. ONNX는 decode_in_inference false다.
6. 산출물: `yolox-model.zip` 하나 (model.onnx + manifest.json). manifest에 데이터셋 버전·예제
   수·클래스 순서·학습 설정·증강 해시·환경 잠금 파일 해시·출발 가중치·상업 사용 분류·ONNX sha256을
   남긴다.

로드 (`YoloxLoader.load`, 골든셋 평가와 배포 후 프리라벨 `deployed.deployed_predictors`):
- 묶음을 풀어 manifest 형식·ONNX sha256·클래스 순서·입력 크기(지금 prelabel 정책과 같아야 함)·상업
  사용 분류(지금 허용 목록 안)와 출발 가중치 계보의 지금 분류(`check_base_lineage`: models.yaml에서
  forbidden이 되었거나 빠졌으면 거절)를 확인한다. 하나라도 어긋나면 `ModelUnavailableError`
  (배포 단계는 기본 어댑터로 넘어간다).
- `dlp_prelabel.adapters.yolox_tools.YoloxTools`의 탐지기 자리에 그 ONNX(`dlp_models.yolox.Yolox`)를
  끼운 Predictor를 돌려준다. 이름은 `trained-<과제>`(라벨 ID 접두사)이고, 추적·거르기·착용자 몸
  억제는 그 어댑터 그대로다. 골든셋 예측은 메모리에서만 쓴다 (루프가 DB에 쓰지 않는다).

CPU stub: 학습은 `runner`·`has_gpu`를, 로드는 `detector`를 주입해 PyTorch 없이 시험한다 (테스트).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from dlp_models.datasets import dataset_violations, load_datasets
from dlp_models.registry import load_registry, resolve
from dlp_models.yolox import Yolox
from dlp_prelabel.adapters.yolox_tools import ToolDetector, YoloxTools
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_schema import repo_root
from dlp_schema.predictor import ModelUnavailableError, Predictor
from dlp_train.augment import policy_digest
from dlp_train.base import LoadContext, TrainingError, TrainOutput
from dlp_train.extract import TrainingData
from dlp_train.policy import TrainingPolicy, YoloxTrainingPolicy, load_policy
from dlp_train.yolox_data import BlurredVideo, CocoStats, build_coco

NAME = "yolox"  # 등록 이름 (TRAINERS·LOADERS 키, model_versions.trainer)
ARTIFACT = "yolox-model.zip"  # 산출물 파일 이름 (루프가 sha256을 재고 올린다)
MODEL_FILE = "model.onnx"
MANIFEST_FILE = "manifest.json"
MANIFEST_KIND = "dlp-train-yolox-model"
MANIFEST_FORMAT = 1
DECODE_IN_INFERENCE = False  # 격자 상대값 출력 (yolox_tools가 strides 8·16·32로 디코드)
# 묶음 안 파일 시각 (같은 내용이면 같은 바이트)
_ZIP_TIME = (2020, 1, 1, 0, 0, 0)
# 과제 params로 덮을 수 있는 키 → 정책 yolox 절의 필드 (allow_cpu·max_iters는 학습기 옵션)
PARAM_KEYS = frozenset(
    {
        "base_model", "epochs", "batch_size", "lr_per_img", "warmup_iters", "fp16", "seed",
        "allow_cpu", "max_iters",
    }
)  # fmt: skip
# 루프가 넣는 파라미터 (학습기 설정이 아니다)
_LOOP_KEYS = frozenset({"augmentation"})
# YOLOX exps/default/<model>.py의 depth·width (scripts/pretrain/train_yolox.py MODEL_SIZES와 같다)
MODEL_SIZES = {
    "yolox_s": (0.33, 0.50),
    "yolox_m": (0.67, 0.75),
    "yolox_l": (1.0, 1.0),
    "yolox_x": (1.33, 1.25),
}

# (명령, 작업 디렉터리, 시간 상한 초) → 표준 출력. 실패하면 TrainingError를 낸다
Runner = Callable[[list[str], Path, int], str]


class YoloxTrainingError(TrainingError):
    """YOLOX 학습을 진행할 수 없음 (공급자 없음, GPU 없음, 라이선스 위반, 환경 실패 등)."""


def sha256_bytes(data: bytes) -> str:
    """바이트의 sha256 (16진 64자)."""
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    """파일 sha256 (1 MiB씩)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nvidia_gpu() -> bool:
    """`nvidia-smi -L`이 GPU를 보이면 참 (드라이버가 없거나 실패하면 거짓)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        proc = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "GPU" in proc.stdout


def run_subprocess(cmd: list[str], cwd: Path, timeout: int) -> str:
    """명령을 돌려 표준 출력을 돌려준다.

    Raises:
        YoloxTrainingError: 실패(표준 오류 끝부분 포함)·시간 초과.
    """
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )  # fmt: skip
    except subprocess.TimeoutExpired as exc:
        raise YoloxTrainingError(f"YOLOX 학습 환경이 {timeout}초 안에 끝나지 않았습니다") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-30:])
        raise YoloxTrainingError(f"YOLOX 학습 환경 실패 (종료 {proc.returncode}):\n{tail}")
    return proc.stdout


@dataclass(frozen=True)
class Settings:
    """한 번의 학습 설정 (정책 + params, 장치 반영)."""

    base_model: str
    model: str
    epochs: int
    batch_size: int
    lr_per_img: float
    warmup_iters: int
    fp16: bool
    seed: int
    max_iters: int | None  # 반복 수 상한 (CPU면 정책 상한)
    device: str  # cuda | cpu
    extra: str  # 환경 extra (PyTorch 판)


def _flag(params: Mapping[str, Any], key: str, default: bool) -> bool:
    """불리언 param 하나. JSON 불리언(`true`·`false`)만 받는다 (F-ETA-24).

    `--param allow_cpu=False`·`no`는 JSON 해석에 실패해 문자열이 되고, `bool("False")`는 참이다.
    문자열·숫자를 참거짓으로 짐작하지 않고 거절한다.

    Raises:
        YoloxTrainingError: 값이 불리언이 아닐 때.
    """
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise YoloxTrainingError(
            f"yolox 파라미터 {key}는 불리언이어야 합니다 (받은 값 {value!r}). "
            f"--param {key}=true 또는 {key}=false로 쓴다"
        )
    return value


def settings(policy: YoloxTrainingPolicy, params: Mapping[str, Any], gpu: bool) -> Settings:
    """정책 yolox 절 + 과제 params → 설정.

    Raises:
        YoloxTrainingError: 모르는 param, 불리언이 아닌 allow_cpu·fp16, GPU가 없는데 CPU 허용이
            없을 때.
    """
    unknown = sorted(set(params) - PARAM_KEYS - _LOOP_KEYS - {"jitter_px"})
    if unknown:
        raise YoloxTrainingError(
            f"yolox 학습기가 모르는 파라미터: {unknown} (허용 {sorted(PARAM_KEYS)})"
        )
    p = dict(params)
    allow_cpu = _flag(p, "allow_cpu", False)  # GPU가 있어도 형식은 먼저 본다
    max_iters = int(p["max_iters"]) if p.get("max_iters") is not None else None
    s = Settings(
        base_model=str(p.get("base_model", policy.base_model)),
        model=policy.model,
        epochs=int(p.get("epochs", policy.epochs)),
        batch_size=int(p.get("batch_size", policy.batch_size)),
        lr_per_img=float(p.get("lr_per_img", policy.lr_per_img)),
        warmup_iters=int(p.get("warmup_iters", policy.warmup_iters)),
        fp16=_flag(p, "fp16", policy.fp16),
        seed=int(p.get("seed", policy.seed)),
        max_iters=max_iters,
        device="cuda",
        extra=policy.env.cuda_extra,
    )
    if gpu:
        return s
    cpu = policy.cpu
    if not (cpu.allowed or allow_cpu):
        raise YoloxTrainingError(
            "GPU(nvidia-smi)가 없습니다. YOLOX 학습은 GPU 학습 서버에서 돌린다. "
            f"CPU 스모크(매우 느림, epoch {cpu.max_epochs}·반복 {cpu.max_iters} 상한)는 "
            "--param allow_cpu=true로 명시한다"
        )
    print(
        "경고: GPU가 없어 CPU로 학습합니다 (매우 느림). "
        f"epoch {cpu.max_epochs}·반복 {cpu.max_iters}·배치 {cpu.batch_size} 상한으로 자릅니다. "
        "이 모델은 스모크용이며 성능을 기대하지 않습니다",
        file=sys.stderr,
    )
    iters = cpu.max_iters if max_iters is None else min(max_iters, cpu.max_iters)
    return replace(
        s,
        epochs=min(s.epochs, cpu.max_epochs),
        batch_size=min(s.batch_size, cpu.batch_size),
        fp16=False,
        max_iters=iters,
        device="cpu",
        extra=policy.env.cpu_extra,
    )


def base_lineage_problems(root: Path, name: str) -> list[str]:
    """출발 가중치 `name`과 그 계보(`base_model` 사슬)의 지금 분류 위반 (사람이 읽는 메시지).

    사슬의 가중치마다 models.yaml에 있고 상업 사용 분류가 허용 목록(`accept`) 안이어야 하며,
    공개 데이터셋 분류 위반(`dataset_violations`)이 없어야 한다. 위반 메시지는 정확히 그 가중치
    이름(`<이름>: `)으로 고른다 (이름이 앞부분만 같은 다른 가중치의 위반을 섞지 않는다). 학습할
    때(`base_weights`)와 로드할 때(`check_base_lineage`) 같은 기준을 쓴다 (F-ETA-22).
    """
    registry = load_registry(root)
    by_model: dict[str, list[str]] = {}
    for v in dataset_violations(registry, load_datasets(root)):
        by_model.setdefault(v.split(": ", 1)[0], []).append(v)
    problems: list[str] = []
    seen: set[str] = set()
    current: str | None = name
    while current is not None and current not in seen:
        seen.add(current)
        problems += registry.violations({"training.yolox.base_model": current})
        problems += by_model.get(current, [])
        spec = registry.models.get(current)
        current = spec.base_model if spec is not None else None
    if current is not None:
        problems.append(f"{name}: 출발 가중치 계보가 순환합니다 ({current})")
    return problems


def base_weights(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """출발 가중치 경로와 manifest에 남길 계보 (이름·버전·sha256·라이선스·분류).

    Raises:
        YoloxTrainingError: 목록에 없음·.pth가 아님·허용 분류 밖·공개 데이터셋 분류 위반·파일 없음.
    """
    registry = load_registry(root)
    spec = registry.models.get(name)
    if spec is None:
        raise YoloxTrainingError(f"출발 가중치 {name}이 config/models.yaml에 없습니다")
    if not spec.path.endswith(".pth"):
        raise YoloxTrainingError(
            f"출발 가중치 {name}은 PyTorch .pth 체크포인트여야 합니다 ({spec.path})"
        )
    problems = base_lineage_problems(root, name)
    if problems:
        raise YoloxTrainingError(f"출발 가중치 라이선스 위반: {problems}")
    try:
        path, version = resolve(root, name)
    except ModelUnavailableError as exc:
        raise YoloxTrainingError(str(exc)) from exc
    lineage = {
        "name": name,
        "version": version,
        "sha256": sha256_path(path),
        "license": spec.license,
        "training_data": list(spec.training_data),
        "training_datasets": list(spec.training_datasets),
        "base_model": spec.base_model,
        "commercial": spec.commercial,
    }
    return path, lineage


def env_command(root: Path, policy: YoloxTrainingPolicy, extra: str, spec: Path) -> list[str]:
    """격리 환경 실행 명령 (`--locked --isolated`: 잠금 파일 그대로, 실행마다 일회용 환경)."""
    return [
        "uv", "run", "--project", str(root / policy.env.project), "--locked", "--isolated",
        "--extra", extra, "--python", policy.env.python, "python", str(root / policy.env.script),
        "--spec", str(spec),
    ]  # fmt: skip


def env_lineage(root: Path, policy: YoloxTrainingPolicy, extra: str) -> dict[str, Any]:
    """환경 재현 기록: 잠금 파일·pyproject·학습 스크립트·공용 YOLOX 도구의 sha256."""
    project = root / policy.env.project
    return {
        "project": policy.env.project,
        "extra": extra,
        "python": policy.env.python,
        "lock_sha256": sha256_path(project / "uv.lock"),
        "pyproject_sha256": sha256_path(project / "pyproject.toml"),
        "script_sha256": sha256_path(root / policy.env.script),
        "pretrain_tools_sha256": sha256_path(root / "scripts" / "pretrain" / "train_yolox.py"),
    }


def _write_bundle(path: Path, onnx: Path, manifest: dict[str, Any]) -> None:
    """model.onnx + manifest.json을 시각 고정 zip(무압축)으로 쓴다."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in (
            (MODEL_FILE, onnx.read_bytes()),
            (
                MANIFEST_FILE,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode(),
            ),
        ):
            z.writestr(zipfile.ZipInfo(name, _ZIP_TIME), data)


class YoloxTrainer:
    """`yolox` 학습기. 모듈 docstring의 학습 1~6단계."""

    name = NAME

    def __init__(
        self,
        videos: BlurredVideo | None = None,
        *,
        root: Path | None = None,
        runner: Runner = run_subprocess,
        has_gpu: Callable[[], bool] = nvidia_gpu,
    ) -> None:
        """videos: 블러본 공급자 (`dlp train run`이 넘긴다. None이면 학습을 거절한다).

        root: 저장소 루트 (기본 `repo_root()`).
        runner·has_gpu: 격리 환경 실행·GPU 확인 (테스트 주입).
        """
        self.videos, self.root, self.runner, self.has_gpu = videos, root, runner, has_gpu

    def _policies(self, params: Mapping[str, Any]) -> tuple[Path, TrainingPolicy, PrelabelPolicy]:
        """저장소 루트·학습 정책·프리라벨 정책. 증강 해시가 지금 정책과 다르면 거절한다."""
        root = self.root or repo_root()
        policy = load_policy(root)
        digest = params.get("augmentation")
        if digest is not None and digest != policy_digest(policy.augmentation):
            raise YoloxTrainingError(
                f"증강 설정 해시 {digest}가 지금 training.yaml augmentation과 다릅니다"
            )
        return root, policy, load_prelabel_policy(root)

    def train(self, data: TrainingData, params: Mapping[str, Any], out_dir: Path) -> TrainOutput:
        """학습해 `out_dir/yolox-model.zip`을 쓴다.

        Raises:
            YoloxTrainingError: 공급자 없음, 과제가 objects가 아님, 설정·GPU·라이선스 문제,
                학습 이미지 없음, 환경 실패, 산출물 없음.
        """
        if self.videos is None:
            raise YoloxTrainingError(
                "블러본 공급자가 없습니다 (dlp train run이 YoloxTrainer(videos=...)로 넘긴다)"
            )
        if data.task != "objects":
            raise YoloxTrainingError(f"yolox 학습기는 objects 과제만 학습합니다 ({data.task})")
        root, policy, prelabel = self._policies(params)
        cfg = policy.yolox
        run = settings(cfg, params, self.has_gpu())
        weights, base = base_weights(root, run.base_model)
        sec = prelabel.closed_vocab_tools
        work = out_dir / "yolox"
        coco = work / "coco"
        aug = policy.augmentation if params.get("augmentation") is not None else None
        stats = build_coco(
            data, sec.classes, self.videos, coco, splits=policy.splits, augmentation=aug,
            jpeg_quality=cfg.jpeg_quality, min_box_px=cfg.min_box_px, work=work / "blurred",
        )  # fmt: skip
        if not stats.images.get("train"):
            raise YoloxTrainingError("학습 분할 이미지가 없습니다 (블러본 프레임·박스 예제 확인)")
        spec = self._spec(root, cfg, run, sec.classes, sec.input_size, weights, coco, work / "out")
        spec_path = work / "spec.json"
        spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        self.runner(env_command(root, cfg, run.extra, spec_path), root, cfg.timeout_s)
        onnx, report = work / "out" / MODEL_FILE, work / "out" / "report.json"
        if not onnx.is_file() or not report.is_file():
            raise YoloxTrainingError("학습 환경이 model.onnx·report.json을 남기지 않았습니다")
        env_report: dict[str, Any] = json.loads(report.read_text("utf-8"))
        env = env_lineage(root, cfg, run.extra)
        manifest = self._manifest(
            data,
            params,
            policy,
            run,
            sec.classes,
            sec.input_size,
            base,
            stats,
            env,
            onnx,
            env_report,
        )
        artifact = out_dir / ARTIFACT
        _write_bundle(artifact, onnx, manifest)
        return TrainOutput(artifact, _metrics(data, stats, env_report, run))

    def _spec(
        self,
        root: Path,
        cfg: YoloxTrainingPolicy,
        run: Settings,
        classes: tuple[str, ...],
        input_size: int,
        weights: Path | None,
        coco: Path,
        out: Path,
    ) -> dict[str, Any]:
        """격리 환경 학습 스크립트 입력 (`scripts/yolox-train-env/finetune.py --spec`)."""
        depth, width = MODEL_SIZES[run.model]
        return {
            "format": 1, "classes": list(classes), "input_size": input_size,
            "model": run.model, "depth": depth, "width": width,
            "data_dir": str(coco), "out_dir": str(out),
            "base_weights": str(weights) if weights is not None else None,
            "yolox_src": str(root / cfg.source_dir), "yolox_commit": cfg.yolox_commit,
            "pretrain_tools": str(root / "scripts" / "pretrain"),
            "epochs": run.epochs, "max_iters": run.max_iters,
            "batch_size": run.batch_size, "lr_per_img": run.lr_per_img,
            "warmup_iters": run.warmup_iters, "fp16": run.fp16, "workers": cfg.workers,
            "seed": run.seed, "device": run.device, "opset": cfg.opset,
            "decode_in_inference": DECODE_IN_INFERENCE,
        }  # fmt: skip

    def _manifest(
        self,
        data: TrainingData,
        params: Mapping[str, Any],
        policy: TrainingPolicy,
        run: Settings,
        classes: tuple[str, ...],
        input_size: int,
        base: dict[str, Any],
        stats: CocoStats,
        env: dict[str, Any],
        onnx: Path,
        report: dict[str, Any],
    ) -> dict[str, Any]:
        """산출물 manifest (재현·계보·라이선스)."""
        augmentation = params.get("augmentation")
        return {
            "kind": MANIFEST_KIND,
            "format": MANIFEST_FORMAT,
            "task": data.task,
            "dataset_version_id": data.dataset_version_id,
            "examples": {"total": len(data.examples), **data.counts()},
            "classes": list(classes),
            "input_size": input_size,
            "coco": stats.to_json(),
            "train_settings": asdict(run),
            "params_sha256": sha256_bytes(json.dumps(dict(params), sort_keys=True).encode()),
            "yolox_policy_sha256": sha256_bytes(
                json.dumps(policy.yolox.model_dump(mode="json"), sort_keys=True).encode()
            ),
            "augmentation": {
                "digest": augmentation,
                "copies": policy.augmentation.copies if augmentation is not None else 0,
            },
            "yolox": {"commit": policy.yolox.yolox_commit},
            "env": env,
            "environment": report.get("environment", {}),
            "training": report.get("training", {}),
            "onnx_check": report.get("onnx_check", {}),
            "base_model": base,
            # 학습 모델의 상업 사용 분류 = 출발 가중치 분류 (우리 데이터셋 버전은 allowed, ADR 0058)
            "commercial": base["commercial"],
            "training_data": f"dataset version {data.dataset_version_id} (train·val 블러본)",
            "onnx": {
                "file": MODEL_FILE,
                "sha256": sha256_path(onnx),
                "decode_in_inference": DECODE_IN_INFERENCE,
                "input_name": "images",
                "output_name": "output",
            },
        }


def _metrics(
    data: TrainingData, stats: CocoStats, report: Mapping[str, Any], run: Settings
) -> dict[str, float]:
    """MLflow train/ 지표 (예제·이미지·손실·반복)."""
    out: dict[str, float] = {"examples": float(len(data.examples))}
    out |= {f"examples/{k}": float(v) for k, v in data.counts().items()}
    out |= {f"images/{k}": float(v) for k, v in stats.images.items()}
    out |= {f"annotations/{k}": float(v) for k, v in stats.annotations.items()}
    out["images/augmented"] = float(stats.augmented_images)
    out["frames/skipped"] = float(stats.skipped_frames)
    out["device/cpu"] = float(run.device == "cpu")
    training: Mapping[str, Any] = report.get("training", {})
    for key in ("iterations", "final_loss", "seconds"):
        value = training.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            out[f"fit/{key}"] = float(value)
    return out


# ---------------------------------------------------------------- 로더


def read_bundle(artifact: Path, dest: Path) -> tuple[dict[str, Any], Path]:
    """묶음을 풀어 (manifest, ONNX 경로). ONNX sha256이 manifest와 다르면 ModelUnavailableError."""
    try:
        with zipfile.ZipFile(artifact) as z:
            manifest: dict[str, Any] = json.loads(z.read(MANIFEST_FILE))
            onnx_bytes = z.read(MODEL_FILE)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(
            f"{artifact.name}: yolox 산출물 묶음이 아닙니다 ({exc})"
        ) from exc
    if manifest.get("kind") != MANIFEST_KIND or manifest.get("format") != MANIFEST_FORMAT:
        raise ModelUnavailableError(f"{artifact.name}: 모르는 manifest 형식")
    if sha256_bytes(onnx_bytes) != manifest.get("onnx", {}).get("sha256"):
        raise ModelUnavailableError(f"{artifact.name}: ONNX 해시가 manifest와 다릅니다")
    dest.mkdir(parents=True, exist_ok=True)
    onnx = dest / MODEL_FILE
    onnx.write_bytes(onnx_bytes)
    return manifest, onnx


def check_manifest(
    manifest: Mapping[str, Any], prelabel: PrelabelPolicy, accept: tuple[str, ...]
) -> None:
    """지금 정책으로 쓸 수 있는 산출물인지 본다. 아니면 ModelUnavailableError.

    클래스 순서·입력 크기가 prelabel.yaml closed_vocab_tools와 같아야 하고(출력 열의 이름이다),
    decode_in_inference가 거짓이며, 물려받은 상업 사용 분류가 지금 허용 목록(models.yaml accept)
    안이어야 한다.
    """
    sec = prelabel.closed_vocab_tools
    if tuple(manifest.get("classes", ())) != sec.classes:
        raise ModelUnavailableError(
            "학습 모델의 클래스 순서가 지금 prelabel.yaml closed_vocab_tools.classes와 "
            "다릅니다 (다시 학습)"
        )
    if manifest.get("input_size") != sec.input_size:
        raise ModelUnavailableError(
            "학습 모델 입력 크기가 closed_vocab_tools.input_size와 다릅니다"
        )
    if manifest.get("onnx", {}).get("decode_in_inference") is not False:
        raise ModelUnavailableError("decode_in_inference로 내보낸 학습 모델은 받지 않습니다")
    if manifest.get("commercial") not in accept:
        raise ModelUnavailableError(
            f"학습 모델의 상업 사용 분류 {manifest.get('commercial')}가 "
            f"허용 목록 {list(accept)} 밖입니다"
        )


def check_base_lineage(root: Path, manifest: Mapping[str, Any]) -> None:
    """manifest의 출발 가중치가 지금 models.yaml에서도 쓸 수 있는지. 아니면 ModelUnavailableError.

    manifest의 분류는 학습 당시 값이다. 그 뒤 출발 가중치(또는 그 계보의 가중치·학습 데이터셋)가
    forbidden으로 재분류되거나 목록에서 빠지면 그 가중치로 학습한 모델도 쓰지 않는다 (F-ETA-22).
    출발 가중치 이름이 없는 manifest는 계보를 확인할 수 없어 받지 않는다.
    """
    base: Any = manifest.get("base_model")
    name: Any = None
    if isinstance(base, dict):
        name = cast(dict[str, Any], base).get("name")
    if not isinstance(name, str) or not name:
        raise ModelUnavailableError(
            "학습 모델 manifest에 출발 가중치 이름이 없습니다 (계보 확인 불가)"
        )
    problems = base_lineage_problems(root, name)
    if problems:
        raise ModelUnavailableError(
            f"학습 모델의 출발 가중치 {name}을 지금 분류로 쓸 수 없습니다: {problems}"
        )


# (ONNX 경로, 클래스 수, closed_vocab_tools 정책) → 탐지기. 테스트는 가짜를 넣는다
DetectorFactory = Callable[[Path, int, PrelabelPolicy], ToolDetector]


def onnx_detector(onnx: Path, num_classes: int, policy: PrelabelPolicy) -> ToolDetector:
    """학습 ONNX를 여는 실제 탐지기 (`dlp_models.yolox.Yolox`, 격자 디코드)."""
    sec = policy.closed_vocab_tools
    return Yolox(
        onnx,
        num_classes,
        input_size=sec.input_size,
        decoded=DECODE_IN_INFERENCE,
        nms_iou=sec.nms_iou,
    )


class YoloxLoader:
    """`yolox` 산출물 로더. 모듈 docstring의 로드 절."""

    def __init__(self, root: Path | None = None, detector: DetectorFactory = onnx_detector) -> None:
        """root: 저장소 루트 (기본 `repo_root()`). detector: 탐지기 생성 (테스트 주입)."""
        self.root, self.detector = root, detector

    def load(self, artifact: Path, *, version: str, ctx: LoadContext) -> Predictor:
        """산출물 → `trained-<과제>` Predictor (`YoloxTools`에 학습 ONNX를 끼움).

        Raises:
            ModelUnavailableError: 묶음·해시·클래스·입력 크기·분류가 맞지 않을 때.
        """
        root = self.root or repo_root()
        manifest, onnx = read_bundle(artifact, artifact.parent / f"{artifact.stem}-{version}")
        prelabel = load_prelabel_policy(root)
        check_manifest(manifest, prelabel, load_registry(root).accept)
        check_base_lineage(root, manifest)
        n = len(prelabel.closed_vocab_tools.classes)
        factory = self.detector

        def open_model() -> ToolDetector:
            return factory(onnx, n, prelabel)

        predictor = YoloxTools(
            prelabel, open_model, version, ontology_version=ctx.ontology_version, now=ctx.now
        )
        # 기본 어댑터(closed_tools)와 라벨 ID 접두사가 겹치지 않게 한다 (replaces로 대신한다)
        predictor.name = f"trained-{manifest['task']}"
        return predictor
