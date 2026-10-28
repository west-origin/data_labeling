"""YOLOX 실제 학습 스모크 (격리 환경, CPU 1~2 반복, ADR 0062). `make test-isolated`에서만 돈다.

처음 실행하면 uv가 `scripts/yolox-train-env/uv.lock`대로 PyTorch CPU판 환경(약 1 GB)을 받고, YOLOX
0.3.0을 GitHub에서 clone한다 (training.yaml yolox.source_dir). 출발 가중치는 받지 않고 무작위
초기화로 시작한다 (`base_weights` 대역): 이 시험은 성능이 아니라 "COCO 내보내기 → 격리 환경 학습
반복 → ONNX 내보내기 → 로더가 그 ONNX로 박스를 낸다"는 경로가 실제 PyTorch·YOLOX로 이어지는지만
본다.

정답: 합성 블러본(흰 사각형)과 그 위치의 박스 예제. 학습 반복 수 = 정책 cpu.max_iters, ONNX 출력
모양 = (1, 격자 칸 수, 5 + 클래스 수), PyTorch와 ONNX Runtime 출력 차이가 작다.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_schema.dataset import Split
from dlp_schema.predictor import Clip
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_train import yolox as yolox_mod
from dlp_train.base import LoadContext
from dlp_train.extract import Example, TrainingData
from dlp_train.policy import load_policy
from dlp_train.yolox import MANIFEST_FILE, YoloxLoader, YoloxTrainer

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.isolated_env


def _video(path: Path) -> Path:
    """PTS 0~600 ms(50 ms 간격) 320x240 합성 영상, (40, 40, 32, 32)에 흰 사각형."""
    import numpy as np

    from dlp_fixtures.video import write_video

    def frames() -> Any:
        for i in range(13):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            img[40:72, 40:72] = 255
            yield i * 50, img

    write_video(path, frames(), width=320, height=240)
    return path


def test_cpu_smoke_trains_exports_and_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU 스모크: 실제 격리 환경에서 2 반복 학습·ONNX 내보내기, 로더가 그 ONNX를 연다."""

    def no_base(root: Path, name: str) -> tuple[None, dict[str, Any]]:
        return None, {"name": "random-init", "commercial": "allowed"}

    real_problems = yolox_mod.base_lineage_problems

    def lineage(root: Path, name: str) -> list[str]:
        # 로더는 출발 가중치 계보를 지금 models.yaml로 다시 본다 (F-ETA-22). 무작위 초기화는 목록에
        # 없는 이 테스트 전용 이름이라 그것만 통과시킨다
        return [] if name == "random-init" else real_problems(root, name)

    monkeypatch.setattr(yolox_mod, "base_weights", no_base)
    monkeypatch.setattr(yolox_mod, "base_lineage_problems", lineage)
    payload = {
        "kind": "box_track",
        "entity_id": "bucket_1",
        "class_id": "bucket",
        "keyframes": [{"t_ms": t, "x": 40, "y": 40, "w": 32, "h": 32} for t in (0, 100, 200)],
    }
    label = make_label(payload, label_id="s1-a", session_id="s1", stream_id="bodycam")
    data = TrainingData("dv-smoke", "objects", [Example("s1", Split.TRAIN, "added", label, None)])

    def videos(session_id: str, stream_id: str, work: Path) -> Path:
        work.mkdir(parents=True, exist_ok=True)
        return _video(work / f"{session_id}__{stream_id}.mp4")

    trainer = YoloxTrainer(videos, root=ROOT, has_gpu=lambda: False)
    out = trainer.train(data, {"allow_cpu": True}, tmp_path)
    with zipfile.ZipFile(out.artifact) as z:
        manifest = json.loads(z.read(MANIFEST_FILE))
    cpu = load_policy(ROOT).yolox.cpu
    assert manifest["training"]["iterations"] == cpu.max_iters
    assert manifest["onnx_check"]["onnx_checker"] == "ok"
    sec = load_prelabel_policy(ROOT).closed_vocab_tools
    n = sum((sec.input_size // s) ** 2 for s in (8, 16, 32))
    assert manifest["onnx_check"]["output_shape"] == [1, n, 5 + len(sec.classes)]
    assert manifest["onnx_check"]["max_abs_diff_vs_torch"] < 1e-2
    ctx = LoadContext(now=FIXED_TIME, ontology_version="1.0.0")
    predictor = YoloxLoader(root=ROOT).load(out.artifact, version="objects-yolox-smoke", ctx=ctx)
    # 2 반복 학습 모델의 탐지는 기대하지 않는다. 실제 ONNX가 어댑터 안에서 돌면 된다
    labels = predictor.run(Clip("s1", "bodycam", _video(tmp_path / "raw.mp4")))
    assert all(x.label_id.startswith("s1-bodycam-trained-objects-") for x in labels)
