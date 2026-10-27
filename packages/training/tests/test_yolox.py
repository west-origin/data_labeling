"""YOLOX 실제 학습기·로더 시험 (ADR 0062). PyTorch 없이 도는 부분만 (실제 학습은 isolated_env 시험).

정답:
- COCO 내보내기: 위치를 아는 합성 블러본(PTS 50 ms 간격, 흰 사각형)과 키프레임을 아는 박스 예제.
  이미지 시각·박스(보간 포함)·범주 순서·건너뛴 시각·클래스 밖 예제·지운 예제(배경 이미지)·증강본
  수가 맞아야 한다.
- 학습기: 격리 환경 실행을 가짜로 바꿔 명령(잠금·일회용 환경·extra)·spec·manifest(데이터셋 버전,
  클래스 순서, 증강 해시, 잠금 파일 해시, 물려받은 분류, ONNX sha256)를 본다. GPU가 없으면 명시 옵션
  없이 거절한다.
- 로더: 손으로 부호화한 아주 작은 ONNX(출력이 상수인 Identity 그래프)가 알려진 격자 칸에 박스 하나를
  낸다. `YoloxTools`가 그것을 `trained-objects` 박스 트랙으로 만들어야 한다 (원래 픽셀 좌표).
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_fixtures.video import write_video
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_schema.dataset import Split
from dlp_schema.history import Change
from dlp_schema.labels import BoxTrackPayload
from dlp_schema.predictor import Clip, ModelUnavailableError
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_train import yolox as yolox_mod
from dlp_train.augment import AugmentationInputError, policy_digest
from dlp_train.base import LoadContext, TrainingError
from dlp_train.extract import Example, TrainingData
from dlp_train.policy import load_policy
from dlp_train.trainers import LOADERS, TRAINERS
from dlp_train.yolox import (
    ARTIFACT,
    MANIFEST_FILE,
    MODEL_FILE,
    YoloxLoader,
    YoloxTrainer,
    YoloxTrainingError,
    check_manifest,
    read_bundle,
    settings,
)
from dlp_train.yolox_data import box_at, build_coco

ROOT = Path(__file__).resolve().parents[3]
W, H = 320, 240
STEP = 50  # 합성 블러본 프레임 간격 (ms)
FRAMES = 13  # 0~600 ms


def _video(path: Path, square: tuple[int, int, int, int] = (40, 40, 32, 32)) -> Path:
    """PTS 0, 50, … 600 ms 합성 영상 (검은 화면에 흰 사각형)."""
    x, y, w, h = square

    def frames() -> Iterator[tuple[int, Any]]:
        for i in range(FRAMES):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[y : y + h, x : x + w] = 255
            yield i * STEP, img

    write_video(path, frames(), width=W, height=H)
    return path


def _box(cls: str, keyframes: list[tuple[int, float, float, bool]]) -> dict[str, Any]:
    """box_track 페이로드 (키프레임 (t, x, y, outside), 크기 32x32)."""
    return {
        "kind": "box_track",
        "entity_id": f"{cls}_1",
        "class_id": cls,
        "keyframes": [
            {"t_ms": t, "x": x, "y": y, "w": 32, "h": 32, "outside": out}
            for t, x, y, out in keyframes
        ],
    }


def _example(sid: str, lid: str, payload: dict[str, Any], split: Split, change: Change) -> Example:
    label = make_label(payload, label_id=f"{sid}-{lid}", session_id=sid, stream_id="bodycam")
    return Example(sid, split, change, label, None)


def _data() -> TrainingData:
    """학습 세션 s1 (bucket 이동 + mop + 클래스 밖 person + 지운 오탐), 검증 세션 s2."""
    examples = [
        # 0 → 100 ms 사이 x가 40 → 60으로 움직인다 (50 ms 이미지에서 보간 50)
        _example(
            "s1",
            "a",
            _box("bucket", [(0, 40, 40, False), (100, 60, 40, False)]),
            Split.TRAIN,
            "accepted",
        ),
        # 25 ms에 나타나 300 ms부터 화면 밖 (25 ms 키프레임은 블러본에 그 시각 프레임이 없다)
        _example(
            "s1",
            "b",
            _box("mop", [(25, 200, 100, False), (200, 200, 100, False), (300, 200, 100, True)]),
            Split.TRAIN,
            "corrected",
        ),
        _example("s1", "c", _box("person", [(0, 100, 100, False)]), Split.TRAIN, "added"),
        # 지운 오탐: 400 ms 이미지는 들어가지만 박스는 없다
        _example("s1", "d", _box("cup", [(400, 10, 10, False)]), Split.TRAIN, "deleted"),
        _example("s2", "a", _box("bucket", [(0, 40, 40, False)]), Split.VAL, "added"),
    ]
    return TrainingData("dv-1", "objects", examples)


@pytest.fixture
def videos(tmp_path: Path) -> Any:
    """블러본 공급자 (세션마다 같은 합성 영상, 받은 횟수 기록)."""
    calls: list[tuple[str, str]] = []

    def fetch(session_id: str, stream_id: str, work: Path) -> Path:
        calls.append((session_id, stream_id))
        work.mkdir(parents=True, exist_ok=True)
        return _video(work / f"{session_id}__{stream_id}.mp4")

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


def test_box_at_interpolates_and_respects_outside() -> None:
    """키프레임 사이 선형 보간, outside 구간·범위 밖은 None (CVAT 트랙 규칙)."""
    p = make_label(
        _box("mop", [(0, 0, 0, False), (100, 100, 0, False), (200, 0, 0, True)]),
        stream_id="bodycam",
    ).payload
    assert isinstance(p, BoxTrackPayload)
    kfs = p.keyframes
    assert box_at(kfs, 50) == (50.0, 0.0, 32.0, 32.0)
    assert box_at(kfs, 150) == (100.0, 0.0, 32.0, 32.0)  # 다음이 outside면 앞 키프레임 그대로
    assert box_at(kfs, 200) is None
    assert box_at(kfs, 250) is None


def test_build_coco_frames_boxes_and_augmentation(tmp_path: Path, videos: Any) -> None:
    """키프레임 시각의 PTS 프레임만, 보간 박스, 클래스 밖·지운 예제, 증강본 수."""
    policy = load_policy(ROOT)
    classes = load_prelabel_policy(ROOT).closed_vocab_tools.classes
    root = tmp_path / "coco"
    stats = build_coco(
        _data(),
        classes,
        videos,
        root,
        splits=policy.splits,
        augmentation=policy.augmentation,
        jpeg_quality=95,
        min_box_px=4,
        work=tmp_path / "work",
    )
    train = json.loads((root / "annotations" / "train.json").read_text())
    val = json.loads((root / "annotations" / "val.json").read_text())
    assert [c["name"] for c in train["categories"]] == list(classes)
    # s1 시각: 0, 25(프레임 없음), 100, 200, 300, 400 → 이미지 5장 + 증강 5 x copies
    originals = [im for im in train["images"] if "__a" not in im["file_name"]]
    assert sorted(im["file_name"].split("__")[2][:-4] for im in originals) == [
        f"{t:09d}" for t in (0, 100, 200, 300, 400)
    ]
    assert stats.skipped_frames == 1
    assert stats.augmented_images == 5 * policy.augmentation.copies
    assert stats.images == {"train": 5 + stats.augmented_images, "val": 1}
    assert stats.skipped_classes == {"person": 1}
    by_name = {im["file_name"]: im["id"] for im in originals}
    cat = {c["name"]: c["id"] for c in train["categories"]}

    def anns(t: int) -> list[tuple[int, list[float]]]:
        image = by_name[f"s1__bodycam__{t:09d}.jpg"]
        return sorted(
            (a["category_id"], a["bbox"]) for a in train["annotations"] if a["image_id"] == image
        )

    assert anns(0) == [(cat["bucket"], [40.0, 40.0, 32.0, 32.0])]
    # 100 ms: bucket 키프레임, mop은 25 → 200 사이 보간 (25 ms 키프레임 자체는 이미지가 없다)
    assert anns(100) == sorted(
        [(cat["bucket"], [60.0, 40.0, 32.0, 32.0]), (cat["mop"], [200.0, 100.0, 32.0, 32.0])]
    )
    # 200 ms: bucket은 마지막 키프레임(100) 뒤라 없고 mop만, 300 ms는 mop outside → 배경
    assert anns(200) == [(cat["mop"], [200.0, 100.0, 32.0, 32.0])]
    assert anns(300) == [] and anns(400) == []
    assert stats.negative_images == 2
    assert [a["bbox"] for a in val["annotations"]] == [[40.0, 40.0, 32.0, 32.0]]
    assert all((root / "train2017" / im["file_name"]).is_file() for im in train["images"])
    # 증강본 박스는 화면 안이다
    for a in train["annotations"]:
        x, y, w, h = a["bbox"]
        assert x >= 0 and y >= 0 and x + w <= W + 1e-6 and y + h <= H + 1e-6


def test_build_coco_is_deterministic_and_rejects_golden(tmp_path: Path, videos: Any) -> None:
    """같은 입력이면 같은 COCO JSON. 골든 예제가 섞이면 AugmentationInputError."""
    policy = load_policy(ROOT)
    classes = load_prelabel_policy(ROOT).closed_vocab_tools.classes
    docs: list[str] = []
    for name in ("a", "b"):
        root = tmp_path / name
        build_coco(
            _data(),
            classes,
            videos,
            root,
            splits=policy.splits,
            augmentation=policy.augmentation,
            jpeg_quality=95,
            min_box_px=4,
            work=tmp_path / f"w{name}",
        )
        docs.append((root / "annotations" / "train.json").read_text())
    assert docs[0] == docs[1]
    data = _data()
    golden = TrainingData(
        "dv-1",
        "objects",
        [
            *data.examples,
            _example("g1", "a", _box("bucket", [(0, 40, 40, False)]), Split.GOLDEN, "added"),
        ],
    )
    with pytest.raises(AugmentationInputError):
        build_coco(
            golden,
            classes,
            videos,
            tmp_path / "g",
            splits=policy.splits,
            augmentation=None,
            jpeg_quality=95,
            min_box_px=4,
            work=tmp_path / "wg",
        )


def test_settings_cpu_requires_explicit_option_and_caps() -> None:
    """GPU 없음: 명시 옵션 없으면 거절, 있으면 정책 상한으로 자르고 CPU판 extra.

    모르는 param은 거절한다.
    """
    cfg = load_policy(ROOT).yolox
    with pytest.raises(YoloxTrainingError, match="allow_cpu"):
        settings(cfg, {}, gpu=False)
    s = settings(cfg, {"allow_cpu": True, "epochs": 50, "batch_size": 64}, gpu=False)
    assert (s.device, s.extra, s.fp16) == ("cpu", cfg.env.cpu_extra, False)
    assert s.epochs == cfg.cpu.max_epochs and s.batch_size == cfg.cpu.batch_size
    assert s.max_iters == cfg.cpu.max_iters
    g = settings(cfg, {"epochs": 3}, gpu=True)
    assert (g.device, g.extra, g.epochs, g.max_iters) == ("cuda", cfg.env.cuda_extra, 3, None)
    with pytest.raises(YoloxTrainingError, match="모르는 파라미터"):
        settings(cfg, {"learning_rate": 0.1}, gpu=True)
    assert issubclass(YoloxTrainingError, TrainingError)


def test_registered_and_commit_matches_pretrain_config() -> None:
    """같은 이름으로 학습기·로더 등록, YOLOX 커밋이 1차 학습 설정과 같다.

    출발 가중치는 .pth 등록 항목이다.
    """
    import yaml

    assert "yolox" in TRAINERS and "yolox" in LOADERS
    pretrain = yaml.safe_load((ROOT / "config/pretrain/objects.yaml").read_text("utf-8"))
    policy = load_policy(ROOT)
    assert policy.yolox.yolox_commit == pretrain["train"]["yolox_commit"]
    models = yaml.safe_load((ROOT / "config/models.yaml").read_text("utf-8"))["models"]
    assert models[policy.yolox.base_model]["path"].endswith(".pth")
    assert (ROOT / policy.yolox.env.project / "uv.lock").is_file()
    assert "closed_tools" in policy.tasks["objects"].replaces


def _fake_base(root: Path, name: str) -> tuple[Path, dict[str, Any]]:
    """출발 가중치 대역 (파일을 받지 않는다). 분류는 models.yaml 항목 그대로."""
    import yaml

    spec = yaml.safe_load((root / "config/models.yaml").read_text("utf-8"))["models"][name]
    return Path("/nonexistent/base.pth"), {
        "name": name,
        "commercial": spec["commercial"],
        "sha256": spec["sha256"],
    }


def test_trainer_writes_bundle_with_manifest(
    tmp_path: Path, videos: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """격리 환경 대역으로 학습 → 묶음(model.onnx + manifest)과 재현 기록."""
    monkeypatch.setattr(yolox_mod, "base_weights", _fake_base)
    seen: dict[str, Any] = {}

    def runner(cmd: list[str], cwd: Path, timeout: int) -> str:
        seen["cmd"] = cmd
        spec = json.loads(Path(cmd[cmd.index("--spec") + 1]).read_text())
        seen["spec"] = spec
        out = Path(spec["out_dir"])
        out.mkdir(parents=True)
        (out / MODEL_FILE).write_bytes(b"fake-onnx")
        (out / "report.json").write_text(
            json.dumps(
                {
                    "training": {"iterations": 7, "final_loss": 1.5, "seconds": 2.0},
                    "onnx_check": {"onnx_checker": "ok"},
                    "environment": {"torch": "2.11.0+cpu"},
                }
            )
        )
        return ""

    policy = load_policy(ROOT)
    digest = policy_digest(policy.augmentation)
    trainer = YoloxTrainer(videos, root=ROOT, runner=runner, has_gpu=lambda: True)
    out = trainer.train(_data(), {"augmentation": digest, "epochs": 2, "jitter_px": 0.0}, tmp_path)
    assert out.artifact.name == ARTIFACT
    cmd = seen["cmd"]
    assert cmd[:2] == ["uv", "run"] and "--locked" in cmd and "--isolated" in cmd
    assert cmd[cmd.index("--extra") + 1] == policy.yolox.env.cuda_extra
    spec = seen["spec"]
    classes = list(load_prelabel_policy(ROOT).closed_vocab_tools.classes)
    assert spec["classes"] == classes and spec["decode_in_inference"] is False
    assert spec["epochs"] == 2 and spec["device"] == "cuda"
    with zipfile.ZipFile(out.artifact) as z:
        assert sorted(z.namelist()) == sorted([MODEL_FILE, MANIFEST_FILE])
        manifest = json.loads(z.read(MANIFEST_FILE))
    assert manifest["dataset_version_id"] == "dv-1"
    assert manifest["examples"]["total"] == 5
    assert manifest["classes"] == classes
    assert manifest["augmentation"] == {"digest": digest, "copies": policy.augmentation.copies}
    lock = ROOT / policy.yolox.env.project / "uv.lock"
    assert manifest["env"]["lock_sha256"] == yolox_mod.sha256_path(lock)
    assert manifest["commercial"] == "review"  # yolox_s COCO 분류를 물려받는다
    assert manifest["onnx"]["sha256"] == yolox_mod.sha256_bytes(b"fake-onnx")
    assert manifest["onnx"]["decode_in_inference"] is False
    assert out.metrics["fit/iterations"] == 7 and out.metrics["examples"] == 5


def test_trainer_refuses_without_videos_gpu_or_matching_augmentation(
    tmp_path: Path, videos: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """거절: 블러본 공급자 없음, GPU 없음(옵션 없음), 다른 증강 해시, objects 밖 과제."""
    monkeypatch.setattr(yolox_mod, "base_weights", _fake_base)

    def never(cmd: list[str], cwd: Path, timeout: int) -> str:
        raise AssertionError("격리 환경을 부르면 안 된다")

    with pytest.raises(YoloxTrainingError, match="블러본 공급자"):
        YoloxTrainer(root=ROOT, runner=never).train(_data(), {}, tmp_path)
    no_gpu = YoloxTrainer(videos, root=ROOT, runner=never, has_gpu=lambda: False)
    with pytest.raises(YoloxTrainingError, match="GPU"):
        no_gpu.train(_data(), {}, tmp_path)
    gpu = YoloxTrainer(videos, root=ROOT, runner=never, has_gpu=lambda: True)
    with pytest.raises(YoloxTrainingError, match="증강 설정 해시"):
        gpu.train(_data(), {"augmentation": "000000000000"}, tmp_path)
    with pytest.raises(YoloxTrainingError, match="objects"):
        gpu.train(TrainingData("dv-1", "hands", []), {}, tmp_path)


def test_base_weights_license_and_file_checks(tmp_path: Path) -> None:
    """출발 가중치: 목록에 없음·.pth 아님은 거절, 파일이 없으면 make models 안내."""
    with pytest.raises(YoloxTrainingError, match="없습니다"):
        yolox_mod.base_weights(ROOT, "no_such_model")
    with pytest.raises(YoloxTrainingError, match=r"\.pth"):
        yolox_mod.base_weights(ROOT, "yolox_m_coco")
    if not (ROOT / "data/models/yolox/yolox_s_coco.pth").exists():
        with pytest.raises(YoloxTrainingError, match="make models"):
            yolox_mod.base_weights(ROOT, "yolox_s_coco_pth")


# ---------------------------------------------------------------- 로더 (아주 작은 ONNX)


def _varint(n: int) -> bytes:
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
    """ValueInfoProto (float 텐서)."""
    dims = b"".join(_field(1, _int(1, d)) for d in shape)
    return _field(1, name.encode()) + _field(2, _field(1, _int(1, 1) + _field(2, dims)))


def tiny_onnx(output: np.ndarray[Any, Any], size: int) -> bytes:
    """입력 images (1, 3, size, size)를 무시하고 상수 output을 내는 ONNX (Identity 하나, opset 13).

    onnx 패키지 없이 protobuf 형식을 직접 쓴다 (ModelProto 7 graph·8 opset_import, GraphProto 1
    node· 5 initializer·11 input·12 output, TensorProto 1 dims·2 data_type·8 name·9 raw_data).
    """
    raw = np.ascontiguousarray(output, dtype="<f4")
    init = b"".join(_int(1, d) for d in raw.shape) + _int(2, 1) + _field(8, b"const")
    init += _field(9, raw.tobytes())
    node = _field(1, b"const") + _field(2, b"output") + _field(4, b"Identity")
    graph = _field(1, node) + _field(2, b"tiny") + _field(5, init)
    graph += _field(11, _value_info("images", (1, 3, size, size)))
    graph += _field(12, _value_info("output", raw.shape))
    return _int(1, 8) + _field(7, graph) + _field(8, _field(1, b"") + _int(2, 13))


def _bundle(tmp_path: Path, **manifest_overrides: Any) -> Path:
    """알려진 격자 칸에 bucket 박스 하나를 내는 ONNX 묶음.

    320x240 → 640 입력 비율 r = 2. 원래 박스 (40, 40, 32, 32) = 입력 중심 (112, 112), 크기 64 →
    stride 8 격자 (14, 14)의 행 1134, 오프셋 0, log(64/8).
    """
    sec = load_prelabel_policy(ROOT).closed_vocab_tools
    size, c = sec.input_size, len(sec.classes)
    n = sum((size // s) ** 2 for s in (8, 16, 32))
    out = np.zeros((1, n, 5 + c), dtype=np.float32)
    row = 14 * (size // 8) + 14
    out[0, row, :5] = (0.0, 0.0, math.log(8.0), math.log(8.0), 0.95)
    out[0, row, 5 + sec.classes.index("bucket")] = 0.95
    onnx = tiny_onnx(out, size)
    manifest: dict[str, Any] = {
        "kind": yolox_mod.MANIFEST_KIND,
        "format": yolox_mod.MANIFEST_FORMAT,
        "task": "objects",
        "classes": list(sec.classes),
        "input_size": size,
        "commercial": "review",
        "onnx": {
            "file": MODEL_FILE,
            "sha256": yolox_mod.sha256_bytes(onnx),
            "decode_in_inference": False,
        },
    } | manifest_overrides
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / ARTIFACT
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(MODEL_FILE, onnx)
        z.writestr(MANIFEST_FILE, json.dumps(manifest))
    return path


def test_loader_runs_tiny_onnx_through_yolox_tools(tmp_path: Path) -> None:
    """로더가 학습 ONNX를 closed_tools 어댑터 자리에 끼워 trained-objects 박스 트랙을 낸다."""
    ctx = LoadContext(now=FIXED_TIME, ontology_version="1.0.0")
    predictor = YoloxLoader(root=ROOT).load(_bundle(tmp_path), version="objects-yolox-abc", ctx=ctx)
    assert predictor.name == "trained-objects"
    assert predictor.version.startswith("yolox-objects-yolox-abc+p")
    video = _video(tmp_path / "raw.mp4")
    labels = predictor.run(Clip("s9", "bodycam", video))
    assert len(labels) == 1
    x = labels[0]
    assert x.label_id.startswith("s9-bodycam-trained-objects-")
    assert isinstance(x.payload, BoxTrackPayload)
    assert x.payload.class_id == "bucket"
    for k in x.payload.keyframes:
        assert (k.x, k.y, k.w, k.h) == pytest.approx((40, 40, 32, 32), abs=1e-3)
    assert x.provenance.model_version == predictor.version
    assert x.stream_id == "bodycam"


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"classes": ["bucket", "mop"]}, "클래스 순서"),
        ({"input_size": 320}, "입력 크기"),
        ({"commercial": "forbidden"}, "상업 사용"),
        ({"kind": "other"}, "manifest 형식"),
    ],
)
def test_loader_rejects_mismatched_manifest(
    tmp_path: Path, override: dict[str, Any], match: str
) -> None:
    """클래스 순서·입력 크기·분류·형식이 지금 정책과 맞지 않으면 ModelUnavailableError."""
    ctx = LoadContext(now=FIXED_TIME, ontology_version="1.0.0")
    with pytest.raises(ModelUnavailableError, match=match):
        YoloxLoader(root=ROOT).load(_bundle(tmp_path, **override), version="v", ctx=ctx)


def test_bundle_hash_mismatch_rejected(tmp_path: Path) -> None:
    """manifest의 ONNX sha256과 다른 묶음은 받지 않는다."""
    bad = _bundle(
        tmp_path, onnx={"file": MODEL_FILE, "sha256": "0" * 64, "decode_in_inference": False}
    )
    with pytest.raises(ModelUnavailableError, match="해시"):
        read_bundle(bad, tmp_path / "x")
    manifest, onnx = read_bundle(_bundle(tmp_path / "ok"), tmp_path / "y")
    assert onnx.is_file()
    check_manifest(manifest, load_prelabel_policy(ROOT), ("allowed", "review"))
