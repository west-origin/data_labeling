"""외부 학습 가중치 가져오기: manifest 검사, models.yaml 항목·분류, 확인 기록, 비교 버튼 조건."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dlp_console.imports import ImportStore
from dlp_console.pretrain import entries_yaml, plan_entries
from dlp_console.steps import ParamError
from dlp_models.datasets import load_pretrain_objects
from dlp_schema import repo_root

from .conftest import Api

# 공식 yolox_s.pth sha256 (config/models.yaml yolox_s_coco_pth와 같다)
YOLOX_S_SHA = "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"


def _bundle(directory: Path, **change: Any) -> tuple[Path, Path, Path]:
    """합성 묶음 (가짜 ONNX 바이트 + 학습 스크립트와 같은 형식의 manifest 두 개)."""
    classes = list(load_pretrain_objects(repo_root()).classes)
    onnx = directory / "model.onnx"
    onnx.write_bytes(b"not-a-real-onnx " * 64)
    data = {
        "kind": "dlp-pretrain-objects-data",
        "dry_run": False,
        "datasets": {"open_images_v7": {"name": "open_images_v7", "version": "v7"},
                     "lvis_v1": {"name": "lvis_v1", "version": "v1"}},
        "partial_annotations": False,
        "classes": classes,
    }  # fmt: skip
    data.update(change.pop("data", {}))
    data_path = directory / "data_manifest.json"
    data_path.write_text(json.dumps(data), "utf-8")
    model = {
        "kind": "dlp-pretrain-objects-model",
        "task": "objects",
        "data_manifest_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_settings": {"model": "yolox_s", "input_size": 640, "classes": classes},
        "start_weights": {"url": "https://example.invalid/yolox_s.pth", "sha256": YOLOX_S_SHA},
        "onnx": {
            "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
            "decode_in_inference": False,
            "classes": classes,
        },
    }
    for part in ("onnx", "train_settings", "start_weights"):
        model[part].update(change.pop(part, {}))  # type: ignore[union-attr]
    model_path = directory / "model_manifest.json"
    model_path.write_text(json.dumps(model), "utf-8")
    return onnx, model_path, data_path


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """설정은 저장소 것을 쓰고 data/models는 임시 디렉터리에 쓰는 루트."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "config").symlink_to(repo_root() / "config")
    return root


def _store(fake_root: Path, tmp_path: Path) -> tuple[ImportStore, str, Path]:
    store = ImportStore(fake_root, tmp_path / "work", "data/models/pretrain")
    import_id, path = store.new_dir()
    return store, import_id, path


def test_valid_bundle_plans_registry_entry_and_records(fake_root: Path, tmp_path: Path) -> None:
    store, import_id, path = _store(fake_root, tmp_path)
    _bundle(path)
    plan = store.check(import_id)
    assert plan.ok, [c for c in plan.checks if not c.ok]
    entry = plan.entries[plan.model_name]
    assert entry["training_datasets"] == ["lvis_v1", "open_images_v7"]
    # 출발 가중치: 같은 sha256으로 등록된 항목(yolox 학습기의 출발 가중치 이름)을 가리킨다
    assert entry["base_model"] == "yolox_s_coco_pth"
    registered = yaml.safe_load((repo_root() / "config" / "models.yaml").read_text("utf-8"))
    if "yolox_s_coco_pth" in registered["models"]:
        assert set(plan.entries) == {plan.model_name}  # 이미 있으면 새로 만들지 않는다
    assert plan.commercial == "review"  # LVIS(review)·출발 가중치(review) 중 가장 엄한 값
    assert plan.license_problems == []
    assert plan.models_patch.startswith("--- a/config/models.yaml")
    assert f"+  {plan.model_name}:" in plan.models_patch
    assert "closed_vocab_tools: " + plan.model_name in plan.activation_patch
    record = store.confirm(import_id, "tester")
    dest = fake_root / plan.dest
    assert dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == plan.onnx_sha256
    assert record["next"][0].startswith("git apply ")
    assert store.records()[0]["model_name"] == plan.model_name
    # 콘솔은 Git이 관리하는 설정을 고치지 않는다
    registry = yaml.safe_load((repo_root() / "config" / "models.yaml").read_text("utf-8"))
    assert plan.model_name not in registry["models"]


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"onnx": {"decode_in_inference": True}}, "decode_in_inference 끔"),
        ({"onnx": {"sha256": "0" * 64}}, "ONNX sha256"),
        ({"onnx": {"classes": ["rag", "mop"]}}, "클래스 순서 (ONNX = objects.yaml)"),
        ({"data": {"dry_run": True}}, "데이터 manifest 종류"),
        ({"data": {"datasets": {"arctic": {"version": "1"}}}}, "forbidden 데이터셋 없음"),
    ],
)
def test_bad_bundles_fail_and_cannot_be_confirmed(
    fake_root: Path, tmp_path: Path, change: dict[str, Any], failed: str
) -> None:
    store, import_id, path = _store(fake_root, tmp_path)
    _bundle(path, **change)
    plan = store.check(import_id)
    assert not plan.ok
    assert failed in {c.name for c in plan.checks if not c.ok}
    with pytest.raises(ParamError):
        store.confirm(import_id, "tester")


INJECTED_ARCH = (
    "yolox_s\n  injected_model:\n    path: data/models/x.onnx\n    url: https://evil.example/x\n"
    "    sha256: '00'\n    license: forbidden-license\n    commercial: allowed\n  #"
)


@pytest.mark.parametrize(
    ("change", "failed"),
    [
        ({"train_settings": {"model": "../../../../tmp/x"}}, "구조 이름 (train_settings.model)"),
        ({"train_settings": {"model": INJECTED_ARCH}}, "구조 이름 (train_settings.model)"),
        ({"train_settings": {"model": ["yolox_s"]}}, "구조 이름 (train_settings.model)"),
        ({"start_weights": {"url": "https://evil.example/a b\nc"}}, "출발 가중치 형식"),
        ({"start_weights": {"url": "file:///etc/passwd"}}, "출발 가중치 형식"),
        ({"start_weights": {"sha256": "ab\n" * 20}}, "출발 가중치 형식"),
    ],
)
def test_manifest_strings_cannot_escape_or_inject(
    fake_root: Path, tmp_path: Path, change: dict[str, Any], failed: str
) -> None:
    """F-COE-77: 구조 이름·출발 가중치 문자열은 경로 이탈·models.yaml 주입을 만들지 못한다."""
    store, import_id, path = _store(fake_root, tmp_path)
    _bundle(path, **change)
    plan = store.check(import_id)
    assert not plan.ok
    assert failed in {c.name for c in plan.checks if not c.ok}
    assert "injected_model" not in plan.models_patch
    with pytest.raises(ParamError):
        store.confirm(import_id, "tester")
    assert not (tmp_path / "tmp").exists()


@pytest.mark.parametrize("arch", ["../../../../tmp/x", INJECTED_ARCH, "/abs", "Yolox"])
def test_plan_entries_rejects_bad_arch(arch: str) -> None:
    """함수 수준 재현 (F-COE-77): 이전에는 dest가 저장소 밖이고 위반 목록이 비었다."""
    model = {"train_settings": {"model": arch}, "start_weights": {"sha256": YOLOX_S_SHA}}
    data = {"datasets": {"open_images_v7": {"version": "v7"}}}
    name, dest, entries, _, problems = plan_entries(
        repo_root(), model, data, "a" * 64, "data/models/pretrain"
    )
    assert (name, dest, entries) == ("", "", {})
    assert problems


def test_plan_entries_rejects_models_dir_outside_root() -> None:
    model = {"train_settings": {"model": "yolox_s"}, "start_weights": {"sha256": YOLOX_S_SHA}}
    data = {"datasets": {"open_images_v7": {"version": "v7"}}}
    *_, problems = plan_entries(repo_root(), model, data, "a" * 64, "../outside")
    assert problems


def test_confirm_rechecks_tampered_plan_dest(fake_root: Path, tmp_path: Path) -> None:
    """plan.json의 dest를 바꿔도 confirm은 models_dir 밖에 쓰지 않는다."""
    store, import_id, path = _store(fake_root, tmp_path)
    _bundle(path)
    assert store.check(import_id).ok
    plan = json.loads((path / "plan.json").read_text("utf-8"))
    plan["dest"] = "../../escape.onnx"
    (path / "plan.json").write_text(json.dumps(plan), "utf-8")
    with pytest.raises(ParamError):
        store.confirm(import_id, "tester")
    assert not (tmp_path / "escape.onnx").exists()


def test_entries_yaml_note_cannot_break_comment() -> None:
    text = entries_yaml({"m": {"path": "p", "_note": "a\n  evil: 1\n#"}})
    assert yaml.safe_load("models:\n" + text) == {"models": {"m": {"path": "p"}}}


def test_compare_is_disabled_without_trainer(client: Api, tmp_path: Path) -> None:
    files = {}
    paths = _bundle(tmp_path)
    for name, p in zip(("onnx", "model_manifest", "data_manifest"), paths, strict=True):
        files[name] = (p.name, p.read_bytes(), "application/octet-stream")
    plan = client.post("/api/pretrain/check", files=files).json()
    assert plan["ok"] is True
    res = client.post(f"/api/pretrain/{plan['import_id']}/compare", json={"dataset_version": "d1"})
    assert res.status_code == 400 and "F-COE-74" in res.json()["detail"]
    data = client.get("/api/pretrain").json()
    assert data["gate_passed"] is False and data["compare_trainer"] is None
