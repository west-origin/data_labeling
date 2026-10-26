"""공개 데이터셋 목록·1차 학습 설정 검사 (`dlp_models.datasets`, ADR 0060).

정답: 손으로 만든 작은 레지스트리. 가중치 분류가 학습 데이터·출발 가중치 중 가장 엄한 분류보다
느슨하면 위반이고, forbidden 데이터셋은 쓰기만 해도 위반이다.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dlp_models.datasets import (
    DatasetRegistry,
    ExternalDataset,
    PretrainObjects,
    dataset_violations,
    pretrain_violations,
)
from dlp_models.registry import ModelRegistry


def _ds(commercial: str, **kw: Any) -> dict[str, Any]:
    """데이터셋 항목 하나."""
    return {"version": "v1", "homepage": "https://example.org", "annotation_license": "CC BY 4.0",
            "media_license": "CC BY 2.0", "terms": "-", "access": "open", "commercial": commercial,
            **kw}  # fmt: skip


def _model(commercial: str, **kw: Any) -> dict[str, Any]:
    """가중치 항목 하나."""
    return {"path": "data/models/x.onnx", "sha256": "0" * 64, "url": "https://example.org/x",
            "license": "Apache-2.0", "training_data": [], "commercial": commercial,
            **kw}  # fmt: skip


DATASETS = DatasetRegistry.model_validate(
    {"version": 1, "datasets": {"ok": _ds("allowed"), "nc": _ds("review"), "no": _ds("forbidden")}}
)


def test_weight_class_must_not_be_looser_than_its_data_or_base() -> None:
    """allowed 데이터만이면 allowed 가능, review 데이터·review 출발 가중치가 섞이면 review 이상."""
    models = ModelRegistry.model_validate(
        {"version": 2, "accept": ["allowed", "review"], "models": {
            "base": _model("review"),
            "a": _model("allowed", training_datasets=["ok"]),
            "b": _model("allowed", training_datasets=["ok", "nc"]),
            "c": _model("allowed", training_datasets=["ok"], base_model="base"),
            "d": _model("review", training_datasets=["ok", "nc"], base_model="base"),
            "e": _model("forbidden", training_datasets=["no"]),
            "f": _model("review", training_datasets=["missing"], base_model="gone"),
        }}
    )  # fmt: skip
    found = dataset_violations(models, DATASETS)
    assert not any(m.startswith(("a:", "d:", "base:")) for m in found)
    assert any(m.startswith("b:") and "느슨" in m for m in found)
    assert any(m.startswith("c:") and "느슨" in m for m in found)
    assert any(m.startswith("e:") and "forbidden 데이터셋" in m for m in found)
    assert {m for m in found if m.startswith("f:")} == {
        "f: 공개 데이터셋 목록에 없는 missing",
        "f: 출발 가중치 gone이 모델 목록에 없습니다",
    }


def _pretrain(datasets: list[str], classes: dict[str, Any]) -> dict[str, Any]:
    """1차 학습 설정 하나."""
    return {"version": 1, "datasets": datasets, "classes": classes,
            "build": {"max_images_per_class": 10, "val_fraction": 0.1, "seed": 0, "min_box_px": 4},
            "train": {"yolox_commit": "a" * 40, "model": "yolox_s", "input_size": 640, "epochs": 1,
                      "batch_size": 1, "fp16": False}}  # fmt: skip


def test_pretrain_config_rules() -> None:
    """클래스 대응은 datasets에 든 데이터셋만, 객체마다 원래 클래스 하나 이상.

    forbidden·미등록 데이터셋은 위반이다.
    """
    ok = PretrainObjects.model_validate(_pretrain(["ok"], {"mop": {"ok": ["mop"]}}))
    assert pretrain_violations(ok, DATASETS) == []
    bad = PretrainObjects.model_validate(_pretrain(["no", "zz"], {"mop": {"no": ["mop"]}}))
    assert pretrain_violations(bad, DATASETS) == [
        "pretrain objects: forbidden 데이터셋 no",
        "pretrain objects: 공개 데이터셋 목록에 없는 zz",
    ]
    with pytest.raises(ValidationError, match="datasets에 없는"):
        PretrainObjects.model_validate(_pretrain(["ok"], {"mop": {"nc": ["mop"]}}))
    with pytest.raises(ValidationError, match="원래 클래스가 하나도"):
        PretrainObjects.model_validate(_pretrain(["ok"], {"mop": {"ok": []}}))


def test_verified_needs_sources() -> None:
    """원문 확인일을 적으면 근거 주소가 있어야 한다."""
    with pytest.raises(ValidationError, match="sources"):
        ExternalDataset.model_validate(_ds("allowed", verified="2026-10-26"))
    assert (
        ExternalDataset.model_validate(
            _ds("allowed", verified="2026-10-26", sources=["https://example.org/license"])
        ).verified
        is not None
    )
