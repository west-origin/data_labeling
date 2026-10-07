"""정책이 쓰는 모델의 상업 사용 분류 검사 (ADR 0010).

라벨링 데이터를 판매하므로, 정책 YAML이 실제로 쓰는 모델이 `config/models.yaml`에 있고 허용 분류
(`accept`)에 드는지 확인한다. `make licenses`(`dlp models licenses`)와 같은 판정을 테스트로
고정한다.
"""

from __future__ import annotations

from pathlib import Path

from dlp_cli.models_cmds import used_models
from dlp_models.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


def test_policies_only_use_accepted_models() -> None:
    """지금 정책이 쓰는 모든 모델이 허용 분류다. 반사면 탐지기가 쓰는 하위 탐지기(OWLv2, YuNet)까지
    `used_models`가 따라간다.
    """
    registry = load_registry(ROOT)
    used = used_models(ROOT)
    assert registry.violations(used) == []
    # 반사면 탐지기가 쓰는 하위 탐지기까지 따라간다
    assert used["privacy.detectors.open_vocab.model"] == "owlv2_onnx"
    assert used["privacy.detectors.yunet.model"] == "yunet"


def test_no_forbidden_model_and_humanart_detector_is_gone() -> None:
    """`forbidden`(비상업 가중치)은 허용 목록에 없고, 비상업 Human-Art 데이터로 학습한 모델은
    목록에서 빠졌다."""
    registry = load_registry(ROOT)
    assert "forbidden" not in registry.accept
    assert all("Human-Art" not in " ".join(m.training_data) for m in registry.models.values())


def test_stricter_acceptance_flags_review_models() -> None:
    """법무 확인 뒤 review를 빼면 review 모델을 쓰는 정책이 걸린다.

    허용 분류를 `allowed`만으로 좁히면 YuNet·전신(RTMPose) 정책은 위반이 되고, MediaPipe 손은
    그대로다. 목록에 없는 모델 이름은 "모델 목록에 없는" 위반이다.
    """
    registry = load_registry(ROOT).model_copy(update={"accept": ("allowed",)})
    flagged = {p.split(":")[0] for p in registry.violations(used_models(ROOT))}
    assert "privacy.detectors.yunet.model" in flagged
    assert "prelabel.models.body" in flagged
    assert "prelabel.models.hands" not in flagged
    assert registry.violations({"x": "no_such_model"}) == ["x: 모델 목록에 없는 no_such_model"]
