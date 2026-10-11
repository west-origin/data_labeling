"""준비 검사·모델 명령·검수 정책은 오타/부분 준비를 성공으로 숨기지 않는다."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from dlp_cli.models_cmds import cmd_fetch, used_models
from dlp_cli.pilot_cmds import cmd_check
from dlp_cli.review_config import load_review_policy
from dlp_privacy.detectors import build_detectors
from dlp_privacy.policy import load_policy as privacy_policy
from dlp_review.ops.policy import load_policy as review_policy
from dlp_schema import load_ontology, repo_root
from dlp_schema.predictor import ModelUnavailableError
from dlp_train.deployed import Deployed


def test_unknown_model_fails_without_downloading() -> None:
    with patch("dlp_cli.models_cmds.fetch") as download:
        assert cmd_fetch(argparse.Namespace(names=["misspelled-model"])) == 1
    download.assert_not_called()


def test_custom_review_policy_is_explicit_and_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo_root()
    data = yaml.safe_load((root / "config/policies/review.yaml").read_text())
    data["reviewers"]["privacy"] = ["local-privacy"]
    selected = tmp_path / "review.yaml"
    selected.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(selected))
    assert load_review_policy(root).reviewers.privacy == ("local-privacy",)
    assert review_policy(root).reviewers.privacy == ()
    selected.unlink()
    with pytest.raises(FileNotFoundError):
        load_review_policy(root)


def test_detector_cycle_is_reported_and_license_walk_terminates() -> None:
    root = repo_root()
    policy = privacy_policy(root)
    name = next(n for n, spec in policy.detectors.items() if spec.kind == "reflection")
    detectors = dict(policy.detectors)
    detectors[name] = detectors[name].model_copy(update={"region_detector": name})
    policy = policy.model_copy(update={"detectors": detectors})
    with patch("dlp_cli.models_cmds.load_privacy_policy", return_value=policy):
        assert used_models(root)
    with pytest.raises(ValueError, match="순환"):
        build_detectors(policy, root)


def test_missing_models_make_readiness_fail_without_service_calls() -> None:
    args = argparse.Namespace(profile="hands", models_only=True, video=None, url=None)
    with (
        patch("dlp_cli.pilot_cmds.resolve", side_effect=FileNotFoundError("missing")),
        patch("dlp_cli.pilot_cmds.service_checks") as services,
    ):
        assert cmd_check(args) == 1
    services.assert_not_called()


@pytest.mark.parametrize("allow_partial", [False, True])
def test_missing_selected_prelabel_model_requires_explicit_partial_run(
    tmp_path: Path, allow_partial: bool
) -> None:
    from dlp_cli.prelabel_cmds import cmd_run

    args = argparse.Namespace(
        session_id="s001",
        store=f"local:{tmp_path}",
        url="sqlite://",
        skip=["body", "objects", "tools", "depth3d"],
        allow_missing_models=allow_partial,
    )
    with (
        patch(
            "dlp_cli.prelabel_cmds.session_ontology",
            return_value=load_ontology(repo_root() / "config/ontology/v1"),
        ),
        patch("dlp_cli.prelabel_cmds.MediaPipeHands", name="missing-hands") as hands,
        patch("dlp_cli.prelabel_cmds.deployed_predictors", return_value=Deployed()),
        patch("dlp_cli.prelabel_cmds.list_model_versions", return_value=[]),
        patch("dlp_cli.prelabel_cmds.raw_store"),
        patch("dlp_cli.prelabel_cmds.run_prelabel") as run,
    ):
        hands.name = "hands"
        hands.side_effect = ModelUnavailableError("weights missing")
        run.return_value = SimpleNamespace(
            produced={}, skipped=[], lifted=0, contacts=0, wearer=None
        )
        assert cmd_run(args) == (0 if allow_partial else 1)
    assert run.call_count == int(allow_partial)
