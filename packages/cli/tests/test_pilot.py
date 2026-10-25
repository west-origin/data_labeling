"""준비 검사·모델 명령·검수 정책은 오타/부분 준비를 성공으로 숨기지 않는다."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from dlp_cli import pilot_cmds
from dlp_cli.models_cmds import cmd_fetch, used_models
from dlp_cli.pilot_cmds import cmd_check
from dlp_cli.review_config import load_review_policy
from dlp_privacy.detectors import build_detectors
from dlp_privacy.policy import load_policy as privacy_policy
from dlp_review.ops.policy import LabelStudioPolicy
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
        skip=["body", "objects", "tools", "depth3d", "masks"],
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
            produced={}, skipped=[], lifted=0, contacts=0, person_contacts=0, wearer=None,
            poses=0, world=0, pose_note=None, pose_lost_at_ms=None, trajectories_inherited=0,
            trajectories_retracted=0, lift_blocked=0, focal_estimate=None, wearer_suppressed={},
            arms={}, masks=0, mask_report=None, masks_blocked=0,
        )  # fmt: skip
        assert cmd_run(args) == (0 if allow_partial else 1)
    assert run.call_count == int(allow_partial)


def test_pilot_check_requires_labeler_label_studio_account() -> None:
    """준비 검사는 일반 검수자의 Label Studio 계정 연결이 서비스 계정이 아닌 실제 계정인지 본다.

    연결이 없거나, 연결한 이메일이 Label Studio에 없거나, 서비스 계정이면 실패다 (F-REV-06: 수집은
    담당자 계정의 주석만 받으므로 연결 없이는 시간 검수를 수집할 수 없다).
    """
    ls = Mock()
    ls.service_user_id.return_value = 1
    ls.find_user_id.side_effect = {"lab@example.com": 7, "admin@example.com": 1}.get
    base = review_policy(repo_root())
    check = pilot_cmds._ls_account_check  # pyright: ignore[reportPrivateUsage]
    cases = {"lab@example.com": True, "ghost@example.com": False, "admin@example.com": False}
    for email, ready in cases.items():
        accounts = LabelStudioPolicy(users={"pilot-labeler": email})
        review = base.model_copy(update={"label_studio": accounts})
        result = check(ls, review, "pilot-labeler")
        assert (result.name, result.ok) == ("label-studio-account", ready)
    assert not check(ls, base, "pilot-labeler").ok  # 기본 정책: 연결 없음
