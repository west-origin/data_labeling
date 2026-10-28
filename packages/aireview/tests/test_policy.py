"""AI 1차 검수 정책 테스트 (`dlp_aireview.policy`, WP17.2).

판정 버전에 들어가는 digest가 "어느 단위를 묻나·어떻게 돌리나" 절(selection·kinds·pending·server·
gate·estimate)을 바꿔도 그대로이고, 판정 결과를 바꾸는 절을 바꾸면 달라지는지 본다. 결정으로 고정한
값(블러 판정 금지, 센서 출처 제외, 외부 백엔드 금지, 타임라인 종류 삭제 금지)이 로드 오류인지 본다.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_aireview.policy import (
    PROMPT_VERSION,
    AiReviewPolicy,
    load_policy,
    panel_models,
    select_model,
)

from .conftest import ROOT, with_policy

UNHASHED: dict[str, dict[str, Any]] = {
    "selection": {"mode": "flagged", "audit_ratio": 0.3, "seed": 7},
    "kinds": {"temporal": ["action", "gap"], "spatial": ["box_track"]},
    "pending": {"quiet_minutes": 5},
    "server": {"concurrency_per_endpoint": 2, "timeout_s": 5, "backoff_s": [1]},
    "gate": {"min_wrong_recall": 0.9},
    "estimate": {"ok_tokens": 3},
}
HASHED: list[tuple[str, dict[str, Any]]] = [
    ("backend", {"model": "Qwen/Qwen2.5-VL-32B-Instruct"}),
    ("backend", {"served_name": "judge"}),
    ("temporal", {"frames": 6}),
    ("temporal", {"timestamp_overlay": False}),
    ("spatial", {"max_side_px": 512}),
    ("image", {"jpeg_quality": 90}),
    ("response", {"max_retries": 2}),
    # 문맥 초과 분할 횟수는 끝내 미응답이 되는 대상을 바꾼다 (결과를 바꾸므로 해시)
    ("response", {"max_context_splits": 1}),
    ("response", {"finding_overlap_iou": 0.5}),
    ("sensor", {"enabled": False}),
]


def test_repo_policy_loads_with_user_decisions(policy: AiReviewPolicy) -> None:
    """저장소 정책: 판정 모델은 행동 분류 모델과 다르다(자기 검수 상관 오류 방지), 전량 판정, 사내
    서버만, 장갑 센서 출처 제외, 크기 정렬은 Qwen3-VL(32)."""
    actions_model = load_actions_policy(ROOT).vlm.model
    assert policy.backend.model == "Qwen/Qwen3-VL-8B-Instruct" != actions_model
    assert policy.selection.mode == "all"
    assert policy.server.allow_external is False
    assert policy.kinds.sources == ("model",)
    assert policy.image.align_px == 32
    assert policy.prompt_version == PROMPT_VERSION
    assert policy.response.max_findings == 0
    assert re.fullmatch(r"aireview-[0-9a-f]{12}\+vlm-\S+", policy.verdict_version)
    assert policy.verdict_version.endswith("+vlm-Qwen/Qwen3-VL-8B-Instruct")
    assert policy.served_model == policy.backend.model


@pytest.mark.parametrize("section", sorted(UNHASHED))
def test_digest_ignores_run_and_selection_sections(policy: AiReviewPolicy, section: str) -> None:
    """선택·종류·대기·서버·게이트·추정 절은 판정 버전을 바꾸지 않는다 (대상 집합은 단위 키)."""
    changed = with_policy(policy, **{section: UNHASHED[section]})
    assert changed != policy
    assert changed.digest == policy.digest and changed.verdict_version == policy.verdict_version


@pytest.mark.parametrize(("section", "values"), HASHED)
def test_digest_follows_result_sections(
    policy: AiReviewPolicy, section: str, values: dict[str, Any]
) -> None:
    """모델·프레임·이미지·응답(문맥 초과 분할 포함)·센서 절을 바꾸면 판정 버전이 바뀐다 (전부 다시
    판정)."""
    changed = with_policy(policy, **{section: values})
    assert changed.digest != policy.digest


@pytest.mark.parametrize(
    ("section", "values", "message"),
    [
        ("kinds", {"spatial": ["box_track", "blur_track"]}, "블러"),
        ("kinds", {"temporal": ["relation"]}, "판정할 수 없는"),
        ("kinds", {"sources": ["sensor"]}, "sources"),
        ("server", {"allow_external": True}, "allow_external"),
        ("response", {"suggest": {"kind_change": True, "delete_kinds": ["action"]}}, "삭제"),
        ("response", {"reason_max_chars": 80}, "reason_max_chars"),
        ("temporal", {"frames": 20}, "max_frames"),
        ("backend", {"kind": "anthropic"}, "kind"),
        ("selection", {"reasons": ["low_confidense"]}, "reasons"),
        ("selection", {"reasons": ["low_confidence", "ai_wrong"]}, "AI 판정 사유"),
        ("selection", {"reasons": ["ai_disagreement"]}, "AI 판정 사유"),
        ("backend", {"panel": [{"model": "m", "fingerprint": "f"}] * 2}, "두 번"),
        ("server", {"max_context_splits": 2}, "max_context_splits"),
        ("gate", {"inject": {"verb_swapp": 0.1}}, "모르는 교란 종류"),
        ("gate", {"inject": {"verb_swap": 1.5}}, "0~1"),
    ],
)
def test_fixed_decisions_are_load_errors(
    policy: AiReviewPolicy, section: str, values: dict[str, Any], message: str
) -> None:
    """블러 판정, 검수 화면 없는 종류, 센서 출처, 외부 백엔드, 타임라인 종류 삭제, 계약보다 긴 이유,
    프레임 상한 초과, 모르는 선택 사유(오타)와 AI 판정 사유, 해시 밖 절의 결과 바꾸는 키, 골든
    게이트의 모르는 교란 종류·범위 밖 비율은 로드 오류다 (평가 도중 traceback 대신)."""
    with pytest.raises(ValidationError, match=message):
        with_policy(policy, **{section: values})


def test_unknown_key_and_prompt_version_mismatch(policy: AiReviewPolicy) -> None:
    """모르는 키는 오류, 정책 prompt_version이 코드 틀 버전과 다르면 오류."""
    data = policy.model_dump()
    with pytest.raises(ValidationError):
        AiReviewPolicy.model_validate(data | {"unknown": 1})
    with pytest.raises(ValidationError, match="prompt_version"):
        AiReviewPolicy.model_validate(data | {"prompt_version": PROMPT_VERSION + 1})


# ---------------------------------------------------------------- 판정단 (ADR 0063)

D3 = {"model": "vllm-sr/d3-turbo", "served_name": "d3-turbo", "fingerprint": "vllm-bf16-r1"}
AMOS = {"model": "mukaddamzaid/amos-1-large", "fingerprint": "vllm-bf16"}


def test_repo_panel_is_empty_by_default(policy: AiReviewPolicy) -> None:
    """저장소 정책의 판정단은 비어 있다 (운영 동작 불변): 고를 수 있는 모델은 backend.model 하나."""
    assert policy.backend.panel == ()
    assert panel_models(policy) == (policy.backend.model,)
    assert select_model(policy, policy.backend.model) is policy


def test_panel_list_is_not_hashed_and_selecting_changes_version(policy: AiReviewPolicy) -> None:
    """판정단 목록은 digest에 들지 않는다 (후보를 더해도 지금 버전 그대로). 판정단 모델을 고르면
    backend.model·served_name·fingerprint만 바뀌고 버전이 `+vlm-<그 모델>`로 달라진다. 요청 형식
    (프레임·응답 등)은 같다."""
    paneled = with_policy(policy, backend={"panel": [D3, AMOS]})
    assert paneled.digest == policy.digest and paneled.verdict_version == policy.verdict_version
    d3 = select_model(paneled, "vllm-sr/d3-turbo")
    assert d3.verdict_version.endswith("+vlm-vllm-sr/d3-turbo")
    assert d3.verdict_version != policy.verdict_version
    assert d3.served_model == "d3-turbo" and d3.backend.fingerprint == "vllm-bf16-r1"
    assert d3.backend.model_dump(exclude={"model", "served_name", "fingerprint"}) == (
        paneled.backend.model_dump(exclude={"model", "served_name", "fingerprint"})
    )
    assert d3.model_dump(exclude={"backend"}) == policy.model_dump(exclude={"backend"})
    amos = select_model(paneled, "mukaddamzaid/amos-1-large")
    assert amos.served_model == "mukaddamzaid/amos-1-large"
    assert len({policy.verdict_version, d3.verdict_version, amos.verdict_version}) == 3
    # 같은 판정단에서 같은 모델을 다시 고르면 같은 버전 (멱등·재개)
    assert select_model(paneled, "vllm-sr/d3-turbo").verdict_version == d3.verdict_version


def test_select_model_outside_panel_is_refused(policy: AiReviewPolicy) -> None:
    """판정단에 없는 모델은 고를 수 없다 (고를 수 있는 이름을 알린다)."""
    paneled = with_policy(policy, backend={"panel": [D3]})
    with pytest.raises(ValueError, match=r"판정단.*없는 모델: other/model"):
        select_model(paneled, "other/model")
    with pytest.raises(ValueError, match="vllm-sr/d3-turbo"):
        select_model(paneled, "mukaddamzaid/amos-1-large")


def _tmp_root(tmp_path: Path, panel: list[dict[str, Any]]) -> Path:
    """판정단을 넣은 aireview.yaml과 저장소 models.yaml을 둔 임시 저장소 루트."""
    (tmp_path / "config" / "policies").mkdir(parents=True)
    shutil.copy(ROOT / "config" / "models.yaml", tmp_path / "config" / "models.yaml")
    data = yaml.safe_load((ROOT / "config/policies/aireview.yaml").read_text("utf-8"))
    data["backend"]["panel"] = panel
    text = yaml.safe_dump(data, allow_unicode=True)
    (tmp_path / "config" / "policies" / "aireview.yaml").write_text(text, "utf-8")
    return tmp_path


def test_load_checks_panel_licenses_and_selects(tmp_path: Path, policy: AiReviewPolicy) -> None:
    """판정단 모델은 로드 때 models.yaml 라이선스 검사를 받는다: 등록된 review 모델은 통과하고
    `model`로 고를 수 있다. 목록에 없는 모델·forbidden 모델은 로드 오류다 (make licenses와 같은
    검사)."""
    root = _tmp_root(tmp_path / "ok", [D3, AMOS])
    loaded = load_policy(root)
    assert loaded.verdict_version == policy.verdict_version
    assert load_policy(root, model="vllm-sr/d3-turbo").verdict_version.endswith(
        "+vlm-vllm-sr/d3-turbo"
    )
    with pytest.raises(ValueError, match="없는 모델"):
        load_policy(root, model="nope/model")
    unknown = _tmp_root(tmp_path / "unknown", [{"model": "nope/model", "fingerprint": "x"}])
    with pytest.raises(ValueError, match="모델 목록에 없는 nope/model"):
        load_policy(unknown)
    forbidden = _tmp_root(
        tmp_path / "forbidden", [{"model": "Qwen/Qwen2.5-VL-3B-Instruct", "fingerprint": "x"}]
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_policy(forbidden)
