"""AI 1차 검수 정책 테스트 (`dlp_aireview.policy`, WP17.2).

판정 버전에 들어가는 digest가 "어느 단위를 묻나·어떻게 돌리나" 절(selection·kinds·pending·server·
gate·estimate)을 바꿔도 그대로이고, 판정 결과를 바꾸는 절을 바꾸면 달라지는지 본다. 결정으로 고정한
값(블러 판정 금지, 센서 출처 제외, 외부 백엔드 금지, 타임라인 종류 삭제 금지)이 로드 오류인지 본다.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_aireview.policy import PROMPT_VERSION, AiReviewPolicy

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
        ("server", {"max_context_splits": 2}, "max_context_splits"),
    ],
)
def test_fixed_decisions_are_load_errors(
    policy: AiReviewPolicy, section: str, values: dict[str, Any], message: str
) -> None:
    """블러 판정, 검수 화면 없는 종류, 센서 출처, 외부 백엔드, 타임라인 종류 삭제, 계약보다 긴 이유,
    프레임 상한 초과, 모르는 선택 사유(오타)와 AI 판정 사유, 해시 밖 절의 결과 바꾸는 키는 로드
    오류다."""
    with pytest.raises(ValidationError, match=message):
        with_policy(policy, **{section: values})


def test_unknown_key_and_prompt_version_mismatch(policy: AiReviewPolicy) -> None:
    """모르는 키는 오류, 정책 prompt_version이 코드 틀 버전과 다르면 오류."""
    data = policy.model_dump()
    with pytest.raises(ValidationError):
        AiReviewPolicy.model_validate(data | {"unknown": 1})
    with pytest.raises(ValidationError, match="prompt_version"):
        AiReviewPolicy.model_validate(data | {"prompt_version": PROMPT_VERSION + 1})
