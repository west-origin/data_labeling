"""정책 파일 사이의 값이 서로 맞는지 검사한다."""

from __future__ import annotations

from pathlib import Path

from dlp_prelabel.policy import load_policy as load_prelabel
from dlp_review.ops.policy import load_policy as load_review

ROOT = Path(__file__).resolve().parents[1]


def test_contact_mismatch_threshold_separates_single_source_contacts() -> None:
    """검수 우선순위의 접촉 불일치 문턱은 한쪽 출처(장갑만·영상만)는 잡고 융합은 놓아야 한다."""
    conf = load_prelabel(ROOT).contact.confidence
    threshold = load_review(ROOT).priority.contact_mismatch_max_confidence
    assert max(conf.glove, conf.video) <= threshold < conf.fused
