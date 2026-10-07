"""정책 파일 사이의 값이 서로 맞는지 검사한다.

서로 다른 패키지의 정책이 한 값을 기준으로 맞물려 있을 때, 한쪽만 바꿔 조용히 깨지는 것을 막는다.
"""

from __future__ import annotations

from pathlib import Path

from dlp_prelabel.policy import load_policy as load_prelabel
from dlp_review.ops.policy import load_policy as load_review

ROOT = Path(__file__).resolve().parents[1]


def test_contact_mismatch_threshold_separates_single_source_contacts() -> None:
    """검수 우선순위의 접촉 불일치 문턱은 한쪽 출처(장갑만·영상만)는 잡고 융합은 놓아야 한다.

    `prelabel.yaml contact.confidence`의 출처별 신뢰도와 `review.yaml
    priority.contact_mismatch_max_confidence`를 비교한다: max(장갑, 영상) ≤ 문턱 < 융합.
    """
    conf = load_prelabel(ROOT).contact.confidence
    threshold = load_review(ROOT).priority.contact_mismatch_max_confidence
    assert max(conf.glove, conf.video) <= threshold < conf.fused
