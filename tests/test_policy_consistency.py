"""정책 파일 사이의 값이 서로 맞는지 검사한다.

서로 다른 패키지의 정책이 한 값을 기준으로 맞물려 있을 때, 한쪽만 바꿔 조용히 깨지는 것을 막는다.
"""

from __future__ import annotations

from pathlib import Path

from dlp_actions.policy import load_policy as load_actions
from dlp_aireview.policy import load_policy as load_aireview
from dlp_ops.policy import load_policy as load_ops
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


def test_aireview_policy_matches_neighbours() -> None:
    """AI 1차 검수 정책과 이웃 정책 (WP17, ADR 0040).

    - 대상을 모르는 접촉 ID는 행동 단계와 같다 (`actions.yaml unresolved_entity_ids`).
    - 판정 모델은 행동 분류 VLM과 다르다 (같으면 행동 라벨 판정에 상관 오류, 사용자 결정).
    - 장갑 Parquet을 읽는 원본 접근 용도 `aireview.run`이 서비스 용도로 등록돼 있다 (월간 감사가
      이상 접근으로 표시하지 않게).
    """
    aireview, actions = load_aireview(ROOT), load_actions(ROOT)
    assert set(aireview.unresolved_entity_ids) == set(actions.unresolved_entity_ids)
    assert aireview.backend.model != actions.vlm.model
    assert "aireview.run" in load_ops(ROOT).audit.service_purposes
