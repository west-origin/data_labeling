"""정책 파일 사이의 값이 서로 맞는지 검사한다.

서로 다른 패키지의 정책이 한 값을 기준으로 맞물려 있을 때, 한쪽만 바꿔 조용히 깨지는 것을 막는다.
설정 주석이 코드의 저장 위치와 어긋나 운영자를 잘못 이끄는 경우(버킷 설명)도 함께 본다.
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


def _key_comment(text: str, key: str) -> str:
    """YAML 텍스트에서 `  <key>: 값  # 주석` 줄의 주석 부분 (그 키 줄이 정확히 하나여야 한다)."""
    lines = [ln for ln in text.splitlines() if ln.startswith(f"  {key}: ")]
    assert len(lines) == 1, (key, lines)
    return lines[0].partition("#")[2]


def test_bucket_comments_keep_proxy_in_raw_bucket() -> None:
    """F-PMS-04: 버킷 설명 주석은 프록시(블러 전 영상)를 원본 버킷에 둔다고 적어야 한다.

    수집(`dlp_media.ingest`)은 프록시를 원본 저장소의 `sessions/<세션>/derived/`에 쓰고, 블러 검수
    (`dlp_review.tasks.proxy_key`)도 원본 버킷에서 읽는다. 주석을 믿고 프록시를 라벨링 버킷에 두면
    일반 라벨러 자격 증명으로 블러 전 영상이 열린다. 정답: defaults.yaml `buckets.raw` 줄 주석에는
    "프록시"가 있고 `buckets.labeling` 줄 주석에는 없다.
    """
    from dlp_review.tasks import proxy_key

    assert proxy_key("s1", "bodycam") == "sessions/s1/derived/bodycam.proxy.mp4"
    text = (ROOT / "config" / "defaults.yaml").read_text(encoding="utf-8")
    assert "프록시" in _key_comment(text, "raw")
    assert "프록시" not in _key_comment(text, "labeling")
