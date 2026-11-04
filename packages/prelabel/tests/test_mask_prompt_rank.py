"""마스크 프롬프트 순위: 신뢰도 0 트랙을 1.0으로 올려 세우지 않는다 (F-PAR-91)."""

from __future__ import annotations

from dlp_prelabel.masks import PromptTrack
from dlp_schema.labels import Hand, HandStatePayload
from dlp_schema.testing import make_label


def _track(confidence: float | None) -> PromptTrack:
    payload = HandStatePayload(
        hand=Hand.RIGHT, contact_target_kind="object", target_id="x", role="active"
    )
    return PromptTrack(make_label(payload, confidence=confidence), "cup_01", "cup", False, (0,))


def test_zero_confidence_track_ranks_last_f_par_91() -> None:
    assert _track(0.0).rank == 0.0
    assert _track(0.0).rank < _track(0.4).rank < _track(None).rank == 1.0
