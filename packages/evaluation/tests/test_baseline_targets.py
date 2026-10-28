"""비교 버전 '예측 없음' 검사가 과제의 평가 대상만 세는지 (F-ETA-21).

DB 없이 `GoldenSession`을 직접 만들어 `load_golden_merged(require_predictions=True)`를 부른다.
정답 근거: 골든 세션 하나에 사람 hand21 정답, 손 모델(mp-hands-v1) hand21 예측, 전신 모델
(rtm-body-v1) coco17 예측, 휴리스틱 접촉 모델의 접촉 없음(none) hand_state만 둔다. hands 과제에
전신 버전을 주거나 contact 과제에 접촉 구간 없는 버전을 주면 평가할 예측이 없다.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import sqlalchemy as sa

from dlp_eval.runner import (
    GoldenSession,
    MissingPredictionsError,
    is_task_target,
    load_golden_merged,
)
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.testing import make_label, make_session


def _kp(skeleton: str, n: int, hand: str | None = None) -> dict[str, Any]:
    """점 n개 키포인트 트랙 (100 ms 간격 3개)."""
    p: dict[str, Any] = {
        "kind": "keypoint_track",
        "entity_id": f"{hand or 'person'}_1",
        "skeleton": skeleton,
        "keyframes": [
            {
                "t_ms": t * 100,
                "points": [{"x": 100 + 7 * j, "y": 80 + 5 * j, "visibility": 2} for j in range(n)],
            }
            for t in range(3)
        ],
    }
    if hand:
        p["hand"] = hand
    return p


def _model(lid: str, payload: dict[str, Any], version: str, **kw: Any) -> LabelRecord:
    return make_label(
        payload,
        label_id=f"gd0-{lid}",
        session_id="gd0",
        provenance=Provenance(source=Source.MODEL, model_version=version),
        confidence=0.8,
        **kw,
    )


def _golden() -> list[GoldenSession]:
    truth = make_label(
        _kp("hand21", 21, "right"), label_id="gd0-t-hand", session_id="gd0", stream_id="bodycam"
    )
    no_contact = {
        "kind": "hand_state",
        "hand": "right",
        "contact_target_kind": "none",
        "role": "inactive",
    }
    labels = [
        truth,
        _model("mp", _kp("hand21", 21, "right"), "mp-hands-v1", stream_id="bodycam"),
        _model("rtm", _kp("coco17", 17), "rtm-body-v1", stream_id="bodycam"),
        _model("c", no_contact, "contact-none-v1", stream_id=None, t_end_ms=300),
    ]
    session = make_session("gd0")
    return [GoldenSession(session, labels, [truth], {"site": session.site_id})]


def _load(task: Any, version: str) -> None:
    # sessions를 주면 DB를 읽지 않는다
    load_golden_merged(
        cast(sa.Connection, None),
        "golden-v1",
        {task: [version]},
        require_predictions=True,
        sessions=_golden(),
    )


def test_right_skeleton_baseline_is_accepted() -> None:
    _load("hands", "mp-hands-v1")


def test_wrong_skeleton_baseline_is_refused() -> None:
    """hands에 전신(coco17) 버전을 주면 keypoint_track은 있어도 hand21이 없어 멈춘다."""
    with pytest.raises(MissingPredictionsError, match="rtm-body-v1"):
        _load("hands", "rtm-body-v1")
    with pytest.raises(MissingPredictionsError, match="mp-hands-v1"):
        _load("body", "mp-hands-v1")


def test_contact_baseline_without_contact_spans_is_refused() -> None:
    """접촉 없음(none) hand_state만 낸 버전은 접촉 평가에 쓸 예측이 없다."""
    with pytest.raises(MissingPredictionsError, match="contact-none-v1"):
        _load("contact", "contact-none-v1")
    assert not any(is_task_target("contact", x) for x in _golden()[0].labels)
