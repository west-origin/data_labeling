"""착용자 몸 판정 규칙 (`dlp_schema.wearer_body`, ADR 0055·0058).

정답은 합성 픽스처 `dlp_fixtures.arm.generate_arm_scenario`: 착용자 손(hand21), 화면 위쪽의 다른
사람(머리 보임), 착용자 몸을 덮은 사람 박스, 다른 사람 박스, 가짜 착용자·다른 사람 자세.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.arm import generate_arm_scenario
from dlp_schema.labels import BoxTrackPayload, Keypoint
from dlp_schema.wearer_body import WearerEvidence, wearer_box_frame, wearer_pose_frame


def _points(xy: NDArray[np.float64], scores: NDArray[np.float64]) -> list[Keypoint]:
    """가짜 자세 → Keypoint (점수 0.3 이상 보임)."""
    return [
        Keypoint(x=float(xy[i, 0]), y=float(xy[i, 1]), visibility=2 if scores[i] >= 0.3 else 1)
        for i in range(len(scores))
    ]


def test_evidence_and_box_and_pose_rules() -> None:
    """착용자 박스·자세는 착용자로, 다른 사람 박스·자세는 아니다. 증거 없는 시각은 아니다.

    박스 규칙의 아래 가장자리 조건은 프레임 높이를 알 때만 본다 (모르면 None).
    """
    scene = generate_arm_scenario(1)
    ev = WearerEvidence.from_labels([*scene.hand_labels, scene.other_person], "bodycam")
    assert not ev.empty and ev.hand_times == scene.frame_times
    assert WearerEvidence.from_labels(scene.hand_labels, "other-stream").empty
    t = scene.frame_times[0]
    hands, heads = ev.hands_near(t, 0), ev.heads_near(t, 0)
    assert {h.hand.value for h in hands} == {"left", "right"} and len(heads) == 5
    h = scene.size[1]
    for box in scene.boxes:
        p = box.payload
        assert isinstance(p, BoxTrackPayload)
        k = p.keyframes[0]
        got = wearer_box_frame(
            (k.x, k.y, k.w, k.h), hands, heads, margin=0.1, frame_h=h, bottom_band=0.15
        )
        # 사람 박스 규칙만 보면 몸통을 덮은 큰 수건 박스도 착용자 프레임이다 (면적 문턱은 프리라벨)
        assert got == (box.label_id.endswith(("wearer", "towel")))
    wearer = _points(*scene.wearer_poses[t])
    other = _points(*scene.other_pose)
    assert wearer_pose_frame(wearer, hands, frame_h=h, bottom_band=0.15, wrist_match_palms=1.0)
    assert not wearer_pose_frame(other, hands, frame_h=h, bottom_band=0.15, wrist_match_palms=1.0)
    assert not wearer_pose_frame(wearer, [], frame_h=h, bottom_band=0.15, wrist_match_palms=1.0)
    # 손목은 맞아도 화면 아래에 닿지 않는 자세는 착용자가 아니다
    assert not wearer_pose_frame(
        wearer, hands, frame_h=h * 3, bottom_band=0.15, wrist_match_palms=1.0
    )
