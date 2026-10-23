"""손 크기 초점 거리 추정(`dlp_prelabel.fov`)과 3D 단계 연동 시험 (ADR 0056).

정답: `dlp_fixtures.hand3d` 장면을 알려진 초점 거리(초광각을 흉내 낸 110 px, 320x240 가로 화각 약
111도)로 투영했다. 정책 기본 화각(90도, 160 px) 근사로 맞추면 거리가 틀리고, 손 크기 사전값(정답
손바닥 0.09 m)과 깊이로 추정한 초점 거리는 정답의 5% 안이어야 한다. 실제 캘리브레이션이 있으면
추정하지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.hand3d import generate_hand3d_scenario
from dlp_prelabel.fov import estimate_focal, focal_samples
from dlp_prelabel.lift3d import DepthLifter, DepthModel
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_schema.labels import IntrinsicsSource, Trajectory3DPayload

ROOT = Path(__file__).resolve().parents[3]
TRUE_F = 110.0


def _policy(**focal: object) -> PrelabelPolicy:
    """저장소 정책 (focal_estimate를 켜고 일부를 바꾼 사본, 3D 간격은 매 프레임)."""
    p = load_policy(ROOT)
    fe = p.depth.focal_estimate.model_copy(
        update={"enabled": True, "frame_stride_ms": 0, "min_samples": 10, **focal}
    )
    depth = p.depth.model_copy(update={"focal_estimate": fe, "frame_stride_ms": 0})
    return p.model_copy(update={"depth": depth})


def test_focal_from_hand_size_recovers_true_focal() -> None:
    """정답 110 px 장면: 근사 160 px로 잰 손바닥 길이 비로 110 px ±5%를 낸다."""
    s = generate_hand3d_scenario(3, n_frames=40, focal_px=TRUE_F)
    policy = _policy()
    samples, size = focal_samples(s.frames(), s.depth, s.labels, policy)
    assert size is not None and size == (320, 240) and len(samples) >= 30
    est = estimate_focal(samples, 0.09, size, policy)
    assert est is not None
    assert est.intrinsics.fx == pytest.approx(TRUE_F, rel=0.05)
    assert est.hfov_deg == pytest.approx(math.degrees(2 * math.atan(160 / TRUE_F)), abs=3.0)
    # 손 크기 사전값이 10% 크면 초점 거리도 그만큼 작게 나온다 (사전값 오차가 그대로 남는다)
    off = estimate_focal(samples, 0.099, size, policy)
    assert off is not None and off.intrinsics.fx == pytest.approx(TRUE_F / 1.1, rel=0.05)


def test_focal_estimate_rejects_too_few_or_out_of_range() -> None:
    """표본이 모자라거나 화각이 범위 밖이면 추정하지 않는다 (화각 근사로 남는다)."""
    s = generate_hand3d_scenario(3, n_frames=40, focal_px=TRUE_F)
    policy = _policy()
    samples, size = focal_samples(s.frames(), s.depth, s.labels, policy)
    assert size is not None
    assert estimate_focal(samples[:5], 0.09, size, policy) is None
    narrow = _policy(hfov_range_deg=(40.0, 80.0))
    assert estimate_focal(samples, 0.09, size, narrow) is None


class _Lifter(DepthLifter):
    """가중치 없는 시험용 3D 단계 (합성 깊이 모델, 합성 프레임)."""

    def __init__(self, policy: PrelabelPolicy, depth: DepthModel) -> None:
        """정책과 깊이 모델만 둔다."""
        self.policy, self.now = policy, datetime(2026, 10, 23, tzinfo=UTC)
        self.version = "depth3d-test+hf5+fe1+p0"
        self.model = depth
        self.last_focal = None


def test_lifter_uses_estimate_only_without_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """캘리브레이션 없음 → 추정 초점 거리로 올리고 출처 estimated_hand_size, 정답 손목 거리 3% 안.
    캘리브레이션 있음 → 추정하지 않고 calibrated."""
    import dlp_prelabel.lift3d as lift3d

    s = generate_hand3d_scenario(3, n_frames=40, focal_px=TRUE_F)

    def frames(_video: Path) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """합성 프레임 (영상 파일 대신)."""
        return s.frames()

    monkeypatch.setattr(lift3d, "iter_frames", frames)
    lifter = _Lifter(_policy(), s.depth)
    labels = lifter.run(
        Path("v.mp4"), session_id=s.session_id, stream_id="bodycam", tracks=s.labels,
        calib=None, ontology_version="1.0.0",
    )  # fmt: skip
    assert lifter.last_focal is not None
    wrist = next(
        x.payload for x in labels
        if isinstance(x.payload, Trajectory3DPayload) and x.payload.part == "wrist"
    )  # fmt: skip
    assert wrist.intrinsics is IntrinsicsSource.ESTIMATED_HAND_SIZE
    errs: list[float] = []
    for q in wrist.samples:
        truth = math.dist(tuple(float(c) for c in s.joints_cam[q.t_ms][0]), (0.0, 0.0, 0.0))
        errs.append(abs(math.dist((q.x, q.y, q.z), (0.0, 0.0, 0.0)) - truth) / truth)
    assert float(np.median(errs)) < 0.03
    calibrated = lifter.run(
        Path("v.mp4"), session_id=s.session_id, stream_id="bodycam", tracks=s.labels,
        calib=s.intrinsics, ontology_version="1.0.0",
    )  # fmt: skip
    assert lifter.last_focal is None
    p = calibrated[0].payload
    assert isinstance(p, Trajectory3DPayload) and p.intrinsics is IntrinsicsSource.CALIBRATED


def test_focal_estimate_is_off_by_default() -> None:
    """저장소 정책은 추정을 끈다 (메트릭 깊이 모델의 화각 사전값을 되돌려 줌, F-PAR-65)."""
    assert not load_policy(ROOT).depth.focal_estimate.enabled
