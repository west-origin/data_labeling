"""`dlp_prelabel.lift3d` 단위 테스트: 깊이 역투영, 추론 간격, 내부 파라미터 비례 조정, 같은 개체
트랙이 여럿일 때의 입력 고르기(F-PAR-01).

실제 깊이 모델 대신 왼쪽 절반 1 m·오른쪽 절반 2 m인 가짜 깊이(`PlaneDepth`)를 써서 3D 좌표 정답을
손으로 계산할 수 있게 한다. DB·서비스 불필요.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from dlp_fixtures.video import write_video
from dlp_models.depth import Intrinsics
from dlp_prelabel.common import Image, model_label
from dlp_prelabel.lift3d import DepthLifter, intrinsics_for, lift_tracks
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import pick_lift_tracks
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    IntrinsicsSource,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Trajectory3DPayload,
    Verification,
    VerificationState,
)
from dlp_schema.session import CameraIntrinsics

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 7, tzinfo=UTC)
W, H = 320, 240


class PlaneDepth:
    """왼쪽 절반 1 m, 오른쪽 절반 2 m인 깊이 맵을 내는 가짜 모델."""

    def __init__(self) -> None:
        """호출 횟수(`calls`)를 0으로 둔다."""
        self.calls = 0

    def predict(self, image: Image) -> NDArray[np.float32]:
        """호출 횟수를 세고 (H, W) 깊이 맵(왼쪽 1 m, 오른쪽 2 m)을 돌려준다."""
        self.calls += 1
        d = np.full(image.shape[:2], 1.0, dtype=np.float32)
        d[:, W // 2 :] = 2.0
        return d


def _tracks() -> list[LabelRecord]:
    """손 트랙(모든 관절이 (80, 120), 100 ms 간격 10프레임)과 버킷 박스(0 ms에 보임, 500 ms에
    화면 밖)를 모델 라벨로 만든다.
    """
    points = tuple(Keypoint(x=80.0, y=120.0, visibility=2) for _ in range(21))
    hand = KeypointTrackPayload(
        entity_id="right_hand",
        skeleton="hand21",
        hand=Hand.RIGHT,
        keyframes=tuple(KeypointFrame(t_ms=t, points=points) for t in range(0, 1_000, 100)),
    )
    box = BoxTrackPayload(
        entity_id="ov_bucket_01",
        class_id="bucket",
        keyframes=(
            BoxKeyframe(t_ms=0, x=200, y=100, w=40, h=40),
            BoxKeyframe(t_ms=500, x=0, y=0, w=0, h=0, outside=True),
        ),
    )
    return [
        model_label(
            label_id=f"l{i}",
            session_id="s",
            stream_id="bodycam",
            t_start_ms=0,
            t_end_ms=900,
            ontology_version="1.0.0",
            model_version="m",
            confidence=0.9,
            payload=p,
            now=NOW,
        )
        for i, p in enumerate((hand, box))
    ]


def test_lifting_unprojects_with_depth_and_respects_stride() -> None:
    """frame_stride_ms=200이면 100 ms 간격 프레임 중 0, 200, ..., 800에서만 깊이를 만들고 역투영
    좌표가 정답과 같은지 본다.

    정답 근거: fx=160, cx=160. 손목 (80, 120)은 왼쪽 절반이라 z=1 → x=(80-160)*1/160=-0.5, y=0.
    버킷 중심 (220, 120)은 오른쪽 절반이라 z=2 → x=(220-160)*2/160=0.75. 화면 밖 키프레임(500
    ms)은 빠진다. 정책 hand_points [0, 4, 8, 12, 16, 20] → 부위 손목과 다섯 손가락 끝. 이 트랙에는
    손 중심 3D 점(world)이 없어 관절마다 깊이만 쓴다 (손가락 끝도 내는 point_depth 대체로 본다.
    기본 drop은 손바닥 관절만 낸다: `test_default_fallback_lifts_only_palm_joints`).
    """
    policy = load_policy(ROOT).depth.model_copy(
        update={"frame_stride_ms": 200, "fingertip_fallback": "point_depth"}
    )
    frames = [(t, np.zeros((H, W, 3), dtype=np.uint8)) for t in range(0, 1_000, 100)]
    depth = PlaneDepth()
    calib = CameraIntrinsics(width=W, height=H, fx=160, fy=160, cx=160, cy=120)
    out = lift_tracks(frames, depth, _tracks(), calib, policy)

    assert depth.calls == 5  # 0, 200, ..., 800
    wrist = out[("right_hand", "wrist")]
    assert [s.t_ms for s in wrist] == [0, 200, 400, 600, 800]
    assert (wrist[0].x, wrist[0].y, wrist[0].z) == pytest.approx((-0.5, 0.0, 1.0))
    assert {p for e, p in out if e == "right_hand"} == {
        "wrist", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
    }  # fmt: skip
    [bucket] = out[("ov_bucket_01", None)]  # 화면 밖 키프레임은 건너뛴다
    assert (bucket.t_ms, bucket.x, bucket.z) == (0, pytest.approx(0.75), pytest.approx(2.0))


def test_default_fallback_lifts_only_palm_joints() -> None:
    """world가 없는 손 트랙을 기본 정책(fingertip_fallback drop)으로 올리면 손바닥 관절인 손목만
    나온다 (손가락 끝 픽셀 깊이는 배경을 읽기 쉬워 내지 않는다, F-PAR-40). 박스 중심은 그대로다."""
    policy = load_policy(ROOT).depth.model_copy(update={"frame_stride_ms": 200})
    assert policy.fingertip_fallback == "drop"
    frames = [(t, np.zeros((H, W, 3), dtype=np.uint8)) for t in range(0, 1_000, 100)]
    out = lift_tracks(frames, PlaneDepth(), _tracks(), None, policy)
    assert set(out) == {("right_hand", "wrist"), ("ov_bucket_01", None)}


def test_intrinsics_scale_with_resolution_or_fall_back_to_hfov() -> None:
    """캘리브레이션 해상도(1920x1080)의 절반 프레임이면 fx·fy·cx·cy가 절반이 되고, 캘리브레이션이
    없으면 화각 90도 근사(fx = 640/2/tan 45도 = 320, 주점은 중앙)가 되는지 본다.
    """
    tol = load_policy(ROOT).depth.max_intrinsics_aspect_mismatch
    calib = CameraIntrinsics(width=1920, height=1080, fx=1000, fy=1000, cx=960, cy=540)
    assert intrinsics_for(calib, 960, 540, 90, tol) == Intrinsics(500, 500, 480, 270)
    approx = intrinsics_for(None, 640, 480, 90, tol)
    assert approx.fx == pytest.approx(320) and (approx.cx, approx.cy) == (320, 240)


def test_calibration_of_another_aspect_ratio_falls_back_to_hfov(tmp_path: Path) -> None:
    """4:3 모드(4000x3000, fx = fy = 2000)에서 잰 캘리브레이션을 16:9 영상(1920x1080)에 쓰면 가로·
    세로를 따로 늘려 fx/fy = 1.33(비정사각 픽셀)이 되던 것을 막는다 (F-PAR-44): 화각 근사로
    떨어지고, 3D 라벨의 내부 파라미터 출처와 표본 신뢰도도 화각 근사다. 같은 화면비(2000x1500)는
    비례 조정한다.
    """
    policy = load_policy(ROOT)
    tol = policy.depth.max_intrinsics_aspect_mismatch
    calib = CameraIntrinsics(width=4000, height=3000, fx=2000, fy=2000, cx=2000, cy=1500)
    assert intrinsics_for(calib, 1920, 1080, 90, tol) == Intrinsics.from_hfov(1920, 1080, 90)
    assert intrinsics_for(calib, 2000, 1500, 90, tol) == Intrinsics(1000, 1000, 1000, 750)
    video = tmp_path / "bodycam.mp4"
    write_video(
        video, ((t, np.zeros((H, W, 3), dtype=np.uint8)) for t in range(0, 300, 100)),
        width=W, height=H,
    )  # fmt: skip
    wide = CameraIntrinsics(width=640, height=360, fx=320, fy=320, cx=320, cy=180)  # 16:9
    labels = PlaneLifter(policy).run(
        video, session_id="s", stream_id="bodycam", tracks=_tracks()[1:], calib=wide,
        ontology_version="1.0.0",
    )  # fmt: skip
    [box] = labels
    assert isinstance(box.payload, Trajectory3DPayload)
    assert box.payload.intrinsics is IntrinsicsSource.HFOV_APPROX
    c = policy.depth.sample_confidence
    assert box.payload.samples[0].confidence == round(c.point_depth * c.hfov_approx, 3)


class PlaneLifter(DepthLifter):
    """가중치 파일 없이 `PlaneDepth`로 도는 깊이 3D 단계 (가중치 확인·ONNX 세션을 건너뛴다)."""

    def __init__(self, policy: PrelabelPolicy) -> None:  # pyright: ignore[reportMissingSuperCall]
        """`DepthLifter.__init__`의 가중치 확인 대신 가짜 모델과 버전을 직접 둔다."""
        self.path = Path("unused.onnx")
        self.version = f"depth3d-plane+p{policy.digest('depth')}"
        self.policy, self.now = policy, NOW
        self.model = PlaneDepth()


def _left_hand(label_id: str, x: float, **kw: object) -> LabelRecord:
    """왼손 hand21 모델 트랙 (모든 관절이 (x, 120), 0~900 ms 100 ms 간격). kw로 검수 상태 등."""
    points = tuple(Keypoint(x=x, y=120.0, visibility=2) for _ in range(21))
    payload = KeypointTrackPayload(
        entity_id="left_hand",
        skeleton="hand21",
        hand=Hand.LEFT,
        keyframes=tuple(KeypointFrame(t_ms=t, points=points) for t in range(0, 1_000, 100)),
    )
    label = model_label(
        label_id=label_id,
        session_id="s",
        stream_id="bodycam",
        t_start_ms=0,
        t_end_ms=900,
        ontology_version="1.0.0",
        model_version=label_id,
        confidence=0.9,
        payload=payload,
        now=NOW,
    )
    return label.model_copy(update=kw)


def test_tracks_sharing_an_entity_are_rejected() -> None:
    """같은 (개체, 트랙 종류) 트랙이 둘이면 표본을 섞지 않고 ValueError로 알린다.

    F-PAR-01: 예전에는 두 트랙의 같은 시각 표본을 한 궤적에 두 번 넣어, 라벨을 만들 때 "키프레임
    시각은 중복 없이 오름차순" 검증이 실패했다. 러너는 `pick_lift_tracks`로 하나만 넘긴다.
    """
    policy = load_policy(ROOT).depth
    frames = [(0, np.zeros((H, W, 3), dtype=np.uint8))]
    tracks = [_left_hand("a", 80.0), _left_hand("b", 240.0)]
    with pytest.raises(ValueError, match="left_hand"):
        lift_tracks(frames, PlaneDepth(), tracks, None, policy)


def test_repeated_frame_times_give_one_sample() -> None:
    """PTS 반올림으로 같은 ms 프레임이 이어져도 (frame_stride_ms=0) 표본은 시각마다 하나다."""
    policy = load_policy(ROOT).depth.model_copy(update={"frame_stride_ms": 0})
    img = np.zeros((H, W, 3), dtype=np.uint8)
    frames = [(0, img), (0, img), (100, img), (100, img)]
    out = lift_tracks(frames, PlaneDepth(), [_left_hand("a", 80.0)], None, policy)
    assert [s.t_ms for s in out[("left_hand", "wrist")]] == [0, 100]


def test_coexisting_hand_versions_lift_from_the_reviewed_track(tmp_path: Path) -> None:
    """F-PAR-01 회귀: 승인된 이전 버전 왼손 트랙과 새 버전 왼손 트랙이 함께 현재 라벨이어도 3D
    라벨이 만들어지고, 승인 트랙의 위치로 올린다.

    정답 근거: 승인 트랙 관절 (80, 120)은 왼쪽 절반이라 z=1 → x=(80-160)*1/160=-0.5. 새 트랙
    (240, 120)이 골렸다면 z=2 → x=1.0이 된다. stride 200 ms → 0, 200, ..., 800 다섯 표본.
    """
    video = tmp_path / "bodycam.mp4"
    write_video(
        video,
        ((t, np.zeros((H, W, 3), dtype=np.uint8)) for t in range(0, 1_000, 100)),
        width=W,
        height=H,
    )
    approved = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="rev01", reviewed_at=NOW
    )
    reviewed = _left_hand("left-v1", 80.0, verification=approved)
    newer = _left_hand("left-v2", 240.0, created_at=NOW.replace(day=8))
    base = load_policy(ROOT)
    # 손가락 끝까지 픽셀 깊이로 올려 관절 수만큼 궤적을 본다 (world 없는 트랙)
    depth = base.depth.model_copy(update={"fingertip_fallback": "point_depth"})
    policy = base.model_copy(update={"depth": depth})
    lifter = PlaneLifter(policy)
    calib = CameraIntrinsics(width=W, height=H, fx=160, fy=160, cx=160, cy=120)
    labels = lifter.run(
        video,
        session_id="s",
        stream_id="bodycam",
        tracks=pick_lift_tracks([newer, reviewed], "bodycam"),
        calib=calib,
        ontology_version="1.0.0",
    )
    assert len(labels) == len(policy.depth.hand_points)
    for x in labels:
        assert isinstance(x.payload, Trajectory3DPayload)
        assert [s.t_ms for s in x.payload.samples] == [0, 200, 400, 600, 800]
        assert {(s.x, s.z) for s in x.payload.samples} == {(-0.5, 1.0)}
        # 올린 2D 트랙이 출처로 남는다 (3D가 그 트랙의 검수를 물려받는 근거, ADR 0048)
        assert x.payload.derived_from == ("left-v1",)
