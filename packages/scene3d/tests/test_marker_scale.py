"""마커로 미터 척도 정하기 시험 (합성 방 사진에 원근에 맞게 그린 ArUco 마커, ADR 0067).

정답: 마커의 실제 한 변(0.3 m)과, 정답 카메라·점에 곱한 임의 배율(3.7). 마커 척도가 그 배율의 역수를
되찾아야 한다. 마커가 없거나 추정이 불안정하면 척도를 지어내지 않아야 한다 (metric=False와 이유).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlp_fixtures.scene3d import ScenePair, floor_marker, generate_scene_pair, write_scene_pair
from dlp_scene3d.backends import InitPointsSplat, SfmResult
from dlp_scene3d.geometry import Camera
from dlp_scene3d.markers import MarkerObservation, detect_markers, estimate_scale, scale_sfm
from dlp_scene3d.oracle import load_oracle, oracle_objects
from dlp_scene3d.pipeline import (
    Scene3dError,
    compare,
    current_reconstruction_record,
    reconstruct,
)
from dlp_scene3d.policy import MarkerPolicy

from .world import NOW, POLICY, Stores, build_scene, gate, open_stores, with_marker

SID = "s-marker"
SIDE_M = 0.3
UNIT_SCALE = 3.7
MARKER = floor_marker(7, (0.15, 0.2), SIDE_M, 25.0)
MARKED = with_marker(POLICY, side_m=SIDE_M)
MP = MARKED.reconstruction.marker


@pytest.fixture(scope="module")
def pair() -> ScenePair:
    """바닥에 마커 하나를 둔 전후 장면."""
    return generate_scene_pair(seed=0, markers=[MARKER])


def _observe(
    pair: ScenePair, unit: float, policy: MarkerPolicy = MP
) -> tuple[list[MarkerObservation], dict[str, Camera]]:
    """before 사진의 마커 관측과, 길이 단위에 unit을 곱한 정답 카메라."""
    cams = {c.name: Camera(c.name, c.k, c.r, c.t * unit, 640, 480) for c in pair.before.cameras}
    obs = [
        MarkerObservation(name, marker_id, quad)
        for name in sorted(cams)
        for marker_id, quad in detect_markers(pair.before.images[name], policy)
    ]
    return obs, cams


def test_marker_corners_are_found_where_they_were_drawn(pair: ScenePair) -> None:
    """마커를 찾은 사진에서 네 변의 화소 길이가 정답 투영과 2% 안에서 맞는다 (모서리 다듬기
    apriltag는 모서리 전체를 같은 쪽으로 반 화소쯤 옮기지만 변 길이는 그대로다)."""
    obs, cams = _observe(pair, 1.0)
    assert len(obs) >= MP.min_photos and {o.marker_id for o in obs} == {7}
    for o in obs:
        truth = cams[o.photo_id].project(MARKER.corners)
        want = np.linalg.norm(truth - np.roll(truth, -1, axis=0), axis=1)
        got = np.linalg.norm(o.corners - np.roll(o.corners, -1, axis=0), axis=1)
        assert np.abs(got / want - 1).max() < 0.02
        assert np.linalg.norm(o.corners - truth, axis=1).max() < 1.0


def test_scale_recovers_arbitrary_sfm_unit(pair: ScenePair) -> None:
    """카메라 좌표에 3.7을 곱한(미터가 아닌) 복원에서 마커 척도가 1/3.7을 1% 안에서 되찾는다."""
    obs, cams = _observe(pair, UNIT_SCALE)
    est = estimate_scale(obs, cams, MP)
    assert est.factor is not None and est.reason is None
    assert est.factor * UNIT_SCALE == pytest.approx(1.0, abs=0.01)
    assert est.marker_ids == (7,) and est.photos >= MP.min_photos
    assert est.side_spread is not None and est.side_spread < 0.02


def test_scaled_sfm_keeps_pixels(pair: ScenePair) -> None:
    """척도를 곱한 SfM 결과는 같은 점을 같은 화소에 찍고 길이만 바뀐다."""
    _, cams = _observe(pair, UNIT_SCALE)
    points = pair.before.landmarks * UNIT_SCALE
    result = SfmResult(
        cams, points, pair.before.descriptors, np.zeros((len(points), 3), np.uint8), False
    )
    scaled = scale_sfm(result, 1 / UNIT_SCALE)
    assert scaled.metric and np.allclose(scaled.points, pair.before.landmarks)
    name = sorted(cams)[0]
    front = cams[name].depth(points) > 0
    assert np.allclose(
        scaled.cameras[name].project(scaled.points)[front], cams[name].project(points)[front]
    )
    assert np.allclose(scaled.cameras[name].center * UNIT_SCALE, cams[name].center)


def test_occluded_view_is_dropped_not_averaged(pair: ScenePair) -> None:
    """한 사진의 모서리가 틀리면(가려진 마커) 그 사진만 빼고 같은 척도를 낸다."""
    obs, cams = _observe(pair, UNIT_SCALE)
    clean = estimate_scale(obs, cams, MP)
    bad = MarkerObservation(obs[0].photo_id, 7, obs[0].corners + np.array([[9.0, -7.0]] * 4))
    est = estimate_scale([bad, *obs[1:]], cams, MP)
    assert est.factor is not None and clean.factor is not None
    assert est.photos == clean.photos - 1
    assert est.factor == pytest.approx(clean.factor, rel=0.01)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"min_photos": 14}, "모서리가 맞는 사진"),
        ({"max_side_spread": 1e-7}, "변 길이 흩어짐"),
        ({"min_parallax_deg": 179.0}, "시선 사이 각"),
        ({"ids": [3]}, "마커를 찾지 못했습니다"),
        ({"dictionary": "DICT_6X6_250"}, "마커를 찾지 못했습니다"),
    ],
)
def test_unstable_or_missing_marker_gives_no_scale(
    pair: ScenePair, change: dict[str, object], reason: str
) -> None:
    """사진 수 미달, 변 길이 흩어짐 초과, 시차 부족, 허용 ID·사전 밖이면 척도를 내지 않는다."""
    policy = with_marker(MARKED, **change).reconstruction.marker
    obs, cams = _observe(pair, UNIT_SCALE, policy)
    est = estimate_scale(obs, cams, policy)
    assert est.factor is None and est.reason is not None and reason in est.reason


def test_same_marker_id_twice_in_one_photo_is_not_used(pair: ScenePair) -> None:
    """한 사진에 같은 ID가 둘이면(같은 인쇄물 두 장) 어느 것인지 알 수 없어 그 사진에서는 쓰지
    않는다."""
    name = next(n for n in sorted(pair.before.images) if detect_markers(pair.before.images[n], MP))
    image = pair.before.images[name]
    double = np.concatenate([image, image], axis=1)
    assert detect_markers(image, MP) and detect_markers(double, MP) == []


@pytest.fixture
def stores(tmp_path: Path) -> Stores:
    return open_stores(tmp_path / "store")


def test_marker_scale_makes_non_metric_sfm_metric_and_moves_match(
    pair: ScenePair, stores: Stores, tmp_path: Path
) -> None:
    """미터가 아닌 SfM(배율 3.7) + 마커 → 두 단계 모두 metric이 되고, 그 척도로 한 전후 비교의 이동
    벡터가 정답(미터)과 2 cm 안에서 맞는다. 정렬의 척도는 1에 가깝다 (두 단계가 모두 미터)."""
    truth = write_scene_pair(pair, tmp_path / "photos", sfm_unit_scale=UNIT_SCALE)
    build_scene(truth, stores, SID, tmp_path / "w", MARKED)
    lab = stores.labeling
    _, before = current_reconstruction_record(lab, SID, "before")
    _, after = current_reconstruction_record(lab, SID, "after")
    s_after = pair.after_to_before[0]
    for rec, want in ((before, 1 / UNIT_SCALE), (after, s_after / UNIT_SCALE)):
        assert rec.metric and rec.scale is not None and rec.scale.source == "marker"
        assert rec.scale.factor == pytest.approx(want, rel=0.01)
        assert rec.scale.marker_ids == (7,) and rec.scale.marker_side_m == SIDE_M
        assert rec.scale.marker_photos >= MP.min_photos and rec.scale.reason is None

    report, _ = compare(lab, SID, oracle_objects(truth), MARKED, tmp_path / "c", NOW)
    assert report.metric and report.alignment.accepted
    assert report.alignment.scale == pytest.approx(1.0, abs=0.01)
    centers = {o.key: o.center for o in pair.before.objects}
    moved = 0
    for c in report.changes:
        if c.kind != "moved":
            continue
        assert c.before_position is not None and c.displacement is not None
        at = np.asarray(c.before_position, dtype=np.float64)
        key = min(centers, key=lambda k, at=at: float(np.linalg.norm(centers[k] - at)))
        assert np.linalg.norm(centers[key] - at) < 0.05
        assert np.linalg.norm(np.subtract(c.displacement, pair.moves[key])) < 0.02, key
        moved += 1
    assert moved == len(pair.moves)
    assert sorted(c.kind for c in report.changes if c.kind != "moved") == [
        "added",
        "removed",
        "unchanged",
        "unchanged",
    ]


def test_without_marker_scale_is_not_invented(stores: Stores, tmp_path: Path) -> None:
    """마커가 없는 사진의 미터 아닌 복원은 metric=False로 남고(이유 기록) 비교가 거부된다."""
    plain = generate_scene_pair(seed=0)
    truth = write_scene_pair(plain, tmp_path / "photos", sfm_unit_scale=UNIT_SCALE)
    build_scene(truth, stores, SID, tmp_path / "w", MARKED)
    _, before = current_reconstruction_record(stores.labeling, SID, "before")
    assert not before.metric and before.scale is not None
    assert before.scale.source == "none" and before.scale.factor == 1.0
    assert before.scale.reason == "마커를 찾지 못했습니다" and before.scale.marker_factor is None
    with pytest.raises(Scene3dError, match=r"미터 단위가 아닙니다.*마커를 찾지 못했습니다"):
        compare(stores.labeling, SID, oracle_objects(truth), MARKED, tmp_path / "c", NOW)


def test_metric_sfm_is_left_alone_and_checked_against_marker(
    pair: ScenePair, stores: Stores, tmp_path: Path
) -> None:
    """이미 미터인 SfM 결과(정답 stub)는 건드리지 않고 마커 배율이 1에 가까운지만 확인한다. 마커
    한 변 길이를 틀리게 적으면(0.25 m) 배율이 1에서 벗어나 복원을 거부한다."""
    truth = write_scene_pair(pair, tmp_path / "photos")
    gate(truth, stores, SID, "before", tmp_path / "g")
    sfm, splat = load_oracle(truth, "before").sfm, InitPointsSplat(0.01)
    lab = stores.labeling
    rec, _ = reconstruct(lab, SID, "before", sfm, splat, MARKED, tmp_path / "r1", NOW)
    assert rec.metric and rec.scale is not None and rec.scale.source == "sfm"
    assert rec.scale.factor == 1.0 and rec.scale.marker_factor == pytest.approx(1.0, abs=0.01)
    with pytest.raises(Scene3dError, match="마커 척도 배율"):
        reconstruct(
            lab, SID, "before", sfm, splat, with_marker(MARKED, side_m=0.25), tmp_path / "r2", NOW
        )
    off, _ = reconstruct(
        lab, SID, "before", sfm, splat, with_marker(MARKED, enabled=False), tmp_path / "r3", NOW
    )
    assert off.metric and off.scale is not None and off.scale.marker_side_m is None
    assert off.scale.reason is not None and "쓰지 않습니다" in off.scale.reason
    with pytest.raises(Scene3dError, match="마커 사전"):
        reconstruct(
            lab, SID, "before", sfm, splat, with_marker(MARKED, dictionary="DICT_NOPE"),
            tmp_path / "r4", NOW,
        )  # fmt: skip
