"""순수 기하·대응·분류 시험 (합성 정답, ADR 0066)."""

from __future__ import annotations

import numpy as np

from dlp_fixtures.scene3d import generate_scene_pair
from dlp_scene3d.alignment import align_scenes
from dlp_scene3d.change import compare_objects
from dlp_scene3d.geometry import Camera, Similarity, ransac_similarity, triangulate, umeyama
from dlp_scene3d.policy import load_policy
from dlp_schema import repo_root
from dlp_schema.scene3d import ObjectInstance

POLICY = load_policy(repo_root())


def _rot(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def test_umeyama_recovers_known_similarity() -> None:
    """알려진 닮음 변환(척도 포함)을 정확히 되찾는다."""
    rng = np.random.default_rng(1)
    src = rng.normal(size=(50, 3))
    truth = Similarity(1.7, _rot(40), np.array([0.3, -2.0, 1.0]))
    fit = umeyama(src, truth.apply(src))
    assert abs(fit.s - 1.7) < 1e-9
    assert np.allclose(fit.r, truth.r) and np.allclose(fit.t, truth.t)
    back = fit.inverse().apply(truth.apply(src))
    assert np.allclose(back, src)


def test_ransac_rejects_moved_points() -> None:
    """30% 이상점(움직인 물체 점)이 있어도 내점만으로 변환을 찾는다."""
    rng = np.random.default_rng(2)
    src = rng.uniform(-3, 3, size=(100, 3))
    truth = Similarity(0.8, _rot(-25), np.array([1.0, 0.5, 0.0]))
    dst = truth.apply(src)
    dst[:30] += rng.uniform(0.3, 1.0, size=(30, 3))
    res = ransac_similarity(src, dst, threshold=0.01, iterations=200, seed=0)
    assert res.transform is not None
    assert set(res.inliers.tolist()) == set(range(30, 100))
    assert abs(res.transform.s - 0.8) < 1e-6


def test_triangulate_two_and_many_views() -> None:
    """다시점 DLT가 정답 점을 되찾는다."""
    pair = generate_scene_pair(seed=3)
    cams = [Camera(c.name, c.k, c.r, c.t, 640, 480) for c in pair.before.cameras]
    x = np.array([0.2, -0.3, 0.4])
    uv = np.stack([c.project(x[None])[0] for c in cams])
    assert np.allclose(triangulate(cams[:2], uv[:2]), x, atol=1e-6)
    assert np.allclose(triangulate(cams, uv), x, atol=1e-6)


def test_scene_alignment_recovers_fixture_transform() -> None:
    """고정 구조 점 대응으로 after → before 변환(픽스처 정답)을 찾고, 움직인 물체 점은 빠진다."""
    pair = generate_scene_pair(seed=0)
    al, fit = align_scenes(
        pair.before.landmarks,
        pair.before.descriptors,
        pair.after.landmarks,
        pair.after.descriptors,
        POLICY.alignment,
        "t",
    )
    s, r, t = pair.after_to_before
    assert al.accepted and fit is not None
    assert (
        abs(fit.s - s) < 1e-3
        and np.allclose(fit.r, r, atol=1e-3)
        and np.allclose(fit.t, t, atol=1e-3)
    )
    assert al.rms_m is not None and al.rms_m < 1e-3


def test_alignment_refuses_without_static_structure() -> None:
    """대응이 무작위면(고정 구조 없음) 정렬을 거부한다."""
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=(80, 3)), rng.normal(size=(80, 3))
    desc = rng.normal(size=(80, 32))
    al, fit = align_scenes(b, desc, a, desc, POLICY.alignment, "t")
    assert fit is None and not al.accepted and al.reason


def _inst(
    iid: str, label: str, pos: tuple[float, float, float], app: tuple[float, ...]
) -> ObjectInstance:
    return ObjectInstance(
        instance_id=iid,
        phase="before" if iid.startswith("b") else "after",
        label=label,
        position=pos,
        size=0.4,
        appearance=app,
        views=4,
        photo_ids=("p1", "p2"),
        reproj_px=0.5,
        confidence=0.9,
    )


def test_change_classification_kinds() -> None:
    """같은 라벨 두 개는 외형으로 구분해 대응하고, 이동·그대로·치움·새로 둠을 나눈다."""
    red, blue, green = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    before = [
        _inst("b1", "bucket", (0, 0, 0), red),
        _inst("b2", "bucket", (1, 0, 0), blue),
        _inst("b3", "towel", (2, 2, 0), green),
    ]
    after = [
        _inst("a1", "bucket", (1.02, 0, 0), red),
        _inst("a2", "bucket", (1, 0.01, 0), blue),
        _inst("a3", "cup", (0, 3, 0), green),
    ]
    got = {
        (c.kind, c.before_instance_id, c.after_instance_id)
        for c in compare_objects(before, after, POLICY.change)
    }
    assert got == {
        ("moved", "b1", "a1"),
        ("unchanged", "b2", "a2"),
        ("removed", "b3", None),
        ("added", None, "a3"),
    }
