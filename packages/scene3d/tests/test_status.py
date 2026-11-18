"""`dlp scene3d status`의 상태 읽기 시험: 전후가 다 갖춰졌는지, 읽기 전용인지 (ADR 0067).

정답은 시험이 저장소에 만들어 둔 단계다 (합성 방, CPU stub).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dlp_fixtures.scene3d import generate_ego_scene, write_scene_pair
from dlp_scene3d.localize import localize
from dlp_scene3d.oracle import load_oracle, oracle_features, oracle_objects
from dlp_scene3d.pipeline import compare
from dlp_scene3d.policy import Scene3dPolicy
from dlp_scene3d.privacy_gate import OraclePhotoDetector, detect_capture
from dlp_scene3d.status import format_status, scene_status

from .world import (
    NOW,
    POLICY,
    PRIVACY,
    RAW_BUCKET,
    Stores,
    bodycam_session,
    build_scene,
    ego_intrinsics,
    gate,
    open_stores,
)

SID = "s-status"


@pytest.fixture(scope="module")
def truth(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """바디캠 경로가 있는 합성 장면의 truth.json."""
    pair, ego = generate_ego_scene(0)
    return write_scene_pair(pair, tmp_path_factory.mktemp("status") / "photos", ego=ego)


def _snapshot(root: Path) -> dict[str, str]:
    """저장소의 모든 파일 → 내용 해시 (읽기 전용 확인)."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _status(stores: Stores, policy: Scene3dPolicy = POLICY):
    return scene_status(stores.labeling, SID, policy, stores.raw)


def test_status_follows_the_pipeline_and_never_writes(truth: Path, tmp_path: Path) -> None:
    """단계를 하나씩 밟으며 빠진 것이 줄어든다. 전후 복원과 받아들인 정렬이 있어야 갖춰진 것이고,
    상태를 읽는 동안 저장소 파일과 원본 접근 기록이 그대로다."""
    stores = open_stores(tmp_path / "store")
    empty = _status(stores)
    assert not empty.ready and empty.before.photos is None and empty.before.detected is False
    assert [m.split(":")[0] for m in empty.missing] == [
        "before 사진 수집 없음",
        "after 사진 수집 없음",
    ]

    build_scene(truth, stores, SID, tmp_path / "w", phases=("before",))
    half = _status(stores)
    assert half.before.recon == "current" and half.before.render == "current"
    assert half.before.photos == 14 and half.before.detected and half.before.approved
    assert half.before.reconstruction is not None and half.before.reconstruction.metric
    assert not half.ready and len(half.missing) == 1 and "after 사진 수집 없음" in half.missing[0]

    gate(truth, stores, SID, "after", tmp_path / "w")
    rendered = _status(stores)
    assert rendered.after.render == "current" and rendered.after.recon == "missing"
    assert "dlp scene3d reconstruct s-status after" in rendered.missing[0]

    build_scene(truth, stores, SID, tmp_path / "w2", phases=("after",))
    both = _status(stores)
    assert both.after.recon == "current" and both.alignment is None
    assert not both.ready and both.missing[0].startswith("정렬 없음")
    assert (both.change, both.localization) == ("missing", "missing")

    compare(stores.labeling, SID, oracle_objects(truth), POLICY, tmp_path / "c", NOW)
    engine = bodycam_session(stores, SID, truth, tmp_path / "db", intrinsics=ego_intrinsics(truth))
    localize(
        engine, stores.labeling, SID, oracle_features(truth), POLICY, PRIVACY, tmp_path / "l", NOW,
        raw_bucket=RAW_BUCKET,
    )  # fmt: skip
    files, events = _snapshot(stores.root), len(stores.sink.events)
    done = _status(stores)
    assert _snapshot(stores.root) == files and len(stores.sink.events) == events
    assert done.ready and done.missing == [] and done.notes == []
    assert done.alignment is not None and done.alignment.accepted
    assert (done.change, done.localization) == ("current", "current")
    text = "\n".join(format_status(done))
    assert "| 수집 | 사진 14장 | 사진 14장 |" in text and "갖춰짐" in text
    assert "| 척도 | 미터 (SfM 결과: 마커를 찾지 못했습니다) | 미터 아님 (척도 없음" in text
    assert "| 위치 추정 | 있음: 프레임" in text and "after 기준" in text


def test_status_reports_rejected_alignment_and_stale_products(truth: Path, tmp_path: Path) -> None:
    """정렬 정책이 바뀌어 거부되면 갖춰지지 않은 것이다. 다시 탐지해 승인이 낡으면 그 단계의 렌더가
    무효, 복원·비교가 낡음으로 나온다."""
    stores = open_stores(tmp_path / "store")
    build_scene(truth, stores, SID, tmp_path / "w")
    compare(stores.labeling, SID, oracle_objects(truth), POLICY, tmp_path / "c", NOW)
    assert _status(stores).ready

    strict = Scene3dPolicy.model_validate(
        {
            **POLICY.model_dump(mode="json"),
            "alignment": {**POLICY.alignment.model_dump(mode="json"), "min_inliers": 100_000},
        }
    )
    # 새 정렬 정책의 정렬은 아직 없다 (예전 정책의 정렬을 지금 것으로 보지 않는다)
    assert _status(stores, strict).alignment is None
    with pytest.raises(Exception, match="정렬 거부"):
        compare(stores.labeling, SID, oracle_objects(truth), strict, tmp_path / "c2", NOW)
    rejected = _status(stores, strict)
    assert rejected.alignment is not None and not rejected.alignment.accepted
    assert not rejected.ready and rejected.missing[0].startswith("정렬 거부")

    other = OraclePhotoDetector(load_oracle(truth, "before").privacy.boxes, version="oracle-2")
    detect_capture(
        stores.raw, stores.labeling, SID, "before", {"oracle": other}, [], PRIVACY,
        tmp_path / "d", NOW,
    )  # fmt: skip
    stale = _status(stores)
    assert stale.before.render == "invalid" and stale.before.recon == "stale"
    assert stale.after.recon == "current" and stale.alignment is None
    assert stale.change == "stale" and not stale.ready
    assert stale.missing == [
        "before 승인된 블러 사진 무효 (재탐지로 승인이 낡음): "
        "dlp scene3d privacy detect → review → approve → render s-status before"
    ]
    text = "\n".join(format_status(stale))
    assert "| 블러 렌더 | 무효 (재탐지로 승인이 낡음) | 유효" in text
    assert "빠짐: before 승인된 블러 사진 무효" in text and "(낡음: 복원이 바뀜)" in text


def test_status_without_raw_store_and_non_metric_note(truth: Path, tmp_path: Path) -> None:
    """원본 저장소를 주지 않으면 블러 탐지·승인 표시는 확인하지 않는다. before 복원이 미터가 아니면
    갖춤 판정과 별개로 알린다."""
    import json

    data = json.loads(truth.read_text("utf-8"))
    data["sfm_unit_scale"] = 2.0
    scaled = truth.parent / "truth-scaled.json"
    scaled.write_text(json.dumps(data), "utf-8")
    stores = open_stores(tmp_path / "store")
    build_scene(scaled, stores, SID, tmp_path / "w")
    status = scene_status(stores.labeling, SID, POLICY)
    assert status.before.detected is None and status.before.approved is None
    assert status.before.recon == "current" and status.before.reconstruction is not None
    assert not status.before.reconstruction.metric
    assert len(status.notes) == 1 and "미터가 아닙니다 (마커를 찾지 못했습니다)" in status.notes[0]
    text = "\n".join(format_status(status))
    assert "| 블러 탐지 | 확인 안 함 | 확인 안 함 |" in text and "주의: before 복원이" in text
