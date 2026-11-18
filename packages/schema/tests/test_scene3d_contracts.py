"""청소 전후 3D·바디캠 위치 계약의 규칙 테스트 (ADR 0066·0067).

척도 출처(`SceneScale`)와 복원 기록의 일관성, 예전 복원 기록(JSON)의 호환, 위치 기록
(`EgoLocalization`)의 검증, 산출물 기록 종류와 표의 열을 본다.

정답 근거: 계약 정의 자체 (각 검증기가 거부해야 하는 최소 반례)와 `tables.scene3d_records`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, get_args

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema.db.scene3d_repository import record_scene3d, scene3d_records_for
from dlp_schema.db.tables import metadata, scene3d_records
from dlp_schema.scene3d import (
    Alignment,
    EgoLocalization,
    FramePose,
    Reconstruction,
    Scene3dRecord,
    Scene3dRecordKind,
    SceneScale,
)
from dlp_schema.session import CameraIntrinsics

NOW = datetime(2026, 11, 14, tzinfo=UTC)
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# ADR 0067 이전의 복원 기록 (scale 필드가 없다)
OLD_RECONSTRUCTION: dict[str, Any] = {
    "session_id": "s1",
    "phase": "before",
    "model_version": "scene3d-recon-1+sfm:oracle-sfm-1+splat:init-points-1+p0123456789",
    "render_tag": "r",
    "metric": True,
    "photo_count": 14,
    "registered": 14,
    "point_count": 372,
    "cameras_key": "c",
    "points_key": "p",
    "splat_key": "s",
    "created_at": "2026-11-04T00:00:00Z",
}


def _alignment(accepted: bool = True) -> Alignment:
    return Alignment(
        model_version="scene3d-align-1+p0",
        scale=1.0,
        rotation=IDENTITY,
        translation=(0.0, 0.0, 0.0),
        matches=10,
        inliers=10,
        rms_m=0.0,
        accepted=accepted,
        reason=None if accepted else "내점 부족",
    )


def _pose(t_ms: int) -> FramePose:
    return FramePose(
        t_ms=t_ms,
        reference="before",
        rotation=IDENTITY,
        translation=(0.0, 0.0, 0.0),
        matches=30,
        inliers=25,
        reproj_px=0.7,
    )


def _localization(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "session_id": "s1",
        "stream_id": "bodycam",
        "model_version": "scene3d-localize-1+feat:oracle-features-1+p0+scene3d-align-1+p0",
        "before_reconstruction": "b",
        "after_reconstruction": "a",
        "alignment_tag": "al",
        "alignment": _alignment(),
        "render_hash": "0" * 64,
        "intrinsics": CameraIntrinsics(width=640, height=480, fx=400, fy=400, cx=320, cy=240),
        "intrinsics_origin": "session",
        "metric": True,
        "frame_count": 3,
        "localized": 2,
        "poses": (_pose(0), _pose(200)),
        "created_at": NOW,
    }
    data.update(overrides)
    return data


def test_old_reconstruction_json_still_loads() -> None:
    """척도 출처가 없던 예전 복원 기록도 읽힌다 (scale은 None)."""
    rec = Reconstruction.model_validate(OLD_RECONSTRUCTION)
    assert rec.scale is None and rec.metric
    assert Reconstruction.model_validate_json(rec.model_dump_json()) == rec


def test_scale_source_must_agree_with_factor_and_metric() -> None:
    """마커로 정했으면 곱한 배율이 마커 배율이고, 아니면 좌표를 건드리지 않았어야 한다. 복원의
    metric은 척도 출처가 있을 때만 참이다."""
    marker = SceneScale(
        source="marker",
        factor=0.27,
        marker_side_m=0.16,
        marker_factor=0.27,
        marker_ids=(7,),
        marker_photos=9,
        side_spread=0.004,
    )
    assert Reconstruction.model_validate({**OLD_RECONSTRUCTION, "scale": marker}).scale == marker
    with pytest.raises(ValidationError, match="marker_factor와 같아야"):
        SceneScale(source="marker", factor=0.27, marker_factor=0.3)
    with pytest.raises(ValidationError, match="marker_factor와 같아야"):
        SceneScale(source="marker", factor=0.27)
    with pytest.raises(ValidationError, match="factor는 1이어야"):
        SceneScale(source="sfm", factor=0.9, marker_factor=0.9)
    none = SceneScale(source="none", factor=1.0, reason="마커를 찾지 못했습니다")
    with pytest.raises(ValidationError, match="척도 출처"):
        Reconstruction.model_validate({**OLD_RECONSTRUCTION, "scale": none})
    assert not Reconstruction.model_validate(
        {**OLD_RECONSTRUCTION, "metric": False, "scale": none}
    ).metric
    with pytest.raises(ValidationError, match="척도 출처"):
        Reconstruction.model_validate({**OLD_RECONSTRUCTION, "metric": False, "scale": marker})


def test_localization_round_trip_and_rules() -> None:
    """위치 기록은 JSON으로 왕복하고, 자세 수·시각 순서·받아들인 정렬 규칙을 지킨다."""
    record = EgoLocalization.model_validate(_localization())
    assert EgoLocalization.model_validate_json(record.model_dump_json()) == record
    assert record.blurred_sha256 is None
    for bad, message in (
        ({"localized": 3}, "poses 수와 같고"),
        ({"frame_count": 1}, "frame_count 이하"),
        ({"poses": (_pose(200), _pose(0))}, "시각 순"),
        ({"poses": (_pose(200), _pose(200))}, "같은 시각이 없어야"),
        ({"alignment": _alignment(accepted=False)}, "거부된 정렬"),
    ):
        with pytest.raises(ValidationError, match=message):
            EgoLocalization.model_validate(_localization(**bad))
    with pytest.raises(ValidationError):
        EgoLocalization.model_validate(_localization(render_hash="not-a-hash"))
    with pytest.raises(ValidationError):
        FramePose.model_validate({**_pose(0).model_dump(), "inliers": 3})
    with pytest.raises(ValidationError):
        FramePose.model_validate({**_pose(0).model_dump(), "t_ms": -1})


def test_new_record_kind_needs_no_migration() -> None:
    """기록 종류 `localization`은 표를 바꾸지 않고 들어간다: kind 열은 값 제약이 없는 문자열이고
    모든 종류 이름이 그 길이 안이다."""
    kind = scene3d_records.c.kind.type
    assert isinstance(kind, sa.String) and kind.length == 32
    assert "localization" in get_args(Scene3dRecordKind)
    assert max(len(k) for k in get_args(Scene3dRecordKind)) <= kind.length
    assert not [c for c in scene3d_records.constraints if isinstance(c, sa.CheckConstraint)]
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)
    rec = Scene3dRecord(
        session_id="s1",
        phase=None,
        kind="localization",
        model_version="scene3d-localize-1+feat:oracle-features-1+p0+scene3d-align-1+p0",
        tag="0123456789ab",
        bucket="labeling",
        storage_key="sessions/s1/scene3d/localization/0123456789ab.json",
        actor="svc",
        created_at=NOW,
    )
    with engine.begin() as conn:
        assert record_scene3d(conn, rec) is True
        assert record_scene3d(conn, rec) is False
        assert scene3d_records_for(conn, "s1") == [rec]
