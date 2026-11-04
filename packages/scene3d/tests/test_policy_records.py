"""정책·라이선스·기록 표 시험 (ADR 0066)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import sqlalchemy as sa
import yaml

from dlp_cli.models_cmds import used_models
from dlp_models.registry import load_registry
from dlp_scene3d.policy import load_policy
from dlp_schema import repo_root
from dlp_schema.db.scene3d_repository import record_scene3d, scene3d_records_for
from dlp_schema.db.tables import metadata
from dlp_schema.scene3d import Scene3dRecord

ROOT = repo_root()


def test_object_queries_are_ontology_objects() -> None:
    """물체 질의 라벨은 온톨로지 objects 키다 (검수·내보내기에서 같은 사전을 쓴다)."""
    data = cast(
        dict[str, Any],
        yaml.safe_load((ROOT / "config/ontology/v1/objects.yaml").read_text("utf-8")),
    )
    assert set(load_policy(ROOT).objects.queries) <= set(cast(dict[str, Any], data["objects"]))


def test_scene3d_tools_are_license_checked_and_inria_is_forbidden() -> None:
    """SfM·스플랫 도구도 라이선스 검사를 받고, Inria 원본 3DGS를 고르면 위반이다."""
    registry = load_registry(ROOT)
    used = used_models(ROOT)
    assert used["scene3d.reconstruction.splat_tool"] == "gsplat"
    assert registry.violations(used) == []
    bad = registry.violations({"x": "graphdeco-inria/gaussian-splatting"})
    assert bad and "forbidden" in bad[0]


def test_records_are_idempotent() -> None:
    """같은 (종류, 태그, 키) 기록은 한 번만 남는다 (멱등 재실행이 기록을 늘리지 않는다)."""
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)
    rec = Scene3dRecord(
        session_id="s1",
        phase="before",
        kind="reconstruction",
        model_version="v",
        tag="t",
        bucket="labeling",
        storage_key="k",
        actor="a",
        created_at=datetime(2026, 11, 4, tzinfo=UTC),
    )
    with engine.begin() as conn:
        assert record_scene3d(conn, rec) is True
        assert record_scene3d(conn, rec) is False
        assert scene3d_records_for(conn, "s1") == [rec]
