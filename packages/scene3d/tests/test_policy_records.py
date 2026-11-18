"""정책·라이선스·기록 표·격리 환경 어댑터 시험 (ADR 0066, 0067)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest
import sqlalchemy as sa
import yaml

from dlp_cli.models_cmds import used_models
from dlp_models.registry import load_registry
from dlp_scene3d import backends
from dlp_scene3d.backends import (
    FrameImage,
    GsplatTrainer,
    IsolatedFrameFeatures,
    IsolatedSfm,
)
from dlp_scene3d.policy import load_policy, section_digest
from dlp_scene3d.privacy_gate import BlurredPhoto
from dlp_schema import repo_root
from dlp_schema.db.scene3d_repository import record_scene3d, scene3d_records_for
from dlp_schema.db.tables import metadata
from dlp_schema.predictor import ModelUnavailableError
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


def test_frame_feature_tool_is_license_checked() -> None:
    """위치 추정의 프레임 특징 추출 도구(pycolmap)도 라이선스 검사를 받는다 (ADR 0067)."""
    used = used_models(ROOT)
    assert used["scene3d.localization.features_tool"] == "colmap"
    assert load_registry(ROOT).violations(used) == []


def test_interpolation_limit_is_not_part_of_any_stage_version() -> None:
    """읽을 때만 쓰는 보간 한계는 위치 추정 절 밖에 있다: 바꿔도 위치 추정 절 해시가 그대로다."""
    policy = load_policy(ROOT)
    data = policy.model_dump(mode="json")
    data["interpolation"]["max_gap_ms"] = policy.interpolation.max_gap_ms + 1000
    changed = type(policy).model_validate(data)
    assert section_digest(changed.localization) == section_digest(policy.localization)
    assert "max_gap_ms" not in policy.localization.model_dump()


def test_isolated_adapters_convert_pixel_convention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """격리 환경(COLMAP·gsplat: 왼쪽 위 화소의 중심이 (0.5, 0.5))과 주고받는 주점·특징 위치를
    작업공간 규약(중심이 (0, 0))으로 바꾼다. 스크립트 대신 같은 파일을 쓰는 가짜를 끼운다."""
    sent: dict[str, Any] = {}

    def fake_run(env: Path, script: str, args: list[str], timeout_s: int) -> None:
        out = Path(args[args.index("--out") + 1])
        if script == "run_sfm.py":
            out.mkdir(parents=True)
            pose = {
                "photo_id": "p0", "width": 640, "height": 480, "fx": 500.0, "fy": 501.0,
                "cx": 320.5, "cy": 240.5, "rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "translation": [0.1, 0.2, 0.3],
            }  # fmt: skip
            (out / "cameras.json").write_text(json.dumps([pose]), "utf-8")
            np.savez(
                out / "points.npz",
                points=np.zeros((2, 3)),
                descriptors=np.zeros((2, 128)),
                colors=np.zeros((2, 3), np.uint8),
            )
        elif script == "train_splat.py":
            sfm_dir = Path(args[args.index("--sfm") + 1])
            sent["cameras"] = json.loads((sfm_dir / "cameras.json").read_text("utf-8"))
            out.write_bytes(b"ply")
        else:
            images = Path(args[args.index("--images") + 1])
            sent["frames"] = sorted(p.name for p in images.iterdir())
            sent["max_features"] = args[args.index("--max-features") + 1]
            # 둘째 프레임(2000)은 결과에 없다: 특징이 없는 프레임으로 본다
            np.savez(
                out,
                kp_1000=np.array([[10.5, 20.5], [0.5, 0.5]], np.float32),
                desc_1000=np.full((2, 128), 7, np.uint8),
            )

    monkeypatch.setattr(backends, "_run_env", fake_run)
    photo = tmp_path / "p0.png"
    cv2.imwrite(str(photo), np.zeros((480, 640, 3), np.uint8))
    photos = [BlurredPhoto("p0", photo, 640, 480)]
    result = IsolatedSfm(tmp_path / "env").reconstruct(photos, tmp_path / "sfm")
    cam = result.cameras["p0"]
    assert (cam.k[0, 2], cam.k[1, 2]) == (320.0, 240.0) and not result.metric
    assert (cam.k[0, 0], cam.k[1, 1]) == (500.0, 501.0) and cam.t.tolist() == [0.1, 0.2, 0.3]

    (tmp_path / "out").mkdir()
    GsplatTrainer(tmp_path / "env", 10).train(photos, result, tmp_path / "out" / "splat.ply")
    assert (sent["cameras"][0]["cx"], sent["cameras"][0]["cy"]) == (320.5, 240.5)

    frames = [FrameImage(t, np.zeros((48, 64, 3), np.uint8)) for t in (1000, 2000)]
    features = IsolatedFrameFeatures(tmp_path / "env", 123).extract(frames, tmp_path / "feat")
    assert sent["frames"] == ["1000.png", "2000.png"] and sent["max_features"] == "123"
    assert set(features) == {1000}
    assert features[1000].keypoints.tolist() == [[10.0, 20.0], [0.0, 0.0]]
    assert features[1000].descriptors.shape == (2, 128)
    assert features[1000].descriptors.dtype == np.float64
    assert not (tmp_path / "feat" / "frames").exists()


def test_isolated_frame_features_need_the_environment(tmp_path: Path) -> None:
    """격리 환경이 없으면 ModelUnavailableError다 (CLI 종료 코드 4). stub으로 조용히 내려가지
    않는다."""
    frames = [FrameImage(0, np.zeros((48, 64, 3), np.uint8))]
    with pytest.raises(ModelUnavailableError, match="격리 환경이 없습니다"):
        IsolatedFrameFeatures(tmp_path / "no-env", 10).extract(frames, tmp_path / "feat")
