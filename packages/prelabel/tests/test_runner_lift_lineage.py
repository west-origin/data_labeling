"""3D 보호는 출처 2D 계보로 정한다 (F-PAR-92, F-PAR-87의 남은 부분).

박스 번호(`<클래스>_<nn>`)는 모델 버전마다 다시 매겨, 승인된 이전 버전 박스 A와 새 버전 박스 B가
같은 cup_01로 다른 개체를 가리킬 수 있다. 3D 단계는 둘을 다른 칸으로 올린다 (`lift3d.lift_slots`).
A의 3D를 사람이 직접 검수했으면 A의 새 3D만 막아야 하고 B의 3D는 들어가야 한다.
SQLite + 로컬 저장소로 `run_prelabel`을 끝까지 돈다 (서비스 없이).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa
import yaml
from numpy.typing import NDArray

from dlp_fixtures.shapes import generate_shape_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import LocalStore
from dlp_prelabel.adapters.mediapipe_models import boxes_to_labels
from dlp_prelabel.common import BoxTrack, Image
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import run_prelabel
from dlp_schema.db.repository import get_labels, record_review, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BoxTrackPayload,
    LabelRecord,
    Trajectory3DPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.testing import FIXED_TIME, sqlite_db

ROOT = Path(__file__).resolve().parents[3]


class _UnitDepth:
    """깊이를 늘 1 m로 내는 가짜 깊이 모델."""

    def predict(self, image: Image) -> NDArray[np.float32]:
        """(H, W) 1 m 깊이 맵."""
        return np.ones(image.shape[:2], dtype=np.float32)


class _UnitLifter(DepthLifter):
    """가중치 없이 `_UnitDepth`로 도는 3D 단계 (판은 version으로 바꾼다)."""

    def __init__(self, policy: PrelabelPolicy, version: str) -> None:  # pyright: ignore[reportMissingSuperCall]
        """가중치 확인 대신 가짜 모델·버전을 둔다."""
        self.path = Path("unused.onnx")
        self.weights_version = "unit"
        self.version = f"depth3d-{version}+p{policy.digest('depth')}"
        self.policy, self.now = policy, FIXED_TIME
        self.model = _UnitDepth()


class _Cups:
    """정해진 컵 박스 트랙을 내는 객체 예측기 (버전마다 첫 등장 순 번호, `boxes_to_labels`)."""

    name = "objects"

    def __init__(self, version: str, x: float, times: list[int]) -> None:
        """version: 모델 버전. x: 멈춰 있는 컵 박스의 왼쪽 위치."""
        self.version = version
        self.track = BoxTrack("cup", {t: ((x, 100.0, 40.0, 40.0), 0.9) for t in times})

    def run(self, clip: Clip) -> list[LabelRecord]:
        """박스 트랙 → box_track 라벨 (cup_01)."""
        return boxes_to_labels(
            [self.track], clip, self.version, "1.0.0", FIXED_TIME, prefix="objects"
        )


@pytest.fixture
def ontology() -> Ontology:
    """저장소 온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


def _session(tmp_path: Path, ontology: Ontology) -> tuple[sa.Engine, LocalStore, list[int]]:
    """정답 모양 영상 하나짜리 세션(shapes)을 SQLite·로컬 원본 저장소에 수집한다."""
    sc = generate_shape_scenario(0)
    sc.write(tmp_path / "bodycam.mp4")
    engine = sqlite_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as conn:
        register_ontology(conn, ontology)
    manifest = {
        "session_id": "shapes", "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(),
        "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    with engine.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    return engine, raw, sc.frame_times


def _current(conn: sa.Connection, kind: str) -> list[LabelRecord]:
    """세션 shapes의 현재 라벨 (종류 하나)."""
    return current_labels(get_labels(conn, "shapes", kinds=[kind]))


def test_reviewed_3d_of_one_box_track_does_not_block_a_same_numbered_other_track(
    tmp_path: Path, ontology: Ontology
) -> None:
    """F-PAR-92: 승인된 v1 박스 A(cup_01)의 3D를 사람이 직접 검수한 뒤, v2가 다른 자리의 컵 B를
    같은 cup_01로 내면 B의 3D는 새로 들어간다 (예전에는 (개체, 부위, 좌표계)가 같아 막혔다).
    A의 검수된 3D는 그대로 남고, 새 판으로 다시 계산한 A의 3D만 막힌다 (lift_blocked 1).
    """
    policy = load_policy(ROOT)
    engine, raw, times = _session(tmp_path, ontology)
    with engine.begin() as conn:
        run_prelabel(
            conn, "shapes", raw, [_Cups("objs-v1", 40.0, times)], policy, ontology,
            FIXED_TIME, lifter=_UnitLifter(policy, "a"),
        )  # fmt: skip
        (box_a,) = _current(conn, "box_track")
        (traj_a,) = _current(conn, "trajectory3d")
        record_review(conn, box_a.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        record_review(conn, traj_a.label_id, VerificationState.HUMAN_CORRECTED, "lead", FIXED_TIME)
    later = FIXED_TIME + timedelta(minutes=1)
    with engine.begin() as conn:
        s = run_prelabel(
            conn, "shapes", raw, [_Cups("objs-v2", 220.0, times)], policy, ontology,
            later, lifter=_UnitLifter(policy, "b"),
        )  # fmt: skip
        boxes = _current(conn, "box_track")
        trajs = _current(conn, "trajectory3d")
    assert {x.payload.entity_id for x in boxes if isinstance(x.payload, BoxTrackPayload)} == {
        "cup_01"
    }
    (box_b,) = [x for x in boxes if x.label_id != box_a.label_id]
    sources = {
        x.label_id: x.payload.derived_from or ()
        for x in trajs
        if isinstance(x.payload, Trajectory3DPayload)
    }
    assert sources[traj_a.label_id] == (box_a.label_id,)
    assert sorted(sources.values()) == sorted([(box_a.label_id,), (box_b.label_id,)])
    assert (s.lifted, s.lift_blocked) == (1, 1)
