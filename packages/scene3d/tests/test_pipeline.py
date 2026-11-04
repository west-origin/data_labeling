"""수집 → 블러 승인 → 복원 → 전후 비교 종단 시험 (로컬 저장소, CPU stub, ADR 0066).

정답은 `dlp_fixtures.scene3d`의 합성 방이다. after는 before와 다른 좌표계(알려진 닮음 변환)라
정렬을 거쳐야 이동 벡터가 맞는다.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dlp_fixtures.scene3d import generate_scene_pair, write_scene_pair
from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.storage import LocalStore
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_scene3d.backends import InitPointsSplat, OracleSfm, SfmResult
from dlp_scene3d.oracle import load_oracle, oracle_objects
from dlp_scene3d.photos import decode_photo, ingest_capture, photo_id_for
from dlp_scene3d.pipeline import compare, current_report, reconstruct, report_rows
from dlp_scene3d.policy import load_policy
from dlp_scene3d.privacy_gate import (
    BlurredPhoto,
    OraclePhotoDetector,
    approve_capture,
    detect_capture,
    latest_detection,
    open_blurred,
    render_capture,
)
from dlp_schema import repo_root
from dlp_schema.scene3d import ScenePhase

ROOT = repo_root()
POLICY = load_policy(ROOT)
PRIVACY = load_privacy_policy(ROOT)
NOW = datetime(2026, 11, 4, tzinfo=UTC)
SID = "s-scene"


@dataclass
class SpySfm:
    """받은 사진 경로를 기록하는 SfM (하류가 블러본만 받는지 본다)."""

    inner: OracleSfm
    seen: list[Path] = field(default_factory=list[Path])
    name: str = "spy"
    version: str = "spy-1"

    def reconstruct(self, photos: Sequence[BlurredPhoto], workdir: Path) -> SfmResult:
        self.seen += [p.path for p in photos]
        return self.inner.reconstruct(photos, workdir)


@pytest.fixture
def world(tmp_path: Path) -> tuple[Path, AuditedStore, LocalStore, MemorySink]:
    truth = write_scene_pair(generate_scene_pair(seed=0), tmp_path / "photos")
    sink = MemorySink()
    raw = AuditedStore(LocalStore(tmp_path / "store", "raw"), sink, actor="svc", purpose="t")
    return truth, raw, LocalStore(tmp_path / "store", "labeling"), sink


def _gate(truth: Path, raw: AuditedStore, lab: LocalStore, phase: ScenePhase, tmp: Path) -> None:
    """수집·탐지·승인(탐지 상자 그대로)·렌더."""
    ingest_capture(raw, lab, SID, phase, truth.parent / phase, POLICY.capture, NOW)
    o = load_oracle(truth, phase)
    detect_capture(raw, lab, SID, phase, {"oracle": o.privacy}, [], PRIVACY, tmp / f"d{phase}", NOW)
    det = latest_detection(raw, SID, phase)
    boxes = {k: list(v) for k, v in det.boxes.items()}
    approve_capture(raw, SID, phase, "privacy-1", boxes, det.model_version, NOW)
    render_capture(raw, lab, SID, phase, PRIVACY, tmp / f"r{phase}", NOW)


def test_recovers_object_moves_in_before_frame(
    world: tuple[Path, AuditedStore, LocalStore, MemorySink], tmp_path: Path
) -> None:
    """이동 벡터(before 좌표, 미터)가 정답과 2 cm 안에서 맞고 치움·새로 둠·그대로를 나눈다."""
    truth, raw, lab, sink = world
    spies: dict[str, SpySfm] = {}
    for phase in ("before", "after"):
        _gate(truth, raw, lab, phase, tmp_path)
        spies[phase] = SpySfm(load_oracle(truth, phase).sfm)
        rec, new = reconstruct(
            lab,
            SID,
            phase,
            spies[phase],
            InitPointsSplat(0.01),
            POLICY,
            tmp_path / f"w{phase}",
            NOW,
        )
        assert new and rec.registered == rec.photo_count == 14
    assert all("/blurred/" in str(p) for s in spies.values() for p in s.seen)
    report, new = compare(lab, SID, oracle_objects(truth), POLICY, tmp_path / "c", NOW)
    assert new and report.alignment.accepted
    kinds = {}
    pair = generate_scene_pair(seed=0)
    centers = {o.key: o.center for o in pair.before.objects}
    for c in report.changes:
        assert c.before_position is not None or c.kind == "added"
        key = next(
            (
                k
                for k, v in centers.items()
                if c.before_position and np.linalg.norm(np.subtract(c.before_position, v)) < 0.05
            ),
            None,
        )
        kinds[key or c.label] = c.kind
        if c.kind == "moved":
            assert key is not None and c.displacement is not None
            assert np.linalg.norm(np.subtract(c.displacement, pair.moves[key])) < 0.02, key
    assert kinds == {
        "pillow": "moved",
        "cup": "moved",
        "bucket_orange": "moved",
        "bucket_blue": "unchanged",
        "trash_bin": "unchanged",
        "towel": "removed",
        "spray_bottle": "added",
    }
    assert current_report(lab, SID) == report
    assert len(report_rows(report)) == 7
    assert compare(lab, SID, oracle_objects(truth), POLICY, tmp_path / "c2", NOW)[1] is False
    assert sink.events and all(e.key.startswith(f"sessions/{SID}/scene3d/") for e in sink.events)


def test_blurred_photos_hide_faces_and_redetect_invalidates(
    world: tuple[Path, AuditedStore, LocalStore, MemorySink], tmp_path: Path
) -> None:
    """블러본의 얼굴 상자는 원본과 다르고, 새 탐지 세대가 생기면 하류가 블러본을 받지 못한다."""
    truth, raw, lab, _ = world
    _gate(truth, raw, lab, "before", tmp_path)
    tag, photos = open_blurred(lab, SID, "before", tmp_path / "o")
    o = load_oracle(truth, "before")
    face_photo = next(p for p in photos if o.privacy.boxes[p.photo_id])
    x, y, w, h = (int(v) for v in o.privacy.boxes[face_photo.photo_id][0][1])
    by_id = {
        photo_id_for(hashlib.sha256(f.read_bytes()).hexdigest()): f
        for f in (truth.parent / "before").glob("*.png")
    }
    raw_img = decode_photo(by_id[face_photo.photo_id])[0]
    blurred = face_photo.load()
    assert not np.array_equal(
        raw_img[y + h // 3 : y + h // 3 + 3, x : x + w],
        blurred[y + h // 3 : y + h // 3 + 3, x : x + w],
    )
    assert render_capture(raw, lab, SID, "before", PRIVACY, tmp_path / "r2", NOW)[1] is False
    other = OraclePhotoDetector({}, version="oracle-photo-2")
    detect_capture(raw, lab, SID, "before", {"oracle": other}, [], PRIVACY, tmp_path / "d2", NOW)
    with pytest.raises(RenderNotCurrentError):
        open_blurred(lab, SID, "before", tmp_path / "o2")
    assert tag


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    """EXIF 방향 6(시계 방향 90도)이면 읽은 배열의 가로·세로가 바뀐다. 원본 바이트는 그대로 둔다."""
    img = Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6
    path = tmp_path / "a.jpg"
    img.save(path, exif=exif)
    arr, orientation = decode_photo(path)
    assert orientation == 6 and arr.shape[:2] == (40, 20)
