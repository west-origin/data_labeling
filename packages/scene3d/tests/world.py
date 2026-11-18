"""scene3d 시험 공용 준비: 로컬 저장소의 전후 사진 게이트·복원과, DB에 넣은 바디캠 세션·블러본.

시험 파일이 아니다 (`test_*.py`가 불러 쓴다). 정답은 `dlp_fixtures.scene3d`의 합성 방이다.
- `gate`: 한 단계의 수집 → 탐지 → 승인(탐지 상자 그대로) → 렌더.
- `build_scene`: 두 단계 모두 게이트를 지나 정답 stub으로 복원한다.
- `bodycam_session`: 시간대 보존 SQLite에 승인된 세션을 넣고, 합성 바디캠 영상을 그 세션의
  블러본으로 라벨링 버킷에 올린 뒤 지금 DB 상태의 렌더 해시로 렌더 기록을 쓴다
  (`dlp privacy render`가 남기는 모양).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import sqlalchemy as sa

from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_scene3d.backends import InitPointsSplat
from dlp_scene3d.oracle import load_oracle
from dlp_scene3d.photos import ingest_capture
from dlp_scene3d.pipeline import reconstruct
from dlp_scene3d.policy import Scene3dPolicy, load_policy
from dlp_scene3d.privacy_gate import (
    approve_capture,
    detect_capture,
    latest_detection,
    render_capture,
)
from dlp_schema import repo_root
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.ontology import load_ontology
from dlp_schema.scene3d import ScenePhase
from dlp_schema.session import Calibration, CameraIntrinsics, PrivacyState
from dlp_schema.testing import make_session, sqlite_db

ROOT = repo_root()
POLICY = load_policy(ROOT)
PRIVACY = load_privacy_policy(ROOT)
NOW = datetime(2026, 11, 14, tzinfo=UTC)
# 시험 저장소의 버킷 이름 (원본, 라벨링)
RAW_BUCKET, LABELING_BUCKET = "raw", "labeling"
PHASES: tuple[ScenePhase, ScenePhase] = ("before", "after")


def with_marker(policy: Scene3dPolicy, **marker: object) -> Scene3dPolicy:
    """`reconstruction.marker` 값을 바꾼 정책 (검증을 다시 거친다)."""
    data = policy.model_dump(mode="json")
    data["reconstruction"]["marker"].update(marker)
    return Scene3dPolicy.model_validate(data)


def with_localization(policy: Scene3dPolicy, **values: object) -> Scene3dPolicy:
    """`localization` 값을 바꾼 정책 (검증을 다시 거친다)."""
    data = policy.model_dump(mode="json")
    data["localization"].update(values)
    return Scene3dPolicy.model_validate(data)


@dataclass
class Stores:
    """시험 저장소: 감사 원본 저장소, 라벨링 저장소, 감사 기록."""

    root: Path
    raw: AuditedStore
    labeling: LocalStore
    sink: MemorySink


def open_stores(root: Path) -> Stores:
    """root 아래의 원본·라벨링 로컬 저장소."""
    sink = MemorySink()
    raw = AuditedStore(LocalStore(root, RAW_BUCKET), sink, actor="svc", purpose="t")
    return Stores(root, raw, LocalStore(root, LABELING_BUCKET), sink)


def gate(truth: Path, stores: Stores, sid: str, phase: ScenePhase, tmp: Path) -> None:
    """수집·탐지·승인(탐지 상자 그대로)·렌더."""
    raw, lab = stores.raw, stores.labeling
    ingest_capture(raw, lab, sid, phase, truth.parent / phase, POLICY.capture, NOW)
    oracle = load_oracle(truth, phase)
    work = tmp / f"gate-{phase}"
    detect_capture(raw, lab, sid, phase, {"oracle": oracle.privacy}, [], PRIVACY, work / "d", NOW)
    det = latest_detection(raw, sid, phase)
    boxes = {k: list(v) for k, v in det.boxes.items()}
    approve_capture(raw, sid, phase, "privacy-1", boxes, det.model_version, NOW)
    render_capture(raw, lab, sid, phase, PRIVACY, work / "r", NOW)


def build_scene(
    truth: Path,
    stores: Stores,
    sid: str,
    tmp: Path,
    policy: Scene3dPolicy = POLICY,
    phases: tuple[ScenePhase, ...] = PHASES,
) -> None:
    """고른 단계를 게이트에 통과시키고 정답 stub SfM으로 복원한다."""
    for phase in phases:
        gate(truth, stores, sid, phase, tmp)
        reconstruct(
            stores.labeling,
            sid,
            phase,
            load_oracle(truth, phase).sfm,
            InitPointsSplat(policy.reconstruction.init_scale),
            policy,
            tmp / f"recon-{phase}",
            NOW,
        )


def ego_truth(truth: Path) -> dict[str, Any]:
    """truth.json의 바디캠 정답 절: 프레임별 자세 `frames`, 바뀐 시각 `change_ms`, 내부 파라미터."""
    return cast(dict[str, Any], json.loads(truth.read_text("utf-8"))["ego"])


def ego_intrinsics(truth: Path) -> CameraIntrinsics:
    """truth.json의 바디캠 정답 내부 파라미터."""
    ego = ego_truth(truth)
    k = cast(list[list[float]], ego["k"])
    return CameraIntrinsics(
        width=int(ego["width"]),
        height=int(ego["height"]),
        fx=k[0][0],
        fy=k[1][1],
        cx=k[0][2],
        cy=k[1][2],
    )


def publish_bodycam(engine: sa.Engine, stores: Stores, sid: str, video: Path, tmp: Path) -> str:
    """영상을 그 세션 바디캠의 블러본으로 올리고 지금 렌더 해시로 렌더 기록을 쓴다 (해시)."""
    sha = sha256_file(video)
    stores.labeling.put_file(blurred_key(sid, "bodycam"), video, sha)
    with engine.connect() as conn:
        digest = expected_render_hash(conn, sid, "bodycam", PRIVACY)
    tmp.mkdir(parents=True, exist_ok=True)
    write_render_meta(stores.labeling, sid, "bodycam", RenderMeta(digest, sha), tmp)
    return digest


def bodycam_session(
    stores: Stores,
    sid: str,
    truth: Path,
    tmp: Path,
    *,
    intrinsics: CameraIntrinsics | None,
    privacy: PrivacyState = PrivacyState.APPROVED,
    publish: bool = True,
) -> sa.Engine:
    """세션을 SQLite에 넣고 (publish면) 합성 바디캠 영상을 블러본으로 올린다. 엔진을 돌려준다.

    Args:
        intrinsics: 세션 매니페스트에 적을 바디캠 내부 파라미터 (None이면 등록 없음).
        privacy: 세션의 프라이버시 상태.
        publish: 블러본과 렌더 기록을 올릴지 (승인 상태일 때만 뜻이 있다).
    """
    tmp.mkdir(parents=True, exist_ok=True)
    engine = sqlite_db(f"sqlite:///{tmp / 'scene.db'}")
    session = make_session(
        sid, calibration=Calibration(intrinsics=intrinsics), privacy_state=privacy
    )
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, session)
    if publish and privacy is PrivacyState.APPROVED:
        publish_bodycam(engine, stores, sid, truth.parent / "ego" / "bodycam.mp4", tmp / "pub")
    return engine
