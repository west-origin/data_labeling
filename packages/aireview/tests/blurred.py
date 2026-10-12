"""DB·블러본이 있는 합성 세션 (실행기·블러본 경계 테스트 공용, WP17.5).

`dlp_fixtures.aireview.aireview_scene`의 세션·라벨을 시간대 보존 SQLite 파일 DB에 넣고, 영상
스트림마다 합성 프레임(`scene_frame`, 사람·글자 없음)으로 MP4를 써서 라벨링 버킷(로컬 저장소)에
올린 뒤, 지금 DB 상태의 렌더 해시로 렌더 기록을 쓴다 (`dlp privacy render`가 남기는 것과 같은
모양). 블러 라벨이 없어 렌더 해시는 렌더 정책만의 해시다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa

from dlp_fixtures.aireview import AiReviewScene, aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import insert_labels, insert_session, register_ontology
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import load_ontology
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import sqlite_db

LABELING = load_config(repo_root() / "config/defaults.yaml").buckets.labeling
RAW = load_config(repo_root() / "config/defaults.yaml").buckets.raw


@dataclass(frozen=True)
class DbScene:
    """DB·라벨링 버킷이 준비된 합성 세션.

    Attributes:
        scene: 합성 장면 (세션·라벨·프레임 시각).
        engine: 시간대 보존 SQLite 파일 엔진 (여러 연결·스레드).
        labeling: 라벨링 버킷 로컬 저장소. privacy: 프라이버시 정책. root: 저장소 상위 디렉터리.
    """

    scene: AiReviewScene
    engine: sa.Engine
    labeling: LocalStore
    privacy: PrivacyPolicy
    root: Path

    @property
    def session_id(self) -> str:
        """세션 ID."""
        return self.scene.session.session_id


def publish_blurred(db: DbScene, stream: str, work: Path) -> str:
    """스트림의 합성 블러본을 라벨링 버킷에 올리고 지금 렌더 해시로 렌더 기록을 쓴다 (해시)."""
    sc = db.scene
    video = work / f"{sc.session.session_id}-{stream}.mp4"
    frames = ((t, scene_frame(sc, stream, t)) for t in sc.frame_times)
    write_video(video, frames, width=sc.width, height=sc.height)
    sha = sha256_file(video)
    db.labeling.put_file(blurred_key(sc.session.session_id, stream), video, sha)
    with db.engine.connect() as conn:
        digest = expected_render_hash(conn, sc.session.session_id, stream, db.privacy)
    write_render_meta(db.labeling, sc.session.session_id, stream, RenderMeta(digest, sha), work)
    return digest


def db_scene(
    tmp_path: Path,
    *,
    session_id: str = "s001",
    n_units: int = 2,
    labels: list[LabelRecord] | None = None,
    privacy: PrivacyState = PrivacyState.APPROVED,
    lifecycle: LifecycleState = LifecycleState.PRELABELED,
    publish: bool = True,
    engine: sa.Engine | None = None,
) -> DbScene:
    """합성 세션을 DB에 넣고 (publish면) 영상 스트림 블러본·렌더 기록을 올린다.

    Args:
        labels: 넣을 라벨 (None이면 장면 라벨 전부).
        engine: 쓸 DB (None이면 tmp_path의 시간대 보존 SQLite 파일. 마이그레이션된
            PostgreSQL도 된다).
    """
    scene = aireview_scene(0, session_id=session_id, n_units=n_units)
    engine = engine or sqlite_db(f"sqlite:///{tmp_path / 'aireview.db'}")
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        session = scene.session.model_copy(
            update={"lifecycle_state": lifecycle, "privacy_state": privacy}
        )
        insert_session(conn, session)
        insert_labels(conn, list(scene.labels) if labels is None else labels)
    db = DbScene(
        scene,
        engine,
        LocalStore(tmp_path / "store", LABELING),
        load_privacy_policy(repo_root()),
        tmp_path / "store",
    )
    if publish and privacy is PrivacyState.APPROVED:
        work = tmp_path / "publish"
        work.mkdir(exist_ok=True)
        for stream in ("bodycam", "tp1"):
            publish_blurred(db, stream, work)
    return db
