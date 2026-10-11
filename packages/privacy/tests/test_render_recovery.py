"""손상된 블러본/렌더 기록은 소비자가 거부하고 재렌더로 복구할 수 있어야 한다."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from dlp_fixtures.video import generate_blur_scenario
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.render import render_blurred
from dlp_privacy.rendering import (
    assert_render_current,
    check_fetched,
    render_meta_key,
    render_session,
)
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.db.tables import metadata
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import make_session


@pytest.mark.parametrize(
    "damage",
    [
        "intact",
        "bytes",
        "object-hash",
        "missing-hash",
        "legacy-record",
        "invalid-json",
        "json-array",
        "invalid-utf8",
    ],
)
def test_rerender_repairs_damaged_release(
    tmp_path: Path, policy: PrivacyPolicy, damage: str
) -> None:
    """같은 라벨·정책이라도 실제 출력과 기록이 유효할 때만 렌더를 건너뛴다."""
    video = tmp_path / "source.mp4"
    generate_blur_scenario(0, duration_ms=100).write(video)
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    raw.put_file("s001/bodycam.mp4", video, sha256_file(video))
    session = make_session(
        privacy_state=PrivacyState.APPROVED,
        lifecycle_state=LifecycleState.PRIVACY_APPROVED,
    )
    stream = session.streams[0].model_copy(update={"uri": raw.uri("s001/bodycam.mp4")})
    session = session.model_copy(update={"streams": (stream,)})
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)
    try:
        with (
            engine.begin() as conn,
            patch("dlp_privacy.rendering.get_session", return_value=session),
        ):
            register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
            insert_session(conn, session)
            render_session(conn, "s001", raw, labeling, policy)
        key = blurred_key("s001", "bodycam")
        release = labeling.root / key
        original = release.read_bytes()
        sidecar = release.with_name(release.name + ".sha256")
        meta = labeling.root / render_meta_key("s001", "bodycam")
        if damage == "bytes":
            release.write_bytes(b"damaged video")  # 메타데이터 해시는 그대로인 손상도 재현한다.
        elif damage == "object-hash":
            release.write_bytes(b"replacement video")
            sidecar.write_text(sha256_file(release))
        elif damage == "missing-hash":
            sidecar.unlink()
        elif damage == "legacy-record":
            record = json.loads(meta.read_text())
            record.pop("blurred_sha256")
            meta.write_text(json.dumps(record))
        elif damage == "invalid-utf8":
            meta.write_bytes(b"\xff")
        elif damage != "intact":
            meta.write_text("{" if damage == "invalid-json" else "[]")
        # SQLite는 timezone을 지운다. 이 단위 검사는 세션 조회 경계만 고정하고
        # 실제 디코딩·렌더·로컬 객체 저장소·파일 해시 검증을 수행한다.
        with (
            engine.begin() as conn,
            patch("dlp_privacy.rendering.get_session", return_value=session),
            patch("dlp_privacy.rendering.render_blurred", wraps=render_blurred) as encode,
        ):
            render_session(conn, "s001", raw, labeling, policy)
            expected = assert_render_current(conn, labeling, "s001", "bodycam", policy)
        assert encode.call_count == (0 if damage == "intact" else 1)
        assert expected is not None
        check_fetched(release, expected, "s001", "bodycam")
        assert release.read_bytes() == original
        head = labeling.head(key)
        assert head is not None and head.sha256 == expected
    finally:
        engine.dispose()
