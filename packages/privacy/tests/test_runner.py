from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import av
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import BlurScenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import PrivacyPolicy, TargetPolicy
from dlp_privacy.runner import PrivacyGateError, approve_session, detect_session, render_session
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_labels, get_session, record_review, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_detect_review_approve_render(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    scenario, video = blur
    # 3인칭 영상 자리에는 오디오가 있는 다른 영상을 둔다 (블러본에서 오디오가 빠지는지 본다)
    sync = generate_sync_scenario(2, recorded_at=FIXED_TIME, duration_ms=3_000)
    sync.write_video(tmp_path / "third.mp4", "bodycam")
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": str(video)},
            {"stream_id": "third_person", "kind": "third_person", "path": "third.mp4"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw, labeling = S3Store.from_env("dlp-raw"), S3Store.from_env("dlp-labeling")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)

    oracle = OracleDetector("oracle", scenario.labels)
    oracle_policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    with pg.begin() as conn:
        with pytest.raises(PrivacyGateError, match="자동 탐지"):
            approve_session(conn, sid)
        summary = detect_session(conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME)
    assert summary.detected["bodycam"] == 6 and set(summary.detected) == {"bodycam", "third_person"}
    with pg.connect() as conn:
        assert get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED

    # 같은 모델 버전으로 다시 실행하면 건너뛴다
    with pg.begin() as conn:
        again = detect_session(conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME)
    assert sorted(again.skipped) == ["bodycam", "third_person"]

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        raw.get_file(f"sessions/{sid}/derived/privacy_review/bodycam.json", dest)
        segments = json.loads(dest.read_text(encoding="utf-8"))
    assert any(s["reason"] == "reflection" for s in segments)

    with pg.begin() as conn:
        with pytest.raises(PrivacyGateError, match="승인 전"):
            render_session(conn, sid, raw, labeling, oracle_policy)
        with pytest.raises(PrivacyGateError, match="사람 검수"):
            approve_session(conn, sid)
        for label in get_labels(conn, sid, kinds=["blur_track"]):
            record_review(
                conn, label.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME
            )
        approved = approve_session(conn, sid)
    assert approved.privacy_state is PrivacyState.APPROVED
    assert approved.lifecycle_state is LifecycleState.PRIVACY_APPROVED

    with pg.begin() as conn:
        uris = render_session(conn, sid, raw, labeling, oracle_policy)
        assert render_session(conn, sid, raw, labeling, oracle_policy) == uris  # 멱등
        with pytest.raises(PrivacyGateError, match="라벨링 버킷"):
            render_session(conn, sid, raw, raw, oracle_policy)
    assert uris["bodycam"] == f"s3://dlp-labeling/sessions/{sid}/blurred/bodycam.mp4"
    assert all("dlp-raw" not in u for u in uris.values())
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "b.mp4"
        labeling.get_file(f"sessions/{sid}/blurred/third_person.mp4", dest)
        with av.open(str(dest)) as c:
            assert c.streams.video and not c.streams.audio


class TrainedBlur:
    """배포된 재학습 블러 모델 자리 (정답 블러를 그대로 낸다)."""

    name = "trained-privacy"

    def __init__(self, labels: list[LabelRecord], version: str) -> None:
        self.labels, self.version = labels, version

    def run(self, clip: Clip) -> list[LabelRecord]:
        tag = self.version.replace(".", "_")
        return [
            x.model_copy(
                update={
                    "label_id": f"{clip.session_id}-{clip.stream_id}-{self.name}-{tag}-{i}",
                    "session_id": clip.session_id,
                    "stream_id": clip.stream_id,
                    "provenance": Provenance(source=Source.MODEL, model_version=self.version),
                    "confidence": 0.9,
                }
            )
            for i, x in enumerate(self.labels)
        ]


def test_deployed_blur_model_is_unioned_and_versioned(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    scenario, video = blur
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": str(video)}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    oracle = OracleDetector("oracle", scenario.labels)
    oracle_policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    blurs = [x for x in scenario.labels if x.kind == "blur_track"]
    v1, v2 = TrainedBlur(blurs, "trained-v1"), TrainedBlur(blurs, "trained-v2")

    def current_versions(conn: sa.Connection) -> dict[str, int]:
        out: dict[str, int] = {}
        for x in current_labels(get_labels(conn, sid, kinds=["blur_track"])):
            v = x.provenance.model_version or ""
            out[v] = out.get(v, 0) + 1
        return out

    with pg.begin() as conn:
        first = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v1]
        )
        before = current_versions(conn)
    assert first.detected["bodycam"] == 6 + len(blurs)  # 탐지기 + 재학습 모델 (합집합)
    assert before["trained-v1"] == len(blurs)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        raw.get_file(f"sessions/{sid}/derived/privacy_review/bodycam.json", dest)
        reasons = {s["reason"] for s in json.loads(dest.read_text(encoding="utf-8"))}
    assert "trained_model" in reasons

    with pg.begin() as conn:
        again = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v1]
        )
        assert again.skipped == ["bodycam"]
        # 모델이 바뀌면 이전 모델의 검수 전 블러만 바뀌고 탐지기 결과는 그대로다
        swapped = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v2]
        )
        after = current_versions(conn)
    assert swapped.detected["bodycam"] == len(blurs)
    assert "trained-v1" not in after and after["trained-v2"] == len(blurs)
    assert sum(n for v, n in after.items() if not v.startswith("trained")) == 6
