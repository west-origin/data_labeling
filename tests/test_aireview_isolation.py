"""AI 판정은 라벨·검수·학습·내보내기·스냅샷에 섞이지 않는다 (격리 회귀, WP17.5, ADR 0040 I3·I4·I9).

합성 세션(승인된 블러본, SQLite 파일 DB)에서 판정 실행기를 돌려 AI 표에 행을 쓴 뒤, 다른 단계가 보는
값이 판정 전과 **같은지** 본다:

- 라벨 이력(`get_labels`)과 운영 현재 라벨(`current_labels`).
- 내보내기 라벨 선택(`dlp_export.source.select_labels`, 미검수 포함 옵션까지).
- 학습 예제 추출(`dlp_train.extract.extract_examples`, 모든 과제).
- 데이터셋 스냅샷 `labels.jsonl` (`build_dataset_version`, 로컬 스냅샷 저장소).
- 사람 검수 판정(`verify_session`): AI가 모두 ok여도 사람 검수 없이는 human_verified가 되지 않는다.
- AI 판정 실행자 접두사(`aireview:`)로는 검수 상태를 기록할 수 없다.

정답 근거: 판정 전에 같은 함수로 계산한 값 (서비스 불필요).
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_aireview.oracle import OracleJudge
from dlp_aireview.policy import load_policy
from dlp_aireview.runner import RunOptions, run_from_db
from dlp_datasets.build import build_dataset_version
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_eval.policy import Task
from dlp_export.policy import load_policy as load_export_policy
from dlp_export.source import label_states, select_labels
from dlp_fixtures.aireview import aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_review.verify import verify_session
from dlp_schema import load_config, repo_root
from dlp_schema.dataset import Split
from dlp_schema.db.aireview_repository import list_ai_units
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import ActionPayload, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, sqlite_db
from dlp_train.extract import extract_examples
from dlp_train.policy import load_policy as load_training_policy

ROOT = repo_root()
BUCKETS = load_config(ROOT / "config/defaults.yaml").buckets
SID = "iso1"


def _world(tmp_path: Path) -> tuple[sa.Engine, LocalStore, list[Any]]:
    """승인된 합성 세션 하나 (블러본·렌더 기록), 사람이 승인한 행동 라벨 하나."""
    scene = aireview_scene(0, session_id=SID, n_units=2)
    engine = sqlite_db(f"sqlite:///{tmp_path / 'iso.db'}")
    session = scene.session.model_copy(
        update={
            "lifecycle_state": LifecycleState.PRELABELED,
            "privacy_state": PrivacyState.APPROVED,
        }
    )
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, session)
        insert_labels(conn, list(scene.labels))
        action = next(x for x in scene.labels if isinstance(x.payload, ActionPayload))
        record_review(conn, action.label_id, VerificationState.HUMAN_APPROVED, "rev1", FIXED_TIME)
    labeling = LocalStore(tmp_path / "store", BUCKETS.labeling)
    privacy = load_privacy_policy(ROOT)
    for stream in ("bodycam", "tp1"):
        video = tmp_path / f"{stream}.mp4"
        frames = ((t, scene_frame(scene, stream, t)) for t in scene.frame_times)
        write_video(video, frames, width=scene.width, height=scene.height)
        sha = sha256_file(video)
        labeling.put_file(blurred_key(SID, stream), video, sha)
        with engine.connect() as conn:
            digest = expected_render_hash(conn, SID, stream, privacy)
        write_render_meta(labeling, SID, stream, RenderMeta(digest, sha), tmp_path)
    return engine, labeling, list(scene.labels)


def _downstream(engine: sa.Engine, tmp_path: Path) -> dict[str, Any]:
    """다른 단계가 보는 값 (판정 전후 비교)."""
    export = load_export_policy(ROOT)
    training = load_training_policy(ROOT)
    with engine.connect() as conn:
        history = get_labels(conn, SID)
    with engine.connect() as conn:
        result = verify_session(conn, SID, FIXED_TIME, "tester")
        build = build_dataset_version(
            conn, LocalSnapshotStore(tmp_path / "snap"), load_dataset_policy(ROOT),
            version_id="iso-v1", ontology_version="1.0.0", golden_set_version=None,
            now=FIXED_TIME,
        )  # fmt: skip
        conn.rollback()  # 비교용: 데이터셋 버전·생애주기 전이를 남기지 않는다
    snapshot = tmp_path / "labels.jsonl"
    LocalSnapshotStore(tmp_path / "snap").read(build.version.snapshot_uri, "labels.jsonl", snapshot)
    tasks: tuple[Task, ...] = typing.get_args(Task)
    return {
        "history": [x.model_dump_json() for x in history],
        "current": [x.model_dump_json() for x in current_labels(history)],
        "export": [
            [x.label_id for x in select_labels(history, export, label_states(export, flag), ())]
            for flag in (False, True)
        ],
        "train": {
            t: [repr(e) for e in extract_examples(history, {SID: Split.TRAIN}, t, training)]
            for t in tasks
        },
        "snapshot": snapshot.read_text("utf-8"),
        "verify": (result.verified, tuple(result.reasons)),
    }


def test_ai_verdicts_change_nothing_downstream(tmp_path: Path) -> None:
    """AI 표에 판정(모두 ok)이 쌓여도 라벨·내보내기·학습·스냅샷·검수 판정이 그대로다. AI가 모두
    ok여도 사람 검수가 끝나지 않은 세션은 human_verified가 되지 않는다 (I9)."""
    engine, labeling, labels = _world(tmp_path)
    before = _downstream(engine, tmp_path / "before")
    summary = run_from_db(
        engine, labeling, SID, OracleJudge(labels), load_policy(ROOT), load_privacy_policy(ROOT),
        raw_bucket=BUCKETS.raw, options=RunOptions(workers=2),
    )  # fmt: skip
    assert summary is not None and summary.status == "completed"
    with engine.connect() as conn:
        units = list_ai_units(conn, SID)
    assert units and all(v.verdict.value == "ok" for u in units for v in u.verdicts)
    after = _downstream(engine, tmp_path / "after")
    assert after == before
    assert after["verify"][0] is False and after["verify"][1]
    assert after["train"]["actions"]  # 사람이 승인한 행동은 학습 예제 (판정 전후 같다)


def test_ai_reviewer_cannot_record_review(tmp_path: Path) -> None:
    """AI 판정 실행자(`aireview:` 접두사)로는 검수 상태를 기록할 수 없다 (I3)."""
    engine, _, labels = _world(tmp_path)
    with engine.begin() as conn, pytest.raises(ValueError, match="AI 판정은 검수가 아닙니다"):
        record_review(
            conn, labels[1].label_id, VerificationState.HUMAN_APPROVED, "aireview:run", FIXED_TIME
        )
