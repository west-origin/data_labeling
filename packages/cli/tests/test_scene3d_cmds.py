"""`dlp scene3d …` 오프라인 종단 시험 (로컬 저장소, SQLite 파일 DB, 정답 stub, ADR 0066·0067).

합성 픽스처(`dlp scene3d fixture`: 척도 마커, 좌표에 3.7을 곱하는 stub SfM, 방을 지나는 바디캠
영상)로 수집 → 블러 게이트 → 복원(마커 척도) → 비교 → 위치 추정 → 상태 보기를 명령으로 돈다.

정답: 픽스처의 truth.json (이동 벡터, 바디캠 프레임별 자세는 before 좌표·미터). 정책은 저장소 값
그대로 쓰되 마커 한 변만 픽스처 마커(0.3 m)에 맞춘다. 검수 정책은 `DLP_REVIEW_POLICY_FILE`로 원본
접근 권한자 한 명을 둔 사본을 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import sqlalchemy
import yaml

from dlp_cli import scene3d_cmds
from dlp_cli.main import main
from dlp_cli.raw_access import OFFLINE_LOG
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_scene3d.localize import stored_localization
from dlp_scene3d.pipeline import current_report
from dlp_scene3d.policy import Scene3dPolicy, load_policy
from dlp_schema import load_config, repo_root
from dlp_schema.calibration import CalibrationMethod
from dlp_schema.db.calibration_repository import register_camera_calibration
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.db.scene3d_repository import scene3d_records_for
from dlp_schema.db.tables import scene3d_records
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Calibration, CameraIntrinsics, PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

ROOT = repo_root()
BUCKETS = load_config(ROOT / "config" / "defaults.yaml").buckets
SID = "s-cli"
SIDE_M = 0.3
UNIT_SCALE = 3.7
PHASES = ("before", "after")


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """표를 만든 SQLite 파일 DB URL과, 시험용 정책·검수자 설정."""
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    sqlite_db(db).dispose()
    monkeypatch.setattr(scene3d_cmds, "sa", _AwareSqlalchemy())
    data = load_policy(ROOT).model_dump(mode="json")
    data["reconstruction"]["marker"]["side_m"] = SIDE_M
    policy = Scene3dPolicy.model_validate(data)

    def marked_policy(root: Path) -> Scene3dPolicy:
        return policy

    monkeypatch.setattr(scene3d_cmds, "load_policy", marked_policy)
    review = cast(
        dict[str, Any], yaml.safe_load((ROOT / "config/policies/review.yaml").read_text("utf-8"))
    )
    review["reviewers"]["privacy"] = ["privacy-1"]
    (tmp_path / "review.yaml").write_text(yaml.safe_dump(review, allow_unicode=True), "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(tmp_path / "review.yaml"))
    monkeypatch.setenv("DLP_ACTOR", "svc-pipeline")
    return db


def _fixture(out: Path, *extra: str) -> Path:
    assert main(["scene3d", "fixture", "--out", str(out), *extra]) == 0
    return out / "truth.json"


def _photo_stages(truth: Path, common: list[str], work: Path) -> None:
    """두 단계의 수집 → 탐지 → 검수 묶음 → 승인 → 렌더 → 복원."""
    for phase in PHASES:
        target = [SID, phase, *common]
        pack = work / f"pack-{phase}"
        assert main(["scene3d", "ingest", SID, phase, str(truth.parent / phase), *common]) == 0
        assert main(["scene3d", "privacy", "detect", *target]) == 0
        reviewer = ["--reviewer", "privacy-1"]
        assert main(["scene3d", "privacy", "review", *target, *reviewer, "--out", str(pack)]) == 0
        boxes = ["--boxes", str(pack / "boxes.json")]
        assert main(["scene3d", "privacy", "approve", *target, *reviewer, *boxes]) == 0
        assert main(["scene3d", "privacy", "render", *target]) == 0
        assert main(["scene3d", "reconstruct", *target]) == 0


def _register_bodycam(url: str, store: Path, truth: Path, work: Path) -> None:
    """세션(기기 cam-ego)과 그 기기의 캘리브레이션을 DB에 넣고 바디캠 블러본·렌더 기록을 올린다."""
    ego = cast(dict[str, Any], json.loads(truth.read_text("utf-8"))["ego"])
    k = cast(list[list[float]], ego["k"])
    intrinsics = CameraIntrinsics(
        width=int(ego["width"]), height=int(ego["height"]), fx=k[0][0], fy=k[1][1],
        cx=k[0][2], cy=k[1][2],
    )  # fmt: skip
    session = make_session(
        SID, calibration=Calibration(camera_id="cam-ego"), privacy_state=PrivacyState.APPROVED
    )
    engine = sqlite_db(url, create=False)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, session)
        register_camera_calibration(
            conn, "cam-ego", intrinsics, method=CalibrationMethod.IMPORTED, now=FIXED_TIME
        )
    labeling = LocalStore(store, BUCKETS.labeling)
    video = truth.parent / "ego" / "bodycam.mp4"
    sha = sha256_file(video)
    labeling.put_file(blurred_key(SID, "bodycam"), video, sha)
    with engine.connect() as conn:
        digest = expected_render_hash(conn, SID, "bodycam", load_privacy_policy(ROOT))
    work.mkdir(parents=True, exist_ok=True)
    write_render_meta(labeling, SID, "bodycam", RenderMeta(digest, sha), work)
    engine.dispose()


def test_offline_flow_marker_scale_compare_localize_status(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """마커로 미터가 된 전후 복원에서 이동 벡터와 바디캠 자세(before 좌표, 미터)가 정답과 맞고,
    status의 종료 코드가 전후 복원·정렬이 갖춰진 뒤에만 0이다. 다시 돌리면 건너뛴다."""
    truth = _fixture(
        tmp_path / "fx", "--marker-side-m", str(SIDE_M), "--unit-scale", str(UNIT_SCALE), "--ego"
    )
    store = tmp_path / "store"
    common = ["--store", f"local:{store}", "--url", url, "--oracle", str(truth)]
    assert main(["scene3d", "status", SID, *common]) == scene3d_cmds.INCOMPLETE_EXIT
    assert "빠짐: before 사진 수집 없음" in capsys.readouterr().out

    _photo_stages(truth, common, tmp_path / "work")
    text = capsys.readouterr().out
    assert text.count("metric=True") == 2 and text.count("척도: 마커 [7] (한 변 0.3 m") == 2
    assert main(["scene3d", "status", SID, *common]) == scene3d_cmds.INCOMPLETE_EXIT
    assert "빠짐: 정렬 없음" in capsys.readouterr().out

    assert main(["scene3d", "compare", SID, *common]) == 0
    assert main(["scene3d", "status", SID, *common]) == 0
    text = capsys.readouterr().out
    assert "갖춰짐" in text and "| 위치 추정 | 없음 |" in text and "| 척도 | 미터 (마커 [7]" in text
    labeling = LocalStore(store, BUCKETS.labeling)
    data = cast(dict[str, Any], json.loads(truth.read_text("utf-8")))
    report = current_report(labeling, SID)
    assert report.metric and report.alignment.scale == pytest.approx(1.0, abs=0.01)
    truth_moves = cast(dict[str, list[float]], data["moves"])
    moves = sorted(float(np.linalg.norm(np.asarray(v))) for v in truth_moves.values())
    got = sorted(c.distance_m for c in report.changes if c.kind == "moved" and c.distance_m)
    assert got == pytest.approx(moves, abs=0.02)

    # 세션이 등록되기 전에는 위치를 추정하지 않는다
    assert main(["scene3d", "localize", SID, *common]) == scene3d_cmds.REFUSED_EXIT
    assert "세션이 등록돼 있지 않습니다" in capsys.readouterr().err
    _register_bodycam(url, store, truth, tmp_path / "pub")
    assert main(["scene3d", "localize", SID, *common]) == 0
    assert "위치 추정: 프레임" in capsys.readouterr().out
    assert main(["scene3d", "localize", SID, *common]) == 0
    assert "같은 위치 추정(건너뜀)" in capsys.readouterr().out
    assert main(["scene3d", "status", SID, *common]) == 0
    assert "| 위치 추정 | 있음: 프레임" in capsys.readouterr().out

    found = stored_localization(labeling, SID)
    assert found is not None
    tag, record = found
    assert record.metric and record.intrinsics_origin.startswith("registry:cal-")
    assert record.localized > 0.8 * record.frame_count
    assert {p.reference for p in record.poses} == {"before", "after"}
    frames = {int(f["t_ms"]): f for f in data["ego"]["frames"]}
    for pose in record.poses:
        r = np.asarray(pose.rotation).reshape(3, 3)
        want_r, want_t = np.asarray(frames[pose.t_ms]["r"]), np.asarray(frames[pose.t_ms]["t"])
        centre, want = -r.T @ np.asarray(pose.translation), -want_r.T @ want_t
        assert np.linalg.norm(centre - want) < 0.03, pose.t_ms
        assert np.trace(r @ want_r.T) > 3 - 2e-4  # 회전 오차 약 0.8도 이내

    # DB 기록: 종류마다 남고, 태그가 열 길이(64자) 안이다 (F-PAR-97)
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        records = scene3d_records_for(conn, SID)
    kinds = [r.kind for r in records]
    assert {k: kinds.count(k) for k in sorted(set(kinds))} == {
        "alignment": 1,
        "capture": 2,
        "change_report": 1,
        "localization": 1,
        "privacy_approval": 2,
        "privacy_detection": 2,
        "reconstruction": 2,
        "render": 2,
    }
    tag_type = scene3d_records.c.tag.type
    assert isinstance(tag_type, sqlalchemy.String) and tag_type.length == 64
    assert all(len(r.tag) <= tag_type.length for r in records)
    located = next(r for r in records if r.kind == "localization")
    assert located.tag == tag and located.storage_key.endswith(f"localization/{tag}.json")
    assert located.bucket == "labeling" and located.phase is None
    assert all(labeling.head(r.storage_key) is not None for r in records if r.bucket == "labeling")


def test_no_db_photo_stages_and_read_only_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """DB 없이(--no-db) 사진 단계와 status가 돈다. status는 원본 접근 기록(로컬 JSON Lines)과
    저장소를 바꾸지 않는다. 마커가 없는 미터 아닌 복원은 갖춤 판정과 별개로 알린다. localize에는
    --no-db가 없다."""
    review = cast(
        dict[str, Any], yaml.safe_load((ROOT / "config/policies/review.yaml").read_text("utf-8"))
    )
    review["reviewers"]["privacy"] = ["privacy-1"]
    (tmp_path / "review.yaml").write_text(yaml.safe_dump(review, allow_unicode=True), "utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(tmp_path / "review.yaml"))
    truth = _fixture(tmp_path / "fx", "--unit-scale", "2")
    store = tmp_path / "store"
    common = ["--store", f"local:{store}", "--no-db", "--oracle", str(truth)]
    _photo_stages(truth, common, tmp_path / "work")
    text = capsys.readouterr().out
    assert (
        "metric=False" in text and "척도: 정하지 못함 (미터 아님) (마커를 찾지 못했습니다)" in text
    )
    # 미터가 아니면 비교는 거부된다 (척도를 지어내지 않는다)
    assert main(["scene3d", "compare", SID, *common]) == scene3d_cmds.REFUSED_EXIT
    assert "미터 단위가 아닙니다" in capsys.readouterr().err

    log = (store / OFFLINE_LOG).read_bytes()
    files = sorted(str(p) for p in store.rglob("*") if p.is_file())
    assert main(["scene3d", "status", SID, *common]) == scene3d_cmds.INCOMPLETE_EXIT
    out = capsys.readouterr().out
    assert "빠짐: 정렬 없음" in out and "주의: before 복원이 미터가 아닙니다" in out
    assert "| 블러 탐지 | 표시 있음 | 표시 있음 |" in out
    assert (store / OFFLINE_LOG).read_bytes() == log
    assert sorted(str(p) for p in store.rglob("*") if p.is_file()) == files
    with pytest.raises(SystemExit) as exc:
        main(["scene3d", "localize", SID, "--store", f"local:{store}", "--no-db"])
    assert exc.value.code == 2
