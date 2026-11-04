"""API: 토큰·Host 검사, 업로드 → 매니페스트 → 작업, 세션 상태·단계, 결과 보기, 원본 위치 비노출."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from dlp_console import app as console_app
from dlp_console.settings import ConsoleSettings
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.ingest import load_manifest
from dlp_media.storage import LocalStore, S3Store, blurred_key, sha256_file
from dlp_privacy.rendering import (
    RenderMeta,
    expected_render_hash,
    invalidate_render,
    write_render_meta,
)
from dlp_schema.db.repository import insert_session, insert_withdrawal, set_privacy_state
from dlp_schema.lineage import Withdrawal
from dlp_schema.session import PrivacyState, StreamKind, SyncMethod
from dlp_schema.testing import make_session

from .conftest import Api, FakeServices

RAW_URI = "s3://dlp-raw/s1/bodycam.mp4"


def _add_session(services: FakeServices, sid: str = "s1") -> None:
    session = make_session(
        sid,
        streams=[
            {
                "stream_id": "bodycam",
                "kind": StreamKind.BODYCAM,
                "uri": RAW_URI,
                "pts_index_uri": "s3://dlp-raw/s1/derived/bodycam.pts.parquet",
                "sync_method": SyncMethod.REFERENCE,
            }
        ],
    )
    with services.engine.begin() as conn:
        insert_session(conn, session)


def _wait(client: Api, services: FakeServices, job_id: str) -> dict[str, Any]:
    services.runner.wait(job_id, 30)
    data: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
    return data


def test_token_and_host_are_required(client: Api) -> None:
    assert client.get("/api/overview", headers={"X-Console-Token": ""}).status_code == 401
    assert client.get("/api/overview", headers={"X-Console-Token": "wrong"}).status_code == 401
    assert client.get("/api/overview").status_code == 200
    assert client.get("/api/overview", headers={"Host": "evil.example"}).status_code == 400
    index = client.get("/", headers={"X-Console-Token": ""})
    assert index.status_code == 200 and "운영 콘솔" in index.text  # 화면 틀에는 자료가 없다
    assert client.get("/static/app.js").status_code == 200


def test_overview_has_theme_from_review_ui(client: Api) -> None:
    data = client.get("/api/overview").json()
    assert data["theme"]["modes"]["dark"]["accent"].startswith("#")
    assert data["theme"]["logo"].startswith("data:image/png;base64,")
    assert "kitchen" in data["scenes"] and data["privacy_reviewers"] == ["priv01"]


def test_upload_writes_manifest_and_runs_prepare_then_ingest(
    client: Api, services: FakeServices, settings: ConsoleSettings, tmp_path: Path
) -> None:
    video = tmp_path / "clip.mp4"
    generate_blur_scenario(duration_ms=600).write(video)  # 합성 픽스처 (실제 영상 아님)
    form = {
        "session_id": "syn-01",
        "worker_id": "w-01",
        "site_id": "site-01",
        "domain": "cleaning",
        "consent_version": "consent-v2",
        "recorded_at": "2026-10-25T14:30:00+09:00",
        "camera_id": "cam-01",
        "scene": "kitchen",
        "display_rotation": "180",
    }
    with video.open("rb") as f:
        res = client.post("/api/upload", data=form, files={"file": ("clip.mp4", f, "video/mp4")})
    assert res.status_code == 200, res.text
    job = res.json()
    assert job["session_id"] == "syn-01" and job["step"] == "ingest"
    upload_dir = settings.work_dir / "uploads" / "syn-01"
    manifest_path = upload_dir / "session.yaml"
    assert job["commands"] == [
        f"dlp media prepare {upload_dir / 'original.mp4'} --out {upload_dir / 'prepared.mp4'} "
        "--display-rotation 180",
        f"dlp ingest {manifest_path}",
    ]
    manifest, base = load_manifest(manifest_path)  # 수집 코드의 모델로 다시 읽힌다
    assert base == upload_dir and manifest.streams[0].path == "prepared.mp4"
    assert manifest.calibration.camera_id == "cam-01" and manifest.consent_version == "consent-v2"
    raw = yaml.safe_load(manifest_path.read_text("utf-8"))
    assert raw["recorded_at"].startswith("2026-10-25T14:30:00")
    done = _wait(client, services, str(job["job_id"]))
    assert done["status"] == "succeeded"
    assert not (upload_dir / "original.mp4").exists()  # 수집 성공 뒤 업로드 사본을 지운다
    assert services.prefs("syn-01") == {"scene": "kitchen"}


def test_upload_rejects_missing_consent_and_bad_ids(client: Api, tmp_path: Path) -> None:
    base = {"session_id": "syn-02", "worker_id": "w", "site_id": "s", "domain": "cleaning"}
    files = {"file": ("a.mp4", b"\x00" * 10, "video/mp4")}
    res = client.post("/api/upload", data={**base, "consent_version": " "}, files=files)
    assert res.status_code == 400 and "동의서" in res.json()["detail"]
    bad = {**base, "consent_version": "c1", "worker_id": "--url"}
    assert client.post("/api/upload", data=bad, files=files).status_code == 400
    exe = {**base, "consent_version": "c1"}
    res = client.post("/api/upload", data=exe, files={"file": ("a.exe", b"x", "x")})
    assert res.status_code == 400


def test_session_view_never_contains_raw_locations(client: Api, services: FakeServices) -> None:
    _add_session(services)
    res = client.get("/api/sessions/s1")
    assert res.status_code == 200
    assert "dlp-raw" not in res.text and "pts_index_uri" not in res.text
    steps = {s["id"]: s for s in res.json()["steps"]}
    assert steps["privacy_detect"]["status"] == "ready"
    assert steps["privacy_approve"]["status"] == "blocked"
    assert client.get("/api/sessions/없음").status_code == 400
    assert client.get("/api/sessions/nope").status_code == 404


def test_step_buttons_run_dlp_or_explain_why_not(client: Api, services: FakeServices) -> None:
    _add_session(services)
    blocked = client.post("/api/sessions/s1/steps/privacy_approve", json={})
    assert blocked.status_code == 400 and "수집" in blocked.json()["detail"]
    res = client.post("/api/sessions/s1/steps/privacy_detect", json={})
    assert res.status_code == 200
    job = res.json()
    assert job["commands"] == ["dlp privacy detect s1"]
    done = _wait(client, services, str(job["job_id"]))
    assert done["status"] == "succeeded" and "args: privacy detect s1" in done["log"]


def test_failed_job_shows_error_line_and_next_action_without_raw_uri(
    client: Api, services: FakeServices
) -> None:
    job = services.runner.submit("labeling_review", "검수 작업", [["fail"]], lock="session:s9")
    data = _wait(client, services, job.job_id)
    assert data["status"] == "failed"
    assert "CVAT" in str(data["next_action"]) and "Connection refused" in str(data["error_line"])
    text = client.get(f"/api/jobs/{job.job_id}").text
    assert "dlp-raw" not in text and "dlp-dev-password" not in text


def test_preview_only_after_approval_and_render(client: Api, services: FakeServices) -> None:
    _add_session(services)
    res = client.get("/api/sessions/s1/preview?stream=bodycam")
    assert res.status_code == 409 and "승인" in res.json()["detail"]
    with services.engine.begin() as conn:
        set_privacy_state(conn, "s1", PrivacyState.APPROVED)
    assert client.get("/api/sessions/s1/preview?stream=bodycam").status_code == 409  # 렌더 전
    services.rendered_map[("s1", "bodycam")] = True
    res = client.get("/api/sessions/s1/preview?stream=bodycam")
    assert res.status_code == 200
    data = res.json()
    assert data["url"].startswith("https://labeling.example/sessions/s1/blurred/bodycam.mp4")
    assert data["tracks"] == [] and "dlp-raw" not in res.text


def test_preview_refused_for_withdrawn_session_f_coe_85(
    client: Api, services: FakeServices
) -> None:
    """F-COE-85: 승인·렌더된 세션이라도 사용 중지 뒤에는 블러본 서명 URL을 내지 않는다."""
    _add_session(services)
    with services.engine.begin() as conn:
        set_privacy_state(conn, "s1", PrivacyState.APPROVED)
    services.rendered_map[("s1", "bodycam")] = True
    assert client.get("/api/sessions/s1/preview?stream=bodycam").status_code == 200
    with services.engine.begin() as conn:
        insert_withdrawal(conn, Withdrawal(session_id="s1", reason="동의 철회",
                                           withdrawn_at=datetime.now(UTC)))  # fmt: skip
    res = client.get("/api/sessions/s1/preview?stream=bodycam")
    assert res.status_code == 409 and "사용 중지" in res.json()["detail"]
    assert "labeling.example" not in res.text


@pytest.fixture
def labeling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalStore:
    """콘솔의 라벨러 저장소를 로컬 라벨링 버킷으로 바꾼다."""
    store = LocalStore(tmp_path / "buckets", "dlp-labeling")

    def labeler(_bucket: str, *, public: bool = True) -> LocalStore:
        return store

    monkeypatch.setattr(S3Store, "labeler_from_env", labeler)
    return store


def _render(services: FakeServices, store: LocalStore, tmp_path: Path, content: bytes) -> None:
    """`dlp privacy render`처럼 블러본과 지금 블러 라벨·정책의 렌더 기록을 쓴다."""
    video = tmp_path / "blurred.mp4"
    video.write_bytes(content)
    store.put_file(blurred_key("s1", "bodycam"), video, sha256_file(video))
    with services.engine.connect() as conn:
        expected = expected_render_hash(conn, "s1", "bodycam", services.privacy_policy)
    write_render_meta(store, "s1", "bodycam", RenderMeta(expected, sha256_file(video)), tmp_path)


def _gate(services: FakeServices) -> bool:
    """가짜가 아닌 콘솔의 렌더 확인 (`Services.rendered`, 렌더 게이트)."""
    with services.engine.connect() as conn:
        return console_app.Services.rendered(services, conn, "s1", "bodycam")


def _approved(services: FakeServices) -> None:
    _add_session(services)
    with services.engine.begin() as conn:
        set_privacy_state(conn, "s1", PrivacyState.APPROVED)


def test_rendered_needs_a_current_render_record_f_coe_85(
    services: FakeServices, labeling: LocalStore, tmp_path: Path
) -> None:
    """F-COE-85: 렌더 기록이 무효(블러 변경)거나 블러본이 렌더 밖에서 바뀌면 렌더 안 됨이다."""
    _approved(services)
    assert _gate(services) is False
    _render(services, labeling, tmp_path, b"blurred-v1")
    assert _gate(services) is True
    invalidate_render(labeling, "s1", "bodycam", "collected:task-1", tmp_path)
    assert _gate(services) is False
    _render(services, labeling, tmp_path, b"blurred-v1")
    other = tmp_path / "other.mp4"
    other.write_bytes(b"replaced-outside-render")
    labeling.put_file(blurred_key("s1", "bodycam"), other, sha256_file(other))
    assert _gate(services) is False


def test_rendered_rejects_blur_made_under_old_render_policy_f_coe_85(
    services: FakeServices, labeling: LocalStore, tmp_path: Path
) -> None:
    """F-COE-85: 기록은 유효해도 렌더 정책만 바뀌면 낡은 블러본이라 렌더 안 됨이다."""
    _approved(services)
    _render(services, labeling, tmp_path, b"blurred-v1")
    assert _gate(services) is True
    policy = services.privacy_policy
    render = policy.render.model_copy(update={"crf": policy.render.crf + 1})
    services.privacy_policy = policy.model_copy(update={"render": render})
    assert _gate(services) is False


def test_training_view_without_rounds(client: Api) -> None:
    data = client.get("/api/training").json()
    assert data["rounds"] == [] and "objects" in data["trainers"]
    assert data["weekly"] is None
    res = client.post("/api/global/train_run", json={"task": "nope", "dataset_version": "d"})
    assert res.status_code == 400


def test_remove_upload_copy(client: Api, settings: ConsoleSettings) -> None:
    directory = settings.work_dir / "uploads" / "syn-09"
    directory.mkdir(parents=True)
    (directory / "original.mp4").write_bytes(b"x")
    res = client.delete("/api/uploads/syn-09")
    assert res.status_code == 200 and res.json() == {"removed": True}
    assert not directory.exists()
