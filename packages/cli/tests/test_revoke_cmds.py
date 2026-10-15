"""검수·프라이버시 CLI의 매체 회수 연결 (ADR 0039).

시간대 보존 SQLite 파일 DB와 LocalStore로 돈다 (서비스 없음).

정답 근거: 회수 실행은 수집·탐지 커밋 뒤에 돌고, 대기가 남으면 종료 코드 4다. `--dry-run`은 쓰지
않는다. `--pending`은 모든 세션의 대기를 실행한다. CVAT 접속이 실패해도(회수 전용 설정) 매체
삭제는 한다.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pytest
import sqlalchemy
import yaml

from dlp_cli import dataset_cmds, privacy_cmds, review_cmds, review_config
from dlp_cli.review_config import REVOCATION_PENDING_EXIT
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.pipeline import StreamResult
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, read_render_meta, write_render_meta
from dlp_privacy.runner import (
    PrivacyGateError,
    RenderNotCurrentError,
    assert_render_current,
    render_hash,
    request_media_revocation,
)
from dlp_review.revoke import RevocationReport
from dlp_review.tasks import ReviewSetup, labeling_media_key
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    active_privacy_hold,
    get_review_task,
    get_session,
    insert_labels,
    insert_review_task,
    insert_session,
    list_privacy_holds,
    mark_review_task_collected,
    mark_task_media_revoked,
    mark_task_tool_revoked,
    register_ontology,
    request_task_revocation,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_label, make_session, sqlite_db

NOW = FIXED_TIME + timedelta(hours=1)


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """세션 둘(s001, s002, 승인됨)을 넣은 SQLite 파일 DB URL. CLI 엔진도 시간대를 보존한다."""
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(db)
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    video = tmp_path / "v.bin"
    video.write_bytes(b"synthetic")
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for sid in ("s001", "s002"):
            key = f"sessions/{sid}/raw/bodycam.mp4"
            raw.put_file(key, video, sha256_file(video))
            stream = {"stream_id": "bodycam", "kind": "bodycam", "sync_method": "reference",
                      "uri": raw.uri(key)}  # fmt: skip
            insert_session(
                conn, make_session(sid, privacy_state=PrivacyState.APPROVED, streams=[stream])
            )
    engine.dispose()
    for module in (review_cmds, privacy_cmds):
        monkeypatch.setattr(module, "sa", _AwareSqlalchemy())
    return db


def _engine(url: str) -> sqlalchemy.Engine:
    """시험 쪽에서 DB를 읽고 쓸 엔진."""
    return sqlite_db(url, create=False)


def _labeling(tmp_path: Path) -> LocalStore:
    """라벨링 버킷 (로컬)."""
    return LocalStore(tmp_path / "store", "dlp-labeling")


def _media_at(tmp_path: Path, key: str) -> str:
    """라벨링 버킷의 key에 워터마크 사본을 둔다. URI."""
    store = _labeling(tmp_path)
    src = tmp_path / "m.bin"
    src.write_bytes(b"marked")
    store.put_file(key, src, sha256_file(src))
    return store.uri(key)


def _media(tmp_path: Path, sid: str, name: str) -> str:
    """labeler01의 워터마크 사본 `sessions/<세션>/review/labeler01/<name>.mp4`. URI."""
    return _media_at(tmp_path, f"sessions/{sid}/review/labeler01/{name}.mp4")


def _task(key: str, sid: str, uri: str, rhash: str | None) -> ReviewTask:
    """바디캠 라벨링 CVAT 작업."""
    return ReviewTask(
        task_key=key, tool=ReviewTool.CVAT, external_id=key.split(":")[1], session_id=sid,
        stream_id="bodycam", stage=ReviewStage.LABELING, media_uri=uri,
        label_kinds=("box_track",), created_at=FIXED_TIME, render_hash=rhash, assignee="labeler01",
    )  # fmt: skip


def _privacy_task(key: str, sid: str, tmp_path: Path) -> ReviewTask:
    """블러 검수 작업 (원본 프록시 URI, 회수 대상 아님)."""
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    return _task(key, sid, raw.uri(f"sessions/{sid}/derived/bodycam.proxy.mp4"), None).model_copy(
        update={"stage": ReviewStage.PRIVACY, "label_kinds": ("blur_track",)}
    )


def _face(sid: str = "s001", label_id: str = "b-new") -> Any:
    """바디캠 얼굴 블러 (사람, 작업보다 늦게 만든 것)."""
    payload = {"kind": "blur_track", "target": "face",
               "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}]}  # fmt: skip
    return make_label(
        payload, label_id=label_id, session_id=sid, stream_id="bodycam", t_end_ms=100,
        created_at=NOW,
    )  # fmt: skip


def _local_setup(tmp_path: Path) -> ReviewSetup:
    """도구 없는 로컬 검수 설정 (CVAT 로그인 실패를 견딘 것과 같다)."""
    return ReviewSetup(
        raw=LocalStore(tmp_path / "store", "dlp-raw"), labeling=_labeling(tmp_path),
        labeling_reader=Mock(), ontology=None, cvat=None, label_studio=None,
        privacy=load_privacy_policy(repo_root()),
    )  # fmt: skip


def _revocation_only(tmp_path: Path) -> Any:
    """`review_setup` 대역: 회수 전용(`revocation_only`) 설정만 받고 CVAT 접속 실패 경고를 낸다."""

    def fake(purpose: str, url: str | None, **kw: Any) -> tuple[ReviewSetup, list[str]]:
        assert kw.get("revocation_only") is True
        return _local_setup(tmp_path), ["CVAT에 접속하지 못했습니다 (ConnectError)"]

    return fake


def _revoke_args(url: str, **kw: Any) -> argparse.Namespace:
    """`dlp review revoke` 인자 (기본값)."""
    values: dict[str, Any] = dict(
        session_id=None, stream=[], task=[], all=False, include_legacy=False, dry_run=False,
        reason=None, pending=False, mark_tool_gone=None, note=None, url=url, orphans=False,
        apply=False,
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_collect_runs_revocations_after_commit_and_exits_4_when_pending(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """이미 수집한 작업(결과 None)이어도 커밋 뒤 그 세션의 회수를 실행하고, 대기가 남으면 4."""
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "bodycam"), None))
        mark_review_task_collected(conn, "cvat:2", NOW)
    setup = _local_setup(tmp_path)
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=setup))
    run = Mock(return_value=RevocationReport(pending=["label_studio:9"]))
    monkeypatch.setattr(review_cmds, "run_revocations", run)
    args = argparse.Namespace(task_key="cvat:2", reviewer="labeler01", url=url)
    assert review_cmds.cmd_collect(args) == REVOCATION_PENDING_EXIT
    assert run.call_args.args[1:] == ("s001", setup)


def test_collect_requests_in_transaction_and_revokes_media_after_commit(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """블러 수집의 회수 요청이 커밋된 뒤 실제 실행기가 돌아 라벨링 버킷 사본을 지운다.

    실행기는 별도 연결로 커밋된 요청만 본다. `run_revocations`를 수집 트랜잭션 안에서 부르면 요청이
    아직 보이지 않아 사본이 남고(다음 cron까지 노출), 이 시험이 실패한다.
    """
    uri = _media(tmp_path, "s001", "bodycam")
    with _engine(url).begin() as conn:
        insert_review_task(conn, _privacy_task("cvat:1", "s001", tmp_path))
        insert_review_task(conn, _task("cvat:2", "s001", uri, None))
    policy = load_privacy_policy(repo_root())

    def collect(conn: sqlalchemy.Connection, *_: Any) -> Any:
        """블러 검수 수집 대역: 얼굴을 더하고 같은 트랜잭션에서 회수를 요청한다."""
        insert_labels(conn, [_face()])
        request_media_revocation(
            conn, "s001", policy, NOW, reason="blur_changed:cvat:1", streams={"bodycam"},
            legacy=True,
        )  # fmt: skip
        return None

    seen: list[bool] = []
    real_run = review_config.run_revocations

    def run(engine: sqlalchemy.Engine, sid: str | None, *a: Any, **kw: Any) -> RevocationReport:
        # 다른 엔진(다른 연결)에서 요청이 보여야 커밋 뒤다
        with _engine(url).connect() as conn:
            seen.append(get_review_task(conn, "cvat:2").revoke_requested_at is not None)
        return real_run(engine, sid, *a, **kw)

    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=_local_setup(tmp_path)))
    monkeypatch.setattr(review_cmds, "collect_task", collect)
    monkeypatch.setattr(review_cmds, "run_revocations", run)
    args = argparse.Namespace(task_key="cvat:1", reviewer="privacy01", url=url)
    assert review_cmds.cmd_collect(args) == REVOCATION_PENDING_EXIT  # CVAT 클라이언트 없음
    assert seen == [True]
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.media_revoked_at is not None and task.tool_revoked_at is None
    assert _labeling(tmp_path).head("sessions/s001/review/labeler01/bodycam.mp4") is None


def test_revoke_dry_run_writes_nothing_and_estimates_legacy(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--dry-run`은 대상과 이유만 보이고 쓰지 않는다. 구 작업에는 생성 시점 추정을 붙인다."""
    empty = render_hash([], load_privacy_policy(repo_root()))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "a"), empty))
        insert_review_task(conn, _task("cvat:3", "s001", _media(tmp_path, "s001", "b"), None))
        insert_labels(conn, [_face()])
    args = _revoke_args(url, session_id="s001", dry_run=True, include_legacy=True)
    assert review_cmds.cmd_revoke(args) == 0
    out = capsys.readouterr().out
    assert "cvat:2" in out and "hash" in out and "추정: 생성 시점 블러가 지금과 다름" in out
    with _engine(url).connect() as conn:
        assert get_review_task(conn, "cvat:2").revoke_requested_at is None
        assert get_review_task(conn, "cvat:3").status is ReviewTaskStatus.OPEN


def test_revoke_requires_reason_for_task_and_all(url: str) -> None:
    """`--task`·`--all`은 사유 없이 쓰지 않는다 (사고 대응 기록)."""
    with pytest.raises(SystemExit, match="--reason"):
        review_cmds.cmd_revoke(_revoke_args(url, session_id="s001", all=True))


def test_revoke_dry_run_never_writes_with_pending_or_mark_tool_gone(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--pending --dry-run`은 대기만 보이고 지우지 않는다. `--mark-tool-gone`은 `--dry-run`과 함께
    쓸 수 없다 (함께 쓰면 실제로 표시돼 노출이 남은 채 대기에서 빠질 수 있었다)."""
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "bodycam"), None))
        request_task_revocation(conn, "cvat:2", reason="sweep", at=NOW)
    no_setup = Mock(side_effect=AssertionError("미리보기는 저장소·도구 설정을 만들지 않는다"))
    monkeypatch.setattr(review_config, "review_setup", no_setup)
    assert review_cmds.cmd_revoke(_revoke_args(url, pending=True, dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "cvat:2" in out and "매체·도구" in out and "회수 대기 1개 (쓰지 않음)" in out
    gone = _revoke_args(url, mark_tool_gone="cvat:2", note="CVAT 폐기", dry_run=True)
    with pytest.raises(SystemExit, match="--dry-run"):
        review_cmds.cmd_revoke(gone)
    with pytest.raises(SystemExit, match="--stream"):
        review_cmds.cmd_revoke(_revoke_args(url, pending=True, stream=["bodycam"]))
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.media_revoked_at is None and task.tool_revoked_at is None
    assert _labeling(tmp_path).head("sessions/s001/review/labeler01/bodycam.mp4") is not None


def test_revoke_task_refuses_keys_it_cannot_request(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--task`가 다른 세션·없는 작업·블러 검수 작업이면 아무것도 쓰지 않고 거부한다.

    예전에는 조용히 빠져 출력 없이 0으로 끝나, 사고 대응 운영자가 회수됐다고 오해할 수 있었다.
    이미 회수 요청된 작업은 그렇다고 알리고 실행을 계속한다.
    """
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", _media(tmp_path, "s001", "a"), None))
        insert_review_task(conn, _task("cvat:3", "s002", _media(tmp_path, "s002", "a"), None))
        insert_review_task(conn, _privacy_task("cvat:4", "s001", tmp_path))
    bad_keys = {"cvat:3": "세션 s002의 작업", "cvat:9": "없는 작업", "cvat:4": "블러 검수"}
    for bad, why in bad_keys.items():
        args = _revoke_args(url, session_id="s001", task=["cvat:2", bad], reason="incident")
        with pytest.raises(SystemExit, match=why):
            review_cmds.cmd_revoke(args)
        with pytest.raises(SystemExit, match=why):
            review_cmds.cmd_revoke(argparse.Namespace(**{**vars(args), "dry_run": True}))
    with _engine(url).connect() as conn:
        for key in ("cvat:2", "cvat:3"):
            assert get_review_task(conn, key).revoke_requested_at is None
    with _engine(url).begin() as conn:
        request_task_revocation(conn, "cvat:2", reason="sweep", at=NOW)
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    args = _revoke_args(url, session_id="s001", task=["cvat:2"], reason="incident")
    assert review_cmds.cmd_revoke(args) == REVOCATION_PENDING_EXIT  # CVAT 클라이언트 없음
    assert "cvat:2: 이미 회수 요청됨 (sweep" in capsys.readouterr().out


def test_revoke_task_records_manual_reason_only_for_named_task(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--task K --reason R`는 K를 `manual:<행위자>:R`로 요청하고, 낡은 다른 작업은 sweep 사유로
    따로 요청한다 (사고 대응 기록에 sweep 대상이 섞이지 않는다)."""
    empty = render_hash([], load_privacy_policy(repo_root()))
    stale = _media_at(tmp_path, labeling_media_key("s001", "labeler01", "bodycam", empty))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", stale, empty))  # 블러가 바뀌어 낡음
        insert_review_task(conn, _task("cvat:3", "s001", _media(tmp_path, "s001", "b"), None))
        insert_labels(conn, [_face()])
    monkeypatch.setenv("DLP_ACTOR", "ops")
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    args = _revoke_args(url, session_id="s001", task=["cvat:3"], reason="incident")
    assert review_cmds.cmd_revoke(args) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "회수 요청 cvat:3 (bodycam, open, 수동)" in out
    assert "회수 요청 cvat:2 (bodycam, open, sweep)" in out
    with _engine(url).connect() as conn:
        assert get_review_task(conn, "cvat:3").revoke_reason == "manual:ops:incident"
        assert get_review_task(conn, "cvat:2").revoke_reason == "sweep"


def test_revoke_pending_runs_every_session_and_tolerates_missing_tools(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--pending`은 대기가 있는 모든 세션의 매체를 지운다. 도구가 없으면 도구 작업은 대기다."""
    keys = {"s001": "cvat:2", "s002": "cvat:3"}
    with _engine(url).begin() as conn:
        for sid, key in keys.items():
            insert_review_task(conn, _task(key, sid, _media(tmp_path, sid, "bodycam"), None))
            request_task_revocation(conn, key, reason="sweep", at=NOW)
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    assert review_cmds.cmd_revoke(_revoke_args(url, pending=True)) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "CVAT에 접속하지 못했습니다" in out and "회수 대기 2개" in out
    with _engine(url).connect() as conn:
        for key in keys.values():
            task = get_review_task(conn, key)
            assert task.media_revoked_at is not None and task.tool_revoked_at is None
    for sid in keys:
        assert _labeling(tmp_path).head(f"sessions/{sid}/review/labeler01/bodycam.mp4") is None


def test_detect_requests_and_runs_revocation_even_without_tools(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """탐지로 블러가 생기면 같은 커밋에 회수를 요청하고 커밋 뒤 실행한다.

    도구가 없어도 매체는 지운다.
    """
    empty = render_hash([], load_privacy_policy(repo_root()))
    uri = _media_at(tmp_path, labeling_media_key("s001", "labeler01", "bodycam", empty))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, empty))
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    model = Provenance(source=Source.MODEL, model_version="m1")
    found = [_face().model_copy(update={"provenance": model, "confidence": 0.9})]
    args = argparse.Namespace(session_id="s001", store="unused", url=url)
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    with (
        patch("dlp_cli.privacy_cmds.raw_store", return_value=raw),
        patch("dlp_cli.privacy_cmds.build_detectors", return_value=({}, {})),
        patch("dlp_cli.privacy_cmds.store_from_spec", return_value=_labeling(tmp_path)),
        patch(
            "dlp_cli.privacy_cmds.deployed_predictors",
            return_value=SimpleNamespace(notes=[], predictors=[]),
        ),
        patch("dlp_privacy.runner.detect_video", return_value=StreamResult(found, [], {}, 3)),
    ):
        assert privacy_cmds.cmd_detect(args) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "라벨링 작업 1개 회수: cvat:2" in out and "회수 대기 1개" in out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "privacy_detect"
    assert task.media_revoked_at is not None


def _create_args(url: str, **kw: Any) -> argparse.Namespace:
    """`dlp review create --stage labeling` 인자."""
    values: dict[str, Any] = dict(
        session_id="s001", stage="labeling", assignee="labeler01", stream=[], tool=None,
        no_cvat=False, no_ls=False, url=url, review_command="create",
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_create_passes_stream_and_tool_and_stops_on_pending(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--stream`·`--tool`을 작업 생성에 넘기고 먼저 sweep 회수를 돈다. 대기가 남으면 4로 멈춘다."""
    monkeypatch.setattr(review_cmds, "_setup", Mock(return_value=_local_setup(tmp_path)))
    run = Mock(return_value=RevocationReport())
    create = Mock(return_value=[])
    monkeypatch.setattr(review_cmds, "run_revocations", run)
    monkeypatch.setattr(review_cmds, "create_labeling_tasks", create)
    args = _create_args(url, stream=["bodycam"], tool="label_studio")
    assert review_cmds.cmd_create(args) == 0
    assert create.call_args.kwargs == {"streams": {"bodycam"}, "tools": {ReviewTool.LABEL_STUDIO}}
    assert run.call_args.kwargs == {"sweep": True}
    run.return_value = RevocationReport(pending=["cvat:2"])
    create.reset_mock()
    assert review_cmds.cmd_create(_create_args(url)) == REVOCATION_PENDING_EXIT
    create.assert_not_called()


def test_render_backstop_requests_tasks_made_under_other_render_policy(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """렌더 정책(여기서는 화질 crf)을 바꾼 뒤 렌더하면 이전 정책의 블러본으로 만든 작업을 회수
    요청한다.

    렌더 해시에는 render 절 전체(모자이크 크기, 인코더 crf·encoder_rate)가 들어가 블러와 무관한
    인코더 설정만 바꿔도 회수된다 (ADR 0039 운영상 결과). 같은 정책으로 다시 렌더하면 해시가 같아
    요청하지 않는다 (렌더 백스톱).
    """
    from dlp_fixtures.video import write_video

    raw = LocalStore(tmp_path / "store", "dlp-raw")
    video = tmp_path / "tiny.mp4"
    frames = [(t, np.zeros((48, 64, 3), dtype=np.uint8)) for t in (0, 100)]
    write_video(video, frames, width=64, height=48)
    raw.put_file("sessions/s001/raw/bodycam.mp4", video, sha256_file(video))
    policy = load_privacy_policy(repo_root())
    other_crf = policy.model_copy(
        update={"render": policy.render.model_copy(update={"crf": policy.render.crf + 1})}
    )
    old = render_hash([], policy)
    uri = _media_at(tmp_path, labeling_media_key("s001", "labeler01", "bodycam", old))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, old))
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    monkeypatch.setattr(privacy_cmds, "raw_store", Mock(return_value=raw))
    monkeypatch.setattr(privacy_cmds, "store_from_spec", Mock(return_value=_labeling(tmp_path)))
    args = argparse.Namespace(session_id="s001", store="unused", url=url)
    monkeypatch.setattr(privacy_cmds, "load_policy", Mock(return_value=policy))
    assert privacy_cmds.cmd_render(args) == 0
    with _engine(url).connect() as conn:
        assert get_review_task(conn, "cvat:2").revoke_requested_at is None
    monkeypatch.setattr(privacy_cmds, "load_policy", Mock(return_value=other_crf))
    assert privacy_cmds.cmd_render(args) == REVOCATION_PENDING_EXIT
    assert "렌더 해시가 바뀐 라벨링 작업 1개 회수: cvat:2" in capsys.readouterr().out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
    assert task.status is ReviewTaskStatus.REVOKED and task.revoke_reason == "render"
    assert task.media_revoked_at is not None


def _orphan_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_age_s: int) -> None:
    """기본 review.yaml에서 `orphans.min_age_s`만 바꾼 파일을 `DLP_REVIEW_POLICY_FILE`로 고른다."""
    data = yaml.safe_load((repo_root() / "config/policies/review.yaml").read_text("utf-8"))
    data["orphans"]["min_age_s"] = min_age_s
    path = tmp_path / "review-orphans.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("DLP_REVIEW_POLICY_FILE", str(path))


def test_revoke_orphans_reports_by_default_and_deletes_only_with_apply(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-REV-15: `--orphans`는 기본이 보고만(종료 코드 4)이고 `--apply`일 때만 지운다 (그 뒤 0).

    DB 행이 있는 사본과 시계열 CSV는 그대로다. 나이 하한은 고른 정책 파일에서 읽는다 (하한을
    늘리면 막 만든 사본은 보지 않아 0). 도구가 없으면 그 채널은 확인하지 못함으로 4다.
    """
    live = _media(tmp_path, "s001", "bodycam")
    orphan_key = "sessions/s001/review/labeler02/bodycam.mp4"
    _media_at(tmp_path, orphan_key)
    _media_at(tmp_path, "sessions/s001/review/timeseries.csv")
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", live, None))
    setup = _local_setup(tmp_path)
    no_tools = Mock(return_value=(setup, []))
    monkeypatch.setattr(review_cmds, "review_setup", no_tools)
    _orphan_policy(tmp_path, monkeypatch, 0)
    args = _revoke_args(url, orphans=True)
    assert review_cmds.cmd_revoke(args) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert f"남음 media        {orphan_key} (s001, no_db_row" in out
    assert "보고만" in out and "확인하지 못함: cvat: 클라이언트 없음" in out
    assert no_tools.call_args.kwargs["revocation_only"] is True
    assert _labeling(tmp_path).head(orphan_key) is not None
    assert review_cmds.cmd_revoke(_revoke_args(url, orphans=True, apply=True)) == 4
    assert f"지움 media        {orphan_key}" in capsys.readouterr().out
    assert _labeling(tmp_path).head(orphan_key) is None
    assert _labeling(tmp_path).head("sessions/s001/review/labeler01/bodycam.mp4") is not None
    assert _labeling(tmp_path).head("sessions/s001/review/timeseries.csv") is not None

    _media_at(tmp_path, orphan_key)
    cvat, ls = Mock(), Mock()
    cvat.list_tasks.return_value = []
    ls.list_tasks.return_value = []
    with_tools = replace(setup, cvat=cvat, label_studio=ls)
    monkeypatch.setattr(review_cmds, "review_setup", Mock(return_value=(with_tools, [])))
    _orphan_policy(tmp_path, monkeypatch, 3600)
    assert review_cmds.cmd_revoke(_revoke_args(url, orphans=True, apply=True)) == 0
    assert "어려 보지 않음 1개" in capsys.readouterr().out
    assert _labeling(tmp_path).head(orphan_key) is not None
    _orphan_policy(tmp_path, monkeypatch, 0)
    assert review_cmds.cmd_revoke(_revoke_args(url, orphans=True, apply=True)) == 0
    assert _labeling(tmp_path).head(orphan_key) is None


def test_revoke_orphans_argument_rules(url: str) -> None:
    """`--apply`는 `--orphans`와만, `--orphans`는 회수 선택 인자·`--pending`과 함께 쓰지 않는다."""
    bad = {
        "--apply는 --orphans와만": {"session_id": "s001", "apply": True},
        "--apply와 --dry-run": {"orphans": True, "apply": True, "dry_run": True},
        "--orphans는 --all, --reason, --pending": {
            "orphans": True, "all": True, "reason": "x", "pending": True,
        },
    }  # fmt: skip
    for why, kw in bad.items():
        with pytest.raises(SystemExit, match=why):
            review_cmds.cmd_revoke(_revoke_args(url, **kw))


def test_dataset_withdraw_requests_and_runs_revocation_after_commit(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-REV-17: `dlp dataset withdraw`는 사용 중지와 같은 커밋에 배포된 작업 회수를 요청하고, 커밋
    뒤 실행기를 돌려 라벨링 버킷 사본을 지운다. 도구가 없으면 도구 작업은 대기로 남아 종료 코드 4다.

    블러 검수 작업은 회수 대상이 아니다. 회수할 작업이 없는 세션은 실행기 설정을 만들지 않고 0이다.
    """
    uri = _media(tmp_path, "s001", "bodycam")
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, None))
        insert_review_task(conn, _privacy_task("cvat:1", "s001", tmp_path))
    monkeypatch.setattr(dataset_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    args = argparse.Namespace(session_id="s001", reason="동의 철회", actor="dpo01", url=url)
    assert dataset_cmds.cmd_withdraw(args) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "검수 작업 1개 회수 요청: cvat:2" in out and "회수 대기 1개" in out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
        assert task.revoke_reason == "withdrawn:dpo01" and task.media_revoked_at is not None
        assert get_review_task(conn, "cvat:1").revoke_requested_at is None
    assert _labeling(tmp_path).head("sessions/s001/review/labeler01/bodycam.mp4") is None
    no_setup = Mock(side_effect=AssertionError("회수할 작업이 없으면 설정을 만들지 않는다"))
    monkeypatch.setattr(review_config, "review_setup", no_setup)
    args = argparse.Namespace(session_id="s002", reason="동의 철회", actor="dpo01", url=url)
    assert dataset_cmds.cmd_withdraw(args) == 0


def test_dataset_withdraw_deletes_timeseries_csv_after_all_tasks_were_already_revoked(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """사용 중지 전에 모든 작업의 회수가 이미 끝났어도 `dlp dataset withdraw`가 시계열 CSV를 지운다.

    정답 근거: s001의 Label Studio 작업은 블러 변경으로 매체·도구 회수가 모두 끝났다. 사용 중지는
    새로 요청할 작업이 없어 회수 대기가 비지만, CSV(동의를 철회한 작업자의 센서 값, 서명 URL이
    가리킨다)는 남아 있다. 회수 실행기를 돌려(도구 클라이언트 없이) CSV를 지우고 0이다. 다시 돌리면
    지울 것이 없어 0이다. 회수 대기도 Label Studio 작업도 없는 사용 중지 세션은 설정을 만들지
    않는다.
    """
    csv = "sessions/s001/review/timeseries.csv"
    _media_at(tmp_path, csv)
    ls = ReviewTask(
        task_key="label_studio:3", tool=ReviewTool.LABEL_STUDIO, external_id="3",
        session_id="s001", stream_id="bodycam", stage=ReviewStage.LABELING,
        media_uri=_media(tmp_path, "s001", "bodycam"), label_kinds=("action",),
        created_at=FIXED_TIME, render_hash=None, assignee="labeler01",
    )  # fmt: skip
    with _engine(url).begin() as conn:
        insert_review_task(conn, ls)
        assert request_task_revocation(conn, ls.task_key, reason="blur_changed:cvat:1", at=NOW)
        mark_task_media_revoked(conn, "s001", ls.media_uri, NOW)
        assert mark_task_tool_revoked(conn, ls.task_key, NOW)
    seen: list[dict[str, Any]] = []

    def fake(purpose: str, url: str | None, **kw: Any) -> tuple[ReviewSetup, list[str]]:
        seen.append(kw)
        return _local_setup(tmp_path), []

    monkeypatch.setattr(dataset_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(review_config, "review_setup", fake)
    args = argparse.Namespace(session_id="s001", reason="동의 철회", actor="dpo01", url=url)
    assert dataset_cmds.cmd_withdraw(args) == 0
    assert "회수: 매체 1개" in capsys.readouterr().out
    assert _labeling(tmp_path).head(csv) is None
    assert seen == [{"revocation_only": True, "cvat": False, "label_studio": False}]
    assert dataset_cmds.cmd_withdraw(args) == 0  # 멱등: 지울 것이 없다
    no_setup = Mock(side_effect=AssertionError("CSV를 확인할 세션이 없으면 설정을 만들지 않는다"))
    monkeypatch.setattr(review_config, "review_setup", no_setup)
    args = argparse.Namespace(session_id="s002", reason="동의 철회", actor="dpo01", url=url)
    assert dataset_cmds.cmd_withdraw(args) == 0


def test_print_revocations_counts_a_left_timeseries_csv_as_pending(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """지우지 못한 사용 중지 세션 CSV만 남아도 대기(종료 코드 4)다: 작업 행 표시가 없어 pending에
    들지 않지만 서명 URL이 계속 열린다."""
    csv = "sessions/s001/review/timeseries.csv"
    errors = [f"[매체 노출 남음] {csv}: 디스크 오류"]
    report = RevocationReport(errors=errors, timeseries_left=[csv])
    assert not report.complete and report.remaining == [csv]
    assert review_config.print_revocations(report, "s001") == REVOCATION_PENDING_EXIT
    assert f"회수 대기 1개: {csv} (dlp review revoke s001)" in capsys.readouterr().out


def test_privacy_hold_revokes_current_tasks_and_release_needs_reapproval(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-REV-18: `dlp privacy hold`는 렌더 해시가 지금과 같은 작업도 회수 요청하고 커밋 뒤 사본을
    지운다 (도구가 없으면 4). 보류 중에는 승인이 거부된다. `--release`는 기록만 남기고 승인은 풀린
    채다. 보류 중이 아니면 해제를 거부한다."""
    current = render_hash([], load_privacy_policy(repo_root()))
    uri = _media_at(tmp_path, labeling_media_key("s001", "labeler01", "bodycam", current))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, current))
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    monkeypatch.setattr(privacy_cmds, "store_from_spec", Mock(return_value=_labeling(tmp_path)))
    hold = argparse.Namespace(
        session_id="s001", reason="labeler report", release=False, actor="ops01", url=url,
        store="unused",
    )  # fmt: skip
    assert privacy_cmds.cmd_hold(hold) == REVOCATION_PENDING_EXIT
    out = capsys.readouterr().out
    assert "승인 보류 (새 보류, ops01: labeler report" in out and "회수: cvat:2" in out
    with _engine(url).connect() as conn:
        task = get_review_task(conn, "cvat:2")
        assert task.revoke_reason == "hold:ops01:labeler report"
        assert task.media_revoked_at is not None
        assert get_session(conn, "s001").privacy_state is PrivacyState.AUTO_BLURRED
    approve = argparse.Namespace(session_id="s001", url=url)
    with pytest.raises(PrivacyGateError, match="승인 보류 중"):
        privacy_cmds.cmd_approve(approve)
    release = argparse.Namespace(**{**vars(hold), "release": True, "reason": "re-reviewed"})
    assert privacy_cmds.cmd_hold(release) == 0
    assert "재승인" in capsys.readouterr().out
    with _engine(url).connect() as conn:
        assert get_session(conn, "s001").privacy_state is PrivacyState.AUTO_BLURRED
    with pytest.raises(PrivacyGateError, match="보류 중이 아닙니다"):
        privacy_cmds.cmd_hold(release)


class _BrokenLabeling(LocalStore):
    """렌더 기록 쓰기가 실패하는 라벨링 버킷 (저장소 장애 흉내)."""

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        raise OSError(f"{key}: 저장소 쓰기 실패")


def test_privacy_hold_commits_db_even_when_render_invalidation_fails(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-PMS-14: 렌더 기록 무효화(저장소 쓰기)가 실패해도 보류·승인 해제·회수 요청은 커밋된다.

    정답: 저장소 장애는 DB를 되돌리지 않는다 — 보류 기록이 남고 AUTO_BLURRED, 작업 회수 요청이
    있고, 렌더 게이트(`assert_render_current`)가 거부한다. CLI는 무효화 실패를 알리고 0이 아닌
    코드로 끝난다. 다시 실행하면(저장소 복구) 보류 기록은 더하지 않고 렌더 기록을 무효로 둔다.
    """
    policy = load_privacy_policy(repo_root())
    current = render_hash([], policy)
    labeling = _labeling(tmp_path)
    _media_at(tmp_path, blurred_key("s001", "bodycam"))
    write_render_meta(labeling, "s001", "bodycam", RenderMeta(current, "0" * 64), tmp_path)
    uri = _media_at(tmp_path, labeling_media_key("s001", "labeler01", "bodycam", current))
    with _engine(url).begin() as conn:
        insert_review_task(conn, _task("cvat:2", "s001", uri, current))
    monkeypatch.setattr(review_config, "review_setup", _revocation_only(tmp_path))
    broken = _BrokenLabeling(tmp_path / "store", "dlp-labeling")
    monkeypatch.setattr(privacy_cmds, "store_from_spec", Mock(return_value=broken))
    hold = argparse.Namespace(
        session_id="s001", reason="labeler report", release=False, actor="ops01", url=url,
        store="unused",
    )  # fmt: skip
    assert privacy_cmds.cmd_hold(hold) not in (0, None)
    assert "렌더 기록 무효화 실패" in capsys.readouterr().out
    with _engine(url).connect() as conn:
        assert active_privacy_hold(conn, "s001") is not None
        assert get_session(conn, "s001").privacy_state is PrivacyState.AUTO_BLURRED
        assert get_review_task(conn, "cvat:2").revoke_reason == "hold:ops01:labeler report"
        with pytest.raises(RenderNotCurrentError):
            assert_render_current(conn, labeling, "s001", "bodycam", policy)
    # 렌더 기록은 아직 그대로다 (무효화 실패) → 다시 실행하면 무효로 둔다
    meta = read_render_meta(labeling, "s001", "bodycam", tmp_path / "m1")
    assert meta is not None and meta.render_hash == current
    monkeypatch.setattr(privacy_cmds, "store_from_spec", Mock(return_value=labeling))
    privacy_cmds.cmd_hold(hold)
    assert "렌더 기록 무효화 실패" not in capsys.readouterr().out
    meta = read_render_meta(labeling, "s001", "bodycam", tmp_path / "m2")
    assert meta is not None and meta.render_hash is None
    with _engine(url).connect() as conn:
        assert len(list_privacy_holds(conn, "s001")) == 1
