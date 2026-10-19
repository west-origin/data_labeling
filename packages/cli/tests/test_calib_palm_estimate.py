"""`dlp calib palm-estimate|palm-revoke` 명령 경로 (ADR 0045·0052, F-COE-71, SQLite 파일 DB).

깊이 모델 가중치 대신 정답 깊이(`dlp_fixtures.hand3d.SceneDepth`)를 내는 가짜 `DepthLifter`를
끼우고, 원본은 `local:` 저장소(버킷 dlp-raw)에 실제 수집 경로로 올린다. 원본 읽기는 실제 감사
저장소(`raw_store`)를 거친다. 디코드한 H.264 프레임은 첫 픽셀의 프레임 번호를 보존하지 않으므로
프레임 순회만 픽스처 프레임으로 바꾼다. 서비스는 필요 없다.

정답: 작업자 손바닥 0.072 m(`hand_scale=0.8`), 사용 중지·캘리브레이션 없음·world 없는 트랙·원본 없음
세션은 원본을 읽지 않는다(감사 기록 없음), 잰 값은 명시 옵션 없이 덮지 않는다, 표본을 낸 세션 ID가
등록에 남고 그 세션을 사용 중지하면 등록이 결정에서 빠진다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy
import yaml
from numpy.typing import NDArray

from dlp_cli import calib_cmds, raw_access
from dlp_cli.main import main
from dlp_fixtures.hand3d import Hand3DScenario, generate_hand3d_scenario
from dlp_fixtures.video import write_video
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import store_from_spec
from dlp_schema import repo_root
from dlp_schema.calibration import CalibrationMethod
from dlp_schema.db.calibration_repository import (
    get_worker_hand_sizes,
    hand_sizes_to_review,
    register_camera_calibration,
    session_palm,
)
from dlp_schema.db.repository import (
    get_session,
    insert_labels,
    insert_withdrawal,
    list_raw_access,
    register_ontology,
)
from dlp_schema.labels import KeypointTrackPayload, without_world
from dlp_schema.lineage import Withdrawal
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, sqlite_db


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@dataclass
class _World:
    """시험 환경: DB URL, 원본 저장소 위치, 세션 ID → 장면, 지금 읽는 세션."""

    url: str
    store: str
    scenes: dict[str, Hand3DScenario] = field(default_factory=dict[str, Hand3DScenario])
    current: list[str] = field(default_factory=list[str])

    def run(self, *args: str) -> int:
        return main(["calib", *args, "--url", self.url])

    def estimate(self, *args: str) -> int:
        return self.run("palm-estimate", "w01", "--store", self.store, *args)

    def raw_reads(self) -> list[str | None]:
        with sqlite_db(self.url, create=False).connect() as conn:
            return [
                e.session_id for e in list_raw_access(conn) if e.purpose == "calib.palm_estimate"
            ]


class _FakeDepth:
    def __init__(self, world: _World) -> None:
        self.world = world

    def predict(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        return self.world.scenes[self.world.current[-1]].depth.predict(image)


def _ingest(world: _World, tmp: Path, seed: int, camera_id: str | None, *, no_world: bool) -> str:
    """합성 손 세션 하나를 수집하고 손 트랙을 넣는다. 반환: 세션 ID."""
    s = generate_hand3d_scenario(seed, hand_scale=0.8, n_frames=200)
    d = tmp / f"s{seed}"
    d.mkdir()
    gray = np.full((240, 320, 3), 90, np.uint8)
    write_video(d / "bodycam.mp4", ((t, gray) for t in s.frame_times), width=320, height=240)
    manifest: dict[str, Any] = {
        "session_id": s.session_id, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    if camera_id:
        manifest["calibration"] = {"camera_id": camera_id}
    (d / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    labels = s.labels
    if no_world:
        labels = [
            x.model_copy(update={"payload": without_world(x.payload)})
            if isinstance(x.payload, KeypointTrackPayload)
            else x
            for x in labels
        ]
    raw = store_from_spec(world.store, "dlp-raw")
    eng = sqlite_db(world.url, create=False)
    with eng.begin() as conn:
        ingest_session(*load_manifest(d / "m.yaml"), raw, conn)
        insert_labels(conn, labels)
    eng.dispose()
    world.scenes[s.session_id] = s
    return s.session_id


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_World]:
    """작업자 w01의 세션 넷: 캘리브레이션(003), 사용 중지(004), 캘리브레이션 없음(005),
    world 없음(006).
    """
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    eng = sqlite_db(url)
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
    eng.dispose()
    monkeypatch.setattr(calib_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(raw_access, "sa", _AwareSqlalchemy())
    w = _World(url, f"local:{tmp_path / 'raw'}")
    calibrated = _ingest(w, tmp_path, 3, "cam-a", no_world=False)
    gone = _ingest(w, tmp_path, 4, "cam-a", no_world=False)
    _ingest(w, tmp_path, 5, None, no_world=False)
    _ingest(w, tmp_path, 6, "cam-a", no_world=True)
    eng = sqlite_db(url, create=False)
    with eng.begin() as conn:
        k = w.scenes[calibrated].intrinsics
        register_camera_calibration(conn, "cam-a", k, method=CalibrationMethod.IMPORTED,
                                    now=FIXED_TIME)  # fmt: skip
        insert_withdrawal(conn, Withdrawal(session_id=gone, reason="동의 철회",
                                           withdrawn_at=datetime.now(UTC)))  # fmt: skip
    eng.dispose()

    class _FakeLifter:
        version = "depth3d-fake+hf3+p0"

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def depth_model(self) -> _FakeDepth:
            return _FakeDepth(w)

    def _frames(path: Path) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        sid = path.name.split("__")[1]  # sessions__<세션>__raw__bodycam.mp4
        w.current.append(sid)
        return w.scenes[sid].frames()

    monkeypatch.setattr(calib_cmds, "DepthLifter", _FakeLifter)
    monkeypatch.setattr(calib_cmds, "iter_frames", _frames)
    yield w


def test_estimate_reads_only_usable_raw_and_records_sources(
    world: _World, capsys: pytest.CaptureFixture[str]
) -> None:
    """보고만 → --apply → 다시 --apply(멱등). 캘리브레이션된 세션의 원본만 감사 저장소로 읽고,
    표본을 낸 세션 ID와 추정기 판이 등록에 남는다 (F-COE-70·71, F-PMS-29).
    """
    assert world.estimate() == 0
    out = capsys.readouterr().out
    assert "hand3d-005: 캘리브레이션이 없어 건너뜀" in out
    assert "hand3d-006: 손 3D 점(world)" in out and "[미등록]" in out
    assert world.raw_reads() == ["hand3d-003"]  # 사용 중지·캘리브레이션 없음·world 없음은 읽지 않음
    assert world.estimate("--apply") == 0
    assert world.estimate("--apply") == 0
    assert "건너뜀" in capsys.readouterr().out
    with sqlite_db(world.url, create=False).connect() as conn:
        [h] = get_worker_hand_sizes(conn, "w01")
    assert h.palm_m == pytest.approx(0.072, abs=5e-4) and h.source.value == "estimated"
    assert h.source_session_ids == ("hand3d-003",) and h.estimator
    assert h.estimator.startswith("depth3d-fake") and "+pe" in h.estimator
    assert h.n_samples is not None and h.n_samples >= 30


def test_estimate_skips_missing_raw_and_reports_no_sessions(
    world: _World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """원본이 없는 세션은 감사 기록 없이 건너뛰고(F-COE-73), 세션 없는 작업자는 따로 알린다
    (F-COE-72).
    """
    (tmp_path / "raw" / "dlp-raw" / "sessions" / "hand3d-003" / "raw" / "bodycam.mp4").unlink()
    assert world.estimate() == calib_cmds.EXIT_NO_SAMPLES
    assert "hand3d-003: 원본이 없어 건너뜀" in capsys.readouterr().out
    assert world.raw_reads() == []
    assert world.run("palm-estimate", "nobody", "--store", world.store) == 3
    assert "[세션 없음]" in capsys.readouterr().out


def test_estimate_does_not_override_measured_without_flag(
    world: _World, capsys: pytest.CaptureFixture[str]
) -> None:
    """지금 쓰이는 값이 잰 값이면 --apply는 종료 코드 4로 거부하고, --override-measured면
    쓴다 (F-PMS-28).
    """
    assert world.run("palm", "w01", "0.084") == 0
    assert world.estimate("--apply") == calib_cmds.EXIT_MEASURED_EXISTS
    assert "[거부]" in capsys.readouterr().out
    with sqlite_db(world.url, create=False).connect() as conn:
        assert [h.source.value for h in get_worker_hand_sizes(conn, "w01")] == ["measured"]
    assert world.estimate("--apply", "--override-measured") == 0
    with sqlite_db(world.url, create=False).connect() as conn:
        assert [h.source.value for h in get_worker_hand_sizes(conn, "w01")] == [
            "measured", "estimated",
        ]  # fmt: skip


def test_withdrawal_during_estimate_blocks_the_write(
    world: _World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """추정 중 출처 세션이 사용 중지되면 잠근 뒤 다시 읽어 쓰지 않는다 (종료 코드 5, F-PMS-29)."""
    real = calib_cmds.share_lock_sessions

    def _withdraw_then_lock(conn: sqlalchemy.Connection, ids: Any) -> None:
        # 다른 트랜잭션의 사용 중지가 잠금 직전에 커밋된 것을 흉내 낸다
        insert_withdrawal(conn, Withdrawal(session_id="hand3d-003", reason="철회",
                                           withdrawn_at=datetime.now(UTC)))  # fmt: skip
        real(conn, ids)

    monkeypatch.setattr(calib_cmds, "share_lock_sessions", _withdraw_then_lock)
    assert world.estimate("--apply") == calib_cmds.EXIT_WITHDRAWN_DURING_RUN
    assert "[막힘]" in capsys.readouterr().out
    with sqlite_db(world.url, create=False).connect() as conn:
        assert get_worker_hand_sizes(conn, "w01") == []


def test_withdrawn_source_and_revoke_fall_back_to_policy(
    world: _World, capsys: pytest.CaptureFixture[str]
) -> None:
    """출처 세션을 사용 중지하면 그 추정이 빠지고, palm-revoke는 정책 기본값으로 되돌린다.

    무효는 행을 지우지 않는다. 세션이 모두 사용 중지되면 보관 알림 대상이 된다 (F-PMS-29·30).
    """
    assert world.estimate("--apply") == 0
    eng = sqlite_db(world.url, create=False)
    with eng.begin() as conn:
        session = get_session(conn, "hand3d-005")
        assert session_palm(conn, session).source == "w"
    assert world.run("palm", "w01", "0.084") == 0
    assert world.run("palm-revoke", "w01", "--reason", "동의 철회") == 0
    assert "지금 값: 정책 기본값" in capsys.readouterr().out
    assert world.run("palm-revoke", "w01", "--reason", "다시") == 0
    assert "건너뜀" in capsys.readouterr().out
    assert world.run("palm-revoke", "w01", "--reason", "x", "--record", "palm-nope") == 2
    with eng.begin() as conn:
        assert session_palm(conn, session).source == "p"
        assert len(get_worker_hand_sizes(conn, "w01")) == 2  # 지우지 않았다
        # 무효 뒤 새로 잰 값은 다시 쓰인다
    assert world.run("palm", "w01", "0.081") == 0
    with eng.begin() as conn:
        assert session_palm(conn, session).palm_m == 0.081
        for sid in ("hand3d-003", "hand3d-005", "hand3d-006"):
            insert_withdrawal(conn, Withdrawal(session_id=sid, reason="동의 철회",
                                               withdrawn_at=datetime.now(UTC)))  # fmt: skip
        [item] = hand_sizes_to_review(conn)
    assert item.worker_id == "w01"
    assert world.run("show", "--worker", "w01") == 0
    out = capsys.readouterr().out
    assert "쓰지 않음" in out and "무효" in out
    eng.dispose()


def test_reviewed_only_uses_only_reviewed_hand_tracks(
    world: _World, capsys: pytest.CaptureFixture[str]
) -> None:
    """--reviewed-only면 검수 전 손 트랙은 쓰지 않아 원본도 읽지 않는다 (F-PMS-27 완화)."""
    assert world.estimate("--reviewed-only") == calib_cmds.EXIT_NO_SAMPLES
    assert "hand3d-003: 손 3D 점(world)" in capsys.readouterr().out
    assert world.raw_reads() == []
