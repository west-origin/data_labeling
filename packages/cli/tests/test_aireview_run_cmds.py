"""`dlp aireview run|status|show|waive` CLI 종단 테스트 (WP17.5, ADR 0040 5.2·5.3).

서버는 가짜 OpenAI 호환 서버(`FakeOpenAIServer`, 127.0.0.1), DB는 시간대를 보존하는 SQLite 파일,
라벨링·원본 버킷은 로컬 저장소(`--store local:<디렉터리>`)다. 블러본은 합성 프레임 MP4에 지금 렌더
해시로 렌더 기록을 쓴 것이다 (서비스 불필요).

정답 근거: 종료 코드 표(모듈 docstring), AI 표 행 수, 판정 전후 라벨 이력, 가짜 서버가 받은 요청,
프리라벨과 같은 함수(`glove_contact_intervals`)로 직접 계산한 장갑 접촉 해시, 원본 접근 감사 기록.
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy

from dlp_aireview.policy import load_policy
from dlp_aireview.sensors import SensorContext
from dlp_aireview.testing import FakeOpenAIServer
from dlp_cli import aireview_cmds, aireview_run_cmds
from dlp_cli.main import main
from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.aireview import aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_prelabel.contact import glove_contact_intervals
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_schema import load_config, repo_root
from dlp_schema.db.aireview_repository import (
    list_ai_run_events,
    list_ai_runs,
    list_ai_units,
)
from dlp_schema.db.repository import (
    insert_labels,
    insert_session,
    record_privacy_hold,
    register_ontology,
)
from dlp_schema.db.tables import label_records, raw_access_log
from dlp_schema.labels import Hand, HandStatePayload
from dlp_schema.ontology import load_ontology
from dlp_schema.session import LifecycleState, PrivacyState, Session, StreamKind, SyncMethod
from dlp_schema.testing import sqlite_db
from dlp_sync.signals import glove_series

BUCKETS = load_config(repo_root() / "config/defaults.yaml").buckets
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


class World:
    """DB·로컬 버킷 하나와 세션을 더하는 도우미."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.url = f"sqlite:///{tmp_path / 'cli.db'}"
        self.engine = sqlite_db(self.url)
        self.store = tmp_path / "store"
        self.labeling = LocalStore(self.store, BUCKETS.labeling)
        self.raw = LocalStore(self.store, BUCKETS.raw)
        self.privacy = load_privacy_policy(repo_root())
        with self.engine.begin() as conn:
            register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))

    @property
    def args(self) -> list[str]:
        """공통 인자 (DB·저장소)."""
        return ["--url", self.url, "--store", f"local:{self.store}"]

    def add(
        self,
        sid: str,
        *,
        lifecycle: LifecycleState = LifecycleState.PRELABELED,
        privacy: PrivacyState = PrivacyState.APPROVED,
        glove: bool = False,
        upload_glove: bool = True,
        publish: bool = True,
    ) -> Session:
        """합성 세션 하나 (장갑이면 오른손 장갑 스트림과 원본 버킷 Parquet)."""
        scene = aireview_scene(0, session_id=sid, n_units=2)
        streams: list[dict[str, Any]] = [s.model_dump() for s in scene.session.streams]
        if glove:
            key = f"sessions/{sid}/glove_right.parquet"
            if upload_glove:
                out = self.tmp / "glove" / sid
                generate_action_scenario(0, session_id=sid, n_units=2).write(out)
                path = out / "glove_right.parquet"
                self.raw.put_file(key, path, sha256_file(path))
            streams.append({
                "stream_id": "glove_r", "kind": StreamKind.GLOVE_RIGHT, "uri": self.raw.uri(key),
                "sample_rate_hz": 100.0, "sync_method": SyncMethod.SHARED_CLOCK,
            })  # fmt: skip
        session = Session.model_validate(
            scene.session.model_dump()
            | {"streams": streams, "lifecycle_state": lifecycle, "privacy_state": privacy}
        )
        with self.engine.begin() as conn:
            insert_session(conn, session)
            insert_labels(conn, list(scene.labels))
        if publish and privacy is PrivacyState.APPROVED:
            for stream in ("bodycam", "tp1"):
                self._publish(scene, stream)
        return session

    def _publish(self, scene: Any, stream: str) -> None:
        """합성 블러본과 지금 렌더 해시의 렌더 기록."""
        sid = scene.session.session_id
        video = self.tmp / f"{sid}-{stream}.mp4"
        frames = ((t, scene_frame(scene, stream, t)) for t in scene.frame_times)
        write_video(video, frames, width=scene.width, height=scene.height)
        sha = sha256_file(video)
        self.labeling.put_file(blurred_key(sid, stream), video, sha)
        with self.engine.connect() as conn:
            digest = expected_render_hash(conn, sid, stream, self.privacy)
        write_render_meta(self.labeling, sid, stream, RenderMeta(digest, sha), self.tmp)

    def label_rows(self) -> list[tuple[Any, ...]]:
        """라벨 이력 표 전체."""
        with self.engine.connect() as conn:
            q = sqlalchemy.select(label_records).order_by("label_id")
            return [tuple(r) for r in conn.execute(q)]

    def units(self, sid: str) -> int:
        """AI 판정 단위 수."""
        with self.engine.connect() as conn:
            return len(list_ai_units(conn, sid))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    """빈 DB·버킷. CLI 엔진은 시간대 보존, 서버 주소 환경 변수는 비운다."""
    for name in ("DLP_VLM_URLS", "DLP_VLM_URL", "DLP_VLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(aireview_run_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(aireview_cmds, "sa", _AwareSqlalchemy())
    return World(tmp_path)


def _summaries(out: str) -> list[dict[str, Any]]:
    """표준 출력의 실행 요약 JSON 줄."""
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def _closed_port() -> int:
    """아무도 듣지 않는 127.0.0.1 포트."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_run_without_url_or_outside_host_writes_nothing(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """주소 없음 2 (DLP_VLM_URL은 쓰지 않는다), 사내 대역 밖 5, 세션 없이 실행 5. 모두 아무것도
    쓰지 않는다."""
    world.add("s1")
    monkeypatch.setenv("DLP_VLM_URL", "http://127.0.0.1:1")
    assert main(["aireview", "run", "s1", *world.args]) == 2
    assert "[미연동]" in capsys.readouterr().out
    assert main(["aireview", "run", "s1", "--vlm-url", "http://8.8.8.8:8000", *world.args]) == 5
    assert "사내 대역 밖" in capsys.readouterr().out
    assert main(["aireview", "run", *world.args]) == 5
    assert world.units("s1") == 0


def test_run_end_to_end_and_idempotent(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """가짜 서버로 판정: 계획 단위가 모두 AI 표에 들어가고 라벨 이력은 그대로다. 같은 명령을 다시
    돌리면 서버 채팅 호출 0번, 할 일 없음 (종료 0)."""
    world.add("s1")
    before = world.label_rows()
    with FakeOpenAIServer(MODEL) as server:
        assert (
            main(["aireview", "run", "s1", "--vlm-url", f"spark-01={server.url}", *world.args]) == 0
        )
        [summary] = _summaries(capsys.readouterr().out)
        assert summary["status"] == "completed" and summary["written"] == summary["planned"] > 0
        assert len(server.chat_calls) == summary["planned"]
        assert (
            main(["aireview", "run", "s1", "--vlm-url", f"spark-01={server.url}", *world.args]) == 0
        )
        assert _summaries(capsys.readouterr().out)[0]["status"] == "nothing_to_do"
        assert len(server.chat_calls) == summary["planned"]
    assert world.units("s1") == summary["planned"]
    assert world.label_rows() == before
    with world.engine.connect() as conn:
        [run] = list_ai_runs(conn, "s1")
    assert run.backend.endpoints == ("spark-01",) and run.requested_by
    assert "http" not in run.model_dump_json()  # 엔드포인트는 이름만 (URL 아님)


def test_run_exit_codes_for_server_problems(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """서버에 정책 모델이 없으면 5, 모든 엔드포인트가 응답하지 않으면 3. 둘 다 아무것도 쓰지 않는다
    (시작 전 확인)."""
    world.add("s1")
    with FakeOpenAIServer("other/model") as server:
        assert main(["aireview", "run", "s1", "--vlm-url", server.url, *world.args]) == 5
        assert server.chat_calls == []
    port = _closed_port()
    assert (
        main(["aireview", "run", "s1", "--vlm-url", f"http://127.0.0.1:{port}", *world.args]) == 3
    )
    assert "응답하는 엔드포인트가 없습니다" in capsys.readouterr().out
    assert world.units("s1") == 0
    with world.engine.connect() as conn:
        assert list_ai_runs(conn, "s1") == []


def test_session_preconditions_exit_6_and_pending_skips_them(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """승인 전·사용 중지·승인 보류·렌더 기록 없음은 종료 6 (아무것도 쓰지 않음). --pending은 그런
    세션을 고르지 않거나 목록으로 알리고 0으로 끝낸다."""
    world.add("unapproved", privacy=PrivacyState.AUTO_BLURRED)
    world.add("withdrawn", lifecycle=LifecycleState.WITHDRAWN)
    world.add("held")
    world.add("unrendered", publish=False)
    with world.engine.begin() as conn:
        record_privacy_hold(
            conn, "held", "hold", reason="노출 신고", actor="ops", at=datetime.now(UTC)
        )
    with FakeOpenAIServer(MODEL) as server:
        url = ["--vlm-url", server.url]
        for sid in ("unapproved", "withdrawn", "held", "unrendered"):
            assert main(["aireview", "run", sid, *url, *world.args]) == 6, sid
        assert main(["aireview", "run", "nope", *url, *world.args]) == 6
        out = capsys.readouterr().out
        assert "사용 중지" in out and "승인 보류" in out and "승인 전" in out and "렌더" in out
        assert main(["aireview", "run", "--pending", *url, *world.args]) == 0
        assert "[블러본 게이트] unrendered" in capsys.readouterr().out
        assert server.chat_calls == []


def test_glove_sensor_reads_raw_only_for_aireview_run(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """장갑 Parquet은 감사 원본 저장소로 용도 `aireview.run`만 읽고, 접촉 해시는 프리라벨과 같은
    함수·입력으로 직접 계산한 값과 같다. Parquet이 없으면 6, --no-sensor면 센서 없이 진행한다."""
    world.add("g1", glove=True)
    world.add("g2", glove=True, upload_glove=False)
    with FakeOpenAIServer(MODEL) as server:
        assert main(["aireview", "run", "g1", "--vlm-url", server.url, *world.args]) == 0
        assert main(["aireview", "run", "g2", "--vlm-url", server.url, *world.args]) == 6
        assert "장갑 Parquet이 없습니다" in capsys.readouterr().out
        assert (
            main(["aireview", "run", "g2", "--no-sensor", "--vlm-url", server.url, *world.args])
            == 0
        )
    path = world.store / BUCKETS.raw / "sessions/g1/glove_right.parquet"
    series = glove_series(path, ["pressure"])
    glove = load_prelabel_policy(repo_root()).contact.glove
    found = glove_contact_intervals(np.asarray(series.t_ms, dtype=float), series.values, glove)
    want = SensorContext.of({Hand.RIGHT: [(c.start_ms, c.end_ms) for c in found]})
    with world.engine.connect() as conn:
        [run1] = list_ai_runs(conn, "g1")
        [run2] = list_ai_runs(conn, "g2")
        units = list_ai_units(conn, "g1")
        access = [dict(r) for r in conn.execute(sqlalchemy.select(raw_access_log)).mappings()]
    assert run1.sensor_sha256 == want.sha256 and run2.sensor_sha256 == "absent"
    hand_units = [u for u in units if u.sensor_sha256 != "absent"]
    assert hand_units and all(u.sensor_sha256 == want.sha256 for u in hand_units)
    assert access and {(a["purpose"], a["action"], a["key"]) for a in access} == {
        ("aireview.run", "read", "sessions/g1/glove_right.parquet")
    }
    labels = {x.label_id: x for x in aireview_scene(0, session_id="g1", n_units=2).labels}
    assert all(
        any(isinstance(labels[i].payload, HandStatePayload) for i in u.target_label_ids)
        for u in hand_units
    )


def test_pending_selects_quiet_uncovered_sessions(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """--pending: prelabeled·승인·보류 아님·조용함·판정 없는 대상이 남은 세션만. 판정한 뒤에는
    아무것도 고르지 않는다. --limit은 고르는 수를 줄인다."""
    world.add("p1")
    world.add("p2")
    world.add("raw", lifecycle=LifecycleState.RAW_INGESTED)
    world.add("busy")
    scene = aireview_scene(0, session_id="busy", n_units=2)
    fresh = scene.labels[0].model_copy(
        update={"label_id": "busy-fresh", "created_at": datetime.now(UTC)}
    )
    with world.engine.begin() as conn:
        insert_labels(conn, [fresh])  # 방금 라벨을 쓴 세션 (조용한 시간 전)
    with FakeOpenAIServer(MODEL) as server:
        url = ["--vlm-url", server.url]
        assert main(["aireview", "run", "--pending", "--limit", "1", *url, *world.args]) == 0
        out = capsys.readouterr().out
        assert "판정할 세션 2개 중 1개: p1" in out
        assert world.units("p1") > 0 and world.units("p2") == 0
        assert main(["aireview", "run", "--pending", *url, *world.args]) == 0
        assert "판정할 세션 1개 중 1개: p2" in capsys.readouterr().out
        assert main(["aireview", "status", "--pending", "--url", world.url]) == 0
        assert "볼 세션이 없습니다" in capsys.readouterr().out
    assert world.units("raw") == world.units("busy") == 0


def test_status_show_and_waive(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """판정 뒤 status는 판정률 100%와 마지막 이벤트를, show는 단위 감사 기록과 라벨의 유효한 판정을
    보여 준다. waive는 건너뛰기 이벤트를 남긴다 (사유·실행자)."""
    world.add("s1")
    with FakeOpenAIServer(MODEL) as server:
        assert (
            main(["aireview", "run", "s1", "--vlm-url", f"spark-01={server.url}", *world.args]) == 0
        )
    capsys.readouterr()
    assert main(["aireview", "status", "s1", "--json", "--url", world.url]) == 0
    [row] = json.loads(capsys.readouterr().out)
    assert row["coverage"] == 1.0 and row["last_event"] == "completed" and row["units"] > 0
    assert row["verdicts"]["ok"] == row["targets"] and not row["presumed_interrupted"]
    with world.engine.connect() as conn:
        unit = list_ai_units(conn, "s1")[0]
    assert main(["aireview", "show", unit.unit_id, "--url", world.url]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["unit"]["unit_id"] == unit.unit_id and shown["run"]["requested_by"]
    assert shown["unit"]["frames"] and shown["unit"]["prompt_digest"] == unit.prompt_digest
    label = unit.target_label_ids[0]
    assert main(["aireview", "show", f"s1:{label}", "--url", world.url]) == 0
    by_label = json.loads(capsys.readouterr().out)
    assert by_label["valid"]["unit_id"] == unit.unit_id and by_label["valid"]["verdict"] == "ok"
    assert main(["aireview", "show", "s1:nope", "--url", world.url]) == 0
    assert json.loads(capsys.readouterr().out)["units"] == []
    assert main(["aireview", "show", "air-" + "f" * 24, "--url", world.url]) == 1
    assert main(["aireview", "waive", "s1", "--reason", "서버 점검", "--url", world.url]) == 0
    assert main(["aireview", "waive", "nope", "--reason", "x", "--url", world.url]) == 1
    with world.engine.connect() as conn:
        last = list_ai_run_events(conn, "s1")[-1]
    assert last.event.value == "waived" and last.reason == "manual: 서버 점검" and last.actor
    capsys.readouterr()
    assert main(["aireview", "status", "s1", "--url", world.url]) == 0
    assert "건너뜀 기록" in capsys.readouterr().out


def test_dry_run_needs_no_server(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """--dry-run은 서버 주소 없이 계획 표만 내고 아무것도 쓰지 않는다."""
    world.add("s1")
    assert main(["aireview", "run", "s1", "--dry-run", *world.args]) == 0
    out = capsys.readouterr().out
    [summary] = _summaries(out)
    assert summary["status"] == "dry_run" and "s1 계획: temporal" in out
    assert world.units("s1") == 0


def test_locked_session_exit_7_and_pending_counts_zero(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """다른 실행기가 잠금을 쥐고 있으면 7 (--pending은 알리고 0). SQLite에는 잠금이 없어 잠금 함수를
    바꿔 흉내 낸다 (PostgreSQL 경합은 services 테스트)."""
    from dlp_aireview import runner

    world.add("s1")

    def held(conn: sqlalchemy.Connection, session_id: str) -> bool:
        return False

    monkeypatch.setattr(runner, "try_lock_aireview", held)
    with FakeOpenAIServer(MODEL) as server:
        assert main(["aireview", "run", "s1", "--vlm-url", server.url, *world.args]) == 7
        assert main(["aireview", "run", "--pending", "--vlm-url", server.url, *world.args]) == 0
        assert server.chat_calls == []
    assert "다른 실행기" in capsys.readouterr().out


def test_max_minutes_exit_8_and_parallel_sessions(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """--max-minutes가 지나면 8 (그 뒤 세션은 시작하지 않는다). --parallel-sessions 2는 두 세션을
    함께 끝낸다."""
    world.add("a1")
    world.add("a2")
    with FakeOpenAIServer(MODEL) as server:
        url = ["--vlm-url", server.url]
        assert (
            main(["aireview", "run", "a1", "a2", "--max-minutes", "0.000001", *url, *world.args])
            == 8
        )
        out = capsys.readouterr().out
        assert "[건너뜀] a2" in out and "max_minutes" in out
        assert (
            main(["aireview", "run", "a1", "a2", "--parallel-sessions", "2", *url, *world.args])
            == 0
        )
        done = _summaries(capsys.readouterr().out)
    assert {s["session_id"]: s["status"] for s in done} == {"a1": "completed", "a2": "completed"}
    assert load_policy(repo_root()).server.concurrency_per_endpoint >= 1


def test_bench_session_source_uses_blurred_gate(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """bench --source session:<id>: 그 세션의 승인된 블러본으로 잰다 (보고서 source, 계획 단위).
    승인 전 세션은 블러본을 받기 전에 종료 6이고 서버에 묻지 않는다."""
    world.add("b1")
    world.add("b2", privacy=PrivacyState.AUTO_BLURRED)
    tiny = [
        "--concurrency", "1", "--groups", "temporal", "--requests", "2", "--warmup", "0",
        "--ttft", "0", "--no-capacity",
    ]  # fmt: skip
    out = tmp_path / "bench.json"
    with FakeOpenAIServer(MODEL) as server:
        url = ["--vlm-url", server.url]
        assert main(["aireview", "bench", "--source", "session:b2", *url, *tiny, *world.args]) == 6
        assert server.chat_calls == []
        code = main(
            ["aireview", "bench", "--source", "session:b1", *url, *tiny, *world.args,
             "--out", str(out)]
        )  # fmt: skip
        assert code == 0
    report = json.loads(out.read_text())
    assert report["source"] == "session:b1"
    assert report["configs"][0]["planned_units"] > 0
    assert world.units("b1") == 0  # 벤치는 DB에 쓰지 않는다
