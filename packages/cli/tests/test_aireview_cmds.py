"""`dlp aireview probe|bench|estimate` CLI 종단 테스트 (WP17.3, ADR 0040 5.6~5.8).

서버는 가짜 OpenAI 호환 서버(`FakeOpenAIServer`, 127.0.0.1), DB는 시간대를 보존하는 SQLite 파일이다
(서비스 불필요). 정답 근거: 가짜 서버 설정에 따른 종료 코드, 같은 입력으로 직접 계산한
`session_load`, 손 계산한 장비 대수 공식.
"""

from __future__ import annotations

import json
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_aireview.bench import (
    ApproxLoad,
    BenchConfig,
    BenchPoint,
    BenchReport,
    DecodeRun,
    PrefillRun,
    group_settings,
)
from dlp_aireview.estimate import counting_policy, group_throughput, session_load
from dlp_aireview.policy import AiReviewPolicy, load_policy
from dlp_aireview.probe import ServerInfo
from dlp_aireview.testing import FakeOpenAIServer
from dlp_cli import aireview_cmds
from dlp_cli.main import main
from dlp_cli.review_config import load_review_policy
from dlp_fixtures.aireview import aireview_scene
from dlp_review.ops.priority import flag_unit, units_for
from dlp_schema import repo_root
from dlp_schema.aireview import AiGroup
from dlp_schema.db.repository import insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewReason
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import sqlite_db

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
# 작은 벤치 (temporal만, 동시성 1·2, 단위 3개, 320x240 합성 프레임)
TINY = [
    "--concurrency", "1,2", "--frames", "8", "--side", "448", "--groups", "temporal",
    "--requests", "3", "--warmup", "0", "--ttft", "1", "--scene-units", "2",
    "--frame-size", "320x240",
]  # fmt: skip


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """서버 주소·키 환경 변수를 비운다 (개발 셸 값이 섞이지 않게)."""
    for name in ("DLP_VLM_URLS", "DLP_VLM_URL", "DLP_VLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _closed_port() -> int:
    """아무도 듣지 않는 127.0.0.1 포트."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_probe_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """주소 없음 2 (DLP_VLM_URL은 쓰지 않는다), 통과 0, 구조화 출력 미강제 5, 서버 없음 3."""
    monkeypatch.setenv("DLP_VLM_URL", "http://127.0.0.1:1")
    assert main(["aireview", "probe"]) == 2
    assert "[미연동]" in capsys.readouterr().out
    out = tmp_path / "probe.json"
    with FakeOpenAIServer(MODEL) as server:
        assert (
            main(["aireview", "probe", "--vlm-url", f"spark-01={server.url}", "--out", str(out)])
            == 0
        )
    [report] = json.loads(out.read_text())
    assert report["kind"] == "aireview-probe" and report["server"]["endpoint"] == "spark-01"
    with FakeOpenAIServer(MODEL, enforce_schema=False) as server:
        monkeypatch.setenv("DLP_VLM_URLS", f"spark-02={server.url}")
        assert main(["aireview", "probe"]) == 5
    assert "schema_enforcement: prompt" in capsys.readouterr().out
    monkeypatch.delenv("DLP_VLM_URLS")
    assert main(["aireview", "probe", "--vlm-url", f"http://127.0.0.1:{_closed_port()}"]) == 3
    assert main(["aireview", "probe", "--vlm-url", "ftp://x"]) == 5
    # 설정 불일치 뒤 서버 장애: 가장 심각한 5 (장애 3이 아니다)
    with FakeOpenAIServer(MODEL, enforce_schema=False) as server:
        server.fail(503, times=10, after=3)
        assert main(["aireview", "probe", "--vlm-url", server.url]) == 5
    printed = capsys.readouterr().out
    assert "[fail] schema_enforced" in printed and "[fail] unavailable" in printed
    # 점검 요청 4xx는 traceback이 아니라 실패 점검 (종료 5)
    with FakeOpenAIServer(MODEL) as server:
        server.fail(400, times=1, after=2)
        assert main(["aireview", "probe", "--vlm-url", server.url]) == 5


def test_bench_writes_report_with_decode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """가짜 서버로 벤치 보고서를 쓴다 (무릎 지점, 서버 정보, 디코드 fps). DB는 쓰지 않는다."""
    out = tmp_path / "bench.json"
    with FakeOpenAIServer(MODEL, latency_s=0.01) as server:
        code = main(
            [
                "aireview",
                "bench",
                "--vlm-url",
                f"t1={server.url}",
                *TINY,
                "--decode",
                "--decode-seconds",
                "1",
                "--out",
                str(out),
            ]
        )
    assert code == 0
    report = BenchReport.model_validate_json(out.read_text())
    assert report.device == "t1" and report.server.models == (MODEL,)
    [config] = report.configs  # 격자 8·448 = 지금 정책 값 (정책 값은 늘 잰다)
    assert config.group is AiGroup.TEMPORAL and config.knee is not None
    assert [p.concurrency for p in config.points] == [1, 2]
    assert config.prefill is not None and config.decode is not None and config.decode.honored
    assert config.decode.forced_tokens == 256 and config.approx is not None
    assert report.grid["capacity"] is True
    assert report.decode is not None and report.decode.fps > 0 and report.decode.width == 320
    printed = capsys.readouterr().out
    assert "단위/s" in printed and "용량 프리필 근사" in printed and str(out) in printed


def test_bench_default_grid_is_policy_values(tmp_path: Path) -> None:
    """--frames·--side를 주지 않으면 묶음마다 지금 정책 값만 잰다 (코드에 박은 기본 격자가 없다)."""
    out = tmp_path / "bench.json"
    tiny = ["--concurrency", "1", "--requests", "1", "--warmup", "0", "--ttft", "0",
            "--scene-units", "2", "--frame-size", "320x240", "--no-capacity"]  # fmt: skip
    with FakeOpenAIServer(MODEL) as server:
        assert main(["aireview", "bench", "--vlm-url", server.url, *tiny, "--out", str(out)]) == 0
    report = BenchReport.model_validate_json(out.read_text())
    policy = load_policy(repo_root())
    assert [(c.group, c.frames, c.side) for c in report.configs] == [
        (g, *group_settings(policy, g)) for g in AiGroup
    ]
    assert all(c.prefill is None and c.decode is None for c in report.configs)


def test_bench_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """주소 없음 2, 엔드포인트 여럿인데 고르지 않음 5, 서버에 정책 모델 없음 5, 서버가 멈춤 3."""
    assert main(["aireview", "bench", *TINY]) == 2
    out = str(tmp_path / "b.json")
    with FakeOpenAIServer("other/model") as server:
        many = ["--vlm-url", f"a={server.url}", "--vlm-url", f"b={server.url}"]
        assert main(["aireview", "bench", *many, *TINY, "--out", out]) == 5
        assert main(["aireview", "bench", *many, "--endpoint", "a", *TINY, "--out", out]) == 5
        assert server.chat_calls == []
    with FakeOpenAIServer(MODEL) as server:
        server.fail(503, times=1000)
        assert main(["aireview", "bench", "--vlm-url", server.url, *TINY, "--out", out]) == 3
    assert json.loads(Path(out).read_text())["server_down"] is True
    port = _closed_port()
    assert main(["aireview", "bench", "--vlm-url", f"http://127.0.0.1:{port}", *TINY]) == 3


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def db_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """합성 장면 세 개: e1(prelabeled), e2(raw_ingested), e3(사용 중지). CLI 엔진도 시간대 보존."""
    url = f"sqlite:///{tmp_path / 'est.db'}"
    engine = sqlite_db(url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for sid, state in (
            ("e1", LifecycleState.PRELABELED),
            ("e2", LifecycleState.RAW_INGESTED),
            ("e3", LifecycleState.WITHDRAWN),
        ):
            scene = aireview_scene(0, session_id=sid, n_units=3)
            session = scene.session.model_copy(
                update={"lifecycle_state": state, "privacy_state": PrivacyState.APPROVED}
            )
            insert_session(conn, session)
            insert_labels(conn, scene.labels)
    engine.dispose()
    monkeypatch.setattr(aireview_cmds, "sa", _AwareSqlalchemy())
    return url


def _bench_file(tmp_path: Path) -> Path:
    """temporal·temporal_long·spatial 무릎 지점이 1 단위/s인 보고서 (지금 정책 설정 값)."""
    policy = load_policy(repo_root())
    point = BenchPoint(
        concurrency=8, units=10, ok=10, unavailable=0, wall_s=10.0, units_per_s=1.0,
        latency_p95_ms=500,
    )  # fmt: skip
    settings = {
        AiGroup.TEMPORAL: (policy.temporal.frames, policy.temporal.max_side_px),
        AiGroup.TEMPORAL_LONG: (policy.temporal.long_frames, policy.temporal.long_max_side_px),
        AiGroup.SPATIAL: (policy.spatial.frames, policy.spatial.max_side_px),
    }
    report = BenchReport(
        created_at=datetime(2026, 10, 12, tzinfo=UTC), device="spark-01", source="synthetic",
        server=ServerInfo(endpoint="spark-01", served_model=MODEL), policy={"model": MODEL},
        grid={},
        configs=tuple(
            BenchConfig(group=g, frames=f, side=s, planned_units=1, points=(point,), knee=point)
            for g, (f, s) in settings.items()
        ),
    )  # fmt: skip
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json())
    return path


def test_estimate_counts_sessions_and_devices(
    db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--prelabeled는 e1만, 작업량은 같은 입력의 session_load와 같고, 대수는 공식대로다."""
    bench = _bench_file(tmp_path)
    args = ["aireview", "estimate", "--prelabeled", "--url", db_url, "--include-actions",
            "--bench", str(bench), "--actions-req-s", "2", "--json"]  # fmt: skip
    assert main(args) == 0
    data = json.loads(capsys.readouterr().out)
    workload, [cap] = data["workload"], data["capacity"]
    assert [s["session_id"] for s in workload["sessions"]] == ["e1"]
    policy = load_policy(repo_root())
    scene = aireview_scene(0, session_id="e1", n_units=3)
    ontology = load_ontology(repo_root() / "config/ontology/v1")
    expected = session_load(scene.session, list(scene.labels), policy, ontology)
    for group, load in expected.groups.items():
        assert workload["groups"][group.value]["units"] == load.units
    hours = expected.video_hours
    units = sum(g.units for g in expected.groups.values())
    actions = workload["actions_requests"]
    assert actions and actions > 0
    d = units / hours + actions / hours / 2
    assert cap["device_s_per_hour"] == pytest.approx(d)
    assert cap["devices"] == -(-164 * 1.2 * d // (86_400 * 0.7))
    assert cap["missing"] == []


def test_estimate_recounts_workload_with_report_counting_values(
    db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """보고서의 셈법 값(align_px 28)이 지금 정책(32)과 다르면 그 보고서용 작업량을 보고서 셈법으로
    다시 센다 (벤치·작업량의 근사 토큰이 같은 셈법이어야 비율이 맞다). 용량 측정이 든 보고서다."""
    policy = load_policy(repo_root())
    point = BenchPoint(
        concurrency=8, units=10, ok=10, unavailable=0, wall_s=10.0, units_per_s=1.0,
        latency_p95_ms=500, prompt_tokens=42_000, cached_tokens=30_000, completion_tokens=300,
        cache_hit_ratio=30_000 / 42_000,
    )  # fmt: skip
    approx = ApproxLoad(image=1000.0, text=200.0, static=3000.0)
    prefill = PrefillRun(concurrency=8, requests=10, ok=10, unavailable=0, wall_s=10.0,
                         fresh_tokens=12_000.0, static_tokens=30_000.0)  # fmt: skip
    decode = DecodeRun(concurrency=8, requests=16, forced_tokens=256, ok=16, wall_s=10.0,
                       completion_tokens=4096, decode_tok_s=409.6, honored=True)  # fmt: skip
    settings = {g: group_settings(policy, g) for g in AiGroup}
    report = BenchReport(
        created_at=datetime(2026, 10, 12, tzinfo=UTC), device="spark-01", source="synthetic",
        server=ServerInfo(endpoint="spark-01", served_model=MODEL),
        policy={"model": MODEL, "align_px": 28, "text_bytes_per_token": 3}, grid={},
        configs=tuple(
            BenchConfig(group=g, frames=f, side=side, planned_units=1, approx=approx,
                        points=(point,), knee=point, prefill=prefill, decode=decode)
            for g, (f, side) in settings.items()
        ),
    )  # fmt: skip
    path = tmp_path / "align28.json"
    path.write_text(report.model_dump_json())
    assert (
        main(["aireview", "estimate", "e1", "--url", db_url, "--bench", str(path), "--json"]) == 0
    )
    [cap] = json.loads(capsys.readouterr().out)["capacity"]
    counted = counting_policy(policy, report)
    assert counted.image.align_px == 28
    scene = aireview_scene(0, session_id="e1", n_units=3)
    ontology = load_ontology(repo_root() / "config/ontology/v1")
    recount = session_load(scene.session, list(scene.labels), counted, ontology)
    as_is = session_load(scene.session, list(scene.labels), policy, ontology)
    for share in cap["groups"]:
        group = AiGroup(share["group"])
        want = group_throughput(report, counted, group, recount.groups[group])
        assert want is not None and share["units_per_s"] == pytest.approx(want.units_per_s)
    t_load, t_as_is = recount.groups[AiGroup.TEMPORAL], as_is.groups[AiGroup.TEMPORAL]
    assert t_load.image_tokens != t_as_is.image_tokens  # 정렬 단위가 달라 근사 토큰이 다르다
    assert cap["warnings"] == []


def test_estimate_named_sessions_text_output(
    db_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """세션을 지정하면 그 세션만 (사용 중지 세션은 빼고 알린다), 표와 대수를 출력한다."""
    bench = _bench_file(tmp_path)
    assert main(["aireview", "estimate", "e1", "e3", "--url", db_url, "--bench", str(bench)]) == 0
    out = capsys.readouterr().out
    assert "e3: 사용 중지된 세션이라 뺍니다" in out
    assert "세션 1개" in out and "필요 대수" in out
    assert main(["aireview", "estimate", "--url", db_url]) == 0
    assert "추정할 세션이 없습니다" in capsys.readouterr().out


def test_estimate_flagged_mode_uses_review_flags(
    db_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """flagged 모드는 검수 운영 정책의 `flag_unit` 사유 구간과 겹치는 단위만 센다: 같은 입력으로
    직접 계산한 사유 구간을 넣은 `session_load`와 같고, 전량보다 적다 (사유 구간이 있어야 0이
    아니다).
    합성 장면의 박스 클래스는 다른 세션에서 사람이 확인한 적이 없어 new_object 사유가 붙는다."""
    assert main(["aireview", "estimate", "e1", "--url", db_url, "--json"]) == 0
    full = json.loads(capsys.readouterr().out)["workload"]["groups"]
    assert main(["aireview", "estimate", "e1", "--url", db_url, "--mode", "flagged", "--json"]) == 0
    picked = json.loads(capsys.readouterr().out)["workload"]["groups"]
    review_policy = load_review_policy(repo_root())
    scene = aireview_scene(0, session_id="e1", n_units=3)
    current = current_labels(list(scene.labels))
    spans = [
        span
        for unit in units_for(scene.session, review_policy)
        for span in flag_unit(
            unit.select(current), review_policy, known_classes=set(), glove_session=False
        )
    ]
    assert spans and {x.reason for x in spans} == {ReviewReason.NEW_OBJECT}
    policy = load_policy(repo_root())
    flagged_policy = AiReviewPolicy.model_validate(
        policy.model_dump() | {"selection": policy.selection.model_dump() | {"mode": "flagged"}}
    )
    ontology = load_ontology(repo_root() / "config/ontology/v1")
    expected = session_load(scene.session, current, flagged_policy, ontology, flagged=spans)
    for group in AiGroup:
        want = expected.groups[group].units if group in expected.groups else 0
        assert picked[group.value]["units"] == want
    units = sum(g["units"] for g in picked.values())
    assert 0 < units < sum(g["units"] for g in full.values())
