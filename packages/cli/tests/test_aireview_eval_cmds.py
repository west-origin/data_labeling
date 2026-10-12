"""`dlp aireview eval` CLI 종단 (WP17.8 완료 기준: eval이 DB에 쓰지 않음, 게이트는 경고만).

시간대 보존 SQLite 파일 DB(쓰기 감시를 붙인다)와 로컬 라벨링 버킷의 합성 블러본, 가짜 OpenAI 호환
서버(`FakeOpenAIServer`, 판정 요청에 모두 ok로 답한다)로 돈다.

정답 근거: 가짜 서버가 모두 ok로 답하므로 교란한 라벨을 하나도 잡지 못한다 → 탐지 재현율 0,
틀린 라벨을 ok라 한 비율 1, ok 놓침률 P(틀림 | AI ok) = 틀린 라벨 수 / 대상 수(교란 유병률),
오지적 0. 게이트는 실패지만 `gate.block: false`(결정)라 종료 0이고 경고만 낸다.
block을 true로 바꾸면 종료 1이다. 평가 내내 DB에 INSERT·UPDATE·DELETE가 한 번도 없다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import event

from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.policy import load_policy as load_aireview_policy
from dlp_aireview.testing import FakeOpenAIServer
from dlp_cli import aireview_eval_cmds
from dlp_cli.main import main
from dlp_fixtures.aireview import aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import RenderMeta, expected_render_hash, write_render_meta
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import (
    get_golden_set,
    insert_golden_set,
    insert_labels,
    insert_session,
    register_ontology,
)
from dlp_schema.labels import Provenance, Source
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME, sqlite_db

BUCKETS = load_config(repo_root() / "config/defaults.yaml").buckets
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


class _Watched:
    """CLI 모듈의 `sa`를 대신한다: 시간대 보존 SQLite 엔진에 쓰기 문장 감시를 붙인다."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        engine = sqlite_db(str(url), create=False)

        def watch(conn: Any, cursor: Any, statement: str, *rest: Any) -> None:
            if not statement.lstrip().upper().startswith(("SELECT", "PRAGMA", "WITH")):
                self.writes.append(statement)

        event.listen(engine, "before_cursor_execute", watch)
        return engine


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], _Watched]:
    """골든셋 g1(이전 형식, 정답 고정 없음)·g2(정답 고정)(세션 s1: 승인·렌더, 사람이 만든 정답
    라벨) + CLI 공통 인자."""
    for name in ("DLP_VLM_URLS", "DLP_VLM_URL", "DLP_VLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(url)
    store = tmp_path / "store"
    labeling = LocalStore(store, BUCKETS.labeling)
    scene = aireview_scene(0, session_id="s1", n_units=3)
    truth = [
        x.model_copy(update={"provenance": Provenance(source=Source.HUMAN), "confidence": None})
        for x in scene.labels
    ]
    session = scene.session.model_copy(
        update={
            "lifecycle_state": LifecycleState.PRELABELED,
            "privacy_state": PrivacyState.APPROVED,
        }
    )
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, session)
        insert_labels(conn, truth)
        insert_golden_set(conn, GoldenSet(
            version="g1", domain=Domain.CLEANING, session_ids=("s1",), created_at=FIXED_TIME,
        ))  # fmt: skip
        # 같은 세션, 정답 ID를 고정한 버전 (ADR 0041)
        insert_golden_set(conn, GoldenSet(
            version="g2", domain=Domain.CLEANING, session_ids=("s1",), created_at=FIXED_TIME,
            truth_label_ids=tuple(sorted(x.label_id for x in truth)),
        ))  # fmt: skip
    privacy = load_privacy_policy(repo_root())
    for stream in ("bodycam", "tp1"):
        video = tmp_path / f"s1-{stream}.mp4"
        frames = ((t, scene_frame(scene, stream, t)) for t in scene.frame_times)
        write_video(video, frames, width=scene.width, height=scene.height)
        sha = sha256_file(video)
        labeling.put_file(blurred_key("s1", stream), video, sha)
        with engine.connect() as conn:
            digest = expected_render_hash(conn, "s1", stream, privacy)
        write_render_meta(labeling, "s1", stream, RenderMeta(digest, sha), tmp_path)
    engine.dispose()
    watched = _Watched()
    monkeypatch.setattr(aireview_eval_cmds, "sa", watched)
    return ["--url", url, "--store", f"local:{store}"], watched


def test_eval_writes_nothing_and_gate_only_warns(
    world: tuple[list[str], _Watched], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """모두 ok로 답하는 서버: 탐지 0·틀린 라벨을 ok라 함 1 → 게이트 실패지만 경고만(종료 0). DB
    쓰기 0건."""
    args, watched = world
    out = tmp_path / "report.json"
    with FakeOpenAIServer(MODEL) as server:
        code = main([
            "aireview", "eval", "--golden", "g1", "--gate", "--seed", "3", "--out", str(out),
            "--inject", "verb_swap=0.5,class_swap=0.5", "--vlm-url", f"spark-01={server.url}",
            *args,
        ])  # fmt: skip
        calls = len(server.chat_calls)
    text = capsys.readouterr().out
    assert code == 0, text
    assert "[게이트 실패 — 경고만 (gate.block: false)]" in text
    assert watched.writes == [] and calls > 0
    report = json.loads(out.read_text("utf-8"))
    assert report["wrong_recall"] == 0.0 and report["wrong_called_ok_rate"] == 1.0
    wrong_ok = sum(v["ok"] for k, v in report["by_error"].items() if k != "clean")
    all_ok = sum(v["ok"] for v in report["by_error"].values())
    assert all_ok == report["targets"] and 0 < wrong_ok < all_ok
    assert report["ok_miss_rate"] == pytest.approx(wrong_ok / all_ok)
    assert report["false_flag_rate"] == 0.0 and report["units"] == calls
    assert report["gate"]["passed"] is False and report["gate"]["block"] is False
    assert set(report["by_error"]) >= {"clean", "verb_swap"}


def test_eval_groups_count_only_targets_of_chosen_groups(
    world: tuple[list[str], _Watched], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--groups temporal`이면 공간 대상(박스)은 분모에 들지 않고, 미판정·미계획이 0이다.

    정답 근거: 예전에는 다른 묶음 대상이 판정되지 않은 채 분모에 남아 재현율이 떨어지고 오지적률
    분모가 커졌다 (판정 버전이 같아 보고서로 구별할 수 없다).
    """
    args, _ = world
    reports: dict[str, Any] = {}
    with FakeOpenAIServer(MODEL) as server:
        for name, extra in (("all", []), ("temporal", ["--groups", "temporal,temporal_long"])):
            out = tmp_path / f"{name}.json"
            code = main([
                "aireview", "eval", "--golden", "g1", "--seed", "3", "--out", str(out),
                "--inject", "verb_swap=0.5,class_swap=0.5", "--vlm-url", f"spark-01={server.url}",
                *extra, *args,
            ])  # fmt: skip
            assert code == 0, capsys.readouterr().out
            reports[name] = json.loads(out.read_text("utf-8"))
    full, narrowed = reports["all"], reports["temporal"]
    assert "box_track" in full["by_kind"] and "box_track" not in narrowed["by_kind"]
    assert "class_swap" not in narrowed["by_error"] and 0 < narrowed["targets"] < full["targets"]
    for report in (full, narrowed):
        assert all(v["unjudged"] == 0 and v["unplanned"] == 0 for v in report["by_error"].values())


def test_eval_report_records_truth_basis_and_warns_on_legacy_golden(
    world: tuple[list[str], _Watched], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """보고서에 정답 근거(`truth_source`, `golden_truth`)를 남기고, 이전 형식 골든셋은 경고한다.

    정답 근거(ADR 0041): 정답을 고정하지 않은 g1은 현재 라벨로 평가하므로 `dlp eval golden`처럼
    경고하고 다시 고정하는 명령을 안내한다. 고정한 g2는 경고가 없고 고정 근거의 해시가 같은
    정답 ID 집합의 sha256이다. history 방식은 사람 검수 사슬의 지금 상태가 정답이라고 알린다
    (이 골든 세션에는 사람이 검수한 모델 레코드가 없어 판정할 대상이 없다 → 6).
    """
    args, _ = world
    reports: dict[str, Any] = {}
    texts: dict[str, str] = {}
    with FakeOpenAIServer(MODEL) as server:
        for name, extra in (
            ("g1", ["--golden", "g1"]),
            ("g2", ["--golden", "g2"]),
            ("history", ["--golden", "g2", "--mode", "history"]),
        ):
            out = tmp_path / f"{name}.json"
            code = main([
                "aireview", "eval", *extra, "--out", str(out), "--vlm-url",
                f"spark-01={server.url}", *args,
            ])  # fmt: skip
            texts[name] = capsys.readouterr().out
            assert code == (6 if name == "history" else 0), texts[name]
            if out.exists():
                reports[name] = json.loads(out.read_text("utf-8"))
    legacy, frozen = reports["g1"], reports["g2"]
    assert legacy["truth_source"] == "current_labels"
    assert legacy["golden_truth"]["truth_frozen"] is False
    assert "이전 형식" in texts["g1"] and "dlp dataset golden --from g1" in texts["g1"]
    assert frozen["truth_source"] == "golden_frozen" and "이전 형식" not in texts["g2"]
    truth = frozen["golden_truth"]
    assert truth["truth_frozen"] is True and truth["sessions"] == ["s1"]
    assert truth["truth_label_ids_sha256"] == legacy["golden_truth"]["truth_label_ids_sha256"]
    assert (
        truth["truth_label_ids_sha256"]
        == hashlib.sha256("\n".join(sorted(_truth_ids(world))).encode()).hexdigest()
    )
    assert "지금 상태를 정답" in texts["history"] and "history" not in reports


def _truth_ids(world: tuple[list[str], _Watched]) -> list[str]:
    """골든셋 g2에 고정한 정답 라벨 ID (DB에서 읽는다)."""
    url = world[0][1]
    engine = sqlite_db(url, create=False)
    try:
        with engine.connect() as conn:
            return list(get_golden_set(conn, "g2").truth_label_ids or ())
    finally:
        engine.dispose()


def test_eval_gate_blocks_only_when_policy_says_so(
    world: tuple[list[str], _Watched],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`gate.block: true`면 게이트 실패가 종료 1이다 (파일럿 뒤 결정을 바꿀 때의 동작)."""
    args, _ = world
    original = load_aireview_policy

    def blocking(root: Path) -> AiReviewPolicy:
        data = original(root).model_dump()
        data["gate"]["block"] = True
        return AiReviewPolicy.model_validate(data)

    monkeypatch.setattr(aireview_eval_cmds, "load_policy", blocking)
    with FakeOpenAIServer(MODEL) as server:
        code = main([
            "aireview", "eval", "--golden", "g1", "--gate", "--out", str(tmp_path / "r.json"),
            "--vlm-url", f"spark-01={server.url}", *args,
        ])  # fmt: skip
    assert code == 1 and "[게이트 실패 — 차단]" in capsys.readouterr().out


def test_eval_without_server_or_golden(
    world: tuple[list[str], _Watched], capsys: pytest.CaptureFixture[str]
) -> None:
    """주소 없음 2 (아무것도 하지 않음), 없는 골든셋 6, 모르는 교란 종류는 인자 오류."""
    args, watched = world
    assert main(["aireview", "eval", "--golden", "g1", *args]) == 2
    with FakeOpenAIServer(MODEL) as server:
        url = f"spark-01={server.url}"
        assert main(["aireview", "eval", "--golden", "nope", "--vlm-url", url, *args]) == 6
    assert "골든셋 nope이 없습니다" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["aireview", "eval", "--golden", "g1", "--inject", "nope=0.1", *args])
    assert watched.writes == []
