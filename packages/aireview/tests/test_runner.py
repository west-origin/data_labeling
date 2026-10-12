"""판정 실행기 테스트 (`runner.run_session`, `MemorySink`, WP17.5, ADR 0040 5.1·5.3·9장).

합성 장면(`conftest.make_scene`, 단색 합성 프레임)을 `OracleJudge`(정답 라벨로 답하는 stub)나 가짜
OpenAI 호환 서버(`FakeOpenAIServer`, 127.0.0.1) + `EndpointPool`로 판정한다. DB 없이 `MemorySink`에
쓴다 (DB 경로는 `test_runner_db.py`).

정답 근거: 같은 입력으로 직접 계산한 계획(`runner.planned_units`), 동시성 1의 결과, 가짜 서버가 받은
요청 수와 단위 키 헤더, 정책 값(차단기·백오프·중단 시간).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_aireview.backend import (
    BackendInfo,
    EndpointPool,
    JudgeBackend,
    JudgeReply,
    JudgeRequest,
    OpenAICompatibleJudge,
    PoolMember,
)
from dlp_aireview.media import BlurredFrame
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.perturb import perturb
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.runner import (
    MemorySink,
    RunInput,
    RunOptions,
    RunSummary,
    plan_digest,
    planned_units,
    run_session,
)
from dlp_aireview.testing import FakeOpenAIServer
from dlp_models.vlm import VlmUnavailableError
from dlp_privacy.errors import RenderNotCurrentError
from dlp_schema.aireview import AiGroup, AiReviewUnit, AiRunEventKind
from dlp_schema.ontology import Ontology

from .conftest import RENDER, SIZE, Scene, with_policy

TEMPORAL = frozenset({AiGroup.TEMPORAL})


class Frames:
    """합성 프레임 공급 (요청받은 번호를 기록한다)."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.requests: list[tuple[str, tuple[int, ...]]] = []
        self._lock = threading.Lock()

    def __call__(self, stream: str, idx: Sequence[int]) -> Iterator[BlurredFrame]:
        with self._lock:
            self.requests.append((stream, tuple(idx)))
        w, h = SIZE
        for i in idx:
            t = self.scene.frame_times[i]
            yield BlurredFrame(stream, i, t, np.full((h, w, 3), 40 + i % 200, dtype=np.uint8))


def run_input(scene: Scene, o: Ontology, labels: Sequence[Any] | None = None) -> RunInput:
    """장면 → 실행 입력 (두 영상 스트림, 합성 프레임)."""
    return RunInput(
        session=scene.session, current=tuple(scene.labels if labels is None else labels),
        ontology=o, media=scene.media, blurred_sha256={s: None for s in scene.media},
        frames=Frames(scene),
    )  # fmt: skip


def fast(policy: AiReviewPolicy, **server: object) -> AiReviewPolicy:
    """실행 시험용 정책 (짧은 백오프·쉼·중단, 자주 기록·확인)."""
    values: dict[str, object] = {
        "backoff_s": (0.0, 0.01), "breaker_failures": 2, "breaker_cooldown_s": 0.05,
        "abort_after_s": 0.4, "commit_every": 3, "commit_interval_s": 0.05,
        "render_check_interval_s": 0.05, "progress_interval_s": 0.05,
        "concurrency_per_endpoint": 4,
    }  # fmt: skip
    return with_policy(policy, server=values | server)


def judge_pool(policy: AiReviewPolicy, *servers: FakeOpenAIServer) -> EndpointPool:
    """가짜 서버마다 엔드포인트 하나 (ping은 /v1/models)."""
    members: list[PoolMember] = []
    for k, server in enumerate(servers):
        judge = OpenAICompatibleJudge(f"fake-{k}", server.url, policy)
        members.append(PoolMember(f"fake-{k}", judge, judge.ping))
    return EndpointPool(members, policy.server)


def comparable(units: dict[str, AiReviewUnit]) -> dict[str, Any]:
    """실행마다 달라지는 값(실행 ID·시각·지연·엔드포인트)을 뺀 단위 기록."""
    skip = {"run_id", "created_at", "latency_ms", "endpoint"}
    return {k: u.model_dump(mode="json", exclude=skip) for k, u in units.items()}


def events(sink: MemorySink) -> list[tuple[str, str | None]]:
    """(이벤트, 사유) 순서."""
    return [(e.event.value, e.reason) for e in sink.events]


def test_concurrency_does_not_change_results(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """작업자 8개의 결과가 작업자 1개와 같다 (단위 키·판정·프롬프트 다이제스트·프레임). 넣은 오류는
    wrong으로 잡히고 계획 단위가 모두 기록되며 completed다. 프레임은 스트림마다 순차로 한 번만
    요청한다."""
    p = perturb(scene.labels, {"verb_swap": 0.5, "class_swap": 0.5}, seed=1, o=ontology)
    results: list[dict[str, Any]] = []
    for workers in (1, 8):
        sink = MemorySink()
        inp = run_input(scene, ontology, p.labels)
        summary = run_session(
            inp, OracleJudge(scene.labels), sink, fast(policy),
            options=RunOptions(workers=workers),
        )  # fmt: skip
        plan = planned_units(inp, policy, RunOptions())
        assert summary.status == "completed" and summary.exit_code == 0
        assert set(sink.units) == {u.unit_id for u in plan}
        assert summary.written == summary.planned == summary.todo == len(plan)
        assert events(sink) == [("started", None), ("completed", None)]
        assert sink.runs[0].planned_units == len(plan)
        assert sink.runs[0].plan_digest == plan_digest(plan)
        frames = inp.frames
        assert isinstance(frames, Frames)
        assert sorted(s for s, _ in frames.requests) == ["bodycam", "tp1"]
        for stream, idx in frames.requests:
            want = sorted({i for u in plan if u.stream_id == stream for i in u.frame_idx})
            assert list(idx) == want
        results.append(comparable(sink.units))
    assert results[0] == results[1]
    wrong = {
        v["label_id"] for u in results[0].values() for v in u["verdicts"] if v["verdict"] == "wrong"
    }
    planned_targets = {i for u in results[0].values() for i in u["target_label_ids"]}
    assert wrong == set(p.injected) & planned_targets and wrong


def test_same_version_rerun_calls_nothing_and_writes_nothing(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """같은 버전·입력으로 다시 실행하면 서버 호출 0번, 새 기록 0건 (nothing_to_do)."""
    sink, judge = MemorySink(), OracleJudge(scene.labels)
    first = run_session(run_input(scene, ontology), judge, sink, fast(policy))
    calls, runs, n_events = judge.calls, len(sink.runs), len(sink.events)
    again = run_session(run_input(scene, ontology), judge, sink, fast(policy))
    assert first.status == "completed" and again.status == "nothing_to_do"
    assert again.exit_code == 0 and again.todo == 0
    assert judge.calls == calls and len(sink.runs) == runs and len(sink.events) == n_events


def test_hashed_policy_change_judges_again_unhashed_does_not(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """해시 절(response)을 바꾸면 새 버전이라 모든 단위를 다시 판정하고 이전 행은 남는다. 해시 밖
    절(selection 감사 비율, server)을 바꾸면 같은 버전이라 할 일이 없다."""
    sink = MemorySink()
    base = fast(policy)
    run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, base)
    n = len(sink.units)
    same = with_policy(base, selection={"audit_ratio": 0.5}, server={"commit_every": 7})
    assert same.verdict_version == base.verdict_version
    again = run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, same)
    assert again.status == "nothing_to_do"
    changed = with_policy(base, response={"reason_max_chars": 50})
    assert changed.verdict_version != base.verdict_version
    summary = run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, changed)
    assert summary.status == "completed" and summary.written == n
    assert len(sink.units) == 2 * n
    versions = {u.version for u in sink.units.values()}
    assert versions == {base.verdict_version, changed.verdict_version}


def test_context_overflow_split_is_one_row_and_rerun_calls_nothing(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """문맥 초과 두 번(프레임 절반 → 대상 절반)으로 나눈 단위도 계획 단위 한 행이고, 재실행은 서버를
    부르지 않는다 (C13)."""
    sink = MemorySink()
    judge = OracleJudge(scene.labels, faults=("context_overflow", "context_overflow"))
    opts = RunOptions(groups=TEMPORAL)
    inp = run_input(scene, ontology)
    summary = run_session(inp, judge, sink, fast(policy), options=opts)
    plan = planned_units(inp, policy, opts)
    assert summary.status == "completed" and len(sink.units) == len(plan)
    split = [u for u in sink.units.values() if u.sub_requests > 1]
    assert len(split) == 1 and split[0].sub_requests == 2 and split[0].attempts == 4
    assert all(v.answered for v in split[0].verdicts)
    calls = judge.calls
    again = run_session(run_input(scene, ontology), judge, sink, fast(policy), options=opts)
    assert again.status == "nothing_to_do" and judge.calls == calls


def test_outage_interrupts_keeps_done_units_and_resumes(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """k번째 요청 뒤 503이 이어지면: 차단기가 열리고 (ping은 되지만 시험 요청이 실패해 다시 열림)
    모두 막힌 지 abort_after_s가 지나면 interrupted:unavailable(종료 3), 완료한 k개는 남는다. 서버가
    돌아오면 남은 단위만 묻고 completed, 그다음은 호출 0번."""
    k = 2
    sink = MemorySink()
    p = fast(policy, abort_after_s=0.05)
    opts = RunOptions(workers=1)
    with FakeOpenAIServer() as server:
        server.fail(503, times=10**6, after=k)
        pool = judge_pool(p, server)
        first = run_session(run_input(scene, ontology), pool, sink, p, options=opts)
    assert (first.status, first.reason, first.exit_code) == ("interrupted", "unavailable", 3)
    assert len(sink.units) == first.written == k
    assert events(sink)[-1] == ("interrupted", "unavailable")
    assert first.detail is not None and "응답하지 않습니다" in first.detail
    done = set(sink.units)
    with FakeOpenAIServer() as server:
        second = run_session(run_input(scene, ontology), judge_pool(p, server), sink, p)
        asked = {c.unit_id for c in server.chat_calls}
    assert second.status == "completed" and second.todo == first.planned - k
    assert asked == set(sink.units) - done and not asked & done
    with FakeOpenAIServer() as server:
        third = run_session(run_input(scene, ontology), judge_pool(p, server), sink, p)
        assert server.chat_calls == []
    assert third.status == "nothing_to_do"


def test_breaker_moves_work_to_healthy_endpoint(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """한 엔드포인트가 계속 503이면 차단기가 열리고 다른 엔드포인트가 모두 처리해 completed다."""
    sink = MemorySink()
    p = fast(policy, breaker_cooldown_s=60)
    with FakeOpenAIServer() as bad, FakeOpenAIServer() as good:
        bad.fail(503, times=10**6)
        pool = judge_pool(p, bad, good)
        summary = run_session(
            run_input(scene, ontology), pool, sink, p,
            options=RunOptions(workers=8, endpoint_states=pool.states),
        )  # fmt: skip
        assert len(bad.chat_calls) <= p.server.breaker_failures + 8
        assert len(good.chat_calls) >= summary.planned
    assert summary.status == "completed" and len(sink.units) == summary.planned
    assert pool.states() == {"fake-0": "open", "fake-1": "ok"}
    assert {u.endpoint for u in sink.units.values()} == {"fake-1"}


def test_model_mismatch_aborts_with_config(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """서버가 보고한 모델이 정책 모델과 다르면 aborted:config (종료 5), 단위를 쓰지 않는다."""
    sink = MemorySink()
    with FakeOpenAIServer(served_model="other/model") as server:
        summary = run_session(
            run_input(scene, ontology), judge_pool(fast(policy), server), sink, fast(policy)
        )
    assert (summary.status, summary.reason, summary.exit_code) == ("aborted", "config", 5)
    assert sink.units == {} and events(sink) == [("started", None), ("aborted", "config")]
    assert summary.detail is not None and "other/model" in summary.detail


@pytest.mark.parametrize("change", ["hash", "unapproved"])
def test_render_change_aborts(
    change: str, scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """실행 중 렌더 재확인에서 해시가 바뀌었거나 승인이 풀렸으면 aborted:render_changed (종료 6)."""

    def check() -> dict[str, str]:
        if change == "unapproved":
            raise RenderNotCurrentError("승인 상태가 아닙니다")
        return {**RENDER, "tp1": "c" * 64}

    sink = MemorySink()
    slow = _Slow(OracleJudge(scene.labels), 0.02)
    summary = run_session(run_input(scene, ontology), slow, sink, fast(policy), render_check=check)
    assert (summary.status, summary.reason, summary.exit_code) == ("aborted", "render_changed", 6)
    assert len(sink.units) < summary.planned
    assert events(sink)[-1] == ("aborted", "render_changed")


def test_final_render_check_blocks_completion(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """모두 판정했어도 마지막 렌더 확인에서 바뀌었으면 completed가 아니다."""
    calls: list[int] = []

    def check() -> dict[str, str]:
        calls.append(1)
        return {**RENDER, "bodycam": "d" * 64}

    p = fast(policy, render_check_interval_s=3600)  # 주기 확인은 돌지 않고 마지막 확인만 돈다
    sink = MemorySink()
    summary = run_session(
        run_input(scene, ontology), OracleJudge(scene.labels), sink, p, render_check=check
    )
    assert summary.reason == "render_changed" and summary.status == "aborted"
    assert len(sink.units) == summary.planned and len(calls) == 1
    assert events(sink)[-1] == ("aborted", "render_changed")


class _Slow:
    """요청마다 기다리는 백엔드 감싸기."""

    def __init__(self, inner: JudgeBackend, delay: float) -> None:
        self.inner, self.delay, self.info = inner, delay, inner.info

    def judge(self, request: JudgeRequest) -> JudgeReply:
        time.sleep(self.delay)
        return self.inner.judge(request)


class _DownFor:
    """정해 둔 단위만 늘 장애인 백엔드 감싸기 (재시도가 끝나도 장애인 단위)."""

    def __init__(self, inner: JudgeBackend, unit_ids: set[str]) -> None:
        self.inner, self.unit_ids = inner, unit_ids
        self.info = BackendInfo("stub", "m", "f", ("stub",))

    def judge(self, request: JudgeRequest) -> JudgeReply:
        if request.unit_id in self.unit_ids:
            raise VlmUnavailableError("이 단위만 장애")
        return self.inner.judge(request)


def test_unit_outage_is_skipped_not_recorded_and_retried(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """재시도 끝에도 장애인 단위는 기록하지 않고(unsure로 쓰지 않음) 나머지를 끝낸 뒤
    interrupted:unavailable. 다음 실행이 그 단위만 묻는다."""
    inp = run_input(scene, ontology)
    plan = planned_units(inp, policy, RunOptions())
    bad = plan[1].unit_id
    sink = MemorySink()
    first = run_session(inp, _DownFor(OracleJudge(scene.labels), {bad}), sink, fast(policy))
    assert (first.status, first.reason, first.skipped) == ("interrupted", "unavailable", 1)
    assert bad not in sink.units and len(sink.units) == len(plan) - 1
    judge = OracleJudge(scene.labels)
    second = run_session(run_input(scene, ontology), judge, sink, fast(policy))
    assert second.status == "completed" and second.todo == 1
    assert {r.unit_id for r in judge.requests} == {bad}


def test_max_minutes_interrupts(scene: Scene, policy: AiReviewPolicy, ontology: Ontology) -> None:
    """마감 시각이 지나면 새 요청을 보내지 않고 interrupted:max_minutes (종료 8)."""
    sink, judge = MemorySink(), OracleJudge(scene.labels)
    summary = run_session(
        run_input(scene, ontology), judge, sink, fast(policy),
        options=RunOptions(deadline=time.monotonic() - 1),
    )  # fmt: skip
    assert (summary.status, summary.reason, summary.exit_code) == ("interrupted", "max_minutes", 8)
    assert judge.calls == 0 and sink.units == {}


def test_dry_run_plans_without_writing(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """계획만: 묶음별 단위 수를 세고 서버·기록 대상을 쓰지 않는다."""
    sink, judge = MemorySink(), OracleJudge(scene.labels)
    inp = run_input(scene, ontology)
    summary = run_session(inp, judge, sink, fast(policy), options=RunOptions(dry_run=True))
    plan = planned_units(inp, policy, RunOptions())
    assert summary.status == "dry_run" and summary.exit_code == 0
    assert summary.planned == summary.todo == len(plan)
    assert summary.groups == {
        g.value: sum(u.group is g for u in plan) for g in AiGroup if any(u.group is g for u in plan)
    }
    assert judge.calls == 0 and sink.runs == [] and sink.units == {}


def test_closed_plan_without_completed_event_gets_one(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """단위는 모두 있는데 마감 이벤트 없이 끝난 실행(강제 종료)이면, 재실행은 서버를 부르지 않고
    완료 기록(started → completed)만 남긴다. 그다음은 아무것도 쓰지 않는다."""
    sink = MemorySink()
    run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, fast(policy))
    sink.events = sink.events[:-1]  # completed 이벤트를 쓰기 전에 죽은 실행
    judge = OracleJudge(scene.labels)
    summary = run_session(run_input(scene, ontology), judge, sink, fast(policy))
    assert summary.status == "completed" and summary.written == 0 and judge.calls == 0
    assert events(sink) == [("started", None), ("started", None), ("completed", None)]
    assert run_session(run_input(scene, ontology), judge, sink, fast(policy)).status == (
        "nothing_to_do"
    )


def test_debug_dir_keeps_only_invalid_responses(
    tmp_path: Path, scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """--debug-dir: 형식이 맞지 않던 응답의 원문만 로컬 파일로 남긴다. DB 행에는 sha만 있다."""
    judge = OracleJudge(scene.labels, faults=("not_json",))
    sink = MemorySink()
    run_session(
        run_input(scene, ontology), judge, sink, fast(policy),
        options=RunOptions(groups=TEMPORAL, debug_dir=tmp_path),
    )  # fmt: skip
    files = list((tmp_path / scene.session.session_id).glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text("utf-8"))
    unit = sink.units[data["unit_id"]]
    assert data["responses"][0]["content"] == "이 구간은 판정할 수 없습니다"
    assert len(data["responses"]) == 2 and unit.response_sha256 is not None
    assert "이 구간은" not in unit.model_dump_json()


def test_progress_lines_text_and_json(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """진행 줄: 글은 `[aireview] <세션> n/m 단위(%)`로 시작하고, JSON은 한 줄 객체다."""
    for as_json in (False, True):
        lines: list[str] = []
        opts = RunOptions(json_progress=as_json, endpoint_states=lambda: {"o": "ok"})
        summary = run_session(
            run_input(scene, ontology), _Slow(OracleJudge(scene.labels), 0.01), MemorySink(),
            fast(policy), options=opts, progress=lines.append,
        )  # fmt: skip
        assert lines and summary.status == "completed"
        if as_json:
            last = json.loads(lines[-1])
            assert last["done"] == last["planned"] == summary.planned
            assert last["endpoints"] == {"o": "ok"}
        else:
            assert lines[-1].startswith(f"[aireview] s001 {summary.planned}/{summary.planned} 단위")
            assert "o ok" in lines[-1]


def test_summary_json_and_exit_codes() -> None:
    """요약 JSON과 사유별 종료 코드."""
    base = RunSummary("s001", "v", "completed")
    assert base.exit_code == 0 and base.to_json()["exit_code"] == 0
    codes = {
        "unavailable": 3, "max_minutes": 8, "config": 5, "render_changed": 6, "error": 1,
        "signal": 130, "session_not_ready": 6,
    }  # fmt: skip
    for reason, code in codes.items():
        assert RunSummary("s001", "v", "aborted", reason=reason).exit_code == code


def test_event_counts_record_written_and_verdicts(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """마감 이벤트 집계: 계획·기록 수, 판정 분포 (실행 기록은 엔드포인트 이름만)."""
    sink = MemorySink()
    summary = run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, fast(policy))
    done = sink.events[-1]
    assert done.event is AiRunEventKind.COMPLETED
    assert done.counts["planned"] == done.counts["written"] == summary.planned
    assert done.counts["verdict_ok"] == sum(summary.verdicts.values())
    assert sink.runs[0].backend.endpoints == ("oracle",)
    assert sink.runs[0].render_hashes == RENDER


class _Trip:
    """n번째 판정 요청이 끝나면 바깥 멈춤 신호를 켜는 백엔드 감싸기 (CLI의 Ctrl-C 신호 흉내)."""

    def __init__(self, inner: JudgeBackend, cancel: threading.Event, after: int) -> None:
        self.inner, self.cancel, self.after, self.info = inner, cancel, after, inner.info
        self.calls = 0
        self._lock = threading.Lock()

    def judge(self, request: JudgeRequest) -> JudgeReply:
        reply = self.inner.judge(request)
        with self._lock:
            self.calls += 1
            if self.calls >= self.after:
                self.cancel.set()
        return reply


def test_external_cancel_closes_with_signal(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """바깥 멈춤 신호(CLI가 여러 세션에 함께 주는 Ctrl-C 신호)가 실행 중 켜지면 새 단위를 묻지 않고
    interrupted:signal (종료 130). 이미 받은 판정은 기록한다. 시작 전에 이미 켜져 있으면 아무것도
    쓰지 않는다. 마지막 단위를 판정한 뒤에 켜졌으면 계획 단위가 모두 있으므로 completed다."""
    cancel = threading.Event()
    trip = _Trip(OracleJudge(scene.labels), cancel, after=2)
    sink = MemorySink()
    opts = RunOptions(workers=1, cancel=cancel)
    summary = run_session(run_input(scene, ontology), trip, sink, fast(policy), options=opts)
    assert (summary.status, summary.reason, summary.exit_code) == ("interrupted", "signal", 130)
    assert trip.calls == 2 and len(sink.units) == 2 < summary.planned
    assert events(sink)[-1] == ("interrupted", "signal")
    fresh, judge = MemorySink(), OracleJudge(scene.labels)
    before = run_session(run_input(scene, ontology), judge, fresh, fast(policy), options=opts)
    assert (before.status, before.reason) == ("interrupted", "signal")
    assert judge.calls == 0 and fresh.runs == [] and fresh.events == [] and fresh.units == {}
    planned = len(planned_units(run_input(scene, ontology), policy, RunOptions()))
    late = threading.Event()
    last = _Trip(OracleJudge(scene.labels), late, after=planned)
    done = MemorySink()
    summary = run_session(
        run_input(scene, ontology), last, done, fast(policy),
        options=RunOptions(workers=1, cancel=late),
    )  # fmt: skip
    assert summary.status == "completed" and late.is_set() and len(done.units) == planned


def test_deadline_stops_workers_waiting_on_open_endpoints(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """모든 엔드포인트가 열려 작업자가 보낼 곳을 기다리는 중에 --max-minutes 마감이 지나면
    abort_after_s(60초)를 기다리지 않고 interrupted:max_minutes로 곧 끝난다. 마감 뒤에는 요청을
    보내지 않는다 (기록자가 마감을 보고 멈추면 묶음에서 기다리던 요청도 멈춤 신호를 본다)."""
    p = fast(policy, breaker_failures=1, breaker_cooldown_s=60, abort_after_s=60, backoff_s=(0.0,))
    with FakeOpenAIServer() as server:
        server.fail(503, times=10**6)
        pool = judge_pool(p, server)
        start = time.monotonic()
        summary = run_session(
            run_input(scene, ontology), pool, MemorySink(), p,
            options=RunOptions(workers=2, deadline=time.monotonic() + 0.5),
        )  # fmt: skip
        took = time.monotonic() - start
        asked = len(server.chat_calls)
    assert (summary.status, summary.reason, summary.exit_code) == ("interrupted", "max_minutes", 8)
    assert took < 5 and asked <= 2 and not pool.dead


def test_debug_dir_write_failure_does_not_kill_workers(
    tmp_path: Path, scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """--debug-dir에 쓸 수 없어도(디렉터리 자리에 파일) 작업자가 죽지 않는다: 경고 한 번을 내고 더
    남기지 않으며 실행은 끝까지 가 completed다 (작업자가 모두 조용히 죽으면 생산자가 가득 찬 큐
    앞에서 영영 기다리고 실행이 세션 잠금을 쥔 채 멈춘다)."""
    blocker = tmp_path / "file"
    blocker.write_text("x")
    judge = OracleJudge(scene.labels, faults=("not_json",) * 1000)
    sink, lines = MemorySink(), list[str]()
    result: dict[str, RunSummary] = {}

    def go() -> None:
        result["s"] = run_session(
            run_input(scene, ontology), judge, sink, fast(policy),
            options=RunOptions(workers=1, debug_dir=blocker), progress=lines.append,
        )  # fmt: skip

    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout=60)
    assert not t.is_alive(), "실행이 멈추지 않았다"
    summary = result["s"]
    assert summary.status == "completed" and len(sink.units) == summary.planned
    assert sum("디버그 응답을 남기지 못해" in line for line in lines) == 1


class _BlindSink(MemorySink):
    """잠금이 뚫린 경쟁 실행 흉내: 계획할 때만 이미 쓴 단위를 모르는 것처럼 답한다 (그사이 다른
    실행이 같은 단위를 썼다. 쓰기·마감 확인은 같은 규칙)."""

    planned = False

    def done_unit_ids(self, session_id: str, version: str) -> set[str]:
        if not self.planned:
            self.planned = True
            return set()
        return super().done_unit_ids(session_id, version)

    def is_complete(self, session_id: str, version: str, plan_digest: str, *, full: bool) -> bool:
        return False


def test_written_counts_judged_units_inserted_counts_new_rows(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """written은 판정해 기록 대상에 넘긴 단위 수, inserted는 실제로 새로 들어간 행 수다. 다른
    실행이 같은 단위를 먼저 썼으면(`ON CONFLICT DO NOTHING`) inserted는 0이고 마감 이벤트에도
    그대로 남는다."""
    sink = MemorySink()
    first = run_session(run_input(scene, ontology), OracleJudge(scene.labels), sink, fast(policy))
    assert first.inserted == first.written == first.planned
    blind = _BlindSink()
    blind.units = dict(sink.units)
    again = run_session(run_input(scene, ontology), OracleJudge(scene.labels), blind, fast(policy))
    assert again.status == "completed" and again.written == again.planned and again.inserted == 0
    assert blind.events[-1].counts["written"] == again.planned
    assert blind.events[-1].counts["inserted"] == 0 and again.to_json()["inserted"] == 0
