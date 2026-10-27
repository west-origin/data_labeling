"""작업 실행기: 상태 전이·로그·실패 처리·동시 실행 방지·중단 복구·가리기."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dlp_console.jobs import JobConflictError, JobRunner, JobStatus
from dlp_console.redact import RAW_MASK, SECRET_MASK, Redactor
from dlp_console.settings import ConsoleSettings


def _runner(settings: ConsoleSettings) -> JobRunner:
    return JobRunner(settings, Redactor("dlp-raw", settings.env))


def test_success_runs_commands_in_order_and_cleans_up(settings: ConsoleSettings) -> None:
    runner = _runner(settings)
    junk = settings.work_dir / "upload.mp4"
    junk.write_bytes(b"x")
    job = runner.submit(
        "sync", "동기화", [["sync", "run", "s1"], ["privacy", "detect", "s1"]],
        lock="session:s1", session_id="s1", cleanup=[junk],
    )  # fmt: skip
    done = runner.wait(job.job_id, 30)
    assert done is not None and done.status is JobStatus.SUCCEEDED and done.exit_code == 0
    log = runner.tail(job.job_id)
    assert log[0] == "$ dlp sync run s1" and "args: privacy detect s1" in log
    assert not junk.exists()  # 성공했을 때만 지운다


def test_failure_stops_and_keeps_files_and_redacts(settings: ConsoleSettings) -> None:
    runner = _runner(settings)
    keep = settings.work_dir / "keep.mp4"
    keep.write_bytes(b"x")
    job = runner.submit(
        "x", "실패", [["fail"], ["never"]], lock="session:s1", session_id="s1", cleanup=[keep]
    )
    done = runner.wait(job.job_id, 30)
    assert done is not None and done.status is JobStatus.FAILED
    assert done.exit_code == 3 and done.failed_command == 0
    text = "\n".join(runner.tail(job.job_id))
    assert "never" not in text  # 뒤 명령은 돌지 않는다
    assert "dlp-raw" not in text and RAW_MASK in text  # 원본 위치를 가린다
    assert "dlp-dev-password" not in text and SECRET_MASK in text
    assert keep.exists()


def test_same_lock_is_rejected_while_running(settings: ConsoleSettings) -> None:
    runner = _runner(settings)
    first = runner.submit("a", "느린 작업", [["sleep", "1.5"]], lock="session:s1")
    with pytest.raises(JobConflictError):
        runner.submit("b", "같은 세션", [["sync", "run", "s1"]], lock="session:s1")
    other = runner.submit("c", "다른 세션", [["sync", "run", "s2"]], lock="session:s2")
    assert runner.wait(other.job_id, 30) is not None
    runner.wait(first.job_id, 30)
    again = runner.submit("b", "끝난 뒤", [["sync", "run", "s1"]], lock="session:s1")
    assert runner.wait(again.job_id, 30).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]


def test_state_is_folded_from_append_only_events_and_running_becomes_interrupted(
    settings: ConsoleSettings,
) -> None:
    runner = _runner(settings)
    job = runner.submit("a", "끝남", [["sync", "run", "s1"]], lock="session:s1", session_id="s1")
    runner.wait(job.job_id, 30)
    # 콘솔이 꺼진 사이 끝 사건이 없는 작업을 흉내 낸다
    events = settings.work_dir / "jobs.jsonl"
    created = {
        "event": "created", "job_id": "j-lost", "step": "privacy_detect", "title": "탐지",
        "session_id": "s1", "lock": "session:s1", "commands": [["privacy", "detect", "s1"]],
        "created_at": "2026-10-27T00:00:00+00:00", "status": "running", "exit_code": None,
        "failed_command": None, "finished_at": None, "cleanup": [],
    }  # fmt: skip
    with events.open("a", encoding="utf-8") as f:
        f.write(json.dumps(created) + "\n")
        f.write("{깨진 줄\n")
    lines_before = events.read_text("utf-8").count("\n")
    again = _runner(settings)
    lost = again.get("j-lost")
    assert lost is not None and lost.status is JobStatus.INTERRUPTED
    assert again.get(job.job_id).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]
    # 사건은 추가만 한다 (앞 줄을 고치지 않는다)
    text = events.read_text("utf-8")
    assert text.count("\n") == lines_before + 1 and '"interrupted"' in text.splitlines()[-1]


def test_launch_failure_is_reported(settings: ConsoleSettings, tmp_path: Path) -> None:
    settings.dlp_argv = [str(tmp_path / "없는-실행-파일")]
    runner = _runner(settings)
    job = runner.submit("a", "시작 실패", [["sync", "run", "s1"]], lock="x")
    done = runner.wait(job.job_id, 30)
    assert done is not None and done.status is JobStatus.FAILED and done.exit_code == 127
    assert any("명령을 시작하지 못했습니다" in x for x in runner.tail(job.job_id))
