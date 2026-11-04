"""작업 실행기: 상태 전이·로그·실패 처리·동시 실행 방지·중단 복구·가리기."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dlp_console.app import create_app
from dlp_console.jobs import JobConflictError, JobRunner, JobStatus, group_alive
from dlp_console.redact import RAW_MASK, SECRET_MASK, Redactor
from dlp_console.settings import ConsoleSettings, JobsConfig


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


# 앞 콘솔 흉내: 작업 하나(자식은 손주 sleep을 띄우고 기다림)를 맡기고 손주 pid를 알린 뒤 잠든다
_CONSOLE = """
import sys, time
from pathlib import Path
from dlp_console.jobs import JobRunner
from dlp_console.redact import Redactor
from dlp_console.settings import ConsoleSettings, load_console_config
from dlp_schema import repo_root
root = repo_root()
s = ConsoleSettings(root=root, work_dir=Path(sys.argv[1]), config=load_console_config(root),
                    dlp_argv=["/bin/sh", "-c", 'sleep 60 & echo $! > "$0"; wait', sys.argv[2]],
                    env={"PATH": "/usr/bin:/bin"})
r = JobRunner(s, Redactor("dlp-raw", {}))
r.submit("prelabel", "AI", [["prelabel"]], lock="session:s1", session_id="s1")
print("ready", flush=True)
time.sleep(600)
"""


def _wait_for(cond: Callable[[], bool], timeout: float = 20) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


def _fast(settings: ConsoleSettings) -> ConsoleSettings:
    jobs = JobsConfig(stop_grace_s=2, orphan_poll_s=0.1)
    settings.config = settings.config.model_copy(update={"jobs": jobs})
    return settings


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _zombie(pid: int) -> bool:
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip().startswith("Z")


_PR_SET_CHILD_SUBREAPER = 36


@pytest.fixture
def non_reaping_init() -> Iterator[None]:
    """고아를 늦게 거두거나 아예 거두지 않는 PID 1(컨테이너) 흉내 (F-COE-91).

    시험 프로세스를 child subreaper로 만들고 거두지 않는다. 그러면 콘솔 자식 그룹의 고아(손주,
    죽은 앞 콘솔의 자식)가 좀비로 남는다. 리눅스가 아니면 아무것도 하지 않는다 (시험은 그대로 돈다).
    끝날 때 남은 좀비 자식을 거두고 원래대로 돌린다.
    """
    libc = ctypes.CDLL(None, use_errno=True) if os.uname().sysname == "Linux" else None
    if libc is not None:
        assert libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    try:
        yield
    finally:
        if libc is not None:
            libc.prctl(_PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)
            out = subprocess.run(
                ["ps", "-A", "-o", "pid=,ppid=,stat="], capture_output=True, text=True
            ).stdout
            for pid, ppid, stat in (line.split()[:3] for line in out.splitlines()):
                if int(ppid) == os.getpid() and stat.startswith("Z"):
                    with contextlib.suppress(ChildProcessError):
                        os.waitpid(int(pid), 0)


def _exited_unreaped_group() -> subprocess.Popen[bytes]:
    """끝났지만 아무도 거두지 않은 자식 그룹 (리더 하나, 좀비)."""
    proc = subprocess.Popen(["true"], start_new_session=True)
    assert _wait_for(lambda: _zombie(proc.pid))
    return proc


def test_group_alive_ignores_zombie_members() -> None:
    """F-COE-91: 끝났지만 거두지 않은(좀비) 프로세스만 남은 그룹은 살아 있지 않다."""
    proc = _exited_unreaped_group()
    try:
        assert not group_alive(proc.pid)
    finally:
        proc.wait()


def test_group_alive_true_for_running_group() -> None:
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        assert group_alive(proc.pid)
    finally:
        proc.kill()
        proc.wait()
    assert not group_alive(proc.pid)


def test_group_alive_reads_state_past_parenthesised_command_name(tmp_path: Path) -> None:
    """F-COE-91: 명령 이름에 `)`·공백이 있어도 상태·그룹을 제대로 읽는다 (`/proc/<pid>/stat`)."""
    odd = tmp_path / "x) Z 1 (y"
    odd.symlink_to(
        subprocess.run(["which", "sleep"], capture_output=True, text=True).stdout.strip()
    )
    proc = subprocess.Popen([str(odd), "30"], start_new_session=True)
    try:
        assert _wait_for(lambda: group_alive(proc.pid), 5)
    finally:
        proc.kill()
        proc.wait()


def test_recover_does_not_adopt_zombie_group(settings: ConsoleSettings) -> None:
    """F-COE-91: 다시 켠 콘솔은 좀비만 남은 그룹을 맡지 않는다 (interrupted, 잠금 풀림)."""
    proc = _exited_unreaped_group()
    try:
        events = settings.work_dir / "jobs.jsonl"
        created = {
            "event": "created", "job_id": "j-z", "step": "prelabel", "title": "AI",
            "session_id": "s1", "lock": "session:s1", "commands": [["prelabel"]],
            "created_at": "2026-11-04T00:00:00+00:00", "status": "running", "exit_code": None,
            "failed_command": None, "finished_at": None, "cleanup": [],
        }  # fmt: skip
        spawned = {"event": "spawned", "job_id": "j-z", "pid": proc.pid, "pgid": proc.pid}
        events.write_text(json.dumps(created) + "\n" + json.dumps(spawned) + "\n", "utf-8")
        runner = _runner(_fast(settings))
        assert _wait_for(lambda: runner.get("j-z").status is JobStatus.INTERRUPTED)  # type: ignore[union-attr]
        again = runner.submit("prelabel", "다시", [["sync", "run", "s1"]], lock="session:s1")
        assert runner.wait(again.job_id, 30).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]
    finally:
        proc.wait()


@pytest.mark.usefixtures("non_reaping_init")
def test_sigterm_console_keeps_lock_while_orphan_child_runs(
    settings: ConsoleSettings, tmp_path: Path
) -> None:
    """F-COE-79: SIGTERM으로 죽은 콘솔의 자식이 살아 있으면 다시 켠 콘솔은 같은 잠금을 안 받는다.

    F-COE-91: 고아를 거두지 않는 init 아래에서도, 그룹이 좀비만 남으면 끝난 것으로 본다.
    """
    script = tmp_path / "console.py"
    script.write_text(_CONSOLE, "utf-8")
    pidfile = tmp_path / "grandchild.pid"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(settings.work_dir), str(pidfile)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
        assert _wait_for(lambda: pidfile.is_file() and pidfile.read_text().strip() != "")
        grandchild = int(pidfile.read_text().strip())
        proc.send_signal(signal.SIGTERM)  # 신호 처리 없는 콘솔: 자식 그룹은 남는다
        proc.wait(10)
        assert _alive(grandchild)
        runner = _runner(_fast(settings))
        (job,) = runner.jobs()
        assert job.status is JobStatus.RUNNING and job.pgid is not None
        with pytest.raises(JobConflictError):
            runner.submit("prelabel", "AI", [["prelabel"]], lock="session:s1", session_id="s1")
        # 남은 자식 그룹이 끝나면 interrupted로 남고 잠금이 풀린다
        os.killpg(job.pgid, signal.SIGKILL)
        assert _wait_for(lambda: runner.get(job.job_id).status is JobStatus.INTERRUPTED)  # type: ignore[union-attr]
        assert not _alive(grandchild)  # 고아를 물려받은 콘솔은 그 좀비를 거둔다
        again = runner.submit("a", "다시", [["sync", "run", "s1"]], lock="session:s1")
        assert runner.wait(again.job_id, 30).status is JobStatus.SUCCEEDED  # type: ignore[union-attr]
    finally:
        proc.kill()
        if pidfile.is_file() and pidfile.read_text().strip():
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(pidfile.read_text().strip()), signal.SIGKILL)


@pytest.mark.usefixtures("non_reaping_init")
def test_shutdown_stops_child_group_and_records_interrupted(
    settings: ConsoleSettings, tmp_path: Path
) -> None:
    """F-COE-79: 콘솔 종료(shutdown)는 자식 그룹(손주까지)을 멈추고 작업을 interrupted로 남긴다.

    손주가 끝난 뒤 좀비로 남을 수 있어(고아를 거두지 않는 init) 그룹 생존으로 판정한다 (F-COE-91).
    """
    pidfile = tmp_path / "grandchild.pid"
    settings.dlp_argv = ["/bin/sh", "-c", 'sleep 60 & echo $! > "$0"; wait', str(pidfile)]
    runner = _runner(_fast(settings))
    job = runner.submit("a", "길게", [["x"], ["y"]], lock="session:s1")
    assert _wait_for(lambda: pidfile.is_file() and pidfile.read_text().strip() != "")
    pgid = runner.get(job.job_id).pgid  # type: ignore[union-attr]
    assert pgid is not None and group_alive(pgid)
    runner.shutdown()
    assert _wait_for(lambda: not group_alive(pgid))
    done = runner.get(job.job_id)
    assert done is not None and done.status is JobStatus.INTERRUPTED
    events = (settings.work_dir / "jobs.jsonl").read_text("utf-8")
    assert '"spawned"' in events
    assert "$ dlp y" not in "\n".join(runner.tail(job.job_id))  # 다음 명령은 시작하지 않는다
    assert _runner(settings).get(job.job_id).status is JobStatus.INTERRUPTED  # type: ignore[union-attr]


@pytest.mark.usefixtures("non_reaping_init")
def test_app_lifespan_shutdown_stops_jobs(services: Any, tmp_path: Path) -> None:
    """uvicorn 종료(SIGTERM)는 앱 lifespan 종료 → runner.shutdown으로 이어진다.

    손주가 좀비로 남을 수 있어 그룹 생존으로 판정한다 (F-COE-91).
    """
    _fast(services.settings)
    pidfile = tmp_path / "grandchild.pid"
    services.settings.dlp_argv = ["/bin/sh", "-c", 'sleep 60 & echo $! > "$0"; wait', str(pidfile)]
    with TestClient(create_app(services.settings, services)):
        job = services.runner.submit("a", "길게", [["x"]], lock="session:s1")
        assert _wait_for(lambda: pidfile.is_file() and pidfile.read_text().strip() != "")
        pgid = services.runner.get(job.job_id).pgid
    assert pgid is not None
    assert _wait_for(lambda: not group_alive(pgid))
    assert services.runner.get(job.job_id).status is JobStatus.INTERRUPTED
