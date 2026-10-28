"""작업(job) 실행기: 기존 `dlp <단계>` 명령을 자식 프로세스로 돌리고 상태·로그를 남긴다.

ADR 0061 2절.

- 작업 하나는 `dlp` 명령 1개 이상을 차례로 돈다. 앞 명령이 실패(0이 아닌 종료 코드)하면 멈춘다.
- 상태는 작업 디렉터리 `jobs.jsonl`의 사건(created·started·finished·interrupted)을 추가만 해서
  남기고, 읽을 때 접어서 만든다. 로그는 `logs/<작업 ID>.log`.
- 같은 잠금 키(세션 하나, 또는 데이터셋·학습 같은 전역 단계)의 작업이 돌고 있으면 새 작업을 받지
  않는다 (`JobConflictError`). 단계 명령은 멱등이라 끝난 뒤 다시 눌러도 안전하다.
- 명령마다 자식을 새 프로세스 그룹(새 세션)으로 띄우고 `spawned` 사건에 pid·pgid를 남긴다
  (F-COE-79).
- 콘솔이 꺼질 때(`shutdown`, uvicorn 종료 = SIGTERM·Ctrl-C) 돌고 있는 자식 그룹에 SIGTERM, 기다려도
  남으면 SIGKILL을 보내고 그 작업을 interrupted로 남긴다.
- 끝 사건 없이 다음 시작을 맞은 작업: 마지막 자식 그룹이 아직 살아 있으면(SIGKILL 등으로 콘솔만
  죽은 경우) `adopted` 사건을 남기고 실행 중으로 둔다 (잠금 유지). 그 그룹이 끝나면 interrupted로
  남긴다. 그룹이 없으면 바로 interrupted다 (자식 프로세스 결과는 모른다).
- `cleanup`: 모든 명령이 성공했을 때만 지울 파일 (업로드 사본).
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from dlp_console.redact import Redactor
from dlp_console.settings import ConsoleSettings

# 명령을 시작하지 못했을 때(실행 파일 없음 등) 남기는 종료 코드 (셸 관례)
LAUNCH_FAILED = 127


class JobStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"  # 콘솔이 꺼져 결과를 모름


class JobConflictError(RuntimeError):
    """같은 잠금 키의 작업이 이미 돌고 있다."""

    def __init__(self, running: Job) -> None:
        super().__init__(f"이미 실행 중인 작업이 있습니다: {running.title} ({running.job_id})")
        self.running = running


@dataclass(frozen=True)
class Job:
    """작업 하나의 지금 상태 (사건을 접은 값)."""

    job_id: str
    step: str
    title: str
    session_id: str | None
    lock: str
    commands: tuple[tuple[str, ...], ...]
    created_at: str
    status: JobStatus = JobStatus.RUNNING
    exit_code: int | None = None
    failed_command: int | None = None
    finished_at: str | None = None
    cleanup: tuple[str, ...] = field(default=())
    pid: int | None = None  # 마지막으로 띄운 자식 (spawned 사건)
    pgid: int | None = None

    def display_commands(self) -> list[str]:
        """화면용 명령 줄 (`dlp …`)."""
        return ["dlp " + shlex.join(c) for c in self.commands]

    def public(self) -> dict[str, Any]:
        """API 응답용 dict (지울 파일 경로는 싣지 않는다)."""
        data = asdict(self)
        data.pop("cleanup")
        data["commands"] = self.display_commands()
        data["status"] = self.status.value
        return data


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobRunner:
    """작업 실행·기록. 스레드 안전 (상태 접기·잠금 확인은 한 잠금 아래)."""

    def __init__(self, settings: ConsoleSettings, redactor: Redactor) -> None:
        self.settings = settings
        self.redactor = redactor
        self.events_path = settings.work_dir / "jobs.jsonl"
        self.logs_dir = settings.work_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._stopping = threading.Event()
        self._jobs: dict[str, Job] = self._fold()
        self._recover()

    # ── 기록 ──
    def _append(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _fold(self) -> dict[str, Job]:
        """`jobs.jsonl` 사건 → 작업 ID별 지금 상태 (깨진 줄은 건너뛴다)."""
        jobs: dict[str, Job] = {}
        if not self.events_path.is_file():
            return jobs
        for line in self.events_path.read_text("utf-8").splitlines():
            try:
                ev: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue
            _apply(jobs, ev)
        return jobs

    def _recover(self) -> None:
        """앞 콘솔이 끝 사건 없이 남긴 작업: 자식 그룹이 살아 있으면 맡고, 없으면 interrupted."""
        for job in list(self._jobs.values()):
            if job.status is not JobStatus.RUNNING:
                continue
            if job.pgid is not None and group_alive(job.pgid):
                ev = {"event": "adopted", "job_id": job.job_id, "at": _now(), "pgid": job.pgid}
                self._append(ev)
                thread = threading.Thread(
                    target=self._watch_orphan, args=(job.job_id, job.pgid), daemon=True
                )
                self._threads[job.job_id] = thread
                thread.start()
                continue
            self._interrupt(job.job_id)

    def _interrupt(self, job_id: str) -> None:
        ev = {"event": "interrupted", "job_id": job_id, "at": _now()}
        self._append(ev)
        _apply(self._jobs, ev)

    def _watch_orphan(self, job_id: str, pgid: int) -> None:
        """앞 콘솔의 자식 그룹이 끝날 때까지 잠금을 쥐고 있다가 interrupted로 남긴다."""
        poll = self.settings.config.jobs.orphan_poll_s
        while group_alive(pgid):
            if self._stopping.wait(poll):
                return  # 콘솔이 꺼진다: 다음 시작이 다시 본다
        with self._lock:
            self._interrupt(job_id)

    # ── 조회 ──
    def jobs(self, session_id: str | None = None) -> list[Job]:
        """작업 목록 (새것 먼저). session_id를 주면 그 세션 것만."""
        with self._lock:
            items = list(self._jobs.values())
        if session_id is not None:
            items = [j for j in items if j.session_id == session_id]
        return sorted(items, key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def running(self, lock: str) -> Job | None:
        """그 잠금 키로 돌고 있는 작업."""
        with self._lock:
            return next(
                (
                    j
                    for j in self._jobs.values()
                    if j.lock == lock and j.status is JobStatus.RUNNING
                ),
                None,
            )

    def log_path(self, job_id: str) -> Path:
        return self.logs_dir / f"{job_id}.log"

    def tail(self, job_id: str, lines: int | None = None) -> list[str]:
        """로그 마지막 줄들 (원본 위치·비밀값을 가림)."""
        path = self.log_path(job_id)
        if not path.is_file():
            return []
        n = lines or self.settings.config.logs.tail_lines
        text = path.read_text("utf-8", errors="replace").splitlines()[-n:]
        return [self.redactor.apply(x) for x in text]

    # ── 실행 ──
    def submit(
        self,
        step: str,
        title: str,
        commands: Sequence[Sequence[str]],
        *,
        lock: str,
        session_id: str | None = None,
        cleanup: Sequence[Path] = (),
    ) -> Job:
        """작업을 만들고 바로 시작한다. 예외: 같은 잠금 키가 돌면 JobConflictError."""
        if not commands:
            raise ValueError("실행할 명령이 없습니다")
        with self._lock:
            busy = next(
                (
                    j
                    for j in self._jobs.values()
                    if j.lock == lock and j.status is JobStatus.RUNNING
                ),
                None,
            )
            if busy is not None:
                raise JobConflictError(busy)
            job = Job(
                job_id=f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}",
                step=step,
                title=title,
                session_id=session_id,
                lock=lock,
                commands=tuple(tuple(c) for c in commands),
                created_at=_now(),
                cleanup=tuple(str(p) for p in cleanup),
            )
            ev = {"event": "created", **_job_record(job)}
            self._append(ev)
            _apply(self._jobs, ev)
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        self._threads[job.job_id] = thread
        thread.start()
        return job

    def shutdown(self) -> None:
        """콘솔 종료: 돌고 있는 자식 그룹을 멈추고 그 작업을 interrupted로 남긴다 (F-COE-79).

        SIGTERM을 보내고 `jobs.stop_grace_s`만큼 기다린 뒤 남은 그룹에 SIGKILL. 앞 콘솔에서 맡은
        그룹(adopted)은 건드리지 않는다 (다음 시작이 다시 맡는다).
        """
        self._stopping.set()
        with self._lock:
            procs = list(self._procs.values())
        for proc in procs:
            signal_group(proc.pid, signal.SIGTERM)
        grace = self.settings.config.jobs.stop_grace_s
        for proc in procs:
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                signal_group(proc.pid, signal.SIGKILL)
                proc.wait()
        for thread in list(self._threads.values()):
            thread.join(grace)

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        """작업 스레드가 끝날 때까지 기다린다 (시험·종료용)."""
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)
        return self.get(job_id)

    def _run(self, job: Job) -> None:
        code, failed = 0, None
        with self.log_path(job.job_id).open("a", encoding="utf-8") as log:
            for i, cmd in enumerate(job.commands):
                if self._stopping.is_set():
                    break
                log.write(f"$ dlp {shlex.join(cmd)}\n")
                log.flush()
                code = self._run_one(job.job_id, cmd, log)
                if self._stopping.is_set():
                    log.write("[콘솔 종료로 멈춤]\n")
                    break
                if code != 0:
                    failed = i
                    log.write(f"[종료 코드 {code}]\n")
                    break
        if self._stopping.is_set():
            with self._lock:
                self._interrupt(job.job_id)
            return
        if code == 0:
            for p in job.cleanup:
                Path(p).unlink(missing_ok=True)
        status = JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED
        ev = {
            "event": "finished",
            "job_id": job.job_id,
            "at": _now(),
            "status": status.value,
            "exit_code": code,
            "failed_command": failed,
        }
        with self._lock:
            self._append(ev)
            _apply(self._jobs, ev)

    def _run_one(self, job_id: str, cmd: Sequence[str], log: Any) -> int:
        """명령 하나를 돌려 종료 코드를 낸다. 시작하지 못하면 로그에 이유를 쓰고 LAUNCH_FAILED.

        자식은 새 세션(= 새 프로세스 그룹, pgid = pid)으로 띄우고 pid·pgid를 `spawned` 사건으로
        남긴다. 콘솔 종료 때 그룹째 멈추고, 다음 시작이 살아 있는 그룹을 알아보게 한다.
        """
        with self._lock:
            if self._stopping.is_set():
                return LAUNCH_FAILED
            try:
                proc = subprocess.Popen(  # 고정 접두 + 콘솔이 만든 인자 목록, 셸 없음
                    [*self.settings.dlp_argv, *cmd],
                    cwd=self.settings.root,
                    env=self.settings.child_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                log.write(f"명령을 시작하지 못했습니다: {exc}\n")
                return LAUNCH_FAILED
            self._procs[job_id] = proc
            ev = {"event": "spawned", "job_id": job_id, "at": _now(), "pid": proc.pid,
                  "pgid": proc.pid}  # fmt: skip
            self._append(ev)
            _apply(self._jobs, ev)
        try:
            return proc.wait()
        finally:
            with self._lock:
                self._procs.pop(job_id, None)


def group_alive(pgid: int) -> bool:
    """그 프로세스 그룹에 프로세스가 남아 있는가 (신호 0). 다른 사용자의 그룹이면 있다고 본다."""
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def signal_group(pgid: int, sig: signal.Signals) -> None:
    """그룹에 신호를 보낸다 (이미 없으면 조용히 넘어간다). 콘솔 자신의 그룹에는 보내지 않는다."""
    if pgid <= 1 or pgid == os.getpgrp():
        return
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        return


def _job_record(job: Job) -> dict[str, Any]:
    data = asdict(job)
    data["status"] = job.status.value
    data["commands"] = [list(c) for c in job.commands]
    data["cleanup"] = list(job.cleanup)
    return data


def _apply(jobs: dict[str, Job], ev: dict[str, Any]) -> None:
    """사건 하나를 작업 상태에 반영한다 (모르는 사건·작업은 무시)."""
    kind = ev.get("event")
    if kind == "created":
        data = {k: v for k, v in ev.items() if k != "event"}
        data["commands"] = tuple(tuple(c) for c in data["commands"])
        data["cleanup"] = tuple(data.get("cleanup", ()))
        data["status"] = JobStatus(data["status"])
        jobs[data["job_id"]] = Job(**data)
        return
    job = jobs.get(str(ev.get("job_id")))
    if job is None:
        return
    if kind == "finished":
        jobs[job.job_id] = replace(
            job,
            status=JobStatus(ev["status"]),
            exit_code=ev.get("exit_code"),
            failed_command=ev.get("failed_command"),
            finished_at=ev.get("at"),
        )
    elif kind == "spawned":
        jobs[job.job_id] = replace(job, pid=ev.get("pid"), pgid=ev.get("pgid"))
    elif kind == "interrupted":
        jobs[job.job_id] = replace(job, status=JobStatus.INTERRUPTED, finished_at=ev.get("at"))
