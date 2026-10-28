"""콘솔 설정·실행 환경 (ADR 0061).

- `ConsoleConfig`: `config/console.yaml` 모델 (모르는 키 거부).
- `ConsoleSettings`: 한 콘솔 프로세스의 실행 값 (저장소 루트, 작업 디렉터리, `dlp` 실행 접두, 토큰,
  자식 프로세스 환경).
- `load_console_config`·`load_dotenv`·`default_settings`.

비밀값: `.env`를 읽어 자식 프로세스(`dlp …`) 환경에만 넣는다. 화면·로그에는 값을 내지 않는다
(`dlp_console.redact`가 로그 꼬리에서 비밀값을 가린다).
"""

from __future__ import annotations

import os
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract

# 작업 디렉터리 환경 변수와 기본값 (저장소 밖. 업로드 사본·작업 로그·가져오기 기록이 여기 있다)
WORK_DIR_ENV = "DLP_CONSOLE_DIR"
DEFAULT_WORK_DIR = "~/.dlp-console"


class ServerConfig(Contract):
    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)


class UploadConfig(Contract):
    max_mb: int = Field(ge=1)
    suffixes: tuple[str, ...] = Field(min_length=1)
    delete_after_ingest: bool


class LogsConfig(Contract):
    tail_lines: int = Field(ge=10, le=5000)


class JobsConfig(Contract):
    """자식 `dlp` 프로세스 다루기 (F-COE-79)."""

    stop_grace_s: float = Field(default=10.0, gt=0, le=600)
    orphan_poll_s: float = Field(default=2.0, gt=0, le=60)


class PreviewConfig(Contract):
    signed_url_ttl_s: int = Field(ge=60, le=7 * 24 * 3600)
    max_tracks: int = Field(ge=1)


class CompareConfig(Contract):
    trainer: str | None = None
    params: dict[str, str] = Field(default_factory=dict[str, str])


class PretrainImportConfig(Contract):
    models_dir: str
    compare: CompareConfig


class ConsoleConfig(Contract):
    """`config/console.yaml` 전체."""

    version: int
    server: ServerConfig
    upload: UploadConfig
    logs: LogsConfig
    jobs: JobsConfig = Field(default_factory=JobsConfig)
    preview: PreviewConfig
    pretrain_import: PretrainImportConfig


def load_console_config(root: Path) -> ConsoleConfig:
    """저장소 루트의 `config/console.yaml`을 읽어 검증한다. 예외: OSError, ValidationError."""
    data: Any = yaml.safe_load((root / "config" / "console.yaml").read_text("utf-8"))
    return ConsoleConfig.model_validate(data)


def load_dotenv(path: Path) -> dict[str, str]:
    """`.env`(KEY=VALUE 줄)를 읽는다. 없으면 빈 dict. 따옴표 한 겹은 벗기고 주석은 건너뛴다."""
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


@dataclass
class ConsoleSettings:
    """콘솔 프로세스 하나의 실행 값.

    - root: 저장소 루트 (정책·설정을 읽고 `dlp`를 이 디렉터리에서 돌린다).
    - work_dir: 작업 디렉터리 (저장소 밖).
    - config: `config/console.yaml`.
    - dlp_argv: `dlp` 명령 접두 (기본 `python -m dlp_cli.main`). 시험은 가짜 명령으로 바꾼다.
    - env: 자식 프로세스 환경 (os.environ 위에 .env, 이미 있는 변수가 앞선다).
    - token: 화면·API 요청 토큰 (시작 때 무작위, 콘솔 URL에 싣는다).
    """

    root: Path
    work_dir: Path
    config: ConsoleConfig
    dlp_argv: list[str] = field(default_factory=lambda: [sys.executable, "-m", "dlp_cli.main"])
    env: dict[str, str] = field(default_factory=dict[str, str])
    token: str = field(default_factory=lambda: secrets.token_urlsafe(24))

    def child_env(self) -> dict[str, str]:
        """`dlp` 자식 프로세스 환경 (사본)."""
        return dict(self.env)


def default_settings(root: Path, work_dir: Path | None = None) -> ConsoleSettings:
    """저장소 루트와 환경 변수로 기본 설정을 만든다. 작업 디렉터리를 만든다(권한 0700)."""
    base = work_dir or Path(os.environ.get(WORK_DIR_ENV, DEFAULT_WORK_DIR)).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    base.chmod(0o700)
    env = {**load_dotenv(root / ".env"), **os.environ}
    # 자식 출력이 바로 로그에 쌓이게 (꼬리 보기)
    env["PYTHONUNBUFFERED"] = "1"
    return ConsoleSettings(root=root, work_dir=base, config=load_console_config(root), env=env)
