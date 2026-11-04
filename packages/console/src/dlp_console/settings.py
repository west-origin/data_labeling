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
    """`dlp console serve`의 기본 묶을 주소와 포트 (명령 옵션 `--host`·`--port`가 앞선다)."""

    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)


class UploadConfig(Contract):
    """영상 올리기.

    - max_mb: 한 파일 상한 (MB, 1 MB = 1024*1024 바이트). 넘으면 받던 파일을 지우고 400.
    - suffixes: 받는 확장자 (소문자, 점 포함. 예: `.mp4`).
    - delete_after_ingest: 수집 작업이 모두 성공하면 올린 사본·준비 사본을 지운다.
    """

    max_mb: int = Field(ge=1)
    suffixes: tuple[str, ...] = Field(min_length=1)
    delete_after_ingest: bool


class LogsConfig(Contract):
    """작업 로그 꼬리. tail_lines: 화면에 보이는 마지막 줄 수 (10~5000)."""

    tail_lines: int = Field(ge=10, le=5000)


class JobsConfig(Contract):
    """자식 `dlp` 프로세스 다루기 (F-COE-79).

    - stop_grace_s: 콘솔 종료 때 자식 그룹에 SIGTERM을 보낸 뒤 SIGKILL까지 기다리는 시간 (초).
    - orphan_poll_s: 앞 콘솔이 남긴 자식 그룹이 끝났는지 보는 간격 (초).
    """

    stop_grace_s: float = Field(default=10.0, gt=0, le=600)
    orphan_poll_s: float = Field(default=2.0, gt=0, le=60)


class PreviewConfig(Contract):
    """결과 보기.

    - signed_url_ttl_s: 블러본 서명 URL 유효 시간 (초, 60초~7일).
    - max_tracks: 한 번에 그리는 박스·키포인트 트랙 상한 (신뢰도 높은 순).
    """

    signed_url_ttl_s: int = Field(ge=60, le=7 * 24 * 3600)
    max_tracks: int = Field(ge=1)


class CompareConfig(Contract):
    """가져온 가중치의 골든셋 비교 명령 (`dlp train run objects … --trainer`).

    - trainer: 비교에 쓸 학습기 이름. None이면 비교 버튼이 꺼진다 (F-COE-74).
    - params: `--param 키=값`으로 넘길 값. 값의 `{model_name}`은 가져온 모델 이름으로 바뀐다.
    """

    trainer: str | None = None
    params: dict[str, str] = Field(default_factory=dict[str, str])


class PretrainImportConfig(Contract):
    """외부 가중치 가져오기.

    - models_dir: 가중치를 둘 디렉터리 (저장소 루트 기준 상대 경로, 저장소 안이어야 한다).
    - compare: 골든셋 비교 명령 설정.
    """

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
    - env: 자식 프로세스 환경. `.env` 값 위에 os.environ을 얹는다 (같은 이름이면 이미 있는 환경
      변수가 이긴다).
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
