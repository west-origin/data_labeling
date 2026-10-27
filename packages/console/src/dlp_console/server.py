"""`dlp console serve` 실행 (uvicorn, ADR 0061 4절).

기본은 127.0.0.1에만 묶는다. 다른 주소는 `allow_remote`가 있어야 하고 경고를 찍는다 (인증은 토큰
하나, 통신은 암호화되지 않는다). 시작 때 만든 토큰을 실은 콘솔 주소를 한 번 찍는다 (토큰은 로그
파일에 남기지 않는다. 다시 시작하면 새 토큰이다).
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from dlp_console.app import create_app
from dlp_console.settings import default_settings, load_dotenv

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class RemoteBindError(ValueError):
    """명시 허용 없이 이 컴퓨터 밖에서 닿는 주소에 묶으려 했다."""


def serve(root: Path, host: str | None, port: int | None, *, allow_remote: bool = False) -> None:
    """콘솔을 띄운다 (Ctrl-C로 끝낸다). 예외: RemoteBindError."""
    # S3Store 등 라이브러리는 os.environ을 읽으므로 .env 값을 (이미 있는 변수는 그대로 두고) 넣는다
    for key, value in load_dotenv(root / ".env").items():
        os.environ.setdefault(key, value)
    settings = default_settings(root)
    bind = host or settings.config.server.host
    bind_port = port or settings.config.server.port
    if bind not in LOCAL_HOSTS and not allow_remote:
        raise RemoteBindError(
            f"{bind}에 묶으면 다른 컴퓨터에서 콘솔에 닿습니다. "
            "정말 필요하면 --allow-remote를 함께 주세요"
        )
    if bind not in LOCAL_HOSTS:
        print(
            "경고: 콘솔을 이 컴퓨터 밖에 엽니다. 토큰 하나로만 막고 통신은 암호화되지 않습니다. "
            "믿을 수 있는 사내망에서만, 쓰고 나면 바로 끄세요."
        )
    app = create_app(settings, bind_host=bind)
    shown = "localhost" if bind in LOCAL_HOSTS else bind
    print(f"운영 콘솔: http://{shown}:{bind_port}/#token={settings.token}")
    print(f"작업 디렉터리: {settings.work_dir} (업로드 사본·작업 로그)")
    uvicorn.run(app, host=bind, port=bind_port, log_level="warning", access_log=False)
