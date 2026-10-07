# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""검수 도구 웹훅. 작업이 끝나면 수집을 일으킨다.

- CVAT: X-Signature-256 = "sha256=" + HMAC-SHA256(비밀, 본문). 작업(job) 상태가 completed로 바뀔 때.
- Label Studio: 사용자 정의 헤더 X-DLP-Secret = 비밀. 주석이 만들어지거나 갱신될 때.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class CollectRequest:
    task_key: str
    reviewer: str


class WebhookAuthError(PermissionError):
    pass


def parse_event(
    tool: str, headers: Mapping[str, str], body: bytes, secret: str
) -> CollectRequest | None:
    lower = {k.lower(): v for k, v in headers.items()}
    if tool == "cvat":
        expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(lower.get("x-signature-256", ""), expected):
            raise WebhookAuthError("CVAT 웹훅 서명이 맞지 않습니다")
        event: Any = json.loads(body)
        job = event.get("job") or {}
        if event.get("event") != "update:job" or job.get("state") != "completed":
            return None
        assignee = (job.get("assignee") or {}).get("username") or (event.get("sender") or {}).get(
            "username"
        )
        return CollectRequest(f"cvat:{job['task_id']}", assignee or "unknown")
    if tool == "label_studio":
        if not hmac.compare_digest(lower.get("x-dlp-secret", ""), secret):
            raise WebhookAuthError("Label Studio 웹훅 비밀이 맞지 않습니다")
        event = json.loads(body)
        if event.get("action") not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            return None
        user = (event.get("annotation") or {}).get("completed_by")
        reviewer = user.get("email") if isinstance(user, dict) else str(user or "unknown")
        return CollectRequest(f"label_studio:{event['task']['id']}", reviewer or "unknown")
    raise ValueError(f"알 수 없는 도구: {tool}")


def serve(
    port: int, secret: str, on_collect: Callable[[CollectRequest], None]
) -> ThreadingHTTPServer:
    """POST /webhooks/cvat, /webhooks/label_studio를 받는 서버 (호출자가 serve_forever)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            tool = self.path.removeprefix("/webhooks/")
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                request = parse_event(tool, dict(self.headers.items()), body, secret)
            except WebhookAuthError:
                self.send_response(401)
                self.end_headers()
                return
            except (ValueError, KeyError):
                self.send_response(400)
                self.end_headers()
                return
            if request is not None:
                on_collect(request)
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)
