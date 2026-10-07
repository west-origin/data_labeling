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

from dlp_schema.review import ReviewTask


@dataclass(frozen=True)
class CollectRequest:
    """reviewer: 도구가 알려 준 사용자 이름 (CVAT 작업 담당자). Label Studio는 숫자 ID만 주므로
    user_id에 싣고 reviewer는 비운다. 실제 검수자는 resolve_reviewer로 작업 담당자에서 정한다."""

    task_key: str
    reviewer: str | None
    user_id: int | None = None


class ReviewerMismatchError(PermissionError):
    pass


def resolve_reviewer(
    request: CollectRequest, task: ReviewTask, service_user_id: int | None = None
) -> str | None:
    """웹훅 요청의 검수자. None이면 수집하지 않는다 (서비스 계정이 만든 주석).

    - 작업에 담당자가 있으면 그 담당자다. 도구가 사용자 이름을 알려 주면(CVAT) 담당자와 같아야 한다.
    - Label Studio의 completed_by는 도구 내부 숫자 ID라 검수자 ID로 쓰지 않는다.
    """
    if service_user_id is not None and request.user_id == service_user_id:
        return None
    if task.assignee is not None:
        if request.reviewer is not None and request.reviewer != task.assignee:
            raise ReviewerMismatchError(
                f"{task.task_key}: 담당자 {task.assignee}가 아닌 {request.reviewer}가 끝냈습니다"
            )
        return task.assignee
    if request.reviewer is None:
        raise ReviewerMismatchError(f"{task.task_key}: 담당자가 없고 검수자를 알 수 없습니다")
    return request.reviewer


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
        # 상태를 바꾼 사람(sender)은 관리자일 수 있어 쓰지 않는다. 작업(job) 담당자만 본다
        assignee = (job.get("assignee") or {}).get("username")
        return CollectRequest(f"cvat:{job['task_id']}", assignee or None)
    if tool == "label_studio":
        if not hmac.compare_digest(lower.get("x-dlp-secret", ""), secret):
            raise WebhookAuthError("Label Studio 웹훅 비밀이 맞지 않습니다")
        event = json.loads(body)
        if event.get("action") not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
            return None
        user = (event.get("annotation") or {}).get("completed_by")
        if isinstance(user, dict):
            user = user.get("id")
        user_id = int(user) if isinstance(user, int | str) and str(user).isdigit() else None
        return CollectRequest(f"label_studio:{event['task']['id']}", None, user_id)
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
