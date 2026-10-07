# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""검수 도구 웹훅. 작업이 끝나면 수집을 일으킨다.

- CVAT: X-Signature-256 = "sha256=" + HMAC-SHA256(비밀, 본문). 작업(job) 상태가 completed로 바뀔 때.
- Label Studio: 사용자 정의 헤더 X-DLP-Secret = 비밀. 주석이 만들어지거나 갱신될 때.

파이프라인 위치: `dlp review serve`(웹훅 서버)가 `serve`로 서버를 띄우고, 요청마다
`parse_event` → (CLI 콜백에서) `resolve_reviewer` → `dlp_review.collect.collect_task` 순으로
수집한다. `dlp review register-webhooks`는 `dlp_review.clients`의 `add_webhook`으로 도구에 웹훅을
등록한다.
관련: WP6, ADR 0006, ADR 0024(CVAT job 담당자와 계정 연결 검사).

공개 이름:
- `CollectRequest`: 웹훅에서 꺼낸 수집 요청 (작업 키, 도구 사용자).
- `ReviewerMismatchError`: 웹훅의 사용자와 작업 담당자가 맞지 않는다.
- `resolve_reviewer`: 작업 담당자와 도구 사용자를 맞춰 실제 검수자를 정한다.
- `WebhookAuthError`: 서명·비밀이 맞지 않는다 (HTTP 401).
- `parse_event`: 헤더·본문 → `CollectRequest` (수집할 사건이 아니면 None).
- `serve`: POST /webhooks/<도구>를 받는 HTTP 서버를 만든다.

보안 주의: 비밀 비교는 `hmac.compare_digest`(상수 시간)로 한다. 웹훅 사용자 정보는 작업 담당자
검사에만 쓰고, 담당자가 있는 작업이면 기록되는 검수자는 늘 그 담당자다 (담당자가 없는 작업만 CVAT가
알려 준 사용자 이름을 쓴다).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool


@dataclass(frozen=True)
class CollectRequest:
    """reviewer: 도구가 알려 준 사용자 이름 (CVAT 작업 담당자). Label Studio는 숫자 ID만 주므로
    user_id에 싣고 reviewer는 비운다. 실제 검수자는 resolve_reviewer로 작업 담당자에서 정한다."""

    # 작업 키 `"<도구>:<도구 작업 ID>"` (예: `cvat:7`, `label_studio:3`) = review_tasks.task_key
    task_key: str
    reviewer: str | None
    # Label Studio 주석을 만든 사용자의 도구 내부 숫자 ID (서비스 계정 주석을 거르는 데만 쓴다)
    user_id: int | None = None


class ReviewerMismatchError(PermissionError):
    """웹훅의 도구 사용자와 작업 담당자(또는 그 CVAT 계정)가 맞지 않아 수집하지 않는다."""


def resolve_reviewer(
    request: CollectRequest,
    task: ReviewTask,
    service_user_id: int | None = None,
    cvat_users: Mapping[str, str] | None = None,
) -> str | None:
    """웹훅 요청의 검수자. None이면 수집하지 않는다 (서비스 계정이 만든 주석).

    - 작업에 담당자가 있으면 그 담당자다. 도구가 사용자 이름을 알려 주면(CVAT) 담당자와 같아야 한다.
    - CVAT 작업은 담당자의 CVAT 계정(review.yaml cvat.users)과 job 담당자를 맞춘다. 연결이 있으면
      job 담당자가 없거나 다르면 받지 않는다. 블러 검수는 연결이 없어도 받지 않는다 (ADR 0024).
    - Label Studio의 completed_by는 도구 내부 숫자 ID라 검수자 ID로 쓰지 않는다.

    인자:
    - request: `parse_event` 결과.
    - task: DB의 검수 작업 (`review_tasks`).
    - service_user_id: 우리 서비스 계정의 Label Studio 사용자 ID. 이 사용자의 주석은 무시한다.
    - cvat_users: dlp 검수자 ID → CVAT 사용자 이름 (`review.yaml cvat.users`).

    반환: 기록할 검수자 ID (dlp 검수자 ID), 또는 None(수집하지 않음).
    예외: `ReviewerMismatchError` — 담당자·계정이 맞지 않거나 검수자를 알 수 없을 때.
    """
    # 서비스 계정(우리가 올린 주석)이 일으킨 웹훅은 사람 검수가 아니다
    if service_user_id is not None and request.user_id == service_user_id:
        return None
    if task.tool is ReviewTool.CVAT and task.assignee is not None:
        expected = (cvat_users or {}).get(task.assignee)
        if expected is None and task.stage is ReviewStage.PRIVACY:
            # 블러 검수는 원본을 보므로 계정 연결이 없으면 누가 했는지 확인할 수 없어 거부한다
            raise ReviewerMismatchError(
                f"{task.task_key}: 블러 검수 담당자 {task.assignee}의 CVAT 계정 연결이 없습니다"
            )
        if expected is not None and request.reviewer != expected:
            raise ReviewerMismatchError(
                f"{task.task_key}: CVAT job 담당자({request.reviewer or '없음'})가 "
                f"담당자 {task.assignee}의 계정 {expected}이 아닙니다"
            )
        if expected is not None:
            return task.assignee
        # 연결이 없는 작업 라벨 CVAT 작업은 아래의 이름 비교로 넘어간다 (예전 방식)
    if task.assignee is not None:
        if request.reviewer is not None and request.reviewer != task.assignee:
            raise ReviewerMismatchError(
                f"{task.task_key}: 담당자 {task.assignee}가 아닌 {request.reviewer}가 끝냈습니다"
            )
        return task.assignee
    # 담당자가 없는 작업은 도구가 알려 준 사용자 이름을 그대로 쓴다 (없으면 거부)
    if request.reviewer is None:
        raise ReviewerMismatchError(f"{task.task_key}: 담당자가 없고 검수자를 알 수 없습니다")
    return request.reviewer


class WebhookAuthError(PermissionError):
    """웹훅 서명(CVAT)이나 공유 비밀(Label Studio)이 맞지 않는다. 서버는 401로 답한다."""


def parse_event(
    tool: str, headers: Mapping[str, str], body: bytes, secret: str
) -> CollectRequest | None:
    """웹훅 요청을 확인하고 수집 요청으로 바꾼다.

    인자:
    - tool: `"cvat"` 또는 `"label_studio"` (URL 경로 /webhooks/<tool>).
    - headers: HTTP 헤더 (이름 대소문자 무시).
    - body: 원본 본문 바이트 (CVAT 서명은 이 바이트로 계산한다).
    - secret: 웹훅 공유 비밀 (등록할 때 쓴 값).

    반환: `CollectRequest`, 또는 수집할 사건이 아니면 None
    (CVAT: `update:job`이면서 job 상태가 completed일 때만, Label Studio: 주석 생성·갱신일 때만).
    예외: `WebhookAuthError`(인증 실패), `ValueError`(알 수 없는 도구·JSON 오류),
    `KeyError`(필수 필드 없음). 서버는 앞의 것은 401, 뒤 둘은 400으로 답한다.
    """
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
        # completed_by는 숫자 ID 또는 사용자 객체({"id": …})로 온다
        user = (event.get("annotation") or {}).get("completed_by")
        if isinstance(user, dict):
            user = user.get("id")
        user_id = int(user) if isinstance(user, int | str) and str(user).isdigit() else None
        return CollectRequest(f"label_studio:{event['task']['id']}", None, user_id)
    raise ValueError(f"알 수 없는 도구: {tool}")


def serve(
    port: int, secret: str, on_collect: Callable[[CollectRequest], None]
) -> ThreadingHTTPServer:
    """POST /webhooks/cvat, /webhooks/label_studio를 받는 서버 (호출자가 serve_forever).

    인자:
    - port: 들을 포트 (모든 인터페이스 0.0.0.0에 묶는다).
    - secret: 웹훅 공유 비밀.
    - on_collect: 수집 요청마다 부를 콜백. 요청 스레드에서 동기로 불린다. 예외 처리는 콜백이
      맡는다 (`dlp_cli.review_cmds`의 콜백이 담당자 불일치·작업 오류를 잡아 출력만 한다).

    응답: 성공·무시 204, 인증 실패 401, 형식 오류 400.
    """

    class Handler(BaseHTTPRequestHandler):
        """요청 하나를 처리한다 (ThreadingHTTPServer가 요청마다 스레드를 만든다)."""

        def do_POST(self) -> None:
            """웹훅 본문을 읽어 확인하고, 수집할 사건이면 콜백을 부른 뒤 204로 답한다."""
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
            """기본 접근 로그(stderr)를 끈다."""
            pass

    return ThreadingHTTPServer(("0.0.0.0", port), Handler)
