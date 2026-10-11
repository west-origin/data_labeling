"""실제 HTTP 경계: 잘못된 입력·느린 연결·과부하·수집 실패가 성공으로 보이지 않아야 한다."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import pytest

from dlp_review.ops.policy import WebhookPolicy
from dlp_review.webhook import CollectRequest, ReviewerMismatchError, serve

BODY = json.dumps({"action": "ANNOTATION_UPDATED", "task": {"id": 3}}).encode()


@contextmanager
def running(
    collect: Callable[[CollectRequest], None],
    *,
    workers: int = 4,
    timeout: float = 0.5,
    max_body: int = 256,
) -> Generator[ThreadingHTTPServer]:
    policy = WebhookPolicy(
        bind_host="127.0.0.1",
        max_body_bytes=max_body,
        request_timeout_s=timeout,
        max_workers=workers,
        retry_after_s=30,
    )
    server = serve(0, "secret", collect, policy=policy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def post(
    server: ThreadingHTTPServer,
    body: bytes = BODY,
    *,
    secret: str = "secret",
    path: str = "/webhooks/label_studio",
) -> tuple[int, str | None]:
    with closing(http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)) as client:
        client.request("POST", path, body, {"X-DLP-Secret": secret})
        response = client.getresponse()
        response.read()
        return response.status, response.getheader("Retry-After")


def post_after_worker_release(server: ThreadingHTTPServer) -> int:
    """소켓 EOF 뒤 서버 finally의 슬롯 반환을 제한 시간 안에서 확인한다.

    재시도 중에는 503만 허용한다. 영구 슬롯 누수나 다른 오류 응답은 계속 실패한다.
    """
    deadline = time.monotonic() + 2
    while (status := post(server)[0]) == 503 and time.monotonic() < deadline:
        time.sleep(0.01)
    return status


def test_failure_is_retryable_and_retry_collects() -> None:
    calls: list[str] = []

    def collect(request: CollectRequest) -> None:
        calls.append(request.task_key)
        if len(calls) == 1:
            raise RuntimeError("내부 오류·민감정보")

    with running(collect) as server:
        assert post(server) == (503, "30")
        assert post(server) == (204, None)
    assert calls == ["label_studio:3", "label_studio:3"]


def test_auth_shape_size_and_route_never_collect() -> None:
    calls: list[CollectRequest] = []
    with running(calls.append) as server:
        assert post(server, secret="wrong")[0] == 401
        assert post(server, b"x" * 257)[0] == 413
        assert post(server, path="/other/label_studio")[0] == 404
        for malformed in (
            b"[]",
            b"null",
            b"not-json",
            b'{"action":"ANNOTATION_UPDATED","task":[]}',
            b'{"action":"ANNOTATION_UPDATED","task":{"id":true}}',
            b'{"action":"ANNOTATION_UPDATED","task":{"id":-1}}',
        ):
            assert post(server, malformed)[0] == 400
        assert post(server, b'{"action":"PING"}')[0] == 204
    assert calls == []


def test_reviewer_rejection_is_forbidden() -> None:
    def collect(request: CollectRequest) -> None:
        raise ReviewerMismatchError("담당자 불일치")

    with running(collect) as server:
        assert post(server) == (403, None)


def test_deep_json_is_bad_request_without_breaking_next_request() -> None:
    calls: list[CollectRequest] = []
    with running(calls.append, max_body=4096) as server:
        assert post(server, b"[" * 1500 + b"0" + b"]" * 1500)[0] == 400
        assert post_after_worker_release(server) == 204
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        (b"", 411),
        (b"Content-Length: -1\r\n", 400),
        (b"Content-Length: nope\r\n", 400),
        (b"Content-Length: 0\r\nContent-Length: 1\r\n", 400),
        (b"Transfer-Encoding: chunked\r\n", 400),
        (b"Content-Length: 257\r\n", 413),
        (b"Content-Length: " + b"9" * 5000 + b"\r\n", 413),
    ],
)
def test_length_validation(headers: bytes, status: int) -> None:
    calls: list[CollectRequest] = []
    with (
        running(calls.append) as server,
        socket.create_connection(("127.0.0.1", server.server_port), timeout=2) as client,
    ):
        client.sendall(
            b"POST /webhooks/label_studio HTTP/1.1\r\nHost: localhost\r\n" + headers + b"\r\n"
        )
        assert client.recv(4096).split()[1] == str(status).encode()
    assert calls == []


def test_partial_body_times_out_and_releases_connection() -> None:
    calls: list[CollectRequest] = []
    with running(calls.append, workers=1, timeout=0.1) as server:
        closed, release = threading.Event(), threading.Event()
        original_shutdown = server.shutdown_request

        def delayed_shutdown(request: socket.socket) -> None:
            original_shutdown(request)
            if not closed.is_set():
                closed.set()
                # 소켓은 닫혔지만 워커 finally는 아직 실행되지 않은 순서를 강제로 만든다.
                release.wait(timeout=2)

        with patch.object(server, "shutdown_request", side_effect=delayed_shutdown):
            try:
                with socket.create_connection(
                    ("127.0.0.1", server.server_port), timeout=2
                ) as client:
                    client.sendall(
                        b"POST /webhooks/label_studio HTTP/1.1\r\nContent-Length: 10\r\n\r\nx"
                    )
                    assert client.recv(4096).split()[1] == b"408"
                    while client.recv(4096):
                        pass
                assert closed.wait(timeout=1)
                assert post(server) == (503, "30")
            finally:
                release.set()
            assert post_after_worker_release(server) == 204
    assert len(calls) == 1


def test_worker_limit_returns_retryable_overload() -> None:
    entered, release = threading.Event(), threading.Event()

    def collect(request: CollectRequest) -> None:
        entered.set()
        release.wait(timeout=3)

    with running(collect, workers=1) as server:
        first = threading.Thread(target=post, args=(server,))
        first.start()
        try:
            assert entered.wait(timeout=1)
            # 과부하는 본문을 읽기 전에 응답하므로 본문 없는 요청으로 HTTP 응답을 확인한다.
            assert post(server, b"") == (503, "30")
        finally:
            release.set()
            first.join(timeout=2)
        assert not first.is_alive()


def test_empty_secret_is_rejected_before_listening() -> None:
    policy = WebhookPolicy(
        bind_host="127.0.0.1",
        max_body_bytes=256,
        request_timeout_s=1,
        max_workers=1,
        retry_after_s=30,
    )
    with pytest.raises(ValueError, match="비어"):
        serve(0, "", lambda request: None, policy=policy)
