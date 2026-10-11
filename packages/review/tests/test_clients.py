"""검수 도구 클라이언트의 회수용 메서드 (ADR 0039). 실제 도구 대신 httpx.MockTransport로 답한다.

정답: 삭제·존재 확인은 404를 "없음"으로, 2xx를 성공으로, 그 밖은 `ToolError`로 본다. 배정 해제는
작업과 모든 job의 담당자를 비운다. Label Studio 스냅숏에는 작업 data(서명 URL)가 없다.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from dlp_review.clients import CvatClient, LabelStudioClient, ToolError

Handler = Callable[[httpx.Request], httpx.Response]


def _cvat(handler: Handler) -> CvatClient:
    """로그인 없이 가짜 전송을 쓰는 CVAT 클라이언트."""
    client = CvatClient.__new__(CvatClient)
    client.http = httpx.Client(base_url="http://cvat", transport=httpx.MockTransport(handler))
    return client


def _ls(handler: Handler) -> LabelStudioClient:
    """가짜 전송을 쓰는 Label Studio 클라이언트."""
    client = LabelStudioClient("http://ls", "token")
    client.http = httpx.Client(base_url="http://ls", transport=httpx.MockTransport(handler))
    return client


@pytest.mark.parametrize(("status", "expected"), [(204, True), (404, False), (500, None)])
def test_delete_task_404_is_gone_and_other_errors_raise(status: int, expected: bool | None) -> None:
    """DELETE 204 → True(지움), 404 → False(이미 없음), 500 → ToolError (두 도구 모두)."""
    seen: list[tuple[str, str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append((req.method, req.url.path))
        return httpx.Response(status)

    for client, path in ((_cvat(handler), "/api/tasks/5"), (_ls(handler), "/api/tasks/5/")):
        if expected is None:
            with pytest.raises(ToolError):
                client.delete_task(5)
        else:
            assert client.delete_task(5) is expected
        assert seen[-1] == ("DELETE", path)


def test_task_exists_reads_404_as_missing() -> None:
    """GET 200 → 있음, 404 → 없음, 403 → ToolError (권한 문제를 "없음"으로 숨기지 않는다)."""
    answers = {"/api/tasks/1": 200, "/api/tasks/2": 404, "/api/tasks/3": 403}

    def handler(req: httpx.Request) -> httpx.Response:
        code = answers[req.url.path.rstrip("/")]
        return httpx.Response(code, json={"id": 1} if code == 200 else None)

    for client in (_cvat(handler), _ls(handler)):
        assert client.task_exists(1) is True
        assert client.task_exists(2) is False
        with pytest.raises(ToolError):
            client.task_exists(3)


def test_cvat_unassign_clears_task_and_every_job() -> None:
    """작업 담당자(assignee_id)와 job마다 담당자(assignee)를 None으로 PATCH한다."""
    patches: list[tuple[str, object]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/api/jobs":
            return httpx.Response(200, json={"results": [{"id": 12}, {"id": 11}]})
        patches.append((req.url.path, json.loads(req.content)))
        return httpx.Response(200, json={})

    _cvat(handler).unassign(7)
    assert patches == [
        ("/api/tasks/7", {"assignee_id": None}),
        ("/api/jobs/11", {"assignee": None}),
        ("/api/jobs/12", {"assignee": None}),
    ]


def test_snapshots_keep_annotations_but_not_media_urls() -> None:
    """스냅숏은 주석을 담고 서명 URL(작업 data)은 담지 않는다. CVAT는 프레임 수를 함께 담는다."""
    url = "http://s3/dlp-labeling/sessions/s/review/a/bodycam.mp4?X-Amz-Signature=x"

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/tasks/3/annotations":
            return httpx.Response(200, json=[{"id": 1, "result": [{"value": {}}]}])
        if path == "/api/tasks/3/":
            return httpx.Response(200, json={"id": 3, "data": {"video": url}, "drafts": []})
        if path == "/api/tasks/4":
            return httpx.Response(200, json={"id": 4, "name": "s/bodycam/a", "data": url})
        if path == "/api/tasks/4/annotations":
            return httpx.Response(200, json={"tracks": [{"id": 9}], "shapes": [], "tags": []})
        if path == "/api/tasks/4/data/meta":
            return httpx.Response(200, json={"size": 30})
        return httpx.Response(404)

    ls = _ls(handler).task_snapshot(3)
    assert ls["annotations"] == [{"id": 1, "result": [{"value": {}}]}] and ls["drafts"] == []
    cvat = _cvat(handler).task_snapshot(4)
    assert cvat["frame_count"] == 30 and cvat["annotations"]["tracks"] == [{"id": 9}]
    assert cvat["task"]["name"] == "s/bodycam/a"
    for snap in (ls, cvat):
        assert "X-Amz-Signature" not in json.dumps(snap)
