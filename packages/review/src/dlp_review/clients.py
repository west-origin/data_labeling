# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""CVAT·Label Studio REST 클라이언트 (필요한 호출만)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx


class ToolError(RuntimeError):
    pass


def _check(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        raise ToolError(
            f"{resp.request.method} {resp.request.url} → {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json() if resp.content else None


class CvatClient:
    def __init__(self, base_url: str, username: str, password: str) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=120)
        key = _check(
            self.http.post("/api/auth/login", json={"username": username, "password": password})
        )
        self.http.headers["Authorization"] = f"Token {key['key']}"

    @classmethod
    def from_env(cls) -> CvatClient:
        return cls(
            f"http://localhost:{os.environ.get('DLP_CVAT_PORT', '8080')}",
            os.environ.get("DLP_CVAT_ADMIN_USER", "admin"),
            os.environ.get("DLP_CVAT_ADMIN_PASSWORD", "dlp-dev-password"),
        )

    def find_project(self, name: str) -> dict[str, Any] | None:
        res = _check(self.http.get("/api/projects", params={"name": name, "page_size": 100}))
        return next((p for p in res["results"] if p["name"] == name), None)

    def create_project(self, name: str, labels: list[dict[str, Any]]) -> int:
        return int(
            _check(self.http.post("/api/projects", json={"name": name, "labels": labels}))["id"]
        )

    def project_labels(self, project_id: int) -> list[dict[str, Any]]:
        res = _check(
            self.http.get("/api/labels", params={"project_id": project_id, "page_size": 500})
        )
        return list(res["results"])

    def create_task(self, name: str, project_id: int, video: Path) -> int:
        task = _check(self.http.post("/api/tasks", json={"name": name, "project_id": project_id}))
        with video.open("rb") as f:
            rq = _check(
                self.http.post(
                    f"/api/tasks/{task['id']}/data",
                    data={"image_quality": "95", "sorting_method": "predefined"},
                    files={"client_files[0]": (video.name, f, "video/mp4")},
                )
            )
        self._wait(rq["rq_id"])
        return int(task["id"])

    def _wait(self, rq_id: str, timeout_s: float = 300) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = _check(self.http.get(f"/api/requests/{rq_id}"))["status"]
            if status == "finished":
                return
            if status == "failed":
                raise ToolError(f"CVAT 작업 실패: {rq_id}")
            time.sleep(1)
        raise ToolError(f"CVAT 작업 대기 시간 초과: {rq_id}")

    def frame_count(self, task_id: int) -> int:
        return int(_check(self.http.get(f"/api/tasks/{task_id}/data/meta"))["size"])

    def put_tracks(self, task_id: int, tracks: list[dict[str, Any]]) -> None:
        body = {"version": 0, "tags": [], "shapes": [], "tracks": tracks}
        _check(self.http.put(f"/api/tasks/{task_id}/annotations", json=body))

    def get_tracks(self, task_id: int) -> list[dict[str, Any]]:
        return list(_check(self.http.get(f"/api/tasks/{task_id}/annotations"))["tracks"])

    def add_webhook(self, project_id: int, url: str, secret: str) -> int:
        body = {
            "target_url": url, "type": "project", "project_id": project_id,
            "events": ["update:job"], "secret": secret, "content_type": "application/json",
            "enable_ssl": False, "is_active": True,
        }  # fmt: skip
        return int(_check(self.http.post("/api/webhooks", json=body))["id"])


class LabelStudioClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.http = httpx.Client(
            base_url=base_url, timeout=60, headers={"Authorization": f"Token {token}"}
        )

    @classmethod
    def from_env(cls) -> LabelStudioClient:
        return cls(
            f"http://localhost:{os.environ.get('DLP_LABEL_STUDIO_PORT', '8081')}",
            os.environ.get("DLP_LABEL_STUDIO_TOKEN", "dlpdevlabelstudiotoken0000000000000000"),
        )

    def find_project(self, title: str) -> dict[str, Any] | None:
        res = _check(self.http.get("/api/projects", params={"title": title, "page_size": 100}))
        items = res["results"] if isinstance(res, dict) else res
        return next((p for p in items if p["title"] == title), None)

    def create_project(self, title: str, label_config: str) -> int:
        body = {"title": title, "label_config": label_config}
        return int(_check(self.http.post("/api/projects", json=body))["id"])

    def create_task(
        self, project_id: int, data: dict[str, Any], results: list[dict[str, Any]]
    ) -> int:
        task = _check(self.http.post("/api/tasks", json={"project": project_id, "data": data}))
        if results:
            body = {"result": results, "ground_truth": False}
            _check(self.http.post(f"/api/tasks/{task['id']}/annotations", json=body))
        return int(task["id"])

    def latest_results(self, task_id: int) -> list[dict[str, Any]]:
        anns = _check(self.http.get(f"/api/tasks/{task_id}/annotations"))
        if not anns:
            return []
        latest = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        return list(latest["result"])

    def add_webhook(self, project_id: int, url: str, secret: str) -> int:
        body = {
            "project": project_id, "url": url, "send_payload": True, "send_for_all_actions": False,
            "actions": ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
            "headers": {"X-DLP-Secret": secret}, "is_active": True,
        }  # fmt: skip
        return int(_check(self.http.post("/api/webhooks", json=body))["id"])
