# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""CVAT·Label Studio REST 클라이언트 (필요한 호출만).

WP6, ADR 0006. `dlp_review.tasks`(작업 만들기), `dlp_review.collect`(결과 받기),
`dlp review register-webhooks`(웹훅 등록)가 쓴다.

공개 이름:
- `ToolError`: 도구 API가 4xx·5xx로 답했거나 비동기 작업이 실패·시간 초과했다.
- `TaskOwner`, `cvat_task_name`: 도구 작업이 어느 세션·스트림·담당자의 것인지 (회수 전 대조).
- `PRELABEL_VERSION`, `PRELABEL_SETTINGS`: Label Studio 프리라벨(예측) 표시 설정.
- `CvatClient`: CVAT REST (프로젝트·작업·주석·job 담당자·이슈·웹훅, 회수용 주인 확인·배정 해제·
  스냅숏·삭제).
- `LabelStudioClient`: Label Studio REST (프로젝트·작업·예측·주석·웹훅, 회수용 주인 확인·스냅숏·
  삭제).

회수(ADR 0039)용 메서드는 `dlp_review.revoke`가 쓴다. 삭제·존재 확인은 404를 "없음"으로 보고
(이미 지운 작업의 재시도가 성공한다), 그 밖의 4xx·5xx는 `ToolError`다. 실행기는 지우기 전에
`task_owner`로 도구 작업이 그 작업 행의 것인지 대조한다 (도구 인스턴스를 새로 만들면 같은 번호가
다른 세션 작업일 수 있다).

환경 변수 (개발 기본값은 `services/` compose와 맞춘다):
- CVAT: `DLP_CVAT_PORT`(8080), `DLP_CVAT_ADMIN_USER`, `DLP_CVAT_ADMIN_PASSWORD`.
- Label Studio: `DLP_LABEL_STUDIO_PORT`(8081), `DLP_LABEL_STUDIO_TOKEN`.
기본 비밀번호·토큰은 개발용 값이다. 운영 환경에서는 반드시 환경 변수로 바꾼다.

주의: 이 클라이언트는 서비스(관리자) 계정으로 동작한다. 검수자 계정과 섞이지 않도록 프리라벨은
Label Studio 예측으로만 올린다 (`LabelStudioClient.create_task`).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class ToolError(RuntimeError):
    """검수 도구 API 호출 실패 (HTTP 4xx·5xx, CVAT 비동기 작업 실패·시간 초과)."""


@dataclass(frozen=True)
class TaskOwner:
    """도구에 저장된 작업의 주인: 세션·스트림·담당자. 회수 실행기가 지우기 전에 작업 행과 대조한다.

    알 수 없는 칸은 None이다. Label Studio 작업은 스트림을 담지 않는다(바디캠 한 스트림).
    """

    session_id: str | None
    stream_id: str | None
    assignee: str | None
    # 사람이 읽는 원문 (CVAT 작업 이름, Label Studio data의 세션·담당자). 오류 메시지에 쓴다
    label: str


def cvat_task_name(session_id: str, stream_id: str, assignee: str) -> str:
    """CVAT 작업 이름 `<세션>/<스트림>/<담당자>` (블러 검수 작업은 담당자 자리에 `privacy`).

    `CvatClient.task_owner`가 이 모양을 되읽는다. 식별자에는 `/`가 없다 (`dlp_schema.common`).
    """
    return f"{session_id}/{stream_id}/{assignee}"


PRELABEL_VERSION = (
    "dlp-prelabel"  # Label Studio 예측의 모델 버전 이름 (프로젝트 설정과 같아야 보인다)
)
# 예측을 검수 화면에 미리 채워 보인다. model_version은 그 버전의 예측이 생긴 뒤에만 정할 수 있다
PRELABEL_SETTINGS = {"show_collab_predictions": True}


def _user(annotation: dict[str, Any]) -> int | None:
    """주석을 만든 사용자 ID (completed_by는 ID 또는 사용자 객체)."""
    user = annotation.get("completed_by")
    if isinstance(user, dict):
        user = user.get("id")
    return int(user) if user is not None else None


def _gone(resp: httpx.Response) -> bool:
    """404면 True (작업이 없다). 그 밖의 실패는 `ToolError`, 성공은 False."""
    if resp.status_code == 404:
        return True
    _check(resp)
    return False


def _check(resp: httpx.Response) -> Any:
    """응답 상태를 확인하고 JSON 본문을 돌려준다 (본문이 비었으면 None).

    예외: 상태 코드 ≥ 400이면 `ToolError` (메서드·URL·상태·본문 앞 500자를 담는다).
    """
    if resp.status_code >= 400:
        raise ToolError(
            f"{resp.request.method} {resp.request.url} → {resp.status_code}: {resp.text[:500]}"
        )
    return resp.json() if resp.content else None


class CvatClient:
    """CVAT REST 클라이언트. 생성할 때 로그인해 토큰을 헤더에 싣는다."""

    def __init__(self, base_url: str, username: str, password: str) -> None:
        """base_url의 CVAT에 username·password로 로그인한다.

        예외: 로그인 실패면 `ToolError`, 서버에 닿지 않으면 `httpx.HTTPError`.
        """
        # 원본 영상 업로드·처리가 길어질 수 있어 시간 제한을 넉넉히 둔다 (초)
        self.http = httpx.Client(base_url=base_url, timeout=120)
        key = _check(
            self.http.post("/api/auth/login", json={"username": username, "password": password})
        )
        self.http.headers["Authorization"] = f"Token {key['key']}"

    @classmethod
    def from_env(cls) -> CvatClient:
        """환경 변수(`DLP_CVAT_PORT`, `DLP_CVAT_ADMIN_USER`, `DLP_CVAT_ADMIN_PASSWORD`)로 만든다."""
        return cls(
            f"http://localhost:{os.environ.get('DLP_CVAT_PORT', '8080')}",
            os.environ.get("DLP_CVAT_ADMIN_USER", "admin"),
            os.environ.get("DLP_CVAT_ADMIN_PASSWORD", "dlp-dev-password"),
        )

    def find_project(self, name: str) -> dict[str, Any] | None:
        """이름이 정확히 같은 프로젝트 (없으면 None). 첫 100개 결과에서만 찾는다."""
        res = _check(self.http.get("/api/projects", params={"name": name, "page_size": 100}))
        return next((p for p in res["results"] if p["name"] == name), None)

    def create_project(self, name: str, labels: list[dict[str, Any]]) -> int:
        """라벨 정의(`cvat.label_spec`)로 프로젝트를 만들고 ID를 돌려준다."""
        return int(
            _check(self.http.post("/api/projects", json={"name": name, "labels": labels}))["id"]
        )

    def project_labels(self, project_id: int) -> list[dict[str, Any]]:
        """프로젝트 라벨 정의 (ID·속성 ID 포함). `CvatSchema.from_labels`의 입력이다."""
        res = _check(
            self.http.get("/api/labels", params={"project_id": project_id, "page_size": 500})
        )
        return list(res["results"])

    def set_label_color(self, label_id: int, color: str) -> None:
        """라벨 하나의 화면 색(#rrggbb)을 바꾼다. 라벨 ID·이름·속성은 그대로다."""
        _check(self.http.patch(f"/api/labels/{label_id}", json={"color": color}))

    def create_task(self, name: str, project_id: int, video: Path) -> int:
        """영상 하나로 작업을 만들고 데이터 처리가 끝날 때까지 기다린다. 작업 ID.

        `sorting_method=predefined`로 프레임 순서를 영상 그대로 둔다 (프레임 번호 ↔ PTS 대응 유지).
        부작용: 영상 파일을 CVAT에 올린다 (블러 검수면 원본 프록시이므로 호출자가 먼저 감사 기록).
        """
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
        """CVAT 비동기 요청(rq_id)이 끝날 때까지 1초마다 확인한다.

        예외: 실패면 `ToolError`, timeout_s초 안에 끝나지 않으면 `ToolError`.
        """
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
        """작업 영상의 프레임 수 (CVAT 메타)."""
        return int(_check(self.http.get(f"/api/tasks/{task_id}/data/meta"))["size"])

    def put_tracks(self, task_id: int, tracks: list[dict[str, Any]]) -> None:
        """작업 주석을 트랙 목록으로 통째로 바꾼다 (모양·태그는 비운다)."""
        body = {"version": 0, "tags": [], "shapes": [], "tracks": tracks}
        _check(self.http.put(f"/api/tasks/{task_id}/annotations", json=body))

    def get_tracks(self, task_id: int) -> list[dict[str, Any]]:
        """트랙만. 수집은 get_annotations를 쓴다 (모양·태그도 봐야 한다)."""
        return list(self.get_annotations(task_id)["tracks"])

    def get_annotations(self, task_id: int) -> dict[str, Any]:
        """작업의 모든 주석: tracks(트랙 모드), shapes(모양 모드, CVAT 기본), tags(프레임 태그)."""
        return dict(_check(self.http.get(f"/api/tasks/{task_id}/annotations")))

    def find_user_id(self, username: str) -> int | None:
        """CVAT 사용자 이름 → 사용자 ID (정확히 같은 이름만)."""
        res = _check(self.http.get("/api/users", params={"search": username, "page_size": 100}))
        return next((int(u["id"]) for u in res["results"] if u["username"] == username), None)

    def is_non_admin_reviewer(self, user_id: int) -> bool:
        """활성 일반 계정인지 확인한다. 관리자 계정으로 원본/라벨러 권한을 분리할 수 없다."""
        user = _check(self.http.get(f"/api/users/{user_id}"))
        return (
            user.get("is_active") is True
            and user.get("is_staff") is False
            and user.get("is_superuser") is False
            and "admin" not in user.get("groups", ["admin"])
        )

    def jobs(self, task_id: int) -> list[dict[str, Any]]:
        """작업의 job 목록 (ID 순). 각 job에 start_frame·stop_frame·assignee가 있다."""
        res = _check(self.http.get("/api/jobs", params={"task_id": task_id, "page_size": 500}))
        return sorted(res["results"], key=lambda j: int(j["id"]))

    def job_ids(self, task_id: int) -> list[int]:
        """작업의 job ID 목록 (오름차순)."""
        return [int(j["id"]) for j in self.jobs(task_id)]

    def job_assignees(self, task_id: int) -> dict[int, str | None]:
        """작업(job) → 담당자 사용자 이름."""
        return {
            int(j["id"]): (j.get("assignee") or {}).get("username") or None
            for j in self.jobs(task_id)
        }

    def assign(self, task_id: int, user_id: int) -> None:
        """작업(task)과 그 모든 job의 담당자를 정한다.

        실제 열람 범위는 CVAT의 프로젝트·조직·사용자 권한에도 달려 있다.
        담당자 지정만으로 다른 작업의 접근 권한을 회수하지는 않는다.
        """
        _check(self.http.patch(f"/api/tasks/{task_id}", json={"assignee_id": user_id}))
        for job in self.job_ids(task_id):
            _check(self.http.patch(f"/api/jobs/{job}", json={"assignee": user_id}))

    def unassign(self, task_id: int) -> None:
        """작업(task)과 그 모든 job의 담당자를 비운다 (회수 전에 라벨러 화면에서 먼저 뺀다).

        주의: job 담당자 변경은 CVAT `update:job` 웹훅을 일으킨다. 웹훅 서버는 열린 작업만
        수집하므로 회수된 작업의 웹훅은 무시한다 (`dlp review serve`).
        """
        _check(self.http.patch(f"/api/tasks/{task_id}", json={"assignee_id": None}))
        for job in self.job_ids(task_id):
            _check(self.http.patch(f"/api/jobs/{job}", json={"assignee": None}))

    def task_exists(self, task_id: int) -> bool:
        """작업이 있으면 True, 404면 False. 그 밖의 실패는 `ToolError`."""
        return not _gone(self.http.get(f"/api/tasks/{task_id}"))

    def task_owner(self, task_id: int) -> TaskOwner:
        """작업 이름(`cvat_task_name`)에서 세션·스트림·담당자를 읽는다. 모양이 다르면 칸이 None.

        예외: 작업이 없거나(404 포함) 응답이 실패하면 `ToolError`.
        """
        name = str(_check(self.http.get(f"/api/tasks/{task_id}"))["name"])
        parts = name.split("/")
        if len(parts) != 3 or not all(parts):
            return TaskOwner(None, None, None, name)
        return TaskOwner(parts[0], parts[1], parts[2], name)

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        """지우기 전에 남길 작업 사본: 작업 메타 일부, 주석 전체, 프레임 수.

        영상·데이터 URL은 넣지 않는다 (라벨링 버킷의 JSON 사본에 매체를 다시 담지 않는다).
        프레임 시각은 넣지 않는다. 필요하면 원본 PTS 인덱스로 다시 만든다 (ADR 0039).
        """
        task = _check(self.http.get(f"/api/tasks/{task_id}"))
        meta = {
            k: task.get(k)
            for k in ("id", "name", "project_id", "status", "created_date", "updated_date")
        }
        meta["assignee"] = (task.get("assignee") or {}).get("username")
        return {
            "task": meta,
            "annotations": self.get_annotations(task_id),
            "frame_count": self.frame_count(task_id),
        }

    def delete_task(self, task_id: int) -> bool:
        """작업을 지운다 (CVAT가 올린 영상·주석도 함께 지운다). 지웠으면 True, 이미 없으면 False."""
        return not _gone(self.http.delete(f"/api/tasks/{task_id}"))

    def add_issue(self, job_id: int, frame: int, position: list[float], message: str) -> int:
        """검수자에게 보이는 이슈 (먼저 볼 구간 안내).

        frame: job 안의 프레임 번호(작업 전체 기준), position: 이슈 표시 위치 [x1, y1, x2, y2] 화소.
        반환: 이슈 ID.
        """
        body = {"job": job_id, "frame": frame, "position": position, "message": message}
        return int(_check(self.http.post("/api/issues", json=body))["id"])

    def add_webhook(self, project_id: int, url: str, secret: str) -> int:
        """프로젝트에 `update:job` 웹훅을 등록한다 (서명 비밀 secret). 웹훅 ID.

        CVAT는 본문 HMAC-SHA256을 X-Signature-256 헤더로 보낸다 (`webhook.parse_event`가 확인).
        """
        body = {
            "target_url": url, "type": "project", "project_id": project_id,
            "events": ["update:job"], "secret": secret, "content_type": "application/json",
            "enable_ssl": False, "is_active": True,
        }  # fmt: skip
        return int(_check(self.http.post("/api/webhooks", json=body))["id"])


class LabelStudioClient:
    """Label Studio REST 클라이언트 (토큰 인증)."""

    def __init__(self, base_url: str, token: str) -> None:
        """base_url의 Label Studio에 API 토큰으로 접속한다 (만들 때는 요청하지 않는다)."""
        self.http = httpx.Client(
            base_url=base_url, timeout=60, headers={"Authorization": f"Token {token}"}
        )
        # service_user_id 캐시
        self._user_id: int | None = None
        self._versioned: set[int] = set()  # 예측 버전을 맞춘 프로젝트

    @classmethod
    def from_env(cls) -> LabelStudioClient:
        """환경 변수(`DLP_LABEL_STUDIO_PORT`, `DLP_LABEL_STUDIO_TOKEN`)로 만든다."""
        return cls(
            f"http://localhost:{os.environ.get('DLP_LABEL_STUDIO_PORT', '8081')}",
            os.environ.get("DLP_LABEL_STUDIO_TOKEN", "dlpdevlabelstudiotoken0000000000000000"),
        )

    def find_project(self, title: str) -> dict[str, Any] | None:
        """제목이 정확히 같은 프로젝트 (없으면 None).

        Label Studio 버전에 따라 목록이 페이지 객체(`results`) 또는 배열로 와서 둘 다 받는다.
        """
        res = _check(self.http.get("/api/projects", params={"title": title, "page_size": 100}))
        items = res["results"] if isinstance(res, dict) else res
        return next((p for p in items if p["title"] == title), None)

    def create_project(self, title: str, label_config: str) -> int:
        """라벨 설정 XML(`labelstudio.label_config`)과 프리라벨 표시 설정으로 프로젝트를 만든다."""
        body = {"title": title, "label_config": label_config, **PRELABEL_SETTINGS}
        return int(_check(self.http.post("/api/projects", json=body))["id"])

    def ensure_label_config(self, project_id: int, label_config: str) -> None:
        """기존 프로젝트의 라벨 설정 XML이 지금 코드가 만드는 것과 다르면 바꾼다.

        설정을 바꾼 뒤(예: 이벤트 속성 입력란 추가) 이전에 만든 프로젝트에도 새 입력란이
        보이게 한다.
        Label Studio는 기존 주석이 쓰는 라벨이 새 설정에 없으면 거부한다(HTTP 400) — 라벨을 지우는
        변경은 새 프로젝트로 해야 한다.
        """
        project = _check(self.http.get(f"/api/projects/{project_id}"))
        if project.get("label_config") != label_config:
            body = {"label_config": label_config}
            _check(self.http.patch(f"/api/projects/{project_id}", json=body))

    def ensure_prelabel_settings(self, project_id: int) -> None:
        """예측(프리라벨)을 검수 화면에 미리 채워 보이게 하는 프로젝트 설정."""
        _check(self.http.patch(f"/api/projects/{project_id}", json=PRELABEL_SETTINGS))

    def service_user_id(self) -> int:
        """이 클라이언트(서비스 계정)의 Label Studio 사용자 ID."""
        if self._user_id is None:
            self._user_id = int(_check(self.http.get("/api/current-user/whoami"))["id"])
        return self._user_id

    def create_task(
        self, project_id: int, data: dict[str, Any], results: list[dict[str, Any]]
    ) -> int:
        """프리라벨은 주석(annotation)이 아니라 예측(prediction)으로 올린다.

        주석으로 올리면 서비스 계정이 만든 결과가 사람 검수 결과처럼 수집될 수 있다.

        인자: data(작업 데이터: 서명 URL 등),
        results(`labelstudio.to_ls_results` 결과, 비면 예측 없음).
        반환: 작업 ID.
        """
        task = _check(self.http.post("/api/tasks", json={"project": project_id, "data": data}))
        if results:
            body = {"task": task["id"], "result": results, "model_version": PRELABEL_VERSION}
            _check(self.http.post("/api/predictions", json=body))
            self._use_prelabel_version(project_id)
        return int(task["id"])

    def _use_prelabel_version(self, project_id: int) -> None:
        """프로젝트가 보여 줄 예측 버전을 프리라벨 버전으로 맞춘다 (예측이 생긴 뒤에만 가능)."""
        if project_id in self._versioned:
            return
        project = _check(self.http.get(f"/api/projects/{project_id}"))
        if project.get("model_version") != PRELABEL_VERSION:
            body = {"model_version": PRELABEL_VERSION}
            _check(self.http.patch(f"/api/projects/{project_id}", json=body))
        self._versioned.add(project_id)

    def prediction_results(self, task_id: int) -> list[dict[str, Any]]:
        """작업에 올린 프리라벨 (예측) 결과."""
        res = _check(self.http.get("/api/predictions", params={"task": task_id}))
        preds = res["results"] if isinstance(res, dict) else res
        # 필터가 무시되는 버전에 대비해 작업 ID를 다시 확인한다
        return [r for p in preds if int(p["task"]) == task_id for r in p["result"]]

    def latest_results(
        self, task_id: int, *, exclude_user: int | None = None
    ) -> list[dict[str, Any]] | None:
        """사람이 제출한 최신 주석의 결과. 제출한 주석이 없으면 None (빈 목록과 구별한다).

        취소(건너뛰기)한 주석과 exclude_user(예: 서비스 계정)가 만든 주석은 보지 않는다.
        """
        anns = _check(self.http.get(f"/api/tasks/{task_id}/annotations"))
        anns = [
            a
            for a in anns
            if not a.get("was_cancelled") and (exclude_user is None or _user(a) != exclude_user)
        ]
        if not anns:
            return None
        # ISO 시각 문자열 비교로 가장 최근 것 (갱신 시각이 없으면 생성 시각)
        latest = max(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        return list(latest["result"])

    def lead_seconds(self, task_id: int) -> float:
        """작업에 쓴 시간(초): 주석들의 lead_time 합 (Label Studio가 편집 화면에서 잰다)."""
        anns = _check(self.http.get(f"/api/tasks/{task_id}/annotations"))
        return float(sum(float(a.get("lead_time") or 0.0) for a in anns))

    def task_exists(self, task_id: int) -> bool:
        """작업이 있으면 True, 404면 False. 그 밖의 실패는 `ToolError`."""
        return not _gone(self.http.get(f"/api/tasks/{task_id}/"))

    def task_owner(self, task_id: int) -> TaskOwner:
        """작업 data의 `session_id`·`assignee` (`create_labeling_tasks`가 넣는다). 스트림은 None.

        서명 URL은 읽기만 하고 돌려주지 않는다. 예외: 작업이 없거나 응답이 실패하면 `ToolError`.
        """
        data = _check(self.http.get(f"/api/tasks/{task_id}/")).get("data") or {}
        sid, who = data.get("session_id"), data.get("assignee")
        return TaskOwner(
            str(sid) if sid is not None else None,
            None,
            str(who) if who is not None else None,
            f"session_id={sid}, assignee={who}",
        )

    def task_snapshot(self, task_id: int) -> dict[str, Any]:
        """지우기 전에 남길 작업 사본: 제출된 주석 전체와 임시 저장(drafts, 없으면 None).

        작업 data(라벨러용 서명 URL 등)는 넣지 않는다 (사본이 다시 매체를 열게 하지 않는다).
        """
        annotations = _check(self.http.get(f"/api/tasks/{task_id}/annotations"))
        task = _check(self.http.get(f"/api/tasks/{task_id}/"))
        return {
            "task": {"id": task.get("id")},
            "annotations": annotations,
            "drafts": task.get("drafts"),
        }

    def delete_task(self, task_id: int) -> bool:
        """작업을 지운다 (주석·예측 포함). 지웠으면 True, 이미 없으면 False."""
        return not _gone(self.http.delete(f"/api/tasks/{task_id}/"))

    def add_webhook(self, project_id: int, url: str, secret: str) -> int:
        """주석 생성·갱신 웹훅을 등록한다. 비밀은 X-DLP-Secret 헤더로 실려 온다. 웹훅 ID."""
        body = {
            "project": project_id, "url": url, "send_payload": True, "send_for_all_actions": False,
            "actions": ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"],
            "headers": {"X-DLP-Secret": secret}, "is_active": True,
        }  # fmt: skip
        return int(_check(self.http.post("/api/webhooks", json=body))["id"])
