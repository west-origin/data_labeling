"""운영 콘솔 웹 앱 (FastAPI, ADR 0061).

- 정적 화면(`static/`)은 토큰 없이 받지만 데이터는 없다. 모든 `/api/*` 요청은 `X-Console-Token`
  머리글이 시작 때 만든 토큰과 같아야 한다 (CSRF: 다른 사이트는 이 머리글을 붙일 수 없고 CORS를 열지
  않는다). Host 머리글은 묶은 주소·localhost만 받는다 (DNS 재바인딩 방지).
- 쓰기는 모두 `dlp` 명령 작업(`JobRunner`)이다. DB는 읽기만 한다.
- 응답에는 원본 위치(스트림 uri·검수 매체 uri)를 싣지 않는다. 로그 꼬리는 `Redactor`로 가린다.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterable
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any, cast

import sqlalchemy as sa
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import NoResultFound

from dlp_console.global_steps import GlobalContext, build_global
from dlp_console.hints import last_error_line, next_action
from dlp_console.imports import FILES, ImportStore
from dlp_console.jobs import Job, JobConflictError, JobRunner, JobStatus
from dlp_console.preview import PreviewUnavailableError, blurred_url, overlay_tracks
from dlp_console.redact import Redactor
from dlp_console.settings import ConsoleSettings
from dlp_console.status import (
    SessionState,
    label_counts,
    list_sessions,
    read_state,
    render_record_key,
    session_view,
    task_view,
    vlm_servers,
)
from dlp_console.steps import IDENT, STEPS_BY_ID, ParamError, StepContext, build_command, evaluate
from dlp_console.training import ReportCache, latest_weekly, policy_trainers, rounds
from dlp_console.upload import UploadForm, prepare_upload, remove_upload
from dlp_media.storage import S3Store
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_review.ui_theme import load_ui_theme, logo_data_uri
from dlp_schema.config import load_config
from dlp_schema.db.repository import list_golden_sets

STATIC = Path(__file__).parent / "static"
TOKEN_HEADER = "x-console-token"
# Host 머리글로 늘 받는 이름 (이 컴퓨터) / 모든 주소에 묶는 값 (--allow-remote)
LOCAL_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
WILDCARD_BINDS = frozenset({"0.0.0.0", "::"})
# DLP_DATABASE_URL이 없을 때 (`dlp` CLI 기본값과 같은 개발 DB, .env.example)
DEV_DATABASE_URL = "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"


class Services:
    """앱이 쓰는 외부 자원 (시험은 가짜로 바꾼다)."""

    def __init__(self, settings: ConsoleSettings) -> None:
        self.settings = settings
        root = settings.root
        self.buckets = load_config(root / "config" / "defaults.yaml").buckets
        self.engine = sa.create_engine(settings.env.get("DLP_DATABASE_URL", DEV_DATABASE_URL))
        self.redactor = Redactor(raw_bucket=self.buckets.raw, env=settings.env)
        self.runner = JobRunner(settings, self.redactor)
        self.imports = ImportStore(
            root, settings.work_dir, settings.config.pretrain_import.models_dir
        )
        self.reports_dir = settings.work_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.prefs_dir = settings.work_dir / "sessions"
        self.prefs_dir.mkdir(parents=True, exist_ok=True)

    # 라벨링 버킷(블러본) 접근: 라벨러 자격 증명만 (읽기 전용)
    def rendered(self, session_id: str, stream_id: str) -> bool:
        store = S3Store.labeler_from_env(self.buckets.labeling, public=False)
        try:
            return store.head(render_record_key(session_id, stream_id)) is not None
        except Exception:  # 저장소 장애: 렌더 안 됨으로 보이고 단계 버튼이 안내한다
            return False

    def sign(self, key: str) -> str:
        store = S3Store.labeler_from_env(self.buckets.labeling, public=True)
        return store.presign(key, self.settings.config.preview.signed_url_ttl_s)

    def fetch_report(self, uri: str, dest: Path) -> None:
        """MLflow 산출물 버킷의 평가 리포트를 받는다 (원본 버킷 URI는 거부)."""
        prefix = f"s3://{self.buckets.mlflow}/"
        if not uri.startswith(prefix):
            raise ValueError("MLflow 산출물 버킷의 리포트만 읽는다")
        S3Store.from_env(self.buckets.mlflow).get_file(uri.removeprefix(prefix), dest)

    @contextmanager
    def connect(self) -> Generator[sa.Connection]:
        with self.engine.connect() as conn:
            yield conn

    def privacy_reviewers(self) -> tuple[str, ...]:
        selected = self.settings.env.get("DLP_REVIEW_POLICY_FILE")
        try:
            policy = load_review_policy(self.settings.root, Path(selected) if selected else None)
        except Exception:  # 정책 파일 오류는 단계 실행(dlp review)이 자세히 알린다
            return ()
        return tuple(policy.reviewers.privacy)

    def scenes(self) -> tuple[str, ...]:
        data: Any = yaml.safe_load(
            (self.settings.root / "config" / "policies" / "prelabel.yaml").read_text("utf-8")
        )
        scene = cast(dict[str, Any], data.get("scene") or {})
        return tuple(str(x) for x in cast(list[Any], scene.get("labels") or []))

    def prefs(self, session_id: str) -> dict[str, Any]:
        path = self.prefs_dir / f"{session_id}.json"
        if not path.is_file():
            return {}
        data: dict[str, Any] = json.loads(path.read_text("utf-8"))
        return data

    def save_prefs(self, session_id: str, values: dict[str, Any]) -> None:
        path = self.prefs_dir / f"{session_id}.json"
        path.write_text(
            json.dumps({**self.prefs(session_id), **values}, ensure_ascii=False), "utf-8"
        )


def job_view(services: Services, job: Job, *, tail: bool = False) -> dict[str, Any]:
    """작업 화면 값. 실패면 마지막 오류 줄과 다음 할 일 안내를 붙인다."""
    data = job.public()
    lines = services.runner.tail(job.job_id) if tail or job.status is JobStatus.FAILED else []
    if tail:
        data["log"] = lines
        # 막지 않는 촬영 점검 경고 등 (dlp_media.capture_check, `경고[코드]: …` 줄)
        data["warnings"] = sorted({x for x in lines if x.startswith("경고[")})
    if job.status in (JobStatus.FAILED, JobStatus.INTERRUPTED):
        step = STEPS_BY_ID.get(job.step)
        default = (
            step.help
            if step
            else "로그를 확인한 뒤 다시 누르세요 (단계 명령은 다시 눌러도 안전합니다)."
        )
        if job.status is JobStatus.INTERRUPTED:
            default = "콘솔이 꺼져 결과를 모릅니다. 상태를 확인하고 다시 누르세요 (멱등)."
        data["error_line"] = last_error_line(lines)
        data["next_action"] = next_action(lines, default)
    return data


def _state(services: Services, conn: sa.Connection, session_id: str) -> SessionState:
    if not IDENT.match(session_id):
        raise HTTPException(400, "세션 ID가 올바르지 않습니다")
    try:
        return read_state(conn, session_id, services.rendered)
    except NoResultFound as exc:
        raise HTTPException(404, f"세션이 없습니다: {session_id}") from exc


def _context(services: Services, state: SessionState) -> StepContext:
    sid = state.session.session_id
    done = frozenset(j.step for j in services.runner.jobs(sid) if j.status is JobStatus.SUCCEEDED)
    return StepContext(
        state=state,
        vlm=vlm_servers(services.settings.env),
        privacy_reviewers=services.privacy_reviewers(),
        scenes=services.scenes(),
        succeeded=done,
    )


def _submit(
    services: Services, title: str, step: str, commands: list[list[str]], **kw: Any
) -> dict[str, Any]:
    try:
        job = services.runner.submit(step, title, commands, **kw)
    except JobConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    return job_view(services, job)


def _theme(root: Path) -> dict[str, Any]:
    theme = load_ui_theme(root)
    return {
        "name": theme.brand.name,
        "logo": logo_data_uri(theme, root),
        "default_mode": theme.default_mode,
        "modes": theme.modes.model_dump(),
    }


def host_name(header: str) -> str:
    """Host 머리글 → 이름 (포트를 떼고 소문자, IPv6 `[::1]:8090` → `::1`, IP는 표준형)."""
    text = header.strip().lower()
    if text.startswith("["):
        name = text[1 : text.find("]")] if "]" in text else ""
    elif text.count(":") == 1:
        name = text.split(":", 1)[0]
    else:
        name = text
    try:
        return str(ipaddress.ip_address(name))
    except ValueError:
        return name


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def host_allowed(header: str, bind_host: str, extra: frozenset[str] = frozenset()) -> bool:
    """Host 머리글 검사 (ADR 0061 4절: 묶은 주소·localhost만, DNS 재바인딩 방지).

    - localhost·127.0.0.1·::1과 묶은 주소(IP는 표준형으로 비교)를 받는다.
    - 모든 주소(0.0.0.0·::)에 묶었으면(`--allow-remote`) 묶은 주소는 이 컴퓨터의 어느 주소든이라
      IP 리터럴 Host를 모두 받는다. 도메인 이름 Host는 받지 않는다 (재바인딩 공격은 공격자 도메인
      이름으로 온다. IP 리터럴 Host는 그 IP를 주소창에 연 같은 출처뿐이다).
    - `extra`: 앱을 만들 때 준 이름 (시험 도구의 `testserver` 등, 운영 서버는 비움).
    """
    name = host_name(header)
    if not name:
        return False
    bound = host_name(bind_host)
    if name in LOCAL_NAMES or name == bound or name in extra:
        return True
    return bound in WILDCARD_BINDS and _is_ip(name)


def create_app(
    settings: ConsoleSettings,
    services: Services | None = None,
    bind_host: str = "127.0.0.1",
    extra_hosts: Iterable[str] = (),
) -> FastAPI:
    """콘솔 앱을 만든다. services를 주지 않으면 실제 DB·S3를 쓴다.

    extra_hosts: Host 검사에서 더 받을 이름. 시험 도구(TestClient `testserver`)만 쓴다. 운영
    서버(`server.serve`)는 주지 않는다 (F-COE-81).
    """
    svc = services or Services(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        yield
        # 콘솔 종료(SIGTERM·Ctrl-C → uvicorn 종료): 자식 dlp 그룹을 멈추고 interrupted로 남긴다
        svc.runner.shutdown()

    app = FastAPI(
        title="dlp 운영 콘솔", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.services = svc
    extra = frozenset(host_name(h) for h in extra_hosts)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not host_allowed(request.headers.get("host") or "", bind_host, extra):
            return JSONResponse({"detail": "허용하지 않은 Host"}, status_code=400)
        if request.url.path.startswith("/api/"):
            # 바이트로 비교한다: str의 compare_digest는 비ASCII 문자에 TypeError(500)를 낸다
            token = request.headers.get(TOKEN_HEADER, "").encode("utf-8")
            if not secrets.compare_digest(token, settings.token.encode("utf-8")):
                return JSONResponse({"detail": "콘솔 토큰이 없거나 틀립니다"}, status_code=401)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ParamError)
    async def param_error(_request: Request, exc: ParamError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    _session_routes(app, svc)
    _job_routes(app, svc)
    _global_routes(app, svc)
    _pretrain_routes(app, svc)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _session_routes(app: FastAPI, svc: Services) -> None:
    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        with svc.connect() as conn:
            sessions = list_sessions(conn)
        running = [job_view(svc, j) for j in svc.runner.jobs() if j.status is JobStatus.RUNNING]
        return {
            "theme": _theme(svc.settings.root),
            "sessions": sessions,
            "running": running,
            "vlm": vlm_servers(svc.settings.env),
            "privacy_reviewers": list(svc.privacy_reviewers()),
            "scenes": list(svc.scenes()),
            "labeler": svc.settings.env.get("DLP_CONSOLE_LABELER", ""),
            "domains": ["cleaning", "caregiving", "nursing"],
        }

    @app.get("/api/sessions/{session_id}")
    def session(session_id: str) -> dict[str, Any]:
        with svc.connect() as conn:
            state = _state(svc, conn, session_id)
            counts = label_counts(conn, session_id)
        ctx = _context(svc, state)
        env = svc.settings.env
        return {
            "session": session_view(state.session),
            "steps": evaluate(ctx),
            "tasks": [task_view(t, env) for t in state.tasks],
            "labels": counts,
            "rendered": state.rendered,
            "held": state.held,
            "withdrawn": state.withdrawn,
            "prefs": svc.prefs(session_id),
            "jobs": [job_view(svc, j) for j in svc.runner.jobs(session_id)[:20]],
        }

    @app.post("/api/sessions/{session_id}/steps/{step_id}")
    async def run_step(session_id: str, step_id: str, request: Request) -> dict[str, Any]:
        params = await _json_params(request)
        with svc.connect() as conn:
            state = _state(svc, conn, session_id)
        if step_id == "prelabel" and not params.get("scene"):
            params["scene"] = svc.prefs(session_id).get("scene")
        step, commands = build_command(_context(svc, state), step_id, params)
        return _submit(
            svc, step.title, step.id, commands, lock=f"session:{session_id}", session_id=session_id
        )

    @app.get("/api/sessions/{session_id}/preview")
    def preview(session_id: str, stream: str = "bodycam") -> dict[str, Any]:
        with svc.connect() as conn:
            state = _state(svc, conn, session_id)
            try:
                url = blurred_url(state, stream, svc.sign)
            except PreviewUnavailableError as exc:
                raise HTTPException(409, str(exc)) from exc
            tracks = overlay_tracks(
                conn, session_id, stream, svc.settings.config.preview.max_tracks
            )
        return {"url": url, **tracks}

    @app.post("/api/upload")
    def upload(
        file: Annotated[UploadFile, File()],
        session_id: Annotated[str, Form()],
        worker_id: Annotated[str, Form()],
        site_id: Annotated[str, Form()],
        domain: Annotated[str, Form()],
        consent_version: Annotated[str, Form()],
        recorded_at: Annotated[str, Form()] = "",
        camera_id: Annotated[str, Form()] = "",
        scene: Annotated[str, Form()] = "",
        display_rotation: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        form = UploadForm(
            session_id, worker_id, site_id, domain, consent_version, recorded_at, camera_id, scene,
            display_rotation,
        )  # fmt: skip
        cfg = svc.settings.config.upload
        if scene.strip() and scene.strip() not in svc.scenes():
            raise ParamError(f"장면은 {', '.join(svc.scenes())} 중 하나입니다")
        lock = f"session:{session_id.strip()}"
        if svc.runner.running(lock) is not None:
            raise HTTPException(409, "이 세션의 작업이 이미 돌고 있습니다")
        prepared = prepare_upload(
            svc.settings.work_dir, form, file.filename or "", file.file,
            max_mb=cfg.max_mb, suffixes=cfg.suffixes, delete_after_ingest=cfg.delete_after_ingest,
        )  # fmt: skip
        if prepared.scene:
            svc.save_prefs(prepared.session_id, {"scene": prepared.scene})
        return _submit(
            svc, "영상 준비·수집", "ingest", prepared.commands,
            lock=lock, session_id=prepared.session_id, cleanup=prepared.cleanup,
        )  # fmt: skip

    @app.delete("/api/uploads/{session_id}")
    def delete_upload(session_id: str) -> dict[str, Any]:
        # 실행 중 검사와 삭제가 같은 정규화 값을 쓴다 (올리기도 strip한 ID로 잠근다, F-COE-80)
        sid = _ident(session_id, "세션 ID")
        if svc.runner.running(f"session:{sid}") is not None:
            raise HTTPException(409, "작업이 끝난 뒤 지우세요")
        return {"removed": remove_upload(svc.settings.work_dir, sid)}


def _job_routes(app: FastAPI, svc: Services) -> None:
    @app.get("/api/jobs")
    def jobs(session: str | None = None) -> list[dict[str, Any]]:
        return [job_view(svc, j) for j in svc.runner.jobs(session)[:50]]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        found = svc.runner.get(job_id)
        if found is None:
            raise HTTPException(404, "작업이 없습니다")
        return job_view(svc, found, tail=True)


def _global_routes(app: FastAPI, svc: Services) -> None:
    def gctx() -> GlobalContext:
        return GlobalContext(
            reports_dir=svc.reports_dir, train_tasks=tuple(policy_trainers(svc.settings.root))
        )

    @app.post("/api/global/{step_id}")
    async def run_global(step_id: str, request: Request) -> dict[str, Any]:
        params = await _json_params(request)
        try:
            step, commands = build_global(gctx(), step_id, params)
        except ValueError as exc:
            raise ParamError(str(exc)) from exc
        return _submit(svc, step.title, step.id, commands, lock=f"global:{step.group}")

    @app.get("/api/training")
    def training() -> dict[str, Any]:
        trainers = policy_trainers(svc.settings.root)
        cache = ReportCache(svc.settings.work_dir, svc.fetch_report)
        with svc.connect() as conn:
            items = rounds(conn, cache, trainers)
            golden = [g.version for g in list_golden_sets(conn)]
        return {
            "rounds": items,
            "trainers": trainers,
            "golden_sets": golden,
            "weekly": latest_weekly(svc.reports_dir),
            "jobs": [job_view(svc, j) for j in svc.runner.jobs() if j.session_id is None][:20],
        }


def _pretrain_routes(app: FastAPI, svc: Services) -> None:
    compare = svc.settings.config.pretrain_import.compare

    @app.get("/api/pretrain")
    def pretrain() -> dict[str, Any]:
        trainers = policy_trainers(svc.settings.root)
        cache = ReportCache(svc.settings.work_dir, svc.fetch_report)
        with svc.connect() as conn:
            items = [
                r for r in rounds(conn, cache, trainers)
                if r["task"] == "objects" and compare.trainer and r["trainer"] == compare.trainer
            ]  # fmt: skip
        passed = any(r["status"] in ("passed", "deployed") for r in items)
        return {
            "imports": svc.imports.records(),
            "compare_trainer": compare.trainer,
            "compare_rounds": items,
            "gate_passed": passed,
        }

    @app.post("/api/pretrain/check")
    def check(
        onnx: Annotated[UploadFile, File()],
        model_manifest: Annotated[UploadFile, File()],
        data_manifest: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        import_id, path = svc.imports.new_dir()
        for name, upload in zip(FILES, (onnx, model_manifest, data_manifest), strict=True):
            with (path / name).open("wb") as f:
                while chunk := upload.file.read(1024 * 1024):
                    f.write(chunk)
        return svc.imports.check(import_id).public()

    @app.post("/api/pretrain/{import_id}/confirm")
    def confirm(import_id: str) -> dict[str, Any]:
        actor = svc.settings.env.get("DLP_ACTOR") or os.environ.get("USER", "operator")
        return svc.imports.confirm(import_id, actor)

    @app.post("/api/pretrain/{import_id}/compare")
    async def run_compare(import_id: str, request: Request) -> dict[str, Any]:
        params = await _json_params(request)
        if not compare.trainer:
            raise ParamError(
                "골든셋 비교 학습기가 아직 없습니다 "
                "(config/console.yaml pretrain_import.compare.trainer, F-COE-74)"
            )
        plan = svc.imports.plan(import_id)
        if not plan.get("ok"):
            raise ParamError("검사를 통과한 가져오기만 비교합니다")
        cmd = ["train", "run", "objects", _ident(params.get("dataset_version"), "데이터셋 버전")]
        cmd += ["--trainer", _ident(compare.trainer, "학습기")]
        for key, value in compare.params.items():
            cmd += [
                "--param",
                f"{_ident(key, '파라미터')}="
                f"{_ident(value.format(model_name=plan['model_name']), '값')}",
            ]
        for item in str(params.get("baseline_versions") or "").split():
            cmd += [
                "--baseline-version",
                item if IDENT.match(item.removesuffix("*")) else _ident(item, "기존 버전"),
            ]
        return _submit(
            svc, "가져온 가중치 골든셋 비교", "pretrain_compare", [cmd], lock="global:train"
        )


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ParamError(f"숫자가 범위를 벗어났습니다: {text}")
    return value


def _reject_constant(text: str) -> Any:
    raise ParamError(f"숫자가 아닙니다: {text}")


async def _json_params(request: Request) -> dict[str, Any]:
    """요청 본문 JSON 객체 → 단계 값 (비면 {}). 잘못된 JSON·객체가 아님·NaN·무한대(1e400)는
    ParamError(400)다 (F-COE-84)."""
    body = await request.body()
    if not body:
        return {}
    try:
        data: Any = json.loads(body, parse_float=_finite_float, parse_constant=_reject_constant)
    except ParamError:
        raise
    except (ValueError, UnicodeDecodeError) as exc:
        raise ParamError("요청 본문이 올바른 JSON이 아닙니다") from exc
    if not isinstance(data, dict):
        raise ParamError("요청 본문은 JSON 객체여야 합니다")
    return cast(dict[str, Any], data)


def _ident(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not IDENT.match(text):
        raise ParamError(f"{name} 값이 올바르지 않습니다")
    return text
