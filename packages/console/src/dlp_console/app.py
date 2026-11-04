"""운영 콘솔 웹 앱 (FastAPI, ADR 0061).

`dlp console serve`(`dlp_console.server.serve`)가 `create_app`으로 앱을 만들어 uvicorn에 넘긴다.
브라우저 화면(`static/app.js`)이 아래 `/api/*` 경로를 부른다. 이 모듈은 요청을 검사하고, 다른 모듈
(`status`·`steps`·`global_steps`·`upload`·`preview`·`training`·`imports`)의 결과를 JSON으로 묶는다.
도메인 판정은 하지 않는다.

경로 목록:

- `GET /`: 화면 HTML. `/static/*`: 화면 JS·CSS (토큰 없이 받지만 데이터는 없다).
- `GET /api/overview`, `GET /api/sessions/{세션}`: 세션 목록·세션 상세 (읽기).
- `POST /api/sessions/{세션}/steps/{단계}`: 세션 단계 작업 시작.
- `GET /api/sessions/{세션}/preview`: 블러본 서명 URL과 겹쳐 그릴 트랙.
- `POST /api/upload`, `DELETE /api/uploads/{세션}`: 영상 올리기 작업 시작, 업로드 사본 지우기.
- `GET /api/jobs`, `GET /api/jobs/{작업}`: 작업 목록·작업 하나 (로그 꼬리 포함).
- `POST /api/global/{단계}`, `GET /api/training`: 세션 밖 단계 작업 시작, 재학습 화면 자료.
- `GET /api/pretrain`, `POST /api/pretrain/check`, `POST /api/pretrain/{가져오기}/confirm`,
  `POST /api/pretrain/{가져오기}/compare`: 외부 가중치 가져오기.

지키는 경계:

- 모든 `/api/*` 요청은 `X-Console-Token` 머리글이 시작 때 만든 토큰과 같아야 한다 (401). CSRF를
  막는 방법이다. 다른 사이트의 페이지는 이 머리글을 붙일 수 없고, 콘솔은 CORS를 열지 않는다.
- Host 머리글은 묶은 주소·localhost만 받는다 (400, DNS 재바인딩 방지, `host_allowed`).
- 올리기 경로(`/api/upload`, `/api/pretrain/check`)는 본문을 읽기 전에 Content-Length를 경로 전체
  상한(`body_limits`)과 비교한다. 넘으면 413, 길이가 없으면(청크 전송) 411이다 (F-COE-90).
- DB는 읽기만 한다. 파이프라인 쓰기는 모두 `dlp` 명령 작업(`JobRunner`)이다. 콘솔이 직접 쓰는
  것은 작업 디렉터리의 파일(업로드 사본, 세션별 화면 기억값, 가져오기 기록, 리포트 캐시)과, 가져오기
  확인 때 Git 밖 `pretrain_import.models_dir`(`data/models/` 아래)에 두는 가중치 사본뿐이다.
- 응답에는 원본 위치(스트림 uri·검수 매체 uri)를 싣지 않는다. 로그 꼬리는 `Redactor`로 가린다.
- 같은 잠금 키(`session:<세션>`, `global:<묶음>`)의 작업이 돌고 있으면 새 작업은 409다.
- 화면 값 오류(`ParamError`)는 400 JSON `{"detail": …}`으로 바꾼다.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import secrets
import shutil
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Iterable, Mapping
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
from dlp_console.settings import ConsoleConfig, ConsoleSettings
from dlp_console.status import (
    SessionState,
    label_counts,
    list_sessions,
    read_state,
    session_view,
    task_view,
    vlm_servers,
)
from dlp_console.steps import IDENT, STEPS_BY_ID, ParamError, StepContext, build_command, evaluate
from dlp_console.training import ReportCache, latest_weekly, policy_trainers, rounds
from dlp_console.upload import UploadForm, prepare_upload, remove_upload, save_stream
from dlp_media.storage import S3Store
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.rendering import assert_render_current
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
MIB = 1024 * 1024


class Services:
    """앱이 쓰는 외부 자원 묶음 (DB 엔진, S3 서명, 작업 실행기, 정책 읽기).

    경로 처리기는 DB 연결·S3 서명·작업 실행을 이 객체로 얻는다. 시험
    (`packages/console/tests/conftest.py` `FakeServices`)은 이 클래스를 상속해 DB·S3를 바꾼다.
    """

    def __init__(self, settings: ConsoleSettings) -> None:
        """시작 때 한 번 만든다. 작업 디렉터리 아래 `reports/`, `sessions/`를 만든다.

        `config/defaults.yaml`(버킷 이름)과 `config/policies/privacy.yaml`(렌더 게이트 정책)을 이때
        읽는다. 콘솔을 다시 시작해야 정책 변경이 렌더 판정에 반영된다. DB 엔진은 연결을 미리 열지
        않는다 (첫 요청 때 연결한다).
        """
        self.settings = settings
        root = settings.root
        self.buckets = load_config(root / "config" / "defaults.yaml").buckets
        # 블러본 렌더 게이트의 정책 (CLI와 같은 config/policies/privacy.yaml, 시작 때 한 번)
        self.privacy_policy = load_privacy_policy(root)
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
    def rendered(self, conn: sa.Connection, session_id: str, stream_id: str) -> bool:
        """블러본이 지금 승인된 블러 라벨·렌더 정책으로 렌더한 것인가 (ADR 0024 게이트).

        서명 URL은 이 게이트를 통과한 스트림에만 낸다. 무효로 덮어쓴 기록, 블러·렌더 정책이 바뀐
        뒤의 낡은 블러본, 렌더 밖에서 덮어쓴 블러본은 렌더 안 됨이다 (F-COE-85).
        """
        store = S3Store.labeler_from_env(self.buckets.labeling, public=False)
        try:
            assert_render_current(conn, store, session_id, stream_id, self.privacy_policy)
        except Exception:  # 게이트 불통과(RenderNotCurrentError)·저장소 장애: 단계 버튼이 안내한다
            return False
        return True

    def sign(self, key: str) -> str:
        """라벨링 버킷 키의 서명 GET URL (라벨러 자격 증명, 유효 `preview.signed_url_ttl_s`초).

        브라우저가 닿는 주소(`DLP_S3_PUBLIC_ENDPOINT`)로 서명한다. 라벨러 자격 증명은 원본 버킷을
        읽지 못하므로 잘못된 키를 서명해도 원본은 열리지 않는다. 부르는 쪽(`preview.blurred_url`)이
        먼저 렌더 게이트를 확인한다.
        """
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
        """DB 연결 하나 (읽기용, 커밋하지 않는다)."""
        with self.engine.connect() as conn:
            yield conn

    def privacy_reviewers(self) -> tuple[str, ...]:
        """검수 정책 `reviewers.privacy`(원본 접근 권한자) 목록. 정책을 못 읽으면 빈 값.

        `DLP_REVIEW_POLICY_FILE`이 있으면 그 파일을, 없으면 `config/policies/review.yaml`을 읽는다.
        요청마다 다시 읽는다. 블러 검수 담당자 선택지와 단계 활성 판정에 쓴다.
        """
        selected = self.settings.env.get("DLP_REVIEW_POLICY_FILE")
        try:
            policy = load_review_policy(self.settings.root, Path(selected) if selected else None)
        except Exception:  # 정책 파일 오류는 단계 실행(dlp review)이 자세히 알린다
            return ()
        return tuple(policy.reviewers.privacy)

    def scenes(self) -> tuple[str, ...]:
        """`prelabel.yaml scene.labels`의 장면 이름 (올리기·AI 라벨 단계의 선택지)."""
        data: Any = yaml.safe_load(
            (self.settings.root / "config" / "policies" / "prelabel.yaml").read_text("utf-8")
        )
        scene = cast(dict[str, Any], data.get("scene") or {})
        return tuple(str(x) for x in cast(list[Any], scene.get("labels") or []))

    def prefs(self, session_id: str) -> dict[str, Any]:
        """세션별 화면 기억값 (`<작업 디렉터리>/sessions/<세션>.json`, 없으면 {}).

        지금은 올릴 때 고른 장면(`scene`)만 담는다. AI 라벨 단계에서 장면을 비우면 이 값을 쓴다.
        DB에 넣지 않는 콘솔 편의 값이다.
        """
        path = self.prefs_dir / f"{session_id}.json"
        if not path.is_file():
            return {}
        data: dict[str, Any] = json.loads(path.read_text("utf-8"))
        return data

    def save_prefs(self, session_id: str, values: dict[str, Any]) -> None:
        """기억값에 `values`를 덮어 합쳐 쓴다 (있던 다른 키는 남긴다)."""
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
    """경로의 세션 ID를 검사하고 세션 상태를 읽는다. 모양이 틀리면 400, 없으면 404."""
    if not IDENT.match(session_id):
        raise HTTPException(400, "세션 ID가 올바르지 않습니다")
    try:
        return read_state(conn, session_id, services.rendered)
    except NoResultFound as exc:
        raise HTTPException(404, f"세션이 없습니다: {session_id}") from exc


def _context(services: Services, state: SessionState) -> StepContext:
    """단계 활성 판정 입력. 콘솔 작업 기록에서 이 세션의 성공한 단계 ID도 모은다."""
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
    """작업을 시작하고 작업 화면 값을 돌려준다. 같은 잠금 키의 작업이 돌고 있으면 409."""
    try:
        job = services.runner.submit(step, title, commands, **kw)
    except JobConflictError as exc:
        raise HTTPException(409, str(exc)) from exc
    return job_view(services, job)


def _theme(root: Path) -> dict[str, Any]:
    """검수 화면 테마(`config/review_ui.yaml`)의 이름·로고·색 모드 (CVAT 테마와 같은 값)."""
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
    """IPv4·IPv6 리터럴인가 (도메인 이름이면 False)."""
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


def body_limits(config: ConsoleConfig) -> dict[str, int]:
    """본문을 읽기 전에 막는 경로별 요청 전체 상한 (바이트, F-COE-90).

    상한은 그 경로 파일 상한의 합 + `request_body.multipart_overhead_kb`다. Starlette 1.3은 처리기
    전에 multipart 본문 전체를 임시 파일로 받고, 파일 부분의 크기를 막는 설정이 없다
    (`max_part_size`는 파일 아닌 폼 값만). 그래서 받는 동안의 임시 디스크는 이 상한으로만 지킨다.
    """
    overhead = config.request_body.multipart_overhead_kb * 1024
    pre = config.pretrain_import
    return {
        "/api/upload": config.upload.max_mb * MIB + overhead,
        "/api/pretrain/check": pre.max_mb * MIB + 2 * pre.manifest_max_kb * 1024 + overhead,
    }


def body_refusal(
    method: str, path: str, headers: Mapping[str, str], config: ConsoleConfig
) -> JSONResponse | None:
    """올리기 경로의 POST를 본문을 읽기 전에 거를 응답. 받을 요청이면 None (F-COE-90).

    서버(uvicorn)는 선언한 Content-Length보다 많이 읽지 않으므로 길이 검사가 받는 양의 상한이 된다.
    청크 전송(`Transfer-Encoding`)은 끝까지 받기 전에는 크기를 알 수 없어 411로 받지 않는다.
    headers는 소문자 이름으로 찾는 요청 머리글이다 (Starlette `Headers`).
    """
    limit = body_limits(config).get(path)
    if limit is None or method != "POST":
        return None
    length = headers.get("content-length")
    if "transfer-encoding" in headers or length is None:
        detail = "Content-Length가 필요합니다 (청크 전송은 받지 않습니다)"
        return JSONResponse({"detail": detail}, status_code=411)
    if not length.isascii() or not length.isdigit():
        return JSONResponse({"detail": "Content-Length 값이 잘못되었습니다"}, status_code=400)
    if int(length) > limit:
        detail = f"요청 본문이 상한({math.ceil(limit / MIB)} MB, 파일 상한 합 + 여유분)보다 큽니다"
        return JSONResponse({"detail": detail}, status_code=413)
    return None


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
        """앱 수명. 시작 때는 할 일이 없고, 끝날 때 돌고 있는 작업을 멈춘다."""
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
        """모든 요청의 문지기.

        1. Host 머리글이 허용 이름이 아니면 400 (`host_allowed`). 정적 파일도 같다.
        2. `/api/*`는 `X-Console-Token`이 토큰과 다르거나 없으면 401.
        3. 올리기 경로는 본문을 읽기 전에 길이를 본다 (411·413, `body_refusal`). 이 미들웨어는
           본문을 읽지 않으므로 거른 요청의 본문은 임시 파일로 받지 않는다.
        4. 통과한 응답에 캐시 금지(`no-store`), MIME 추측 금지, Referer 금지 머리글을 붙인다.
           서명 URL·작업 자료가 브라우저 캐시에 남지 않게 한다.
        """
        if not host_allowed(request.headers.get("host") or "", bind_host, extra):
            return JSONResponse({"detail": "허용하지 않은 Host"}, status_code=400)
        if request.url.path.startswith("/api/"):
            # 바이트로 비교한다: str의 compare_digest는 비ASCII 문자에 TypeError(500)를 낸다
            token = request.headers.get(TOKEN_HEADER, "").encode("utf-8")
            if not secrets.compare_digest(token, settings.token.encode("utf-8")):
                return JSONResponse({"detail": "콘솔 토큰이 없거나 틀립니다"}, status_code=401)
        refusal = body_refusal(request.method, request.url.path, request.headers, settings.config)
        if refusal is not None:
            return refusal
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ParamError)
    async def param_error(_request: Request, exc: ParamError) -> JSONResponse:
        """`ParamError`(화면 값 오류, 막힌 단계) → 400 `{"detail": 한국어 안내}`."""
        return JSONResponse({"detail": str(exc)}, status_code=400)

    _session_routes(app, svc)
    _job_routes(app, svc)
    _global_routes(app, svc)
    _pretrain_routes(app, svc)

    @app.get("/")
    def index() -> FileResponse:
        """`GET /`: 화면 HTML (데이터 없음, 토큰 검사 없음. 토큰은 주소 `#token=`으로 받는다)."""
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _session_routes(app: FastAPI, svc: Services) -> None:
    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        """`GET /api/overview`: 첫 화면 자료. 작업을 시작하지 않는다.

        DB에서 모든 세션의 짧은 값(최근 촬영 먼저)을 읽는다. 함께 싣는 것은 테마, 실행 중 작업,
        VLM 서버 설정 여부(주소는 빼고), 블러 검수 담당자 선택지, 장면 선택지, 라벨 검수 담당자
        기본값(`DLP_CONSOLE_LABELER`), 도메인 선택지다. 오류 응답은 401·400(문지기)뿐이다.
        """
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
        """`GET /api/sessions/{session_id}`: 세션 상세. 작업을 시작하지 않는다.

        DB에서 세션·탐지 세대·검수 작업·보류·사용 중지·라벨 수를 읽고, 영상 스트림마다 렌더
        게이트를 확인한다. 응답은 단계 목록(상태·막힌 이유), 검수 작업(도구 주소 포함, 매체 위치
        제외), 라벨 수, 렌더 여부, 기억값, 이 세션의 최근 작업 20개다.

        Raises:
            HTTPException: 세션 ID 모양이 틀리면 400, 세션이 없으면 404.
        """
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
        """`POST /api/sessions/{session_id}/steps/{step_id}`: 세션 단계 작업을 시작한다.

        본문은 화면 입력 칸 값의 JSON 객체다 (비어도 된다). `steps.build_command`가 단계 판정과
        값 검사를 한 뒤 `dlp` 명령을 만든다. AI 라벨(`prelabel`)에서 장면을 비우면 올릴 때 고른
        장면을 쓴다. 잠금 키는 `session:<세션>`이라 한 세션에는 작업이 하나씩만 돈다.

        Returns:
            시작한 작업의 화면 값 (`job_view`). 작업은 백그라운드에서 돈다.

        Raises:
            HTTPException: 세션 ID 400·세션 없음 404·같은 세션 작업 실행 중 409.
            ParamError: 본문 JSON 오류·모르는 단계·막힌(blocked·skipped) 단계·값 오류 (400).
        """
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
        """`GET /api/sessions/{session_id}/preview?stream=…`: 결과 보기 자료 (작업 시작 없음).

        렌더 게이트를 통과한 블러본의 서명 URL과, 그 스트림의 운영 라벨 박스·키포인트 트랙(신뢰도
        높은 순 `preview.max_tracks`개)을 돌려준다. 원본 버킷은 읽지 않는다.

        Raises:
            HTTPException: 세션 ID 400·세션 없음 404. 사용 중지·블러 승인 전·렌더 안 됨·영상
                스트림 아님이면 409 (`PreviewUnavailableError`).
        """
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
        """`POST /api/upload` (multipart): 영상 하나를 받아 준비·수집 작업을 시작한다.

        파일을 `<작업 디렉터리>/uploads/<세션>/`에 저장하고 매니페스트를 쓴 뒤
        `dlp media prepare` → `dlp ingest` 작업(잠금 `session:<세션>`)을 시작한다. 장면을 고르면
        세션 기억값에 남긴다. DB에는 직접 쓰지 않는다 (수집 명령이 쓴다).

        Raises:
            HTTPException: 같은 세션 작업이 이미 돌면 409 (파일을 받기 전에 확인한다).
            ParamError: 장면이 선택지에 없음, 확장자·크기 상한·폼 값 오류, 이미 올린 사본이
                있음 (400). 요청 전체 상한은 본문을 받기 전에 문지기가 본다 (411·413, F-COE-90).
        """
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
        """`DELETE /api/uploads/{session_id}`: 그 세션의 업로드 사본 디렉터리를 지운다.

        실패한 올리기를 다시 하려고 쓴다. 원본 버킷·DB는 건드리지 않는다. 응답 `removed`는
        디렉터리가 있었는지다. 세션 ID 모양 오류는 400, 그 세션 작업이 돌고 있으면 409.
        """
        # 실행 중 검사와 삭제가 같은 정규화 값을 쓴다 (올리기도 strip한 ID로 잠근다, F-COE-80)
        sid = _ident(session_id, "세션 ID")
        if svc.runner.running(f"session:{sid}") is not None:
            raise HTTPException(409, "작업이 끝난 뒤 지우세요")
        return {"removed": remove_upload(svc.settings.work_dir, sid)}


def _job_routes(app: FastAPI, svc: Services) -> None:
    @app.get("/api/jobs")
    def jobs(session: str | None = None) -> list[dict[str, Any]]:
        """`GET /api/jobs?session=…`: 작업 목록 (새것 먼저 50개, 세션을 주면 그 세션 것만).

        콘솔 작업 기록(`jobs.jsonl`)만 읽는다. 실패·중단 작업에는 마지막 오류 줄과 안내를 붙인다.
        """
        return [job_view(svc, j) for j in svc.runner.jobs(session)[:50]]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        """`GET /api/jobs/{job_id}`: 작업 하나와 가린 로그 꼬리·경고 줄. 없으면 404."""
        found = svc.runner.get(job_id)
        if found is None:
            raise HTTPException(404, "작업이 없습니다")
        return job_view(svc, found, tail=True)


def _global_routes(app: FastAPI, svc: Services) -> None:
    def gctx() -> GlobalContext:
        """요청마다 `training.yaml` 과제 목록을 다시 읽어 세션 밖 단계 입력을 만든다."""
        return GlobalContext(
            reports_dir=svc.reports_dir, train_tasks=tuple(policy_trainers(svc.settings.root))
        )

    @app.post("/api/global/{step_id}")
    async def run_global(step_id: str, request: Request) -> dict[str, Any]:
        """`POST /api/global/{step_id}`: 세션 밖 단계(데이터셋·재학습·배포 승인·평가·리포트) 시작.

        본문 JSON 객체를 `global_steps.build_global`이 `dlp` 명령으로 바꾼다. 잠금 키는
        `global:<묶음>`이다 (예: 재학습과 배포 승인은 `global:train`을 같이 쓴다).

        Raises:
            HTTPException: 같은 묶음 작업이 돌고 있으면 409.
            ParamError: 본문 JSON 오류·모르는 단계·값 오류 (400).
        """
        params = await _json_params(request)
        try:
            step, commands = build_global(gctx(), step_id, params)
        except ValueError as exc:
            raise ParamError(str(exc)) from exc
        return _submit(svc, step.title, step.id, commands, lock=f"global:{step.group}")

    @app.get("/api/training")
    def training() -> dict[str, Any]:
        """`GET /api/training`: 재학습·성과 화면 자료. 작업을 시작하지 않는다.

        DB `model_versions` 회차와 각 회차의 골든셋 평가 리포트(MLflow 산출물 버킷, 작업 디렉터리에
        캐시), 골든셋 버전 목록, 가장 최근 주간 지표 JSON, 세션 밖 작업 최근 20개를 돌려준다.
        리포트를 받지 못한 회차는 `report: null`이다 (오류로 바꾸지 않는다).
        """
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
        """`GET /api/pretrain`: 가져오기 화면 자료. 작업을 시작하지 않는다.

        확인한 가져오기 기록(새것 먼저), 비교 학습기 이름(`pretrain_import.compare.trainer`), 그
        학습기로 돈 objects 회차, 그중 게이트를 통과(passed·deployed)한 회차가 있는지를 돌려준다.
        """
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
        """`POST /api/pretrain/check` (multipart): 가중치 묶음 세 파일을 받아 검사한다.

        새 가져오기 디렉터리(`<작업 디렉터리>/pretrain/<가져오기 ID>/`)에 ONNX와 두 manifest를
        저장하고 `ImportStore.check`로 검사·적용 미리보기를 만든다. 작업을 시작하지 않고 설정
        파일도 고치지 않는다. 검사 실패도 200이며 응답의 `ok`·`checks`가 이유를 보인다.
        파일마다 크기 상한(ONNX `pretrain_import.max_mb`, manifest `manifest_max_kb`)을 두어
        작업 디렉터리 디스크를 지킨다 (F-COE-89). 요청 전체 상한은 본문을 받기 전에 문지기가
        본다 (411·413, F-COE-90).

        Raises:
            ParamError: 한 파일이라도 상한을 넘음 (400, 가져오기 디렉터리째 지운다).
        """
        cfg = svc.settings.config.pretrain_import
        limits = (cfg.max_mb * 1024 * 1024, cfg.manifest_max_kb * 1024, cfg.manifest_max_kb * 1024)
        import_id, path = svc.imports.new_dir()
        uploads = (onnx, model_manifest, data_manifest)
        for name, upload, limit in zip(FILES, uploads, limits, strict=True):
            try:
                save_stream(upload.file, path / name, limit)
            except ParamError as exc:
                shutil.rmtree(path, ignore_errors=True)
                raise ParamError(f"{name}: {exc}") from exc
        return svc.imports.check(import_id).public()

    @app.post("/api/pretrain/{import_id}/confirm")
    def confirm(import_id: str) -> dict[str, Any]:
        """`POST /api/pretrain/{import_id}/confirm`: 검사를 통과한 가져오기를 확인한다.

        가중치를 `pretrain_import.models_dir`에 복사(sha256 재확인)하고, 패치 파일과 가져오기 기록
        (`pretrain_imports.jsonl`, 추가만)을 남긴다. 기록의 actor는 `DLP_ACTOR`, 없으면 `USER`다.
        Git이 관리하는 설정 파일은 고치지 않고 운영자가 할 명령(`next`)을 돌려준다.

        Raises:
            ParamError: ID 모양 오류·모르는 가져오기·검사 결과 없음·검사 불통과·같은 자리에 다른
                파일·복사본 해시 불일치 (400).
        """
        actor = svc.settings.env.get("DLP_ACTOR") or os.environ.get("USER", "operator")
        return svc.imports.confirm(import_id, actor)

    @app.post("/api/pretrain/{import_id}/compare")
    async def run_compare(import_id: str, request: Request) -> dict[str, Any]:
        """`POST /api/pretrain/{import_id}/compare`: 가져온 가중치의 골든셋 비교 작업을 시작한다.

        본문 `dataset_version`(필수)과 `baseline_versions`(공백으로 나눈 모델 버전, 끝 `*` 허용)를
        받아 `dlp train run objects <버전> --trainer <비교 학습기> --param …`을 돌린다. `params`
        값의 `{model_name}`은 가져온 모델 이름으로 바뀐다. 잠금 키는 재학습과 같은 `global:train`.

        Raises:
            ParamError: 비교 학습기 미설정(F-COE-74)·검사 불통과·값 오류·본문 JSON 오류 (400).
            HTTPException: 재학습 작업이 이미 돌면 409.
        """
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
    """JSON 실수 해석기. 1e400처럼 무한대로 넘치는 값은 ParamError."""
    value = float(text)
    if not math.isfinite(value):
        raise ParamError(f"숫자가 범위를 벗어났습니다: {text}")
    return value


def _reject_constant(text: str) -> Any:
    """JSON 상수(NaN·Infinity·-Infinity)를 거부한다 (표준 JSON이 아니다)."""
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
    """앞뒤 공백을 떼고 `IDENT` 모양인지 본다. 아니면 ParamError(400)."""
    text = str(value or "").strip()
    if not IDENT.match(text):
        raise ParamError(f"{name} 값이 올바르지 않습니다")
    return text
