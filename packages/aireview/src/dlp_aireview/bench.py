"""처리량 벤치 `dlp aireview bench` (WP17.3, ADR 0040 5.6·8.3). DB에 쓰지 않는다.

장비 1대(엔드포인트 하나)가 판정 요청을 얼마나 처리하는지 실제 계획기·프롬프트·스키마·재검사
(`judge_unit`)로 잰다. 결과(지속 단위/s, 프리필·생성 tok/s)는 `dlp aireview estimate --bench`가 장비
대수 공식에 쓴다.

재는 방법
- 설정 = 묶음(temporal·temporal_long·spatial), 프레임 수(temporal만, `temporal.frames`), 긴 변
  (그 묶음의 `max_side_px`)의 조합. 설정마다 정책 사본(`bench_policy`)으로 계획(`plan_session`)하고
  그 묶음 단위만 쓴다. 입력은 `BenchInput`(세션·라벨·PTS 인덱스·프레임 공급 함수)이다. 합성 입력은
  CLI가 `dlp_fixtures.aireview`로 만든다 (영상·사람·글자 없음).
- 설정마다: 준비 요청(`warmup`, 측정 안 함) → 첫 토큰 시간(`max_tokens=1`, 동시성 1) → 동시성
  단계마다 같은 단위 집합 `requests`개를 판정한다. 요청마다 고정 블록 끝에 측정 표식 한 줄을 넣어
  (`_Marked`) 고정 접두사(system + 온톨로지 블록)는 캐시에 맞고 이미지·가변 글은 매번 새로 계산하게
  한다 (실제 운영에서 단위는 늘 새 입력이다. 같은 창 분할 요청의 이미지 공유는 빠지므로 보수적이다).
- 점(동시성 하나)마다: 단위/s, p50·p95 지연(단위 하나의 재시도·분할 포함 왕복), 프리필 tok/s
  (= (prompt - cached) 합 / 벽시계), 생성 tok/s, 캐시 적중 비율, 스키마 적합률(위반 없는 응답 /
  받은 응답), 재시도율, fallback율, 서버 장애 수.
- 무릎 지점(`knee`): 서버 장애 0이고 p95가 정책 `server.timeout_s`보다 작은 점 중 단위/s가 가장 큰
  점 (같으면 낮은 동시성). 그 동시성이 `server.concurrency_per_endpoint` 후보다. 동시성은 httpx 연결
  풀 기본 상한(100) 이하로 준다 (넘으면 클라이언트 쪽에서 기다린 시간이 지연에 섞인다).
- `cache_probe`: 같은 요청을 그대로 다시 보낸 지연·적중 토큰과, system 맨 앞에 난수 한 줄을 넣어
  캐시를 모두 깬 요청의 지연·적중 토큰을 비교한다.
- 서버 장애(`VlmUnavailableError`)는 그 점의 오류로 센다 (재시도하지 않는다. 과부하 503도 오류다).
  한 점의 모든 단위가 장애면 서버가 멈춘 것으로 보고 벤치를 멈춘다(`server_down`). 설정 오류
  (`VlmRequestError`: 4xx, 서버 모델 불일치)는 그대로 올린다 (CLI 종료 5).

보고서(`BenchReport`)는 JSON으로 `reports/aireview-bench/<장비>-<시각>.json`(Git 제외)에 남긴다.
`/v1/models`, 서버 모델, 엔진 버전, 정책 fingerprint, 장비 이름을 함께 담는다. 디코드
fps(`decode`)는 CLI가 잰다 (이 패키지는 영상을 열지 않는다, 블러본 경계는 `media.py`).
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime

from dlp_aireview.backend import BackendInfo, JudgeBackend, JudgeReply, JudgeRequest
from dlp_aireview.judge import JudgedUnit, judge_unit, request_for
from dlp_aireview.overlay import render_unit_images
from dlp_aireview.plan import PlannedUnit, StreamMedia, plan_session
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import ServerInfo
from dlp_aireview.prompt import PromptImage, build_parts
from dlp_aireview.schema import response_schema
from dlp_aireview.sensors import SensorContext
from dlp_models.vlm import VlmUnavailableError
from dlp_schema.aireview import AiGroup
from dlp_schema.common import Contract
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.session import Session

Image = NDArray[np.uint8]
# 프레임 공급 함수: (스트림, 프레임 번호들) → 원 해상도 RGB 프레임 (번호 순)
FrameSource = Callable[[str, Sequence[int]], Sequence[Image]]


# ---------------------------------------------------------------- 입력·설정


@dataclass(frozen=True)
class BenchInput:
    """벤치 입력 (실제 계획기에 넣는 세션 하나).

    Attributes:
        session: 세션. labels: 현재 운영 라벨. media: 스트림 → PTS 인덱스·렌더 해시.
        ontology: 온톨로지. sensor: 장갑 접촉 구간. frames: 프레임 공급 함수.
        source: 입력 설명 (`synthetic` 등, 보고서에 남긴다).
    """

    session: Session
    labels: tuple[LabelRecord, ...]
    media: Mapping[str, StreamMedia]
    ontology: Ontology
    sensor: SensorContext
    frames: FrameSource
    source: str


@dataclass(frozen=True)
class BenchGrid:
    """잴 설정 격자 (CLI 인자).

    Attributes:
        concurrency: 동시 요청 단계. frames: temporal 프레임 수 후보. sides: 긴 변 후보.
        groups: 잴 묶음. requests: 점마다 판정할 단위 수 (계획 단위를 돌려 쓴다).
        warmup: 설정마다 측정 전 준비 단위 수. ttft: 첫 토큰 시간 표본 수.
        cache_probe: 캐시 비교를 할지.
    """

    concurrency: tuple[int, ...]
    frames: tuple[int, ...]
    sides: tuple[int, ...]
    groups: tuple[AiGroup, ...]
    requests: int
    warmup: int
    ttft: int
    cache_probe: bool


# ---------------------------------------------------------------- 보고서


class BenchPoint(Contract):
    """동시성 하나의 측정값."""

    concurrency: int
    units: int
    ok: int
    unavailable: int
    wall_s: float
    units_per_s: float
    latency_p50_ms: int | None = None
    latency_p95_ms: int | None = None
    requests: int = 0
    prompt_tokens: int = 0
    cached_tokens: int | None = None
    completion_tokens: int = 0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    cache_hit_ratio: float | None = None
    schema_ok_ratio: float | None = None
    retry_ratio: float | None = None
    fallback_ratio: float | None = None
    context_errors: int = 0


class CacheProbe(Contract):
    """캐시 비교 (표본 중앙값). warm = 같은 요청 다시, cold = 맨 앞 난수로 캐시를 깬 요청."""

    samples: int
    warm_latency_ms: int | None = None
    cold_latency_ms: int | None = None
    warm_cached_tokens: int | None = None
    cold_cached_tokens: int | None = None
    prompt_tokens: int | None = None


class BenchConfig(Contract):
    """설정 하나(묶음, 프레임 수, 긴 변 조합)의 결과."""

    group: AiGroup
    frames: int
    side: int
    planned_units: int
    mean_targets: float = 0.0
    mean_images: float = 0.0
    ttft_p50_ms: int | None = None
    single_prefill_tok_s: float | None = None
    cache: CacheProbe | None = None
    points: tuple[BenchPoint, ...] = ()
    knee: BenchPoint | None = None
    stopped: str | None = None


class DecodeMeasure(Contract):
    """블러본 순차 디코드 속도 (CLI가 잰다)."""

    fps: float
    frames: int
    width: int
    height: int
    max_side: int
    method: Literal["cpu"] = "cpu"


class BenchReport(Contract):
    """`dlp aireview bench` 보고서."""

    kind: Literal["aireview-bench"] = "aireview-bench"
    created_at: AwareDatetime
    device: str
    source: str
    server: ServerInfo
    policy: dict[str, Any]
    grid: dict[str, Any]
    configs: tuple[BenchConfig, ...]
    decode: DecodeMeasure | None = None
    server_down: bool = False

    def config(self, group: AiGroup, frames: int, side: int) -> BenchConfig | None:
        """그 묶음·프레임 수·긴 변의 설정 결과 (없으면 None)."""
        return next(
            (c for c in self.configs if (c.group, c.frames, c.side) == (group, frames, side)),
            None,
        )


# ---------------------------------------------------------------- 정책 사본


def group_settings(policy: AiReviewPolicy, group: AiGroup) -> tuple[int, int]:
    """그 묶음이 정책에서 쓰는 (프레임 수, 긴 변) (벤치 설정과 추정이 맞춰 보는 값)."""
    if group is AiGroup.TEMPORAL:
        return policy.temporal.frames, policy.temporal.max_side_px
    if group is AiGroup.TEMPORAL_LONG:
        return policy.temporal.long_frames, policy.temporal.long_max_side_px
    return policy.spatial.frames, policy.spatial.max_side_px


def bench_policy(
    policy: AiReviewPolicy, group: AiGroup, frames: int | None, side: int
) -> AiReviewPolicy:
    """벤치 설정의 정책 사본 (검증을 다시 거친다).

    temporal: `temporal.frames = frames`(필요하면 `max_frames`를 올린다), `max_side_px = side`.
    temporal_long: `long_max_side_px = side`. spatial: `spatial.max_side_px = side`.
    프레임 수는 temporal에만 적용한다 (나머지 묶음은 정책 값).
    """
    data = policy.model_dump()
    t, s = data["temporal"], data["spatial"]
    if group is AiGroup.TEMPORAL:
        if frames is not None:
            t["frames"] = frames
            t["max_frames"] = max(t["max_frames"], frames)
        t["max_side_px"] = side
    elif group is AiGroup.TEMPORAL_LONG:
        t["long_max_side_px"] = side
    else:
        s["max_side_px"] = side
    return AiReviewPolicy.model_validate(data)


def _settings(
    policy: AiReviewPolicy, grid: BenchGrid
) -> Iterator[tuple[AiGroup, int, int, AiReviewPolicy]]:
    """(묶음, 그 묶음 프레임 수, 긴 변, 정책 사본) — 프레임 수 후보는 temporal에만."""
    for group in grid.groups:
        options: tuple[int | None, ...] = grid.frames if group is AiGroup.TEMPORAL else (None,)
        for frames in options:
            for side in grid.sides:
                p = bench_policy(policy, group, frames, side)
                yield group, group_settings(p, group)[0], side, p


# ---------------------------------------------------------------- 요청 꾸미기


class _Marked:
    """요청마다 고정 블록 끝에 측정 표식을 넣는 백엔드 감싸기 (모듈 docstring)."""

    def __init__(self, inner: JudgeBackend, mark: str, *, break_cache: bool = False) -> None:
        """mark: 표식 문자열. break_cache: system 맨 앞에 넣어 접두사 캐시를 모두 깬다."""
        self.inner = inner
        self.mark = mark
        self.break_cache = break_cache
        self.info: BackendInfo = inner.info

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """표식을 넣어 보낸다."""
        parts = request.parts
        if self.break_cache:
            parts = replace(parts, system=f"(측정 {self.mark})\n{parts.system}")
        else:
            parts = replace(parts, ontology=f"{parts.ontology}\n(측정 {self.mark})")
        return self.inner.judge(replace(request, parts=parts))


@dataclass(frozen=True)
class _Case:
    """준비한 단위 하나 (계획 단위 + 보낼 이미지)."""

    unit: PlannedUnit
    images: tuple[PromptImage, ...]


@dataclass(frozen=True)
class _Ctx:
    """설정 하나의 실행 문맥."""

    backend: JudgeBackend
    data: BenchInput
    policy: AiReviewPolicy
    token: str
    clock: Callable[[], float]


def _full_request(ctx: _Ctx, case: _Case, max_tokens: int | None = None) -> JudgeRequest:
    """대상 전부를 한 번에 묻는 판정 요청 (judge_unit의 첫 요청과 같은 모양)."""
    unit, o, policy = case.unit, ctx.data.ontology, ctx.policy
    positions = tuple(range(len(unit.targets)))
    parts = build_parts(unit, case.images, positions, o, policy, ctx.data.sensor)
    schema = response_schema(o, unit.group, policy)
    return request_for(unit, parts, positions, schema=schema, policy=policy, max_tokens=max_tokens)


# ---------------------------------------------------------------- 측정


@dataclass(frozen=True)
class _Outcome:
    """단위 하나의 결과 (장애면 judged None)."""

    judged: JudgedUnit | None
    seconds: float


def _one(ctx: _Ctx, case: _Case, mark: str) -> _Outcome:
    """단위 하나를 판정한다 (장애는 결과로, 설정 오류는 올린다)."""
    start = ctx.clock()
    try:
        judged = judge_unit(
            _Marked(ctx.backend, mark),
            case.unit,
            case.images,
            ontology=ctx.data.ontology,
            policy=ctx.policy,
            sensor=ctx.data.sensor,
        )
    except VlmUnavailableError:
        return _Outcome(None, ctx.clock() - start)
    return _Outcome(judged, ctx.clock() - start)


def _pct(values: Sequence[float], q: float) -> int | None:
    """백분위 ms (정수, 없으면 None)."""
    return round(float(np.percentile(values, q))) if values else None


def _ratio(num: float, den: float) -> float | None:
    """비율 (분모 0이면 None)."""
    return num / den if den else None


def summarize(concurrency: int, outcomes: Sequence[_Outcome], wall_s: float) -> BenchPoint:
    """점 하나의 결과 → 측정값 (모듈 docstring의 정의)."""
    done = [o.judged for o in outcomes if o.judged is not None]
    lat = [o.seconds * 1000 for o in outcomes if o.judged is not None]
    prompt = sum(j.prompt_tokens or 0 for j in done)
    cached_known = [j.cached_tokens for j in done if j.cached_tokens is not None]
    cached = sum(cached_known) if cached_known else None
    completion = sum(j.completion_tokens or 0 for j in done)
    responses = sum(j.responses for j in done)
    wall = max(wall_s, 1e-9)
    return BenchPoint(
        concurrency=concurrency,
        units=len(outcomes),
        ok=len(done),
        unavailable=len(outcomes) - len(done),
        wall_s=round(wall_s, 3),
        units_per_s=len(done) / wall,
        latency_p50_ms=_pct(lat, 50),
        latency_p95_ms=_pct(lat, 95),
        requests=sum(j.attempts for j in done),
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=completion,
        prefill_tok_s=(prompt - (cached or 0)) / wall,
        decode_tok_s=completion / wall,
        cache_hit_ratio=_ratio(cached, prompt) if cached is not None else None,
        schema_ok_ratio=_ratio(responses - sum(j.invalid_responses for j in done), responses),
        retry_ratio=_ratio(sum(j.retries for j in done), len(done)),
        fallback_ratio=_ratio(sum(j.outcome == "fallback" for j in done), len(done)),
        context_errors=sum(j.context_errors for j in done),
    )


def _point(ctx: _Ctx, cases: Sequence[_Case], concurrency: int, n: int, key: str) -> BenchPoint:
    """동시성 하나: 같은 단위 집합 n개(돌려 쓰기)를 판정한다."""
    jobs = [(cases[k % len(cases)], f"{ctx.token}-{key}-{k}") for k in range(n)]
    start = ctx.clock()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one, ctx, case, mark) for case, mark in jobs]
        try:
            outcomes = [f.result() for f in futures]
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return summarize(concurrency, outcomes, ctx.clock() - start)


def _ttft(ctx: _Ctx, cases: Sequence[_Case], n: int, key: str) -> tuple[int | None, float | None]:
    """첫 토큰 시간 (생성 1토큰 요청의 지연 중앙값)과 동시성 1 프리필 tok/s."""
    lat: list[float] = []
    fresh = 0
    for k, case in enumerate(cases[:n]):
        reply = _Marked(ctx.backend, f"{ctx.token}-{key}-ttft-{k}").judge(
            _full_request(ctx, case, max_tokens=1)
        )
        lat.append(reply.latency_ms)
        fresh += (reply.prompt_tokens or 0) - (reply.cached_tokens or 0)
    seconds = sum(lat) / 1000
    return _pct(lat, 50), (fresh / seconds if seconds and fresh else None)


def _median(values: Sequence[int | None]) -> int | None:
    """None을 뺀 중앙값 (정수)."""
    present = [v for v in values if v is not None]
    return _pct(present, 50)


def _cache_probe(ctx: _Ctx, cases: Sequence[_Case], n: int, key: str) -> CacheProbe:
    """캐시 비교: 새 요청 → 같은 요청 다시(warm) → 맨 앞 난수로 캐시를 깬 요청(cold)."""
    warm: list[JudgeReply] = []
    cold: list[JudgeReply] = []
    for k, case in enumerate(cases[:n]):
        request = _full_request(ctx, case)
        mark = f"{ctx.token}-{key}-cache-{k}"
        fresh = _Marked(ctx.backend, mark)
        fresh.judge(request)
        warm.append(fresh.judge(request))
        cold.append(_Marked(ctx.backend, mark, break_cache=True).judge(request))
    return CacheProbe(
        samples=len(warm),
        warm_latency_ms=_median([r.latency_ms for r in warm]),
        cold_latency_ms=_median([r.latency_ms for r in cold]),
        warm_cached_tokens=_median([r.cached_tokens for r in warm]),
        cold_cached_tokens=_median([r.cached_tokens for r in cold]),
        prompt_tokens=_median([r.prompt_tokens for r in warm]),
    )


def knee(points: Sequence[BenchPoint], timeout_s: float) -> BenchPoint | None:
    """무릎 지점: 장애 0, p95 < 시간 초과인 점 중 단위/s 최대 (같으면 낮은 동시성)."""
    valid = [
        p
        for p in points
        if p.ok and p.unavailable == 0 and (p.latency_p95_ms or 0) < timeout_s * 1000
    ]
    return max(valid, key=lambda p: (p.units_per_s, -p.concurrency), default=None)


def _prepare(data: BenchInput, policy: AiReviewPolicy, group: AiGroup, limit: int) -> list[_Case]:
    """설정의 계획 단위 (그 묶음만, 앞에서 limit개) + 보낼 이미지."""
    units = [
        u
        for u in plan_session(
            data.session,
            list(data.labels),
            policy,
            data.ontology,
            media=data.media,
            sensor=data.sensor,
        )
        if u.group is group
    ][:limit]
    return [
        _Case(u, render_unit_images(u, list(data.frames(u.stream_id, u.frame_idx)), policy))
        for u in units
    ]


@dataclass
class _Run:
    """벤치 실행 상태."""

    grid: BenchGrid
    progress: Callable[[str], None]
    server_down: bool = False


def _measure(run: _Run, ctx: _Ctx, cases: list[_Case], head: Mapping[str, Any]) -> BenchConfig:
    """설정 하나의 측정 (준비 → 첫 토큰 → 캐시 → 동시성 단계).

    서버가 멈추면(한 점의 모든 단위가 장애, 또는 첫 토큰·캐시 요청 장애) 그때까지의 결과에
    `stopped`를 달고 `run.server_down`을 켠다.
    """
    grid = run.grid
    base = {
        **head,
        "planned_units": len(cases),
        "mean_targets": float(np.mean([len(c.unit.targets) for c in cases])),
        "mean_images": float(np.mean([len(c.images) for c in cases])),
    }
    for k in range(grid.warmup):
        _one(ctx, cases[k % len(cases)], f"{ctx.token}-{head['key']}-warm-{k}")
    try:
        ttft, single = _ttft(ctx, cases, grid.ttft, head["key"]) if grid.ttft else (None, None)
        cache = (
            _cache_probe(ctx, cases, min(5, len(cases)), head["key"]) if grid.cache_probe else None
        )
    except VlmUnavailableError as exc:
        run.server_down = True
        return BenchConfig(**_fields(base), stopped=f"서버 장애: {exc}")
    points: list[BenchPoint] = []
    for c in grid.concurrency:
        point = _point(ctx, cases, c, grid.requests, f"{head['key']}-c{c}")
        points.append(point)
        run.progress(
            f"[bench] {head['key']} 동시성 {c}: {point.units_per_s:.2f} 단위/s, "
            f"p95 {point.latency_p95_ms} ms, 장애 {point.unavailable}"
        )
        if point.ok == 0 and point.unavailable:
            run.server_down = True
            stopped = "서버 장애 (모든 단위 실패)"
            return BenchConfig(**_fields(base), points=tuple(points), stopped=stopped)
    return BenchConfig(
        **_fields(base),
        ttft_p50_ms=ttft,
        single_prefill_tok_s=single,
        cache=cache,
        points=tuple(points),
        knee=knee(points, ctx.policy.server.timeout_s),
    )


def _fields(base: Mapping[str, Any]) -> dict[str, Any]:
    """설정 머리 값에서 BenchConfig 필드만 (진행 키 제외)."""
    return {k: v for k, v in base.items() if k != "key"}


def run_bench(
    backend: JudgeBackend,
    data: BenchInput,
    policy: AiReviewPolicy,
    grid: BenchGrid,
    *,
    server: ServerInfo,
    device: str,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = lambda line: None,
) -> BenchReport:
    """벤치를 돌린다 (모듈 docstring). 외부 서비스 호출.

    Args:
        backend: 판정 백엔드 (엔드포인트 하나). data: 입력. policy: 정책 (설정마다 사본).
        grid: 설정 격자. server: 서버 정보 (`probe.server_info`). device: 장비 이름.
        now, clock: 보고서 시각·벽시계 (테스트). progress: 진행 한 줄 출력.

    Raises:
        VlmRequestError: 설정 오류 (4xx, 서버 모델 불일치).
    """
    run = _Run(grid, progress)
    token = secrets.token_hex(4)
    configs: list[BenchConfig] = []
    for group, frames, side, p in _settings(policy, grid):
        head: dict[str, Any] = {"group": group, "frames": frames, "side": side}
        cases = _prepare(data, p, group, grid.requests)
        if not cases:
            configs.append(BenchConfig(**head, planned_units=0))
            continue
        key = f"{group.value}-f{frames}-s{side}"
        config = _measure(run, _Ctx(backend, data, p, token, clock), cases, {**head, "key": key})
        configs.append(config)
        if config.stopped:
            break
    b = policy.backend
    policy_info: dict[str, Any] = {
        "verdict_version": policy.verdict_version, "model": b.model,
        "fingerprint": b.fingerprint, "schema_enforcement": b.schema_enforcement,
        "schema_profile": b.schema_profile, "max_tokens": b.max_tokens,
        "align_px": policy.image.align_px, "jpeg_quality": policy.image.jpeg_quality,
        "timeout_s": policy.server.timeout_s,
    }  # fmt: skip
    grid_info: dict[str, Any] = {
        "concurrency": list(grid.concurrency), "frames": list(grid.frames),
        "sides": list(grid.sides), "groups": [g.value for g in grid.groups],
        "requests": grid.requests, "warmup": grid.warmup, "ttft": grid.ttft,
        "cache_probe": grid.cache_probe,
    }  # fmt: skip
    return BenchReport(
        created_at=now(),
        device=device,
        source=data.source,
        server=server,
        policy=policy_info,
        grid=grid_info,
        configs=tuple(configs),
        server_down=run.server_down,
    )
