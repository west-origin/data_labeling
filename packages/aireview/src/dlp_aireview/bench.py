"""처리량 벤치 `dlp aireview bench` (WP17.3, ADR 0040 5.6·8.3). DB에 쓰지 않는다.

장비 1대(엔드포인트 하나)가 판정 요청을 얼마나 처리하는지 실제 계획기·프롬프트·스키마·재검사
(`judge_unit`)로 잰다. 결과(무릎 처리량, 프리필·생성 용량, 근사 토큰)는 `dlp aireview estimate
--bench`가 장비 대수 공식에 쓴다.

재는 방법
- 설정 = 묶음(temporal·temporal_long·spatial), 프레임 수(temporal만, `temporal.frames`), 긴 변
  (그 묶음의 `max_side_px`)의 조합. 격자 후보에 지금 정책 값(`group_settings`)을 늘 더한다 (추정은
  정책 값과 같은 설정의 측정만 쓴다). 격자가 비면 정책 값만 잰다. 설정마다 정책 사본
  (`bench_policy`)으로 계획(`plan_session`)하고 그 묶음 단위만 쓴다. 입력은 `BenchInput`(세션·라벨·
  PTS 인덱스·프레임 공급 함수)이다. 합성 입력은 CLI가 `dlp_fixtures.aireview`로 만든다 (영상·사람·
  글자 없음).
- 설정마다: 준비 요청(`warmup`, 측정 안 함) → 첫 토큰 시간(`max_tokens=1`, 동시성 1) → 캐시 비교 →
  동시성 단계마다 단위 `requests`개 판정 → 무릎 동시성에서 프리필·생성 용량(`capacity`).
- 요청마다 새 이미지: 계획 단위가 `requests`보다 적으면 단위를 돌려 쓴다. 같은 이미지를 다시 보내면
  서버의 멀티모달 전처리·인코더 캐시(vLLM `--mm-processor-cache-gb`, 이미지 해시 기반 재사용)가 비전
  인코딩을 건너뛰어 처리량이 부풀려진다. 그래서 측정 요청마다 이미지 오른쪽 아래 모서리에 표식별
  잡음 조각을 넣어 다시 인코딩한다 (`_vary`, 시간 측정 전에 만든다). 점마다 서로 다른 이미지 묶음
  수(`distinct_image_sets`)를 남기고, 요청 수보다 적으면 진행 줄에 경고한다.
- 측정 표식: 요청마다 고정 블록(온톨로지) 끝에 표식 한 줄을 넣는다 (`_Marked`). system만 접두사
  캐시에 맞고 온톨로지 뒤(이미지·가변 글)는 매번 새로 계산한다. 운영에서 단위는 늘 새 입력이고
  온톨로지 블록까지 캐시에 맞으며 같은 창의 분할 요청은 이미지를 공유하므로, 벤치의 요청당 계산량은
  운영보다 조금 크다 (보수적). 단, 추정이 근사로 세는 작업량 쪽 오차는 `estimate` 문서를 본다.
- 점(동시성 하나)마다: 단위/s, p50·p95 지연(단위 하나의 재시도·분할 포함 왕복), 관측 프리필 tok/s
  (= 서버 (prompt - cached) 합 / 벽시계), 관측 생성 tok/s, 캐시 적중 비율, 스키마 적합률(위반 없는
  응답 / 받은 응답), 재시도율, fallback율, 문맥 초과 수, 서버 장애 수. 관측 tok/s는 그 혼합 실행에서
  나온 속도이지 용량이 아니다 (묶이지 않은 자원의 속도는 낮게 나온다).
- 무릎 지점(`knee`): 서버 장애 0이고 p95가 정책 `server.timeout_s`보다 작은 점 중 단위/s가 가장 큰
  점 (같으면 낮은 동시성). 그 동시성이 `server.concurrency_per_endpoint` 후보다. 동시성은 httpx 연결
  풀 기본 상한(100) 이하로 준다 (넘으면 클라이언트 쪽에서 기다린 시간이 지연에 섞인다).
- 용량(`grid.capacity`, 무릎 동시성에서 따로):
  - 프리필(`PrefillRun`): 생성 1토큰 요청 `requests`개 (요청마다 새 이미지). 근사 새 토큰(이미지 +
    가변 글)·고정 접두사 토큰 합과 벽시계를 남긴다.
  - 생성(`DecodeRun`): 단위 하나를 먼저 한 번 보내 캐시에 올린 뒤, 가변 글 끝 표식만 다른 요청
    2 * 동시성개를 보낸다 (프롬프트 대부분이 캐시에 맞아 생성이 시간을 정한다). 구조화 출력 없이
    `max_tokens = min_tokens = decode_tokens`와 `ignore_eos`(vLLM 확장)로 생성 길이를 고정한다.
    서버가 확장을 무시하면(생성 토큰이 요청 길이와 다르면) `honored=False`, 거부하면(4xx) `error`다.
    추정은 그런 측정을 쓰지 않는다.
  - 추정(`estimate.group_throughput`)은 무릎 처리량을 두 용량의 시간 모형으로 작업량 구성(단위당 새
    토큰·출력 토큰)에 맞춘다.
- 근사 토큰(`ApproxLoad`): 단위당 이미지·가변 글·고정 접두사 토큰을 추정과 같은 셈법
  (`probe.image_tokens`, `probe.approx_text_tokens`)으로 센다. 확대 조각도 센다.
- `cache_probe`: 새 요청 → 같은 요청 다시(warm) → system 맨 앞 난수 줄 + 새 이미지로 캐시를 모두 깬
  요청(cold)의 지연·적중 토큰을 비교한다.
- 서버 장애(`VlmUnavailableError`)는 그 점의 오류로 센다 (재시도하지 않는다. 과부하 503도 오류다).
  한 점의 모든 단위가 장애면 서버가 멈춘 것으로 보고 벤치를 멈춘다(`server_down`). 설정 오류
  (`VlmRequestError`: 4xx, 서버 모델 불일치)는 그대로 올린다 (CLI 종료 5). 생성 용량 요청의 4xx만
  예외로 `DecodeRun.error`에 남긴다.

보고서(`BenchReport`)는 JSON으로 `reports/aireview-bench/<장비>-<시각>.json`(Git 제외)에 남긴다.
`/v1/models`, 서버 모델, 엔진 버전, 정책 모델·fingerprint·align_px, 장비 이름을 함께 담는다. 디코드
fps(`decode`)는 CLI가 잰다 (이 패키지는 영상을 열지 않는다, 블러본 경계는 `media.py`).
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import AwareDatetime

from dlp_aireview.backend import BackendInfo, JudgeBackend, JudgeReply, JudgeRequest
from dlp_aireview.judge import JudgedUnit, judge_unit, request_for
from dlp_aireview.overlay import render_unit_images
from dlp_aireview.plan import PlannedUnit, StreamMedia, plan_session
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import ServerInfo, approx_text_tokens, image_tokens
from dlp_aireview.prompt import PromptImage, build_parts
from dlp_aireview.schema import response_schema
from dlp_aireview.sensors import SensorContext
from dlp_models.vlm import ChatImage, VlmRequestError, VlmUnavailableError, encode_jpeg
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
            frames·sides에는 지금 정책 값이 늘 더해진다 (비면 정책 값만).
        groups: 잴 묶음. requests: 점마다 판정할 단위 수 (계획 단위를 돌려 쓰되 이미지는 요청마다
            새로 만든다). warmup: 설정마다 측정 전 준비 단위 수. ttft: 첫 토큰 시간 표본 수.
        cache_probe: 캐시 비교를 할지. capacity: 무릎 동시성에서 프리필·생성 용량을 따로 잴지.
        decode_tokens: 생성 용량 측정의 요청당 생성 길이.
    """

    concurrency: tuple[int, ...]
    frames: tuple[int, ...]
    sides: tuple[int, ...]
    groups: tuple[AiGroup, ...]
    requests: int
    warmup: int
    ttft: int
    cache_probe: bool
    capacity: bool = True
    decode_tokens: int = 256


# ---------------------------------------------------------------- 보고서


class BenchPoint(Contract):
    """동시성 하나의 측정값 (tok/s는 이 혼합 실행의 관측 속도, 용량이 아니다)."""

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
    distinct_image_sets: int = 0


class ApproxLoad(Contract):
    """단위당 근사 토큰 (추정과 같은 셈법). 측정 요청에 돌려 쓴 단위들의 평균이다.

    image: 이미지(확대 조각 포함). text: 가변 글. static: 고정 접두사(system + 온톨로지 블록).
    """

    image: float
    text: float
    static: float


class PrefillRun(Contract):
    """프리필 용량 측정: 무릎 동시성에서 생성 1토큰 요청 (요청마다 새 이미지).

    fresh_tokens·static_tokens: 응답을 받은 요청의 근사 새 토큰(이미지 + 가변 글)과 근사 고정 접두사
    합. prompt_tokens·cached_tokens: 서버 usage 합 (정보). wall_s: 벽시계 초.
    """

    concurrency: int
    requests: int
    ok: int
    unavailable: int
    wall_s: float
    fresh_tokens: float
    static_tokens: float
    prompt_tokens: int = 0
    cached_tokens: int | None = None


class DecodeRun(Contract):
    """생성 용량 측정: 무릎 동시성에서 프롬프트가 캐시에 맞는 요청, 생성 길이 고정.

    honored: 응답마다 생성 토큰이 정확히 `forced_tokens`였는가 (서버가 `ignore_eos`·`min_tokens`를
    따랐는가). error: 서버가 길이 고정 요청을 거부한 사유 (4xx 또는 장애).
    """

    concurrency: int
    requests: int
    forced_tokens: int
    ok: int = 0
    unavailable: int = 0
    wall_s: float = 0.0
    completion_tokens: int = 0
    decode_tok_s: float = 0.0
    honored: bool = False
    error: str | None = None


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
    approx: ApproxLoad | None = None
    ttft_p50_ms: int | None = None
    single_prefill_tok_s: float | None = None
    cache: CacheProbe | None = None
    points: tuple[BenchPoint, ...] = ()
    knee: BenchPoint | None = None
    prefill: PrefillRun | None = None
    decode: DecodeRun | None = None
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


def _with_policy_value(values: Sequence[int], policy_value: int) -> tuple[int, ...]:
    """격자 후보 + 정책 값 (중복 없이 오름차순)."""
    return tuple(sorted({*values, policy_value}))


def bench_settings(
    policy: AiReviewPolicy, grid: BenchGrid
) -> Iterator[tuple[AiGroup, int, int, AiReviewPolicy]]:
    """(묶음, 그 묶음 프레임 수, 긴 변, 정책 사본) — 프레임 수 후보는 temporal에만.

    묶음마다 지금 정책 값(`group_settings`)을 후보에 늘 더한다.
    """
    for group in grid.groups:
        frames0, side0 = group_settings(policy, group)
        options: tuple[int | None, ...] = (
            _with_policy_value(grid.frames, frames0) if group is AiGroup.TEMPORAL else (None,)
        )
        for frames in options:
            for side in _with_policy_value(grid.sides, side0):
                p = bench_policy(policy, group, frames, side)
                yield group, group_settings(p, group)[0], side, p


# ---------------------------------------------------------------- 요청 꾸미기

# 측정 표식을 넣는 곳: ontology(고정 블록 끝, 기본) | system(맨 앞, 캐시 모두 깨기)
#   | tail(가변 글 끝)
_Where = Literal["ontology", "system", "tail"]


class _Marked:
    """요청마다 측정 표식 한 줄을 넣는 백엔드 감싸기 (모듈 docstring)."""

    def __init__(self, inner: JudgeBackend, mark: str, *, where: _Where = "ontology") -> None:
        """mark: 표식 문자열. where: 넣는 곳."""
        self.inner = inner
        self.mark = mark
        self.where = where
        self.info: BackendInfo = inner.info

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """표식을 넣어 보낸다."""
        parts, line = request.parts, f"(측정 {self.mark})"
        if self.where == "system":
            parts = replace(parts, system=f"{line}\n{parts.system}")
        elif self.where == "tail":
            parts = replace(parts, text=f"{parts.text}\n{line}")
        else:
            parts = replace(parts, ontology=f"{parts.ontology}\n{line}")
        return self.inner.judge(replace(request, parts=parts))


def _noisy(image: ChatImage, seed: str, quality: int) -> ChatImage:
    """오른쪽 아래 모서리(짧은 변의 1/8, 최소 8px)에 seed로 정한 잡음 조각을 넣어 다시 인코딩한다.

    픽셀이 달라지므로 서버의 이미지 해시 기반 캐시가 맞지 않는다.

    Raises:
        ValueError: JPEG를 읽지 못할 때.
    """
    bgr = cv2.imdecode(np.frombuffer(image.jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG를 읽지 못했습니다")
    rgb: Image = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    h, w = rgb.shape[:2]
    side = min(h, w, max(8, min(h, w) // 8))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    rgb[h - side :, w - side :] = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)
    return encode_jpeg(rgb, quality)


def _vary(images: Sequence[PromptImage], seed: str, quality: int) -> tuple[PromptImage, ...]:
    """이미지마다 `_noisy` (크기·번호·시각은 그대로)."""
    return tuple(
        replace(im, image=_noisy(im.image, f"{seed}:{k}", quality)) for k, im in enumerate(images)
    )


@dataclass(frozen=True)
class _Case:
    """준비한 단위 하나 (계획 단위 + 보낼 이미지 + 근사 토큰)."""

    unit: PlannedUnit
    images: tuple[PromptImage, ...]
    approx: ApproxLoad

    @property
    def image_key(self) -> tuple[str, ...]:
        """보낼 이미지 묶음의 키 (sha256 목록)."""
        return tuple(im.image.sha256 for im in self.images)


@dataclass(frozen=True)
class _Ctx:
    """설정 하나의 실행 문맥."""

    backend: JudgeBackend
    data: BenchInput
    policy: AiReviewPolicy
    token: str
    clock: Callable[[], float]

    def fresh(self, case: _Case, mark: str) -> tuple[_Case, str]:
        """표식 `mark`의 새 이미지로 바꾼 단위와 그 표식 (측정 요청 하나)."""
        images = _vary(case.images, mark, self.policy.image.jpeg_quality)
        return replace(case, images=images), mark


def _approx(
    unit: PlannedUnit, images: tuple[PromptImage, ...], data: BenchInput, policy: AiReviewPolicy
) -> ApproxLoad:
    """단위 하나의 근사 토큰 (추정과 같은 셈법: `probe.image_tokens`·`approx_text_tokens`)."""
    parts = build_parts(unit, images, range(len(unit.targets)), data.ontology, policy, data.sensor)
    align = policy.image.align_px
    return ApproxLoad(
        image=float(sum(image_tokens(im.w, im.h, max(im.w, im.h), align) for im in images)),
        text=float(approx_text_tokens(parts.text, policy)),
        static=float(approx_text_tokens(parts.system + parts.ontology, policy)),
    )


def _mean_approx(cases: Sequence[_Case], n: int) -> ApproxLoad:
    """측정 요청 n개(단위를 차례로 돌려 씀)의 근사 토큰 평균."""
    picked = [cases[k % len(cases)].approx for k in range(max(n, 1))]
    return ApproxLoad(
        image=float(np.mean([a.image for a in picked])),
        text=float(np.mean([a.text for a in picked])),
        static=float(np.mean([a.static for a in picked])),
    )


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


def _ask(ctx: _Ctx, request: JudgeRequest, mark: str, where: _Where) -> JudgeReply | None:
    """요청 하나를 그대로 보낸다 (재검사 없음, 장애면 None, 설정 오류는 올린다)."""
    try:
        return _Marked(ctx.backend, mark, where=where).judge(request)
    except VlmUnavailableError:
        return None


def _parallel[T](concurrency: int, calls: Sequence[Callable[[], T]]) -> list[T]:
    """호출들을 동시성 `concurrency`로 돌려 순서대로 결과를 모은다 (예외면 남은 호출을 취소)."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call) for call in calls]
        try:
            return [f.result() for f in futures]
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise


def _pct(values: Sequence[float], q: float) -> int | None:
    """백분위 ms (정수, 없으면 None)."""
    return round(float(np.percentile(values, q))) if values else None


def _ratio(num: float, den: float) -> float | None:
    """비율 (분모 0이면 None)."""
    return num / den if den else None


def summarize(
    concurrency: int, outcomes: Sequence[_Outcome], wall_s: float, *, distinct: int = 0
) -> BenchPoint:
    """점 하나의 결과 → 측정값 (모듈 docstring의 정의).

    스키마 적합률 = (받은 응답 - 위반 응답) / 받은 응답, 재시도율 = 재시도 수 / 판정 단위,
    fallback율 = fallback 단위 / 판정 단위, 문맥 초과 = 거절된 요청 수 (`JudgedUnit` 집계).
    """
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
        distinct_image_sets=distinct,
    )


def _point(ctx: _Ctx, cases: Sequence[_Case], concurrency: int, n: int, key: str) -> BenchPoint:
    """동시성 하나: 단위 n개(계획 단위를 돌려 쓰되 이미지는 요청마다 새것)를 판정한다.

    새 이미지는 시간 측정 전에 만든다.
    """
    jobs = [ctx.fresh(cases[k % len(cases)], f"{ctx.token}-{key}-{k}") for k in range(n)]
    distinct = len({case.image_key for case, _ in jobs})
    start = ctx.clock()
    outcomes = _parallel(concurrency, [partial(_one, ctx, case, mark) for case, mark in jobs])
    return summarize(concurrency, outcomes, ctx.clock() - start, distinct=distinct)


def _ttft(ctx: _Ctx, cases: Sequence[_Case], n: int, key: str) -> tuple[int | None, float | None]:
    """첫 토큰 시간 (생성 1토큰 요청의 지연 중앙값)과 동시성 1 프리필 tok/s (요청마다 새 이미지)."""
    lat: list[float] = []
    fresh = 0
    for k, case in enumerate(cases[:n]):
        varied, mark = ctx.fresh(case, f"{ctx.token}-{key}-ttft-{k}")
        reply = _Marked(ctx.backend, mark).judge(_full_request(ctx, varied, max_tokens=1))
        lat.append(reply.latency_ms)
        fresh += (reply.prompt_tokens or 0) - (reply.cached_tokens or 0)
    seconds = sum(lat) / 1000
    return _pct(lat, 50), (fresh / seconds if seconds and fresh else None)


def _median(values: Sequence[int | None]) -> int | None:
    """None을 뺀 중앙값 (정수)."""
    present = [v for v in values if v is not None]
    return _pct(present, 50)


def _cache_probe(ctx: _Ctx, cases: Sequence[_Case], n: int, key: str) -> CacheProbe:
    """캐시 비교: 새 요청 → 같은 요청 다시(warm) → system 맨 앞 난수 + 새 이미지 요청(cold)."""
    warm: list[JudgeReply] = []
    cold: list[JudgeReply] = []
    for k, case in enumerate(cases[:n]):
        varied, mark = ctx.fresh(case, f"{ctx.token}-{key}-cache-{k}")
        request = _full_request(ctx, varied)
        fresh = _Marked(ctx.backend, mark)
        fresh.judge(request)
        warm.append(fresh.judge(request))
        cold_case, cold_mark = ctx.fresh(case, f"{mark}-cold")
        cold.append(
            _Marked(ctx.backend, cold_mark, where="system").judge(_full_request(ctx, cold_case))
        )
    return CacheProbe(
        samples=len(warm),
        warm_latency_ms=_median([r.latency_ms for r in warm]),
        cold_latency_ms=_median([r.latency_ms for r in cold]),
        warm_cached_tokens=_median([r.cached_tokens for r in warm]),
        cold_cached_tokens=_median([r.cached_tokens for r in cold]),
        prompt_tokens=_median([r.prompt_tokens for r in warm]),
    )


def _prefill_run(
    ctx: _Ctx, cases: Sequence[_Case], concurrency: int, n: int, key: str
) -> PrefillRun:
    """프리필 용량: 생성 1토큰 요청 n개 (요청마다 새 이미지, 근사 토큰 합과 벽시계)."""
    jobs = [ctx.fresh(cases[k % len(cases)], f"{ctx.token}-{key}-prefill-{k}") for k in range(n)]
    requests = [(_full_request(ctx, case, max_tokens=1), mark, case) for case, mark in jobs]
    start = ctx.clock()
    replies = _parallel(
        concurrency, [partial(_ask, ctx, req, mark, "ontology") for req, mark, _ in requests]
    )
    wall = ctx.clock() - start
    done = [(r, case.approx) for r, (_, _, case) in zip(replies, requests, strict=True) if r]
    cached = [r.cached_tokens for r, _ in done if r.cached_tokens is not None]
    return PrefillRun(
        concurrency=concurrency,
        requests=n,
        ok=len(done),
        unavailable=n - len(done),
        wall_s=wall,
        fresh_tokens=sum(a.image + a.text for _, a in done),
        static_tokens=sum(a.static for _, a in done),
        prompt_tokens=sum(r.prompt_tokens or 0 for r, _ in done),
        cached_tokens=sum(cached) if cached else None,
    )


def _decode_run(ctx: _Ctx, case: _Case, concurrency: int, tokens: int, key: str) -> DecodeRun:
    """생성 용량: 단위 하나를 캐시에 올린 뒤 가변 글 끝 표식만 다른 요청 2 * 동시성개.

    구조화 출력 없이 생성 길이를 `tokens`로 고정한다 (`min_tokens`·`ignore_eos`, vLLM 확장). 서버가
    길이 고정 요청을 거부하거나(4xx) 첫 요청이 장애면 `error`만 남긴다.
    """
    n = 2 * concurrency
    base = _full_request(ctx, case, max_tokens=tokens)
    request = replace(
        base, schema_enforcement="prompt", extra_body={"min_tokens": tokens, "ignore_eos": True}
    )
    head: dict[str, Any] = {"concurrency": concurrency, "requests": n, "forced_tokens": tokens}
    try:
        _Marked(ctx.backend, f"{ctx.token}-{key}-prime", where="tail").judge(request)
    except (VlmRequestError, VlmUnavailableError) as exc:
        return DecodeRun(**head, error=f"생성 길이 고정 요청 실패: {exc}")
    marks = [f"{ctx.token}-{key}-decode-{k}" for k in range(n)]
    start = ctx.clock()
    replies = _parallel(concurrency, [partial(_ask, ctx, request, m, "tail") for m in marks])
    wall = ctx.clock() - start
    done = [r for r in replies if r is not None]
    completion = sum(r.completion_tokens or 0 for r in done)
    return DecodeRun(
        **head,
        ok=len(done),
        unavailable=n - len(done),
        wall_s=wall,
        completion_tokens=completion,
        decode_tok_s=completion / wall if wall > 0 else 0.0,
        honored=bool(done) and all(r.completion_tokens == tokens for r in done),
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
    """설정의 계획 단위 (그 묶음만, 앞에서 limit개) + 보낼 이미지 + 근사 토큰."""
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
    cases: list[_Case] = []
    for u in units:
        images = render_unit_images(u, list(data.frames(u.stream_id, u.frame_idx)), policy)
        cases.append(_Case(u, images, _approx(u, images, data, policy)))
    return cases


@dataclass
class _Run:
    """벤치 실행 상태."""

    grid: BenchGrid
    progress: Callable[[str], None]
    server_down: bool = False


def _points(
    run: _Run, ctx: _Ctx, cases: list[_Case], key: str
) -> tuple[list[BenchPoint], str | None]:
    """동시성 단계마다 점을 잰다. 한 점의 모든 단위가 장애면 멈추고 사유를 돌려준다."""
    points: list[BenchPoint] = []
    for c in run.grid.concurrency:
        point = _point(ctx, cases, c, run.grid.requests, f"{key}-c{c}")
        points.append(point)
        reused = (
            f", 경고: 이미지 묶음 {point.distinct_image_sets}/{point.units}만 서로 다름"
            if point.distinct_image_sets < point.units
            else ""
        )
        run.progress(
            f"[bench] {key} 동시성 {c}: {point.units_per_s:.2f} 단위/s, "
            f"p95 {point.latency_p95_ms} ms, 장애 {point.unavailable}{reused}"
        )
        if point.ok == 0 and point.unavailable:
            run.server_down = True
            return points, "서버 장애 (모든 단위 실패)"
    return points, None


def _capacity(
    run: _Run, ctx: _Ctx, cases: list[_Case], point: BenchPoint, key: str
) -> tuple[PrefillRun, DecodeRun]:
    """무릎 동시성에서 프리필·생성 용량을 따로 잰다."""
    c = point.concurrency
    prefill = _prefill_run(ctx, cases, c, run.grid.requests, key)
    decode = _decode_run(ctx, cases[0], c, run.grid.decode_tokens, key)
    note = decode.error or ("" if decode.honored else " (서버가 생성 길이 고정을 따르지 않음)")
    run.progress(
        f"[bench] {key} 용량 (동시성 {c}): 프리필 근사 "
        f"{prefill.fresh_tokens / max(prefill.wall_s, 1e-9):.0f} tok/s, "
        f"생성 {decode.decode_tok_s:.0f} tok/s{note}"
    )
    return prefill, decode


def _measure(run: _Run, ctx: _Ctx, cases: list[_Case], head: Mapping[str, Any]) -> BenchConfig:
    """설정 하나의 측정 (준비 → 첫 토큰 → 캐시 → 동시성 단계 → 용량).

    서버가 멈추면(한 점의 모든 단위가 장애, 또는 첫 토큰·캐시 요청 장애) 그때까지의 결과에
    `stopped`를 달고 `run.server_down`을 켠다.
    """
    grid, key = run.grid, head["key"]
    base = {
        **head,
        "planned_units": len(cases),
        "mean_targets": float(np.mean([len(c.unit.targets) for c in cases])),
        "mean_images": float(np.mean([len(c.images) for c in cases])),
        "approx": _mean_approx(cases, grid.requests),
    }
    for k in range(grid.warmup):
        _one(ctx, cases[k % len(cases)], f"{ctx.token}-{key}-warm-{k}")
    try:
        ttft, single = _ttft(ctx, cases, grid.ttft, key) if grid.ttft else (None, None)
        cache = _cache_probe(ctx, cases, min(5, len(cases)), key) if grid.cache_probe else None
    except VlmUnavailableError as exc:
        run.server_down = True
        return BenchConfig(**_fields(base), stopped=f"서버 장애: {exc}")
    points, stopped = _points(run, ctx, cases, key)
    if stopped:
        return BenchConfig(**_fields(base), points=tuple(points), stopped=stopped)
    best = knee(points, ctx.policy.server.timeout_s)
    prefill, decode = (
        _capacity(run, ctx, cases, best, key)
        if best is not None and grid.capacity
        else (None, None)
    )
    return BenchConfig(
        **_fields(base),
        ttft_p50_ms=ttft,
        single_prefill_tok_s=single,
        cache=cache,
        points=tuple(points),
        knee=best,
        prefill=prefill,
        decode=decode,
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
    for group, frames, side, p in bench_settings(policy, grid):
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
        "text_bytes_per_token": policy.estimate.text_bytes_per_token,
        "timeout_s": policy.server.timeout_s,
    }  # fmt: skip
    grid_info: dict[str, Any] = {
        "concurrency": list(grid.concurrency), "frames": list(grid.frames),
        "sides": list(grid.sides), "groups": [g.value for g in grid.groups],
        "requests": grid.requests, "warmup": grid.warmup, "ttft": grid.ttft,
        "cache_probe": grid.cache_probe, "capacity": grid.capacity,
        "decode_tokens": grid.decode_tokens,
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
