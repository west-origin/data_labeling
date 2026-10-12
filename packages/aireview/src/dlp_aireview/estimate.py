"""작업량·장비 대수 추정 `dlp aireview estimate` (WP17.3, ADR 0040 5.8·8.3). 순수 함수.

서버에 묻지 않는다. 입력은 CLI가 DB에서 읽은 세션과 현재 운영 라벨이다 (이 모듈은 DB를 열지 않는다).

세는 것 (세션마다, 묶음별)
- 단위: 실행 경로와 같은 창 논리(`plan.plan_windows`, 선택 모드 포함)로 센다. 영상을 열지 않으므로
  프레임 수는 원하는 시각을 정수 ms로 모은 수(상한)다.
- 이미지 토큰: 프레임 하나 = ⌈w/align⌉·⌈h/align⌉ (긴 변 상한·정렬 단위로 줄인 크기,
  `probe.image_tokens`). 영상 크기는 기준 스트림이면 세션 캘리브레이션, 없으면 `estimate.frame_px`.
  확대 조각은 세지 않는다 (작은 박스가 있으면 요청당 최대 `spatial.crops.max`장이 더해진다).
- 글 토큰: 가변 글(`prompt.unit_text`, 단위마다 다르다)을 실제로 만들어 UTF-8 바이트 /
  `estimate.text_bytes_per_token`. 고정 접두사(system + 온톨로지 블록)는 따로 센다 (접두사 캐시에
  맞는다고 본다). 장갑 문장은 넣지 않는다 (손 상태 단위의 한 줄).
- 예상 출력 토큰: 대상 수 * (`ok_tokens` + `flag_ratio` * `item_tokens`).
- 행동 VLM 요청 수(선택): CLI가 `dlp_actions`의 후보 구간 수로 센다
  (`SessionLoad.actions_requests`).

영상 1시간당 값 = 합 / 영상 시간(세션 길이, 마스터 시계). 라벨이 아직 없는 미래 분량은 이 밀도로
추정한다.

장비 대수 (`capacity`, 벤치 보고서마다)
    T_s = min(프리필 한계, 디코드 한계)   (묶음 s, 벤치 무릎 지점의 tok/s로)
      프리필 한계 = 프리필 tok/s / 단위당 새로 계산할 프롬프트 토큰 (이미지 + 가변 글. 벤치가 캐시
                   적중을 재지 못했으면 고정 접두사도 더한다)
      디코드 한계 = 생성 tok/s / 단위당 예상 출력 토큰
      (tok/s를 모르면 무릎 지점의 단위/s를 그대로 쓴다)
    D = 합_s (영상 1시간당 단위_s / T_s) + 행동 VLM 영상 1시간당 요청 / `actions_req_s`
    대수 = ceil(H * (1 + 재실행 여유) * D / (86,400 * 가동률))
    디코드 확인: H * video_fps * 영상 스트림 수 / (디코드 fps * decode_workers) <= 24 시간
벤치 설정은 지금 정책 값(묶음별 프레임 수·긴 변, `bench.group_settings`)과 같은 것만 쓴다. 측정이
없는 묶음은 `missing`으로 남기고 대수를 내지 않는다 (다른 설정 값으로 대신하지 않는다).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dlp_aireview.bench import BenchReport, group_settings
from dlp_aireview.plan import (
    VIDEO_KINDS,
    PlannedUnit,
    PlannedWindow,
    plan_windows,
    unit_sensor_sha,
)
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import image_tokens
from dlp_aireview.prompt import PromptImage, ontology_text, system_text, unit_text
from dlp_aireview.sensors import SensorContext
from dlp_models.vlm import ChatImage
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import Session

# 하루 초, 1시간 ms
_DAY_S = 86_400
_HOUR_MS = 3_600_000
# 가변 글을 만들 때 쓰는 빈 이미지 (글에는 프레임 번호·시각만 들어간다)
_EMPTY = ChatImage(b"", "")


@dataclass(frozen=True)
class GroupLoad:
    """한 묶음의 작업량 합.

    Attributes:
        units: 단위(요청) 수. targets: 대상 수. frames: 프레임 수(상한).
        image_tokens: 이미지 토큰. text_tokens: 가변 글 토큰. static_tokens: 고정 접두사 토큰 합
            (요청마다 같은 접두사, 캐시 적중 대상). output_tokens: 예상 출력 토큰.
    """

    units: int = 0
    targets: int = 0
    frames: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    static_tokens: int = 0
    output_tokens: float = 0.0

    def __add__(self, other: GroupLoad) -> GroupLoad:
        """두 합을 더한다."""
        return GroupLoad(
            *(a + b for a, b in zip(asdict(self).values(), asdict(other).values(), strict=True))
        )

    def per_unit(self, value: float) -> float:
        """단위당 값 (단위가 없으면 0)."""
        return value / self.units if self.units else 0.0


@dataclass(frozen=True)
class SessionLoad:
    """세션 하나의 작업량.

    Attributes:
        session_id: 세션. video_hours: 영상 시간 (세션 길이). video_streams: 영상 스트림 수.
        groups: 묶음 → 작업량. actions_requests: 행동 VLM 요청 수 (세지 않았으면 None).
    """

    session_id: str
    video_hours: float
    video_streams: int
    groups: Mapping[AiGroup, GroupLoad]
    actions_requests: int | None = None


@dataclass(frozen=True)
class Workload:
    """세션들의 작업량."""

    sessions: tuple[SessionLoad, ...]

    @property
    def video_hours(self) -> float:
        """영상 시간 합."""
        return sum(s.video_hours for s in self.sessions)

    def total(self, group: AiGroup) -> GroupLoad:
        """묶음의 작업량 합."""
        out = GroupLoad()
        for s in self.sessions:
            out = out + s.groups.get(group, GroupLoad())
        return out

    def per_hour(self, value: float) -> float:
        """영상 1시간당 값 (영상이 없으면 0)."""
        return value / self.video_hours if self.video_hours else 0.0

    @property
    def actions_requests(self) -> int | None:
        """행동 VLM 요청 수 합 (한 세션이라도 세지 않았으면 None)."""
        counts = [s.actions_requests for s in self.sessions]
        return None if not counts or None in counts else sum(c or 0 for c in counts)

    @property
    def streams_per_hour(self) -> float:
        """영상 시간으로 가중한 영상 스트림 수 (디코드 확인용)."""
        return self.per_hour(sum(s.video_hours * s.video_streams for s in self.sessions))


def _frame_size(session: Session, stream_id: str, policy: AiReviewPolicy) -> tuple[int, int]:
    """스트림 프레임 크기 (기준 스트림은 캘리브레이션, 없으면 정책 가정 값)."""
    intr = session.calibration.intrinsics
    if intr is not None and stream_id == session.reference_stream.stream_id:
        return intr.width, intr.height
    return policy.estimate.frame_px


def _tokens(text: str, policy: AiReviewPolicy) -> int:
    """글 토큰 근사 (UTF-8 바이트 / text_bytes_per_token, 올림)."""
    return math.ceil(len(text.encode()) / policy.estimate.text_bytes_per_token)


def _window_text(
    w: PlannedWindow, ms: Sequence[int], o: Ontology, policy: AiReviewPolicy, sensor: SensorContext
) -> str:
    """창의 가변 글 (실제 `unit_text`, 이미지 자리는 빈 이미지)."""
    unit = PlannedUnit(
        window=w,
        unit_id="estimate",
        version=policy.verdict_version,
        frame_idx=tuple(range(len(ms))),
        frame_ms=tuple(ms),
        render_hash="",
        sensor_sha256=unit_sensor_sha(w, policy, sensor),
        ontology_version=o.version,
    )
    images = [PromptImage(_EMPTY, t, 0, 0, k, (0, 0)) for k, t in enumerate(ms)]
    return unit_text(unit, images, range(len(w.targets)), o, sensor)


def session_load(
    session: Session,
    current: Sequence[LabelRecord],
    policy: AiReviewPolicy,
    o: Ontology,
    *,
    flagged: Sequence[FlaggedSpan] = (),
    actions_requests: int | None = None,
) -> SessionLoad:
    """세션 하나의 작업량 (모듈 docstring의 세는 법).

    Args:
        session: 세션. current: 현재 운영 라벨 (`current_labels`). policy: 정책 (선택 모드 포함).
        o: 세션 온톨로지. flagged: flagged 모드의 우선순위 사유 구간 (CLI가 계산).
        actions_requests: 행동 VLM 요청 수 (CLI가 셌으면).
    """
    sensor = SensorContext.absent()
    static = {
        g: _tokens(system_text(policy) + ontology_text(o, g, policy), policy) for g in AiGroup
    }
    out: dict[AiGroup, GroupLoad] = {}
    est = policy.estimate
    for w in plan_windows(session, current, policy, o, flagged=flagged):
        ms = sorted({round(t) for t in w.wanted_ms})
        width, height = _frame_size(session, w.stream_id, policy)
        per_frame = image_tokens(width, height, w.max_side_px, policy.image.align_px)
        load = GroupLoad(
            units=1,
            targets=len(w.targets),
            frames=len(ms),
            image_tokens=per_frame * len(ms),
            text_tokens=_tokens(_window_text(w, ms, o, policy, sensor), policy),
            static_tokens=static[w.group],
            output_tokens=len(w.targets) * (est.ok_tokens + est.flag_ratio * est.item_tokens),
        )
        out[w.group] = out.get(w.group, GroupLoad()) + load
    video = sum(1 for s in session.streams if s.kind in VIDEO_KINDS)
    return SessionLoad(
        session.session_id, session.duration_ms / _HOUR_MS, video, out, actions_requests
    )


# ---------------------------------------------------------------- 장비 대수


@dataclass(frozen=True)
class Throughput:
    """벤치에서 읽은 묶음 하나의 지속 처리량.

    Attributes:
        units_per_s: 이 작업량에 맞춘 T_s. limit: 정한 한계 (prefill | decode | knee).
        knee_units_per_s: 벤치 무릎 지점의 단위/s. concurrency: 무릎 동시성.
        prefill_tok_s, decode_tok_s: 무릎 지점의 프리필·생성 tok/s.
    """

    units_per_s: float
    limit: Literal["prefill", "decode", "knee"]
    knee_units_per_s: float
    concurrency: int
    prefill_tok_s: float
    decode_tok_s: float


def group_throughput(
    report: BenchReport, policy: AiReviewPolicy, group: AiGroup, load: GroupLoad
) -> Throughput | None:
    """지금 정책 설정과 같은 벤치 설정의 무릎 지점으로 T_s를 정한다 (없으면 None)."""
    frames, side = group_settings(policy, group)
    config = report.config(group, frames, side)
    point = config.knee if config is not None else None
    if point is None:
        return None
    fresh = load.per_unit(load.image_tokens + load.text_tokens)
    if point.cached_tokens is None:
        fresh += load.per_unit(load.static_tokens)
    limits: dict[Literal["prefill", "decode", "knee"], float] = {}
    if point.prefill_tok_s > 0 and fresh > 0:
        limits["prefill"] = point.prefill_tok_s / fresh
    out_tokens = load.per_unit(load.output_tokens)
    if point.decode_tok_s > 0 and out_tokens > 0:
        limits["decode"] = point.decode_tok_s / out_tokens
    if not limits:
        limits["knee"] = point.units_per_s
    limit = min(limits, key=lambda k: limits[k])
    return Throughput(
        units_per_s=limits[limit],
        limit=limit,
        knee_units_per_s=point.units_per_s,
        concurrency=point.concurrency,
        prefill_tok_s=point.prefill_tok_s,
        decode_tok_s=point.decode_tok_s,
    )


@dataclass(frozen=True)
class GroupCapacity:
    """묶음 하나의 몫.

    Attributes:
        group: 묶음 (행동 VLM이면 "actions"). units_per_hour: 영상 1시간당 단위(요청).
        units_per_s: T_s (모르면 None). limit: 정한 한계. device_s_per_hour: 영상 1시간당 장비 초.
    """

    group: str
    units_per_hour: float
    units_per_s: float | None
    limit: str | None
    device_s_per_hour: float | None


@dataclass(frozen=True)
class Capacity:
    """벤치 보고서 하나로 낸 장비 대수.

    Attributes:
        label: 보고서 설명 (장비·시각·모델). groups: 묶음별 몫 (행동 VLM 포함).
        device_s_per_hour: D (모르는 묶음은 빼고 더한 값). devices: 필요 대수 (모르는 묶음이 있으면
            None). missing: 측정이 없는 묶음. decode_hours_per_day: 하루 디코드 시간 (벤치에
            디코드 측정이 없으면 None). params: 쓴 가정 값.
    """

    label: str
    groups: tuple[GroupCapacity, ...]
    device_s_per_hour: float
    devices: int | None
    missing: tuple[str, ...]
    decode_hours_per_day: float | None
    params: dict[str, float] = field(default_factory=dict[str, float])


def _share(group: str, per_hour: float, t: float | None, limit: str | None) -> GroupCapacity:
    """몫 하나 (T를 모르면 장비 초 None)."""
    return GroupCapacity(group, per_hour, t, limit, per_hour / t if t else None)


def capacity(
    workload: Workload,
    report: BenchReport,
    policy: AiReviewPolicy,
    *,
    hours_per_day: float,
    rerun_margin: float,
    utilization: float,
    decode_workers: int,
    actions_req_s: float | None = None,
) -> Capacity:
    """장비 대수 (모듈 docstring의 공식). 작업량이 없는 묶음은 빼고, 측정이 없는 묶음은 missing.

    Args:
        workload: 작업량. report: 벤치 보고서. policy: 정책 (묶음별 설정 값).
        hours_per_day, rerun_margin, utilization, decode_workers: 가정 값 (CLI가 정책 값 또는 인자).
        actions_req_s: 행동 VLM 지속 요청/s (작업량에 행동 VLM이 있을 때. 모르면 missing).
    """
    shares: list[GroupCapacity] = []
    for group in AiGroup:
        load = workload.total(group)
        if not load.units:
            continue
        tp = group_throughput(report, policy, group, load)
        per_hour = workload.per_hour(load.units)
        shares.append(
            _share(group.value, per_hour, tp.units_per_s if tp else None, tp.limit if tp else None)
        )
    actions = workload.actions_requests
    if actions:
        shares.append(_share("actions", workload.per_hour(actions), actions_req_s, "measured"))
    missing = tuple(s.group for s in shares if s.device_s_per_hour is None)
    d = sum(s.device_s_per_hour or 0.0 for s in shares)
    devices = math.ceil(hours_per_day * (1 + rerun_margin) * d / (_DAY_S * utilization))
    decode = None
    if report.decode is not None and report.decode.fps > 0:
        decode = (
            hours_per_day
            * policy.estimate.video_fps
            * workload.streams_per_hour
            / (report.decode.fps * decode_workers)
        )
    params = {
        "hours_per_day": hours_per_day, "rerun_margin": rerun_margin,
        "utilization": utilization, "decode_workers": float(decode_workers),
    }  # fmt: skip
    return Capacity(
        label=f"{report.device} {report.created_at:%Y-%m-%d %H:%M} {report.policy.get('model')}",
        groups=tuple(shares),
        device_s_per_hour=d,
        devices=None if missing else devices,
        missing=missing,
        decode_hours_per_day=decode,
        params=params,
    )


def to_json(value: Workload | Capacity) -> dict[str, Any]:
    """보고서 JSON (묶음 열거형은 문자열 키)."""
    if isinstance(value, Capacity):
        return asdict(value)
    return {
        "video_hours": value.video_hours,
        "groups": {g.value: asdict(value.total(g)) for g in AiGroup},
        "actions_requests": value.actions_requests,
        "sessions": [
            {
                "session_id": s.session_id,
                "video_hours": s.video_hours,
                "video_streams": s.video_streams,
                "groups": {g.value: asdict(load) for g, load in s.groups.items()},
                "actions_requests": s.actions_requests,
            }
            for s in value.sessions
        ],
    }
