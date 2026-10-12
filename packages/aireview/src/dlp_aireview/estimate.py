"""작업량·장비 대수 추정 `dlp aireview estimate` (WP17.3, ADR 0040 5.8·8.3). 순수 함수.

서버에 묻지 않는다. 입력은 CLI가 DB에서 읽은 세션과 현재 운영 라벨이다 (이 모듈은 DB를 열지 않는다).

세는 것 (세션마다, 묶음별)
- 단위: 실행 경로와 같은 창 논리(`plan.plan_windows`, 선택 모드 포함)로 센다. 영상을 열지 않으므로
  프레임 수는 원하는 시각을 정수 ms로 모은 수(상한)다.
- 이미지 토큰 근사: 프레임 하나 = ⌈w/align⌉·⌈h/align⌉ (긴 변 상한·정렬 단위로 줄인 크기,
  `probe.image_tokens`). 영상 크기는 기준 스트림이면 세션 캘리브레이션, 없으면 `estimate.frame_px`.
  공간 단위의 확대 조각도 실행과 같은 규칙(`overlay.crop_plan`: 작은 박스, 요청당 `crops.max`장)으로
  세어 더한다 (`GroupLoad.crops`).
- 글 토큰 근사: 가변 글(`prompt.unit_text`, 단위마다 다르다)을 실제로 만들어
  `probe.approx_text_tokens`(UTF-8 바이트 / `estimate.text_bytes_per_token`)로 센다. 고정
  접두사(system + 온톨로지 블록)는 따로 센다.
  장갑 문장은 넣지 않는다 (손 상태 단위의 한 줄).
- 예상 출력 토큰: 대상 수 * (`ok_tokens` + `flag_ratio` * `item_tokens`).
- 행동 VLM 요청 수(선택): CLI가 `dlp_actions`의 후보 구간 수로 센다
  (`SessionLoad.actions_requests`).

영상 1시간당 값 = 합 / 영상 시간(세션 길이, 마스터 시계). 라벨이 아직 없는 미래 분량은 이 밀도로
추정한다.

장비 대수 (`capacity`, 벤치 보고서마다)
    T_s = 무릎 단위/s * t(벤치) / t(작업량)        (묶음 s, `group_throughput`)
      t(x) = P_x / C_p + O_x / C_d   (단위 하나의 장비 시간 모형)
      P: 단위당 새로 계산할 프롬프트 근사 토큰 = 이미지 + 가변 글 + (1 - f) * 고정 접두사
         (f = 고정 접두사가 캐시에 맞은 비율, `static_cached_fraction`. 서버가 cached_tokens를 주지
         않거나 0이면 f = 0)
      O: 단위당 출력 토큰 (벤치는 무릎 지점 실측, 작업량은 예상 출력)
      C_p, C_d: 벤치가 무릎 동시성에서 따로 잰 프리필 용량(근사 토큰/s, `PrefillRun`)과 생성 용량
         (생성 토큰/s, `DecodeRun`, 서버가 길이 고정을 따른 측정만)
      작업량 구성이 벤치와 같으면 T_s = 무릎 단위/s다. 용량 측정이 없으면 보정 없이 무릎 단위/s를
      쓰고(limit "knee") 경고한다.
    D = 합_s (영상 1시간당 단위_s / T_s) + 행동 VLM 영상 1시간당 요청 / `actions_req_s`
    대수 = ceil(H * (1 + 재실행 여유) * D / (86,400 * 가동률))
    디코드 확인: H * video_fps * 영상 스트림 수 / (디코드 fps * decode_workers) <= 24 시간

벤치·추정의 근사 셈법이 같으므로(같은 align_px·text_bytes_per_token) 서버 토크나이저와의 차이는
t(벤치)/t(작업량) 비율에서 지워진다. 그래서 CLI는 보고서의 셈법 값으로 작업량을 다시 센다
(`counting_policy`). 모델·fingerprint가 지금 정책과 다른 보고서는 경고한다 (`report_mismatch`, 후보
모델 비교에 쓴다). 벤치 설정은 지금 정책 값(묶음별 프레임 수·긴 변, `bench.group_settings`)과 같은
것만 쓴다. 측정이 없는 묶음은 `missing`으로 남기고 대수를 내지 않는다 (다른 설정 값으로 대신하지
않는다).

설계(8.3)의 `min(프리필 한계, 디코드 한계)`를 바꾼 이유: 혼합 실행의 무릎 지점에서 나온 프리필·
생성 tok/s는 그 실행의 관측 속도이지 용량이 아니다. 무릎이 프리필에 묶였을 때 생성 tok/s
(= 단위/s * 단위당 출력)를 "디코드 한계"로 쓰면 묶이지 않은 자원이 T를 깎는다. 그래서 용량을 따로
재고, 실측한 무릎을 기준으로 작업량 구성 차이만 시간 모형으로 옮긴다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from dlp_aireview.bench import ApproxLoad, BenchConfig, BenchPoint, BenchReport, group_settings
from dlp_aireview.overlay import crop_plan
from dlp_aireview.plan import (
    VIDEO_KINDS,
    PlannedUnit,
    PlannedWindow,
    plan_windows,
    unit_sensor_sha,
)
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.probe import approx_text_tokens, image_tokens
from dlp_aireview.prompt import PromptImage, mark_tag, ontology_text, system_text, unit_text
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
        image_tokens: 이미지 토큰 (확대 조각 포함). text_tokens: 가변 글 토큰. static_tokens: 고정
            접두사 토큰 합 (요청마다 같은 접두사, 캐시 적중 대상). output_tokens: 예상 출력 토큰.
        crops: 확대 조각 수 (`overlay.crop_plan`).
    """

    units: int = 0
    targets: int = 0
    frames: int = 0
    image_tokens: int = 0
    text_tokens: int = 0
    static_tokens: int = 0
    output_tokens: float = 0.0
    crops: int = 0

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


def _window_text(
    w: PlannedWindow,
    ms: Sequence[int],
    crops: Sequence[tuple[int, int]],
    o: Ontology,
    policy: AiReviewPolicy,
    sensor: SensorContext,
) -> str:
    """창의 가변 글 (실제 `unit_text`, 이미지 자리는 빈 이미지, 확대 조각 설명 포함)."""
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
    images += [
        PromptImage(_EMPTY, ms[k], 0, 0, k, (0, 0), crop=True, label=mark_tag(w.targets[n], n))
        for k, n in crops
    ]
    return unit_text(unit, images, range(len(w.targets)), o, sensor)


def _window_load(
    w: PlannedWindow,
    session: Session,
    o: Ontology,
    policy: AiReviewPolicy,
    static: int,
) -> GroupLoad:
    """창 하나의 작업량 (모듈 docstring의 세는 법)."""
    est, align = policy.estimate, policy.image.align_px
    ms = sorted({round(t) for t in w.wanted_ms})
    width, height = _frame_size(session, w.stream_id, policy)
    crops: tuple[tuple[int, int], ...] = ()
    if w.group is AiGroup.SPATIAL:
        crops = crop_plan(w.targets, ms, (width, height), w.max_side_px, policy)
    side = policy.spatial.crops.side_px
    per_frame = image_tokens(width, height, w.max_side_px, align)
    per_crop = image_tokens(side, side, side, align)
    text = _window_text(w, ms, crops, o, policy, SensorContext.absent())
    return GroupLoad(
        units=1,
        targets=len(w.targets),
        frames=len(ms),
        image_tokens=per_frame * len(ms) + per_crop * len(crops),
        text_tokens=approx_text_tokens(text, policy),
        static_tokens=static,
        output_tokens=len(w.targets) * (est.ok_tokens + est.flag_ratio * est.item_tokens),
        crops=len(crops),
    )


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
    static = {
        g: approx_text_tokens(system_text(policy) + ontology_text(o, g, policy), policy)
        for g in AiGroup
    }
    out: dict[AiGroup, GroupLoad] = {}
    for w in plan_windows(session, current, policy, o, flagged=flagged):
        load = _window_load(w, session, o, policy, static[w.group])
        out[w.group] = out.get(w.group, GroupLoad()) + load
    video = sum(1 for s in session.streams if s.kind in VIDEO_KINDS)
    return SessionLoad(
        session.session_id, session.duration_ms / _HOUR_MS, video, out, actions_requests
    )


# ---------------------------------------------------------------- 장비 대수


@dataclass(frozen=True)
class Throughput:
    """벤치에서 읽은 묶음 하나의 지속 처리량 (이 작업량에 맞춘 값).

    Attributes:
        units_per_s: T_s. limit: 작업량 단위 시간에서 큰 몫 (prefill | decode). 용량 측정이 없어
            보정하지 않았으면 knee. knee_units_per_s: 벤치 무릎 지점의 단위/s. concurrency: 무릎
            동시성. prefill_tok_s, decode_tok_s: 따로 잰 프리필 용량(근사 토큰/s)·생성 용량
            (생성 토큰/s), 없으면 None. mix: t(벤치) / t(작업량) (T = 무릎 * mix), 없으면 None.
        static_cached: 고정 접두사가 캐시에 맞은 비율 추정 (0~1).
    """

    units_per_s: float
    limit: Literal["prefill", "decode", "knee"]
    knee_units_per_s: float
    concurrency: int
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    mix: float | None = None
    static_cached: float = 0.0


def static_cached_fraction(point: BenchPoint, approx: ApproxLoad) -> float:
    """고정 접두사 중 서버 캐시에 맞은 비율 (0~1).

    서버가 cached_tokens를 주지 않았거나(None) 0이면 0이다 (접두사도 매번 계산했다고 본다). 그
    밖에는 캐시 적중 비율(cached / prompt) ÷ 근사 고정 접두사 비율(static / (static + image +
    text))을 1에서 자른다.
    """
    total = approx.static + approx.image + approx.text
    if point.cache_hit_ratio is None or approx.static <= 0 or total <= 0:
        return 0.0
    return min(1.0, point.cache_hit_ratio / (approx.static / total))


def _capacities(config: BenchConfig, cached: float) -> tuple[float, float] | None:
    """따로 잰 (프리필 근사 토큰/s, 생성 토큰/s). 없거나 쓸 수 없는 측정이면 None.

    프리필 용량의 토큰에는 캐시에 맞지 않은 고정 접두사 몫((1 - cached) * 접두사)도 넣는다. 생성
    용량은 서버가 생성 길이 고정을 따른 측정(`DecodeRun.honored`)만 쓴다.
    """
    pre, dec = config.prefill, config.decode
    if pre is None or dec is None or not dec.honored or not pre.ok:
        return None
    if pre.wall_s <= 0 or dec.wall_s <= 0:
        return None
    c_p = (pre.fresh_tokens + (1 - cached) * pre.static_tokens) / pre.wall_s
    c_d = dec.completion_tokens / dec.wall_s
    return (c_p, c_d) if c_p > 0 and c_d > 0 else None


def group_throughput(
    report: BenchReport, policy: AiReviewPolicy, group: AiGroup, load: GroupLoad
) -> Throughput | None:
    """지금 정책 설정과 같은 벤치 설정의 무릎 지점을 작업량 구성에 맞춘 T_s (모듈 docstring).

    측정이 없으면 None. 용량 측정(`prefill`·`decode`)이나 근사 토큰(`approx`)이 없으면 보정 없이
    무릎 단위/s (limit "knee").
    """
    frames, side = group_settings(policy, group)
    config = report.config(group, frames, side)
    point = config.knee if config is not None else None
    if config is None or point is None:
        return None
    base = Throughput(point.units_per_s, "knee", point.units_per_s, point.concurrency)
    approx = config.approx
    if approx is None or not point.ok or not load.units:
        return base
    cached = static_cached_fraction(point, approx)
    caps = _capacities(config, cached)
    if caps is None:
        return replace(base, static_cached=cached)
    c_p, c_d = caps
    p_bench = approx.image + approx.text + (1 - cached) * approx.static
    t_bench = p_bench / c_p + (point.completion_tokens / point.ok) / c_d
    prefill = load.per_unit(load.image_tokens + load.text_tokens) + (1 - cached) * load.per_unit(
        load.static_tokens
    )
    t_prefill, t_decode = prefill / c_p, load.per_unit(load.output_tokens) / c_d
    mix = t_bench / (t_prefill + t_decode)
    return Throughput(
        units_per_s=point.units_per_s * mix,
        limit="prefill" if t_prefill >= t_decode else "decode",
        knee_units_per_s=point.units_per_s,
        concurrency=point.concurrency,
        prefill_tok_s=c_p,
        decode_tok_s=c_d,
        mix=mix,
        static_cached=cached,
    )


def report_mismatch(report: BenchReport, policy: AiReviewPolicy) -> list[str]:
    """보고서를 만든 정책 값이 지금 정책과 다른 곳 (모델·fingerprint·align_px·text_bytes_per_token).

    보고서에 그 값이 없으면(이전 형식) 비교하지 않는다. 셈법 값(align_px·text_bytes_per_token)은
    CLI가 `counting_policy`로 맞춰 작업량을 다시 세므로, 그렇게 넘긴 정책이면 모델·fingerprint만
    남는다.
    """
    now: dict[str, Any] = {
        "model": policy.backend.model,
        "fingerprint": policy.backend.fingerprint,
        "align_px": policy.image.align_px,
        "text_bytes_per_token": policy.estimate.text_bytes_per_token,
    }
    return [
        f"{key}: 보고서 {report.policy[key]} ≠ 지금 정책 {value}"
        for key, value in now.items()
        if key in report.policy and report.policy[key] != value
    ]


def counting_policy(policy: AiReviewPolicy, report: BenchReport) -> AiReviewPolicy:
    """보고서의 셈법 값(`image.align_px`, `estimate.text_bytes_per_token`)으로 바꾼 정책 사본.

    작업량을 벤치와 같은 근사 셈법으로 세야 t(벤치)/t(작업량) 비율이 맞다 (후보 모델마다 align_px가
    다를 수 있다). 값이 없거나 같으면 정책을 그대로 돌려준다.
    """
    data = policy.model_dump()
    align, bpt = report.policy.get("align_px"), report.policy.get("text_bytes_per_token")
    if isinstance(align, int) and align != policy.image.align_px:
        data["image"]["align_px"] = align
    if isinstance(bpt, int | float) and bpt != policy.estimate.text_bytes_per_token:
        data["estimate"]["text_bytes_per_token"] = bpt
    return policy if data == policy.model_dump() else AiReviewPolicy.model_validate(data)


@dataclass(frozen=True)
class GroupCapacity:
    """묶음 하나의 몫.

    Attributes:
        group: 묶음 (행동 VLM이면 "actions"). units_per_hour: 영상 1시간당 단위(요청).
        units_per_s: T_s (모르면 None). limit: 큰 몫 (prefill | decode | knee | measured).
        device_s_per_hour: 영상 1시간당 장비 초. prefill_tok_s, decode_tok_s: 쓴 용량 (없으면 None).
    """

    group: str
    units_per_hour: float
    units_per_s: float | None
    limit: str | None
    device_s_per_hour: float | None
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None


@dataclass(frozen=True)
class Capacity:
    """벤치 보고서 하나로 낸 장비 대수.

    Attributes:
        label: 보고서 설명 (장비·시각·모델). groups: 묶음별 몫 (행동 VLM 포함).
        device_s_per_hour: D (모르는 묶음은 빼고 더한 값). devices: 필요 대수 (모르는 묶음이 있으면
            None). missing: 측정이 없는 묶음. decode_hours_per_day: 하루 디코드 시간 (벤치에
            디코드 측정이 없으면 None). params: 쓴 가정 값. warnings: 보고서·정책 불일치, 보정하지
            못한 묶음.
    """

    label: str
    groups: tuple[GroupCapacity, ...]
    device_s_per_hour: float
    devices: int | None
    missing: tuple[str, ...]
    decode_hours_per_day: float | None
    params: dict[str, float] = field(default_factory=dict[str, float])
    warnings: tuple[str, ...] = ()


def _share(group: str, per_hour: float, tp: Throughput | None) -> GroupCapacity:
    """묶음 몫 하나 (T를 모르면 장비 초 None)."""
    if tp is None:
        return GroupCapacity(group, per_hour, None, None, None)
    t = tp.units_per_s
    return GroupCapacity(
        group, per_hour, t, tp.limit, per_hour / t if t else None, tp.prefill_tok_s, tp.decode_tok_s
    )


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
        workload: 작업량 (`policy`의 셈법으로 센 것). report: 벤치 보고서.
        policy: 정책 (묶음별 설정 값, 셈법 값은 보고서와 같아야 한다: `counting_policy`).
        hours_per_day, rerun_margin, utilization, decode_workers: 가정 값 (CLI가 정책 값 또는 인자).
        actions_req_s: 행동 VLM 지속 요청/s (작업량에 행동 VLM이 있을 때. 모르면 missing).
    """
    shares: list[GroupCapacity] = []
    warnings = [f"벤치 보고서와 정책이 다르다 — {m}" for m in report_mismatch(report, policy)]
    for group in AiGroup:
        load = workload.total(group)
        if not load.units:
            continue
        tp = group_throughput(report, policy, group, load)
        shares.append(_share(group.value, workload.per_hour(load.units), tp))
        if tp is not None and tp.limit == "knee":
            warnings.append(
                f"{group.value}: 프리필·생성 용량 측정이 없어 무릎 처리량을 작업량 구성으로 "
                "보정하지 않았다 (벤치와 작업량의 단위당 토큰이 다르면 틀린다)"
            )
    actions = workload.actions_requests
    if actions:
        per_hour = workload.per_hour(actions)
        shares.append(
            GroupCapacity(
                "actions",
                per_hour,
                actions_req_s,
                "measured",
                per_hour / actions_req_s if actions_req_s else None,
            )
        )
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
        warnings=tuple(warnings),
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
