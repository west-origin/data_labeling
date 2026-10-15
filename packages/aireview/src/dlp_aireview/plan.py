"""판정 요청 단위 계획 (순수·결정적, WP17.2, ADR 0040 2.1·2.2·4.3).

입력은 세션, 현재 운영 라벨, 정책, 온톨로지, (프레임을 고를 때) 스트림별 블러본 PTS 인덱스와
렌더 해시다. 영상·DB·서버를 읽지 않는다.

단계
1. `targets`: 판정 대상 = 검수 전 자동(model) 운영 라벨 중 판정 종류 (평가 경로는 `eval_targets`로
   대상을 직접 준다: 골든·과거 라벨은 검수됐기 때문이다). 측정·오류 삽입 레코드,
   원본 접근 이벤트(raw_access), 블러 트랙, 장갑 센서에서 나온 라벨(`glove_derived`),
   `long_max_span_ms`보다 긴 상위 구간은 뺀다.
   정책 `kinds`는 대상만 고른다. 맥락·개체표·프롬프트·응답 스키마는 판정할 수 있는 종류 전체
   (`TEMPORAL_KINDS`, `SPATIAL_KINDS`) 기준이라 `kinds`를 바꿔도 남은 단위의 키와 요청이 그대로다.
2. `plan_windows`: 묶음별 창을 만든다 (영상 없이 같은 결과라 `estimate_units`,
   `uncovered_targets`도 쓴다).
   - temporal: 0부터 `window_ms` 격자. 짧은 대상은 중점이 든 창 하나에만 든다. 증거 구간 = 창과
     대상을 덮고 앞뒤 `context_ms`를 더한 구간(세션 길이로 자름). 대상이 `max_targets`를 넘으면
     같은 프레임으로 요청을 나눈다 (분할 순번).
   - temporal_long: 긴 라벨(상위 구간 전부, `long_label_ms`보다 긴 라벨,
     `long_object_state_ms`보다 긴 객체 상태) 하나가 한 단위다.
   - spatial: 영상 스트림마다 `stride_ms` 격자와 트랙 첫 키프레임에 스냅샷 창을 두고(가까운
     시각은 합침), 창 안에 표식이 그려지는 키프레임이 있는 트랙이 대상이다 (트랙당
     `max_windows_per_track`개까지). 프레임은 대상 트랙의 키프레임 시각에서 골라 대상마다 적어도 한
     프레임에 표식이 그려지게 한다 (`spatial.frames`장으로 다 덮지 못하면 같은 구간의 다음 창).
   - `selection.mode=flagged`면 지적 단위 → 감사 표본 순으로 고른다.
3. `plan_session`: 창마다 프레임을 고르고(`dlp_media.frames.pick_frames`) 단위 키를 만든다.

시각: temporal 묶음은 마스터 ms이고 기준 스트림(바디캠, offset 0·배율 1)의 PTS ms와 같다.
spatial은 그 스트림 PTS ms다 (ADR 0019). 프레임 번호와 간격의 곱으로 시각을 계산하지 않는다.

단위 키(`unit_id`)는 판정 입력(대상·맥락 라벨 ID·프레임·렌더·센서·온톨로지 버전과 내용 해시)의
해시다. 온톨로지 내용 해시(`dlp_schema.ontology.ontology_digest`)는 같은 버전의 draft 덧붙이기로
프롬프트 개체표·응답 스키마가 바뀐 것을 구별한다 (F-AIR-32). 라벨 하나를
고치면 그 창만, 렌더 해시가 바뀌면 그 스트림만 다시 판정한다. 선택 방식은 키에 들어가지 않는다
(all↔flagged 전환이 다시 판정을 부르지 않게). 맥락 라벨의 검수 상태(프롬프트 맥락 표의 "(확정)"
표시, `prompt._confirmed`)도 키 밖이다: 승인은 같은 label_id 행의 검수 상태만 바꾸므로, 승인만으로는
다시 판정하지 않는다 (검수가 진행될 때마다 재판정이 늘지 않게 한 결정, ADR 0040 4절). 같은 키의
단위가 "(확정)" 표시만 다른 프롬프트로 판정될 수 있고, 실제로 보낸 요청은 `prompt_digest`가 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from dlp_aireview.policy import SPATIAL_KINDS, TEMPORAL_KINDS, AiReviewPolicy
from dlp_aireview.sensors import ABSENT, SensorContext
from dlp_aireview.tracks import Timeline
from dlp_media.frames import frame_ms, pick_frames
from dlp_media.pts import PtsIndex
from dlp_schema.aireview import (
    AiGroup,
    AiReviewUnit,
    is_judge_target,
    raw_access_event,
)
from dlp_schema.episode import entity_refs
from dlp_schema.labels import (
    BoxTrackPayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    SegmentPayload,
)
from dlp_schema.ontology import Ontology, ontology_digest
from dlp_schema.review import FlaggedSpan, ReviewReason
from dlp_schema.session import Session, StreamKind

# 공간 판정을 하는 영상 스트림 종류 (블러본이 있는 스트림)
VIDEO_KINDS = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)
# 시간 묶음 창 개체표에 넣는 개체 참조 종류 (손 키포인트의 손 ID는 넣지 않는다)
_ENTITY_KINDS = ("action", "hand_state", "event", "object_state", "box_track")
Selection = Literal["all", "flagged", "unknown_gap", "audit"]


@dataclass(frozen=True)
class StreamMedia:
    """한 스트림의 블러본 정보 (프레임 고르기용).

    Attributes:
        pts: 블러본 PTS 인덱스.
        render_hash: 지금 승인된 블러본의 렌더 해시.
    """

    pts: PtsIndex
    render_hash: str


@dataclass(frozen=True)
class PlannedWindow:
    """프레임을 고르기 전의 요청 단위 (영상 없이 정해진다).

    Attributes:
        session_id, group, stream_id: 세션, 묶음, 프레임을 뽑을 스트림.
        t_start_ms, t_end_ms: 증거 구간 (temporal 마스터 ms, spatial 스트림 PTS ms).
        split_index: 같은 창을 대상 상한으로 나눈 순번 (같은 프레임을 쓴다).
        targets: 판정 대상 (순서 = 응답 번호 i).
        context: 맥락 라벨 (텍스트로만 준다, label_id 정렬).
        entities: 창 개체표 (개체 ID, 클래스) — 응답의 target·tool 번호는 이 순서다.
        wanted_ms: 프레임을 고를 시각 (오름차순). temporal은 그 시각에 화면에 떠 있는 프레임,
            spatial은 가장 가까운 프레임으로 맞춘다 (spatial 시각은 대상 키프레임 = 그 스트림 PTS
            시각).
        window_key: 묶음·스트림·창 시작 (같은 창의 분할 요청을 같은 엔드포인트로, 감사 표본 해시).
        selection: 고른 이유 (all | flagged | unknown_gap | audit).
        max_side_px: 프레임 긴 변 상한.
    """

    session_id: str
    group: AiGroup
    stream_id: str
    t_start_ms: int
    t_end_ms: int
    split_index: int
    targets: tuple[LabelRecord, ...]
    context: tuple[LabelRecord, ...]
    entities: tuple[tuple[str, str | None], ...]
    wanted_ms: tuple[float, ...]
    window_key: str
    selection: Selection
    max_side_px: int


@dataclass(frozen=True)
class PlannedUnit:
    """프레임까지 고른 요청 단위 하나 (= 판정 기록 한 행).

    Attributes:
        window: 창 (대상·맥락·개체표·증거 구간).
        unit_id: 단위 키 (`unit_id`).
        version: 판정 버전.
        frame_idx, frame_ms: 고른 프레임 번호와 그 PTS 정수 ms (오름차순).
        render_hash: 그 스트림 블러본의 렌더 해시.
        sensor_sha256: 이 단위가 장갑 문장을 쓰면 그 해시, 아니면 "absent".
        ontology_version: 세션 온톨로지 버전.
        ontology_digest: 세션 온톨로지 내용 해시 (`ontology_digest`, F-AIR-32).
    """

    window: PlannedWindow
    unit_id: str
    version: str
    frame_idx: tuple[int, ...]
    frame_ms: tuple[int, ...]
    render_hash: str
    sensor_sha256: str
    ontology_version: str
    ontology_digest: str

    @property
    def group(self) -> AiGroup:
        """묶음."""
        return self.window.group

    @property
    def stream_id(self) -> str:
        """스트림."""
        return self.window.stream_id

    @property
    def targets(self) -> tuple[LabelRecord, ...]:
        """판정 대상 (응답 번호 순)."""
        return self.window.targets

    @property
    def span(self) -> tuple[int, int]:
        """증거 구간 (시작 ms, 끝 ms)."""
        return self.window.t_start_ms, self.window.t_end_ms


@dataclass(frozen=True)
class UnitEstimate:
    """작업량 추정용 단위 요약 (영상 없이 센다).

    Attributes:
        group, stream_id: 묶음과 스트림.
        targets: 대상 수.
        frames: 프레임 수 상한 (같은 프레임으로 합쳐지면 실제는 더 적다).
        max_side_px: 프레임 긴 변 상한.
    """

    group: AiGroup
    stream_id: str
    targets: int
    frames: int
    max_side_px: int


# ---------------------------------------------------------------- 대상·맥락


def _operational(label: LabelRecord, o: Ontology) -> bool:
    """대상·맥락 공통 조건: 측정·오류 삽입 레코드, 블러, 원본 접근 이벤트가 아니다.

    `current_labels`가 이미 측정·오류 삽입 레코드를 빼지만 한 번 더 거른다.
    """
    return (
        not label.seeded_error
        and label.measurement is None
        and label.kind != "blur_track"
        and not raw_access_event(label, o)
    )


def targets(
    current: Sequence[LabelRecord], policy: AiReviewPolicy, o: Ontology
) -> list[LabelRecord]:
    """판정 대상: 검수 전 자동 운영 라벨 중 정책 종류·출처 (시작 시각, ID 순).

    조건은 `dlp_schema.aireview.is_judge_target`(정책 값은 `policy.target_rule`)이다. 장갑 센서에서
    나온 라벨(`glove_derived`)과 `long_max_span_ms`보다 긴 상위 구간은 뺀다. 검수 화면의 판정률
    분모(`dlp_review.ai_display.exposure_for`)도 같은 함수를 쓴다. 모델 버전·신뢰도·검수자·우선순위
    사유는 판정 입력에 넣지 않는다 (자동 라벨의 자신감에 끌리지 않게).
    """
    rule = policy.target_rule
    out = [x for x in current if is_judge_target(x, o, rule)]
    return sorted(out, key=lambda x: (x.t_start_ms, x.label_id))


def eval_targets(
    only: Sequence[LabelRecord], policy: AiReviewPolicy, o: Ontology
) -> list[LabelRecord]:
    """평가 경로(`dlp aireview eval`, `evaluate`)가 직접 준 대상 중 판정할 수 있는 것 (시작 시각,
    ID 순).

    골든·과거 라벨은 검수된 라벨이라 `targets`의 검수 상태·출처 조건을 통과하지 못하므로 그 두
    조건만 빼고 같은 조건을 쓴다: 정책 종류, 측정·오류 삽입 레코드·블러·원본 접근 이벤트 아님,
    `long_max_span_ms`보다 긴 상위 구간 아님. 장갑 센서 근거 제외(`glove_derived`)는 대상을 고르는
    쪽(평가 사례 만들기)이 정한다 (사람이 그린 골든 손 상태는 센서 출처가 아니다).
    """
    rule = policy.target_rule
    out = [
        x
        for x in only
        if x.kind in rule.kinds
        and _operational(x, o)
        and not (
            isinstance(x.payload, SegmentPayload)
            and x.t_end_ms - x.t_start_ms > rule.long_max_span_ms
        )
    ]
    return sorted(out, key=lambda x: (x.t_start_ms, x.label_id))


def entity_classes(current: Iterable[LabelRecord]) -> dict[str, str]:
    """개체 ID → 객체 클래스 (박스·마스크 트랙, 객체 상태에서)."""
    out: dict[str, str] = {}
    for x in current:
        p = x.payload
        if isinstance(p, BoxTrackPayload | MaskTrackPayload | ObjectStatePayload):
            out.setdefault(p.entity_id, p.class_id)
    return out


def _overlaps(label: LabelRecord, start: int, end: int) -> bool:
    """라벨 구간이 [start, end]와 겹치는가 (시점 라벨은 구간 안이면)."""
    if label.t_start_ms == label.t_end_ms:
        return start <= label.t_start_ms <= end
    return label.t_start_ms < end and label.t_end_ms > start


def _entity_table(
    labels: Iterable[LabelRecord], classes: Mapping[str, str], unresolved: set[str]
) -> tuple[tuple[str, str | None], ...]:
    """창 개체표: 라벨들이 참조하는 개체 (ID 정렬, 대상을 모르는 접촉 ID는 뺀다)."""
    ids = {
        r for x in labels if x.kind in _ENTITY_KINDS for r in entity_refs(x) if r not in unresolved
    }
    return tuple((i, classes.get(i)) for i in sorted(ids))


# ---------------------------------------------------------------- 프레임 시각


def _uniform(start: float, end: float, n: int) -> list[float]:
    """구간을 n등분한 칸의 가운데 시각."""
    return [start + (i + 0.5) * (end - start) / n for i in range(n)]


def _spread(
    base: Iterable[float], extras: Iterable[float], span: tuple[int, int], min_gap: int, limit: int
) -> tuple[float, ...]:
    """프레임 시각 고르기. 최대 `limit`개, 오름차순.

    기본 시각을 순서대로(간격 `min_gap` 이상) 넣고, 추가 후보는 이미 고른 시각에서 가장 먼 것부터
    넣는다 (간격 이상, 같으면 이른 시각).
    """
    a, b = span
    picked: list[float] = []
    for t in base:
        t = min(max(t, a), b)
        if len(picked) < limit and all(abs(t - s) >= min_gap for s in picked):
            picked.append(t)
    pool = sorted({min(max(t, a), b) for t in extras})
    while len(picked) < limit and pool:
        dist = [min((abs(c - s) for s in picked), default=math.inf) for c in pool]
        best = max(range(len(pool)), key=lambda k: (dist[k], -pool[k]))
        if dist[best] < min_gap:
            break
        picked.append(pool.pop(best))
    return tuple(sorted(picked))


def _temporal_extras(targets_: Sequence[LabelRecord], policy: AiReviewPolicy) -> list[float]:
    """대상 경계와 시점 이벤트 초점 시각 (정책 boundary_frames·point_event_focus_ms)."""
    tp = policy.temporal
    out: list[float] = []
    for x in targets_:
        if isinstance(x.payload, EventPayload) and x.t_start_ms == x.t_end_ms:
            t = x.t_start_ms
            out += [t - tp.point_event_focus_ms, t, t + tp.point_event_focus_ms]
        elif tp.boundary_frames:
            out += [x.t_start_ms, x.t_end_ms]
    return out


# ---------------------------------------------------------------- 창 만들기


def _is_long(label: LabelRecord, policy: AiReviewPolicy) -> bool:
    """temporal_long 대상인가 (상위 구간 전부, 긴 객체 상태, 긴 라벨)."""
    tp = policy.temporal
    length = label.t_end_ms - label.t_start_ms
    if isinstance(label.payload, SegmentPayload):
        return True
    if isinstance(label.payload, ObjectStatePayload):
        return length > tp.long_object_state_ms
    return length > tp.long_label_ms


@dataclass(frozen=True)
class _Ctx:
    """창 만들기 공통 입력."""

    session: Session
    policy: AiReviewPolicy
    pool: tuple[LabelRecord, ...]  # 맥락 후보 (운영 라벨, 판정 종류)
    classes: Mapping[str, str]
    unresolved: frozenset[str]


def _windows_from(
    ctx: _Ctx,
    group: AiGroup,
    stream_id: str,
    span: tuple[int, int],
    members: Sequence[LabelRecord],
    context: Sequence[LabelRecord],
    wanted: tuple[float, ...],
    key: str,
) -> list[PlannedWindow]:
    """한 창의 대상을 상한으로 나눠 분할 창들을 만든다 (같은 증거 구간·프레임·맥락·개체표)."""
    limit = (
        ctx.policy.spatial.max_targets
        if group is AiGroup.SPATIAL
        else ctx.policy.temporal.max_targets
    )
    side = {
        AiGroup.TEMPORAL: ctx.policy.temporal.max_side_px,
        AiGroup.TEMPORAL_LONG: ctx.policy.temporal.long_max_side_px,
        AiGroup.SPATIAL: ctx.policy.spatial.max_side_px,
    }[group]
    ids = {x.label_id for x in members}
    ctx_labels = tuple(
        sorted((x for x in context if x.label_id not in ids), key=lambda x: x.label_id)
    )
    entities = () if group is AiGroup.SPATIAL else _entity_table(
        [*members, *ctx_labels], ctx.classes, set(ctx.unresolved)
    )  # fmt: skip
    return [
        PlannedWindow(
            session_id=ctx.session.session_id,
            group=group,
            stream_id=stream_id,
            t_start_ms=span[0],
            t_end_ms=span[1],
            split_index=s,
            targets=tuple(members[s * limit : (s + 1) * limit]),
            context=ctx_labels,
            entities=entities,
            wanted_ms=wanted,
            window_key=key,
            selection="all",
            max_side_px=side,
        )
        for s in range(math.ceil(len(members) / limit))
    ]


def _temporal_context(ctx: _Ctx, span: tuple[int, int], ref_id: str) -> list[LabelRecord]:
    """시간 창의 맥락: 증거 구간과 겹치는 시간 라벨 + 기준 스트림 박스 트랙(개체표용).

    판정할 수 있는 시간 종류 전체(`TEMPORAL_KINDS`)를 본다. 정책 `kinds`는 대상만 고른다 (바꿔도
    남은 단위의 맥락·키·프롬프트가 그대로).
    """
    temporal = set(TEMPORAL_KINDS)
    return [
        x
        for x in ctx.pool
        if (x.kind in temporal or (x.kind == "box_track" and x.stream_id == ref_id))
        and _overlaps(x, *span)
    ]


def _temporal_windows(ctx: _Ctx, short: Sequence[LabelRecord]) -> list[PlannedWindow]:
    """temporal 묶음: 격자 창마다 중점이 든 짧은 대상."""
    tp, duration = ctx.policy.temporal, ctx.session.duration_ms
    ref = ctx.session.reference_stream.stream_id
    by_window: dict[int, list[LabelRecord]] = {}
    for x in short:
        by_window.setdefault((x.t_start_ms + x.t_end_ms) // 2 // tp.window_ms, []).append(x)
    out: list[PlannedWindow] = []
    for k in sorted(by_window):
        members = by_window[k]
        w0, w1 = k * tp.window_ms, min((k + 1) * tp.window_ms, duration)
        a = max(0, min(w0, *(x.t_start_ms for x in members)) - tp.context_ms)
        b = min(duration, max(w1, *(x.t_end_ms for x in members)) + tp.context_ms)
        span = (a, max(a, b))
        wanted = _spread(
            _uniform(*span, tp.frames), _temporal_extras(members, ctx.policy), span,
            tp.min_frame_gap_ms, tp.max_frames,
        )  # fmt: skip
        context = _temporal_context(ctx, span, ref)
        out += _windows_from(
            ctx, AiGroup.TEMPORAL, ref, span, members, context, wanted, f"temporal|{ref}|{w0}"
        )
    return out


def _long_windows(ctx: _Ctx, long: Sequence[LabelRecord]) -> list[PlannedWindow]:
    """temporal_long 묶음: 긴 라벨 하나 = 한 단위 (고르게, 객체 상태는 전이 근처 추가)."""
    tp, duration = ctx.policy.temporal, ctx.session.duration_ms
    ref = ctx.session.reference_stream.stream_id
    out: list[PlannedWindow] = []
    for x in long:
        span = (max(0, x.t_start_ms - tp.context_ms), min(duration, x.t_end_ms + tp.context_ms))
        span = (span[0], max(span))
        extras = [x.t_start_ms, x.t_end_ms] if isinstance(x.payload, ObjectStatePayload) else []
        wanted = _spread(
            _uniform(x.t_start_ms, x.t_end_ms, tp.long_frames), extras, span,
            tp.min_frame_gap_ms, tp.max_frames,
        )  # fmt: skip
        context = _temporal_context(ctx, span, ref)
        out += _windows_from(
            ctx, AiGroup.TEMPORAL_LONG, ref, span, [x], context, wanted,
            f"temporal_long|{ref}|{x.label_id}",
        )  # fmt: skip
    return out


@dataclass(frozen=True)
class _Track:
    """공간 대상·맥락 트랙과 표식 시간표 (창마다 다시 계산하지 않게)."""

    label: LabelRecord
    timeline: Timeline
    drawn: tuple[int, ...]  # 표식이 그려지는 키프레임 시각 (오름차순)

    @classmethod
    def of(cls, label: LabelRecord) -> _Track:
        """라벨 → 트랙."""
        timeline = Timeline.of(label)
        return cls(label, timeline, tuple(timeline.drawn_times()))

    def drawn_in(self, span: tuple[int, int]) -> bool:
        """구간 안에 표식이 그려지는 키프레임이 있는가."""
        k = bisect_left(self.drawn, span[0])
        return k < len(self.drawn) and self.drawn[k] <= span[1]


def _snapshot_times(tracks: Sequence[_Track], policy: AiReviewPolicy) -> list[int]:
    """스냅샷 창 시작 시각: 격자와 트랙 첫 키프레임 (`snapshot_ms` 안의 가까운 시각은 앞 것으로)."""
    sp = policy.spatial
    times: set[int] = set()
    for x in tracks:
        seen = x.drawn
        if not seen:
            continue
        times |= {
            k * sp.stride_ms for k in range(seen[0] // sp.stride_ms, seen[-1] // sp.stride_ms + 1)
        }
        if sp.track_start_snapshot:
            times.add(seen[0])
    merged: list[int] = []
    for t in sorted(times):
        if not merged or t >= merged[-1] + sp.snapshot_ms:
            merged.append(t)
    return merged


def _cover(
    members: Sequence[_Track], span: tuple[int, int], budget: int
) -> tuple[list[int], list[_Track], list[_Track]]:
    """대상마다 표식이 적어도 한 프레임에 그려지게 시각을 `budget`개까지 고른다 (탐욕 집합 덮기).

    후보는 창 안에서 대상 표식이 그려지는 키프레임 시각이다 (공간 키프레임은 그 스트림 PTS 시각이라
    가장 가까운 프레임이 바로 그 키프레임이다). 아직 덮이지 않은 대상을 가장 많이 그리는 시각부터
    (같으면 이른 시각) 고른다.

    Returns: (고른 시각, 덮은 대상, 덮지 못한 대상). 대상 순서는 입력 순서를 지킨다.
    """
    cands = sorted({t for m in members for t in m.drawn if span[0] <= t <= span[1]})
    left = {m.label.label_id for m in members}
    picked: list[int] = []
    while left and len(picked) < budget and cands:
        gains = [
            (
                t,
                {
                    m.label.label_id
                    for m in members
                    if m.label.label_id in left and m.timeline.drawn_at(t)
                },
            )
            for t in cands
        ]
        best, gain = max(gains, key=lambda g: (len(g[1]), -g[0]))
        if not gain:
            break
        picked.append(best)
        left -= gain
    covered = [m for m in members if m.label.label_id not in left]
    return picked, covered, [m for m in members if m.label.label_id in left]


def _spatial_windows(
    ctx: _Ctx, spatial: Sequence[LabelRecord], stream_id: str
) -> list[PlannedWindow]:
    """spatial 묶음 (한 스트림): 스냅샷 창마다 표식이 그려지는 키프레임이 있는 대상 트랙.

    프레임 시각은 대상 트랙의 키프레임 시각에서 고른다 (`_cover`, 대상마다 적어도 한 프레임에 표식).
    `spatial.frames`장으로 창의 대상을 다 덮지 못하면 남은 대상으로 같은 증거 구간의 다음 창을
    만든다 (창 키 끝에 `|<순번>`). 덮고 남은 자리는 창에 고르게 채운다 (`_spread`).
    """
    sp = ctx.policy.spatial
    tracks = [_Track.of(x) for x in spatial if x.stream_id == stream_id]
    # 맥락은 판정할 수 있는 공간 종류 전체 (정책 kinds는 대상만 고른다)
    others = [
        _Track.of(x) for x in ctx.pool if x.stream_id == stream_id and x.kind in SPATIAL_KINDS
    ]
    used: dict[str, int] = {}
    out: list[PlannedWindow] = []
    for t0 in _snapshot_times(tracks, ctx.policy):
        span = (t0, t0 + sp.snapshot_ms)
        members = [
            x
            for x in tracks
            if x.drawn_in(span) and used.get(x.label.label_id, 0) < sp.max_windows_per_track
        ]
        context = [x.label for x in others if x.drawn_in(span)]
        part = 0
        while members:
            times, covered, members = _cover(members, span, sp.frames)
            if not covered:
                break
            for x in covered:
                used[x.label.label_id] = used.get(x.label.label_id, 0) + 1
            wanted = _spread(times, _uniform(*span, sp.frames), span, 1, sp.frames)
            key = f"spatial|{stream_id}|{t0}" + (f"|{part}" if part else "")
            out += _windows_from(
                ctx, AiGroup.SPATIAL, stream_id, span, [x.label for x in covered], context,
                wanted, key,
            )  # fmt: skip
            part += 1
    return out


# ---------------------------------------------------------------- 선택


def _audit_draw(session_id: str, key: str, seed: int) -> float:
    """감사 표본 추첨 값 [0, 1): sha256(세션|창 키|seed)의 앞 32비트."""
    digest = hashlib.sha256(f"{session_id}|{key}|{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 2**32


def _is_flagged(
    w: PlannedWindow, flagged: Sequence[FlaggedSpan], reasons: set[ReviewReason]
) -> bool:
    """우선순위 사유 구간이 이 창의 대상을 가리키거나(라벨 ID), 라벨 없이 시간 창과 겹치는가."""
    ids = {x.label_id for x in w.targets}
    for f in flagged:
        if f.reason not in reasons:
            continue
        if set(f.label_ids) & ids:
            return True
        if (
            not f.label_ids
            and w.group is not AiGroup.SPATIAL
            and (f.t_start_ms < w.t_end_ms and f.t_end_ms > w.t_start_ms)
        ):
            return True
    return False


def _select(
    windows: list[PlannedWindow], policy: AiReviewPolicy, flagged: Sequence[FlaggedSpan]
) -> list[PlannedWindow]:
    """선택 모드 적용. all은 그대로. flagged는 지적·미상 단위 → 감사 표본 순."""
    sel = policy.selection
    if sel.mode == "all":
        return windows
    reasons = set(sel.reasons)
    first: list[PlannedWindow] = []
    audit: list[PlannedWindow] = []
    for w in windows:
        unknown = any(
            isinstance(x.payload, GapPayload) and x.payload.gap_type == "unknown" for x in w.targets
        )
        if _is_flagged(w, flagged, reasons):
            first.append(replace(w, selection="flagged"))
        elif sel.unknown_gaps and unknown:
            first.append(replace(w, selection="unknown_gap"))
        elif _audit_draw(w.session_id, w.window_key, sel.seed) < sel.audit_ratio:
            audit.append(replace(w, selection="audit"))
    return first + audit


def plan_windows(
    session: Session,
    current: Sequence[LabelRecord],
    policy: AiReviewPolicy,
    o: Ontology,
    *,
    flagged: Sequence[FlaggedSpan] = (),
    only: Sequence[LabelRecord] | None = None,
) -> list[PlannedWindow]:
    """세션의 요청 창 (프레임 전). 순서: temporal → temporal_long → 스트림별 spatial.

    선택 모드(`selection.mode`)를 적용한 결과다. 평가 경로(only가 있음)는 선택 모드를 쓰지 않는다:
    지정한 대상을 모두 판정해야 지표 분모가 맞고, 평가에는 우선순위 사유 구간이 없어 flagged면 감사
    표본 창만 남는다 (selection은 판정 버전 밖이라 보고서로 구별할 수 없다).

    Args:
        session: 세션 (길이, 기준 스트림, 영상 스트림).
        current: 현재 운영 라벨 (`current_labels`). 맥락 후보다.
        policy: 정책.
        o: 세션 온톨로지.
        flagged: flagged 모드에서 쓸 우선순위 사유 구간 (CLI가 `flag_unit`으로 계산).
        only: 대상 지정 (평가 경로, `eval_targets`로 거른다). None이면 `targets(current)`.
    """
    found = targets(current, policy, o) if only is None else eval_targets(only, policy, o)
    ctx = _Ctx(
        session=session,
        policy=policy,
        pool=tuple(x for x in current if _operational(x, o)),
        classes=entity_classes(current),
        unresolved=frozenset(policy.unresolved_entity_ids),
    )
    temporal = [x for x in found if x.kind in policy.kinds.temporal]
    spatial = [x for x in found if x.kind in policy.kinds.spatial]
    windows = _temporal_windows(ctx, [x for x in temporal if not _is_long(x, policy)])
    windows += _long_windows(ctx, [x for x in temporal if _is_long(x, policy)])
    for stream in session.streams:
        if stream.kind in VIDEO_KINDS:
            windows += _spatial_windows(ctx, spatial, stream.stream_id)
    return windows if only is not None else _select(windows, policy, flagged)


# ---------------------------------------------------------------- 단위


def unit_id(
    version: str,
    session_id: str,
    w: PlannedWindow,
    frame_ms_: Sequence[int],
    render_hash: str,
    sensor_sha256: str,
    ontology_version: str,
    ontology_digest: str,
) -> str:
    """단위 키 `air-<sha256 24자>`.

    해시 입력: [버전, 세션, 묶음, 스트림, 구간 시작·끝, 분할 순번, 대상 ID(순서), 맥락 ID(정렬),
    프레임 ms, 렌더 해시, 센서 해시, 온톨로지 버전, 온톨로지 내용 해시]의 JSON.
    내용 해시가 들어가므로 이 변경(F-AIR-32) 전에 기록한 단위와 키가 다르다: 한 번 다시 판정한다.

    문맥 초과 분할은 단위 안 일이라 키에 들어가지 않는다.
    """
    payload = [
        version, session_id, w.group.value, w.stream_id, w.t_start_ms, w.t_end_ms, w.split_index,
        [x.label_id for x in w.targets], sorted(x.label_id for x in w.context), list(frame_ms_),
        render_hash, sensor_sha256, ontology_version, ontology_digest,
    ]  # fmt: skip
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return "air-" + hashlib.sha256(text.encode()).hexdigest()[:24]


def unit_sensor_sha(w: PlannedWindow, policy: AiReviewPolicy, sensor: SensorContext) -> str:
    """장갑 문장을 쓰는 단위(시간 묶음, 손 상태 대상, 센서 켬, 장갑 있음)면 센서 해시, 아니면
    "absent"."""
    uses = (
        policy.sensor.enabled
        and sensor.present
        and w.group is not AiGroup.SPATIAL
        and any(isinstance(x.payload, HandStatePayload) for x in w.targets)
    )
    return sensor.sha256 if uses else ABSENT


def plan_session(
    session: Session,
    current: Sequence[LabelRecord],
    policy: AiReviewPolicy,
    o: Ontology,
    *,
    media: Mapping[str, StreamMedia],
    sensor: SensorContext,
    flagged: Sequence[FlaggedSpan] = (),
    only: Sequence[LabelRecord] | None = None,
) -> list[PlannedUnit]:
    """세션의 판정 단위 (`plan_windows` + 프레임 고르기 + 단위 키).

    블러본이 없는 스트림(`media`에 없음)의 창은 계획하지 않는다 (그 대상은 판정되지 않은 채 남는다).

    Args:
        media: 스트림 → 블러본 PTS 인덱스·렌더 해시 (temporal은 기준 스트림을 쓴다).
        sensor: 장갑 접촉 구간 (없으면 `SensorContext.absent()`).
        only: 대상 지정 (평가 경로). (나머지는 `plan_windows`와 같다.)
    """
    version, digest = policy.verdict_version, ontology_digest(o)
    out: list[PlannedUnit] = []
    for w in plan_windows(session, current, policy, o, flagged=flagged, only=only):
        m = media.get(w.stream_id)
        if m is None:
            continue
        idx = tuple(pick_frames(m.pts, w.wanted_ms, nearest=w.group is AiGroup.SPATIAL))
        ms = tuple(frame_ms(m.pts, i) for i in idx)
        sensor_sha = unit_sensor_sha(w, policy, sensor)
        uid = unit_id(
            version, session.session_id, w, ms, m.render_hash, sensor_sha, o.version, digest
        )
        out.append(
            PlannedUnit(w, uid, version, idx, ms, m.render_hash, sensor_sha, o.version, digest)
        )
    return out


def estimate_units(
    session: Session,
    current: Sequence[LabelRecord],
    policy: AiReviewPolicy,
    o: Ontology,
    *,
    flagged: Sequence[FlaggedSpan] = (),
) -> list[UnitEstimate]:
    """영상 없이 단위·대상·프레임 수 상한을 센다 (실행 경로와 같은 창 논리, `aireview estimate`)."""
    return [
        UnitEstimate(w.group, w.stream_id, len(w.targets), len(w.wanted_ms), w.max_side_px)
        for w in plan_windows(session, current, policy, o, flagged=flagged)
    ]


def uncovered_targets(
    windows: Iterable[PlannedWindow],
    units: Iterable[AiReviewUnit],
    *,
    version: str,
    render_hashes: Mapping[str, str],
    ontology_digest: str | None = None,
) -> list[str]:
    """계획 창의 대상 중 판정되지 않은 라벨 (라벨 ID 정렬).

    "판정됨" = 지금 버전이고 그 스트림의 지금 렌더 해시인 단위에 대상으로 든 적이 있다.
    ontology_digest를 주면 그 온톨로지 내용 해시로 기록한 단위만 센다 (F-AIR-32. 내용 해시가 없는
    이전 단위는 세지 않는다: 다시 판정하면 내용 해시가 생긴다).
    `--pending`이 세션을 고르는 조건이다. 맥락 변화는 보지 않는다 (사람 검수가 진행될 때마다 다시
    고르지 않게).
    """
    covered = {
        i
        for u in units
        if u.version == version
        and render_hashes.get(u.stream_id) == u.render_hash
        and (ontology_digest is None or u.ontology_digest == ontology_digest)
        for i in u.target_label_ids
    }
    return sorted({x.label_id for w in windows for x in w.targets} - covered)
