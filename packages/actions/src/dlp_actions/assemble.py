"""분류된 후보 구간을 행동·사이 구간 라벨로 만든다.

- 인접한 같은 분류(같은 동사·대상·도구, 또는 같은 사이 구간 종류)는 병합한다.
- 행동 구간: 접근 시작 = 구간 시작, 종료 = 구간 끝. 접촉 시작·종료는 접촉 신호가 구간 안에서 바뀌는
  시각이다. 구간이 접촉 중에 시작하면 접촉 유지(contact_held)다.
- 행동이 아닌 구간은 대기·미상·범위 외로 채운다. 후보 구간이 전체를 나누므로 타임라인에 공백이 없다.
- VLM 설명은 행동마다 description 레코드로 둔다 (검수자가 고친다).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from dlp_actions.vlm import Classified
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    Evidence,
    GapPayload,
    Hand,
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
)


def version_tag(model_version: str) -> str:
    """버전마다 다른 짧은 ID 조각 (정책·VLM이 바뀌면 새 레코드 ID)."""
    return hashlib.sha256(model_version.encode()).hexdigest()[:8]


@dataclass
class Span:
    start_ms: int
    end_ms: int
    key: tuple[str, str | None, str | None, str | None]  # (label, verb|gap_type, target, tool)
    confidence: float
    description: str | None
    fallback: bool


def _key(c: Classified) -> tuple[str, str | None, str | None, str | None]:
    a = c.answer
    if a.label == "action":
        return ("action", a.verb, a.target_id, a.tool_id)
    return ("gap", a.gap_type, None, None)


def merge_spans(
    items: list[Classified],
    default_confidence: float,
    contacts: list[tuple[int, int]],
    cut_reasons: dict[int, str],
) -> list[Span]:
    """같은 분류의 인접 구간을 병합한다.

    단, 병합한 구간 안에 접촉이 둘 이상이면 같은 대상에 같은 행동을 연달아 한 것이므로, 접촉 사이의
    경계 후보(속도 골짜기 우선, 없으면 가운데에 가까운 것)에서 다시 나눈다.
    """
    pieces = sorted(items, key=lambda c: c.request.start_ms)
    out: list[Span] = []
    for c in pieces:
        conf = c.answer.confidence if c.answer.confidence is not None else default_confidence
        key = _key(c)
        if (
            out
            and out[-1].key == key
            and out[-1].end_ms == c.request.start_ms
            and not _splits(
                out[-1].start_ms, c.request.start_ms, c.request.end_ms, contacts, cut_reasons
            )
        ):
            last = out[-1]
            last.end_ms = c.request.end_ms
            last.confidence = min(last.confidence, conf)
            last.description = last.description or c.answer.description
            last.fallback = last.fallback or c.fallback
        else:
            out.append(
                Span(
                    c.request.start_ms,
                    c.request.end_ms,
                    key,
                    conf,
                    c.answer.description,
                    c.fallback,
                )
            )
    return out


def _splits(
    span_start: int,
    cut: int,
    piece_end: int,
    contacts: list[tuple[int, int]],
    reasons: dict[int, str],
) -> bool:
    """cut에서 나눠야 하는가.

    왼쪽에서 끝난 접촉과 오른쪽 이후 시작하는 접촉 사이의 대표 경계이면 나눈다.
    """
    ended = [e for _, e in contacts if span_start < e <= cut]
    if not ended:
        return False
    gap_start = max(ended)
    nxt = [s for s, _ in contacts if s >= cut]
    if not nxt:
        return False
    gap_end = min(nxt)
    inside = sorted(t for t in reasons if gap_start < t < gap_end)
    if not inside:
        return cut == gap_end
    valleys = [t for t in inside if reasons[t] == "valley"]
    mid = (gap_start + gap_end) / 2
    chosen = min(valleys or inside, key=lambda t: abs(t - mid))
    return cut == chosen


def _contact_times(
    start: int, end: int, contacts: list[tuple[int, int]]
) -> tuple[int | None, int | None, bool]:
    """(접촉 시작, 접촉 종료, 접촉 유지). 구간 안에서 바뀌는 시각만 쓴다.

    구간이 접촉 중에 시작하면 접촉 유지이고 접촉 시작은 앞 행동에 있다. 구간 안에 접촉이 여러 번
    있으면 첫 시작과 마지막 종료를 쓴다.
    """
    held = any(s < start < e for s, e in contacts)
    starts = [s for s, _ in contacts if start <= s < end]
    c_start = None if held else (min(starts) if starts else None)
    ends = [e for _, e in contacts if start < e <= end and (c_start is None or e >= c_start)]
    c_end = max(ends) if ends and (held or c_start is not None) else None
    return c_start, c_end, held


def to_labels(
    spans: list[Span],
    *,
    session_id: str,
    hand: Hand,
    contacts: list[tuple[int, int]],
    ontology_version: str,
    model_version: str,
    now: datetime,
) -> list[LabelRecord]:
    out: list[LabelRecord] = []

    def record(payload: LabelPayload, s: Span, suffix: str, evidence: Evidence) -> LabelRecord:
        return LabelRecord(
            label_id=f"{session_id}-{hand.value}-{version_tag(model_version)}-{s.start_ms}-{suffix}",
            session_id=session_id,
            t_start_ms=s.start_ms,
            t_end_ms=s.end_ms,
            ontology_version=ontology_version,
            provenance=Provenance(source=Source.MODEL, model_version=model_version),
            evidence=evidence,
            confidence=round(s.confidence, 4),
            created_at=now,
            payload=payload,
        )

    for s in spans:
        label, value, target, tool = s.key
        if label == "gap":
            out.append(
                record(
                    GapPayload(hand=hand, gap_type=value or "unknown"), s, "gap", Evidence.INFERRED
                )
            )
            continue
        c_start, c_end, held = _contact_times(s.start_ms, s.end_ms, contacts)
        action_id = f"{session_id}-{hand.value}-{version_tag(model_version)}-a{s.start_ms}"
        payload = ActionPayload(
            action_id=action_id,
            hand=hand,
            verb=value or "",
            target_id=target,
            tool_id=tool,
            t_approach_ms=s.start_ms,
            t_contact_start_ms=c_start,
            t_contact_end_ms=c_end,
            t_end_ms=s.end_ms,
            contact_held=held,
        )
        out.append(record(payload, s, "action", Evidence.OBSERVED))
        if s.description:
            text = DescriptionPayload(segment_id=action_id, text=s.description)
            out.append(record(text, s, "desc", Evidence.INFERRED))
    return out
