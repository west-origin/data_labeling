"""시드가 정해진 라벨 교란 9종 (통제된 오류, WP17.2, ADR 0040 7.4).

골든 평가(`dlp aireview eval --mode perturb`, WP17.8)와 합성 오류 탐지 테스트가 쓴다. 라벨을
메모리에서만 바꾸고 DB에 쓰지 않는다. 바꾼 라벨은 ID를 그대로 두고(정답과 대조하려고) 페이로드만
바꾼다. 거짓 이벤트는 새 ID(`pert-ev-NNNN-<세션>`, 128자에서 자름)로 더한다. 세션을 넣어 여러 골든
세션의 거짓 이벤트가 같은 ID가 되지 않게 한다 (평가가 판정을 라벨 ID로 모으던 때 다른 세션의
판정이 합쳐졌다). 교란한 라벨도 `check_label`을 통과하는 값만 고른다.

교란 종류
    - verb_swap: 행동 동사 (같은 비접촉 여부 안에서)
    - target_swap·tool_swap: 행동 대상·도구 개체
    - hand_swap: 행동 손
    - gap_to_action: 손 있는 사이 구간 → 행동
    - action_to_gap: 행동 → 대기 사이 구간
    - contact_target_swap: 손 상태 접촉 대상 개체
    - class_swap: 박스 트랙 클래스
    - false_event: 행동 중점의 거짓 시점 이벤트

라벨마다 정해진 순서로 종류를 보고 처음 뽑힌 하나만 적용한다. 같은 입력·seed면 같은 결과다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    EventPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    LabelPayload,
    LabelRecord,
)
from dlp_schema.ontology import Ontology, VerbLevel
from dlp_schema.validation import check_label

KINDS = (
    "verb_swap", "target_swap", "tool_swap", "hand_swap", "gap_to_action", "action_to_gap",
    "contact_target_swap", "class_swap", "false_event",
)  # fmt: skip


@dataclass(frozen=True)
class Perturbed:
    """교란 결과.

    Attributes:
        labels: 교란한 라벨 집합 (원래 순서, 거짓 이벤트는 끝에).
        injected: 라벨 ID → 넣은 교란 종류.
        truth: 바꾼 라벨의 원래 레코드 (거짓 이벤트는 없다).
    """

    labels: list[LabelRecord]
    injected: dict[str, str]
    truth: dict[str, LabelRecord]


@dataclass(frozen=True)
class _Pools:
    """교란 후보 값 (정렬해 결정적으로 고른다)."""

    verbs: tuple[str, ...]
    contactless: frozenset[str]
    entities: tuple[str, ...]
    classes: tuple[str, ...]
    events: tuple[str, ...]


def _pools(labels: Sequence[LabelRecord], o: Ontology, entities: Sequence[str] | None) -> _Pools:
    """온톨로지·라벨에서 교란 후보 값을 모은다."""
    refs: set[str] = set(entities or ())
    if entities is None:
        for x in labels:
            p = x.payload
            if isinstance(p, ActionPayload):
                refs |= {r for r in (p.target_id, p.tool_id) if r}
            elif isinstance(p, BoxTrackPayload):
                refs.add(p.entity_id)
    events = tuple(
        e for e, s in sorted(o.events.items())
        if s.form == "point" and not s.raw_access and not s.requires_object and not s.severity
    )  # fmt: skip
    return _Pools(
        verbs=tuple(sorted(v for v, s in o.verbs.items() if s.level is VerbLevel.PRIMITIVE)),
        contactless=frozenset(v for v, s in o.verbs.items() if s.contactless),
        entities=tuple(sorted(refs)),
        classes=tuple(sorted(o.objects)),
        events=events,
    )


def _other(rng: np.random.Generator, values: Sequence[Any], current: Any) -> Any:
    """`current`가 아닌 값 하나 (없으면 None)."""
    choices = [v for v in values if v != current]
    return choices[int(rng.integers(len(choices)))] if choices else None


def _with(label: LabelRecord, payload: LabelPayload) -> LabelRecord:
    """페이로드를 바꾼 사본 (검증을 다시 거친다)."""
    return label.model_validate(label.model_dump() | {"payload": payload.model_dump()})


def _action_edit(
    p: ActionPayload, kind: str, rng: np.random.Generator, pools: _Pools
) -> dict[str, Any] | None:
    """행동 페이로드 교란 (동사·대상·도구·손). 적용할 수 없으면 None."""
    if kind == "verb_swap":
        same = [v for v in pools.verbs if (v in pools.contactless) == (p.verb in pools.contactless)]
        return {"verb": _other(rng, same, p.verb)}
    if kind == "target_swap" and p.target_id:
        return {"target_id": _other(rng, pools.entities, p.target_id)}
    if kind == "tool_swap" and p.tool_id:
        return {"tool_id": _other(rng, pools.entities, p.tool_id)}
    if kind == "hand_swap":
        return {"hand": Hand.LEFT if p.hand is Hand.RIGHT else Hand.RIGHT}
    return None


def _edit(
    label: LabelRecord, kind: str, rng: np.random.Generator, pools: _Pools
) -> LabelRecord | None:
    """라벨 하나에 교란 종류 하나를 적용한 사본 (적용할 수 없으면 None)."""
    p = label.payload
    up: dict[str, Any] | None = None
    if isinstance(p, ActionPayload):
        if kind == "action_to_gap":
            return _with(label, GapPayload(hand=p.hand, gap_type="idle"))
        up = _action_edit(p, kind, rng, pools)
    elif isinstance(p, GapPayload) and kind == "gap_to_action" and p.hand is not None:
        verb = _other(rng, [v for v in pools.verbs if v not in pools.contactless], None)
        action = ActionPayload(
            action_id=f"pert-{label.label_id}"[:128], hand=p.hand, verb=verb,
            t_approach_ms=label.t_start_ms, t_end_ms=label.t_end_ms,
        )  # fmt: skip
        return _with(label, action)
    elif isinstance(p, HandStatePayload) and kind == "contact_target_swap" and p.target_id:
        up = {"target_id": _other(rng, pools.entities, p.target_id)}
    elif isinstance(p, BoxTrackPayload) and kind == "class_swap":
        up = {"class_id": _other(rng, pools.classes, p.class_id)}
    if not up or any(v is None for v in up.values()):
        return None
    return _with(label, type(p).model_validate(p.model_dump() | up))


def _false_event(
    label: LabelRecord, n: int, rng: np.random.Generator, pools: _Pools
) -> LabelRecord | None:
    """행동 중점의 거짓 시점 이벤트 (관련 행동 = 그 행동)."""
    p = label.payload
    if not isinstance(p, ActionPayload) or not pools.events:
        return None
    t = (label.t_start_ms + label.t_end_ms) // 2
    event = EventPayload(event_type=_other(rng, pools.events, None), related_action_id=p.action_id)
    return label.model_validate(
        label.model_dump()
        | {
            "label_id": f"pert-ev-{n:04d}-{label.session_id}"[:128],
            "t_start_ms": t,
            "t_end_ms": t,
            "payload": event.model_dump(),
        }
    )


def perturb(
    labels: Sequence[LabelRecord],
    rates: Mapping[str, float],
    *,
    seed: int,
    o: Ontology,
    entities: Sequence[str] | None = None,
) -> Perturbed:
    """라벨을 교란한다 (모듈 docstring).

    Args:
        labels: 원래 라벨 (보통 골든·정답 라벨).
        rates: 교란 종류 → 라벨마다 적용 확률 (0~1, 없는 종류는 0).
        seed: 난수 seed.
        o: 온톨로지 (교란 값 후보와 검사).
        entities: 대상·도구로 바꿔 넣을 개체 ID (None이면 라벨에서 모은다).

    Raises:
        ValueError: 모르는 교란 종류나 범위 밖 비율.
    """
    unknown = set(rates) - set(KINDS)
    if unknown or any(not 0 <= r <= 1 for r in rates.values()):
        raise ValueError(f"교란 종류·비율 오류: {sorted(unknown) or rates}")
    rng = np.random.default_rng(seed)
    pools = _pools(labels, o, entities)
    out: list[LabelRecord] = []
    added: list[LabelRecord] = []
    injected: dict[str, str] = {}
    truth: dict[str, LabelRecord] = {}
    edits: list[str] = [k for k in KINDS if k != "false_event"]
    for label in labels:
        changed = _first_edit(label, edits, rates, rng, pools, o)
        if changed is not None:
            kind, new = changed
            injected[label.label_id], truth[label.label_id] = kind, label
            out.append(new)
        else:
            out.append(label)
        if rng.random() < rates.get("false_event", 0.0):
            event = _false_event(label, len(added), rng, pools)
            if event is not None and not check_label(event, o):
                added.append(event)
                injected[event.label_id] = "false_event"
    return Perturbed(out + added, injected, truth)


def _first_edit(
    label: LabelRecord,
    edits: Sequence[str],
    rates: Mapping[str, float],
    rng: np.random.Generator,
    pools: _Pools,
    o: Ontology,
) -> tuple[str, LabelRecord] | None:
    """정해진 순서로 종류를 보며 처음 뽑히고 온톨로지 검사를 통과한 교란 하나."""
    draw: Callable[[], float] = rng.random
    for kind in edits:
        if draw() >= rates.get(kind, 0.0):
            continue
        new = _edit(label, kind, rng, pools)
        if new is not None and new.payload != label.payload and not check_label(new, o):
            return kind, new
    return None
