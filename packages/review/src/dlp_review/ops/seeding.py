"""오류 삽입 과제: 정답을 아는 단위의 라벨 사본에 오류를 넣고, 검수 뒤 발견 여부를 판정한다.

- 사본은 모두 seeded_error=True이고 ID가 "seed-<배정>-" 로 시작한다. 원래 라벨을 parent로
  가리키지 않으므로 운영 라벨 이력을 건드리지 않는다. 검수자가 고친 레코드도 오류 삽입 계보라
  학습에서 빠진다.
- 오류 종류
  - boundary_shift: 시간 구간의 시작을 앞으로(또는 끝을 뒤로) boundary_shift_ms 범위만큼 옮긴다.
  - class_swap: 박스·마스크 클래스나 행동 동사를 같은 종류의 다른 값으로 바꾼다.
  - blur_deletion: 블러 트랙 하나를 사본에서 뺀다.
- 발견 판정
  - boundary_shift: 사본을 고친 레코드의 해당 경계가 원래 값에서 detect_tolerance_ms 안
  - class_swap: 사본을 고친 레코드의 클래스·동사가 원래 값
  - blur_deletion: 검수자가 새로 그린 블러 트랙이 원래 트랙과 시간이 절반 이상 겹치고 대상이 같음
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dlp_review.ops.policy import ErrorType, SeedingPolicy
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    GapPayload,
    HandStatePayload,
    LabelRecord,
    MaskTrackPayload,
    ObjectStatePayload,
    RelationPayload,
    SegmentPayload,
    Source,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import InjectedError

INTERVAL_TYPES = (
    ActionPayload,
    HandStatePayload,
    GapPayload,
    SegmentPayload,
    ObjectStatePayload,
    RelationPayload,
)


def seed_prefix(assignment_id: str) -> str:
    return f"seed-{assignment_id}-"


def _rng(seed: int, assignment_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{assignment_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _candidates(x: LabelRecord, error: ErrorType) -> bool:
    p = x.payload
    if error == "boundary_shift":
        return isinstance(p, INTERVAL_TYPES)
    if error == "class_swap":
        return isinstance(p, BoxTrackPayload | MaskTrackPayload | ActionPayload)
    return isinstance(p, BlurTrackPayload)


def _shift(x: LabelRecord, delta: int) -> tuple[LabelRecord, dict[str, str | int | float]]:
    """시작을 delta만큼 앞으로 (0 아래로 못 가면 끝을 뒤로)."""
    if x.t_start_ms - delta >= 0:
        edge, start, end = "start", x.t_start_ms - delta, x.t_end_ms
    else:
        edge, start, end = "end", x.t_start_ms, x.t_end_ms + delta
    update: dict[str, object] = {"t_start_ms": start, "t_end_ms": end}
    p = x.payload
    if isinstance(p, ActionPayload):
        update["payload"] = p.model_copy(update={"t_approach_ms": start, "t_end_ms": end})
    original = x.t_start_ms if edge == "start" else x.t_end_ms
    return x.model_copy(update=update), {"edge": edge, "original_ms": original, "shift_ms": delta}


def _swap(
    x: LabelRecord, ontology: Ontology, rng: np.random.Generator
) -> tuple[LabelRecord, dict[str, str | int | float]]:
    p = x.payload
    if isinstance(p, ActionPayload):
        options = sorted(
            v for v, s in ontology.verbs.items() if s.level == "primitive" and v != p.verb
        )
        new = options[int(rng.integers(len(options)))]
        return x.model_copy(update={"payload": p.model_copy(update={"verb": new})}), {
            "field": "verb", "original": p.verb, "seeded": new,
        }  # fmt: skip
    assert isinstance(p, BoxTrackPayload | MaskTrackPayload)
    options = sorted(c for c in ontology.objects if c != p.class_id)
    new = options[int(rng.integers(len(options)))]
    return x.model_copy(update={"payload": p.model_copy(update={"class_id": new})}), {
        "field": "class_id", "original": p.class_id, "seeded": new,
    }  # fmt: skip


@dataclass
class SeededTask:
    labels: list[LabelRecord]  # 검수자에게 보낼 사본 (오류 포함)
    injected: list[InjectedError]


def seed_labels(
    truth: list[LabelRecord],
    *,
    assignment_id: str,
    ontology: Ontology,
    policy: SeedingPolicy,
    seed: int,
    now: datetime,
) -> SeededTask:
    rng = _rng(seed, assignment_id)
    prefix = seed_prefix(assignment_id)
    ordered = sorted(truth, key=lambda x: x.label_id)
    copies = {
        x.label_id: x.model_copy(
            update={
                "label_id": f"{prefix}{i:04d}", "parent_label_id": None, "retracted": False,
                "seeded_error": True, "created_at": now,
            }
        )
        for i, x in enumerate(ordered)
    }  # fmt: skip
    injected: list[InjectedError] = []
    used: set[str] = set()
    types: list[ErrorType] = [t for t in policy.types if any(_candidates(x, t) for x in ordered)]
    for n in range(policy.errors_per_task):
        if not types:
            break
        error = types[n % len(types)]
        pool = [x for x in ordered if _candidates(x, error) and x.label_id not in used]
        if not pool:
            continue
        original = pool[int(rng.integers(len(pool)))]
        used.add(original.label_id)
        copy = copies[original.label_id]
        if error == "blur_deletion":
            del copies[original.label_id]
            injected.append(InjectedError(error_type=error, original_label_id=original.label_id))
            continue
        if error == "boundary_shift":
            lo, hi = policy.boundary_shift_ms
            modified, detail = _shift(copy, int(rng.integers(lo, hi + 1)))
        else:
            modified, detail = _swap(copy, ontology, rng)
        copies[original.label_id] = modified
        injected.append(
            InjectedError(
                error_type=error, original_label_id=original.label_id,
                seeded_label_id=copy.label_id, detail=detail,
            )
        )  # fmt: skip
    return SeededTask(sorted(copies.values(), key=lambda x: x.label_id), injected)


def _overlap_ratio(a: LabelRecord, b: LabelRecord) -> float:
    inter = max(0, min(a.t_end_ms, b.t_end_ms) - max(a.t_start_ms, b.t_start_ms))
    return inter / max(1, a.t_end_ms - a.t_start_ms)


def detected(
    error: InjectedError,
    labels: list[LabelRecord],
    reviewer_id: str | None,
    tolerance_ms: int,
) -> bool:
    """검수 결과(사람 출처, 오류 삽입 계보)를 보고 발견했는지 판정한다.

    labels: 세션의 모든 레코드.
    """
    by_id = {x.label_id: x for x in labels}
    original = by_id.get(error.original_label_id)
    if original is None:
        return False
    human = [x for x in labels if x.provenance.source is Source.HUMAN and x.seeded_error]
    if reviewer_id is not None:
        human = [x for x in human if x.verification.reviewer_id in (None, reviewer_id)]
    if error.error_type == "blur_deletion":
        p = original.payload
        assert isinstance(p, BlurTrackPayload)
        return any(
            x.parent_label_id is None
            and isinstance(x.payload, BlurTrackPayload)
            and x.payload.target == p.target
            and _overlap_ratio(original, x) >= 0.5
            for x in human
        )
    fixes = [x for x in human if x.parent_label_id == error.seeded_label_id and not x.retracted]
    for fix in fixes:
        if error.error_type == "boundary_shift":
            edge = error.detail.get("edge")
            got = fix.t_start_ms if edge == "start" else fix.t_end_ms
            if abs(got - int(error.detail["original_ms"])) <= tolerance_ms:
                return True
        else:
            field = str(error.detail["field"])
            if getattr(fix.payload, field, None) == error.detail["original"]:
                return True
    return False
