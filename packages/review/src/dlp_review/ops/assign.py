"""배정 계획.

검수 단위마다 표준 배정을 두고, 정책 비율대로 블라인드·이중·오류 삽입·QA 배정을 더한다.
- 비율은 config/defaults.yaml review (blind_task_ratio, double_annotation_ratio,
  seeded_error_task_ratio, qa_sample_ratio)이고, 표준 배정 수에 대한 비율이다.
- 뽑기는 (seed, 단위, 방식)의 해시로 정해 같은 입력이면 같은 계획이 나온다 (재실행 멱등).
- 담당자는 배정이 가장 적은 사람. 블라인드·이중은 표준 담당자와 다른 사람에게 준다.
- 오류 삽입 과제는 정답을 아는 단위 풀(골든셋 등)에서 고른다. 표준 배정과 섞여 구별되지 않는다.
- QA는 끝난 표준 배정에서 뽑아 선임에게 준다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit
from dlp_schema.review import FlaggedSpan, ReviewAssignment, ReviewMode


def draw(seed: int, key: str, mode: str) -> float:
    """[0, 1) 균등 난수 (결정적)."""
    digest = hashlib.sha256(f"{seed}:{key}:{mode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assignment_id(unit: Unit, mode: ReviewMode, tag: str = "") -> str:
    suffix = f":{tag}" if tag else ""
    return f"{unit.unit_id}:{mode.value}{suffix}"


@dataclass(frozen=True)
class PlannedUnit:
    unit: Unit
    flagged: tuple[FlaggedSpan, ...]
    priority: float
    sample_label_ids: tuple[str, ...] = ()
    withheld_label_ids: tuple[str, ...] = ()
    # 단위 입력 세대 (블러 단위: 탐지·블러 라벨 집합 해시). 배정 ID에 붙여 재탐지·승인 취소 뒤
    # 다시 계획하면 새 배정이 생기게 한다. 비어 있으면 붙이지 않는다.
    generation: str = ""


class Loads:
    """담당자별 배정 수. 가장 적은 사람(같으면 이름순)을 고른다."""

    def __init__(self, reviewers: Sequence[str], existing: Counter[str] | None = None) -> None:
        self.counts: Counter[str] = Counter({r: 0 for r in reviewers})
        self.counts.update(
            {r: n for r, n in (existing or Counter[str]()).items() if r in self.counts}
        )

    def pick(self, exclude: set[str] | None = None) -> str | None:
        options = [r for r in self.counts if r not in (exclude or set())]
        if not options:
            return None
        chosen = min(options, key=lambda r: (self.counts[r], r))
        self.counts[chosen] += 1
        return chosen


def plan(
    planned: Sequence[PlannedUnit],
    reviewers: Sequence[str],
    policy: ReviewOpsPolicy,
    *,
    seed: int,
    now: datetime,
    seed_pool: Sequence[Unit] = (),
    loads: Loads | None = None,
) -> list[ReviewAssignment]:
    ratios = policy.ratios
    loads = loads or Loads(reviewers)
    out: list[ReviewAssignment] = []

    def make(
        unit: Unit, mode: ReviewMode, assignee: str | None, generation: str = "", **kw: object
    ) -> ReviewAssignment:
        return ReviewAssignment.model_validate(
            {
                "assignment_id": kw.pop("assignment_id", None)
                or assignment_id(unit, mode, generation),
                "session_id": unit.session_id,
                "stream_id": unit.stream_id,
                "label_kinds": unit.kinds,
                "mode": mode,
                "assignee": assignee,
                "created_at": now,
                **kw,
            }
        )

    for p in sorted(planned, key=lambda p: (-p.priority, p.unit.unit_id)):
        unit = p.unit
        standard_to = loads.pick()
        gen = p.generation
        standard = make(
            unit, ReviewMode.STANDARD, standard_to, gen, priority=p.priority, flagged=p.flagged,
            sample_label_ids=p.sample_label_ids, withheld_label_ids=p.withheld_label_ids,
        )  # fmt: skip
        out.append(standard)
        taken = {standard_to} if standard_to else set[str]()
        if draw(seed, unit.unit_id, "blind") < ratios.blind_task_ratio:
            to = loads.pick(taken)
            if to is not None:
                taken.add(to)
                out.append(
                    make(
                        unit,
                        ReviewMode.BLIND,
                        to,
                        gen,
                        priority=p.priority,
                        pair_id=standard.assignment_id,
                    )
                )
        if draw(seed, unit.unit_id, "double") < ratios.double_annotation_ratio:
            to = loads.pick(taken)
            if to is not None:
                out.append(
                    make(
                        unit,
                        ReviewMode.DOUBLE,
                        to,
                        gen,
                        priority=p.priority,
                        pair_id=standard.assignment_id,
                    )
                )
        if seed_pool and draw(seed, unit.unit_id, "seeded") < ratios.seeded_error_task_ratio:
            idx = int(draw(seed, unit.unit_id, "seeded-pick") * len(seed_pool))
            source = seed_pool[idx]
            out.append(
                make(
                    source, ReviewMode.SEEDED_ERROR, loads.pick(), priority=p.priority,
                    assignment_id=assignment_id(
                        source,
                        ReviewMode.SEEDED_ERROR,
                        tag=unit.unit_id.replace(":", ".") + (f".{gen}" if gen else ""),
                    ),
                )
            )  # fmt: skip
    return out


def plan_qa(
    done: Sequence[ReviewAssignment], policy: ReviewOpsPolicy, *, seed: int, now: datetime
) -> list[ReviewAssignment]:
    """끝난 표준 배정 중 qa_sample_ratio만큼 선임 재검수를 만든다."""
    seniors = Loads(policy.reviewers.senior)
    out: list[ReviewAssignment] = []
    for a in sorted(done, key=lambda a: a.assignment_id):
        if a.mode is not ReviewMode.STANDARD or a.label_kinds == ("blur_track",):
            continue  # 블러 검수는 원본 영상을 열어 일반 QA 대상이 아니다 (잔여 누락 감사가 맡는다)
        if draw(seed, a.assignment_id, "qa") >= policy.ratios.qa_sample_ratio:
            continue
        out.append(
            a.model_copy(
                update={
                    "assignment_id": f"{a.assignment_id}:qa",
                    "mode": ReviewMode.QA,
                    "assignee": seniors.pick({a.assignee} if a.assignee else None),
                    "pair_id": a.assignment_id,
                    "sample_label_ids": (),
                    "task_key": None,
                    "status": "open",
                    "created_at": now,
                    "completed_at": None,
                }
            )
        )
    return out
