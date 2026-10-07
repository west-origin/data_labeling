"""배정 계획.

검수 단위마다 표준 배정을 두고, 정책 비율대로 블라인드·이중·오류 삽입·QA 배정을 더한다.
- 비율은 config/defaults.yaml review (blind_task_ratio, double_annotation_ratio,
  seeded_error_task_ratio, qa_sample_ratio)이고, 표준 배정 수에 대한 비율이다.
- 뽑기는 (seed, 단위, 방식)의 해시로 정해 같은 입력이면 같은 계획이 나온다 (재실행 멱등).
- 담당자는 배정이 가장 적은 사람. 블라인드·이중은 표준 담당자와 다른 사람에게 준다.
- 오류 삽입 과제는 정답을 아는 단위 풀(골든셋 등)에서 고른다. 표준 배정과 섞여 구별되지 않는다.
- QA는 끝난 표준 배정에서 뽑아 선임에게 준다.

WP12, ADR 0014. `ops.runner.plan_session`(`dlp review plan`)이 `plan`을, `dlp review qa`가
`plan_qa`를 부른다. 이 모듈은 DB를 쓰지 않는다 (배정 저장은 runner가 한다).

공개 이름: `draw`, `assignment_id`, `PlannedUnit`, `Loads`, `plan`, `plan_qa`.

배정 ID 규칙 (재계획 때 같은 ID가 나와야 중복 삽입을 건너뛸 수 있다):
- `<단위 ID>:<방식>[:<세대>]` (표준·블라인드·이중).
- 오류 삽입: `<원천 단위 ID>:seeded_error:<대상 단위 ID의 ':'를 '.'로>[.<세대>]`.
- QA: `<표준 배정 ID>:qa`, 재검수: `<배정 ID>:resample` (runner).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.priority import Unit
from dlp_schema.review import AssignmentStatus, FlaggedSpan, ReviewAssignment, ReviewMode


def draw(seed: int, key: str, mode: str) -> float:
    """[0, 1) 균등 난수 (결정적).

    sha256(`"<seed>:<key>:<mode>"`)의 앞 8바이트를 2^64로 나눈다. key는 단위·배정 ID, mode는
    뽑기 용도("blind", "double", "seeded", "seeded-pick", "qa")라서 용도마다 독립적인 값이 나온다.
    """
    digest = hashlib.sha256(f"{seed}:{key}:{mode}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assignment_id(unit: Unit, mode: ReviewMode, tag: str = "") -> str:
    """배정 ID `"<단위 ID>:<방식>[:<tag>]"`. tag는 세대(블러 단위) 또는 오류 삽입 대상 표시."""
    suffix = f":{tag}" if tag else ""
    return f"{unit.unit_id}:{mode.value}{suffix}"


@dataclass(frozen=True)
class PlannedUnit:
    """계획할 검수 단위와 그 우선순위 정보 (runner가 만든다)."""

    unit: Unit
    # 먼저 볼 구간 (우선순위 사유 + 표본 구간)
    flagged: tuple[FlaggedSpan, ...]
    # 단위 우선순위 (`priority.unit_priority`). 클수록 먼저
    priority: float
    # 표본 검수로 보낼 라벨 ID
    sample_label_ids: tuple[str, ...] = ()
    # 표본에 들지 않아 보내지 않을 높은 신뢰도 라벨 ID (표본 판정 뒤 처리)
    withheld_label_ids: tuple[str, ...] = ()
    # 단위 입력 세대 (블러 단위: 탐지·블러 라벨 집합 해시). 배정 ID에 붙여 재탐지·승인 취소 뒤
    # 다시 계획하면 새 배정이 생기게 한다. 비어 있으면 붙이지 않는다.
    generation: str = ""


class Loads:
    """담당자별 배정 수. 가장 적은 사람(같으면 이름순)을 고른다."""

    def __init__(self, reviewers: Sequence[str], existing: Counter[str] | None = None) -> None:
        """reviewers: 고를 수 있는 담당자. existing: 이미 열린 배정 수 (명단에 없는 사람은 무시)."""
        self.counts: Counter[str] = Counter({r: 0 for r in reviewers})
        self.counts.update(
            {r: n for r, n in (existing or Counter[str]()).items() if r in self.counts}
        )

    def pick(self, exclude: set[str] | None = None) -> str | None:
        """exclude를 뺀 사람 중 배정이 가장 적은 사람을 골라 수를 1 늘린다 (없으면 None)."""
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
    """단위별 배정 목록을 만든다 (DB에 쓰지 않는다).

    인자:
    - planned: 계획할 단위. 우선순위 내림차순(같으면 단위 ID 순)으로 처리해 바쁜 사람 쏠림을 줄인다.
    - reviewers: 담당자 후보.
    - policy: `ratios`(배정 비율)를 쓴다.
    - seed: 뽑기 시드 (같은 seed·같은 입력이면 같은 계획).
    - now: 배정 `created_at` (시간대 필수).
    - seed_pool: 오류 삽입 원천 단위 (정답을 아는 단위). 비면 오류 삽입 배정을 만들지 않는다.
    - loads: 기존 부하 (없으면 0부터). 호출 뒤 갱신된다.

    반환: 표준 배정(단위마다 하나) + 확률적으로 블라인드·이중·오류 삽입 배정.
    블라인드·이중은 `pair_id`로 표준 배정을 가리키고, 다른 담당자를 고를 수 없으면 만들지 않는다.
    오류 삽입 배정의 단위(세션·스트림·종류)는 원천 단위의 것이다 (검수자는 구별하지 못한다).
    """
    ratios = policy.ratios
    loads = loads or Loads(reviewers)
    out: list[ReviewAssignment] = []

    def make(
        unit: Unit, mode: ReviewMode, assignee: str | None, generation: str = "", **kw: object
    ) -> ReviewAssignment:
        """단위·방식으로 배정 하나를 만든다. kw의 `assignment_id`가 있으면 그것을 쓴다."""
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
        # 같은 단위의 측정 배정(블라인드·이중)은 서로 다른 사람에게 준다
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
            # 원천 단위를 결정적으로 고른다 ([0, 1) 난수 * 풀 크기)
            idx = int(draw(seed, unit.unit_id, "seeded-pick") * len(seed_pool))
            source = seed_pool[idx]
            # 배정 ID에 대상 단위 ID를 붙여 같은 원천을 여러 단위가 골라도 ID가 겹치지 않게 한다
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
    """끝난 표준 배정 중 qa_sample_ratio만큼 선임 재검수를 만든다.

    인자:
    - done: 끝난 배정 (표준이 아닌 것과 블러 배정은 건너뛴다).
    - policy: `ratios.qa_sample_ratio`, `reviewers.senior`를 쓴다.
    - seed: 뽑기 시드 (배정 ID별로 결정적).
    - now: 새 QA 배정 `created_at`.

    반환: QA 배정 목록 (ID `<표준 배정 ID>:qa`, pair_id = 원래 배정). 선임 중 원래 담당자가 아닌
    사람에게 고르게 준다. 선임이 없거나 원래 담당자뿐이면 담당자가 None이다. DB에 쓰지 않는다.
    """
    seniors = Loads(policy.reviewers.senior)
    out: list[ReviewAssignment] = []
    for a in sorted(done, key=lambda a: a.assignment_id):
        if a.mode is not ReviewMode.STANDARD or a.label_kinds == ("blur_track",):
            continue  # 블러 검수는 원본 영상을 열어 일반 QA 대상이 아니다 (잔여 누락 감사가 맡는다)
        if draw(seed, a.assignment_id, "qa") >= policy.ratios.qa_sample_ratio:
            continue
        # 원래 배정을 복사해 단위·라벨 범위(only/withheld 등)를 그대로 잇고 상태만 새로 연다.
        # model_copy는 검증하지 않으므로 값은 반드시 enum으로 넣는다. 회귀: 문자열 "open"을 넣어
        # `status is AssignmentStatus.OPEN` 비교가 거짓이 되고 직렬화 경고가 났다.
        out.append(
            a.model_copy(
                update={
                    "assignment_id": f"{a.assignment_id}:qa",
                    "mode": ReviewMode.QA,
                    "assignee": seniors.pick({a.assignee} if a.assignee else None),
                    "pair_id": a.assignment_id,
                    "sample_label_ids": (),
                    "task_key": None,
                    "status": AssignmentStatus.OPEN,
                    "created_at": now,
                    "completed_at": None,
                }
            )
        )
    return out
