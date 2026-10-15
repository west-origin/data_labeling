"""설명 라벨이 그것이 설명하는 행동·상위 구간의 검수 결과를 따른다 (F-PAR-15, ADR 0043 3절).

설명 라벨(`DescriptionPayload`)은 검수 도구(CVAT·Label Studio)에 경로가 없다. 그래서 모델 설명
(행동 단계의 한국어 원문, 번역 단계의 영어 번역)은 사람이 승인할 수 없어 `dlp review verify`를
막았다. 사용자가 고른 규칙은 "행동을 승인할 때 설명도 같이"다. 이 모듈은 세션 이력만 보고 각
설명이 따라야 할 결과를 정하는 순수 계산이다. DB 적용은 `dlp_schema.db.description_repository`가
맡는다.

주인(owner)
    설명의 `segment_id`와 같은 ID를 가진 행동(`ActionPayload.action_id`)·상위 구간
    (`SegmentPayload.segment_id`) 레코드. 양손 행동처럼 주인 레코드가 여럿일 수 있다. 오류 삽입·
    측정 레코드(`non_operational_ids`)는 주인으로 보지 않는다.

규칙 (현재 운영 라벨 중 모델 출처·미검수 설명마다, `plan_description_follow`)
    1. 주인 ID가 이력에 한 번도 없으면 건드리지 않는다 (주인을 모른다).
    2. 현재 주인이 하나도 없다(검수자·단계가 행동을 지움) → 설명을 지운다.
    3. 현재 주인 중 하나라도 의미가 바뀌었다(`meaning`: 시각 필드를 뺀 페이로드가 비교 기준의 그
       조상과 다르다. 동사·손·대상·도구·방향·전후 상태, 상위 구간의 수준·참조·자식), 또는 비교
       기준에서 나오지 않은 주인이 생겼다 → 설명이 다른 행동을 말하므로 지운다.
    3b. 부분 삭제(F-PAR-16): 비교 기준 레코드 중 지금 현재 후손이 없는 것이 있다 → 의미가 바뀐
       것으로 보고 지운다. 양손 행동에서 한 손 레코드를 지우고 다른 손을 승인하면 "양손 동작"
       설명이 한 손 동작을 말하게 된다.
       고친 행동의 설명은 이 모듈이 다시 만들지 않는다. 사람이 의미를 고친 행동의 한국어 설명은
       `dlp_actions.describe`(`dlp actions describe`)가 행동 VLM으로 다시 만든다 (ADR 0043 3절).
    4. 번역(source_label_id 있음)인데 원문이 현재 라벨이 아니면 건드리지 않는다 (번역 단계가
       낡은 번역으로 지운다).
    5. 현재 주인이 모두 사람이 확인한 상태(승인·표본 검증·사람 수정, 또는 사람 출처)이고 의미가
       같으면(시각만 고침 포함) 설명도 검수된 것으로 기록한다. 하나라도 검수 전이면 기다린다.
       - 상태: 주인 중 하나라도 표본 검증이면 sample_verified, 아니면 human_approved (시각만 고친
         행동의 설명은 내용을 고치지 않았으므로 human_approved).
       - 검수자: 그 상태를 준 주인의 검수자 (여럿이면 가장 늦게 검수한 주인).
       - 시각: max(주인 검수 시각, 설명 생성 시각). 행동 승인 뒤에 만든 번역은 만든 시각에
         물려받는다 (검수 시각이 생성보다 앞서지 않게).

비교 기준 (설명이 설명한 주인 레코드)
    - 행동 단계 원문: 같은 단계 실행이 함께 만든 주인 레코드 = 주인 ID가 같고 모델 버전이 설명과
      같은 운영 레코드 (시각을 보지 않아 같은 ms에 만든 레코드도 정확히 가른다).
    - 다시 만든 설명(`REGENERATED_PREFIX`): 만든 시각에 현재였던 주인 레코드 (사람이 고친 레코드).
    - 번역: 원문(`source_label_id`)의 기준.
    - 기준을 찾지 못하면(이전 형식): 사슬의 처음 레코드와 비교하고 부분 삭제는 보지 않는다.

물려받은 검수만 있는 설명 (규칙 6, 2026-10-15 사용자 결정 "추천안대로", ADR 0043 3절)
    물려받은 승인은 사람이 설명문을 읽은 것이 아니다. 물려받은 뒤 다음 검수 회차에서 주인의 의미가
    다시 바뀌거나(규칙 3) 주인이 지워지거나(규칙 2) 일부만 지워지면(규칙 3b) 그 설명(한국어·영어)은
    보호를 잃고 검수 전 설명처럼 지운다. 승인을 다시 물려주지는 않는다 (이미 검수 상태다).
    `dlp actions describe`가 고친 행동의 설명을 다시 만들고 번역이 따라와 새로 물려받는다.
    "물려받은 검수만 있음"(`inheritance_only`)은 이력으로 다시 계산한다: 모델 출처 설명이고, 상태가
    승인·표본 검증이며, `inherited_owner`가 물려준 주인을 찾는다. 찾지 못하면(직접 검수, 또는 물려준
    뒤 주인 레코드가 제자리에서 다시 검수돼 시각이 바뀐 경우) 보호를 유지한다 (안전한 쪽).

그 밖의 검수된 설명(직접 검수)과 사람 출처 설명(사람이 쓰거나 고친 설명)은 어떤 경우에도 건드리지
않는다. 결과는 결정적이고 멱등이다 (적용한 뒤 다시 계산하면 빈 결과).

출처 감사 (`inherited_owner`)
    설명에는 검수 도구 경로가 없으므로 검수된 모델 설명은 이 규칙으로 물려받았거나 `record_review`로
    직접 검수한 것이다. 물려준 주인은 이력에서 다시 계산한다 (검수자가 같고 시각이 위 규칙과 같은
    주인). DB 열·계약은 바꾸지 않는다.

공개 이름: `FOLLOW_VERSION`, `REGENERATED_PREFIX`, `INHERITED_STATES`, `owner_id`, `meaning`,
`human_confirmed`, `InheritedReview`, `DescriptionFollow`, `plan_description_follow`,
`inherited_owner`, `inheritance_only`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import (
    ActionPayload,
    DescriptionPayload,
    LabelRecord,
    SegmentPayload,
    Source,
    VerificationState,
)

# 이 규칙이 쓰는 삭제 레코드의 출처 버전 (규칙 버전, 정책 값이 없다). 모델 출처라서 저장 검증이
# 사람이 확인한 설명을 지우지 못하게 한 번 더 막는다 (`ProtectedLabelRetractionError`)
FOLLOW_VERSION = "review-follow-1"

# 사람이 고친 행동에 검수 뒤 다시 만든 설명의 모델 버전 접두사 (`dlp_actions.describe`,
# ADR 0043 3절).
# 이 설명의 비교 기준은 만든 시각에 현재였던(고친) 주인 레코드다
REGENERATED_PREFIX = "describe-"

# 설명이 물려받을 수 있는 검수 상태 (`_inherit`가 쓰는 상태. 사람 수정은 물려주지 않는다)
INHERITED_STATES = (VerificationState.HUMAN_APPROVED, VerificationState.SAMPLE_VERIFIED)

# 의미 비교에서 빼는 행동 페이로드의 시각 필드 (경계만 고친 행동의 설명은 그대로 맞다)
TIME_FIELDS = frozenset(
    {"t_approach_ms", "t_contact_start_ms", "t_contact_end_ms", "t_end_ms", "contact_held"}
)


def owner_id(x: LabelRecord) -> str | None:
    """설명이 가리킬 수 있는 ID (행동 action_id, 상위 구간 segment_id). 그 밖의 라벨은 None."""
    p = x.payload
    if isinstance(p, ActionPayload):
        return p.action_id
    if isinstance(p, SegmentPayload):
        return p.segment_id
    return None


def meaning(x: LabelRecord) -> dict[str, Any]:
    """설명이 말하는 내용과 관련된 페이로드 값 (시각 필드를 뺀 페이로드)."""
    return x.payload.model_dump(mode="json", exclude=set(TIME_FIELDS))


@dataclass(frozen=True)
class InheritedReview:
    """설명에 기록할 검수 상태 (`record_review` 인자)와 물려준 주인."""

    label_id: str
    state: VerificationState
    reviewer_id: str
    reviewed_at: datetime
    owner_label_id: str


@dataclass
class DescriptionFollow:
    """설명이 따를 결과. approve는 검수 상태 기록, retract는 지울 설명 레코드(원래 레코드)."""

    approve: list[InheritedReview] = field(default_factory=list[InheritedReview])
    retract: list[LabelRecord] = field(default_factory=list[LabelRecord])


def _root(x: LabelRecord, by_id: dict[str, LabelRecord]) -> LabelRecord:
    """parent 사슬의 처음 레코드 (모델이 설명한 레코드)."""
    seen: set[str] = set()
    while x.parent_label_id and x.parent_label_id in by_id and x.label_id not in seen:
        seen.add(x.label_id)
        x = by_id[x.parent_label_id]
    return x


def human_confirmed(x: LabelRecord) -> bool:
    """사람이 확인한 주인인가 (검수자·시각이 있어야 물려줄 수 있다)."""
    v = x.verification
    if v.reviewer_id is None or v.reviewed_at is None:
        return False
    return x.provenance.source is Source.HUMAN or v.state is not VerificationState.UNREVIEWED


def _inherit(d: LabelRecord, owners: Sequence[LabelRecord]) -> InheritedReview | None:
    """주인이 모두 확인됐으면 설명이 물려받을 검수 (모듈 docstring 규칙 5), 아니면 None."""
    if not owners or not all(human_confirmed(o) for o in owners):
        return None
    sampled = [o for o in owners if o.verification.state is VerificationState.SAMPLE_VERIFIED]
    state = VerificationState.SAMPLE_VERIFIED if sampled else VerificationState.HUMAN_APPROVED
    pool = sampled or list(owners)
    # 여럿이면 가장 늦게 검수한 주인 (같으면 라벨 ID 큰 것, 결정적)
    giver = max(pool, key=lambda o: (o.verification.reviewed_at, o.label_id))
    reviewer, at = giver.verification.reviewer_id, giver.verification.reviewed_at
    assert reviewer is not None and at is not None  # human_confirmed가 보장한다
    return InheritedReview(d.label_id, state, reviewer, max(at, d.created_at), giver.label_id)


def _chain(x: LabelRecord, by_id: dict[str, LabelRecord]) -> list[LabelRecord]:
    """x에서 parent를 따라 올라간 사슬 (x 먼저, 처음 레코드가 마지막)."""
    out = [x]
    seen = {x.label_id}
    while x.parent_label_id and x.parent_label_id in by_id and x.parent_label_id not in seen:
        x = by_id[x.parent_label_id]
        seen.add(x.label_id)
        out.append(x)
    return out


class _Baseline:
    """설명이 설명한 주인 레코드 (규칙 3·3b의 비교 기준, 모듈 docstring "비교 기준").

    기준을 찾지 못하면(이전 형식 등) None: 사슬의 처음 레코드와 비교하고 부분 삭제는 보지 않는다.
    """

    def __init__(self, labels: list[LabelRecord], by_id: dict[str, LabelRecord]) -> None:
        self.operational, self.by_id = labels, by_id
        self._at: dict[datetime, list[LabelRecord]] = {}

    def _current_at(self, when: datetime) -> list[LabelRecord]:
        """when 시점의 현재 운영 라벨 (시각별로 한 번만 계산한다)."""
        found = self._at.get(when)
        if found is None:
            found = current_labels([x for x in self.operational if x.created_at <= when])
            self._at[when] = found
        return found

    def of(self, d: LabelRecord, p: DescriptionPayload) -> dict[str, LabelRecord] | None:
        """설명 d가 설명한 주인 레코드 (라벨 ID → 레코드). 모르면 None."""
        source = self.by_id.get(p.source_label_id) if p.source_label_id is not None else None
        ref = source or d
        version = ref.provenance.model_version
        if version is None:
            return None
        if version.startswith(REGENERATED_PREFIX):
            pool = self._current_at(ref.created_at)
        else:
            pool = [
                x
                for x in self.operational
                if x.provenance.model_version == version and not x.retracted
            ]
        found = {x.label_id: x for x in pool if owner_id(x) == p.segment_id}
        return found or None


def _changed(
    now: Sequence[LabelRecord], base: dict[str, LabelRecord] | None, by_id: dict[str, LabelRecord]
) -> bool:
    """규칙 3·3b: 현재 주인의 의미가 기준과 다르거나, 기준에서 나오지 않은 주인이 생겼거나, 기준
    레코드 중 현재 후손이 없는 것이 있다 (부분 삭제)."""
    if base is None:
        return any(meaning(o) != meaning(_root(o, by_id)) for o in now)
    reached: set[str] = set()
    for o in now:
        ancestor = next((a for a in _chain(o, by_id) if a.label_id in base), None)
        if ancestor is None or meaning(o) != meaning(ancestor):
            return True
        reached.add(ancestor.label_id)
    return bool(set(base) - reached)


def _must_retract(
    d: LabelRecord,
    p: DescriptionPayload,
    now: Sequence[LabelRecord],
    by_id: dict[str, LabelRecord],
    baseline: _Baseline,
) -> bool:
    """규칙 2·3·3b: 현재 주인이 없거나, 의미가 바뀌었거나, 일부 가지가 지워졌다."""
    return not now or _changed(now, baseline.of(d, p), by_id)


def plan_description_follow(history: Sequence[LabelRecord]) -> DescriptionFollow:
    """세션 이력에서 설명이 따를 결과를 정한다 (모듈 docstring 규칙). 순수 함수, DB를 쓰지 않는다.

    Args:
        history: 세션의 라벨 이력 전체 (`get_labels`, 종류를 거르지 않는다).

    Returns:
        `DescriptionFollow`. 입력 순서를 따른다.
    """
    labels = list(history)
    excluded = non_operational_ids(labels)
    by_id = {x.label_id: x for x in labels}
    current = current_labels(labels)
    current_ids = {x.label_id for x in current}
    ever = {k for x in labels if x.label_id not in excluded and (k := owner_id(x)) is not None}
    owners: dict[str, list[LabelRecord]] = defaultdict(list)
    for x in current:
        if (k := owner_id(x)) is not None:
            owners[k].append(x)
    operational = [x for x in labels if x.label_id not in excluded]
    baseline = _Baseline(operational, by_id)
    givers = _givers(operational)
    out = DescriptionFollow()
    for d in current:
        p = d.payload
        if not isinstance(p, DescriptionPayload) or p.segment_id not in ever:
            continue
        if d.provenance.source is Source.HUMAN:
            continue
        reviewed = d.verification.state is not VerificationState.UNREVIEWED
        if reviewed and _inherited_from(d, givers) is None:
            continue  # 직접 검수했거나 물려준 주인을 다시 계산할 수 없다 (보호 유지)
        now = owners.get(p.segment_id, [])
        if _must_retract(d, p, now, by_id, baseline):
            out.retract.append(d)  # 규칙 2·3·3b, 물려받은 검수만 있는 설명도 (규칙 6)
            continue
        if reviewed or (p.source_label_id is not None and p.source_label_id not in current_ids):
            continue
        review = _inherit(d, now)
        if review is not None:
            out.approve.append(review)
    return out


def _givers(operational: Sequence[LabelRecord]) -> dict[str, list[LabelRecord]]:
    """주인 ID → 검수를 물려줄 수 있는 운영 주인 레코드 (현재가 아니어도 된다, 검수 시각 있음)."""
    out: dict[str, list[LabelRecord]] = defaultdict(list)
    for o in operational:
        if (k := owner_id(o)) is not None and o.verification.reviewed_at is not None:
            out[k].append(o)
    return out


def _inherited_from(d: LabelRecord, givers: dict[str, list[LabelRecord]]) -> LabelRecord | None:
    """`inherited_owner`의 계산 (주인 후보를 미리 모은 판)."""
    p, v = d.payload, d.verification
    if not isinstance(p, DescriptionPayload) or d.provenance.source is Source.HUMAN:
        return None
    if v.state not in INHERITED_STATES or v.reviewed_at is None:
        return None
    found: list[LabelRecord] = []
    for o in givers.get(p.segment_id, []):
        at = o.verification.reviewed_at
        if at is None or o.verification.reviewer_id != v.reviewer_id:
            continue
        if max(at, d.created_at) == v.reviewed_at:
            found.append(o)
    if not found:
        return None
    return max(found, key=lambda o: (o.verification.reviewed_at, o.label_id))


def inherited_owner(d: LabelRecord, history: Sequence[LabelRecord]) -> LabelRecord | None:
    """검수된 모델 설명에 검수를 물려준 주인 레코드 (감사용). 물려받은 것이 아니면 None.

    주인 후보는 설명의 segment_id를 가진 운영 레코드(현재가 아니어도 된다) 중 검수자가 같고
    `max(주인 검수 시각, 설명 생성 시각)`이 설명의 검수 시각과 같은 것이다. 여럿이면 가장 늦게
    검수한 주인. 설명의 상태가 물려받을 수 있는 상태(`INHERITED_STATES`)가 아니면 None.
    """
    labels = list(history)
    excluded = non_operational_ids(labels)
    return _inherited_from(d, _givers([x for x in labels if x.label_id not in excluded]))


def inheritance_only(d: LabelRecord, history: Sequence[LabelRecord]) -> bool:
    """d의 보호가 물려받은 검수뿐인가 (규칙 6: 주인이 다시 바뀌면 설명 따르기가 지울 수 있다).

    모델 출처 설명이고, 승인·표본 검증 상태이며, 물려준 주인을 이력에서 다시 계산할 수 있을 때만
    참이다. 사람 출처·사람 수정 상태·직접 검수한 설명은 거짓(보호 유지).
    """
    return inherited_owner(d, history) is not None
