"""높은 신뢰도 라벨의 표본 검수 (합격 판정 샘플링).

묶음(lot) = 같은 세션·라벨 종류·모델 버전의, 신뢰도가 high_confidence 이상인 미검수 모델 라벨.
묶음마다 max(min_sample, ⌈ratio·N⌉)개를 뽑아 검수한다 (같은 seed면 같은 표본). 표본에서 사람이
고치거나 지운 비율이 max_defect_ratio 이하이면 묶음의 나머지 미검수 라벨을 "표본 검증"으로 둔다.
넘으면 묶음 전체를 다시 검수한다.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from dlp_review.ops.policy import SamplingPolicy
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Source, VerificationState


@dataclass(frozen=True)
class Lot:
    session_id: str
    kind: str
    model_version: str
    label_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.session_id}:{self.kind}:{self.model_version}"


def lots(labels: list[LabelRecord], policy: SamplingPolicy) -> list[Lot]:
    """labels: 현재 운영 라벨."""
    groups: dict[tuple[str, str, str], list[str]] = {}
    for x in labels:
        if (
            x.provenance.source is Source.MODEL
            and x.verification.state is VerificationState.UNREVIEWED
            and x.confidence is not None
            and x.confidence >= policy.high_confidence
        ):
            key = (x.session_id, x.kind, x.provenance.model_version or "")
            groups.setdefault(key, []).append(x.label_id)
    return [Lot(s, k, v, tuple(sorted(ids))) for (s, k, v), ids in sorted(groups.items())]


def sample_size(n: int, policy: SamplingPolicy) -> int:
    return min(n, max(policy.min_sample, math.ceil(policy.ratio * n)))


def draw_sample(lot: Lot, policy: SamplingPolicy, seed: int) -> tuple[str, ...]:
    digest = hashlib.sha256(f"{seed}:{lot.key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    k = sample_size(len(lot.label_ids), policy)
    picked = rng.choice(len(lot.label_ids), size=k, replace=False)
    return tuple(sorted(lot.label_ids[int(i)] for i in picked))


@dataclass(frozen=True)
class SamplingVerdict:
    sampled: int
    defects: int
    pending: int  # 아직 검수하지 않은 표본
    accepted: bool | None  # None: 표본 검수가 끝나지 않음
    to_verify: tuple[str, ...]  # 합격이면 표본 검증으로 둘 라벨


def judge(
    lot_ids: tuple[str, ...],
    sample_ids: tuple[str, ...],
    labels: list[LabelRecord],
    policy: SamplingPolicy,
) -> SamplingVerdict:
    """labels: 세션의 모든 레코드 (이력 포함)."""
    by_id = {x.label_id: x for x in labels}
    corrected = {
        x.parent_label_id
        for x in labels
        if x.parent_label_id and x.provenance.source is Source.HUMAN and x.measurement is None
    }
    current = {x.label_id for x in current_labels(labels)}
    # 검수 전에 다른 단계(모델 재실행)가 지운 표본은 판정에서 뺀다
    kept = [i for i in sample_ids if i in corrected or i in current]
    defects = sum(i in corrected for i in kept)
    approved = sum(
        by_id[i].verification.state is VerificationState.HUMAN_APPROVED
        for i in kept
        if i not in corrected
    )
    pending = len(kept) - defects - approved
    if pending:
        return SamplingVerdict(len(kept), defects, pending, None, ())
    accepted = defects / max(1, len(kept)) <= policy.max_defect_ratio
    rest = tuple(
        i
        for i in lot_ids
        if i not in sample_ids
        and i in by_id
        and i not in corrected
        and i in current
        and by_id[i].verification.state is VerificationState.UNREVIEWED
    )
    return SamplingVerdict(len(kept), defects, 0, accepted, rest if accepted else ())
