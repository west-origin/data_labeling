"""클래스별 검수자 수정률.

클래스 키는 "라벨 종류/클래스"다 (예: box_track/cup, action/fold_sheet). 개별 검수(사람 승인·수정)된
모델 라벨과 사람이 추가한 라벨만 센다. 검수 수가 적은 클래스는 전체 수정률 쪽으로 당긴다
(베이즈 평활).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from dlp_active.policy import ActivePolicy
from dlp_schema.history import label_class, review_changes
from dlp_schema.labels import LabelRecord, VerificationState

# 개별 검수 상태. 표본 검증은 표본만 본 것이라 세지 않는다
REVIEWED = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)


def class_key(label: LabelRecord) -> str:
    return f"{label.kind}/{label_class(label)}"


@dataclass(frozen=True)
class ClassRate:
    reviewed: int
    changed: int
    rate: float  # 평활한 수정률


@dataclass(frozen=True)
class RateTable:
    overall: float
    classes: dict[str, ClassRate]

    def rate(self, key: str) -> float:
        """처음 보는 클래스는 전체 수정률 (아직 모르는 것)."""
        c = self.classes.get(key)
        return c.rate if c is not None else self.overall


def correction_rates(histories: Iterable[list[LabelRecord]], policy: ActivePolicy) -> RateTable:
    """세션별 라벨 이력들 → 클래스별 수정률."""
    reviewed: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    for history in histories:
        for c in review_changes(history, REVIEWED):
            if c.label.kind in policy.excluded_kinds:
                continue
            if c.change == "added" and not policy.correction.count_added:
                continue
            key = class_key(c.label)
            reviewed[key] += 1
            changed[key] += c.change != "accepted"
    total = sum(reviewed.values())
    overall = sum(changed.values()) / total if total else 0.0
    w = policy.correction.prior_weight
    classes = {
        k: ClassRate(n, changed[k], (changed[k] + w * overall) / (n + w) if n + w else overall)
        for k, n in sorted(reviewed.items())
    }
    return RateTable(overall, classes)
