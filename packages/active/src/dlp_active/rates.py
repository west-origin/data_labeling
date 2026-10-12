"""클래스별 검수자 수정률.

클래스 키는 "라벨 종류/클래스"다 (예: box_track/cup, action/fold_sheet). 개별 검수(사람
승인·수정)된 모델 라벨, 사람이 지운 모델 라벨, 사람이 추가한 라벨만 센다 (표본 검증·센서 라벨은
세지 않는다). 검수 수가 적은 클래스는 전체 수정률 쪽으로 당긴다 (베이즈 평활).

어느 클래스로 세는가: 모델이 낸 것(승인·수정·삭제)은 모델 원본(`ReviewChange.origin`, 수정 사슬의
맨 처음 모델 레코드)의 클래스, 사람이 추가한 것(원본 없음)은 그 라벨의 클래스다. 모델이 cup으로 낸
것을 사람이 mug로 고치면 cup의 수정으로 센다 (F-ETA-06: 예전에는 고친 뒤 클래스 mug로 세어 cup
수정률이 0이었다). 점수(`terms.CorrectionRateTerm`)가 검수 대기 모델 라벨의 예측 클래스로 수정률을
찾으므로 같은 기준이어야 한다.

수식 (active.yaml `correction`):
- 검수 수 n_c = 그 클래스의 accepted + corrected + deleted (+ added, `count_added`면)
- 바뀐 수 k_c = n_c - accepted
- 전체 수정률 p = Σk / Σn
- 평활 수정률 r_c = (k_c + w * p) / (n_c + w), w = `prior_weight`
처음 보는 클래스는 p를 쓴다.

검수 결과 분류(accepted·corrected·added·deleted)는 `dlp_schema.history.review_changes`가 정한다
(오류 삽입·측정 레코드는 거기서 빠진다). 블러 등 `excluded_kinds`는 여기서 뺀다.
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
    """수정률을 세는 단위 "종류/클래스" (예: "box_track/cup", "keypoint_track/hand21/right")."""
    return f"{label.kind}/{label_class(label)}"


@dataclass(frozen=True)
class ClassRate:
    """한 클래스의 수정률."""

    reviewed: int  # 개별 검수 수 n_c
    changed: int  # 바뀐 수 k_c (고침·지움·추가)
    rate: float  # 평활한 수정률


@dataclass(frozen=True)
class RateTable:
    """클래스별 수정률 표."""

    overall: float  # 전체 수정률 p (검수가 하나도 없으면 0)
    classes: dict[str, ClassRate]  # 클래스 키 → 수정률 (키 순)

    def rate(self, key: str) -> float:
        """처음 보는 클래스는 전체 수정률 (아직 모르는 것)."""
        c = self.classes.get(key)
        return c.rate if c is not None else self.overall


def correction_rates(histories: Iterable[list[LabelRecord]], policy: ActivePolicy) -> RateTable:
    """세션별 라벨 이력들 → 클래스별 수정률.

    검수 결과(`review_changes`)마다 모델 원본(origin, 없으면 그 라벨)의 클래스 키에 검수 1을 더하고,
    accepted가 아니면 바뀜 1을 더한다.

    Args:
        histories: 세션마다 전체 라벨 이력 (`get_labels`). 골든셋 세션은 호출자가 빼야 한다
            (사람이 처음부터 만든 정답이 "추가"로 세어져 수정률이 부풀려진다).
        policy: 액티브 러닝 정책 (`correction`, `excluded_kinds`).
    """
    reviewed: Counter[str] = Counter()
    changed: Counter[str] = Counter()
    for history in histories:
        for c in review_changes(history, REVIEWED):
            # 모델 원본의 클래스로 센다 (사람이 추가한 것은 원본이 없어 그 라벨로, F-ETA-06)
            counted = c.origin or c.label
            if counted.kind in policy.excluded_kinds:
                continue
            if c.change == "added" and not policy.correction.count_added:
                continue
            key = class_key(counted)
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
