"""높은 신뢰도 라벨의 표본 검수 (합격 판정 샘플링).

묶음(lot) = 같은 세션·라벨 종류·모델 버전의, 신뢰도가 high_confidence 이상인 미검수 모델 라벨.
묶음마다 max(min_sample, ⌈ratio·N⌉)개를 뽑아 검수한다 (같은 seed면 같은 표본). 표본에서 사람이
고치거나 지운 비율이 max_defect_ratio 이하이면 묶음의 나머지 미검수 라벨을 "표본 검증"으로 둔다.
넘으면(또는 검수 전에 표본이 모두 지워져 판정할 표본이 없으면) 묶음 전체를 다시 검수한다.

WP12, ADR 0014. 정책은 `config/policies/review.yaml` `sampling` 절.

흐름:
1. 계획(`ops.runner.plan_session`): `lots`로 묶음을 만들고 `draw_sample`로 표본을 뽑아 배정의
   `sample_label_ids`(검수에 보냄)와 `withheld_label_ids`(보내지 않음)에 나눠 둔다.
2. 마무리(`ops.runner.finish_assignment`): 수집이 끝나면 `judge`로 합격 여부를 정한다.
   합격이면 나머지를 `sample_verified`로 기록, 불합격이면 나머지만 다시 보는 재검수 배정을 만든다.

공개 이름: `Lot`, `lots`, `sample_size`, `draw_sample`, `SamplingVerdict`, `judge`.
모두 순수 함수다 (DB를 쓰지 않는다).
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
    """표본 검수 묶음. 같은 세션·종류·모델 버전의 높은 신뢰도 미검수 모델 라벨."""

    session_id: str
    # 라벨 종류 (LabelRecord.kind)
    kind: str
    # 모델 버전 (provenance.model_version, 없으면 빈 문자열)
    model_version: str
    # 묶음에 든 라벨 ID (정렬됨)
    label_ids: tuple[str, ...]

    @property
    def key(self) -> str:
        """묶음 식별 문자열 `"<세션>:<종류>:<모델 버전>"` (표본 난수 시드에 쓴다)."""
        return f"{self.session_id}:{self.kind}:{self.model_version}"


def lots(labels: list[LabelRecord], policy: SamplingPolicy) -> list[Lot]:
    """labels: 현재 운영 라벨.

    반환: 묶음 목록 (세션·종류·모델 버전 순으로 정렬, 묶음 안 ID도 정렬 → 결정적).
    묶음에 드는 라벨: 모델 출처, 미검수(unreviewed), 신뢰도가 있고 `policy.high_confidence` 이상.
    """
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
    """묶음 크기 n에서 뽑을 표본 수 = min(n, max(min_sample, ⌈ratio·n⌉))."""
    return min(n, max(policy.min_sample, math.ceil(policy.ratio * n)))


def draw_sample(lot: Lot, policy: SamplingPolicy, seed: int) -> tuple[str, ...]:
    """묶음에서 표본 라벨 ID를 비복원으로 뽑는다 (정렬해 돌려준다).

    시드는 sha256(`"<seed>:<lot.key>"`)의 앞 8바이트라서 같은 seed·같은 묶음이면 늘 같은 표본이다
    (재계획 멱등). 묶음마다 시드가 달라 묶음끼리 표본 위치가 겹치지 않는다.
    """
    digest = hashlib.sha256(f"{seed}:{lot.key}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    k = sample_size(len(lot.label_ids), policy)
    picked = rng.choice(len(lot.label_ids), size=k, replace=False)
    return tuple(sorted(lot.label_ids[int(i)] for i in picked))


@dataclass(frozen=True)
class SamplingVerdict:
    """표본 판정 결과."""

    # 판정에 쓴 표본 수 (검수 전에 다른 단계가 지운 표본은 뺀다)
    sampled: int
    # 사람이 고치거나 지운 표본 수
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
    """labels: 세션의 모든 레코드 (이력 포함).

    인자:
    - lot_ids: 묶음 전체 라벨 ID (표본 + 보류).
    - sample_ids: 표본 라벨 ID.
    - policy: `max_defect_ratio`를 쓴다.

    판정:
    - 결함 = 표본을 parent로 하는 사람 레코드(측정 레코드 제외)가 있다 (수정 또는 삭제).
    - 승인 = 결함이 아니고 검수 상태가 human_approved.
    - 그 밖의 표본이 있으면 아직 검수 중(accepted=None).
    - 결함 비율 ≤ max_defect_ratio이면 합격. 합격이면 표본이 아닌 묶음 라벨 중 아직 현재 운영
      라벨이고 미검수인 것만 to_verify에 넣는다 (그 사이 고쳐지거나 지워진 것은 건드리지 않는다).
    - 판정에 쓸 표본이 하나도 남지 않았으면(모두 검수 전에 지워짐) 불합격이다. 사람이 아무것도 보지
      않았으므로 나머지를 표본 검증으로 둘 근거가 없다 (호출자가 보류 라벨 전수 재검수 배정을
      만든다).
    """
    by_id = {x.label_id: x for x in labels}
    # 사람(측정 레코드 제외)이 자식 레코드를 만든 라벨 = 고쳤거나 지웠다
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
    # 회귀: 표본이 모두 사라졌을 때(kept가 빔) 분모를 1로 두어 결함 0 → 합격으로 봐서, 사람이 한
    # 건도 보지 않은 나머지 라벨이 sample_verified가 됐다. 이제는 불합격(전수 재검수)이다.
    accepted = bool(kept) and defects / len(kept) <= policy.max_defect_ratio
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
