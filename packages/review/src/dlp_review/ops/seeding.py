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
  - blur_deletion: 그 배정에서 검수자가 새로 그린 블러 트랙(수집 때 ID에 사본 접두사가 붙는다)이
    원래 트랙과 같은 스트림·대상이고 시간이 blur_overlap 이상 겹침

WP12, ADR 0014·0015(감사 정정). 정책은 `config/policies/review.yaml` `seeding` 절.
흐름: `ops.runner.plan_session`이 오류 삽입 배정을 계획할 때 `seed_labels`로 사본을 만들어 DB에
넣고, 수집(`collect.collect_task`) 뒤 `ops.runner.quality_report`가 `detected`로 발견율을 센다.

공개 이름: `INTERVAL_TYPES`, `seed_prefix`, `SeededTask`, `seed_labels`, `detected`.
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
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import InjectedError

# boundary_shift 대상이 되는 시간 구간 라벨 페이로드
INTERVAL_TYPES = (
    ActionPayload,
    HandStatePayload,
    GapPayload,
    SegmentPayload,
    ObjectStatePayload,
    RelationPayload,
)


def seed_prefix(assignment_id: str) -> str:
    """오류 삽입 배정의 사본 ID 접두사 `"seed-<배정 ID>-"`.

    사본(`seed-<배정>-0000`…)과 수집 때 새로 그린 레코드(`seed-<배정>-new-<해시>`)가 이 접두사를
    쓴다. 선택(`ops.selection`)과 발견 판정이 이 접두사로 그 배정의 레코드를 가린다.
    """
    return f"seed-{assignment_id}-"


def _rng(seed: int, assignment_id: str) -> np.random.Generator:
    """배정별 결정적 난수 생성기 (sha256(`"<seed>:<배정>"`) 앞 8바이트를 시드로)."""
    digest = hashlib.sha256(f"{seed}:{assignment_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _candidates(x: LabelRecord, error: ErrorType) -> bool:
    """라벨 x가 오류 종류 error의 대상이 될 수 있는가."""
    p = x.payload
    if error == "boundary_shift":
        return isinstance(p, INTERVAL_TYPES)
    if error == "class_swap":
        return isinstance(p, BoxTrackPayload | MaskTrackPayload | ActionPayload)
    return isinstance(p, BlurTrackPayload)


def _shift(x: LabelRecord, delta: int) -> tuple[LabelRecord, dict[str, str | int | float]]:
    """시작을 delta만큼 앞으로 (0 아래로 못 가면 끝을 뒤로).

    인자: x(사본 라벨), delta(옮길 ms, 양수).
    반환: (옮긴 라벨, 판정용 detail). detail = {"edge": "start"|"end", "original_ms": 원래 경계 ms,
    "shift_ms": delta}.
    행동 라벨은 페이로드의 접근 시작(`t_approach_ms`)·종료(`t_end_ms`)도 새 구간에 맞춘다.
    구간을 넓히기만 하므로 접촉 시각은 여전히 구간 안에 있다.
    """
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
    """행동 동사나 박스·마스크 클래스를 온톨로지의 다른 값으로 바꾼다.

    행동은 원시 동작(primitive) 동사 중에서, 박스·마스크는 객체 클래스 중에서 고른다
    (후보를 정렬한 뒤 rng로 골라 결정적이다).
    반환: (바꾼 라벨, detail = {"field": "verb"|"class_id", "original": 원래 값, "seeded": 새 값}).
    """
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
    """오류 삽입 과제 하나."""

    labels: list[LabelRecord]  # 검수자에게 보낼 사본 (오류 포함)
    # 넣은 오류 목록 (배정 `injected`에 저장되어 발견 판정에 쓴다)
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
    """정답 라벨(truth)의 사본을 만들고 오류를 넣는다.

    인자:
    - truth: 정답을 아는 단위의 현재 운영 라벨 (모두 사람이 만들었거나 승인·수정한 것).
    - assignment_id: 오류 삽입 배정 ID (사본 ID 접두사와 난수 시드에 쓴다).
    - ontology: class_swap의 대체 값 후보.
    - policy: `errors_per_task`, `types`, `boundary_shift_ms`.
    - seed: 난수 시드 (같은 seed·배정이면 같은 사본·오류).
    - now: 사본 `created_at`.

    반환: `SeededTask` (사본은 ID 순 정렬). 사본은 `seeded_error=True`, parent 없음, 검수 상태
    초기화(unreviewed). 후보가 있는 오류 종류를 `types` 순서대로 돌아가며 넣고, 한 라벨에는
    오류를 하나만 넣는다. DB에 쓰지 않는다 (runner가 `insert_labels`로 넣는다).
    """
    rng = _rng(seed, assignment_id)
    prefix = seed_prefix(assignment_id)
    # ID 순으로 정렬해 사본 번호(0000…)와 뽑기가 입력 순서에 좌우되지 않게 한다
    ordered = sorted(truth, key=lambda x: x.label_id)
    copies = {
        x.label_id: x.model_copy(
            update={
                "label_id": f"{prefix}{i:04d}", "parent_label_id": None, "retracted": False,
                "seeded_error": True, "created_at": now, "verification": Verification(),
            }
        )
        for i, x in enumerate(ordered)
    }  # fmt: skip
    injected: list[InjectedError] = []
    used: set[str] = set()
    # 이 단위에 후보가 하나도 없는 오류 종류는 뺀다 (예: 시간 단위에는 blur_deletion 후보가 없다)
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
            # 발견 판정을 이 배정에서 새로 그린 블러로 한정하려고 배정 ID를 남긴다
            injected.append(
                InjectedError(
                    error_type=error, original_label_id=original.label_id,
                    detail={"assignment_id": assignment_id},
                )
            )  # fmt: skip
            continue
        if error == "boundary_shift":
            lo, hi = policy.boundary_shift_ms
            # [lo, hi] 양끝 포함 (integers의 상한은 배타적이라 +1)
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
    """a 구간 길이 대비 a·b 겹친 길이 (a 길이가 0이면 1 ms로 본다)."""
    inter = max(0, min(a.t_end_ms, b.t_end_ms) - max(a.t_start_ms, b.t_start_ms))
    return inter / max(1, a.t_end_ms - a.t_start_ms)


def detected(
    error: InjectedError,
    labels: list[LabelRecord],
    reviewer_id: str | None,
    tolerance_ms: int,
    blur_overlap: float,
) -> bool:
    """검수 결과(사람 출처, 오류 삽입 계보)를 보고 발견했는지 판정한다.

    labels: 세션의 모든 레코드.

    인자:
    - error: 넣은 오류 (배정 `injected`).
    - reviewer_id: 그 배정의 담당자. 있으면 그 사람이 남긴 레코드만 센다. None이면 모두.
    - tolerance_ms: 경계 허용 오차 (`seeding.detect_tolerance_ms`).
    - blur_overlap: 다시 그린 블러의 최소 겹침 비율 (`seeding.blur_overlap`).

    반환: 발견했으면 True. 원래 라벨이 이력에 없으면 False.
    """
    by_id = {x.label_id: x for x in labels}
    original = by_id.get(error.original_label_id)
    if original is None:
        return False
    # 검수자가 이 과제에서 남긴 레코드만 본다. 오류 삽입 사본은 검수 상태가 비어 있으므로 빠진다
    # (사본은 만들 때 검수 상태를 지운다. 검수자가 그 사본을 승인만 하면 고친 것이 아니다).
    human = [
        x
        for x in labels
        if x.provenance.source is Source.HUMAN
        and x.seeded_error
        and x.verification.state is VerificationState.HUMAN_CORRECTED
    ]
    if reviewer_id is not None:
        human = [x for x in human if x.verification.reviewer_id == reviewer_id]
    if error.error_type == "blur_deletion":
        p = original.payload
        assert isinstance(p, BlurTrackPayload)
        # 같은 스트림, 그리고 이 배정의 수집 결과(사본 접두사 ID)로 새로 그린 블러만 센다
        # (같은 세션의 다른 배정·스트림에서 그린 블러로 발견을 인정하지 않는다)
        aid = error.detail.get("assignment_id")
        # 예전 기록(detail에 배정 ID가 없음)은 접두사 검사 없이 판정한다
        prefix = seed_prefix(str(aid)) if aid else None
        return any(
            x.parent_label_id is None
            and x.stream_id == original.stream_id
            and (prefix is None or x.label_id.startswith(prefix))
            and isinstance(x.payload, BlurTrackPayload)
            and x.payload.target == p.target
            and _overlap_ratio(original, x) >= blur_overlap
            for x in human
        )
    # 경계·클래스 오류: 오류가 든 사본을 parent로 하는 수정 레코드(삭제 제외)를 본다
    fixes = [x for x in human if x.parent_label_id == error.seeded_label_id and not x.retracted]
    for fix in fixes:
        if error.error_type == "boundary_shift":
            edge = error.detail.get("edge")
            got = fix.t_start_ms if edge == "start" else fix.t_end_ms
            if abs(got - int(error.detail["original_ms"])) <= tolerance_ms:
                return True
        else:
            # class_swap: 바꾼 필드(verb 또는 class_id)가 원래 값으로 돌아왔는가
            field = str(error.detail["field"])
            if getattr(fix.payload, field, None) == error.detail["original"]:
                return True
    return False
