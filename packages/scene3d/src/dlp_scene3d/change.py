"""전후 물체 대응(헝가리안)과 변화 분류.

after 물체를 정렬 변환으로 before 좌표에 옮긴 뒤(크기도 척도를 곱한다) 같은 라벨끼리만 비용을
매긴다. 외형 거리(Hellinger)나 크기 비가 관문을 넘는 쌍은 대응하지 않는다. 비용은
`w_a · 외형 거리 + w_s · |log 크기 비| + w_d · min(거리 / distance_scale_m, 1)`이다. 거리 항은
같은 모양 물체가 여럿일 때 가까운 쪽을 고르게 하는 약한 항이다 (움직인 물체도 대응해야 하므로).

분류: 대응한 쌍은 이동 거리 ≤ `unchanged_tolerance_m`이면 unchanged, 넘으면 moved. 대응하지 못한
before 물체는 removed, after 물체는 added. 신뢰도는 대응 쌍이면 두 물체 신뢰도의 작은 값 x
(1 - 외형 거리), 아니면 그 물체의 신뢰도다. 가려져 after에서 안 보인 물체도 removed로 나오므로
검수가 필요하다 (ADR 0066).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from dlp_scene3d.geometry import F64, Similarity
from dlp_scene3d.objects import hellinger
from dlp_scene3d.policy import ChangePolicy
from dlp_scene3d.scipy_typed import assign
from dlp_schema.scene3d import ObjectChange, ObjectInstance

# 대응 금지 쌍의 비용 (헝가리안 입력은 유한해야 한다)
_FORBIDDEN = 1e9


def to_before_frame(obj: ObjectInstance, after_to_before: Similarity) -> ObjectInstance:
    """after 물체의 위치·크기를 before 좌표로 옮긴 사본."""
    p = after_to_before.apply(np.asarray([obj.position], dtype=np.float64))[0]
    return obj.model_copy(
        update={
            "position": (float(p[0]), float(p[1]), float(p[2])),
            "size": obj.size * after_to_before.s,
        }
    )


def _pair_cost(b: ObjectInstance, a: ObjectInstance, policy: ChangePolicy) -> float | None:
    """대응 비용. 라벨이 다르거나 관문을 넘으면 None."""
    if b.label != a.label:
        return None
    app = hellinger(b.appearance, a.appearance)
    ratio = max(b.size, a.size) / min(b.size, a.size)
    if app > policy.max_appearance_distance or ratio > policy.max_size_ratio:
        return None
    dist = float(np.linalg.norm(np.subtract(a.position, b.position)))
    w = policy.weights
    return (
        w.appearance * app
        + w.size * float(np.log(ratio))
        + w.distance * min(dist / policy.distance_scale_m, 1.0)
    )


def _change_id(kind: str, before: str | None, after: str | None) -> str:
    """변화 ID (물체 ID에서 결정적으로)."""
    return "chg-" + hashlib.sha256(f"{kind}|{before}|{after}".encode()).hexdigest()[:12]


def _vec(v: F64) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _matched(
    b: ObjectInstance, a: ObjectInstance, cost: float, policy: ChangePolicy
) -> ObjectChange:
    """대응한 쌍 → moved 또는 unchanged."""
    d = np.subtract(a.position, b.position)
    dist = float(np.linalg.norm(d))
    kind = "unchanged" if dist <= policy.unchanged_tolerance_m else "moved"
    conf = min(b.confidence, a.confidence) * (1.0 - hellinger(b.appearance, a.appearance))
    return ObjectChange(
        change_id=_change_id(kind, b.instance_id, a.instance_id),
        kind=kind,
        label=b.label,
        before_instance_id=b.instance_id,
        after_instance_id=a.instance_id,
        before_position=b.position,
        after_position=a.position,
        displacement=_vec(d),
        distance_m=dist,
        match_cost=cost,
        confidence=max(0.0, min(1.0, conf)),
    )


def _single(obj: ObjectInstance, kind: str) -> ObjectChange:
    """대응하지 못한 물체 → removed(before) 또는 added(after)."""
    before = obj.instance_id if kind == "removed" else None
    after = obj.instance_id if kind == "added" else None
    return ObjectChange(
        change_id=_change_id(kind, before, after),
        kind="removed" if kind == "removed" else "added",
        label=obj.label,
        before_instance_id=before,
        after_instance_id=after,
        before_position=obj.position if before else None,
        after_position=obj.position if after else None,
        displacement=None,
        confidence=obj.confidence,
    )


def compare_objects(
    before: Sequence[ObjectInstance],
    after_in_before: Sequence[ObjectInstance],
    policy: ChangePolicy,
) -> list[ObjectChange]:
    """before 물체와 (before 좌표로 옮긴) after 물체를 대응해 변화 목록을 낸다 (종류·라벨 순)."""
    out: list[ObjectChange] = []
    matched_b: set[int] = set()
    matched_a: set[int] = set()
    if before and after_in_before:
        cost = np.full((len(before), len(after_in_before)), _FORBIDDEN)
        for i, b in enumerate(before):
            for j, a in enumerate(after_in_before):
                c = _pair_cost(b, a, policy)
                if c is not None:
                    cost[i, j] = c
        for i, j in assign(cost):
            if cost[i, j] >= _FORBIDDEN:
                continue
            matched_b.add(i)
            matched_a.add(j)
            out.append(_matched(before[i], after_in_before[j], float(cost[i, j]), policy))
    out += [_single(b, "removed") for i, b in enumerate(before) if i not in matched_b]
    out += [_single(a, "added") for j, a in enumerate(after_in_before) if j not in matched_a]
    return sorted(out, key=lambda c: (c.kind, c.label, c.change_id))
