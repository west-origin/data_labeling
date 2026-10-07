"""작업자·장소 단위 분할.

같은 작업자나 같은 장소가 학습과 평가에 함께 들어가면 성능이 부풀려진다. 그래서 분할의 단위는
세션이 아니라 작업자와 장소다. 작업자가 여러 장소를 오가므로 작업자·장소 관계는 사슬처럼 이어지고,
그대로 두면 거의 모든 세션이 한 덩어리가 된다. 그래서 다음 규칙으로 나눈다.

1. 골든셋 세션의 작업자·장소를 "골든 쪽"으로 묶는다. 그 작업자나 장소가 나오는 다른 세션은
   어느 쪽에도 넣지 않는다 (holdout). 후보에 들지 않은 골든셋 세션(프라이버시 미승인, 다른 온톨로지
   등)도 골든 쪽 작업자·장소를 정한다 (golden_sessions): 그 세션이 나중에 평가에 쓰여도
   학습과 겹치지 않게.
2. 나머지에서 검증용 작업자를 하나씩 고른다. 작업자를 고르면 그의 장소도 검증 쪽이 된다.
   작업자와 장소가 모두 검증 쪽인 세션은 val, 모두 학습 쪽인 세션은 train, 걸친 세션은 holdout이다.
   매 단계에서 holdout이 가장 적게 늘어나는 작업자를 고르고(같으면 seed 해시 순), 검증 세션이
   목표 비율에 이르면 멈춘다.
결과는 check_isolation으로 다시 검사한다.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from dlp_schema.dataset import Split
from dlp_schema.session import Session


@dataclass(frozen=True)
class SplitReport:
    counts: dict[str, int]
    val_ratio: float  # (val) / (train + val)
    holdout_ratio: float  # holdout / 전체


def _h(seed: int, s: str) -> str:
    return hashlib.sha256(f"{seed}|{s}".encode()).hexdigest()


def assign_splits(
    sessions: list[Session],
    golden_ids: Iterable[str],
    *,
    val_ratio: float,
    seed: int = 0,
    golden_sessions: Iterable[Session] = (),
) -> tuple[dict[str, Split], SplitReport]:
    """golden_sessions: 골든셋의 모든 세션 (후보 밖 세션 포함).

    그 작업자·장소는 학습·검증에 못 들어간다.
    """
    golden = set(golden_ids)
    splits: dict[str, Split] = {}
    gold = [s for s in sessions if s.session_id in golden] + list(golden_sessions)
    gold_workers = {s.worker_id for s in gold}
    gold_sites = {s.site_id for s in gold}
    pool: list[Session] = []
    for s in sessions:
        if s.session_id in golden:
            splits[s.session_id] = Split.GOLDEN
        elif s.worker_id in gold_workers or s.site_id in gold_sites:
            splits[s.session_id] = Split.HOLDOUT
        else:
            pool.append(s)

    val_workers: set[str] = set()
    val_sites: set[str] = set()

    def classify(vw: set[str], vs: set[str]) -> tuple[int, int]:
        val = mixed = 0
        for s in pool:
            w, site = s.worker_id in vw, s.site_id in vs
            if w and site:
                val += 1
            elif w or site:
                mixed += 1
        return val, mixed

    target = val_ratio * len(pool)
    workers = sorted({s.worker_id for s in pool}, key=lambda w: _h(seed, w))
    sites_of: dict[str, set[str]] = {}
    for s in pool:
        sites_of.setdefault(s.worker_id, set()).add(s.site_id)
    val_count = 0
    while val_count < target:
        best: tuple[int, int, str] | None = None
        for w in workers:
            if w in val_workers:
                continue
            v, m = classify(val_workers | {w}, val_sites | sites_of[w])
            if v == val_count:
                continue  # 검증 세션을 늘리지 못하는 후보
            key = (m, -v, w)
            if best is None or key < best:
                best = key
        if best is None:
            break
        _, _, chosen = best
        val_workers.add(chosen)
        val_sites |= sites_of[chosen]
        val_count, _ = classify(val_workers, val_sites)

    for s in pool:
        w, site = s.worker_id in val_workers, s.site_id in val_sites
        splits[s.session_id] = (
            Split.VAL if w and site else Split.HOLDOUT if w or site else Split.TRAIN
        )

    counts = Counter(v.value for v in splits.values())
    tv = counts[Split.TRAIN.value] + counts[Split.VAL.value]
    report = SplitReport(
        counts=dict(counts),
        val_ratio=counts[Split.VAL.value] / tv if tv else 0.0,
        holdout_ratio=counts[Split.HOLDOUT.value] / len(sessions) if sessions else 0.0,
    )
    return splits, report


def check_isolation(sessions: list[Session], splits: dict[str, Split]) -> list[str]:
    """서로 다른 분할(golden, train, val) 사이에 작업자나 장소가 겹치면 그 설명 목록."""
    used = {Split.GOLDEN, Split.TRAIN, Split.VAL}
    owners: dict[tuple[str, str], set[Split]] = {}
    for s in sessions:
        split = splits.get(s.session_id)
        if split not in used or split is None:
            continue
        owners.setdefault(("worker", s.worker_id), set()).add(split)
        owners.setdefault(("site", s.site_id), set()).add(split)
    return [
        f"{kind} {key}: {sorted(x.value for x in sp)}"
        for (kind, key), sp in sorted(owners.items())
        if len(sp) > 1
    ]


def propose_golden(
    sessions: list[Session], domain: str, target: int, *, exclude: Iterable[str] = (), seed: int = 0
) -> list[str]:
    """골든셋 후보 세션. 도메인 안에서 작업자를 골라 그 작업자의 그 도메인 세션을 모은다.

    골든 작업자의 장소에 있는 다른 작업자 세션은 holdout이 되므로, 장소를 적게 공유하는 작업자를
    먼저 고른다. 최종 확정은 사람이 한다.
    """
    banned = set(exclude)
    candidates = [s for s in sessions if s.domain.value == domain and s.session_id not in banned]
    by_worker: dict[str, list[Session]] = {}
    for s in candidates:
        by_worker.setdefault(s.worker_id, []).append(s)
    site_load = Counter(s.site_id for s in sessions)

    def cost(w: str) -> tuple[float, str]:
        own = len(by_worker[w])
        shared = sum(site_load[site] for site in {s.site_id for s in by_worker[w]}) - own
        return (shared / own, _h(seed, w))

    chosen: list[str] = []
    for w in sorted(by_worker, key=cost):
        if len(chosen) >= target:
            break
        chosen += [s.session_id for s in by_worker[w]]
    return sorted(chosen)
