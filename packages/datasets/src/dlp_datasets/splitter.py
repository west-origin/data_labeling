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

예: 작업자 A가 장소 X·Y에서, 작업자 B가 장소 Y에서 일했다. A를 검증으로 고르면 X·Y가 검증 쪽이 되어
B의 Y 세션은 (작업자 학습 쪽, 장소 검증 쪽) → holdout이다.

holdout은 학습·검증·평가 어디에도 쓰지 않는다 (정보 누수 방지, `dlp_schema.dataset.Split`).
결정적이다: 같은 세션·골든·seed면 같은 분할이 나온다 (세션 순서와 무관하게 seed 해시로 동점 처리).

공개 함수: `assign_splits`, `check_isolation`, `propose_golden`. 이 모듈은 DB를 모른다 (순수 함수).
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
    """분할 요약 (CLI 출력·매니페스트용)."""

    counts: dict[str, int]  # 분할 이름 → 세션 수
    val_ratio: float  # (val) / (train + val)
    holdout_ratio: float  # holdout / 전체


def _h(seed: int, s: str) -> str:
    """seed를 섞은 결정적 정렬 키 (작업자 순서를 ID 사전순 편향 없이 섞는다)."""
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

    Args:
        sessions: 분할할 후보 세션 (데이터셋에 들어갈 세션 전부).
        golden_ids: 골든셋 세션 ID. `sessions`에 있는 것은 golden 분할이 된다.
        val_ratio: 골든 쪽을 뺀 나머지(pool) 중 검증 세션 목표 비율 (dataset.yaml `val_ratio`).
        seed: 동점 처리 순서 seed.
        golden_sessions: 후보 밖 골든 세션 (작업자·장소만 골든 쪽으로 막는 데 쓴다,
            분할에는 안 넣는다).

    Returns:
        (세션 ID → 분할, 요약). `sessions`의 모든 세션이 키로 들어간다.

    알고리즘(탐욕): 목표 val 세션 수에 이를 때까지, 아직 고르지 않은 작업자 중 그를 더했을 때
    "걸친 세션(holdout) 수"가 가장 작고, 같으면 val 수가 큰 작업자를 고른다. val을 늘리지 못하는
    작업자는 건너뛰고, 고를 작업자가 없으면 멈춘다. 매 단계 pool 전체를 다시 세므로
    O(작업자² * 세션)이다.
    """
    golden = set(golden_ids)
    splits: dict[str, Split] = {}
    gold = [s for s in sessions if s.session_id in golden] + list(golden_sessions)
    gold_workers = {s.worker_id for s in gold}
    gold_sites = {s.site_id for s in gold}
    # 1단계: 골든과 골든 쪽 작업자·장소를 공유하는 세션(holdout)을 떼고, 나머지를 pool로
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
        """검증 쪽 작업자·장소가 (vw, vs)일 때 pool의 (val 세션 수, 걸친 세션 수)."""
        val = mixed = 0
        for s in pool:
            w, site = s.worker_id in vw, s.site_id in vs
            if w and site:
                val += 1
            elif w or site:
                mixed += 1
        return val, mixed

    # 2단계: 검증 작업자를 탐욕적으로 고른다
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
            # 정렬 키: 걸친 세션 적은 것 → val 많은 것 → 작업자 ID
            key = (m, -v, w)
            if best is None or key < best:
                best = key
        if best is None:
            break
        _, _, chosen = best
        val_workers.add(chosen)
        val_sites |= sites_of[chosen]
        val_count, _ = classify(val_workers, val_sites)

    # 3단계: pool 세션을 최종 분류 (둘 다 검증 쪽 → val, 하나만 → holdout, 둘 다 아님 → train)
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
    """서로 다른 분할(golden, train, val) 사이에 작업자나 장소가 겹치면 그 설명 목록.

    holdout과 분할이 없는 세션은 보지 않는다. 빈 목록이면 격리가 지켜진 것이다.

    Returns:
        "worker <ID>: ['train', 'val']" 같은 문자열 목록 (종류·ID 순).
    """
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

    Args:
        sessions: 전체 세션 (장소 공유 비용은 이 전체로 센다).
        domain: 도메인 값 (예: "cleaning").
        target: 목표 세션 수. 작업자 단위로 통째로 더하므로 조금 넘을 수 있다.
        exclude: 후보에서 뺄 세션 ID (프라이버시 미승인·사용 중지 등).
        seed: 동점 처리 seed.

    Returns:
        고른 세션 ID (정렬).

    비용 = (그 작업자 장소들의 전체 세션 수 - 그 작업자 몫) / 그 작업자 몫. 작을수록 holdout을 적게
    만든다.
    """
    banned = set(exclude)
    candidates = [s for s in sessions if s.domain.value == domain and s.session_id not in banned]
    by_worker: dict[str, list[Session]] = {}
    for s in candidates:
        by_worker.setdefault(s.worker_id, []).append(s)
    site_load = Counter(s.site_id for s in sessions)

    def cost(w: str) -> tuple[float, str]:
        """(작업자 세션 하나당 장소 공유 세션 수, seed 해시) — 작은 것부터 고른다."""
        own = len(by_worker[w])
        shared = sum(site_load[site] for site in {s.site_id for s in by_worker[w]}) - own
        return (shared / own, _h(seed, w))

    chosen: list[str] = []
    for w in sorted(by_worker, key=cost):
        if len(chosen) >= target:
            break
        chosen += [s.session_id for s in by_worker[w]]
    return sorted(chosen)
