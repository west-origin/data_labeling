from __future__ import annotations

from pathlib import Path

import pytest

from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_datasets.splitter import assign_splits, check_isolation, propose_golden
from dlp_fixtures.sessions import generate_sessions
from dlp_schema.dataset import Split


@pytest.mark.parametrize("seed", range(6))
def test_golden_train_val_share_no_worker_or_site(seed: int) -> None:
    sessions = generate_sessions(300, seed=seed)
    golden = propose_golden(sessions, "cleaning", 20, seed=seed)
    splits, report = assign_splits(sessions, golden, val_ratio=0.1, seed=seed)
    assert check_isolation(sessions, splits) == []  # 완료 기준: 교집합 0
    assert set(splits) == {s.session_id for s in sessions}
    assert all(splits[g] is Split.GOLDEN for g in golden)
    for name in ("golden", "train", "val"):
        assert report.counts.get(name, 0) > 0
    assert 0.08 <= report.val_ratio <= 0.25
    assert report.holdout_ratio < 0.4


def test_golden_proposal_stays_in_domain_and_takes_whole_workers() -> None:
    sessions = generate_sessions(300, seed=2)
    golden = set(propose_golden(sessions, "caregiving", 15, seed=2))
    by_id = {s.session_id: s for s in sessions}
    assert golden and all(by_id[g].domain.value == "caregiving" for g in golden)
    workers = {by_id[g].worker_id for g in golden}
    # 고른 작업자의 그 도메인 세션은 모두 골든에 들어간다
    assert all(
        s.session_id in golden
        for s in sessions
        if s.worker_id in workers and s.domain.value == "caregiving"
    )


def test_isolation_check_reports_leaks() -> None:
    sessions = generate_sessions(30, seed=1)
    splits = {s.session_id: Split.TRAIN for s in sessions}
    splits[sessions[0].session_id] = Split.VAL
    twin = next(
        s
        for s in sessions[1:]
        if s.worker_id == sessions[0].worker_id or s.site_id == sessions[0].site_id
    )
    assert twin and check_isolation(sessions, splits)


def test_local_snapshot_is_content_addressed(tmp_path: Path) -> None:
    store = LocalSnapshotStore(tmp_path / "snap")
    f = tmp_path / "a.txt"
    f.write_text("v1")
    uri1 = store.commit("datasets/d1", {"a.txt": f}, "m", {})
    assert uri1 == store.commit("datasets/d1", {"a.txt": f}, "m", {})
    f.write_text("v2")
    uri2 = store.commit("datasets/d1", {"a.txt": f}, "m", {})
    assert uri2 != uri1
    out = tmp_path / "out.txt"
    store.read(uri1, "a.txt", out)
    assert out.read_text() == "v1"
