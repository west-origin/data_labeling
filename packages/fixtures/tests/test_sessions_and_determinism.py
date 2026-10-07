from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from dlp_fixtures import generate_all
from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.sessions import generate_sessions
from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_schema.session import StreamKind
from dlp_schema.testing import FIXED_TIME


def test_sessions_have_realistic_worker_site_structure() -> None:
    sessions = generate_sessions(300, seed=1)
    assert len({s.session_id for s in sessions}) == 300
    assert all(s.reference_stream.kind is StreamKind.BODYCAM for s in sessions)
    workers_per_site: dict[str, set[str]] = defaultdict(set)
    for s in sessions:
        workers_per_site[s.site_id].add(s.worker_id)
    # 작업자·장소 단위 분할을 시험할 수 있도록 여러 작업자가 공유하는 장소가 있어야 한다
    assert any(len(w) > 1 for w in workers_per_site.values())
    assert {s.domain.value for s in sessions} == {"cleaning", "caregiving", "nursing"}
    kinds = {st.kind for s in sessions for st in s.streams}
    assert StreamKind.GLOVE_RIGHT in kinds and StreamKind.THIRD_PERSON in kinds


def test_generators_are_deterministic() -> None:
    assert generate_sessions(50, seed=7) == generate_sessions(50, seed=7)
    assert generate_sessions(50, seed=7) != generate_sessions(50, seed=8)
    a1, a2 = generate_action_scenario(7), generate_action_scenario(7)
    assert a1.labels == a2.labels
    assert np.array_equal(a1.glove_pressure, a2.glove_pressure)
    s1 = generate_sync_scenario(7, recorded_at=FIXED_TIME)
    s2 = generate_sync_scenario(7, recorded_at=FIXED_TIME)
    assert s1.truth() == s2.truth()
    assert np.array_equal(s1.audio["third_person"], s2.audio["third_person"])
    b1, b2 = generate_blur_scenario(7), generate_blur_scenario(7)
    assert b1.labels == b2.labels
    assert all(np.array_equal(x, y) for x, y in zip(b1.frames, b2.frames, strict=True))


def test_generate_all_writes_every_fixture(tmp_path: Path) -> None:
    paths = generate_all(tmp_path, seed=2, n_sessions=20)
    assert sum(1 for _ in paths["sessions"].open()) == 20
    for name in ("bodycam.mp4", "third_person.mp4", "bodycam.wav", "glove_right.parquet",
                 "imu.parquet", "truth.json"):  # fmt: skip
        assert (paths["sync"] / name).stat().st_size > 0
    assert (paths["blur"] / "bodycam.mp4").stat().st_size > 0
    assert (paths["actions"] / "labels.jsonl").stat().st_size > 0
