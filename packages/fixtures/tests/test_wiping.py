from __future__ import annotations

from dlp_fixtures.wiping import CORNER_PARTS, generate_wiping_scenario
from dlp_schema.labels import Trajectory3DPayload


def test_wiping_scenario_is_deterministic_and_consistent() -> None:
    a, b = generate_wiping_scenario(3), generate_wiping_scenario(3)
    assert [x.model_dump() for x in a.labels] == [x.model_dump() for x in b.labels]
    assert a.coverage == b.coverage and 0 < a.coverage < 1
    assert len(a.contacts) == len(a.strokes) and len(a.truth_relations) == len(a.contacts) + 1
    grasp_start, grasp_end = a.grasp
    assert all(grasp_start < s < e < grasp_end for s, e in a.contacts)
    parts = {
        x.payload.part for x in a.labels if isinstance(x.payload, Trajectory3DPayload)
    }  # fmt: skip
    assert parts == {"cloth_face", *CORNER_PARTS}


def test_more_rows_cover_more_of_the_surface() -> None:
    by_rows: dict[int, list[float]] = {}
    for seed in range(8):
        w = generate_wiping_scenario(seed)
        by_rows.setdefault(len(w.strokes), []).append(w.coverage)
    assert min(by_rows) < max(by_rows)
    assert min(by_rows[min(by_rows)]) < max(by_rows[max(by_rows)])
