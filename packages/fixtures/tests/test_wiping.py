from __future__ import annotations

from pathlib import Path

from dlp_fixtures.wiping import CORNER_PARTS, generate_wiping_scenario
from dlp_schema.labels import Trajectory3DPayload
from dlp_schema.ontology import load_ontology
from dlp_schema.validation import check_label

ROOT = Path(__file__).resolve().parents[3]


def test_wiping_labels_match_ontology() -> None:
    """3D 궤적의 부분(작용부, 표면 꼭짓점)과 관계의 부분이 온톨로지 사전 안에 있다."""
    ontology = load_ontology(ROOT / "config" / "ontology" / "v1")
    w = generate_wiping_scenario(0)
    for label in [*w.labels, *w.truth_relations]:
        assert check_label(label, ontology) == [], label.label_id


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
