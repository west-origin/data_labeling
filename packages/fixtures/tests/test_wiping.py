"""걸레질 시나리오 생성기(`generate_wiping_scenario`) 자체 테스트 (WP2, WP9 전제).

라벨이 온톨로지에 맞고, 정답(접촉 구간·선분·관계·커버리지)이 서로 일관되며, 줄이 많을수록 커버리지가
커지는지 본다.
"""

from __future__ import annotations

from pathlib import Path

from dlp_fixtures.wiping import CORNER_PARTS, generate_wiping_scenario
from dlp_schema.labels import Trajectory3DPayload
from dlp_schema.ontology import load_ontology
from dlp_schema.validation import check_label

# 저장소 루트 (온톨로지 경로용)
ROOT = Path(__file__).resolve().parents[3]


def test_wiping_labels_match_ontology() -> None:
    """3D 궤적의 부분(작용부, 표면 꼭짓점)과 관계의 부분이 온톨로지 사전 안에 있다."""
    ontology = load_ontology(ROOT / "config" / "ontology" / "v1")
    w = generate_wiping_scenario(0)
    for label in [*w.labels, *w.truth_relations]:
        assert check_label(label, ontology) == [], label.label_id


def test_wiping_scenario_is_deterministic_and_consistent() -> None:
    """같은 seed면 같은 라벨·커버리지이고, 접촉 수 = 줄 수, 정답 관계 = 접촉 + 파지 1개, 모든
    접촉이 파지 구간 안, 3D 궤적 부분이 작용부와 꼭짓점 넷인지 검증한다.
    """
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
    """seed 0~7에서 줄 수가 가장 적은 시나리오의 최소 커버리지가 가장 많은 시나리오의 최대
    커버리지보다 작은지 검증한다.
    """
    by_rows: dict[int, list[float]] = {}
    for seed in range(8):
        w = generate_wiping_scenario(seed)
        by_rows.setdefault(len(w.strokes), []).append(w.coverage)
    assert min(by_rows) < max(by_rows)
    assert min(by_rows[min(by_rows)]) < max(by_rows[max(by_rows)])
