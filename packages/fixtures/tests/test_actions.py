from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from dlp_fixtures.actions import ActionScenario, generate_action_scenario
from dlp_schema.episode import EpisodeGraph
from dlp_schema.labels import ActionPayload, GapPayload, HandStatePayload, LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.validation import check_label


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def scenario(request: pytest.FixtureRequest) -> ActionScenario:
    return generate_action_scenario(int(request.param), n_units=10)


def _intervals(labels: list[LabelRecord], kinds: tuple[type, ...]) -> list[tuple[int, int]]:
    return sorted((x.t_start_ms, x.t_end_ms) for x in labels if isinstance(x.payload, kinds))


def test_labels_are_valid_and_form_an_episode(scenario: ActionScenario, ontology: Ontology) -> None:
    for label in scenario.labels:
        assert check_label(label, ontology) == [], label.label_id
    graph = EpisodeGraph(
        episode_id="e", session_id=scenario.session_id, ontology_version="1.0.0",
        t_start_ms=0, t_end_ms=scenario.duration_ms,
        entities=tuple(scenario.entities), labels=tuple(scenario.labels),
    )  # fmt: skip
    assert len(graph.events) > 10


def test_actions_and_gaps_tile_the_timeline_without_holes(scenario: ActionScenario) -> None:
    for kinds in ((ActionPayload, GapPayload), (HandStatePayload,)):
        spans = _intervals(scenario.labels, kinds)
        assert spans[0][0] == 0 and spans[-1][1] == scenario.duration_ms
        for (_, end), (start, _) in pairwise(spans):
            assert start == end


def test_contact_chain_markers(scenario: ActionScenario) -> None:
    actions = scenario.actions
    for i, a in enumerate(actions):
        if a.verb == "grasp":
            carry, release = actions[i + 1], actions[i + 2]
            assert a.t_contact_start_ms is not None and a.t_contact_end_ms is None
            assert carry.contact_held and carry.t_contact_start_ms is None
            assert release.contact_held and release.t_contact_end_ms is not None
        elif a.verb not in ("carry", "release"):
            assert a.t_contact_start_ms is not None and a.t_contact_end_ms is not None


def test_glove_pressure_matches_contact_intervals(scenario: ActionScenario) -> None:
    t, p = scenario.glove_t_ms, scenario.glove_pressure
    in_contact = np.zeros(t.size, dtype=bool)
    edge = np.zeros(t.size, dtype=bool)
    for start, end in _intervals(
        [x for x in scenario.labels if isinstance(x.payload, HandStatePayload)
         and x.payload.contact_target_kind != "none"],
        (HandStatePayload,),
    ):  # fmt: skip
        in_contact |= (t >= start) & (t <= end)
        edge |= (np.abs(t - start) < 40) | (np.abs(t - end) < 40)
    assert np.all(p[in_contact & ~edge] > 0.3)
    assert np.all(p[~in_contact & ~edge] < 0.1)


def test_wrist_moves_fast_on_approach_and_rests_in_gaps(scenario: ActionScenario) -> None:
    times = np.asarray(scenario.frame_times, dtype=float)
    speed = np.linalg.norm(np.diff(scenario.wrist, axis=0), axis=1) / np.diff(times) * 1000
    mid = (times[1:] + times[:-1]) / 2

    def mean_speed(start: float, end: float) -> float:
        sel = (mid > start) & (mid < end)
        return float(speed[sel].mean())

    for a in scenario.actions:
        if a.t_contact_start_ms is not None:
            span = a.t_contact_start_ms - a.t_approach_ms
            assert mean_speed(a.t_approach_ms + span * 0.3, a.t_approach_ms + span * 0.7) > 80
    for start, end in _intervals(scenario.labels, (GapPayload,)):
        if end - start > 200:
            assert mean_speed(start + 50, end - 50) < 30
