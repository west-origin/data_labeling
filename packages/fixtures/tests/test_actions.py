"""행동 시퀀스 생성기(`generate_action_scenario`) 자체 테스트 (WP2).

정답 라벨이 계약·온톨로지에 맞고, 타임라인을 빈틈없이 덮으며, 신호(손목 속도·장갑 압력)가
정답 경계와 맞는지 본다. 행동 구간 모듈(WP10)이 이 정답에 기대므로 생성기의 전제를 여기서 고정한다.
seed 0~3, 단위 10개로 돈다.
"""

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
    """seed 0~3 각각의 행동 시나리오 (단위 10개, 모듈 범위 캐시)."""
    return generate_action_scenario(int(request.param), n_units=10)


def _intervals(labels: list[LabelRecord], kinds: tuple[type, ...]) -> list[tuple[int, int]]:
    """`kinds` payload 라벨의 (시작, 끝) ms 구간 목록 (정렬)."""
    return sorted((x.t_start_ms, x.t_end_ms) for x in labels if isinstance(x.payload, kinds))


def test_labels_are_valid_and_form_an_episode(scenario: ActionScenario, ontology: Ontology) -> None:
    """모든 정답 라벨이 온톨로지 검증(`check_label`)을 통과하고, 개체와 함께 에피소드 그래프로
    묶이며 이벤트가 10개 넘게 나오는지 검증한다.
    """
    for label in scenario.labels:
        assert check_label(label, ontology) == [], label.label_id
    graph = EpisodeGraph(
        episode_id="e", session_id=scenario.session_id, ontology_version="1.0.0",
        t_start_ms=0, t_end_ms=scenario.duration_ms,
        entities=tuple(scenario.entities), labels=tuple(scenario.labels),
    )  # fmt: skip
    assert len(graph.events) > 10


def test_actions_and_gaps_tile_the_timeline_without_holes(scenario: ActionScenario) -> None:
    """행동 + 사이 구간, 그리고 손 상태가 각각 0부터 끝까지 빈틈·겹침 없이 이어지는지 검증한다."""
    for kinds in ((ActionPayload, GapPayload), (HandStatePayload,)):
        spans = _intervals(scenario.labels, kinds)
        assert spans[0][0] == 0 and spans[-1][1] == scenario.duration_ms
        for (_, end), (start, _) in pairwise(spans):
            assert start == end


def test_contact_chain_markers(scenario: ActionScenario) -> None:
    """잡다 → 옮기다 → 놓다 묶음은 접촉 시작이 잡다에만, 접촉 종료가 놓다에만 있고(`contact_held`),
    단일 동사는 접촉 시작·종료를 모두 갖는지 검증한다.
    """
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
    """장갑 압력이 정답 접촉 구간 안에서는 0.3 초과, 밖에서는 0.1 미만인지 검증한다.

    압력 경사(30 ms)가 걸치는 경계 ±40 ms는 판정에서 뺀다.
    """
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
    """접근 중에는 손목이 빠르고 대기 중에는 멈춰 있는지 검증한다.

    접근 국면 가운데(30~70%)의 평균 속도 > 80 px/s, 200 ms 넘는 대기 구간 안쪽(양 끝 50 ms 제외)
    < 30 px/s.

    경계 후보 생성기가 속도로 경계를 찾을 수 있다는 전제다.
    """
    times = np.asarray(scenario.frame_times, dtype=float)
    speed = np.linalg.norm(np.diff(scenario.wrist, axis=0), axis=1) / np.diff(times) * 1000
    mid = (times[1:] + times[:-1]) / 2

    def mean_speed(start: float, end: float) -> float:
        """(start, end) ms 안 프레임 사이 속도의 평균 px/s."""
        sel = (mid > start) & (mid < end)
        return float(speed[sel].mean())

    for a in scenario.actions:
        if a.t_contact_start_ms is not None:
            span = a.t_contact_start_ms - a.t_approach_ms
            assert mean_speed(a.t_approach_ms + span * 0.3, a.t_approach_ms + span * 0.7) > 80
    for start, end in _intervals(scenario.labels, (GapPayload,)):
        if end - start > 200:
            assert mean_speed(start + 50, end - 50) < 30
