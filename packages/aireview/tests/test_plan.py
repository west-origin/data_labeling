"""판정 단위 계획 테스트 (`dlp_aireview.plan`, WP17.2, ADR 0040 2.1·2.2·4.3).

정답을 아는 합성 장면(conftest)에서 다음을 본다.
- 시간 대상은 정확히 한 단위에 한 번 든다 (긴 라벨은 temporal_long 한 단위). 공간 트랙은 창 상한 안.
- 대상 상한으로 나눈 요청은 같은 증거 구간·프레임을 쓴다.
- flagged·감사 선택이 결정적이고, 선택 방식은 단위 키에 들어가지 않는다.
- 라벨 하나를 고치면 그 라벨이 든 단위 키만 바뀌고, 렌더 해시를 바꾸면 그 스트림 단위만 바뀐다.
- 측정·오류 삽입·사람·검수된 라벨, 원본 접근 이벤트, 블러는 대상에 없고, 측정·오류 삽입·블러·원본
  접근 이벤트는 맥락에도 없다.
- 프레임 시각은 PTS 인덱스에서 나온다 (VFR).
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from fractions import Fraction

import numpy as np

from dlp_aireview.judge import judge_unit
from dlp_aireview.oracle import OracleJudge
from dlp_aireview.plan import (
    PlannedWindow,
    StreamMedia,
    estimate_units,
    plan_session,
    plan_windows,
    targets,
    uncovered_targets,
)
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.sensors import SensorContext
from dlp_fixtures.video import vfr_times
from dlp_media.pts import PtsIndex
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxKeyframe,
    EventPayload,
    GapPayload,
    Hand,
    LabelRecord,
    ObjectStatePayload,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.review import FlaggedSpan, ReviewReason
from dlp_schema.testing import FIXED_TIME

from .conftest import RENDER, SID, Scene, images_for, with_policy

ABSENT = SensorContext.absent()


def _ids(ws: list[PlannedWindow], *groups: AiGroup) -> Counter[str]:
    """창 대상 라벨 ID 개수 (묶음 제한)."""
    return Counter(x.label_id for w in ws if w.group in groups for x in w.targets)


def _label(label_id: str, payload: object, start: int, end: int, **kw: object) -> LabelRecord:
    """모델 출처 라벨 하나 (세션 s001)."""
    data: dict[str, object] = {
        "label_id": label_id, "session_id": SID, "t_start_ms": start, "t_end_ms": end,
        "ontology_version": "1.0.0",
        "provenance": Provenance(source=Source.MODEL, model_version="m-1"), "confidence": 0.8,
        "created_at": FIXED_TIME, "payload": payload,
    }  # fmt: skip
    data.update(kw)
    return LabelRecord.model_validate(data)


def test_each_temporal_target_planned_exactly_once(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """시간 대상은 temporal·temporal_long 단위 중 정확히 하나에 든다. 공간 트랙은 1개 이상, 트랙당
    창 상한 이하에 든다. 계획은 결정적이다."""
    ws = plan_windows(scene.session, scene.labels, policy, ontology)
    found = targets(scene.labels, policy, ontology)
    temporal = {x.label_id for x in found if x.kind in policy.kinds.temporal}
    spatial = {x.label_id for x in found if x.kind in policy.kinds.spatial}
    counts = _ids(ws, AiGroup.TEMPORAL, AiGroup.TEMPORAL_LONG)
    assert set(counts) == temporal and set(counts.values()) == {1}
    spatial_counts = _ids(ws, AiGroup.SPATIAL)
    assert set(spatial_counts) == spatial
    assert max(spatial_counts.values()) <= policy.spatial.max_windows_per_track
    assert ws == plan_windows(scene.session, scene.labels, policy, ontology)
    # 짧은 대상의 중점은 자기 창 격자 안에 있다 (증거 구간은 대상을 덮는다)
    for w in ws:
        for x in w.targets:
            if w.group is AiGroup.TEMPORAL:
                assert w.t_start_ms <= x.t_start_ms and x.t_end_ms <= w.t_end_ms


def test_long_labels_go_to_their_own_unit(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """상위 구간 전부, `long_label_ms`보다 긴 라벨, `long_object_state_ms`보다 긴 객체 상태는 라벨
    하나가 temporal_long 한 단위다. 짧은 객체 상태는 temporal 창에 든다."""
    long_gap = _label("long-gap", GapPayload(hand=Hand.LEFT, gap_type="idle"), 0, 12_000)
    obj = ObjectStatePayload(
        entity_id="cup_01", class_id="cup", attribute="fill_state", value="full"
    )
    long_state = _label("state-long", obj, 0, 9_000)
    short_state = _label("state-short", obj, 9_000, 12_000)
    policy_ = with_policy(policy, temporal={"long_label_ms": 11_000})
    ws = plan_windows(
        scene.session, [*scene.labels, long_gap, long_state, short_state], policy_, ontology
    )
    long_ids = [[x.label_id for x in w.targets] for w in ws if w.group is AiGroup.TEMPORAL_LONG]
    assert ["long-gap"] in long_ids and ["state-long"] in long_ids
    assert any(
        x.kind == "segment" for w in ws if w.group is AiGroup.TEMPORAL_LONG for x in w.targets
    )
    assert "state-short" in _ids(ws, AiGroup.TEMPORAL)
    for w in ws:
        if w.group is AiGroup.TEMPORAL_LONG:
            assert len(w.targets) == 1 and len(w.wanted_ms) <= policy.temporal.max_frames


def test_split_requests_share_frames(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """대상이 상한을 넘으면 같은 증거 구간·프레임·맥락으로 요청을 나눈다 (분할 순번 0, 1, …)."""
    small = with_policy(policy, temporal={"max_targets": 4})
    ws = [
        w
        for w in plan_windows(scene.session, scene.labels, small, ontology)
        if w.group is AiGroup.TEMPORAL
    ]
    by_key: dict[str, list[PlannedWindow]] = {}
    for w in ws:
        by_key.setdefault(w.window_key, []).append(w)
    assert any(len(v) > 1 for v in by_key.values())
    for group in by_key.values():
        assert [w.split_index for w in group] == list(range(len(group)))
        shared = {
            (
                w.t_start_ms,
                w.t_end_ms,
                w.wanted_ms,
                tuple(x.label_id for x in w.context),
                w.entities,
            )
            for w in group
        }
        assert len(shared) == 1
        assert all(len(w.targets) <= 4 for w in group)
    units = plan_session(
        scene.session, scene.labels, small, ontology, media=scene.media, sensor=ABSENT
    )
    frames = {u.window.window_key: set[tuple[int, ...]]() for u in units}
    for u in units:
        frames[u.window.window_key].add(u.frame_ms)
    assert all(len(f) == 1 for f in frames.values())


def test_flagged_selection_is_deterministic_and_not_keyed(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """flagged: 우선순위 사유가 가리키는 단위와 미상 사이 구간 단위가 먼저, 그다음 감사 표본.

    같은 입력이면 같은 선택이고, 선택 방식은 단위 키를 바꾸지 않는다.
    """
    all_units = plan_session(
        scene.session, scene.labels, policy, ontology, media=scene.media, sensor=ABSENT
    )
    target = next(u for u in all_units if u.group is AiGroup.SPATIAL).targets[0]
    flag = FlaggedSpan(
        reason=ReviewReason.LOW_CONFIDENCE, t_start_ms=0, t_end_ms=1, label_ids=(target.label_id,)
    )
    unknown = _label("gap-unknown", GapPayload(hand=Hand.LEFT, gap_type="unknown"), 1_000, 2_000)
    flagged = with_policy(policy, selection={"mode": "flagged", "audit_ratio": 0.0})
    labels = [*scene.labels, unknown]
    picked = plan_windows(scene.session, labels, flagged, ontology, flagged=[flag])
    assert [w.selection for w in picked] == ["unknown_gap", "flagged"]
    assert picked[1].targets[0].label_id == target.label_id
    # 다른 사유는 고르지 않는다 (flagged 정책의 reasons 밖)
    other = flag.model_copy(update={"reason": ReviewReason.SAMPLE})
    assert [
        w.selection for w in plan_windows(scene.session, labels, flagged, ontology, flagged=[other])
    ] == ["unknown_gap"]
    everything = with_policy(
        policy, selection={"mode": "flagged", "audit_ratio": 1.0, "unknown_gaps": False}
    )
    audited = plan_windows(scene.session, labels, everything, ontology, flagged=[flag])
    assert audited[0].selection == "flagged"
    assert {w.selection for w in audited[1:]} == {"audit"}
    assert len(audited) == len(plan_windows(scene.session, labels, policy, ontology))
    half = with_policy(
        policy, selection={"mode": "flagged", "audit_ratio": 0.5, "unknown_gaps": False}
    )
    first = plan_windows(scene.session, labels, half, ontology)
    assert first == plan_windows(scene.session, labels, half, ontology)
    # 선택 방식은 단위 키에 없다: flagged로 고른 단위의 키는 all로 계획한 같은 단위의 키와 같다
    keyed = plan_session(
        scene.session, labels, flagged, ontology, media=scene.media, sensor=ABSENT, flagged=[flag]
    )
    full = {
        u.unit_id
        for u in plan_session(
            scene.session, labels, policy, ontology, media=scene.media, sensor=ABSENT
        )
    }
    assert {u.unit_id for u in keyed} <= full


def test_unit_ids_change_only_where_inputs_change(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """라벨 하나를 고치면 그 라벨이 대상·맥락으로 든 단위의 키만 바뀐다. 한 스트림의 렌더 해시를
    바꾸면 그 스트림 단위의 키만 바뀐다. 정책 판정 버전을 바꾸면 모두 바뀐다."""
    base = plan_session(
        scene.session, scene.labels, policy, ontology, media=scene.media, sensor=ABSENT
    )
    action = next(x for x in scene.labels if isinstance(x.payload, ActionPayload))
    fixed = action.model_copy(
        update={"label_id": action.label_id + "-fix", "parent_label_id": action.label_id}
    )
    labels = [fixed if x.label_id == action.label_id else x for x in scene.labels]
    after = plan_session(scene.session, labels, policy, ontology, media=scene.media, sensor=ABSENT)
    assert len(after) == len(base)
    for old, new in zip(base, after, strict=True):
        touched = action.label_id in {x.label_id for x in (*old.targets, *old.window.context)}
        assert (old.unit_id != new.unit_id) == touched
    assert any(old.unit_id == new.unit_id for old, new in zip(base, after, strict=True))

    media = dict(scene.media) | {"tp1": StreamMedia(scene.media["tp1"].pts, "c" * 64)}
    rerender = plan_session(
        scene.session, scene.labels, policy, ontology, media=media, sensor=ABSENT
    )
    for old, new in zip(base, rerender, strict=True):
        assert (old.unit_id != new.unit_id) == (old.stream_id == "tp1")

    other = with_policy(policy, temporal={"frames": 7})
    again = plan_session(
        scene.session, scene.labels, other, ontology, media=scene.media, sensor=ABSENT
    )
    assert not {u.unit_id for u in again} & {u.unit_id for u in base}


def test_excluded_labels_never_targets_and_measurements_never_context(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """대상: 모델 출처·검수 전·운영 라벨만. 블라인드·오류 삽입 레코드, 원본 접근 이벤트(불편 반응),
    블러 트랙은 맥락에도 없다. 사람·검수된 라벨은 맥락에 "확정"으로 들어갈 수 있다."""
    grasp = ActionPayload(
        action_id="x1", hand=Hand.LEFT, verb="grasp", t_approach_ms=1_000, t_end_ms=2_000
    )
    reviewed = Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )
    blur = BlurTrackPayload(target="face", keyframes=(BoxKeyframe(t_ms=1_000, x=0, y=0, w=5, h=5),))
    extra = [
        _label("blind", grasp, 1_000, 2_000, measurement="blind"),
        _label("seeded", grasp, 1_000, 2_000, seeded_error=True),
        _label(
            "human",
            grasp,
            1_000,
            2_000,
            provenance=Provenance(source=Source.HUMAN),
            confidence=None,
        ),
        _label("done", grasp, 1_000, 2_000, verification=reviewed),
        _label("discomfort", EventPayload(event_type="discomfort", severity=1), 1_000, 1_500),
        _label("blur", blur, 1_000, 1_000, stream_id="bodycam"),
    ]
    ws = plan_windows(scene.session, [*scene.labels, *extra], policy, ontology)
    in_targets = {x.label_id for w in ws for x in w.targets}
    in_context = {x.label_id for w in ws for x in w.context}
    assert not in_targets & {x.label_id for x in extra}
    assert not in_context & {"blind", "seeded", "discomfort", "blur"}
    assert {"human", "done"} <= in_context


def test_frames_come_from_pts_index_vfr(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """VFR PTS 인덱스: 고른 프레임 시각은 인덱스의 실제 프레임 시각이다. 상한·최소 간격을 지킨다."""
    times = vfr_times(np.random.default_rng(5), scene.session.duration_ms)
    pts = PtsIndex(
        Fraction(1, 1000), np.asarray(times, dtype=np.int64), np.ones(len(times), dtype=bool)
    )
    media = {s: StreamMedia(pts, RENDER[s]) for s in RENDER}
    units = plan_session(scene.session, scene.labels, policy, ontology, media=media, sensor=ABSENT)
    for u in units:
        assert set(u.frame_ms) <= set(times)
        assert list(u.frame_ms) == sorted(set(u.frame_ms))
        assert u.frame_ms == tuple(times[i] for i in u.frame_idx)
        limit = policy.spatial.frames if u.group is AiGroup.SPATIAL else policy.temporal.max_frames
        assert 1 <= len(u.frame_ms) <= limit


def test_estimate_and_media_free_planning_agree(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """추정은 영상 없이 같은 창 논리로 센다. 블러본이 없는 스트림은 계획하지 않는다."""
    est = estimate_units(scene.session, scene.labels, policy, ontology)
    ws = plan_windows(scene.session, scene.labels, policy, ontology)
    assert [(e.group, e.stream_id, e.targets) for e in est] == [
        (w.group, w.stream_id, len(w.targets)) for w in ws
    ]
    only_body = {"bodycam": scene.media["bodycam"]}
    units = plan_session(
        scene.session, scene.labels, policy, ontology, media=only_body, sensor=ABSENT
    )
    assert {u.stream_id for u in units} == {"bodycam"}


def test_sensor_hash_only_for_hand_state_units(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """장갑 문장을 쓰는 단위(손 상태 대상이 있는 시간 단위)만 센서 해시가 단위 키에 들어간다."""
    sensor = SensorContext.of({Hand.RIGHT: [(1_200.0, 2_800.4)]})
    units = plan_session(
        scene.session, scene.labels, policy, ontology, media=scene.media, sensor=sensor
    )
    for u in units:
        uses = u.group is not AiGroup.SPATIAL and any(x.kind == "hand_state" for x in u.targets)
        assert u.sensor_sha256 == (sensor.sha256 if uses else "absent")
    assert any(u.sensor_sha256 != "absent" for u in units)


def test_uncovered_targets_by_version_and_render(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """판정된 단위에 대상으로 든 라벨은 덮였고, 렌더 해시·버전이 다르면 덮이지 않은 것으로 본다."""
    units = plan_session(
        scene.session, scene.labels, policy, ontology, media=scene.media, sensor=ABSENT
    )
    oracle = OracleJudge(scene.labels)
    done = [
        judge_unit(
            oracle, u, images_for(u, policy), ontology=ontology, policy=policy, sensor=ABSENT
        )
        .record("air-run-" + "0" * 24, FIXED_TIME + timedelta(minutes=k))
        for k, u in enumerate(units[:2])
    ]  # fmt: skip
    ws = plan_windows(scene.session, scene.labels, policy, ontology)
    version = policy.verdict_version
    left = uncovered_targets(ws, done, version=version, render_hashes=RENDER)
    covered = {i for u in done for i in u.target_label_ids}
    assert set(left) == {x.label_id for w in ws for x in w.targets} - covered
    assert left == sorted(left)
    assert (
        set(
            uncovered_targets(ws, done, version="aireview-000000000000+vlm-x", render_hashes=RENDER)
        )
        >= covered
    )
    moved = {"bodycam": "d" * 64, "tp1": RENDER["tp1"]}
    assert set(uncovered_targets(ws, done, version=version, render_hashes=moved)) >= covered
