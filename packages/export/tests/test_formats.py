from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dlp_export.coco import write_coco
from dlp_export.intervals import write_intervals
from dlp_export.lerobot import Vocab, build_episode, state_names
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.source import (
    ExportError,
    ExportSession,
    ExportSource,
    assert_no_raw,
    label_states,
    select_labels,
)
from dlp_media.pts import build_pts_index
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.export import IntervalFile
from dlp_schema.labels import VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME

from .conftest import ROOT, Scenario


def source(sc: Scenario, policy: ExportPolicy, include_unreviewed: bool = False) -> ExportSource:
    states = label_states(policy, include_unreviewed)
    version = DatasetVersion(
        version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="local-snapshot://x/dv1", splits={"s1": Split.TRAIN},
    )  # fmt: skip
    return ExportSource(
        version,
        [ExportSession(sc.session, Split.TRAIN, select_labels(sc.labels, policy, states))],
        states,
    )


def test_verification_policy(scenario: Scenario, policy: ExportPolicy) -> None:
    default = {x.label_id for x in source(scenario, policy).sessions[0].labels}
    # 미검수·블러·오류 삽입은 기본으로 빠지고, 표본 검증과 사람 라벨은 들어간다
    assert default == {f"s1-{k}" for k in ("box", "kp", "hs", "tip", "tsc", "act", "task")}
    allx = {
        x.label_id for x in source(scenario, policy, include_unreviewed=True).sessions[0].labels
    }
    assert allx == default | {"s1-unrev"}  # 블러·오류 삽입은 옵션을 켜도 빠진다
    assert label_states(policy, True)[-1] is VerificationState.UNREVIEWED


def test_interval_json_validates_against_published_schema(
    scenario: Scenario, policy: ExportPolicy, tmp_path: Path
) -> None:
    counts = write_intervals(
        source(scenario, policy), policy, tmp_path, export_id="e1", now=FIXED_TIME,
        ids=Pseudonymizer.for_export("e1", b"secret", enabled=True),
    )  # fmt: skip
    assert counts == {"s1": 4}  # 손 상태, 관계, 행동, 작업 구간 (공간 라벨은 COCO·LeRobot으로)
    data = json.loads((tmp_path / "intervals" / "s1.json").read_text())
    schema = json.loads((ROOT / "schemas" / "export_intervals.schema.json").read_text())
    jsonschema.validate(data, schema)
    f = IntervalFile.model_validate(data)
    assert [x.verification for x in f.labels] == [
        VerificationState.HUMAN_CORRECTED, VerificationState.SAMPLE_VERIFIED,
        VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED,
    ]  # fmt: skip
    text = (tmp_path / "intervals" / "s1.json").read_text()
    assert "reviewer-7" not in text  # 검수자 ID는 내보내지 않는다
    # 작업자·장소는 이 내보내기의 가명
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    assert (f.worker_id, f.site_id) == (ids.worker("w01"), ids.site("site01"))
    assert "w01" not in text and "site01" not in text


def test_pseudonyms_are_per_export() -> None:
    a = Pseudonymizer.for_export("export-a", b"secret", enabled=True)
    b = Pseudonymizer.for_export("export-b", b"secret", enabled=True)
    assert a.worker("w1") == a.worker("w1") != a.worker("w2")  # 한 내보내기 안에서 일관
    assert a.worker("w1") != b.worker("w1")  # 내보내기 사이에서는 이어지지 않는다
    assert a.worker("w1") != a.site("w1")  # 종류가 다르면 다르다
    # 비밀값을 아는 내부만 다시 계산할 수 있다
    assert Pseudonymizer.for_export("export-a", b"secret", enabled=True).worker("w1") == a.worker(
        "w1"
    )
    assert Pseudonymizer.for_export("export-a", b"other", enabled=True).worker("w1") != a.worker(
        "w1"
    )
    # 비밀값이 없으면 실행마다 임의 값
    assert Pseudonymizer.for_export("export-a", None, enabled=True).worker("w1") != a.worker("w1")
    assert Pseudonymizer.for_export("export-a", None, enabled=False).worker("w1") == "w1"


def test_coco_loads_with_pycocotools(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    r = write_coco(
        source(scenario, policy), policy, ontology, scenario.labeling, out, tmp_path / "w",
        export_id="e1", now=FIXED_TIME,
    )  # fmt: skip
    assert r.dropped == {"keyframe_between_frames": 1}
    with contextlib.redirect_stdout(io.StringIO()):
        coco: Any = COCO(str(out / "coco" / "annotations.json"))
    t = scenario.times
    assert sorted(img["t_ms"] for img in coco.dataset["images"]) == [t[2], t[4], t[6]]
    for img in coco.dataset["images"]:
        assert (out / "coco" / img["file_name"]).stat().st_size > 0
    cup = coco.getCatIds(catNms=["cup"])[0]
    hand = coco.getCatIds(catNms=["hand"])[0]
    boxes = coco.loadAnns(coco.getAnnIds(catIds=[cup]))
    assert [a["bbox"] for a in boxes] == [[8, 8, 16, 12], [10, 8, 16, 12]]
    assert {a["verification"] for a in boxes} == {"human_approved"}
    kps = coco.loadAnns(coco.getAnnIds(catIds=[hand]))
    assert [a["num_keypoints"] for a in kps] == [21, 21]
    assert kps[0]["keypoints"][-1] == 1  # 가려진 관절은 v=1
    assert not coco.getAnnIds(catIds=coco.getCatIds(catNms=["sponge"]))  # 미검수 제외
    # 정답을 그대로 예측으로 넣으면 COCO 평가가 AP 1
    dets: list[dict[str, Any]] = [{**a, "score": 1.0} for a in boxes]
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "bbox")
        ev.params.catIds = [cup]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert ev.stats[0] == pytest.approx(1.0)


def test_coco_person_boxes_work_with_keypoint_eval(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """객체 person 범주에는 coco17 키포인트가 붙는다. 박스만의 person 주석도 keypoints(0)와
    num_keypoints 0을 가져야 COCO 키포인트 평가가 돈다."""
    from dlp_schema.testing import make_label

    t = scenario.times
    person_box = make_label(
        {"kind": "box_track", "entity_id": "person_1", "class_id": "person",
         "keyframes": [{"t_ms": t[2], "x": 2, "y": 2, "w": 40, "h": 40}]},
        label_id="s1-person", session_id="s1", stream_id="bodycam", t_start_ms=t[2],
        t_end_ms=t[2],
    )  # fmt: skip
    body = make_label(
        {"kind": "keypoint_track", "entity_id": "person_1", "skeleton": "coco17",
         "keyframes": [{"t_ms": t[2], "points": [{"x": 5 + i, "y": 6 + i, "visibility": 2}
                                                 for i in range(17)]}]},
        label_id="s1-body", session_id="s1", stream_id="bodycam", t_start_ms=t[2],
        t_end_ms=t[2],
    )  # fmt: skip
    sc = Scenario(scenario.session, [person_box, body], scenario.times, scenario.labeling)
    out = tmp_path / "out"
    write_coco(
        source(sc, policy), policy, ontology, sc.labeling, out, tmp_path / "w",
        export_id="e1", now=FIXED_TIME,
    )  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        coco: Any = COCO(str(out / "coco" / "annotations.json"))
    person = coco.getCatIds(catNms=["person"])
    assert len(person) == 1 and len(coco.loadCats(person)[0]["keypoints"]) == 17
    anns = coco.loadAnns(coco.getAnnIds(catIds=person))
    box_only = next(a for a in anns if a["label_id"] == "s1-person")
    assert box_only["num_keypoints"] == 0 and box_only["keypoints"] == [0] * 51
    dets = [{**a, "score": 1.0} for a in anns if a["num_keypoints"] > 0]
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(dets), "keypoints")
        ev.params.catIds = person
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    assert ev.stats[0] == pytest.approx(1.0)


def test_lerobot_frame_features(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    lp = policy.lerobot
    vocab = Vocab.from_ontology(ontology)
    video = tmp_path / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    index = build_pts_index(video)
    stream = scenario.session.streams[0]
    ep = build_episode(
        scenario.session, stream, index, (64, 48), source(scenario, policy).sessions[0].labels,
        vocab, lp,
    )  # fmt: skip
    t = scenario.times
    # 고정 30 fps: 블러본 길이 안의 시각마다, 그 시각에 보이던 프레임 (PTS)
    assert np.allclose(np.diff(ep.times_ms), 1000 / 30)
    for k in (0, 5, 17):
        tk = ep.times_ms[k]
        assert (
            t[ep.frame_index[k]]
            <= tk
            < (t[ep.frame_index[k] + 1] if ep.frame_index[k] + 1 < len(t) else 1e9)
        )
    names = state_names(lp)
    col = {n: i for i, n in enumerate(names)}

    def at(ms: float) -> int:
        return int(np.argmin(np.abs(ep.times_ms - ms)))

    # 손 2D: 두 키프레임 사이 보간 (간격이 interp_max_gap_ms 안이면)
    assert t[6] - t[2] <= lp.interp_max_gap_ms  # 시나리오 전제
    k = at((t[2] + t[6]) / 2)
    tk = ep.times_ms[k]
    expected_x = (10 + 20 * (tk - t[2]) / (t[6] - t[2])) / 64
    assert ep.state[k, col["right.wrist.v"]] == 2.0
    assert ep.state[k, col["right.wrist.x"]] == pytest.approx(expected_x, abs=1e-5)
    assert ep.state[k, col["right.pinky_tip.v"]] == 1.0  # 가려진 관절은 v=1 그대로
    assert ep.state[k, col["left.wrist.v"]] == 0.0
    assert ep.state[at(t[6] + 300), col["right.wrist.v"]] == 0.0  # 키프레임 밖
    # 손 상태·쥔 도구·작용부 3D·도구-표면 접촉·동사·작업 (400 ms: 모두 걸림)
    k = at(400)
    hs = ep.hand_state[k]
    assert list(hs[4:]) == [
        1,
        vocab.contact_target_kinds.index("tool"),
        vocab.grasp_types.index("tool_grip"),
        vocab.hand_roles.index("active"),
    ]
    assert list(hs[:4]) == [-1, -1, -1, -1]
    assert ep.state[k, col["right.tool_tip.present"]] == 1.0
    assert ep.state[k, col["right.tool_tip.x3d"]] == pytest.approx(ep.times_ms[k] / 1000, abs=1e-4)
    assert list(ep.tool_surface_contact[k]) == [-1, 1]
    assert list(ep.verb[k]) == [-1, vocab.verbs.index("carry")]
    # 검증 등급 [2D, 3D, 손 상태, 행동, 도구-표면 접촉, 작업]:
    # 3D 사람(3), 손 상태 표본 검증(1), 행동 사람 승인(2), 접촉 관계·작업 구간 사람(3)
    assert list(ep.verification[k][1:]) == [3, 1, 2, 3, 3]
    assert ep.tasks[k] == "floor_sweep_mop"
    assert ep.tasks[at(800)] == "cleaning"  # 작업 구간 밖은 도메인
    assert list(ep.tool_surface_contact[at(800)]) == [-1, -1]  # 쥔 도구 없음
    assert np.array_equal(ep.action[:-1], ep.state[1:])
    # 실제로 프레임 특징에 들어간 라벨만 쓴 라벨로 센다 (박스는 LeRobot 특징이 아니다)
    assert ep.used == {f"s1-{k}" for k in ("kp", "hs", "tip", "tsc", "act", "task")}


def test_lerobot_counts_only_written_labels(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """LeRobot 종류라도 어느 프레임에도 쓰지 않은 라벨(영상 밖 시각, 다른 스트림)은 세지 않는다."""
    from dlp_schema.testing import make_label

    video = tmp_path / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    labels = source(scenario, policy).sessions[0].labels
    late = make_label(
        {"kind": "action", "action_id": "late", "hand": "left", "verb": "carry",
         "t_approach_ms": 10_000_000, "t_end_ms": 10_000_500},
        label_id="s1-late", session_id="s1", t_start_ms=10_000_000, t_end_ms=10_000_500,
    )  # fmt: skip
    other = next(x for x in labels if x.label_id == "s1-kp").model_copy(
        update={"label_id": "s1-kp-third", "stream_id": "third"}
    )
    ep = build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48),
        [*labels, late, other], Vocab.from_ontology(ontology), policy.lerobot,
    )  # fmt: skip
    assert "s1-late" not in ep.used and "s1-kp-third" not in ep.used
    assert "s1-act" in ep.used


def test_raw_uri_guard(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"uri": "s3://dlp-raw/sessions/x/bodycam.mp4"}')
    with pytest.raises(ExportError, match="원본"):
        assert_no_raw(tmp_path, "dlp-raw")


def test_offset_third_person_keyframes_use_stream_time(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """공간 라벨 키프레임은 그 스트림 영상의 PTS 시각이다 (ADR 0019). 3인칭 스트림이 마스터와
    어긋나 있어도(오프셋·클럭 배율) 키프레임은 그 영상 프레임에 그대로 맞는다."""
    from dlp_schema.session import Stream, StreamKind, SyncMethod

    from .conftest import scenario_labels

    third = Stream(
        stream_id="third", kind=StreamKind.THIRD_PERSON, uri="s3://dlp-raw/x/third.mp4",
        sync_method=SyncMethod.QR_SLATE, offset_ms=1234.567, clock_scale=1.0001,
    )  # fmt: skip
    session = scenario.session.model_copy(update={"streams": (*scenario.session.streams, third)})
    t = scenario.times
    labels = [
        x.model_copy(update={"stream_id": "third", "label_id": f"{x.label_id}-3"})
        for x in scenario_labels("s1", t)
        if x.label_id in ("s1-box", "s1-kp")
    ]
    # 3인칭 블러본 자리에 같은 VFR 영상을 둔다 (키프레임은 이 영상의 PTS 시각)
    video = tmp_path / "third_src.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    from dlp_media.storage import sha256_file

    scenario.labeling.put_file("sessions/s1/blurred/third.mp4", video, sha256_file(video))
    states = label_states(policy, False)
    version = DatasetVersion(
        version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="local-snapshot://x/dv1", splits={"s1": Split.TRAIN},
    )  # fmt: skip
    src = ExportSource(
        version,
        [ExportSession(session, Split.TRAIN, select_labels(labels, policy, states))],
        states,
    )
    out = tmp_path / "out"
    r = write_coco(src, policy, ontology, scenario.labeling, out, tmp_path / "w",
                   export_id="e1", now=FIXED_TIME)  # fmt: skip
    data = json.loads((out / "coco" / "annotations.json").read_text())
    third_imgs = [i for i in data["images"] if i["stream_id"] == "third"]
    assert sorted(i["t_ms"] for i in third_imgs) == [t[2], t[4], t[6]]
    assert r.dropped == {"keyframe_between_frames": 1}  # 원래 프레임 사이에 둔 키프레임 하나만
    for img in data["images"]:
        assert (out / "coco" / img["file_name"]).exists()
    assert len({i["file_name"] for i in data["images"]}) == len(data["images"])
