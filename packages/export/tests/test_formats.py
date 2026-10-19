"""내보내기 형식 단위 테스트 (DB 없이).

검증 정책, 구간 JSON, 가명, COCO, LeRobot 특징, 원본 위치 검사를 본다.

정답은 conftest의 합성 시나리오(프레임 시각을 아는 VFR 블러본과 라벨)에서 온다. DB를 쓰지 않도록
`source()`가 `load_source` 대신 `ExportSource`를 직접 만든다.
"""

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
from dlp_export.intervals import exported, write_intervals
from dlp_export.lerobot import Vocab, build_episode, state_names
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer, check_secret
from dlp_export.source import (
    ExportError,
    ExportSession,
    ExportSource,
    assert_no_raw,
    label_states,
    select_labels,
)
from dlp_media.pts import build_pts_index
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderMeta, operational_blur, render_hash, write_render_meta
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.export import IntervalFile
from dlp_schema.labels import DescriptionPayload, VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label

from .conftest import ROOT, Scenario, model

NOIDS = Pseudonymizer(None)  # 가명 처리 없이 (내부 ID를 그대로 확인하는 테스트)


def test_segment_children_match_exported_action_ids() -> None:
    """구간의 child_ids도 가명으로 바뀌어 행동 참조가 이어지고 내부 세션 ID가 남지 않는다."""
    sid = "private-session-001"
    action_id = f"{sid}-right-a0"
    segment = make_label(
        {"kind": "segment", "segment_id": f"{sid}-task", "level": "task",
         "ref_id": "floor_sweep_mop", "child_ids": [action_id]}, session_id=sid,
    )  # fmt: skip
    action = make_label(action_payload(action_id=action_id), session_id=sid)
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    seg = exported(segment, ids).model_dump(mode="json")
    act = exported(action, ids).model_dump(mode="json")
    assert seg["payload"]["child_ids"] == [act["payload"]["action_id"]]
    assert sid not in json.dumps([seg, act])
    # 중첩 참조도 바꾸지만 온톨로지 값·서술 문자열은 같은 문자열이어도 그대로 둔다.
    nested = ids.payload_ids(sid, {"nested": [{"child_ids": (action_id,)}], "text": sid})
    assert nested == {"nested": [{"child_ids": [ids.ref(sid, action_id)]}], "text": sid}
    # 합법적인 세션 ID가 온톨로지 과제 ID와 같아도 과제 정의 참조는 보존한다.
    taxonomy_sid = "floor_sweep_mop"
    taxonomy = segment.model_copy(
        update={
            "session_id": taxonomy_sid,
            "payload": segment.payload.model_copy(
                update={
                    "segment_id": f"{taxonomy_sid}-task",
                    "child_ids": (f"{taxonomy_sid}-skill",),
                }
            ),
        }
    )
    payload = exported(taxonomy, ids).model_dump(mode="json")["payload"]
    assert payload["ref_id"] == taxonomy_sid
    assert payload["child_ids"] == [ids.ref(taxonomy_sid, f"{taxonomy_sid}-skill")]
    box = make_label(
        {
            "kind": "box_track",
            "entity_id": "mop_01",
            "class_id": "mop",
            "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 10, "h": 10}],
        },
        session_id="mop",
        stream_id="bodycam",
    )
    assert exported(box, ids).model_dump(mode="json")["payload"]["class_id"] == "mop"


def source(sc: Scenario, policy: ExportPolicy, include_unreviewed: bool = False) -> ExportSource:
    """시나리오 → `ExportSource` (세션 s1 하나, 학습 분할, 정책으로 고른 라벨). DB 없이 만든다."""
    states = label_states(policy, include_unreviewed)
    version = DatasetVersion(
        version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="local-snapshot://x/dv1", splits={"s1": Split.TRAIN},
    )  # fmt: skip
    return ExportSource(
        version,
        [
            ExportSession(
                sc.session,
                Split.TRAIN,
                select_labels(sc.labels, policy, states, ()),
                sc.render_hashes,
            )
        ],
        states,
    )


def test_verification_policy(scenario: Scenario, policy: ExportPolicy) -> None:
    """기본 검증 정책과 --include-unreviewed가 고르는 라벨.

    정답: 기본은 사람 라벨(kp·tip·tsc·task)과 승인·표본 검증 모델 라벨(box·act·hs).
    옵션을 켜면 미검수(unrev)만 더해지고 블러·오류 삽입·측정은 여전히 빠진다.
    """
    default = {x.label_id for x in source(scenario, policy).sessions[0].labels}
    # 미검수·블러·오류 삽입은 기본으로 빠지고, 표본 검증과 사람 라벨은 들어간다
    assert default == {f"s1-{k}" for k in ("box", "kp", "hs", "tip", "tsc", "act", "task")}
    allx = {
        x.label_id for x in source(scenario, policy, include_unreviewed=True).sessions[0].labels
    }
    assert allx == default | {"s1-unrev"}  # 블러·오류 삽입은 옵션을 켜도 빠진다
    assert label_states(policy, True)[-1] is VerificationState.UNREVIEWED


def test_interval_json_validates_against_published_schema(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """구간 JSON이 공개 JSON Schema를 통과하고, 내부 ID·검수자 ID가 결과에 없다.

    정답: 구간 종류 라벨 4개(hs·tsc·act·task)가 시작 시각 순(task 0, hs 100, act 200, tsc 300)으로
    나오고 검증 상태도 그 순서. 작업자·장소·세션·라벨 ID는 이 내보내기의 가명.
    """
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    counts = write_intervals(
        source(scenario, policy), policy, tmp_path, export_id="e1", now=FIXED_TIME, ids=ids,
        ontology=ontology,
    )  # fmt: skip
    sid = ids.session("s1")
    assert counts == {sid: 4}  # 손 상태, 관계, 행동, 작업 구간 (공간 라벨은 COCO·LeRobot으로)
    data = json.loads((tmp_path / "intervals" / f"{sid}.json").read_text())
    schema = json.loads((ROOT / "schemas" / "export_intervals.schema.json").read_text())
    jsonschema.validate(data, schema)
    f = IntervalFile.model_validate(data)
    assert [x.verification for x in f.labels] == [
        VerificationState.HUMAN_CORRECTED, VerificationState.SAMPLE_VERIFIED,
        VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED,
    ]  # fmt: skip
    text = (tmp_path / "intervals" / f"{sid}.json").read_text()
    assert "reviewer-7" not in text  # 검수자 ID는 내보내지 않는다
    # 작업자·장소·세션·라벨은 이 내보내기의 가명
    assert (f.worker_id, f.site_id) == (ids.worker("w01"), ids.site("site01"))
    assert "w01" not in text and "site01" not in text
    assert f.session_id == sid and "s1-" not in text and '"s1"' not in text
    assert {x.label_id for x in f.labels} == {
        ids.label(f"s1-{k}") for k in ("hs", "tsc", "act", "task")
    }
    # 형식 1.1: 온톨로지 한·영 표시 이름 표 (ADR 0043)
    assert f.term_names == ontology.display_names()
    assert f.term_names["verbs:rub"].en == "rub" and f.term_names["verbs:rub"].ko == "문지르다"
    # 형식 1.2: 객체 물리 사전값 표. 측정값이 아니라 사전값이다 (ADR 0049)
    assert f.format_version == "1.2"
    priors = data["physical_priors"]
    assert priors["measured"] is False and "NOT measurements" in priors["conventions"]["nature"]
    assert priors["materials"]["ceramic"] == {"ko": "도자기·세라믹", "en": "Ceramic or porcelain"}
    assert priors["tactile_properties"]["slippery_when_wet"]["en"] == "Slippery when wet"
    bucket = priors["objects"]["bucket"]
    assert bucket["source"] == "prior_estimate" and bucket["material"] == "plastic"
    assert bucket["mass_kg"]["by_state"]["fill_state"]["full"] == {"min": 5.0, "max": 16.0}
    assert "person" not in priors["objects"]  # 사전값이 없는 클래스는 싣지 않는다
    assert all(o["source"].startswith("prior_") for o in priors["objects"].values())


def test_bilingual_descriptions_are_exported_with_linked_ids(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """한국어 설명과 영어 번역이 둘 다 나가고, 번역의 원문 참조도 같은 가명 라벨 ID다 (ADR 0043).

    정답: 설명 2개(ko 원문 s1-dko, en 번역 s1-den). en의 source_label_id = ko의 가명 label_id,
    segment_id는 둘 다 행동 ID의 가명. 내부 라벨 ID는 결과에 없다.
    """
    act_id = "s1-right-a0"
    ko = make_label(
        {"kind": "description", "segment_id": act_id, "text": "걸레로 바닥을 닦는다"},
        label_id="s1-dko", session_id="s1", t_start_ms=200, t_end_ms=900,
    )  # fmt: skip
    en = make_label(
        {"kind": "description", "segment_id": act_id, "text": "Wipes the floor with a cloth",
         "language": "en", "source_label_id": "s1-dko"},
        label_id="s1-dko:en1a2b3c4d", session_id="s1", t_start_ms=200, t_end_ms=900,
    )  # fmt: skip
    src = source(scenario, policy)
    sess = src.sessions[0]
    src = ExportSource(
        src.version,
        [ExportSession(sess.session, sess.split, [*sess.labels, ko, en], sess.render_hashes)],
        src.label_states,
    )
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    write_intervals(
        src, policy, tmp_path, export_id="e1", now=FIXED_TIME, ids=ids, ontology=ontology
    )
    path = tmp_path / "intervals" / f"{ids.session('s1')}.json"
    f = IntervalFile.model_validate_json(path.read_text())
    descs = {x.payload.language: x for x in f.labels if isinstance(x.payload, DescriptionPayload)}
    assert set(descs) == {"ko", "en"}
    en_payload = descs["en"].payload
    assert isinstance(en_payload, DescriptionPayload)
    assert en_payload.source_label_id == descs["ko"].label_id == ids.label("s1-dko")
    assert en_payload.text == "Wipes the floor with a cloth"
    assert en_payload.segment_id == ids.ref("s1", act_id)
    assert "s1-dko" not in path.read_text()
    # 가명 처리를 끄면 원래 ID 그대로
    off = Pseudonymizer(None)
    assert exported(en, off).payload == en.payload


def test_pseudonyms_are_per_export() -> None:
    """가명 성질 (ADR 0021·0027).

    내보내기 안 일관, 내보내기 사이 단절, 종류별 분리, 비밀값 재현성, 비밀값 없으면 임의,
    꺼지면 원래 값, 세션 ID가 든 다른 ID는 세션 부분만 바뀜(`ref`·`payload_ids`).
    """
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
    # 세션·라벨 ID도 내보내기마다 다르고, 세션 ID가 들어간 다른 ID는 그 부분만 바뀐다
    assert a.session("s1") != b.session("s1") and a.label("s1-a") != b.label("s1-a")
    assert a.ref("s1", "s1-right-a100") == f"{a.session('s1')}-right-a100"
    assert a.ref("s1", "cup_1") == "cup_1"
    got = a.payload_ids("s1", {"action_id": "s1-x", "verb": "s1", "n": [{"segment_id": "s1"}]})
    assert got == {
        "action_id": f"{a.session('s1')}-x", "verb": "s1", "n": [{"segment_id": a.session("s1")}]
    }  # fmt: skip
    off = Pseudonymizer.for_export("export-a", None, enabled=False)
    assert (off.session("s1"), off.label("l"), off.ref("s1", "s1-x")) == ("s1", "l", "s1-x")


def test_ref_replaces_only_whole_session_tokens() -> None:
    """`ref`는 세션 ID가 온전한 토큰(앞뒤가 시작·끝 또는 구분자)일 때만 바꾼다.

    감사 회귀: 예전에는 부분 문자열 치환이라 세션 "s1"이 "s10-right-…"의 앞부분을 바꿔
    "session-…0-right-…"가 됐다.
    """
    a = Pseudonymizer.for_export("export-a", b"secret", enabled=True)
    alias = a.session("s1")
    assert a.ref("s1", "s10-right-a100") == "s10-right-a100"  # 다른 세션 ID의 일부
    assert a.ref("s1", "xs1-a") == "xs1-a"  # 앞에 영숫자가 붙은 경우
    assert a.ref("s1", "s1") == alias  # 정확히 일치
    assert a.ref("s1", "s1-right-a100") == f"{alias}-right-a100"  # 접두사 + 구분자
    assert a.ref("s1", "person_01:s1") == f"person_01:{alias}"  # 구분자 뒤 토큰
    # 특수문자("." 등)가 든 세션 ID도 글자 그대로 맞춘다 (정규식 이스케이프)
    assert a.ref("s.1", "s.1-a") == f"{a.session('s.1')}-a"


def test_dev_secret_only_in_dev(policy: ExportPolicy) -> None:
    """.env.example의 개발용 비밀값은 DLP_ENV=dev에서만 받는다 (공개 값이라 가명을 되짚는다)."""
    dev = policy.ids.dev_secrets[0]
    check_secret(dev, policy.ids, {"DLP_ENV": "dev"})
    check_secret("real-secret", policy.ids, {})
    check_secret(None, policy.ids, {})
    for env in ({}, {"DLP_ENV": "prod"}):
        with pytest.raises(ValueError, match="개발용"):
            check_secret(dev, policy.ids, env)


def test_coco_loads_with_pycocotools(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """COCO 결과를 pycocotools로 읽고 평가할 수 있다.

    정답: 이미지는 박스·관절 키프레임 프레임 t[2]·t[4]·t[6] 셋. 박스 키프레임 t[3]+5는 프레임
    사이라 1건 버림. 미검수 sponge 박스는 없음. 정답을 그대로 예측으로 넣으면 bbox AP = 1.
    """
    out = tmp_path / "out"
    ids = Pseudonymizer.for_export("e1", b"secret", enabled=True)
    r = write_coco(
        source(scenario, policy), policy, ontology, scenario.labeling, out, tmp_path / "w",
        export_id="e1", now=FIXED_TIME, ids=ids,
    )  # fmt: skip
    assert r.dropped == {"keyframe_between_frames": 1}
    assert r.sessions == {"s1"}  # 이력용 결과는 내부 ID
    text = (out / "coco" / "annotations.json").read_text()
    assert "s1-" not in text and '"s1"' not in text
    with contextlib.redirect_stdout(io.StringIO()):
        coco: Any = COCO(str(out / "coco" / "annotations.json"))
    t = scenario.times
    assert sorted(img["t_ms"] for img in coco.dataset["images"]) == [t[2], t[4], t[6]]
    for img in coco.dataset["images"]:
        assert (out / "coco" / img["file_name"]).stat().st_size > 0
        assert img["session_id"] == ids.session("s1")
        assert img["file_name"].startswith(f"images/{ids.session('s1')}__")
    cup = coco.getCatIds(catNms=["cup"])[0]
    hand = coco.getCatIds(catNms=["hand"])[0]
    boxes = coco.loadAnns(coco.getAnnIds(catIds=[cup]))
    assert [a["bbox"] for a in boxes] == [[8, 8, 16, 12], [10, 8, 16, 12]]
    assert {a["verification"] for a in boxes} == {"human_approved"}
    assert {a["label_id"] for a in boxes} == {ids.label("s1-box")}
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
    # 범주에 한·영 표시 이름이 붙는다 (name은 온톨로지 ID 그대로, ADR 0043)
    cat = coco.loadCats([cup])[0]
    assert (cat["name"], cat["name_ko"], cat["name_en"]) == ("cup", "컵", "Cup")
    # 객체 범주에 물리 사전값(측정값 아님)이, 최상위에 단위·재질·촉감 이름이 붙는다 (ADR 0049)
    assert cat["physical_priors"]["measured"] is False
    assert cat["physical_priors"]["source"] == "prior_estimate"
    assert cat["physical_priors"]["material"] == "ceramic"
    assert "physical_priors" not in coco.loadCats(coco.getCatIds(catNms=["person"]))[0]
    top = coco.dataset["physical_priors"]
    assert top["measured"] is False and "objects" not in top
    assert top["materials"]["ceramic"]["en"] == "Ceramic or porcelain"


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
    sc = Scenario(
        scenario.session, [person_box, body], scenario.times, scenario.labeling,
        scenario.render_hashes,
    )  # fmt: skip
    out = tmp_path / "out"
    write_coco(
        source(sc, policy), policy, ontology, sc.labeling, out, tmp_path / "w",
        export_id="e1", now=FIXED_TIME, ids=NOIDS,
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
    """LeRobot 프레임 특징이 시나리오 정답과 같다 (격리 환경 없이 `build_episode`만).

    정답 근거: 프레임은 그 시각에 보이던 블러본 프레임(PTS), 손 2D는 t[2]·t[6] 키프레임의 선형 보간,
    400 ms에는 손 상태(대걸레 쥠)·작용부 3D(x = t/1000)·도구-표면 접촉·동사 carry·작업
    floor_sweep_mop이 모두 걸리고, 검증 등급은 각 라벨의 상태에서 나온다.
    """
    lp = policy.lerobot
    vocab = Vocab.from_ontology(ontology)
    # 어휘의 한·영 표시 이름은 어휘 순서와 같다 (ADR 0043)
    names = vocab.as_json()["term_names"]
    assert len(names["verb"]) == len(vocab.verbs)
    rub = vocab.verbs.index("rub")
    assert names["verb"][rub] == {"ko": "문지르다", "en": "rub"}
    assert len(names["task"]) == len(vocab.tasks)
    # 물리 사전값 표 (측정값 아님, ADR 0049)
    priors = vocab.as_json()["physical_priors"]
    assert priors["measured"] is False
    assert priors["objects"]["mop"]["part_materials"] == {"mop_head": "fabric"}
    assert priors["objects"]["floor"]["static_friction"]["by_state"]["wetness"]["wet"]["max"] == 0.5
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
        """마스터 시각 ms에 가장 가까운 에피소드 프레임 번호."""
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


def test_lerobot_hand_point_confidence_follows_sample_confidence(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """ADR 0045: 손 3D 점마다 표본 신뢰도를 `annotation.hand_points_3d_confidence`에 싣는다.

    시나리오: 오른손 손목 3D 표본 100 ms(신뢰도 0.4)·300 ms(0.8), 엄지 끝 3D는 신뢰도 없는 표본
    (예전·사람 궤적). 정답: 200 ms는 두 표본의 선형 보간 0.6, 엄지 끝은 값이 있어도(present 1)
    신뢰도 -1, 왼손·궤적 밖 시각은 -1. 열 순서는 손(왼, 오른) x hand_points_3d.
    """
    lp = policy.lerobot
    video = tmp_path / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    base = {"session_id": "s1", "stream_id": "bodycam", "ontology_version": "1.0.0"}
    wrist = make_label(
        {"kind": "trajectory3d", "entity_id": "right_hand", "part": "wrist", "frame": "camera",
         "source_3d": "mono_depth", "intrinsics": "calibrated",
         "samples": [{"t_ms": 100, "x": 0.1, "y": 0.0, "z": 0.5, "method": "hand_fit",
                      "confidence": 0.4},
                     {"t_ms": 300, "x": 0.3, "y": 0.0, "z": 0.5, "method": "hand_fit",
                      "confidence": 0.8}]},
        label_id="s1-wrist3d", t_start_ms=100, t_end_ms=300, **base,
    )  # fmt: skip
    thumb = make_label(
        {"kind": "trajectory3d", "entity_id": "right_hand", "part": "thumb_tip", "frame": "camera",
         "source_3d": "mono_depth",
         "samples": [{"t_ms": 100, "x": 0.1, "y": 0.0, "z": 0.5},
                     {"t_ms": 300, "x": 0.3, "y": 0.0, "z": 0.5}]},
        label_id="s1-thumb3d", t_start_ms=100, t_end_ms=300, **base,
    )  # fmt: skip
    labels = [*source(scenario, policy).sessions[0].labels, wrist, thumb]
    ep = build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48), labels,
        Vocab.from_ontology(ontology), lp,
    )  # fmt: skip
    n = len(lp.hand_points_3d)
    assert ep.hand_points_3d_confidence.shape == (len(ep.times_ms), 2 * n)
    col = {name: i for i, name in enumerate(state_names(lp))}
    k = int(np.argmin(np.abs(ep.times_ms - 200)))
    w, th = lp.hand_points_3d.index("wrist"), lp.hand_points_3d.index("thumb_tip")
    right = ep.hand_points_3d_confidence[k, n:]
    expected = 0.4 + 0.4 * (ep.times_ms[k] - 100) / 200
    assert right[w] == pytest.approx(expected, abs=1e-5)
    assert ep.state[k, col["right.thumb_tip.present"]] == 1.0 and right[th] == -1.0
    assert np.all(ep.hand_points_3d_confidence[k, :n] == -1.0)
    late = int(np.argmin(np.abs(ep.times_ms - 700)))
    assert np.all(ep.hand_points_3d_confidence[late] == -1.0)


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
    """결과 파일에 원본 버킷 URI가 있으면 `assert_no_raw`가 실패한다."""
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

    sha = sha256_file(video)
    scenario.labeling.put_file("sessions/s1/blurred/third.mp4", video, sha)
    digest = render_hash(operational_blur(labels, "third"), load_privacy_policy(ROOT))
    write_render_meta(scenario.labeling, "s1", "third", RenderMeta(digest, sha), tmp_path)
    states = label_states(policy, False)
    version = DatasetVersion(
        version_id="dv1", ontology_version="1.0.0", created_at=FIXED_TIME,
        snapshot_uri="local-snapshot://x/dv1", splits={"s1": Split.TRAIN},
    )  # fmt: skip
    src = ExportSource(
        version,
        [
            ExportSession(
                session, Split.TRAIN, select_labels(labels, policy, states, ()), {"third": digest}
            )
        ],
        states,
    )
    out = tmp_path / "out"
    r = write_coco(src, policy, ontology, scenario.labeling, out, tmp_path / "w",
                   export_id="e1", now=FIXED_TIME, ids=NOIDS)  # fmt: skip
    data = json.loads((out / "coco" / "annotations.json").read_text())
    third_imgs = [i for i in data["images"] if i["stream_id"] == "third"]
    assert sorted(i["t_ms"] for i in third_imgs) == [t[2], t[4], t[6]]
    assert r.dropped == {"keyframe_between_frames": 1}  # 원래 프레임 사이에 둔 키프레임 하나만
    for img in data["images"]:
        assert (out / "coco" / img["file_name"]).exists()
    assert len({i["file_name"] for i in data["images"]}) == len(data["images"])


def test_coco_skips_images_without_annotations(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """주석이 모두 버려진 프레임·세션은 이미지도 내지 않는다 (내보낸 세션 목록과 어긋나지 않게)."""
    from dlp_schema.testing import make_label

    t = scenario.times
    hidden = make_label(
        {"kind": "keypoint_track", "entity_id": "right_hand", "skeleton": "hand21",
         "hand": "right",
         "keyframes": [{"t_ms": t[2], "points": [{"x": 0, "y": 0, "visibility": 0}] * 21}]},
        label_id="s1-hidden", session_id="s1", stream_id="bodycam", t_start_ms=t[2],
        t_end_ms=t[2],
    )  # fmt: skip
    unknown = make_label(
        {"kind": "box_track", "entity_id": "x_1", "class_id": "not_in_ontology",
         "keyframes": [{"t_ms": t[4], "x": 1, "y": 1, "w": 4, "h": 4}]},
        label_id="s1-unknown", session_id="s1", stream_id="bodycam", t_start_ms=t[4],
        t_end_ms=t[4],
    )  # fmt: skip
    out = tmp_path / "out"
    sc = Scenario(
        scenario.session,
        [hidden, unknown],
        scenario.times,
        scenario.labeling,
        scenario.render_hashes,
    )
    r = write_coco(
        source(sc, policy), policy, ontology, sc.labeling, out, tmp_path / "w",
        export_id="e1", now=FIXED_TIME, ids=NOIDS,
    )  # fmt: skip
    data = json.loads((out / "coco" / "annotations.json").read_text())
    assert r.dropped == {"no_visible_keypoints": 1, "unknown_class": 1}
    assert data["images"] == [] and r.images == 0 and r.sessions == set()
    assert not list((out / "coco" / "images").iterdir())
    # 같은 세션에 남는 주석이 있으면 그 프레임의 이미지만 낸다
    box = next(x for x in scenario.labels if x.label_id == "s1-box")
    sc = Scenario(
        scenario.session,
        [hidden, unknown, box],
        scenario.times,
        scenario.labeling,
        scenario.render_hashes,
    )
    out2 = tmp_path / "out2"
    r = write_coco(
        source(sc, policy), policy, ontology, sc.labeling, out2, tmp_path / "w2",
        export_id="e1", now=FIXED_TIME, ids=NOIDS,
    )  # fmt: skip
    data = json.loads((out2 / "coco" / "annotations.json").read_text())
    assert sorted(i["t_ms"] for i in data["images"]) == [t[2], t[4]]  # 박스 키프레임 둘
    assert len(list((out2 / "coco" / "images").iterdir())) == 2 and r.sessions == {"s1"}
    assert {a["image_id"] for a in data["annotations"]} == {i["id"] for i in data["images"]}


# ---------------------------------------------------------------- 묶음 C (ADR 0053)


def _traj(
    label_id: str,
    part: str,
    times: list[int],
    methods: list[str | None] | None = None,
    intrinsics: str | None = "calibrated",
    entity: str = "right_hand",
) -> Any:
    """카메라 좌표 3D 궤적 라벨 (모델 출처, 사람 승인)."""
    samples: list[dict[str, Any]] = []
    for i, t in enumerate(times):
        s: dict[str, Any] = {"t_ms": t, "x": t / 1000, "y": 0.0, "z": 0.5, "confidence": 0.5}
        m = methods[i] if methods is not None else "hand_fit"
        if m is not None:
            s["method"] = m
        samples.append(s)
    payload: dict[str, Any] = {
        "kind": "trajectory3d", "entity_id": entity, "part": part, "frame": "camera",
        "source_3d": "mono_depth", "samples": samples,
        "derived_from": ["s1-bodycam-kp-right"],
    }  # fmt: skip
    if intrinsics is not None:
        payload["intrinsics"] = intrinsics
    return make_label(
        payload, label_id=label_id, session_id="s1", stream_id="bodycam", t_start_ms=times[0],
        t_end_ms=times[-1], ontology_version="1.0.0",
        **model(VerificationState.HUMAN_APPROVED),
    )  # fmt: skip


def _ep(scenario: Scenario, policy: ExportPolicy, ontology: Ontology, extra: list[Any], tmp: Path):
    """시나리오 라벨 + extra로 에피소드를 만든다."""
    video = tmp / "v.mp4"
    scenario.labeling.get_file("sessions/s1/blurred/bodycam.mp4", video)
    labels = [*source(scenario, policy).sessions[0].labels, *extra]
    return build_episode(
        scenario.session, scenario.session.streams[0], build_pts_index(video), (64, 48), labels,
        Vocab.from_ontology(ontology), policy.lerobot,
    )  # fmt: skip


@pytest.mark.parametrize(
    "times",
    [
        [0, 200, 400, 601, 801],  # 29.97 fps: 깊이 간격 200 ms 뒤 첫 프레임 (200/201 ms)
        [0, 208, 417, 625, 834],  # 24 fps (208/209 ms)
        [0, 233, 450, 690, 920],  # VFR
    ],
)
def test_lerobot_hand_3d_survives_real_frame_rates(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path, times: list[int]
) -> None:
    """3D 표본 간격이 깊이 간격(200 ms) + 한 프레임일 때도 궤적 구간 안 모든 에피소드 프레임에 손
    3D가 있다(present 1, 신뢰도 ≥ 0, 채움 비율 1). 2D 보간 한도(200 ms)를 3D에 쓰던 때는 29.97
    fps에서 약 20%, 24 fps·VFR에서 거의 모두 빠졌다 (F-COE-50)."""
    lp = policy.lerobot
    ep = _ep(scenario, policy, ontology, [_traj("s1-w3d", "wrist", times)], tmp_path)
    col = {name: i for i, name in enumerate(state_names(lp))}["right.wrist.present"]
    inside = (ep.times_ms >= times[0]) & (ep.times_ms <= times[-1])
    w = len(lp.hand_points_3d) + lp.hand_points_3d.index("wrist")
    assert ep.state[inside, col].all()
    assert (ep.hand_points_3d_confidence[inside, w] >= 0).all()
    narrow = policy.model_copy(
        update={"lerobot": lp.model_copy(update={"interp_max_gap_3d_ms": 200})}
    )
    old = _ep(scenario, narrow, ontology, [_traj("s1-w3d", "wrist", times)], tmp_path)
    assert old.hand_points_3d_coverage is not None
    assert old.hand_points_3d_coverage < lp.min_hand_points_3d_coverage  # 경고 대상


def test_lerobot_exports_3d_method_intrinsics_and_contact_object_class(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """손 3D 점마다 추정 방식(hand_fit 0, point_depth 1, 보간 양끝이 다르면 mixed 2)과 내부 파라미터
    출처(calibrated 0, hfov_approx 1, 없음 -1)를 싣고, 손마다 접촉 대상 객체 클래스 번호를 싣는다
    (신뢰도 범위가 방식끼리 겹쳐 신뢰도만으로 관절별 깊이 대체 표본을 가를 수 없었다, F-COE-51;
    LeRobot에 객체 클래스가 없어 객체별 사전값을 프레임에 이을 수 없었다, F-COE-52)."""
    lp = policy.lerobot
    times = [100, 300, 500]
    wrist = _traj("s1-w3d", "wrist", times, ["hand_fit", "hand_fit", "point_depth"], "calibrated")
    thumb = _traj("s1-t3d", "thumb_tip", times, [None, None, None], "hfov_approx")
    index = _traj("s1-i3d", "index_tip", times, intrinsics=None)
    mop = make_label(
        {"kind": "box_track", "entity_id": "mop_1", "class_id": "mop",
         "keyframes": [{"t_ms": 100, "x": 1, "y": 1, "w": 5, "h": 5}]},
        label_id="s1-mopbox", session_id="s1", stream_id="bodycam", t_start_ms=100, t_end_ms=100,
        ontology_version="1.0.0", **model(VerificationState.HUMAN_APPROVED),
    )  # fmt: skip
    ep = _ep(scenario, policy, ontology, [wrist, thumb, index, mop], tmp_path)
    vocab = Vocab.from_ontology(ontology).as_json()
    n = len(lp.hand_points_3d)
    w, th, ix = (n + lp.hand_points_3d.index(p) for p in ("wrist", "thumb_tip", "index_tip"))
    k2 = int(np.argmin(np.abs(ep.times_ms - 200)))  # hand_fit 둘 사이
    k4 = int(np.argmin(np.abs(ep.times_ms - 400)))  # hand_fit ~ point_depth 사이
    assert vocab["hand_point_3d_method"] == ["hand_fit", "point_depth", "mixed"]
    assert vocab["hand_point_3d_intrinsics"] == ["calibrated", "hfov_approx"]
    assert ep.hand_points_3d_method[k2, w] == 0 and ep.hand_points_3d_method[k4, w] == 2
    assert ep.hand_points_3d_method[k2, th] == -1 and ep.hand_points_3d_method[k2, ix] == 0
    assert ep.hand_points_3d_intrinsics[k2, w] == 0 and ep.hand_points_3d_intrinsics[k2, th] == 1
    assert ep.hand_points_3d_intrinsics[k2, ix] == -1
    assert np.all(ep.hand_points_3d_method[k2, :n] == -1)  # 왼손 3D 없음
    # 손 상태 100~600 ms 오른손이 mop_1(도구)에 닿는다 → 객체 클래스 mop, 왼손은 -1
    k3 = int(np.argmin(np.abs(ep.times_ms - 300)))
    mop_index = vocab["object_class"].index("mop")
    assert ep.contact_object_class[k3, 1] == mop_index and ep.contact_object_class[k3, 0] == -1
    assert "s1-mopbox" in ep.used
    assert set(vocab["object_class"]) >= set(vocab["physical_priors"]["objects"])
    late = int(np.argmin(np.abs(ep.times_ms - 800)))
    assert ep.contact_object_class[late, 1] == -1


def test_physical_prior_table_states_base_meaning_and_content_hash(ontology: Ontology) -> None:
    """구매자 규약에 기본 범위의 뜻(젖음은 건조, 그 밖은 모든 상태 포괄)·도구 기준면(파지부)이 있고,
    표에 내용 해시가 있어 같은 버전의 정정을 가릴 수 있다 (F-SCH-23·21)."""
    from dlp_schema.export import physical_prior_table, prior_content_sha256

    table = physical_prior_table(ontology)
    assert (
        "dry" in table.conventions["base_range"] and "spans all" in table.conventions["base_range"]
    )
    assert "grip part" in table.conventions["reference_surface"]
    assert table.content_sha256 == prior_content_sha256(table) and len(table.content_sha256) == 64
    data = ontology.model_dump(mode="json")
    data["objects"]["cup"]["physical_priors"]["mass_kg"]["max"] = 9.0
    changed = physical_prior_table(Ontology.model_validate(data))
    assert changed.content_sha256 != table.content_sha256


def test_interval_json_new_schema_reads_older_files(
    scenario: Scenario, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """하위 호환의 뜻: 1.2 공개 스키마와 리더(`IntervalFile`)가 1.0·1.1 모양 파일(term_names·
    physical_priors 없음)을 받는다. 옛 스키마가 새 파일을 거부하는 것(최상위 additionalProperties
    false)은 약속하지 않는다 (F-SCH-24)."""
    write_intervals(
        source(scenario, policy), policy, tmp_path, export_id="e1", now=FIXED_TIME, ids=NOIDS,
        ontology=ontology,
    )  # fmt: skip
    path = tmp_path / "intervals" / "s1.json"
    data = json.loads(path.read_text("utf-8"))
    schema = json.loads((ROOT / "schemas" / "export_intervals.schema.json").read_text("utf-8"))
    for version, drop in (
        ("1.1", ("physical_priors",)),
        ("1.0", ("physical_priors", "term_names")),
    ):
        old = {k: v for k, v in data.items() if k not in drop}
        old["format_version"] = version
        jsonschema.validate(old, schema)
        IntervalFile.model_validate(old)


def test_trajectory_derived_from_is_pseudonymized_like_label_ids() -> None:
    """3D 궤적 `derived_from`(내부 라벨 ID, 세션 ID 접두)은 라벨 가명으로 통째로 바뀌고, 결과 JSON에
    내부 세션 ID가 남지 않는다 (F-COE-53)."""
    ids = Pseudonymizer.for_export("export-x", b"secret", enabled=True)
    x = _traj("s1-bodycam-3d-w", "wrist", [100, 200])
    out = exported(x, ids).model_dump(mode="json")
    assert out["payload"]["derived_from"] == [ids.label("s1-bodycam-kp-right")]
    assert "s1-" not in json.dumps(out)


def test_every_label_id_reference_field_is_pseudonymized() -> None:
    """계약의 모든 페이로드에서 Identifier 필드는 `_id`·`_ids` 이름 규칙이거나
    `pseudonym.LABEL_REF_FIELDS`에 있다 (새 필드가 가명 규칙을 조용히 벗어나지 않게, F-COE-53)."""
    import typing

    from dlp_export.pseudonym import LABEL_REF_FIELDS
    from dlp_schema import labels as schema_labels

    payloads = typing.get_args(typing.get_args(schema_labels.LabelPayload)[0])
    assert payloads
    unnamed: list[str] = []
    for model_cls in payloads:
        for name, f in model_cls.model_fields.items():
            text = repr(f.annotation) + repr(f.metadata)
            is_id = "A-Za-z0-9" in text or "Identifier" in text
            if is_id and not (name.endswith(("_id", "_ids")) or name in LABEL_REF_FIELDS):
                unnamed.append(f"{model_cls.__name__}.{name}")
    assert unnamed == [], unnamed


@pytest.mark.parametrize(
    ("points", "match"),
    [([], "at least 1"), (["wrist", "wrist"], "두 번")],
)
def test_lerobot_hand_points_policy_is_validated(
    policy: ExportPolicy, points: list[str], match: str
) -> None:
    """hand_points_3d는 1개 이상·중복 없음 (F-COE-54)."""
    from pydantic import ValidationError

    data = policy.model_dump(mode="json")
    data["lerobot"]["hand_points_3d"] = points
    with pytest.raises(ValidationError, match=match):
        ExportPolicy.model_validate(data)


def test_unknown_hand_part_fails_before_export(policy: ExportPolicy, ontology: Ontology) -> None:
    """온톨로지 hand_joints에 없는 손 3D 점·리타기팅 부위는 내보내기 전에 거부한다 (오타면 열이 전부
    빈 값으로 팔렸다, F-COE-54)."""
    from dlp_retarget.policy import load_policy as load_retarget

    from dlp_export.runner import check_hand_parts

    check_hand_parts(ontology, policy.lerobot, load_retarget(ROOT))
    typo = policy.lerobot.model_copy(update={"hand_points_3d": ("wrist", "middle_tips")})
    with pytest.raises(ExportError, match="middle_tips"):
        check_hand_parts(ontology, typo, None)
    rp = load_retarget(ROOT)
    bad = rp.model_copy(update={"hand_parts": rp.hand_parts.model_copy(update={"index": "indx"})})
    with pytest.raises(ExportError, match="indx"):
        check_hand_parts(ontology, policy.lerobot, bad)
