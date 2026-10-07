from __future__ import annotations

import hashlib
import hmac
import json
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.video import TARGET_COLORS, generate_blur_scenario
from dlp_media.proxy import make_proxy
from dlp_media.pts import build_pts_index
from dlp_media.storage import LocalStore, sha256_file
from dlp_review.collect import drop_retracted
from dlp_review.cvat import CvatSchema, from_cvat_tracks, label_spec, quantize, to_cvat_tracks
from dlp_review.labelstudio import (
    LS_KINDS,
    from_ls_results,
    label_config,
    label_names,
    to_ls_results,
)
from dlp_review.reconcile import ReviewedItem, reconcile
from dlp_review.roles import RawAccessError, check_stage_uris
from dlp_review.tasks import frame_times, label_scale, video_size
from dlp_review.timeseries import write_timeseries_csv
from dlp_review.watermark import burn_watermark
from dlp_review.webhook import (
    CollectRequest,
    ReviewerMismatchError,
    WebhookAuthError,
    parse_event,
    resolve_reviewer,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    GapPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import Stream, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session
from dlp_sync.signals import Series

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return load_ontology(ROOT / "config" / "ontology" / "v1")


def fake_schema(names: list[str]) -> CvatSchema:
    """CVAT가 부여하는 것처럼 라벨·속성에 ID를 매긴다."""
    labels: list[dict[str, Any]] = []
    next_id = 100
    for i, spec in enumerate(label_spec(names)):
        attrs: list[dict[str, Any]] = []
        for a in spec["attributes"]:
            attrs.append({**a, "id": next_id})
            next_id += 1
        labels.append({**spec, "id": i + 1, "attributes": attrs})
    return CvatSchema.from_labels(labels)


# ---------------------------------------------------------------- 변환기


def test_cvat_roundtrip_offline_including_keypoints() -> None:
    blur = generate_blur_scenario(2)
    actions = generate_action_scenario(2)
    kp = next(x for x in actions.labels if x.kind == "keypoint_track")
    names = ["face", "reflection", "document", "screen", "photo", "shipping_label", "kp_hand21"]
    schema = fake_schema(names)
    for labels, times in ((blur.labels, blur.frame_times), ([kp], actions.frame_times)):
        tracks = to_cvat_tracks(labels, times, schema)
        back = from_cvat_tracks(
            tracks, times, schema, labels[0].stream_id or "bodycam", new_box_kind="blur_track"
        )
        for original, item in zip(labels, back, strict=True):
            q = quantize(original)
            assert item.origin_label_id == original.label_id
            assert (item.payload, item.t_start_ms, item.t_end_ms) == (
                q.payload,
                q.t_start_ms,
                q.t_end_ms,
            )


def test_cvat_new_track_and_bad_frame_time() -> None:
    blur = generate_blur_scenario(2)
    schema = fake_schema(["face", "reflection", "document", "screen", "photo", "shipping_label"])
    [track] = to_cvat_tracks(blur.labels[:1], blur.frame_times, schema)
    track["attributes"] = []  # 검수자가 새로 그린 트랙에는 ID가 없다
    [item] = from_cvat_tracks(
        [track], blur.frame_times, schema, "bodycam", new_box_kind="blur_track"
    )
    assert item.origin_label_id is None and isinstance(item.payload, BlurTrackPayload)
    with pytest.raises(ValueError, match="프레임 시각"):
        to_cvat_tracks(blur.labels[:1], [t + 1 for t in blur.frame_times], schema)


def test_label_studio_roundtrip_offline_and_new_regions(ontology: Ontology) -> None:
    labels = [x for x in generate_action_scenario(3).labels if x.kind in LS_KINDS]
    results = to_ls_results(labels)
    back = from_ls_results(results, {x.label_id for x in labels})
    for original, item in zip(labels, back, strict=True):
        assert item.origin_label_id == original.label_id
        assert (item.payload, item.t_start_ms, item.t_end_ms, item.stream_id) == (
            original.payload, original.t_start_ms, original.t_end_ms, original.stream_id,
        )  # fmt: skip
    names = set(label_names(ontology))
    assert all(r["value"]["timeserieslabels"][0] in names for r in results)

    value = {"start": 100.0, "end": 900.0, "timeserieslabels": ["action.left:press"]}
    drawn = {"id": "xyz", "type": "timeserieslabels", "value": value}
    [item] = from_ls_results([drawn], set())
    assert item.origin_label_id is None
    assert isinstance(item.payload, ActionPayload) and item.payload.hand.value == "left"
    assert (item.payload.t_approach_ms, item.payload.t_end_ms) == (100, 900)

    state = {**drawn, "value": {**value, "timeserieslabels": ["state:wetness=wet"]}}
    with pytest.raises(ValueError, match="객체 상태"):
        from_ls_results([state], set())


def test_moving_an_action_keeps_contact_inside(ontology: Ontology) -> None:
    label = make_label(action_payload())  # 0~1000, 접촉 400~900
    [r] = to_ls_results([label])
    r["value"]["start"], r["value"]["end"] = 500, 800
    [item] = from_ls_results([r], {label.label_id})
    p = item.payload
    assert isinstance(p, ActionPayload)
    assert (p.t_approach_ms, p.t_contact_start_ms, p.t_contact_end_ms, p.t_end_ms) == (
        500,
        500,
        800,
        800,
    )


def test_label_config_is_valid_xml(ontology: Ontology) -> None:
    root = ET.fromstring(label_config(ontology))
    assert root.find("TimeSeries") is not None and root.find("Video") is not None
    values = {e.get("value") for e in root.iter("Label")}
    assert "action.right:grasp" in values and "gap.left:unknown" in values


# ---------------------------------------------------------------- reconcile


def test_reconcile_approves_corrects_retracts_and_adds() -> None:
    a = make_label(action_payload(action_id="a1"), label_id="a")
    b = make_label(action_payload(action_id="b1"), label_id="b")
    c = make_label(action_payload(action_id="c1"), label_id="c")
    moved = action_payload(action_id="b1", t_contact_start_ms=300)
    reviewed = [
        ReviewedItem("a", None, 0, 1000, a.payload),
        ReviewedItem("b", None, 0, 1000, ActionPayload.model_validate(moved)),
        ReviewedItem(None, None, 2000, 3000, ActionPayload.model_validate(
            action_payload(action_id="n1", t_approach_ms=2000, t_contact_start_ms=2100,
                           t_contact_end_ms=2900, t_end_ms=3000))),
    ]  # fmt: skip
    out = reconcile([a, b, c], reviewed, session_id="s001", ontology_version="1.0.0",
                    reviewer_id="rev1", now=FIXED_TIME)  # fmt: skip
    assert out.approved == ["a"]
    assert (out.corrected, out.retracted, out.added) == (1, 1, 1)
    by_parent = {r.parent_label_id: r for r in out.new_records}
    assert by_parent["b"].verification.state is VerificationState.HUMAN_CORRECTED
    assert by_parent["c"].retracted
    assert by_parent[None].provenance.source.value == "human"
    again = reconcile([a, b, c], reviewed, session_id="s001", ontology_version="1.0.0",
                      reviewer_id="rev1", now=FIXED_TIME)  # fmt: skip
    assert [r.label_id for r in again.new_records] == [r.label_id for r in out.new_records]


def test_reconcile_compares_against_normalized_originals() -> None:
    blur = generate_blur_scenario(2)
    original = blur.labels[0]
    p = original.payload
    assert isinstance(p, BlurTrackPayload)
    jittered = original.model_copy(
        update={"payload": p.model_copy(update={"keyframes": tuple(
            k.model_copy(update={"x": k.x + 0.0001}) for k in p.keyframes)})}
    )  # fmt: skip
    q = quantize(jittered)
    item = ReviewedItem(original.label_id, q.stream_id, q.t_start_ms, q.t_end_ms, q.payload)
    out = reconcile([jittered], [item], session_id="syn-blur", ontology_version="1.0.0",
                    reviewer_id="r", now=FIXED_TIME, normalize=quantize)  # fmt: skip
    assert out.approved == [original.label_id] and not out.new_records


# ---------------------------------------------------------------- 그 밖


def test_timeseries_csv_uses_synced_master_time(tmp_path: Path) -> None:
    session = make_session()
    glove = session.streams[1].model_copy(
        update={"stream_id": "glove_right", "kind": StreamKind.GLOVE_RIGHT, "offset_ms": 1_000.0,
                "sync_method": SyncMethod.TAP_EVENT}
    )  # fmt: skip
    session = session.model_copy(
        update={"streams": (*session.streams, glove), "duration_ms": 4_000}
    )
    t = np.arange(0, 3_000, 10.0)
    series = {"glove_right": Series(t, np.where((t >= 500) & (t < 600), 1.0, 0.0))}
    out = tmp_path / "ts.csv"
    rows = write_timeseries_csv(session, series, out, rate_hz=100)
    data = np.genfromtxt(out, delimiter=",", names=True)
    assert rows == 400
    on = data["time_ms"][data["glove_right"] > 0.5]
    assert on.min() == pytest.approx(1_500) and on.max() == pytest.approx(
        1_590
    )  # 스트림 500 ms + 오프셋 1000 ms
    assert np.all(data["glove_left"] == 0)


def test_watermark_marks_frames_and_keeps_pts(tmp_path: Path) -> None:
    blur = generate_blur_scenario(1)
    src, dst = tmp_path / "a.mp4", tmp_path / "b.mp4"
    blur.write(src)
    burn_watermark(src, dst, "labeler-07 syn-blur", opacity=0.3, crf=20, encoder_rate=30)
    assert build_pts_index(dst).ms.tolist() == build_pts_index(src).ms.tolist()
    import av

    with av.open(str(src)) as a, av.open(str(dst)) as b:
        fa = next(a.decode(video=0)).to_ndarray(format="rgb24").astype(int)
        fb = next(b.decode(video=0)).to_ndarray(format="rgb24").astype(int)
    changed = (np.abs(fa - fb).max(axis=2) > 25).mean()
    assert 0.01 < changed < 0.4  # 글씨가 화면 곳곳에 있지만 화면을 덮지는 않는다


def test_stage_uri_guard() -> None:
    check_stage_uris(ReviewStage.PRIVACY, ["s3://dlp-raw/sessions/s/derived/a.mp4"], "dlp-raw")
    check_stage_uris(ReviewStage.LABELING, ["s3://dlp-labeling/x.mp4"], "dlp-raw")
    check_stage_uris(
        ReviewStage.LABELING, ["http://localhost:8333/dlp-labeling/x.mp4?X-Amz=1"], "dlp-raw"
    )
    for leaked in (
        "s3://dlp-raw/x.mp4",
        "http://localhost:8333/dlp-raw/x.mp4?X-Amz-Signature=abc",  # path-style 서명 URL
        "https://dlp-raw.s3.example.com/x.mp4",  # virtual-host style
    ):
        with pytest.raises(RawAccessError):
            check_stage_uris(ReviewStage.LABELING, [leaked], "dlp-raw")


def test_webhook_parsing_and_auth() -> None:
    job = {"task_id": 7, "state": "completed", "assignee": {"username": "rev1"}}
    body = json.dumps({"event": "update:job", "job": job}).encode()
    sig = "sha256=" + hmac.new(b"s3cret", body, hashlib.sha256).hexdigest()
    req = parse_event("cvat", {"X-Signature-256": sig}, body, "s3cret")
    assert req is not None and (req.task_key, req.reviewer) == ("cvat:7", "rev1")
    with pytest.raises(WebhookAuthError):
        parse_event("cvat", {"X-Signature-256": "sha256=bad"}, body, "s3cret")
    in_progress = json.dumps(
        {"event": "update:job", "job": {"task_id": 7, "state": "in progress"}}
    ).encode()
    sig2 = "sha256=" + hmac.new(b"s3cret", in_progress, hashlib.sha256).hexdigest()
    assert parse_event("cvat", {"X-Signature-256": sig2}, in_progress, "s3cret") is None

    ls = json.dumps({"action": "ANNOTATION_UPDATED", "task": {"id": 3},
                     "annotation": {"completed_by": 12}}).encode()  # fmt: skip
    req = parse_event("label_studio", {"X-DLP-Secret": "s3cret"}, ls, "s3cret")
    # completed_by는 도구 내부 숫자 ID라 검수자로 쓰지 않는다
    assert req is not None and (req.task_key, req.reviewer, req.user_id) == (
        "label_studio:3", None, 12,
    )  # fmt: skip
    with pytest.raises(WebhookAuthError):
        parse_event("label_studio", {}, ls, "s3cret")


def _task(key: str, assignee: str | None) -> ReviewTask:
    return ReviewTask(
        task_key=key, tool=ReviewTool.LABEL_STUDIO, external_id="3", session_id="s001",
        stream_id="bodycam", stage=ReviewStage.LABELING, assignee=assignee,
        media_uri="s3://dlp-labeling/x.mp4", label_kinds=("action",), created_at=FIXED_TIME,
    )  # fmt: skip


def test_webhook_reviewer_comes_from_task_assignee() -> None:
    """회귀: Label Studio 숫자 ID를 검수자로 쓰지 않고, 서비스 계정 주석은 무시하고,
    CVAT 작업 담당자가 다르면 받지 않는다."""
    task = _task("label_studio:3", "labeler01")
    assert resolve_reviewer(CollectRequest("label_studio:3", None, 12), task, 1) == "labeler01"
    assert resolve_reviewer(CollectRequest("label_studio:3", None, 1), task, 1) is None
    assert resolve_reviewer(CollectRequest("cvat:7", "labeler01"), task) == "labeler01"
    with pytest.raises(ReviewerMismatchError):
        resolve_reviewer(CollectRequest("cvat:7", "someone"), task)
    with pytest.raises(ReviewerMismatchError):
        resolve_reviewer(CollectRequest("label_studio:3", None, 12), _task("label_studio:3", None))


# ---------------------------------------------------------------- 감사 회귀


def test_privacy_boxes_land_on_the_downscaled_proxy(tmp_path: Path) -> None:
    """회귀: 블러 라벨(원본 720p 화소)을 480p 프록시 검수 화면 좌표로 바꿔 보내고 되돌린다."""
    blur = generate_blur_scenario(3, duration_ms=400, width=1280, height=720)
    store = LocalStore(tmp_path / "store", "dlp-raw")
    src = tmp_path / "raw.mp4"
    blur.write(src)
    store.put_file("sessions/s/raw/bodycam.mp4", src, sha256_file(src))
    proxy = tmp_path / "proxy.mp4"
    make_proxy(src, proxy)
    stream = Stream(
        stream_id="bodycam", kind=StreamKind.BODYCAM, uri=store.uri("sessions/s/raw/bodycam.mp4")
    )
    work = tmp_path / "work"
    work.mkdir()
    scale = label_scale(store, stream, proxy, work)
    assert video_size(proxy)[1] == 480 and scale[1] == pytest.approx(480 / 720)

    names = ["face", "reflection", "document", "screen", "photo", "shipping_label"]
    schema = fake_schema(names)
    times = frame_times(proxy)
    tracks = to_cvat_tracks(blur.labels, times, schema, scale=scale)
    # 프록시 화면에서 박스 가운데가 그 대상의 색이다 (원본 좌표 그대로면 다른 곳을 가리킨다)
    with av.open(str(proxy)) as c:
        first = next(c.decode(video=0)).to_ndarray(format="rgb24").astype(int)
    for label, track in zip(blur.labels, tracks, strict=True):
        assert isinstance(label.payload, BlurTrackPayload)
        shape = track["shapes"][0]
        if shape["outside"] or label.payload.target == "reflection":
            continue
        x1, y1, x2, y2 = shape["points"]
        assert x2 <= 854 and y2 <= 480
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = np.array(TARGET_COLORS[label.payload.target])
        assert np.abs(first[cy, cx] - color).max() < 30, label.payload.target

    back = from_cvat_tracks(
        tracks, times, schema, "bodycam", new_box_kind="blur_track", scale=scale
    )
    for original, item in zip(blur.labels, back, strict=True):
        assert item.payload == quantize(original, scale).payload  # 고치지 않으면 승인만 된다
        assert isinstance(item.payload, BlurTrackPayload)
        assert isinstance(original.payload, BlurTrackPayload)
        for a, b in zip(item.payload.keyframes, original.payload.keyframes, strict=True):
            assert abs(a.x - b.x) < 0.01 and abs(a.w - b.w) < 0.01
    out = reconcile(blur.labels, back, session_id="syn-blur", ontology_version="1.0.0",
                    reviewer_id="r", now=FIXED_TIME,
                    normalize=partial(quantize, scale=scale))  # fmt: skip
    assert not out.new_records and len(out.approved) == len(blur.labels)


def test_new_cvat_box_kind_follows_task_stage() -> None:
    """회귀: 작업 라벨(객체) 작업에서 새로 그린 박스는 블러가 아니라 객체 박스 트랙이다."""
    blur = generate_blur_scenario(2)
    schema = fake_schema(["cup", "reflection"])
    track = to_cvat_tracks(blur.labels[:1], blur.frame_times, schema)[0]
    track["label_id"] = schema.label_ids["cup"]
    track["attributes"] = []
    [box] = from_cvat_tracks([track], blur.frame_times, schema, "bodycam", new_box_kind="box_track")
    assert isinstance(box.payload, BoxTrackPayload) and box.payload.class_id == "cup"
    [priv] = from_cvat_tracks(
        [track], blur.frame_times, schema, "bodycam", new_box_kind="blur_track"
    )
    assert isinstance(priv.payload, BlurTrackPayload)


def test_label_studio_relabel_to_another_kind_is_delete_and_add() -> None:
    """회귀: 행동 구간을 사이 구간으로 바꿔 붙이면 남은 행동 필드 때문에 수집이 깨졌다."""
    label = make_label(action_payload())
    [r] = to_ls_results([label])
    r["value"]["timeserieslabels"] = ["gap.right:unknown"]
    [item] = from_ls_results([r], {label.label_id})
    assert item.origin_label_id is None and isinstance(item.payload, GapPayload)
    out = reconcile([label], [item], session_id="s001", ontology_version="1.0.0",
                    reviewer_id="r", now=FIXED_TIME)  # fmt: skip
    assert (out.retracted, out.added) == (1, 1)


def test_correction_of_a_retracted_label_does_not_revive_it() -> None:
    """회귀: 작업을 보낸 뒤 다른 단계가 지운 라벨을 검수자가 고쳐도 되살리지 않는다."""
    a = make_label(action_payload(action_id="a1"), label_id="a")
    b = make_label(action_payload(action_id="b1"), label_id="b")
    gone = a.model_copy(
        update={"label_id": "a:retracted", "parent_label_id": "a", "retracted": True}
    )
    moved = ActionPayload.model_validate(action_payload(action_id="a1", t_contact_start_ms=300))
    reviewed = [
        ReviewedItem("a", None, 0, 1000, moved),
        ReviewedItem("b", None, 0, 1000, b.payload),
    ]
    out = reconcile([a, b], reviewed, session_id="s001", ontology_version="1.0.0",
                    reviewer_id="r", now=FIXED_TIME)  # fmt: skip
    assert drop_retracted(out, [a, b, gone]) == ["a"]
    assert out.approved == ["b"] and not out.new_records
    assert {x.label_id for x in current_labels([a, b, gone, *out.new_records])} == {"b"}
