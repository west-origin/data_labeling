"""검수 연동 단위 테스트 (WP6, ADR 0006·0024). 서비스(DB·도구) 없이 돈다.

- 변환기: CVAT·Label Studio 무손실 왕복, 새로 그린 트랙·구간, 모양(Shape) 모드, 지원하지 않는 주석.
- reconcile: 승인·수정·삭제·추가 판정과 ID 멱등, 좌표 반올림 기준 비교, 보낸 뒤 바뀐 라벨 제외.
- 그 밖: 시계열 CSV의 마스터 시각, 워터마크 PTS 유지, 원본 URI 차단, 웹훅 서명·검수자 판정.
정답은 합성 픽스처(`dlp_fixtures.video.generate_blur_scenario`, `dlp_fixtures.actions`)의
라벨·프레임 시각과 `dlp_schema.testing`의 고정 라벨에서 온다.
"""

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
from dlp_privacy.render import blur_tracks
from dlp_review.collect import drop_retracted
from dlp_review.cvat import (
    CvatFormatError,
    CvatSchema,
    annotation_tracks,
    from_cvat_tracks,
    label_spec,
    quantize,
    to_cvat_tracks,
)
from dlp_review.labelstudio import (
    LS_KINDS,
    from_ls_results,
    label_config,
    label_names,
    to_ls_results,
)
from dlp_review.reconcile import ReviewedItem, reconcile
from dlp_review.roles import RawAccessError, check_stage_uris
from dlp_review.tasks import ReviewSetup, frame_times, label_scale, video_size
from dlp_review.timeseries import write_timeseries_csv
from dlp_review.watermark import burn_watermark
from dlp_review.webhook import (
    CollectRequest,
    ReviewerMismatchError,
    WebhookAuthError,
    parse_event,
    resolve_reviewer,
)
from dlp_schema import repo_root
from dlp_schema.config import load_config
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    GapPayload,
    KeypointTrackPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import Stream, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import Series

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1 (Label Studio 라벨 이름·설정 XML 시험용)."""
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
    """블러 박스·키포인트 트랙이 CVAT 형식을 거쳐도 그대로 돌아오는지 본다 (ADR 0006 무손실 왕복).

    정답 근거: 합성 블러 시나리오의 블러 라벨과 행동 시나리오의 손 키포인트 트랙, 각 시나리오의
    프레임 시각. 돌아온 항목은 원래 라벨 ID를 가리키고, 내용·구간이 `quantize(원래 라벨)`과 같다.
    """
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


def test_cvat_keypoint_visibility_must_match_points() -> None:
    """회귀: dlp_visibility 값이 점 수보다 짧으면 IndexError가 나 수집이 내부 오류로 멈췄다.

    정답 근거: 행동 시나리오의 손 키포인트 트랙(21점)을 CVAT 형식으로 보낸 뒤 첫 모양의 가시성
    값을 하나 빼거나(20개), 하나 더하거나(22개), 정수가 아니게 바꾸면 모두 `CvatFormatError`다.
    값이 비어 있으면 모든 점을 보임(2)으로 본다.
    """
    actions = generate_action_scenario(2)
    kp = next(x for x in actions.labels if x.kind == "keypoint_track")
    schema = fake_schema(["kp_hand21"])
    vis_id = schema.attr_ids[("kp_hand21", "dlp_visibility")]

    def with_visibility(value: str) -> list[dict[str, Any]]:
        """첫 모양의 가시성 속성만 value로 바꾼 트랙 목록."""
        [track] = to_cvat_tracks([kp], actions.frame_times, schema)
        track["shapes"][0]["attributes"] = [{"spec_id": vis_id, "value": value}]
        return [track]

    for bad in (",".join(["2"] * 20), ",".join(["2"] * 22), "2,x"):
        with pytest.raises(CvatFormatError, match="dlp_visibility"):
            from_cvat_tracks(
                with_visibility(bad), actions.frame_times, schema, "bodycam",
                new_box_kind="box_track",
            )  # fmt: skip
    [item] = from_cvat_tracks(
        with_visibility(""), actions.frame_times, schema, "bodycam", new_box_kind="box_track"
    )
    assert isinstance(item.payload, KeypointTrackPayload)
    assert {p.visibility for p in item.payload.keyframes[0].points} == {2}


def test_cvat_new_track_and_bad_frame_time() -> None:
    """새로 그린 트랙과 프레임 시각이 어긋난 키프레임을 본다.

    정답 근거: 속성(dlp_label_id)을 지운 트랙은 원래 ID 없는 블러 항목이 된다. 프레임 시각을 1 ms씩
    밀어 키프레임 시각과 맞지 않으면 보낼 때 ValueError("프레임 시각").
    """
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
    """시간 라벨의 Label Studio 왕복과 화면에서 새로 그린 구간을 본다.

    정답 근거: 합성 행동 시나리오의 시간 라벨이 ID·내용·구간·스트림 그대로 돌아오고, 결과의 라벨
    이름이 모두 `label_names(온톨로지)`에 있다. 새로 그린 `action.left:press` 100~900 ms는 원래 ID
    없는 왼손 행동(접근 시작·종료 = 구간)이 되고, 새로 그린 객체 상태 구간은 ValueError.
    """
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
    """행동 구간을 줄이면 접촉 시각이 새 구간 안으로 들어오는지 본다.

    정답 근거: 고정 행동 라벨(0~1000 ms, 접촉 400~900 ms)을 500~800 ms로 옮기면
    접근 시작 500, 접촉 500~800, 종료 800.
    """
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
    """프로젝트 설정 XML이 파싱되고 시계열·영상·대표 라벨 이름을 담는지 본다."""
    root = ET.fromstring(label_config(ontology))
    assert root.find("TimeSeries") is not None and root.find("Video") is not None
    values = {e.get("value") for e in root.iter("Label")}
    assert "action.right:grasp" in values and "gap.left:unknown" in values


# ---------------------------------------------------------------- reconcile


def test_reconcile_approves_corrects_retracts_and_adds() -> None:
    """reconcile의 네 판정(승인·수정·삭제·추가)과 ID 멱등을 본다.

    시나리오: a는 그대로, b는 접촉 시작을 300으로 옮김, c는 결과에 없음, 새 행동 하나 추가.
    정답 근거: 승인 [a], 수정·삭제·추가 각 1, b의 수정은 human_corrected, c는 retracted, 새 레코드는
    사람 출처. 같은 입력으로 다시 돌리면 새 레코드 ID가 같다.
    """
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
    """좌표 반올림 왕복 값과 비교하므로 미세한 차이는 승인으로 보는지 본다.

    시나리오: 원래 블러 박스 x에 0.0001을 더한 라벨을 보냈고 `quantize` 결과가 그대로 돌아왔다.
    정답 근거: normalize=quantize이면 새 레코드 없이 승인만 된다.
    """
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
    """시계열 CSV가 동기화 오프셋을 적용한 마스터 시각 격자인지 본다.

    시나리오: 오른손 장갑 스트림 offset 1000 ms, 스트림 시각 500~600 ms에만 신호 1,
    세션 4초, 100 Hz.
    정답 근거: 행 수 400(4초 * 100 Hz), 신호가 마스터 1500~1590 ms에 나타나고 없는 왼손 열은 0.
    """
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
    """워터마크가 화면 일부에만 들어가고 PTS를 그대로 두는지 본다.

    정답 근거: 합성 블러 영상과 워터마크 영상의 PTS 목록이 같고, 첫 프레임에서 크게 바뀐 화소 비율이
    1%~40% (글씨가 곳곳에 있지만 화면을 덮지는 않는다).
    """
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
    """원본 버킷 URI 차단 검사를 본다 (CLAUDE.md 원본 노출 금지 규칙).

    정답 근거: 프라이버시 단계는 원본 URI를 허용, 작업 라벨 단계는 라벨링 버킷 URI만 허용하고
    원본 버킷의 s3://, path-style 서명 URL, virtual-host style URL은 모두 RawAccessError.
    """
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
    """웹훅 서명·비밀 확인과 수집 요청 파싱을 본다.

    정답 근거: CVAT는 본문 HMAC-SHA256 서명이 맞고 job이 completed일 때만 요청(작업 키 cvat:7, job
    담당자 rev1), 틀린 서명은 WebhookAuthError, 진행 중 상태는 None. Label Studio는 X-DLP-Secret이
    맞을 때 요청(검수자는 비우고 숫자 ID 12만 싣는다), 비밀이 없으면 WebhookAuthError.
    """
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
    """시험용 Label Studio 작업 라벨 작업 (작업 키 key, 담당자 assignee)."""
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
    make_proxy(src, proxy, load_config(repo_root() / "config" / "defaults.yaml").media.proxy)
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


# ---------------------------------------------------------------- 감사 4차 회귀


def _shape(label_id: int, frame: int, kind: str = "rectangle", **kw: Any) -> dict[str, Any]:
    """시험용 CVAT 모양 dict (기본: 직사각형 10~60 화소, 수동). kw로 필드를 덮어쓴다."""
    return {
        "type": kind, "frame": frame, "label_id": label_id, "points": [10, 10, 60, 60],
        "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0, "attributes": [],
        "group": 0, "source": "manual", **kw,
    }  # fmt: skip


def test_cvat_shape_mode_rectangle_becomes_single_frame_track() -> None:
    """회귀(감사 4-1): CVAT 기본 모양(Shape) 모드로 그린 직사각형을 버려 놓친 얼굴이 블러 없이
    승인·렌더됐다. 그 프레임 키프레임 + 다음 프레임 outside 트랙으로 받는다."""
    blur = generate_blur_scenario(2)
    times = blur.frame_times
    names = ["face", "reflection", "document", "screen", "photo", "shipping_label"]
    schema = fake_schema(names)
    existing = to_cvat_tracks(blur.labels[:1], times, schema)
    ann = {"tracks": existing, "tags": [], "shapes": [_shape(schema.label_ids["face"], 3)]}
    tracks = annotation_tracks(ann, len(times), schema)
    items = from_cvat_tracks(tracks, times, schema, "bodycam", new_box_kind="blur_track")
    assert len(items) == 2
    new = items[1]
    assert new.origin_label_id is None and isinstance(new.payload, BlurTrackPayload)
    k0, k1 = new.payload.keyframes
    assert (k0.t_ms, k0.outside, k0.x, k0.w) == (times[3], False, 10, 50)
    assert (k1.t_ms, k1.outside) == (times[4], True)
    # 렌더도 그 프레임에만 블러를 둔다
    [track] = blur_tracks(
        [make_label(new.payload, t_start_ms=times[3], t_end_ms=times[4], stream_id="bodycam")],
        times,
    )
    assert track.box_at(times[3]) is not None and track.box_at(times[4]) is None
    # 마지막 프레임 모양은 키프레임 하나 (뒤에 프레임이 없다)
    [last] = annotation_tracks(
        {"tracks": [], "shapes": [_shape(1, len(times) - 1)]}, len(times), schema
    )
    assert len(last["shapes"]) == 1
    # 작업 라벨 작업에서는 객체 박스 트랙
    obj = fake_schema(["cup", "kp_hand21"])
    [box] = from_cvat_tracks(
        annotation_tracks({"shapes": [_shape(obj.label_ids["cup"], 2)]}, len(times), obj),
        times, obj, "bodycam", new_box_kind="box_track",
    )  # fmt: skip
    assert isinstance(box.payload, BoxTrackPayload) and box.payload.class_id == "cup"


def test_cvat_unsupported_annotations_fail_closed() -> None:
    """옮길 수 없는 주석(태그·다각형·회전 박스·점 모양 박스)은 조용히 버리지 않고 수집을 멈춘다."""
    schema = fake_schema(["face", "kp_hand21"])
    face = schema.label_ids["face"]
    bad: list[dict[str, Any]] = [
        {"tags": [{"frame": 1, "label_id": face, "attributes": []}]},
        {"shapes": [_shape(face, 1, "polygon", points=[0, 0, 5, 0, 5, 5])]},
        {"shapes": [_shape(face, 1, rotation=30.0)]},
        {"shapes": [_shape(face, 1, "points", points=[1, 1])]},
        {"shapes": [_shape(schema.label_ids["kp_hand21"], 1)]},
        {"shapes": [_shape(face, 99)]},
        {
            "tracks": [
                {
                    "frame": 0,
                    "label_id": face,
                    "attributes": [],
                    "shapes": [_shape(face, 0, "ellipse")],
                }
            ]
        },
        {"shapes": [_shape(999, 1)]},
    ]
    for ann in bad:
        with pytest.raises(CvatFormatError):
            annotation_tracks(ann, 10, schema)


def _cvat_task(stage: ReviewStage, assignee: str | None) -> ReviewTask:
    """시험용 CVAT 작업 (단계 stage에 맞는 버킷의 매체 URI, 담당자 assignee)."""
    return ReviewTask(
        task_key="cvat:7", tool=ReviewTool.CVAT, external_id="7", session_id="s001",
        stream_id="bodycam", stage=stage, assignee=assignee,
        media_uri="s3://dlp-raw/x.mp4" if stage is ReviewStage.PRIVACY else "s3://dlp-labeling/x",
        label_kinds=("blur_track",), created_at=FIXED_TIME,
    )  # fmt: skip


def test_cvat_webhook_checks_job_assignee_against_account_mapping() -> None:
    """회귀(감사 4-4): job 담당자가 늘 비어 있어 담당자 검사가 의미가 없었다. 블러 검수는 담당자의
    CVAT 계정과 job 담당자가 같아야만 받는다."""
    users = {"rev01": "cvat-rev01", "labeler01": "cvat-lab01"}
    priv = _cvat_task(ReviewStage.PRIVACY, "rev01")
    assert resolve_reviewer(CollectRequest("cvat:7", "cvat-rev01"), priv, None, users) == "rev01"
    for who in (None, "cvat-lab01", "rev01"):
        with pytest.raises(ReviewerMismatchError):
            resolve_reviewer(CollectRequest("cvat:7", who), priv, None, users)
    with pytest.raises(ReviewerMismatchError, match="계정 연결"):
        resolve_reviewer(CollectRequest("cvat:7", "rev01"), priv, None, {})
    lab = _cvat_task(ReviewStage.LABELING, "labeler01")
    assert resolve_reviewer(CollectRequest("cvat:7", "cvat-lab01"), lab, None, users) == "labeler01"
    with pytest.raises(ReviewerMismatchError):
        resolve_reviewer(CollectRequest("cvat:7", None), lab, None, users)
    # 연결이 없는 작업 라벨 작업은 이전처럼 담당자 이름으로 본다
    assert resolve_reviewer(CollectRequest("cvat:7", None), lab, None, {}) == "labeler01"


def test_second_task_correction_of_already_corrected_label_is_dropped() -> None:
    """회귀(감사 4-9): 같은 라벨을 보낸 두 작업이 차례로 고치면 한 라벨에 자식이 둘인 갈래 이력이
    생겨 두 수정본이 모두 현재 라벨이 됐다. 먼저 고친 것만 남기고 나중 것은 뺀다."""

    def box(x: float) -> BoxTrackPayload:
        """x 위치만 다른 박스 트랙 페이로드 (개체 cup_1, 키프레임 하나)."""
        return BoxTrackPayload(
            entity_id="cup_1",
            class_id="cup",
            keyframes=(BoxKeyframe(t_ms=0, x=x, y=0, w=10, h=10),),
        )

    original = make_label(box(0), "L", 0, 0, stream_id="bodycam")
    first = reconcile([original], [ReviewedItem("L", "bodycam", 0, 0, box(5))], session_id="s001",
                      ontology_version="1.0.0", reviewer_id="r1", now=FIXED_TIME)  # fmt: skip
    history = [original, *first.new_records]
    second = reconcile([original], [ReviewedItem("L", "bodycam", 0, 0, box(9))],
                       session_id="s001", ontology_version="1.0.0", reviewer_id="r2",
                       now=FIXED_TIME)  # fmt: skip
    assert drop_retracted(second, history) == ["L"] and not second.new_records
    current = current_labels([*history, *second.new_records])
    assert len(current) == 1 and isinstance(current[0].payload, BoxTrackPayload)
    assert current[0].payload.keyframes[0].x == 5
    # 승인만 한 경우도 이미 고친 라벨은 승인하지 않는다
    approve = reconcile([original], [ReviewedItem("L", "bodycam", 0, 0, box(0))],
                        session_id="s001", ontology_version="1.0.0", reviewer_id="r2",
                        now=FIXED_TIME)  # fmt: skip
    assert approve.approved == ["L"]
    assert drop_retracted(approve, history) == ["L"] and approve.approved == []


def test_glove_series_uses_policy_prefixes(tmp_path: Path) -> None:
    """검수 시계열 CSV는 sync 정책의 압력 채널 접두사를 명시해 넘긴다 (암묵적으로 읽지 않는다)."""
    setup = ReviewSetup(
        raw=LocalStore(tmp_path, "dlp-raw"), labeling=LocalStore(tmp_path, "dlp-labeling"),
        labeling_reader=None,  # type: ignore[arg-type]
        ontology=None,  # type: ignore[arg-type]
    )  # fmt: skip
    expected = load_sync_policy(ROOT / "config" / "policies" / "sync.yaml").glove.pressure_prefixes
    assert setup.pressure_prefixes() == expected
