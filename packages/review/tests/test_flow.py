from __future__ import annotations

import copy
import os
import uuid
from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import PrivacyPolicy, TargetPolicy, load_policy
from dlp_privacy.runner import approve_session, detect_session, render_session
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import LS_KINDS
from dlp_review.ops.policy import CvatPolicy
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.ops.runner import plan_session
from dlp_review.roles import AccessError
from dlp_review.tasks import (
    ReviewSetup,
    TaskError,
    create_labeling_tasks,
    create_privacy_tasks,
    object_key,
    video_size,
)
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    insert_review_task,
    list_review_tasks,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    BlurTrackPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
# 블러 검수 담당자와 원본 접근 권한자 (review.yaml reviewers.privacy 자리)
PRIVACY: dict[str, Any] = {"assignee": "rev01", "privacy_reviewers": ("rev01",)}
_LS = {ReviewTool.LABEL_STUDIO}  # 작업 라벨 작업을 Label Studio만 만든다 (빠르게)


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


@pytest.fixture
def setup(cvat_config: CvatPolicy) -> ReviewSetup:
    """CVAT는 이미지가 커서 CI 기본 서비스 작업에서 띄우지 않는다. 닿지 않으면 CVAT 없이 만든다
    (CVAT가 필요한 테스트만 건너뛰고 Label Studio 테스트는 돈다)."""
    try:
        cvat: CvatClient | None = CvatClient.from_env()
    except httpx.HTTPError:
        cvat = None
    return ReviewSetup(
        raw=S3Store.from_env("dlp-raw"),
        labeling=S3Store.from_env("dlp-labeling"),
        labeling_reader=S3Store.labeler_from_env("dlp-labeling"),
        ontology=load_ontology(ROOT / "config" / "ontology" / "v1"),
        cvat=cvat,
        label_studio=LabelStudioClient.from_env(),
        cvat_config=cvat_config,
    )


def _need_cvat(setup: ReviewSetup) -> CvatClient:
    if setup.cvat is None:
        pytest.skip("CVAT에 연결할 수 없습니다 (make cvat-up)")
    return setup.cvat


def _approve_blur(conn: sa.Connection, sid: str) -> None:
    """CVAT 없이 블러 검수가 끝난 것으로 둔다 (Label Studio 쪽만 시험할 때)."""
    for x in get_labels(conn, sid, kinds=["blur_track"]):
        record_review(conn, x.label_id, VerificationState.HUMAN_APPROVED, "privacy01", FIXED_TIME)
    insert_review_task(
        conn,
        ReviewTask(
            task_key=f"cvat:{sid}-offline", tool=ReviewTool.CVAT, external_id="0",
            session_id=sid, stream_id="bodycam", stage=ReviewStage.PRIVACY,
            assignee="privacy01", media_uri=f"s3://dlp-raw/sessions/{sid}/derived/bodycam.mp4",
            label_kinds=("blur_track",), created_at=FIXED_TIME,
            status=ReviewTaskStatus.COLLECTED, collected_at=FIXED_TIME,
        ),
    )  # fmt: skip


def _submit_prelabels(ls: LabelStudioClient, task_id: str) -> None:
    """Label Studio 검수자가 프리라벨(예측)을 고치지 않고 제출한다."""
    results = ls.prediction_results(int(task_id))
    ls.http.post(f"/api/tasks/{task_id}/annotations", json={"result": results}).raise_for_status()


def _oracle_policy() -> PrivacyPolicy:
    policy = load_policy(ROOT)
    return policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )


def _session(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path, detect: bool = True, **video: int
) -> tuple[str, list[LabelRecord]]:
    """블러 픽스처 영상 + 장갑 데이터로 세션을 만들고 오라클로 블러 프리라벨까지 한다.

    detect가 False면 오라클이 아무것도 찾지 못한다 (탐지 0개 스트림).
    video: generate_blur_scenario의 width·height·duration_ms (해상도를 바꿔 시험할 때).
    """
    blur = generate_blur_scenario(5, **video)  # pyright: ignore[reportArgumentType]
    blur.write(tmp_path / "bodycam.mp4")
    generate_sync_scenario(1, recorded_at=FIXED_TIME, duration_ms=3_000).write(
        tmp_path, videos=False
    )
    sid = f"rev-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    policy = _oracle_policy()
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), setup.raw, conn)
        found = blur.labels if detect else []
        oracle: dict[str, FrameDetector] = {"oracle": OracleDetector("oracle", found)}
        detect_session(conn, sid, setup.raw, oracle, {}, policy, FIXED_TIME)
        return sid, current_labels(get_labels(conn, sid, kinds=["blur_track"]))


def test_unchanged_privacy_review_round_trips_losslessly_through_cvat(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """완료 기준: 도구를 거쳐도 ID·시각·속성이 그대로라 고치지 않으면 모두 승인만 된다 (CVAT)."""
    _need_cvat(setup)
    sid, blur = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        [privacy] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
        outcome = collect_task(conn, privacy.task_key, setup, "rev01", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(x.label_id for x in blur)


def test_unchanged_labeling_review_round_trips_losslessly_through_label_studio(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """완료 기준 (Label Studio): 시간 라벨이 도구를 거쳐도 그대로라 고치지 않으면 승인만 된다."""
    sid, _ = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        _approve_blur(conn, sid)
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        temporal = [
            x.model_copy(update={"session_id": sid, "label_id": f"{sid}-{x.label_id}"})
            for x in generate_action_scenario(4).labels
            if x.kind in LS_KINDS
        ]
        insert_labels(conn, temporal)
        tasks = create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
    ls_task = next(t for t in tasks if t.task_key.startswith("label_studio:"))
    ls = setup.label_studio
    assert ls is not None
    # 회귀: 프리라벨은 예측으로 올라가 사람 주석이 아니다. 제출 전에는 수집하지 않는다
    assert ls.latest_results(int(ls_task.external_id)) is None
    with pg.begin() as conn, pytest.raises(TaskError, match="주석이 아직"):
        collect_task(conn, ls_task.task_key, setup, "labeler01", FIXED_TIME)
    _submit_prelabels(ls, ls_task.external_id)
    with pg.begin() as conn, pytest.raises(TaskError, match="담당자"):
        collect_task(conn, ls_task.task_key, setup, "labeler02", FIXED_TIME)  # 다른 사람
    with pg.begin() as conn:
        outcome = collect_task(conn, ls_task.task_key, setup, "labeler01", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(x.label_id for x in temporal)


def test_privacy_review_edits_become_label_history(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    _need_cvat(setup)
    sid, blur = _session(pg, setup, tmp_path)
    assert setup.cvat is not None
    with pg.begin() as conn:
        [task] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
    assert task.stage is ReviewStage.PRIVACY and task.media_uri.startswith("s3://dlp-raw/")

    # 검수자: 첫 트랙을 5 px 옮기고, 하나를 지우고, 하나를 새로 그린다
    tid = int(task.external_id)
    tracks = setup.cvat.get_tracks(tid)
    for shape in tracks[0]["shapes"]:
        shape["points"] = [
            shape["points"][0] + 5,
            shape["points"][1],
            shape["points"][2] + 5,
            shape["points"][3],
        ]
    new_track = copy.deepcopy(tracks[1])
    new_track["attributes"] = []
    for t in (new_track, *tracks):
        t.pop("id", None)
        for s in t["shapes"]:
            s.pop("id", None)
    kept = [tracks[0], *tracks[1:-1], new_track]  # 마지막 트랙을 지운다
    setup.cvat.put_tracks(tid, kept)

    with pg.begin() as conn:
        outcome = collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME)
        assert collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME) is None  # 한 번만
        [stored] = list_review_tasks(conn, sid, ReviewStage.PRIVACY)
        current = current_labels(get_labels(conn, sid, kinds=["blur_track"]))
    assert outcome is not None
    assert (outcome.corrected, outcome.retracted, outcome.added) == (1, 1, 1)
    assert len(outcome.approved) == len(blur) - 2
    assert stored.status is ReviewTaskStatus.COLLECTED
    assert len(current) == len(blur)  # 수정 1(대체), 삭제 1, 추가 1
    assert all(x.verification.state is not VerificationState.UNREVIEWED for x in current)
    # CVAT는 속성 순서를 보장하지 않으므로 값으로 원래 라벨 ID를 찾는다
    original_ids = {x.label_id for x in blur}
    moved_from = next(a["value"] for a in tracks[0]["attributes"] if a["value"] in original_ids)
    moved = next(x for x in current if x.parent_label_id == moved_from)
    assert isinstance(moved.payload, BlurTrackPayload)
    with pg.begin() as conn:
        assert approve_session(conn, sid).privacy_state.value == "approved"

    # 회귀: 승인 뒤 다시 블러 검수에서 고치면 승인이 풀린다 (다시 승인해야 블러본을 새로 렌더)
    later = FIXED_TIME + timedelta(hours=1)
    with pg.begin() as conn:
        [again] = create_privacy_tasks(conn, sid, setup, later, **PRIVACY)
    tracks = setup.cvat.get_tracks(int(again.external_id))
    for t in tracks:
        t.pop("id", None)
        for s in t["shapes"]:
            s.pop("id", None)
    setup.cvat.put_tracks(int(again.external_id), tracks[:-1])
    with pg.begin() as conn:
        collect_task(conn, again.task_key, setup, "rev01", later)
        assert get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED


def test_privacy_review_scales_boxes_to_downscaled_proxy(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀: 720p 원본 블러 박스를 480p 프록시 좌표로 보내고, 고치지 않으면 승인만 된다."""
    cvat = _need_cvat(setup)
    sid, blur = _session(pg, setup, tmp_path, width=1280, height=720, duration_ms=600)
    with pg.begin() as conn:
        [task] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
    proxy = tmp_path / "proxy.mp4"
    setup.raw.get_file(object_key(setup.raw, task.media_uri), proxy)
    pw, ph = video_size(proxy)
    assert (pw, ph) == (852, 480)  # 짝수로 내림 (yuv420p)
    sent = cvat.get_tracks(int(task.external_id))
    by_id = {x.label_id: x for x in blur}
    for track in sent:
        origin = next(a["value"] for a in track["attributes"] if a["value"] in by_id)
        p = by_id[origin].payload
        assert isinstance(p, BlurTrackPayload)
        shown = [k for k in p.keyframes if not k.outside]
        shape = next(s for s in track["shapes"] if not s["outside"])
        k = next(k for k in shown if k.t_ms == min(x.t_ms for x in shown))
        assert shape["points"][0] == pytest.approx(k.x * pw / 1280, abs=0.01)
        assert shape["points"][3] == pytest.approx((k.y + k.h) * ph / 720, abs=0.01)
    with pg.begin() as conn:
        outcome = collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(by_id)


class _FailingCvat:
    """원본을 올리다 실패하는 CVAT (감사 기록이 올리기 전에 남는지 본다)."""

    def __init__(self, real: CvatClient) -> None:
        self.real = real

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)

    def create_task(self, name: str, project_id: int, video: Path) -> int:
        raise RuntimeError("업로드 실패")


def test_privacy_task_checks_reviewer_and_records_grant_before_upload(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀: 원본 접근 권한자가 아닌 담당자를 막고, grant 감사 기록은 원본을 올리기 전에 남긴다."""
    cvat = _need_cvat(setup)
    sid, _ = _session(pg, setup, tmp_path)
    sink = MemorySink()
    audited = AuditedStore(setup.raw, sink, actor="svc", purpose="review.create")
    with pg.begin() as conn, pytest.raises(AccessError):
        create_privacy_tasks(
            conn, sid, replace(setup, raw=audited), FIXED_TIME,
            assignee="labeler01", privacy_reviewers=("rev01",),
        )  # fmt: skip
    assert sink.events == []  # 권한 확인이 원본을 읽기 전에 막는다
    failing = replace(setup, raw=audited, cvat=cast(CvatClient, _FailingCvat(cvat)))
    with pg.begin() as conn, pytest.raises(RuntimeError, match="업로드 실패"):
        create_privacy_tasks(conn, sid, failing, FIXED_TIME, **PRIVACY)
    grants = [e for e in sink.events if e.action == "grant"]
    assert [(e.actor, e.key) for e in grants] == [
        ("rev01", f"sessions/{sid}/derived/bodycam.proxy.mp4")
    ]


def test_labelers_only_get_watermarked_blurred_media(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    sid, _ = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        if setup.cvat is not None:
            [privacy] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
            collect_task(conn, privacy.task_key, setup, "rev01", FIXED_TIME)
        else:
            _approve_blur(conn, sid)
        with pytest.raises(Exception, match="프라이버시 승인"):
            create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        tasks = create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME)
    expected = {"cvat", "label_studio"} if setup.cvat is not None else {"label_studio"}
    assert {t.tool.value for t in tasks} == expected
    assert all(
        t.media_uri.startswith(f"s3://dlp-labeling/sessions/{sid}/review/labeler01/") for t in tasks
    )

    # Label Studio 작업의 URL은 라벨러 자격 증명으로 서명돼 라벨링 버킷만 열린다
    assert setup.label_studio is not None
    ls_task = next(t for t in tasks if t.task_key.startswith("label_studio:"))
    data = setup.label_studio.http.get(f"/api/tasks/{ls_task.external_id}").json()["data"]
    assert "dlp-raw" not in str(data)
    assert httpx.get(data["video"]).status_code == 200
    assert httpx.get(data["timeseries"]).text.startswith("time_ms,")
    raw_url = S3Store.labeler_from_env("dlp-raw").presign(f"sessions/{sid}/raw/bodycam.mp4")
    assert httpx.get(raw_url).status_code == 403


# ---------------------------------------------------------------- 감사 4차 회귀


def test_cvat_shape_mode_box_is_collected_and_stale_blurred_is_refused(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀(감사 4-1, 4-2, 4-4, 4-8).

    - 검수자가 CVAT 기본 모양(Shape) 모드로 그린 직사각형을 버리지 않고 블러로 받는다.
    - 블러 검수 작업은 담당자의 CVAT 계정에 배정되고, 검수 우선 구간이 이슈로 보인다.
    - 승인 뒤 블러 검수로 블러가 바뀌면 이전 블러본은 다시 승인해도 다시 렌더하기 전에는
      쓰지 않는다.
    """
    cvat = _need_cvat(setup)
    sid, _ = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        [first] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
        collect_task(conn, first.task_key, setup, "rev01", FIXED_TIME)
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME, tools=_LS)

    later = FIXED_TIME + timedelta(hours=2)
    with pg.begin() as conn:
        [task] = create_privacy_tasks(conn, sid, setup, later, **PRIVACY)
    tid = int(task.external_id)
    # 담당자의 CVAT 계정에 모든 job이 배정된다 (일반 사용자는 배정된 job만 본다)
    expected_user = setup.cvat_policy().users["rev01"]
    assert set(cvat.job_assignees(tid).values()) == {expected_user}
    issues = cvat.http.get("/api/issues", params={"task_id": tid, "page_size": 500}).json()
    assert issues["count"] > 0  # 검수 우선 구간이 CVAT 이슈로 보인다
    # 검수자가 놓친 얼굴을 모양 모드 직사각형 하나로 그린다
    label_id = cvat.get_tracks(tid)[0]["label_id"]
    body = {"version": 0, "tags": [], "tracks": [], "shapes": [{
        "type": "rectangle", "frame": 3, "label_id": label_id, "points": [10, 10, 60, 60],
        "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0, "attributes": [],
        "group": 0, "source": "manual",
    }]}  # fmt: skip
    r = cvat.http.patch(f"/api/tasks/{tid}/annotations", params={"action": "create"}, json=body)
    r.raise_for_status()
    with pg.begin() as conn:
        out = collect_task(conn, task.task_key, setup, "rev01", later)
        assert out is not None and out.added == 1
        [new] = out.new_records
        assert isinstance(new.payload, BlurTrackPayload)
        k0, k1 = new.payload.keyframes
        assert not k0.outside and k1.outside  # 그 프레임에만 보인다
        # 승인이 풀리고 이전 블러본은 무효
        assert get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED
        with pytest.raises(TaskError, match="승인"):
            create_labeling_tasks(conn, sid, setup, "labeler01", later, tools=_LS)
        approve_session(conn, sid)
        # 다시 승인해도 다시 렌더하기 전에는 이전 블러본으로 작업을 만들지 않는다
        with pytest.raises(TaskError, match="무효"):
            create_labeling_tasks(conn, sid, setup, "labeler01", later, tools=_LS)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
        assert create_labeling_tasks(conn, sid, setup, "labeler01", later, tools=_LS)


def test_unsupported_cvat_annotation_stops_collection(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """옮길 수 없는 CVAT 주석(다각형)은 조용히 버리지 않고 수집을 멈춘다 (승인도 못 한다)."""
    cvat = _need_cvat(setup)
    sid, _ = _session(pg, setup, tmp_path)
    with pg.begin() as conn:
        [task] = create_privacy_tasks(conn, sid, setup, FIXED_TIME, **PRIVACY)
    tid = int(task.external_id)
    label_id = cvat.get_tracks(tid)[0]["label_id"]
    body = {"version": 0, "tags": [], "tracks": [], "shapes": [{
        "type": "polygon", "frame": 2, "label_id": label_id,
        "points": [10, 10, 60, 10, 60, 60], "occluded": False, "outside": False, "z_order": 0,
        "rotation": 0.0, "attributes": [], "group": 0, "source": "manual",
    }]}  # fmt: skip
    r = cvat.http.patch(f"/api/tasks/{tid}/annotations", params={"action": "create"}, json=body)
    r.raise_for_status()
    with pg.begin() as conn, pytest.raises(TaskError, match="polygon"):
        collect_task(conn, task.task_key, setup, "rev01", FIXED_TIME)


def test_privacy_task_needs_cvat_account_of_assignee(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀(감사 4-4): 담당자의 CVAT 계정 연결이 없으면 원본을 올리기 전에 멈춘다."""
    _need_cvat(setup)
    sid, _ = _session(pg, setup, tmp_path)
    sink = MemorySink()
    audited = AuditedStore(setup.raw, sink, actor="svc", purpose="review.create")
    no_map = replace(setup, raw=audited, cvat_config=CvatPolicy(users={}, privacy_issue_limit=5))
    with pg.begin() as conn, pytest.raises(TaskError, match="CVAT 계정"):
        create_privacy_tasks(conn, sid, no_map, FIXED_TIME, **PRIVACY)
    assert [e for e in sink.events if e.action == "grant"] == []


def test_stale_blurred_refused_without_invalidation(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀(감사 4-2, 재현 그대로): 렌더 기록을 무효로 두지 못한 경로로 블러가 바뀌고 다시
    승인돼도, 블러본 해시가 지금 블러 라벨과 다르면 작업 라벨 작업을 만들지 않는다."""
    sid, blur = _session(pg, setup, tmp_path)
    policy = load_policy(ROOT)
    with pg.begin() as conn:
        _approve_blur(conn, sid)
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, policy)
    later = FIXED_TIME + timedelta(hours=2)
    with pg.begin() as conn:
        extra = blur[0].model_copy(
            update={
                "label_id": f"{sid}-h-new", "provenance": Provenance(source=Source.HUMAN),
                "confidence": None, "created_at": later,
                "verification": Verification(
                    state=VerificationState.HUMAN_CORRECTED, reviewer_id="rev01", reviewed_at=later
                ),
            }
        )  # fmt: skip
        insert_labels(conn, [extra])
        insert_review_task(
            conn,
            ReviewTask(
                task_key=f"cvat:{sid}-2", tool=ReviewTool.CVAT, external_id="0", session_id=sid,
                stream_id="bodycam", stage=ReviewStage.PRIVACY, assignee="rev01",
                media_uri=f"s3://dlp-raw/sessions/{sid}/derived/bodycam.proxy.mp4",
                label_kinds=("blur_track",), created_at=later,
                status=ReviewTaskStatus.COLLECTED, collected_at=later,
            ),
        )  # fmt: skip
        approve_session(conn, sid)
        with pytest.raises(TaskError, match="승인된 블러 라벨"):
            create_labeling_tasks(conn, sid, setup, "labeler01", later, tools=_LS)


def test_privacy_plan_covers_streams_without_detections_and_replans_per_generation(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """회귀(감사 4-5): 탐지가 0개인 스트림은 블러 배정이 없어 승인할 수 없었고, 배정 ID가 고정이라
    재탐지 뒤 다시 계획해도 새 배정이 생기지 않았다."""
    sid, blur = _session(pg, setup, tmp_path, detect=False)
    assert blur == []
    pol = load_ops_policy(ROOT)
    pol = pol.model_copy(
        update={"reviewers": pol.reviewers.model_copy(update={"privacy": ("rev01",)})}
    )
    ontology = load_ontology(ROOT / "config" / "ontology" / "v1")
    with pg.begin() as conn:
        created = plan_session(
            conn, sid, ["rev01"], pol, ontology, seed=0, now=FIXED_TIME, privacy=True
        )
        standard = [a for a in created if a.mode is ReviewMode.STANDARD]
        assert [a.stream_id for a in standard] == ["bodycam"]
        assert (
            plan_session(conn, sid, ["rev01"], pol, ontology, seed=0, now=FIXED_TIME, privacy=True)
            == []
        )  # 같은 상태에서 다시 계획하면 멱등
        # 재탐지로 블러가 생기면 새 세대 배정
        policy = _oracle_policy()
        later = FIXED_TIME + timedelta(hours=1)
        found = OracleDetector("oracle", generate_blur_scenario(5).labels)
        found.version = "oracle-2"
        detect_session(conn, sid, setup.raw, {"oracle": found}, {}, policy, later)
        again = plan_session(conn, sid, ["rev01"], pol, ontology, seed=0, now=later, privacy=True)
        assert [a.stream_id for a in again if a.mode is ReviewMode.STANDARD] == ["bodycam"]
        assert {a.assignment_id for a in again}.isdisjoint(a.assignment_id for a in created)
