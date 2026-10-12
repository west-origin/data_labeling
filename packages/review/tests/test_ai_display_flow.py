"""AI 1차 검수 표시 종단 테스트 (`@pytest.mark.services`, WP17.6, ADR 0040).

실행 중인 PostgreSQL·SeaweedFS·Label Studio(필수)와 CVAT(선택)로, 승인·렌더한 합성 세션에 AI 판정
단위를 넣고 작업 생성 → 도구 화면 데이터(예측·이슈) → 노출 기록 → 검수자 제출 → 수집을 확인한다.
판정은 `dlp aireview run` 대신 계약 객체로 직접 넣는다 (실행기는 `packages/aireview/tests`가 본다).
테스트마다 일회용 DB를 만들고 지운다.

정답 근거
- 판정 단위의 렌더 해시는 그 스트림의 지금 렌더 해시(`expected_render_hash`)다. 다른 값이면
  표시하지 않는다 (I10).
- 프리라벨 예측 결과(`labels` 컨트롤)와 작업 data는 AI 표시가 있어도 없을 때와 같다 (서명 URL은
  서명 시각만 다르고 같은 객체를 가리킨다).
- 검수자가 AI 결과가 섞인 주석을 그대로 제출해도 수집 결과는 AI가 없을 때와 같다 (모두 승인).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.actions import generate_action_scenario
from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import PrivacyPolicy, TargetPolicy, load_policy
from dlp_privacy.runner import (
    approve_session,
    detect_session,
    expected_render_hash,
    render_session,
)
from dlp_review.ai_display import AiDisplayConfig
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.collect import collect_task
from dlp_review.labelstudio import AI_DECISION, AI_FLAGS, AI_NOTE, AI_SUMMARY
from dlp_review.ops.policy import AiReviewDisplayPolicy, CvatPolicy, LabelStudioPolicy
from dlp_review.tasks import ReviewSetup, create_labeling_tasks
from dlp_schema.aireview import AiArm, AiSuggestion, AiVerdict
from dlp_schema.db.aireview_repository import get_ai_exposure, insert_ai_run, insert_ai_units
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_review_task,
    latest_privacy_detection,
    record_review,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.testing import (
    AI_VERSION,
    FIXED_TIME,
    make_ai_run,
    make_ai_unit,
    make_ai_verdict,
    make_label,
)

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
_LS = {ReviewTool.LABEL_STUDIO}
_CVAT = {ReviewTool.CVAT}


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """일회용 PostgreSQL DB (마이그레이션 + 온톨로지 v1). 테스트 뒤 지운다."""
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
def setup(cvat_config: CvatPolicy, ls_config: LabelStudioPolicy) -> ReviewSetup:
    """CVAT는 닿지 않으면 없이 (CVAT 테스트만 건너뛴다). AI 표시는 테스트가 켠다 (`_on`)."""
    try:
        cvat: CvatClient | None = CvatClient.from_env()
    except httpx.HTTPError:
        cvat = None
    return ReviewSetup(
        raw=S3Store.from_env("dlp-raw"),
        labeling=S3Store.from_env("dlp-labeling"),
        labeling_reader=S3Store.labeler_from_env("dlp-labeling"),
        ontology=None,
        cvat=cvat,
        label_studio=LabelStudioClient.from_env(),
        cvat_config=cvat_config,
        label_studio_config=ls_config,
    )


def _on(setup: ReviewSetup, **policy: Any) -> ReviewSetup:
    """AI 표시를 켠 설정 (advisory, 대조군 없음, 허용 버전 `AI_VERSION`)."""
    data: dict[str, Any] = {"mode": "advisory", "holdout_ratio": 0.0} | policy
    cfg = AiDisplayConfig(AiReviewDisplayPolicy.model_validate(data), (AI_VERSION,))
    return replace(setup, ai=cfg)


def _oracle_policy() -> PrivacyPolicy:
    """모든 블러 대상을 오라클 탐지기로 (정답 박스 그대로)."""
    policy = load_policy(ROOT)
    return policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )


def _approved_session(pg: sa.Engine, setup: ReviewSetup, tmp_path: Path) -> tuple[str, list[int]]:
    """합성 블러 영상 + 장갑 세션을 수집·탐지하고, 블러 검수가 끝난 것으로 두고 승인·렌더한다.

    반환: (세션, 바디캠 프레임 PTS ms 목록).
    """
    blur = generate_blur_scenario(5)
    blur.write(tmp_path / "bodycam.mp4")
    generate_sync_scenario(1, recorded_at=FIXED_TIME, duration_ms=3_000).write(
        tmp_path, videos=False
    )
    sid = f"air-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"},
            {"stream_id": "glove_right", "kind": "glove_right", "path": "glove_right.parquet"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), setup.raw, conn)
        oracle: dict[str, FrameDetector] = {"oracle": OracleDetector("oracle", blur.labels)}
        detect_session(conn, sid, setup.raw, oracle, {}, _oracle_policy(), FIXED_TIME)
        for x in get_labels(conn, sid, kinds=["blur_track"]):
            record_review(conn, x.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        run = latest_privacy_detection(conn, sid, "bodycam")
        insert_review_task(conn, ReviewTask(
            privacy_detection_id=run.event_id if run else None,
            task_key=f"cvat:{sid}-offline", tool=ReviewTool.CVAT, external_id="0",
            session_id=sid, stream_id="bodycam", stage=ReviewStage.PRIVACY, assignee="rev01",
            media_uri=f"s3://dlp-raw/sessions/{sid}/derived/bodycam.mp4",
            label_kinds=("blur_track",), created_at=FIXED_TIME,
            status=ReviewTaskStatus.COLLECTED, collected_at=FIXED_TIME,
        ))  # fmt: skip
        approve_session(conn, sid)
        render_session(conn, sid, setup.raw, setup.labeling, load_policy(ROOT))
    return sid, blur.frame_times


def _model(x: LabelRecord, sid: str) -> LabelRecord:
    """정답 라벨 → 검수 전 모델 출처 사본 (세션·ID 접두사)."""
    return x.model_copy(update={
        "session_id": sid, "label_id": f"{sid}-{x.label_id}", "confidence": 0.9,
        "provenance": Provenance(source=Source.MODEL, model_version="actions-1"),
    })  # fmt: skip


def _temporal(pg: sa.Engine, sid: str) -> list[LabelRecord]:
    """합성 행동 시나리오의 시간 라벨(행동·사이 구간)을 모델 출처로 넣는다."""
    labels = [
        _model(x, sid) for x in generate_action_scenario(4).labels if x.kind in ("action", "gap")
    ]
    with pg.begin() as conn:
        insert_labels(conn, labels)
    return labels


def _judge(pg: sa.Engine, sid: str, verdicts: list[AiVerdict], **unit: Any) -> None:
    """판정 실행 하나와 단위 하나를 넣는다 (렌더 해시 = 그 스트림의 지금 렌더 해시)."""
    stream = unit.get("stream_id", "bodycam")
    with pg.begin() as conn:
        rhash = expected_render_hash(conn, sid, stream, load_policy(ROOT))
        run_id = "air-run-" + uuid.uuid4().hex[:24]
        insert_ai_run(conn, make_ai_run(sid, run_id, render_hashes={stream: rhash}))
        uid = "air-" + uuid.uuid4().hex[:24]
        unit.setdefault("render_hash", rhash)
        insert_ai_units(conn, [make_ai_unit(uid, verdicts, run_id=run_id, session_id=sid, **unit)])


def _ls_task(pg: sa.Engine, setup: ReviewSetup, sid: str, who: str) -> ReviewTask:
    """Label Studio 시간 작업 하나 (직접 작업)."""
    with pg.begin() as conn:
        [task] = create_labeling_tasks(conn, sid, setup, who, FIXED_TIME, tools=_LS)
    return task


def _predictions(setup: ReviewSetup, task: ReviewTask) -> list[dict[str, Any]]:
    """작업의 예측 결과."""
    assert setup.label_studio is not None
    return setup.label_studio.prediction_results(int(task.external_id))


def _data(setup: ReviewSetup, task: ReviewTask) -> dict[str, Any]:
    """작업 data (서명 URL은 경로만 남긴다: 서명 시각이 다르다)."""
    assert setup.label_studio is not None
    data: dict[str, Any] = setup.label_studio.http.get(f"/api/tasks/{task.external_id}/").json()[
        "data"
    ]
    for k in ("video", "timeseries"):
        path: str = urlparse(str(data[k])).path
        data[k] = path.replace(f"/{task.assignee}/", "/<assignee>/")
    return data | {"assignee": "<assignee>"}


def _no_ids(value: Any) -> Any:
    """CVAT 주석에서 서버가 매기는 ID를 뺀다 (두 작업의 트랙 내용 비교)."""
    if isinstance(value, dict):
        return {k: _no_ids(v) for k, v in value.items() if k != "id"}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_no_ids(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def test_label_studio_shows_valid_verdicts_only_and_collection_is_unchanged(
    pg: sa.Engine,
    setup: ReviewSetup,
    tmp_path: Path,
    ls_as: Callable[[LabelStudioClient, str], LabelStudioClient],
) -> None:
    """지금 렌더의 판정만 AI 구간·메모·요약으로 보이고(맞음은 구간 없음), 프리라벨 예측·작업 data는
    AI가 없을 때와 같다. 노출 기록이 같은 트랜잭션에 남고, 검수자가 AI 결과·결정이 섞인 주석을
    그대로 내도 수집은 AI 없을 때처럼 모두 승인이다 (F-AIR-02, 적용은 WP17.7)."""
    sid, _ = _approved_session(pg, setup, tmp_path)
    labels = _temporal(pg, sid)
    actions = [x for x in labels if x.kind == "action"]
    gap = next(x for x in labels if x.kind == "gap")
    plain = _ls_task(pg, setup, sid, "labeler01")  # AI 끔
    swap = actions[0].payload.model_copy(update={"verb": "push"})
    _judge(pg, sid, [
        make_ai_verdict(actions[0].label_id, "wrong", evidence_t_ms=actions[0].t_start_ms,
                        suggestion=AiSuggestion(op="set", fields=("verb",), payload=swap)),
        make_ai_verdict(gap.label_id, "suspect", kind="gap", problems=("should_be_action",)),
        make_ai_verdict(actions[1].label_id, "ok"),
    ])  # fmt: skip
    # 낡은 렌더 해시의 판정은 표시하지 않는다 (I10)
    _judge(pg, sid, [make_ai_verdict(actions[2].label_id, "wrong")], render_hash="d" * 64)
    shown = _ls_task(pg, _on(setup), sid, "labeler02")

    base, with_ai = _predictions(setup, plain), _predictions(setup, shown)
    ai_names = {AI_FLAGS, AI_NOTE, AI_SUMMARY}
    assert [r for r in with_ai if r["from_name"] not in ai_names] == base
    assert _data(setup, plain) == _data(setup, shown)
    flags = [r for r in with_ai if r["from_name"] == AI_FLAGS]
    assert sorted(r["value"]["timeserieslabels"][0] for r in flags) == ["AI:의심", "AI:틀림"]
    assert all(r["readonly"] for r in flags)
    notes = {r["id"] for r in with_ai if r["from_name"] == AI_NOTE}
    assert notes == {actions[0].label_id, gap.label_id}
    [summary] = [r["value"]["text"][0] for r in with_ai if r["from_name"] == AI_SUMMARY]
    assert summary.startswith("AI 1차 검수(참고용·사람 검수 아님): 틀림 1 · 의심 1")
    with pg.connect() as conn:
        exposure = get_ai_exposure(conn, shown.task_key)
        assert get_ai_exposure(conn, plain.task_key) is None
    assert exposure is not None and exposure.arm is AiArm.SHOWN
    assert set(exposure.shown) == {actions[0].label_id, gap.label_id}
    assert set(exposure.hidden) == {actions[1].label_id}
    assert exposure.render_hash == shown.render_hash and exposure.arm_key == shown.task_key

    # 검수자가 예측(AI 결과 포함)을 그대로 내고 AI 결정 하나를 고른다
    ls = setup.label_studio
    assert ls is not None
    decision = {"id": actions[0].label_id, "from_name": AI_DECISION, "to_name": "ts",
                "type": "choices", "value": {"choices": ["apply"]}}  # fmt: skip
    resp = ls_as(ls, "labeler02").http.post(
        f"/api/tasks/{shown.external_id}/annotations", json={"result": [*with_ai, decision]}
    )
    resp.raise_for_status()
    with pg.begin() as conn:
        outcome = collect_task(conn, shown.task_key, _on(setup), "labeler02", FIXED_TIME)
    assert outcome is not None and not outcome.new_records
    assert sorted(outcome.approved) == sorted(shown.sent_label_ids or ())


def test_holdout_and_missing_verdicts_show_only_the_same_empty_summary(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """대조군과 판정 없음은 화면이 같다 ("AI 1차 검수: 없음"만). 대조군의 판정은 노출 기록
    hidden에만 남는다 (편향 없는 품질 측정)."""
    sid, _ = _approved_session(pg, setup, tmp_path)
    labels = _temporal(pg, sid)
    target = next(x for x in labels if x.kind == "action")
    _judge(pg, sid, [make_ai_verdict(target.label_id, "wrong")])
    held = _ls_task(pg, _on(setup, holdout_ratio=1.0), sid, "labeler01")
    none = _ls_task(pg, replace(_on(setup), ai=AiDisplayConfig(
        AiReviewDisplayPolicy(mode="advisory", holdout_ratio=0.0),
        ("aireview-ffffffffffff+vlm-other",),
    )), sid, "labeler02")  # fmt: skip
    for task in (held, none):
        ai = [r for r in _predictions(setup, task) if r["from_name"] in {AI_FLAGS, AI_NOTE}]
        [summary] = [r for r in _predictions(setup, task) if r["from_name"] == AI_SUMMARY]
        assert ai == [] and summary["value"]["text"] == ["AI 1차 검수: 없음"]
    with pg.connect() as conn:
        held_e, none_e = get_ai_exposure(conn, held.task_key), get_ai_exposure(conn, none.task_key)
    assert held_e is not None and held_e.arm is AiArm.HOLDOUT
    assert set(held_e.hidden) == {target.label_id} and not held_e.shown
    assert none_e is not None and none_e.arm is AiArm.NO_VERDICTS and not none_e.hidden


def test_cvat_gets_ai_issues_without_project_changes(
    pg: sa.Engine, setup: ReviewSetup, tmp_path: Path
) -> None:
    """CVAT 1단계: 프로젝트 라벨·속성은 그대로이고 보인 판정만 근거 시각 프레임의 이슈가 된다.
    트랙(주석)은 AI 없을 때와 같다."""
    if setup.cvat is None:
        pytest.skip("CVAT에 연결할 수 없습니다 (make cvat-up)")
    cvat = setup.cvat
    sid, times = _approved_session(pg, setup, tmp_path)
    first, last, mid = times[0], times[-1], len(times) // 2
    box = {"kind": "box_track", "entity_id": "cup_01", "class_id": "cup", "keyframes": [
        {"t_ms": first, "x": 20, "y": 30, "w": 40, "h": 40},
        {"t_ms": last, "x": 120, "y": 30, "w": 40, "h": 40},
    ]}  # fmt: skip
    track = make_label(
        box, label_id=f"{sid}-cup", t_start_ms=first, t_end_ms=last, session_id=sid,
        stream_id="bodycam",
        confidence=0.9, provenance=Provenance(source=Source.MODEL, model_version="det-1"),
    )  # fmt: skip
    with pg.begin() as conn:
        insert_labels(conn, [track])
    bowl = track.payload.model_copy(update={"class_id": "bucket"})
    _judge(pg, sid, [make_ai_verdict(
        track.label_id, "wrong", kind="box_track", problems=("wrong_class",),
        evidence_t_ms=times[mid], suggestion=AiSuggestion(op="set", fields=("class_id",),
                                                          payload=bowl),
    )], group="spatial", t_start_ms=first, t_end_ms=last)  # fmt: skip
    with pg.begin() as conn:
        [plain] = create_labeling_tasks(conn, sid, setup, "labeler01", FIXED_TIME, tools=_CVAT)
        [task] = create_labeling_tasks(conn, sid, _on(setup), "rev01", FIXED_TIME, tools=_CVAT)

    def issues(t: ReviewTask) -> list[dict[str, Any]]:
        """작업의 이슈와 첫 댓글(메시지)."""
        res = cvat.http.get("/api/issues", params={"task_id": t.external_id, "page_size": 100})
        out: list[dict[str, Any]] = []
        for issue in res.json()["results"]:
            [first] = cvat.http.get("/api/comments", params={"issue_id": issue["id"]}).json()[
                "results"
            ][:1]
            out.append(issue | {"message": first["message"]})
        return out

    o = load_ontology(ROOT / "config" / "ontology" / "v1")
    assert issues(plain) == []
    [issue] = issues(task)
    cup, bucket = o.objects["cup"].ko, o.objects["bucket"].ko
    assert issue["message"].startswith(f"[AI 틀림·참고] 클래스 {cup}→{bucket} 제안")
    assert issue["message"].endswith(f"{times[mid] / 1000:.1f}s · 라벨 {track.label_id}")
    assert issue["frame"] == mid  # 근거 시각에 떠 있는 프레임 (PTS 인덱스)
    assert _no_ids(cvat.get_tracks(int(plain.external_id))) == _no_ids(
        cvat.get_tracks(int(task.external_id))
    )
    with pg.connect() as conn:
        exposure = get_ai_exposure(conn, task.task_key)
    assert exposure is not None and exposure.arm is AiArm.SHOWN
    assert list(exposure.shown) == [track.label_id]
