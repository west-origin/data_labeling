"""단계 활성 규칙과 명령 (앞 단계가 끝나지 않으면 막고 이유를 보인다)."""

from __future__ import annotations

from typing import Any

import pytest

from dlp_console.global_steps import GlobalContext, build_global
from dlp_console.status import SessionState
from dlp_console.steps import ParamError, StepContext, build_command, evaluate
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session

BODYCAM = [
    {
        "stream_id": "bodycam",
        "kind": StreamKind.BODYCAM,
        "uri": "s3://dlp-raw/s1/bodycam.mp4",
        "sync_method": SyncMethod.REFERENCE,
    }
]


def _task(stage: ReviewStage, status: ReviewTaskStatus, det: int | None = 1) -> ReviewTask:
    labeling = stage is ReviewStage.LABELING
    return ReviewTask(
        task_key=f"cvat:{stage.value}",
        tool=ReviewTool.CVAT,
        external_id="7",
        session_id="s1",
        stream_id="bodycam",
        stage=stage,
        assignee="priv01" if not labeling else "lab01",
        media_uri="s3://dlp-raw/s1/proxy.mp4" if not labeling else "s3://dlp-labeling/x.mp4",
        label_kinds=("blur_track",),
        privacy_detection_id=None if labeling else det,
        status=status,
        created_at=FIXED_TIME,
        collected_at=FIXED_TIME if status is ReviewTaskStatus.COLLECTED else None,
        render_hash="0" * 64 if labeling else None,
    )


def _ctx(**kw: Any) -> StepContext:
    session = make_session(
        "s1",
        streams=BODYCAM,
        privacy_state=kw.pop("privacy", PrivacyState.PENDING),
        lifecycle_state=kw.pop("lifecycle", LifecycleState.RAW_INGESTED),
    )
    state = SessionState(
        session=session,
        video_streams=("bodycam",),
        detections=kw.pop("detections", {}),
        tasks=kw.pop("tasks", []),
        rendered=kw.pop("rendered", {"bodycam": False}),
        withdrawn=kw.pop("withdrawn", False),
    )
    return StepContext(
        state=state,
        vlm=kw.pop("vlm", {"actions": False, "aireview": False}),
        privacy_reviewers=kw.pop("reviewers", ("priv01",)),
        scenes=("kitchen", "bathroom", "unknown"),
    )


def _status(ctx: StepContext) -> dict[str, tuple[str, str | None]]:
    return {s["id"]: (s["status"], s["reason"]) for s in evaluate(ctx)}


def test_fresh_session_only_allows_detection() -> None:
    st = _status(_ctx())
    assert st["sync"][0] == "done"  # 스트림 하나
    assert st["privacy_detect"][0] == "ready"
    assert st["privacy_review"] == ("blocked", "블러 탐지를 먼저 끝내야 합니다")
    assert st["privacy_approve"][0] == "blocked"
    assert st["prelabel"] == ("blocked", "블러 승인 뒤에 AI 라벨을 만듭니다")
    assert st["labeling_review"][0] == "blocked"
    assert st["aireview"] == ("skipped", "사내 VLM 서버(DLP_VLM_URLS) 없음, 건너뜀")
    with pytest.raises(ParamError, match="블러 승인"):
        build_command(_ctx(), "prelabel", {})


def test_privacy_review_needs_raw_access_reviewer() -> None:
    st = _status(_ctx(detections={"bodycam": 1}, reviewers=()))
    assert st["privacy_review"][0] == "blocked" and "reviewers.privacy" in str(
        st["privacy_review"][1]
    )
    ctx = _ctx(detections={"bodycam": 1})
    _, cmds = build_command(ctx, "privacy_review", {"assignee": "priv01"})
    assert cmds == [["review", "create", "s1", "--stage", "privacy", "--assignee", "priv01"]]
    with pytest.raises(ParamError):
        build_command(ctx, "privacy_review", {"assignee": "someone-else"})


def test_approve_waits_for_collected_current_generation() -> None:
    open_task = _task(ReviewStage.PRIVACY, ReviewTaskStatus.OPEN)
    st = _status(_ctx(detections={"bodycam": 1}, tasks=[open_task]))
    assert st["privacy_review"][0] == "done" and st["privacy_collect"][0] == "ready"
    assert st["privacy_approve"][0] == "blocked"
    _, cmds = build_command(
        _ctx(detections={"bodycam": 1}, tasks=[open_task]), "privacy_collect", {}
    )
    assert cmds == [["review", "collect", "cvat:privacy", "--reviewer", "priv01"]]
    done = _task(ReviewStage.PRIVACY, ReviewTaskStatus.COLLECTED)
    assert _status(_ctx(detections={"bodycam": 1}, tasks=[done]))["privacy_approve"][0] == "ready"
    # 재탐지(새 세대 2)가 있으면 옛 세대 검수로는 승인하지 않는다
    st2 = _status(_ctx(detections={"bodycam": 2}, tasks=[done]))
    assert st2["privacy_review"][0] == "ready" and st2["privacy_approve"][0] == "blocked"


def test_labeling_needs_prelabel_and_render() -> None:
    approved = {"privacy": PrivacyState.APPROVED, "detections": {"bodycam": 1}}
    st = _status(_ctx(lifecycle=LifecycleState.PRIVACY_APPROVED, **approved))
    assert st["privacy_render"][0] == "ready" and st["prelabel"][0] == "ready"
    st = _status(_ctx(lifecycle=LifecycleState.PRELABELED, **approved))
    assert st["labeling_review"] == ("blocked", "블러본 렌더를 먼저 끝내야 합니다")
    ctx = _ctx(lifecycle=LifecycleState.PRELABELED, rendered={"bodycam": True}, **approved)
    assert _status(ctx)["labeling_review"][0] == "ready"
    _, cmds = build_command(ctx, "labeling_review", {"assignee": "lab01", "tool": "label_studio"})
    assert cmds[0][-1] == "--no-cvat"
    _, cmds = build_command(ctx, "prelabel", {"scene": "kitchen", "quick": True})
    assert cmds == [
        ["prelabel", "run", "s1", "--scene", "kitchen", "--skip", "body", "--skip", "depth3d"]
    ]
    with pytest.raises(ParamError):  # 옵션처럼 읽히는 값은 받지 않는다
        build_command(ctx, "labeling_review", {"assignee": "--no-ls"})
    with pytest.raises(ParamError):
        build_command(ctx, "prelabel", {"scene": "garage"})


def test_verify_after_all_labeling_collected_and_withdrawn_blocks_all() -> None:
    base = {
        "privacy": PrivacyState.APPROVED,
        "lifecycle": LifecycleState.PRELABELED,
        "rendered": {"bodycam": True},
    }
    open_lab = _task(ReviewStage.LABELING, ReviewTaskStatus.OPEN)
    assert _status(_ctx(tasks=[open_lab], **base))["verify"][0] == "blocked"
    done_lab = _task(ReviewStage.LABELING, ReviewTaskStatus.COLLECTED)
    assert _status(_ctx(tasks=[done_lab], **base))["verify"][0] == "ready"
    st = _status(_ctx(withdrawn=True, **base))
    assert all(v == ("blocked", "사용 중지된 세션입니다") for v in st.values())


def test_vlm_steps_need_server() -> None:
    ctx = _ctx(
        privacy=PrivacyState.APPROVED,
        lifecycle=LifecycleState.PRELABELED,
        vlm={"actions": True, "aireview": True},
    )
    st = _status(ctx)
    assert st["actions"][0] == "ready" and st["aireview"][0] == "ready"
    assert build_command(ctx, "aireview", {})[1] == [["aireview", "run", "s1"]]


def test_global_steps(tmp_path: Any) -> None:
    ctx = GlobalContext(reports_dir=tmp_path, train_tasks=("objects", "hands"))
    _, cmds = build_global(
        ctx, "dataset_build", {"version_id": "ds1", "golden": "g1", "domain": ""}
    )
    assert cmds == [["dataset", "build", "ds1", "--golden", "g1"]]
    _, cmds = build_global(
        ctx, "train_run", {"task": "objects", "dataset_version": "ds1", "baseline_versions": "a* b"}
    )
    assert cmds == [
        ["train", "run", "objects", "ds1", "--baseline-version", "a*", "--baseline-version", "b"]
    ]
    with pytest.raises(ParamError):
        build_global(ctx, "train_run", {"task": "rm -rf", "dataset_version": "ds1"})
    _, cmds = build_global(ctx, "ops_weekly", {"weeks": "4"})
    assert cmds[0][:4] == ["ops", "weekly", "--weeks", "4"] and str(tmp_path) in cmds[0][-1]
