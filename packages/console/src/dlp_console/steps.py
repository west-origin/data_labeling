"""세션 파이프라인 단계와 활성 규칙 (ADR 0061 3절).

단계마다 (1) 지금 상태(done·ready·blocked·skipped)와 막힌 이유, (2) 누르면 돌릴 `dlp` 명령을 정한다.
도메인 판정은 하지 않는다: 활성 규칙은 "앞 단계가 끝났는가"를 화면에서 미리 보이는 안내일 뿐이고,
실제 검사(블러 승인 조건, 검수 완료 조건, 원본 접근 권한)는 각 `dlp` 명령이 그대로 한다.
프라이버시 승인은 사람이 누르는 버튼이며 콘솔은 `dlp privacy approve`를 그대로 부른다.

명령 인자로 들어가는 값(세션 ID·담당자·장면 등)은 `IDENT`(계약 Identifier와 같은 모양, 첫 글자가
영숫자라 옵션으로 읽히지 않는다)나 정해진 선택지만 받는다. 셸을 거치지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from dlp_console.status import SessionState, lifecycle_at_least
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus
from dlp_schema.session import LifecycleState, PrivacyState

# 계약 Identifier와 같은 모양 (dlp_schema.common.Identifier)
IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class StepStatus(StrEnum):
    DONE = "done"
    READY = "ready"
    BLOCKED = "blocked"
    SKIPPED = "skipped"  # 조건(서버 없음 등)이 없어 건너뜀


class ParamError(ValueError):
    """화면에서 받은 값이 형식에 맞지 않는다."""


@dataclass(frozen=True)
class StepContext:
    """활성 규칙이 보는 값."""

    state: SessionState
    vlm: Mapping[str, bool]
    privacy_reviewers: tuple[str, ...]
    scenes: tuple[str, ...]
    # 이 세션에서 성공한 작업의 단계 ID (콘솔 작업 기록, DB에 흔적이 없는 단계의 완료 판정)
    succeeded: frozenset[str] = frozenset()


Verdict = tuple[StepStatus, str | None]
Params = Mapping[str, Any]


@dataclass(frozen=True)
class Step:
    """세션 단계 하나."""

    id: str
    title: str
    help: str  # 무엇을 하는지·실패 때 기본 안내
    judge: Callable[[StepContext], Verdict]
    command: Callable[[StepContext, Params], list[list[str]]]
    # 화면 입력 칸 이름 (privacy_assignee, assignee, scene, quick, tool). 새 이름을 쓰면
    # static/app.js `inputFor`(이름별 칸 그리기)에도 추가해야 화면에 칸이 나온다.
    inputs: tuple[str, ...] = ()
    confirm: str | None = None  # 누르기 전에 묻는 말
    links: tuple[str, ...] = field(default=())  # 보일 검수 작업 단계 (privacy, labeling)


def ident(value: Any, name: str) -> str:
    """IDENT 모양 값. 예외: ParamError."""
    text = str(value or "").strip()
    if not IDENT.match(text):
        raise ParamError(f"{name} 값이 올바르지 않습니다 (영문·숫자·_.:- , 첫 글자는 영문·숫자)")
    return text


def choice(value: Any, options: tuple[str, ...], name: str) -> str:
    text = str(value or "").strip()
    if text not in options:
        raise ParamError(f"{name}은 {', '.join(options)} 중 하나여야 합니다")
    return text


# ── 판정 도우미 ──
def _sid(ctx: StepContext) -> str:
    return ctx.state.session.session_id


def _approved(ctx: StepContext) -> bool:
    return ctx.state.session.privacy_state is PrivacyState.APPROVED


def _prelabeled(ctx: StepContext) -> bool:
    return lifecycle_at_least(ctx.state.session.lifecycle_state, LifecycleState.PRELABELED)


def _all_detected(ctx: StepContext) -> bool:
    return all(s in ctx.state.detections for s in ctx.state.video_streams)


def _all_rendered(ctx: StepContext) -> bool:
    return bool(ctx.state.rendered) and all(ctx.state.rendered.values())


def _privacy_collected(ctx: StepContext) -> bool:
    tasks = ctx.state.current_privacy_tasks()
    covered = {t.stream_id for t in tasks}
    done = all(t.status is ReviewTaskStatus.COLLECTED for t in tasks)
    return bool(tasks) and done and covered >= set(ctx.state.video_streams)


def _labeling_tasks(ctx: StepContext) -> list[ReviewTask]:
    return ctx.state.stage_tasks(ReviewStage.LABELING)


def _open(tasks: list[ReviewTask]) -> list[ReviewTask]:
    return [t for t in tasks if t.status is ReviewTaskStatus.OPEN]


# ── 단계별 판정 ──
def judge_sync(ctx: StepContext) -> Verdict:
    if "sync" in ctx.succeeded:
        return StepStatus.DONE, None
    if len(ctx.state.session.streams) == 1:
        return StepStatus.DONE, "스트림이 하나라 맞출 것이 없습니다 (눌러도 됩니다)"
    return StepStatus.READY, None


def judge_detect(ctx: StepContext) -> Verdict:
    return (StepStatus.DONE, None) if _all_detected(ctx) else (StepStatus.READY, None)


def judge_privacy_review(ctx: StepContext) -> Verdict:
    if not _all_detected(ctx):
        return StepStatus.BLOCKED, "블러 탐지를 먼저 끝내야 합니다"
    if not ctx.privacy_reviewers:
        return StepStatus.BLOCKED, (
            "검수 정책 reviewers.privacy에 원본 접근 권한자가 없습니다 (DLP_REVIEW_POLICY_FILE)"
        )
    covered = {t.stream_id for t in ctx.state.current_privacy_tasks()}
    if covered >= set(ctx.state.video_streams):
        return StepStatus.DONE, None
    return StepStatus.READY, None


def _judge_collect(tasks: list[ReviewTask], none_reason: str) -> Verdict:
    if not tasks:
        return StepStatus.BLOCKED, none_reason
    if _open(tasks):
        return StepStatus.READY, "검수 도구에서 작업을 끝낸 뒤 누르세요"
    return StepStatus.DONE, None


def judge_privacy_collect(ctx: StepContext) -> Verdict:
    return _judge_collect(
        ctx.state.current_privacy_tasks(), "블러 검수 작업을 먼저 만들어야 합니다"
    )


def judge_approve(ctx: StepContext) -> Verdict:
    if _approved(ctx):
        return StepStatus.DONE, None
    if ctx.state.held:
        return StepStatus.BLOCKED, "사고 대응 보류 중입니다 (dlp privacy hold --release 뒤 재승인)"
    if not _privacy_collected(ctx):
        return StepStatus.BLOCKED, "모든 영상의 블러 검수 결과를 먼저 수집해야 합니다"
    return StepStatus.READY, None


def judge_render(ctx: StepContext) -> Verdict:
    if not _approved(ctx):
        return StepStatus.BLOCKED, "블러를 사람이 승인해야 렌더할 수 있습니다"
    return (StepStatus.DONE, None) if _all_rendered(ctx) else (StepStatus.READY, None)


def judge_prelabel(ctx: StepContext) -> Verdict:
    if _prelabeled(ctx):
        return StepStatus.DONE, "다시 누르면 바뀐 단계만 다시 돕니다"
    if not _approved(ctx):
        return StepStatus.BLOCKED, "블러 승인 뒤에 AI 라벨을 만듭니다"
    return StepStatus.READY, None


def _judge_after_prelabel(ctx: StepContext, step_id: str) -> Verdict:
    if not _prelabeled(ctx):
        return StepStatus.BLOCKED, "AI 라벨을 먼저 만들어야 합니다"
    return (StepStatus.DONE, None) if step_id in ctx.succeeded else (StepStatus.READY, None)


def judge_relations(ctx: StepContext) -> Verdict:
    return _judge_after_prelabel(ctx, "relations")


def judge_actions(ctx: StepContext) -> Verdict:
    if not ctx.vlm.get("actions"):
        return StepStatus.SKIPPED, "사내 VLM 서버(DLP_VLM_URL) 없음, 건너뜀"
    return _judge_after_prelabel(ctx, "actions")


def judge_aireview(ctx: StepContext) -> Verdict:
    if not ctx.vlm.get("aireview"):
        return StepStatus.SKIPPED, "사내 VLM 서버(DLP_VLM_URLS) 없음, 건너뜀"
    return _judge_after_prelabel(ctx, "aireview")


def judge_labeling_review(ctx: StepContext) -> Verdict:
    if not _prelabeled(ctx):
        return StepStatus.BLOCKED, "AI 라벨을 먼저 만들어야 합니다"
    if not _all_rendered(ctx):
        return StepStatus.BLOCKED, "블러본 렌더를 먼저 끝내야 합니다"
    if _labeling_tasks(ctx):
        return StepStatus.DONE, "작업이 있습니다 (다시 누르면 새 작업을 만듭니다)"
    return StepStatus.READY, None


def judge_labeling_collect(ctx: StepContext) -> Verdict:
    return _judge_collect(_labeling_tasks(ctx), "라벨 검수 작업을 먼저 만들어야 합니다")


def judge_verify(ctx: StepContext) -> Verdict:
    if lifecycle_at_least(ctx.state.session.lifecycle_state, LifecycleState.HUMAN_VERIFIED):
        return StepStatus.DONE, None
    tasks = _labeling_tasks(ctx)
    if not tasks or _open(tasks):
        return StepStatus.BLOCKED, "라벨 검수 결과를 모두 수집해야 합니다"
    return StepStatus.READY, None


# ── 명령 ──
def _collect_commands(tasks: list[ReviewTask]) -> list[list[str]]:
    open_tasks = _open(tasks)
    if not open_tasks:
        raise ParamError("수집할 열린 작업이 없습니다")
    return [
        ["review", "collect", t.task_key, "--reviewer", ident(t.assignee, "담당자")]
        for t in open_tasks
    ]


def cmd_privacy_review(ctx: StepContext, p: Params) -> list[list[str]]:
    who = choice(p.get("assignee"), ctx.privacy_reviewers, "담당자")
    return [["review", "create", _sid(ctx), "--stage", "privacy", "--assignee", who]]


def cmd_prelabel(ctx: StepContext, p: Params) -> list[list[str]]:
    cmd = ["prelabel", "run", _sid(ctx)]
    scene = str(p.get("scene") or "").strip()
    if scene:
        cmd += ["--scene", choice(scene, ctx.scenes, "장면")]
    if p.get("quick"):
        # 첫 실행을 짧게: 전신·깊이 3D를 뺀다 (긴 영상은 나눠 돌리기를 권한다, 파일럿 기록)
        cmd += ["--skip", "body", "--skip", "depth3d"]
    return [cmd]


_TOOL_FLAGS = {"both": [], "label_studio": ["--no-cvat"], "cvat": ["--no-ls"]}


def cmd_labeling_review(ctx: StepContext, p: Params) -> list[list[str]]:
    who = ident(p.get("assignee"), "담당자")
    tool = choice(p.get("tool") or "both", tuple(_TOOL_FLAGS), "검수 도구")
    return [
        [
            "review",
            "create",
            _sid(ctx),
            "--stage",
            "labeling",
            "--assignee",
            who,
            *_TOOL_FLAGS[tool],
        ]
    ]


def _simple(*parts: str) -> Callable[[StepContext, Params], list[list[str]]]:
    def build(ctx: StepContext, _p: Params) -> list[list[str]]:
        return [[*parts[:2], _sid(ctx), *parts[2:]]]

    return build


SESSION_STEPS: tuple[Step, ...] = (
    Step("sync", "동기화", "여러 스트림의 시각을 맞춥니다.", judge_sync, _simple("sync", "run")),
    Step(
        "privacy_detect",
        "블러 탐지",
        "얼굴·문서·화면 등 가릴 대상을 자동으로 찾습니다 "
        "(CPU에서 영상 길이의 수십 배 걸릴 수 있음).",
        judge_detect,
        _simple("privacy", "detect"),
    ),
    Step(
        "privacy_review",
        "블러 검수 작업 만들기",
        "원본 접근 권한자에게 CVAT 블러 검수 작업을 만듭니다. CVAT가 켜져 있어야 합니다.",
        judge_privacy_review,
        cmd_privacy_review,
        inputs=("privacy_assignee",),
        links=("privacy",),
    ),
    Step(
        "privacy_collect",
        "블러 검수 결과 수집",
        "CVAT에서 영상 전체를 확인·수정하고 작업을 완료한 뒤 누릅니다.",
        judge_privacy_collect,
        lambda ctx, _p: _collect_commands(ctx.state.current_privacy_tasks()),
    ),
    Step(
        "privacy_approve",
        "블러 승인",
        "사람이 블러 검수를 끝냈음을 승인합니다. 승인 조건은 dlp privacy approve가 검사합니다.",
        judge_approve,
        _simple("privacy", "approve"),
        confirm=(
            "영상 전체의 블러를 직접 확인했습니까? 승인하면 블러본을 라벨러에게 보일 수 있습니다."
        ),
    ),
    Step(
        "privacy_render",
        "블러본 만들기",
        "승인된 블러로 라벨링용 블러본을 만듭니다.",
        judge_render,
        _simple("privacy", "render"),
    ),
    Step(
        "prelabel",
        "AI 1차 라벨",
        "손·객체·도구 등 자동 라벨 초안을 만듭니다. 긴 영상은 '빠르게'로 먼저 돌려 보세요.",
        judge_prelabel,
        cmd_prelabel,
        inputs=("scene", "quick"),
    ),
    Step(
        "relations",
        "관계 도출",
        "접촉 등 관계를 규칙으로 만듭니다.",
        judge_relations,
        _simple("relations", "run"),
    ),
    Step(
        "actions",
        "행동 구간",
        "사내 VLM 서버로 행동 구간과 설명을 만듭니다.",
        judge_actions,
        _simple("actions", "run"),
    ),
    Step(
        "aireview",
        "AI 1차 검수",
        "사내 VLM이 라벨마다 참고용 판정을 남깁니다 (라벨·검수 상태는 바꾸지 않음).",
        judge_aireview,
        _simple("aireview", "run"),
    ),
    Step(
        "labeling_review",
        "라벨 검수 작업 만들기",
        "블러본으로 CVAT(공간 라벨)·Label Studio(시간 구간) 검수 작업을 만듭니다.",
        judge_labeling_review,
        cmd_labeling_review,
        inputs=("assignee", "tool"),
        links=("labeling",),
    ),
    Step(
        "labeling_collect",
        "라벨 검수 결과 수집",
        "검수 도구에서 수정·제출을 끝낸 뒤 누릅니다. 담당자 계정의 결과만 받습니다.",
        judge_labeling_collect,
        lambda ctx, _p: _collect_commands(_labeling_tasks(ctx)),
    ),
    Step(
        "verify",
        "검수 완료",
        "검수가 끝났는지 검사하고 끝났으면 검수 완료로 표시합니다. 남은 사유가 있으면 보여 줍니다.",
        judge_verify,
        _simple("review", "verify"),
    ),
)

STEPS_BY_ID = {s.id: s for s in SESSION_STEPS}


def evaluate(ctx: StepContext) -> list[dict[str, Any]]:
    """화면용 단계 목록 (순서대로, 상태·이유 포함)."""
    out: list[dict[str, Any]] = []
    for step in SESSION_STEPS:
        status, reason = step.judge(ctx)
        if ctx.state.withdrawn:
            status, reason = StepStatus.BLOCKED, "사용 중지된 세션입니다"
        out.append(
            {
                "id": step.id,
                "title": step.title,
                "help": step.help,
                "status": status.value,
                "reason": reason,
                "inputs": list(step.inputs),
                "confirm": step.confirm,
                "links": list(step.links),
            }
        )
    return out


def build_command(ctx: StepContext, step_id: str, params: Params) -> tuple[Step, list[list[str]]]:
    """단계 명령. 예외: 모르는 단계·막힌 단계·값 오류는 ParamError."""
    step = STEPS_BY_ID.get(step_id)
    if step is None:
        raise ParamError(f"모르는 단계: {step_id}")
    status, reason = step.judge(ctx)
    if ctx.state.withdrawn:
        raise ParamError("사용 중지된 세션입니다")
    if status in (StepStatus.BLOCKED, StepStatus.SKIPPED):
        raise ParamError(reason or "지금은 실행할 수 없습니다")
    return step, step.command(ctx, params)
