"""세션 파이프라인 단계와 활성 규칙 (ADR 0061 3절).

단계마다 (1) 지금 상태(done·ready·blocked·skipped)와 막힌 이유, (2) 누르면 돌릴 `dlp` 명령을 정한다.
도메인 판정은 하지 않는다: 활성 규칙은 "앞 단계가 끝났는가"를 화면에서 미리 보이는 안내일 뿐이고,
실제 검사(블러 승인 조건, 검수 완료 조건, 원본 접근 권한)는 각 `dlp` 명령이 그대로 한다.
프라이버시 승인은 사람이 누르는 버튼이며 콘솔은 `dlp privacy approve`를 그대로 부른다.

명령 인자로 들어가는 값(세션 ID·담당자·장면 등)은 `IDENT`(계약 Identifier와 같은 모양, 첫 글자가
영숫자라 옵션으로 읽히지 않는다)나 정해진 선택지만 받는다. 셸을 거치지 않는다.

주요 이름:

- `SESSION_STEPS`·`STEPS_BY_ID`: 화면 순서대로의 세션 단계 표. 새 단계는 여기에 `Step`으로
  넣는다.
- `evaluate`: 화면용 단계 목록 (상태·이유). `build_command`: 누른 단계의 명령 (막힌 단계는 거부).
- `judge_*`: 단계별 상태 판정. `cmd_*`: 화면 값이 필요한 단계의 명령 만들기.
- `ident`·`choice`: 화면 값 검사 (`ParamError`, `global_steps`도 쓴다).

판정 근거 두 가지:

- DB 상태 (`SessionState`): 탐지 세대, 검수 작업, 생애주기, 블러 승인, 렌더 게이트, 보류.
- 콘솔 작업 기록 (`StepContext.succeeded`): DB에 단계 흔적이 남지 않는 동기화·관계·행동·AI
  검수는 이 콘솔에서 성공한 작업이 있으면 done이다. 그래서 CLI로 직접 돌린 단계는 ready로 보일
  수 있다. 다시 눌러도 명령이 멱등이라 안전하다.

사용 중지(withdrawn) 세션은 모든 단계가 blocked이고 명령을 만들지 않는다.
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
    """단계 상태. ready·done은 누를 수 있고, blocked·skipped는 `build_command`가 거부한다."""

    DONE = "done"
    READY = "ready"
    BLOCKED = "blocked"
    SKIPPED = "skipped"  # 조건(서버 없음 등)이 없어 건너뜀


class ParamError(ValueError):
    """화면에서 받은 값이 형식에 맞지 않는다."""


@dataclass(frozen=True)
class StepContext:
    """활성 규칙이 보는 값.

    - state: DB에서 읽은 세션 상태.
    - vlm: 사내 VLM 서버 설정 여부 (`actions`, `aireview` 키, `status.vlm_servers`).
    - privacy_reviewers: 블러 검수 담당자 선택지 (검수 정책 `reviewers.privacy`).
    - scenes: 장면 선택지 (`prelabel.yaml scene.labels`).
    """

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
    """세션 단계 하나.

    - id: 경로 `/api/sessions/{세션}/steps/{id}`와 작업 기록 `step`에 쓰는 이름.
    - judge: 지금 상태와 막힌 이유(화면 안내, 없으면 None).
    - command: (판정 입력, 화면 값) → `dlp` 뒤에 붙일 인자 목록들. 값이 틀리면 ParamError.
    """

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
    """앞뒤 공백을 뗀 값이 `options` 중 하나인지 본다. 예외: ParamError (`name`은 안내에 쓴다)."""
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
    """모든 영상 스트림이 지금 탐지 세대의 블러 검수 작업을 갖고, 그 작업이 모두 수집됐는가."""
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
    """동기화: 콘솔에서 성공했거나 스트림이 하나면 done, 아니면 ready (막는 조건 없음)."""
    if "sync" in ctx.succeeded:
        return StepStatus.DONE, None
    if len(ctx.state.session.streams) == 1:
        return StepStatus.DONE, "스트림이 하나라 맞출 것이 없습니다 (눌러도 됩니다)"
    return StepStatus.READY, None


def judge_detect(ctx: StepContext) -> Verdict:
    """블러 탐지: 모든 영상 스트림에 완료 탐지 세대가 있으면 done, 아니면 ready."""
    return (StepStatus.DONE, None) if _all_detected(ctx) else (StepStatus.READY, None)


def judge_privacy_review(ctx: StepContext) -> Verdict:
    """블러 검수 작업 만들기.

    탐지가 덜 끝났거나 원본 접근 권한자가 없으면 blocked. 모든 영상 스트림에 지금 탐지 세대를
    본 작업이 있으면 done. 탐지를 다시 돌려 세대가 바뀌면 다시 ready가 된다.
    """
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
    """수집 단계 공통: 작업이 없으면 blocked, 열린 작업이 있으면 ready, 모두 닫혔으면 done."""
    if not tasks:
        return StepStatus.BLOCKED, none_reason
    if _open(tasks):
        return StepStatus.READY, "검수 도구에서 작업을 끝낸 뒤 누르세요"
    return StepStatus.DONE, None


def judge_privacy_collect(ctx: StepContext) -> Verdict:
    """블러 검수 결과 수집: 지금 탐지 세대의 블러 검수 작업을 본다 (`_judge_collect`)."""
    return _judge_collect(
        ctx.state.current_privacy_tasks(), "블러 검수 작업을 먼저 만들어야 합니다"
    )


def judge_approve(ctx: StepContext) -> Verdict:
    """블러 승인: 승인됐으면 done. 보류 중이거나 블러 검수 수집이 덜 됐으면 blocked.

    ready여도 실제 승인 조건은 `dlp privacy approve`가 다시 검사한다.
    """
    if _approved(ctx):
        return StepStatus.DONE, None
    if ctx.state.held:
        return StepStatus.BLOCKED, "사고 대응 보류 중입니다 (dlp privacy hold --release 뒤 재승인)"
    if not _privacy_collected(ctx):
        return StepStatus.BLOCKED, "모든 영상의 블러 검수 결과를 먼저 수집해야 합니다"
    return StepStatus.READY, None


def judge_render(ctx: StepContext) -> Verdict:
    """블러본 만들기: 승인 전 blocked. 모든 영상 스트림이 렌더 게이트를 통과하면 done."""
    if not _approved(ctx):
        return StepStatus.BLOCKED, "블러를 사람이 승인해야 렌더할 수 있습니다"
    return (StepStatus.DONE, None) if _all_rendered(ctx) else (StepStatus.READY, None)


def judge_prelabel(ctx: StepContext) -> Verdict:
    """AI 1차 라벨: 생애주기가 prelabeled 이상이면 done(다시 눌러도 됨), 승인 전 blocked."""
    if _prelabeled(ctx):
        return StepStatus.DONE, "다시 누르면 바뀐 단계만 다시 돕니다"
    if not _approved(ctx):
        return StepStatus.BLOCKED, "블러 승인 뒤에 AI 라벨을 만듭니다"
    return StepStatus.READY, None


def _judge_after_prelabel(ctx: StepContext, step_id: str) -> Verdict:
    """프리라벨 뒤 단계 공통: 프리라벨 전 blocked, 콘솔에서 성공한 적 있으면 done."""
    if not _prelabeled(ctx):
        return StepStatus.BLOCKED, "AI 라벨을 먼저 만들어야 합니다"
    return (StepStatus.DONE, None) if step_id in ctx.succeeded else (StepStatus.READY, None)


def judge_relations(ctx: StepContext) -> Verdict:
    """관계 도출: `_judge_after_prelabel`."""
    return _judge_after_prelabel(ctx, "relations")


def judge_actions(ctx: StepContext) -> Verdict:
    """행동 구간: `DLP_VLM_URL`이 없으면 skipped, 있으면 `_judge_after_prelabel`."""
    if not ctx.vlm.get("actions"):
        return StepStatus.SKIPPED, "사내 VLM 서버(DLP_VLM_URL) 없음, 건너뜀"
    return _judge_after_prelabel(ctx, "actions")


def judge_aireview(ctx: StepContext) -> Verdict:
    """AI 1차 검수: `DLP_VLM_URLS`가 없으면 skipped, 있으면 `_judge_after_prelabel`."""
    if not ctx.vlm.get("aireview"):
        return StepStatus.SKIPPED, "사내 VLM 서버(DLP_VLM_URLS) 없음, 건너뜀"
    return _judge_after_prelabel(ctx, "aireview")


def judge_labeling_review(ctx: StepContext) -> Verdict:
    """라벨 검수 작업 만들기.

    프리라벨 전이나 렌더 전이면 blocked. 회수(revoked)되지 않은 라벨 검수 작업이 하나라도 있으면
    done이다. done에서 다시 누르면 새 작업을 만든다 (`dlp review create`).
    """
    if not _prelabeled(ctx):
        return StepStatus.BLOCKED, "AI 라벨을 먼저 만들어야 합니다"
    if not _all_rendered(ctx):
        return StepStatus.BLOCKED, "블러본 렌더를 먼저 끝내야 합니다"
    if _labeling_tasks(ctx):
        return StepStatus.DONE, "작업이 있습니다 (다시 누르면 새 작업을 만듭니다)"
    return StepStatus.READY, None


def judge_labeling_collect(ctx: StepContext) -> Verdict:
    """라벨 검수 결과 수집: 회수(revoked)되지 않은 라벨 검수 작업을 본다 (`_judge_collect`)."""
    return _judge_collect(_labeling_tasks(ctx), "라벨 검수 작업을 먼저 만들어야 합니다")


def judge_verify(ctx: StepContext) -> Verdict:
    """검수 완료: human_verified 이상이면 done. 라벨 검수 작업이 없거나 열린 것이 있으면 blocked.

    ready여도 완료 판정(남은 사유)은 `dlp review verify`가 한다.
    """
    if lifecycle_at_least(ctx.state.session.lifecycle_state, LifecycleState.HUMAN_VERIFIED):
        return StepStatus.DONE, None
    tasks = _labeling_tasks(ctx)
    if not tasks or _open(tasks):
        return StepStatus.BLOCKED, "라벨 검수 결과를 모두 수집해야 합니다"
    return StepStatus.READY, None


# ── 명령 ──
def _collect_commands(tasks: list[ReviewTask]) -> list[list[str]]:
    """열린 작업마다 `review collect <작업 키> --reviewer <담당자>` (담당자 계정 결과만 받는다)."""
    open_tasks = _open(tasks)
    if not open_tasks:
        raise ParamError("수집할 열린 작업이 없습니다")
    return [
        ["review", "collect", t.task_key, "--reviewer", ident(t.assignee, "담당자")]
        for t in open_tasks
    ]


def cmd_privacy_review(ctx: StepContext, p: Params) -> list[list[str]]:
    """`review create <세션> --stage privacy --assignee <담당자>`.

    담당자(`assignee`)는 원본 접근 권한자 목록에서만 고른다. 블러 검수는 원본 영상을 보기 때문이다.
    """
    who = choice(p.get("assignee"), ctx.privacy_reviewers, "담당자")
    return [["review", "create", _sid(ctx), "--stage", "privacy", "--assignee", who]]


def cmd_prelabel(ctx: StepContext, p: Params) -> list[list[str]]:
    """`prelabel run <세션> [--scene <장면>] [--skip body --skip depth3d]`.

    화면 값: `scene`(비우면 자동, 장면 선택지 중 하나), `quick`(참이면 전신·깊이 3D를 뺀다).
    """
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
    """`review create <세션> --stage labeling --assignee <담당자> [--no-cvat | --no-ls]`.

    화면 값: `assignee`(IDENT 모양), `tool`(both·label_studio·cvat, 기본 both).
    """
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
    """화면 값이 없는 단계의 명령 만들기. `_simple("sync", "run")` → `sync run <세션>`."""

    def build(ctx: StepContext, _p: Params) -> list[list[str]]:
        """`parts` 앞 두 개 뒤에 세션 ID를 넣는다 (화면 값은 보지 않는다)."""
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
    """화면용 단계 목록 (`SESSION_STEPS` 순서, 상태·이유·입력 칸·확인 문구·검수 링크 포함)."""
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
    """누른 단계의 `dlp` 명령을 만든다 (실행은 하지 않는다).

    화면이 보인 상태와 상관없이 판정을 다시 한다. 화면이 낡았어도 막힌 단계는 돌지 않는다.

    Raises:
        ParamError: 모르는 단계, 사용 중지 세션, blocked·skipped 단계(이유를 메시지로), 값 오류.
    """
    step = STEPS_BY_ID.get(step_id)
    if step is None:
        raise ParamError(f"모르는 단계: {step_id}")
    status, reason = step.judge(ctx)
    if ctx.state.withdrawn:
        raise ParamError("사용 중지된 세션입니다")
    if status in (StepStatus.BLOCKED, StepStatus.SKIPPED):
        raise ParamError(reason or "지금은 실행할 수 없습니다")
    return step, step.command(ctx, params)
