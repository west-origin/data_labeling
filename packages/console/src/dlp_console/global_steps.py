"""세션 밖 단계: 데이터셋 버전·재학습·평가·운영 지표 (ADR 0061 3절).

모두 기존 명령을 그대로 부른다 (`dlp dataset build`, `dlp train run|approve`, `dlp eval golden`,
`dlp review quality`, `dlp ops weekly`). 리포트 파일은 작업 디렉터리 `reports/`에 쓴다 (저장소 밖).
같은 묶음(데이터셋·학습·평가·리포트)은 한 번에 하나씩만 돈다 (잠금 키 `global:<묶음>`).

`POST /api/global/{단계}`(`dlp_console.app`)가 `build_global`로 명령을 만든다. 단계 표는
`GLOBAL_STEPS`이고 명령 만들기는 `cmd_*`다. 세션 단계와 달리 활성 판정이 없다. 게이트 통과·배포
조건 같은 검사는 각 `dlp` 명령이 한다.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dlp_console.steps import ParamError, choice, ident

DOMAINS = ("cleaning", "caregiving", "nursing")

Params = Mapping[str, Any]


@dataclass(frozen=True)
class GlobalContext:
    """세션 밖 단계가 보는 값."""

    reports_dir: Path
    train_tasks: tuple[str, ...]  # training.yaml tasks 키


@dataclass(frozen=True)
class GlobalStep:
    """세션 밖 단계 하나.

    - id: 경로 `/api/global/{id}`와 작업 기록 `step`에 쓰는 이름.
    - group: 잠금 묶음 (`global:<group>`). 같은 묶음 단계끼리는 동시에 돌지 않는다.
    - command: (입력, 화면 값) → `dlp` 뒤에 붙일 인자 목록들. 값이 틀리면 ParamError.
    - confirm: 누르기 전에 묻는 말 (없으면 바로 시작).
    """

    id: str
    title: str
    group: str  # 잠금 묶음
    command: Callable[[GlobalContext, Params], list[list[str]]]
    confirm: str | None = None


def _stamp() -> str:
    """리포트 파일 이름에 넣는 UTC 시각 (`20261104T091500`, 이름순 = 시각순)."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _optional(p: Params, key: str, name: str) -> str | None:
    """비어 있으면 None, 아니면 IDENT 모양 검사를 통과한 값."""
    value = str(p.get(key) or "").strip()
    return ident(value, name) if value else None


def cmd_dataset_build(_ctx: GlobalContext, p: Params) -> list[list[str]]:
    """`dataset build <버전> [--golden <골든셋>] [--domain <도메인>] [--parent <이전 버전>]`.

    화면 값: `version_id`(필수), `golden`·`parent`(IDENT, 선택), `domain`(`DOMAINS` 중, 선택).
    """
    cmd = ["dataset", "build", ident(p.get("version_id"), "데이터셋 버전")]
    golden = _optional(p, "golden", "골든셋")
    if golden:
        cmd += ["--golden", golden]
    domain = str(p.get("domain") or "").strip()
    if domain:
        cmd += ["--domain", choice(domain, DOMAINS, "도메인")]
    parent = _optional(p, "parent", "이전 버전")
    if parent:
        cmd += ["--parent", parent]
    return [cmd]


def cmd_train_run(ctx: GlobalContext, p: Params) -> list[list[str]]:
    """`train run <과제> <데이터셋 버전> [--baseline-version <버전>]... [--force]`.

    화면 값: `task`(`training.yaml` 과제 중), `dataset_version`, `baseline_versions`(공백으로
    나눈 모델 버전, 끝 `*` 허용), `force`(참이면 `--force`).
    """
    task = choice(p.get("task"), ctx.train_tasks, "과제")
    cmd = ["train", "run", task, ident(p.get("dataset_version"), "데이터셋 버전")]
    for item in str(p.get("baseline_versions") or "").split():
        cmd += ["--baseline-version", _version_pattern(item)]
    if p.get("force"):
        cmd.append("--force")
    return [cmd]


def _version_pattern(item: str) -> str:
    """모델 버전 또는 앞부분+`*` (dlp eval golden·train run 규약)."""
    body = item.removesuffix("*")
    ident(body, "모델 버전")
    return item


def cmd_train_approve(_ctx: GlobalContext, p: Params) -> list[list[str]]:
    """`train approve <모델 버전>`. 게이트 통과 여부는 `dlp train approve`가 검사한다."""
    return [["train", "approve", ident(p.get("model_version"), "모델 버전")]]


def cmd_eval_golden(ctx: GlobalContext, p: Params) -> list[list[str]]:
    """`eval golden <골든셋> --model <과제>=<버전>... [--baseline …] --out <리포트>`.

    화면 값: `golden`, `models`(필수, 공백으로 나눈 `<과제>=<버전>`), `baseline`(같은 형식, 선택).
    버전은 끝 `*`를 허용한다. 리포트는 작업 디렉터리 `reports/eval-<시각>.json`에 쓴다.
    """
    cmd = ["eval", "golden", ident(p.get("golden"), "골든셋")]
    models = str(p.get("models") or "").split()
    if not models:
        raise ParamError("평가할 모델을 <과제>=<버전> 형식으로 하나 이상 적으세요")
    for flag, items in (("--model", models), ("--baseline", str(p.get("baseline") or "").split())):
        for item in items:
            task, _, version = item.partition("=")
            cmd += [flag, f"{ident(task, '과제')}={_version_pattern(version)}"]
    out = ctx.reports_dir / f"eval-{_stamp()}.json"
    return [[*cmd, "--out", str(out)]]


def cmd_review_quality(_ctx: GlobalContext, _p: Params) -> list[list[str]]:
    """`review quality` (화면 값 없음, 결과는 작업 로그로 본다)."""
    return [["review", "quality"]]


def cmd_ops_weekly(ctx: GlobalContext, p: Params) -> list[list[str]]:
    """`ops weekly --weeks <주 수> --out <reports/weekly-시각.md>`.

    화면 값: `weeks`(1~52, 기본 8). `dlp ops weekly`는 `.md` 옆에 같은 이름의 `.json`도 쓰고,
    재학습 화면(`training.latest_weekly`)이 그 JSON을 읽는다.
    """
    try:
        weeks = int(p.get("weeks") or 8)
    except (TypeError, ValueError, OverflowError) as exc:  # 문자열·무한대 등
        raise ParamError("주 수는 1~52입니다") from exc
    if not 1 <= weeks <= 52:
        raise ParamError("주 수는 1~52입니다")
    out = ctx.reports_dir / f"weekly-{_stamp()}.md"
    return [["ops", "weekly", "--weeks", str(weeks), "--out", str(out)]]


GLOBAL_STEPS: dict[str, GlobalStep] = {
    s.id: s
    for s in (
        GlobalStep("dataset_build", "데이터셋 버전 만들기", "dataset", cmd_dataset_build),
        GlobalStep("train_run", "재학습 (골든셋 게이트)", "train", cmd_train_run),
        GlobalStep(
            "train_approve",
            "배포 승인",
            "train",
            cmd_train_approve,
            confirm="게이트를 통과한 이 모델을 배포합니다. 다음 AI 라벨부터 이 모델을 씁니다.",
        ),
        GlobalStep("eval_golden", "골든셋 평가", "eval", cmd_eval_golden),
        GlobalStep("review_quality", "검수 품질", "report", cmd_review_quality),
        GlobalStep("ops_weekly", "주간 운영 지표", "report", cmd_ops_weekly),
    )
}


def build_global(
    ctx: GlobalContext, step_id: str, params: Params
) -> tuple[GlobalStep, list[list[str]]]:
    """세션 밖 단계의 `dlp` 명령을 만든다 (실행은 하지 않는다).

    예외: 모르는 단계·값 오류는 ParamError.
    """
    step = GLOBAL_STEPS.get(step_id)
    if step is None:
        raise ParamError(f"모르는 단계: {step_id}")
    return step, step.command(ctx, params)
