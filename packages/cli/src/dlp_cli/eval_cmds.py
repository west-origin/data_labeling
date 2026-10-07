"""평가 하위 명령."""

from __future__ import annotations

import argparse
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_eval.gate import decide
from dlp_eval.harness import evaluate
from dlp_eval.policy import TASKS, Task, load_policy
from dlp_eval.runner import load_golden, write_report
from dlp_schema import repo_root


def _models(items: list[str]) -> dict[Task, str]:
    out: dict[Task, str] = {}
    for item in items:
        task, _, version = item.partition("=")
        if task not in TASKS or not version:
            raise SystemExit(
                f"--model은 <과제>=<모델 버전> 형식입니다 (과제: {', '.join(TASKS)}): {item}"
            )
        out[task] = version
    return out


def cmd_golden(args: argparse.Namespace) -> int:
    policy = load_policy(repo_root())
    models, baseline_models = _models(args.model), _models(args.baseline)
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        golden = load_golden(conn, args.golden, models)
        old = load_golden(conn, args.golden, baseline_models) if baseline_models else None
    engine.dispose()
    report = evaluate(
        golden,
        policy,
        golden_version=args.golden,
        model_versions={str(k): v for k, v in models.items()},
    )
    baseline = None
    if old is not None:
        versions = {str(k): v for k, v in baseline_models.items()}
        baseline = evaluate(old, policy, golden_version=args.golden, model_versions=versions)
    decision = decide(report, baseline, policy)
    write_report(Path(args.out), report, decision)
    for task, d in decision.tasks.items():
        print(
            f"{task}: {'통과' if d.passed else '실패'}" + "".join(f"\n  - {r}" for r in d.reasons)
        )
    print(f"리포트: {args.out} (+ .md). 배포 게이트 {'통과' if decision.passed else '실패'}")
    return 0 if decision.passed else 1


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    ev = sub.add_parser("eval", help="골든셋 평가와 배포 게이트")
    esub = ev.add_subparsers(dest="eval_command", required=True)
    golden = esub.add_parser(
        "golden", help="골든셋으로 모델 버전을 평가 (게이트 실패 시 종료 코드 1)"
    )
    golden.add_argument("golden", help="골든셋 버전")
    golden.add_argument(
        "--model",
        action="append",
        default=[],
        help="<과제>=<모델 버전> (여러 번). 관계·커버리지는 relations-* 처럼 앞부분+*",
    )
    golden.add_argument(
        "--baseline", action="append", default=[], help="비교할 기존 모델 <과제>=<버전>"
    )
    golden.add_argument("--out", default="reports/eval.json")
    golden.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    golden.set_defaults(func=cmd_golden)
