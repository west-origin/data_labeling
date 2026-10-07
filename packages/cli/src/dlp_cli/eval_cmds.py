"""평가 하위 명령."""

from __future__ import annotations

import argparse
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_eval.gate import decide
from dlp_eval.harness import evaluate
from dlp_eval.policy import TASKS, Task, load_policy
from dlp_eval.runner import MissingPredictionsError, load_golden_merged, write_report
from dlp_schema import repo_root


def _models(items: list[str], flag: str) -> dict[Task, list[str]]:
    """<과제>=<버전> 목록 → 과제별 버전들. 같은 과제를 여러 번 주면 예측을 합친다."""
    out: dict[Task, list[str]] = {}
    for item in items:
        task, _, version = item.partition("=")
        if task not in TASKS or not version:
            raise SystemExit(
                f"{flag}은 <과제>=<모델 버전> 형식입니다 (과제: {', '.join(TASKS)}): {item}"
            )
        versions = out.setdefault(task, [])
        if version not in versions:
            versions.append(version)
    return out


def _names(models: dict[Task, list[str]]) -> dict[str, str]:
    return {str(k): "+".join(v) for k, v in models.items()}


def cmd_golden(args: argparse.Namespace) -> int:
    policy = load_policy(repo_root())
    models = _models(args.model, "--model")
    baseline_models = _models(args.baseline, "--baseline")
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            golden = load_golden_merged(conn, args.golden, models)
            # 기존 모델 버전을 잘못 적으면 예측이 비어 기존 지표가 0이 되고 어떤 후보든 통과한다
            old = (
                load_golden_merged(conn, args.golden, baseline_models, require_predictions=True)
                if baseline_models
                else None
            )
    except MissingPredictionsError as exc:
        raise SystemExit(f"--baseline: {exc}") from exc
    finally:
        engine.dispose()
    report = evaluate(golden, policy, golden_version=args.golden, model_versions=_names(models))
    baseline = None
    if old is not None:
        baseline = evaluate(
            old, policy, golden_version=args.golden, model_versions=_names(baseline_models)
        )
    decision = decide(report, baseline, policy)
    write_report(Path(args.out), report, decision)
    for task, d in decision.tasks.items():
        print(
            f"{task}: {'통과' if d.passed else '실패'}" + "".join(f"\n  - {r}" for r in d.reasons)
        )
    print(f"리포트: {args.out} (+ .md). 배포 게이트 {'통과' if decision.passed else '실패'}")
    return 0 if decision.passed else 1


VERSION_HELP = (
    "<과제>=<모델 버전> (여러 번). 한 과제에 여러 번 주면 예측을 세션별로 합친다 "
    "(예: objects=<COCO 버전> objects=<OWLv2 도구 버전>). 버전 끝에 *를 붙이면 그 앞부분으로 "
    "시작하는 버전 전부다. 접촉·행동·3D 궤적은 버전에 세션 입력 해시가 들어가 세션마다 다르므로 "
    "contact=contact-heuristic-1* 처럼 앞부분+*로 준다. 관계·커버리지도 relations-* 처럼"
)


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    ev = sub.add_parser("eval", help="골든셋 평가와 배포 게이트")
    esub = ev.add_subparsers(dest="eval_command", required=True)
    golden = esub.add_parser(
        "golden", help="골든셋으로 모델 버전을 평가 (게이트 실패 시 종료 코드 1)"
    )
    golden.add_argument("golden", help="골든셋 버전")
    golden.add_argument("--model", action="append", default=[], help=VERSION_HELP)
    golden.add_argument(
        "--baseline",
        action="append",
        default=[],
        help=(
            "비교할 기존 모델 <과제>=<버전> (--model과 같은 형식). "
            "골든셋에 예측이 없는 버전이면 실패"
        ),
    )
    golden.add_argument("--out", default="reports/eval.json")
    golden.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    golden.set_defaults(func=cmd_golden)
