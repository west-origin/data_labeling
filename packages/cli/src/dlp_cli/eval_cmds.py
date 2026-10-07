"""골든셋 평가 하위 명령 (WP11, ADR 0013·0025).

등록하는 명령:
- `dlp eval golden <골든셋 버전> --model <과제>=<버전> … [--baseline …] [--out]` — DB에 있는 모델
  출처 라벨(예측)을 골든셋의 사람 정답과 비교해 과제별 지표(mAP·HOTA·PCK·구간 F1 등)와 하위 집단
  리포트를 만들고, 배포 게이트(절대 기준 + 기존 모델 대비 회귀)를 판정한다.

입력 → 출력: DB `golden_sets`·`label_records` → `<out>`(JSON)과 같은 이름의 `.md` 리포트.
종료 코드: 게이트 통과 0, 실패 1. 형식 오류는 `SystemExit` 메시지.

주의:
- 이 명령은 DB에 이미 기록된 예측만 평가한다(읽기 전용). 재학습 후보 모델의 골든셋 평가는
  `dlp train run`이 하며, 그 예측은 DB에 쓰지 않는다.
- `--baseline`을 잘못 적으면 예측이 비어 기존 지표가 0이 되고 어떤 후보든 통과하므로,
  기존 모델은 `require_predictions=True`로 읽어 예측이 없으면 실패시킨다.

정책 출처: `config/policies/evaluation.yaml` (과제별 지표·임계값·게이트). 게이트에 모르는 지표
이름을 적으면 정책을 읽을 때 실패한다 (`dlp_eval.policy.TASK_METRICS`, ADR 0031).
"""

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
    """<과제>=<버전> 목록 → 과제별 버전들. 같은 과제를 여러 번 주면 예측을 합친다.

    인자:
        items: 명령줄 값 목록. 과제는 `dlp_eval.policy.TASKS` 중 하나. 버전 끝의 `*`는 앞부분
            일치(해석은 `load_golden_merged` 쪽)이며 여기서는 문자열 그대로 둔다.
        flag: 오류 메시지에 쓸 옵션 이름 (`--model`/`--baseline`).

    반환: 과제 → 중복 없는 버전 목록(입력 순서 유지).
    예외: 형식이 틀리거나 모르는 과제면 `SystemExit`.
    """
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
    """리포트에 적을 과제 → 모델 버전 이름. 여러 버전은 `+`로 잇는다."""
    return {str(k): "+".join(v) for k, v in models.items()}


def cmd_golden(args: argparse.Namespace) -> int:
    """`dlp eval golden`: 골든셋 평가와 배포 게이트 판정.

    흐름:
    1. `--model`·`--baseline`을 과제별 버전 목록으로 해석한다.
    2. 읽기 전용 연결로 골든셋 정답과 예측을 세션별로 합쳐 읽는다.
    3. 후보(`--model`)와 기존(`--baseline`, 있으면)을 같은 정책으로 평가한다.
    4. `decide`로 과제별 통과/실패와 이유를 정하고 리포트를 쓴다.

    반환: 게이트 통과 0, 실패 1.
    부작용: `args.out`(기본 `reports/eval.json`)과 `.md` 파일 쓰기. DB 쓰기 없음.
    """
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


# `--model` 도움말. 세션마다 입력 해시가 붙는 버전(접촉·행동·3D 궤적·관계)은 앞부분+`*`로 준다는
# 규약을 사용자에게 알린다 (버전 형식은 각 단계 runner의 model_version 규칙, ADR 0026)
VERSION_HELP = (
    "<과제>=<모델 버전> (여러 번). 한 과제에 여러 번 주면 예측을 세션별로 합친다 "
    "(예: objects=<COCO 버전> objects=<OWLv2 도구 버전>). 버전 끝에 *를 붙이면 그 앞부분으로 "
    "시작하는 버전 전부다. 접촉·행동·3D 궤적은 버전에 세션 입력 해시가 들어가 세션마다 다르므로 "
    "contact=contact-heuristic-1* 처럼 앞부분+*로 준다. 관계·커버리지도 relations-* 처럼"
)


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`eval golden` 하위 명령을 등록한다."""
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
