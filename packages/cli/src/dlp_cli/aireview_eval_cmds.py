"""`dlp aireview eval`: 골든셋 판정 정확도와 게이트 (WP17.8, ADR 0040 7.4). DB에 쓰지 않는다.

`aireview_cmds.add_commands`가 등록한다.

`dlp aireview eval --golden <버전> [--mode perturb|history] [--inject verb_swap=0.1,…] [--seed N]
[--groups …] [--vlm-url URL …] [--store s3] [--gate] [--out FILE] [--url]`
    - 세션은 `dlp_eval.runner.golden_sessions`(사용 중지 제외)로 고르고, 사례는
      `dlp_aireview.evaluate.perturb_case`(기본, 골든 정답을 메모리에서 교란, 비율은 `--inject` 또는
      `aireview.yaml gate.inject`) 또는 `history_case`(사람이 검수한 과거 모델 레코드)로 만든다.
    - 프레임은 운영과 같은 블러본 게이트(`open_blurred`)로만 얻고 판정은 실행기(`run_session`)로
      하되 기록은 메모리다 (AI 표·라벨 이력에 쓰지 않는다). 장갑 원본은 읽지 않는다.
    - 선택 모드(`aireview.yaml selection.mode`)는 쓰지 않는다 (대상을 모두 판정). `--groups`면 그
      묶음 창에 든 대상만 분모에 넣는다 (`evaluate.restrict_groups`). 판정이 없는 대상은 미판정
      (계획했지만 장애·중단)과 미계획으로 나눠 센다.
    - 보고서는 `reports/aireview-eval/<골든>-<방식>-<시각>.json`(Git 제외, `--out`으로 바꾼다).
      MLflow 기록은 하지 않는다 (후속: 실측 단계에서 필요하면 더한다).
    - 게이트(`--gate`, `aireview.yaml gate`): 실패하면 경고를 출력한다. `gate.block`이 false(결정)라
      종료 코드는 0이다 (차단은 파일럿 WP17.9 뒤 결정. true로 바꾸면 종료 1).
    - 판정 모델이 행동 분류 모델(`actions.yaml vlm.model`)과 같으면 경고하고 행동 라벨 지표를
      따로 보인다 (자기 검수 상관 오류).
    - ok 놓침률은 P(틀린 라벨 | AI ok)다 (품질 보고와 같은 정의, `evaluate` docstring). 틀린 라벨을
      ok라 한 비율 P(AI ok | 틀린 라벨)은 참고로 함께 보인다.
    - 정답 근거(ADR 0041): 보고서에 `truth_source`와 `golden_truth`(`dlp_eval.runner.truth_anchor`,
      perturb만)를 남긴다. 정답을 고정하지 않은 이전 형식 골든셋이면 `dlp eval golden`처럼 경고하고
      다시 고정하는 명령을 안내한다. history는 사람 검수 사슬의 지금 상태가 정답이라고 알린다.
    종료: 0 완료(게이트 경고 포함) / 1 게이트 실패(block이 true일 때만) / 2 서버 주소 없음, 또는
    인자 오류(`--inject`의 모르는 교란 종류·형식 오류 등, argparse 사용법 오류) / 3 서버 장애로 일부
    미판정 / 5 설정 불일치(4xx·모델 불일치·사내 주소 아님·원본 버킷 저장소) / 6 골든셋 없음·판정할
    세션 없음 / 130 Ctrl-C. `aireview.yaml gate.inject`의 모르는 교란 종류는 정책 로드 오류다
    (`GatePolicy`).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import NoResultFound

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_aireview.backend import EndpointPool, JudgeBackend
from dlp_aireview.evaluate import (
    EvalCase,
    EvalReport,
    TruthSource,
    eval_input,
    history_case,
    judge_cases,
    perturb_case,
    restrict_groups,
    score,
)
from dlp_aireview.media import MediaBoundaryError
from dlp_aireview.perturb import KINDS
from dlp_aireview.policy import AiReviewPolicy, load_policy
from dlp_aireview.runner import RunInput, RunOptions, SessionNotReadyError
from dlp_cli.aireview_common import CONFIG, UNAVAILABLE, add_url_arg, parse_groups
from dlp_cli.aireview_run_cmds import open_pool
from dlp_cli.schema_cmds import database_url
from dlp_eval.runner import (
    GoldenSession,
    golden_sessions,
    refreeze_hint,
    truth_anchor,
    truth_lag,
)
from dlp_media.audit import current_actor
from dlp_media.storage import store_from_spec
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema import load_config, repo_root
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import get_golden_set
from dlp_schema.lineage import GoldenSet

# 골든셋이 없거나 판정할 세션이 없음 (dlp aireview run의 전제 실패 코드와 같다)
_NOTHING = 6
# 실행 요약 사유 → 종료 코드 (설정 > 장애 > Ctrl-C)
_CODES = {"config": CONFIG, "unavailable": UNAVAILABLE, "signal": 130}


def parse_inject(text: str) -> dict[str, float]:
    """`verb_swap=0.1,class_swap=0.2` → 사전 (argparse 형식 함수, 종류는 `perturb.KINDS`)."""
    out: dict[str, float] = {}
    for part in (p.strip() for p in text.split(",") if p.strip()):
        name, sep, value = part.partition("=")
        try:
            rate = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"<종류>=<비율> 형식이 아닙니다: {part}") from exc
        if not sep or name not in KINDS or not 0 <= rate <= 1:
            raise argparse.ArgumentTypeError(
                f"교란 종류는 {', '.join(KINDS)} 중 하나, 비율은 0~1입니다: {part}"
            )
        out[name] = rate
    return out


@dataclass(frozen=True)
class _Truth:
    """평가 정답 근거 (보고서 `truth_source`·`golden_truth`)."""

    source: TruthSource
    anchor: dict[str, Any] | None


def _truth_basis(
    args: argparse.Namespace, golden: GoldenSet, sessions: Sequence[GoldenSession]
) -> _Truth:
    """정답 근거를 정하고 경고를 출력한다 (`dlp eval golden`과 같은 경고, ADR 0041).

    perturb는 골든 정답을 교란하므로 고정 근거(`truth_anchor`)를 남긴다. 고정하지 않은 이전 형식이면
    현재 라벨로 정답을 정했다고 경고하고 다시 고정하는 명령을 안내한다. 고정한 정답이 지금 정답과
    다르면 차이를 경고한다 (평가는 고정한 정답으로 한다). history는 설계상 사람 검수 사슬의 지금
    상태를 정답으로 쓴다 (골든 정답을 쓰지 않는다).
    """
    if args.mode == "history":
        print(
            f"참고: history 방식은 사람 검수 결과의 지금 상태를 정답으로 씁니다 (골든셋 "
            f"{args.golden}의 고정 정답과 무관, 검수가 진행되면 같은 버전이어도 정답이 바뀐다)"
        )
        return _Truth("history_current", None)
    anchor = truth_anchor(golden, sessions)
    if not anchor["truth_frozen"]:
        print(
            f"경고: 골든셋 {args.golden}은 정답을 고정하지 않은 이전 형식이라 현재 라벨로 "
            f"평가합니다 (같은 버전이어도 정답이 바뀔 수 있다). {refreeze_hint(args.golden)}"
        )
        return _Truth("current_labels", anchor)
    lag = truth_lag(golden, sessions)
    if lag is not None:
        print(f"경고: {lag.message()}")
    return _Truth("golden_frozen", anchor)


def _cases(
    args: argparse.Namespace, engine: sa.Engine, policy: AiReviewPolicy
) -> tuple[list[EvalCase], _Truth] | int:
    """골든 세션 → 평가 사례와 정답 근거 (읽기 전용). 골든셋이 없으면 종료 6.

    골든셋과 세션을 한 연결에서 한 번 읽어 정답 근거와 사례에 함께 쓴다 (`dlp eval golden`과 같다).
    """
    rates = args.inject if args.inject is not None else dict(policy.gate.inject)
    out: list[EvalCase] = []
    with engine.connect() as conn:
        try:
            golden = get_golden_set(conn, args.golden)
            sessions = golden_sessions(conn, args.golden)
        except NoResultFound:
            print(f"[전제 실패] 골든셋 {args.golden}이 없습니다")
            return _NOTHING
        basis = _truth_basis(args, golden, sessions)
        for i, g in enumerate(sessions):
            o = get_session_ontology(conn, g.session.session_id)
            if args.mode == "history":
                out.append(history_case(g.session, g.labels, o=o, policy=policy))
            else:
                out.append(
                    perturb_case(g.session, g.truth, rates, seed=args.seed + i, o=o, policy=policy)
                )
    return out, basis


def _inputs(
    args: argparse.Namespace,
    engine: sa.Engine,
    cases: Sequence[EvalCase],
    policy: AiReviewPolicy,
    work: Path,
) -> tuple[list[RunInput], list[EvalCase], dict[str, str]]:
    """사례마다 블러본 게이트로 실행기 입력을 만든다. (입력, 입력이 된 사례, 건너뛴 세션 → 사유).

    Raises:
        MediaBoundaryError: 라벨링 저장소가 원본 버킷일 때 (호출자가 종료 5).
    """
    buckets = load_config(repo_root() / "config/defaults.yaml").buckets
    labeling = store_from_spec(args.store, buckets.labeling)
    privacy = load_privacy_policy(repo_root())
    inputs: list[RunInput] = []
    kept: list[EvalCase] = []
    skipped: dict[str, str] = {}
    for case in cases:
        sid = case.session.session_id
        if args.groups:
            # 다른 묶음 대상을 분모에 남기지 않는다 (판정되지 않은 채 재현율·오지적률을 틀어지게
            # 한다)
            case = restrict_groups(case, policy, set(args.groups))
        if not case.targets:
            skipped[sid] = "판정할 대상 없음" + (" (고른 묶음)" if args.groups else "")
            continue
        try:
            inp = eval_input(engine, labeling, case, policy, privacy, work, raw_bucket=buckets.raw)
        except (SessionNotReadyError, RenderNotCurrentError) as exc:
            skipped[sid] = str(exc)
            continue
        inputs.append(inp)
        kept.append(case)
    return inputs, kept, skipped


def _print_report(report: EvalReport, out: Path, gate: bool) -> None:
    """평가 요약 출력."""

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{v:.1%}"

    print(
        f"[eval] {report.golden_version} {report.mode} · 판정 버전 {report.verdict_version} · "
        f"세션 {len(report.sessions)} · 대상 {report.targets} · 단위 {report.units}"
    )
    print(
        f"  틀린 라벨 탐지 재현율 {pct(report.wrong_recall)} · 깨끗한 라벨 오지적률 "
        f"{pct(report.false_flag_rate)} · ok 놓침률(AI ok 중 틀린 라벨) "
        f"{pct(report.ok_miss_rate)} · 틀린 라벨을 ok라 함 {pct(report.wrong_called_ok_rate)} · "
        f"제안 정확 일치 "
        f"{pct(report.suggestion_exact_rate)} · ECE {pct(report.ece)}"
    )
    for name, s in report.by_error.items():
        print(f"  {name}: {s.flagged}/{s.total} 지적, ok {s.ok}, 판단 불가 {s.unsure}, "
              f"미판정 {s.unjudged}(장애·중단), 미계획 {s.unplanned}, "
              f"제안 일치 {s.exact}/{s.expected} (표현 불가 {s.unattainable})")  # fmt: skip
    for kind, k in report.by_kind.items():
        print(f"  종류 {kind}: 틀린 라벨 {k.wrong_flagged}/{k.wrong} 지적 · 깨끗한 라벨 "
              f"{k.clean_flagged}/{k.clean} 지적")  # fmt: skip
    for sid, reason in report.skipped.items():
        print(f"  건너뜀 {sid}: {reason}")
    if report.same_model_as_actions:
        print(
            "[경고] 판정 모델이 행동 분류 모델과 같습니다: 행동 라벨(종류 action) 지표를 따로 본다"
        )
    if gate:
        if report.gate.passed:
            print("  게이트: 통과")
        else:
            kind = "차단" if report.gate.block else "경고만 (gate.block: false)"
            print(f"[게이트 실패 — {kind}] " + "; ".join(report.gate.failures))
    print(f"보고서: {out}")


def _options(policy: AiReviewPolicy, backend: JudgeBackend, args: argparse.Namespace) -> RunOptions:
    """실행 설정 (작업자 = 엔드포인트 수 * 엔드포인트별 동시성)."""
    endpoints = max(1, len(backend.info.endpoints))
    return RunOptions(
        workers=endpoints * policy.server.concurrency_per_endpoint,
        groups=frozenset(args.groups) if args.groups else None, actor=current_actor(),
        same_model_as_actions=load_actions_policy(repo_root()).vlm.model == policy.backend.model,
        endpoint_states=backend.states if isinstance(backend, EndpointPool) else None,
    )  # fmt: skip


def _code(report_codes: Sequence[str | None]) -> int:
    """실행 요약 사유 중 가장 심각한 종료 코드 (없으면 0)."""
    for reason in ("config", "unavailable", "signal"):
        if reason in report_codes:
            return _CODES[reason]
    return 0


def _evaluate(
    args: argparse.Namespace, engine: sa.Engine, policy: AiReviewPolicy, backend: JudgeBackend
) -> tuple[EvalReport, int] | int:
    """사례 → 입력 → 판정 → 보고서. (보고서, 실행 종료 코드) 또는 바로 끝낼 종료 코드."""
    found = _cases(args, engine, policy)
    if isinstance(found, int):
        return found
    cases, basis = found
    options = _options(policy, backend, args)
    with tempfile.TemporaryDirectory(prefix="aireview-eval-") as tmp:
        try:
            inputs, kept, skipped = _inputs(args, engine, cases, policy, Path(tmp))
        except MediaBoundaryError as exc:
            print(f"[설정 오류] {exc}")
            return CONFIG
        if not inputs:
            print(f"[전제 실패] {args.golden}: 판정할 세션이 없습니다 ({len(skipped)}개 건너뜀)")
            for sid, reason in skipped.items():
                print(f"  {sid}: {reason}")
            return _NOTHING
        judged = judge_cases(inputs, backend, policy, options=options, progress=print)
    report = score(
        kept, judged, policy, mode=args.mode, golden_version=args.golden,
        same_model_as_actions=options.same_model_as_actions, skipped=skipped,
        truth_source=basis.source, golden_truth=basis.anchor,
    )  # fmt: skip
    return report, _code([s.reason for s in judged.summaries])


def cmd_eval(args: argparse.Namespace) -> int:
    """`dlp aireview eval` (모듈 docstring). 부작용: VLM 서버 호출(사내 주소만), 라벨링 버킷 블러본
    읽기(임시 디렉터리), 보고서 JSON 파일. DB에는 쓰지 않는다."""
    root = repo_root()
    policy = load_policy(root)
    opened = open_pool(args, policy)
    if isinstance(opened, int):
        return opened
    engine = sa.create_engine(database_url(args.url))
    try:
        result = _evaluate(args, engine, policy, opened)
    finally:
        engine.dispose()
    if isinstance(result, int):
        return result
    report, code = result
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else (
        root / "reports" / "aireview-eval" / f"{args.golden}-{args.mode}-{stamp}.json"
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = report.model_dump(mode="json")
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_report(report, out, args.gate)
    if code:
        return code
    return 1 if args.gate and report.gate.block and not report.gate.passed else 0


def add_eval_command(asub: Any) -> None:
    """`aireview eval` 인자."""
    e = asub.add_parser("eval", help="골든셋 판정 정확도·게이트 (DB에 쓰지 않음)")
    e.add_argument("--golden", required=True, help="골든셋 버전")
    e.add_argument("--mode", choices=["perturb", "history"], default="perturb")
    e.add_argument(
        "--inject", type=parse_inject,
        help="교란 비율 (예: verb_swap=0.1,class_swap=0.2, 기본: aireview.yaml gate.inject)",
    )  # fmt: skip
    e.add_argument("--seed", type=int, default=0, help="교란 seed (세션마다 +1)")
    e.add_argument("--groups", type=parse_groups, help="이 묶음만 (기본 전부)")
    e.add_argument("--gate", action="store_true", help="게이트 판정을 출력한다 (실패해도 경고만)")
    e.add_argument("--out", help="보고서 JSON (기본 reports/aireview-eval/…)")
    e.add_argument("--store", default="s3", help="라벨링 버킷 ('s3' 또는 'local:<디렉터리>')")
    e.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    add_url_arg(e)
    e.set_defaults(func=cmd_eval)
