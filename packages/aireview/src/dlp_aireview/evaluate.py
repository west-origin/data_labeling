"""골든셋 판정 정확도 (`dlp aireview eval --golden <버전>`, WP17.8, ADR 0040 7.4). DB에 쓰지 않는다.

판정 모델·프롬프트·프레임 설정을 바꿀 때 골든셋에서 판정이 얼마나 맞는지 잰다. 판정은 실제
실행기(`runner.run_session`)로 하되 기록 대상은 메모리(`MemorySink`)라 AI 표에도 라벨 이력에도 쓰지
않는다. 프레임은 운영과 같은 블러본 게이트(`media.open_blurred`)로만 얻는다 (I1).

두 방식 (`EvalCase`를 만드는 함수)
    - `perturb_case` (통제된 오류): 골든 정답 라벨을 메모리에서 교란한다(`perturb`). 교란한 라벨의
      정답은 wrong(+ 원래 값 제안, 거짓 이벤트는 삭제), 나머지는 ok다.
    - `history_case` (실제 오류 분포): 사람이 검수한 과거 모델 레코드를 대상으로 한다. 승인 → ok,
      사람 수정 → wrong, 사람 삭제 → wrong(삭제), 표본 검증·미검수·모델끼리의 교체는 뺀다. 정답은
      대상 후손 사슬의 지금 상태다: 후손 중 현재 라벨(QA가 다시 고쳤으면 그 마지막 수정본)의
      페이로드가 정답 제안이고, 사슬이 삭제로 끝났으면 삭제다 (같은 구간을 다른 종류로 다시 그린
      사람 레코드가 있으면 replaced, 그 지금 상태의 페이로드). 고쳤다가 원래 값으로 되돌린 라벨은
      맞음(approved)이다.
      맥락은 대상, 대상의 후손, 정답으로 쓴 레코드(다시 그린 레코드와 그 후손)를 뺀 현재 라벨이다
      (정답이 맥락으로 새지 않게). 한계: 검수자가 새로 그린 다른 레코드(정답으로 쓰지 않은 추가)는
      판정 시점에 없던 정보지만 맥락에 남는다.
    두 방식 모두 대상·맥락 라벨을 "검수 전 자동 라벨" 사본(출처 model, 미검수)으로 바꿔 넣는다:
    운영에서 판정이 보는 맥락은 대개 검수 전 자동 라벨이라 "확정" 표시가 붙지 않는다. 골든 라벨은
    검수된 라벨이라 일반 대상 조건(`plan.targets`)을 통과하지 못하므로 대상 목록을 직접 넘긴다
    (`plan.eval_targets`). 장갑 센서 근거 라벨(`glove_derived`)은 대상에서 뺀다 (사용자 결정).
    장갑 접촉 문장은 넣지 않는다 (평가는 원본 버킷을 읽지 않는다: 손 상태의 센서 대조 문제 유형은
    평가하지 않는다).

지표 (`score`)
    - 오류 종류별·라벨 종류별: 탐지(wrong·suspect) 재현율, ok 수(틀린 라벨을 ok라 함), 제안 정확
      일치(제안 페이로드가 정답과 같음, 삭제는 삭제 제안).
    - 전체: 틀린 라벨 탐지 재현율(`wrong_recall`), 깨끗한 라벨 오지적률(`false_flag_rate`), **ok
      놓침률(`ok_miss_rate`) = P(틀린 라벨 | AI ok)** = 틀린 라벨의 ok 수 / 모든 ok 수 (설계 7.2와
      `dlp_review.ops.ai_quality`의 ok 놓침률 P(바뀜 | AI ok)와 같은 정의. `review.yaml
      sampling.max_defect_ratio`와 비교할 수 있는 "AI가 맞다고 한 라벨 중 결함 비율"이다), 틀린
      라벨을 ok라 한 비율(`wrong_called_ok_rate` = P(AI ok | 틀린 라벨), 참고), 제안 정확 일치율,
      확신도 ECE(`dlp_eval.metrics.classification.ece`, 맞음 = 지적 여부가 정답과 같음), 판단
      불가·미응답·판정 안 됨 수, 단위/s. perturb 방식의 ok 놓침률은 교란 비율(틀린 라벨의 유병률)에
      따라 달라지므로 같은 `gate.inject`로 잰 값끼리 비교한다.
    - 게이트(`gate_result`, `aireview.yaml gate`): 탐지 재현율 하한, 오지적률 상한, ok 놓침률 상한.
      자료가 없는 지표(분모 0)는 통과로 보지 않는다. `gate.block`이 false(결정)면 경고만 한다.
    - 판정 모델이 행동 분류 모델과 같으면(`same_model_as_actions`) 행동 라벨 판정에 상관 오류가 생길
      수 있으므로 라벨 종류별 표(`by_kind["action"]`)를 따로 본다.

평가 범위: 선택 모드(`selection.mode`)는 쓰지 않는다 (`plan.plan_windows`의 평가 경로, 대상을 모두
판정한다). `--groups`면 `restrict_groups`로 그 묶음 창의 대상만 분모에 넣는다.

공개 이름: `Truth`, `EvalCase`, `LabelKey`, `Judged`, `ErrorStat`, `KindStat`, `GateResult`,
`EvalReport`, `perturb_case`, `history_case`, `restrict_groups`, `eval_input`, `judge_cases`,
`score`, `gate_result`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa
from pydantic import Field

from dlp_aireview.backend import JudgeBackend
from dlp_aireview.media import BlurredFrame, frames_for, open_blurred
from dlp_aireview.perturb import perturb
from dlp_aireview.plan import eval_targets, plan_windows
from dlp_aireview.policy import AiReviewPolicy, GatePolicy
from dlp_aireview.runner import (
    MemorySink,
    RunInput,
    RunOptions,
    RunSummary,
    SessionNotReadyError,
    planned_units,
    run_session,
    session_problem,
)
from dlp_eval.metrics.classification import ece
from dlp_media.storage import ObjectStore
from dlp_privacy.policy import PrivacyPolicy
from dlp_schema.aireview import (
    AiGroup,
    AiReviewUnit,
    AiVerdict,
    AiVerdictValue,
    glove_derived,
    worst,
)
from dlp_schema.common import Contract
from dlp_schema.db.repository import get_session
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelPayload, LabelRecord, Provenance, Source, Verification
from dlp_schema.labels import VerificationState as State
from dlp_schema.ontology import Ontology
from dlp_schema.session import Session

# 평가 사본의 출처 모델 버전 (판정 입력에는 들어가지 않는다: 모델 버전·신뢰도는 보내지 않는다)
_EVAL_MODEL = "golden-eval"
# 지적 (사람 확인이 필요하다고 본 판정)
_FLAGS = (AiVerdictValue.WRONG, AiVerdictValue.SUSPECT)
Mode = Literal["history", "perturb"]
# 정답의 출처 (보고서 `truth_source`): 골든 버전에 고정한 정답(perturb, 고정한 버전), 지금
# 라벨로 정한 정답(perturb, 정답을 고정하지 않은 이전 형식 골든셋 — 사람이 고치면 같은 버전이어도
# 바뀐다), 사람 검수 사슬의 지금 상태(history, 설계상 늘 지금 상태)
TruthSource = Literal["golden_frozen", "current_labels", "history_current"]


@dataclass(frozen=True)
class Truth:
    """대상 라벨 하나의 정답.

    Attributes:
        label_id: 대상 라벨.
        wrong: 틀린 라벨인가 (교란했거나 사람이 고치거나 지운 라벨).
        error: 오류 종류 (교란 종류, 과거 결과 corrected·deleted·replaced) 또는 깨끗하면 clean.
        expected: 정답 제안 페이로드 (원래 값·사람 수정본). 없으면 None.
        delete: 정답 제안이 삭제인가 (거짓 이벤트, 사람이 지운 라벨).
    """

    label_id: str
    wrong: bool
    error: str
    expected: LabelPayload | None = None
    delete: bool = False


@dataclass(frozen=True)
class EvalCase:
    """세션 하나의 평가 입력 (메모리).

    Attributes:
        session: 골든 세션. ontology: 그 세션 온톨로지.
        labels: 계획 입력 (대상 + 맥락, 검수 전 자동 라벨 사본).
        targets: 판정 대상 (`labels`의 부분 집합).
        truth: 라벨 ID → 정답.
    """

    session: Session
    ontology: Ontology
    labels: tuple[LabelRecord, ...]
    targets: tuple[LabelRecord, ...]
    truth: Mapping[str, Truth]


def _unreviewed(x: LabelRecord) -> LabelRecord:
    """검수 전 자동 라벨 사본 (출처 model, 미검수). 같은 ID·페이로드·구간이다."""
    return x.model_copy(
        update={
            "provenance": Provenance(source=Source.MODEL, model_version=_EVAL_MODEL),
            "confidence": x.confidence if x.confidence is not None else 0.5,
            "verification": Verification(),
        }
    )


def perturb_case(
    session: Session,
    truth_labels: Sequence[LabelRecord],
    rates: Mapping[str, float],
    *,
    seed: int,
    o: Ontology,
    policy: AiReviewPolicy,
) -> EvalCase:
    """골든 정답 라벨을 교란한 평가 입력 (`perturb`, 모듈 docstring).

    Args:
        truth_labels: 골든 정답 (`dlp_eval.runner.GoldenSession.truth`).
        rates: 교란 종류 → 라벨마다 적용 확률 (`--inject verb_swap=0.1,…`).
        seed: 교란 seed (세션마다 다르게 주면 세션마다 다른 교란).
    """
    p = perturb(truth_labels, rates, seed=seed, o=o)
    allowed = {x.label_id for x in p.labels if not glove_derived(x)}
    labels = tuple(_unreviewed(x) for x in p.labels)
    targets = tuple(x for x in eval_targets(labels, policy, o) if x.label_id in allowed)
    truth: dict[str, Truth] = {}
    for x in targets:
        kind = p.injected.get(x.label_id)
        if kind is None:
            truth[x.label_id] = Truth(x.label_id, False, "clean")
            continue
        original = p.truth.get(x.label_id)
        truth[x.label_id] = Truth(
            x.label_id, True, kind, expected=original.payload if original is not None else None,
            delete=kind == "false_event",
        )  # fmt: skip
    return EvalCase(session, o, labels, targets, truth)


def _operational(history: Sequence[LabelRecord]) -> list[LabelRecord]:
    """측정·오류 삽입 레코드를 뺀 이력."""
    return [x for x in history if not x.seeded_error and x.measurement is None]


def _same_span(a: LabelRecord, b: LabelRecord) -> bool:
    """같은 구간인가 (스트림은 둘 다 있을 때만 비교한다: Label Studio에서 새로 그린 시간 구간은
    스트림이 None이다)."""
    streams = (a.stream_id, b.stream_id)
    return (a.t_start_ms, a.t_end_ms) == (b.t_start_ms, b.t_end_ms) and (
        None in streams or streams[0] == streams[1]
    )


def _redrawn_as(r: LabelRecord, x: LabelRecord) -> bool:
    """r이 x와 같은 손·같은 구간을 다른 종류로 다시 그린 레코드인가 (행동 ↔ 사이 구간 등)."""
    hand = getattr(x.payload, "hand", None)
    return _same_span(r, x) and r.kind != x.kind and getattr(r.payload, "hand", None) == hand


def _children(history: Sequence[LabelRecord]) -> dict[str, list[LabelRecord]]:
    """부모 라벨 ID → 직접 자식 레코드."""
    out: dict[str, list[LabelRecord]] = {}
    for x in history:
        if x.parent_label_id:
            out.setdefault(x.parent_label_id, []).append(x)
    return out


def _descendants(roots: set[str], children: Mapping[str, Sequence[LabelRecord]]) -> set[str]:
    """roots의 모든 후손 ID (부모 사슬을 따라 내려간다, children = `_children` 결과)."""
    out: set[str] = set()
    stack = list(roots)
    while stack:
        for c in children.get(stack.pop(), []):
            if c.label_id not in out:
                out.add(c.label_id)
                stack.append(c.label_id)
    return out


@dataclass(frozen=True)
class _History:
    """history 방식의 이력 색인: 직접 자식, 현재 라벨, 원 라벨 없는 사람 레코드."""

    children: dict[str, list[LabelRecord]]
    current: dict[str, LabelRecord]
    fresh: tuple[LabelRecord, ...]


def _now_of(root: LabelRecord, h: _History) -> LabelRecord | None:
    """root 사슬의 지금 상태: root 자신이나 후손 중 현재 라벨 (여럿이면 가장 늦게 만든 것). 사슬이
    삭제로 끝났으면 None."""
    chain = {root.label_id, *_descendants({root.label_id}, h.children)}
    found = [h.current[i] for i in chain if i in h.current]
    return max(found, key=lambda x: (x.created_at, x.label_id)) if found else None


def _history_truth(x: LabelRecord, h: _History) -> tuple[Truth, set[str]] | None:
    """과거 모델 레코드 하나의 정답과 정답으로 쓴 레코드 ID (사람이 본 결과가 아니면 None).

    정답은 대상 후손 사슬의 지금 상태다 (모듈 docstring). 정답으로 쓴 레코드는 맥락에서 뺀다.
    """
    kids = h.children.get(x.label_id, [])
    if not any(k.provenance.source is Source.HUMAN for k in kids):
        approved = not kids and x.verification.state is State.HUMAN_APPROVED
        return (Truth(x.label_id, False, "approved"), set()) if approved else None
    now = _now_of(x, h)
    if now is not None:
        if now.payload == x.payload and _same_span(now, x):  # 고쳤다가 원래 값으로 되돌렸다
            return Truth(x.label_id, False, "approved"), {now.label_id}
        return Truth(x.label_id, True, "corrected", expected=now.payload), {now.label_id}
    for r in h.fresh:
        # Label Studio에서 같은 구간을 다른 종류로 바꿔 붙였다 (삭제 + 새 사람 레코드)
        if not _redrawn_as(r, x):
            continue
        redrawn = _now_of(r, h)
        if redrawn is not None:
            return (
                Truth(x.label_id, True, "replaced", expected=redrawn.payload),
                {r.label_id, redrawn.label_id},
            )
    return Truth(x.label_id, True, "deleted", delete=True), set()


def history_case(
    session: Session, history: Sequence[LabelRecord], *, o: Ontology, policy: AiReviewPolicy
) -> EvalCase:
    """사람이 검수한 과거 모델 레코드의 평가 입력 (모듈 docstring).

    Args:
        history: 세션의 전체 라벨 이력 (`GoldenSession.labels`).
    """
    ops = _operational(history)
    h = _History(
        children=_children(ops),
        current={x.label_id: x for x in current_labels(list(history))},
        fresh=tuple(
            x for x in ops
            if x.parent_label_id is None and x.provenance.source is Source.HUMAN
            and not x.retracted
        ),
    )  # fmt: skip
    truth: dict[str, Truth] = {}
    chosen: list[LabelRecord] = []
    answers: set[str] = set()
    for x in ops:
        if x.provenance.source is not Source.MODEL or x.retracted or glove_derived(x):
            continue
        found = _history_truth(x, h)
        if found is not None:
            truth[x.label_id] = found[0]
            answers |= found[1]
            chosen.append(x)
    ids = {x.label_id for x in chosen}
    roots = ids | answers
    hidden = roots | _descendants(roots, h.children)
    context = [x for x in h.current.values() if x.label_id not in hidden]
    labels = tuple(_unreviewed(x) for x in (*chosen, *context))
    targets = tuple(x for x in eval_targets(labels, policy, o) if x.label_id in ids)
    return EvalCase(
        session, o, labels, targets, {i: truth[i] for i in (x.label_id for x in targets)}
    )


def restrict_groups(
    case: EvalCase, policy: AiReviewPolicy, groups: Collection[AiGroup]
) -> EvalCase:
    """`--groups`: 그 묶음의 창에 드는 대상만 남긴 사례 (정답도 같이 줄인다).

    다른 묶음 대상을 남기면 판정되지 않은 채 재현율·오지적률 분모에 들어간다 (판정 버전이 같아
    보고서로 구별할 수 없다). 창은 평가 경로와 같은 `plan_windows(only=…)`로 고른다.
    """
    windows = plan_windows(case.session, case.labels, policy, case.ontology, only=case.targets)
    keep = {x.label_id for w in windows if w.group in groups for x in w.targets}
    targets = tuple(x for x in case.targets if x.label_id in keep)
    truth = {i: t for i, t in case.truth.items() if i in keep}
    return EvalCase(case.session, case.ontology, case.labels, targets, truth)


def eval_input(
    engine: sa.Engine,
    labeling: ObjectStore,
    case: EvalCase,
    policy: AiReviewPolicy,
    privacy_policy: PrivacyPolicy,
    work: Path,
    *,
    raw_bucket: str,
) -> RunInput:
    """평가 사례 → 실행기 입력. 판정할 창이 있는 스트림만 블러본 게이트(`open_blurred`)로 받는다.

    DB는 읽기만 한다 (세션 상태·렌더 해시). 장갑 문장은 넣지 않는다.

    Raises:
        SessionNotReadyError: 사용 중지·승인 보류·승인 전.
        MediaBoundaryError, RenderNotCurrentError: `open_blurred`와 같다.
    """
    sid = case.session.session_id
    with engine.connect() as conn:
        problem = session_problem(conn, get_session(conn, sid))
    if problem is not None:
        raise SessionNotReadyError(problem)
    windows = plan_windows(case.session, case.labels, policy, case.ontology, only=case.targets)
    medias = {
        s: open_blurred(engine, labeling, sid, s, privacy_policy, work, raw_bucket=raw_bucket)
        for s in sorted({w.stream_id for w in windows})
    }

    def frames(stream: str, idx: Sequence[int]) -> Iterator[BlurredFrame]:
        return frames_for(medias[stream], idx)

    return RunInput(
        session=case.session, current=case.labels, ontology=case.ontology,
        media={s: m.stream_media for s, m in medias.items()},
        blurred_sha256={s: m.blurred_sha256 for s, m in medias.items()},
        frames=frames, targets=case.targets,
    )  # fmt: skip


# 판정을 모으는 키: (세션 ID, 라벨 ID). 여러 골든 세션의 메모리 라벨(교란 사본의 거짓 이벤트 등)이
# 같은 ID를 써도 판정이 섞이지 않게 세션까지 넣는다
LabelKey = tuple[str, str]


@dataclass(frozen=True)
class Judged:
    """`judge_cases` 결과: (세션, 라벨 ID) → 가장 나쁜 판정, 단위 기록, 세션별 실행 요약, 초.

    planned는 계획한 단위에 든 대상 (판정이 없으면 장애·중단 = 미판정, 계획에 없으면 미계획).
    None이면 모른다 (판정 없는 대상을 모두 미판정으로 센다).
    """

    verdicts: dict[LabelKey, AiVerdict]
    units: tuple[AiReviewUnit, ...]
    summaries: tuple[RunSummary, ...]
    elapsed_s: float
    planned: frozenset[LabelKey] | None = None


def judge_cases(
    inputs: Sequence[RunInput],
    backend: JudgeBackend,
    policy: AiReviewPolicy,
    *,
    options: RunOptions | None = None,
    progress: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Judged:
    """평가 입력마다 실행기로 판정한다 (기록 대상은 메모리, DB에 쓰지 않는다).

    설정 오류·Ctrl-C로 멈춘 실행이 있으면 남은 사례는 판정하지 않는다 (요약의 reason으로 알린다).
    """
    start = clock()
    sink = MemorySink()
    summaries: list[RunSummary] = []
    opts = options or RunOptions()
    planned = frozenset(
        (inp.session.session_id, x.label_id)
        for inp in inputs
        for u in planned_units(inp, policy, opts)
        for x in u.window.targets
    )
    for inp in inputs:
        summary = run_session(inp, backend, sink, policy, options=options, progress=progress)
        summaries.append(summary)
        if summary.reason in ("config", "signal"):
            break
    units = tuple(sink.units.values())
    found: dict[LabelKey, list[tuple[AiVerdict, AiReviewUnit]]] = {}
    for u in units:
        for v in u.verdicts:
            found.setdefault((u.session_id, v.label_id), []).append((v, u))
    verdicts = {key: worst(c)[0] for key, c in found.items()}
    return Judged(verdicts, units, tuple(summaries), clock() - start, planned)


# ---------------------------------------------------------------- 지표


def _ratio(num: int, den: int) -> float | None:
    """num / den (분모 0이면 None: 자료 없음)."""
    return num / den if den else None


class ErrorStat(Contract):
    """오류 종류 하나(또는 깨끗한 라벨)의 집계."""

    # total: 대상 수. flagged: wrong·suspect 수. ok: AI ok 수. unsure: 판단 불가 수.
    # unjudged: 계획했지만 판정이 없는 수 (장애·중단). unplanned: 계획되지 않아 판정이 없는 수
    # (블러본 없는 스트림 등). expected / exact: 정답 제안이 있는 수와 제안이 그 정답과 같은 수.
    total: int = 0
    flagged: int = 0
    ok: int = 0
    unsure: int = 0
    unjudged: int = 0
    unplanned: int = 0
    expected: int = 0
    exact: int = 0

    @property
    def recall(self) -> float | None:
        """flagged / total."""
        return _ratio(self.flagged, self.total)


class KindStat(Contract):
    """라벨 종류 하나의 집계 (틀린 라벨 탐지 재현율, 깨끗한 라벨 오지적)."""

    wrong: int = 0
    wrong_flagged: int = 0
    clean: int = 0
    clean_flagged: int = 0


class GateResult(Contract):
    """골든 게이트 결과 (`aireview.yaml gate`)."""

    passed: bool
    failures: tuple[str, ...] = ()
    block: bool = False


class EvalReport(Contract):
    """골든 평가 보고서 (`reports/aireview-eval/*.json`, DB에 쓰지 않는다)."""

    mode: Literal["history", "perturb"]
    golden_version: str
    verdict_version: str
    model: str
    same_model_as_actions: bool
    sessions: tuple[str, ...]
    skipped: dict[str, str] = Field(default_factory=dict[str, str])
    targets: int
    units: int
    fallback_units: int
    unanswered: int
    elapsed_s: float
    units_per_s: float | None
    by_error: dict[str, ErrorStat]
    by_kind: dict[str, KindStat]
    wrong_recall: float | None
    false_flag_rate: float | None
    # P(틀린 라벨 | AI ok): 게이트 `max_ok_miss_rate`와 비교한다
    ok_miss_rate: float | None
    # P(AI ok | 틀린 라벨): 참고 (게이트에 쓰지 않는다)
    wrong_called_ok_rate: float | None
    suggestion_exact_rate: float | None
    ece: float | None
    gate: GateResult
    # 정답 근거: 출처와 골든 정답 고정 근거(`dlp_eval.runner.truth_anchor`, `dlp eval golden`의
    # golden_truth와 같은 모양. history 방식은 골든 정답을 쓰지 않아 None). 같은 골든 버전으로
    # 다시 잰 두 보고서의 정답이 같았는지 해시로 확인한다 (ADR 0041)
    truth_source: TruthSource | None = None
    golden_truth: dict[str, Any] | None = None


def _matches(v: AiVerdict, t: Truth) -> bool:
    """제안이 정답 제안과 같은가."""
    s = v.suggestion
    if s is None:
        return False
    if t.delete:
        return s.op == "delete"
    return s.payload is not None and s.payload == t.expected


def _add(stat: ErrorStat, v: AiVerdict | None, t: Truth, planned: bool) -> ErrorStat:
    """집계에 대상 하나를 더한 새 값. planned: 그 대상이 계획한 단위에 들었는가."""
    data = stat.model_dump()
    data["total"] += 1
    if v is None:
        data["unjudged" if planned else "unplanned"] += 1
    else:
        data["flagged"] += v.verdict in _FLAGS
        data["ok"] += v.verdict is AiVerdictValue.OK
        data["unsure"] += v.verdict is AiVerdictValue.UNSURE
    if t.wrong and (t.delete or t.expected is not None):
        data["expected"] += 1
        data["exact"] += v is not None and _matches(v, t)
    return ErrorStat.model_validate(data)


def _kinds(
    cases: Sequence[EvalCase], verdicts: Mapping[LabelKey, AiVerdict]
) -> dict[str, KindStat]:
    """라벨 종류별 집계."""
    out: dict[str, dict[str, int]] = {}
    for case in cases:
        sid = case.session.session_id
        for x in case.targets:
            t, v = case.truth[x.label_id], verdicts.get((sid, x.label_id))
            flagged = v is not None and v.verdict in _FLAGS
            cell = out.setdefault(x.kind, KindStat().model_dump())
            key = "wrong" if t.wrong else "clean"
            cell[key] += 1
            cell[f"{key}_flagged"] += flagged
    return {k: KindStat.model_validate(v) for k, v in sorted(out.items())}


def gate_result(report: EvalReport, gate: GatePolicy) -> GateResult:
    """골든 게이트 판정 (자료가 없는 지표는 실패로 센다)."""
    failures: list[str] = []
    checks = (
        ("틀린 라벨 탐지 재현율", report.wrong_recall, gate.min_wrong_recall, True),
        ("깨끗한 라벨 오지적률", report.false_flag_rate, gate.max_false_flag_rate, False),
        ("ok 놓침률", report.ok_miss_rate, gate.max_ok_miss_rate, False),
    )
    for name, value, bound, at_least in checks:
        if value is None:
            failures.append(f"{name}: 자료 없음 (분모 0)")
        elif (value < bound) if at_least else (value > bound):
            sign = "≥" if at_least else "≤"
            failures.append(f"{name} {value:.1%} (기준 {sign} {bound:.1%})")
    return GateResult(passed=not failures, failures=tuple(failures), block=gate.block)


def score(
    cases: Sequence[EvalCase],
    judged: Judged,
    policy: AiReviewPolicy,
    *,
    mode: Mode,
    golden_version: str,
    same_model_as_actions: bool,
    skipped: Mapping[str, str] | None = None,
    truth_source: TruthSource | None = None,
    golden_truth: Mapping[str, Any] | None = None,
) -> EvalReport:
    """판정 결과 → 평가 보고서 (모듈 docstring 지표). 순수 함수.

    truth_source·golden_truth: 정답 근거 (보고서에 그대로 남긴다, `EvalReport`).
    """
    by_error: dict[str, ErrorStat] = {}
    conf: list[float] = []
    correct: list[bool] = []
    for case in cases:
        sid = case.session.session_id
        for x in case.targets:
            key = (sid, x.label_id)
            t, v = case.truth[x.label_id], judged.verdicts.get(key)
            planned = judged.planned is None or key in judged.planned
            by_error[t.error] = _add(by_error.get(t.error, ErrorStat()), v, t, planned)
            if (
                v is not None
                and v.confidence is not None
                and v.verdict is not AiVerdictValue.UNSURE
            ):
                conf.append(v.confidence)
                correct.append((v.verdict in _FLAGS) == t.wrong)
    wrong = [s for k, s in by_error.items() if k not in ("clean", "approved")]
    clean = [s for k, s in by_error.items() if k in ("clean", "approved")]
    n_wrong = sum(s.total for s in wrong)
    n_clean = sum(s.total for s in clean)
    units = len(judged.units)
    report = EvalReport(
        mode=mode, golden_version=golden_version, verdict_version=policy.verdict_version,
        model=policy.backend.model, same_model_as_actions=same_model_as_actions,
        sessions=tuple(c.session.session_id for c in cases), skipped=dict(skipped or {}),
        targets=sum(len(c.targets) for c in cases), units=units,
        fallback_units=sum(1 for u in judged.units if u.outcome == "fallback"),
        unanswered=sum(1 for v in judged.verdicts.values() if not v.answered),
        elapsed_s=round(judged.elapsed_s, 3),
        units_per_s=units / judged.elapsed_s if judged.elapsed_s > 0 else None,
        by_error=dict(sorted(by_error.items())), by_kind=_kinds(cases, judged.verdicts),
        wrong_recall=_ratio(sum(s.flagged for s in wrong), n_wrong),
        false_flag_rate=_ratio(sum(s.flagged for s in clean), n_clean),
        ok_miss_rate=_ratio(sum(s.ok for s in wrong), sum(s.ok for s in by_error.values())),
        wrong_called_ok_rate=_ratio(sum(s.ok for s in wrong), n_wrong),
        suggestion_exact_rate=_ratio(sum(s.exact for s in wrong), sum(s.expected for s in wrong)),
        ece=None if not conf or math.isnan(e := ece(conf, correct)) else e,
        gate=GateResult(passed=False), truth_source=truth_source,
        golden_truth=dict(golden_truth) if golden_truth is not None else None,
    )  # fmt: skip
    return report.model_copy(update={"gate": gate_result(report, policy.gate)})
