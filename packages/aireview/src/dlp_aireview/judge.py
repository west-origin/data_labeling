"""단위 하나를 판정한다: 재시도, 부분 수용, 미응답 대체, 단위 안 문맥 초과 분할 (WP17.2).

ADR 0040 3.2·5.3.

`judge_unit(backend, unit, images, ...)` 흐름
1. 전체 대상·프레임으로 묻는다. 응답을 재검사(`validate.validate_answer`)하고, 파싱 실패이거나 위반
   비율이 `retry_invalid_ratio`를 넘으면 직전 응답과 위반 목록을 대화 끝에 붙여 `max_retries`번까지
   다시 묻는다. 그 밖에는 위반 항목만 unsure + response_invalid로 낮추고 나머지는 받는다(부분 수용).
   여러 번 파싱에 성공했으면 위반이 가장 적은(같으면 나중) 응답을 받는다.
2. 문맥 초과(`VlmContextError`)면 같은 단위 안에서 프레임을 절반으로 줄이고, 그래도 안 되면 대상을
   절반으로 나눈다 (`server.max_context_splits`). 결과는 계획 단위 한 행에 합친다(`sub_requests`).
   끝내 실패한 대상은 unsure + response_invalid(answered=False)다.
3. 한 번도 형식에 맞는 응답을 받지 못하면 단위 `outcome="fallback"`(모든 대상 unsure). 응답을 받은
   결과이므로 완료로 기록한다.
4. 서버 장애(`VlmUnavailableError`)와 설정 오류(`VlmRequestError`)는 결과로 기록하지 않고 올린다
   (장애를 unsure로 쓰면 같은 버전 재실행이 건너뛰어 영구히 남는다).

제안은 재검사(`validate`)가 `suggest.materialize`로 완성 페이로드를 만든다. 온톨로지 검사에 걸리면
제안만 버린다(`suggestion_error`). 응답 원문은 기록하지 않고 sha256만 남긴다
(`JudgedUnit.response_sha256`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from dlp_aireview.backend import JudgeBackend, JudgeReply, JudgeRequest
from dlp_aireview.plan import PlannedUnit
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptImage, build_parts, retry_text
from dlp_aireview.schema import response_schema
from dlp_aireview.sensors import SensorContext
from dlp_aireview.validate import (
    ItemDraft,
    RawFinding,
    RequestView,
    ValidatedAnswer,
    missing_item,
    validate_answer,
)
from dlp_models.vlm import VlmContextError
from dlp_schema.aireview import (
    AiFinding,
    AiFrameRef,
    AiGroup,
    AiReviewUnit,
    AiVerdict,
)
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology

# 기록에 남기는 오류 문장 상한 (단위 하나)
_MAX_ERRORS = 50


@dataclass(frozen=True)
class JudgedUnit:
    """판정한 단위 하나 (DB 행 `AiReviewUnit`의 run_id·created_at 앞 단계).

    Attributes는 `AiReviewUnit`과 같은 이름이다. `record`로 계약 객체를 만든다.
    """

    unit: PlannedUnit
    outcome: Literal["answered", "fallback"]
    attempts: int
    sub_requests: int
    errors: tuple[str, ...]
    verdicts: tuple[AiVerdict, ...]
    findings: tuple[AiFinding, ...]
    frames: tuple[AiFrameRef, ...]
    prompt_digest: str
    response_sha256: str | None
    served_model: str | None
    endpoint: str | None
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None

    def record(
        self, run_id: str, created_at: datetime, blurred_sha256: str | None = None
    ) -> AiReviewUnit:
        """계약 객체 (DB에 쓸 행). `created_at`은 시간대가 있어야 한다."""
        u, w = self.unit, self.unit.window
        return AiReviewUnit(
            unit_id=u.unit_id, run_id=run_id, session_id=w.session_id, version=u.version,
            group=w.group, stream_id=w.stream_id, t_start_ms=w.t_start_ms, t_end_ms=w.t_end_ms,
            target_label_ids=tuple(x.label_id for x in w.targets),
            context_label_ids=tuple(x.label_id for x in w.context),
            render_hash=u.render_hash, blurred_sha256=blurred_sha256,
            sensor_sha256=u.sensor_sha256, ontology_version=u.ontology_version,
            frames=self.frames, prompt_digest=self.prompt_digest, outcome=self.outcome,
            attempts=self.attempts, sub_requests=self.sub_requests, errors=self.errors,
            verdicts=self.verdicts, findings=self.findings, response_sha256=self.response_sha256,
            served_model=self.served_model, endpoint=self.endpoint,
            prompt_tokens=self.prompt_tokens, cached_tokens=self.cached_tokens,
            completion_tokens=self.completion_tokens, latency_ms=self.latency_ms,
            created_at=created_at,
        )  # fmt: skip


@dataclass(frozen=True)
class _Part:
    """요청 묶음 하나의 결과 (문맥 초과로 나눈 조각 포함).

    positions: 단위 안 대상 위치. frame_map: 요청 프레임 번호 → 단위 프레임 번호.
    answer: 받아들인 재검사 결과 (None이면 형식에 맞는 응답을 끝내 받지 못함).
    """

    positions: tuple[int, ...]
    frame_map: tuple[int, ...]
    answer: ValidatedAnswer | None


@dataclass
class _Run:
    """한 단위 판정 중의 공통 입력과 누적값."""

    backend: JudgeBackend
    unit: PlannedUnit
    o: Ontology
    policy: AiReviewPolicy
    sensor: SensorContext
    attempts: int = 0
    errors: list[str] = field(default_factory=list[str])
    raws: list[str] = field(default_factory=list[str])
    replies: list[JudgeReply] = field(default_factory=list[JudgeReply])

    def note(self, message: str) -> None:
        """오류 문장을 남긴다 (상한까지)."""
        if len(self.errors) < _MAX_ERRORS:
            self.errors.append(message[:300])


def _ask(
    run: _Run,
    images: tuple[PromptImage, ...],
    frame_map: tuple[int, ...],
    positions: tuple[int, ...],
) -> _Part:
    """대상 `positions`를 한 요청으로 묻는다 (재시도 포함). 문맥 초과는 호출자에게 올린다."""
    unit, policy = run.unit, run.policy
    targets = tuple(unit.targets[p] for p in positions)
    parts = build_parts(unit, images, positions, run.o, policy, run.sensor)
    view = RequestView(unit.group, targets, len(frame_map), unit.window.entities)
    schema = response_schema(run.o, unit.group, policy)
    history: list[tuple[str, str]] = []
    best: tuple[ValidatedAnswer, str] | None = None
    for attempt in range(policy.response.max_retries + 1):
        request = JudgeRequest(
            unit_id=unit.unit_id, affinity=unit.window.window_key, group=unit.group, parts=parts,
            history=tuple(history), schema=schema,
            schema_enforcement=policy.backend.schema_enforcement,
            max_tokens=policy.backend.max_tokens, targets=targets, entities=unit.window.entities,
            frame_ms=tuple(unit.frame_ms[i] for i in frame_map),
        )  # fmt: skip
        reply = run.backend.judge(request)
        run.attempts += 1
        run.replies.append(reply)
        answer = validate_answer(reply.content, view, run.o, policy)
        for v in answer.violations:
            run.note(f"시도 {attempt + 1}: {v}")
        if answer.parsed and (best is None or answer.invalid <= best[0].invalid):
            best = (answer, reply.content)
        if not answer.retry_worthy(len(targets), policy.response.retry_invalid_ratio):
            break
        history.append((reply.content, retry_text(answer.violations)))
    if best is not None:
        run.raws.append(best[1])
    return _Part(positions, frame_map, best[0] if best else None)


def _halve_frames(
    images: tuple[PromptImage, ...], frame_map: tuple[int, ...]
) -> tuple[tuple[PromptImage, ...], tuple[int, ...]]:
    """프레임을 하나 걸러 절반으로 (확대 조각은 남은 프레임 것만, 원 프레임 번호를 다시 매김)."""
    frames = [im for im in images if im.crop_of is None]
    keep = list(range(0, len(frames), 2))
    new_index = {old: new for new, old in enumerate(keep)}
    crops = [
        PromptImage(c.image, c.t_ms, c.w, c.h, new_index[c.crop_of], c.label)
        for c in images
        if c.crop_of is not None and c.crop_of in new_index
    ]
    return (*(frames[k] for k in keep), *crops), tuple(frame_map[k] for k in keep)


def _ask_split(
    run: _Run,
    images: tuple[PromptImage, ...],
    frame_map: tuple[int, ...],
    positions: tuple[int, ...],
    splits_left: int,
    halved: bool,
) -> list[_Part]:
    """묻고, 문맥 초과면 프레임 절반 → 대상 절반 순으로 나눠 다시 묻는다 (`splits_left`번까지)."""
    try:
        return [_ask(run, images, frame_map, positions)]
    except VlmContextError as exc:
        run.attempts += 1
        run.note(f"문맥 초과 (프레임 {len(frame_map)}, 대상 {len(positions)}): {exc}")
        if splits_left > 0 and not halved and len(frame_map) > 1:
            fewer, fmap = _halve_frames(images, frame_map)
            return _ask_split(run, fewer, fmap, positions, splits_left - 1, True)
        if splits_left > 0 and len(positions) > 1:
            mid = (len(positions) + 1) // 2
            return [
                *_ask_split(run, images, frame_map, positions[:mid], splits_left - 1, halved),
                *_ask_split(run, images, frame_map, positions[mid:], splits_left - 1, halved),
            ]
        return [_Part(positions, frame_map, None)]


def _to_verdict(
    run: _Run, label: LabelRecord, draft: ItemDraft, frame_map: tuple[int, ...]
) -> AiVerdict:
    """초안 → 판정 (근거 프레임 번호 → 그 프레임 PTS ms)."""
    evidence = None if draft.frame is None else run.unit.frame_ms[frame_map[draft.frame]]
    return AiVerdict(
        label_id=label.label_id,
        kind=label.kind,  # pyright: ignore[reportArgumentType]
        verdict=draft.verdict,
        problems=draft.problems,
        reason_ko=draft.reason,
        confidence=draft.confidence,
        evidence_t_ms=evidence,
        answered=draft.answered,
        suggestion=draft.suggestion,
        suggestion_error=draft.suggestion_error,
    )


def _finding(run: _Run, raw: RawFinding, frame_map: tuple[int, ...]) -> AiFinding:
    """누락 후보 초안 → 계약 (요청 프레임 번호 → 시각)."""
    spatial = run.unit.group is AiGroup.SPATIAL
    t0, t1 = (run.unit.frame_ms[frame_map[f]] for f in (raw.f0, raw.f1))
    return AiFinding(
        kind="box_track" if spatial else "event",
        stream_id=run.unit.stream_id if spatial else None,
        t_start_ms=t0,
        t_end_ms=t1,
        value=raw.value,
        box=raw.box if spatial else None,
        reason_ko=raw.reason[:60],
    )


def _sum(values: Sequence[int | None]) -> int | None:
    """None을 뺀 합 (모두 None이면 None)."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None


def _frame_refs(images: Sequence[PromptImage]) -> tuple[AiFrameRef, ...]:
    """보낸 이미지 감사 기록 (프레임 F0… 다음 확대 조각)."""
    return tuple(
        AiFrameRef(i=k, t_ms=im.t_ms, sha256=im.image.sha256, w=im.w, h=im.h, crop_of=im.crop_of)
        for k, im in enumerate(images)
    )


def judge_unit(
    backend: JudgeBackend,
    unit: PlannedUnit,
    images: Sequence[PromptImage],
    *,
    ontology: Ontology,
    policy: AiReviewPolicy,
    sensor: SensorContext,
) -> JudgedUnit:
    """단위 하나를 판정한다 (모듈 docstring).

    Args:
        backend: 판정 백엔드.
        unit: 계획 단위.
        images: `overlay.render_unit_images` 결과 (프레임 F0… = `unit.frame_ms` 순, 다음 확대 조각).
        ontology: 세션 온톨로지.
        policy: 정책.
        sensor: 장갑 접촉 구간.

    Raises:
        VlmUnavailableError: 서버 장애 (결과로 기록하지 않는다).
        VlmRequestError: 설정 오류·서버 모델 불일치.
        ValueError: 이미지의 프레임 수가 단위 프레임 수와 다를 때.
    """
    frames = tuple(im for im in images if im.crop_of is None)
    if len(frames) != len(unit.frame_ms):
        raise ValueError("이미지의 프레임 수가 단위 프레임 수와 다릅니다")
    run = _Run(backend, unit, ontology, policy, sensor)
    all_positions = tuple(range(len(unit.targets)))
    frame_map = tuple(range(len(frames)))
    parts = _ask_split(
        run, tuple(images), frame_map, all_positions, policy.server.max_context_splits, False
    )
    by_position: dict[int, AiVerdict] = {}
    findings: list[AiFinding] = []
    for part in parts:
        for k, pos in enumerate(part.positions):
            draft = missing_item(k) if part.answer is None else part.answer.items[k]
            by_position[pos] = _to_verdict(run, unit.targets[pos], draft, part.frame_map)
        if part.answer is not None:
            findings += [_finding(run, f, part.frame_map) for f in part.answer.findings]
    replies = run.replies
    digest = build_parts(unit, tuple(images), all_positions, ontology, policy, sensor).digest
    return JudgedUnit(
        unit=unit,
        outcome="answered" if any(p.answer is not None for p in parts) else "fallback",
        attempts=run.attempts,
        sub_requests=len(parts),
        errors=tuple(run.errors),
        verdicts=tuple(by_position[p] for p in all_positions),
        findings=tuple(findings),
        frames=_frame_refs(images),
        prompt_digest=digest,
        response_sha256=(
            hashlib.sha256("\x00".join(run.raws).encode()).hexdigest() if run.raws else None
        ),
        served_model=next((r.served_model for r in reversed(replies) if r.served_model), None),
        endpoint=replies[-1].endpoint if replies else None,
        prompt_tokens=_sum([r.prompt_tokens for r in replies]),
        cached_tokens=_sum([r.cached_tokens for r in replies]),
        completion_tokens=_sum([r.completion_tokens for r in replies]),
        latency_ms=_sum([r.latency_ms for r in replies]),
    )
