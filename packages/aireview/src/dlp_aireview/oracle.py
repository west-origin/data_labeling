"""정답 라벨로 답하는 판정 stub (CPU·CI용 `JudgeBackend`, WP17.2, ADR 0040).

`OracleJudge(truth, faults)`는 요청의 대상 라벨을 같은 ID의 정답 라벨과 비교해 응답 JSON을 만든다.
프레임 픽셀은 보지 않는다 (배관 검사용: 정답을 아는 합성 오류를 `perturb`로 넣고, 계획 → 프롬프트 →
응답 재검사 → 제안 완성 경로가 그 오류를 100% wrong으로 잡고 원래 값을 제안하는지 본다).

- 정답에 같은 ID가 없으면(넣은 거짓 이벤트 등) wrong: 이벤트는 false_event, 공간 트랙은
  false_positive이고 삭제를 제안한다.
- 정답과 페이로드가 같으면 ok. 다르면 바뀐 값마다 문제 유형과 원래 값 제안 (경계는 제안하지 않는다).
- faults: 앞 몇 번의 호출에서 낼 고장 (순서대로 소비).
  `not_json`, `bad_ref`(목록 밖 번호), `dup_ref`(같은 번호 두 번), `bad_value`(온톨로지 밖 값),
  `missing_ref`(번호 하나 빠뜨림), `unavailable`(장애), `bad_request`(설정 오류),
  `context_overflow`(문맥 초과).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from dlp_aireview.backend import BackendInfo, JudgeReply, JudgeRequest
from dlp_aireview.prompt import PROBLEM_KO
from dlp_aireview.schema import SPATIAL_FIELDS, TEMPORAL_FIELDS
from dlp_models.vlm import VlmContextError, VlmRequestError, VlmUnavailableError
from dlp_schema.aireview import AiGroup, AiProblem
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    ObjectStatePayload,
    SegmentPayload,
)

P = AiProblem
FAULTS = (
    "not_json", "bad_ref", "dup_ref", "bad_value", "missing_ref", "unavailable", "bad_request",
    "context_overflow",
)  # fmt: skip
Diff = tuple[list[AiProblem], dict[str, Any]]


def _index(entities: Sequence[tuple[str, str | None]], eid: str | None) -> int | None:
    """개체 ID → 개체표 번호 (없으면 None)."""
    return next((k for k, (e, _) in enumerate(entities) if e == eid), None)


def _action_diff(
    t: ActionPayload, g: ActionPayload, entities: Sequence[tuple[str, str | None]]
) -> Diff:
    """행동: 동사·대상·도구는 원래 값 제안, 손은 지적만."""
    problems: list[AiProblem] = []
    s: dict[str, Any] = {}
    if t.verb != g.verb:
        problems.append(P.WRONG_VERB)
        s["verb"] = g.verb
    for field, key, problem in (
        ("target_id", "target", P.WRONG_TARGET),
        ("tool_id", "tool", P.WRONG_TOOL),
    ):
        if getattr(t, field) != getattr(g, field):
            problems.append(problem)
            idx = _index(entities, getattr(g, field))
            if idx is not None:
                s[key] = idx
    if t.hand != g.hand:
        problems.append(P.WRONG_HAND)
    return problems, s


def _hand_diff(
    t: HandStatePayload, g: HandStatePayload, entities: Sequence[tuple[str, str | None]]
) -> Diff:
    """손 상태: 접촉 종류·대상·파지·신체 부위."""
    problems: list[AiProblem] = []
    s: dict[str, Any] = {}
    if t.contact_target_kind != g.contact_target_kind:
        problems.append(P.WRONG_CONTACT_KIND)
        s["contact_kind"] = g.contact_target_kind
    if t.target_id != g.target_id:
        problems.append(P.WRONG_CONTACT_TARGET)
        idx = _index(entities, g.target_id)
        if idx is not None:
            s["target"] = idx
    if t.grasp_type != g.grasp_type:
        problems.append(P.WRONG_GRASP)
        if g.grasp_type is not None:
            s["grasp"] = g.grasp_type
    if t.body_part != g.body_part and g.body_part is not None:
        s["body_part"] = g.body_part
    return problems, s


def _simple_diff(t: Any, g: Any) -> Diff:
    """값 하나짜리 종류: (문제, 제안 키, 필드) 표대로 비교."""
    table: dict[type, list[tuple[AiProblem, str, str]]] = {
        GapPayload: [(P.WRONG_GAP_TYPE, "gap_type", "gap_type")],
        EventPayload: [
            (P.WRONG_EVENT_TYPE, "event_type", "event_type"),
            (P.WRONG_SEVERITY, "severity", "severity"),
        ],
        SegmentPayload: [(P.WRONG_REF, "ref_id", "ref_id")],
        ObjectStatePayload: [(P.WRONG_STATE_VALUE, "value", "value")],
        BoxTrackPayload: [(P.WRONG_CLASS, "class_id", "class_id")],
        KeypointTrackPayload: [(P.HAND_SIDE_SWAPPED, "hand", "hand")],
    }  # fmt: skip
    problems: list[AiProblem] = []
    s: dict[str, Any] = {}
    kind: type = type(t)  # pyright: ignore[reportUnknownVariableType]
    rows = table.get(kind, [])
    for problem, key, field in rows:
        if getattr(t, field) != getattr(g, field):
            problems.append(problem)
            value = getattr(g, field)
            s[key] = value.value if hasattr(value, "value") else value
    return problems, s


def _kind_change(
    t: LabelRecord, g: LabelRecord, entities: Sequence[tuple[str, str | None]]
) -> Diff:
    """action↔gap: 종류 변경 제안 (원래 종류의 값)."""
    gp = g.payload
    if isinstance(gp, GapPayload):
        return [P.SHOULD_BE_GAP], {"kind": "gap", "gap_type": gp.gap_type}
    assert isinstance(gp, ActionPayload)
    s: dict[str, Any] = {"kind": "action", "verb": gp.verb}
    for key, eid in (("target", gp.target_id), ("tool", gp.tool_id)):
        idx = _index(entities, eid)
        if idx is not None:
            s[key] = idx
    return [P.SHOULD_BE_ACTION], s


def _diff(t: LabelRecord, g: LabelRecord, entities: Sequence[tuple[str, str | None]]) -> Diff:
    """대상과 정답의 차이 → (문제 유형, 제안 키)."""
    tp, gp = t.payload, g.payload
    if t.kind != g.kind:
        if {t.kind, g.kind} == {"action", "gap"}:
            return _kind_change(t, g, entities)
        return [P.OTHER], {}
    if isinstance(tp, ActionPayload) and isinstance(gp, ActionPayload):
        problems, s = _action_diff(tp, gp, entities)
    elif isinstance(tp, HandStatePayload) and isinstance(gp, HandStatePayload):
        problems, s = _hand_diff(tp, gp, entities)
    else:
        problems, s = _simple_diff(tp, gp)
    if isinstance(tp, BoxTrackPayload) and isinstance(gp, BoxTrackPayload):
        if tp.keyframes != gp.keyframes:
            problems.append(P.BOX_MISALIGNED)
    elif (
        isinstance(tp, KeypointTrackPayload)
        and isinstance(gp, KeypointTrackPayload)
        and tp.keyframes != gp.keyframes
    ):
        problems.append(P.KEYPOINT_MISPLACED)
    return (problems or [P.OTHER]), s


class OracleJudge:
    """정답 라벨로 답하는 판정 stub (모듈 docstring).

    Attributes:
        info: 백엔드 정보 (kind "oracle").
        calls: judge 호출 수.
        requests: 받은 요청 (테스트에서 프롬프트·대상 확인).
    """

    def __init__(
        self,
        truth: Sequence[LabelRecord],
        faults: Sequence[str] = (),
        *,
        model: str = "oracle-judge",
    ) -> None:
        """정답 라벨과 고장 목록.

        Raises:
            ValueError: 모르는 고장 이름.
        """
        unknown = set(faults) - set(FAULTS)
        if unknown:
            raise ValueError(f"모르는 고장: {sorted(unknown)}")
        self.truth = {x.label_id: x for x in truth}
        self.faults = list(faults)
        self.model = model
        self.calls = 0
        self.requests: list[JudgeRequest] = []
        self.info = BackendInfo(
            kind="oracle", model=model, fingerprint="oracle", endpoints=("oracle",)
        )

    def _item(self, i: int, t: LabelRecord, request: JudgeRequest) -> dict[str, Any] | None:
        """대상 하나의 응답 항목 (맞으면 None)."""
        truth = self.truth.get(t.label_id)
        if truth is not None and truth.payload == t.payload:
            return None
        if truth is None:
            problem = P.FALSE_EVENT if isinstance(t.payload, EventPayload) else P.FALSE_POSITIVE
            deletable = isinstance(t.payload, EventPayload | BoxTrackPayload | KeypointTrackPayload)
            problems, s = [problem], ({"delete": True} if deletable else {})
            if not deletable:
                problems = [P.OTHER]
        else:
            problems, s = _diff(t, truth, request.entities)
        keys = SPATIAL_FIELDS if request.group is AiGroup.SPATIAL else TEMPORAL_FIELDS
        full = {k: s.get(k, False if k == "delete" else None) for k in keys} if s else None
        return {
            "i": i, "v": "wrong", "p": [p.value for p in problems[:3]],
            "r": PROBLEM_KO[problems[0]], "c": 0.9,
            "f": request.frame_nos[0] if request.frame_nos else None, "s": full,
        }  # fmt: skip

    def answer(self, request: JudgeRequest) -> dict[str, Any]:
        """고장 없는 정답 응답 사전."""
        ok: list[int] = []
        items: list[dict[str, Any]] = []
        for i, t in enumerate(request.targets):
            item = self._item(i, t, request)
            if item is None:
                ok.append(i)
            else:
                items.append(item)
        return {"ok": ok, "items": items, "missing": []}

    def _faulty(self, fault: str, data: dict[str, Any], n: int) -> str:
        """고장 응답 원문."""
        if fault == "not_json":
            return "이 구간은 판정할 수 없습니다"
        if fault == "bad_ref":
            data["ok"] = [*data["ok"], n + 5]
        elif fault == "dup_ref" and data["ok"]:
            data["items"] = [
                *data["items"],
                {
                    "i": data["ok"][0],
                    "v": "unsure",
                    "p": ["other"],
                    "r": "모름",
                    "c": None,
                    "f": None,
                    "s": None,
                },
            ]
        elif fault == "bad_value":
            data["items"] = [
                {
                    "i": 0,
                    "v": "wrong",
                    "p": ["dance_error"],
                    "r": "춤춘다",
                    "c": 0.5,
                    "f": 0,
                    "s": None,
                },
                *(x for x in data["items"] if x["i"] != 0),
            ]
            data["ok"] = [i for i in data["ok"] if i != 0]
        elif fault == "missing_ref":
            data["ok"] = [i for i in data["ok"] if i != 0]
            data["items"] = [x for x in data["items"] if x["i"] != 0]
        return json.dumps(data, ensure_ascii=False)

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """정답 응답 (고장이 남아 있으면 먼저 그 고장).

        Raises:
            VlmUnavailableError, VlmRequestError, VlmContextError: 해당 고장일 때.
        """
        self.calls += 1
        self.requests.append(request)
        data = self.answer(request)
        content = json.dumps(data, ensure_ascii=False)
        if self.faults:
            fault = self.faults.pop(0)
            if fault == "unavailable":
                raise VlmUnavailableError("oracle: 장애")
            if fault == "bad_request":
                raise VlmRequestError("oracle: 설정 오류", status=401)
            if fault == "context_overflow":
                raise VlmContextError("oracle: maximum context length exceeded", status=400)
            content = self._faulty(fault, data, len(request.targets))
        return JudgeReply(content=content, served_model=self.model, endpoint="oracle")
