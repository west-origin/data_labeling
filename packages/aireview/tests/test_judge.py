"""단위 판정 경로 테스트 (`dlp_aireview.judge`, 정해 둔 응답을 내는 백엔드).

- 문맥 초과로 프레임을 줄인 요청: 이미지 모서리 도장·프레임 지도·응답 f가 모두 단위 프레임 번호이고
  (F0, F2, …), 근거 시각이 그 프레임 시각이다. 단위 기록의 frames·prompt_digest는 실제로 보낸 요청의
  것이다.
- 재시도 뒤 위반이 가장 적은 응답을 받는다.
- 단위 기록의 오류 문장에 모델이 쓴 값·서버 응답 본문이 없다.
- 누락 후보를 켜도 이상한 후보가 단위 판정을 멈추지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from dlp_aireview.backend import BackendInfo, JudgeReply, JudgeRequest
from dlp_aireview.judge import JudgedUnit, judge_unit, request_digest
from dlp_aireview.plan import PlannedUnit, plan_session
from dlp_aireview.policy import AiReviewPolicy
from dlp_aireview.prompt import PromptImage, build_parts
from dlp_aireview.schema import TEMPORAL_FIELDS
from dlp_aireview.sensors import SensorContext
from dlp_models.vlm import VlmContextError
from dlp_schema.aireview import AiGroup, AiVerdictValue
from dlp_schema.ontology import Ontology
from dlp_schema.testing import FIXED_TIME

from .conftest import Scene, images_for, with_policy

ABSENT = SensorContext.absent()
RUN = "air-run-" + "2" * 24


class ScriptedJudge:
    """정해 둔 응답(문자열) 또는 예외를 차례로 내는 판정 백엔드. 받은 요청을 남긴다."""

    def __init__(self, script: Sequence[str | Exception]) -> None:
        """응답 목록 (다 쓰면 마지막 것을 되풀이)."""
        self.script = list(script)
        self.requests: list[JudgeRequest] = []
        self.info = BackendInfo(kind="scripted", model="m", fingerprint="f", endpoints=("s",))

    def judge(self, request: JudgeRequest) -> JudgeReply:
        """다음 응답 (예외면 올린다)."""
        self.requests.append(request)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return JudgeReply(content=step, served_model="m", endpoint="s")


def _temporal_unit(scene: Scene, policy: AiReviewPolicy, o: Ontology) -> PlannedUnit:
    """프레임이 4장 이상인 첫 시간 단위."""
    units = plan_session(scene.session, scene.labels, policy, o, media=scene.media, sensor=ABSENT)
    return next(u for u in units if u.group is AiGroup.TEMPORAL and len(u.frame_ms) >= 4)


def _answer(n: int, items: Sequence[dict[str, Any]] = (), missing: Sequence[Any] = ()) -> str:
    """대상 n개 중 items에 없는 번호는 ok인 응답 원문."""
    flagged = {it["i"] for it in items}
    ok = [i for i in range(n) if i not in flagged]
    return json.dumps(
        {"ok": ok, "items": list(items), "missing": list(missing)}, ensure_ascii=False
    )


def _wrong(i: int, f: int | None, r: str = "동작이 다르다", **kw: Any) -> dict[str, Any]:
    """판정 항목 하나 (wrong, 문제 other, 제안 없음)."""
    return {"i": i, "v": "wrong", "p": ["other"], "r": r, "c": 0.7, "f": f, "s": None} | kw


def _judge(
    backend: ScriptedJudge,
    unit: PlannedUnit,
    policy: AiReviewPolicy,
    o: Ontology,
    images: tuple[PromptImage, ...] | None = None,
) -> JudgedUnit:
    """단위 하나를 판정한다."""
    imgs = images if images is not None else images_for(unit, policy)
    return judge_unit(backend, unit, imgs, ontology=o, policy=policy, sensor=ABSENT)


def test_halved_frames_keep_unit_numbers_everywhere(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """문맥 초과로 프레임을 절반으로 줄인 요청: 보낸 이미지의 번호(도장 번호)와 프레임 지도의 번호가
    같은 단위 번호(F0, F2, …)이고, 응답 f=2는 단위 프레임 2의 시각이 된다. 기록에는 실제로 보낸
    이미지와 그 요청의 다이제스트만 남는다."""
    unit = _temporal_unit(scene, policy, ontology)
    n = len(unit.targets)
    images = images_for(unit, policy)
    backend = ScriptedJudge(
        [VlmContextError("maximum context length", status=400), _answer(n, [_wrong(0, 2)])]
    )
    judged = _judge(backend, unit, policy, ontology, images)
    sent = backend.requests[-1]
    kept = tuple(range(0, len(unit.frame_ms), 2))
    assert sent.frame_nos == kept
    assert sent.frame_ms == tuple(unit.frame_ms[k] for k in kept)
    sent_images = tuple(im for im in sent.parts.images if not im.crop)
    assert tuple(im.frame_no for im in sent_images) == kept  # 도장 F#는 frame_no로 그렸다
    frame_map = next(line for line in sent.parts.text.splitlines() if line.startswith("프레임:"))
    assert [p.split()[0] for p in frame_map.removeprefix("프레임: ").split(", ")] == [
        f"F{k}" for k in kept
    ]
    assert judged.verdicts[0].verdict is AiVerdictValue.WRONG
    assert judged.verdicts[0].evidence_t_ms == unit.frame_ms[2]
    assert [f.i for f in judged.frames] == list(kept)
    assert {f.sha256 for f in judged.frames} == {im.image.sha256 for im in sent_images}
    halved = tuple(im for im in images if im.frame_no in kept)
    expected = build_parts(unit, halved, range(n), ontology, policy, ABSENT).digest
    assert judged.prompt_digest == expected
    full = build_parts(unit, images, range(n), ontology, policy, ABSENT).digest
    assert judged.prompt_digest != full
    record = judged.record(RUN, FIXED_TIME)
    assert len(record.frames) == len(kept) and record.sub_requests == 1


def test_split_targets_digest_covers_each_request(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """대상까지 나눈 단위의 다이제스트는 요청별 다이제스트 목록의 해시다."""
    small = with_policy(policy, temporal={"max_targets": 2})
    unit = next(
        u
        for u in plan_session(
            scene.session, scene.labels, small, ontology, media=scene.media, sensor=ABSENT
        )
        if u.group is AiGroup.TEMPORAL and len(u.targets) == 2
    )
    overflow = VlmContextError("context length", status=400)
    backend = ScriptedJudge([overflow, overflow, _answer(1)])
    judged = _judge(backend, unit, small, ontology)
    assert judged.sub_requests == 2
    digests = [request_digest(r.parts, r.history) for r in backend.requests[-2:]]
    assert judged.prompt_digest == hashlib.sha256(json.dumps(digests).encode()).hexdigest()


def test_retry_keeps_the_answer_with_fewest_violations(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """위반이 많아 다시 물었는데 재시도 응답이 더 나쁘면 첫 응답을 받는다. 다이제스트·원문 해시는
    받은 응답을 낸 요청의 것이다."""
    unit = _temporal_unit(scene, policy, ontology)
    n = len(unit.targets)
    assert n >= 4
    # 첫 응답: 대상 절반 넘게 빠짐 (재시도 조건), 둘째 응답: 전부 빠짐 + 목록 밖 번호
    first = json.dumps({"ok": [0], "items": [], "missing": []})
    worse = json.dumps({"ok": [n + 3], "items": [], "missing": []})
    retried = _judge(ScriptedJudge([first, worse]), unit, policy, ontology)
    assert retried.attempts == 2
    assert retried.verdicts[0].verdict is AiVerdictValue.OK and retried.verdicts[0].answered
    assert all(not v.answered for v in retried.verdicts[1:])
    assert retried.response_sha256 == hashlib.sha256(first.encode()).hexdigest()
    parts = build_parts(unit, images_for(unit, policy), range(n), ontology, policy, ABSENT)
    assert retried.prompt_digest == parts.digest  # 첫 요청(재시도 대화 없음)
    better = _judge(ScriptedJudge([first, _answer(n)]), unit, policy, ontology)
    assert all(v.answered for v in better.verdicts)
    assert better.prompt_digest != parts.digest  # 재시도 대화가 든 요청


def test_unit_errors_hold_no_model_values_or_server_bodies(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """단위 기록의 오류 문장(추가 전용 표)에는 모델이 쓴 값과 서버 오류 본문이 없다."""
    unit = _temporal_unit(scene, policy, ontology)
    n = len(unit.targets)
    pii = "김철수 환자 302호 010-1234-5678"
    bad = {k: None for k in TEMPORAL_FIELDS} | {"verb": [pii], "delete": False}
    content = _answer(
        n,
        [
            {"i": 0, "v": pii, "p": [], "r": pii, "c": None, "f": None, "s": None},
            _wrong(1, None, p=[pii]),
            _wrong(2, None, p=["wrong_verb"], s=bad),
        ],
    )
    overflow = VlmContextError(f"context too long: {pii}", status=400)
    judged = _judge(ScriptedJudge([overflow, content]), unit, policy, ontology)
    assert judged.errors
    joined = " ".join(judged.errors)
    assert pii not in joined and "302" not in joined
    assert any("HTTP 400" in e for e in judged.errors)


def test_findings_enabled_never_stop_the_unit(
    scene: Scene, policy: AiReviewPolicy, ontology: Ontology
) -> None:
    """누락 후보를 켜고(max_findings > 0) 이상한 후보를 내도 단위 판정이 끝나고, 받은 후보의 이유는
    안전 검사를 거친다."""
    response = policy.response.model_copy(update={"max_findings": 3})
    on = policy.model_copy(update={"response": response})
    unit = _temporal_unit(scene, on, ontology)
    missing = [
        {"event_type": "drop", "f0": 0, "f1": 1, "r": "김철수 씨 http://evil 01012345678 얼굴"},
        {"event_type": "discomfort", "f0": 0, "f1": 1, "r": "불편해 보임"},
        {"event_type": "Not An Id!", "f0": 0, "f1": 1, "r": "이상"},
    ]
    judged = _judge(
        ScriptedJudge([_answer(len(unit.targets), missing=missing)]), unit, on, ontology
    )
    assert [(f.value, f.reason_ko) for f in judged.findings] == [("drop", "AI: 빠진 이벤트")]
    assert judged.findings[0].t_start_ms == unit.frame_ms[0]
    assert judged.record(RUN, FIXED_TIME).findings == judged.findings
