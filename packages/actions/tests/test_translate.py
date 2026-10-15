"""한국어 설명 → 영어 번역 단계 테스트 (ADR 0043). 서비스 없이 시간대 보존 SQLite로 돈다.

정답 근거: 직접 만든 한국어 설명 라벨(행동 2개의 설명)과 정답 번역을 아는 stub 번역기
(`GlossaryStubTranslator`). 서버 어댑터는 httpx `MockTransport`의 가짜 서버로 요청 본문·재시도·
장애를 본다. 확인하는 규칙: 원문·번역이 둘 다 현재 운영 라벨, 멱등, 원문 수정 → 검수 전 번역만
교체, 검수된 번역은 남김, 번역기 버전 교체, 서버 장애 → 대기 목록(예외 없음), 원문 삭제 → 번역
삭제.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_actions.translate import (
    glossary_entries,
    glossary_for,
    is_machine_translation,
    plan_translation,
    run_translate,
    translation_id,
    translation_version,
    untranslated,
)
from dlp_actions.translate_policy import TranslatePolicy, load_policy
from dlp_actions.translators import (
    GlossaryEntry,
    GlossaryStubTranslator,
    OpenAICompatibleTranslator,
    TranslationInvalidError,
    TranslationUnavailableError,
    check_reply,
)
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels, retractions, version_tag
from dlp_schema.label_validation import validate_session_labels
from dlp_schema.labels import (
    DescriptionPayload,
    LabelRecord,
    Provenance,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

ROOT = Path(__file__).resolve().parents[3]
SID = "s001"
# 행동 설명 원문 (행동 단계가 낸 모델 설명을 흉내 낸다)
KO = {"s001-a0": "걸레로 바닥을 닦는다", "s001-a1": "양동이를 들어 올린다"}
EN = {
    "걸레로 바닥을 닦는다": "Wipes the floor with a cloth",
    "양동이를 들어 올린다": "Lifts the bucket",
}
ACTIONS_VERSION = "actions-abc+oracle-vlm-1+i00000000"


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    """온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


@pytest.fixture(scope="module")
def policy() -> TranslatePolicy:
    """저장소의 translate.yaml."""
    return load_policy(ROOT)


def _ko(action_id: str, text: str, **overrides: Any) -> LabelRecord:
    """행동 단계가 낸 것 같은 한국어 설명 (모델 출처, 신뢰도 0.9)."""
    data: dict[str, Any] = {
        "provenance": Provenance(source=Source.MODEL, model_version=ACTIONS_VERSION),
        "confidence": 0.9,
    }
    data.update(overrides)
    return make_label(
        {"kind": "description", "segment_id": action_id, "text": text},
        label_id=f"{action_id}-desc", session_id=SID, t_start_ms=100, t_end_ms=900, **data,
    )  # fmt: skip


@pytest.fixture
def db(ontology: Ontology) -> Iterator[sa.Engine]:
    """온톨로지 v1·세션 s001과 한국어 설명 2개를 넣은 메모리 DB."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session(SID))
        insert_labels(conn, [_ko(a, t) for a, t in KO.items()])
    yield engine
    engine.dispose()


def _current(engine: sa.Engine) -> list[LabelRecord]:
    with engine.connect() as conn:
        return current_labels(get_labels(conn, SID, kinds=["description"]))


def _english(labels: list[LabelRecord]) -> dict[str, LabelRecord]:
    """원문 라벨 ID → 현재 영어 번역."""
    out: dict[str, LabelRecord] = {}
    for x in labels:
        p = x.payload
        if isinstance(p, DescriptionPayload) and p.language == "en" and p.source_label_id:
            out[p.source_label_id] = x
    return out


def _run(engine: sa.Engine, ontology: Ontology, policy: TranslatePolicy, translator: Any) -> Any:
    with engine.begin() as conn:
        return run_translate(conn, SID, translator, ontology, policy, FIXED_TIME)


def test_each_korean_description_gets_an_english_sibling(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """원문마다 영어 형제 레코드가 생기고, 원문·번역이 둘 다 현재 운영 라벨이다.

    정답: 번역 2개. 같은 segment_id·구간, language en, source_label_id = 원문 ID, parent 없음,
    ID에 version_tag(번역 버전), 버전은 translate-<정책 해시>로 시작, 신뢰도는 원문 값.
    """
    stub = GlossaryStubTranslator(EN)
    s = _run(db, ontology, policy, stub)
    assert (s.translated, s.retracted, s.pending) == (2, 0, [])
    assert s.version.startswith(f"translate-{policy.digest}+stub-glossary-1+g")
    current = _current(db)
    assert len(current) == 4  # 원문 2 + 번역 2 (원문이 현재 라벨에서 빠지지 않는다)
    english = _english(current)
    for action_id, text in KO.items():
        en = english[f"{action_id}-desc"]
        p = en.payload
        assert isinstance(p, DescriptionPayload)
        assert (p.segment_id, p.text, p.language) == (action_id, EN[text], "en")
        assert en.parent_label_id is None
        assert version_tag(s.version) in en.label_id
        assert (en.t_start_ms, en.t_end_ms, en.confidence) == (100, 900, 0.9)
        assert en.provenance.model_version == s.version and is_machine_translation(en)


def test_rerun_is_idempotent(db: sa.Engine, ontology: Ontology, policy: TranslatePolicy) -> None:
    """같은 버전으로 다시 돌리면 아무것도 쓰지 않고 번역기도 부르지 않는다."""
    _run(db, ontology, policy, GlossaryStubTranslator(EN))
    stub = GlossaryStubTranslator(EN)
    again = _run(db, ontology, policy, stub)
    assert (again.translated, again.retracted, again.covered, stub.calls) == (0, 0, 2, 0)


def _approved_action(engine: sa.Engine, action_id: str, reviewed_at: datetime) -> None:
    """설명 원문이 가리키는 행동(모델)을 넣고 검수자 r1이 승인한다 (F-PAR-15)."""
    action = make_label(
        action_payload(action_id=action_id), label_id=f"{action_id}-act", session_id=SID,
        provenance=Provenance(source=Source.MODEL, model_version=ACTIONS_VERSION), confidence=0.9,
    )  # fmt: skip
    with engine.begin() as conn:
        insert_labels(conn, [action])
        record_review(conn, action.label_id, VerificationState.HUMAN_APPROVED, "r1", reviewed_at)


def test_translation_of_approved_action_inherits_review(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """행동을 승인한 뒤에 만든 번역은 그 승인을 물려받는다 (F-PAR-15, ADR 0043 3절).

    정답: s001-a0 행동만 r1이 FIXED_TIME에 승인. 번역 시각 later에 원문 s001-a0-desc와 그 번역이
    human_approved(검수자 r1). 원문은 행동 검수 시각, 번역은 자기 생성 시각(검수가 생성보다 앞서지
    않는다). 행동이 없는 s001-a1의 원문·번역은 미검수 그대로.
    """
    later = FIXED_TIME + timedelta(hours=2)
    _approved_action(db, "s001-a0", FIXED_TIME)
    with db.begin() as conn:
        s = run_translate(conn, SID, GlossaryStubTranslator(EN), ontology, policy, later)
    assert (s.translated, s.inherited) == (2, 2)
    current = {x.label_id: x for x in _current(db)}
    english = _english(list(current.values()))
    source = current["s001-a0-desc"].verification
    assert (source.state, source.reviewer_id, source.reviewed_at) == (
        VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME,
    )  # fmt: skip
    en = english["s001-a0-desc"].verification
    assert (en.state, en.reviewer_id, en.reviewed_at) == (
        VerificationState.HUMAN_APPROVED, "r1", later,
    )  # fmt: skip
    assert current["s001-a1-desc"].verification.state is VerificationState.UNREVIEWED
    assert english["s001-a1-desc"].verification.state is VerificationState.UNREVIEWED


def test_corrected_source_replaces_only_unreviewed_translation(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """원문이 고쳐지면(새 레코드 + parent) 검수 전 번역은 지우고 새 원문을 번역한다.

    검수자가 승인한 번역의 원문이 고쳐지면 그 번역은 남기고 새 번역을 만들지 않는다
    (kept_reviewed로 알린다).
    """
    first = _run(db, ontology, policy, GlossaryStubTranslator(EN))
    english = _english(_current(db))
    approved = english["s001-a1-desc"].label_id
    with db.begin() as conn:
        record_review(conn, approved, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
        fixed = [
            make_label(
                {"kind": "description", "segment_id": a, "text": f"{KO[a]} (수정)"},
                label_id=f"{a}-desc-fix",
                session_id=SID,
                t_start_ms=100,
                t_end_ms=900,
                parent_label_id=f"{a}-desc",
            )
            for a in KO
        ]
        insert_labels(conn, fixed)
    s = _run(db, ontology, policy, GlossaryStubTranslator())
    assert s.retracted == 1 and s.translated == 1
    assert s.kept_reviewed == ["s001-a1-desc-fix"]
    current = _current(db)
    english = _english(current)
    assert set(english) == {"s001-a0-desc-fix", "s001-a1-desc"}
    assert english["s001-a1-desc"].label_id == approved  # 검수된 번역은 남는다
    new = english["s001-a0-desc-fix"]
    assert isinstance(new.payload, DescriptionPayload)
    assert new.payload.text.startswith("[stub-en] ")
    # 사람이 고친 원문은 신뢰도가 없어 정책 기본값을 쓴다
    assert new.confidence == policy.default_confidence
    assert first.version == s.version


def test_new_translator_version_replaces_only_unreviewed(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """번역기 버전이 바뀌면 검수 전 이전 번역만 지우고 새 버전으로 만든다. 검수된 번역은 남는다."""
    old = _run(db, ontology, policy, GlossaryStubTranslator(EN))
    kept = _english(_current(db))["s001-a1-desc"].label_id
    with db.begin() as conn:
        record_review(conn, kept, VerificationState.SAMPLE_VERIFIED, "r1", FIXED_TIME)
    s = _run(db, ontology, policy, GlossaryStubTranslator(EN, version="stub-glossary-2"))
    assert s.version != old.version
    assert (s.translated, s.retracted, s.covered) == (1, 1, 1)
    english = _english(_current(db))
    assert english["s001-a1-desc"].label_id == kept
    assert english["s001-a0-desc"].provenance.model_version == s.version
    assert len(_current(db)) == 4


def test_server_outage_leaves_pending_without_raising(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """서버 장애면 예외 없이 남은 원문을 대기 목록에 두고, 다음 실행이 채운다.

    정답: 첫 원문(a0)에서 장애 → a0·a1 모두 대기, 번역 0, stopped=unavailable. 응답 위반은 그 원문만
    대기로 남는다.
    """
    down = GlossaryStubTranslator(EN, fail_unavailable=list(KO.values()))
    s = _run(db, ontology, policy, down)
    assert (s.translated, s.stopped, down.calls) == (0, "unavailable", 1)
    assert s.pending == ["s001-a0-desc", "s001-a1-desc"]
    with db.connect() as conn:
        history = get_labels(conn, SID, kinds=["description"])
    version = translation_version(policy, down.version, ontology)
    assert [x.label_id for x in untranslated(history, policy, version)] == s.pending
    bad = GlossaryStubTranslator(EN, fail_invalid=[KO["s001-a1"]])
    s2 = _run(db, ontology, policy, bad)
    assert (s2.translated, s2.invalid, s2.pending, s2.stopped) == (
        1, ["s001-a1-desc"], ["s001-a1-desc"], None
    )  # fmt: skip
    s3 = _run(db, ontology, policy, GlossaryStubTranslator(EN))
    assert (s3.translated, s3.pending) == (1, [])


def test_retracted_source_drops_its_unreviewed_translation(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """원문이 지워지면(삭제 레코드) 검수 전 번역도 지운다. 다시 돌리면 아무것도 하지 않는다."""
    s = _run(db, ontology, policy, GlossaryStubTranslator(EN))
    with db.begin() as conn:
        source = next(x for x in get_labels(conn, SID) if x.label_id == "s001-a0-desc")
        insert_labels(conn, retractions([source], "actions-next", FIXED_TIME))
    s2 = _run(db, ontology, policy, GlossaryStubTranslator(EN))
    assert (s2.retracted, s2.translated) == (1, 0)
    assert set(_english(_current(db))) == {"s001-a1-desc"}
    assert s2.version == s.version


def test_reviewer_deleted_translation_is_not_revived(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """검수자가 지운 번역은 같은 버전으로 다시 돌려도 되살리지 않는다 (이력으로 멱등)."""
    _run(db, ontology, policy, GlossaryStubTranslator(EN))
    with db.begin() as conn:
        en = _english(_current(db))["s001-a0-desc"]
        human = retractions([en], "x", FIXED_TIME + timedelta(seconds=1))[0]
        insert_labels(
            conn,
            [human.model_copy(update={"provenance": Provenance(source=Source.HUMAN)})],
        )
    stub = GlossaryStubTranslator(EN)
    s = _run(db, ontology, policy, stub)
    assert (s.translated, s.deleted, stub.calls) == (0, 1, 0)


def test_session_validation_accepts_ko_and_en_for_one_segment(ontology: Ontology) -> None:
    """같은 구간에 한국어 설명과 영어 번역이 함께 있어도 저장 검증·현재 라벨 판정이 받는다."""
    ko = _ko("s001-a0", KO["s001-a0"])
    en = make_label(
        {"kind": "description", "segment_id": "s001-a0", "text": "Wipes the floor",
         "language": "en", "source_label_id": ko.label_id},
        label_id="s001-a0-desc:en00000000", session_id=SID, t_start_ms=100, t_end_ms=900,
        provenance=Provenance(source=Source.MODEL, model_version="translate-x"), confidence=0.9,
    )  # fmt: skip
    validate_session_labels(ontology, [], [ko, en])
    assert {x.label_id for x in current_labels([ko, en])} == {ko.label_id, en.label_id}


def test_glossary_comes_from_ontology_names(ontology: Ontology, policy: TranslatePolicy) -> None:
    """용어집은 온톨로지 ko → en 이름이고, matched는 원문에 나온 이름만 넣는다."""
    entries = glossary_entries(ontology)
    assert GlossaryEntry("걸레", "Cleaning cloth") in entries
    assert GlossaryEntry("양동이", "Bucket") in entries
    assert len({e.ko for e in entries}) == len(entries)
    picked = glossary_for("걸레로 바닥을 닦는다", entries, "matched")
    assert {e.ko for e in picked} >= {"걸레", "바닥"}
    assert all(e.ko in "걸레로 바닥을 닦는다" for e in picked)
    assert glossary_for("걸레", entries, "off") == []
    assert len(glossary_for("걸레", entries, "all")) == len(entries)
    stub = GlossaryStubTranslator()
    assert (
        stub.translate("양동이", [*picked, GlossaryEntry("양동이", "Bucket")]) == "[stub-en] Bucket"
    )
    # 용어집 해시가 버전에 들어간다 (끄면 고정값)
    off = policy.model_copy(
        update={"backend": policy.backend.model_copy(update={"glossary": "off"})}
    )
    assert translation_version(policy, stub.version, ontology) != translation_version(
        off, stub.version, ontology
    )


def test_plan_ids_are_deterministic(policy: TranslatePolicy) -> None:
    """번역 ID는 원문 ID·버전으로 정해지고, 버전이 다르면 다르다. 긴 ID는 128자 안으로 줄인다."""
    ko = _ko("s001-a0", "닦는다")
    a, b = translation_id(ko, policy, "v1"), translation_id(ko, policy, "v2")
    assert a == f"{ko.label_id}:en{version_tag('v1')}" and a != b
    long = ko.model_copy(update={"label_id": "x" * 125})
    assert len(translation_id(long, policy, "v1")) <= 128
    plan = plan_translation([ko], policy, "v1")
    assert [s.label_id for s, _ in plan.todo] == [ko.label_id] and plan.stale == []


def test_policy_digest_and_prompt_version(policy: TranslatePolicy) -> None:
    """server 절은 해시에 들어가지 않고, 프롬프트 틀 버전이 코드와 다르면 적재 오류다."""
    data = policy.model_dump(mode="json")
    data["server"]["timeout_s"] = 5
    assert TranslatePolicy.model_validate(data).digest == policy.digest
    data["backend"]["max_chars"] = 10
    assert TranslatePolicy.model_validate(data).digest != policy.digest
    data["prompt_version"] = 99
    with pytest.raises(ValidationError):
        TranslatePolicy.model_validate(data)
    data["prompt_version"] = policy.prompt_version
    data["target_language"] = "ko"
    with pytest.raises(ValidationError):
        TranslatePolicy.model_validate(data)


# ---------------------------------------------------------------- 서버 어댑터 (가짜 서버)


def _reply(content: str, model: str) -> dict[str, Any]:
    return {"model": model, "choices": [{"message": {"content": content}}]}


def test_server_adapter_sends_text_only_and_retries_violations(policy: TranslatePolicy) -> None:
    """요청은 글만(이미지 없음), 모델·용어집·원문·응답 스키마를 싣는다. 한글이 남은 응답은 위반을
    알려 다시 묻는다."""
    bodies: list[dict[str, Any]] = []
    model = policy.backend.model
    replies = [json.dumps({"text": "걸레 wipes"}), json.dumps({"text": "Wipes with a cloth"})]

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json=_reply(replies[len(bodies) - 1], model))

    t = OpenAICompatibleTranslator(
        "http://10.0.0.11:8000", policy, api_key="k", transport=httpx.MockTransport(handler)
    )
    out = t.translate("걸레로 닦는다", [GlossaryEntry("걸레", "Cleaning cloth")])
    assert out == "Wipes with a cloth" and len(bodies) == 2
    first = bodies[0]
    assert first["model"] == model and first["temperature"] == 0
    user = first["messages"][1]["content"]
    assert [part["type"] for part in user] == ["text"]
    assert "걸레 => Cleaning cloth" in user[0]["text"] and "걸레로 닦는다" in user[0]["text"]
    assert first["response_format"]["json_schema"]["name"] == "translation"
    retry = bodies[1]["messages"]
    assert retry[2]["role"] == "assistant" and "Korean" in retry[3]["content"][0]["text"]
    assert t.version == f"llm-{model}"


def test_server_adapter_outage_and_persistent_violation(policy: TranslatePolicy) -> None:
    """5xx가 백오프 뒤에도 이어지면 TranslationUnavailableError, 계속 위반이면
    TranslationInvalidError."""
    waits: list[float] = []
    down = OpenAICompatibleTranslator(
        "http://10.0.0.11:8000", policy,
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="busy")),
        sleep=waits.append,
    )  # fmt: skip
    with pytest.raises(TranslationUnavailableError):
        down.translate("닦는다", [])
    assert waits == list(policy.server.backoff_s)
    model = policy.backend.model
    not_json = httpx.MockTransport(lambda r: httpx.Response(200, json=_reply("not json", model)))
    bad = OpenAICompatibleTranslator("http://10.0.0.11:8000", policy, transport=not_json)
    with pytest.raises(TranslationInvalidError):
        bad.translate("닦는다", [])


def test_check_reply_rules() -> None:
    """응답 검사: JSON·비지 않음·길이·한글."""
    assert check_reply('{"text": " Wipes "}', 10, True) == ("Wipes", [])
    assert check_reply("x", 10, True)[0] is None
    assert check_reply('{"text": ""}', 10, True)[0] is None
    assert check_reply('{"text": "aaaaaaaaaaaa"}', 10, True)[0] is None
    assert check_reply('{"text": "닦다"}', 10, True)[0] is None
    assert check_reply('{"text": "닦다"}', 10, False) == ("닦다", [])


def test_description_added_during_translation_stays_pending(
    db: sa.Engine, ontology: Ontology, policy: TranslatePolicy
) -> None:
    """F-PAR-17: 번역하는 동안(잠금 전) 한국어 설명이 새로 커밋되면 이번 실행은 그 원문을 번역하지
    않았으므로 요약의 대기 목록에 들어야 한다 (전에는 pending=[]라 CLI가 0으로 끝났다).

    정답: 처음 계획의 원문 2개는 번역되고, 번역 중 들어온 a2 설명은 대기다 (CLI 종료 코드는
    `summary.pending`으로 정한다). 다음 실행이 a2를 번역하고 대기가 없다.
    """
    late = _ko("s001-a2", "걸레를 짠다")

    class Racing(GlossaryStubTranslator):
        """첫 번역 요청 때 다른 실행(행동 단계)이 새 설명을 커밋한 것처럼 넣는다."""

        conn: sa.Connection | None = None

        def translate(self, text: str, glossary: Any) -> str:
            if self.calls == 0 and self.conn is not None:
                insert_labels(self.conn, [late])
            return super().translate(text, glossary)

    stub = Racing({**EN, "걸레를 짠다": "Wrings the cloth"})
    with db.begin() as conn:
        stub.conn = conn
        s = run_translate(conn, SID, stub, ontology, policy, FIXED_TIME)
    assert s.translated == 2 and s.pending == [late.label_id]
    again = _run(db, ontology, policy, GlossaryStubTranslator({"걸레를 짠다": "Wrings the cloth"}))
    assert (again.translated, again.pending) == (1, [])
