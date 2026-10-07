from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_active.policy import ActivePolicy, load_policy
from dlp_active.rates import correction_rates
from dlp_active.select import score_sessions
from dlp_active.terms import TERMS, SessionContext, register_term
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.session import Session
from dlp_schema.testing import FIXED_TIME, make_session
from dlp_schema.testing import make_label as _make_label

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> ActivePolicy:
    return load_policy(ROOT)


def action(verb: str) -> dict[str, Any]:
    return {"kind": "action", "action_id": "a", "hand": "right", "verb": verb,
            "t_approach_ms": 0, "t_end_ms": 1000}  # fmt: skip


def box(cls: str) -> dict[str, Any]:
    return {"kind": "box_track", "entity_id": f"{cls}_1", "class_id": cls,
            "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]}  # fmt: skip


def blur() -> dict[str, Any]:
    return {"kind": "blur_track", "target": "face",
            "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]}  # fmt: skip


def make_label(payload: dict[str, Any], **kw: Any) -> LabelRecord:
    """공간 라벨에는 스트림을 붙인다."""
    if payload["kind"] != "action":
        kw.setdefault("stream_id", "bodycam")
    return _make_label(payload, **kw)


def model(sid: str, lid: str, payload: dict[str, Any], **kw: Any) -> LabelRecord:
    return make_label(
        payload,
        label_id=f"{sid}-{lid}",
        session_id=sid,
        provenance=Provenance(source=Source.MODEL, model_version="m1"),
        confidence=kw.pop("confidence", 0.9),
        **kw,
    )


def reviewed(state: VerificationState) -> Verification:
    return Verification(state=state, reviewer_id="r1", reviewed_at=FIXED_TIME)


def reviewed_history(
    sid: str, payload: dict[str, Any], n: int, corrected: int
) -> list[LabelRecord]:
    """모델 라벨 n개를 검수: corrected개는 사람이 고치고 나머지는 그대로 승인."""
    out: list[LabelRecord] = []
    for i in range(n):
        if i < corrected:
            out.append(model(sid, f"m{i}", payload))
            out.append(
                make_label(
                    payload,
                    label_id=f"{sid}-h{i}",
                    session_id=sid,
                    parent_label_id=f"{sid}-m{i}",
                    verification=reviewed(VerificationState.HUMAN_CORRECTED),
                )
            )
        else:
            out.append(
                model(
                    sid, f"m{i}", payload, verification=reviewed(VerificationState.HUMAN_APPROVED)
                )
            )
    return out


def pending(
    sid: str, counts: dict[str, int], payloads: dict[str, dict[str, Any]]
) -> list[LabelRecord]:
    return [
        model(sid, f"p-{name}-{i}", payloads[name]) for name, n in counts.items() for i in range(n)
    ]


PAYLOADS = {"fold": action("fold_sheet"), "cup": box("cup"), "mop": box("mop"), "blur": blur()}


def test_sessions_with_high_correction_classes_come_first(policy: ActivePolicy) -> None:
    # 합성 수정률 분포: 시트 접기 30%, 컵 0.5% (기준 문서의 예), 대걸레 10%
    histories = [
        reviewed_history("r1", PAYLOADS["fold"], 400, 120),
        reviewed_history("r2", PAYLOADS["cup"], 2000, 10),
        reviewed_history("r3", PAYLOADS["mop"], 500, 50),
    ]
    rates = correction_rates(histories, policy)
    # 평활 때문에 전체 수정률(약 6%) 쪽으로 조금 당겨진다
    assert rates.rate("action/fold_sheet") == pytest.approx(0.3, abs=0.015)
    assert rates.rate("box_track/cup") == pytest.approx(0.005, abs=0.002)

    # 후보 세션: 담은 클래스가 다르다. 예상 수정 수 = Σ 수정률 → 기대 순서
    mixes = {
        "s-cups": {"cup": 40},  # 40 * 0.005 ≈ 0.2
        "s-fold": {"fold": 5},  # 5 * 0.3 = 1.5
        "s-mixed": {"fold": 2, "mop": 4, "cup": 10},  # 0.6 + 0.4 + 0.05 ≈ 1.05
        "s-mops": {"mop": 6},  # 0.6
        "s-blur": {"blur": 50},  # 블러는 점수에 넣지 않는다 → 0
    }
    sessions: list[tuple[Session, list[LabelRecord]]] = [
        (make_session(sid), pending(sid, m, PAYLOADS)) for sid, m in mixes.items()
    ]
    ranked = score_sessions(sessions, rates, policy)
    assert [s.session_id for s in ranked] == ["s-fold", "s-mixed", "s-mops", "s-cups", "s-blur"]
    expected = {
        sid: sum(n * rates.rate(k) for k, n in [
            ("action/fold_sheet", m.get("fold", 0)),
            ("box_track/mop", m.get("mop", 0)),
            ("box_track/cup", m.get("cup", 0)),
        ])
        for sid, m in mixes.items()
    }  # fmt: skip
    for s in ranked:
        assert s.score == pytest.approx(expected[s.session_id])
    assert ranked[0].top_classes[0][0] == "action/fold_sheet"
    assert ranked[-1].pending == 0


def test_random_distribution_matches_expected_order(policy: ActivePolicy) -> None:
    """임의의 클래스 수정률·세션 구성에서도 순서가 예상 수정 수 순서와 같다."""
    rng = np.random.default_rng(7)
    classes = [f"c{i}" for i in range(8)]
    true_rates = rng.uniform(0.0, 0.5, len(classes))
    histories = [
        reviewed_history(f"r{i}", box(c), 300, round(300 * true_rates[i]))
        for i, c in enumerate(classes)
    ]
    rates = correction_rates(histories, policy)
    payloads = {c: box(c) for c in classes}
    sessions: list[tuple[Session, list[LabelRecord]]] = []
    for j in range(30):
        mix = {c: int(n) for c, n in zip(classes, rng.integers(0, 6, len(classes)), strict=True)}
        sessions.append((make_session(f"s{j:02d}"), pending(f"s{j:02d}", mix, payloads)))
    ranked = score_sessions(sessions, rates, policy)
    expected = sorted(
        sessions,
        key=lambda s: (
            -sum(rates.rate(f"box_track/{x.payload.class_id}") for x in s[1]),  # type: ignore[union-attr]
            s[0].session_id,
        ),
    )
    assert [s.session_id for s in ranked] == [s.session_id for s, _ in expected]


def test_rates_count_only_individual_reviews_and_smooth_small_classes(
    policy: ActivePolicy,
) -> None:
    sid = "r1"
    history = [
        *reviewed_history(sid, PAYLOADS["mop"], 4, 4),  # 4개 중 4개 수정 (표본 적음)
        *reviewed_history("r2", PAYLOADS["cup"], 96, 0),
        # 표본 검증은 개별 검수가 아니므로 세지 않는다
        model(sid, "sv", PAYLOADS["mop"], verification=reviewed(VerificationState.SAMPLE_VERIFIED)),
        # 사람이 지운 오탐은 수정
        model(sid, "fp", PAYLOADS["cup"]),
        make_label(PAYLOADS["cup"], label_id=f"{sid}-fp-del", session_id=sid,
                   parent_label_id=f"{sid}-fp", retracted=True,
                   verification=reviewed(VerificationState.HUMAN_CORRECTED)),
        # 블러와 오류 삽입 레코드는 세지 않는다
        make_label(PAYLOADS["blur"], label_id=f"{sid}-b", session_id=sid,
                   verification=reviewed(VerificationState.HUMAN_CORRECTED)),
        make_label(PAYLOADS["cup"], label_id=f"{sid}-seed", session_id=sid, seeded_error=True,
                   verification=reviewed(VerificationState.HUMAN_CORRECTED)),
    ]  # fmt: skip
    rates = correction_rates([history], policy)
    assert set(rates.classes) == {"box_track/mop", "box_track/cup"}
    assert rates.classes["box_track/mop"].reviewed == 4  # 표본 검증 제외
    assert rates.classes["box_track/cup"].changed == 1  # 지운 오탐
    overall = 5 / 101
    assert rates.overall == pytest.approx(overall)
    w = policy.correction.prior_weight
    # 4/4로 고쳤어도 표본이 적어 전체 수정률 쪽으로 당겨진다
    assert rates.rate("box_track/mop") == pytest.approx((4 + w * overall) / (4 + w))
    assert rates.rate("box_track/unseen") == pytest.approx(overall)


def test_score_term_plugins(policy: ActivePolicy) -> None:
    rates = correction_rates([reviewed_history("r", PAYLOADS["cup"], 10, 1)], policy)
    sessions = [
        (make_session("a"), [model("a", "1", PAYLOADS["cup"], confidence=0.2)]),
        (make_session("b"), [model("b", f"{i}", PAYLOADS["cup"]) for i in range(3)]),
    ]
    assert [s.session_id for s in score_sessions(sessions, rates, policy)] == ["b", "a"]
    # 불확실성 항목을 켜면 신뢰도 낮은 세션이 앞선다
    unc = policy.model_copy(
        update={"score": policy.score.model_copy(update={"terms": {"uncertainty": 1.0}})}
    )
    assert [s.session_id for s in score_sessions(sessions, rates, unc)] == ["a", "b"]

    # 새 항목은 등록만 하면 정책에서 켤 수 있다
    class LongSession:
        name = "long_session"

        def score(self, ctx: SessionContext) -> float:
            return ctx.session.duration_ms / 1000

        def explain(self, ctx: SessionContext) -> dict[str, float]:
            return {}

    if "long_session" not in TERMS:
        register_term("long_session")(lambda p: LongSession())
    longp = policy.model_copy(
        update={"score": policy.score.model_copy(update={"terms": {"long_session": 1.0}})}
    )
    sessions[0] = (make_session("a", duration_ms=120_000), sessions[0][1])
    ranked = score_sessions(sessions, rates, longp)
    assert ranked[0].session_id == "a" and ranked[0].score == 120
    with pytest.raises(ValueError, match="등록되지 않은"):
        score_sessions(
            sessions,
            rates,
            policy.model_copy(
                update={"score": policy.score.model_copy(update={"terms": {"nope": 1.0}})}
            ),
        )
    with pytest.raises(ValueError, match="겹칩니다"):
        register_term("correction_rate")(lambda p: LongSession())


def test_per_minute_normalization(policy: ActivePolicy) -> None:
    rates = correction_rates([reviewed_history("r", PAYLOADS["mop"], 100, 10)], policy)
    per_min = policy.model_copy(
        update={"score": policy.score.model_copy(update={"normalize": "per_minute"})}
    )
    sessions = [
        (make_session("long", duration_ms=600_000), pending("long", {"mop": 10}, PAYLOADS)),
        (make_session("short", duration_ms=60_000), pending("short", {"mop": 3}, PAYLOADS)),
    ]
    assert [s.session_id for s in score_sessions(sessions, rates, policy)] == ["long", "short"]
    assert [s.session_id for s in score_sessions(sessions, rates, per_min)] == ["short", "long"]
