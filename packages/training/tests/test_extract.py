"""학습 예제 추출(`dlp_train.extract`), 학습 정책 검증, oracle-stub 학습기·로더의 단위 테스트.

WP13, ADR 0016. DB 없이 라벨 이력(`history`)을 직접 만든다. 이력 하나에 검수 결과 네 가지(승인·수정·
추가·삭제)와 빠져야 할 레코드(미검수, 모델 버전 교체로 지운 것, 오류 삽입, 측정, 다른 과제)가 들어
있어 기대 예제 집합을 미리 안다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlp_schema.dataset import Split
from dlp_schema.labels import (
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.predictor import Clip, ModelUnavailableError
from dlp_schema.testing import FIXED_TIME, make_label
from dlp_train.extract import TrainingData, extract_examples
from dlp_train.policy import TrainingPolicy, load_policy
from dlp_train.trainers import LoadContext, OracleStubLoader, OracleStubTrainer

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def policy() -> TrainingPolicy:
    """저장소의 실제 학습 정책 (config/policies/training.yaml)."""
    return load_policy(ROOT)


def box(cls: str, x: float = 10.0) -> dict[str, Any]:
    """키프레임 하나짜리 박스 트랙 페이로드 (개체 `<cls>_01`)."""
    return {
        "kind": "box_track",
        "entity_id": f"{cls}_01",
        "class_id": cls,
        "keyframes": [{"t_ms": 0, "x": x, "y": 10, "w": 30, "h": 30}],
    }


def model(label_id: str, payload: dict[str, Any], session: str = "s1", **kw: Any) -> LabelRecord:
    """모델(base-v1)이 낸 바디캠 라벨 (기본 미검수)."""
    return make_label(
        payload,
        label_id=label_id,
        session_id=session,
        stream_id="bodycam",
        provenance=Provenance(source=Source.MODEL, model_version="base-v1"),
        confidence=0.8,
        **kw,
    )


def human(label_id: str, payload: dict[str, Any], session: str = "s1", **kw: Any) -> LabelRecord:
    """사람이 만든(수정한) 바디캠 라벨."""
    return make_label(
        payload,
        label_id=label_id,
        session_id=session,
        stream_id="bodycam",
        verification=Verification(
            state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=FIXED_TIME
        ),
        **kw,
    )


def approved() -> Verification:
    """검수자 r1이 승인한 검증 상태."""
    return Verification(
        state=VerificationState.HUMAN_APPROVED, reviewer_id="r1", reviewed_at=FIXED_TIME
    )


def history(session: str = "s1") -> list[LabelRecord]:
    """세션 하나의 라벨 이력. 줄마다 주석이 기대 처리(예제 종류 또는 제외)다."""
    return [
        model("m-cup", box("cup"), session, verification=approved()),  # 그대로 승인
        model("m-bucket", box("bucket"), session),  # 사람이 고침
        human("h-bucket", box("bucket", 14), session, parent_label_id="m-bucket"),
        human("h-mop", box("mop"), session),  # 모델이 놓친 것 추가
        model("m-fp", box("sponge"), session),  # 사람이 지움 (오탐)
        human("h-fp", box("sponge"), session, parent_label_id="m-fp", retracted=True),
        model("m-unrev", box("broom"), session),  # 미검수 → 제외
        # 모델 버전이 바뀌어 지운 것은 사람의 삭제가 아니다
        model("m-old", box("towel"), session),
        model("m-old:retracted", box("towel"), session, parent_label_id="m-old", retracted=True),
        # 오류 삽입 사본과 측정 레코드 → 제외
        human("seed-1", box("cup", 99), session, seeded_error=True),
        human("blind-1", box("cup", 50), session, measurement="blind"),
        # 다른 과제
        make_label(
            {
                "kind": "action",
                "action_id": "a1",
                "hand": "right",
                "verb": "wipe",
                "t_approach_ms": 0,
                "t_end_ms": 1000,
            },
            label_id="act",
            session_id=session,
        ),
    ]


def test_extract_examples_with_auto_vs_corrected_diff(policy: TrainingPolicy) -> None:
    """예제마다 변화 종류와 모델 원본(origin)이 붙고, 제외 대상은 하나도 들지 않는다."""
    ex = extract_examples(history(), {"s1": Split.TRAIN}, "objects", policy)
    got = {(e.change, e.label.label_id, e.origin.label_id if e.origin else None) for e in ex}
    assert got == {
        ("accepted", "m-cup", "m-cup"),
        ("corrected", "h-bucket", "m-bucket"),
        ("added", "h-mop", None),
        ("deleted", "m-fp", "m-fp"),
    }


def test_golden_and_holdout_sessions_never_become_examples(policy: TrainingPolicy) -> None:
    """골든·holdout·분할 밖 세션은 예제가 되지 않는다 (val 세션만 남는다)."""
    labels = history("s1") + history("g1") + history("h1") + history("x1")
    splits = {"s1": Split.VAL, "g1": Split.GOLDEN, "h1": Split.HOLDOUT}  # x1은 분할 밖
    ex = extract_examples(labels, splits, "objects", policy)
    assert {e.session_id for e in ex} == {"s1"}
    assert {e.split for e in ex} == {Split.VAL}


def test_policy_rejects_golden_split_and_unreviewed(policy: TrainingPolicy) -> None:
    """정책 검증기: 학습 분할에 golden, 학습 상태에 unreviewed가 있으면 거부한다."""
    data = policy.model_dump(mode="json")
    with pytest.raises(ValidationError, match="골든"):
        TrainingPolicy.model_validate({**data, "splits": ["train", "golden"]})
    with pytest.raises(ValidationError, match="미검수"):
        TrainingPolicy.model_validate({**data, "trainable_states": ["unreviewed"]})


def test_oracle_stub_learns_only_seen_classes(policy: TrainingPolicy, tmp_path: Path) -> None:
    """oracle-stub은 배운 클래스만 예측하고(지운 예제 제외), 정답 없이 로드하면 쓸 수 없다."""
    ex = extract_examples(history(), {"s1": Split.TRAIN}, "objects", policy)
    out = OracleStubTrainer().train(
        TrainingData("dv1", "objects", ex), {"jitter_px": 0.0}, tmp_path
    )
    spec = json.loads(out.artifact.read_text())
    # 지운 예제(sponge)는 배울 클래스가 아니다
    assert spec["classes"] == ["bucket", "cup", "mop"]
    assert out.metrics["examples"] == 4

    truth = [human(f"t-{c}", box(c), "g1") for c in ("cup", "mop", "broom")]
    ctx = LoadContext(
        now=FIXED_TIME,
        ontology_version="1.0.0",
        truth=lambda sid, stream: [x for x in truth if x.session_id == sid],
    )
    predictor = OracleStubLoader().load(out.artifact, version="objects-v1", ctx=ctx)
    pred = predictor.run(Clip("g1", "bodycam", tmp_path / "none.mp4"))
    assert sorted(p.payload.class_id for p in pred) == ["cup", "mop"]  # type: ignore[union-attr]
    assert all(p.provenance.model_version == "objects-v1" for p in pred)
    assert all(p.label_id.startswith("g1-bodycam-trained-objects-") for p in pred)

    with pytest.raises(ModelUnavailableError):
        OracleStubLoader().load(
            out.artifact, version="v", ctx=LoadContext(now=FIXED_TIME, ontology_version="1.0.0")
        )
