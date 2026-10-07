"""골든셋 평가 실행기(`dlp_eval.runner`)의 DB 통합 테스트 (WP11, ADR 0013·0015·0025).

PostgreSQL이 필요하다 (`make up`, `@pytest.mark.services`). 테스트마다 일회용 DB를 만들고 지운다.
정답은 합성 행동 시나리오(`dlp_fixtures.actions.generate_action_scenario`)의 행동 라벨이고, 예측은
그 정답을 모델 출처로 복사한 것(완벽한 예측)과 일부를 뺀 것이다. 그래서 기대 지표를 미리 안다.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_eval.gate import decide
from dlp_eval.harness import evaluate
from dlp_eval.policy import load_policy
from dlp_eval.runner import load_golden, write_report
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_golden_set,
    insert_labels,
    insert_session,
    insert_withdrawal,
    record_review,
    register_ontology,
)
from dlp_schema.episode import retractions
from dlp_schema.labels import ActionPayload, LabelRecord, Provenance, Source, VerificationState
from dlp_schema.lineage import GoldenSet, Withdrawal
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Domain, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """일회용 PostgreSQL DB (마이그레이션 적용). 끝나면 강제로 지운다.

    접속 정보는 `DLP_DATABASE_URL`, 없으면 개발 기본값 (`make up`).
    """
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _model(labels: list[LabelRecord], version: str, suffix: str) -> list[LabelRecord]:
    """라벨을 모델 `version`이 낸 예측으로 복사한다 (ID에 접미사, 신뢰도 0.8)."""
    return [
        x.model_copy(
            update={
                "label_id": f"{x.label_id}-{suffix}",
                "provenance": Provenance(source=Source.MODEL, model_version=version),
                "confidence": 0.8,
            }
        )
        for x in labels
    ]


def test_golden_evaluation_from_database(pg: sa.Engine, tmp_path: Path) -> None:
    """DB에서 정답·예측을 모아 평가하고, 나쁜 후보를 게이트가 막으며 리포트 파일이 써진다.

    시나리오: 골든 세션 두 개 (하나는 장갑 스트림 있음). 정답 = 사람 행동 라벨 + 검수자가 승인한
    모델 라벨(첫 행동). actions-good은 정답 그대로, actions-bad는 첫 행동을 놓친다. 오류 삽입 사본은
    모델 버전을 달고 있어도 예측에서 빠져야 한다 (ADR 0015 감사 회귀).
    정답 근거: good은 정답과 같으므로 segment_f1 1.0, bad는 놓친 행동 때문에 1.0 미만.
    """
    glove_stream: dict[str, Any] = {
        "stream_id": "glove_right", "kind": StreamKind.GLOVE_RIGHT,
        "uri": "s3://dlp-raw/x/glove.parquet", "sync_method": SyncMethod.SHARED_CLOCK,
    }  # fmt: skip
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        for i, sid in enumerate(("g-0001", "g-0002")):
            session = make_session(sid)
            if i == 0:
                session = session.model_copy(
                    update={
                        "streams": (
                            *session.streams,
                            type(session.streams[0]).model_validate(glove_stream),
                        )
                    }
                )
            insert_session(conn, session)
            truth = [
                x for x in generate_action_scenario(i, session_id=sid).labels if x.kind == "action"
            ]
            good = _model(truth, "actions-good", "good")
            bad = _model(truth[1:], "actions-bad", "bad")  # 첫 행동을 놓침
            # 첫 행동의 정답은 검수자가 승인한 모델 라벨이다 (사람 라벨과 같이 정답으로 본다)
            # 오류 삽입 사본은 모델 버전을 그대로 달고 있어도 예측이 아니다 (감사 회귀, ADR 0015)
            seeded = good[1].model_copy(
                update={
                    "label_id": f"seed-x-{sid}",
                    "seeded_error": True,
                    "t_end_ms": good[1].t_end_ms,
                }
            )
            insert_labels(conn, [*truth[1:], *good, *bad, seeded])
            record_review(
                conn, good[0].label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME
            )
        insert_golden_set(
            conn,
            GoldenSet(
                version="cleaning-g1",
                domain=Domain.CLEANING,
                session_ids=("g-0001", "g-0002"),
                created_at=FIXED_TIME,
            ),
        )

    policy = load_policy(ROOT)
    with pg.connect() as conn:
        good = load_golden(conn, "cleaning-g1", {"actions": "actions-good"})
        bad = load_golden(conn, "cleaning-g1", {"actions": "actions-bad"})
    assert [s.groups["glove"] for s in good["actions"]] == ["glove", "bare"]
    good_report = evaluate(
        good, policy, golden_version="cleaning-g1", model_versions={"actions": "actions-good"}
    )
    bad_report = evaluate(
        bad, policy, golden_version="cleaning-g1", model_versions={"actions": "actions-bad"}
    )
    assert good_report.overall["actions"].metrics["segment_f1_0.5"] == 1.0
    assert bad_report.overall["actions"].metrics["segment_f1_0.5"] < 1.0

    decision = decide(bad_report, good_report, policy)
    assert not decision.passed
    write_report(tmp_path / "eval.json", bad_report, decision)
    data = json.loads((tmp_path / "eval.json").read_text("utf-8"))
    assert data["gate"]["passed"] is False and "glove=glove" in data["subgroups"]
    md = (tmp_path / "eval.md").read_text("utf-8")
    assert "배포 게이트: 실패" in md and "segment_f1_0.5" in md
    assert all(isinstance(x.payload, ActionPayload) for s in good["actions"] for x in s.truth)


def test_prefix_predictions_keep_reviewer_deleted_records_and_skip_withdrawn(
    pg: sa.Engine,
) -> None:
    """버전 앞부분(`actions-*`)으로 고른 예측도 전체 이력에서 뽑는다.

    검수자가 지운 오탐은 예측에 남고(사라지면 오탐이 안 세진다), 새 모델 버전이 지운 옛 레코드는
    빠진다. 사용 중지된 세션은 골든셋에 있어도 평가하지 않는다.
    """
    with pg.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        for sid in ("p-0001", "p-0002"):
            insert_session(conn, make_session(sid))
        truth = [
            x for x in generate_action_scenario(0, session_id="p-0001").labels if x.kind == "action"
        ]
        v1 = _model(truth, "actions-v1", "v1")
        fp = v1[0].model_copy(update={"label_id": "fp-v1"})  # 같은 구간 중복 예측 = 오탐
        # 검수자가 오탐을 지움 (사람 삭제 레코드)
        deleted = fp.model_copy(
            update={
                "label_id": "fp-v1-del",
                "parent_label_id": "fp-v1",
                "retracted": True,
                "provenance": Provenance(source=Source.HUMAN),
                "confidence": None,
            }
        )
        # 새 버전(actions-v2)이 옛 레코드 하나를 지우고 다시 냄 (모델 삭제 레코드)
        stale = _model(truth[:1], "actions-v0", "v0")
        insert_labels(
            conn,
            [*truth, *v1, fp, deleted, *stale, *retractions(stale, "actions-v2", FIXED_TIME)],
        )
        other = [
            x for x in generate_action_scenario(1, session_id="p-0002").labels if x.kind == "action"
        ]
        insert_labels(conn, [*other, *_model(other, "actions-v1", "v1")])
        insert_withdrawal(conn, Withdrawal(session_id="p-0002", reason="동의 철회",
                                           withdrawn_at=FIXED_TIME))  # fmt: skip
        insert_golden_set(
            conn,
            GoldenSet(
                version="cleaning-g2",
                domain=Domain.CLEANING,
                session_ids=("p-0001", "p-0002"),
                created_at=FIXED_TIME,
            ),
        )
    with pg.connect() as conn:
        data = load_golden(conn, "cleaning-g2", {"actions": "actions-*"})["actions"]
    assert [s.session_id for s in data] == ["p-0001"]
    pred_ids = {x.label_id for x in data[0].pred}
    assert "fp-v1" in pred_ids  # 검수자가 지운 오탐도 예측이다
    assert not any(i.endswith("-v0") for i in pred_ids)  # 새 버전이 지운 옛 레코드는 빠진다
    assert pred_ids == {x.label_id for x in v1} | {"fp-v1"}
