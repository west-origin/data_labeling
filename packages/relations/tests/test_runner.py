"""관계 실행기 통합 테스트 (`@pytest.mark.services`: PostgreSQL 필요, `make up`).

테스트마다 새 DB에 닦기 시나리오 세션·라벨을 넣고 `run_relations`의 멱등성, 규칙 변경 반영,
검수자 삭제 존중, 승인 보존, 중복 궤적 처리를 본다 (ADR 0011, 0015, 0026).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa

from dlp_fixtures.wiping import generate_wiping_scenario
from dlp_relations.policy import load_policy
from dlp_relations.runner import run_relations
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    CoveragePayload,
    LabelRecord,
    Provenance,
    RelationPayload,
    Source,
    Trajectory3DPayload,
)
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]
SID = "wipe-0000"


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """테스트 전용 DB를 만들고 마이그레이션·온톨로지·닦기 시나리오(시드 0)를 넣는다. 끝나면
    지운다.
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
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, make_session(SID))
        insert_labels(conn, generate_wiping_scenario(0, session_id=SID).labels)
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def _current(conn: sa.Connection) -> list[LabelRecord]:
    """세션의 현재 관계·커버리지 레코드."""
    return current_labels(get_labels(conn, SID, kinds=["relation", "coverage"]))


def test_rerun_is_idempotent_and_rule_changes_regenerate(pg: sa.Engine) -> None:
    """첫 실행은 관계 + 커버리지 1개를 넣고 재실행은 모두 그대로 둔다. 손 파지 규칙을 빼면 그
    관계만 삭제 표시되고, 되돌리면 같은 내용이 새 ID(`-1` 접미사)로 다시 생긴다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    wiping = generate_wiping_scenario(0, session_id=SID)
    with pg.begin() as conn:
        first = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        again = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert first.inserted == len(wiping.truth_relations) + 1  # 관계 + 커버리지 1
    assert (again.inserted, again.retracted, again.kept) == (0, 0, first.inserted)
    [coverage] = [x.payload for x in current if isinstance(x.payload, CoveragePayload)]
    assert coverage.ratio == pytest.approx(wiping.coverage, abs=0.03)
    assert all(x.provenance.model_version == first.version for x in current)

    # 규칙 변경: 손 파지 규칙을 빼면 파지 관계만 삭제 표시되고 나머지는 그대로다
    no_grasp = policy.model_copy(
        update={"rules": tuple(r for r in policy.rules if r.id != "hand_grasp")}
    )
    with pg.begin() as conn:
        changed = run_relations(conn, SID, ontology, no_grasp, FIXED_TIME)
        assert run_relations(conn, SID, ontology, no_grasp, FIXED_TIME).inserted == 0
        current = _current(conn)
    assert (changed.inserted, changed.retracted) == (0, 1)
    assert not any(
        isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
        for x in current
    )

    # 규칙을 되돌리면 파지 관계가 새 ID로 다시 생긴다
    with pg.begin() as conn:
        restored = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert (restored.inserted, restored.retracted) == (1, 0)
    [grasp] = [
        x for x in current
        if isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
    ]  # fmt: skip
    assert grasp.label_id.endswith("-1")

    # 한 번 더 빼고 되돌리면 `-2` (감사 회귀: 예전에는 `<기본>-1:retracted`까지 세어 `-3`이었다)
    with pg.begin() as conn:
        assert run_relations(conn, SID, ontology, no_grasp, FIXED_TIME).retracted == 1
        again_restored = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert again_restored.inserted == 1
    [grasp2] = [
        x for x in current
        if isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
    ]  # fmt: skip
    assert grasp2.label_id == grasp.label_id[: -len("-1")] + "-2"


def test_reviewer_deleted_relation_is_not_reinserted(pg: sa.Engine) -> None:
    """검수자가 지운 도구-표면 관계는 다시 실행해도 넣지 않는다 (skipped_by_review=1)."""
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    with pg.begin() as conn:
        run_relations(conn, SID, ontology, policy, FIXED_TIME)
        [contact, *_] = [
            x
            for x in _current(conn)
            if isinstance(x.payload, RelationPayload)
            and x.payload.derived_by == "tool_surface_contact"
        ]
        deletion = contact.model_copy(
            update={
                "label_id": f"{contact.label_id}:rev",
                "parent_label_id": contact.label_id,
                "retracted": True,
                "provenance": Provenance(source=Source.HUMAN),
                "confidence": None,
            }
        )
        insert_labels(conn, [deletion])
        summary = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        ids = {x.label_id for x in _current(conn)}
    assert (summary.inserted, summary.skipped_by_review) == (0, 1)
    assert contact.label_id not in ids


def test_rule_change_keeps_approved_relations(pg: sa.Engine) -> None:
    """승인된 관계는 그 규칙을 빼도 삭제 표시하지 않는다 (ADR 0015)."""
    from dlp_schema.db.repository import record_review
    from dlp_schema.labels import VerificationState

    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    with pg.begin() as conn:
        run_relations(conn, SID, ontology, policy, FIXED_TIME)
        [grasp] = [
            x for x in _current(conn)
            if isinstance(x.payload, RelationPayload) and x.payload.derived_by == "hand_grasp"
        ]  # fmt: skip
        record_review(conn, grasp.label_id, VerificationState.HUMAN_APPROVED, "r1", FIXED_TIME)
    no_grasp = policy.model_copy(
        update={"rules": tuple(r for r in policy.rules if r.id != "hand_grasp")}
    )
    with pg.begin() as conn:
        changed = run_relations(conn, SID, ontology, no_grasp, FIXED_TIME)
        ids = {x.label_id for x in _current(conn)}
    # 승인된 관계는 규칙에서 빠져도 지우지 않는다
    assert changed.retracted == 0 and grasp.label_id in ids


def test_duplicate_tool_trajectory_does_not_crash_the_run(pg: sa.Engine) -> None:
    """감사 회귀 (4차): 같은 도구 작용부 궤적이 새 ID로 다시 나와도 같은 커버리지 ID를 두 번 넣지
    않는다 (UniqueViolation 없음).
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    wiping = generate_wiping_scenario(0, session_id=SID)
    tool = [
        x
        for x in wiping.labels
        if isinstance(x.payload, Trajectory3DPayload) and x.payload.entity_id == "rag_01"
    ]
    with pg.begin() as conn:
        insert_labels(conn, [x.model_copy(update={"label_id": f"{x.label_id}-v2"}) for x in tool])
    with pg.begin() as conn:
        first = run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert first.inserted == len(wiping.truth_relations) + 1
    assert sum(isinstance(x.payload, CoveragePayload) for x in current) == 1


def test_long_session_id_retractions_fit_identifier(pg: sa.Engine) -> None:
    """세션 ID가 길어 `<라벨 ID>:retracted`가 128자를 넘어도 규칙 변경 재실행이 성공한다.

    감사 회귀: 예전에는 삭제 레코드를 직접 만들어(검증 없는 model_copy) 128자 넘는 ID를 넣으려 했다.
    지금은 `dlp_schema.episode.retractions`가 해시로 줄인다.
    """
    sid = "w" * 100  # 기본 라벨 ID = 100 + "-rel-" + 16 = 121자, ":retracted"를 붙이면 131자
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    no_grasp = policy.model_copy(
        update={"rules": tuple(r for r in policy.rules if r.id != "hand_grasp")}
    )
    with pg.begin() as conn:
        insert_session(conn, make_session(sid))
        insert_labels(conn, generate_wiping_scenario(0, session_id=sid).labels)
        run_relations(conn, sid, ontology, policy, FIXED_TIME)
        changed = run_relations(conn, sid, ontology, no_grasp, FIXED_TIME)
        removed = [x for x in get_labels(conn, sid, kinds=["relation"]) if x.retracted]
    assert changed.retracted == 1
    [r] = removed
    assert len(r.label_id) <= 128 and r.label_id.endswith(":retracted")


def test_new_records_use_the_session_ontology_version(pg: sa.Engine) -> None:
    """새 관계·커버리지의 온톨로지 버전은 세션의 것이다 (시각 순 첫 라벨의 것이 아니다).

    시각 0에 옛 온톨로지 버전(0.9.0, 이관 전 라벨을 흉내)의 라벨 하나를 앞에 둔다. 감사 회귀:
    예전에는 그 버전을 새 레코드에 썼다.
    """
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    with pg.begin() as conn:
        register_ontology(conn, ontology.model_copy(update={"version": "0.9.0"}))
        insert_labels(
            conn,
            [
                make_label(
                    action_payload(),
                    label_id="0000-old",
                    session_id=SID,
                    ontology_version="0.9.0",
                )
            ],
        )
        assert get_labels(conn, SID)[0].ontology_version == "0.9.0"  # 시각 순 첫 라벨
        run_relations(conn, SID, ontology, policy, FIXED_TIME)
        current = _current(conn)
    assert current and {x.ontology_version for x in current} == {"1.0.0"}
