"""골든셋 정답 고정 계약·저장소·마이그레이션 0018 (ADR 0041).

SQLite 테스트는 서비스 없이 돈다. PostgreSQL 테스트(`@pytest.mark.services`)는 같은 성질을 JSONB
열에서 다시 본다.

정답 근거: 계약 정의(`GoldenSet.truth_label_ids`는 중복 없는 ID 튜플 또는 None), 0018 마이그레이션
docstring(이전 행은 NULL, 정답을 고정한 골든셋이 있으면 되돌리기 거부), `insert_golden_set`
docstring(정답 ID는 그 골든셋 세션의 라벨이어야 한다).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema.db.migrate import downgrade, upgrade
from dlp_schema.db.repository import (
    get_golden_set,
    insert_golden_set,
    insert_labels,
    insert_session,
    register_ontology,
)
from dlp_schema.lineage import GoldenSet
from dlp_schema.ontology import Ontology
from dlp_schema.session import Domain
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db


def _golden(version: str, ids: tuple[str, ...] | None) -> GoldenSet:
    """세션 s001 하나짜리 골든셋 버전 (ids = 고정한 정답 라벨 ID, None이면 이전 형식)."""
    return GoldenSet(
        version=version,
        domain=Domain.CLEANING,
        session_ids=("s001",),
        created_at=FIXED_TIME,
        truth_label_ids=ids,
    )


def _null_rows(engine: sa.Engine) -> list[str]:
    """`truth_label_ids`가 SQL NULL인 골든셋 버전 (JSON null이 아니라 열이 비어 있는 행)."""
    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("SELECT version FROM golden_sets WHERE truth_label_ids IS NULL")
        ).scalars()
        return sorted(rows)


def test_truth_label_ids_contract() -> None:
    """None(이전 형식)·빈 튜플(정답 없음으로 고정)·ID 목록은 받고, 중복 ID와 빈 ID는 거부한다."""
    for ids in (None, (), ("l001", "l002")):
        golden = _golden("g", ids)
        assert GoldenSet.model_validate_json(golden.model_dump_json()) == golden
        assert golden.truth_label_ids == ids
    # 필드를 빼면 이전 형식(None)이다 (0018 이전 계약으로 만든 값도 그대로 읽힌다)
    legacy = _golden("g", None).model_dump(mode="json")
    del legacy["truth_label_ids"]
    assert GoldenSet.model_validate(legacy).truth_label_ids is None
    with pytest.raises(ValidationError, match="중복"):
        _golden("g", ("l001", "l001"))
    with pytest.raises(ValidationError):
        _golden("g", ("",))


def test_migration_0018_keeps_legacy_null_and_refuses_downgrade_with_frozen_rows(
    tmp_path: Path, ontology: Ontology
) -> None:
    """이전 형식 행만 있으면 0018을 내렸다 다시 올릴 수 있고, 고정한 행이 있으면 거부한다.

    고정한 행은 빈 튜플(라벨링 전에 고정)이어도 고정이다. 이전 형식 행은 SQL NULL로 저장되고
    None으로 읽힌다. 고정한 행은 내리기를 거부해도 그대로 남는다.
    """
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    engine = sqlite_db(url, create=False)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
        insert_labels(conn, [make_label(action_payload())])
        insert_golden_set(conn, _golden("golden-legacy", None))
    assert _null_rows(engine) == ["golden-legacy"]
    engine.dispose()

    # 이전 형식만: 0017로 내렸다가 다시 올려도 행은 그대로 이전 형식이다
    downgrade(url, "0017")
    upgrade(url)
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        assert get_golden_set(conn, "golden-legacy").truth_label_ids is None

    for version, ids in (("golden-empty", ()), ("golden-v1", ("l001",))):
        with engine.begin() as conn:
            insert_golden_set(conn, _golden(version, ids))
        with engine.connect() as conn:
            assert get_golden_set(conn, version).truth_label_ids == ids
    assert _null_rows(engine) == ["golden-legacy"]
    engine.dispose()
    with pytest.raises(RuntimeError, match="0018"):
        downgrade(url, "0017")
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        assert get_golden_set(conn, "golden-v1").truth_label_ids == ("l001",)
    engine.dispose()


def test_insert_golden_set_rejects_truth_outside_its_sessions(ontology: Ontology) -> None:
    """정답 ID는 그 골든셋 세션의 라벨이어야 한다 (없는 ID, 다른 세션의 라벨은 거부)."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
        insert_session(conn, make_session("s002"))
        insert_labels(
            conn,
            [
                make_label(action_payload()),
                make_label(action_payload(), label_id="other", session_id="s002"),
            ],
        )
    for ids in (("nope",), ("l001", "other")):
        with pytest.raises(ValueError, match="골든셋 세션에 없습니다"), engine.begin() as conn:
            insert_golden_set(conn, _golden("golden-bad", ids))
    with engine.begin() as conn:
        insert_golden_set(conn, _golden("golden-v1", ("l001",)))
    with engine.connect() as conn:
        assert get_golden_set(conn, "golden-v1").truth_label_ids == ("l001",)
        assert conn.execute(sa.text("SELECT count(*) FROM golden_sets")).scalar_one() == 1
    engine.dispose()


@pytest.mark.services
def test_migration_0018_on_postgresql(pg_url: str, ontology: Ontology) -> None:
    """PostgreSQL(JSONB)에서도 이전 형식은 SQL NULL, 고정한 행이 있으면 0018 내리기를 거부한다.

    고정한 행을 지우면 내리고 다시 올릴 수 있다 (운영자가 직접 정리한 뒤의 경로).
    """
    engine = sa.create_engine(pg_url)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, make_session())
        insert_labels(conn, [make_label(action_payload())])
        insert_golden_set(conn, _golden("golden-legacy", None))
        insert_golden_set(conn, _golden("golden-v1", ("l001",)))
    with engine.connect() as conn:
        assert get_golden_set(conn, "golden-legacy").truth_label_ids is None
        assert get_golden_set(conn, "golden-v1").truth_label_ids == ("l001",)
    assert _null_rows(engine) == ["golden-legacy"]
    with pytest.raises(RuntimeError, match="0018"):
        downgrade(pg_url, "0017")
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM golden_sets WHERE truth_label_ids IS NOT NULL"))
    engine.dispose()
    downgrade(pg_url, "0017")
    upgrade(pg_url)
    engine = sa.create_engine(pg_url)
    with engine.connect() as conn:
        assert get_golden_set(conn, "golden-legacy").truth_label_ids is None
    engine.dispose()
