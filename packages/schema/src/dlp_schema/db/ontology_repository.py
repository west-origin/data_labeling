"""DB에 등록된 사전을 세션·데이터셋 버전에 맞춰 읽는 공통 경계."""

import sqlalchemy as sa
from pydantic import ValidationError

from dlp_schema.db.tables import ontology_versions, sessions
from dlp_schema.ontology import Ontology


class LegacyOntologyContentError(ValueError):
    """등록된 사전이 영어 이름(en) 도입 전 내용이다 (ADR 0043). 저장소 사전을 다시 등록해야 한다."""


def get_ontology(conn: sa.Connection, version: str) -> Ontology:
    """등록된 사전을 읽는다. 없는 버전이나 손상된 등록 내용은 실패하며 v1로 대체하지 않는다.

    Raises:
        LegacyOntologyContentError: 영어 이름(en) 도입 전에 등록된 내용일 때. 저장소 사전을 다시
            등록하면(`make db-upgrade` = `dlp ontology register`, 초안 덧붙이기) 풀린다 (ADR 0043).
    """
    content = conn.execute(
        sa.select(ontology_versions.c.content).where(ontology_versions.c.version == version)
    ).scalar_one()
    try:
        ontology = Ontology.model_validate(content)
    except ValidationError as exc:
        if any(e["type"] == "missing" and e["loc"][-1:] == ("en",) for e in exc.errors()):
            raise LegacyOntologyContentError(
                f"등록된 온톨로지 {version}에 영어 이름(en)이 없습니다 (ADR 0043 이전 등록). "
                f"`make db-upgrade`(dlp ontology register {version})로 다시 등록하세요"
            ) from exc
        raise
    if ontology.version != version:
        raise ValueError(f"등록된 온톨로지 버전 불일치: {version} != {ontology.version}")
    return ontology


def get_session_ontology(conn: sa.Connection, session_id: str) -> Ontology:
    """세션의 버전에 해당하는 DB 사전을 읽는다. 버전 없는 세션에는 라벨을 생성하지 않는다."""
    version = conn.execute(
        sa.select(sessions.c.ontology_version).where(sessions.c.session_id == session_id)
    ).scalar_one()
    if version is None:
        raise ValueError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    return get_ontology(conn, version)
