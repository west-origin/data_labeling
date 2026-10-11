"""DB에 등록된 사전을 세션·데이터셋 버전에 맞춰 읽는 공통 경계."""

import sqlalchemy as sa

from dlp_schema.db.tables import ontology_versions, sessions
from dlp_schema.ontology import Ontology


def get_ontology(conn: sa.Connection, version: str) -> Ontology:
    """등록된 사전을 읽는다. 없는 버전이나 손상된 등록 내용은 실패하며 v1로 대체하지 않는다."""
    content = conn.execute(
        sa.select(ontology_versions.c.content).where(ontology_versions.c.version == version)
    ).scalar_one()
    ontology = Ontology.model_validate(content)
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
