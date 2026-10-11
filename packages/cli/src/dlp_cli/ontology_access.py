"""CLI에서 세션·데이터셋 버전에 맞는 등록 사전을 읽는다. 연결은 항상 닫는다."""

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_schema.db.repository import get_dataset_version, get_ontology, get_session_ontology
from dlp_schema.ontology import Ontology


def session_ontology(url: str | None, session_id: str) -> Ontology:
    """모델 준비 전에 세션 사전을 읽는다. 실행 단계도 저장 시 버전을 다시 검사한다."""
    engine = sa.create_engine(database_url(url))
    try:
        with engine.connect() as conn:
            return get_session_ontology(conn, session_id)
    finally:
        engine.dispose()


def dataset_ontology(url: str | None, version_id: str) -> Ontology:
    """내보내기 대상 데이터셋 버전의 등록 사전을 읽는다."""
    engine = sa.create_engine(database_url(url))
    try:
        with engine.connect() as conn:
            return get_ontology(conn, get_dataset_version(conn, version_id).ontology_version)
    finally:
        engine.dispose()
