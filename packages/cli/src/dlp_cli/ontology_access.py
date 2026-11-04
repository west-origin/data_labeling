"""CLI에서 세션·데이터셋 버전에 맞는 등록 사전을 읽는다. 연결은 항상 닫는다.

설정 파일(`config/ontology/`)이 아니라 DB `ontology_versions`에 등록된 사전을 읽는다. 세션은 수집 때
정한 온톨로지 버전을 쓰므로, 설정 파일이 바뀌어도 그 세션의 라벨은 등록된 사전으로 검사한다.

공개 함수:
- `session_ontology(url, session_id)` — 세션 버전의 사전 (`prelabel`, `relations`, `actions`가
  쓴다).
- `dataset_ontology(url, version_id)` — 데이터셋 버전의 사전 (`export`가 쓴다).
"""

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
