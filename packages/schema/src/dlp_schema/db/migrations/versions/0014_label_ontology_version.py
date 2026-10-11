"""새 라벨의 세션 온톨로지 버전 일치 강제 (ADR 0037).

기존 이력은 수정하지 않는다. 삭제 레코드는 과거 부모의 버전을 유지할 수 있다.
의미 검증은 공통 저장 함수, 버전 검증은 직접 SQL에도 적용한다.
"""

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """신규 INSERT만 검사한다. 기존 위반은 감사 명령에서 찾아 이력 추가로 정리한다."""
    if op.get_bind().dialect.name == "postgresql":
        op.execute("""
            CREATE FUNCTION dlp_label_ontology_version() RETURNS trigger AS $$
            DECLARE session_version text;
            BEGIN
                SELECT ontology_version INTO session_version FROM sessions
                    WHERE session_id = NEW.session_id FOR SHARE;
                IF NOT NEW.retracted AND
                    NEW.ontology_version IS DISTINCT FROM session_version THEN
                    RAISE EXCEPTION 'session and label ontology versions differ'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
        """)
        op.execute("""
            CREATE TRIGGER label_ontology_version BEFORE INSERT ON label_records
            FOR EACH ROW EXECUTE FUNCTION dlp_label_ontology_version()
        """)
    elif op.get_bind().dialect.name == "sqlite":
        op.execute("""
            CREATE TRIGGER label_ontology_version BEFORE INSERT ON label_records
            WHEN NOT NEW.retracted AND NEW.ontology_version IS NOT
                (SELECT ontology_version FROM sessions WHERE session_id = NEW.session_id)
            BEGIN
                SELECT RAISE(ABORT, 'session and label ontology versions differ');
            END
        """)


def downgrade() -> None:
    """이 리비전 소유의 트리거와 함수만 제거한다."""
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TRIGGER label_ontology_version ON label_records")
        op.execute("DROP FUNCTION dlp_label_ontology_version()")
    elif op.get_bind().dialect.name == "sqlite":
        op.execute("DROP TRIGGER label_ontology_version")
