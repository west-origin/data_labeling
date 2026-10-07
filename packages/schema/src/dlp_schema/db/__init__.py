"""DB 계층: Core 테이블(`tables`), 계약 ↔ 행 저장소(`repository`), Alembic 실행(`migrate`).

연결과 트랜잭션은 호출자가 관리한다. 스키마 변경은 `tables`와 새 Alembic 리비전을 함께 바꾼다.
"""
