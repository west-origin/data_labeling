"""`dlp dataset build`가 빌드 거절을 traceback 없이 알린다 (F-ETA-31).

시간대 보존 SQLite 파일 DB와 로컬 스냅샷 저장소로 돈다 (서비스 없음).
정답 근거: 같은 버전 ID로 다시 빌드하면 한국어 오류를 stderr에 내고 종료 코드 1이며, 스냅샷
디렉터리에 두 번째 커밋이 생기지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_cli import dataset_cmds
from dlp_cli.main import main
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import make_session, sqlite_db


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


def test_rebuilding_an_existing_version_exits_1_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-ETA-31: 같은 버전 ID 재빌드는 오류 문구와 종료 코드 1, 새 스냅샷 커밋 없음."""
    url = f"sqlite:///{tmp_path / 'build.db'}"
    engine = sqlite_db(url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for i in range(2):
            insert_session(
                conn,
                make_session(
                    f"b{i}", worker_id=f"w{i}", site_id=f"site{i}",
                    privacy_state=PrivacyState.APPROVED,
                ),
            )  # fmt: skip
    engine.dispose()
    root = tmp_path / "snapshots"

    class _Snapshots:
        @staticmethod
        def from_env(**_: Any) -> LocalSnapshotStore:
            return LocalSnapshotStore(root)

    monkeypatch.setattr(dataset_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.setattr(dataset_cmds, "LakeFSSnapshotStore", _Snapshots)

    assert main(["dataset", "build", "dv1", "--url", url]) == 0
    committed = sorted(p for p in root.rglob("*") if p.is_file())
    capsys.readouterr()
    assert main(["dataset", "build", "dv1", "--url", url]) == 1
    err = capsys.readouterr().err
    assert "오류: 데이터셋 버전 dv1이(가) 이미 있습니다" in err
    assert "Traceback" not in err
    assert sorted(p for p in root.rglob("*") if p.is_file()) == committed
