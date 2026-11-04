"""같은 버전 ID로 다시 빌드하면 스냅샷을 커밋하기 전에 거절한다 (F-ETA-31).

시간대 보존 SQLite 파일 DB로 돈다 (서비스 없음). 스냅샷 저장소는 커밋을 기록하는 가짜다.
정답 근거: 스냅샷 커밋은 DB 트랜잭션과 함께 롤백되지 않으므로, 중복 버전 ID는 커밋 전에
`DatasetBuildError`로 끝나야 하고 두 번째 빌드의 커밋 수는 0이다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dlp_datasets.build import BuildResult, DatasetBuildError, build_dataset_version
from dlp_datasets.policy import load_policy
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import get_dataset_version, insert_session, register_ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db


class RecordingSnapshots:
    """커밋한 접두사를 기록하는 가짜 스냅샷 저장소."""

    def __init__(self) -> None:
        self.commits: list[str] = []

    def commit(
        self, prefix: str, files: dict[str, Path], message: str, metadata: dict[str, str]
    ) -> str:
        """접두사를 기록하고 가짜 URI를 돌려준다."""
        self.commits.append(prefix)
        return f"local-snapshot://test/{prefix}/{len(self.commits)}"

    def read(self, snapshot_uri: str, path: str, dest: Path) -> None:
        """이 테스트는 읽지 않는다."""
        raise NotImplementedError


def test_duplicate_version_id_is_refused_before_snapshot_commit(tmp_path: Path) -> None:
    """F-ETA-31: 첫 빌드는 커밋 1번, 같은 ID 재빌드는 DatasetBuildError이고 커밋하지 않는다."""
    engine = sqlite_db(f"sqlite:///{tmp_path / 'ds.db'}")
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for i in range(2):
            insert_session(
                conn,
                make_session(
                    f"s{i}", worker_id=f"w{i}", site_id=f"site{i}",
                    privacy_state=PrivacyState.APPROVED,
                ),
            )  # fmt: skip
    policy = load_policy(repo_root())
    snapshots = RecordingSnapshots()

    def build() -> BuildResult:
        """dv1을 한 트랜잭션에서 빌드한다."""
        with engine.begin() as conn:
            return build_dataset_version(
                conn, snapshots, policy, version_id="dv1", ontology_version="1.0.0",
                golden_set_version=None, now=FIXED_TIME,
            )  # fmt: skip

    first = build()
    assert snapshots.commits == ["datasets/dv1"]

    with pytest.raises(DatasetBuildError, match="dv1"):
        build()
    assert snapshots.commits == ["datasets/dv1"]
    with engine.connect() as conn:
        assert get_dataset_version(conn, "dv1").snapshot_uri == first.version.snapshot_uri
    engine.dispose()
