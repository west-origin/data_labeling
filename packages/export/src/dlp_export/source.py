"""내보낼 세션과 라벨을 고른다 (모든 형식 공용).

- 세션: 데이터셋 버전의 정책 분할(기본 학습·검증) 세션 중 지금 사용 중지되지 않은 것.
- 라벨: 데이터셋 버전 스냅샷의 이력에서 운영 현재 라벨(오류 삽입·측정 제외) 중
  사람이 만든 것과 검증 정책 상태의 모델 라벨. 블러 등 제외 종류는 뺀다.
- 영상: 라벨링 버킷의 블러본 (sessions/<세션>/blurred/<스트림>.mp4).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import sqlalchemy as sa

from dlp_datasets.snapshot import SnapshotError, SnapshotStore
from dlp_export.policy import ExportPolicy
from dlp_media.storage import ObjectStore, blurred_key, sha256_file
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db.repository import get_dataset_version, get_session, withdrawn_session_ids
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.session import Session, Stream


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportSession:
    session: Session
    split: Split
    labels: list[LabelRecord]  # 내보낼 라벨 (정책 적용 후)


@dataclass(frozen=True)
class ExportSource:
    version: DatasetVersion
    sessions: list[ExportSession]
    label_states: tuple[VerificationState, ...]  # 실제로 적용한 검증 정책


def label_states(policy: ExportPolicy, include_unreviewed: bool) -> tuple[VerificationState, ...]:
    extra = (VerificationState.UNREVIEWED,) if include_unreviewed else ()
    return (*policy.label_states, *extra)


def select_labels(
    history: list[LabelRecord], policy: ExportPolicy, states: tuple[VerificationState, ...]
) -> list[LabelRecord]:
    return [
        x
        for x in current_labels(history)
        if x.kind not in policy.excluded_kinds
        and (x.provenance.source is Source.HUMAN or x.verification.state in states)
    ]


def load_source(
    conn: sa.Connection,
    snapshots: SnapshotStore,
    version_id: str,
    policy: ExportPolicy,
    *,
    include_unreviewed: bool,
    splits: tuple[Split, ...] | None = None,
) -> ExportSource:
    version = get_dataset_version(conn, version_id)
    wanted = splits or policy.splits
    withdrawn = withdrawn_session_ids(conn)
    chosen = {
        sid: sp for sid, sp in version.splits.items() if sp in wanted and sid not in withdrawn
    }
    by_session: dict[str, list[LabelRecord]] = {sid: [] for sid in chosen}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.jsonl"
        snapshots.read(version.snapshot_uri, "labels.jsonl", path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = LabelRecord.model_validate_json(line)
                    if x.session_id in by_session:
                        by_session[x.session_id].append(x)
    pinned: dict[str, Session] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sessions.jsonl"
        try:
            snapshots.read(version.snapshot_uri, "sessions.jsonl", path)
        except (FileNotFoundError, SnapshotError):
            pass  # 세션을 고정하기 전에 만든 버전: DB의 현재 세션을 쓴다
        else:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    s = Session.model_validate_json(line)
                    pinned[s.session_id] = s
    states = label_states(policy, include_unreviewed)
    sessions = [
        ExportSession(
            pinned.get(sid) or get_session(conn, sid),
            chosen[sid],
            select_labels(by_session[sid], policy, states),
        )
        for sid in sorted(chosen)
    ]
    return ExportSource(version, sessions, states)


def fetch_blurred(labeling: ObjectStore, session: Session, stream: Stream, work: Path) -> Path:
    """블러본을 받는다. 없으면(렌더 전) 실패한다. 원본 버킷에서는 절대 읽지 않는다."""
    key = blurred_key(session.session_id, stream.stream_id)
    head = labeling.head(key)
    if head is None:
        raise ExportError(
            f"{session.session_id}/{stream.stream_id}: 블러본이 없습니다 (dlp privacy render)"
        )
    dest = work / "blurred" / session.session_id / f"{stream.stream_id}.mp4"
    if not dest.exists() or (head.sha256 is not None and sha256_file(dest) != head.sha256):
        dest.parent.mkdir(parents=True, exist_ok=True)
        labeling.get_file(key, dest)
    return dest


def assert_no_raw(out: Path, raw_bucket: str) -> None:
    """내보내기 결과의 모든 파일에 원본 위치가 없는지 본다 (parquet·영상 포함, 바이트로)."""
    needles = [
        n.encode()
        for n in (f"s3://{raw_bucket}", f"local://{raw_bucket}", f"{raw_bucket}/sessions/")
    ]
    for p in out.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            hit = next((n.decode() for n in needles if n in data), None)
            if hit:
                raise ExportError(f"내보내기에 원본 위치가 있습니다: {p.relative_to(out)} ({hit})")


def walk_files(root: Path) -> Iterator[Path]:
    return (p for p in sorted(root.rglob("*")) if p.is_file())
