"""내보낼 세션과 라벨을 고른다 (모든 형식 공용).

- 세션: 데이터셋 버전의 정책 분할(기본 학습·검증) 세션 중 지금 사용 중지되지 않은 것.
- 라벨: 데이터셋 버전 스냅샷의 이력에서 운영 현재 라벨(오류 삽입·측정 제외) 중
  사람이 만든 것과 검증 정책 상태의 모델 라벨. 블러 등 제외 종류는 뺀다.
- 영상: 라벨링 버킷의 블러본 (sessions/<세션>/blurred/<스트림>.mp4).
  스냅샷이 아니라 **지금** DB 상태로 세션이 프라이버시 승인 상태여야 하고, 블러본은 지금 운영 블러
  라벨·렌더 정책으로 렌더한 것이어야 한다 (승인 취소·재승인 뒤 이전 블러본 금지, ADR 0024).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import sqlalchemy as sa

from dlp_datasets.snapshot import SnapshotError, SnapshotStore
from dlp_export.policy import ExportPolicy
from dlp_media.storage import ObjectStore, blurred_key, sha256_file
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import (
    VIDEO_KINDS,
    RenderNotCurrentError,
    check_fetched,
    check_render_meta,
    expected_render_hash,
    read_render_meta,
)
from dlp_schema import repo_root
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
    # 스트림 → 지금 DB 상태로 본 블러본 렌더 해시 (load_source가 승인 상태를 확인하고 채운다).
    # 여기 없는 스트림의 블러본은 받지 않는다
    render_hashes: dict[str, str] = field(default_factory=dict[str, str])


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
    privacy: PrivacyPolicy | None = None,
) -> ExportSource:
    """privacy: 블러본 해시를 계산할 프라이버시 정책 (없으면 config/policies/privacy.yaml)."""
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
    privacy = privacy or load_privacy_policy(repo_root())
    sessions: list[ExportSession] = []
    for sid in sorted(chosen):
        session = pinned.get(sid) or get_session(conn, sid)
        hashes: dict[str, str] = {}
        for stream in (s for s in session.streams if s.kind in VIDEO_KINDS):
            try:
                # 스냅샷의 세션 상태가 아니라 지금 상태: 그 뒤 승인이 풀렸으면 내보내지 않는다
                hashes[stream.stream_id] = expected_render_hash(
                    conn, sid, stream.stream_id, privacy
                )
            except RenderNotCurrentError as exc:
                raise ExportError(str(exc)) from exc
        sessions.append(
            ExportSession(
                session, chosen[sid], select_labels(by_session[sid], policy, states), hashes
            )
        )
    return ExportSource(version, sessions, states)


def fetch_blurred(labeling: ObjectStore, es: ExportSession, stream: Stream, work: Path) -> Path:
    """블러본을 받는다. 원본 버킷에서는 절대 읽지 않는다.

    렌더 기록이 load_source가 지금 DB 상태로 계산한 해시와 같아야 한다 (없거나 무효이거나
    다르면 실패: 렌더 전이거나 승인 취소·블러 변경 뒤 다시 렌더하지 않은 블러본).
    """
    session = es.session
    where = f"{session.session_id}/{stream.stream_id}"
    key = blurred_key(session.session_id, stream.stream_id)
    head = labeling.head(key)
    if head is None:
        raise ExportError(f"{where}: 블러본이 없습니다 (dlp privacy render)")
    expected = es.render_hashes.get(stream.stream_id)
    if expected is None:
        raise ExportError(f"{where}: 프라이버시 승인 상태를 확인하지 않은 스트림입니다")
    dest = work / "blurred" / session.session_id / f"{stream.stream_id}.mp4"
    try:
        meta = read_render_meta(labeling, session.session_id, stream.stream_id, work)
        rendered = check_render_meta(meta, expected, labeling, session.session_id, stream.stream_id)
        if not dest.exists() or (head.sha256 is not None and sha256_file(dest) != head.sha256):
            dest.parent.mkdir(parents=True, exist_ok=True)
            labeling.get_file(key, dest)
        check_fetched(dest, rendered, session.session_id, stream.stream_id)
    except RenderNotCurrentError as exc:
        raise ExportError(str(exc)) from exc
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
