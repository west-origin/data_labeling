"""내보낼 세션과 라벨을 고른다 (모든 형식 공용, WP15, ADR 0018·0021·0024).

- 세션: 데이터셋 버전의 정책 분할(기본 학습·검증) 세션 중 지금 사용 중지되지 않은 것.
  사용 중지는 스냅샷이 아니라 **내보내는 시점의 DB**(`withdrawals`)로 본다: 버전을 만든 뒤
  동의를 철회한 세션도 빠진다.
- 라벨: 데이터셋 버전 스냅샷(labels.jsonl)의 이력에서 운영 현재 라벨(`current_labels`: 오류 삽입·
  측정 제외) 중 사람이 만든 것과 검증 정책 상태의 모델 라벨. 블러 등 제외 종류는 뺀다.
- 세션 메타데이터: 스냅샷의 sessions.jsonl(빌드 때 고정)을 쓴다. 그 전 버전은 DB의 지금 세션.
- 영상: 라벨링 버킷의 블러본 (sessions/<세션>/blurred/<스트림>.mp4).
  스냅샷이 아니라 **지금** DB 상태로 세션이 프라이버시 승인 상태여야 하고, 블러본은 지금 운영 블러
  라벨·렌더 정책으로 렌더한 것이어야 한다 (승인 취소·재승인 뒤 이전 블러본 금지, ADR 0024).
- 결과 검사: `assert_no_raw`가 결과 파일 바이트에서 원본 버킷 위치를 찾으면 실패한다.

주요 공개 이름: `ExportError`, `ExportSession`, `ExportSource`, `label_states`, `select_labels`,
`load_source`, `fetch_blurred`, `assert_no_raw`, `walk_files`.
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
    """내보내기를 진행할 수 없는 상태 (승인 안 됨, 블러본 없음·불일치, 원본 위치 노출 등).

    CLI는 이 오류를 사용자 메시지로 보여 주고 실패 종료한다.
    """


@dataclass(frozen=True)
class ExportSession:
    """내보낼 세션 하나와 그 세션에서 내보낼 라벨."""

    session: Session  # 스냅샷에 고정된 세션 메타데이터 (없으면 DB의 지금 세션)
    split: Split  # 데이터셋 버전에서의 분할
    labels: list[LabelRecord]  # 내보낼 라벨 (정책 적용 후)
    # 스트림 → 지금 DB 상태로 본 블러본 렌더 해시 (load_source가 승인 상태를 확인하고 채운다).
    # 여기 없는 스트림의 블러본은 받지 않는다
    render_hashes: dict[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class ExportSource:
    """형식별 쓰기 함수가 받는 입력 전체."""

    version: DatasetVersion  # 대상 데이터셋 버전 (DB `dataset_versions`)
    sessions: list[ExportSession]  # 세션 ID 순
    label_states: tuple[VerificationState, ...]  # 실제로 적용한 검증 정책


def label_states(policy: ExportPolicy, include_unreviewed: bool) -> tuple[VerificationState, ...]:
    """이번 내보내기에 적용할 모델 라벨 검증 상태.

    정책 `label_states`에, 옵션을 켰을 때만 `unreviewed`를 맨 끝에 붙인다.
    """
    extra = (VerificationState.UNREVIEWED,) if include_unreviewed else ()
    return (*policy.label_states, *extra)


def select_labels(
    history: list[LabelRecord], policy: ExportPolicy, states: tuple[VerificationState, ...]
) -> list[LabelRecord]:
    """세션 라벨 이력 → 내보낼 라벨.

    1. `current_labels`: 수정·삭제가 반영된 운영 현재 라벨만 (오류 삽입 레코드와 그 후손,
       블라인드·이중 측정 레코드 제외).
    2. 제외 종류(`excluded_kinds`, 블러) 빼기.
    3. 사람이 만든 라벨은 상태와 관계없이 넣고, 모델 라벨은 `states`에 든 것만.

    Args:
        history: 한 세션의 라벨 레코드 전체 (스냅샷 labels.jsonl에서 그 세션 몫).
        policy: 내보내기 정책.
        states: `label_states()` 결과.
    """
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
    """데이터셋 버전에서 내보낼 세션·라벨을 고르고, 각 영상 스트림의 블러본 렌더 해시를 확인한다.

    Args:
        conn: DB 연결 (읽기만 한다).
        snapshots: 스냅샷 저장소 (lakeFS 또는 로컬). labels.jsonl·sessions.jsonl을 읽는다.
        version_id: 데이터셋 버전 ID.
        policy: 내보내기 정책.
        include_unreviewed: 미검수 모델 라벨도 넣을지 (명시적 옵션).
        splits: 내보낼 분할. None이면 정책 `splits`(기본 train·val).
        privacy: 블러본 해시를 계산할 프라이버시 정책 (없으면 config/policies/privacy.yaml).

    Returns:
        `ExportSource` (세션은 ID 순).

    Raises:
        ExportError: 영상 스트림이 지금 프라이버시 승인 상태가 아닐 때 (`RenderNotCurrentError`).
        SnapshotError / FileNotFoundError: labels.jsonl을 읽을 수 없을 때.
        KeyError 등: 버전이 없을 때 (`get_dataset_version`).
    """
    version = get_dataset_version(conn, version_id)
    wanted = splits or policy.splits
    # 사용 중지는 스냅샷이 아니라 지금 DB 기준 (버전을 만든 뒤 철회한 세션도 뺀다)
    withdrawn = withdrawn_session_ids(conn)
    chosen = {
        sid: sp for sid, sp in version.splits.items() if sp in wanted and sid not in withdrawn
    }
    by_session: dict[str, list[LabelRecord]] = {sid: [] for sid in chosen}
    # 라벨 이력: 스냅샷 labels.jsonl을 임시 파일로 받아 한 줄씩 읽는다 (고른 세션 몫만 남긴다)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.jsonl"
        snapshots.read(version.snapshot_uri, "labels.jsonl", path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = LabelRecord.model_validate_json(line)
                    if x.session_id in by_session:
                        by_session[x.session_id].append(x)
    # 세션 메타데이터(스트림·동기화 오프셋)도 스냅샷 것을 쓴다: 빌드 뒤 다시 동기화해도 같은 결과
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

    Args:
        labeling: 라벨링 버킷 저장소 (블러본·렌더 기록이 있는 곳).
        es: 내보낼 세션 (`render_hashes`에 이 스트림이 있어야 한다).
        stream: 받을 영상 스트림.
        work: 작업 디렉터리. `work/blurred/<세션>/<스트림>.mp4`에 받는다. 이미 있고 sha256이
            저장소 기록과 같으면 다시 받지 않는다 (COCO·LeRobot이 같은 세션을 여러 번 부른다).

    Returns:
        받은 블러본 로컬 경로.

    Raises:
        ExportError: 블러본이 없거나, 승인 상태를 확인하지 않은 스트림이거나, 렌더 기록·파일 해시가
            지금 기대값과 다를 때.
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
        # 1) 렌더 기록(블러 라벨·정책 해시 + 파일 해시)이 지금 기대값과 같은지
        meta = read_render_meta(labeling, session.session_id, stream.stream_id, work)
        rendered = check_render_meta(meta, expected, labeling, session.session_id, stream.stream_id)
        # 2) 받기 (캐시가 저장소 해시와 다르면 다시 받는다)
        if not dest.exists() or (head.sha256 is not None and sha256_file(dest) != head.sha256):
            dest.parent.mkdir(parents=True, exist_ok=True)
            labeling.get_file(key, dest)
        # 3) 받은 파일 자체가 렌더 기록의 파일 해시와 같은지 (렌더 밖에서 덮어쓴 블러본 거부)
        check_fetched(dest, rendered, session.session_id, stream.stream_id)
    except RenderNotCurrentError as exc:
        raise ExportError(str(exc)) from exc
    return dest


def assert_no_raw(out: Path, raw_bucket: str) -> None:
    """내보내기 결과의 모든 파일에 원본 위치가 없는지 본다 (parquet·영상 포함, 바이트로).

    찾는 문자열: `s3://<원본 버킷>`, `local://<원본 버킷>`, `<원본 버킷>/sessions/`.
    텍스트 형식만이 아니라 바이너리(parquet·mp4 메타데이터)에 섞여 들어간 경우도 잡으려고
    바이트로 본다.

    Raises:
        ExportError: 한 파일이라도 원본 위치를 담고 있을 때 (첫 번째 것을 알린다).
    """
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
    """`root` 아래 모든 파일을 경로 순으로 (manifest 파일 목록·올리기 순서가 늘 같게)."""
    return (p for p in sorted(root.rglob("*")) if p.is_file())
