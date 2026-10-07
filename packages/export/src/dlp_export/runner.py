"""내보내기 실행 (`dlp export <형식> <데이터셋 버전>`).

1. 데이터셋 버전에서 세션·라벨을 고른다 (검증 정책, 사용 중지 제외, 블러 라벨 제외).
2. 형식별로 쓴다: coco / intervals / lerobot
   (lerobot은 격리 환경에서 공식 API로 쓰고 공식 로더로 다시 읽는다).
3. manifest.json (대상, 일시, 형식, 검증 정책, 세션, 라벨 수)을 넣고, 원본 버킷 위치가 없는지
   확인한다.
4. 데이터셋 버킷 exports/<내보내기 ID>/ 에 올리고 내보내기 이력(exports)을 남긴다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa

from dlp_datasets.lineage import record_export
from dlp_datasets.snapshot import SnapshotStore
from dlp_export.coco import write_coco
from dlp_export.intervals import write_intervals
from dlp_export.lerobot import Episode, Vocab, build_episode, run_script, write_package
from dlp_export.policy import ExportPolicy
from dlp_export.source import (
    ExportError,
    ExportSource,
    assert_no_raw,
    fetch_blurred,
    load_source,
    walk_files,
)
from dlp_media.probe import probe
from dlp_media.pts import build_pts_index
from dlp_media.storage import ObjectStore, sha256_file
from dlp_schema.dataset import Split
from dlp_schema.db.repository import withdrawn_session_ids
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.lineage import ExportRecord
from dlp_schema.ontology import Ontology
from dlp_schema.session import Session

Format = Literal["coco", "intervals", "lerobot"]


@dataclass
class ExportResult:
    record: ExportRecord
    files: int
    label_counts: dict[str, int]  # "종류/검증 상태" → 수
    details: dict[str, Any] = field(default_factory=dict[str, Any])


def export_id_for(
    version_id: str,
    fmt: str,
    target: str,
    states: tuple[VerificationState, ...],
    splits: tuple[Split, ...],
    now: datetime,
) -> str:
    """같은 시각이라도 형식·대상·검증 정책·분할이 다르면 다른 ID (결과 폴더가 겹치지 않게)."""
    key = "|".join([version_id, fmt, target, ",".join(states), ",".join(splits), now.isoformat()])
    tag = hashlib.sha256(key.encode()).hexdigest()[:10]
    return f"export-{fmt}-{tag}"


def _label_counts(labels: list[LabelRecord]) -> dict[str, int]:
    c = Counter(f"{x.kind}/{x.verification.state.value}" for x in labels)
    return dict(sorted(c.items()))


# LeRobot 특징을 만드는 라벨 종류 (내보낸 라벨 수를 셀 때 쓴다)
LEROBOT_KINDS = ("keypoint_track", "trajectory3d", "hand_state", "action", "relation", "segment")


def _lerobot(
    root: Path,
    src: ExportSource,
    policy: ExportPolicy,
    ontology: Ontology,
    labeling: ObjectStore,
    out: Path,
    work: Path,
) -> tuple[list[LabelRecord], set[str], dict[str, Any]]:
    """(내보낸 라벨, 내보낸 세션, 세부 정보)."""
    vocab = Vocab.from_ontology(ontology)
    episodes: list[tuple[Session, Path, Episode]] = []
    size: tuple[int, int] | None = None
    sessions: list[dict[str, Any]] = []
    for es in src.sessions:
        stream = next(
            (s for s in es.session.streams if s.kind.value == policy.lerobot.video_stream), None
        )
        if stream is None:
            continue
        video = fetch_blurred(labeling, es.session, stream, work)
        info = probe(video).video
        assert info is not None
        if size is None:
            size = (info.width, info.height)
        elif abs(info.width / info.height - size[0] / size[1]) > 0.01:
            # 다른 화면비를 한 크기로 맞추면 영상이 찌그러진다
            # (2D 관절은 정규화라 맞지만 화면은 틀린다)
            raise ExportError(
                f"{es.session.session_id}: 화면비가 다른 에피소드는 한 LeRobot 데이터셋에 "
                f"넣지 않습니다 ({info.width}x{info.height} vs {size[0]}x{size[1]}). "
                "화면비별로 나눠 내보내세요"
            )
        ep = build_episode(
            es.session, stream, build_pts_index(video), (info.width, info.height), es.labels, vocab,
            policy.lerobot,
        )  # fmt: skip
        episodes.append((es.session, video, ep))
        sessions.append(
            {
                "episode_index": len(sessions),
                "session_id": es.session.session_id,
                "split": es.split.value,
                "stream_id": stream.stream_id,
                "start_ms": float(ep.times_ms[0]),
                "frames": len(ep.times_ms),
            }
        )
    if not episodes or size is None:
        raise ExportError(f"{policy.lerobot.video_stream} 스트림이 있는 세션이 없습니다")
    written = {s["session_id"] for s in sessions}
    pkg = write_package(episodes, policy, size, work / "lerobot_pkg")
    dest = out / "lerobot"
    run_script(root, policy.lerobot, "lerobot_write.py", str(pkg), str(dest))
    check = json.loads(
        run_script(root, policy.lerobot, "lerobot_check.py", str(dest), policy.lerobot.repo_id)
    )
    if check["episodes"] != len(episodes):
        raise ExportError(f"LeRobot 로더가 읽은 에피소드 수가 다릅니다: {check['episodes']}")
    (dest / "meta" / "dlp_vocab.json").write_text(
        json.dumps(vocab.as_json(), ensure_ascii=False, indent=2), "utf-8"
    )
    (dest / "meta" / "dlp_episodes.json").write_text(
        json.dumps(sessions, ensure_ascii=False, indent=2), "utf-8"
    )
    used = [
        x
        for es in src.sessions
        if es.session.session_id in written
        for x in es.labels
        if x.kind in LEROBOT_KINDS
    ]
    return (
        used,
        written,
        {"episodes": len(episodes), "frames": check["frames"], "loader_check": check},
    )


def run_export(
    conn: sa.Connection,
    *,
    root: Path,
    version_id: str,
    fmt: Format,
    target: str,
    snapshots: SnapshotStore,
    labeling: ObjectStore,
    datasets: ObjectStore,
    raw_bucket: str,
    policy: ExportPolicy,
    ontology: Ontology,
    include_unreviewed: bool,
    splits: tuple[Split, ...] | None,
    now: datetime,
) -> ExportResult:
    if labeling.bucket == raw_bucket or datasets.bucket == raw_bucket:
        raise ExportError("내보내기는 원본 버킷을 읽거나 쓰지 않습니다")
    src = load_source(
        conn, snapshots, version_id, policy, include_unreviewed=include_unreviewed, splits=splits
    )
    if not src.sessions:
        raise ExportError(f"{version_id}: 내보낼 세션이 없습니다")
    export_id = export_id_for(
        version_id, fmt, target, src.label_states, splits or policy.splits, now
    )
    with tempfile.TemporaryDirectory() as tmp:
        out, work = Path(tmp) / "out", Path(tmp) / "work"
        out.mkdir()
        details: dict[str, Any]
        if fmt == "intervals":
            details = {
                "labels_per_session": write_intervals(
                    src, policy, out, export_id=export_id, now=now
                )
            }
            written = {es.session.session_id for es in src.sessions}
            used = [x for es in src.sessions for x in es.labels if x.kind in policy.intervals.kinds]
        elif fmt == "coco":
            r = write_coco(src, policy, ontology, labeling, out, work, export_id=export_id, now=now)
            details = {"images": r.images, "annotations": r.annotations, "dropped": r.dropped}
            written, used = r.sessions, list(r.labels.values())
        else:
            used, written, details = _lerobot(root, src, policy, ontology, labeling, out, work)
        if not written:
            raise ExportError(f"{version_id}: 이 형식으로 쓴 세션이 없습니다")
        counts = _label_counts(used)
        manifest = {
            "export_id": export_id,
            "dataset_version_id": version_id,
            "ontology_version": src.version.ontology_version,
            "format": fmt,
            "target": target,
            "created_at": now.isoformat(),
            "verification_policy": {
                "label_states": [s.value for s in src.label_states],
                "human_created": "always",
                "include_unreviewed": include_unreviewed,
            },
            "splits": [s.value for s in (splits or policy.splits)],
            "sessions": [
                {"session_id": es.session.session_id, "split": es.split.value}
                for es in src.sessions
                if es.session.session_id in written
            ],
            "label_counts": counts,
            "details": details,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8"
        )
        assert_no_raw(out, raw_bucket)
        # 내보내는 동안 사용 중지된 세션이 생겼으면 아무것도 올리지 않는다
        late = written & withdrawn_session_ids(conn)
        if late:
            raise ExportError(f"내보내는 동안 사용 중지된 세션: {sorted(late)}. 다시 내보내세요")
        # 이력을 먼저 남기고(같은 트랜잭션), manifest.json은 마지막에 올린다. 중간에 실패하면
        # 트랜잭션이 되돌아가고 manifest 없는 폴더만 남는다 (manifest 없는 폴더 = 미완성 내보내기).
        record = record_export(
            conn,
            export_id=export_id,
            dataset_version_id=version_id,
            target=target,
            format=fmt,
            uri=datasets.uri(f"exports/{export_id}"),
            splits=splits or policy.splits,
            label_states=src.label_states,
            session_ids=tuple(sorted(written)),
            now=now,
        )
        files = 0
        manifest_path = out / "manifest.json"
        for p in [*(q for q in walk_files(out) if q != manifest_path), manifest_path]:
            datasets.put_file(
                f"exports/{export_id}/{p.relative_to(out).as_posix()}", p, sha256_file(p)
            )
            files += 1
    return ExportResult(record, files, counts, details)
