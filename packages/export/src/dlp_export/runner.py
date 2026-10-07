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
from dlp_schema.labels import VerificationState
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


def _label_counts(src: ExportSource) -> dict[str, int]:
    c = Counter(f"{x.kind}/{x.verification.state.value}" for es in src.sessions for x in es.labels)
    return dict(sorted(c.items()))


def _lerobot(
    root: Path,
    src: ExportSource,
    policy: ExportPolicy,
    ontology: Ontology,
    labeling: ObjectStore,
    out: Path,
    work: Path,
) -> dict[str, Any]:
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
        size = size or (info.width, info.height)
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
    return {"episodes": len(episodes), "frames": check["frames"], "loader_check": check}


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
        if fmt == "intervals":
            details: dict[str, Any] = {
                "labels_per_session": write_intervals(
                    src, policy, out, export_id=export_id, now=now
                )
            }
        elif fmt == "coco":
            r = write_coco(src, policy, ontology, labeling, out, work, export_id=export_id, now=now)
            details = {"images": r.images, "annotations": r.annotations, "dropped": r.dropped}
        else:
            details = _lerobot(root, src, policy, ontology, labeling, out, work)
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
            ],
            "label_counts": _label_counts(src),
            "details": details,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8"
        )
        assert_no_raw(out, raw_bucket)
        files = 0
        for p in walk_files(out):
            datasets.put_file(
                f"exports/{export_id}/{p.relative_to(out).as_posix()}", p, sha256_file(p)
            )
            files += 1
    record = record_export(
        conn,
        export_id=export_id,
        dataset_version_id=version_id,
        target=target,
        format=fmt,
        uri=datasets.uri(f"exports/{export_id}"),
        splits=splits or policy.splits,
        label_states=src.label_states,
        now=now,
    )
    if set(record.session_ids) != {es.session.session_id for es in src.sessions}:
        raise ExportError("내보낸 세션과 이력의 세션이 다릅니다")
    return ExportResult(record, files, _label_counts(src), details)
