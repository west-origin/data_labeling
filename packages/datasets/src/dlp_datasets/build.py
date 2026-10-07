"""데이터셋 버전 빌드.

1. 후보: 온톨로지 버전이 같고 프라이버시 승인된 세션 (도메인을 주면 그 도메인만).
2. 사용 중지된 세션은 빼고 excluded_sessions에 적는다.
3. 골든셋 세션은 golden, 나머지는 작업자·장소 단위로 train / val / holdout.
4. 스냅샷: 포함 세션의 라벨 레코드 전체(수정 이력 포함, 오류 삽입 과제 제외), 세션 메타데이터
   (스트림·동기화)와 매니페스트를 커밋한다.
5. 데이터셋 버전과 분할을 DB에 쓰고, 사람 검증을 마친 세션은 생애주기를
   분할 배정으로 옮긴다.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_datasets.policy import DatasetPolicy
from dlp_datasets.snapshot import SnapshotStore
from dlp_datasets.splitter import SplitReport, assign_splits, check_isolation
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.db.repository import (
    get_golden_set,
    get_labels,
    get_session,
    insert_dataset_version,
    list_session_ids,
    set_lifecycle,
    withdrawn_session_ids,
)
from dlp_schema.episode import non_operational_ids
from dlp_schema.session import LifecycleState, Session


class DatasetBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    version: DatasetVersion
    report: SplitReport
    label_counts: dict[str, int]


def build_dataset_version(
    conn: sa.Connection,
    snapshots: SnapshotStore,
    policy: DatasetPolicy,
    *,
    version_id: str,
    ontology_version: str,
    golden_set_version: str | None,
    domain: str | None = None,
    parent_version_id: str | None = None,
    seed: int = 0,
    now: datetime,
) -> BuildResult:
    withdrawn = withdrawn_session_ids(conn)
    candidates: list[Session] = []
    excluded: list[str] = []
    for sid in list_session_ids(conn, ontology_version):
        s = get_session(conn, sid)
        if domain is not None and s.domain.value != domain:
            continue
        if s.privacy_state.value != policy.eligible_privacy_state:
            continue
        if sid in withdrawn or s.lifecycle_state is LifecycleState.WITHDRAWN:
            excluded.append(sid)
            continue
        candidates.append(s)
    if not candidates:
        raise DatasetBuildError("데이터셋에 넣을 세션이 없습니다")

    golden: list[str] = []
    if golden_set_version is not None:
        g = get_golden_set(conn, golden_set_version)
        if domain is not None and g.domain.value != domain:
            raise DatasetBuildError(f"골든셋 {g.version}의 도메인({g.domain.value})이 다릅니다")
        golden = [sid for sid in g.session_ids if sid not in withdrawn]
    splits, report = assign_splits(candidates, golden, val_ratio=policy.val_ratio, seed=seed)
    leaks = check_isolation(candidates, splits)
    if leaks:
        raise DatasetBuildError(f"분할 사이 작업자·장소가 겹칩니다: {leaks[:5]}")

    label_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = Path(tmp) / "labels.jsonl"
        with labels_path.open("w", encoding="utf-8") as f:
            for s in sorted(candidates, key=lambda x: x.session_id):
                labels = get_labels(conn, s.session_id)
                # 오류 삽입 레코드와 그 후손, 블라인드·이중 라벨링 측정 레코드는 넣지 않는다
                excluded_ids = non_operational_ids(labels)
                for label in labels:
                    if label.label_id in excluded_ids:
                        continue
                    f.write(label.model_dump_json() + "\n")
                    label_counts[f"{label.kind}/{label.verification.state.value}"] += 1
        manifest = {
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "ontology_version": ontology_version,
            "golden_set_version": golden_set_version,
            "domain": domain,
            "created_at": now.isoformat(),
            "splits": {k: v.value for k, v in sorted(splits.items())},
            "split_counts": report.counts,
            "excluded_sessions": sorted(excluded),
            "label_counts": dict(sorted(label_counts.items())),
        }
        # 세션 메타데이터(스트림·동기화)도 고정한다:
        # 같은 버전으로 내보내면 그 뒤에 다시 동기화했어도 같은 결과가 나온다
        sessions_path = Path(tmp) / "sessions.jsonl"
        sessions_path.write_text(
            "".join(
                s.model_dump_json() + "\n" for s in sorted(candidates, key=lambda x: x.session_id)
            ),
            encoding="utf-8",
        )
        manifest_path = Path(tmp) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        uri = snapshots.commit(
            f"datasets/{version_id}",
            {
                "labels.jsonl": labels_path,
                "sessions.jsonl": sessions_path,
                "manifest.json": manifest_path,
            },
            f"dataset {version_id}",
            {"version_id": version_id, "ontology_version": ontology_version},
        )

    version = DatasetVersion(
        version_id=version_id,
        parent_version_id=parent_version_id,
        ontology_version=ontology_version,
        created_at=now,
        snapshot_uri=uri,
        splits=splits,
        excluded_sessions=tuple(sorted(excluded)),
        golden_set_version=golden_set_version,
    )
    insert_dataset_version(conn, version)
    for s in candidates:
        if (
            s.lifecycle_state is LifecycleState.HUMAN_VERIFIED
            and splits[s.session_id] is not Split.HOLDOUT
        ):
            set_lifecycle(conn, s.session_id, LifecycleState.SPLIT_ASSIGNED)
    return BuildResult(version, report, dict(label_counts))
