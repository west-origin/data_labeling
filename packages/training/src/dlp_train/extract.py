"""데이터셋 버전에서 과제별 학습 예제를 뽑는다.

- 분할: 정책의 splits(학습·검증)에 든 세션만. 골든·holdout 세션은 절대 넣지 않는다.
- 라벨: 운영 라벨(`current_labels`)만. 오류 삽입 레코드와 그 후손, 측정용 레코드는 빠진다.
- 예제가 되는 라벨: 사람이 만든 라벨, 또는 정책의 trainable_states(승인·수정·표본 검증) 상태의 라벨.
- 자동 원본과 수정본의 차이: 각 예제에 처음 모델이 낸 레코드(origin)와 변화 종류를 붙인다.
  - accepted: 모델 라벨이 그대로 승인·표본 검증됨
  - corrected: 모델 라벨을 사람이 고침 (origin = 모델 원본)
  - added: 모델이 놓친 것을 사람이 추가함 (origin 없음)
  - deleted: 모델 라벨을 사람이 지움 (오탐, label = 지운 레코드)
"""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dlp_datasets.snapshot import SnapshotStore
from dlp_eval.policy import Task
from dlp_eval.runner import TASK_KINDS
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import (
    ActionPayload,
    BlurTrackPayload,
    BoxTrackPayload,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    Source,
)
from dlp_train.policy import TrainingPolicy

Change = Literal["accepted", "corrected", "added", "deleted"]


@dataclass(frozen=True)
class Example:
    session_id: str
    split: Split
    change: Change
    label: LabelRecord
    origin: LabelRecord | None  # 처음 모델이 낸 레코드 (사람이 추가한 것은 없음)


@dataclass(frozen=True)
class TrainingData:
    dataset_version_id: str
    task: Task
    examples: list[Example]

    @property
    def session_ids(self) -> set[str]:
        return {e.session_id for e in self.examples}

    def counts(self) -> dict[str, int]:
        c = Counter(f"{e.split.value}/{e.change}" for e in self.examples)
        return dict(sorted(c.items()))


def matches(task: Task, label: LabelRecord) -> bool:
    """라벨이 과제의 학습·평가 대상인가 (종류, 손·전신은 골격까지)."""
    if label.kind not in TASK_KINDS[task]:
        return False
    p = label.payload
    if isinstance(p, KeypointTrackPayload):
        return p.skeleton == ("hand21" if task == "hands" else "coco17")
    if isinstance(p, HandStatePayload):
        return p.contact_target_kind != "none"  # 평가와 같게 접촉 구간만
    return True


def class_key(label: LabelRecord) -> str:
    """학습기가 배우는 클래스 (stub 학습기는 본 클래스만 예측한다)."""
    p = label.payload
    match p:
        case BoxTrackPayload():
            return p.class_id
        case BlurTrackPayload():
            return p.target
        case KeypointTrackPayload():
            return f"{p.skeleton}/{p.hand.value if p.hand else 'body'}"
        case HandStatePayload():
            return p.grasp_type or p.contact_target_kind
        case ActionPayload():
            return p.verb
        case _:
            return label.kind


def _root(label: LabelRecord, by_id: dict[str, LabelRecord]) -> LabelRecord:
    seen: set[str] = set()
    x = label
    while x.parent_label_id and x.parent_label_id in by_id and x.label_id not in seen:
        seen.add(x.label_id)
        x = by_id[x.parent_label_id]
    return x


def extract_examples(
    labels: list[LabelRecord],
    splits: dict[str, Split],
    task: Task,
    policy: TrainingPolicy,
) -> list[Example]:
    """세션 여러 개의 라벨 이력에서 학습 예제를 뽑는다 (분할에 없는 세션은 버린다)."""
    by_session: dict[str, list[LabelRecord]] = {}
    for x in labels:
        by_session.setdefault(x.session_id, []).append(x)
    out: list[Example] = []
    for sid in sorted(by_session):
        split = splits.get(sid)
        if split is None or split not in policy.splits:
            continue
        history = by_session[sid]
        by_id = {x.label_id: x for x in history}
        excluded = non_operational_ids(history)
        for x in current_labels(history):
            if not matches(task, x):
                continue
            human = x.provenance.source is Source.HUMAN
            if not human and x.verification.state not in policy.trainable_states:
                continue
            root = _root(x, by_id)
            if root.provenance.source is Source.MODEL:
                change: Change = "accepted" if root.label_id == x.label_id else "corrected"
                out.append(Example(sid, split, change, x, root))
            else:
                out.append(Example(sid, split, "added", x, None))
        for r in history:
            # 사람이 지운 모델 라벨 = 오탐 예제. 모델 버전이 바뀌어 지운 것(출처 model)은 아니다
            if (
                not r.retracted
                or r.provenance.source is not Source.HUMAN
                or r.label_id in excluded
                or r.parent_label_id not in by_id
            ):
                continue
            gone = by_id[r.parent_label_id]
            root = _root(gone, by_id)
            if matches(task, gone) and root.provenance.source is Source.MODEL:
                out.append(Example(sid, split, "deleted", gone, root))
    return out


def load_training_data(
    snapshots: SnapshotStore, version: DatasetVersion, task: Task, policy: TrainingPolicy
) -> TrainingData:
    """데이터셋 버전 스냅샷(labels.jsonl)과 버전의 분할로 학습 예제를 만든다."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.jsonl"
        snapshots.read(version.snapshot_uri, "labels.jsonl", path)
        with path.open(encoding="utf-8") as f:
            labels = [LabelRecord.model_validate_json(line) for line in f if line.strip()]
    examples = extract_examples(labels, dict(version.splits), task, policy)
    return TrainingData(version.version_id, task, examples)
