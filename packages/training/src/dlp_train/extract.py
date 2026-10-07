"""데이터셋 버전에서 과제별 학습 예제를 뽑는다.

- 분할: 정책의 splits(학습·검증)에 든 세션만. 골든·holdout 세션은 절대 넣지 않는다.
  사용 중지(동의 철회)된 세션은 버전에 있어도 뺀다.
- 라벨: 운영 라벨(`current_labels`)만. 오류 삽입 레코드와 그 후손, 측정용 레코드는 빠진다.
- 예제가 되는 라벨: 사람이 만든 라벨, 또는 정책의 trainable_states(승인·수정·표본 검증) 상태의 모델
  라벨. 센서 출처 현재 라벨(장갑 압력 접촉 등)은 검수 결과가 아니므로 예제가 되지 않는다.
- 자동 원본과 수정본의 차이: 각 예제에 처음 모델이 낸 레코드(origin)와 변화 종류를 붙인다.
  - accepted: 모델 라벨이 그대로 승인·표본 검증됨
  - corrected: 모델 라벨을 사람이 고침 (origin = 모델 원본)
  - added: 모델이 놓친 것을 사람이 추가함 (origin 없음)
  - deleted: 모델 라벨을 사람이 지움 (오탐, label = 지운 레코드)

WP13, ADR 0016·0025. 변화 종류 판정 자체는 `dlp_schema.history.review_changes`가 한다 (액티브 러닝의
수정률과 같은 코드). 입력은 데이터셋 버전 스냅샷의 labels.jsonl이고(`dlp_datasets.snapshot`), DB에는
접근하지 않는다. 원본 버킷에도 접근하지 않는다.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

from dlp_datasets.snapshot import SnapshotStore
from dlp_eval.policy import Task
from dlp_eval.runner import TASK_KINDS
from dlp_schema.dataset import DatasetVersion, Split
from dlp_schema.history import Change, label_class, review_changes
from dlp_schema.labels import HandStatePayload, KeypointTrackPayload, LabelRecord
from dlp_train.policy import TrainingPolicy


@dataclass(frozen=True)
class Example:
    """학습 예제 하나."""

    session_id: str
    split: Split  # train 또는 val
    change: Change  # accepted | corrected | added | deleted
    label: LabelRecord  # 학습에 쓸 레코드 (deleted면 지워진 모델 레코드 = 음성 예제)
    origin: LabelRecord | None  # 처음 모델이 낸 레코드 (사람이 추가한 것은 없음)


@dataclass(frozen=True)
class TrainingData:
    """과제 하나의 학습 데이터 (데이터셋 버전 하나에서)."""

    dataset_version_id: str
    task: Task
    examples: list[Example]

    @property
    def session_ids(self) -> set[str]:
        """예제가 나온 세션 ID (골든셋 누출 검사에 쓴다)."""
        return {e.session_id for e in self.examples}

    def counts(self) -> dict[str, int]:
        """`<분할>/<변화 종류>` → 예제 수 (키 정렬). MLflow 지표·루프 결과에 남긴다."""
        c = Counter(f"{e.split.value}/{e.change}" for e in self.examples)
        return dict(sorted(c.items()))


def matches(task: Task, label: LabelRecord) -> bool:
    """라벨이 과제의 학습·평가 대상인가 (종류, 손·전신은 골격까지).

    - 종류가 `runner.TASK_KINDS[task]`에 있어야 한다.
    - keypoint_track은 hands면 hand21, 그 밖(body)이면 coco17 골격만.
    - hand_state는 접촉 구간(contact_target_kind != "none")만 (`harness._contacts`와 같은 기준).
    """
    if label.kind not in TASK_KINDS[task]:
        return False
    p = label.payload
    if isinstance(p, KeypointTrackPayload):
        return p.skeleton == ("hand21" if task == "hands" else "coco17")
    if isinstance(p, HandStatePayload):
        return p.contact_target_kind != "none"  # 평가와 같게 접촉 구간만
    return True


def class_key(label: LabelRecord) -> str:
    """학습기가 배우는 클래스 (stub 학습기는 본 클래스만 예측한다).

    `dlp_schema.history.label_class`와 같다 (박스는 class_id, 블러는 target, 키포인트는 "골격/손"
    등).
    """
    return label_class(label)


def extract_examples(
    labels: list[LabelRecord],
    splits: dict[str, Split],
    task: Task,
    policy: TrainingPolicy,
    excluded: Set[str] = frozenset(),
) -> list[Example]:
    """세션 여러 개의 라벨 이력에서 학습 예제를 뽑는다.

    분할에 없는 세션과 excluded(사용 중지된 세션)는 버린다.

    Args:
        labels: 여러 세션의 전체 라벨 이력 (삭제·오류 삽입·측정 레코드 포함 그대로).
        splits: 세션 → 분할 (데이터셋 버전의 분할). 정책 splits에 없는 분할
            (golden·holdout)은 버린다.
        task: 과제 (`matches`로 거른다).
        policy: 학습 정책 (`splits`, `trainable_states`).
        excluded: 뺄 세션 ID.

    Returns:
        세션 ID 순으로 정렬된 예제 목록. 순수 함수 (부작용 없음).
    """
    by_session: dict[str, list[LabelRecord]] = {}
    for x in labels:
        by_session.setdefault(x.session_id, []).append(x)
    out: list[Example] = []
    for sid in sorted(by_session):
        split = splits.get(sid)
        if split is None or split not in policy.splits or sid in excluded:
            continue
        for c in review_changes(by_session[sid], policy.trainable_states):
            if matches(task, c.label):
                out.append(Example(sid, split, c.change, c.label, c.origin))
    return out


def load_training_data(
    snapshots: SnapshotStore,
    version: DatasetVersion,
    task: Task,
    policy: TrainingPolicy,
    *,
    excluded: Set[str] = frozenset(),
) -> TrainingData:
    """데이터셋 버전 스냅샷(labels.jsonl)과 버전의 분할로 학습 예제를 만든다.

    excluded: 버전을 만든 뒤 사용 중지(동의 철회 등)된 세션 (`loop.withdrawn_sessions`). 뺀다.

    부작용: 스냅샷 저장소에서 labels.jsonl을 임시 디렉터리로 받는다 (끝나면 지운다). 파일 전체를
    메모리에 읽는다.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.jsonl"
        snapshots.read(version.snapshot_uri, "labels.jsonl", path)
        with path.open(encoding="utf-8") as f:
            labels = [LabelRecord.model_validate_json(line) for line in f if line.strip()]
    examples = extract_examples(labels, dict(version.splits), task, policy, excluded)
    return TrainingData(version.version_id, task, examples)
