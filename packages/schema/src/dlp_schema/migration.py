"""온톨로지 버전 간 라벨 이관.

매핑 표(config/ontology/migrations/<from>__<to>.yaml)로 ID를 바꾼다. 원본 라벨은 두고
새 버전의 새 레코드를 만들며(parent_label_id로 연결), 검증 상태는 미검수로 되돌린다.
골든셋은 이관 후 재검수를 거쳐야 평가에 쓸 수 있기 때문이다.

입력은 세션의 전체 수정 이력이어도 된다. 이관 대상은 이력의 현재 라벨뿐이다
(`current_labels(operational=False)`): 이미 수정된 레코드, 삭제된 레코드, 삭제 레코드 자체를
다시 이관하면 지운 라벨이 되살아나기 때문이다. 오류 삽입·측정 레코드는 표시를 그대로 지닌 채
이관되므로 이관 후에도 운영 라벨이 아니다.

부분 ID(parts: 도구 작용부·파지부, 표면 부분, 손 관절)는 mask_track·trajectory3d의 part와
relation의 subject_part·object_part에서 바꾼다.

상태 속성과 값(pre_state, post_state, object_state.attribute/value)의
이름 변경은 아직 지원하지 않는다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError

from dlp_schema.common import Contract, OntologyId, SemVer, derived_id
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Verification

Category = Literal[
    "verbs",
    "tasks",
    "substeps",
    "objects",
    "privacy_targets",
    "events",
    "gap_types",
    "grasp_types",
    "body_parts",
    "contact_target_kinds",
    "hand_roles",
    "parts",
]

# 페이로드 종류별로 어떤 필드가 어떤 사전을 참조하는지
_FIELDS: dict[str, dict[str, Category]] = {
    "action": {"verb": "verbs", "target_body_part": "body_parts"},
    "box_track": {"class_id": "objects"},
    "mask_track": {"class_id": "objects", "part": "parts"},
    "trajectory3d": {"part": "parts"},
    "relation": {"subject_part": "parts", "object_part": "parts"},
    "object_state": {"class_id": "objects"},
    "blur_track": {"target": "privacy_targets"},
    "hand_state": {
        "grasp_type": "grasp_types",
        "body_part": "body_parts",
        "contact_target_kind": "contact_target_kinds",
        "role": "hand_roles",
    },
    "event": {"event_type": "events"},
    "gap": {"gap_type": "gap_types"},
}
_SEGMENT_CATEGORY: dict[str, Category] = {
    "skill": "verbs",
    "task": "tasks",
    "substep": "substeps",
}


class OntologyMigration(Contract):
    from_version: SemVer
    to_version: SemVer
    renames: dict[Category, dict[OntologyId, OntologyId]] = Field(default={})
    removed: dict[Category, tuple[OntologyId, ...]] = Field(
        default={}, description="새 버전에 대응 ID가 없어 사람이 다시 라벨링할 ID"
    )


class MigrationResult(Contract):
    migrated: tuple[LabelRecord, ...]
    needs_review: tuple[tuple[str, str], ...] = Field(description="(label_id, 사유)")


def load_migration(path: Path) -> OntologyMigration:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OntologyMigration.model_validate(data)


def migrate_labels(
    labels: list[LabelRecord], migration: OntologyMigration, now: datetime
) -> MigrationResult:
    """labels: 세션의 라벨 이력 (현재 라벨만 이관한다)."""
    migrated: list[LabelRecord] = []
    review: list[tuple[str, str]] = []
    for label in current_labels(labels, operational=False):
        if label.ontology_version == migration.to_version:
            continue  # 이미 새 버전인 라벨 (이관을 다시 돌린 경우)
        if label.ontology_version != migration.from_version:
            review.append((label.label_id, f"버전 {label.ontology_version}은 이관 대상이 아닙니다"))
            continue
        payload: dict[str, Any] = label.payload.model_dump(mode="json")
        fields = dict(_FIELDS.get(label.kind, {}))
        if label.kind == "segment":
            fields["ref_id"] = _SEGMENT_CATEGORY[payload["level"]]
        reason = _apply(payload, fields, migration)
        if reason:
            review.append((label.label_id, reason))
            continue
        record: dict[str, Any] = label.model_dump(mode="json")
        record.update(
            label_id=derived_id(label.label_id, f"v{migration.to_version}"),
            parent_label_id=label.label_id,
            ontology_version=migration.to_version,
            verification=Verification().model_dump(mode="json"),
            created_at=now.isoformat(),
            payload=payload,
        )
        try:
            migrated.append(LabelRecord.model_validate(record))
        except ValidationError as exc:
            review.append((label.label_id, f"이관 후 검증 실패: {exc.error_count()}건"))
    return MigrationResult(migrated=tuple(migrated), needs_review=tuple(review))


def _apply(payload: dict[str, Any], fields: dict[str, Category], m: OntologyMigration) -> str:
    for field, category in fields.items():
        value = payload.get(field)
        if value is None:
            continue
        if value in m.removed.get(category, ()):
            return f"{category}.{value}가 새 버전에서 제거되었습니다"
        payload[field] = m.renames.get(category, {}).get(value, value)
    return ""
