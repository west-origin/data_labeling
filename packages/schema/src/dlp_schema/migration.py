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

위치
    WP1(온톨로지 이관 프레임워크), ADR 0002·0028. 같은 버전의 초안에 키를 덧붙이기만 하는 변경은
    이관 없이 `db.repository.register_ontology`가 받는다. 키 삭제·이름 변경은 새 버전 + 이 모듈의
    이관이 필요하다.

주요 이름
    - `Category`: 매핑 표에서 쓸 수 있는 사전 범주.
    - `OntologyMigration`: 매핑 표 (YAML 한 파일).
    - `MigrationResult`: 이관된 새 레코드와 사람이 다시 볼 라벨 목록.
    - `load_migration`, `migrate_labels`.

주의
    - 순수 함수다. DB에 쓰지 않는다. 호출자가 결과 레코드를 `insert_labels`로 저장한다.
    - 같은 입력을 다시 이관하면 같은 ID(`derived_id(<원래>, "v<새 버전>")`)가 나와 멱등이다.
      이미 새 버전인 현재 라벨은 건너뛴다.
    - 객체 클래스 이름이 바뀌면 `object_state.class_id`는 바뀌지만 `attribute`·`value`는 그대로다.
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

# 매핑 표의 범주. 대부분 `Ontology`의 같은 이름 사전이고, 예외는 둘이다.
#   substeps: 작업 사전 안의 하위 단계 ID (segment level=substep의 ref_id).
#   parts: 부분 ID 전체 (도구 작용부·파지부, 표면 부분, 손 관절. `Ontology.known_parts`).
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

# 페이로드 종류별로 어떤 필드가 어떤 사전을 참조하는지.
# 여기에 없는 종류·필드는 이관 때 그대로 복사된다 (예: keypoint_track, coverage, description,
# action.pre_state/post_state, object_state.attribute/value).
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
# segment의 ref_id는 level에 따라 참조하는 사전이 다르다.
_SEGMENT_CATEGORY: dict[str, Category] = {
    "skill": "verbs",
    "task": "tasks",
    "substep": "substeps",
}


# 온톨로지 버전 간 매핑 표 (config/ontology/migrations/<from>__<to>.yaml).
#   from_version / to_version: 이관 전·후 온톨로지 버전.
#   renames: 범주 → {옛 ID: 새 ID}. 목록에 없는 ID는 그대로 둔다.
#   removed: 범주 → 새 버전에서 없어진 ID. 이런 값을 가진
#     라벨은 이관하지 않고 needs_review로 보낸다.
class OntologyMigration(Contract):
    from_version: SemVer
    to_version: SemVer
    renames: dict[Category, dict[OntologyId, OntologyId]] = Field(default={})
    removed: dict[Category, tuple[OntologyId, ...]] = Field(
        default={}, description="새 버전에 대응 ID가 없어 사람이 다시 라벨링할 ID"
    )


# 이관 결과.
#   migrated: 새 버전의 새 레코드 (parent_label_id = 원래 라벨, 검증 상태 미검수).
#   needs_review: (원래 label_id, 사유) — 이관하지 못해 사람이 다시 라벨링·확인할 라벨.
class MigrationResult(Contract):
    migrated: tuple[LabelRecord, ...]
    needs_review: tuple[tuple[str, str], ...] = Field(description="(label_id, 사유)")


def load_migration(path: Path) -> OntologyMigration:
    """매핑 표 YAML을 읽어 검증한다.

    Raises:
        FileNotFoundError, yaml.YAMLError, pydantic.ValidationError (모르는 범주 등).
    """
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return OntologyMigration.model_validate(data)


def migrate_labels(
    labels: list[LabelRecord], migration: OntologyMigration, now: datetime
) -> MigrationResult:
    """이력의 현재 라벨을 매핑 표대로 새 온톨로지 버전의 새 레코드로 이관한다.

    Args:
        labels: 세션의 라벨 이력 (전체 이력이어도 된다. 여러 세션이 섞여도 동작은 하지만
            보통 세션 하나).
        migration: 매핑 표.
        now: 새 레코드의 created_at (시간대 필수. 없으면 새 레코드 검증이 실패해
            needs_review로 간다).

    Returns:
        `MigrationResult`. 이관 대상이 아닌 옛 버전 라벨, 제거된 ID를 쓴 라벨, 이관 후 계약 검증에
        실패한 라벨은 needs_review에 사유와 함께 들어간다.

    부작용: 없음 (DB에 쓰지 않는다).
    """
    migrated: list[LabelRecord] = []
    review: list[tuple[str, str]] = []
    for label in current_labels(labels, operational=False):
        if label.ontology_version == migration.to_version:
            continue  # 이미 새 버전인 라벨 (이관을 다시 돌린 경우)
        if label.ontology_version != migration.from_version:
            review.append((label.label_id, f"버전 {label.ontology_version}은 이관 대상이 아닙니다"))
            continue
        # 페이로드를 JSON 사전으로 풀어 ID 필드만 바꾼 뒤 새 레코드로 다시 검증한다.
        payload: dict[str, Any] = label.payload.model_dump(mode="json")
        fields = dict(_FIELDS.get(label.kind, {}))
        if label.kind == "segment":
            fields["ref_id"] = _SEGMENT_CATEGORY[payload["level"]]
        reason = _apply(payload, fields, migration)
        if reason:
            review.append((label.label_id, reason))
            continue
        # 새 레코드: 원래 라벨의 사본 + 새 ID·부모·버전·미검수 검증·새 생성 시각.
        # seeded_error·measurement 표시는 그대로 복사된다 (운영 여부 유지). 현재 라벨만 오므로
        # retracted는 늘 False다.
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
    """페이로드 사전의 ID 필드를 매핑 표대로 바꾼다 (제자리 수정).

    Args:
        payload: JSON으로 푼 페이로드 사전. 이 함수가 값을 바꾼다.
        fields: 필드 이름 → 범주.
        m: 매핑 표.

    Returns:
        제거된 ID를 만나면 그 사유 문자열 (이때 payload는 일부만 바뀐 상태일 수 있다).
        문제가 없으면 빈 문자열.
    """
    for field, category in fields.items():
        value = payload.get(field)
        if value is None:
            continue
        if value in m.removed.get(category, ()):
            return f"{category}.{value}가 새 버전에서 제거되었습니다"
        payload[field] = m.renames.get(category, {}).get(value, value)
    return ""
