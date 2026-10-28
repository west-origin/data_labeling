"""온톨로지 버전 간 라벨 이관.

매핑 표(config/ontology/migrations/<from>__<to>.yaml)로 ID를 바꾼다. 원본 라벨은 두고
새 버전의 새 레코드를 만들며(parent_label_id로 연결), 목적지 사전으로 의미를 검사하고
검증 상태는 미검수로 되돌린다. 잘못된 목적지 ID는 needs_review로 반환한다.
골든셋은 이관 후 재검수를 거쳐야 평가에 쓸 수 있기 때문이다.

입력은 세션의 전체 수정 이력이어도 된다. 이관 대상은 이력의 현재 라벨뿐이다
(`current_labels(operational=False)`): 이미 수정된 레코드, 삭제된 레코드, 삭제 레코드 자체를
다시 이관하면 지운 라벨이 되살아나기 때문이다. 오류 삽입·측정 레코드는 표시를 그대로 지닌 채
이관되므로 이관 후에도 운영 라벨이 아니다.

이관 결과의 구간·부분 연결 검사(`check_label_set`)는 묶음마다 따로 한다: 운영 라벨, 오류 삽입
배정별 사본(과 그 후손), 측정 레코드(방식·검수자별). 사본은 원래 라벨의 action_id·segment_id를
그대로 쓰므로 섞어 검사하면 ID 중복이 되어 운영 라벨까지 재검수로 갔다 (F-SCH-03). 한 묶음이
깨지면 그 묶음의 이번 배치만 재검수로 돌린다.

라벨 사이 참조(`labels.LABEL_REF_FIELDS`: 번역 설명의 source_label_id, 3D 궤적·마스크의
derived_from)는 같은 배치에서 이관되어 남은 라벨을 가리키면 새 ID로 바꾼다 (F-SCH-29). 옛 ID로
두면 이관된 번역이 이관 전 원문을, 이관된 3D가 이관 전 2D를 가리켜, 새 2D 검수를 3D가 물려받지
못한다. 이관되지 못한(재검수로 간) 라벨을 가리키는 참조는 그대로 둔다 (그 라벨이 여전히 현재 라벨).

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

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, ValidationError

from dlp_schema.common import Contract, OntologyId, SemVer, derived_id
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import LABEL_REF_FIELDS, LabelRecord, Verification
from dlp_schema.ontology import Ontology
from dlp_schema.validation import check_label, check_label_set

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
    labels: list[LabelRecord],
    migration: OntologyMigration,
    now: datetime,
    *,
    target_ontology: Ontology,
) -> MigrationResult:
    """이력의 현재 라벨을 매핑 표대로 새 온톨로지 버전의 새 레코드로 이관한다.

    Args:
        labels: 세션의 라벨 이력 (전체 이력이어도 된다. 여러 세션이 섞여도 동작은 하지만
            보통 세션 하나).
        migration: 매핑 표.
        now: 새 레코드의 created_at (시간대 필수. 없으면 새 레코드 검증이 실패해
            needs_review로 간다).
        target_ontology: 목적지 사전. 버전이 이관 표와 다르면 작업을 거부한다.

    Returns:
        `MigrationResult`. 이관 대상이 아닌 옛 버전 라벨, 제거된 ID를 쓴 라벨, 이관 후 계약 검증에
        실패한 라벨은 needs_review에 사유와 함께 들어간다.

    부작용: 없음 (DB에 쓰지 않는다).
    """
    if target_ontology.version != migration.to_version:
        raise ValueError("이관 목적지와 온톨로지 버전이 다릅니다")
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
            candidate = LabelRecord.model_validate(record)
            problems = check_label(candidate, target_ontology)
            if problems:
                review.append(
                    (label.label_id, "이관 후 온톨로지 검증 실패: " + "; ".join(problems))
                )
            else:
                migrated.append(candidate)
        except ValidationError as exc:
            review.append((label.label_id, f"이관 후 검증 실패: {exc.error_count()}건"))
    kept, broken = _check_sets(labels, migrated, migration, target_ontology)
    return MigrationResult(migrated=tuple(_relink(kept)), needs_review=tuple([*review, *broken]))


def _relink(kept: list[LabelRecord]) -> list[LabelRecord]:
    """이관 레코드의 라벨 참조(`LABEL_REF_FIELDS`)를 같은 배치의 새 ID로 바꾼다 (F-SCH-29).

    구간·부분 검사(`_check_sets`)를 통과해 남은 레코드만 대응표에 넣는다. 재검수로 빠진 라벨은
    새 레코드가 없으니 옛 ID(여전히 현재 라벨)를 그대로 가리킨다. 바뀐 페이로드는 다시 검증한다.

    Args:
        kept: 남긴 이관 레코드 (parent_label_id = 원래 라벨).

    Returns:
        같은 순서의 레코드. 참조가 바뀐 것만 새 객체다.
    """
    renamed = {x.parent_label_id: x.label_id for x in kept if x.parent_label_id is not None}
    out: list[LabelRecord] = []
    for label in kept:
        payload: dict[str, Any] = label.payload.model_dump(mode="json")
        changed = {
            field: new
            for field in LABEL_REF_FIELDS & payload.keys()
            if (new := _renamed_ref(payload[field], renamed)) is not None
        }
        if changed:
            record = label.model_dump(mode="json")
            record["payload"] = {**payload, **changed}
            label = LabelRecord.model_validate(record)
        out.append(label)
    return out


def _renamed_ref(value: object, renamed: dict[str, str]) -> str | list[str] | None:
    """참조 값(ID 하나 또는 ID 목록)을 대응표로 바꾼 값. 바뀐 것이 없으면 None."""
    if isinstance(value, str):
        return renamed.get(value)
    if isinstance(value, list):
        ids = [str(v) for v in cast(list[object], value)]
        if any(v in renamed for v in ids):
            return [renamed.get(v, v) for v in ids]
    return None


# 오류 삽입 사본 ID 형식: `seed-<배정>-NNNN`(사본, `dlp_review.ops.seeding.seed_prefix`)과
# `seed-<배정>-new-<sha256 16자>`(오류 삽입 과제에서 검수자가 새로 그린 레코드, 수집 때 붙인다).
# dlp_schema는 dlp_review를 import하지 않으므로 형식만 여기서 읽는다. 형식이 다르면 그 계보
# 하나를 묶음으로 본다 (보수적: 같은 배정의 다른 사본을 못 찾으면 그 사본이 재검수로 간다).
_SEED_ID = re.compile(r"^seed-(?P<assignment>.+)-(?:\d{4}|new-[0-9a-f]{16})$")


def _set_key(label: LabelRecord, by_id: dict[str, LabelRecord], excluded: set[str]) -> str:
    """집합 검사 묶음 키: 운영 라벨, 오류 삽입 배정, 측정 (방식, 검수자).

    비운영 레코드는 계보에서 가장 위의 표시 레코드(seeded_error 또는 measurement가 있는 것)가
    묶음을 정한다. 검수자가 고친 레코드와 이관 레코드는 그 조상의 묶음을 따른다.

    Args:
        label: 묶을 현재 레코드.
        by_id: 이력 + 이번 이관 결과의 label_id 색인 (조상 찾기).
        excluded: 운영 라벨이 아닌 ID (`non_operational_ids`).
    """
    if label.label_id not in excluded:
        return "operational"
    origin = label
    node: LabelRecord | None = label
    seen: set[str] = set()
    # 사슬은 보통 짧지만 잘못된 이력의 순환에 대비해 방문한 ID에서 멈춘다
    while node is not None and node.label_id not in seen:
        seen.add(node.label_id)
        if node.seeded_error or node.measurement is not None:
            origin = node
        node = by_id.get(node.parent_label_id or "")
    if origin.seeded_error:
        match = _SEED_ID.match(origin.label_id)
        return f"seeded:{match['assignment'] if match else origin.label_id}"
    return f"measurement:{origin.measurement}:{origin.verification.reviewer_id or ''}"


def _check_sets(
    labels: list[LabelRecord],
    migrated: list[LabelRecord],
    migration: OntologyMigration,
    target: Ontology,
) -> tuple[list[LabelRecord], list[tuple[str, str]]]:
    """이관 결과의 구간·부분 연결을 묶음(`_set_key`)마다 검사한다.

    자식이 제거/재검수로 빠졌다면 부모만 새 버전으로 확정하지 않는다. 이미 새 버전인 현재 라벨도
    같은 묶음의 참조 문맥으로 넣고, 깨진 묶음의 이번 배치는 전부 재검수로 돌린다.

    Returns:
        (남긴 이관 레코드(입력 순서), 재검수 항목 (원래 label_id, 사유)).
    """
    every = [*labels, *migrated]
    by_id = {x.label_id: x for x in every}
    excluded = non_operational_ids(every)
    context = [
        x
        for x in current_labels(labels, operational=False)
        if x.ontology_version == migration.to_version
    ]
    groups: dict[str, list[LabelRecord]] = {}
    for x in [*context, *migrated]:
        groups.setdefault(_set_key(x, by_id, excluded), []).append(x)
    failed = {
        key: problems
        for key, members in groups.items()
        if (problems := check_label_set(members, target))
    }
    kept: list[LabelRecord] = []
    broken: list[tuple[str, str]] = []
    for label in migrated:
        problems = failed.get(_set_key(label, by_id, excluded))
        if problems:
            reason = "이관 후 구간·부분 검증 실패: " + "; ".join(problems)
            broken.append((label.parent_label_id or label.label_id, reason))
        else:
            kept.append(label)
    return kept, broken


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
