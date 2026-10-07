"""검수 결과를 라벨 이력으로 바꾼다.

검수 도구에서 돌아온 항목(ReviewedItem)을 검수에 보낸 원래 라벨과 비교한다.
- 바뀌지 않음: 원래 라벨의 검수 상태만 human_approved로 갱신한다.
- 바뀜: 원래 라벨을 parent로 하는 새 human 레코드 (human_corrected).
- 없어짐: 원래 라벨을 지우는 retracted 레코드 (human_corrected).
- 새로 생김: parent 없는 새 human 레코드 (human_corrected).
새 레코드 ID는 (원래 ID, 검수자, 시각, 내용)의 해시라 같은 검수 결과를 다시 수집해도 같다.

WP6, ADR 0006·0002(라벨은 덮어쓰지 않는다: 수정은 새 레코드 + parent_label_id).
파이프라인 위치: `dlp_review.collect.collect_task`가 도구 결과를 변환기(`cvat`·`labelstudio`)로
`ReviewedItem`으로 바꾼 뒤 이 모듈의 `reconcile`을 부른다. 순수 함수이며 DB를 쓰지 않는다.

공개 이름: `ReviewedItem`, `ReviewOutcome`, `reconcile`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from dlp_schema.labels import (
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)


@dataclass(frozen=True)
class ReviewedItem:
    """검수 도구에서 돌아온 라벨 하나. origin_label_id가 없으면 검수자가 새로 만든 것이다."""

    # 도구에 보낼 때 실어 둔 원래 라벨 ID (CVAT dlp_label_id 속성, Label Studio 결과 id)
    origin_label_id: str | None
    # 스트림 ID. 공간 라벨은 그 영상 스트림, 시간 라벨은 원래 라벨의 값(대개 None 또는 기준 스트림)
    stream_id: str | None
    # 구간 시작·끝 (정수 ms). 공간 라벨은 스트림 PTS, 시간 라벨은 마스터 타임라인 (ADR 0019)
    t_start_ms: int
    t_end_ms: int
    # 라벨 내용 (종류별 페이로드)
    payload: LabelPayload


@dataclass
class ReviewOutcome:
    """reconcile 결과. collect가 DB에 쓴다."""

    # 바뀌지 않아 human_approved로 기록할 원래 라벨 ID
    approved: list[str] = field(default_factory=list[str])
    # 새로 쓸 사람 레코드 (수정·삭제·추가)
    new_records: list[LabelRecord] = field(default_factory=list[LabelRecord])
    # 작업을 보낸 뒤 이미 자식 레코드가 생겨(다른 단계가 지웠거나 다른 작업이 먼저 고침) 검수 결과를
    # 반영하지 않은 원래 라벨 (`collect.drop_retracted`가 채운다)
    stale: list[str] = field(default_factory=list[str])

    @property
    def corrected(self) -> int:
        """수정 레코드 수 (parent가 있고 retracted가 아닌 새 레코드)."""
        return sum(1 for r in self.new_records if r.parent_label_id and not r.retracted)

    @property
    def retracted(self) -> int:
        """삭제 레코드 수 (retracted=True)."""
        return sum(1 for r in self.new_records if r.retracted)

    @property
    def added(self) -> int:
        """검수자가 새로 그린 레코드 수 (parent 없음)."""
        return sum(1 for r in self.new_records if r.parent_label_id is None)


def _same(a: LabelRecord, b: ReviewedItem) -> bool:
    """도구로 보낸 형태(normalize 적용 후)와 돌아온 항목이 같은가.

    페이로드 전체, 시작·끝 시각, 스트림이 모두 같아야 같다 (검수 상태·출처는 보지 않는다).
    """
    return (
        a.payload == b.payload
        and a.t_start_ms == b.t_start_ms
        and a.t_end_ms == b.t_end_ms
        and a.stream_id == b.stream_id
    )


def _new_id(base: str, reviewer: str, now: datetime, content: str) -> str:
    """새 사람 레코드 ID `"<base>:r<해시 12자>"` (128자를 넘으면 뒤쪽 128자만 남긴다).

    base: 원래 라벨 ID(수정·삭제) 또는 `"<세션>-h"`(새로 그린 것). 해시 입력이 검수자·시각·내용이라
    같은 결과를 다시 수집하면 같은 ID가 나온다 (중복 삽입 방지·멱등).
    """
    digest = hashlib.sha256(f"{base}|{reviewer}|{now.isoformat()}|{content}".encode()).hexdigest()
    return f"{base}:r{digest[:12]}"[-128:]


def reconcile(
    originals: list[LabelRecord],
    reviewed: list[ReviewedItem],
    *,
    session_id: str,
    ontology_version: str,
    reviewer_id: str,
    now: datetime,
    normalize: Callable[[LabelRecord], LabelRecord] | None = None,
) -> ReviewOutcome:
    """normalize: 도구로 보낼 때 적용한 변환 (예: CVAT 좌표 반올림). 비교는 그 결과와 한다.

    인자:
    - originals: 작업에 실제로 보낸 원래 라벨 (`ReviewTask.sent_label_ids`로 찾은 레코드).
      블라인드·이중 작업은 빈 목록이라 돌아온 모든 항목이 "새로 그린 것"이 된다.
    - reviewed: 도구에서 돌아온 항목.
    - session_id: 새 레코드의 세션.
    - ontology_version: 새로 그린 레코드의 온톨로지 버전 (수정·삭제는 원래 라벨의 버전을 잇는다).
    - reviewer_id: 검수자 ID (작업 담당자). 새 레코드 `verification.reviewer_id`에 남는다.
    - now: 검수 시각 (시간대 필수). `created_at`·`reviewed_at`과 ID 해시에 쓴다.

    반환: `ReviewOutcome`. 원래 라벨 중 돌아오지 않은 것은 삭제(retracted) 레코드가 된다.
    예외: 같은 원래 라벨이 결과에 두 번 나오면 ValueError (검수자가 트랙을 복제한 경우 등).
    """
    by_id = {x.label_id: x for x in originals}
    # 비교 기준: 도구로 보낸 형태 (좌표 변환·반올림 왕복 결과)
    sent = {x.label_id: (normalize(x) if normalize else x) for x in originals}
    verification = Verification(
        state=VerificationState.HUMAN_CORRECTED, reviewer_id=reviewer_id, reviewed_at=now
    )
    outcome = ReviewOutcome()
    seen: set[str] = set()

    def record(
        item: ReviewedItem, parent: LabelRecord | None, *, retracted: bool = False
    ) -> LabelRecord:
        """검수 항목으로 새 사람 레코드를 만든다 (parent가 있으면 수정·삭제, 없으면 추가)."""
        # ID 해시 입력: 내용이 다르면 ID도 달라진다 (정렬된 JSON이라 결정적)
        content = json.dumps(
            [
                item.t_start_ms,
                item.t_end_ms,
                item.stream_id,
                item.payload.model_dump(mode="json"),
                retracted,
            ],
            sort_keys=True,
        )
        base = parent.label_id if parent else f"{session_id}-h"
        return LabelRecord(
            label_id=_new_id(base, reviewer_id, now, content),
            session_id=session_id,
            stream_id=item.stream_id,
            t_start_ms=item.t_start_ms,
            t_end_ms=item.t_end_ms,
            ontology_version=parent.ontology_version if parent else ontology_version,
            provenance=Provenance(source=Source.HUMAN),
            verification=verification,
            parent_label_id=parent.label_id if parent else None,
            retracted=retracted,
            created_at=now,
            payload=item.payload,
        )

    for item in reviewed:
        # 원래 ID가 없거나 보낸 라벨이 아니면(다른 작업의 ID 등) 새로 그린 것으로 본다
        origin = by_id.get(item.origin_label_id or "")
        if origin is None:
            outcome.new_records.append(record(item, None))
            continue
        if origin.label_id in seen:
            raise ValueError(f"검수 결과에 같은 원래 라벨이 두 번 나옵니다: {origin.label_id}")
        seen.add(origin.label_id)
        if _same(sent[origin.label_id], item):
            outcome.approved.append(origin.label_id)
        else:
            outcome.new_records.append(record(item, origin))
    # 보냈는데 돌아오지 않은 라벨 = 검수자가 지웠다 → 원래 내용을 담은 retracted 레코드
    for origin in originals:
        if origin.label_id not in seen:
            gone = ReviewedItem(
                origin.label_id,
                origin.stream_id,
                origin.t_start_ms,
                origin.t_end_ms,
                origin.payload,
            )
            outcome.new_records.append(record(gone, origin, retracted=True))
    return outcome
