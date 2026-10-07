"""배정 방식에 따라 검수 작업에 넣을 라벨을 고른다 (WP12, ADR 0014).

작업을 만들 때와 결과를 수집할 때 같은 함수를 쓴다. 수집 때 다르게 고르면 보내지 않은 라벨을
검수자가 지운 것으로 잘못 본다.

사용처:
- `dlp_review.ops.runner.create_assignment_tasks`: 배정의 작업을 만들 때 (`dlp review assign`).
- `dlp_review.collect.collect_task`: `sent_label_ids`가 없는 예전 작업을 수집할 때.

공개 함수:
- `assignment_selector`: 배정(`ReviewAssignment`) → `Selector` (스트림·종류 → 라벨 목록).

배정 방식별 선택:
- 표준·이중·QA: 현재 운영 라벨. `only_label_ids`가 있으면 그것만,
  없으면 `withheld_label_ids`를 뺀다.
- 블라인드: 아무것도 보내지 않는다 (프리라벨 없이 처음부터 그린다).
- 오류 삽입: 계획 때 넣은 그 배정의 사본(`seed-<배정>-` 접두사)만.
"""

from __future__ import annotations

import sqlalchemy as sa

from dlp_review.ops.seeding import seed_prefix
from dlp_review.tasks import Selector, current_for_review
from dlp_schema.db.repository import get_labels
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.review import ReviewAssignment, ReviewMode


def assignment_selector(conn: sa.Connection, a: ReviewAssignment) -> Selector:
    """배정 방식에 맞는 라벨 선택 함수를 돌려준다.

    인자:
    - conn: DB 연결 (읽기만 한다. `label_records`를 조회).
    - a: 검수 배정. `mode`, `session_id`, `only_label_ids`, `withheld_label_ids`,
      `assignment_id`를 본다.

    반환: `(stream_id, kinds) -> list[LabelRecord]`. stream_id가 None이면 세션 단위(시간 라벨)이고,
    스트림 지정 시에는 그 스트림 라벨과 스트림이 없는(세션 단위) 라벨을 함께 고른다.
    부작용 없음. 호출할 때마다 DB를 다시 읽으므로 같은 트랜잭션 안에서 부르는 것이 안전하다.
    """

    def standard(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        """표준·이중·QA 배정: 현재 운영 라벨에서 표본 정책에 따라 거른다."""
        labels = current_for_review(conn, a.session_id, stream_id, kinds)
        if a.only_label_ids:
            # 재검수(resample) 배정: 표본 불합격 묶음의 나머지 라벨만 다시 본다
            keep = set(a.only_label_ids)
            return [x for x in labels if x.label_id in keep]
        # 표본 검수 배정: 표본에 들지 않은 높은 신뢰도 라벨(withheld)은 보내지 않는다.
        # 그 라벨은 표본 판정(finish_assignment)으로 표본 검증되거나 재검수 배정으로 간다.
        withheld = set(a.withheld_label_ids)
        return [x for x in labels if x.label_id not in withheld]

    def blind(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        """블라인드 배정: 프리라벨을 보여 주지 않는다 (편향 측정의 기준)."""
        return []

    def seeded(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        """오류 삽입 배정: 이 배정의 사본(접두사로 구별)만 보낸다.

        사본은 `seeded_error=True`라 운영 라벨이 아니므로 `operational=False`로 이력을 펼친다.
        """
        prefix = seed_prefix(a.assignment_id)
        labels = current_labels(
            get_labels(conn, a.session_id, kinds=list(kinds)), operational=False
        )
        return [
            x
            for x in labels
            if x.label_id.startswith(prefix)
            and (stream_id is None or x.stream_id in (stream_id, None))
        ]

    if a.mode is ReviewMode.BLIND:
        return blind
    if a.mode is ReviewMode.SEEDED_ERROR:
        return seeded
    return standard
