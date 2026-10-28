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

공개 이름: `ReviewedItem`, `ReviewOutcome`, `same_as_sent`, `keep_unchanged_world`, `reconcile`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from dlp_schema.aireview import AiOutcome
from dlp_schema.common import derived_id
from dlp_schema.labels import (
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
    without_world,
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
    # 도구 쪽 영역·트랙 ID (Label Studio 결과 id 등, 원래 라벨이 아닐 때도 남김). 새로 그린 항목의
    # ID 해시 구분자로만 쓴다 (F-REV-35). 모르면 None
    tool_region_id: str | None = None


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
    # 블러 검수 결과로 블러가 바뀌어 매체 회수를 요청한 라벨링 작업 키 (ADR 0039, collect가 채운다)
    revoked: list[str] = field(default_factory=list[str])
    # 회수 요청 당시 열려 있고 배정이 있던 작업의 배정 ID (재승인·렌더 뒤 `dlp review assign`)
    reissue_assignments: list[str] = field(default_factory=list[str])
    # 배정 없이 만든 열린 작업의 (담당자, 스트림, 도구) (`dlp review create … --stream --tool`)
    direct_reissue: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    # AI 1차 검수 결과 기록 (노출 기록이 있는 작업, `ai_collect.record_outcome`이 채운다)
    ai: AiOutcome | None = None
    # 행동·상위 구간 검수 결과를 따라 검수 상태를 물려받은 설명 라벨 ID와 지운 설명 라벨 ID
    # (F-PAR-15, `dlp_schema.description_follow`. collect가 채운다. AI 결과 기록과 무관하다)
    descriptions_approved: list[str] = field(default_factory=list[str])
    descriptions_retracted: list[str] = field(default_factory=list[str])
    # 2D 트랙(손·박스) 검수를 물려받은 3D 궤적 라벨 ID와, 출처 2D가 바뀌어 물려받음이 끊겨 지운
    # 3D 궤적 라벨 ID (다음 `dlp prelabel run`이 고친 2D로 다시 만든다. ADR 0048·0050, collect가
    # 채운다)
    trajectories_approved: list[str] = field(default_factory=list[str])
    trajectories_retracted: list[str] = field(default_factory=list[str])

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


def same_as_sent(a: LabelRecord, b: ReviewedItem) -> bool:
    """도구로 보낸 형태(normalize 적용 후)와 돌아온 항목이 같은가.

    페이로드 전체, 시작·끝 시각, 스트림이 모두 같아야 같다 (검수 상태·출처는 보지 않는다).
    `reconcile`(승인 판정)과 AI 제안 적용(`ai_collect.apply_decisions`, 검수자가 구간을 그대로
    둔 채 적용했는지)이 같은 기준을 쓴다.
    """
    return (
        a.payload == b.payload
        and a.t_start_ms == b.t_start_ms
        and a.t_end_ms == b.t_end_ms
        and a.stream_id == b.stream_id
    )


def keep_unchanged_world(
    payload: KeypointTrackPayload, parent: LabelRecord, sent: LabelRecord
) -> KeypointTrackPayload:
    """사람 레코드 키프레임 중 2D가 parent와 같은 것에만 parent의 손 중심 3D 점을 되살린다.

    보간 표시(`interpolated`, ADR 0054)도 같은 규칙으로 되살린다 (고치지 않은 키프레임의 2D는
    여전히 보간 값이다).

    ADR 0044 1절 (2026-10-16 개정): 검수자가 손 트랙의 일부 키프레임만 고치면, 고치지 않은
    키프레임의 2D는 모델이 낸 2D 그대로다. 그 키프레임의 world는 그 2D와 짝인 모델 출력이므로
    남기고, 고친 키프레임·새로 생긴 시각에는 싣지 않는다. 판정은 같은 시각(t_ms)의 키프레임에서
    도구로 보낸 형태(`sent`, CVAT `quantize` 왕복 결과)의 points(좌표·보임 정도)와 돌아온 points가
    정확히 같을 때다. 그래서 world가 있는 키프레임의 2D는 늘 parent 2D의 양자화 값이다.

    Args:
        payload: world를 뺀 사람 레코드 페이로드 (`without_world` 결과).
        parent: 원래 라벨 (world 원본, 직전 사람 레코드여도 그 world는 같은 규칙으로 남은 것이다).
        sent: parent를 도구로 보낸 형태 (normalize 결과, 없으면 parent 그대로).

    Returns:
        되살릴 키프레임이 없으면 payload 그대로, 있으면 그 키프레임에만 world를 채운 사본. 개체·
        골격·손이 parent와 다르면(다른 손으로 바꿈) 되살리지 않는다.
    """
    p, q = parent.payload, sent.payload
    if not (
        isinstance(p, KeypointTrackPayload)
        and isinstance(q, KeypointTrackPayload)
        and (p.entity_id, p.skeleton, p.hand) == (payload.entity_id, payload.skeleton, payload.hand)
    ):
        return payload
    # 원래 키프레임의 모델 전용 값: world와 보간 표시 (ADR 0044·0054, 도구로 보내지 않는다)
    model_fields = {
        f.t_ms: {"world": f.world, "interpolated": f.interpolated}
        for f in p.keyframes
        if f.world is not None or f.interpolated
    }
    sent_points = {f.t_ms: f.points for f in q.keyframes}
    frames = tuple(
        f.model_copy(update=model_fields[f.t_ms])
        if f.t_ms in model_fields and sent_points.get(f.t_ms) == f.points
        else f
        for f in payload.keyframes
    )
    if all(f.world is None and not f.interpolated for f in frames):
        return payload
    return payload.model_copy(update={"keyframes": frames})


def _new_id(base: str, reviewer: str, now: datetime, content: str) -> str:
    """새 사람 레코드 ID `"<base>:r<해시 12자>"`.

    base: 원래 라벨 ID(수정·삭제) 또는 `"<세션>-h"`(새로 그린 것). 해시 입력이 base·검수자·시각·
    내용이라 넷이 모두 같아야 같은 ID가 나온다(멱등). 수집 시각이 달라지면 ID도 달라진다.
    수집 재시도의 중복 방지는 이 함수가 아니라 저장된 작업의 COLLECTED 상태로 처리한다.

    128자(`Identifier` 최대)를 넘으면 `dlp_schema.common.derived_id`처럼 base 앞부분 + base 해시로
    줄인다: `<base 앞부분>:h<sha256 16자>:r<해시 12자>`. base가 영숫자로 시작하므로 결과도
    `Identifier` 규칙을 지킨다. 회귀(F-REV-05, A-09): 예전에는 뒤쪽 128자만 남겨 `-`·`:`로 시작하는
    잘못된 ID가 생겨 수집이 ValidationError로 멈췄다. 128자 이하이면 예전과 같은 ID다.
    """
    digest = hashlib.sha256(f"{base}|{reviewer}|{now.isoformat()}|{content}".encode()).hexdigest()
    return derived_id(base, f"r{digest[:12]}")


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
    # 새로 그린 항목의 (도구 영역 ID, 내용)별 등장 횟수
    # (F-REV-35: 같은 내용을 두 번 그려도 ID가 갈린다)
    drawn: dict[str, int] = {}

    def record(
        item: ReviewedItem, parent: LabelRecord | None, *, retracted: bool = False
    ) -> LabelRecord:
        """검수 항목으로 새 사람 레코드를 만든다 (parent가 있으면 수정·삭제, 없으면 추가)."""
        payload = item.payload
        if isinstance(payload, KeypointTrackPayload) and not retracted:
            # 사람이 고치거나 그린 골격에는 들어온 손 중심 3D 점을 그대로 싣지 않는다 (ADR 0044).
            # 도구 변환기는 world를 내지 않지만, 다른 경로로 들어와도 고친 2D와 예전 world가 섞이지
            # 않게 먼저 모두 빼고, 2D가 parent와 같은 키프레임에만 parent의 world를 되살린다
            payload = without_world(payload)
            if parent is not None:
                payload = keep_unchanged_world(payload, parent, sent[parent.label_id])
        # ID 해시 입력: 내용이 다르면 ID도 달라진다 (정렬된 JSON이라 결정적)
        content = json.dumps(
            [
                item.t_start_ms,
                item.t_end_ms,
                item.stream_id,
                payload.model_dump(mode="json"),
                retracted,
            ],
            sort_keys=True,
        )
        base = parent.label_id if parent else f"{session_id}-h"
        if parent is None:
            # 새로 그린 항목은 같은 내용이 둘일 수 있다 (F-REV-35). 도구 영역 ID와
            # 같은 (영역 ID, 내용)의 등장 순번을 해시에 넣는다. 같은 입력을 다시 수집하면
            # 순서가 같아 같은 ID다(멱등). 영역 ID가 없고 첫 등장이면 예전과 같은 해시 입력이다
            key = json.dumps([item.tool_region_id, content])
            nth = drawn.get(key, 0)
            drawn[key] = nth + 1
            if item.tool_region_id is not None or nth:
                content = json.dumps([content, item.tool_region_id, nth])
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
            payload=payload,
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
        if same_as_sent(sent[origin.label_id], item):
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
