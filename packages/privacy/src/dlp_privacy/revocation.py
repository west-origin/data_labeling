"""블러가 바뀐 뒤 낡은 라벨링 작업 매체의 회수 요청 (WP5·WP6, ADR 0039).

일반 라벨러 작업(CVAT 공간 검수의 워터마크 사본, Label Studio 작업의 서명 URL)은 그 작업을 만들 때의
블러본으로 만든다. 그 뒤 운영 블러가 바뀌면(블러 재검수 수집, 재탐지) 이미 배포한 매체에는 새로 찾은
얼굴이 블러되지 않은 채 남는다. 이 모듈은 DB만 쓴다: 낡은 작업을 고르고 `review_tasks`에 회수 요청
(outbox)을 남긴다. 외부 삭제(S3 사본·도구 작업)는 커밋 뒤 `dlp_review.revoke` 실행기가 한다.

판정 불변식: 작업에 저장한 `render_hash` ≠ 지금 운영 블러의 `render_hash`. 블러가 지금 바뀐
스트림에서는 렌더 해시가 없는 구 작업도 낡은 것으로 본다(`legacy=True`). 지금 트랜잭션에서 생긴
블러 라벨을 이전 작업이 담았을 수 없기 때문이다. 같은 매체 URI를 쓰는 작업은 함께 요청한다
(회수 단위는 매체 객체).

dlp_privacy가 `review_tasks`를 쓰는 근거: 블러를 바꾸는 트랜잭션(탐지·렌더)이 이 패키지에 있고,
회수 요청은 같은 커밋에 있어야 한다. `generation`도 이미 `review_tasks`를 읽는다. 쓰기는
`dlp_schema` 저장소 함수(`request_task_revocation`)로만 한다
(의존 표: privacy → media, models, schema).

공개 이름:
- `StaleMedia`: 고른 작업과 고른 이유.
- `stale_media_tasks`: 낡은 매체 작업을 고른다 (읽기만).
- `request_media_revocation`: 세션을 잠그고 회수 요청을 남긴다 (DB 쓰기).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import sqlalchemy as sa

from dlp_privacy.generation import VIDEO_KINDS, operational_blur
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import render_hash
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_review_tasks,
    lock_session,
    request_task_revocation,
)
from dlp_schema.review import ReviewStage, ReviewTask

# 고른 이유: 지정한 작업, 사고 대응 강제, 해시가 다름, 해시 없는 구 작업, 같은 매체를 쓰는 작업
Cause = Literal["task", "force", "hash", "legacy", "shared-media"]


@dataclass(frozen=True)
class StaleMedia:
    """회수할 작업 하나. cause가 shared-media면 다른 작업과 같은 매체 객체를 써서 함께 골랐다."""

    task: ReviewTask
    cause: Cause


def _cause(
    task: ReviewTask,
    current: dict[str, str],
    *,
    legacy: bool,
    force: bool,
    task_keys: Collection[str],
) -> Cause | None:
    """작업 하나를 고를 이유 (고르지 않으면 None). current: 대상 스트림 → 지금 렌더 해시.

    수동 선택(task_keys·force)이 있으면 해시·legacy 판정을 하지 않는다. 낡은 작업은 호출자가 따로
    `sweep` 사유로 요청한다 (`dlp review revoke`). 사유 기록이 섞이지 않게 하려는 것이다.
    """
    if task.task_key in task_keys:
        return "task"
    if task.stream_id not in current:
        return None
    if force:
        return "force"
    if task_keys:
        return None
    if task.render_hash is None:
        return "legacy" if legacy else None
    return "hash" if task.render_hash != current[task.stream_id] else None


def stale_media_tasks(
    conn: sa.Connection,
    session_id: str,
    policy: PrivacyPolicy,
    *,
    streams: Collection[str] | None = None,
    legacy: bool = False,
    force: bool = False,
    task_keys: Collection[str] = (),
) -> list[StaleMedia]:
    """회수 요청이 아직 없는 비프라이버시 작업 중 매체가 낡은 것 (만든 순). 읽기만 한다.

    수동 선택이 있으면(task_keys나 force) 그것만 고른다: (a) task_key가 task_keys에 있다(스트림
    무관), (b) force이고 스트림이 대상이다. 수동 선택이 없으면 해시로 고른다: (c) 렌더 해시가 있고
    지금 운영 블러의 렌더 해시와 다르다, (d) 렌더 해시가 없고 legacy다. 수동 요청과 해시 판정을 한
    번에 섞으면 낡은 작업의 사유가 수동 사유로 남는다. 둘 다 하려면 따로 부른다.
    고른 작업과 같은 매체 URI를 쓰는 다른 미요청 작업도 덧붙인다 (`shared-media`).
    상태(open·collected)와 방식(표준·QA·블라인드·이중·오류 삽입)은 가리지 않는다. 모두 블러본을
    보여 주고, 수집한 CVAT 작업도 담당자가 다시 열 수 있으며 Label Studio 서명 URL도 유효하다.

    지금 해시는 `render_hash(operational_blur(...), policy)`로 계산한다. `expected_render_hash`는
    승인 상태를 요구해 승인이 풀린 뒤(블러 재검수 수집 직후)에 쓸 수 없다.

    Args:
        streams: 대상 영상 스트림 (None이면 세션의 모든 영상 스트림).
        legacy: 렌더 해시가 없는 구 작업도 고른다 (블러가 지금 바뀐 스트림). 수동 선택이면 무시.
        force: 해시와 무관하게 대상 스트림의 모든 작업 (사고 대응 `--all`).
        task_keys: 해시·스트림과 무관하게 고를 작업 키 (`--task`). 이 세션의 비프라이버시 미요청
            작업이 아니면 조용히 빠진다 (호출자가 먼저 확인한다).
    """
    session = get_session(conn, session_id)
    video = [s.stream_id for s in session.streams if s.kind in VIDEO_KINDS]
    targets = [s for s in video if streams is None or s in streams]
    history = get_labels(conn, session_id, kinds=["blur_track"])
    current = {s: render_hash(operational_blur(history, s), policy) for s in targets}
    candidates = [
        t
        for t in list_review_tasks(conn, session_id)
        if t.stage is not ReviewStage.PRIVACY and t.revoke_requested_at is None
    ]
    picked: dict[str, StaleMedia] = {}
    for t in candidates:
        cause = _cause(t, current, legacy=legacy, force=force, task_keys=task_keys)
        if cause is not None:
            picked[t.task_key] = StaleMedia(t, cause)
    uris = {m.task.media_uri for m in picked.values()}
    shared = [t for t in candidates if t.task_key not in picked and t.media_uri in uris]
    return [*picked.values(), *(StaleMedia(t, "shared-media") for t in shared)]


def request_media_revocation(
    conn: sa.Connection,
    session_id: str,
    policy: PrivacyPolicy,
    now: datetime,
    *,
    reason: str,
    streams: Collection[str] | None = None,
    legacy: bool = False,
    force: bool = False,
    task_keys: Sequence[str] = (),
) -> list[ReviewTask]:
    """세션을 잠그고 낡은 매체 작업마다 회수를 요청한다. 실제로 요청한 작업(요청 전 사본).

    돌려준 사본의 상태로 호출자가 다시 만들 작업을 안다 (요청 당시 open이었는지, 배정이 있었는지).
    열린 작업은 revoked, 수집한 작업은 상태를 두고 요청 시각·사유만 남는다. 같은 매체를 쓰는 작업의
    사유는 `<reason>:shared-media`다. 배정(`review_assignments`)은 쓰지 않는다: `lock_assignment`가
    회수된 작업을 "작업 없음"으로 본다. 고르는 규칙은 `stale_media_tasks`와 같다 (수동 선택이면
    그것만).

    세션 잠금 아래에서 요청하므로 같은 세션의 작업 생성(`create_labeling_tasks`도 세션을 먼저
    잠근다)과 겹치지 않는다. 다시 불러도 이미 요청한 작업은 바뀌지 않는다 (멱등).

    부작용: `sessions` 행 잠금, `review_tasks` 갱신. 외부 호출은 없다 (커밋 뒤 실행기 몫).
    """
    lock_session(conn, session_id)
    stale = stale_media_tasks(
        conn, session_id, policy, streams=streams, legacy=legacy, force=force, task_keys=task_keys
    )
    requested: list[ReviewTask] = []
    for item in stale:
        why = f"{reason}:shared-media" if item.cause == "shared-media" else reason
        if request_task_revocation(conn, item.task.task_key, reason=why, at=now):
            requested.append(item.task)
    return requested
