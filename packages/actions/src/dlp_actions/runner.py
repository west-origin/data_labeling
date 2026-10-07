"""세션 행동 구간 실행 (`dlp actions run`). 멱등이다 (WP10, ADR 0012·0015·0024·0026).

바디캠의 손마다(hand21 키포인트 트랙이 있는 손) 경계 후보 → VLM 분류 → 병합·채우기를 하고
action·gap·description 레코드를 쓴다. 트랙이 사라진 손은 이 모듈이 낸 검수 전 레코드만 지운다.
VLM 서버가 재시도(백오프) 후에도 응답하지 않은 구간이 있으면 VlmUnavailableError로 세션 전체를
멈춘다 (호출자의 트랜잭션이 되돌려져 아무것도 쓰지 않으므로 다시 실행하면 처음부터 한다).
서버 장애를 미상으로 채워 정상 버전으로 쓰면 재실행이 건너뛰어 미상이 영구히 남기 때문이다.
model_version은 "actions-<정책 해시>+<VLM 버전>+i<입력 해시>"다.
입력 해시는 이 단계가 읽는 현재 라벨(바디캠 손 키포인트, 손 상태, 객체 트랙)의 ID 집합 해시다.
입력이 바뀌면(프리라벨 재실행, 검수자의 접촉 수정) 버전이 바뀌어 다시 만든다.
- 같은 버전 결과가 그 손에 이미 있으면 건너뛴다.
- 버전이 바뀌면 이 모듈이 만든 이전 현재 레코드 중 **검수 전인 것만** 삭제 레코드로 표시하고
  새로 쓴다.
  검수자가 승인·표본 검증한 레코드와 사람이 고치거나 만든 레코드는 남기고(설명이 검수된 행동도
  남긴다: 행동만 지우면 설명이 고아가 된다), 새 결과 중 그와 겹치는 구간은 버린 뒤 남는 빈 시간을
  미상(unknown) 공백으로 채운다 (타임라인 공백 0 유지, ADR 0015).

입력은 `current_labels()`(운영 라벨)에서 고르고, 다시 돌릴지는 전체 이력(`get_labels`)으로 정한다.
DB: `labels` 테이블 읽기·쓰기(추가만). 저장소: 라벨링 버킷의 블러본 읽기(원본 버킷은 읽지 않는다).
트랜잭션은 호출자(CLI)가 연다.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_actions.pipeline import segment_hand
from dlp_actions.policy import ActionsPolicy
from dlp_actions.vlm import VlmClient
from dlp_media.storage import ObjectStore, blurred_key
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import assert_render_current, check_fetched
from dlp_schema import repo_root
from dlp_schema.db.repository import get_labels, get_session, insert_labels
from dlp_schema.episode import current_labels, retractions, version_tag
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    DescriptionPayload,
    Evidence,
    GapPayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology

# 이 단계 모델 버전의 접두사 (`_ours` 판별)
PREFIX = "actions-"
# 이 단계가 쓰는 라벨 종류 (참고용, 코드에서 쓰지 않는다)
KINDS = ("action", "gap", "description")


@dataclass
class ActionsSummary:
    """실행 요약 (CLI가 출력한다).

    Attributes:
        version: 이번 실행의 모델 버전.
        hands: 손 → 개수 (candidates, segments, actions, gaps, kept_reviewed, unknown_fallbacks,
            또는 트랙이 사라진 손이면 retracted_without_track).
        skipped: 같은 버전 결과가 있어 건너뛴 손.
        retracted: 이번에 쓴 삭제 레코드 수.
    """

    version: str
    hands: dict[str, dict[str, int]] = field(default_factory=dict[str, dict[str, int]])
    skipped: list[str] = field(default_factory=list[str])
    retracted: int = 0


def _hand_of(x: LabelRecord) -> Hand | None:
    """행동·사이 구간 라벨의 손 (그 밖의 라벨은 None)."""
    p = x.payload
    if isinstance(p, ActionPayload | GapPayload):
        return p.hand
    return None


def _ours(x: LabelRecord) -> bool:
    """이 단계(`actions-` 접두사 모델 버전)가 낸 모델 라벨인지."""
    return x.provenance.source is Source.MODEL and (x.provenance.model_version or "").startswith(
        PREFIX
    )


def input_digest(labels: list[LabelRecord]) -> str:
    """입력 라벨 ID 집합의 짧은 해시 (라벨은 덮어쓰지 않으므로 수정되면 ID가 바뀐다)."""
    h = hashlib.sha256()
    for label_id in sorted({x.label_id for x in labels}):
        h.update(label_id.encode() + b"\n")
    return h.hexdigest()[:8]


def subtract(span: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """span에서 cuts(정렬됨)를 뺀 조각들.

    예: subtract((0, 100), [(20, 30), (50, 60)]) == [(0, 20), (30, 50), (60, 100)].
    """
    start, end = span
    out: list[tuple[int, int]] = []
    cur = start
    for cs, ce in cuts:
        if ce <= cur or cs >= end:
            continue
        if cs > cur:
            out.append((cur, cs))
        cur = max(cur, ce)
        if cur >= end:
            break
    if cur < end:
        out.append((cur, end))
    return out


def keep_protected(
    labels: list[LabelRecord],
    protected: list[LabelRecord],
    hand: Hand,
    version: str,
    now: datetime,
) -> list[LabelRecord]:
    """새 행동·공백 중 검수된(또는 사람) 구간과 겹치는 것을 버린다.

    버려서 비는 시간은 미상 공백으로 채운다.

    버린 행동의 설명도 버린다. 남은 타임라인은 검수된 구간 + 새 구간 + 채운 공백으로 빈틈이 없다.

    Args:
        labels: 새로 만든 라벨 (`segment_hand` 결과).
        protected: 남길 그 손의 행동·공백 (사람 출처, 검수됨, 설명이 검수된 행동).
        hand: 손.
        version: 이번 모델 버전 (채운 공백 ID의 `version_tag`).
        now: 생성 시각.

    Returns:
        쓸 라벨. 채운 공백은 첫 번째로 버린 라벨을 본떠 만들고(신뢰도 0, 근거 INFERRED),
        ID는 `<세션>-<손>-<tag>-<시작>-fill`이다.
    """
    if not protected:
        return labels
    keep_spans = sorted((x.t_start_ms, x.t_end_ms) for x in protected)

    def overlaps(s: int, e: int) -> bool:
        """[s, e)가 보호 구간 중 하나와 겹치는지."""
        return any(ks < e and s < ke for ks, ke in keep_spans)

    out: list[LabelRecord] = []
    dropped_actions: set[str] = set()
    holes: list[tuple[int, int]] = []
    template: LabelRecord | None = None
    for x in labels:
        timed = isinstance(x.payload, ActionPayload | GapPayload)
        if timed and overlaps(x.t_start_ms, x.t_end_ms):
            holes.append((x.t_start_ms, x.t_end_ms))
            template = template or x
            if isinstance(x.payload, ActionPayload):
                dropped_actions.add(x.payload.action_id)
            continue
        out.append(x)
    out = [
        x
        for x in out
        if not (
            isinstance(x.payload, DescriptionPayload) and x.payload.segment_id in dropped_actions
        )
    ]
    if template is None:
        return out
    tag = version_tag(version)
    for hole in holes:
        for start, end in subtract(hole, keep_spans):
            out.append(
                template.model_copy(
                    update={
                        "label_id": f"{template.session_id}-{hand.value}-{tag}-{start}-fill",
                        "t_start_ms": start,
                        "t_end_ms": end,
                        "confidence": 0.0,
                        "evidence": Evidence.INFERRED,
                        "created_at": now,
                        "payload": GapPayload(hand=hand, gap_type="unknown"),
                    }
                )
            )
    return out


def run_actions(
    conn: sa.Connection,
    session_id: str,
    client: VlmClient,
    ontology: Ontology,
    policy: ActionsPolicy,
    now: datetime,
    labeling: ObjectStore | None = None,
) -> ActionsSummary:
    """세션의 손마다 행동 구간을 만들어 DB에 쓴다 (모듈 docstring의 규칙).

    Args:
        conn: 열린 트랜잭션의 DB 연결 (호출자가 커밋·롤백한다).
        session_id: 세션 ID.
        client: VLM 클라이언트 (`version`이 모델 버전에 들어간다).
        ontology: 온톨로지.
        policy: 행동 구간 정책 (`digest`가 모델 버전에 들어간다).
        now: 생성 시각 (시간대 필수).
        labeling: 라벨링 버킷 저장소. 주면 블러본을 받아 VLM에 프레임을 보낸다. None이면 프레임 없이
            묻는다 (테스트·오라클).

    Returns:
        `ActionsSummary`.

    Raises:
        VlmUnavailableError: VLM 서버가 백오프 후에도 응답하지 않을 때 (아무것도 쓰지 않도록
            호출자가 트랜잭션을 되돌린다).
        RenderNotCurrentError: 블러본이 지금 승인된 블러 라벨로 렌더한 것이 아닐 때 (ADR 0024).

    부작용: `labels`에 삭제 레코드와 새 라벨을 추가한다. 라벨링 버킷에서 블러본을 임시 디렉터리로
    받는다.
    """
    session = get_session(conn, session_id)
    labels = get_labels(conn, session_id)
    # 입력은 운영 라벨(current)에서, 재실행 판단은 전체 이력(labels)으로 한다
    current = current_labels(labels)
    body = session.reference_stream
    track_labels = [
        x
        for x in current
        if x.stream_id == body.stream_id
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "hand21"
        and x.payload.hand is not None
    ]
    # 손마다 트랙 하나 (같은 손이 여럿이면 마지막 것)
    tracks: dict[Hand, KeypointTrackPayload] = {}
    for x in track_labels:
        assert isinstance(x.payload, KeypointTrackPayload) and x.payload.hand is not None
        tracks[x.payload.hand] = x.payload
    object_labels = [
        x for x in current if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
    ]
    state_labels = [x for x in current if isinstance(x.payload, HandStatePayload)]
    # 모델 버전 = 정책 해시 + VLM 버전 + 입력 라벨 ID 집합 해시
    inputs = input_digest([*track_labels, *object_labels, *state_labels])
    version = f"{PREFIX}{policy.digest}+{client.version}+i{inputs}"
    summary = ActionsSummary(version)
    unresolved = set(policy.unresolved_entity_ids)
    # 대상 후보: 객체 트랙 개체와 손 상태의 접촉 대상 (대상을 모르는 접촉 표시 ID는 뺀다)
    entities = tuple(
        sorted(
            (
                {
                    x.payload.entity_id
                    for x in object_labels
                    if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
                }
                | {
                    x.payload.target_id
                    for x in state_labels
                    if isinstance(x.payload, HandStatePayload) and x.payload.target_id
                }
            )
            - unresolved
        )
    )
    descriptions_by_action = {
        x.payload.segment_id: x
        for x in current
        if isinstance(x.payload, DescriptionPayload) and _ours(x)
    }
    # 설명이 검수된(사람이 고쳤거나 승인·표본 검증한) 행동.
    # 행동을 지우면 설명이 고아가 되므로 남긴다
    reviewed_descriptions = {
        x.payload.segment_id
        for x in current
        if isinstance(x.payload, DescriptionPayload)
        and (
            x.provenance.source is Source.HUMAN
            or x.verification.state is not VerificationState.UNREVIEWED
        )
    }

    def described(x: LabelRecord) -> bool:
        """설명이 검수된 행동인지 (그 행동은 지우지 않는다)."""
        return isinstance(x.payload, ActionPayload) and x.payload.action_id in reviewed_descriptions

    def mine_of(hand: Hand) -> list[LabelRecord]:
        """이 단계가 낸 그 손의 검수 전 현재 행동·공백 (설명이 검수된 행동은 뺀다)."""
        return [
            x
            for x in current
            if _ours(x)
            and _hand_of(x) is hand
            and x.verification.state is VerificationState.UNREVIEWED
            and not described(x)
        ]

    def with_descriptions(mine: list[LabelRecord]) -> list[LabelRecord]:
        """지울 행동에 딸린 검수 전 설명까지."""
        return mine + [
            d
            for x in mine
            if isinstance(x.payload, ActionPayload)
            and (d := descriptions_by_action.get(x.payload.action_id)) is not None
            and d.verification.state is VerificationState.UNREVIEWED
        ]

    # 손 트랙이 사라진 손 (예: 손 모델 버전이 바뀌어 그 손을 더 찾지 못함): 새로 만들 입력이
    # 없으므로 이 단계가 냈던 검수 전 행동·공백·설명만 지운다. 검수된 것과 사람 레코드는 남긴다.
    orphans = {h for x in current if _ours(x) and (h := _hand_of(x)) is not None} - set(tracks)
    for hand in sorted(orphans):
        removed = retractions(with_descriptions(mine_of(hand)), version, now)
        if removed:
            insert_labels(conn, removed)
            summary.retracted += len(removed)
            summary.hands[hand.value] = {"retracted_without_track": len(removed)}

    with tempfile.TemporaryDirectory() as tmp:
        video: Path | None = None
        if labeling is not None:
            video = Path(tmp) / "bodycam.mp4"
            # VLM에는 블러본만 보낸다 (원본 프레임이 VLM 서버로 나가지 않게,
            # 설명 초안에 개인정보가 들어가지 않게). 블러본은 프라이버시 승인 후 렌더된다.
            # 지금 승인된 블러 라벨로 렌더한 블러본이 아니면 멈춘다 (ADR 0024)
            rendered = assert_render_current(
                conn, labeling, session_id, body.stream_id, load_privacy_policy(repo_root())
            )
            labeling.get_file(blurred_key(session_id, body.stream_id), video)
            check_fetched(video, rendered, session_id, body.stream_id)
        for hand, track in sorted(tracks.items()):
            # 지울 후보(검수 전 우리 것)와 남길 것(사람·검수됨·설명이 검수된 행동)
            mine = mine_of(hand)
            protected = [
                x
                for x in current
                if _hand_of(x) is hand
                and (
                    x.provenance.source is Source.HUMAN
                    or x.verification.state is not VerificationState.UNREVIEWED
                    or described(x)
                )
            ]
            # 멱등: 이 버전을 낸 적이 있으면 건너뛴다 (검수자가 모두 고쳤거나 다른 버전으로
            # 바뀌었어도). 예전 버전으로 되돌려도 다시 만들지 않는다. 다시 만들려면 버전을 바꾼다.
            if any(x.provenance.model_version == version and _hand_of(x) is hand for x in labels):
                summary.skipped.append(hand.value)
                continue
            stale = with_descriptions(mine)
            # 그 손의 접촉 구간 = 접촉 대상이 있는 손 상태 라벨 (프리라벨·검수 결과)
            contacts = sorted(
                (x.t_start_ms, x.t_end_ms)
                for x in current
                if isinstance(x.payload, HandStatePayload)
                and x.payload.hand is hand
                and x.payload.contact_target_kind != "none"
            )
            result = segment_hand(
                session_id=session_id,
                stream_id=body.stream_id,
                video=video,
                hand=hand,
                track=track,
                contacts=contacts,
                entities=entities,
                start_ms=0,
                end_ms=session.duration_ms,
                client=client,
                ontology=ontology,
                policy=policy,
                model_version=version,
                ontology_version=session.ontology_version or ontology.version,
                now=now,
            )
            # 보호 구간과 겹치는 새 결과는 버리고 빈 곳은 미상으로 채운다.
            # 이전 버전의 검수 전 결과는 삭제 레코드로 지운다
            new = keep_protected(result.labels, protected, hand, version, now)
            removed = retractions(stale, version, now)
            insert_labels(conn, [*removed, *new])
            summary.retracted += len(removed)
            summary.hands[hand.value] = {
                "candidates": len(result.candidates),
                "segments": len(result.classified),
                "actions": sum(isinstance(x.payload, ActionPayload) for x in new),
                "gaps": sum(isinstance(x.payload, GapPayload) for x in new),
                "kept_reviewed": len(protected),
                "unknown_fallbacks": result.fallbacks,
            }
    return summary
