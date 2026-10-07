"""세션 행동 구간 실행 (`dlp actions run`). 멱등이다.

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
  남긴다: 행동만 지우면 설명이 고아가 된다), 새 결과 중 그와 겹치는
  구간은 버린 뒤 남는 빈 시간을 미상(unknown) 공백으로 채운다 (타임라인 공백 0 유지, ADR 0015).
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

PREFIX = "actions-"
KINDS = ("action", "gap", "description")


@dataclass
class ActionsSummary:
    version: str
    hands: dict[str, dict[str, int]] = field(default_factory=dict[str, dict[str, int]])
    skipped: list[str] = field(default_factory=list[str])
    retracted: int = 0


def _hand_of(x: LabelRecord) -> Hand | None:
    p = x.payload
    if isinstance(p, ActionPayload | GapPayload):
        return p.hand
    return None


def _ours(x: LabelRecord) -> bool:
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
    """span에서 cuts(정렬됨)를 뺀 조각들."""
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
    """
    if not protected:
        return labels
    keep_spans = sorted((x.t_start_ms, x.t_end_ms) for x in protected)

    def overlaps(s: int, e: int) -> bool:
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
    session = get_session(conn, session_id)
    labels = get_labels(conn, session_id)
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
    tracks: dict[Hand, KeypointTrackPayload] = {}
    for x in track_labels:
        assert isinstance(x.payload, KeypointTrackPayload) and x.payload.hand is not None
        tracks[x.payload.hand] = x.payload
    object_labels = [
        x for x in current if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
    ]
    state_labels = [x for x in current if isinstance(x.payload, HandStatePayload)]
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
            labeling.get_file(blurred_key(session_id, body.stream_id), video)
        for hand, track in sorted(tracks.items()):
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
