"""세션 행동 구간 실행 (`dlp actions run`). 멱등이다.

바디캠의 손마다(hand21 키포인트 트랙이 있는 손) 경계 후보 → VLM 분류 → 병합·채우기를 하고
action·gap·description 레코드를 쓴다. model_version은 "actions-<정책 해시>+<VLM 버전>"이다.
- 같은 버전 결과가 그 손에 이미 있으면 건너뛴다.
- 버전이 바뀌면 이 모듈이 만든 이전 현재 레코드를 삭제 레코드로 표시하고 새로 쓴다.
  검수자가 고친 레코드(사람 출처 자식이 있는 것)는 이미 현재 레코드가 아니므로 건드리지 않는다.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_actions.pipeline import segment_hand
from dlp_actions.policy import ActionsPolicy
from dlp_actions.vlm import VlmClient
from dlp_media.storage import ObjectStore
from dlp_schema.db.repository import get_labels, get_session, insert_labels
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ActionPayload,
    BoxTrackPayload,
    DescriptionPayload,
    GapPayload,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    Provenance,
    Source,
    Verification,
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


def run_actions(
    conn: sa.Connection,
    session_id: str,
    client: VlmClient,
    ontology: Ontology,
    policy: ActionsPolicy,
    now: datetime,
    raw: ObjectStore | None = None,
) -> ActionsSummary:
    session = get_session(conn, session_id)
    version = f"{PREFIX}{policy.digest}+{client.version}"
    summary = ActionsSummary(version)
    labels = get_labels(conn, session_id)
    current = current_labels(labels)
    body = session.reference_stream
    tracks = {
        x.payload.hand: x.payload
        for x in current
        if x.stream_id == body.stream_id
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "hand21"
        and x.payload.hand is not None
    }
    # 대상 후보: 객체 트랙 개체와 손 상태의 접촉 대상
    entities = tuple(
        sorted(
            {
                x.payload.entity_id
                for x in current
                if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
            }
            | {
                x.payload.target_id
                for x in current
                if isinstance(x.payload, HandStatePayload) and x.payload.target_id
            }
        )
    )
    descriptions_by_action = {
        x.payload.segment_id: x
        for x in current
        if isinstance(x.payload, DescriptionPayload) and _ours(x)
    }
    with tempfile.TemporaryDirectory() as tmp:
        video: Path | None = None
        if raw is not None:
            video = Path(tmp) / "bodycam.mp4"
            raw.get_file(body.uri.removeprefix(raw.uri("")), video)
        for hand, track in sorted(tracks.items()):
            mine = [x for x in current if _ours(x) and _hand_of(x) is hand]
            if any(x.provenance.model_version == version for x in mine):
                summary.skipped.append(hand.value)
                continue
            stale = mine + [
                descriptions_by_action[x.payload.action_id]
                for x in mine
                if isinstance(x.payload, ActionPayload)
                and x.payload.action_id in descriptions_by_action
            ]
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
            retractions = [
                x.model_copy(
                    update={
                        "label_id": f"{x.label_id}:retracted",
                        "parent_label_id": x.label_id,
                        "retracted": True,
                        "verification": Verification(),
                        "provenance": Provenance(source=Source.MODEL, model_version=version),
                        "created_at": now,
                    }
                )
                for x in stale
            ]
            insert_labels(conn, [*retractions, *result.labels])
            summary.retracted += len(retractions)
            summary.hands[hand.value] = {
                "candidates": len(result.candidates),
                "segments": len(result.classified),
                "actions": sum(isinstance(x.payload, ActionPayload) for x in result.labels),
                "gaps": sum(isinstance(x.payload, GapPayload) for x in result.labels),
                "unknown_fallbacks": result.fallbacks,
            }
    return summary
