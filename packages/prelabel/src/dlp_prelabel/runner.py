"""세션 프리라벨 실행.

1. 영상 스트림마다 Predictor를 돌린다 (원본 영상: 모델 추론은 접근 통제된 서버에서 원본으로 한다).
   같은 Predictor·모델 버전 결과가 그 스트림에 이미 있으면 건너뛴다.
2. 깊이 모델이 있으면 바디캠 손 관절·객체 박스를 카메라 좌표 3D 궤적으로 올린다 (lift3d).
3. 바디캠의 손 키포인트·객체 박스와 장갑 신호로 접촉 구간을 만들어 hand_state 라벨로 쓴다.
4. 3인칭 영상이 있으면 바디캠 IMU와 3인칭 인물 손목 속도를 상관시켜 착용자를 찾는다.
   찾은 인물의 키포인트 트랙은 entity_id="wearer"인 새 레코드(parent=원래 트랙)로 남긴다.
5. 프라이버시 승인 상태의 세션은 생애주기를 prelabeled로 옮긴다.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlalchemy as sa

from dlp_media.storage import ObjectStore
from dlp_prelabel.common import model_label
from dlp_prelabel.contact import (
    ContactInterval,
    fuse_contacts,
    glove_contact_intervals,
    video_contact_intervals,
)
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.wearer import match_wearer, wrist_speed
from dlp_schema.db.repository import get_labels, get_session, insert_labels, set_lifecycle
from dlp_schema.episode import current_labels, retractions, version_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    Evidence,
    Hand,
    HandStatePayload,
    KeypointTrackPayload,
    LabelRecord,
    Source,
    VerificationState,
)
from dlp_schema.ontology import Ontology
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.session import LifecycleState, Session, StreamKind, SyncMethod
from dlp_sync.signals import glove_series, imu_series

VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
CONTACT_VERSION = "contact-heuristic-1"
WEARER_VERSION = "wearer-xcorr-1"


@dataclass
class PrelabelSummary:
    produced: dict[str, int] = field(default_factory=dict[str, int])  # "스트림/predictor" → 라벨 수
    skipped: list[str] = field(default_factory=list[str])
    contacts: int = 0
    retracted: int = 0  # 새 버전으로 바뀌며 지운 이전 버전 라벨
    lifted: int = 0
    wearer: str | None = None
    wearer_scores: dict[str, float] = field(default_factory=dict[str, float])


def _fetch(store: ObjectStore, uri: str, work: Path) -> Path:
    key = uri.removeprefix(store.uri(""))
    dest = work / key.replace("/", "__")
    if not dest.exists():
        store.get_file(key, dest)
    return dest


def run_prelabel(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    predictors: list[Predictor],
    policy: PrelabelPolicy,
    ontology: Ontology,
    now: datetime,
    lifter: DepthLifter | None = None,
) -> PrelabelSummary:
    session = get_session(conn, session_id)
    if session.ontology_version is None:
        raise ValueError(f"{session_id}: 세션에 온톨로지 버전이 없습니다")
    summary = PrelabelSummary()
    existing = get_labels(conn, session_id)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in (s for s in session.streams if s.kind in VIDEO):
            for predictor in predictors:
                key = f"{stream.stream_id}/{predictor.name}"
                if any(
                    x.stream_id == stream.stream_id
                    and x.provenance.model_version == predictor.version
                    for x in existing
                ):
                    summary.skipped.append(key)
                    continue
                labels = predictor.run(
                    Clip(session_id, stream.stream_id, _fetch(raw, stream.uri, work))
                )
                # 같은 예측기의 이전 버전 라벨 중 아직 아무도 검수하지 않은 것은 지운다
                prefix = f"{session_id}-{stream.stream_id}-{predictor.name}-"
                stale = _stale(existing, prefix, predictor.version)
                insert_labels(conn, [*retractions(stale, predictor.version, now), *labels])
                summary.produced[key] = len(labels)
                summary.retracted += len(stale)
        history = get_labels(conn, session_id)
        current = current_labels(history)
        if lifter is not None:
            summary.lifted = _lift(conn, session, history, current, raw, work, lifter, now)
        summary.contacts = _contacts(
            conn, session, history, current, raw, work, policy, ontology, now
        )
        _wearer(conn, session, history, current, raw, work, policy, now, summary)
    if session.lifecycle_state is LifecycleState.PRIVACY_APPROVED:
        set_lifecycle(conn, session_id, LifecycleState.PRELABELED)
    return summary


def _stale(labels: list[LabelRecord], prefix: str, version: str) -> list[LabelRecord]:
    """같은 단계의 이전 버전 모델 라벨 중 현재 운영 라벨이고 아직 검수하지 않은 것."""
    return [
        x
        for x in current_labels(labels)
        if x.label_id.startswith(prefix)
        and x.provenance.source is Source.MODEL
        and x.provenance.model_version != version
        and x.verification.state is VerificationState.UNREVIEWED
    ]


def _lift(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    lifter: DepthLifter,
    now: datetime,
) -> int:
    """멱등: 같은 버전을 낸 적이 있으면(검수자가 모두 고쳤어도) 다시 만들지 않는다."""
    body = session.reference_stream
    if any(x.provenance.model_version == lifter.version for x in history):
        return 0
    tracks = [
        x
        for x in current
        if x.stream_id == body.stream_id
        and isinstance(x.payload, KeypointTrackPayload | BoxTrackPayload)
    ]
    if not tracks:
        return 0
    labels = lifter.run(
        _fetch(raw, body.uri, work),
        session_id=session.session_id,
        stream_id=body.stream_id,
        tracks=tracks,
        calib=session.calibration.intrinsics,
        ontology_version=session.ontology_version or "",
    )
    stale = _stale(history, f"{session.session_id}-{body.stream_id}-3d-", lifter.version)
    insert_labels(conn, [*retractions(stale, lifter.version, now), *labels])
    return len(labels)


def _contact_kind(class_id: str | None, ontology: Ontology) -> str:
    obj = ontology.objects.get(class_id or "")
    if obj is None:
        return "object"
    if obj.tool is not None:
        return "tool"
    return "fixed_surface" if obj.surface else "object"


def _contacts(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    policy: PrelabelPolicy,
    ontology: Ontology,
    now: datetime,
) -> int:
    # 멱등: 이력에 이 버전이 있으면 (검수자가 모두 고쳤거나 지웠어도) 다시 만들지 않는다
    if any(x.provenance.model_version == CONTACT_VERSION for x in history):
        return 0
    body = session.reference_stream.stream_id
    hands = {
        x.payload.hand: x.payload
        for x in current
        if x.stream_id == body
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "hand21"
        and x.payload.hand is not None
    }
    objects = [
        x.payload for x in current if x.stream_id == body and isinstance(x.payload, BoxTrackPayload)
    ]
    classes = {o.entity_id: o.class_id for o in objects}
    gloves = {
        Hand.LEFT if s.kind is StreamKind.GLOVE_LEFT else Hand.RIGHT: s
        for s in session.streams
        if s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT)
        and s.sync_method is not SyncMethod.UNSYNCED
    }
    labels: list[LabelRecord] = []
    for hand in (Hand.LEFT, Hand.RIGHT):
        video = (
            video_contact_intervals(hands[hand], objects, policy.contact.video)
            if hand in hands
            else []
        )
        intervals: list[ContactInterval] = video
        if hand in gloves:
            g = gloves[hand]
            series = glove_series(_fetch(raw, g.uri, work))
            master = np.array([g.to_master_ms(float(t)) for t in series.t_ms])
            intervals = fuse_contacts(
                glove_contact_intervals(master, series.values, policy.contact.glove), video
            )
        for i, c in enumerate(intervals):
            kind = _contact_kind(classes.get(c.target_id or ""), ontology)
            payload = HandStatePayload(
                hand=hand,
                contact_target_kind=kind,
                target_id=c.target_id or "unresolved",
                role="active",
            )
            labels.append(
                model_label(
                    label_id=f"{session.session_id}-contact-{version_tag(CONTACT_VERSION)}-{hand.value}-{i:04d}",
                    session_id=session.session_id,
                    stream_id=None,
                    t_start_ms=c.start_ms,
                    t_end_ms=c.end_ms,
                    ontology_version=session.ontology_version or "",
                    model_version=CONTACT_VERSION,
                    confidence=getattr(policy.contact.confidence, c.source),
                    payload=payload,
                    now=now,
                    evidence=Evidence.OBSERVED if c.source != "video" else Evidence.INFERRED,
                )
            )
    insert_labels(conn, labels)
    return len(labels)


def _wearer(
    conn: sa.Connection,
    session: Session,
    history: list[LabelRecord],
    current: list[LabelRecord],
    raw: ObjectStore,
    work: Path,
    policy: PrelabelPolicy,
    now: datetime,
    summary: PrelabelSummary,
) -> None:
    third = next((s for s in session.streams if s.kind is StreamKind.THIRD_PERSON), None)
    imu = next(
        (
            s
            for s in session.streams
            if s.kind is StreamKind.IMU and s.sync_method is SyncMethod.SHARED_CLOCK
        ),
        None,
    )
    if third is None or imu is None or third.sync_method is SyncMethod.UNSYNCED:
        return
    if any(x.provenance.model_version == WEARER_VERSION for x in history):
        return
    people = {
        x.label_id: x
        for x in current
        if x.stream_id == third.stream_id
        and isinstance(x.payload, KeypointTrackPayload)
        and x.payload.skeleton == "coco17"
    }
    if not people:
        return
    ref = imu_series(_fetch(raw, imu.uri, work))
    speeds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label_id, label in people.items():
        assert isinstance(label.payload, KeypointTrackPayload)
        t, v = wrist_speed(label.payload)
        speeds[label_id] = (np.array([third.to_master_ms(float(x)) for x in t]), v)
    match = match_wearer(
        ref.t_ms,
        ref.values,
        speeds,
        rate_hz=policy.wearer_matching.rate_hz,
        min_correlation=policy.wearer_matching.min_correlation,
    )
    summary.wearer_scores = match.scores
    if match.entity_id is None:
        return
    original = people[match.entity_id]
    assert isinstance(original.payload, KeypointTrackPayload)
    relabeled = original.model_copy(
        update={
            "label_id": f"{original.label_id}:wearer",
            "parent_label_id": original.label_id,
            "payload": original.payload.model_copy(update={"entity_id": "wearer"}),
            "provenance": original.provenance.model_copy(update={"model_version": WEARER_VERSION}),
            "confidence": round(max(match.correlation, 0.0), 4),
            "evidence": Evidence.INFERRED,
            "created_at": now,
        }
    )
    insert_labels(conn, [relabeled])
    summary.wearer = match.entity_id
