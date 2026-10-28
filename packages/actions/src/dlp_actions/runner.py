"""세션 행동 구간 실행 (`dlp actions run`). 멱등이다 (WP10, ADR 0012·0015·0024·0026).

바디캠의 손마다(hand21 키포인트 트랙이 있는 손) 경계 후보 → VLM 분류 → 병합·채우기를 하고
action·gap·description 레코드를 쓴다. 트랙이 사라진 손은 이 모듈이 낸 검수 전 레코드만 지운다.
VLM 서버가 재시도(백오프) 후에도 응답하지 않은 구간이 있으면 VlmUnavailableError로 세션 전체를
멈춘다 (호출자의 트랜잭션이 되돌려져 아무것도 쓰지 않으므로 다시 실행하면 처음부터 한다).
서버 장애를 미상으로 채워 정상 버전으로 쓰면 재실행이 건너뛰어 미상이 영구히 남기 때문이다.
model_version은 "actions-<정책 해시>+<VLM 버전>+i<입력 해시>"다.
입력 해시는 이 단계가 읽는 현재 라벨(손마다 고른 바디캠 hand21 트랙, 손 상태, 객체 트랙)의 ID 집합
해시다. 같은 손 트랙이 여럿이면 프리라벨 접촉·3D 단계와 같은 공용 순위(`dlp_schema.episode.
track_rank`: 사람·검수 → 최신 → 긴 구간 → 라벨 ID)로 하나를 고른다 (F-PAR-04).
입력이 바뀌면(프리라벨 재실행, 검수자의 접촉 수정, 승인으로 고르는 트랙이 바뀜) 버전이 바뀌어 다시
만든다.
- 같은 버전 결과가 그 손에 이미 있으면 건너뛴다.
- 버전이 바뀌면 이 모듈이 만든 이전 현재 레코드 중 **검수 전인 것만** 삭제 레코드로 표시하고
  새로 쓴다.
  검수자가 승인·표본 검증한 레코드와 사람이 고치거나 만든 레코드는 남기고(설명이 검수된 행동도
  남긴다: 행동만 지우면 설명이 고아가 된다), 새 결과 중 그와 겹치는 구간은 버린 뒤 남는 빈 시간을
  미상(unknown) 공백으로 채운다 (타임라인 공백 0 유지, ADR 0015).
- 설명의 영어 번역(`dlp_actions.translate`, ADR 0043)은 이 단계 출처가 아니지만, 행동을 지울 때 그
  행동의 검수 전 설명과 함께 검수 전 번역도 지운다 (원문 없는 번역이 남지 않게). 검수된 번역은 그
  행동을 지우지 않게 보호한다 (검수된 설명과 같은 규칙). 번역은 CLI가 이 단계가 커밋된 뒤 따로 돈다.

입력은 `current_labels()`(운영 라벨)에서 고르고, 다시 돌릴지는 전체 이력(`get_labels`)으로 정한다.
작업량 추정용 `vlm_request_count`(손마다 후보 구간 수, 순수 함수)는 같은 트랙·접촉 선택
(`hand_track_labels`, `hand_contacts`)을 쓴다 (`dlp aireview estimate --include-actions`).
잠금 (F-PAR-80·81): 이력 읽기·블러본 받기·VLM 호출은 세션을 잠그지 않는다. 쓰기 직전에 세션을
잠그고 (1) 사용 중지·승인 보류·(프레임을 보냈으면) 블러본 렌더 해시를 다시 본다
(`dlp_actions.write_check.write_blocker`, 설명 재생성·번역과 같은 확인). 막히면 아무것도 쓰지 않고
`blocked`로 알린다. (2) 이력을 다시 읽어 지울 것(검수 전 우리 것)·남길 것(보호 구간)을 다시
계획한다. 묻는 동안 검수자가 고친 행동은 보호 구간이 되어 새 결과와 겹치지 않는다. 그 사이 입력이
바뀌어 버전이 달라졌으면 쓰지 않는다 (다시 실행).
DB: `label_records` 테이블 읽기·쓰기(추가만).
저장소: 라벨링 버킷의 블러본 읽기(원본 버킷은 읽지 않는다).
트랜잭션은 호출자(CLI)가 연다.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_actions.pipeline import HandResult, segment_count, segment_hand
from dlp_actions.policy import ActionsPolicy
from dlp_actions.translate import is_machine_translation
from dlp_actions.vlm import VlmClient
from dlp_actions.write_check import fetch_current_blurred, write_blocker
from dlp_media.storage import ObjectStore
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema import repo_root
from dlp_schema.db.locks import lock_session
from dlp_schema.db.repository import get_labels, get_session, insert_labels
from dlp_schema.episode import current_labels, pick_hand_track_labels, retractions, version_tag
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
from dlp_schema.session import Session

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
        blocked: 쓰기 직전 확인(사용 중지·승인 보류·렌더 변경·입력 변경)이 막은 이유. 있으면
            아무것도 쓰지 않았다 (F-PAR-80·81).
    """

    version: str
    hands: dict[str, dict[str, int]] = field(default_factory=dict[str, dict[str, int]])
    skipped: list[str] = field(default_factory=list[str])
    retracted: int = 0
    blocked: str | None = None


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


def hand_track_labels(current: Sequence[LabelRecord], stream_id: str) -> dict[Hand, LabelRecord]:
    """기준 스트림의 손마다 hand21 트랙 라벨 하나.

    같은 손 트랙이 여럿이면 프리라벨 접촉·3D 단계와 같은 공용 순위(`track_rank`: 사람·검수 →
    최신 → 긴 구간 → 라벨 ID)로 고른다 (F-PAR-04).
    """
    return pick_hand_track_labels(
        [
            x
            for x in current
            if x.stream_id == stream_id
            and isinstance(x.payload, KeypointTrackPayload)
            and x.payload.skeleton == "hand21"
            and x.payload.hand is not None
        ]
    )


def _track(x: LabelRecord) -> KeypointTrackPayload:
    """손 트랙 라벨의 페이로드 (`hand_track_labels`가 고른 라벨)."""
    assert isinstance(x.payload, KeypointTrackPayload)
    return x.payload


def hand_contacts(current: Sequence[LabelRecord], hand: Hand) -> list[tuple[int, int]]:
    """그 손의 접촉 구간 = 접촉 대상이 있는 손 상태 라벨 (프리라벨·검수 결과), 시각 순."""
    return sorted(
        (x.t_start_ms, x.t_end_ms)
        for x in current
        if isinstance(x.payload, HandStatePayload)
        and x.payload.hand is hand
        and x.payload.contact_target_kind != "none"
    )


def vlm_request_count(
    session: Session, current: Sequence[LabelRecord], policy: ActionsPolicy
) -> dict[Hand, int]:
    """손마다 행동 VLM 요청 수 (= 후보 구간 수, `run_actions`와 같은 트랙·접촉 선택). 순수 함수.

    같은 버전 결과가 있어 건너뛸 손도 센다 (작업량 추정은 세션 전체를 처음 처리하는 양을 본다).
    `dlp aireview estimate --include-actions`가 쓴다 (ADR 0040 8.3, F-AIR-01).

    Args:
        session: 세션 (기준 스트림, 길이). current: 현재 운영 라벨. policy: 행동 정책.
    """
    picked = hand_track_labels(current, session.reference_stream.stream_id)
    return {
        hand: segment_count(_track(x), hand_contacts(current, hand), 0, session.duration_ms, policy)
        for hand, x in sorted(picked.items())
    }


@dataclass(frozen=True)
class _Inputs:
    """한 시점의 이력으로 본 이 단계의 입력과 지울·남길 레코드 판단 재료 (`_read_inputs`).

    실행 시작 때 한 번(무엇을 물을지), 쓰기 직전 세션 잠금 아래 한 번 더(무엇을 쓰고 지울지,
    F-PAR-81) 만든다.
    """

    session: Session
    # 전체 이력 (재실행 판단)
    labels: list[LabelRecord]
    # 운영 라벨 (입력·지울 후보·보호 판단)
    current: list[LabelRecord]
    # 손 → 고른 hand21 트랙
    tracks: dict[Hand, KeypointTrackPayload]
    # VLM에 줄 대상 후보 개체 ID
    entities: tuple[str, ...]
    # 이번 모델 버전
    version: str
    # 행동 ID → 이 단계의 설명과 그 기계 번역 (행동을 지우면 둘 다 지운다)
    descriptions_by_action: dict[str, list[LabelRecord]]
    # 설명이 검수된(사람이 고쳤거나 승인·표본 검증한) 행동 ID (그 행동은 지우지 않는다)
    reviewed_descriptions: frozenset[str]

    def described(self, x: LabelRecord) -> bool:
        """설명이 검수된 행동인지 (그 행동은 지우지 않는다: 지우면 설명이 고아가 된다)."""
        p = x.payload
        return isinstance(p, ActionPayload) and p.action_id in self.reviewed_descriptions

    def mine_of(self, hand: Hand) -> list[LabelRecord]:
        """이 단계가 낸 그 손의 검수 전 현재 행동·공백 (설명이 검수된 행동은 뺀다)."""
        return [
            x
            for x in self.current
            if _ours(x)
            and _hand_of(x) is hand
            and x.verification.state is VerificationState.UNREVIEWED
            and not self.described(x)
        ]

    def protected_of(self, hand: Hand) -> list[LabelRecord]:
        """남길 그 손의 행동·공백 (사람 출처, 검수됨, 설명이 검수된 행동)."""
        return [
            x
            for x in self.current
            if _hand_of(x) is hand
            and (
                x.provenance.source is Source.HUMAN
                or x.verification.state is not VerificationState.UNREVIEWED
                or self.described(x)
            )
        ]

    def with_descriptions(self, mine: list[LabelRecord]) -> list[LabelRecord]:
        """지울 행동에 딸린 검수 전 설명(과 그 영어 번역)까지."""
        return mine + [
            d
            for x in mine
            if isinstance(x.payload, ActionPayload)
            for d in self.descriptions_by_action.get(x.payload.action_id, [])
            if d.verification.state is VerificationState.UNREVIEWED
        ]

    def done(self, hand: Hand) -> bool:
        """이 버전을 그 손에 낸 적이 있는지 (멱등: 있으면 건너뛴다).

        검수자가 모두 고쳤거나 다른 버전으로 바뀌었어도 건너뛴다. 예전 버전으로 되돌려도 다시
        만들지 않는다. 다시 만들려면 버전을 바꾼다.
        """
        return any(
            x.provenance.model_version == self.version and _hand_of(x) is hand for x in self.labels
        )

    def orphans(self) -> list[Hand]:
        """이 단계 라벨은 있는데 손 트랙이 사라진 손 (예: 손 모델이 그 손을 더 찾지 못함)."""
        ours = {h for x in self.current if _ours(x) and (h := _hand_of(x)) is not None}
        return sorted(ours - set(self.tracks))


def _read_inputs(
    conn: sa.Connection, session_id: str, vlm_version: str, policy: ActionsPolicy
) -> _Inputs:
    """지금 이력에서 입력·모델 버전·판단 재료를 만든다 (모듈 docstring의 규칙)."""
    session = get_session(conn, session_id)
    labels = get_labels(conn, session_id)
    # 입력은 운영 라벨(current)에서, 재실행 판단은 전체 이력(labels)으로 한다
    current = current_labels(labels)
    picked = hand_track_labels(current, session.reference_stream.stream_id)
    object_labels = [
        x for x in current if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
    ]
    state_labels = [x for x in current if isinstance(x.payload, HandStatePayload)]
    # 모델 버전 = 정책 해시 + VLM 버전 + 입력 라벨 ID 집합 해시. 손 트랙은 실제로 쓰는 고른
    # 트랙만 넣는다: 승인은 라벨 ID를 바꾸지 않으므로, 모든 트랙 ID를 넣으면 승인으로 고르는
    # 트랙이 바뀌어도 버전이 같아 다시 돌지 않는다
    inputs = input_digest([*picked.values(), *object_labels, *state_labels])
    # 대상 후보: 객체 트랙 개체와 손 상태의 접촉 대상 (대상을 모르는 접촉 표시 ID는 뺀다)
    candidates = {
        x.payload.entity_id
        for x in object_labels
        if isinstance(x.payload, BoxTrackPayload | MaskTrackPayload)
    } | {
        x.payload.target_id
        for x in state_labels
        if isinstance(x.payload, HandStatePayload) and x.payload.target_id
    }
    descriptions_by_action: dict[str, list[LabelRecord]] = {}
    for x in current:
        if isinstance(x.payload, DescriptionPayload) and (_ours(x) or is_machine_translation(x)):
            descriptions_by_action.setdefault(x.payload.segment_id, []).append(x)
    reviewed = frozenset(
        x.payload.segment_id
        for x in current
        if isinstance(x.payload, DescriptionPayload)
        and (
            x.provenance.source is Source.HUMAN
            or x.verification.state is not VerificationState.UNREVIEWED
        )
    )
    return _Inputs(
        session=session,
        labels=labels,
        current=current,
        tracks={hand: _track(x) for hand, x in picked.items()},
        entities=tuple(sorted(candidates - set(policy.unresolved_entity_ids))),
        version=f"{PREFIX}{policy.digest}+{vlm_version}+i{inputs}",
        descriptions_by_action=descriptions_by_action,
        reviewed_descriptions=reviewed,
    )


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
        `ActionsSummary`. 쓰기 직전 확인이 막았으면 `blocked`에 이유가 있고 아무것도 쓰지 않았다.

    Raises:
        VlmUnavailableError: VLM 서버가 백오프 후에도 응답하지 않을 때 (아무것도 쓰지 않도록
            호출자가 트랜잭션을 되돌린다).
        RenderNotCurrentError: 블러본이 지금 승인된 블러 라벨로 렌더한 것이 아닐 때 (ADR 0024).

    부작용: `label_records`에 삭제 레코드와 새 라벨을 추가한다 (세션 잠금 아래).
    라벨링 버킷에서 블러본을 임시 디렉터리로 받는다.
    """
    first = _read_inputs(conn, session_id, client.version, policy)
    summary = ActionsSummary(first.version)
    # 사용 중지·승인 보류 세션은 묻지 않는다 (쓰기 직전에 한 번 더 본다)
    summary.blocked = write_blocker(conn, session_id)
    if summary.blocked is not None:
        return summary
    stream_id = first.session.reference_stream.stream_id
    results: dict[Hand, HandResult] = {}
    render: tuple[str, str, PrivacyPolicy] | None = None
    with tempfile.TemporaryDirectory() as tmp:
        video: Path | None = None
        if labeling is not None:
            # VLM에는 블러본만 보낸다 (원본 프레임이 VLM 서버로 나가지 않게,
            # 설명 초안에 개인정보가 들어가지 않게). 블러본은 프라이버시 승인 후 렌더된다.
            # 지금 승인된 블러 라벨로 렌더한 블러본이 아니면 멈춘다 (ADR 0024)
            privacy = load_privacy_policy(repo_root())
            video, before = fetch_current_blurred(
                conn, labeling, session_id, stream_id, Path(tmp), privacy
            )
            render = (stream_id, before, privacy)
        # VLM 호출은 세션을 잠그지 않는다 (긴 잠금을 피한다)
        for hand, track in sorted(first.tracks.items()):
            if first.done(hand):
                continue
            results[hand] = segment_hand(
                session_id=session_id,
                stream_id=stream_id,
                video=video,
                hand=hand,
                track=track,
                contacts=hand_contacts(first.current, hand),
                entities=first.entities,
                start_ms=0,
                end_ms=first.session.duration_ms,
                client=client,
                ontology=ontology,
                policy=policy,
                model_version=first.version,
                ontology_version=first.session.ontology_version or ontology.version,
                now=now,
            )
    _write(conn, session_id, first.version, client.version, policy, results, render, now, summary)
    return summary


def _write(
    conn: sa.Connection,
    session_id: str,
    version: str,
    vlm_version: str,
    policy: ActionsPolicy,
    results: dict[Hand, HandResult],
    render: tuple[str, str, PrivacyPolicy] | None,
    now: datetime,
    summary: ActionsSummary,
) -> None:
    """세션을 잠그고 쓰기 직전 확인을 한 뒤(막히면 아무것도 쓰지 않음, F-PAR-80), 지금 이력으로
    지울 것·남길 것을 다시 계획해 쓴다 (F-PAR-81).

    묻는 동안 검수자가 고친 행동(사람 레코드)·승인한 행동은 보호 구간이 되어 새 결과 중 겹치는
    것은 버리고 빈 곳은 미상으로 채운다 (`keep_protected`, 공백 0·겹침 없음). 그 사이 입력이 바뀌어
    버전이 달라졌으면 지난 입력으로 만든 결과이므로 쓰지 않는다 (다시 실행).
    """
    lock_session(conn, session_id)
    summary.blocked = write_blocker(conn, session_id, render)
    if summary.blocked is not None:
        return
    fresh = _read_inputs(conn, session_id, vlm_version, policy)
    if fresh.version != version:
        summary.blocked = (
            f"{session_id}: 묻는 동안 행동 단계 입력(손 트랙·손 상태·객체 트랙)이 바뀌었습니다. "
            "다시 실행하세요"
        )
        return
    # 손 트랙이 사라진 손: 새로 만들 입력이 없으므로 이 단계가 냈던 검수 전 행동·공백·설명만
    # 지운다. 검수된 것과 사람 레코드는 남긴다
    for hand in fresh.orphans():
        removed = retractions(fresh.with_descriptions(fresh.mine_of(hand)), version, now)
        if removed:
            insert_labels(conn, removed)
            summary.retracted += len(removed)
            summary.hands[hand.value] = {"retracted_without_track": len(removed)}
    for hand in sorted(fresh.tracks):
        result = results.get(hand)
        # 그 사이 같은 버전이 쓰였으면(동시 실행) 건너뛴다. 버전이 같으면 물은 손과 같다
        if fresh.done(hand) or result is None:
            summary.skipped.append(hand.value)
            continue
        protected = fresh.protected_of(hand)
        # 보호 구간과 겹치는 새 결과는 버리고 빈 곳은 미상으로 채운다.
        # 이전 버전의 검수 전 결과는 삭제 레코드로 지운다
        new = keep_protected(result.labels, protected, hand, version, now)
        removed = retractions(fresh.with_descriptions(fresh.mine_of(hand)), version, now)
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
