"""한 손의 행동 구간: 경계 후보 → 후보 구간 VLM 분류 → 병합·채우기 → 라벨 (WP10).

DB를 쓰지 않는 순수 흐름이다. `runner.run_actions`가 손마다 부르고, 테스트는 정답 라벨로 답하는
`OracleVlm`으로 직접 부른다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dlp_actions.assemble import merge_spans, to_labels
from dlp_actions.boundaries import Candidate, boundary_candidates, segments, wrist_series
from dlp_actions.policy import ActionsPolicy
from dlp_actions.vlm import Classified, SegmentRequest, VlmClient, classify
from dlp_schema.labels import Hand, KeypointTrackPayload, LabelRecord
from dlp_schema.ontology import Ontology


@dataclass
class HandResult:
    """한 손의 결과.

    Attributes:
        candidates: 경계 후보.
        classified: 후보 구간별 분류 결과 (시도 횟수·오류 포함).
        labels: 만든 action·gap·description 라벨 (아직 DB에 쓰지 않음).
    """

    candidates: list[Candidate]
    classified: list[Classified]
    labels: list[LabelRecord]

    @property
    def fallbacks(self) -> int:
        """재시도 후에도 응답이 규칙을 어겨 미상으로 둔 구간 수."""
        return sum(c.fallback for c in self.classified)


def segment_hand(
    *,
    session_id: str,
    stream_id: str,
    video: Path | None,
    hand: Hand,
    track: KeypointTrackPayload,
    contacts: list[tuple[int, int]],
    entities: tuple[str, ...],
    start_ms: int,
    end_ms: int,
    client: VlmClient,
    ontology: Ontology,
    policy: ActionsPolicy,
    model_version: str,
    ontology_version: str,
    now: datetime,
) -> HandResult:
    """한 손의 행동·사이 구간 라벨을 만든다.

    Args:
        session_id, stream_id: 세션과 영상 스트림 (바디캠).
        video: VLM에 보낼 프레임을 뽑을 블러본 로컬 경로. None이면 프레임 없이 묻는다.
        hand: 손.
        track: 그 손의 hand21 키포인트 트랙 (바디캠 PTS ms).
        contacts: 그 손의 접촉 구간 [(시작, 끝)] ms (손 상태 라벨).
        entities: 대상·도구 후보 개체 ID (VLM 응답을 이 안으로 제한).
        start_ms, end_ms: 나눌 구간 (보통 0 ~ 세션 길이).
        client: VLM 클라이언트.
        ontology: 온톨로지 (동작·사이 구간 목록).
        policy: 행동 구간 정책.
        model_version: 라벨 출처 모델 버전 (라벨 ID에도 들어간다).
        ontology_version: 라벨의 온톨로지 버전.
        now: 라벨 생성 시각 (시간대 필수).

    Returns:
        `HandResult`. 라벨의 action·gap은 [start_ms, end_ms]를 빈틈없이 덮는다.

    Raises:
        VlmUnavailableError: VLM 서버가 백오프 후에도 응답하지 않을 때.
    """
    times, xy, scale = wrist_series(track)
    candidates = boundary_candidates(times, xy, scale, contacts, policy.boundaries)
    classified: list[Classified] = []
    for s, e in segments(candidates, start_ms, end_ms, policy.boundaries.min_segment_ms):
        request = SegmentRequest(
            session_id=session_id,
            stream_id=stream_id,
            video=video,
            hand=hand,
            start_ms=s,
            end_ms=e,
            in_contact=any(cs < e and ce > s for cs, ce in contacts),
            entities=entities,
        )
        classified.append(
            classify(
                client, request, ontology, policy.vlm.max_retries, policy.vlm.unavailable_backoff_s
            )
        )
    reasons = {c.t_ms: c.reason for c in candidates}
    spans = merge_spans(classified, policy.vlm.default_confidence, contacts, reasons)
    labels = to_labels(
        spans,
        session_id=session_id,
        hand=hand,
        contacts=contacts,
        ontology_version=ontology_version,
        model_version=model_version,
        now=now,
    )
    return HandResult(candidates, classified, labels)
