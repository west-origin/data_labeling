"""한 손의 행동 구간: 경계 후보 → 후보 구간 VLM 분류 → 병합·채우기 → 라벨."""

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
    candidates: list[Candidate]
    classified: list[Classified]
    labels: list[LabelRecord]

    @property
    def fallbacks(self) -> int:
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
