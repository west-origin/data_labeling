"""구간 JSON (`dlp-intervals`): 세션마다 파일 하나 (dlp_schema.export.IntervalFile)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dlp_export.policy import ExportPolicy
from dlp_export.source import ExportSource
from dlp_schema.export import ExportedLabel, ExportedStream, IntervalFile
from dlp_schema.labels import LabelRecord


def exported(x: LabelRecord) -> ExportedLabel:
    """검수자 ID 등 내부 정보는 빼고 검증 상태만 남긴다."""
    return ExportedLabel(
        label_id=x.label_id,
        stream_id=x.stream_id,
        t_start_ms=x.t_start_ms,
        t_end_ms=x.t_end_ms,
        verification=x.verification.state,
        source=x.provenance.source,
        model_version=x.provenance.model_version,
        confidence=x.confidence,
        payload=x.payload,
    )


def write_intervals(
    src: ExportSource, policy: ExportPolicy, out: Path, *, export_id: str, now: datetime
) -> dict[str, int]:
    """out/intervals/<세션>.json. 세션별 라벨 수를 돌려준다."""
    (out / "intervals").mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for es in src.sessions:
        s = es.session
        labels = sorted(
            (x for x in es.labels if x.kind in policy.intervals.kinds),
            key=lambda x: (x.t_start_ms, x.label_id),
        )
        f = IntervalFile(
            export_id=export_id,
            dataset_version_id=src.version.version_id,
            ontology_version=src.version.ontology_version,
            session_id=s.session_id,
            split=es.split,
            domain=s.domain,
            worker_id=s.worker_id,
            site_id=s.site_id,
            duration_ms=s.duration_ms,
            streams=tuple(ExportedStream(stream_id=st.stream_id, kind=st.kind) for st in s.streams),
            labels=tuple(exported(x) for x in labels),
            exported_at=now,
        )
        (out / "intervals" / f"{s.session_id}.json").write_text(
            f.model_dump_json(indent=2), encoding="utf-8"
        )
        counts[s.session_id] = len(labels)
    return counts
