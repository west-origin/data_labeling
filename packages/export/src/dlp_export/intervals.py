"""구간 JSON (`dlp-intervals`): 세션마다 파일 하나 (dlp_schema.export.IntervalFile).

`dlp export intervals <버전>`이 쓴다. 출력: `out/intervals/<세션 가명>.json`.
JSON Schema는 `schemas/export_intervals.schema.json`(`make schemas`로 생성)이고 구매자에게
함께 준다.

- 넣는 라벨: 정책 `intervals.kinds`(행동·구간·손 상태·관계·커버리지 등 시간 구간 라벨).
  박스·키포인트 같은 공간 라벨은 COCO·LeRobot으로 나간다.
- 시각: `t_start_ms`·`t_end_ms`는 마스터 타임라인 시각(정수 ms, ADR 0019).
- 작업자·장소·세션·라벨 ID와 페이로드 속 세션 ID(`*_id`)는 가명이다 (ADR 0021·0027).
- 검수자 ID·검수 시각 같은 내부 정보는 넣지 않고 검증 상태만 남긴다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.source import ExportSource
from dlp_schema.export import ExportedLabel, ExportedStream, IntervalFile
from dlp_schema.labels import LabelRecord


def exported(x: LabelRecord, ids: Pseudonymizer) -> ExportedLabel:
    """검수자 ID 등 내부 정보는 빼고 검증 상태만 남긴다.

    라벨 ID와 페이로드 속 세션 ID는 이 내보내기의 가명으로 바꾼다 (ids).
    가명 처리가 꺼져 있으면 페이로드 객체를 그대로 넣는다.
    """
    payload = (
        x.payload
        if not ids.enabled
        else ids.payload_ids(x.session_id, x.payload.model_dump(mode="json"))
    )
    return ExportedLabel(
        label_id=ids.label(x.label_id),
        stream_id=x.stream_id,
        t_start_ms=x.t_start_ms,
        t_end_ms=x.t_end_ms,
        verification=x.verification.state,
        source=x.provenance.source,
        model_version=x.provenance.model_version,
        confidence=x.confidence,
        payload=payload,
    )


def write_intervals(
    src: ExportSource,
    policy: ExportPolicy,
    out: Path,
    *,
    export_id: str,
    now: datetime,
    ids: Pseudonymizer,
) -> dict[str, int]:
    """out/intervals/<세션 가명>.json. 세션 가명별 라벨 수를 돌려준다.

    작업자·장소·세션·라벨 ID는 ids로 가명 처리한다. 라벨은 (시작 시각, 라벨 ID) 순으로 정렬해
    같은 입력이면 같은 파일이 나온다. 라벨이 0개인 세션도 파일을 쓴다 (세션 목록과 맞추려고).

    Args:
        src: `load_source` 결과.
        policy: 내보내기 정책 (`intervals.kinds`).
        out: 내보내기 결과 루트 (임시 디렉터리).
        export_id: 내보내기 ID (파일에 기록).
        now: 내보내기 시각 (시간대 포함).
        ids: 이 내보내기의 가명 함수.
    """
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
            session_id=ids.session(s.session_id),
            split=es.split,
            domain=s.domain,
            worker_id=ids.worker(s.worker_id),
            site_id=ids.site(s.site_id),
            duration_ms=s.duration_ms,
            streams=tuple(ExportedStream(stream_id=st.stream_id, kind=st.kind) for st in s.streams),
            labels=tuple(exported(x, ids) for x in labels),
            exported_at=now,
        )
        (out / "intervals" / f"{f.session_id}.json").write_text(
            f.model_dump_json(indent=2), encoding="utf-8"
        )
        counts[f.session_id] = len(labels)
    return counts
