"""FiftyOne 연동 (`dlp active fiftyone`): 고른 세션을 FiftyOne 동영상 데이터셋으로 만들어 분석한다.

- 영상은 라벨링 버킷의 블러본만 쓴다 (원본 버킷은 읽지 않는다). FiftyOne은 로컬 파일 경로가 필요해
  캐시 디렉터리에 받는다.
- 라벨 시각(마스터 ms) → 스트림 시각 → 블러본 PTS 인덱스의 가장 가까운 프레임 (허용 오차 안에서만).
  FiftyOne 프레임 번호는 1부터다. 프레임 번호는 FiftyOne 안에서만 쓰고 저장소에는 남기지 않는다.
- 박스는 Detections, 관절은 Keypoints, 시간 구간(행동·손 상태 등)은 TemporalDetections로 넣는다.
  각 라벨에 검증 상태·출처·모델 버전·label_id를 붙여, 수정률 높은 클래스를 FiftyOne에서 바로 거른다.
- 샘플 필드: 세션 점수·순위·항목 점수·기여 상위 클래스. 데이터셋 info에 클래스별 수정률.

FiftyOne(약 1 GB, 내장 MongoDB)은 선택 설치다: make install-curation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from dlp_active.policy import ActivePolicy
from dlp_active.rates import RateTable
from dlp_active.select import SessionScore
from dlp_media.probe import probe
from dlp_media.pts import PtsIndex, build_pts_index
from dlp_media.storage import ObjectStore, sha256_file
from dlp_schema.db.repository import get_labels, get_session
from dlp_schema.episode import current_labels
from dlp_schema.history import label_class
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload, LabelRecord
from dlp_schema.session import Stream, StreamKind

SPATIAL = ("box_track", "keypoint_track")
VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


@dataclass(frozen=True)
class FoLabel:
    kind: str
    label: str  # 클래스
    label_id: str
    source: str
    verification: str
    model_version: str | None
    confidence: float | None
    box: tuple[float, float, float, float] | None = None  # 정규화 [x, y, w, h]
    points: tuple[tuple[float, float], ...] | None = None  # 정규화, 보이지 않는 관절은 NaN
    support: tuple[int, int] | None = None  # 시간 구간 [첫 프레임, 끝 프레임] (1부터)


@dataclass
class FoSample:
    filepath: Path
    session_id: str
    stream_id: str
    fields: dict[str, Any]
    frames: dict[int, list[FoLabel]] = field(default_factory=dict[int, list[FoLabel]])
    temporal: list[FoLabel] = field(default_factory=list[FoLabel])


def _frame(index: PtsIndex, stream: Stream, master_ms: float, tolerance_ms: int) -> int | None:
    """마스터 시각 → 블러본 프레임 번호(1부터). 가까운 프레임이 허용 오차 밖이면 None."""
    t = (master_ms - stream.offset_ms - stream.manual_adjustment_ms) / stream.clock_scale
    i = index.nearest(t)
    return i + 1 if abs(float(index.ms[i]) - t) <= tolerance_ms else None


def _meta(x: LabelRecord) -> dict[str, Any]:
    return {
        "label": label_class(x),
        "label_id": x.label_id,
        "source": x.provenance.source.value,
        "verification": x.verification.state.value,
        "model_version": x.provenance.model_version,
        "confidence": x.confidence,
    }


def stream_sample(
    session_id: str,
    video: Path,
    index: PtsIndex,
    size: tuple[int, int],
    stream: Stream,
    history: list[LabelRecord],
    policy: ActivePolicy,
    fields: dict[str, Any],
) -> FoSample:
    """세션 스트림 하나 → FiftyOne 샘플 (운영 현재 라벨만, 블러 등 제외 종류 빼고)."""
    w, h = size
    tol = policy.fiftyone.frame_tolerance_ms
    sample = FoSample(video, session_id, stream.stream_id, fields)
    for x in current_labels(history):
        if x.kind in policy.excluded_kinds:
            continue
        p = x.payload
        if x.stream_id == stream.stream_id and isinstance(p, BoxTrackPayload):
            for k in p.keyframes:
                f = None if k.outside else _frame(index, stream, k.t_ms, tol)
                if f is not None:
                    box = (k.x / w, k.y / h, k.w / w, k.h / h)
                    sample.frames.setdefault(f, []).append(FoLabel(x.kind, box=box, **_meta(x)))
        elif x.stream_id == stream.stream_id and isinstance(p, KeypointTrackPayload):
            for kf in p.keyframes:
                f = _frame(index, stream, kf.t_ms, tol)
                if f is not None:
                    pts = tuple(
                        (q.x / w, q.y / h) if q.visibility > 0 else (float("nan"), float("nan"))
                        for q in kf.points
                    )
                    sample.frames.setdefault(f, []).append(FoLabel(x.kind, points=pts, **_meta(x)))
        elif x.kind not in SPATIAL and x.stream_id in (None, stream.stream_id):
            a = _frame(index, stream, x.t_start_ms, 10**9)
            b = _frame(index, stream, x.t_end_ms, 10**9)
            if a is not None and b is not None:
                sample.temporal.append(FoLabel(x.kind, support=(a, max(a, b)), **_meta(x)))
    return sample


def score_fields(rank: int, s: SessionScore) -> dict[str, Any]:
    return {
        "active_rank": rank,
        "active_score": s.score,
        "active_pending": s.pending,
        "active_terms": dict(s.terms),
        "active_top_classes": [k for k, _ in s.top_classes],
    }


def rates_info(rates: RateTable) -> dict[str, Any]:
    return {
        "overall_correction_rate": rates.overall,
        "correction_rates": {
            k: {"reviewed": c.reviewed, "changed": c.changed, "rate": c.rate}
            for k, c in rates.classes.items()
        },
    }


class CurationUnavailableError(RuntimeError):
    pass


def push_to_fiftyone(name: str, samples: list[FoSample], info: dict[str, Any]) -> Any:
    """FiftyOne 데이터셋을 (같은 이름이면 덮어써) 만든다. fiftyone.Dataset을 돌려준다."""
    try:
        import fiftyone as fo  # pyright: ignore[reportMissingTypeStubs]
    except ImportError as exc:  # pragma: no cover - 설치 여부에 따라
        raise CurationUnavailableError("FiftyOne이 없습니다: make install-curation") from exc

    ds: Any = fo.Dataset(name, overwrite=True, persistent=True)
    ds.info = info
    out: list[Any] = []
    for s in samples:
        sample: Any = fo.Sample(
            filepath=str(s.filepath), session_id=s.session_id, stream_id=s.stream_id
        )
        for k, v in s.fields.items():
            sample[k] = v
        for f, labels in sorted(s.frames.items()):
            dets = [_detection(fo, x) for x in labels if x.box is not None]
            kps = [_keypoint(fo, x) for x in labels if x.points is not None]
            sample.frames[f] = fo.Frame(
                labels=fo.Detections(detections=dets), keypoints=fo.Keypoints(keypoints=kps)
            )
        sample["segments"] = fo.TemporalDetections(
            detections=[
                fo.TemporalDetection(
                    label=f"{x.kind}/{x.label}", support=list(x.support or (1, 1)), **_attrs(x)
                )
                for x in s.temporal
            ]
        )
        out.append(sample)
    ds.add_samples(out)
    return ds


def _attrs(x: FoLabel) -> dict[str, Any]:
    return {
        "label_id": x.label_id,
        "source": x.source,
        "verification": x.verification,
        "model_version": x.model_version,
        "confidence": x.confidence,
    }


def _detection(fo: Any, x: FoLabel) -> Any:
    return fo.Detection(label=x.label, bounding_box=list(x.box or ()), **_attrs(x))


def _keypoint(fo: Any, x: FoLabel) -> Any:
    return fo.Keypoint(label=x.label, points=[list(p) for p in x.points or ()], **_attrs(x))


def build_samples(
    conn: sa.Connection,
    labeling: ObjectStore,
    cache: Path,
    ranked: list[SessionScore],
    policy: ActivePolicy,
) -> tuple[list[FoSample], list[str]]:
    """고른 세션의 블러본을 받아 샘플을 만든다. 블러본이 없는 스트림은 건너뛰고 이유를 돌려준다."""
    samples: list[FoSample] = []
    notes: list[str] = []
    for rank, s in enumerate(ranked, start=1):
        session = get_session(conn, s.session_id)
        history = get_labels(conn, s.session_id)
        for stream in (x for x in session.streams if x.kind in VIDEO):
            key = f"sessions/{s.session_id}/blurred/{stream.stream_id}.mp4"
            head = labeling.head(key)
            if head is None:
                notes.append(f"{s.session_id}/{stream.stream_id}: 블러본이 없어 건너뜀")
                continue
            video = cache / s.session_id / f"{stream.stream_id}.mp4"
            if not video.exists() or sha256_file(video) != head.sha256:
                video.parent.mkdir(parents=True, exist_ok=True)
                labeling.get_file(key, video)
            info = probe(video).video
            if info is None:
                notes.append(f"{s.session_id}/{stream.stream_id}: 영상 트랙이 없음")
                continue
            samples.append(
                stream_sample(
                    s.session_id,
                    video,
                    build_pts_index(video),
                    (info.width, info.height),
                    stream,
                    history,
                    policy,
                    score_fields(rank, s),
                )
            )
    return samples, notes
