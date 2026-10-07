"""FiftyOne 연동 (`dlp active fiftyone`): 고른 세션을 FiftyOne 동영상 데이터셋으로 만들어 분석한다.

- 영상은 라벨링 버킷의 블러본만 쓴다 (원본 버킷은 읽지 않는다). FiftyOne은 로컬 파일 경로가 필요해
  캐시 디렉터리에 받는다.
- 공간 라벨(박스·관절)의 키프레임 시각은 이미 그 스트림의 PTS 시각이다 (ADR 0019). 그대로 블러본 PTS
  인덱스의 가장 가까운 프레임에 맞춘다 (허용 오차 안에서만). 시간 구간 라벨은 마스터 시각이라
  스트림 시각으로 바꾼 뒤 맞춘다. FiftyOne 프레임 번호는 1부터다. 프레임 번호는 FiftyOne
  안에서만 쓰고 저장소에는 남기지 않는다.
- 박스는 Detections, 관절은 Keypoints, 시간 구간(행동·손 상태 등)은 TemporalDetections로 넣는다.
  각 라벨에 검증 상태·출처·모델 버전·label_id를 붙여, 수정률 높은 클래스를 FiftyOne에서 바로 거른다.
- 샘플 필드: 세션 점수·순위·항목 점수·기여 상위 클래스. 데이터셋 info에 클래스별 수정률.
- 블러본은 지금 승인된 블러 라벨·정책으로 렌더한 것만 쓴다 (`assert_render_current`, ADR 0024).
  승인이 풀렸거나 렌더 기록이 없거나 다르면 그 스트림을 건너뛰고 이유를 남긴다.

FiftyOne(약 1 GB, 내장 MongoDB)은 선택 설치다: make install-curation.

공개 이름: `FoLabel`, `FoSample`, `stream_sample`, `score_fields`, `rates_info`, `push_to_fiftyone`,
`build_samples`, `CurationUnavailableError`.
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
from dlp_media.storage import ObjectStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderNotCurrentError, assert_render_current, check_fetched
from dlp_schema import repo_root
from dlp_schema.db.repository import get_labels, get_session
from dlp_schema.episode import current_labels
from dlp_schema.history import label_class
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload, LabelRecord
from dlp_schema.session import Stream, StreamKind

# 프레임 단위로 넣는 공간 라벨 종류 (나머지는 시간 구간으로 넣는다)
SPATIAL = ("box_track", "keypoint_track")
# 샘플로 만드는 영상 스트림 종류
VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


@dataclass(frozen=True)
class FoLabel:
    """FiftyOne에 넣을 라벨 하나 (fiftyone import 없이 만들 수 있게 중간 표현으로 둔다)."""

    kind: str  # 라벨 종류
    label: str  # 클래스
    label_id: str  # 내부 라벨 ID (FiftyOne은 내부 도구라 가명 처리하지 않는다)
    source: str  # human | model
    verification: str  # 검증 상태
    model_version: str | None
    confidence: float | None
    box: tuple[float, float, float, float] | None = None  # 정규화 [x, y, w, h]
    points: tuple[tuple[float, float], ...] | None = None  # 정규화, 보이지 않는 관절은 NaN
    support: tuple[int, int] | None = None  # 시간 구간 [첫 프레임, 끝 프레임] (1부터)


@dataclass
class FoSample:
    """FiftyOne 동영상 샘플 하나 (세션 스트림 하나)."""

    filepath: Path  # 캐시에 받은 블러본
    session_id: str
    stream_id: str
    fields: dict[str, Any]  # 샘플 필드 (`score_fields`)
    frames: dict[int, list[FoLabel]] = field(default_factory=dict[int, list[FoLabel]])  # 1부터
    temporal: list[FoLabel] = field(default_factory=list[FoLabel])  # 시간 구간 라벨


def _frame(index: PtsIndex, stream_ms: float, tolerance_ms: int) -> int | None:
    """스트림 시각 → 블러본 프레임 번호(1부터). 가까운 프레임이 허용 오차 밖이면 None."""
    i = index.nearest(stream_ms)
    return i + 1 if abs(float(index.ms[i]) - stream_ms) <= tolerance_ms else None


def _stream_ms(stream: Stream, master_ms: float) -> float:
    """마스터 시각 → 스트림 시각 (시간 구간 라벨용)."""
    return (master_ms - stream.offset_ms - stream.manual_adjustment_ms) / stream.clock_scale


def _meta(x: LabelRecord) -> dict[str, Any]:
    """`FoLabel`의 공통 필드 (클래스·ID·출처·검증 상태·모델 버전·신뢰도)."""
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
    """세션 스트림 하나 → FiftyOne 샘플 (운영 현재 라벨만, 블러 등 제외 종류 빼고).

    Args:
        session_id: 세션 ID.
        video: 블러본 로컬 경로.
        index: 블러본 PTS 인덱스.
        size: 블러본 (가로, 세로) 픽셀 (좌표 정규화용).
        stream: 스트림 (시간 구간 라벨의 마스터 → 스트림 시각 변환).
        history: 세션의 전체 라벨 이력 (여기서 `current_labels`로 거른다).
        policy: 정책 (`fiftyone.frame_tolerance_ms`, `excluded_kinds`).
        fields: 샘플 필드.

    규칙: 박스·관절 키프레임이 프레임 사이(허용 오차 밖)면 버린다. 박스의 outside 키프레임도 버린다.
    시간 구간은 시작·끝 시각에 가장 가까운 프레임으로 맞춘다 (허용 오차 사실상 무한, 영상 밖이면
    첫·마지막 프레임으로 붙는다).
    """
    w, h = size
    tol = policy.fiftyone.frame_tolerance_ms
    sample = FoSample(video, session_id, stream.stream_id, fields)
    for x in current_labels(history):
        if x.kind in policy.excluded_kinds:
            continue
        p = x.payload
        if x.stream_id == stream.stream_id and isinstance(p, BoxTrackPayload):
            for k in p.keyframes:
                f = None if k.outside else _frame(index, k.t_ms, tol)
                if f is not None:
                    box = (k.x / w, k.y / h, k.w / w, k.h / h)
                    sample.frames.setdefault(f, []).append(FoLabel(x.kind, box=box, **_meta(x)))
        elif x.stream_id == stream.stream_id and isinstance(p, KeypointTrackPayload):
            for kf in p.keyframes:
                f = _frame(index, kf.t_ms, tol)
                if f is not None:
                    pts = tuple(
                        (q.x / w, q.y / h) if q.visibility > 0 else (float("nan"), float("nan"))
                        for q in kf.points
                    )
                    sample.frames.setdefault(f, []).append(FoLabel(x.kind, points=pts, **_meta(x)))
        elif x.kind not in SPATIAL and x.stream_id in (None, stream.stream_id):
            # 시간 구간 라벨: 허용 오차 10**9 ms = 늘 가장 가까운 프레임
            a = _frame(index, _stream_ms(stream, x.t_start_ms), 10**9)
            b = _frame(index, _stream_ms(stream, x.t_end_ms), 10**9)
            if a is not None and b is not None:
                sample.temporal.append(FoLabel(x.kind, support=(a, max(a, b)), **_meta(x)))
    return sample


def score_fields(rank: int, s: SessionScore) -> dict[str, Any]:
    """세션 점수 → 샘플 필드 (active_rank·active_score·active_pending·active_terms·
    active_top_classes)."""
    return {
        "active_rank": rank,
        "active_score": s.score,
        "active_pending": s.pending,
        "active_terms": dict(s.terms),
        "active_top_classes": [k for k, _ in s.top_classes],
    }


def rates_info(rates: RateTable) -> dict[str, Any]:
    """수정률 표 → FiftyOne 데이터셋 info (전체·클래스별 수정률)."""
    return {
        "overall_correction_rate": rates.overall,
        "correction_rates": {
            k: {"reviewed": c.reviewed, "changed": c.changed, "rate": c.rate}
            for k, c in rates.classes.items()
        },
    }


class CurationUnavailableError(RuntimeError):
    """FiftyOne이 설치되지 않았다 (`make install-curation`)."""


def push_to_fiftyone(name: str, samples: list[FoSample], info: dict[str, Any]) -> Any:
    """FiftyOne 데이터셋을 (같은 이름이면 덮어써) 만든다. fiftyone.Dataset을 돌려준다.

    persistent=True라 FiftyOne 내장 DB에 남는다 (지우려면 `ds.delete()`).
    프레임 필드: `labels`(Detections), `keypoints`(Keypoints).
    샘플 필드 `segments`(TemporalDetections,
    라벨은 "<종류>/<클래스>").

    Raises:
        CurationUnavailableError: fiftyone을 import할 수 없을 때.
    """
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
    """FiftyOne 라벨에 붙이는 사용자 속성 (필터용)."""
    return {
        "label_id": x.label_id,
        "source": x.source,
        "verification": x.verification,
        "model_version": x.model_version,
        "confidence": x.confidence,
    }


def _detection(fo: Any, x: FoLabel) -> Any:
    """박스 라벨 → `fo.Detection` (bounding_box는 정규화 [x, y, w, h])."""
    return fo.Detection(label=x.label, bounding_box=list(x.box or ()), **_attrs(x))


def _keypoint(fo: Any, x: FoLabel) -> Any:
    """관절 라벨 → `fo.Keypoint` (보이지 않는 관절은 NaN 좌표)."""
    return fo.Keypoint(label=x.label, points=[list(p) for p in x.points or ()], **_attrs(x))


def build_samples(
    conn: sa.Connection,
    labeling: ObjectStore,
    cache: Path,
    ranked: list[SessionScore],
    policy: ActivePolicy,
) -> tuple[list[FoSample], list[str]]:
    """고른 세션의 블러본을 받아 샘플을 만든다. 블러본이 없는 스트림은 건너뛰고 이유를 돌려준다.

    Args:
        conn: DB 연결 (읽기만: 세션·라벨·프라이버시 상태).
        labeling: 라벨링 버킷 (블러본·렌더 기록). 원본 버킷은 쓰지 않는다.
        cache: 블러본 캐시 디렉터리 (`<cache>/<세션>/<스트림>.mp4`, sha256이 같으면
            다시 받지 않는다).
        ranked: `rank_sessions` 결과 (순위 = 목록 순서, 1부터).
        policy: 정책.

    Returns:
        (샘플 목록, 건너뛴 스트림과 이유 목록).
    """
    samples: list[FoSample] = []
    notes: list[str] = []
    privacy = load_privacy_policy(repo_root())
    for rank, s in enumerate(ranked, start=1):
        session = get_session(conn, s.session_id)
        history = get_labels(conn, s.session_id)
        for stream in (x for x in session.streams if x.kind in VIDEO):
            key = blurred_key(s.session_id, stream.stream_id)
            head = labeling.head(key)
            if head is None:
                notes.append(f"{s.session_id}/{stream.stream_id}: 블러본이 없어 건너뜀")
                continue
            try:
                # 지금 승인된 블러 라벨로 렌더한 블러본만 쓴다 (승인 취소·재승인 뒤 이전 것 금지)
                rendered = assert_render_current(
                    conn, labeling, s.session_id, stream.stream_id, privacy
                )
            except RenderNotCurrentError as exc:
                notes.append(f"{s.session_id}/{stream.stream_id}: 건너뜀 ({exc})")
                continue
            video = cache / s.session_id / f"{stream.stream_id}.mp4"
            if not video.exists() or sha256_file(video) != head.sha256:
                video.parent.mkdir(parents=True, exist_ok=True)
                labeling.get_file(key, video)
            try:
                # 받은 파일이 렌더 기록의 파일 해시와 같은지 (렌더 밖에서 덮어쓴 블러본 거부)
                check_fetched(video, rendered, s.session_id, stream.stream_id)
            except RenderNotCurrentError as exc:
                notes.append(f"{s.session_id}/{stream.stream_id}: 건너뜀 ({exc})")
                continue
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
