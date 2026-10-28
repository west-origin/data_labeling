"""세션 결과 미리보기 영상 (검수 도구 없이 프리라벨 결과를 눈으로 보기).

개발 DB에 있는 한 세션의 운영 라벨(`episode.current_labels`)을 로컬 영상 위에 그려 작은 H.264 MP4로
쓴다. 검수 도구(Label Studio·CVAT)를 열 수 없는 사람에게 결과를 보여 주려는 일회용 확인 도구이고
파이프라인 단계가 아니다. DB에 아무것도 쓰지 않는다.

그리는 것
- 손 2D 키포인트 (hand21, 왼손·오른손 색 구분, 손끝 다섯 점 강조). 가장 가까운 키프레임이
  `--kp-tol-ms` 안일 때만.
- 손 3D 상태 (아래 패널): 손목 표본의 방식(hand_fit·point_depth)·신뢰도·z(m), 같은 시각 손끝
  표본 수.
- 착용자 팔 (arm3, ADR 0058): 손목 → 팔꿈치 굵은 선과 팔꿈치 점(보일 때만), 패널에 손마다 팔꿈치
  보임·점수와 같은 시각 아래팔 3D 길이(손목 3D ↔ 팔꿈치 3D, m).
- 블러 탐지 (blur_track): 기본은 블러를 적용하지 않고 테두리만 그린다 (오탐을 눈으로 판단하게).
  `--apply-blur`면 원본 크기 프레임에서 블러 박스를 블러본과 같은 모자이크
  (`dlp_privacy.render.apply_blur`, privacy.yaml render 값)로 덮고 테두리는 그리지 않는다
  (밖에 보여 줄 영상용). 덮는 범위는 블러본 렌더가 덮는 범위를 늘 포함한다 (F-COE-78): 프레임마다
  렌더의 박스 규칙(`dlp_privacy.render.blur_tracks`의 `box_at`: outside가 아닌 키프레임 사이는
  간격과 무관하게 보간, 마지막 키프레임 뒤에도 유지)으로 구한 박스와 표시용 박스(키프레임 사이
  `--blur-gap-ms`까지 보간, 마지막 키프레임 뒤로 그만큼 유지)를 합친 외접 박스를 `--blur-pad`
  비율만큼 넓혀 덮는다.
  화면 대부분을 덮는 오탐(F-PMS-31) 때문에 `--blur-max-area`(화면 대비 넓이)를 넘는
  박스는 덮지 않을 수 있다 (`--blur-always`에 든 대상은 크기와 무관하게 덮는다).
  사람이 승인한 블러본이 아니므로 남은 노출이 없는지 사람이 끝까지 본다.
  테두리에는 대상 ID와 탐지기 이름(`privacy.yaml targets.*.detectors`를 표시용으로만 읽음),
  트랙 신뢰도를 적는다.
- 객체 박스 (box_track): 출처(모델 버전의 첫 토막: COCO 객체·OWLv2 등)·클래스·트랙 신뢰도.
  `--min-score` 미만 박스는 그리지 않고 패널에 출처별 개수와 많은 클래스만 적는다.
- 모양 마스크 (mask_track, ADR 0059): 반투명 채움 + 윤곽선. 손은 왼손 초록·오른손 주황, 객체는
  클래스마다 정해진 색(클래스 이름 해시). 마스크는 보간하지 않으므로 가장 가까운 키프레임이
  `--mask-tol-ms` 안일 때만 그 마스크를 그린다 (표시용 유지, 기본 처리 간격의 절반). 패널에 보이는
  마스크 수와 평균 점수를 적는다. `--extra-labels`(JSON Lines `LabelRecord`)로 DB에 쓰지 않은 마스크
  (파일럿 구간 실행 결과)를 더 그릴 수 있다.
- PTS 시각·범례.

시각 규칙: 프레임 시각은 영상의 PTS 인덱스(`dlp_media.pts.build_pts_index`)로만 정한다 (프레임 번호
x 간격 금지). 공간 라벨 키프레임은 그 스트림 PTS를 정수 ms로 반올림한 값이다. 키프레임 사이 박스는
간격이 `--max-interp-ms` 이하일 때 선형 보간하고, 마지막 키프레임 뒤에는 `--hold-ms`까지만 유지한다
(기본 0: 저장된 키프레임 그대로). 출력 영상의 프레임 PTS도 원본 PTS를 그대로 쓴다 (VFR 유지).

영상 입력: 원본 버킷을 읽지 않는다. 수집 전 로컬 사본(같은 PTS)을 `--video`로 주고, 수집 때 쓴
방향과 같게 `--rotate`(저장 픽셀 기준 반시계 각도, 표시 방향 태그는 무시)로 돌린다. 돌린 크기가
라벨 좌표계와 다르면 실패한다. 개인정보가 든 영상이므로 출력은 저장소 밖에 둔다. 오디오는 넣지
않는다.

정책 값을 쓰지 않는다 (문턱·유지 시간은 모두 표시용 옵션).

사용:
    uv run python scripts/preview_session.py pilot-real2 --video data/pilot/real1/test.mp4 \\
        --rotate 180 --start 205 --end 285 --out /tmp/preview.mp4 --hold-ms 466
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import av
import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from dlp_media.pts import PtsIndex, build_pts_index
from dlp_media.rle import decode_rle
from dlp_privacy.geometry import Box
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    Rle,
    Trajectory3DPayload,
)

ROOT = Path(__file__).resolve().parents[1]

Image = NDArray[np.uint8]

# MediaPipe hand21 연결 (손목 0, 엄지 1~4, 검지 5~8, 중지 9~12, 약지 13~16, 새끼 17~20)
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
)  # fmt: skip
FINGERTIPS = (4, 8, 12, 16, 20)
TIP_PARTS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")

# 색 (BGR)
HAND_COLORS = {"left": (80, 220, 80), "right": (40, 150, 255)}  # 초록, 주황
ARM_COLORS = {"left": (255, 230, 120), "right": (120, 230, 255)}  # 하늘, 노랑
BLUR_COLORS = {"face": (0, 0, 255), "reflection": (255, 0, 255), "shipping_label": (203, 160, 255)}
BLUR_COLOR = (40, 80, 230)  # 그 밖의 블러 대상
# 박스 글자 줄임말 (범례에 적는다)
SHORT = {
    "face": "face", "reflection": "refl", "document": "doc", "screen": "scr", "photo": "photo",
    "shipping_label": "ship", "address_sign": "addr", "yunet": "YN", "open_vocab": "OV",
    "codes+open_vocab": "QR+OV",
}  # fmt: skip
SOURCE_COLORS = {"coco": (255, 255, 0), "owlv2": (0, 220, 255)}
OTHER_COLOR = (200, 200, 200)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(frozen=True)
class BoxTrack:
    """그릴 박스 트랙 하나 (블러 또는 객체)."""

    label: str  # 박스 옆 글자 (클래스·대상)
    source: str  # 출처 이름 (블러: 탐지기, 객체: 모델 버전 첫 토막)
    confidence: float
    t_start_ms: int
    t_end_ms: int
    keyframes: tuple[BoxKeyframe, ...]
    times: tuple[int, ...]
    # 원 라벨 (블러 트랙만: 렌더 규칙 박스를 `dlp_privacy.render.blur_tracks`로 구한다)
    record: LabelRecord | None = field(default=None, compare=False, repr=False)


class RenderBoxes(Protocol):
    """블러본 렌더의 트랙 (`dlp_privacy.render.blur_tracks` 결과 하나)."""

    def box_at(self, t_ms: int) -> Box | None: ...


@dataclass(frozen=True)
class HandTrack:
    """손 2D 키포인트 트랙."""

    hand: str
    times: tuple[int, ...]
    points: tuple[NDArray[np.float64], ...]  # 키프레임마다 (21, 3): x, y, visibility


@dataclass(frozen=True)
class ArmTrack:
    """착용자 팔 arm3 트랙 (ADR 0058)."""

    hand: str
    times: tuple[int, ...]
    points: tuple[
        NDArray[np.float64], ...
    ]  # 키프레임마다 (3, 4): x, y, visibility, 신뢰도(-1 없음)


@dataclass(frozen=True)
class MaskTrack:
    """그릴 모양 마스크 트랙 (ADR 0059). hand: 손 마스크면 left/right, 객체면 None."""

    label: str  # 클래스
    hand: str | None
    times: tuple[int, ...]
    rles: tuple[Rle, ...]
    scores: tuple[float | None, ...]


@dataclass(frozen=True)
class Samples3D:
    """한 (손, 부위) 3D 궤적 표본 (여러 궤적을 시각 순으로 합침). 팔은 부위 `arm_<점>`."""

    times: tuple[int, ...]
    methods: tuple[str, ...]
    confidences: tuple[float | None, ...]
    xyz: tuple[tuple[float, float, float], ...]

    @property
    def z(self) -> tuple[float, ...]:
        """표본 z (m)."""
        return tuple(p[2] for p in self.xyz)


@dataclass
class Overlays:
    """한 세션 구간의 그릴 거리."""

    blur: list[BoxTrack] = field(default_factory=list)
    objects: list[BoxTrack] = field(default_factory=list)
    hands: list[HandTrack] = field(default_factory=list)
    arms: list[ArmTrack] = field(default_factory=list)
    hand3d: dict[tuple[str, str], Samples3D] = field(default_factory=dict)
    masks: list[MaskTrack] = field(default_factory=list)
    # blur와 같은 순서의 렌더 규칙 트랙 (`attach_render_boxes`).
    # 없으면 _mosaic이 키프레임 시각을 프레임으로 보고 만든다
    blur_render: list[RenderBoxes | None] = field(default_factory=list[RenderBoxes | None])


@dataclass(frozen=True)
class DrawOptions:
    """표시 옵션 (정책 값이 아니다)."""

    min_score: float = 0.3
    hold_ms: int = 0
    max_interp_ms: int = 1500
    kp_tol_ms: int = 40
    tol_3d_ms: int = 150
    mask_tol_ms: int = 260
    mask_alpha: float = 0.45
    width: int = 540
    # --apply-blur: 블러 박스를 모자이크로 덮는다
    # (넓힘 비율, 보간·유지 ms, 모자이크 최소 블록·블록 수)
    apply_blur: bool = False
    blur_pad: float = 0.15
    blur_gap_ms: int = 1500
    blur_min_block: int = 6
    blur_blocks: int = 5
    # 화면 대비 이 넓이를 넘는 박스는 덮지 않는다 (1.0 = 모두 덮음). blur_always 대상은 예외
    blur_max_area: float = 1.0
    blur_always: frozenset[str] = frozenset()
    # 이 대상의 박스는 중심이 손 위(손 키포인트 외접 박스를 30 % 넓힌 안)면 덮지 않는다
    # (손을 얼굴로 본 오탐 F-PMS-31, 표시용. 라벨은 바꾸지 않는다)
    blur_skip_on_hand: frozenset[str] = frozenset()
    # --clean: 아래 진단 패널 없이 영상만 (발표용)
    clean: bool = False
    # --no-overlays: 라벨을 그리지 않고 (블러만 적용한) 영상만 (검수 도구 시연용 클립)
    no_overlays: bool = False


# ---------------------------------------------------------------- 라벨 → 그릴 거리


def _source_name(model_version: str | None) -> str:
    """모델 버전 문자열에서 출처 이름 (첫 토막, mediapipe 객체 탐지기는 coco)."""
    if not model_version:
        return "human"
    head = model_version.split("-")[0]
    if model_version.startswith("mediapipe-object_detector"):
        return "coco"
    return head


def privacy_detectors(path: Path) -> dict[str, str]:
    """프라이버시 정책의 대상 → 탐지기 이름 (표시용, 파일이 없으면 빈 사전)."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    targets = data.get("targets") or {}
    return {str(k): "+".join(v.get("detectors") or []) for k, v in targets.items()}


def _box_track(
    label: str,
    source: str,
    rec: LabelRecord,
    kfs: tuple[BoxKeyframe, ...],
    *,
    keep_record: bool = False,
) -> BoxTrack:
    return BoxTrack(
        label=label,
        source=source,
        confidence=float(rec.confidence if rec.confidence is not None else 1.0),
        t_start_ms=rec.t_start_ms,
        t_end_ms=rec.t_end_ms,
        keyframes=kfs,
        times=tuple(k.t_ms for k in kfs),
        record=rec if keep_record else None,
    )


def _mask_track(p: MaskTrackPayload, start_ms: int, end_ms: int) -> MaskTrack:
    """mask_track → 그릴 마스크 (구간 근처의 보이는 키프레임만, 손은 entity `<손>_hand`)."""
    kfs = [k for k in p.keyframes if not k.outside and k.rle is not None
           and start_ms - 1000 <= k.t_ms <= end_ms + 1000]  # fmt: skip
    side = p.entity_id.removesuffix("_hand")
    hand = side if p.class_id == "hand" and side in ("left", "right") else None
    return MaskTrack(p.class_id, hand, tuple(k.t_ms for k in kfs),
                     tuple(k.rle for k in kfs if k.rle is not None),
                     tuple(k.score for k in kfs))  # fmt: skip


def class_color(name: str) -> tuple[int, int, int]:
    """클래스 이름 → 정해진 밝은 BGR 색 (해시 색상환, 실행마다 같다)."""
    hue = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") * 2654435761 % 180
    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


Row3D = tuple[int, str, float | None, tuple[float, float, float]]


def _add_3d(acc: dict[tuple[str, str], list[Row3D]], p: Trajectory3DPayload) -> None:
    """손 관절·팔 점 카메라 좌표 궤적 표본을 (손, 부위)별로 모은다 (팔은 `arm_<점>`)."""
    if p.frame.value != "camera" or p.part is None:
        return
    if p.entity_id.endswith("_hand"):
        hand, part = p.entity_id.removesuffix("_hand"), p.part
    elif p.entity_id.endswith("_arm"):
        hand, part = p.entity_id.removesuffix("_arm"), f"arm_{p.part}"
    else:
        return
    for s in p.samples:
        method = s.method.value if s.method is not None else "?"
        acc[(hand, part)].append((s.t_ms, method, s.confidence, (s.x, s.y, s.z)))


def build_overlays(
    labels: Iterable[LabelRecord],
    *,
    stream_id: str,
    start_ms: int,
    end_ms: int,
    detectors: dict[str, str] | None = None,
) -> Overlays:
    """운영 라벨에서 구간 [start_ms, end_ms]에 걸치는 그릴 거리를 고른다.

    Args:
        labels: 운영 현재 라벨 (`current_labels` 결과).
        stream_id: 그릴 영상 스트림.
        start_ms / end_ms: 구간 (그 스트림 PTS ms).
        detectors: 블러 대상 → 탐지기 이름 (표시용).
    """
    det = detectors or {}
    out = Overlays()
    acc3d: dict[tuple[str, str], list[Row3D]] = collections.defaultdict(list)
    for rec in labels:
        if rec.stream_id != stream_id or rec.t_start_ms > end_ms:
            continue
        p = rec.payload
        # 블러는 렌더가 마지막 키프레임 뒤에도 박스를 유지하므로 구간 앞에서 끝난 트랙도
        # 둔다 (F-COE-78)
        if rec.t_end_ms < start_ms - 2000 and not isinstance(p, BlurTrackPayload):
            continue
        match p:
            case BlurTrackPayload():
                out.blur.append(
                    _box_track(p.target, det.get(p.target, "?"), rec, p.keyframes, keep_record=True)
                )
            case BoxTrackPayload():
                src = _source_name(rec.provenance.model_version)
                out.objects.append(_box_track(p.class_id, src, rec, p.keyframes))
            case KeypointTrackPayload() if p.skeleton == "hand21" and p.hand is not None:
                frames = [f for f in p.keyframes if start_ms - 100 <= f.t_ms <= end_ms + 100]
                pts = tuple(
                    np.array([(k.x, k.y, k.visibility) for k in f.points], dtype=np.float64)
                    for f in frames
                )
                out.hands.append(HandTrack(p.hand.value, tuple(f.t_ms for f in frames), pts))
            case KeypointTrackPayload() if p.skeleton == "arm3" and p.hand is not None:
                frames = [f for f in p.keyframes if start_ms - 100 <= f.t_ms <= end_ms + 100]
                arm = tuple(
                    np.array(
                        [(k.x, k.y, k.visibility, -1.0 if k.confidence is None else k.confidence)
                         for k in f.points], dtype=np.float64,
                    )
                    for f in frames
                )  # fmt: skip
                out.arms.append(ArmTrack(p.hand.value, tuple(f.t_ms for f in frames), arm))
            case Trajectory3DPayload():
                _add_3d(acc3d, p)
            case MaskTrackPayload():
                out.masks.append(_mask_track(p, start_ms, end_ms))
            case _:
                pass
    for key, rows in acc3d.items():
        rows.sort(key=lambda r: r[0])
        out.hand3d[key] = Samples3D(
            tuple(r[0] for r in rows), tuple(r[1] for r in rows),
            tuple(r[2] for r in rows), tuple(r[3] for r in rows),
        )  # fmt: skip
    return out


# ---------------------------------------------------------------- 시각별 값


def box_at(
    track: BoxTrack, t_ms: float, opts: DrawOptions
) -> tuple[float, float, float, float] | None:
    """시각 t_ms의 박스 (x, y, w, h, 원본 픽셀) 또는 None (안 보임·범위 밖)."""
    if t_ms < track.t_start_ms or t_ms > track.t_end_ms + opts.hold_ms:
        return None
    i = bisect.bisect_right(track.times, t_ms) - 1
    if i < 0:
        return None
    a = track.keyframes[i]
    if a.outside:
        return None
    if i + 1 < len(track.keyframes):
        b = track.keyframes[i + 1]
        gap = b.t_ms - a.t_ms
        if not b.outside and gap <= opts.max_interp_ms:
            r = (t_ms - a.t_ms) / gap
            return (a.x + r * (b.x - a.x), a.y + r * (b.y - a.y),
                    a.w + r * (b.w - a.w), a.h + r * (b.h - a.h))  # fmt: skip
    if t_ms - a.t_ms > opts.hold_ms:
        return None
    return (a.x, a.y, a.w, a.h)


def attach_render_boxes(ov: Overlays, frame_times: Sequence[int]) -> None:
    """블러 트랙마다 블러본 렌더와 같은 박스 규칙 트랙을 붙인다 (F-COE-78).

    frame_times: 그 영상 전체 프레임 시각 (PTS 인덱스 ms를 반올림한 정수, 렌더와 같은 식).
    """
    from dlp_privacy.render import blur_tracks

    built = iter(
        blur_tracks([tr.record for tr in ov.blur if tr.record is not None], list(frame_times))
    )
    ov.blur_render = [next(built) if tr.record is not None else None for tr in ov.blur]


def _render_box(
    tr: BoxTrack, rt: RenderBoxes | None, t_ms: int
) -> tuple[float, float, float, float] | None:
    """렌더 규칙 박스 (x, y, w, h). 렌더 트랙이 없으면 키프레임 시각을 프레임으로 보고 만든다."""
    if rt is None:
        from dlp_privacy.render import blur_tracks

        if tr.record is None:
            return None
        rt = blur_tracks([tr.record], sorted(set(tr.times)))[0]
    box = rt.box_at(t_ms)
    return None if box is None else (box.x, box.y, box.w, box.h)


def blur_cover_at(
    tr: BoxTrack, t_ms: int, opts: DrawOptions, rt: RenderBoxes | None = None
) -> tuple[float, float, float, float] | None:
    """`--apply-blur`가 덮을 박스 (넓힘 전, x, y, w, h). 렌더가 덮는 프레임은 늘 덮는다.

    렌더 규칙 박스와 표시용 박스(`blur_gap_ms`까지 보간·유지)의 외접 박스.
    """
    loose = replace(opts, hold_ms=opts.blur_gap_ms, max_interp_ms=opts.blur_gap_ms)
    boxes = [b for b in (_render_box(tr, rt, t_ms), box_at(tr, t_ms, loose)) if b is not None]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _nearest(times: Sequence[int], t_ms: float, tol: float) -> int | None:
    """정렬된 times에서 t_ms에 가장 가까운 위치 (tol 밖이면 None)."""
    if not times:
        return None
    i = bisect.bisect_left(times, t_ms)
    best = min((j for j in (i - 1, i) if 0 <= j < len(times)), key=lambda j: abs(times[j] - t_ms))
    return best if abs(times[best] - t_ms) <= tol else None


def hand_status(ov: Overlays, hand: str, t_ms: float, opts: DrawOptions) -> str:
    """손 3D 상태 한 줄 (손목 방식·신뢰도·z, 손끝 표본 수)."""
    wrist = ov.hand3d.get((hand, "wrist"))
    j = _nearest(wrist.times, t_ms, opts.tol_3d_ms) if wrist else None
    if wrist is None or j is None:
        return f"{hand[0].upper()} 3D: -"
    conf = wrist.confidences[j]
    c = f"{conf:.2f}" if conf is not None else "?"
    tips = sum(
        1
        for part in TIP_PARTS
        if (s := ov.hand3d.get((hand, part)))
        and _nearest(s.times, t_ms, opts.tol_3d_ms) is not None
    )
    return f"{hand[0].upper()} 3D wrist {wrist.methods[j]} c={c} z={wrist.z[j]:.2f}m tips {tips}/5"


def arm_status(ov: Overlays, hand: str, t_ms: float, opts: DrawOptions) -> str:
    """착용자 팔 한 줄: 팔꿈치 2D 보임·점수, 같은 시각 아래팔 3D 길이 (ADR 0058)."""
    tag = f"{hand[0].upper()} arm"
    track = next((a for a in ov.arms if a.hand == hand), None)
    j = _nearest(track.times, t_ms, opts.kp_tol_ms) if track else None
    if track is None or j is None:
        return f"{tag}: -"
    elbow = track.points[j][1]
    text = f"{tag}: elbow " + (f"2D s={elbow[3]:.2f}" if elbow[2] > 0 else "not visible")
    wrist, e3 = ov.hand3d.get((hand, "wrist")), ov.hand3d.get((hand, "arm_elbow"))
    if wrist and e3:
        i, k = _nearest(wrist.times, t_ms, opts.tol_3d_ms), _nearest(e3.times, t_ms, opts.tol_3d_ms)
        if i is not None and k is not None and wrist.times[i] == e3.times[k]:
            length = float(np.linalg.norm(np.subtract(wrist.xyz[i], e3.xyz[k])))
            text += f" | forearm 3D {length:.2f} m"
    return text


# ---------------------------------------------------------------- 그리기


def _text(
    img: Image, s: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.33
) -> None:
    """검은 테두리 글자."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def _plain(
    img: Image, s: str, org: tuple[int, int], color: tuple[int, int, int], scale: float
) -> None:
    """테두리 없는 글자 (어두운 패널용)."""
    cv2.putText(img, s, org, FONT, scale, color, 1, cv2.LINE_AA)


def _draw_boxes(
    img: Image,
    tracks: list[BoxTrack],
    t_ms: int,
    sc: tuple[float, float],
    opts: DrawOptions,
    *,
    blur: bool,
) -> collections.Counter[tuple[str, str]]:
    """박스를 그리고, 문턱 미만이라 그리지 않은 (출처, 클래스) 수를 돌려준다."""
    hidden: collections.Counter[tuple[str, str]] = collections.Counter()
    for tr in tracks:
        box = box_at(tr, t_ms, opts)
        if box is None:
            continue
        if not blur and tr.confidence < opts.min_score:
            hidden[(tr.source, tr.label)] += 1
            continue
        x, y, w, h = box[0] * sc[0], box[1] * sc[1], box[2] * sc[0], box[3] * sc[1]
        if blur:
            color = BLUR_COLORS.get(tr.label, BLUR_COLOR)
            det = "RF" if tr.source == "reflection" else SHORT.get(tr.source, tr.source)
            tag = f"{SHORT.get(tr.label, tr.label)}/{det} {tr.confidence:.2f}"
        else:
            color = SOURCE_COLORS.get(tr.source, OTHER_COLOR)
            tag = f"{tr.label} {tr.confidence:.2f}"
        p1, p2 = (round(x), round(y)), (round(x + w), round(y + h))
        cv2.rectangle(img, p1, p2, color, 1, cv2.LINE_AA)
        ty = p1[1] + 10 if blur else max(p1[1] - 3, 9)
        _text(img, tag, (p1[0] + 2, ty), color, 0.3)
    return hidden


def _draw_hands(
    img: Image, ov: Overlays, t_ms: int, sc: tuple[float, float], opts: DrawOptions
) -> set[str]:
    """손 키포인트를 그리고 그린 손 이름을 돌려준다."""
    drawn: set[str] = set()
    for tr in ov.hands:
        j = _nearest(tr.times, t_ms, opts.kp_tol_ms)
        if j is None:
            continue
        color = HAND_COLORS.get(tr.hand, OTHER_COLOR)
        pts = tr.points[j]
        xy = [(round(p[0] * sc[0]), round(p[1] * sc[1])) for p in pts]
        for a, b in HAND_EDGES:
            if pts[a][2] > 0 and pts[b][2] > 0:
                cv2.line(img, xy[a], xy[b], color, 1, cv2.LINE_AA)
        for k, p in enumerate(xy):
            if pts[k][2] <= 0:
                continue
            if k in FINGERTIPS:
                cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, p, 4, color, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, 2, color, -1, cv2.LINE_AA)
        drawn.add(tr.hand)
    return drawn


def _draw_arms(
    img: Image, ov: Overlays, t_ms: int, sc: tuple[float, float], opts: DrawOptions
) -> None:
    """착용자 팔: 보이는 점끼리 굵은 선(손목 → 팔꿈치 → 어깨)과 팔꿈치 점."""
    for tr in ov.arms:
        j = _nearest(tr.times, t_ms, opts.kp_tol_ms)
        if j is None:
            continue
        color = ARM_COLORS.get(tr.hand, OTHER_COLOR)
        pts = tr.points[j]
        xy = [(round(p[0] * sc[0]), round(p[1] * sc[1])) for p in pts]
        for a, b in ((0, 1), (1, 2)):
            if pts[a][2] > 0 and pts[b][2] > 0:
                cv2.line(img, xy[a], xy[b], color, 3, cv2.LINE_AA)
        if pts[1][2] > 0:
            cv2.circle(img, xy[1], 6, color, -1, cv2.LINE_AA)
            cv2.circle(img, xy[1], 6, (40, 40, 40), 1, cv2.LINE_AA)
            _text(img, "elbow", (xy[1][0] + 8, xy[1][1] + 4), color, 0.32)


def _draw_masks(
    img: Image, ov: Overlays, t_ms: int, opts: DrawOptions
) -> list[tuple[MaskTrack, float | None]]:
    """모양 마스크: 반투명 채움 + 윤곽선 (객체 먼저, 손을 위에). 그린 (트랙, 점수)를 돌려준다."""
    h, w = img.shape[:2]
    drawn: list[tuple[MaskTrack, float | None]] = []
    for tr in sorted(ov.masks, key=lambda m: m.hand is not None):
        j = _nearest(tr.times, t_ms, opts.mask_tol_ms)
        if j is None:
            continue
        m = cv2.resize(decode_rle(tr.rles[j]).astype(np.uint8), (w, h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)  # fmt: skip
        if not m.any():
            continue
        color = HAND_COLORS.get(tr.hand, OTHER_COLOR) if tr.hand else class_color(tr.label)
        fill = np.array(color, dtype=np.float32)
        img[m] = (img[m].astype(np.float32) * (1 - opts.mask_alpha) + fill * opts.mask_alpha
                  ).astype(np.uint8)  # fmt: skip
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # fmt: skip
        cv2.drawContours(img, contours, -1, color, 1, cv2.LINE_AA)
        if tr.hand is None:
            ys, xs = np.nonzero(m)
            _text(img, tr.label, (int(xs.min()) + 2, int(ys.max()) - 3), color, 0.3)
        drawn.append((tr, tr.scores[j]))
    return drawn


def _mask_line(drawn: list[tuple[MaskTrack, float | None]]) -> str:
    """보이는 마스크 요약 한 줄 (객체 수, 손, 평균 점수)."""
    objects = [d for d in drawn if d[0].hand is None]
    hands = "".join(sorted(d[0].hand[0].upper() for d in drawn if d[0].hand))
    scores = [s for _, s in drawn if s is not None]
    mean = f"{float(np.mean(scores)):.2f}" if scores else "-"
    top = ", ".join(f"{c} {n}" for c, n in
                    collections.Counter(d[0].label for d in objects).most_common(4))  # fmt: skip
    return f"masks: objects {len(objects)} ({top}) hands {hands or '-'} mean score {mean}"


def _hidden_line(name: str, hidden: collections.Counter[tuple[str, str]]) -> str:
    """문턱 미만 박스 요약 한 줄 (출처 하나)."""
    rows = {k[1]: v for k, v in hidden.items() if k[0] == name}
    top = ", ".join(f"{c} {n}" for c, n in collections.Counter(rows).most_common(3))
    return f"{sum(rows.values())}" + (f" ({top})" if top else "")


def _blur_line(ov: Overlays, t_ms: int, opts: DrawOptions) -> str:
    """지금 보이는 블러 박스 수 (대상별)."""
    c = collections.Counter(tr.label for tr in ov.blur if box_at(tr, t_ms, opts) is not None)
    parts = " ".join(f"{SHORT.get(k, k)} {v}" for k, v in c.most_common())
    return f"privacy det. boxes (NOT blurred) {c.total()}: {parts}"


LEGEND = (
    "hand: green=L orange=R white=fingertip | arm: sky=L yellow=R wrist->elbow (only if seen)",
    "mask: filled+outline, hands green/orange, objects colour per class (nearest keyframe)",
    "obj: cyan=COCO yellow=OWLv2, score=track conf (wearer's own body boxes are suppressed)",
    "privacy: red=face/YN magenta=refl/RF pink=ship/QR+OV orange=doc,scr,photo,addr/OV",
)


def _hand_boxes(ov: Overlays, t_ms: int, tol_ms: int) -> list[tuple[float, float, float, float]]:
    """그 시각 보이는 손 키포인트의 외접 박스 (x0, y0, x1, y1, 원본 픽셀, 30 % 넓힘)."""
    out: list[tuple[float, float, float, float]] = []
    for tr in ov.hands:
        j = _nearest(tr.times, t_ms, tol_ms)
        if j is None:
            continue
        pts = tr.points[j]
        seen = pts[pts[:, 2] > 0]
        if len(seen) < 3:
            continue
        x0, y0 = seen[:, 0].min(), seen[:, 1].min()
        x1, y1 = seen[:, 0].max(), seen[:, 1].max()
        px, py = (x1 - x0) * 0.3, (y1 - y0) * 0.3
        out.append((x0 - px, y0 - py, x1 + px, y1 + py))
    return out


def _mosaic(img: Image, tracks: list[BoxTrack], t_ms: int, opts: DrawOptions, ov: Overlays) -> None:
    """원본 크기 프레임의 블러 박스를 제자리에서 모자이크한다 (`--apply-blur`).

    블러본 렌더와 같은 함수(`dlp_privacy.render.apply_blur`)와 같은 박스 규칙(`blur_cover_at`)을
    쓰고, 표시용 보간·유지(`blur_gap_ms`)와 넓힘(`blur_pad`)으로 렌더보다 넓게 덮는다.
    """
    from dlp_privacy.render import apply_blur

    frame_area = float(img.shape[0] * img.shape[1])
    hands = _hand_boxes(ov, t_ms, 200) if opts.blur_skip_on_hand else []
    aligned = ov.blur_render if tracks is ov.blur and len(ov.blur_render) == len(tracks) else []
    for k, tr in enumerate(tracks):
        box = blur_cover_at(tr, t_ms, opts, aligned[k] if aligned else None)
        if box is None:
            continue
        x, y, w, h = box
        if tr.label not in opts.blur_always and w * h > opts.blur_max_area * frame_area:
            continue
        cx, cy = x + w / 2, y + h / 2
        if tr.label in opts.blur_skip_on_hand and any(
            a <= cx <= c and b <= cy <= d for a, b, c, d in hands
        ):
            continue
        px, py = w * opts.blur_pad, h * opts.blur_pad
        apply_blur(img, Box(x - px, y - py, w + 2 * px, h + 2 * py), "mosaic",
                   opts.blur_min_block, opts.blur_blocks)  # fmt: skip


def draw_frame(
    img: Image, t_ms: float, ov: Overlays, opts: DrawOptions, *, session_id: str, panel_h: int = 190
) -> Image:
    """원본 크기 BGR 프레임 하나에 오버레이를 그린 축소 영상(+아래 패널)을 돌려준다.

    라벨은 정수 ms로 반올림한 프레임 시각(`round(t_ms)`)으로 찾는다 (키프레임 저장 규칙과 같다).
    """
    h0, w0 = img.shape[:2]
    if opts.apply_blur:
        _mosaic(img, ov.blur, round(t_ms), opts, ov)
    vh = round(h0 * opts.width / w0 / 2) * 2
    small = cv2.resize(img, (opts.width, vh), interpolation=cv2.INTER_AREA)
    sc = (opts.width / w0, vh / h0)
    key = round(t_ms)
    if opts.no_overlays:
        return small
    masks = _draw_masks(small, ov, key, opts)
    if not opts.apply_blur:
        _draw_boxes(small, ov.blur, key, sc, opts, blur=True)
    hidden = _draw_boxes(small, ov.objects, key, sc, opts, blur=False)
    _draw_arms(small, ov, key, sc, opts)
    hands = _draw_hands(small, ov, key, sc, opts)
    if opts.clean:
        return small

    panel = np.full((panel_h, opts.width, 3), 24, dtype=np.uint8)
    y = 13
    _plain(panel, f"{session_id}  t={t_ms / 1000:8.3f} s (PTS)  hands 2D: "
          f"{'L' if 'left' in hands else '-'}{'R' if 'right' in hands else '-'}", (6, y),
          (255, 255, 255), 0.38)  # fmt: skip
    for hand in ("left", "right"):
        y += 15
        _plain(panel, hand_status(ov, hand, key, opts), (6, y), HAND_COLORS[hand], 0.36)
    for hand in ("left", "right"):
        y += 15
        _plain(panel, arm_status(ov, hand, key, opts), (6, y), ARM_COLORS[hand], 0.36)
    y += 15
    _plain(panel, _mask_line(masks), (6, y), (230, 230, 120), 0.34)
    y += 15
    _plain(panel, _blur_line(ov, key, opts), (6, y), (120, 120, 255), 0.34)
    for name in ("coco", "owlv2"):
        y += 15
        _plain(panel, f"{name} < {opts.min_score:.2f} hidden: {_hidden_line(name, hidden)}",
              (6, y), SOURCE_COLORS[name], 0.34)  # fmt: skip
    for line in LEGEND:
        y += 13
        _plain(panel, line, (6, y), (190, 190, 190), 0.28)
    return np.vstack([small, panel])


# ---------------------------------------------------------------- 영상 읽기·쓰기


ROTATIONS = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_CLOCKWISE}  # fmt: skip


def iter_frames(
    video: Path, start_ms: float, end_ms: float, rotate: int, index: PtsIndex
) -> Iterator[tuple[int, float, Image]]:
    """[start_ms, end_ms] 프레임을 (PTS, PTS 인덱스 시각 ms, BGR 영상)으로 낸다.

    시각은 PTS 인덱스에서 찾는다 (디코딩 PTS가 인덱스에 없으면 실패). 회전은 저장 픽셀 기준 반시계.

    Raises:
        ValueError: 디코딩 PTS가 인덱스에 없을 때.
    """
    rot = ROTATIONS[rotate]
    with av.open(str(video)) as c:
        stream = c.streams.video[0]
        tb = Fraction(stream.time_base.numerator, stream.time_base.denominator)  # type: ignore[union-attr]
        c.seek(int(Fraction(start_ms) / 1000 / tb), stream=stream, backward=True)
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            i = int(np.searchsorted(index.pts, frame.pts))
            if i >= len(index) or int(index.pts[i]) != frame.pts:
                raise ValueError(f"PTS {frame.pts}가 PTS 인덱스에 없습니다")
            t_ms = float(index.ms[i])
            if t_ms < start_ms:
                continue
            if t_ms > end_ms:
                break
            img = frame.to_ndarray(format="bgr24")
            if rot is not None:
                img = cv2.rotate(img, rot)
            yield int(frame.pts), t_ms, np.ascontiguousarray(img)


def render(
    video: Path,
    out: Path,
    ov: Overlays,
    opts: DrawOptions,
    *,
    session_id: str,
    start_ms: float,
    end_ms: float,
    rotate: int = 0,
    expect_size: tuple[int, int] | None = None,
    crf: int = 26,
) -> int:
    """미리보기 MP4를 쓴다. 반환: 쓴 프레임 수.

    Args:
        expect_size: 돌린 프레임의 (폭, 높이). 라벨 좌표계와 다르면 실패한다.

    Raises:
        ValueError: 구간에 프레임이 없거나 크기가 expect_size와 다를 때.
    """
    index = build_pts_index(video)
    tb = index.time_base
    if opts.apply_blur:
        # 렌더(render_blurred)와 같은 프레임 시각: PTS ms를 반올림한 정수
        attach_render_boxes(ov, [round(float(x)) for x in index.ms])
    n = 0
    with av.open(str(out), "w") as oc:
        vs = oc.add_stream("libx264", rate=30)
        vs.time_base = tb  # type: ignore[assignment]
        vs.codec_context.time_base = tb  # type: ignore[assignment]
        vs.pix_fmt = "yuv420p"
        vs.options = {"crf": str(crf), "preset": "medium"}
        first: int | None = None
        for pts, t_ms, img in iter_frames(video, start_ms, end_ms, rotate, index):
            if expect_size is not None and (img.shape[1], img.shape[0]) != expect_size:
                raise ValueError(f"돌린 크기 {img.shape[1]}x{img.shape[0]} != {expect_size}")
            canvas = draw_frame(img, t_ms, ov, opts, session_id=session_id)
            if first is None:
                first = pts
                vs.width, vs.height = canvas.shape[1], canvas.shape[0]
            vf = av.VideoFrame.from_ndarray(canvas, format="bgr24")
            vf.pts, vf.time_base = pts - first, tb
            oc.mux(vs.encode(vf))
            n += 1
        if n == 0:
            raise ValueError("구간에 프레임이 없습니다")
        oc.mux(vs.encode(None))
    return n


# ---------------------------------------------------------------- 명령줄


def _load_labels(session_id: str, url: str | None) -> list[LabelRecord]:
    """개발 DB에서 세션의 운영 현재 라벨을 읽는다 (쓰지 않음)."""
    import sqlalchemy as sa

    from dlp_cli.schema_cmds import database_url
    from dlp_schema.db.repository import get_labels
    from dlp_schema.episode import current_labels

    engine = sa.create_engine(database_url(url))
    try:
        with engine.connect() as conn:
            return current_labels(get_labels(conn, session_id))
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    """명령줄 진입점. 반환: 성공 0, 입력 오류 2."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("session_id")
    ap.add_argument("--video", type=Path, required=True, help="수집 전 로컬 영상 (같은 PTS)")
    ap.add_argument("--rotate", type=int, choices=sorted(ROTATIONS), default=0,
                    help="저장 픽셀 기준 반시계 각도 (수집 때와 같게)")  # fmt: skip
    ap.add_argument("--start", type=float, required=True, help="시작 (초, PTS)")
    ap.add_argument("--end", type=float, required=True, help="끝 (초, PTS)")
    ap.add_argument("--out", type=Path, required=True, help="출력 MP4 (저장소 밖)")
    ap.add_argument("--stream", default="bodycam")
    ap.add_argument("--width", type=int, default=540)
    ap.add_argument("--min-score", type=float, default=0.3, help="객체 박스 표시 문턱 (표시용)")
    ap.add_argument("--hold-ms", type=int, default=0, help="마지막 키프레임 뒤 박스 유지 (ms)")
    ap.add_argument("--max-interp-ms", type=int, default=1500)
    ap.add_argument("--mask-tol-ms", type=int, default=260,
                    help="마스크 키프레임을 보일 최대 시각 차 (ms, 표시용)")  # fmt: skip
    ap.add_argument("--no-boxes", action="store_true", help="객체 박스를 그리지 않는다")
    ap.add_argument("--no-privacy", action="store_true", help="프라이버시 탐지 박스 숨김")
    ap.add_argument(
        "--apply-blur",
        action="store_true",
        help="블러 박스를 모자이크로 덮는다 (밖에 보여 줄 영상용, 테두리 대신)",
    )
    ap.add_argument("--blur-pad", type=float, default=0.15, help="모자이크 박스 넓힘 비율")
    ap.add_argument(
        "--blur-gap-ms", type=int, default=1500, help="모자이크 키프레임 보간·유지 (ms)"
    )
    ap.add_argument(
        "--blur-max-area",
        type=float,
        default=1.0,
        help="화면 대비 이 넓이를 넘는 블러 박스는 덮지 않는다 (오탐 대비)",
    )
    ap.add_argument(
        "--blur-always",
        default="",
        help="크기와 무관하게 덮을 블러 대상 (쉼표, 예: shipping_label,address_sign)",
    )
    ap.add_argument(
        "--blur-skip-on-hand",
        default="",
        help="중심이 손 위면 덮지 않을 블러 대상 (쉼표, 예: face,reflection)",
    )
    ap.add_argument("--clean", action="store_true", help="진단 패널 없이 영상만 (발표용)")
    ap.add_argument(
        "--no-overlays",
        action="store_true",
        help="라벨을 그리지 않는다 (--apply-blur와 함께: 검수 도구 시연용 클립)",
    )
    ap.add_argument("--extra-labels", type=Path, action="append", default=[],
                    help="더 그릴 라벨 JSON Lines (DB 밖 파일럿 결과)")  # fmt: skip
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--expect-size", default=None, help="돌린 프레임 크기 WxH (예: 1920x1080)")
    ap.add_argument("--url", default=None, help="DB URL (기본: DLP_DATABASE_URL)")
    args = ap.parse_args(argv)
    if args.end <= args.start:
        print("[실패] --end가 --start보다 커야 합니다")
        return 2
    if args.out.resolve().is_relative_to(ROOT):
        print("[실패] 출력은 저장소 밖에 둡니다 (개인정보가 든 영상)")
        return 2
    labels = _load_labels(args.session_id, args.url)
    for path in args.extra_labels:
        with path.open(encoding="utf-8") as f:
            labels += [LabelRecord.model_validate_json(line) for line in f if line.strip()]
    start_ms, end_ms = args.start * 1000, args.end * 1000
    ov = build_overlays(
        labels, stream_id=args.stream, start_ms=int(start_ms), end_ms=int(end_ms),
        detectors=privacy_detectors(ROOT / "config/policies/privacy.yaml"),
    )  # fmt: skip
    if args.no_boxes:
        ov.objects = []
    if args.no_privacy:
        ov.blur = []
    if args.apply_blur and args.no_privacy:
        print("[실패] --apply-blur와 --no-privacy는 함께 쓸 수 없습니다 (블러가 빠진다)")
        return 2
    render_policy = load_privacy_policy(ROOT).render
    opts = DrawOptions(min_score=args.min_score, hold_ms=args.hold_ms,
                       max_interp_ms=args.max_interp_ms, width=args.width,
                       mask_tol_ms=args.mask_tol_ms, apply_blur=args.apply_blur,
                       blur_pad=args.blur_pad, blur_gap_ms=args.blur_gap_ms,
                       blur_min_block=render_policy.min_block_px,
                       blur_blocks=render_policy.blocks_per_box, clean=args.clean,
                       blur_max_area=args.blur_max_area, no_overlays=args.no_overlays,
                       blur_always=frozenset(filter(None, args.blur_always.split(","))),
                       blur_skip_on_hand=frozenset(
                           filter(None, args.blur_skip_on_hand.split(",")))
                       )  # fmt: skip
    size = tuple(int(v) for v in args.expect_size.split("x")) if args.expect_size else None
    n = render(args.video, args.out, ov, opts, session_id=args.session_id, start_ms=start_ms,
               end_ms=end_ms, rotate=args.rotate, expect_size=size, crf=args.crf)  # type: ignore[arg-type]  # fmt: skip
    print(f"{args.out}: 프레임 {n}, 블러 트랙 {len(ov.blur)}, 객체 트랙 {len(ov.objects)}, "
          f"손 트랙 {len(ov.hands)}, 마스크 트랙 {len(ov.masks)}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
