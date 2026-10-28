# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""LabelRecord ↔ CVAT 비디오 트랙.

CVAT는 프레임 번호로 주석을 저장하므로 PTS 인덱스의 프레임 시각(정수 ms)으로 맞바꾼다.
우리 쪽에는 프레임 번호를 남기지 않는다. 라벨마다 트랙 속성 두 개를 붙인다.
- dlp_label_id: 원래 라벨 ID (검수자가 새로 그린 트랙은 비어 있다)
- dlp_meta: 모양·라벨 이름 외의 왕복 보존용 필드 (종류, 개체 ID, 손 등)
이 속성들은 검수 화면에서 수정 불가능한 보안 필드가 아니다.
키포인트의 점별 가시성은 모양 속성 dlp_visibility("2,2,1,…")로 싣는다.

검수자가 CVAT 기본인 모양(Shape) 모드로 그린 직사각형은 한 프레임에만 있는 주석이다. 이것도
놓친 블러·박스이므로 버리지 않고 트랙으로 바꾼다: 그 프레임 키프레임 + 다음 프레임 화면 밖(outside)
키프레임 (CVAT 화면과 같게 그 프레임에만 보인다). 키포인트(points) 모양은 키프레임 하나짜리
키포인트 트랙이다. 직사각형·점·마스크가 아닌 모양(다각형·스켈레톤 등), 회전한 직사각형, 프레임
태그는 우리 라벨로 옮길 수 없으므로 수집을 멈춘다 (CvatFormatError, 조용히 버리지 않는다).

마스크(mask_track, ADR 0059)는 CVAT 트랙이 마스크를 보간하지 않아 모양 모드 마스크로 오간다
(`to_cvat_shapes`): 키프레임마다 모양 하나, 모양 속성에 원래 라벨 ID·메타. 수집은 원래 라벨 ID로
다시 한 트랙으로 묶고, ID 없는(새로 그린) 마스크 모양은 하나가 새 트랙이다. CVAT 마스크 점(외접
박스 안 행 우선 연속 길이 + 박스)과 COCO RLE는 같은 화소를 나타내 무손실이다. 마스크는 원본
해상도 작업(UNIT_SCALE)에서만 오가고, 블러 검수에서는 받지 않는다.

손 골격(hand21)은 손별 라벨(`kp_hand21_left`·`kp_hand21_right`)로 그린다. 검수자가 새로 그린 손
골격도 라벨 이름으로 손을 알 수 있고, 손을 바꾸려면 라벨을 바꾼다 (라벨 이름이 dlp_meta의 손보다
우선한다).
손별 라벨 이전에 만든 프로젝트는 `kp_hand21` 하나뿐이라 작업을 만들 때 빠진 라벨을 덧붙이고
(`tasks._cvat_project`), 덧붙이기 전 작업의 `kp_hand21` 트랙은 dlp_meta의 손으로 그대로 왕복한다.
`kp_hand21`로 새로 그려 손을 알 수 없는 골격은 CvatFormatError다 (F-REV-04).
손이 있는 골격의 개체 ID는 손에서 정한다 (`<손>_hand`·`<손>_arm`, F-REV-33). 모든 점이 보이지
않는(visibility 0) 키포인트 키프레임은 화면 밖(outside)으로 오간다 (F-REV-32).
착용자 팔 골격(arm3, 손목·팔꿈치·어깨, ADR 0058)도 손별 라벨(`kp_arm3_left`·`kp_arm3_right`)로
그린다. CVAT 점(points) 모양이라 점 세 개를 고치고, 보이지 않는 점은 dlp_visibility 0이다 (좌표는
손목 자리). 손목은 같은 손 hand21 손목이 기준이다.

좌표: 라벨은 원본(라벨이 가리키는 영상) 화소 좌표다. 검수 화면 영상이 다른 해상도(예: 프라이버시
검수의 480p 프록시)이면 scale = (화면 영상 너비 / 원본 너비, 화면 영상 높이 / 원본 높이)로 보내고
돌아올 때 나눈다. 둘 다 PRECISION 자리로 반올림하므로, 고치지 않은 라벨의 비교 기준은
quantize(label, scale) (보냈다가 그대로 받은 값)이다.

WP6, ADR 0006(무손실 왕복), ADR 0019(키프레임 시각 = 스트림 PTS ms), ADR 0024(모양 모드·지원하지
않는 주석 감사 정정). 사용처: `tasks`(보내기: `label_spec`, `to_cvat_tracks`),
`collect`(받기: `annotation_tracks`, `from_cvat_tracks`, `quantize`). 마스크: `to_cvat_shapes`,
`mask_to_cvat_points`, `cvat_points_to_mask`.

공개 이름: `CVAT_KINDS`, `TRACK_ATTRS`, `SHAPE_ATTRS`, `PRECISION`, `Scale`, `UNIT_SCALE`,
`NewBoxKind`, `KEYPOINT_LABELS`, `LEGACY_HAND_LABEL`, `label_spec`, `CvatSchema`, `CvatFormatError`,
`annotation_tracks`, `quantize`, `cvat_label_name`, `to_cvat_tracks`, `from_cvat_tracks`,
`handed_entity_id`, `handed_entity_mismatches`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from dlp_media.rle import decode_rle, encode_rle
from dlp_review.reconcile import ReviewedItem
from dlp_schema.labels import (
    HANDED_SKELETONS,
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    MaskKeyframe,
    MaskTrackPayload,
)

# CVAT로 오가는 라벨 종류 (블러 검수의 blur_track 포함). mask_track은 모양(shape) 모드 마스크로
# 오간다 (`to_cvat_shapes`, ADR 0059)
CVAT_KINDS = ("box_track", "blur_track", "keypoint_track", "mask_track")
# 트랙 속성 (프레임별 값 변화 없음, mutable=False): 원래 라벨 ID와 메타 JSON.
# mutable=False는 검수자가 속성을 편집하지 못한다는 뜻이 아니다.
TRACK_ATTRS = ("dlp_label_id", "dlp_meta")
# 모양(키프레임) 속성 (프레임마다 다를 수 있음): 키포인트 점별 가시성
SHAPE_ATTRS = ("dlp_visibility",)
PRECISION = 3  # 좌표 소수 자릿수. 왕복에서 값이 흔들리지 않게 내보낼 때 반올림한다.

Scale = tuple[float, float]  # (가로, 세로) 화면 영상 화소 / 라벨 화소
# 화면 영상과 라벨 좌표계가 같을 때 (작업 라벨 검수: 원본 해상도 블러본)
UNIT_SCALE: Scale = (1.0, 1.0)
# 검수자가 새로 그린 직사각형 트랙의 라벨 종류 (dlp_meta가 비어 있을 때). 작업 단계로 정한다.
NewBoxKind = Literal["blur_track", "box_track"]
# 작업 라벨 프로젝트의 키포인트 골격 라벨 (`kp_<골격>`, 손 골격은 `kp_hand21_<손>`).
# 손을 라벨 이름에 넣어 검수자가 새로 그린 손 골격도 어느 손인지 알 수 있게 한다 (F-REV-04).
# 착용자 팔 골격(arm3, 손목·팔꿈치·어깨, ADR 0058)도 손별 라벨이다 (`kp_arm3_<손>`).
KEYPOINT_LABELS = (
    "kp_hand21_left",
    "kp_hand21_right",
    "kp_coco17",
    "kp_wholebody133",
    "kp_arm3_left",
    "kp_arm3_right",
)
# 손별 라벨 이전의 손 골격 라벨. 그때 만든 프로젝트·작업의 트랙만 쓴다 (손은 dlp_meta에 있다).
LEGACY_HAND_LABEL = "kp_hand21"


def label_spec(
    names: Iterable[str], colors: Mapping[str, str] | None = None
) -> list[dict[str, Any]]:
    """CVAT 프로젝트 라벨 정의.

    names: CVAT 라벨 이름 (블러 대상 종류, 객체 클래스, `KEYPOINT_LABELS`).
    colors: 라벨 이름 → 화면 색(#rrggbb, `ui_theme.cvat_label_color`). 없거나 이름이 빠지면 CVAT가
    색을 고른다. 색은 화면 모양일 뿐 수집·왕복에 쓰지 않는다.
    라벨마다 텍스트 속성 `dlp_label_id`·`dlp_meta`(트랙, 고정)와 `dlp_visibility`(모양, 변경 가능)를
    붙인다. 라벨 type "any"라 직사각형·점 모두 그릴 수 있다 (모양 검사는 수집 때 한다).
    """
    attrs = [
        {"name": n, "mutable": False, "input_type": "text", "default_value": "", "values": [""]}
        for n in TRACK_ATTRS
    ] + [
        {"name": n, "mutable": True, "input_type": "text", "default_value": "", "values": [""]}
        for n in SHAPE_ATTRS
    ]
    out: list[dict[str, Any]] = []
    for n in names:
        spec: dict[str, Any] = {"name": n, "type": "any", "attributes": attrs}
        if colors is not None and n in colors:
            spec["color"] = colors[n]
        out.append(spec)
    return out


@dataclass(frozen=True)
class CvatSchema:
    """프로젝트의 라벨 이름 ↔ ID, (라벨, 속성 이름) ↔ 속성 ID."""

    # 라벨 이름 → CVAT 라벨 ID
    label_ids: dict[str, int]
    # (라벨 이름, 속성 이름) → CVAT 속성(spec) ID
    attr_ids: dict[tuple[str, str], int]

    @classmethod
    def from_labels(cls, labels: list[dict[str, Any]]) -> CvatSchema:
        """`CvatClient.project_labels` 결과(ID가 붙은 라벨 정의)로 만든다."""
        return cls(
            {lb["name"]: lb["id"] for lb in labels},
            {(lb["name"], a["name"]): a["id"] for lb in labels for a in lb["attributes"]},
        )

    def label_name(self, label_id: int) -> str:
        """CVAT 라벨 ID → 이름. 프로젝트에 없으면 `CvatFormatError`."""
        name = next((n for n, i in self.label_ids.items() if i == label_id), None)
        if name is None:
            raise CvatFormatError(f"프로젝트에 없는 CVAT 라벨 ID: {label_id}")
        return name

    def attr_name(self, spec_id: int) -> str:
        """CVAT 속성 ID → 속성 이름. 프로젝트에 없으면 `CvatFormatError`."""
        name = next((n for (_, n), i in self.attr_ids.items() if i == spec_id), None)
        if name is None:
            raise CvatFormatError(f"프로젝트에 없는 CVAT 속성 ID: {spec_id}")
        return name


class CvatFormatError(ValueError):
    """CVAT 주석을 우리 라벨로 옮길 수 없다 (지원하지 않는 모양·태그). 수집하지 않는다."""


def _expected_shape(name: str) -> str:
    """라벨 이름이 요구하는 CVAT 모양: `kp_`로 시작하면 points, 아니면 rectangle (또는 mask)."""
    return "points" if name.startswith("kp_") else "rectangle"


def _check_shape(shape: dict[str, Any], name: str) -> None:
    """모양이 라벨에 맞는지 검사한다. 다른 모양이거나 회전한 직사각형이면 `CvatFormatError`.

    객체 라벨(`kp_`가 아닌 이름)은 직사각형과 마스크(mask, ADR 0059)를 받는다.
    """
    want = _expected_shape(name)
    if want == "rectangle" and shape.get("type") == "mask":
        return
    if shape.get("type") != want:
        raise CvatFormatError(
            f"라벨 {name}에 지원하지 않는 CVAT 모양 {shape.get('type')!r} "
            f"(프레임 {shape.get('frame')}). {want}로 그려야 합니다"
        )
    # 우리 박스는 축 정렬 (x, y, w, h)라서 회전을 담을 수 없다
    if want == "rectangle" and float(shape.get("rotation") or 0.0) != 0.0:
        raise CvatFormatError(
            f"라벨 {name}의 회전한 직사각형 (프레임 {shape.get('frame')})은 지원하지 않습니다"
        )


def _visibility(raw: str, n: int, name: str, shape: dict[str, Any]) -> list[int]:
    """키포인트 모양의 `dlp_visibility` 속성("2,2,1,…")을 점별 가시성 목록으로 바꾼다.

    회귀: 검수자가 점을 더 찍어 가시성 값이 점 수보다 짧으면 `IndexError`가 나 수집이
    `CvatFormatError`(옮길 수 없는 주석)가 아닌 내부 오류로 멈췄다.

    Args:
        raw: 속성 값 (비어 있으면 모든 점이 보임 = 2).
        n: 그 모양의 점 수.
        name: CVAT 라벨 이름 (오류 메시지용).
        shape: CVAT 모양 (오류 메시지의 프레임 번호용).

    Returns:
        길이 n의 가시성 목록 (값 검증은 `Keypoint`가 한다).

    Raises:
        CvatFormatError: 정수가 아닌 값이 있거나 값 수가 점 수와 다를 때 (모자란 점의 가시성을
            지어내거나 남는 값을 조용히 버리지 않는다).
    """
    try:
        vis = [int(v) for v in raw.split(",") if v.strip()]
    except ValueError as e:
        raise CvatFormatError(
            f"라벨 {name}의 dlp_visibility가 정수 목록이 아닙니다 (프레임 {shape.get('frame')}): "
            f"{raw!r}"
        ) from e
    if not vis:
        return [2] * n
    if len(vis) != n:
        raise CvatFormatError(
            f"라벨 {name}의 dlp_visibility 값 {len(vis)}개가 점 {n}개와 맞지 않습니다 "
            f"(프레임 {shape.get('frame')})"
        )
    return vis


def annotation_tracks(
    annotations: dict[str, Any], frame_count: int, schema: CvatSchema
) -> list[dict[str, Any]]:
    """CVAT 작업 주석 전체(tracks·shapes·tags) → 트랙 목록.

    모양 모드 직사각형은 (그 프레임, 다음 프레임 outside) 트랙으로, 점 모양은 키프레임 하나짜리
    트랙으로 바꾼다. 태그나 지원하지 않는 모양이 있으면 CvatFormatError.

    인자: annotations(`CvatClient.get_annotations` 결과), frame_count(화면 영상 프레임 수),
    schema(프로젝트 라벨·속성 ID).
    반환: 기존 트랙(복사본) 뒤에 모양에서 만든 트랙을 (프레임, 라벨 ID) 순으로 붙인 목록.
    """
    tags = annotations.get("tags") or []
    if tags:
        frames = sorted({int(t.get("frame", -1)) for t in tags})
        raise CvatFormatError(f"CVAT 프레임 태그는 지원하지 않습니다 (프레임 {frames[:5]})")
    tracks = [dict(t) for t in annotations.get("tracks") or []]
    for track in tracks:
        name = schema.label_name(track["label_id"])
        for shape in track.get("shapes") or []:
            _check_shape(shape, name)
    masks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for shape in sorted(annotations.get("shapes") or [], key=lambda x: (x["frame"], x["label_id"])):
        name = schema.label_name(shape["label_id"])
        _check_shape(shape, name)
        frame = int(shape["frame"])
        if not 0 <= frame < frame_count:
            raise CvatFormatError(f"영상 밖 프레임의 CVAT 모양: {frame}")
        if shape["type"] == "mask":
            # 마스크 모양은 원래 라벨 ID(모양 속성)마다 한 트랙으로 묶는다. ID가 없으면(새로 그림)
            # 모양 하나가 트랙 하나다 (ADR 0059)
            origin = _shape_attr(shape, schema, "dlp_label_id") or f"new:{len(masks)}"
            masks.setdefault((int(shape["label_id"]), origin), []).append(shape)
            continue
        attrs = list(shape.get("attributes") or [])
        first = {
            "type": shape["type"], "frame": frame, "points": list(shape["points"]),
            "outside": False, "occluded": bool(shape.get("occluded", False)),
            "z_order": 0, "rotation": 0.0, "attributes": attrs,
        }  # fmt: skip
        shapes = [first]
        # 직사각형은 다음 프레임에 outside 키프레임을 붙여 그 프레임에만 보이게 한다.
        # 마지막 프레임이면 다음 프레임이 없어 키프레임 하나로 둔다.
        if shape["type"] == "rectangle" and frame + 1 < frame_count:
            shapes.append({**first, "frame": frame + 1, "outside": True, "attributes": []})
        tracks.append(
            {
                "frame": frame,
                "label_id": shape["label_id"],
                "group": 0,
                "source": shape.get("source", "manual"),
                # 모양에는 트랙 속성(dlp_label_id·dlp_meta)이 없으므로 새로 그린 트랙이 된다
                "attributes": attrs,
                "shapes": shapes,
            }
        )
    tracks += [_mask_track(group) for group in masks.values()]
    return tracks


def _shape_attr(shape: dict[str, Any], schema: CvatSchema, attr: str) -> str:
    """모양 속성 값 (없으면 "")."""
    for a in shape.get("attributes") or []:
        if schema.attr_name(a["spec_id"]) == attr:
            return str(a["value"])
    return ""


def _mask_track(shapes: list[dict[str, Any]]) -> dict[str, Any]:
    """같은 라벨의 마스크 모양 묶음 → 트랙 dict (트랙 속성 = 첫 모양의 속성)."""
    first = shapes[0]
    return {
        "frame": int(first["frame"]),
        "label_id": first["label_id"],
        "group": 0,
        "source": first.get("source", "manual"),
        "attributes": list(first.get("attributes") or []),
        "shapes": [{**sh, "outside": False, "attributes": []} for sh in shapes],
    }


def mask_to_cvat_points(mask: NDArray[np.bool_]) -> list[float]:
    """마스크 → CVAT 마스크 모양 점: 외접 박스 안 행 우선 연속 길이(배경부터) + [왼, 위, 오, 아래].

    빈 마스크면 ValueError (CVAT 마스크 모양은 한 화소 이상이어야 한다).
    """
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("빈 마스크는 CVAT 모양으로 보낼 수 없습니다")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    flat = mask[y0 : y1 + 1, x0 : x1 + 1].reshape(-1)
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
    if flat[0]:
        counts.insert(0, 0)  # CVAT 연속 길이도 배경(0)부터 센다
    return [float(c) for c in counts] + [float(x0), float(y0), float(x1), float(y1)]


def cvat_points_to_mask(points: list[float], size: tuple[int, int]) -> NDArray[np.bool_]:
    """CVAT 마스크 모양 점 → (높이, 너비) 마스크. 길이 합이 박스와 다르거나 박스가 화면 밖이면
    `CvatFormatError`."""
    h, w = size
    if len(points) < 5:
        raise CvatFormatError("CVAT 마스크 점이 모자랍니다")
    x0, y0, x1, y1 = (round(float(v)) for v in points[-4:])
    if not (0 <= x0 <= x1 < w and 0 <= y0 <= y1 < h):
        raise CvatFormatError(f"CVAT 마스크 박스가 화면 밖입니다: {(x0, y0, x1, y1)}")
    runs = [round(float(v)) for v in points[:-4]]
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if sum(runs) != bw * bh or any(r < 0 for r in runs):
        raise CvatFormatError("CVAT 마스크 연속 길이 합이 박스 크기와 다릅니다")
    flat = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
    out = np.zeros((h, w), bool)
    out[y0 : y1 + 1, x0 : x1 + 1] = flat.reshape(bh, bw)
    return out


def _r(v: float) -> float:
    """좌표를 PRECISION 자리로 반올림한다."""
    return round(float(v), PRECISION)


def _box_points(k: BoxKeyframe, scale: Scale) -> list[float]:
    """박스 키프레임 (x, y, w, h, 라벨 화소) → CVAT 직사각형 점 [x1, y1, x2, y2] (화면 화소)."""
    sx, sy = scale
    return [_r(k.x * sx), _r(k.y * sy), _r((k.x + k.w) * sx), _r((k.y + k.h) * sy)]


def _box_from_points(t_ms: int, pts: list[float], outside: bool, scale: Scale) -> BoxKeyframe:
    """CVAT 직사각형 점 [x1, y1, x2, y2] (화면 화소) → 박스 키프레임 (라벨 화소).

    t_ms: 그 프레임의 PTS 시각. 뒤집힌 점(x2 < x1)은 너비·높이를 0으로 둔다.
    """
    sx, sy = scale
    return BoxKeyframe(
        t_ms=t_ms,
        x=_r(pts[0] / sx),
        y=_r(pts[1] / sy),
        w=_r(max(0.0, pts[2] - pts[0]) / sx),
        h=_r(max(0.0, pts[3] - pts[1]) / sy),
        outside=outside,
    )


def _point_to(pt: Keypoint, scale: Scale) -> tuple[float, float]:
    """키포인트 (라벨 화소) → 화면 화소 (x, y), 반올림."""
    return _r(pt.x * scale[0]), _r(pt.y * scale[1])


def quantize(label: LabelRecord, scale: Scale = UNIT_SCALE) -> LabelRecord:
    """CVAT에 보냈다가 고치지 않고 받은 값 (좌표 변환·반올림 왕복). 수집 때 비교 기준이다.

    박스·블러·키포인트 트랙만 바꾸고 그 밖의 라벨은 그대로 돌려준다. 키포인트의 신뢰도와 손 중심
    3D 점(world, ADR 0044)은 도구로 보내지 않으므로 뺀다.
    """
    p = label.payload
    if isinstance(p, BoxTrackPayload | BlurTrackPayload):
        kfs = tuple(
            _box_from_points(k.t_ms, _box_points(k, scale), k.outside, scale) for k in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    if isinstance(p, KeypointTrackPayload):
        sx, sy = scale
        # 보낼 때(_point_to)와 받을 때(from_cvat_tracks의 나눗셈·반올림)를 그대로 재현한다.
        # 손 중심 3D 점(world)은 CVAT로 보내지 않으므로 돌아온 값에도 없다 (ADR 0044). 고치지 않은
        # 트랙은 원래 레코드를 승인하므로 world가 남는다. 고친 트랙의 사람 레코드는 2D가 이 값과
        # 같은 키프레임에만 원래 world를 되살린다 (`reconcile.keep_unchanged_world`)
        kfs = tuple(
            f.model_copy(
                update={
                    "world": None,
                    # 보간 표시(ADR 0054)도 도구로 보내지 않는다
                    "interpolated": None,
                    "points": tuple(
                        Keypoint(
                            x=_r(_point_to(pt, scale)[0] / sx),
                            y=_r(_point_to(pt, scale)[1] / sy),
                            visibility=pt.visibility,
                        )
                        for pt in f.points
                    ),
                }
            )
            for f in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    if isinstance(p, MaskTrackPayload):
        # 마스크는 무손실로 오간다. 점수와 안 보임·빈 키프레임은 도구로 보내지 않는다 (ADR 0059)
        kept = tuple(k.model_copy(update={"score": None}) for k in p.keyframes if _sendable_mask(k))
        if kept:
            return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kept})})
    return label


def _sendable_mask(k: MaskKeyframe) -> bool:
    """CVAT로 보내는 마스크 키프레임인가 (보이고 한 화소 이상)."""
    return not k.outside and k.rle is not None and bool(decode_rle(k.rle).any())


def cvat_label_name(label: LabelRecord) -> str:
    """라벨 → CVAT 라벨 이름 (블러: 대상 종류, 박스: 클래스, 키포인트: `kp_<골격>`,
    손이 있는 키포인트: `kp_<골격>_<손>`, 예: `kp_hand21_left`).

    예외: CVAT로 보낼 수 없는 종류면 ValueError.
    """
    p = label.payload
    if isinstance(p, BlurTrackPayload):
        return p.target
    if isinstance(p, BoxTrackPayload | MaskTrackPayload):
        return p.class_id
    if isinstance(p, KeypointTrackPayload):
        return f"kp_{p.skeleton}" + (f"_{p.hand.value}" if p.hand is not None else "")
    raise ValueError(f"CVAT로 보낼 수 없는 라벨 종류: {label.kind}")


def _project_label(name: str, schema: CvatSchema) -> str:
    """프로젝트에 있는 라벨 이름. 손별 손 골격 라벨이 없는(이전에 만든) 프로젝트면 `kp_hand21`.

    손은 dlp_meta에 실리므로 이전 라벨로 보내도 왕복한다.
    """
    if name not in schema.label_ids and name.startswith(f"{LEGACY_HAND_LABEL}_"):
        return LEGACY_HAND_LABEL
    return name


def _keypoint_label(name: str) -> tuple[str, str | None]:
    """키포인트 라벨 이름 → (골격, 이름에 든 손 또는 None).

    예: `kp_hand21_left` → (hand21, left), `kp_hand21` → (hand21, None),
    `kp_coco17` → (coco17, None).
    """
    body = name.removeprefix("kp_")
    skeleton, _, hand = body.rpartition("_")
    if skeleton and hand in ("left", "right"):
        return skeleton, hand
    return body, None


def to_cvat_tracks(
    labels: list[LabelRecord],
    frame_times: list[int],
    schema: CvatSchema,
    *,
    scale: Scale = UNIT_SCALE,
) -> list[dict[str, Any]]:
    """frame_times: 화면 영상의 프레임 시각. scale: 화면 영상 화소 / 라벨 화소.

    반환: CVAT 트랙 목록 (`CvatClient.put_tracks`에 넘긴다). 키프레임 시각은 frame_times에서
    정확히 같은 값을 찾아 프레임 번호로 바꾼다 (보간·근사하지 않는다). 손 골격은 손별 라벨
    (`kp_hand21_<손>`)로 보내고, 그 라벨이 없는 이전 프로젝트면 `kp_hand21`로 보낸다.
    예외: 키프레임 시각이 화면 영상 프레임 시각에 없으면 ValueError (`_shape`), 프로젝트에
    라벨·속성이 없으면 KeyError.
    """
    # PTS ms → 프레임 번호 (화면 영상 기준)
    frame_of = {t: i for i, t in enumerate(frame_times)}
    tracks: list[dict[str, Any]] = []
    for label in labels:
        p = label.payload
        if isinstance(p, MaskTrackPayload):
            continue  # 마스크는 모양 모드로 보낸다 (`to_cvat_shapes`)
        name = _project_label(cvat_label_name(label), schema)
        meta: dict[str, Any] = {"kind": label.kind}
        shapes: list[dict[str, Any]] = []
        if isinstance(p, BoxTrackPayload | BlurTrackPayload):
            if isinstance(p, BoxTrackPayload):
                meta["entity_id"] = p.entity_id
            for k in p.keyframes:
                shapes.append(
                    _shape(frame_of, k.t_ms, "rectangle", _box_points(k, scale), k.outside)
                )
        elif isinstance(p, KeypointTrackPayload):
            meta |= {"entity_id": p.entity_id, "skeleton": p.skeleton, "hand": p.hand}
            vis_id = schema.attr_ids[(name, "dlp_visibility")]
            for f in p.keyframes:
                # 점 목록을 [x1, y1, x2, y2, …]로 펼친다. 모든 점이 보이지 않는(visibility 0)
                # 키프레임은 화면 밖(outside)으로 보낸다 (돌아오면 다시 모두 0, F-REV-32)
                pts = [c for pt in f.points for c in _point_to(pt, scale)]
                shape = _shape(frame_of, f.t_ms, "points", pts, _all_hidden(f))
                shape["attributes"] = [
                    {"spec_id": vis_id, "value": ",".join(str(pt.visibility) for pt in f.points)}
                ]
                shapes.append(shape)
        tracks.append(
            {
                "frame": shapes[0]["frame"],
                "label_id": schema.label_ids[name],
                "group": 0,
                "source": "auto",
                "attributes": [
                    {"spec_id": schema.attr_ids[(name, "dlp_label_id")], "value": label.label_id},
                    {"spec_id": schema.attr_ids[(name, "dlp_meta")], "value": json.dumps(meta)},
                ],
                "shapes": shapes,
            }
        )
    return tracks


def to_cvat_shapes(
    labels: list[LabelRecord],
    frame_times: list[int],
    schema: CvatSchema,
    *,
    scale: Scale = UNIT_SCALE,
) -> list[dict[str, Any]]:
    """마스크 라벨 → CVAT 모양 모드 마스크 목록 (키프레임마다 모양 하나, ADR 0059).

    CVAT 트랙은 마스크를 보간하지 않으므로 모양 모드로 보낸다. 모양마다 원래 라벨 ID와 메타
    (종류·개체·부분·derived_from·프레임 크기)를 모양 속성으로 붙이고, 수집(`annotation_tracks`)이
    라벨 ID로 다시 한 트랙으로 묶는다. 마스크는 외접 박스 안 연속 길이라 무손실이다. 점수
    (`score`)와 안 보임·빈 키프레임은 보내지 않는다 (`quantize`와 같다). 마스크가 아닌 라벨은
    건너뛴다.
    예외: 마스크가 있는데 scale이 1이 아니면 ValueError (마스크는 원본 해상도 작업에서만 오간다),
    키프레임 시각이 영상 프레임 시각에 없으면 ValueError.
    """
    frame_of = {t: i for i, t in enumerate(frame_times)}
    out: list[dict[str, Any]] = []
    for label in labels:
        p = label.payload
        if not isinstance(p, MaskTrackPayload):
            continue
        if scale != UNIT_SCALE:
            raise ValueError("마스크는 원본 해상도(UNIT_SCALE) 작업에서만 보낼 수 있습니다")
        name = _project_label(cvat_label_name(label), schema)
        sendable = [k for k in p.keyframes if _sendable_mask(k)]
        meta: dict[str, Any] = {
            "kind": label.kind, "entity_id": p.entity_id, "part": p.part,
            "derived_from": list(p.derived_from) if p.derived_from else None,
            "size": list(sendable[0].rle.size) if sendable and sendable[0].rle else None,
        }  # fmt: skip
        attrs = [
            {"spec_id": schema.attr_ids[(name, "dlp_label_id")], "value": label.label_id},
            {"spec_id": schema.attr_ids[(name, "dlp_meta")], "value": json.dumps(meta)},
        ]
        for k in sendable:
            assert k.rle is not None
            shape = _shape(frame_of, k.t_ms, "mask", mask_to_cvat_points(decode_rle(k.rle)), False)
            out.append(
                shape
                | {
                    "label_id": schema.label_ids[name],
                    "group": 0,
                    "source": "auto",
                    "attributes": attrs,
                }
            )
    return out


def _shape(
    frame_of: dict[int, int], t_ms: int, kind: str, points: list[float], outside: bool
) -> dict[str, Any]:
    """키프레임 하나 → CVAT 모양 dict. t_ms가 영상 프레임 시각이 아니면 ValueError."""
    if t_ms not in frame_of:
        raise ValueError(f"키프레임 시각 {t_ms} ms가 영상 프레임 시각과 맞지 않습니다")
    return {
        "type": kind,
        "frame": frame_of[t_ms],
        "points": points,
        "outside": outside,
        "occluded": False,
        "z_order": 0,
        "rotation": 0.0,
        "attributes": [],
    }


def from_cvat_tracks(
    tracks: list[dict[str, Any]],
    frame_times: list[int],
    schema: CvatSchema,
    stream_id: str,
    *,
    new_box_kind: NewBoxKind,
    scale: Scale = UNIT_SCALE,
    frame_size: tuple[int, int] | None = None,
) -> list[ReviewedItem]:
    """new_box_kind: dlp_meta가 없는(검수자가 새로 그린) 직사각형 트랙의 종류.
    프라이버시 작업이면 blur_track, 작업 라벨 작업이면 box_track이다.
    마스크 모양 트랙(ADR 0059)은 mask_track이다 (프라이버시 작업에서는 받지 않는다).
    frame_size: 화면 영상 (높이, 너비). 새로 그린 마스크의 RLE 크기 (보낸 마스크는 메타의 크기).

    인자:
    - tracks: `annotation_tracks` 결과.
    - frame_times: 화면 영상 프레임 시각 (프레임 번호 → PTS ms).
    - stream_id: 작업의 스트림 (모든 항목에 붙인다).
    - scale: 화면 영상 화소 / 라벨 화소 (돌아올 때 나눈다).

    반환: `ReviewedItem` 목록 (트랙 순서). 구간은 첫·마지막 모양의 프레임 시각이다.
    새로 그린 박스·키포인트 트랙의 개체 ID는 `<라벨 이름>_h<트랙 순번>`이다. 손이 있는 골격
    (hand21·arm3)은 메타와 관계없이 늘 손에서 정한다 (`handed_entity_id`, F-REV-33).
    키포인트 가시성 속성이 없으면 모든 점을 2(보임)로 둔다. 화면 밖(outside) 키포인트 키프레임은
    모든 점이 0(라벨 없음)이다 (F-REV-32). 키포인트의 골격·손은 라벨 이름
    (`kp_<골격>[_<손>]`)에서 읽고, 이름에 손이 없으면 dlp_meta의 손을 쓴다.
    예외: 모양 없는 트랙, 지원하지 않는 모양, 영상 밖 프레임, 모르는 라벨·속성 ID, 손을 알 수
    없는 손 골격(`kp_hand21`로 새로 그림), 골격과 맞지 않는 점 수면 `CvatFormatError`.
    """
    items: list[ReviewedItem] = []
    for n, track in enumerate(tracks):
        name = schema.label_name(track["label_id"])
        attrs = {schema.attr_name(a["spec_id"]): a["value"] for a in track.get("attributes", [])}
        origin = attrs.get("dlp_label_id") or None
        meta: dict[str, Any] = json.loads(attrs["dlp_meta"]) if attrs.get("dlp_meta") else {}
        # 종류: 메타가 있으면 그것, 없으면 라벨 이름(kp_)과 작업 단계(new_box_kind)로 정한다
        shapes = sorted(track["shapes"], key=lambda s: s["frame"])
        if not shapes:
            raise CvatFormatError(f"모양이 없는 CVAT 트랙 (라벨 {name})")
        kind = meta.get("kind") or _new_kind(name, shapes[0], new_box_kind)
        for s in shapes:
            _check_shape(s, name)
            if (s["type"] == "mask") != (kind == "mask_track"):
                raise CvatFormatError(
                    f"라벨 {name}의 {kind} 트랙에 맞지 않는 CVAT 모양 {s['type']!r} "
                    f"(프레임 {s['frame']})"
                )
            if not 0 <= int(s["frame"]) < len(frame_times):
                raise CvatFormatError(f"영상 밖 프레임의 CVAT 모양: {s['frame']}")
        # 프레임 번호 → PTS ms (우리 쪽에는 프레임 번호를 남기지 않는다)
        times = [frame_times[s["frame"]] for s in shapes]
        payload: Any
        if kind in ("box_track", "blur_track"):
            kfs = tuple(
                _box_from_points(t, [float(v) for v in s["points"]], bool(s["outside"]), scale)
                for t, s in zip(times, shapes, strict=True)
            )
            if kind == "blur_track":
                payload = BlurTrackPayload(target=name, keyframes=kfs)
            else:
                entity = meta.get("entity_id") or f"{name}_h{n}"
                payload = BoxTrackPayload(entity_id=entity, class_id=name, keyframes=kfs)
        elif kind == "mask_track":
            payload = _mask_payload(name, meta, n, times, shapes, frame_size, new_box_kind)
        else:
            payload = _keypoint_payload(name, meta, n, times, shapes, scale, schema)
        items.append(ReviewedItem(origin, stream_id, times[0], times[-1], payload))
    return items


def _new_kind(name: str, shape: dict[str, Any], new_box_kind: NewBoxKind) -> str:
    """dlp_meta가 없는(새로 그린) 트랙의 종류: kp_ 라벨은 키포인트, 마스크 모양은 마스크, 그 밖은
    작업 단계(new_box_kind)."""
    if name.startswith("kp_"):
        return "keypoint_track"
    return "mask_track" if shape.get("type") == "mask" else new_box_kind


def _mask_payload(
    name: str,
    meta: dict[str, Any],
    n: int,
    times: list[int],
    shapes: list[dict[str, Any]],
    frame_size: tuple[int, int] | None,
    new_box_kind: NewBoxKind,
) -> MaskTrackPayload:
    """마스크 모양 묶음 → mask_track (키프레임마다 COCO RLE, 점수 없음). 개체 ID는 메타 또는
    `<라벨 이름>_h<트랙 순번>`. 예외: 프라이버시 작업의 마스크, 크기를 모름 → `CvatFormatError`."""
    if new_box_kind == "blur_track":
        raise CvatFormatError(f"블러 검수에서는 마스크를 받지 않습니다 (라벨 {name})")
    size = tuple(meta["size"]) if meta.get("size") else frame_size
    if size is None or len(size) != 2:
        raise CvatFormatError(f"라벨 {name}의 마스크 크기를 알 수 없습니다")
    hw = (int(size[0]), int(size[1]))
    kfs = tuple(
        MaskKeyframe(
            t_ms=t, rle=encode_rle(cvat_points_to_mask([float(v) for v in s["points"]], hw))
        )
        for t, s in zip(times, shapes, strict=True)
    )
    derived = meta.get("derived_from")
    return MaskTrackPayload(
        entity_id=meta.get("entity_id") or f"{name}_h{n}", class_id=name, part=meta.get("part"),
        keyframes=kfs, derived_from=tuple(derived) if derived else None,
    )  # fmt: skip


def _keypoint_payload(
    name: str,
    meta: dict[str, Any],
    n: int,
    times: list[int],
    shapes: list[dict[str, Any]],
    scale: Scale,
    schema: CvatSchema,
) -> KeypointTrackPayload:
    """키포인트 점 모양 묶음 → keypoint_track. 예외는 `from_cvat_tracks` 참고."""
    sx, sy = scale
    frames: list[KeypointFrame] = []
    skeleton, hand = _keypoint_label(name)
    # 라벨 이름의 손이 메타(원래 손)보다 우선한다 (검수자가 라벨을 바꿔 손을 고친다)
    hand = hand or meta.get("hand")
    if skeleton in HANDED_SKELETONS and hand is None:
        frames_span = f"{shapes[0]['frame']}~{shapes[-1]['frame']}"
        raise CvatFormatError(
            f"라벨 {name}의 {skeleton} 골격 트랙 (프레임 {frames_span})에 손(left/right)이 "
            f"없습니다. kp_{skeleton}_left 또는 kp_{skeleton}_right로 그려야 합니다"
        )
    for t, s in zip(times, shapes, strict=True):
        vis_attr = {schema.attr_name(a["spec_id"]): a["value"] for a in s.get("attributes", [])}
        pts = s["points"]
        # 점별 가시성 "2,2,1,…" (없으면 점 수만큼 2). 점 수와 맞지 않으면 CvatFormatError
        vis = _visibility(vis_attr.get("dlp_visibility", ""), len(pts) // 2, name, s)
        if s.get("outside"):
            # 화면 밖(outside) 키프레임: CVAT는 직전 모양(좌표·속성)을 복사해 두므로 좌표는 그대로
            # 두고 모든 점을 라벨 없음(0)으로 본다. 계약에 키포인트 outside가 없고, 0인 점은 3D로
            # 올리지 않으며 내보내기에서도 빈 값이다 (F-REV-32)
            vis = [0] * len(vis)
        frames.append(
            KeypointFrame(
                t_ms=t,
                points=tuple(
                    Keypoint(
                        x=_r(pts[2 * i] / sx),
                        y=_r(pts[2 * i + 1] / sy),
                        visibility=vis[i],  # type: ignore[arg-type]
                    )
                    for i in range(len(pts) // 2)
                ),
            )
        )
    # 손이 있는 골격(hand21·arm3)의 개체 ID는 손에서 정한다 (`<손>_hand`·`<손>_arm`, 3D 승격·
    # 내보내기의 규약). 라벨을 바꿔 손을 고쳐도, 새로 그려도 같다 (F-REV-33)
    entity = handed_entity_id(skeleton, hand) or meta.get("entity_id") or f"{name}_h{n}"
    try:
        return KeypointTrackPayload(
            entity_id=entity,
            skeleton=skeleton,  # type: ignore[arg-type]
            hand=hand,  # type: ignore[arg-type]
            keyframes=tuple(frames),
        )
    except ValidationError as e:
        # 예: 새로 그린 골격의 점 수가 골격과 다름. 내부 오류가 아니라 고칠 수 있는 주석이다
        raise CvatFormatError(
            f"라벨 {name}의 키포인트 트랙 (프레임 {shapes[0]['frame']}~"
            f"{shapes[-1]['frame']})을 옮길 수 없습니다: "
            + "; ".join(str(x["msg"]) for x in e.errors())
        ) from e


def _all_hidden(frame: KeypointFrame) -> bool:
    """모든 점이 보이지 않는(visibility 0) 키프레임인가 (CVAT outside로 오간다)."""
    return all(pt.visibility == 0 for pt in frame.points)


# 손이 있는 골격 → 개체 ID 접미사 (`<손>_hand`·`<손>_arm`, ADR 0044·0058)
_HANDED_ENTITY_SUFFIX = {"hand21": "hand", "arm3": "arm"}


def handed_entity_id(skeleton: str, hand: Hand | str | None) -> str | None:
    """손이 있는 골격(hand21·arm3)의 규약 개체 ID (`left_hand`, `right_arm` 등).

    다른 골격이거나 손을 모르면 None. 3D 승격(`dlp_prelabel.lift3d.lift_key`)과 LeRobot 내보내기가
    이 ID로 손·팔을 찾는다.
    """
    suffix = _HANDED_ENTITY_SUFFIX.get(skeleton)
    if suffix is None or hand is None:
        return None
    return f"{Hand(hand).value}_{suffix}"


def handed_entity_mismatches(labels: Iterable[LabelRecord]) -> list[str]:
    """손이 있는 골격 키포인트 라벨 중 개체 ID가 손과 맞지 않는 것의 설명 목록 (없으면 빈 목록).

    수집이 저장 전에 검사한다 (`collect.collect_task`, F-REV-33).
    """
    out: list[str] = []
    for label in labels:
        p = label.payload
        if not isinstance(p, KeypointTrackPayload):
            continue
        want = handed_entity_id(p.skeleton, p.hand)
        if want is not None and p.entity_id != want:
            out.append(
                f"{label.label_id}: {p.skeleton} 골격의 개체 ID {p.entity_id}가 손 규약 {want}와 "
                "다릅니다"
            )
    return out
