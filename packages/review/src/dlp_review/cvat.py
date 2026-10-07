# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""LabelRecord ↔ CVAT 비디오 트랙.

CVAT는 프레임 번호로 주석을 저장하므로 PTS 인덱스의 프레임 시각(정수 ms)으로 맞바꾼다.
우리 쪽에는 프레임 번호를 남기지 않는다. 라벨마다 트랙 속성 두 개를 붙인다.
- dlp_label_id: 원래 라벨 ID (검수자가 새로 그린 트랙은 비어 있다)
- dlp_meta: 화면에서 고치지 않는 필드 (종류, 개체 ID, 손 등)
키포인트의 점별 가시성은 모양 속성 dlp_visibility("2,2,1,…")로 싣는다.

검수자가 CVAT 기본인 모양(Shape) 모드로 그린 직사각형은 한 프레임에만 있는 주석이다. 이것도
놓친 블러·박스이므로 버리지 않고 트랙으로 바꾼다: 그 프레임 키프레임 + 다음 프레임 화면 밖(outside)
키프레임 (CVAT 화면과 같게 그 프레임에만 보인다). 키포인트(points) 모양은 키프레임 하나짜리
키포인트 트랙이다. 직사각형·점이 아닌 모양(다각형·마스크·스켈레톤 등), 회전한 직사각형, 프레임
태그는 우리 라벨로 옮길 수 없으므로 수집을 멈춘다 (CvatFormatError, 조용히 버리지 않는다).

좌표: 라벨은 원본(라벨이 가리키는 영상) 화소 좌표다. 검수 화면 영상이 다른 해상도(예: 프라이버시
검수의 480p 프록시)이면 scale = (화면 영상 너비 / 원본 너비, 화면 영상 높이 / 원본 높이)로 보내고
돌아올 때 나눈다. 둘 다 PRECISION 자리로 반올림하므로, 고치지 않은 라벨의 비교 기준은
quantize(label, scale) (보냈다가 그대로 받은 값)이다.

WP6, ADR 0006(무손실 왕복), ADR 0019(키프레임 시각 = 스트림 PTS ms), ADR 0024(모양 모드·지원하지
않는 주석 감사 정정). 사용처: `tasks`(보내기: `label_spec`, `to_cvat_tracks`),
`collect`(받기: `annotation_tracks`, `from_cvat_tracks`, `quantize`).

공개 이름: `CVAT_KINDS`, `TRACK_ATTRS`, `SHAPE_ATTRS`, `PRECISION`, `Scale`, `UNIT_SCALE`,
`NewBoxKind`, `label_spec`, `CvatSchema`, `CvatFormatError`, `annotation_tracks`, `quantize`,
`cvat_label_name`, `to_cvat_tracks`, `from_cvat_tracks`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from dlp_review.reconcile import ReviewedItem
from dlp_schema.labels import (
    BlurTrackPayload,
    BoxKeyframe,
    BoxTrackPayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
)

# CVAT로 오가는 라벨 종류 (블러 검수의 blur_track 포함)
CVAT_KINDS = ("box_track", "blur_track", "keypoint_track")
# 트랙 속성 (바꿀 수 없음, mutable=False): 원래 라벨 ID와 메타 JSON
TRACK_ATTRS = ("dlp_label_id", "dlp_meta")
# 모양(키프레임) 속성 (프레임마다 다를 수 있음): 키포인트 점별 가시성
SHAPE_ATTRS = ("dlp_visibility",)
PRECISION = 3  # 좌표 소수 자릿수. 왕복에서 값이 흔들리지 않게 내보낼 때 반올림한다.

Scale = tuple[float, float]  # (가로, 세로) 화면 영상 화소 / 라벨 화소
# 화면 영상과 라벨 좌표계가 같을 때 (작업 라벨 검수: 원본 해상도 블러본)
UNIT_SCALE: Scale = (1.0, 1.0)
# 검수자가 새로 그린 직사각형 트랙의 라벨 종류 (dlp_meta가 비어 있을 때). 작업 단계로 정한다.
NewBoxKind = Literal["blur_track", "box_track"]


def label_spec(names: Iterable[str]) -> list[dict[str, Any]]:
    """CVAT 프로젝트 라벨 정의.

    names: CVAT 라벨 이름 (블러 대상 종류, 객체 클래스, `kp_<골격>`).
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
    return [{"name": n, "type": "any", "attributes": attrs} for n in names]


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
    """라벨 이름이 요구하는 CVAT 모양: `kp_`로 시작하면 points, 아니면 rectangle."""
    return "points" if name.startswith("kp_") else "rectangle"


def _check_shape(shape: dict[str, Any], name: str) -> None:
    """모양이 라벨에 맞는지 검사한다. 다른 모양이거나 회전한 직사각형이면 `CvatFormatError`."""
    want = _expected_shape(name)
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
    for shape in sorted(annotations.get("shapes") or [], key=lambda x: (x["frame"], x["label_id"])):
        name = schema.label_name(shape["label_id"])
        _check_shape(shape, name)
        frame = int(shape["frame"])
        if not 0 <= frame < frame_count:
            raise CvatFormatError(f"영상 밖 프레임의 CVAT 모양: {frame}")
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
    return tracks


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

    박스·블러·키포인트 트랙만 바꾸고 그 밖의 라벨은 그대로 돌려준다.
    """
    p = label.payload
    if isinstance(p, BoxTrackPayload | BlurTrackPayload):
        kfs = tuple(
            _box_from_points(k.t_ms, _box_points(k, scale), k.outside, scale) for k in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    if isinstance(p, KeypointTrackPayload):
        sx, sy = scale
        # 보낼 때(_point_to)와 받을 때(from_cvat_tracks의 나눗셈·반올림)를 그대로 재현한다
        kfs = tuple(
            f.model_copy(
                update={
                    "points": tuple(
                        Keypoint(
                            x=_r(_point_to(pt, scale)[0] / sx),
                            y=_r(_point_to(pt, scale)[1] / sy),
                            visibility=pt.visibility,
                        )
                        for pt in f.points
                    )
                }
            )
            for f in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    return label


def cvat_label_name(label: LabelRecord) -> str:
    """라벨 → CVAT 라벨 이름 (블러: 대상 종류, 박스: 클래스, 키포인트: `kp_<골격>`).

    예외: CVAT로 보낼 수 없는 종류면 ValueError.
    """
    p = label.payload
    if isinstance(p, BlurTrackPayload):
        return p.target
    if isinstance(p, BoxTrackPayload):
        return p.class_id
    if isinstance(p, KeypointTrackPayload):
        return f"kp_{p.skeleton}"
    raise ValueError(f"CVAT로 보낼 수 없는 라벨 종류: {label.kind}")


def to_cvat_tracks(
    labels: list[LabelRecord],
    frame_times: list[int],
    schema: CvatSchema,
    *,
    scale: Scale = UNIT_SCALE,
) -> list[dict[str, Any]]:
    """frame_times: 화면 영상의 프레임 시각. scale: 화면 영상 화소 / 라벨 화소.

    반환: CVAT 트랙 목록 (`CvatClient.put_tracks`에 넘긴다). 키프레임 시각은 frame_times에서
    정확히 같은 값을 찾아 프레임 번호로 바꾼다 (보간·근사하지 않는다).
    예외: 키프레임 시각이 화면 영상 프레임 시각에 없으면 ValueError (`_shape`), 프로젝트에
    라벨·속성이 없으면 KeyError.
    """
    # PTS ms → 프레임 번호 (화면 영상 기준)
    frame_of = {t: i for i, t in enumerate(frame_times)}
    tracks: list[dict[str, Any]] = []
    for label in labels:
        p = label.payload
        name = cvat_label_name(label)
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
                # 점 목록을 [x1, y1, x2, y2, …]로 펼친다
                pts = [c for pt in f.points for c in _point_to(pt, scale)]
                shape = _shape(frame_of, f.t_ms, "points", pts, False)
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
) -> list[ReviewedItem]:
    """new_box_kind: dlp_meta가 없는(검수자가 새로 그린) 직사각형 트랙의 종류.
    프라이버시 작업이면 blur_track, 작업 라벨 작업이면 box_track이다.

    인자:
    - tracks: `annotation_tracks` 결과.
    - frame_times: 화면 영상 프레임 시각 (프레임 번호 → PTS ms).
    - stream_id: 작업의 스트림 (모든 항목에 붙인다).
    - scale: 화면 영상 화소 / 라벨 화소 (돌아올 때 나눈다).

    반환: `ReviewedItem` 목록 (트랙 순서). 구간은 첫·마지막 모양의 프레임 시각이다.
    새로 그린 박스·키포인트 트랙의 개체 ID는 `<라벨 이름>_h<트랙 순번>`이다.
    키포인트 가시성 속성이 없으면 모든 점을 2(보임)로 둔다.
    예외: 모양 없는 트랙, 지원하지 않는 모양, 영상 밖 프레임, 모르는 라벨·속성 ID면
    `CvatFormatError`.
    """
    sx, sy = scale
    items: list[ReviewedItem] = []
    for n, track in enumerate(tracks):
        name = schema.label_name(track["label_id"])
        attrs = {schema.attr_name(a["spec_id"]): a["value"] for a in track.get("attributes", [])}
        origin = attrs.get("dlp_label_id") or None
        meta: dict[str, Any] = json.loads(attrs["dlp_meta"]) if attrs.get("dlp_meta") else {}
        # 종류: 메타가 있으면 그것, 없으면 라벨 이름(kp_)과 작업 단계(new_box_kind)로 정한다
        kind = meta.get("kind") or ("keypoint_track" if name.startswith("kp_") else new_box_kind)
        shapes = sorted(track["shapes"], key=lambda s: s["frame"])
        if not shapes:
            raise CvatFormatError(f"모양이 없는 CVAT 트랙 (라벨 {name})")
        for s in shapes:
            _check_shape(s, name)
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
        else:
            frames: list[KeypointFrame] = []
            for t, s in zip(times, shapes, strict=True):
                vis_attr = {
                    schema.attr_name(a["spec_id"]): a["value"] for a in s.get("attributes", [])
                }
                pts = s["points"]
                # 점별 가시성 "2,2,1,…" (없으면 점 수만큼 2). 점 수와 맞지 않으면 CvatFormatError
                vis = _visibility(vis_attr.get("dlp_visibility", ""), len(pts) // 2, name, s)
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
            payload = KeypointTrackPayload(
                entity_id=meta.get("entity_id") or f"{name}_h{n}",
                skeleton=meta.get("skeleton") or name.removeprefix("kp_"),  # type: ignore[arg-type]
                hand=meta.get("hand"),
                keyframes=tuple(frames),
            )
        items.append(ReviewedItem(origin, stream_id, times[0], times[-1], payload))
    return items
