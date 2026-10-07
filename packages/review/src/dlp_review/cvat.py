# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""LabelRecord ↔ CVAT 비디오 트랙.

CVAT는 프레임 번호로 주석을 저장하므로 PTS 인덱스의 프레임 시각(정수 ms)으로 맞바꾼다.
우리 쪽에는 프레임 번호를 남기지 않는다. 라벨마다 트랙 속성 두 개를 붙인다.
- dlp_label_id: 원래 라벨 ID (검수자가 새로 그린 트랙은 비어 있다)
- dlp_meta: 화면에서 고치지 않는 필드 (종류, 개체 ID, 손 등)
키포인트의 점별 가시성은 모양 속성 dlp_visibility("2,2,1,…")로 싣는다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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

CVAT_KINDS = ("box_track", "blur_track", "keypoint_track")
TRACK_ATTRS = ("dlp_label_id", "dlp_meta")
SHAPE_ATTRS = ("dlp_visibility",)
PRECISION = 3  # 좌표 소수 자릿수. 왕복에서 값이 흔들리지 않게 내보낼 때 반올림한다.


def label_spec(names: Iterable[str]) -> list[dict[str, Any]]:
    """CVAT 프로젝트 라벨 정의."""
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

    label_ids: dict[str, int]
    attr_ids: dict[tuple[str, str], int]

    @classmethod
    def from_labels(cls, labels: list[dict[str, Any]]) -> CvatSchema:
        return cls(
            {lb["name"]: lb["id"] for lb in labels},
            {(lb["name"], a["name"]): a["id"] for lb in labels for a in lb["attributes"]},
        )

    def label_name(self, label_id: int) -> str:
        return next(n for n, i in self.label_ids.items() if i == label_id)

    def attr_name(self, spec_id: int) -> str:
        return next(n for (_, n), i in self.attr_ids.items() if i == spec_id)


def _r(v: float) -> float:
    return round(float(v), PRECISION)


def quantize(label: LabelRecord) -> LabelRecord:
    """좌표를 PRECISION 자리로 반올림한 라벨 (CVAT에 보내기 전 기준 값)."""
    p = label.payload
    if isinstance(p, BoxTrackPayload | BlurTrackPayload):
        kfs = tuple(
            k.model_copy(update={"x": _r(k.x), "y": _r(k.y), "w": _r(k.w), "h": _r(k.h)})
            for k in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    if isinstance(p, KeypointTrackPayload):
        kfs = tuple(
            f.model_copy(
                update={
                    "points": tuple(
                        pt.model_copy(update={"x": _r(pt.x), "y": _r(pt.y)}) for pt in f.points
                    )
                }
            )
            for f in p.keyframes
        )
        return label.model_copy(update={"payload": p.model_copy(update={"keyframes": kfs})})
    return label


def cvat_label_name(label: LabelRecord) -> str:
    p = label.payload
    if isinstance(p, BlurTrackPayload):
        return p.target
    if isinstance(p, BoxTrackPayload):
        return p.class_id
    if isinstance(p, KeypointTrackPayload):
        return f"kp_{p.skeleton}"
    raise ValueError(f"CVAT로 보낼 수 없는 라벨 종류: {label.kind}")


def to_cvat_tracks(
    labels: list[LabelRecord], frame_times: list[int], schema: CvatSchema
) -> list[dict[str, Any]]:
    frame_of = {t: i for i, t in enumerate(frame_times)}
    tracks: list[dict[str, Any]] = []
    for label in labels:
        p = quantize(label).payload
        name = cvat_label_name(label)
        meta: dict[str, Any] = {"kind": label.kind}
        shapes: list[dict[str, Any]] = []
        if isinstance(p, BoxTrackPayload | BlurTrackPayload):
            if isinstance(p, BoxTrackPayload):
                meta["entity_id"] = p.entity_id
            for k in p.keyframes:
                shapes.append(
                    _shape(
                        frame_of, k.t_ms, "rectangle", [k.x, k.y, k.x + k.w, k.y + k.h], k.outside
                    )
                )
        elif isinstance(p, KeypointTrackPayload):
            meta |= {"entity_id": p.entity_id, "skeleton": p.skeleton, "hand": p.hand}
            vis_id = schema.attr_ids[(name, "dlp_visibility")]
            for f in p.keyframes:
                pts = [c for pt in f.points for c in (pt.x, pt.y)]
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
    tracks: list[dict[str, Any]], frame_times: list[int], schema: CvatSchema, stream_id: str
) -> list[ReviewedItem]:
    items: list[ReviewedItem] = []
    for n, track in enumerate(tracks):
        name = schema.label_name(track["label_id"])
        attrs = {schema.attr_name(a["spec_id"]): a["value"] for a in track.get("attributes", [])}
        origin = attrs.get("dlp_label_id") or None
        meta: dict[str, Any] = json.loads(attrs["dlp_meta"]) if attrs.get("dlp_meta") else {}
        kind = meta.get("kind") or ("keypoint_track" if name.startswith("kp_") else "blur_track")
        shapes = sorted(track["shapes"], key=lambda s: s["frame"])
        times = [frame_times[s["frame"]] for s in shapes]
        payload: Any
        if kind in ("box_track", "blur_track"):
            kfs = tuple(
                BoxKeyframe(
                    t_ms=t,
                    x=_r(s["points"][0]),
                    y=_r(s["points"][1]),
                    w=_r(s["points"][2] - s["points"][0]),
                    h=_r(s["points"][3] - s["points"][1]),
                    outside=bool(s["outside"]),
                )
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
                vis = [int(v) for v in vis_attr.get("dlp_visibility", "").split(",") if v] or [
                    2
                ] * (len(pts) // 2)
                frames.append(
                    KeypointFrame(
                        t_ms=t,
                        points=tuple(
                            Keypoint(x=_r(pts[2 * i]), y=_r(pts[2 * i + 1]), visibility=vis[i])  # type: ignore[arg-type]
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
