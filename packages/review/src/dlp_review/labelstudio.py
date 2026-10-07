# 외부 도구의 JSON을 다루는 경계 모듈이라 알 수 없는 타입 경고를 이 모듈에서만 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

"""LabelRecord ↔ Label Studio 시계열 구간 라벨, 그리고 프로젝트 설정 XML.

시간 구간 라벨(행동, 사이 구간, 상위 구간, 이벤트, 손 상태, 객체 상태)은 TimeSeriesLabels로 다룬다.
시계열의 시간 열이 마스터 타임라인 ms라서 프레임 변환 없이 시각이 그대로 오간다.
영상과 장갑·IMU 시계열은 sync 그룹으로 함께 재생한다.

라벨 이름은 "<종류>[.<손>]:<핵심 값>"이다
(예: action.right:grasp, gap.left:idle, event:slip,
hand.right:tool, state:cleanliness=clean).
손별 트랙이 있는 종류는 손을 이름에 넣어, 검수자가 화면에서
새로 그린 구간도 어느 손인지 알 수 있게 한다. 화면에서 고치지 않는 나머지 필드는 결과의
meta.text[0]에 JSON으로 싣는다. 결과 id는 원래 라벨 ID다.

화면에서 새로 그린 객체 상태 구간은 어느 개체인지 알 수 없어 받지 않는다 (CVAT에서 개체를 고른다).
"""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import quoteattr

from dlp_review.reconcile import ReviewedItem
from dlp_schema.labels import (
    ActionPayload,
    EventPayload,
    GapPayload,
    HandStatePayload,
    LabelPayload,
    LabelRecord,
    ObjectStatePayload,
    SegmentPayload,
)
from dlp_schema.ontology import Ontology, VerbLevel

LS_KINDS = ("action", "gap", "segment", "event", "hand_state", "object_state")
FROM_NAME, TO_NAME = "labels", "ts"
CHANNELS = ("glove_left", "glove_right", "imu_acc")


HANDS = ("right", "left")


def label_names(o: Ontology) -> list[str]:
    primitives = [v for v, t in o.verbs.items() if t.level is VerbLevel.PRIMITIVE]
    names = [f"action.{h}:{v}" for h in HANDS for v in primitives]
    names += [f"gap.{h}:{g}" for h in HANDS for g in o.gap_types]
    names += [f"hand.{h}:{k}" for h in HANDS for k in o.contact_target_kinds]
    names += [f"skill:{v}" for v, t in o.verbs.items() if t.level is not VerbLevel.PRIMITIVE]
    names += [f"task:{t}" for t in o.tasks]
    names += [f"substep:{s}" for t in o.tasks.values() for s in t.substeps]
    names += [f"event:{e}" for e in o.events]
    names += [f"state:{a}={v}" for a, sa in o.state_attributes.items() for v in sa.values]
    return names


def label_config(o: Ontology) -> str:
    labels = "\n".join(f"      <Label value={quoteattr(n)}/>" for n in label_names(o))
    channels = "\n".join(
        f'      <Channel column="{c}" legend="{c}" strokeColor="{color}"/>'
        for c, color in zip(CHANNELS, ("#1f77b4", "#d62728", "#2ca02c"), strict=True)
    )
    return f"""<View>
  <Header value="온톨로지 v{o.version}: 손별 원시 동작과 사이 구간으로 빈틈없이"/>
  <Video name="video" value="$video" sync="v"/>
  <TimeSeries name="{TO_NAME}" valueType="url" value="$timeseries" sep="," timeColumn="time_ms"
              timeDisplayFormat=",.0f" sync="v">
{channels}
  </TimeSeries>
  <TimeSeriesLabels name="{FROM_NAME}" toName="{TO_NAME}">
{labels}
  </TimeSeriesLabels>
</View>
"""


def _name_and_meta(label: LabelRecord) -> tuple[str, dict[str, Any]]:
    p = label.payload
    meta: dict[str, Any] = p.model_dump(mode="json")
    match p:
        case ActionPayload():
            name = f"action.{p.hand.value}:{p.verb}"
        case GapPayload() if p.hand is not None:
            name = f"gap.{p.hand.value}:{p.gap_type}"
        case GapPayload():
            raise ValueError("손이 정해지지 않은 사이 구간은 Label Studio로 보낼 수 없습니다")
        case SegmentPayload():
            name = f"{p.level.value}:{p.ref_id}"
        case EventPayload():
            name = f"event:{p.event_type}"
        case HandStatePayload():
            name = f"hand.{p.hand.value}:{p.contact_target_kind}"
        case ObjectStatePayload():
            name = f"state:{p.attribute}={p.value}"
        case _:
            raise ValueError(f"Label Studio로 보낼 수 없는 라벨 종류: {label.kind}")
    return name, meta


def to_ls_results(labels: list[LabelRecord]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label in labels:
        name, meta = _name_and_meta(label)
        meta["_stream_id"] = label.stream_id
        out.append(
            {
                "id": label.label_id,
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "timeserieslabels",
                "value": {
                    "start": label.t_start_ms,
                    "end": label.t_end_ms,
                    "instant": label.t_start_ms == label.t_end_ms,
                    "timeserieslabels": [name],
                },
                "meta": {"text": [json.dumps(meta, ensure_ascii=False)]},
            }
        )
    return out


def _apply_name(name: str, meta: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """라벨 이름의 값을 덮어쓰고, 화면에서 새로 그린 구간이면 나머지 필드를 기본값으로 채운다."""
    head, _, key = name.partition(":")
    kind, _, hand = head.partition(".")
    new = "kind" not in meta
    if kind == "action":
        meta |= {"kind": "action", "verb": key, "hand": hand}
        if new:
            meta["action_id"] = f"h-{hand}-{start}"
    elif kind == "gap":
        meta |= {"kind": "gap", "gap_type": key, "hand": hand}
    elif kind in ("skill", "task", "substep"):
        meta |= {"kind": "segment", "level": kind, "ref_id": key}
        if new:
            meta["segment_id"] = f"h-{kind}-{start}"
    elif kind == "event":
        meta |= {"kind": "event", "event_type": key}
    elif kind == "hand":
        meta |= {"kind": "hand_state", "contact_target_kind": key, "hand": hand}
        if new:
            meta["role"] = "active" if key != "none" else "inactive"
    elif kind == "state":
        if new:
            raise ValueError(
                f"화면에서 새로 그린 객체 상태 구간은 받지 않습니다 ({start}~{end} ms)"
            )
        attribute, _, value = key.partition("=")
        meta |= {"kind": "object_state", "attribute": attribute, "value": value}
    else:
        raise ValueError(f"알 수 없는 Label Studio 라벨: {name}")
    return meta


def from_ls_results(results: list[dict[str, Any]], known_ids: set[str]) -> list[ReviewedItem]:
    from pydantic import TypeAdapter

    adapter: TypeAdapter[LabelPayload] = TypeAdapter(LabelPayload)
    items: list[ReviewedItem] = []
    for r in results:
        if r.get("type") != "timeserieslabels":
            continue
        value = r["value"]
        start, end = round(float(value["start"])), round(float(value["end"]))
        texts = (r.get("meta") or {}).get("text") or []
        meta: dict[str, Any] = json.loads(texts[0]) if texts else {}
        stream_id = meta.pop("_stream_id", None)
        meta = _apply_name(value["timeserieslabels"][0], meta, start, end)
        if meta["kind"] == "action":
            # 구간을 옮겼으면 접근 시작·종료를 맞추고 접촉 시각은 새 구간 안으로 넣는다
            meta |= {"t_approach_ms": start, "t_end_ms": end}
            for k in ("t_contact_start_ms", "t_contact_end_ms"):
                if meta.get(k) is not None:
                    meta[k] = min(max(meta[k], start), end)
        origin = r.get("id") if r.get("id") in known_ids else None
        items.append(ReviewedItem(origin, stream_id, start, end, adapter.validate_python(meta)))
    return items
