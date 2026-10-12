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
새로 그린 구간도 어느 손인지 알 수 있게 한다. 라벨 이름 외의 왕복 보존용 필드는 결과의
meta.text[0]에 JSON으로 싣는다. 수정 불가능한 보안 필드는 아니다. 결과 id는 원래 라벨 ID다.

화면에서 새로 그린 객체 상태 구간은 어느 개체인지 알 수 없어 받지 않는다.
기존 객체 상태 라벨은 메타데이터의 개체 참조를 보존해 왕복한다.

라벨로 옮길 수 없는 구간(새로 그린 객체 상태, 필수 입력이 빠진 구간 등)이 있으면 그 구간들의
영역 ID·시각·라벨 이름·이유를 모두 담은 `LsFormatError` 하나로 멈춘다. `collect.collect_task`가
이것을 `TaskError`로 바꾸므로 검수자가 그 구간을 고쳐 다시 제출하면 된다 (F-REV-03).

WP6, ADR 0006(무손실 왕복). 사용처: `tasks.create_labeling_tasks`(보내기: `label_config`,
`to_ls_results`), `collect.collect_task`(받기: `from_ls_results`), `timeseries`(`CHANNELS`),
`ops.policy.UnitsPolicy`(`LS_KINDS` 검사).

공개 이름: `LS_KINDS`, `FROM_NAME`, `TO_NAME`, `CHANNELS`, `HANDS`, `EVENT_SEVERITY`,
`EVENT_TEXT_FIELDS`, `HAND_TARGET`, `HAND_BODY_PART`, `HAND_FIELDS`, `label_names`, `label_config`,
`to_ls_results`, `NAME_KINDS`, `LsFormatError`, `from_ls_results`.

이벤트 속성(심각도, 관련 행동·개체 ID)은 구간에 딸린 입력란(perRegion)으로 주고받는다. 온톨로지가
요구하는 속성을 화면에서 채울 수 없으면 새로 그린 이벤트가 저장 검증에서 거부되고 작업 수집 전체가
롤백되므로(Codex 수정본 검수, 2026-10-11) 입력란을 둔다. 손 상태의 접촉 대상 ID·대상자 부위도 같은
이유로 입력란을 둔다 (F-REV-03): 새로 그린 도구·객체·대상자 접촉, 접촉 대상 종류를 바꾼 구간.
손 상태 입력란은 프리라벨로 미리 채우지 않는다 (현재 값은 구간 메타 JSON에 보인다). 채운 입력란만
원래 값을 덮어쓰고, 비어 있으면 원래 값을 둔다. 접촉 대상 종류를 바꾸면 새 종류에 둘 수 없는 필드를
비운다 (none: 대상 ID·부위·파지 유형, person이 아닌 종류: 부위).

시간 단위: 결과의 start·end는 마스터 타임라인 ms(ADR 0019). 받을 때 반올림해 정수 ms로 만든다.
"""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import escape as xml_escape
from xml.sax.saxutils import quoteattr

from pydantic import TypeAdapter, ValidationError

from dlp_review.reconcile import ReviewedItem
from dlp_review.ui_theme import UiTheme, ls_hotkey, ls_label_color, ls_style
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

# Label Studio 화면이 보여 주는 라벨 종류 (LabelRecord.kind)
LS_KINDS = ("action", "gap", "segment", "event", "hand_state", "object_state")
# 라벨 설정 XML의 TimeSeriesLabels 이름(from_name)과 TimeSeries 이름(to_name)
FROM_NAME, TO_NAME = "labels", "ts"
# 이벤트 구간마다 붙는 입력란 (perRegion 컨트롤 이름 = EventPayload 필드 이름).
# 온톨로지가 이벤트에 심각도·관련 행동·관련 개체를 요구하므로(events.yaml), 검수자가 화면에서 새로
# 그린 이벤트도 이 값을 채울 수 있어야 저장 검증(`insert_labels`)을 통과한다.
EVENT_SEVERITY = "severity"
EVENT_TEXT_FIELDS = ("related_action_id", "related_entity_id")
# 손 상태 구간마다 붙는 입력란 (perRegion 컨트롤 이름 = HandStatePayload 필드 이름).
# object·tool·fixed_surface 접촉은 target_id, person 접촉은 body_part가 필수라(hand.yaml), 검수자가
# 새로 그리거나 접촉 대상 종류를 바꾼 구간도 이 값을 채울 수 있어야 수집된다 (F-REV-03).
# 대상 ID는 글 입력(TextArea), 부위는 온톨로지 body_parts 중 하나를 고른다(Choices).
# 프리라벨로 미리 채우지 않는다: 현재 값은 구간 메타 JSON에 있고, 비어 있는 입력란은 원래 값을
# 유지한다 (입력란이 생기기 전에 만든 작업과 같은 규칙, 고치지 않은 검수의 결과 모양도 그대로).
HAND_TARGET = "target_id"
HAND_BODY_PART = "body_part"
HAND_FIELDS = (HAND_TARGET, HAND_BODY_PART)
# 시계열 CSV 열 = 화면 채널 (timeseries.write_timeseries_csv와 같아야 한다)
CHANNELS = ("glove_left", "glove_right", "imu_acc")


# 손별 라벨 이름에 쓰는 손 값 (Hand enum 값)
HANDS = ("right", "left")


def label_names(o: Ontology) -> list[str]:
    """온톨로지에서 화면에 둘 모든 라벨 이름을 만든다.

    행동(원시 동작, 손별), 사이 구간(손별), 손 상태(접촉 대상 종류, 손별), 상위 구간(skill·task·
    substep), 이벤트, 객체 상태(`state:<속성>=<값>`) 순서다.
    """
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


def label_config(o: Ontology, theme: UiTheme | None = None, *, logo_uri: str = "") -> str:
    """Label Studio 프로젝트 설정 XML.

    영상(`$video`)과 시계열(`$timeseries`, CSV URL)을 sync="v" 그룹으로 함께 재생하고,
    시계열 위에 구간 라벨(TimeSeriesLabels)을 그린다. 시간 열은 `time_ms`(마스터 타임라인 ms).
    구간을 고르면 이벤트 속성 입력란(심각도 Choices, 관련 행동·개체 ID TextArea)과 손 상태 접촉 대상
    입력란(대상 ID TextArea, 대상자 부위 Choices = 온톨로지 body_parts)이 보인다 (모두 perRegion).
    결과는 구간과 같은 id의 별도 결과(type choices·textarea)로 오며 `from_ls_results`가 합친다.
    라벨 값은 `quoteattr`로 XML 속성 이스케이프한다.

    theme(`ui_theme.UiTheme`, config/review_ui.yaml)이 있으면 화면 모양을 덧씌운다: 라벨 색
    (`background`), 원시 동작 단축키(`hotkey`), 채널 선 색, 영상·채널 높이, 화면 CSS(`<Style>`,
    두 모드 색과 머리글 로고). logo_uri는 `ui_theme.logo_data_uri` 결과(비우면 로고 없음).
    모양만 바꾸며 라벨 이름·컨트롤 이름(from_name·to_name)은 theme과 상관없이 같다.

    예외: theme 단축키의 동사가 이 온톨로지의 원시 동작이 아니면 ValueError (오타로 단축키가
    조용히 빠지지 않게).
    """
    if theme is not None:
        primitives = {v for v, t in o.verbs.items() if t.level is VerbLevel.PRIMITIVE}
        unknown = sorted(set(theme.hotkeys) - primitives)
        if unknown:
            raise ValueError(f"review_ui.yaml hotkeys: 원시 동작이 아닌 동사 {unknown}")

    def label(n: str) -> str:
        attrs = f"value={quoteattr(n)}"
        if theme is not None:
            attrs += f' background="{ls_label_color(theme, o, n)}"'
            key = ls_hotkey(theme, n)
            if key is not None:
                attrs += f' hotkey="{key}"'
        return f"      <Label {attrs}/>"

    labels = "\n".join(label(n) for n in label_names(o))
    colors = (
        [theme.channels.get(c, theme.colors.fallback) for c in CHANNELS]
        if theme is not None
        else ["#1f77b4", "#d62728", "#2ca02c"]
    )
    height = f' height="{theme.layout.channel_height}"' if theme is not None else ""
    body_parts = "\n".join(f"      <Choice value={quoteattr(b)}/>" for b in o.body_parts)
    channels = "\n".join(
        f'      <Channel column="{c}" legend="{c}" strokeColor="{color}"{height}/>'
        for c, color in zip(CHANNELS, colors, strict=True)
    )
    if theme is None:
        head = f'  <Header value="온톨로지 v{o.version}: 손별 원시 동작과 사이 구간으로 빈틈없이"/>'
        video = '  <Video name="video" value="$video" sync="v"/>'
        style = ""
        event_class = hand_class = ""
    else:
        title = quoteattr(f"행동 구간 검수 · 블러본(원본 아님) · 온톨로지 v{o.version}")
        hint = quoteattr(
            "손별 원시 동작과 사이 구간으로 빈틈없이 · 단축키: 오른손 숫자, 왼손 shift+숫자"
        )
        head = (
            '  <View className="dlp-brand">\n'
            f'    <Header value={title} size="5"/>\n'
            f'    <Header value={hint} size="6"/>\n'
            "  </View>"
        )
        video = (
            f'  <Video name="video" value="$video" sync="v" height="{theme.layout.video_height}"/>'
        )
        style = f"  <Style>{xml_escape(ls_style(theme, logo_uri))}</Style>\n"
        event_class = ' className="dlp-event"'
        hand_class = ' className="dlp-hand"'
    return f"""<View>
{style}{head}
{video}
  <TimeSeries name="{TO_NAME}" valueType="url" value="$timeseries" sep="," timeColumn="time_ms"
              timeDisplayFormat=",.0f" sync="v">
{channels}
  </TimeSeries>
  <TimeSeriesLabels name="{FROM_NAME}" toName="{TO_NAME}">
{labels}
  </TimeSeriesLabels>
  <View visibleWhen="region-selected"{event_class}>
    <Header value="이벤트 속성 (이벤트 종류가 요구할 때만 채운다)"/>
    <Choices name="{EVENT_SEVERITY}" toName="{TO_NAME}" perRegion="true" choice="single">
      <Choice value="1"/>
      <Choice value="2"/>
      <Choice value="3"/>
    </Choices>
    <TextArea name="related_action_id" toName="{TO_NAME}" perRegion="true" maxSubmissions="1"
              placeholder="관련 행동 ID (related_action_id)"/>
    <TextArea name="related_entity_id" toName="{TO_NAME}" perRegion="true" maxSubmissions="1"
              placeholder="관련 개체 ID (related_entity_id)"/>
  </View>
  <View visibleWhen="region-selected"{hand_class}>
    <Header value="손 상태 접촉 대상 (object·tool·fixed_surface: 대상 ID, person: 부위)"/>
    <TextArea name="{HAND_TARGET}" toName="{TO_NAME}" perRegion="true" maxSubmissions="1"
              placeholder="접촉 대상 개체 ID (target_id)"/>
    <Choices name="{HAND_BODY_PART}" toName="{TO_NAME}" perRegion="true" choice="single"
             layout="select">
{body_parts}
    </Choices>
  </View>
</View>
"""


def _name_and_meta(label: LabelRecord) -> tuple[str, dict[str, Any]]:
    """라벨 → (화면 라벨 이름, 메타 JSON용 페이로드 전체).

    예외: 손이 없는 사이 구간, Label Studio가 다루지 않는 종류면 ValueError.
    """
    p = label.payload
    # 페이로드 전체를 메타로 싣는다 (화면에서 고치지 않는 필드를 되살리는 데 쓴다)
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
    """라벨 → Label Studio 결과 목록 (예측으로 올린다).

    결과 id = 원래 라벨 ID, value.start·end = 구간 ms, 시작=끝이면 instant(순간 이벤트).
    meta.text[0]에 페이로드 JSON과 `_stream_id`(원래 스트림)를 싣는다.
    """
    out: list[dict[str, Any]] = []
    for label in labels:
        name, meta = _name_and_meta(label)
        # 스트림 ID는 페이로드 밖 필드라 메타에 따로 실어 왕복시킨다
        meta["_stream_id"] = label.stream_id
        if isinstance(label.payload, EventPayload):
            # 이벤트 속성은 화면 입력란에도 채워 보낸다 (검수자가 보고 고칠 수 있게)
            out += _event_attribute_results(label.label_id, label.payload)
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


def _event_attribute_results(region_id: str, p: EventPayload) -> list[dict[str, Any]]:
    """이벤트 속성 → 구간에 딸린 perRegion 결과 (값이 있는 것만)."""
    out: list[dict[str, Any]] = []
    if p.severity is not None:
        out.append(_region_result(region_id, EVENT_SEVERITY, "choices", [str(p.severity)]))
    for field in EVENT_TEXT_FIELDS:
        value = getattr(p, field)
        if value is not None:
            out.append(_region_result(region_id, field, "textarea", [value]))
    return out


def _region_result(region_id: str, name: str, kind: str, values: list[str]) -> dict[str, Any]:
    """구간(id)에 딸린 perRegion 컨트롤 결과 하나 (Label Studio 결과 형식)."""
    key = "choices" if kind == "choices" else "text"
    return {
        "id": region_id, "from_name": name, "to_name": TO_NAME, "type": kind,
        "value": {key: values},
    }  # fmt: skip


def _region_inputs(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """perRegion 결과 → 구간 id별 입력란 값 (이벤트 속성과 손 상태 접촉 대상).

    severity는 정수, 부위는 고른 값 그대로, 글 입력(ID)은 앞뒤 공백을 뗀 마지막 값이다.
    빈 문자열·선택 없음은 값 없음으로 본다. 알 수 없는 컨트롤은 무시한다.
    """
    inputs: dict[str, dict[str, Any]] = {}
    for r in results:
        name, rid, value = r.get("from_name"), r.get("id"), r.get("value") or {}
        if rid is None:
            continue
        if name in (EVENT_SEVERITY, HAND_BODY_PART) and r.get("type") == "choices":
            picked = value.get("choices") or []
            if picked:
                choice = int(picked[0]) if name == EVENT_SEVERITY else picked[0]
                inputs.setdefault(str(rid), {})[name] = choice
        elif name in (*EVENT_TEXT_FIELDS, HAND_TARGET) and r.get("type") == "textarea":
            texts = [t.strip() for t in value.get("text") or [] if t and t.strip()]
            if texts:
                inputs.setdefault(str(rid), {})[name] = texts[-1]
    return inputs


def _apply_event_inputs(meta: dict[str, Any], region: dict[str, Any], new: bool) -> None:
    """이벤트 입력란 값(region)을 메타에 덮어쓴다. 화면 입력란의 값이 메타(원래 값)보다 우선한다.

    이벤트 입력란 결과가 하나라도 있거나 새로 그린 구간(new)이면, 결과가 없는 필드는 지운다
    (입력란을 비웠다고 본다). 이벤트 입력란 결과가 하나도 없는 원래 구간은 메타 값을 그대로 둔다.
    """
    fields = (EVENT_SEVERITY, *EVENT_TEXT_FIELDS)
    clear = new or any(f in region for f in fields)
    for field in fields:
        if field in region:
            meta[field] = region[field]
        elif clear:
            meta.pop(field, None)


def _fit_hand_fields(meta: dict[str, Any], previous: str | None) -> None:
    """접촉 대상 종류에 둘 수 없는 손 상태 필드를 비운다 (`HandStatePayload._check`, hand.yaml).

    - none: 대상 ID·대상자 부위·파지 유형을 늘 비운다 (화면 입력란에 남은 값 포함).
    - 종류를 바꿨고(previous와 다름) 새 종류가 person이 아니면 대상자 부위를 비운다.
    대상 ID·파지 유형은 none이 아닌 종류 사이에서 그대로 둔다 (입력란에서 보고 고친다).
    """
    kind = meta["contact_target_kind"]
    if kind == "none":
        for field in ("target_id", "body_part", "grasp_type"):
            meta.pop(field, None)
    elif previous is not None and previous != kind and kind != "person":
        meta.pop("body_part", None)


# 라벨 이름 머리("<종류>" 또는 "<종류>.<손>"의 종류 부분) → LabelRecord.kind
NAME_KINDS = {
    "action": "action",
    "gap": "gap",
    "skill": "segment",
    "task": "segment",
    "substep": "segment",
    "event": "event",
    "hand": "hand_state",
    "state": "object_state",
}


def _kind_of(name: str) -> str | None:
    """라벨 이름이 가리키는 라벨 종류 (알 수 없으면 None)."""
    return NAME_KINDS.get(name.partition(":")[0].partition(".")[0])


def _apply_name(name: str, meta: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """라벨 이름의 값을 덮어쓰고, 화면에서 새로 그린 구간이면 나머지 필드를 기본값으로 채운다.

    인자: name(화면 라벨 이름), meta(원래 페이로드 JSON 또는 빈 dict), start·end(구간 ms).
    반환: 페이로드 검증 직전 dict (제자리에서 고친 meta).
    새로 그린 구간의 ID: 행동 `h-<손>-<시작 ms>`, 상위 구간 `h-<수준>-<시작 ms>`.
    새로 그린 손 상태의 역할: 접촉이 있으면 active, 없으면(none) inactive. 손 상태의 대상 ID·부위는
    여기서 정하지 않는다 (호출자가 입력란 값을 넣은 뒤 `_fit_hand_fields`로 맞춘다).
    예외: 새로 그린 객체 상태 구간, 알 수 없는 라벨 이름이면 ValueError (호출자가 구간 정보를 붙여
    `LsFormatError`로 바꾼다).
    """
    head, _, key = name.partition(":")
    kind, _, hand = head.partition(".")
    # 메타에 kind가 없다 = 원래 라벨이 없는(새로 그린) 구간
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
                "화면에서 새로 그린 객체 상태 구간은 어느 개체인지 알 수 없어 받지 않습니다 "
                "(지우거나 기존 객체 상태 구간을 고친다)"
            )
        attribute, _, value = key.partition("=")
        meta |= {"kind": "object_state", "attribute": attribute, "value": value}
    else:
        raise ValueError(f"알 수 없는 Label Studio 라벨: {name}")
    return meta


class LsFormatError(ValueError):
    """Label Studio 결과를 우리 라벨로 옮길 수 없다. 옮길 수 없는 구간마다 영역 ID·시각·라벨 이름·
    이유를 담는다. 수집하지 않는다 (`collect.collect_task`가 `TaskError`로 바꾼다).

    ValueError의 하위 클래스라 예전처럼 ValueError를 잡는 호출자와도 호환된다.
    """


def _describe(r: dict[str, Any]) -> str:
    """오류 메시지용 구간 설명 `구간 <영역 ID> (<시작>~<끝> ms, <라벨 이름>)`."""
    value = r.get("value") or {}
    names = value.get("timeserieslabels") or ["?"]
    try:
        span = f"{round(float(value['start']))}~{round(float(value['end']))} ms"
    except (KeyError, TypeError, ValueError):
        span = "시각 없음"
    return f"구간 {r.get('id')} ({span}, {names[0]})"


def _reason(exc: Exception) -> str:
    """변환 실패 이유 (pydantic 검증 오류는 위치와 메시지만 짧게)."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" if e["loc"] else e["msg"]
            for e in exc.errors()
        )
    if isinstance(exc, KeyError):
        return f"결과에 필드가 없습니다: {exc}"
    return str(exc)


def _reviewed_item(
    r: dict[str, Any],
    inputs: dict[str, dict[str, Any]],
    known_ids: set[str],
    adapter: TypeAdapter[LabelPayload],
) -> ReviewedItem:
    """timeserieslabels 결과 하나 → 검수 항목.

    예외: 옮길 수 없으면 ValueError(pydantic ValidationError·JSON 오류 포함)·KeyError·IndexError.
    """
    value = r["value"]
    # 화면에서 옮긴 경계는 소수 ms일 수 있어 정수 ms로 반올림한다
    start, end = round(float(value["start"])), round(float(value["end"]))
    texts = (r.get("meta") or {}).get("text") or []
    meta: dict[str, Any] = json.loads(texts[0]) if texts else {}
    stream_id = meta.pop("_stream_id", None)
    # 한 구간에 라벨 하나만 쓴다 (여러 개 고르면 첫 번째)
    name = value["timeserieslabels"][0]
    origin = r.get("id") if r.get("id") in known_ids else None
    if "kind" in meta and meta["kind"] != _kind_of(name):
        # 다른 종류로 바꿔 붙였다 (예: 행동 → 사이 구간): 남은 필드가 새 종류와 맞지 않으므로
        # 원래 라벨을 지우고 새로 그린 것으로 본다
        meta, origin = {}, None
    previous = meta.get("contact_target_kind")  # 손 상태의 원래 접촉 대상 종류
    meta = _apply_name(name, meta, start, end)
    region = inputs.get(str(r.get("id")), {})
    if meta["kind"] == "event":
        _apply_event_inputs(meta, region, origin is None)
    elif meta["kind"] == "hand_state":
        # 채운 입력란만 덮어쓴다 (비운 입력란 = 원래 값 유지).
        # 그 뒤 새 접촉 대상 종류에 둘 수 없는 필드를 비운다
        meta |= {f: region[f] for f in HAND_FIELDS if f in region}
        _fit_hand_fields(meta, previous)
    elif meta["kind"] == "action":
        # 구간을 옮겼으면 접근 시작·종료를 맞추고 접촉 시각은 새 구간 안으로 넣는다
        meta |= {"t_approach_ms": start, "t_end_ms": end}
        for k in ("t_contact_start_ms", "t_contact_end_ms"):
            if meta.get(k) is not None:
                meta[k] = min(max(meta[k], start), end)
    return ReviewedItem(origin, stream_id, start, end, adapter.validate_python(meta))


def from_ls_results(results: list[dict[str, Any]], known_ids: set[str]) -> list[ReviewedItem]:
    """Label Studio 결과 → 검수 항목.

    인자:
    - results: 사람이 제출한 최신 주석의 결과 (`LabelStudioClient.latest_results`).
    - known_ids: 작업에 보낸 원래 라벨 ID. 결과 id가 여기 있어야 원래 라벨로 본다
      (아니면 새로 그린 것).

    반환: `ReviewedItem` 목록. timeserieslabels가 아닌 결과는 건너뛴다.
    예외: 옮길 수 없는 구간(새로 그린 객체 상태, 알 수 없는 라벨 이름, 페이로드 검증 실패 — 예: 대상
    ID 없는 도구 접촉, 깨진 메타 JSON)이 하나라도 있으면 그 구간을 모두 적은 `LsFormatError`.
    """
    adapter: TypeAdapter[LabelPayload] = TypeAdapter(LabelPayload)
    # 입력란(perRegion)은 구간과 같은 id의 별도 결과로 온다
    inputs = _region_inputs(results)
    items: list[ReviewedItem] = []
    problems: list[str] = []
    for r in results:
        if r.get("type") != "timeserieslabels":
            continue
        try:
            items.append(_reviewed_item(r, inputs, known_ids, adapter))
        except (ValueError, KeyError, IndexError) as exc:  # ValidationError도 ValueError다
            # 첫 구간에서 멈추지 않고 모두 모아, 검수자가 한 번에 고칠 수 있게 한다
            problems.append(f"{_describe(r)}: {_reason(exc)}")
    if problems:
        raise LsFormatError(
            f"라벨로 옮길 수 없는 구간 {len(problems)}개 (화면에서 고친 뒤 다시 제출한다): "
            + " | ".join(problems)
        )
    return items
