"""잘라 낸 박스의 AI 확인: 블러·도구 박스 오탐 의심 힌트 (ADR 0056 2절, ADR 0040).

검수자에게 보일 힌트만 낸다. 판정은 라벨도 검수 상태도 아니고 DB의 라벨·검수 기록에 쓰지 않는다
(`dlp review hints --ai`가 힌트 파일에 `ai_crop_no` 이유로 싣는다).

입력 픽셀은 지금 승인된 블러본(`media.open_blurred`·`frames_for`)에서만 나온다 (ADR 0040 I1). 그래서
블러 박스의 가운데는 모자이크다. 블러 박스는 "가운데가 일부러 뭉개졌다"고 알리고 주변 맥락(머리카락·
어깨·몸, 종이 테두리 등)으로 그 대상이 그럴듯한지만 묻는다. 도구 박스는 블러본에서 그대로 보인다.

- `CropAsk`: 물을 박스 하나 (라벨 ID, 종류 privacy|object, 대상·클래스, 프레임 번호, 박스).
- `crop_with_margin`: 박스를 각 변 margin 비율 넓혀 자르고 긴 변 max_side로 줄인다.
- `CropVerifier` 프로토콜, `OracleCropVerifier`(CPU·CI stub: 늘 unsure), `VlmCropVerifier`(사내
  서버, `dlp_models.vlm`, JSON Schema 강제). 외부 API는 쓰지 않는다 (CLI가 `backend.check_hosts`로
  사내 주소만 받는다).
- `to_doubts`: 판정 → 라벨 ID별 의심 정도 (no이고 신뢰도가 min_confidence 이상일 때만).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_aireview.media import BlurredFrame
from dlp_models.vlm import ChatReply, ChatRequest, ChatTurn, encode_jpeg

Answer = Literal["yes", "no", "unsure"]


@dataclass(frozen=True)
class CropAsk:
    """물을 박스 하나."""

    label_id: str
    kind: Literal["privacy", "object"]
    subject: str  # 블러 대상 ID 또는 온톨로지 클래스 ID
    frame_index: int  # 블러본 PTS 인덱스 위치
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class CropVerdict:
    """박스 하나의 답 (yes: 대상 맞음, no: 아님, unsure)."""

    label_id: str
    answer: Answer
    confidence: float


class ChatClient(Protocol):
    """`dlp_models.vlm.OpenAICompatibleChat`의 쓰는 부분."""

    def chat(self, req: ChatRequest) -> ChatReply:
        """요청 하나."""
        ...


class CropVerifier(Protocol):
    """잘라 낸 이미지 → 판정."""

    version: str

    def verify(self, ask: CropAsk, crop: NDArray[np.uint8]) -> CropVerdict:
        """박스 하나."""
        ...


def crop_with_margin(
    rgb: NDArray[np.uint8], box: tuple[float, float, float, float], margin: float, max_side: int
) -> NDArray[np.uint8]:
    """박스를 각 변 margin 비율 넓혀 자른다 (화면 안으로 자름, 긴 변 max_side 이하)."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    x0, y0 = max(0, int(x - bw * margin)), max(0, int(y - bh * margin))
    x1, y1 = min(w, int(x + bw * (1 + margin)) + 1), min(h, int(y + bh * (1 + margin)) + 1)
    out = np.ascontiguousarray(rgb[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)])
    side = max(out.shape[:2])
    if side > max_side:
        s = max_side / side
        out = np.asarray(
            cv2.resize(out, (max(1, round(out.shape[1] * s)), max(1, round(out.shape[0] * s)))),
            dtype=np.uint8,
        )
    return out


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}


def prompt_for(ask: CropAsk) -> str:
    """질문 글 (영어, 대상 ID는 그대로)."""
    subject = ask.subject.replace("_", " ")
    if ask.kind == "privacy":
        return (
            "This crop is from a first-person video. The central region is intentionally pixelated "
            f"for privacy because a detector thought it shows a {subject}. Using only the visible "
            f"surroundings, is it plausible that the pixelated region really covers a {subject}? "
            "Answer yes, no, or unsure."
        )
    return (
        f"This crop is from a first-person video. Does it show a {subject}? "
        "Answer yes, no, or unsure."
    )


def parse_verdict(label_id: str, content: str) -> CropVerdict:
    """응답 → 판정. 형식이 틀리면 unsure (신뢰도 0)."""
    try:
        data: Any = json.loads(content)
        answer, conf = str(data["answer"]), float(data["confidence"])
    except (ValueError, KeyError, TypeError):
        return CropVerdict(label_id, "unsure", 0.0)
    if answer not in ("yes", "no", "unsure") or not 0 <= conf <= 1:
        return CropVerdict(label_id, "unsure", 0.0)
    return CropVerdict(label_id, answer, conf)  # type: ignore[arg-type]


class OracleCropVerifier:
    """CPU·CI stub: 늘 unsure (힌트를 만들지 않는다).

    TODO(real-model): 사내 VLM 서버가 없는 환경에서 쓸 로컬 조각 확인 모델이 없다.
    """

    version = "oracle-crop-1"

    def verify(self, ask: CropAsk, crop: NDArray[np.uint8]) -> CropVerdict:
        """판단하지 않는다."""
        return CropVerdict(ask.label_id, "unsure", 0.0)


class VlmCropVerifier:
    """사내 VLM으로 묻는다 (한 박스 한 요청, JSON Schema 강제, temperature는 서버 기본)."""

    def __init__(self, chat: ChatClient, model: str, jpeg_quality: int) -> None:
        """chat: 사내 서버 클라이언트."""
        self.chat, self.jpeg_quality = chat, jpeg_quality
        self.version = f"crop-vlm-{model}"

    def verify(self, ask: CropAsk, crop: NDArray[np.uint8]) -> CropVerdict:
        """잘라 낸 이미지와 질문을 보낸다. 서버 오류는 그대로 올린다 (호출자가 unsure로 둔다)."""
        image = encode_jpeg(crop, self.jpeg_quality)
        req = ChatRequest(
            system=None,
            turns=(ChatTurn("user", (image, prompt_for(ask))),),
            schema=SCHEMA,
            schema_name="crop_check",
        )
        return parse_verdict(ask.label_id, self.chat.chat(req).content)


def verify_crops(
    asks: Sequence[CropAsk],
    frames: Iterable[BlurredFrame],
    verifier: CropVerifier,
    *,
    margin: float,
    max_side: int,
) -> list[CropVerdict]:
    """블러본 프레임(프레임 번호 순)마다 그 프레임의 박스를 잘라 묻는다.

    서버 장애(RuntimeError)는 그 박스를 unsure로 둔다 (힌트를 만들지 않을 뿐 실패가 아니다).
    """
    by_frame: dict[int, list[CropAsk]] = {}
    for a in asks:
        by_frame.setdefault(a.frame_index, []).append(a)
    out: list[CropVerdict] = []
    for frame in frames:
        for a in by_frame.get(frame.index, []):
            crop = crop_with_margin(frame.rgb, a.box, margin, max_side)
            try:
                out.append(verifier.verify(a, crop))
            except RuntimeError:
                out.append(CropVerdict(a.label_id, "unsure", 0.0))
    return out


def to_doubts(verdicts: Iterable[CropVerdict], min_confidence: float) -> dict[str, float]:
    """라벨 ID → 의심 정도 (no 판정 비율 x 평균 신뢰도). no가 없는 라벨은 뺀다."""
    by: dict[str, list[CropVerdict]] = {}
    for v in verdicts:
        by.setdefault(v.label_id, []).append(v)
    out: dict[str, float] = {}
    for label_id, vs in by.items():
        no = [v for v in vs if v.answer == "no" and v.confidence >= min_confidence]
        if no:
            out[label_id] = round(len(no) / len(vs) * float(np.mean([v.confidence for v in no])), 4)
    return out
