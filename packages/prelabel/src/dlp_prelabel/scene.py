"""장면 분류와 장면별 OWLv2 질의 목록 (ADR 0055).

2026-10-22 파일럿에서 부엌 영상에 욕조·변기 솔·기저귀 질의가 분당 수십~백여 트랙을 냈다(F-PAR-60).
세션 장면(부엌·욕실·침실·거실)을 먼저 정하고, 그 장면에 있을 법한 도구만 OWLv2에 묻는다. 모르면
(`unknown`) 전체 목록이다 (재현을 잃지 않게).

- `SceneClassifier`: 프레임 몇 장 → `SceneResult`(장면, 신뢰도, 출처). 구현은 둘:
  - `FixedScene`: CPU·CI용 stub. 늘 정책 기본 장면(보통 unknown, 전체 질의)을 낸다.
    TODO(real-model): 사내 VLM 서버가 없는 환경에서 쓸 로컬 장면 분류기 (Places365 계열은
    비상업 라이선스라 쓸 수 없다, ADR 0010). 지금은 사내 VLM(`VlmScene`) 또는 사람이 준 장면뿐이다.
  - `VlmScene`: 사내 OpenAI 호환 VLM(`dlp_models.vlm`)에 블러를 씌운 프레임과 장면 목록을 보내
    JSON으로 장면·신뢰도를 받는다. 신뢰도가 `min_confidence` 미만이면 unknown. 외부 API는 쓰지
    않는다 (사내 서버 주소만, CLI가 `DLP_VLM_URL`에서 받는다).
- `blur_boxes`: 보낼 프레임에 그 시각 운영 블러 트랙 박스를 모자이크한다. 블러 렌더가 가리는 곳을
  모두 덮는다 (키프레임 사이는 직전·다음 박스의 외접 박스, 마지막 키프레임 뒤는 유지, F-PAR-72).
  블러 트랙이 하나도 없으면(프라이버시 탐지 전) 프레임을 보내지 않고 정책 default(출처 fallback)를
  낸다 (`classify_video`). `SceneUnavailableError`는 서버 장애일 때다.
- `queries_for`: 장면 → 질의 문장 → 온톨로지 객체 (정책 `open_vocab_objects.scene_queries`).

사람이 장면을 알면 `dlp prelabel run --scene <장면>`이 분류기보다 앞선다 (출처 manual).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.vlm import ChatReply, ChatRequest, ChatTurn, encode_jpeg
from dlp_prelabel.common import Image, iter_frames
from dlp_prelabel.policy import OpenVocabObjectsPolicy, ScenePolicy
from dlp_schema.common import OntologyId
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord

UNKNOWN = "unknown"


class SceneUnavailableError(RuntimeError):
    """장면을 물을 수 없다 (블러 트랙 없음·서버 장애 등). 호출자는 unknown으로 둔다."""


@dataclass(frozen=True)
class SceneResult:
    """장면 판정. source: fixed | vlm | manual | fallback (분류 실패로 unknown)."""

    scene: str
    confidence: float
    source: str


class SceneClassifier(Protocol):
    """장면 분류기. version은 기록용이다 (도구 단계 버전에는 분류 결과 장면이 들어간다)."""

    version: str

    def classify(self, frames: Sequence[Image]) -> SceneResult:
        """프레임들(블러 적용) → 장면."""
        ...


class FixedScene:
    """늘 같은 장면을 내는 stub (CPU·CI, 기본 unknown)."""

    def __init__(self, scene: str = UNKNOWN, source: str = "fixed") -> None:
        """장면과 출처 (사람이 준 장면이면 source=manual)."""
        self.scene, self.source = scene, source
        self.version = f"{source}-{scene}"

    def classify(self, frames: Sequence[Image]) -> SceneResult:
        """프레임을 보지 않고 정해진 장면."""
        return SceneResult(self.scene, 1.0, self.source)


class ChatClient(Protocol):
    """`dlp_models.vlm.OpenAICompatibleChat`의 쓰는 부분."""

    def chat(self, req: ChatRequest) -> ChatReply:
        """요청 하나."""
        ...


def _schema(labels: Sequence[str]) -> dict[str, Any]:
    """응답 JSON Schema: {"scene": 장면 목록 중 하나, "confidence": 0~1}."""
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string", "enum": list(labels)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["scene", "confidence"],
        "additionalProperties": False,
    }


def parse_scene(content: str, policy: ScenePolicy) -> SceneResult:
    """VLM 응답 → 장면. 형식이 틀리거나 목록 밖이거나 신뢰도가 낮으면 unknown (출처 vlm)."""
    try:
        data: Any = json.loads(content)
        scene, conf = str(data["scene"]), float(data["confidence"])
    except (ValueError, KeyError, TypeError):
        return SceneResult(UNKNOWN, 0.0, "vlm")
    if scene not in policy.labels or not 0 <= conf <= 1 or conf < policy.min_confidence:
        return SceneResult(UNKNOWN, conf if 0 <= conf <= 1 else 0.0, "vlm")
    return SceneResult(scene, conf, "vlm")


class VlmScene:
    """사내 VLM 장면 분류 (블러 적용 프레임, JSON Schema 강제)."""

    def __init__(self, chat: ChatClient, model: str, policy: ScenePolicy) -> None:
        """chat: 사내 서버 클라이언트. model: 모델 이름 (버전에 들어간다)."""
        self.chat, self.policy = chat, policy
        self.version = f"vlm-{model}"

    def classify(self, frames: Sequence[Image]) -> SceneResult:
        """프레임을 긴 변 max_side로 줄여 보내고 응답을 해석한다.

        Raises:
            SceneUnavailableError: 서버 장애·요청 오류 (호출자가 unknown으로 둔다).
        """
        images = tuple(encode_jpeg(_shrink(f, self.policy.max_side), 85) for f in frames)
        prompt = (
            "These frames come from one first-person (egocentric) video of household or care "
            f"work. Which room is it? Answer one of: {', '.join(self.policy.labels)}. "
            "Use 'unknown' if unsure."
        )
        req = ChatRequest(
            system=None,
            turns=(ChatTurn("user", (*images, prompt)),),
            schema=_schema(self.policy.labels),
            schema_name="scene",
        )
        try:
            reply = self.chat.chat(req)
        except RuntimeError as exc:  # VlmUnavailableError·VlmRequestError
            raise SceneUnavailableError(str(exc)) from exc
        return parse_scene(reply.content, self.policy)


def _shrink(img: Image, max_side: int) -> NDArray[np.uint8]:
    """긴 변을 max_side 이하로 줄인 RGB."""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s >= 1.0:
        return np.ascontiguousarray(img)
    out = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return np.asarray(out, dtype=np.uint8)


def _boxes_at(
    keyframes: Sequence[BoxKeyframe], t_ms: int, tol_ms: int
) -> list[tuple[float, float, float, float]]:
    """t_ms에 가릴 박스 (x0, y0, x1, y1) 목록. 블러 렌더보다 넓게 가린다 (F-PAR-72).

    렌더(`dlp_privacy.render`)는 직전 키프레임이 보이면 다음 키프레임까지 프레임 번호로 보간하고,
    마지막 키프레임 뒤는 그 박스를 유지한다. 여기서는 프리라벨이 privacy를 import할 수 없어
    (architecture.md 의존 표) 보간 대신 직전·다음 박스의 외접 박스를 쓴다. 선형 보간 박스는 늘 이
    외접 박스 안이므로 렌더가 가리는 곳을 모두 덮는다. 그 밖에 tol_ms 안 가장 가까운 보이는
    키프레임도 덮는다 (첫 키프레임 직전 프레임, 시각 반올림 차).
    """
    if not keyframes:
        return []
    out: list[tuple[float, float, float, float]] = []
    prev = [k for k in keyframes if k.t_ms <= t_ms]
    nxt = [k for k in keyframes if k.t_ms > t_ms]
    if prev and not prev[-1].outside:
        a = prev[-1]
        x0, y0, x1, y1 = a.x, a.y, a.x + a.w, a.y + a.h
        if nxt and not nxt[0].outside:
            b = nxt[0]
            x0, y0 = min(x0, b.x), min(y0, b.y)
            x1, y1 = max(x1, b.x + b.w), max(y1, b.y + b.h)
        out.append((x0, y0, x1, y1))
    near = min(keyframes, key=lambda k: abs(k.t_ms - t_ms))
    if abs(near.t_ms - t_ms) <= tol_ms and not near.outside:
        out.append((near.x, near.y, near.x + near.w, near.y + near.h))
    return out


def blur_boxes(
    frame: Image, t_ms: int, blur: Iterable[LabelRecord], tol_ms: int, cell_px: int
) -> NDArray[np.uint8]:
    """그 시각 블러 트랙 박스를 모자이크(cell_px 칸)한 사본 (가릴 곳은 `_boxes_at`, 렌더 이상)."""
    out = np.array(frame, copy=True)
    h, w = out.shape[:2]
    for x in blur:
        p = x.payload
        if not isinstance(p, BlurTrackPayload):
            continue
        for bx0, by0, bx1, by1 in _boxes_at(p.keyframes, t_ms, tol_ms):
            # 부분 픽셀도 덮도록 바깥쪽으로 정수화한다
            x0, y0 = max(0, int(bx0)), max(0, int(by0))
            x1, y1 = min(w, int(bx1) + 1), min(h, int(by1) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]
            small = cv2.resize(
                roi, (max(1, (x1 - x0) // cell_px), max(1, (y1 - y0) // cell_px)),
                interpolation=cv2.INTER_AREA,
            )  # fmt: skip
            out[y0:y1, x0:x1] = cv2.resize(
                small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
            )
    return out


def sample_times(frame_times: Sequence[int], n: int) -> list[int]:
    """영상 프레임 시각에서 고르게 n개 (처음·끝 10%는 뺀다: 시작 정지·끝 정리 장면)."""
    if not frame_times:
        return []
    lo, hi = int(len(frame_times) * 0.1), max(int(len(frame_times) * 0.9), 1)
    idx = np.linspace(lo, hi - 1, num=min(n, hi - lo), dtype=int)
    return [frame_times[i] for i in idx]


def queries_for(scene: str, policy: OpenVocabObjectsPolicy) -> dict[str, OntologyId]:
    """장면의 질의 문장 → 온톨로지 객체. 장면 목록이 없으면(unknown 등) 전체 질의."""
    names = policy.scene_queries.get(scene)
    if names is None:
        return dict(policy.queries)
    return {q: policy.queries[q] for q in names}


def classify_video(
    video: Path,
    blur: Sequence[LabelRecord],
    classifier: SceneClassifier,
    policy: ScenePolicy,
) -> SceneResult:
    """영상에서 sample_frames장을 골라 블러를 씌우고 장면을 묻는다.

    블러 트랙이 하나도 없으면(프라이버시 탐지 전) 프레임을 보내지 않고 default(출처 fallback)다.
    분류기가 `SceneUnavailableError`를 내도 default(fallback)다.
    """
    if not any(isinstance(x.payload, BlurTrackPayload) for x in blur):
        return SceneResult(policy.default, 0.0, "fallback")
    times = sample_times([t for t, _ in iter_frames(video)], policy.sample_frames)
    wanted = set(times)
    frames = [
        blur_boxes(img, t, blur, policy.blur_tolerance_ms, policy.blur_cell_px)
        for t, img in iter_frames(video)
        if t in wanted
    ]
    try:
        return classifier.classify(frames)
    except SceneUnavailableError:
        return SceneResult(policy.default, 0.0, "fallback")
