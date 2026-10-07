"""VLM 클라이언트: CPU용 stub(정답 기반)과 OpenAI 호환 서버(vLLM 등) 어댑터."""

from __future__ import annotations

# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false
import base64
import json
from typing import Any

import av
import cv2
import httpx
import numpy as np

from dlp_actions.vlm import SegmentRequest
from dlp_media.probe import to_fraction
from dlp_schema.labels import ActionPayload, DescriptionPayload, GapPayload, LabelRecord


class OracleVlm:
    """정답 라벨로 답하는 stub. 구간과 가장 많이 겹치는 정답 행동·사이 구간을 고른다.

    faults: 앞 몇 번의 호출에 일부러 틀린 응답을 낸다 (재시도·미상 처리 시험용).
      "not_json" | "bad_verb" | "bad_target" 을 순서대로 소비한다.
    """

    def __init__(self, truth: list[LabelRecord], faults: list[str] | None = None) -> None:
        self.version = "oracle-vlm-1"
        self.truth = truth
        self.faults = list(faults or [])
        self.calls = 0
        self.descriptions = {
            x.payload.segment_id: x.payload.text
            for x in truth
            if isinstance(x.payload, DescriptionPayload)
        }

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
        self.calls += 1
        if self.faults:
            fault = self.faults.pop(0)
            if fault == "not_json":
                return "이 구간은 문지르기입니다"
            if fault == "bad_verb":
                return json.dumps({"label": "action", "verb": "dance", "description": "춤춘다"})
            if fault == "bad_target":
                return json.dumps(
                    {"label": "action", "verb": "rub", "target_id": "moon_01", "description": "-"}
                )
        best: LabelRecord | None = None
        overlap = 0
        for x in self.truth:
            p = x.payload
            if not isinstance(p, ActionPayload | GapPayload) or (
                p.hand not in (None, request.hand)
            ):
                continue
            o = min(x.t_end_ms, request.end_ms) - max(x.t_start_ms, request.start_ms)
            if o > overlap:
                best, overlap = x, o
        if best is None:
            return json.dumps({"label": "gap", "gap_type": "unknown", "description": None})
        p = best.payload
        if isinstance(p, ActionPayload):
            text = self.descriptions.get(p.action_id, f"{p.verb} {p.target_id or ''}".strip())
            return json.dumps(
                {"label": "action", "verb": p.verb, "target_id": p.target_id, "tool_id": p.tool_id,
                 "description": text, "confidence": 0.9}
            )  # fmt: skip
        assert isinstance(p, GapPayload)
        return json.dumps(
            {"label": "gap", "gap_type": p.gap_type, "description": None, "confidence": 0.9}
        )


def sample_frames(request: SegmentRequest, n: int, max_side: int) -> list[bytes]:
    """구간에서 PTS 기준으로 고르게 n장을 골라 JPEG로."""
    if request.video is None:
        return []
    wanted = list(np.linspace(request.start_ms, request.end_ms, n + 2)[1:-1])
    out: list[bytes] = []
    with av.open(str(request.video)) as c:
        stream = c.streams.video[0]
        tb = to_fraction(stream.time_base)
        c.seek(max(0, int(request.start_ms / 1000 / tb)), stream=stream, backward=True)
        for frame in c.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb * 1000)
            if t > request.end_ms or not wanted:
                break
            if t >= wanted[0]:
                wanted.pop(0)
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]
                s = min(1.0, max_side / max(h, w))
                img = cv2.resize(img, (round(w * s), round(h * s))) if s < 1 else img
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    out.append(jpg.tobytes())
    return out


class OpenAICompatibleVlm:
    """OpenAI 호환 채팅 API (vLLM 등)로 프레임 여러 장 + 프롬프트를 보내고 JSON 응답을 받는다.

    TODO(real-model): 실제 VLM 서버에서 아직 검증하지 않았다 (GPU 필요). 요청 형식만 테스트한다.
    """

    def __init__(
        self, base_url: str, model: str, *, frames: int, max_side: int, timeout_s: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:  # fmt: skip
        self.version = f"vlm-{model}"
        self.model, self.frames, self.max_side = model, frames, max_side
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
        images = [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()},
            }
            for j in sample_frames(request, self.frames, self.max_side)
        ]
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [*images, {"type": "text", "text": prompt}]}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "segment_label", "schema": schema, "strict": True},
            },
        }
        resp = self.http.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data: Any = resp.json()
        return str(data["choices"][0]["message"]["content"])
