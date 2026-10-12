"""장갑 압력 접촉 구간을 판정 프롬프트 문장으로 바꾼다 (WP17.2, ADR 0040 2.5).

계산은 하지 않는다. CLI(`dlp aireview run`, WP17.5)가 프리라벨과 같은 함수
(`dlp_prelabel.contact.glove_contact_intervals`)에 같은 입력(원 샘플을 마스터 ms로 바꾼 것)을 넣어
손별 구간을 구하고 `SensorContext.of`로 넘긴다. 같은 값을 두 곳에서 따로 계산하지 않기 위해서다.

- `SensorContext`: 손별 접촉 구간(마스터 ms)과 그 해시. 장갑이 없으면 `absent()` (해시 "absent").
- `contact_text`: 증거 구간 기준 상대 초 문장
  (예: "장갑 압력 접촉(센서): 오른손 1.2~3.8 s, 왼손 없음").
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dlp_schema.labels import Hand

# 장갑이 없을 때의 해시 자리 값 (계약 AiReviewUnit.sensor_sha256)
ABSENT = "absent"
_HAND_KO = {Hand.LEFT: "왼손", Hand.RIGHT: "오른손"}


@dataclass(frozen=True)
class SensorContext:
    """세션의 장갑 접촉 구간.

    Attributes:
        intervals: 손 → 접촉 구간 [(시작 ms, 끝 ms)] (마스터 ms, 정렬). 장갑을 끼지 않은 손은
            키가 없다. None이면 장갑 스트림이 없거나 센서를 쓰지 않는다.
        sha256: 구간 JSON의 해시 (단위 키에 들어간다), 없으면 "absent".
    """

    intervals: Mapping[Hand, tuple[tuple[int, int], ...]] | None
    sha256: str

    @classmethod
    def absent(cls) -> SensorContext:
        """장갑 없음."""
        return cls(None, ABSENT)

    @classmethod
    def of(cls, intervals: Mapping[Hand, Sequence[tuple[float, float]]]) -> SensorContext:
        """손별 접촉 구간(마스터 ms)에서 만든다. 시각은 정수 ms로 반올림하고 정렬한다.

        해시 = sha256(손 이름 정렬 → [[시작, 끝], ...] JSON). 같은 구간이면 같은 해시다.
        """
        clean = {
            hand: tuple(sorted((round(a), round(b)) for a, b in spans))
            for hand, spans in intervals.items()
        }
        data = {hand.value: [list(s) for s in spans] for hand, spans in sorted(clean.items())}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return cls(clean, digest)

    @property
    def present(self) -> bool:
        """장갑 구간이 있는가."""
        return self.intervals is not None


def contact_text(ctx: SensorContext, span: tuple[int, int]) -> str | None:
    """증거 구간 `span`(마스터 ms) 안의 장갑 접촉 문장. 장갑이 없으면 None.

    구간은 증거 구간으로 자르고 증거 구간 시작 기준 상대 초(소수 한 자리)로 쓴다. 장갑을 낀 손에
    접촉이 없으면 "없음", 장갑을 끼지 않은 손은 쓰지 않는다.
    """
    if ctx.intervals is None:
        return None
    start, end = span
    parts: list[str] = []
    for hand in (Hand.RIGHT, Hand.LEFT):
        spans = ctx.intervals.get(hand)
        if spans is None:
            continue
        inside = [(max(a, start), min(b, end)) for a, b in spans if a < end and b > start]
        text = ", ".join(f"{(a - start) / 1000:.1f}~{(b - start) / 1000:.1f} s" for a, b in inside)
        parts.append(f"{_HAND_KO[hand]} {text or '없음'}")
    return "장갑 압력 접촉(센서): " + ", ".join(parts) if parts else None
