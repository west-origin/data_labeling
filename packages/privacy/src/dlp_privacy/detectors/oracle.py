"""정답 블러 트랙을 그대로 돌려주는 stub 탐지기 (테스트·CI용).

누락률, 위치 흔들림, 오탐을 넣어 추적·보간·유지·렌더 로직을 시험한다. 같은 seed와 시각에는
항상 같은 결과를 낸다 (호출 순서와 무관).
"""

from __future__ import annotations

import numpy as np

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box
from dlp_schema.labels import BlurTrackPayload, LabelRecord


class OracleDetector:
    version = "oracle-1"

    def __init__(
        self,
        name: str,
        truth: list[LabelRecord],
        *,
        targets: set[str] | None = None,
        miss_rate: float = 0.0,
        miss_spans_ms: list[tuple[str, int, int]] | None = None,
        jitter_px: float = 0.0,
        false_positive_rate: float = 0.0,
        score_range: tuple[float, float] = (0.7, 0.99),
        seed: int = 0,
    ) -> None:
        self.name = name
        self.seed = seed
        self.miss_rate = miss_rate
        self.miss_spans = miss_spans_ms or []
        self.jitter = jitter_px
        self.fp_rate = false_positive_rate
        self.score_range = score_range
        self.boxes: dict[int, list[tuple[str, Box]]] = {}
        for label in truth:
            p = label.payload
            if not isinstance(p, BlurTrackPayload) or (targets and p.target not in targets):
                continue
            for k in p.keyframes:
                if not k.outside:
                    self.boxes.setdefault(k.t_ms, []).append((p.target, Box(k.x, k.y, k.w, k.h)))

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        rng = np.random.default_rng([self.seed, t_ms])
        out: list[Detection] = []
        for target, box in self.boxes.get(t_ms, []):
            missed = rng.random() < self.miss_rate or any(
                tgt == target and s <= t_ms <= e for tgt, s, e in self.miss_spans
            )
            jx, jy = rng.normal(0, self.jitter, 2) if self.jitter else (0.0, 0.0)
            score = float(rng.uniform(*self.score_range))
            if not missed and score >= threshold:
                out.append(
                    Detection(target, Box(box.x + jx, box.y + jy, box.w, box.h), score, self.name)
                )
        if rng.random() < self.fp_rate:
            h, w = image.shape[:2]
            x, y = float(rng.uniform(0, w - 20)), float(rng.uniform(0, h - 20))
            out.append(Detection("face", Box(x, y, 20, 20), max(threshold, 0.35), self.name))
        return out
