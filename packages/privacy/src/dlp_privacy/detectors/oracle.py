"""정답 블러 트랙을 그대로 돌려주는 stub 탐지기 (테스트·CI용).

누락률, 위치 흔들림, 오탐을 넣어 추적·보간·유지·렌더 로직을 시험한다. 같은 seed와 시각에는
항상 같은 결과를 낸다 (호출 순서와 무관).

WP2·WP5. 정답은 합성 픽스처(`dlp_fixtures.video.generate_blur_scenario`)의 blur_track 라벨이다.
정책 파일로는 만들지 않고 테스트가 `build_detectors(extra=…)`나 `detectors={"oracle": …}`로 끼운다.
CPU만으로 도는 stub이라 CI의 기본 탐지기 역할을 한다 (CLAUDE.md: 새 모델은 stub과 함께).
"""

from __future__ import annotations

import numpy as np

from dlp_privacy.detection import Detection, Image
from dlp_privacy.geometry import Box
from dlp_schema.labels import BlurTrackPayload, LabelRecord


class OracleDetector:
    """정답 키프레임 박스를 (흔들림·누락·오탐을 넣어) 그대로 내는 `FrameDetector`."""

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
        """
        Args:
            name: 탐지기 이름.
            truth: 정답 라벨 (blur_track 외 종류는 무시). outside가 아닌 키프레임만 쓴다.
            targets: 이 대상만 낸다 (None·빈 집합이면 모두).
            miss_rate: 프레임·박스마다 놓칠 확률 (0~1).
            miss_spans_ms: (대상, 시작 ms, 끝 ms) 구간에서는 그 대상을 항상 놓친다 (양 끝 포함).
            jitter_px: 박스 위치(x, y)에 더할 정규분포 표준편차 (픽셀). 크기는 그대로.
            false_positive_rate: 프레임마다 임의 위치 20x20 "face" 오탐을 낼 확률.
            score_range: 점수를 뽑을 균등분포 범위.
            seed: 난수 시드. 프레임 난수는 (seed, t_ms)로 정해져 호출 순서와 무관하다.
        """
        self.name = name
        self.seed = seed
        self.miss_rate = miss_rate
        self.miss_spans = miss_spans_ms or []
        self.jitter = jitter_px
        self.fp_rate = false_positive_rate
        self.score_range = score_range
        # 프레임 시각(ms) → 그 프레임의 (대상, 정답 박스) 목록
        self.boxes: dict[int, list[tuple[str, Box]]] = {}
        for label in truth:
            p = label.payload
            if not isinstance(p, BlurTrackPayload) or (targets and p.target not in targets):
                continue
            for k in p.keyframes:
                if not k.outside:
                    self.boxes.setdefault(k.t_ms, []).append((p.target, Box(k.x, k.y, k.w, k.h)))

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """t_ms의 정답 박스를 낸다. 놓침·흔들림·점수·오탐은 (seed, t_ms) 난수로 정한다.

        점수가 문턱 미만이면 내지 않는다. 오탐 점수는 max(문턱, 0.35)다.
        """
        # 난수를 소비하는 순서가 박스마다 같아야 결과가 결정적이다 (놓쳐도 jitter·score를 뽑는다)
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
