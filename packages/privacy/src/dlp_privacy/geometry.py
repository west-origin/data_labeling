"""박스 연산. 박스는 (x, y, w, h) 픽셀.

WP5. 탐지·트래커·렌더가 함께 쓰는 축 정렬 직사각형이다. 원점은 이미지 왼쪽 위, x는 오른쪽, y는
아래쪽으로 증가한다. 좌표는 실수 픽셀이다 (렌더할 때 바깥쪽으로 정수화한다).

- `Box.area`, `Box.iou`, `Box.contains`: 넓이, IoU, 포함 여부.
- `Box.padded`: 대상별 여유(margin)를 더한 박스.
- `Box.clipped`: 화면 안으로 자른 박스 (완전히 밖이면 None).
- `Box.lerp`: 두 박스 사이 선형 보간.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """축 정렬 직사각형 (불변)."""

    # 왼쪽 위 모서리 (픽셀)
    x: float
    y: float
    # 너비·높이 (픽셀). 음수면 넓이 0으로 본다.
    w: float
    h: float

    @property
    def area(self) -> float:
        """넓이 (음수 너비·높이는 0으로 본다)."""
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: Box) -> float:
        """두 박스의 IoU (교집합 넓이 / 합집합 넓이). 합집합이 0이면 0."""
        ix = max(0.0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0.0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: Box) -> bool:
        """other가 이 박스 안에 완전히 들어가는가 (경계 포함)."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.x + other.w <= self.x + self.w
            and other.y + other.h <= self.y + self.h
        )

    def padded(self, margin: float) -> Box:
        """가로·세로를 각각 (크기 x margin)만큼 늘린 박스 (중심 유지, 양쪽에 절반씩).

        margin은 privacy.yaml `targets.<대상>.margin` (예: 0.2면 가로·세로 20% 크게).
        """
        dx, dy = self.w * margin / 2, self.h * margin / 2
        return Box(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def clipped(self, width: int, height: int) -> Box | None:
        """[0, width] x [0, height] 화면 안으로 자른 박스. 겹치는 부분이 없으면 None."""
        x1, y1 = max(0.0, self.x), max(0.0, self.y)
        x2, y2 = min(float(width), self.x + self.w), min(float(height), self.y + self.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(x1, y1, x2 - x1, y2 - y1)

    def lerp(self, other: Box, s: float) -> Box:
        """self(s=0)에서 other(s=1)까지 x·y·w·h를 각각 선형 보간한다. s는 범위를 자르지 않는다."""
        return Box(
            self.x + (other.x - self.x) * s,
            self.y + (other.y - self.y) * s,
            self.w + (other.w - self.w) * s,
            self.h + (other.h - self.h) * s,
        )
