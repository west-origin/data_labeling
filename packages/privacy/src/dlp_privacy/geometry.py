"""박스 연산. 박스는 (x, y, w, h) 픽셀."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        iy = max(0.0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: Box) -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and other.x + other.w <= self.x + self.w
            and other.y + other.h <= self.y + self.h
        )

    def padded(self, margin: float) -> Box:
        dx, dy = self.w * margin / 2, self.h * margin / 2
        return Box(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def clipped(self, width: int, height: int) -> Box | None:
        x1, y1 = max(0.0, self.x), max(0.0, self.y)
        x2, y2 = min(float(width), self.x + self.w), min(float(height), self.y + self.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return Box(x1, y1, x2 - x1, y2 - y1)

    def lerp(self, other: Box, s: float) -> Box:
        return Box(
            self.x + (other.x - self.x) * s,
            self.y + (other.y - self.y) * s,
            self.w + (other.w - self.w) * s,
            self.h + (other.h - self.h) * s,
        )
