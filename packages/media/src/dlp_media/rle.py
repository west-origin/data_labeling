"""COCO 압축 RLE 마스크 부호화·복호화 (`dlp_schema.labels.Rle`, ADR 0059).

마스크 프리라벨(`dlp_prelabel.masks`)·CVAT 왕복(`dlp_review.cvat`)·COCO 내보내기·학습 증강 공용이다.

COCO(pycocotools) 형식: 마스크를 열 우선(Fortran 순서)으로 펼쳐 0부터 시작하는 연속 길이 목록을
만들고, 그 목록을 6비트 문자로 줄인 문자열이다 (네 번째 값(인덱스 3)부터는 두 칸 앞 값과의 차를
적는다). size는 (높이, 너비).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import Rle


def _counts_to_string(counts: list[int]) -> str:
    """연속 길이 목록 → COCO 압축 문자열."""
    out: list[str] = []
    for i, value in enumerate(counts):
        x = value - counts[i - 2] if i > 2 else value
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = x != -1 if c & 0x10 else x != 0
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def _string_to_counts(s: str) -> list[int]:
    """COCO 압축 문자열 → 연속 길이 목록."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def encode_rle(mask: NDArray[np.bool_]) -> Rle:
    """(높이, 너비) 불리언 마스크 → Rle."""
    flat = np.asarray(mask, dtype=bool).flatten(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate([[0], edges, [flat.size]])
    counts = [int(n) for n in np.diff(bounds)]
    if flat.size and flat[0]:
        counts.insert(0, 0)  # COCO RLE는 0(배경) 길이부터 센다
    h, w = mask.shape
    return Rle(size=(int(h), int(w)), counts=_counts_to_string(counts))


def decode_rle(rle: Rle) -> NDArray[np.bool_]:
    """Rle → (높이, 너비) 불리언 마스크.

    Raises:
        ValueError: 길이 중 음수가 있거나(손상된 문자열, F-PMS-44) 길이 합이 크기와 다를 때.
    """
    h, w = rle.size
    counts = _string_to_counts(rle.counts)
    bad = next((i for i, n in enumerate(counts) if n < 0), None)
    if bad is not None:
        raise ValueError(f"RLE {bad}번째 길이가 음수 {counts[bad]}")
    if sum(counts) != h * w:
        raise ValueError(f"RLE 길이 합 {sum(counts)} != {h}x{w}")
    flat = np.zeros(h * w, dtype=bool)
    pos, value = 0, False
    for n in counts:
        if value:
            flat[pos : pos + n] = True
        pos += n
        value = not value
    return flat.reshape((h, w), order="F")
