"""libx264 yuv420p 재인코딩 공용 도우미 (블러본·워터마크 사본).

yuv420p는 색차를 2x2 화소마다 하나씩 두므로 libx264가 가로·세로가 홀수인 영상을 받지 않는다.
수집은 홀수 크기 원본도 받으므로(프록시는 `proxy.proxy_size`로 줄여 짝수로 만든다), 원본 화소
좌표를 그대로 써야 하는 사본(블러본 `dlp_privacy.render`, 워터마크 `dlp_review.watermark`)은
늘이거나 자르지 않고 오른쪽·아래에 검은 화소를 1줄 덧대 짝수로 맞춘다 (F-PMS-02).
왼쪽 위 원점과 모든 원본 화소의 좌표가 그대로라, 원본 화소 좌표인 라벨(블러 박스·공간 라벨)이
사본에서도 같은 자리를 가리킨다. 사본은 원본보다 가로·세로가 최대 1픽셀 클 수 있다.

- `even_size`: 덧댄 뒤의 (가로, 세로).
- `pad_to_even`: 이미지 한 장을 덧대 짝수 크기로 만든다.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def even_size(width: int, height: int) -> tuple[int, int]:
    """홀수 변을 1픽셀 늘린 (가로, 세로). 이미 짝수면 그대로다.

    예: (101, 75) → (102, 76), (640, 480) → (640, 480).
    """
    return width + width % 2, height + height % 2


def pad_to_even(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """(H, W, C) 이미지의 오른쪽·아래에 검은(0) 화소를 덧대 가로·세로를 짝수로 만든다.

    원본 화소는 같은 좌표에 그대로 있다. 이미 짝수면 같은 배열을 돌려준다 (복사하지 않는다).
    """
    h, w = img.shape[:2]
    pw, ph = even_size(w, h)
    if (pw, ph) == (w, h):
        return img
    return np.pad(img, ((0, ph - h), (0, pw - w), (0, 0)))
