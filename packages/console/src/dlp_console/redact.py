"""화면에 보내는 글에서 원본 위치와 비밀값을 가린다 (ADR 0061 4절).

콘솔 사용자는 원본 접근 권한자일 수 있지만, 화면은 일반 라벨러도 볼 수 있는 경로로 취급한다.
그래서 원본 버킷 URI(`s3://<원본 버킷>/…`)와 원본 버킷 이름이 든 경로는 늘 가린다 (CLAUDE.md:
원본 버킷 URI를 라벨러 경로에 노출하지 않는다). 비밀값은 이름에 PASSWORD·SECRET·TOKEN·KEY가 든
환경 변수의 값이다 (8자 이상만, 짧은 값은 보통 문자와 겹친다).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

RAW_MASK = "[원본 위치 숨김]"
SECRET_MASK = "[비밀값 숨김]"
_SECRET_NAMES = ("PASSWORD", "SECRET", "TOKEN", "KEY")


class Redactor:
    """원본 버킷 이름과 비밀값으로 만든 가리개. `apply(text)`로 쓴다."""

    def __init__(self, raw_bucket: str, env: Mapping[str, str]) -> None:
        bucket = re.escape(raw_bucket)
        # s3://dlp-raw/..., http(s)://host/dlp-raw/..., 경로 속 /dlp-raw/... 를 공백·따옴표 전까지
        self._raw = re.compile(rf"(?:s3://|https?://[^\s'\"]*/|/)?{bucket}(?:/[^\s'\"),]*)?")
        values = {
            v for k, v in env.items() if len(v) >= 8 and any(n in k.upper() for n in _SECRET_NAMES)
        }
        # 긴 값부터 바꾼다 (한 값이 다른 값의 일부일 때)
        self._secrets = sorted(values, key=len, reverse=True)

    def apply(self, text: str) -> str:
        """가린 글."""
        for value in self._secrets:
            text = text.replace(value, SECRET_MASK)
        return self._raw.sub(RAW_MASK, text)
