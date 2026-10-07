"""내보내기마다 다른 작업자·장소 가명 (export.yaml ids).

가명 = HMAC-SHA256(내보내기 키, "<종류>:<ID>")의 앞 16자.
내보내기 키 = HMAC-SHA256(비밀값, 내보내기 ID)라서
- 한 내보내기 안에서는 같은 작업자·장소가 늘 같은 가명이다 (구매자가 작업자 단위로 나눌 수 있다).
- 다른 내보내기 사이에서는 가명이 달라 서로 이어 붙일 수 없다.
- 비밀값을 아는 내부만 가명을 다시 계산해 원래 ID와 맞춰 볼 수 있다. 비밀값이 없으면 실행마다 임의의
  값을 쓰므로 아무도 되짚을 수 없다.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Pseudonymizer:
    key: bytes | None  # None이면 가명 처리하지 않는다 (정책 ids.pseudonymize: false)

    @classmethod
    def for_export(cls, export_id: str, secret: bytes | None, *, enabled: bool) -> Pseudonymizer:
        if not enabled:
            return cls(None)
        base = secret if secret else os.urandom(32)
        return cls(hmac.new(base, export_id.encode(), hashlib.sha256).digest())

    @property
    def enabled(self) -> bool:
        return self.key is not None

    def __call__(self, kind: str, value: str) -> str:
        if self.key is None:
            return value
        tag = hmac.new(self.key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:16]
        return f"{kind}-{tag}"

    def worker(self, worker_id: str) -> str:
        return self("worker", worker_id)

    def site(self, site_id: str) -> str:
        return self("site", site_id)
