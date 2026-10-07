"""내보내기마다 다른 가명: 작업자·장소·세션·라벨 ID (export.yaml ids).

가명 = HMAC-SHA256(내보내기 키, "<종류>:<ID>")의 앞 16자.
내보내기 키 = HMAC-SHA256(비밀값, 내보내기 ID)라서
- 한 내보내기 안에서는 같은 ID가 늘 같은 가명이다 (구매자가 작업자·세션 단위로 나눌 수 있다).
- 다른 내보내기 사이에서는 가명이 달라 ID로 서로 이어 붙일 수 없다. 내용(시각·값)은 같으므로 같은
  세션을 담은 두 내보내기를 내용으로 맞춰 보는 것까지 막지는 않는다 (ADR 0027).
- 비밀값을 아는 내부만 가명을 다시 계산해 원래 ID와 맞춰 볼 수 있다. 비밀값이 없으면 실행마다 임의의
  값을 쓰므로 아무도 되짚을 수 없다 (세션 대응표는 내부 경로에 따로 남긴다, runner).
- 세션 ID가 들어간 다른 ID(예: 행동 ID "<세션>-right-…")는 그 부분을 세션 가명으로 바꾼다 (ref).
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from dlp_export.policy import IdsPolicy


def check_secret(secret: str | None, policy: IdsPolicy, environ: Mapping[str, str]) -> None:
    """개발용 비밀값(.env.example)은 개발 환경에서만 받아들인다.

    공개된 값이라 그것으로 만든 가명은 누구나 다시 계산해 내부 ID와 맞춰 볼 수 있다.
    """
    env = environ.get(policy.env_var, "")
    if secret in policy.dev_secrets and env not in policy.dev_envs:
        raise ValueError(
            f"{policy.secret_env}가 개발용 값입니다 ({policy.env_var}={env or '(없음)'}). "
            f"운영 비밀값으로 바꾸거나 개발 환경이면 {policy.env_var}={policy.dev_envs[0]}로 둡니다"
        )


@dataclass(frozen=True)
class Pseudonymizer:
    key: bytes | None  # None이면 가명 처리하지 않는다 (정책 ids.pseudonymize: false)
    _cache: dict[str, str] = field(default_factory=dict[str, str], compare=False, repr=False)

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
        k = f"{kind}:{value}"
        if k not in self._cache:
            tag = hmac.new(self.key, k.encode(), hashlib.sha256).hexdigest()[:16]
            self._cache[k] = f"{kind}-{tag}"
        return self._cache[k]

    def worker(self, worker_id: str) -> str:
        return self("worker", worker_id)

    def site(self, site_id: str) -> str:
        return self("site", site_id)

    def session(self, session_id: str) -> str:
        return self("session", session_id)

    def label(self, label_id: str) -> str:
        return self("label", label_id)

    def ref(self, session_id: str, value: str) -> str:
        """세션 ID가 들어간 다른 ID(개체·행동·구간 ID 등)의 세션 부분을 세션 가명으로 바꾼다."""
        if self.key is None or session_id not in value:
            return value
        return value.replace(session_id, self.session(session_id))

    def payload_ids(self, session_id: str, data: Any) -> Any:
        """페이로드 사전에서 이름이 _id로 끝나는 문자열 값에 ref를 적용한다 (중첩 포함)."""
        if self.key is None:
            return data
        if isinstance(data, dict):
            out: dict[str, Any] = {}
            for k, v in data.items():  # pyright: ignore[reportUnknownVariableType]
                if isinstance(k, str) and k.endswith("_id") and isinstance(v, str):
                    out[k] = self.ref(session_id, v)
                else:
                    out[str(k)] = self.payload_ids(session_id, v)  # pyright: ignore[reportUnknownArgumentType]
            return out
        if isinstance(data, list | tuple):
            return [self.payload_ids(session_id, v) for v in data]  # pyright: ignore[reportUnknownVariableType]
        return data
