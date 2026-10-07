"""내보내기마다 다른 가명: 작업자·장소·세션·라벨 ID (export.yaml ids, ADR 0021·0027).

가명 = HMAC-SHA256(내보내기 키, "<종류>:<ID>")의 앞 16자.
내보내기 키 = HMAC-SHA256(비밀값, 내보내기 ID)라서
- 한 내보내기 안에서는 같은 ID가 늘 같은 가명이다 (구매자가 작업자·세션 단위로 나눌 수 있다).
- 다른 내보내기 사이에서는 가명이 달라 ID로 서로 이어 붙일 수 없다. 내용(시각·값)은 같으므로 같은
  세션을 담은 두 내보내기를 내용으로 맞춰 보는 것까지 막지는 않는다 (ADR 0027).
- 비밀값을 아는 내부만 가명을 다시 계산해 원래 ID와 맞춰 볼 수 있다. 비밀값이 없으면 실행마다 임의의
  값을 쓰므로 아무도 되짚을 수 없다 (세션 대응표는 내부 경로에 따로 남긴다, runner).
- 세션 ID가 들어간 다른 ID(예: 행동 ID "<세션>-right-…")는 그 부분을 세션 가명으로 바꾼다 (ref).

가명 형식: "<종류>-<16진 16자>" (예: "worker-1a2b…", "session-…"). 종류를 키에 넣으므로 같은
문자열이라도 작업자 가명과 장소 가명은 다르다.

주요 공개 이름:
- `check_secret`: 개발용(공개) 비밀값을 운영에서 쓰지 못하게 막는다.
- `Pseudonymizer`: 내보내기 하나의 가명 함수 (`for_export`로 만든다).
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

    Args:
        secret: 가명 비밀값 (`policy.secret_env` 환경 변수 값). None이면 검사할 것이 없다
            (실행마다 임의 값을 쓰게 된다).
        policy: export.yaml `ids` 절.
        environ: 환경 변수 (보통 `os.environ`, 테스트에서는 사전).

    Raises:
        ValueError: 비밀값이 `dev_secrets` 중 하나인데 `environ[env_var]`가 `dev_envs` 밖일 때.
    """
    env = environ.get(policy.env_var, "")
    if secret in policy.dev_secrets and env not in policy.dev_envs:
        raise ValueError(
            f"{policy.secret_env}가 개발용 값입니다 ({policy.env_var}={env or '(없음)'}). "
            f"운영 비밀값으로 바꾸거나 개발 환경이면 {policy.env_var}={policy.dev_envs[0]}로 둡니다"
        )


@dataclass(frozen=True)
class Pseudonymizer:
    """내보내기 하나의 가명 함수. 호출하면 `(종류, ID) → "<종류>-<태그>"`.

    `key`가 None이면 모든 메서드가 입력을 그대로 돌려준다 (가명 처리 끔).
    결과는 `_cache`에 기억한다 (같은 ID를 여러 번 바꾸므로). frozen이지만 캐시 사전 자체는 바뀐다.
    """

    key: bytes | None  # None이면 가명 처리하지 않는다 (정책 ids.pseudonymize: false)
    # "<종류>:<ID>" → 가명. 비교·repr에서 뺀다 (같은 키면 같은 객체로 본다)
    _cache: dict[str, str] = field(default_factory=dict[str, str], compare=False, repr=False)

    @classmethod
    def for_export(cls, export_id: str, secret: bytes | None, *, enabled: bool) -> Pseudonymizer:
        """내보내기 ID로 내보내기 키를 만든다.

        Args:
            export_id: 내보내기 ID (`runner.export_id_for`). 내보내기마다 달라 가명도 달라진다.
            secret: 내부 보관 비밀값. None·빈 값이면 32바이트 임의 값을 쓴다 (되짚을 수 없음).
            enabled: 정책 `ids.pseudonymize`. 거짓이면 가명 처리하지 않는 객체를 돌려준다.
        """
        if not enabled:
            return cls(None)
        base = secret if secret else os.urandom(32)
        return cls(hmac.new(base, export_id.encode(), hashlib.sha256).digest())

    @property
    def enabled(self) -> bool:
        """가명 처리가 켜져 있는지."""
        return self.key is not None

    def __call__(self, kind: str, value: str) -> str:
        """`value`의 가명. 종류(`kind`)를 HMAC 입력에 넣어 종류가 다르면 가명도 다르다.

        Args:
            kind: "worker"·"site"·"session"·"label" 등. 가명의 앞부분이 된다.
            value: 내부 ID.

        Returns:
            "<kind>-<HMAC 16진 앞 16자>". 가명 처리가 꺼져 있으면 `value` 그대로.
        """
        if self.key is None:
            return value
        k = f"{kind}:{value}"
        if k not in self._cache:
            tag = hmac.new(self.key, k.encode(), hashlib.sha256).hexdigest()[:16]
            self._cache[k] = f"{kind}-{tag}"
        return self._cache[k]

    def worker(self, worker_id: str) -> str:
        """작업자 ID 가명."""
        return self("worker", worker_id)

    def site(self, site_id: str) -> str:
        """장소 ID 가명."""
        return self("site", site_id)

    def session(self, session_id: str) -> str:
        """세션 ID 가명 (결과 파일 이름·manifest·대응표에도 쓴다)."""
        return self("session", session_id)

    def label(self, label_id: str) -> str:
        """라벨 ID 가명. 라벨 ID에 세션 ID가 들어 있어도 통째로 해시하므로 남지 않는다."""
        return self("label", label_id)

    def ref(self, session_id: str, value: str) -> str:
        """세션 ID가 들어간 다른 ID(개체·행동·구간 ID 등)의 세션 부분을 세션 가명으로 바꾼다.

        개체 ID("cup_1")처럼 세션 ID가 없으면 그대로 둔다. 문자열 치환이라 `value` 안의 모든
        `session_id` 부분 문자열이 바뀐다.
        """
        if self.key is None or session_id not in value:
            return value
        return value.replace(session_id, self.session(session_id))

    def payload_ids(self, session_id: str, data: Any) -> Any:
        """페이로드 사전에서 이름이 _id로 끝나는 문자열 값에 ref를 적용한다 (중첩 포함).

        Args:
            session_id: 이 라벨의 내부 세션 ID.
            data: `payload.model_dump(mode="json")` 결과 (사전·목록·스칼라).

        Returns:
            같은 구조의 새 값. 튜플은 목록이 된다. `_id`로 끝나지 않는 키(예: "verb")의 문자열은
            세션 ID와 같아도 바꾸지 않는다.
        """
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
