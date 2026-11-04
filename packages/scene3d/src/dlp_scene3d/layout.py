"""저장소 키 배치와 JSON 읽고 쓰기 (ADR 0066).

원본 버킷(감사 저장소로만 연다): 원본 사진, 원본 쪽 수집 기록, 블러 탐지·승인 기록.
    sessions/<세션>/scene3d/<단계>/photos/<photo_id><확장자>
    sessions/<세션>/scene3d/<단계>/capture.json
    sessions/<세션>/scene3d/<단계>/privacy/detections/<버전 태그>.json   (추가만)
    sessions/<세션>/scene3d/<단계>/privacy/approvals/<승인 ID>.json     (추가만)
    sessions/<세션>/scene3d/<단계>/privacy/latest.json, approved.json   (최신을 가리키는 표시)
라벨링 버킷(일반 라벨러가 읽는다): 승인된 블러 사진과 그 뒤 모든 산출물. 원본 위치는 담지 않는다.
    sessions/<세션>/scene3d/<단계>/capture.json
    sessions/<세션>/scene3d/<단계>/blurred/<렌더 태그>/<photo_id>.png, blurred/render.json
    sessions/<세션>/scene3d/<단계>/recon/<태그>/{reconstruction.json,cameras.json,points.npz,splat.ply}
    sessions/<세션>/scene3d/<단계>/recon/current.json
    sessions/<세션>/scene3d/changes/<태그>.json, changes/current.json
    sessions/<세션>/scene3d/alignment/<태그>.json

산출물은 버전 태그 키에 한 번만 쓴다 (`dlp_media.storage.put_immutable`). current·latest·render
표시만 바뀐다 (이력이 아니라 지금 무엇을 쓰는지 가리킨다).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from dlp_media.storage import ObjectStore, put_immutable, sha256_file
from dlp_schema.scene3d import ScenePhase


def phase_prefix(session_id: str, phase: ScenePhase) -> str:
    """단계 하나의 키 머리."""
    return f"sessions/{session_id}/scene3d/{phase}"


def session_prefix(session_id: str) -> str:
    """세션 전후 비교의 키 머리."""
    return f"sessions/{session_id}/scene3d"


def short_hash(*parts: str) -> str:
    """버전 태그 (sha256 앞 12자)."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _write_tmp(data: bytes, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as f:
        f.write(data)
        return Path(f.name)


def put_bytes_immutable(store: ObjectStore, key: str, data: bytes, suffix: str = "") -> bool:
    """바이트를 불변 키에 쓴다 (같은 내용이면 건너뛰어 False).

    Raises:
        ImmutableObjectError: 같은 키에 다른 내용이 있을 때.
    """
    path = _write_tmp(data, suffix)
    try:
        return put_immutable(store, key, path)
    finally:
        path.unlink(missing_ok=True)


def put_model_immutable(store: ObjectStore, key: str, model: BaseModel) -> bool:
    """계약 하나를 JSON으로 불변 키에 쓴다."""
    return put_bytes_immutable(store, key, model.model_dump_json(indent=1).encode(), ".json")


def put_pointer(store: ObjectStore, key: str, data: dict[str, object] | BaseModel) -> None:
    """바뀌는 표시(current·latest·render)를 쓴다 (덮어쓴다)."""
    raw = data.model_dump_json() if isinstance(data, BaseModel) else json.dumps(data)
    path = _write_tmp(raw.encode(), ".json")
    try:
        store.put_file(key, path, sha256_file(path))
    finally:
        path.unlink(missing_ok=True)


def get_bytes(store: ObjectStore, key: str) -> bytes | None:
    """키의 내용. 없으면 None (head로 먼저 본다: head는 감사 기록을 남기지 않는다)."""
    if store.head(key) is None:
        return None
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "obj"
        store.get_file(key, dest)
        return dest.read_bytes()


def get_json(store: ObjectStore, key: str) -> object | None:
    """키의 JSON. 없으면 None."""
    raw = get_bytes(store, key)
    return None if raw is None else json.loads(raw)


def get_pointer(store: ObjectStore, key: str) -> dict[str, str] | None:
    """바뀌는 표시(문자열 값 JSON 객체). 없거나 형식이 다르면 None."""
    data = get_json(store, key)
    if not isinstance(data, dict):
        return None
    items = cast(dict[object, object], data).items()
    return {str(k): str(v) for k, v in items}


def get_model[M: BaseModel](store: ObjectStore, key: str, model: type[M]) -> M | None:
    """키의 계약. 없으면 None."""
    raw = get_bytes(store, key)
    return None if raw is None else model.model_validate_json(raw)
