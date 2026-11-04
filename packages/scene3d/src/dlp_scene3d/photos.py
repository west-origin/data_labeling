"""사진 수집: 원본은 불변으로 원본 버킷에 두고, EXIF 방향은 읽을 때 적용한다 (ADR 0066).

`dlp scene3d ingest <세션> <단계> <디렉터리>`가 `ingest_capture`를 부른다. photo_id는 원본 바이트의
sha256에서 만들므로 같은 사진을 다시 넣어도 같은 ID·같은 키다 (멱등). 이미 수집한 단계에 다른
사진 묶음을 넣으면 거부한다 (복원·승인이 어떤 사진에 대한 것인지 흐려지므로).

원본 위치는 원본 버킷의 수집 기록에만 있다. 라벨링 버킷의 수집 기록(같은 `PhotoCapture`)에는
photo_id·크기·방향만 있다. 블러 사진은 PNG로 다시 쓰므로 EXIF(위치·기기 정보)가 따라가지 않는다.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from dlp_media.storage import ObjectStore, put_immutable
from dlp_scene3d.layout import get_model, phase_prefix, put_model_immutable
from dlp_scene3d.policy import CapturePolicy
from dlp_schema.scene3d import PhotoCapture, PhotoEntry, ScenePhase

_EXIF_ORIENTATION = 0x0112


class CaptureError(ValueError):
    """수집할 수 없는 사진 묶음 (개수·형식·이미 다른 묶음이 있음)."""


def photo_id_for(sha256: str) -> str:
    """원본 sha256 → photo_id (`p` + 앞 16자)."""
    return "p" + sha256[:16]


def raw_photo_key(session_id: str, phase: ScenePhase, entry: PhotoEntry) -> str:
    """원본 버킷의 원본 사진 키."""
    return f"{phase_prefix(session_id, phase)}/photos/{entry.photo_id}{entry.ext}"


def capture_key(session_id: str, phase: ScenePhase) -> str:
    """수집 기록 키 (원본·라벨링 버킷 같은 키)."""
    return f"{phase_prefix(session_id, phase)}/capture.json"


def decode_photo(path: Path) -> tuple[NDArray[np.uint8], int]:
    """(EXIF 방향을 적용한 RGB 배열, 원본 EXIF 방향 1~8)."""
    with Image.open(path) as im:
        orientation = int(im.getexif().get(_EXIF_ORIENTATION, 1) or 1)
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return np.array(rgb, dtype=np.uint8), orientation if 1 <= orientation <= 8 else 1


def scan_photos(photo_dir: Path, policy: CapturePolicy) -> list[tuple[Path, PhotoEntry]]:
    """디렉터리의 사진을 읽어 (경로, 항목) 목록을 photo_id 순으로 낸다. 같은 사진은 한 번만.

    Raises:
        CaptureError: 사진 수가 정책 범위 밖일 때.
    """
    found: dict[str, tuple[Path, PhotoEntry]] = {}
    for path in sorted(photo_dir.iterdir()):
        ext = path.suffix.lower()
        if not path.is_file() or ext not in policy.extensions:
            continue
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        img, orientation = decode_photo(path)
        entry = PhotoEntry(
            photo_id=photo_id_for(sha),
            sha256=sha,
            ext=ext,  # pyright: ignore[reportArgumentType]
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            orientation=orientation,
        )
        found.setdefault(entry.photo_id, (path, entry))
    if not policy.min_photos <= len(found) <= policy.max_photos:
        raise CaptureError(
            f"사진 {len(found)}장: 정책 범위 {policy.min_photos}~{policy.max_photos}장 밖입니다"
        )
    return [found[k] for k in sorted(found)]


def ingest_capture(
    raw: ObjectStore,
    labeling: ObjectStore,
    session_id: str,
    phase: ScenePhase,
    photo_dir: Path,
    policy: CapturePolicy,
    now: datetime,
) -> tuple[PhotoCapture, int]:
    """(수집 기록, 새로 올린 사진 수). 이미 같은 묶음이 있으면 그 기록을 그대로 돌려준다.

    Raises:
        CaptureError: 이미 다른 사진 묶음이 수집돼 있을 때, 또는 `scan_photos`의 오류.
    """
    items = scan_photos(photo_dir, policy)
    entries = tuple(e for _, e in items)
    existing = get_model(raw, capture_key(session_id, phase), PhotoCapture)
    if existing is not None and existing.photos != entries:
        raise CaptureError(f"{session_id} {phase}에 다른 사진 묶음이 이미 수집돼 있습니다")
    uploaded = sum(put_immutable(raw, raw_photo_key(session_id, phase, e), p) for p, e in items)
    capture = existing or PhotoCapture(
        session_id=session_id, phase=phase, photos=entries, ingested_at=now
    )
    put_model_immutable(raw, capture_key(session_id, phase), capture)
    put_model_immutable(labeling, capture_key(session_id, phase), capture)
    return capture, uploaded
