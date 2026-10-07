"""검수 단계별 접근 경계.

- 프라이버시 검수(원본 접근 권한자): 원본 버킷의 프록시 영상을 본다.
- 작업 라벨 검수·QA(일반 라벨러, 선임): 라벨링 버킷의 블러본만 본다.
일반 라벨러 경로에 원본 버킷 URI가 섞이면 바로 오류를 낸다. 저장소 수준에서도 라벨러 자격 증명은
라벨링 버킷만 읽을 수 있다 (services/seaweedfs/entrypoint.sh).
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from dlp_schema.review import ReviewStage


class RawAccessError(PermissionError):
    pass


def bucket_of(uri: str) -> str | None:
    """URI가 가리키는 버킷. s3://버킷/키, http(s)://호스트/버킷/키(path-style),
    http(s)://버킷.호스트/키(virtual-host style)를 모두 본다."""
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc
    if parsed.scheme in ("http", "https"):
        first = parsed.path.lstrip("/").split("/", 1)[0]
        return first or parsed.netloc.split(".", 1)[0]
    return None


def check_stage_uris(stage: ReviewStage, uris: Iterable[str], raw_bucket: str) -> None:
    if stage is ReviewStage.PRIVACY:
        return
    leaked = [
        u
        for u in uris
        if bucket_of(u) == raw_bucket or urlparse(u).netloc.startswith(f"{raw_bucket}.")
    ]
    if leaked:
        raise RawAccessError(f"{stage.value} 단계 작업에 원본 URI가 들어 있습니다: {leaked}")


class AccessError(PermissionError):
    """원본 접근 권한이 없는 사람에게 블러(원본 영상) 검수를 배정하려 했다."""


def check_privacy_reviewers(reviewers: Iterable[str | None], allowed: Iterable[str]) -> None:
    """블러 검수 담당자는 모두 원본 접근 권한자여야 한다 (비어 있는 담당자도 막는다)."""
    denied = sorted({r or "(미배정)" for r in reviewers} - set(allowed))
    if denied:
        raise AccessError(
            f"원본 접근 권한자가 아닌 검수자에게 블러 검수를 배정할 수 없습니다: {denied} "
            "(config/policies/review.yaml reviewers.privacy)"
        )
