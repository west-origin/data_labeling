"""검수 단계별 접근 경계.

- 프라이버시 검수(원본 접근 권한자): 원본 버킷의 프록시 영상을 본다.
- 작업 라벨 검수·QA(일반 라벨러, 선임): 라벨링 버킷의 블러본만 본다.
일반 라벨러 경로에 원본 버킷 URI가 섞이면 바로 오류를 낸다. 저장소 수준에서도 라벨러 자격 증명은
라벨링 버킷만 읽을 수 있다 (services/seaweedfs/entrypoint.sh).

관련: WP6, ADR 0006(역할 경계), ADR 0020(원본 접근 감사), ADR 0023(블러 검수 권한자).
CLAUDE.md 규칙 "원본 버킷 URI를 일반 라벨러 경로에 노출하지 않는다"의 검사 지점이다.

공개 이름:
- `RawAccessError`: 일반 라벨러 단계에 원본 URI가 섞였다.
- `bucket_of`: URI → 버킷 이름 (s3://, path-style, virtual-host style URL).
- `check_stage_uris`: 단계별 URI 검사 (`dlp_review.tasks.create_labeling_tasks`가 부른다).
- `AccessError`: 원본 접근 권한이 없는 사람에게 블러 검수를 배정하려 했다.
- `check_privacy_reviewers`: 블러 검수 담당자 권한 검사 (정책 `review.yaml reviewers.privacy`).
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse

from dlp_schema.review import ReviewStage


class RawAccessError(PermissionError):
    """원본 접근 권한이 없는 단계(작업 라벨 검수·QA)의 작업에 원본 버킷 URI가 들어 있다."""


def bucket_of(uri: str) -> str | None:
    """URI가 가리키는 버킷. s3://버킷/키, http(s)://호스트/버킷/키(path-style),
    http(s)://버킷.호스트/키(virtual-host style)를 모두 본다.

    반환: 버킷 이름. 알 수 없는 스킴이면 None.

    주의: http(s) URL에서 경로가 있으면 경로 첫 조각을 버킷으로 본다(path-style 가정).
    virtual-host style URL은 경로 첫 조각이 키의 일부라서 여기서는 버킷을 잘못 돌려줄 수 있다.
    그래서 `check_stage_uris`는 호스트 이름이 `<원본 버킷>.`으로 시작하는지도 따로 본다.
    """
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        return parsed.netloc
    if parsed.scheme in ("http", "https"):
        first = parsed.path.lstrip("/").split("/", 1)[0]
        # 경로가 비었으면 virtual-host style로 보고 호스트 첫 조각을 버킷으로 본다
        return first or parsed.netloc.split(".", 1)[0]
    return None


def check_stage_uris(stage: ReviewStage, uris: Iterable[str], raw_bucket: str) -> None:
    """검수 단계에 맞지 않는 원본 URI가 섞였는지 검사한다.

    인자:
    - stage: 검수 단계. PRIVACY(원본 접근 권한자)는 원본을 봐도 되므로 검사하지 않는다.
    - uris: 작업에 들어가는 모든 URI (매체 URI, 서명 URL 등).
    - raw_bucket: 원본 버킷 이름 (보통 `dlp-raw`, 원본 저장소의 `bucket`).

    예외: `RawAccessError` — 원본 버킷을 가리키는 URI가 하나라도 있으면 (목록을 메시지에 담는다).
    부작용 없음.
    """
    if stage is ReviewStage.PRIVACY:
        return
    leaked = [
        u
        for u in uris
        # path-style·s3:// 판정 + virtual-host style(호스트가 "<버킷>."으로 시작) 판정
        if bucket_of(u) == raw_bucket or urlparse(u).netloc.startswith(f"{raw_bucket}.")
    ]
    if leaked:
        raise RawAccessError(f"{stage.value} 단계 작업에 원본 URI가 들어 있습니다: {leaked}")


class AccessError(PermissionError):
    """원본 접근 권한이 없는 사람에게 블러(원본 영상) 검수를 배정하려 했다."""


def check_privacy_reviewers(reviewers: Iterable[str | None], allowed: Iterable[str]) -> None:
    """블러 검수 담당자는 모두 원본 접근 권한자여야 한다 (비어 있는 담당자도 막는다).

    인자:
    - reviewers: 배정하려는 담당자 ID 목록. None(미배정)은 "(미배정)"으로 보고 거부한다.
    - allowed: 원본 접근 권한자 (`config/policies/review.yaml` `reviewers.privacy`).

    예외: `AccessError` — 권한자가 아닌 담당자가 있으면 (정렬된 목록을 메시지에 담는다).
    원본을 읽거나 검수 도구에 올리기 **전에** 불러야 한다 (ADR 0020: 접근 전에 막고 기록).
    """
    denied = sorted({r or "(미배정)" for r in reviewers} - set(allowed))
    if denied:
        raise AccessError(
            f"원본 접근 권한자가 아닌 검수자에게 블러 검수를 배정할 수 없습니다: {denied} "
            "(config/policies/review.yaml reviewers.privacy)"
        )
