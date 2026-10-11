"""승인된 블러본 렌더와 소비자 검증 (WP5, ADR 0024).

렌더 입력 해시·파일 해시를 함께 확인한다. 실행 이력·검수 세대 판단은 generation에 둔다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import sqlalchemy as sa

from dlp_media.storage import ObjectStore, blurred_key, sha256_file
from dlp_privacy.errors import PrivacyGateError, RenderNotCurrentError
from dlp_privacy.generation import VIDEO_KINDS, operational_blur
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.render import render_blurred
from dlp_schema.db.repository import get_labels, get_session, lock_session
from dlp_schema.labels import LabelRecord
from dlp_schema.session import PrivacyState


def fetch_video(store: ObjectStore, uri: str, work: Path) -> Path:
    """저장소 URI의 객체를 work 아래에 받는다 (이미 받았으면 다시 받지 않는다).

    Args:
        store: 그 URI가 속한 저장소 (원본이면 감사 저장소 → 읽기 기록이 남는다).
        uri: `store.uri(key)` 형식의 URI (Stream.uri).
        work: 임시 작업 디렉터리.

    Returns:
        로컬 파일 경로 (키의 "/"를 "__"로 바꾼 이름).

    Raises:
        ValueError: URI가 이 저장소(버킷)에 있지 않을 때.
    """
    prefix = store.uri("")
    if not uri.startswith(prefix):
        raise ValueError(f"{uri}는 저장소 {store.bucket}에 있지 않습니다")
    key = uri.removeprefix(prefix)
    dest = work / key.replace("/", "__")
    if not dest.exists():
        store.get_file(key, dest)
    return dest


def _blur_labels(conn: sa.Connection, session_id: str, stream_id: str) -> list[LabelRecord]:
    """DB에서 그 스트림의 운영 블러 라벨을 읽는다 (`operational_blur`)."""
    return operational_blur(get_labels(conn, session_id, kinds=["blur_track"]), stream_id)


def render_hash(labels: Sequence[LabelRecord], policy: PrivacyPolicy) -> str:
    """블러본을 정하는 입력의 해시: 블러 라벨 집합(라벨은 불변이라 ID로 충분)과 렌더 정책.

    Args:
        labels: 운영 블러 라벨 (`operational_blur`).
        policy: 렌더 정책(`render`)과 `platform.render_mode`만 쓴다.

    Returns:
        sha256 16진수 64자. 렌더 기록(`RenderMeta.render_hash`)과 비교한다.
    """
    content = json.dumps(
        {
            "labels": sorted(x.label_id for x in labels),
            "mode": policy.platform.render_mode,
            "render": policy.render.model_dump(mode="json"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def render_meta_key(session_id: str, stream_id: str) -> str:
    """블러본 옆에 두는 렌더 입력 기록 (라벨링 버킷). 라벨·렌더 정책 해시와 렌더 파일 해시를 담는다.

    예: `sessions/<세션>/blurred/<스트림>.render.json`.
    """
    return blurred_key(session_id, stream_id).removesuffix(".mp4") + ".render.json"


@dataclass(frozen=True)
class RenderMeta:
    """블러본 옆 렌더 기록. render_hash가 None이면 무효로 둔 기록이다 (승인 취소 등)."""

    # 렌더 당시 `render_hash` 값 (None이면 무효)
    render_hash: str | None
    blurred_sha256: str | None = None  # 렌더한 블러본 파일 해시 (이전 기록에는 없다)


def read_render_meta(
    labeling: ObjectStore, session_id: str, stream_id: str, work: Path
) -> RenderMeta | None:
    """블러본과 유효한 JSON 객체 기록이 둘 다 있으면 기록을, 아니면 None.

    부작용: 라벨링 버킷 읽기, work 디렉터리 생성.
    """
    key = blurred_key(session_id, stream_id)
    meta = render_meta_key(session_id, stream_id)
    if labeling.head(key) is None or labeling.head(meta) is None:
        return None
    work.mkdir(parents=True, exist_ok=True)
    dest = work / f"{stream_id}.render.json"
    labeling.get_file(meta, dest)
    try:
        data: object = json.loads(dest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    record = cast(dict[str, object], data)
    value, sha = record.get("render_hash"), record.get("blurred_sha256")
    # 문자열이 아닌 값(null 등)은 None으로 본다 (무효 기록·이전 형식)
    return RenderMeta(
        value if isinstance(value, str) else None, sha if isinstance(sha, str) else None
    )


def _render_is_current(
    labeling: ObjectStore, session_id: str, stream_id: str, expected: str, work: Path
) -> bool:
    """입력·객체 메타데이터·실제 바이트 해시가 모두 맞아야 재렌더를 건너뛴다.

    예전 기록/객체에 파일 해시가 없으면 이번 실행에서 다시 렌더해 기록을 보완한다.
    저장소 연결/권한 오류는 손상으로 숨기지 않고 호출자에게 전달한다.
    """
    meta = read_render_meta(labeling, session_id, stream_id, work)
    head = labeling.head(blurred_key(session_id, stream_id))
    if meta is None or meta.blurred_sha256 is None or head is None or head.sha256 is None:
        return False
    try:
        sha = check_render_meta(meta, expected, labeling, session_id, stream_id)
        existing = work / f"{stream_id}-existing.mp4"
        labeling.get_file(blurred_key(session_id, stream_id), existing)
        check_fetched(existing, sha, session_id, stream_id)
    except RenderNotCurrentError:
        return False
    return True


def write_render_meta(
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    meta: RenderMeta,
    work: Path,
    **extra: str,
) -> None:
    """렌더 기록 JSON을 라벨링 버킷에 (덮어)쓴다.

    extra: 기록에 덧붙일 문자열 필드 (예: invalidated="blur_changed").
    """
    path = work / f"{stream_id}.render.out.json"
    body = {"render_hash": meta.render_hash, "blurred_sha256": meta.blurred_sha256, **extra}
    path.write_text(json.dumps(body), encoding="utf-8")
    labeling.put_file(render_meta_key(session_id, stream_id), path, sha256_file(path))


def invalidate_render(
    labeling: ObjectStore, session_id: str, stream_id: str, reason: str, work: Path
) -> bool:
    """렌더 기록을 무효로 덮어쓴다 (블러본 파일은 남지만 어떤 단계도 현재 것으로 보지 않는다).

    라벨링 버킷 저장소에는 삭제가 없어 기록을 render_hash=None으로 바꾼다. 무효로 했으면 True.

    Args:
        reason: 무효 사유 (기록의 invalidated 필드, 예: "blur_changed").

    Returns:
        렌더 기록이 있어 무효로 바꿨으면 True, 기록이 없으면 False.
    """
    if labeling.head(render_meta_key(session_id, stream_id)) is None:
        return False
    write_render_meta(labeling, session_id, stream_id, RenderMeta(None), work, invalidated=reason)
    return True


def expected_render_hash(
    conn: sa.Connection, session_id: str, stream_id: str, policy: PrivacyPolicy
) -> str:
    """지금 DB 상태로 본 블러본 해시. 세션이 지금 프라이버시 승인 상태가 아니면 실패한다
    (데이터셋 스냅샷의 세션 상태가 아니라 현재 상태를 본다).

    Raises:
        RenderNotCurrentError: 세션이 지금 APPROVED가 아닐 때.
    """
    session = get_session(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise RenderNotCurrentError(
            f"{session_id}: 지금 프라이버시 승인 상태가 아닙니다 ({session.privacy_state.value}). "
            "블러 검수·승인(dlp privacy approve)·렌더(dlp privacy render)를 다시 하세요"
        )
    return render_hash(_blur_labels(conn, session_id, stream_id), policy)


def check_render_meta(
    meta: RenderMeta | None,
    expected: str,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
) -> str | None:
    """렌더 기록이 기대 해시와 같고 블러본 파일이 그 렌더의 것인지 본다.

    렌더한 블러본 파일 해시를 돌려준다 (이전 렌더 기록이면 None).

    Args:
        meta: `read_render_meta` 결과.
        expected: `expected_render_hash` 결과 (또는 내보내기가 미리 계산해 둔 값).

    Raises:
        RenderNotCurrentError: 기록이 없거나 무효, 해시가 다름, 블러본이 없음, 블러본 파일 해시가
            기록과 다름 (렌더 밖에서 덮어씀).
    """
    where = f"{session_id}/{stream_id}"
    if meta is None or meta.render_hash is None:
        raise RenderNotCurrentError(
            f"{where}: 현재 블러본 렌더 기록이 없거나 무효입니다 (dlp privacy render)"
        )
    if meta.render_hash != expected:
        raise RenderNotCurrentError(
            f"{where}: 블러본이 지금 승인된 블러 라벨·렌더 정책으로 만든 것이 아닙니다 "
            "(dlp privacy render)"
        )
    head = labeling.head(blurred_key(session_id, stream_id))
    if head is None:
        raise RenderNotCurrentError(f"{where}: 블러본이 없습니다 (dlp privacy render)")
    # 둘 다 해시가 있을 때만 비교한다 (이전 형식 기록이나 해시 메타데이터가 없는 객체는 넘어간다)
    if meta.blurred_sha256 and head.sha256 and head.sha256 != meta.blurred_sha256:
        raise RenderNotCurrentError(f"{where}: 블러본 파일이 렌더 기록과 다릅니다")
    return meta.blurred_sha256


def assert_render_current(
    conn: sa.Connection,
    labeling: ObjectStore,
    session_id: str,
    stream_id: str,
    policy: PrivacyPolicy,
) -> str | None:
    """블러본을 읽는 모든 단계(검수 작업·행동 VLM·큐레이션·내보내기)가 먼저 부른다 (ADR 0024).

    세션이 **지금** 프라이버시 승인 상태이고, 블러본 렌더 기록의 해시가 지금 운영 블러 라벨과
    렌더 정책의 해시와 같아야 한다. 아니면 RenderNotCurrentError. 렌더한 블러본 파일의 sha256을
    돌려준다 (이전 렌더 기록이면 None). 받은 파일과 비교해 그 사이 다시 렌더된 것을 막는다.

    사용 예::

        sha = assert_render_current(conn, labeling, sid, "bodycam", policy)
        labeling.get_file(blurred_key(sid, "bodycam"), local)
        check_fetched(local, sha, sid, "bodycam")
    """
    expected = expected_render_hash(conn, session_id, stream_id, policy)
    with tempfile.TemporaryDirectory() as tmp:
        meta = read_render_meta(labeling, session_id, stream_id, Path(tmp))
    return check_render_meta(meta, expected, labeling, session_id, stream_id)


def check_fetched(path: Path, expected_sha: str | None, session_id: str, stream_id: str) -> None:
    """받은 블러본이 렌더 기록의 파일인지 본다 (확인과 받기 사이에 다시 렌더된 경우).

    Args:
        path: 받은 로컬 블러본.
        expected_sha: `assert_render_current`의 반환값. None이면 (이전 기록) 확인하지 않는다.

    Raises:
        RenderNotCurrentError: 파일 해시가 다를 때.
    """
    if expected_sha is not None and sha256_file(path) != expected_sha:
        raise RenderNotCurrentError(
            f"{session_id}/{stream_id}: 받은 블러본이 확인한 렌더 기록과 다릅니다. 다시 시도하세요"
        )


def render_session(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    labeling: ObjectStore,
    policy: PrivacyPolicy,
) -> dict[str, str]:
    """승인된 세션의 블러본을 라벨링 버킷에 만든다. 스트림 → URI.

    같은 블러 라벨 집합·렌더 정책의 유효한 블러본이면 건너뛴다 (멱등).
    실제 파일 해시도 확인해 손상된 출력이나 기록은 다시 렌더한다. 승인·라벨 상태를 읽기 전에
    세션을 잠가 렌더/게시 동안 탐지·검수 수집·승인 취소와 다른 렌더를 직렬화한다.

    Args:
        conn: 호출자가 연 트랜잭션의 DB 연결 (조회·세션 행 잠금, PostgreSQL).
        raw: 원본 버킷 저장소 (감사 저장소 → 원본 읽기 기록).
        labeling: 라벨링 버킷 저장소 (raw와 다른 버킷이어야 한다).
        policy: 프라이버시 정책.

    Returns:
        영상 스트림 ID → 블러본 URI (`s3://dlp-labeling/sessions/<세션>/blurred/<스트림>.mp4`).

    Raises:
        PrivacyGateError: 오디오를 남기는 설정, 같은 버킷, 세션이 승인 상태가 아님.

    부작용: 라벨링 버킷에 블러본과 렌더 기록을 (덮어)쓴다. 블러본을 먼저 쓰고 기록을 나중에 써서,
    중간에 실패하면 기록 해시가 맞지 않아(또는 파일 해시가 달라) 소비자가 쓰지 못한다.
    """
    if not policy.platform.strip_audio_in_release:
        raise PrivacyGateError(
            "오디오를 남기는 블러본은 지원하지 않습니다 (strip_audio_in_release)"
        )
    if labeling.bucket == raw.bucket:
        raise PrivacyGateError("블러본은 원본 버킷이 아닌 라벨링 버킷에 써야 합니다")
    lock_session(conn, session_id)
    session = get_session(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise PrivacyGateError(f"{session_id}: 프라이버시 승인 전에는 블러본을 만들 수 없습니다")
    out: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in session.streams:
            if stream.kind not in VIDEO_KINDS:
                continue
            key = blurred_key(session_id, stream.stream_id)
            labels = _blur_labels(conn, session_id, stream.stream_id)
            digest = render_hash(labels, policy)
            # 입력이 같더라도 실제 출력이 손상되면 복구한다. 정상 출력은 재인코딩하지 않는다.
            if not _render_is_current(labeling, session_id, stream.stream_id, digest, work):
                dst = work / f"{stream.stream_id}-blurred.mp4"
                render_blurred(
                    fetch_video(raw, stream.uri, work),
                    dst,
                    labels,
                    mode=policy.platform.render_mode,
                    min_block_px=policy.render.min_block_px,
                    blocks_per_box=policy.render.blocks_per_box,
                    encoder_rate=policy.render.encoder_rate,
                    crf=policy.render.crf,
                )
                blurred_sha = sha256_file(dst)
                labeling.put_file(key, dst, blurred_sha)
                write_render_meta(
                    labeling, session_id, stream.stream_id, RenderMeta(digest, blurred_sha), work
                )
            out[stream.stream_id] = labeling.uri(key)
    return out
