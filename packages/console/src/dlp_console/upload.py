"""영상 올리기 → 매니페스트 → `dlp media prepare` + `dlp ingest` 작업 (ADR 0061 3절).

- 올린 파일은 작업 디렉터리 `uploads/<세션>/`에만 둔다 (개인정보가 든 영상이다, 저장소·라벨링 버킷
  밖).
- 매니페스트는 수집 코드의 모델(`dlp_media.ingest.SessionManifest`)로 검증해 쓴다. 형식은
  docs/weekend-video-pilot.md "입력 준비와 등록"과 같다 (바디캠 스트림 하나).
- 동의서 버전은 운영자가 확인한 값을 받는다. 콘솔이 지어내지 않는다. 촬영 시각이 없으면 비워 두어
  수집이 영상의 creation_time을 쓰게 한다 (그것도 없으면 수집이 실패한다).
- 작업: `dlp media prepare <원본> --out <준비 사본> [--display-rotation N]` → `dlp ingest
  <매니페스트>`. 수집이 성공하면 원본 사본·준비 사본을 지운다 (`upload.delete_after_ingest`). 원본은
  원본 버킷에 불변으로 남는다.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import yaml
from pydantic import TypeAdapter, ValidationError

from dlp_console.steps import IDENT, ParamError
from dlp_media.ingest import ManifestStream, SessionManifest
from dlp_schema.session import Calibration, ConsentVersion, Domain, StreamKind

# 표시 방향을 사람이 고칠 때의 각도 (dlp media prepare --display-rotation, 90의 배수)
ROTATIONS = (0, 90, 180, 270)
CHUNK = 1024 * 1024
CONSENT_VERSION: TypeAdapter[str] = TypeAdapter(ConsentVersion)


@dataclass(frozen=True)
class UploadForm:
    """올리기 폼 값 (검증 전 문자열)."""

    session_id: str
    worker_id: str
    site_id: str
    domain: str
    consent_version: str
    recorded_at: str = ""
    camera_id: str = ""
    scene: str = ""
    display_rotation: str = ""


@dataclass(frozen=True)
class PreparedUpload:
    """작업에 넘길 값."""

    session_id: str
    directory: Path
    original: Path
    prepared: Path
    manifest_path: Path
    commands: list[list[str]]
    cleanup: list[Path]
    scene: str | None


def _ident(value: str, name: str) -> str:
    text = value.strip()
    if not IDENT.match(text):
        raise ParamError(f"{name} 값이 올바르지 않습니다 (영문·숫자·_.:-, 첫 글자는 영문·숫자)")
    return text


def _rotation(value: str) -> int | None:
    text = value.strip()
    if not text or text == "tag":
        return None
    if not re.fullmatch(r"\d{1,3}", text) or int(text) not in ROTATIONS:
        raise ParamError("촬영 방향 회전은 0·90·180·270 중 하나입니다 (반시계, 태그를 쓰면 비움)")
    return int(text)


def build_manifest(form: UploadForm, prepared_name: str) -> SessionManifest:
    """폼 → 매니페스트 (수집 모델로 검증). 예외: ParamError."""
    consent = form.consent_version.strip()
    if not consent:
        raise ParamError("확인한 동의서 버전을 적어야 합니다 (영상만으로 동의를 추정하지 않습니다)")
    # 동의의 근거라 자르거나 바꾸지 않는다.
    # 계약 길이(ConsentVersion, DB VARCHAR(64))를 넘으면 거부한다 (F-COE-83).
    try:
        CONSENT_VERSION.validate_python(consent)
    except ValidationError as exc:
        raise ParamError(
            "동의서 버전이 너무 깁니다 (64자 이하, 확인한 값을 그대로 적으세요)"
        ) from exc
    try:
        domain = Domain(form.domain.strip())
    except ValueError as exc:
        raise ParamError(f"도메인은 {', '.join(d.value for d in Domain)} 중 하나입니다") from exc
    recorded: datetime | None = None
    if form.recorded_at.strip():
        try:
            recorded = datetime.fromisoformat(form.recorded_at.strip())
        except ValueError as exc:
            raise ParamError("촬영 시각은 ISO 형식입니다 (예: 2026-10-25T14:30:00+09:00)") from exc
        if recorded.tzinfo is None:
            raise ParamError("촬영 시각에 시간대가 있어야 합니다 (예: +09:00)")
    camera = form.camera_id.strip()
    return SessionManifest(
        session_id=_ident(form.session_id, "세션 ID"),
        domain=domain,
        worker_id=_ident(form.worker_id, "작업자 ID"),
        site_id=_ident(form.site_id, "장소 ID"),
        consent_version=consent,
        recorded_at=recorded,
        calibration=Calibration(camera_id=_ident(camera, "카메라 ID") if camera else None),
        ontology_version="1.0.0",
        streams=(ManifestStream(stream_id="bodycam", kind=StreamKind.BODYCAM, path=prepared_name),),
    )


def manifest_yaml(manifest: SessionManifest) -> str:
    """매니페스트 YAML (None 값은 빼서 기본값을 쓰게 한다)."""
    data: dict[str, Any] = manifest.model_dump(mode="json", exclude_none=True)
    if not data.get("calibration"):
        data.pop("calibration", None)
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)


def save_stream(src: BinaryIO, dest: Path, max_bytes: int) -> int:
    """업로드 본문을 파일로 쓴다. 상한을 넘으면 지우고 ParamError."""
    written = 0
    with dest.open("wb") as f:
        while chunk := src.read(CHUNK):
            written += len(chunk)
            if written > max_bytes:
                f.close()
                dest.unlink(missing_ok=True)
                raise ParamError(f"파일이 상한({max_bytes // CHUNK} MB)보다 큽니다")
            f.write(chunk)
    return written


def prepare_upload(
    work_dir: Path,
    form: UploadForm,
    filename: str,
    body: BinaryIO,
    *,
    max_mb: int,
    suffixes: tuple[str, ...],
    delete_after_ingest: bool,
) -> PreparedUpload:
    """파일을 저장하고 매니페스트를 써서 작업 명령을 만든다.

    예외: ParamError (형식 오류, 이미 올린 세션 디렉터리).
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in suffixes:
        raise ParamError(f"받는 확장자: {', '.join(suffixes)}")
    rotation = _rotation(form.display_rotation)
    manifest = build_manifest(form, "prepared.mp4")
    directory = work_dir / "uploads" / manifest.session_id
    if directory.exists():
        raise ParamError(
            f"{manifest.session_id}: 이미 올린 사본이 있습니다. 진행 중이면 기다리고, 실패했으면 "
            "'업로드 사본 지우기' 뒤 다시 올리세요"
        )
    directory.mkdir(parents=True)
    directory.chmod(0o700)
    original = directory / f"original{suffix}"
    prepared = directory / "prepared.mp4"
    try:
        save_stream(body, original, max_mb * CHUNK)
    except ParamError:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    manifest_path = directory / "session.yaml"
    manifest_path.write_text(manifest_yaml(manifest), "utf-8")
    prep = ["media", "prepare", str(original), "--out", str(prepared)]
    if rotation is not None:
        prep += ["--display-rotation", str(rotation)]
    return PreparedUpload(
        session_id=manifest.session_id,
        directory=directory,
        original=original,
        prepared=prepared,
        manifest_path=manifest_path,
        commands=[prep, ["ingest", str(manifest_path)]],
        cleanup=[original, prepared] if delete_after_ingest else [],
        scene=form.scene.strip() or None,
    )


def remove_upload(work_dir: Path, session_id: str) -> bool:
    """올린 사본 디렉터리를 지운다 (있었으면 True)."""
    directory = work_dir / "uploads" / _ident(session_id, "세션 ID")
    if not directory.is_dir():
        return False
    shutil.rmtree(directory)
    return True
