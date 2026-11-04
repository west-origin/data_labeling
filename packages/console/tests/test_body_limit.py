"""F-COE-90: 올리기 경로는 본문을 읽기 전에 Content-Length로 전체 크기를 막는다.

Starlette는 처리기가 돌기 전에 multipart 본문 전체를 임시 파일로 받으므로, 파일별 상한
(F-COE-89)만으로는 받는 동안의 임시 디스크를 지키지 못한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from dlp_console.app import body_limits
from dlp_console.settings import ConsoleSettings

from .conftest import Api

FORM = {
    "session_id": "syn-big",
    "worker_id": "w-01",
    "site_id": "site-01",
    "domain": "cleaning",
    "consent_version": "consent-v2",
}
MIB = 1024 * 1024


def _small_limits(settings: ConsoleSettings) -> None:
    """두 경로의 상한을 1 MB·1 KB로 줄인 설정으로 갈아 끼운다 (요청 때 설정을 읽는다)."""
    cfg = settings.config
    upload = cfg.upload.model_copy(update={"max_mb": 1})
    pretrain = cfg.pretrain_import.model_copy(update={"max_mb": 1, "manifest_max_kb": 1})
    settings.config = cfg.model_copy(update={"upload": upload, "pretrain_import": pretrain})


def _pretrain_files(onnx: bytes) -> dict[str, tuple[str, bytes, str]]:
    return {
        "onnx": ("model.onnx", onnx, "application/octet-stream"),
        "model_manifest": ("model_manifest.json", b"{}", "application/json"),
        "data_manifest": ("data_manifest.json", b"{}", "application/json"),
    }


def _import_dirs(settings: ConsoleSettings) -> list[Path]:
    base = settings.work_dir / "pretrain"
    return list(base.iterdir()) if base.is_dir() else []


def test_endpoint_total_is_file_limits_plus_overhead_f_coe_90(settings: ConsoleSettings) -> None:
    """F-COE-90: 경로 전체 상한 = 그 경로 파일 상한의 합 + multipart 여유분."""
    cfg = settings.config
    overhead = cfg.request_body.multipart_overhead_kb * 1024
    limits = body_limits(cfg)
    assert limits["/api/upload"] == cfg.upload.max_mb * MIB + overhead
    pre = cfg.pretrain_import
    assert limits["/api/pretrain/check"] == pre.max_mb * MIB + 2 * pre.manifest_max_kb * 1024 + (
        overhead
    )


def test_declared_oversize_upload_is_413_before_body_is_read_f_coe_90(
    client: Api, settings: ConsoleSettings
) -> None:
    """F-COE-90: Content-Length가 상한을 넘으면 본문을 받지 않고 413, 업로드 디렉터리가 없다."""
    _small_limits(settings)
    res = client.post(
        "/api/upload",
        data=FORM,
        files={"file": ("a.mp4", b"\0" * (3 * MIB), "video/mp4")},
    )
    assert res.status_code == 413, res.text
    assert "MB" in res.json()["detail"]
    assert not (settings.work_dir / "uploads" / "syn-big").exists()


def test_lying_content_length_on_pretrain_check_is_413_f_coe_90(
    client: Api, settings: ConsoleSettings
) -> None:
    """F-COE-90: 작은 본문이라도 선언한 길이가 상한을 넘으면 처리기를 돌리지 않는다."""
    _small_limits(settings)
    boundary = "dlpboundary"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n{{}}\r\n"
        for name in ("onnx", "model_manifest", "data_manifest")
    ]
    body = ("".join(parts) + f"--{boundary}--\r\n").encode()
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(10 * 1024**3),
    }
    res = client.post("/api/pretrain/check", headers=headers, content=body)
    assert res.status_code == 413, res.text
    assert _import_dirs(settings) == []


def test_chunked_body_without_length_is_411_f_coe_90(
    client: Api, settings: ConsoleSettings
) -> None:
    """F-COE-90: Content-Length 없는(청크) 본문은 크기를 미리 알 수 없어 411로 받지 않는다."""

    def chunks() -> Iterator[bytes]:
        yield b"--x\r\n"
        yield b"--x--\r\n"

    headers = {"Content-Type": "multipart/form-data; boundary=x"}
    for path in ("/api/upload", "/api/pretrain/check"):
        res = client.post(path, headers=headers, content=chunks())
        assert res.status_code == 411, (path, res.text)
        assert "Content-Length" in res.json()["detail"]
    assert _import_dirs(settings) == []


def test_bundle_within_total_is_still_accepted_f_coe_90(
    client: Api, settings: ConsoleSettings
) -> None:
    """F-COE-90: 파일 상한 안의 정상 요청은 전체 상한에 걸리지 않고 처리기까지 간다."""
    _small_limits(settings)
    res = client.post("/api/pretrain/check", files=_pretrain_files(b"\0" * MIB))
    assert res.status_code == 200, res.text
    assert len(_import_dirs(settings)) == 1
