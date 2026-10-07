"""모델 가중치 레지스트리 (config/models.yaml).

정책 파일은 모델을 이름으로 가리키고, 실제 파일 경로·해시 확인은 여기서 한다. 파일이 없거나
해시가 다르면 ModelUnavailableError를 낸다 (어댑터는 이를 받아 stub이나 전수 검수로 넘긴다).
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract
from dlp_schema.predictor import ModelUnavailableError

# 상업 사용 분류 (config/models.yaml 머리말, ADR 0010)
Commercial = Literal["allowed", "review", "forbidden"]


class ModelFile(Contract):
    path: str
    sha256: str
    url: str | None = None
    archive_member: str | None = None
    export: str | None = Field(default=None, description="공개 파일이 없어 직접 변환하는 스크립트")
    license: str = Field(description="가중치 라이선스")
    training_data: tuple[str, ...] = Field(description="직접 학습·미세조정 데이터와 그 라이선스")
    commercial: Commercial


class ExternalModel(Contract):
    """외부 추론 서버(VLM 등)가 가중치를 들고 있는 모델. 이 저장소는 라이선스 검사만 한다."""

    served_by: str = Field(description="어떤 서버로 쓰는지 (예: OpenAI 호환 VLM 서버)")
    license: str
    training_data: tuple[str, ...]
    commercial: Commercial


class ModelRegistry(Contract):
    version: int
    accept: tuple[Commercial, ...] = Field(description="정책이 쓸 수 있는 상업 사용 분류")
    models: dict[str, ModelFile]
    external: dict[str, ExternalModel] = Field(default_factory=dict[str, ExternalModel])

    def violations(self, used: dict[str, str]) -> list[str]:
        """정책이 쓰는 모델(쓰는 곳 → 모델 이름) 중 목록에 없거나 허용 분류가 아닌 것."""
        out: list[str] = []
        for where, name in sorted(used.items()):
            spec = self.models.get(name) or self.external.get(name)
            if spec is None:
                out.append(f"{where}: 모델 목록에 없는 {name}")
            elif spec.commercial not in self.accept:
                out.append(f"{where}: {name}은 상업 사용 분류가 {spec.commercial}")
        return out


@cache
def _digest(path: Path, mtime: float) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(root: Path) -> ModelRegistry:
    data: Any = yaml.safe_load((root / "config" / "models.yaml").read_text(encoding="utf-8"))
    return ModelRegistry.model_validate(data)


def resolve(root: Path, name: str, *, check_hash: bool = True) -> tuple[Path, str]:
    """(파일 경로, 버전 문자열). 압축에서 꺼낸 파일과 직접 변환한 파일은 꺼낸·변환한 결과의 해시가
    등록 해시와 다를 수 있어(zip 해시, 환경별 변환) 존재만 확인한다."""
    registry = load_registry(root)
    spec = registry.models.get(name)
    if spec is None:
        raise ModelUnavailableError(f"모델 목록에 {name}이 없습니다 (config/models.yaml)")
    path = root / spec.path
    if not path.is_file():
        hint = f"make export-models ({spec.export})" if spec.export else "make models"
        raise ModelUnavailableError(f"{name} 가중치가 없습니다: {path} ({hint})")
    if check_hash and spec.archive_member is None and spec.export is None:
        digest = _digest(path, path.stat().st_mtime)
        if digest != spec.sha256:
            raise ModelUnavailableError(f"{name} 가중치 해시가 다릅니다: {digest}")
        return path, f"{name}-{digest[:12]}"
    # 압축에서 꺼냈거나 직접 변환한 파일은 등록 해시(zip·변환 환경)와 다를 수 있다. 라벨의 모델
    # 버전에는 실제로 쓴 파일의 해시를 남긴다 (같은 등록 항목이라도 다른 파일이면 다른 버전이 된다).
    return path, f"{name}-{_digest(path, path.stat().st_mtime)[:12]}"


def fetch(root: Path, name: str, spec: ModelFile) -> str:
    """가중치를 받는다. 결과 메시지를 돌려준다. 변환형은 안내만 한다."""
    path = root / spec.path
    if spec.export is not None:
        return (
            f"[변환 필요] {name}: {spec.export} (make export-models)"
            if not path.is_file()
            else f"[있음] {name}"
        )
    if path.is_file() and (
        spec.archive_member or _digest(path, path.stat().st_mtime) == spec.sha256
    ):
        return f"[있음] {name}"
    assert spec.url is not None
    with urllib.request.urlopen(spec.url, timeout=600) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        return f"[실패] {name}: 해시 불일치 {digest}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.archive_member:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            member = next(n for n in z.namelist() if n.endswith(spec.archive_member))
            path.write_bytes(z.read(member))
    else:
        path.write_bytes(data)
    return f"[받음] {name}: {path}"
