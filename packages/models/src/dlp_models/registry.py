"""모델 가중치 레지스트리 (config/models.yaml).

정책 파일은 모델을 이름으로 가리키고, 실제 파일 경로·해시 확인은 여기서 한다. 파일이 없거나
해시가 다르면 ModelUnavailableError를 낸다. 실행 중단·명시적 부분 실행·전수 검수 여부는 호출자가
정한다 (기본 프리라벨 명령은 선택한 모델 미준비 시 중단한다).

파이프라인 위치:
- `make models` / `dlp models fetch` → `fetch`: 가중치를 `data/models/`에 받고 sha256을 확인한다.
- `make licenses` → `ModelRegistry.violations`: 정책이 쓰는 모델의 상업 사용 분류 검사 (ADR 0010).
  공개 데이터셋으로 1차 학습한 가중치의 분류는 `dlp_models.datasets.dataset_violations` (ADR 0060).
- 각 어댑터(프리라벨·프라이버시) 생성자 → `resolve`: 파일 경로와 라벨에 남길 버전 문자열.
관련: WP8, ADR 0009(실제 모델 레지스트리), ADR 0010(상업 사용).

주의:
- 버전 문자열(`<이름>-<파일 sha256 앞 12자>`)은 라벨의 `provenance.model_version`과 라벨 ID
  (`version_tag`)에 들어간다. 가중치 파일이 바뀌면 버전이 바뀌고, 프리라벨이 다시 돌며 검수 전인
  이전 결과만 지운다 (ADR 0015).
- `config/models.yaml`은 파싱된 값으로만 읽는다 (원문 해시를 쓰지 않으므로 YAML 주석은 버전에 영향이
  없다).
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field

from dlp_schema.common import Contract
from dlp_schema.predictor import ModelUnavailableError

# 상업 사용 분류 (config/models.yaml 머리말, ADR 0010)
#   allowed: 가중치·직접 학습 데이터 모두 상업 사용 가능
#   review: 가중치는 허용하지만 직접 학습 데이터에 비상업·연구용 조건 (판매 전 법무 확인)
#   forbidden: 가중치 자체가 비상업 → 쓰지 않는다
Commercial = Literal["allowed", "review", "forbidden"]


class ModelFile(Contract):
    """`models.yaml models.<이름>` 항목: 저장소가 직접 받아 로컬에서 돌리는 가중치 파일.

    필드:
    - path: 저장소 루트 기준 상대 경로 (`data/models/...`, 저장소에는 넣지 않는다).
    - sha256: 등록 해시. `url` 파일(압축이면 zip 전체)의 해시다. `export`형은 변환한 환경의 값.
    - url: 직접 받을 주소. `export`와 둘 중 하나만 있다 (테스트가 검사).
    - archive_member: url이 zip이면 꺼낼 파일 이름의 끝부분 (예: `end2end.onnx`).
    - artifact_sha256: 압축에서 꺼낸 파일의 등록 해시. ZIP 모델은 캐시 검증에도 필요하다.
    - export: 공개 ONNX가 없어 공식 가중치에서 변환하는 스크립트 (`make export-models`).
    - license / training_data / commercial: 상업 사용 판단 근거 (ADR 0010).
    - training_datasets / base_model: 공개 데이터셋 1차 학습 가중치의 계보 (ADR 0060, 선택).
    """

    path: str
    sha256: str
    url: str | None = None
    archive_member: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    export: str | None = Field(default=None, description="공개 파일이 없어 직접 변환하는 스크립트")
    license: str = Field(description="가중치 라이선스")
    training_data: tuple[str, ...] = Field(description="직접 학습·미세조정 데이터와 그 라이선스")
    commercial: Commercial
    # 우리가 공개 데이터셋으로 1차 학습한 가중치만 적는다 (ADR 0060): 학습에 쓴 데이터셋 이름
    # (`config/external_datasets.yaml`)과 출발 가중치 이름. 분류는 `dlp_models.datasets`가 검사한다.
    training_datasets: tuple[str, ...] = ()
    base_model: str | None = None


class ExternalModel(Contract):
    """외부 추론 서버(VLM 등)가 가중치를 들고 있는 모델. 이 저장소는 라이선스 검사만 한다."""

    served_by: str = Field(description="어떤 서버로 쓰는지 (예: OpenAI 호환 VLM 서버)")
    license: str
    training_data: tuple[str, ...]
    commercial: Commercial


class ModelRegistry(Contract):
    """`config/models.yaml` 전체.

    - version: 파일 형식 버전.
    - accept: 정책이 쓸 수 있는 상업 사용 분류 (지금 `[allowed, review]`). 법무 확인 뒤 review를
      빼면 review 모델을 쓰는 정책이 `make licenses`에서 걸린다.
    - models: 이름 → 로컬 가중치 (`ModelFile`).
    - external: 이름 → 외부 서버 모델 (`ExternalModel`, 예: VLM).
    """

    version: int
    accept: tuple[Commercial, ...] = Field(description="정책이 쓸 수 있는 상업 사용 분류")
    models: dict[str, ModelFile]
    external: dict[str, ExternalModel] = Field(default_factory=dict[str, ExternalModel])

    def violations(self, used: dict[str, str]) -> list[str]:
        """정책이 쓰는 모델(쓰는 곳 → 모델 이름) 중 목록에 없거나 허용 분류가 아닌 것.

        Args:
            used: "쓰는 곳"(예: `prelabel.yaml models.hands`) → 모델 이름. `dlp_cli.models_cmds`가
                각 정책에서 모은다.

        Returns:
            사람이 읽는 위반 메시지 목록 (쓰는 곳 이름순). 비어 있으면 통과.
        """
        out: list[str] = []
        for where, name in sorted(used.items()):
            # 로컬 가중치와 외부 서버 모델을 같은 이름 공간에서 찾는다
            spec = self.models.get(name) or self.external.get(name)
            if spec is None:
                out.append(f"{where}: 모델 목록에 없는 {name}")
            elif spec.commercial not in self.accept:
                out.append(f"{where}: {name}은 상업 사용 분류가 {spec.commercial}")
        return out


def _fingerprint(stat: os.stat_result) -> tuple[int, ...]:
    """내용 변경·원자적 교체를 구별하는 파일 상태. 시각은 나노초 정수로 비교한다."""
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def _digest(path: Path, fingerprint: tuple[int, ...]) -> str:
    """파일 sha256 (16진 64자). 1 MiB씩 읽는다.

    경로·파일 식별자·크기·나노초 수정/변경 시각으로 프로세스 안에서 캐시한다.
    mtime을 보존한 파일 변경도 ctime으로 구별한다. 읽는 동안 변경되면 캐시하지 않고 거부한다.
    검증 뒤의 파일을 다른 프로세스가 수정하지 못하게 하는 운영 권한 설정은 별도로 필요하다.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        if _fingerprint(os.fstat(f.fileno())) != fingerprint:
            raise ModelUnavailableError(f"해시 검사 중 가중치 파일이 바뀌었습니다: {path}")
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
        if _fingerprint(os.fstat(f.fileno())) != fingerprint:
            raise ModelUnavailableError(f"해시 검사 중 가중치 파일이 바뀌었습니다: {path}")
    return h.hexdigest()


def _file_digest(path: Path) -> str:
    """캐시 조회 전후에도 현재 경로가 같은 파일 상태를 가리키는지 확인한다."""
    fingerprint = _fingerprint(path.stat())
    digest = _digest(path, fingerprint)
    if _fingerprint(path.stat()) != fingerprint:
        raise ModelUnavailableError(f"해시 검사 중 가중치 파일이 바뀌었습니다: {path}")
    return digest


def load_registry(root: Path) -> ModelRegistry:
    """`<root>/config/models.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        pydantic.ValidationError: 형식이 맞지 않을 때 (분류 값 오타 등).
    """
    data: Any = yaml.safe_load((root / "config" / "models.yaml").read_text(encoding="utf-8"))
    return ModelRegistry.model_validate(data)


def resolve(root: Path, name: str, *, check_hash: bool = True) -> tuple[Path, str]:
    """(파일 경로, 버전 문자열). ZIP 모델은 추출 파일의 별도 등록 해시로 검증한다.

    직접 변환한 파일은 환경에 따라 바이트가 달라 존재와 실제 버전 해시를 확인한다.

    Args:
        root: 저장소 루트 (`config/models.yaml`과 `data/models/`가 있는 곳).
        name: `models.yaml models`의 이름 (외부 모델은 대상이 아니다).
        check_hash: False면 직접 받은 파일도 등록 해시와 비교하지 않는다.

    Returns:
        (절대 경로, `"<이름>-<실제 파일 sha256 앞 12자>"`). 버전에는 등록 해시가 아니라 실제로
        쓴 파일의 해시를 넣는다.

    Raises:
        ModelUnavailableError: 목록에 없음 / 파일 없음(메시지에 `make models` 또는
            `make export-models` 안내) / 직접 받은 파일의 해시가 등록 해시와 다름.

    부작용: 파일을 끝까지 읽어 해시한다 (처음 한 번, 이후 `_digest` 캐시).
    """
    registry = load_registry(root)
    spec = registry.models.get(name)
    if spec is None:
        raise ModelUnavailableError(f"모델 목록에 {name}이 없습니다 (config/models.yaml)")
    path = root / spec.path
    if not path.is_file():
        hint = f"make export-models ({spec.export})" if spec.export else "make models"
        raise ModelUnavailableError(f"{name} 가중치가 없습니다: {path} ({hint})")
    if check_hash and spec.export is None:
        digest = _file_digest(path)
        expected = spec.artifact_sha256 if spec.archive_member else spec.sha256
        if expected is None or digest != expected:
            raise ModelUnavailableError(f"{name} 가중치 해시가 다릅니다: {digest}")
        return path, f"{name}-{digest[:12]}"
    # 직접 변환하거나 명시적으로 검증을 생략한 경우도 실제 파일의 해시를 버전에 남긴다.
    return path, f"{name}-{_file_digest(path)[:12]}"


def fetch(root: Path, name: str, spec: ModelFile) -> str:
    """가중치를 받는다. 결과 메시지를 돌려준다. 변환형은 안내만 한다.

    멱등: 파일이 이미 있고 해시가 맞으면 다시 받지 않는다. 실패한 다운로드는 공개하지 않는다.

    Args:
        root: 저장소 루트.
        name: 모델 이름 (메시지용).
        spec: `models.yaml`의 항목.

    Returns:
        `[있음]`, `[받음]`, `[변환 필요]`, `[실패] ... 해시 불일치` 중 하나로 시작하는 한 줄.
        해시 불일치는 예외가 아니라 메시지로 알리고 파일을 쓰지 않는다.

    부작용: 외부 HTTP 요청(시간 제한 600초), `root/spec.path`에 파일 쓰기(상위 디렉터리 생성).
    """
    path = root / spec.path
    if spec.export is not None:
        return (
            f"[변환 필요] {name}: {spec.export} (make export-models)"
            if not path.is_file()
            else f"[있음] {name}"
        )
    expected = spec.artifact_sha256 if spec.archive_member else spec.sha256
    if expected is None:
        return f"[실패] {name}: 압축 모델의 artifact_sha256이 없습니다"
    if path.is_file() and _file_digest(path) == expected:
        return f"[있음] {name}"
    # export도 url도 없는 항목은 테스트(test_registry_entries_have_source_and_license)가 막는다
    assert spec.url is not None
    with urllib.request.urlopen(spec.url, timeout=600) as resp:
        data = resp.read()
    # 받은 바이트(압축형은 zip 전체)를 등록 해시와 비교한 뒤에만 디스크에 쓴다
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec.sha256:
        return f"[실패] {name}: 해시 불일치 {digest}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.archive_member:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            # 경로 접두사는 배포판마다 다를 수 있지만 일치 파일은 정확히 하나여야 한다.
            members = [n for n in z.namelist() if n.endswith(spec.archive_member)]
            if len(members) != 1:
                return f"[실패] {name}: 압축 파일의 모델 항목이 유일하지 않습니다"
            data = z.read(members[0])
        if hashlib.sha256(data).hexdigest() != expected:
            return f"[실패] {name}: 추출 파일 해시 불일치"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        candidate = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            candidate.replace(path)
        finally:
            candidate.unlink(missing_ok=True)
    return f"[받음] {name}: {path}"
