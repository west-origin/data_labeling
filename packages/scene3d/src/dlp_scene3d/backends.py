"""복원·탐지 백엔드 어댑터와 CPU stub (ADR 0066).

세 역할마다 프로토콜 하나와 실제 구현·stub을 둔다 (CLAUDE.md: 새 모델은 어댑터 + CPU stub).

- SfM `SfmBackend`: 실제 `IsolatedSfm`(pycolmap, BSD-3, 격리 환경 `scripts/gsplat-env`),
  stub `OracleSfm`(정답 카메라·점, 선택적 잡음).
- 스플랫 `SplatTrainer`: 실제 `GsplatTrainer`(gsplat, Apache-2.0, GPU 격리 환경), stub
  `InitPointsSplat`(SfM 점으로 초기화한 가우시안을 표준 3DGS PLY로 쓴다, 학습 없음).
  Inria 원본 gaussian-splatting 코드는 비상업 라이선스라 쓰지 않는다 (models.yaml forbidden).
- 물체 탐지 `ObjectDetector`: 실제 `OwlObjectDetector`(OWLv2 ONNX, `dlp_models.owlv2`), stub
  `OracleObjectDetector`(정답 상자).

격리 환경 어댑터는 `uv run --project scripts/gsplat-env python <스크립트>`를 하위 프로세스로 부르고
파일(JSON·NPZ·PLY)로만 주고받는다. 환경이 없거나 실패하면 `ModelUnavailableError`.
입력은 언제나 `privacy_gate.open_blurred`가 준 승인된 블러 사진이다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_models.owlv2 import Owlv2
from dlp_scene3d.geometry import F64, Camera
from dlp_scene3d.privacy_gate import BlurredPhoto
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.scene3d import CameraPose


@dataclass(frozen=True)
class SfmResult:
    """SfM 결과. points (N, 3), descriptors (N, D) (정렬 대응용), colors (N, 3) uint8."""

    cameras: dict[str, Camera]
    points: F64
    descriptors: F64
    colors: NDArray[np.uint8]
    metric: bool


class SfmBackend(Protocol):
    """사진 묶음 → 카메라 자세·희소 점."""

    name: str
    version: str

    def reconstruct(self, photos: Sequence[BlurredPhoto], workdir: Path) -> SfmResult:
        """블러 사진으로 복원한다 (자세를 못 찾은 사진은 cameras에 없다)."""
        ...


class SplatTrainer(Protocol):
    """SfM 결과 → 3D 가우시안 스플랫 PLY 파일."""

    name: str
    version: str

    def train(self, photos: Sequence[BlurredPhoto], sfm: SfmResult, out: Path) -> None:
        """out에 PLY를 쓴다."""
        ...


class ObjectDetector(Protocol):
    """사진 한 장의 물체 탐지: (라벨, (x, y, w, h), 점수)."""

    name: str
    version: str

    def detect(
        self, photo_id: str, image: NDArray[np.uint8]
    ) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """문턱 이상 탐지만 낸다."""
        ...


@dataclass
class OracleSfm:
    """정답 카메라·점을 그대로 내는 stub. noise_px는 점·카메라 중심에 더할 잡음(장면 단위)."""

    cameras: dict[str, Camera]
    points: F64
    descriptors: F64
    metric: bool
    noise: float = 0.0
    seed: int = 0
    name: str = "oracle"
    version: str = "oracle-sfm-1"

    def reconstruct(self, photos: Sequence[BlurredPhoto], workdir: Path) -> SfmResult:
        """photos 중 정답이 있는 사진의 카메라만 낸다."""
        rng = np.random.default_rng(self.seed)
        cams = {p.photo_id: self.cameras[p.photo_id] for p in photos if p.photo_id in self.cameras}
        pts = (
            self.points + rng.normal(scale=self.noise, size=self.points.shape)
            if self.noise
            else self.points
        )
        colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        return SfmResult(cams, pts, self.descriptors, colors, self.metric)


def _sh0(colors: NDArray[np.uint8]) -> F64:
    """RGB(0~255) → 0차 구면조화 계수 (3DGS 규약 f_dc = (c - 0.5) / C0)."""
    return (colors.astype(np.float64) / 255.0 - 0.5) / 0.28209479177387814


@dataclass
class InitPointsSplat:
    """학습 없이 SfM 점마다 등방 가우시안 하나를 둔 표준 3DGS PLY (CPU stub).

    TODO(real-model): 실제 스플랫은 `GsplatTrainer`(GPU)로 학습한다. 이 stub은 뷰어 확인·하류
    시험용이며 공간 정보는 SfM 점 이상을 담지 않는다.
    """

    init_scale: float
    name: str = "init_points"
    version: str = "init-points-1"

    def train(self, photos: Sequence[BlurredPhoto], sfm: SfmResult, out: Path) -> None:
        """binary_little_endian PLY (x y z nx ny nz f_dc_0..2 opacity scale_0..2 rot_0..3)."""
        n = len(sfm.points)
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
        names += ["scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.zeros((n, len(names)), dtype=np.float32)
        data[:, 0:3] = sfm.points
        data[:, 6:9] = _sh0(sfm.colors)
        data[:, 9] = 2.197  # logit(0.9)
        data[:, 10:13] = np.log(self.init_scale)
        data[:, 13] = 1.0
        header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
        header += "".join(f"property float {k}\n" for k in names) + "end_header\n"
        out.write_bytes(header.encode() + data.astype("<f4").tobytes())


def _run_env(env: Path, script: str, args: list[str], timeout_s: int) -> None:
    """격리 환경에서 스크립트를 돌린다. uv나 환경이 없거나 실패하면 ModelUnavailableError."""
    uv = shutil.which("uv")
    if uv is None or not (env / "pyproject.toml").exists():
        raise ModelUnavailableError(f"격리 환경이 없습니다: {env} (scripts/gsplat-env/README.md)")
    cmd = [uv, "run", "--project", str(env), "python", str(env / script), *args]
    try:
        subprocess.run(cmd, check=True, timeout=timeout_s, capture_output=True, text=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ModelUnavailableError(f"{script} 실패: {exc}") from exc


def _stage_photos(photos: Sequence[BlurredPhoto], workdir: Path) -> Path:
    """격리 환경에 넘길 블러 사진 디렉터리 (photo_id.png)."""
    images = workdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    for p in photos:
        shutil.copyfile(p.path, images / f"{p.photo_id}.png")
    return images


@dataclass
class IsolatedSfm:
    """pycolmap SfM (격리 환경 `run_sfm.py`). 척도를 모르므로 metric=False.

    TODO(real-model): GPU·pycolmap 환경에서 실측 전이다. 3D 점 기술자는 트랙의 SIFT 기술자 평균이며
    before·after 정렬은 이 기술자 대응 또는 공동 복원(ADR 0066)으로 한다.
    척도 기준 마커 연동도 남았다.
    """

    env: Path
    timeout_s: int = 3600
    name: str = "pycolmap"
    version: str = "pycolmap-1"

    def reconstruct(self, photos: Sequence[BlurredPhoto], workdir: Path) -> SfmResult:
        """run_sfm.py --images DIR --out DIR → cameras.json, points.npz."""
        images = _stage_photos(photos, workdir)
        out = workdir / "sfm"
        _run_env(
            self.env, "run_sfm.py", ["--images", str(images), "--out", str(out)], self.timeout_s
        )
        poses = [
            CameraPose.model_validate(p)
            for p in json.loads((out / "cameras.json").read_text("utf-8"))
        ]
        npz = np.load(out / "points.npz")
        return SfmResult(
            {p.photo_id: Camera.from_pose(p) for p in poses},
            np.asarray(npz["points"], dtype=np.float64),
            np.asarray(npz["descriptors"], dtype=np.float64),
            np.asarray(npz["colors"], dtype=np.uint8),
            metric=False,
        )


@dataclass
class GsplatTrainer:
    """gsplat 3DGS 학습 (격리 환경 `train_splat.py`, GPU).

    TODO(real-model): GPU 서버에서 실측 전이다 (scripts/gsplat-env/README.md).
    """

    env: Path
    iterations: int
    timeout_s: int = 7200
    name: str = "gsplat"
    version: str = "gsplat-1"

    def train(self, photos: Sequence[BlurredPhoto], sfm: SfmResult, out: Path) -> None:
        """SfM 결과를 파일로 넘기고 학습한 PLY를 out에 받는다."""
        work = out.parent / "gsplat"
        images = _stage_photos(photos, work)
        (work / "cameras.json").write_text(
            json.dumps([c.to_pose().model_dump(mode="json") for c in sfm.cameras.values()]), "utf-8"
        )
        np.savez(work / "points.npz", points=sfm.points, colors=sfm.colors)
        args = ["--images", str(images), "--sfm", str(work), "--iterations", str(self.iterations)]
        _run_env(self.env, "train_splat.py", [*args, "--out", str(out)], self.timeout_s)


@dataclass
class OracleObjectDetector:
    """정답 상자를 그대로 내는 stub. boxes: photo_id → (라벨, 상자)."""

    boxes: Mapping[str, list[tuple[str, tuple[float, float, float, float]]]]
    score: float = 0.9
    name: str = "oracle"
    version: str = "oracle-objects-1"

    def detect(
        self, photo_id: str, image: NDArray[np.uint8]
    ) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """그 사진의 정답 상자."""
        return [(lb, b, self.score) for lb, b in self.boxes.get(photo_id, [])]


@dataclass
class OwlObjectDetector:
    """OWLv2 오픈 보캐뷸러리 물체 탐지 (CPU ONNX). queries: 라벨 → 영어 질의."""

    model: Path
    tokenizer: Path
    model_version: str
    queries: Mapping[str, str]
    threshold: float
    nms_iou: float
    name: str = "owlv2"
    version: str = field(init=False)
    _owl: Owlv2 | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """버전 = 가중치 버전 + 질의 수 (질의 내용은 비교 버전의 objects 절 해시에 들어간다)."""
        self.version = f"{self.model_version}+q{len(self.queries)}"

    def detect(
        self, photo_id: str, image: NDArray[np.uint8]
    ) -> list[tuple[str, tuple[float, float, float, float], float]]:
        """질의마다 같은 문턱으로 탐지한다."""
        labels = sorted(self.queries)
        if self._owl is None:
            self._owl = Owlv2(
                self.model, self.tokenizer, [self.queries[k] for k in labels], nms_iou=self.nms_iou
            )
        dets = self._owl.detect(image, [self.threshold] * len(labels))
        return [(labels[d.query_index], d.box, d.score) for d in dets]
