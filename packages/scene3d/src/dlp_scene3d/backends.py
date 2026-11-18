"""복원·탐지·프레임 특징 백엔드 어댑터와 CPU stub (ADR 0066, 0067).

네 역할마다 프로토콜 하나와 실제 구현·stub을 둔다 (CLAUDE.md: 새 모델은 어댑터 + CPU stub).

- SfM `SfmBackend`: 실제 `IsolatedSfm`(pycolmap, BSD-3, 격리 환경 `scripts/gsplat-env`),
  stub `OracleSfm`(정답 카메라·점, 선택적 잡음·길이 단위 배율).
- 스플랫 `SplatTrainer`: 실제 `GsplatTrainer`(gsplat, Apache-2.0, GPU 격리 환경), stub
  `InitPointsSplat`(SfM 점으로 초기화한 가우시안을 표준 3DGS PLY로 쓴다, 학습 없음).
  Inria 원본 gaussian-splatting 코드는 비상업 라이선스라 쓰지 않는다 (models.yaml forbidden).
- 물체 탐지 `ObjectDetector`: 실제 `OwlObjectDetector`(OWLv2 ONNX, `dlp_models.owlv2`), stub
  `OracleObjectDetector`(정답 상자).
- 바디캠 프레임 특징 `FrameFeatureExtractor` (장면 안 위치 추정, ADR 0067): 실제
  `IsolatedFrameFeatures`(pycolmap SIFT, SfM과 같은 격리 환경), stub `OracleFrameFeatures`
  (정답 자세로 복원 점을 투영한 특징 + 화소 잡음 + 틀린 대응을 만드는 방해 특징). 대응·PnP·좌표
  변환은 백엔드가 아니라 작업공간 코드(`geometry`, `pnp`, `localize`)다.

격리 환경 어댑터는 `uv run --project scripts/gsplat-env python <스크립트>`를 하위 프로세스로 부르고
파일(JSON·NPZ·PLY·PNG)로만 주고받는다. 환경이 없거나 실패하면 `ModelUnavailableError`.
입력은 언제나 승인된 블러본이다 (사진은 `privacy_gate.open_blurred`, 영상 프레임은
`ego_video.open_blurred_video`).

화소 좌표 규약: 작업공간 안은 OpenCV 규약(왼쪽 위 화소의 **중심**이 (0, 0),
`dlp_media.calibration`과 같다)이고, COLMAP·gsplat은 왼쪽 위 화소의 중심이 (0.5, 0.5)다. 격리
환경 어댑터가 경계에서 주점과 특징 위치에 0.5를 빼고 더한다 (`IsolatedSfm`, `GsplatTrainer`,
`IsolatedFrameFeatures`).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_models.owlv2 import Owlv2
from dlp_scene3d.geometry import F64, Camera
from dlp_scene3d.privacy_gate import BlurredPhoto
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.scene3d import CameraPose

# COLMAP·gsplat 화소 좌표에서 왼쪽 위 화소 중심의 좌표 (작업공간 규약은 0)
COLMAP_PIXEL_CENTER = 0.5


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


@dataclass(frozen=True)
class FrameImage:
    """바디캠 프레임 한 장 (승인된 블러본에서 꺼낸 것). t_ms는 그 프레임의 PTS 정수 ms."""

    t_ms: int
    image: NDArray[np.uint8] = field(repr=False)


@dataclass(frozen=True)
class FrameFeatures:
    """프레임 한 장의 특징. keypoints (M, 2) 화소(작업공간 규약), descriptors (M, D).

    기술자는 복원 점 기술자(`SfmResult.descriptors`)와 같은 공간이어야 대응된다 (같은 추출기).
    """

    keypoints: F64
    descriptors: F64


class FrameFeatureExtractor(Protocol):
    """바디캠 프레임 → 특징 (2D 위치 + 기술자). 장면 안 위치 추정의 유일한 모델 부분."""

    name: str
    version: str

    def extract(self, frames: Sequence[FrameImage], workdir: Path) -> dict[int, FrameFeatures]:
        """프레임 묶음의 특징. 키는 프레임 시각(t_ms). 특징이 없는 프레임은 빈 배열 또는 키 없음."""
        ...


@dataclass
class OracleSfm:
    """정답 카메라·점을 그대로 내는 stub. noise는 점에 더할 잡음(장면 단위).

    unit_scale이 1이 아니면 점과 카메라 이동에 그 배율을 곱하고 metric=False로 낸다: 척도를 모르는
    실제 SfM을 흉내 내 마커 척도(ADR 0067)를 시험한다.
    """

    cameras: dict[str, Camera]
    points: F64
    descriptors: F64
    metric: bool
    noise: float = 0.0
    seed: int = 0
    unit_scale: float = 1.0
    name: str = "oracle"
    version: str = "oracle-sfm-1"

    def __post_init__(self) -> None:
        """배율을 곱하는 stub은 버전에 배율을 적는다 (같은 사진의 다른 복원으로 구분된다)."""
        if self.unit_scale != 1.0:
            self.version = f"{self.version}+u{self.unit_scale:g}"

    def reconstruct(self, photos: Sequence[BlurredPhoto], workdir: Path) -> SfmResult:
        """photos 중 정답이 있는 사진의 카메라만 낸다."""
        rng = np.random.default_rng(self.seed)
        s = self.unit_scale
        cams = {
            p.photo_id: dataclasses.replace(
                self.cameras[p.photo_id], t=self.cameras[p.photo_id].t * s
            )
            for p in photos
            if p.photo_id in self.cameras
        }
        pts = (
            self.points + rng.normal(scale=self.noise, size=self.points.shape)
            if self.noise
            else self.points
        )
        colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        return SfmResult(cams, pts * s, self.descriptors, colors, self.metric and s == 1.0)


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


def _shift_principal_point(cam: Camera, delta: float) -> Camera:
    """주점 (cx, cy)에 delta를 더한 카메라 (화소 좌표 규약을 바꾼다, 모듈 설명)."""
    k = cam.k.copy()
    k[0, 2] += delta
    k[1, 2] += delta
    return dataclasses.replace(cam, k=k)


@dataclass
class IsolatedSfm:
    """pycolmap SfM (격리 환경 `run_sfm.py`). 척도를 모르므로 metric=False.

    미터 척도는 복원 단계가 이 결과에 마커 척도를 곱해 정한다 (`markers`, ADR 0067). 스크립트가 낸
    카메라의 주점은 COLMAP 화소 규약이라 0.5를 빼 작업공간 규약으로 바꾼다.

    TODO(real-model): GPU·pycolmap 환경에서 실측 전이다. 3D 점 기술자는 트랙의 SIFT 기술자 평균이며
    before·after 정렬과 바디캠 프레임 대응은 이 기술자로 한다 (공동 복원 대안은 ADR 0066).
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
            {
                p.photo_id: _shift_principal_point(Camera.from_pose(p), -COLMAP_PIXEL_CENTER)
                for p in poses
            },
            np.asarray(npz["points"], dtype=np.float64),
            np.asarray(npz["descriptors"], dtype=np.float64),
            np.asarray(npz["colors"], dtype=np.uint8),
            metric=False,
        )


@dataclass
class GsplatTrainer:
    """gsplat 3DGS 학습 (격리 환경 `train_splat.py`, GPU).

    넘기는 카메라의 주점에는 0.5를 더한다 (gsplat은 COLMAP 화소 규약으로 그린다).

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
        poses = [
            _shift_principal_point(c, COLMAP_PIXEL_CENTER).to_pose().model_dump(mode="json")
            for c in sfm.cameras.values()
        ]
        (work / "cameras.json").write_text(json.dumps(poses), "utf-8")
        np.savez(work / "points.npz", points=sfm.points, colors=sfm.colors)
        args = ["--images", str(images), "--sfm", str(work), "--iterations", str(self.iterations)]
        _run_env(self.env, "train_splat.py", [*args, "--out", str(out)], self.timeout_s)


def _stage_frames(frames: Sequence[FrameImage], workdir: Path) -> Path:
    """격리 환경에 넘길 프레임 디렉터리 (`<t_ms>.png`, 앞선 묶음의 파일은 지운다)."""
    images = workdir / "frames"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True)
    for f in frames:
        ok = cv2.imwrite(str(images / f"{f.t_ms}.png"), cv2.cvtColor(f.image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ModelUnavailableError(f"프레임을 쓰지 못했습니다: {f.t_ms}")
    return images


@dataclass
class IsolatedFrameFeatures:
    """pycolmap SIFT 프레임 특징 (격리 환경 `extract_frame_features.py`).

    SfM(`run_sfm.py`)과 같은 추출 설정으로 뽑아야 복원 점 기술자(트랙의 SIFT 기술자 평균)와
    대응된다. 스크립트가 낸 특징 위치는 COLMAP 화소 규약이라 0.5를 빼 작업공간 규약으로 바꾼다.

    TODO(real-model): 운영 격리 환경(GPU 서버, 잠금 파일)과 실제 영상에서 실측 전이다. CPU판
    pycolmap으로 무늬가 있는 합성 방에서만 돌려 봤다 (scripts/gsplat-env/README.md). 실제 바디캠
    프레임의 기술자가 사진 SfM 점 기술자와 얼마나 대응되는지는 재지 않았다.
    """

    env: Path
    max_features: int
    timeout_s: int = 3600
    name: str = "pycolmap-sift"
    version: str = "pycolmap-sift-1"

    def extract(self, frames: Sequence[FrameImage], workdir: Path) -> dict[int, FrameFeatures]:
        """extract_frame_features.py --images DIR --out NPZ --max-features N → 프레임별 특징."""
        images = _stage_frames(frames, workdir)
        out = workdir / "frame_features.npz"
        args = [
            "--images",
            str(images),
            "--out",
            str(out),
            "--max-features",
            str(self.max_features),
        ]
        try:
            _run_env(self.env, "extract_frame_features.py", args, self.timeout_s)
            npz = np.load(out)
            result: dict[int, FrameFeatures] = {}
            for f in frames:
                if f"kp_{f.t_ms}" not in npz.files:
                    continue
                kp = np.asarray(npz[f"kp_{f.t_ms}"], dtype=np.float64).reshape(-1, 2)
                desc = np.asarray(npz[f"desc_{f.t_ms}"], dtype=np.float64)
                result[f.t_ms] = FrameFeatures(kp - COLMAP_PIXEL_CENTER, desc)
            return result
        finally:
            shutil.rmtree(images, ignore_errors=True)
            out.unlink(missing_ok=True)


@dataclass(frozen=True)
class OracleLayer:
    """`OracleFrameFeatures`가 투영할 장면 점 한 묶음과 그 점들이 그 모습으로 있는 시간.

    points (N, 3)는 before 장면 좌표(정답), descriptors (N, D). from_ms 이상 until_ms 미만의
    프레임에만 보인다 (until_ms None은 끝까지). 작업 중 옮긴 물체·닦인 벽은 전 상태와 후 상태가
    따로 묶음이다.
    """

    points: F64
    descriptors: F64
    from_ms: int = 0
    until_ms: int | None = None

    def active(self, t_ms: int) -> bool:
        """그 시각에 이 모습으로 있는가."""
        return self.from_ms <= t_ms and (self.until_ms is None or t_ms < self.until_ms)


@dataclass
class OracleFrameFeatures:
    """정답 카메라 자세로 장면 점을 투영해 프레임 특징을 만드는 stub (합성 픽스처·시험·`--oracle`).

    프레임 그림은 보지 않고 프레임 시각으로 정답 자세를 찾는다. 그 시각에 방에 있는 점(`layers`)을
    정답 내부 파라미터·왜곡으로 투영해 화면 안에 든 것만 특징으로 낸다. 실제 추출기의 흠을 흉내
    낸다:
    - 화소 잡음 `pixel_noise`(표준편차), 기술자 잡음 `descriptor_noise`.
    - 틀린 대응 `wrong_ratio`: 화면에 안 보이는 장면 점의 기술자를 무작위 화소에 둔다 (비슷하게 생긴
      다른 곳). 보이는 특징 수에 대한 비율이다. 기술자가 복원 점과 맞으므로 대응 검사를 통과해 PnP가
      걸러내야 한다.
    - 잡동사니 `clutter_ratio`: 어느 복원 점과도 무관한 무작위 기술자 (손·움직이는 것).
    정답 자세가 없는 시각의 프레임은 특징이 없다.

    TODO(real-model): 실제 프레임 특징은 `IsolatedFrameFeatures`(pycolmap SIFT)로 뽑는다.
    """

    k: F64
    dist: F64
    width: int
    height: int
    poses: Mapping[int, tuple[F64, F64]]
    layers: Sequence[OracleLayer]
    pixel_noise: float = 0.5
    descriptor_noise: float = 0.02
    wrong_ratio: float = 0.3
    clutter_ratio: float = 0.2
    seed: int = 0
    name: str = "oracle"
    version: str = "oracle-features-1"

    def _one(self, t_ms: int) -> FrameFeatures:
        """프레임 한 장의 특징."""
        active = [layer for layer in self.layers if layer.active(t_ms)]
        dim = self.layers[0].descriptors.shape[1] if self.layers else 0
        pose = self.poses.get(t_ms)
        if pose is None or not active:
            return FrameFeatures(np.zeros((0, 2)), np.zeros((0, dim)))
        rng = np.random.default_rng([self.seed, t_ms])
        points = np.concatenate([layer.points for layer in active])
        desc = np.concatenate([layer.descriptors for layer in active])
        r, t = pose
        rvec, _ = cv2.Rodrigues(r)
        proj, _ = cv2.projectPoints(points, rvec, t.reshape(3, 1), self.k, self.dist)
        uv = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        seen = (
            ((points @ r.T + t)[:, 2] > 0.05)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] <= self.width - 1)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] <= self.height - 1)
        )
        n = int(seen.sum())
        hidden = np.flatnonzero(~seen)
        wrong = rng.choice(hidden, min(len(hidden), round(self.wrong_ratio * n)), replace=False)
        n_clutter = round(self.clutter_ratio * n)
        n_fake = len(wrong) + n_clutter
        size = np.array([self.width - 1, self.height - 1], dtype=np.float64)
        keypoints = np.concatenate(
            [
                uv[seen] + rng.normal(scale=self.pixel_noise, size=(n, 2)),
                rng.uniform(0.0, 1.0, size=(n_fake, 2)) * size,
            ]
        )
        descriptors = np.concatenate(
            [desc[seen], desc[wrong], rng.normal(size=(n_clutter, dim))]
        ) + rng.normal(scale=self.descriptor_noise, size=(n + n_fake, dim))
        order = rng.permutation(len(keypoints))
        return FrameFeatures(keypoints[order], descriptors[order])

    def extract(self, frames: Sequence[FrameImage], workdir: Path) -> dict[int, FrameFeatures]:
        """프레임 시각마다 `_one`."""
        return {f.t_ms: self._one(f.t_ms) for f in frames}


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
