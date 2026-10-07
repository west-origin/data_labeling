"""단안 메트릭 깊이로 바디캠의 손 관절·객체를 카메라 좌표 3D 궤적으로 올린다.

frame_stride_ms마다 깊이 맵을 만들고, 그 프레임에 있는 손 키포인트(정책의 hand_points)와 객체 박스
중심의 깊이(주변 패치 중앙값)를 읽어 카메라 내부 파라미터로 역투영한다. 좌표계는 카메라 (광축 Z 앞,
X 오른쪽, Y 아래, 미터)다. 월드 좌표는 카메라 자세(SLAM)가 생긴 뒤에 만든다.

파이프라인 위치: `runner._lift`가 바디캠의 현재 손·박스 트랙을 넘겨 부른다. 결과는 trajectory3d
라벨 (frame=camera, source_3d=mono_depth)이고, 관계 단계(`dlp_relations`)의 도구-표면 접촉이
쓴다. 정책: `prelabel.yaml depth`, 모델: `config/models.yaml depth_metric_indoor_small` (ADR
0010: review). 시간: 3D 샘플 시각은 바디캠 스트림 PTS ms다 (공간 라벨 규약, ADR 0019).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_models.depth import Intrinsics, MetricDepth, sample_depth
from dlp_models.registry import resolve
from dlp_prelabel.common import Image, iter_frames, model_label
from dlp_prelabel.policy import HAND21_JOINTS, DepthPolicy, PrelabelPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    CoordinateFrame,
    KeypointTrackPayload,
    LabelRecord,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
)
from dlp_schema.session import CameraIntrinsics

# hand21 관절 번호 → 이름 (MediaPipe 손 모델 순서)
# policy.HAND21_JOINTS의 별칭 (이 모듈 안에서 읽기 쉽게)
HAND21_NAMES = HAND21_JOINTS


class DepthModel(Protocol):
    """깊이 추론기 인터페이스 (`dlp_models.depth.MetricDepth`, 테스트의 가짜 모델)."""

    def predict(self, image: Image) -> NDArray[np.float32]: ...


def intrinsics_for(
    calib: CameraIntrinsics | None, width: int, height: int, hfov_deg: float
) -> Intrinsics:
    """세션 캘리브레이션(해상도가 다르면 비례 조정) 또는 기본 화각 근사.

    Args:
        calib: 세션 캘리브레이션 내부 파라미터 (없으면 None).
        width, height: 실제 디코드한 프레임 크기(px). 캘리브레이션 해상도와 다르면 fx·cx는 가로
            비율, fy·cy는 세로 비율로 맞춘다 (프록시·다운스케일 영상 대응).
        hfov_deg: calib가 없을 때 쓸 가로 화각 (`depth.default_hfov_deg`).
    """
    if calib is None:
        return Intrinsics.from_hfov(width, height, hfov_deg)
    sx, sy = width / calib.width, height / calib.height
    return Intrinsics(calib.fx * sx, calib.fy * sy, calib.cx * sx, calib.cy * sy)


def lift_tracks(
    frames: Iterable[tuple[int, Image]],
    depth: DepthModel,
    tracks: list[LabelRecord],
    calib: CameraIntrinsics | None,
    policy: DepthPolicy,
) -> dict[tuple[str, str | None], list[Trajectory3DSample]]:
    """(entity_id, part) → 3D 샘플. tracks는 hand21 키포인트 트랙과 박스 트랙.

    hand21 트랙은 `policy.hand_points` 관절마다 part=관절 이름(예: "wrist"), 박스 트랙은 화면 밖이
    아닌 키프레임의 박스 중심을 part=None으로 올린다. 깊이는 키프레임 시각과 PTS가 정확히 같은
    프레임에서만 읽고, 마지막으로 깊이를 만든 프레임에서 frame_stride_ms가 지나기 전 프레임은
    건너뛴다. 깊이가 nan(화면 밖)이거나 [min_depth_m, max_depth_m] 밖이면 그 점은 버린다.

    Args:
        frames: (스트림 PTS ms, RGB 프레임) 순회 (`common.iter_frames`).
        depth: 깊이 추론기 (프레임마다 미터 깊이 맵).
        tracks: 바디캠 현재 라벨 중 키포인트·박스 트랙 (다른 종류는 무시).
        calib: 세션 카메라 내부 파라미터 (None이면 화각 근사). 첫 처리 프레임 크기로 한 번 정한다.
        policy: `prelabel.yaml depth`.

    Returns:
        (entity_id, part) → 시각 순 `Trajectory3DSample` (카메라 좌표 m, 소수 넷째 자리).
    """
    # 1단계: 스트림 PTS ms → 그 시각에 올릴 점 (개체, 부위, u, v 픽셀) 목록
    points: dict[int, list[tuple[str, str | None, float, float]]] = {}
    for label in tracks:
        p = label.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "hand21":
            for f in p.keyframes:
                for i in policy.hand_points:
                    kp = f.points[i]
                    # visibility 0(라벨 없음)인 관절은 올리지 않는다
                    if kp.visibility > 0:
                        name = HAND21_NAMES[i]
                        points.setdefault(f.t_ms, []).append((p.entity_id, name, kp.x, kp.y))
        elif isinstance(p, BoxTrackPayload):
            for k in p.keyframes:
                if not k.outside:
                    # 박스는 중심 한 점만 올린다 (part=None)
                    cx, cy = k.x + k.w / 2, k.y + k.h / 2
                    points.setdefault(k.t_ms, []).append((p.entity_id, None, cx, cy))
    out: dict[tuple[str, str | None], list[Trajectory3DSample]] = {}
    last: int | None = None
    intr: Intrinsics | None = None
    for t, img in frames:
        # 2단계: 점이 있는 프레임 중 frame_stride_ms 간격으로만 깊이를 만든다 (CPU 비용)
        if t not in points or (last is not None and t - last < policy.frame_stride_ms):
            continue
        last = t
        if intr is None:
            # 내부 파라미터는 첫 처리 프레임의 크기로 한 번 정한다 (영상 중간에 해상도가 바뀌지
            # 않는다고 본다)
            intr = intrinsics_for(calib, img.shape[1], img.shape[0], policy.default_hfov_deg)
        dmap = depth.predict(img)
        for entity, part, u, v in points[t]:
            z = sample_depth(dmap, u, v, policy.patch_radius_px)
            # 화면 밖(nan)이나 믿기 어려운 깊이는 버린다 (그 시각 샘플이 빠진다)
            if math.isnan(z) or not policy.min_depth_m <= z <= policy.max_depth_m:
                continue
            x, y, z = intr.unproject(u, v, z)
            out.setdefault((entity, part), []).append(
                Trajectory3DSample(t_ms=t, x=round(x, 4), y=round(y, 4), z=round(z, 4))
            )
    return out


class DepthLifter:
    """메트릭 깊이 3D 단계. 러너가 Predictor 목록과 따로 부른다 (입력이 영상 + 다른 라벨이다).

    name: "depth3d". version: `depth3d-<가중치 버전>+p<depth 절 해시>` (러너가 입력 해시를 더
    붙인다).
    """

    name = "depth3d"

    def __init__(self, root: Path, policy: PrelabelPolicy, *, now: datetime) -> None:
        """가중치를 확인하고 버전을 정한다. 모델 세션은 처음 `run`할 때 연다.

        Raises:
            ModelUnavailableError: 깊이 가중치가 없을 때 (`make export-models`). CLI는 이 단계를
                건너뛴다.
        """
        self.path, version = resolve(root, policy.models.depth)
        self.version = f"depth3d-{version}+p{policy.digest('depth')}"
        self.policy, self.now = policy, now
        self.model: DepthModel | None = None

    def run(
        self,
        video: Path,
        *,
        session_id: str,
        stream_id: str,
        tracks: list[LabelRecord],
        calib: CameraIntrinsics | None,
        ontology_version: str,
        version: str | None = None,
    ) -> list[LabelRecord]:
        """version: 라벨에 쓸 모델 버전 (기본: self.version). 러너가 입력 해시를 붙여 넘긴다.

        Args:
            video: 바디캠 영상 로컬 경로.
            session_id, stream_id: 라벨의 세션·스트림 (바디캠).
            tracks: 올릴 손·박스 트랙 (`lift_tracks` 참고).
            calib: 세션 카메라 내부 파라미터.
            ontology_version: 라벨에 넣을 온톨로지 버전.

        Returns:
            (개체, 부위)마다 trajectory3d 라벨 하나. ID는
                `<세션>-<스트림>-3d-<version_tag>-<개체>-<부위|center>`.
            신뢰도는 `depth.confidence` 고정. DB에는 쓰지 않는다 (러너가 쓴다).
        """
        # 러너는 입력 해시를 붙인 버전을 넘긴다 (라벨 ID의 version_tag도 이 버전으로 만든다)
        model_version = version or self.version
        # ONNX 세션은 처음 쓸 때 한 번만 연다 (가중치가 커서 생성 비용이 크다)
        if self.model is None:
            self.model = MetricDepth(self.path)
        samples = lift_tracks(iter_frames(video), self.model, tracks, calib, self.policy.depth)
        out: list[LabelRecord] = []
        for (entity, part), ss in sorted(
            samples.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
        ):
            payload = Trajectory3DPayload(
                entity_id=entity,
                part=part,
                frame=CoordinateFrame.CAMERA,
                source_3d=Source3D.MONO_DEPTH,
                samples=tuple(ss),
            )
            tag = version_tag(model_version)
            out.append(
                model_label(
                    label_id=f"{session_id}-{stream_id}-3d-{tag}-{entity}-{part or 'center'}",
                    session_id=session_id,
                    stream_id=stream_id,
                    t_start_ms=ss[0].t_ms,
                    t_end_ms=ss[-1].t_ms,
                    ontology_version=ontology_version,
                    model_version=model_version,
                    confidence=self.policy.depth.confidence,
                    payload=payload,
                    now=self.now,
                )
            )
        return out
