"""단안 메트릭 깊이로 바디캠의 손 관절·객체를 카메라 좌표 3D 궤적으로 올린다.

frame_stride_ms마다 깊이 맵을 만들고, 그 프레임에 있는 손 키포인트(정책의 hand_points)와 객체 박스
중심의 깊이(주변 패치 중앙값)를 읽어 카메라 내부 파라미터로 역투영한다. 좌표계는 카메라
(광축 Z 앞, X 오른쪽, Y 아래, 미터)다. 월드 좌표는 카메라 자세(SLAM)가 생긴 뒤에 만든다.
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
from dlp_prelabel.policy import DepthPolicy, PrelabelPolicy
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
HAND21_NAMES = {
    0: "wrist", 4: "thumb_tip", 8: "index_tip", 12: "middle_tip", 16: "ring_tip", 20: "pinky_tip",
}  # fmt: skip


class DepthModel(Protocol):
    def predict(self, image: Image) -> NDArray[np.float32]: ...


def intrinsics_for(
    calib: CameraIntrinsics | None, width: int, height: int, hfov_deg: float
) -> Intrinsics:
    """세션 캘리브레이션(해상도가 다르면 비례 조정) 또는 기본 화각 근사."""
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
    """(entity_id, part) → 3D 샘플. tracks는 hand21 키포인트 트랙과 박스 트랙."""
    points: dict[int, list[tuple[str, str | None, float, float]]] = {}
    for label in tracks:
        p = label.payload
        if isinstance(p, KeypointTrackPayload) and p.skeleton == "hand21":
            for f in p.keyframes:
                for i in policy.hand_points:
                    kp = f.points[i]
                    if kp.visibility > 0:
                        name = HAND21_NAMES.get(i, f"joint_{i}")
                        points.setdefault(f.t_ms, []).append((p.entity_id, name, kp.x, kp.y))
        elif isinstance(p, BoxTrackPayload):
            for k in p.keyframes:
                if not k.outside:
                    cx, cy = k.x + k.w / 2, k.y + k.h / 2
                    points.setdefault(k.t_ms, []).append((p.entity_id, None, cx, cy))
    out: dict[tuple[str, str | None], list[Trajectory3DSample]] = {}
    last: int | None = None
    intr: Intrinsics | None = None
    for t, img in frames:
        if t not in points or (last is not None and t - last < policy.frame_stride_ms):
            continue
        last = t
        if intr is None:
            intr = intrinsics_for(calib, img.shape[1], img.shape[0], policy.default_hfov_deg)
        dmap = depth.predict(img)
        for entity, part, u, v in points[t]:
            z = sample_depth(dmap, u, v, policy.patch_radius_px)
            if math.isnan(z) or not policy.min_depth_m <= z <= policy.max_depth_m:
                continue
            x, y, z = intr.unproject(u, v, z)
            out.setdefault((entity, part), []).append(
                Trajectory3DSample(t_ms=t, x=round(x, 4), y=round(y, 4), z=round(z, 4))
            )
    return out


class DepthLifter:
    name = "depth3d"

    def __init__(self, root: Path, policy: PrelabelPolicy, *, now: datetime) -> None:
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
        """version: 라벨에 쓸 모델 버전 (기본: self.version). 러너가 입력 해시를 붙여 넘긴다."""
        model_version = version or self.version
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
