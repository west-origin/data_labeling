"""카메라 좌표 3D 궤적 → 세션 월드 좌표 궤적 (ADR 0046, F-PAR-26).

`campose`의 카메라 자세(R_wc, p)로 카메라 좌표 표본 X_c를 X_w = R_wc · X_c + p로 바꾼다 (R_wc는
카메라 → 월드 회전, p는 월드 좌표 카메라 중심, 미터. 규약은 `posemath`). 자세는 표본 시각 양쪽 자세
표본 사이를 보간한다 (회전 구면 보간, 위치 선형). 양쪽 간격이 `world_frame.max_pose_gap_ms`보다
크거나 자세 범위 밖이면 그 표본은 바꾸지 않는다 (외삽하지 않는다: 그 사이 머리가 얼마나 움직였는지
모르므로 지어내지 않는다). 시각은 둘 다 바디캠 PTS ms라 맞춰 볼 필요가 없다 (ADR 0019).

결과는 같은 (개체, 부위)의 새 trajectory3d (frame=world, source_3d=원래 값, derived_from=[카메라
좌표 궤적 ID, 자세 ID]). 표본 방식(method)은 원래 값, 신뢰도 = 원래 표본 신뢰도(없으면 1) x 보간한
자세 신뢰도(두 표본 중 낮은 쪽). 원래 카메라 좌표 궤적은 그대로 남는다 (덮어쓰지 않는다).

파이프라인 위치: `runner._world`가 현재 카메라 자세 하나와 바디캠의 현재 카메라 좌표 궤적((개체,
부위)마다 하나)으로 부른다. 정책: `prelabel.yaml world_frame`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from dlp_prelabel.campose import Mat3, PoseSample, Vec3
from dlp_prelabel.common import fit_label_id, model_label
from dlp_prelabel.lift3d import mean_confidence
from dlp_prelabel.posemath import slerp
from dlp_schema.episode import version_tag
from dlp_schema.labels import CoordinateFrame, LabelRecord, Trajectory3DPayload, Trajectory3DSample


def pose_at(
    poses: Sequence[PoseSample], t_ms: int, max_gap_ms: int
) -> tuple[Mat3, Vec3, float] | None:
    """시각 t의 (R_wc, p, 신뢰도). 같은 시각 표본이 있으면 그 값, 두 표본 사이면 보간, 아니면 None.

    poses는 시각 순이어야 한다.
    """
    if not poses:
        return None
    times = [p.t_ms for p in poses]
    # i: t_ms 이상인 첫 자세 표본 자리 (이분 탐색)
    i = int(np.searchsorted(times, t_ms))
    if i < len(times) and times[i] == t_ms:
        s = poses[i]
        return s.rotation, s.position, s.confidence
    if i == 0 or i == len(times):
        return None
    a, b = poses[i - 1], poses[i]
    # 양쪽 자세가 너무 멀면(추적 실패로 비었거나 자세 간격이 김) 보간하지 않는다
    if b.t_ms - a.t_ms > max_gap_ms:
        return None
    w = (t_ms - a.t_ms) / (b.t_ms - a.t_ms)
    r = slerp(a.rotation, b.rotation, w)
    p = a.position * (1 - w) + b.position * w
    # 보간한 자세는 두 표본 중 덜 믿는 쪽만큼만 믿는다
    return r, p, min(a.confidence, b.confidence)


def to_world_samples(
    samples: Sequence[Trajectory3DSample], poses: Sequence[PoseSample], max_gap_ms: int
) -> list[Trajectory3DSample]:
    """카메라 좌표 표본 → 월드 좌표 표본 (자세가 없는 시각은 빠진다, 모듈 설명)."""
    out: list[Trajectory3DSample] = []
    for s in sorted(samples, key=lambda q: q.t_ms):
        got = pose_at(poses, s.t_ms, max_gap_ms)
        if got is None:
            continue
        r, p, pose_conf = got
        # X_w = R_wc · X_c + p (카메라 좌표 점을 돌린 뒤 카메라 중심만큼 옮긴다)
        xyz = r @ np.array([s.x, s.y, s.z]) + p
        # 두 오차가 겹치므로 신뢰도를 곱한다 (카메라 좌표 3D 표본 x 그 시각 자세)
        conf = (1.0 if s.confidence is None else s.confidence) * pose_conf
        out.append(
            Trajectory3DSample(
                t_ms=s.t_ms,
                x=round(float(xyz[0]), 4),
                y=round(float(xyz[1]), 4),
                z=round(float(xyz[2]), 4),
                method=s.method,
                confidence=round(conf, 3),
                # 물리 제약 표시는 표본의 성질이라 좌표계를 바꿔도 남긴다 (F-PAR-78)
                flags=s.flags,
            )
        )
    return out


def world_label(
    camera: LabelRecord,
    pose: LabelRecord,
    poses: Sequence[PoseSample],
    *,
    version: str,
    max_gap_ms: int,
    now: datetime,
) -> LabelRecord | None:
    """카메라 좌표 궤적 라벨 하나 → 월드 좌표 궤적 라벨 (바꾼 표본이 없으면 None).

    ID: `<세션>-<스트림>-world3d-<version_tag>-<개체>-<부위|center>` (128자를 넘으면
    `common.fit_label_id`로 접두사 뒤를 해시로 줄인다).

    Raises:
        ValueError: camera가 카메라 좌표 trajectory3d가 아닐 때.
    """
    p = camera.payload
    if not isinstance(p, Trajectory3DPayload) or p.frame is not CoordinateFrame.CAMERA:
        raise ValueError(f"{camera.label_id}: 카메라 좌표 3D 궤적이 아닙니다")
    samples = to_world_samples(p.samples, poses, max_gap_ms)
    if not samples:
        return None
    payload = Trajectory3DPayload(
        entity_id=p.entity_id,
        part=p.part,
        frame=CoordinateFrame.WORLD,
        source_3d=p.source_3d,
        samples=tuple(samples),
        intrinsics=p.intrinsics,
        derived_from=(camera.label_id, pose.label_id),
    )
    stream = camera.stream_id or ""
    return model_label(
        label_id=fit_label_id(
            f"{camera.session_id}-{stream}-world3d-",
            f"{version_tag(version)}-{p.entity_id}-{p.part or 'center'}",
        ),
        session_id=camera.session_id,
        stream_id=stream,
        t_start_ms=samples[0].t_ms,
        t_end_ms=samples[-1].t_ms,
        ontology_version=camera.ontology_version,
        model_version=version,
        confidence=mean_confidence(samples),
        payload=payload,
        now=now,
    )
