"""단안 메트릭 깊이로 바디캠의 손 관절·객체를 카메라 좌표 3D 궤적으로 올린다.

frame_stride_ms마다 깊이 맵을 만들고, 그 프레임에 있는 손 키포인트(정책의 hand_points: 손목과 다섯
손가락 끝)와 객체 박스 중심을 카메라 좌표로 올린다. 좌표계는 카메라 (광축 Z 앞, X 오른쪽, Y 아래,
미터)다. 이 단계는 카메라 좌표만 만든다. 세션 월드 좌표 궤적은 바디캠 자세(`campose`)와
`worldframe` 단계가 이 결과를 derived_from으로 받아 따로 만든다 (ADR 0046, F-PAR-45).

- 손 키프레임에 손 중심 3D 점(`KeypointFrame.world`, MediaPipe hand_world_landmarks)이 있으면 손
  모양 + 2D 픽셀 + 손바닥 깊이·손 크기 사전값으로 21관절을 함께 맞춘다 (`handfit.fit_hand`,
  ADR 0044). 얇은 손가락 끝에서 배경 깊이를 읽는 문제가 없다.
- world가 없는 키프레임(사람이 고치거나 새로 그린 키프레임, 예전 버전. 사람 수정 레코드도 2D가
  그대로인 키프레임은 parent의 world를 지닌다)이나 맞춤이 실패한 키프레임은 손바닥 관절
  (`hand_fit.anchor_points`, 지금은 손목)만 그 픽셀의 깊이(주변 패치 중앙값)로 역투영한다. 손가락
  끝은 픽셀 깊이가 배경을 읽어 수백 mm 틀리므로 내지 않는다 (`depth.fingertip_fallback: drop`,
  F-PAR-40. point_depth면 예전처럼 올린다). 박스 중심도 픽셀 깊이 방식이다.
- 내부 파라미터: 세션·등록 캘리브레이션을 프레임 크기에 비례 조정한다. 화면비가 다르면(다른 촬영
  모드) 그 캘리브레이션을 쓰지 않고 화각 근사로 떨어지며 표본·궤적 출처도 hfov_approx다
  (`intrinsics_for`, F-PAR-44).

파이프라인 위치: `runner._lift`가 바디캠의 현재 hand21 손·박스 트랙을 (개체, 트랙 종류)마다 하나씩
골라(`runner.pick_lift_tracks`) 넘겨 부른다. 결과는 trajectory3d 라벨 (frame=camera,
source_3d=mono_depth)이고, 관계 단계(`dlp_relations`)의 도구-표면 접촉이 쓴다. 정책:
`prelabel.yaml depth`, 모델: `config/models.yaml depth_metric_indoor_small` (ADR 0010: review).
시간: 3D 샘플 시각은 바디캠 스트림 PTS ms다 (공간 라벨 규약, ADR 0019).
"""

from __future__ import annotations

import math
from collections.abc import Container, Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_models.depth import Intrinsics, MetricDepth, sample_depth
from dlp_models.registry import resolve
from dlp_prelabel.common import Image, iter_frames, model_label
from dlp_prelabel.handfit import HandFit, fit_confidence, fit_hand, point_confidence
from dlp_prelabel.policy import HAND21_JOINTS, DepthPolicy, PrelabelPolicy
from dlp_schema.episode import version_tag
from dlp_schema.labels import (
    BoxTrackPayload,
    CoordinateFrame,
    IntrinsicsSource,
    KeypointFrame,
    KeypointTrackPayload,
    LabelPayload,
    LabelRecord,
    Lift3DMethod,
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

    def predict(self, image: Image) -> NDArray[np.float32]:
        """(H, W, 3) RGB 프레임 → 같은 크기 (H, W) 깊이 맵 (미터)."""
        ...


def calibration_fits(
    calib: CameraIntrinsics, width: int, height: int, max_aspect_mismatch: float
) -> bool:
    """캘리브레이션을 이 프레임 크기에 비례 조정해 써도 되는가 (화면비가 허용 오차 안).

    같은 화면비의 축소·확대(프록시)는 fx·cx와 fy·cy를 같은 비율로 맞추면 정확하다. 화면비가
    다르면(4:3 모드에서 잰 값을 16:9 모드에 씀) 센서를 잘라 쓴 것이라 가로·세로를 따로 늘리면
    비정사각 픽셀과 틀린 주점이 된다 (F-PAR-44).
    """
    # 상대 차이: 1920x1080(16:9) vs 4000x3000(4:3)이면 1.778 / 1.333 - 1 = 0.33
    ratio = (width / height) / (calib.width / calib.height)
    return abs(ratio - 1.0) <= max_aspect_mismatch


def intrinsics_for(
    calib: CameraIntrinsics | None,
    width: int,
    height: int,
    hfov_deg: float,
    max_aspect_mismatch: float,
) -> Intrinsics:
    """세션 캘리브레이션(해상도가 다르면 비례 조정) 또는 기본 화각 근사.

    Args:
        calib: 세션 캘리브레이션 내부 파라미터 (없으면 None).
        width, height: 실제 디코드한 프레임 크기(px). 캘리브레이션 해상도와 다르면 fx·cx는 가로
            비율, fy·cy는 세로 비율로 맞춘다 (프록시·다운스케일 영상 대응).
        hfov_deg: calib가 없거나 쓸 수 없을 때의 가로 화각 (`depth.default_hfov_deg`).
        max_aspect_mismatch: 화면비 허용 오차 (`depth.max_intrinsics_aspect_mismatch`). 넘으면
            calib를 쓰지 않고 화각 근사로 떨어진다 (`calibration_fits`).
    """
    if calib is None or not calibration_fits(calib, width, height, max_aspect_mismatch):
        return Intrinsics.from_hfov(width, height, hfov_deg)
    # 같은 화면비의 축소·확대: 픽셀 좌표가 가로·세로 같은 비율로 늘어나므로 fx·cx는 가로, fy·cy는
    # 세로 비율을 곱한다 (화면비가 같아 sx ≈ sy)
    sx, sy = width / calib.width, height / calib.height
    return Intrinsics(calib.fx * sx, calib.fy * sy, calib.cx * sx, calib.cy * sy)


def lift_key(payload: LabelPayload) -> tuple[str, str] | None:
    """3D로 올리는 트랙이면 (entity_id, 트랙 종류), 아니면 None.

    hand21 키포인트 트랙(관절마다 part=관절 이름)과 박스 트랙(중심, part=None)만 올린다. 같은 키의
    트랙은 `lift_tracks`에 하나만 넘긴다 (둘이면 같은 (개체, 부위)에 표본이 두 번 들어간다).
    """
    if isinstance(payload, KeypointTrackPayload) and payload.skeleton == "hand21":
        return (payload.entity_id, payload.kind)
    if isinstance(payload, BoxTrackPayload):
        return (payload.entity_id, payload.kind)
    return None


# 깊이만으로 올릴 점: (개체, 부위, u, v 픽셀). 부위 None은 박스 중심
_Point = tuple[str, str | None, float, float]


def _points_by_time(
    tracks: list[LabelRecord],
) -> tuple[dict[int, list[_Point]], dict[int, list[tuple[str, KeypointFrame]]]]:
    """스트림 PTS ms → 그 시각에 올릴 박스 중심 점과 손 키프레임 (`lift_tracks` 1단계).

    Returns:
        (박스 중심 점, (개체, 손 키프레임)). 손 관절은 시각마다 키프레임째로 넘겨 `_lift_hand`가
        맞춤(world 있음) 또는 관절별 깊이로 올린다.

    Raises:
        ValueError: 같은 (개체, 트랙 종류) 트랙이 둘 이상일 때 (F-PAR-01). 두 트랙의 표본을 한
            궤적에 섞지 않는다. 러너는 `runner.pick_lift_tracks`로 하나만 고른다.
    """
    seen: dict[tuple[str, str], str] = {}
    points: dict[int, list[_Point]] = {}
    hands: dict[int, list[tuple[str, KeypointFrame]]] = {}
    for label in tracks:
        p = label.payload
        key = lift_key(p)
        if key is None:
            continue
        if key in seen:
            raise ValueError(
                f"{key[0]}: 같은 개체의 {key[1]} 트랙이 둘입니다 ({seen[key]}, {label.label_id}). "
                "개체마다 트랙 하나만 넘깁니다 (`runner.pick_lift_tracks`)"
            )
        seen[key] = label.label_id
        if isinstance(p, KeypointTrackPayload):
            for f in p.keyframes:
                hands.setdefault(f.t_ms, []).append((p.entity_id, f))
        elif isinstance(p, BoxTrackPayload):
            for k in p.keyframes:
                if not k.outside:
                    # 박스는 중심 한 점만 올린다 (part=None). 박스 좌표는 왼쪽 위 (x, y)와 크기 (w,
                    # h) px
                    cx, cy = k.x + k.w / 2, k.y + k.h / 2
                    points.setdefault(k.t_ms, []).append((p.entity_id, None, cx, cy))
    return points, hands


def _unproject_depth(
    dmap: NDArray[np.float32], intr: Intrinsics, u: float, v: float, policy: DepthPolicy
) -> tuple[float, float, float] | None:
    """픽셀 (u, v)의 깊이(주변 패치 중앙값)로 역투영. 깊이가 nan이거나 범위 밖이면 None."""
    # 깊이는 (2r+1)^2 패치 중앙값이다 (경계 한 픽셀이 배경을 읽어도 버틴다). 광축 방향 평면 깊이
    # z(m)
    z = sample_depth(dmap, u, v, policy.patch_radius_px)
    # 화면 밖(nan)이나 믿기 어려운 깊이는 버린다 (그 시각 샘플이 빠진다)
    if math.isnan(z) or not policy.min_depth_m <= z <= policy.max_depth_m:
        return None
    return intr.unproject(u, v, z)


# 표본 하나: (부위 이름, 카메라 좌표, 추정 방식, 신뢰도)
_Lifted = tuple[str, tuple[float, float, float], Lift3DMethod, float]


def hand_fit_of(
    frame: KeypointFrame, dmap: NDArray[np.float32], intr: Intrinsics, policy: DepthPolicy
) -> HandFit | None:
    """손 키프레임의 손 맞춤 (world가 없거나 맞춤이 안 되면 None, `handfit.fit_hand`)."""
    if frame.world is None:
        return None
    world = np.array([[w.x, w.y, w.z] for w in frame.world], dtype=np.float64)
    pixels = np.array([[q.x, q.y] for q in frame.points], dtype=np.float64)
    visible = np.array([q.visibility > 0 for q in frame.points])
    return fit_hand(world, pixels, visible, dmap, intr, policy)


def _lift_hand(
    frame: KeypointFrame,
    dmap: NDArray[np.float32],
    intr: Intrinsics,
    policy: DepthPolicy,
    calibrated: bool,
) -> list[_Lifted]:
    """손 키프레임 하나 → (관절 이름, 카메라 좌표, 방식, 신뢰도) 목록.

    `policy.hand_points` 중 보이는 관절만 낸다.

    world가 있고 맞춤이 되면 `handfit.fit_hand` 결과를 쓴다 (깊이 z가 [min_depth_m, max_depth_m]
    밖인 관절은 뺀다). 아니면 손바닥 관절(`hand_fit.anchor_points`)만 그 픽셀 깊이로 역투영하고,
    손가락 끝은 `fingertip_fallback`이 point_depth일 때만 같은 방식으로 올린다 (기본 drop,
    F-PAR-40). visibility 0(라벨 없음)인 관절은 올리지 않는다. 신뢰도는 맞춤이면
    `fit_confidence`(그 시각 맞춤 품질, 관절마다 같다), 깊이면 `point_confidence` (ADR 0045).
    """
    fitted = hand_fit_of(frame, dmap, intr, policy)
    out: list[_Lifted] = []
    if fitted is not None:
        conf = fit_confidence(fitted, calibrated=calibrated, policy=policy)
    else:
        conf = point_confidence(calibrated=calibrated, policy=policy)
    # 맞춤 없이 픽셀 깊이로 믿고 올릴 수 있는 관절: 손바닥(넓은 면이라 깊이 패치가 손 위에 있다)
    palm = set(policy.hand_fit.anchor_points)
    keep_tips = policy.fingertip_fallback == "point_depth"
    for i in policy.hand_points:
        kp = frame.points[i]
        if kp.visibility == 0 or (fitted is None and i not in palm and not keep_tips):
            continue
        if fitted is not None:
            # 맞춘 21관절 중 이 관절 (카메라 좌표 m). 깊이 범위 밖이면 그 관절만 뺀다
            x, y, z = (float(c) for c in fitted.points[i])
            xyz = (x, y, z) if policy.min_depth_m <= z <= policy.max_depth_m else None
            method = Lift3DMethod.HAND_FIT
        else:
            xyz = _unproject_depth(dmap, intr, kp.x, kp.y, policy)
            method = Lift3DMethod.POINT_DEPTH
        if xyz is not None:
            out.append((HAND21_NAMES[i], xyz, method, conf))
    return out


def _sample(
    t: int, xyz: tuple[float, float, float], method: Lift3DMethod, confidence: float
) -> Trajectory3DSample:
    """3D 표본 (소수 넷째 자리, 0.1 mm)과 그 추정 방식·신뢰도."""
    x, y, z = xyz
    return Trajectory3DSample(
        t_ms=t,
        x=round(x, 4),
        y=round(y, 4),
        z=round(z, 4),
        method=method,
        confidence=confidence,
    )


def depth_frames(
    frames: Iterable[tuple[int, Image]],
    wanted: Container[int],
    depth: DepthModel,
    calib: CameraIntrinsics | None,
    policy: DepthPolicy,
) -> Iterator[tuple[int, NDArray[np.float32], Intrinsics]]:
    """깊이를 만들 프레임마다 (PTS ms, 깊이 맵, 내부 파라미터).

    wanted에 있는 시각의 프레임 중 마지막으로 깊이를 만든 프레임에서 frame_stride_ms가 지나기 전
    프레임은 건너뛴다 (0이어도 1 ms로 두어 같은 시각을 두 번 쓰지 않는다). 내부 파라미터는 첫
    처리 프레임의 크기로 한 번 정한다 (`intrinsics_for`, 영상 중간에 해상도가 바뀌지 않는다고 본다.
    화면비가 다른 캘리브레이션은 화각 근사로 떨어진다).
    `lift_tracks`와 작업자 손 크기 추정(`dlp_prelabel.palm`)이 함께 쓴다.
    """
    min_gap = max(policy.frame_stride_ms, 1)
    last: int | None = None
    intr: Intrinsics | None = None
    for t, img in frames:
        if t not in wanted or (last is not None and t - last < min_gap):
            continue
        last = t
        if intr is None:
            intr = intrinsics_for(
                calib, img.shape[1], img.shape[0], policy.default_hfov_deg,
                policy.max_intrinsics_aspect_mismatch,
            )  # fmt: skip
        yield t, depth.predict(img), intr


def hand_frames_by_time(
    tracks: list[LabelRecord],
) -> dict[int, list[tuple[str, KeypointFrame]]]:
    """스트림 PTS ms → 그 시각 (개체, 손 키프레임).

    hand21 트랙만 보며 `_points_by_time`과 같은 검사(같은 개체 트랙 둘이면 ValueError)를 한다.
    """
    return _points_by_time(tracks)[1]


def lift_tracks(
    frames: Iterable[tuple[int, Image]],
    depth: DepthModel,
    tracks: list[LabelRecord],
    calib: CameraIntrinsics | None,
    policy: DepthPolicy,
) -> dict[tuple[str, str | None], list[Trajectory3DSample]]:
    """(entity_id, part) → 3D 샘플. tracks는 hand21 키포인트 트랙과 박스 트랙.

    hand21 트랙은 `policy.hand_points` 관절마다 part=관절 이름(예: "wrist", "pinky_tip"), 박스
    트랙은 화면 밖이 아닌 키프레임의 박스 중심을 part=None으로 올린다. 손 관절은 키프레임에 world가
    있으면 손 전체 맞춤(`_lift_hand`, ADR 0044), 없으면 관절별 깊이다. 깊이는 키프레임 시각과 PTS가
    정확히 같은 프레임에서만 읽고, 마지막으로 깊이를 만든 프레임에서 frame_stride_ms가 지나기 전
    프레임은 건너뛴다 (0이어도 마지막 프레임보다 늦은 프레임만 쓴다: PTS 반올림으로 같은 ms 프레임이
    이어져도 표본은 시각마다 하나). 깊이가 nan(화면 밖)이거나 [min_depth_m, max_depth_m] 밖이면 그
    점은 버린다. 표본마다 추정 방식(`Lift3DMethod`)과 신뢰도를 싣는다 (ADR 0045).

    Args:
        frames: (스트림 PTS ms, RGB 프레임) 순회 (`common.iter_frames`).
        depth: 깊이 추론기 (프레임마다 미터 깊이 맵).
        tracks: 바디캠 현재 라벨 중 키포인트·박스 트랙 (`lift_key`가 None인 것은 무시).
            (개체, 트랙 종류)마다 하나여야 한다 (러너는 `runner.pick_lift_tracks`로 고른다).
        calib: 카메라 내부 파라미터 (세션 또는 등록된 캘리브레이션, None이면 화각 근사). 첫 처리
            프레임 크기로 한 번 정한다.
        policy: `prelabel.yaml depth` (작업자 손 크기 사전값은 `hand_fit.reference_palm_m`을 바꾼
            사본으로 넘긴다).

    Returns:
        (entity_id, part) → 시각 순(중복 없음) `Trajectory3DSample` (카메라 좌표 m, 소수 넷째 자리).

    Raises:
        ValueError: 같은 (개체, 트랙 종류) 트랙이 둘 이상일 때 (F-PAR-01).
    """
    return lift_tracks_with_source(frames, depth, tracks, calib, policy)[0]


def lift_tracks_with_source(
    frames: Iterable[tuple[int, Image]],
    depth: DepthModel,
    tracks: list[LabelRecord],
    calib: CameraIntrinsics | None,
    policy: DepthPolicy,
) -> tuple[dict[tuple[str, str | None], list[Trajectory3DSample]], bool]:
    """`lift_tracks`와 같고, 실제로 캘리브레이션을 썼는지(화각 근사가 아님)를 함께 돌려준다.

    calib가 있어도 첫 처리 프레임과 화면비가 맞지 않으면 화각 근사를 쓰고 거짓이다 (F-PAR-44).
    처리한 프레임이 없으면 calib 유무를 그대로 돌려준다.
    """
    # 1단계: 스트림 PTS ms → 그 시각에 올릴 박스 중심과 손 키프레임
    points, hands = _points_by_time(tracks)
    calibrated: bool | None = None
    box_conf = point_confidence(calibrated=calib is not None, policy=policy)
    out: dict[tuple[str, str | None], list[Trajectory3DSample]] = {}
    # 2단계: 점이 있는 프레임 중 frame_stride_ms 간격으로만 깊이를 만든다 (CPU 비용)
    for t, dmap, intr in depth_frames(frames, points.keys() | hands.keys(), depth, calib, policy):
        if calibrated is None:
            # 깊이 맵은 프레임과 같은 크기다 (H, W). 화면비가 다르면 화각 근사를 썼다
            h, w = dmap.shape[:2]
            calibrated = calib is not None and calibration_fits(
                calib, w, h, policy.max_intrinsics_aspect_mismatch
            )
            box_conf = point_confidence(calibrated=calibrated, policy=policy)
        for entity, part, u, v in points.get(t, []):
            xyz = _unproject_depth(dmap, intr, u, v, policy)
            if xyz is not None:
                sample = _sample(t, xyz, Lift3DMethod.POINT_DEPTH, box_conf)
                out.setdefault((entity, part), []).append(sample)
        for entity, frame in hands.get(t, []):
            for name, xyz, method, conf in _lift_hand(frame, dmap, intr, policy, calibrated):
                out.setdefault((entity, name), []).append(_sample(t, xyz, method, conf))
    return out, calib is not None if calibrated is None else calibrated


def _source_of(
    inputs: dict[tuple[str, bool], str], entity: str, part: str | None
) -> tuple[str, ...] | None:
    """(개체, 부위) 궤적을 올린 입력 트랙 ID (손 관절이면 손 트랙, 박스 중심이면 박스 트랙)."""
    found = inputs.get((entity, part is not None))
    return None if found is None else (found,)


def with_palm(policy: DepthPolicy, palm_m: float | None) -> DepthPolicy:
    """손 크기 사전값을 작업자 값으로 바꾼 depth 정책 사본 (None이면 그대로, ADR 0045)."""
    if palm_m is None:
        return policy
    fit = policy.hand_fit.model_copy(update={"reference_palm_m": palm_m})
    return policy.model_copy(update={"hand_fit": fit})


def mean_confidence(samples: Iterable[Trajectory3DSample]) -> float:
    """표본 신뢰도의 평균 (소수 셋째 자리). 신뢰도가 없는 표본은 빼고, 모두 없으면 0."""
    values = [x.confidence for x in samples if x.confidence is not None]
    return round(sum(values) / len(values), 3) if values else 0.0


class DepthLifter:
    """메트릭 깊이 3D 단계. 러너가 Predictor 목록과 따로 부른다 (입력이 영상 + 다른 라벨이다).

    name: "depth3d". version: `depth3d-<가중치 버전>+hf4+p<depth 절 해시>` (hf1: 손 중심 3D 점
    맞춤 판, ADR 0044. hf2: 표본별 방식·신뢰도, ADR 0045. hf3: 궤적마다 입력 트랙 ID
    `derived_from`, ADR 0048 — 3D가 2D 손 트랙 검수를 물려받는 근거. hf4: 손 모델 자세 회전
    다듬기·해상도와 무관한 상대 재투영 문턱, 맞춤 없는 손가락 끝 빼기, 화면비가 다른 캘리브레이션
    거부, F-PAR-40·42·43·44. 러너가 입력 해시와 손 크기 사전값 출처 `+hp<w|p>`를 더 붙인다).
    weights_version: 깊이 가중치 버전만 (자세 단계 버전의 깊이 입력, depth 절의 손 맞춤 정책이
    바뀌어도 자세를 다시 추정하지 않게, F-PAR-53).
    """

    name = "depth3d"
    # 깊이 가중치 판 (`__init__`이 정한다. 가중치 없이 만든 시험용 하위 클래스는 빈 값)
    weights_version: str = ""

    def __init__(self, root: Path, policy: PrelabelPolicy, *, now: datetime) -> None:
        """가중치를 확인하고 버전을 정한다. 모델 세션은 처음 `run`할 때 연다.

        Raises:
            ModelUnavailableError: 깊이 가중치가 없을 때 (`make export-models`). CLI는 이 단계를
                건너뛴다.
        """
        self.path, version = resolve(root, policy.models.depth)
        self.weights_version = version
        self.version = f"depth3d-{version}+hf4+p{policy.digest('depth')}"
        self.policy, self.now = policy, now
        self.model: DepthModel | None = None

    def depth_model(self) -> DepthModel:
        """깊이 추론기. ONNX 세션은 처음 쓸 때 한 번만 연다 (가중치가 커서 생성 비용이 크다)."""
        if self.model is None:
            self.model = MetricDepth(self.path)
        return self.model

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
        palm_m: float | None = None,
    ) -> list[LabelRecord]:
        """바디캠 영상에서 트랙을 3D로 올려 trajectory3d 라벨을 만든다.

        Args:
            video: 바디캠 영상 로컬 경로.
            session_id, stream_id: 라벨의 세션·스트림 (바디캠).
            tracks: 올릴 손·박스 트랙. (개체, 트랙 종류)마다 하나 (`lift_tracks` 참고).
            calib: 카메라 내부 파라미터 (세션 또는 등록된 캘리브레이션,
                `calibration_repository.session_intrinsics`). None이거나 영상과 화면비가 맞지
                않으면 화각 근사이고 페이로드 intrinsics는 hfov_approx다 (F-PAR-44).
            ontology_version: 라벨에 넣을 온톨로지 버전.
            version: 라벨에 쓸 모델 버전 (기본: `self.version`). 러너는 입력 해시를 붙여 넘긴다.
            palm_m: 작업자 손 크기 사전값 (m, None이면 정책 `hand_fit.reference_palm_m`, ADR 0045).

        Returns:
            (개체, 부위)마다 trajectory3d 라벨 하나. ID는
            `<세션>-<스트림>-3d-<version_tag>-<개체>-<부위|center>`. 표본마다 방식·신뢰도를 싣고,
            라벨 신뢰도는 표본 신뢰도의 평균(소수 셋째 자리)이다. `derived_from`은 그 개체를 올린
            입력 트랙의 라벨 ID 하나다 (ADR 0048: 손 3D가 그 2D 트랙의 검수를 물려받는 근거).
            DB에는 쓰지 않는다 (러너가 쓴다).

        Raises:
            ValueError: tracks에 같은 (개체, 트랙 종류) 트랙이 둘 이상일 때 (`lift_tracks`).
        """
        # 러너는 입력 해시를 붙인 버전을 넘긴다 (라벨 ID의 version_tag도 이 버전으로 만든다)
        model_version = version or self.version
        policy = with_palm(self.policy.depth, palm_m)
        samples, used = lift_tracks_with_source(
            iter_frames(video), self.depth_model(), tracks, calib, policy
        )
        # 출처는 실제로 쓴 값을 따른다 (화면비가 달라 화각 근사로 떨어졌으면 hfov_approx)
        intrinsics = IntrinsicsSource.CALIBRATED if used else IntrinsicsSource.HFOV_APPROX
        # 개체 → 올린 입력 트랙 (같은 개체의 손·박스 트랙이 함께 오면 part로 가른다: 손은 관절 이름,
        # 박스는 None)
        inputs = {
            (t.payload.entity_id, isinstance(t.payload, KeypointTrackPayload)): t.label_id
            for t in tracks
            if lift_key(t.payload) is not None
            and isinstance(t.payload, KeypointTrackPayload | BoxTrackPayload)
        }
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
                intrinsics=intrinsics,
                derived_from=_source_of(inputs, entity, part),
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
                    confidence=mean_confidence(ss),
                    payload=payload,
                    now=self.now,
                )
            )
        return out
