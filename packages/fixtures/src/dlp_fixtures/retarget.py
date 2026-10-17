"""정답 끝단 궤적을 아는 사람 손 점 (로봇 리타기팅 시험, ADR 0047).

로봇 끝단(도구 중심점) 목표 = 엄지·검지 끝 가운데(잡기 점), 접근 축 = 손목 → 잡기 점, 닫힘 축 = 엄지
→
검지, 그리퍼 열림 = 엄지-검지 끝 거리라는 규약(`dlp_retarget.targets`)을 거꾸로 써서, 알려진 끝단
궤적 (시험이 로봇 정기구학으로 만든 도달 가능한 궤적 등)에서 손 6점을 만든다. 시험은 리타기팅 결과의
끝단 경로를 이 정답과 비교한다.

- `hand_points_for_grasp`: 잡기 점·접근 축·닫힘 축·열림 거리 → 손 6점 (부위 이름 → (n, 3)).
- `world_hand_labels`: 손 6점 → 세션 월드 좌표 trajectory3d 라벨 (월드 단계 출력 흉내, 내보내기
시험).
- `world_hand_sources`: 위 월드 좌표 라벨의 `derived_from`이 가리키는 입력 (2D 손 트랙, 카메라 좌표
  궤적, 카메라 자세). 3D가 2D 손 트랙 검수를 물려받는 시험(ADR 0048)에 쓴다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from dlp_schema.labels import (
    CoordinateFrame,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Lift3DMethod,
    Provenance,
    Source,
    Source3D,
    Trajectory3DPayload,
    Trajectory3DSample,
    Verification,
    VerificationState,
)
from dlp_schema.testing import FIXED_TIME

# 손 6점 부위 (온톨로지 hand_joints)
PARTS = ("wrist", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
# 손목 → 잡기 점 거리 (m)
WRIST_TO_GRASP_M = 0.09


def _unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """행마다 단위 벡터."""
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def hand_points_for_grasp(
    grasp: NDArray[np.float64],
    approach: NDArray[np.float64],
    closing: NDArray[np.float64],
    opening_m: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """잡기 점 (n, 3), 접근 축 (n, 3), 닫힘 축 (n, 3, 접근 축과 수직이 아니면 수직 성분만), 열림
    거리
    (n,) → 부위 → (n, 3).

    엄지 끝 = g - c d/2, 검지 끝 = g + c d/2, 손목 = g - a x 0.09 m. 중지·약지·새끼 끝은 잡기 점
    뒤쪽 손바닥 둘레에 둔다 (리타기팅은 쓰지 않는다).
    """
    a = _unit(approach)
    c = _unit(closing - np.sum(closing * a, axis=1, keepdims=True) * a)
    d = opening_m[:, None]
    side = np.cross(a, c)
    thumb, index = grasp - c * d / 2, grasp + c * d / 2
    out = {
        "wrist": grasp - a * WRIST_TO_GRASP_M,
        "thumb_tip": thumb,
        "index_tip": index,
        "middle_tip": grasp - a * 0.01 + c * (d / 2 + 0.01) + side * 0.01,
        "ring_tip": grasp - a * 0.02 + c * (d / 2 + 0.015) + side * 0.02,
        "pinky_tip": grasp - a * 0.03 + c * (d / 2 + 0.02) + side * 0.03,
    }
    return out


def hand_points_from_joints(
    fk: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    q: NDArray[np.float64],
    approach_axis: Sequence[float],
    closing_axis: Sequence[float],
    opening_m: NDArray[np.float64],
    *,
    base_xyz: Sequence[float],
    base_yaw_rad: float,
    position_scale: float,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """정답 관절 궤적 q (n, dof) → (월드 손 6점, 정답 끝단 위치 (n, 3), 정답 접근 축 (n, 3)).

    끝단 위치·접근 축은 로봇 기준 좌표다.

    fk: 관절 값 → 끝 링크 4x4 자세 (로봇 기준 좌표). world_fixed 배치(`p_b = s · Rz(yaw)ᵀ (g -
    base)`)를 거꾸로 써서 잡기 점 g = base + Rz(yaw) p_b / s를 만든다.
    """
    tips = np.array([fk(row) for row in q])
    pos_b = tips[:, :3, 3]
    a_b = tips[:, :3, :3] @ np.asarray(approach_axis, dtype=np.float64)
    c_b = tips[:, :3, :3] @ np.asarray(closing_axis, dtype=np.float64)
    c, s = math.cos(base_yaw_rad), math.sin(base_yaw_rad)
    r_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    grasp = np.asarray(base_xyz, dtype=np.float64) + (pos_b / position_scale) @ r_wb.T
    return hand_points_for_grasp(grasp, a_b @ r_wb.T, c_b @ r_wb.T, opening_m), pos_b, a_b


def world_hand_labels(
    session_id: str,
    times_ms: Sequence[int],
    points: dict[str, NDArray[np.float64]],
    *,
    hand: str = "right",
    stream_id: str = "bodycam",
    state: VerificationState = VerificationState.HUMAN_APPROVED,
) -> list[LabelRecord]:
    """손 6점 → 월드 좌표 trajectory3d 라벨 (부위마다 하나, 모델 출처, 검증 상태 state).

    NaN인 표본은 넣지 않는다 (그 시각 손 점 없음). 기본 검증 상태는 승인이다 (내보내기 기본 정책이
    검수한 라벨만 넣으므로).
    """
    out: list[LabelRecord] = []
    for part in PARTS:
        pts = points[part]
        samples = tuple(
            Trajectory3DSample(
                t_ms=int(t),
                x=round(float(p[0]), 5),
                y=round(float(p[1]), 5),
                z=round(float(p[2]), 5),
                method=Lift3DMethod.HAND_FIT,
                confidence=0.7,
            )
            for t, p in zip(times_ms, pts, strict=True)
            if not np.isnan(p).any()
        )
        reviewed = state is not VerificationState.UNREVIEWED
        out.append(
            LabelRecord(
                label_id=f"{session_id}-{stream_id}-world3d-fx-{hand}_hand-{part}",
                session_id=session_id,
                stream_id=stream_id,
                t_start_ms=samples[0].t_ms,
                t_end_ms=samples[-1].t_ms,
                ontology_version="1.0.0",
                provenance=Provenance(source=Source.MODEL, model_version="world3d-fixture"),
                confidence=0.7,
                verification=Verification(
                    state=state,
                    reviewer_id="rev1" if reviewed else None,
                    reviewed_at=FIXED_TIME if reviewed else None,
                ),
                created_at=FIXED_TIME,
                payload=Trajectory3DPayload(
                    entity_id=f"{hand}_hand",
                    part=part,
                    frame=CoordinateFrame.WORLD,
                    source_3d=Source3D.MONO_DEPTH,
                    samples=samples,
                    derived_from=(f"{session_id}-cam-{part}", f"{session_id}-pose"),
                ),
            )
        )
    return out


def _model_label(
    label_id: str,
    session_id: str,
    stream_id: str,
    times: tuple[int, int],
    state: VerificationState,
    payload: KeypointTrackPayload | Trajectory3DPayload,
    version: str,
) -> LabelRecord:
    """모델 출처 공간 라벨 (검증 상태 state, 검수됐으면 검수자 rev1·FIXED_TIME)."""
    reviewed = state is not VerificationState.UNREVIEWED
    return LabelRecord(
        label_id=label_id,
        session_id=session_id,
        stream_id=stream_id,
        t_start_ms=times[0],
        t_end_ms=times[1],
        ontology_version="1.0.0",
        provenance=Provenance(source=Source.MODEL, model_version=version),
        confidence=0.7,
        verification=Verification(
            state=state,
            reviewer_id="rev1" if reviewed else None,
            reviewed_at=FIXED_TIME if reviewed else None,
        ),
        created_at=FIXED_TIME,
        payload=payload,
    )


def world_hand_sources(
    session_id: str,
    times_ms: Sequence[int],
    points: dict[str, NDArray[np.float64]],
    *,
    hand: str = "right",
    stream_id: str = "bodycam",
    track_state: VerificationState = VerificationState.UNREVIEWED,
) -> list[LabelRecord]:
    """`world_hand_labels`의 `derived_from` 입력: [2D 손 트랙, 카메라 좌표 궤적 6개, 카메라 자세].

    자세는 늘 원점·단위 회전이라 카메라 좌표 궤적 표본 = 월드 좌표 표본이다 (정답이 같다). 2D 손
    트랙(hand21, `<세션>-hands-fx-<손>`)의 픽셀 값은 시험이 쓰지 않는 자리 값이다. 카메라 좌표 궤적
    (`<세션>-cam-<부위>`)은 그 2D 트랙에서 계산된 것으로 `derived_from`에 적는다 (ADR 0048). 2D
    트랙의 검증 상태는 track_state, 3D 입력은 모두 검수 전이다.
    """
    span = (int(times_ms[0]), int(times_ms[-1]))
    track_id = f"{session_id}-hands-fx-{hand}"
    frames = tuple(
        KeypointFrame(
            t_ms=int(t),
            points=tuple(Keypoint(x=100.0 + j, y=80.0 + j, visibility=2) for j in range(21)),
        )
        for t in times_ms
    )
    entity = f"{hand}_hand"
    track = KeypointTrackPayload(
        entity_id=entity, skeleton="hand21", hand=Hand(hand), keyframes=frames
    )
    out = [_model_label(track_id, session_id, stream_id, span, track_state, track, "hands-fx")]
    for part in PARTS:
        samples = tuple(
            Trajectory3DSample(
                t_ms=int(t), x=round(float(p[0]), 5), y=round(float(p[1]), 5),
                z=round(float(p[2]), 5), method=Lift3DMethod.HAND_FIT, confidence=0.7,
            )
            for t, p in zip(times_ms, points[part], strict=True)
            if not np.isnan(p).any()
        )  # fmt: skip
        payload = Trajectory3DPayload(
            entity_id=entity, part=part, frame=CoordinateFrame.CAMERA,
            source_3d=Source3D.MONO_DEPTH, samples=samples, derived_from=(track_id,),
        )  # fmt: skip
        out.append(
            _model_label(
                f"{session_id}-cam-{part}", session_id, stream_id,
                (samples[0].t_ms, samples[-1].t_ms), VerificationState.UNREVIEWED, payload,
                "depth3d-fixture",
            )
        )  # fmt: skip
    pose = Trajectory3DPayload(
        entity_id="camera", frame=CoordinateFrame.WORLD, source_3d=Source3D.VISUAL_INERTIAL_SLAM,
        samples=tuple(
            Trajectory3DSample(t_ms=int(t), x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
            for t in times_ms
        ),
    )  # fmt: skip
    out.append(
        _model_label(
            f"{session_id}-pose",
            session_id,
            stream_id,
            span,
            VerificationState.UNREVIEWED,
            pose,
            "campose-fixture",
        )
    )
    return out
