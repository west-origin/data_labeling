"""AI 1차 검수 벤치용 합성 장면 (WP17.3, ADR 0040). 실제 영상·사람·글자가 없다.

`dlp aireview bench`가 서버 처리량을 잴 때 실제 계획기·프롬프트·스키마에 넣을 입력이다.

- `aireview_scene`: 정답 경계를 아는 행동 시나리오(`generate_action_scenario`)의 라벨을 검수 전
  자동(model) 라벨로 바꾸고 (손 상태는 영상만 잡은 접촉처럼 증거 INFERRED로 둔다. 장갑 근거
  접촉은 AI 판정 대상이 아니다), 장면 개체(`ENTITIES`)마다 바디캠 박스 트랙, 3인칭 스트림(`tp1`)에
  박스 트랙 하나를 더한 세션이다. 화면 위치·크기는 320x240 기준 값을 영상 크기에 맞춰 늘린다.
- `scene_frame`: 그 스트림·시각의 합성 RGB 프레임 (배경 그라디언트 + 개체 색 사각형 + 시각에 따라
  움직이는 띠). 프레임마다 픽셀이 달라 JPEG도 다르다. 같은 입력이면 같은 픽셀이다.
- `scene_times`: 스트림의 프레임 시각 ms (바디캠은 시나리오 30 fps, 3인칭은 같은 시각을 쓴다).

프레임을 미리 만들어 두지 않는다 (1080p 프레임 수천 장은 메모리를 크게 쓴다). 벤치는 계획이 고른
프레임만 그때그때 그린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_schema.labels import (
    BoxKeyframe,
    BoxTrackPayload,
    Evidence,
    HandStatePayload,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    Provenance,
    Source,
)
from dlp_schema.session import Session, Stream, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session

# RGB 영상 (height, width, 3)
Image = NDArray[np.uint8]
# ENTITIES 화면 위치의 기준 크기 (행동 시나리오의 손목 궤적과 같은 좌표계)
BASE_SIZE = (320, 240)
# 3인칭 스트림 ID와 그 스트림의 박스 개체
THIRD_PERSON = "tp1"
_TP_ENTITY = ("bucket_01", "bucket", (100.0, 100.0))
# 박스 트랙 키프레임 간격 ms (프리라벨 객체 추적 간격과 같은 자릿수)
_BOX_STEP_MS = 500
# 박스 한 변 (기준 크기 px)
_BOX_SIDE = 30.0
# 클래스별 사각형 색 (RGB, 결정적)
_COLORS = ((200, 80, 60), (60, 160, 90), (70, 90, 200), (210, 190, 60), (150, 70, 170))


@dataclass(frozen=True)
class AiReviewScene:
    """합성 장면 하나.

    Attributes:
        session: 바디캠 + IMU + 3인칭(tp1) 세션 (프라이버시·생애주기 상태는 기본값).
        labels: 현재 운영 라벨 (모두 검수 전 모델 출처).
        frame_times: 바디캠 프레임 시각 ms (30 fps). 3인칭도 같은 시각을 쓴다.
        width, height: 영상 크기 px.
    """

    session: Session
    labels: tuple[LabelRecord, ...] = field(repr=False)
    frame_times: tuple[int, ...] = field(repr=False)
    width: int
    height: int


def _as_model(x: LabelRecord, sx: float, sy: float) -> LabelRecord:
    """모델 출처(검수 전) 사본. 손 상태는 영상 근거(INFERRED), 키포인트는 영상 크기로 늘린다."""
    update: dict[str, object] = {
        "provenance": Provenance(source=Source.MODEL, model_version="prelabel-synthetic"),
        "confidence": 0.8,
    }
    p = x.payload
    if isinstance(p, HandStatePayload):
        update["evidence"] = Evidence.INFERRED
    if isinstance(p, KeypointTrackPayload):
        frames = tuple(
            KeypointFrame(
                t_ms=f.t_ms,
                points=tuple(
                    Keypoint(x=k.x * sx, y=k.y * sy, visibility=k.visibility) for k in f.points
                ),
            )
            for f in p.keyframes
        )
        update["payload"] = p.model_copy(update={"keyframes": frames})
    return x.model_copy(update=update)


def _box_track(
    label_id: str,
    session_id: str,
    stream: str,
    entity: tuple[str, str, tuple[float, float]],
    times: list[int],
    scale: tuple[float, float],
) -> LabelRecord:
    """정지한 박스 트랙 (모델 출처, 기준 30x30 px을 영상 크기로 늘린다)."""
    eid, cls, (cx, cy) = entity
    sx, sy = scale
    w, h = _BOX_SIDE * sx, _BOX_SIDE * sy
    kfs = tuple(BoxKeyframe(t_ms=t, x=cx * sx - w / 2, y=cy * sy - h / 2, w=w, h=h) for t in times)
    return LabelRecord(
        label_id=label_id,
        session_id=session_id,
        stream_id=stream,
        t_start_ms=times[0],
        t_end_ms=times[-1],
        ontology_version="1.0.0",
        provenance=Provenance(source=Source.MODEL, model_version="prelabel-synthetic"),
        confidence=0.9,
        created_at=FIXED_TIME,
        payload=BoxTrackPayload(entity_id=eid, class_id=cls, keyframes=kfs),
    )


def _session(session_id: str, duration_ms: int) -> Session:
    """바디캠·IMU(기본) + 3인칭 스트림 tp1 (오프셋 120 ms)."""
    base = make_session(session_id, duration_ms=duration_ms)
    tp = Stream(
        stream_id=THIRD_PERSON,
        kind=StreamKind.THIRD_PERSON,
        uri=f"s3://dlp-raw/{session_id}/{THIRD_PERSON}.mp4",
        sync_method=SyncMethod.QR_SLATE,
        offset_ms=120.0,
    )
    streams = [*(s.model_dump() for s in base.streams), tp.model_dump()]
    return Session.model_validate(base.model_dump() | {"streams": streams})


def aireview_scene(
    seed: int = 0,
    *,
    session_id: str = "syn-aireview",
    n_units: int = 8,
    width: int = 320,
    height: int = 240,
) -> AiReviewScene:
    """합성 장면을 만든다 (결정적).

    Args:
        seed: 행동 시나리오 난수 seed.
        session_id: 세션 ID (라벨 ID 접두사).
        n_units: 행동 단위 수 (길이가 대략 비례한다).
        width, height: 영상 크기 px (짝수).
    """
    sc = generate_action_scenario(seed, session_id=session_id, n_units=n_units)
    sx, sy = width / BASE_SIZE[0], height / BASE_SIZE[1]
    times = list(range(0, sc.duration_ms, _BOX_STEP_MS))
    boxes = [
        _box_track(
            f"{session_id}-box-{eid}", session_id, "bodycam", (eid, cls, pos), times, (sx, sy)
        )
        for eid, cls, _, pos, _ in ENTITIES
    ]
    tp_box = _box_track(
        f"{session_id}-tp-box", session_id, THIRD_PERSON, _TP_ENTITY, times, (sx, sy)
    )
    labels = (*(_as_model(x, sx, sy) for x in sc.labels), *boxes, tp_box)
    return AiReviewScene(
        _session(session_id, sc.duration_ms), labels, tuple(sc.frame_times), width, height
    )


def scene_frame(scene: AiReviewScene, stream_id: str, t_ms: int) -> Image:
    """합성 RGB 프레임 (모듈 docstring). 사람·글자가 없다.

    Args:
        scene: 장면.
        stream_id: 스트림 (3인칭이면 배경 색조가 다르다).
        t_ms: 그 스트림 시각 ms (띠 위치가 시각에 따라 움직인다).
    """
    w, h = scene.width, scene.height
    xs = np.linspace(40, 120, w, dtype=np.float32)
    ys = np.linspace(60, 140, h, dtype=np.float32)
    tint = 40.0 if stream_id == THIRD_PERSON else 0.0
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[..., 0] = xs[np.newaxis, :].astype(np.uint8)
    img[..., 1] = ys[:, np.newaxis].astype(np.uint8)
    img[..., 2] = np.uint8(70 + tint)
    sx, sy = w / BASE_SIZE[0], h / BASE_SIZE[1]
    for k, (_, _, _, (cx, cy), _) in enumerate(ENTITIES):
        x0, y0 = int((cx - _BOX_SIDE / 2) * sx), int((cy - _BOX_SIDE / 2) * sy)
        img[max(0, y0) : y0 + int(_BOX_SIDE * sy), max(0, x0) : x0 + int(_BOX_SIDE * sx)] = _COLORS[
            k % len(_COLORS)
        ]
    band = max(2, h // 30)
    top = int((t_ms / 20) % max(1, h - band))
    img[top : top + band, :] = (230, 230, 230)
    return img


def scene_times(scene: AiReviewScene) -> tuple[int, ...]:
    """프레임 시각 ms (바디캠·3인칭 모두 이 시각을 쓴다)."""
    return scene.frame_times
