"""AI 1차 검수 테스트 공용 픽스처 (WP17.2).

정답을 아는 합성 행동 시나리오(`dlp_fixtures.actions.generate_action_scenario`)의 라벨을 모델 출처
(검수 전 자동 라벨)로 바꾸고, 장면 개체마다 바디캠 박스 트랙을, 3인칭 스트림에는 박스 트랙 하나를
더한다. 블러본 PTS 인덱스는 시나리오 프레임 시각(바디캠, 정수 ms)으로 만든다. 영상·DB·서버는 쓰지
않는다 (이미지는 단색 합성 프레임).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from dlp_aireview.overlay import render_unit_images
from dlp_aireview.plan import PlannedUnit, StreamMedia
from dlp_aireview.policy import AiReviewPolicy, load_policy
from dlp_aireview.prompt import PromptImage
from dlp_fixtures.actions import ENTITIES, generate_action_scenario
from dlp_media.pts import PtsIndex
from dlp_schema import repo_root
from dlp_schema.labels import BoxKeyframe, BoxTrackPayload, LabelRecord, Provenance, Source
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.session import Session, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME, make_session

ROOT = repo_root()
SID = "s001"
RENDER = {"bodycam": "a" * 64, "tp1": "b" * 64}
SIZE = (320, 240)


def as_model(x: LabelRecord, version: str = "prelabel-test-0123") -> LabelRecord:
    """출처를 모델(검수 전)로 바꾼 사본."""
    return x.model_copy(
        update={
            "session_id": SID,
            "provenance": Provenance(source=Source.MODEL, model_version=version),
            "confidence": 0.8,
        }
    )


def box_track(
    label_id: str, entity: str, cls: str, stream: str, xy: tuple[float, float], times: list[int]
) -> LabelRecord:
    """정지한 30x30 박스 트랙 (모델 출처)."""
    kfs = tuple(BoxKeyframe(t_ms=t, x=xy[0] - 15, y=xy[1] - 15, w=30, h=30) for t in times)
    return LabelRecord(
        label_id=label_id,
        session_id=SID,
        stream_id=stream,
        t_start_ms=times[0],
        t_end_ms=times[-1],
        ontology_version="1.0.0",
        provenance=Provenance(source=Source.MODEL, model_version="prelabel-test-0123"),
        confidence=0.9,
        created_at=FIXED_TIME,
        payload=BoxTrackPayload(entity_id=entity, class_id=cls, keyframes=kfs),
    )


@dataclass(frozen=True)
class Scene:
    """합성 세션 하나.

    Attributes:
        session: 바디캠 + IMU + 3인칭(tp1) 세션.
        labels: 현재 운영 라벨 (모델 출처).
        media: 스트림 → 블러본 PTS 인덱스·렌더 해시.
        frame_times: 바디캠 프레임 시각 ms.
    """

    session: Session
    labels: list[LabelRecord]
    media: dict[str, StreamMedia]
    frame_times: list[int]


def _session(duration: int) -> Session:
    """바디캠·IMU(기본) + 3인칭 스트림 tp1."""
    base = make_session(SID, duration_ms=duration)
    tp = {
        "stream_id": "tp1", "kind": StreamKind.THIRD_PERSON, "uri": "s3://dlp-raw/s001/tp1.mp4",
        "sync_method": SyncMethod.QR_SLATE, "offset_ms": 120.0,
    }  # fmt: skip
    return Session.model_validate(
        base.model_dump() | {"streams": [*(s.model_dump() for s in base.streams), tp]}
    )


def make_scene(seed: int = 0, n_units: int = 6) -> Scene:
    """합성 장면 (결정적)."""
    sc = generate_action_scenario(seed, session_id=SID, n_units=n_units)
    times = list(range(0, sc.duration_ms, 500))
    boxes = [
        box_track(f"{SID}-box-{eid}", eid, cls, "bodycam", pos, times)
        for eid, cls, _, pos, _ in ENTITIES
    ]
    tp_box = box_track(f"{SID}-tp-box", "bucket_01", "bucket", "tp1", (100.0, 100.0), times)
    labels = [as_model(x) for x in sc.labels] + boxes + [tp_box]
    pts = PtsIndex(
        Fraction(1, 1000),
        np.asarray(sc.frame_times, dtype=np.int64),
        np.ones(len(sc.frame_times), dtype=bool),
    )
    media = {s: StreamMedia(pts, RENDER[s]) for s in RENDER}
    return Scene(_session(sc.duration_ms), labels, media, list(sc.frame_times))


def images_for(unit: PlannedUnit, policy: AiReviewPolicy) -> tuple[PromptImage, ...]:
    """단위 프레임 수만큼 단색 합성 프레임을 그려 보낼 이미지를 만든다."""
    w, h = SIZE
    frames = [np.full((h, w, 3), 40 + 10 * (k % 10), dtype=np.uint8) for k in unit.frame_ms]
    return render_unit_images(unit, frames, policy)


def with_policy(policy: AiReviewPolicy, **sections: dict[str, Any]) -> AiReviewPolicy:
    """절 단위로 값을 바꾼 정책 사본 (검증을 다시 거친다)."""
    data = policy.model_dump()
    for name, values in sections.items():
        data[name] = data[name] | values
    return AiReviewPolicy.model_validate(data)


@pytest.fixture(scope="session")
def policy() -> AiReviewPolicy:
    """저장소의 aireview 정책."""
    return load_policy(ROOT)


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    """저장소의 온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


@pytest.fixture(scope="session")
def scene() -> Scene:
    """합성 장면 (세션 범위, 바꾸지 않는다)."""
    return make_scene()
