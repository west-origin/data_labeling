"""FiftyOne 샘플 변환과 실제 MongoDB 드라이버 왕복 검사 (WP14).

정답 근거: VFR 합성 블러본(프레임 시각 `times`를 안다)에 공간 라벨은 프레임 시각(스트림 PTS)으로,
시간 구간 라벨은 마스터 시각(= 스트림 시각 + 오프셋)으로 두어, 어느 FiftyOne 프레임(1부터)에
들어가야 하는지 정해진다.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_active.curation import push_to_fiftyone, rates_info, score_fields, stream_sample
from dlp_active.policy import ActivePolicy, load_policy
from dlp_active.rates import correction_rates
from dlp_active.select import SessionScore
from dlp_fixtures.video import vfr_times, write_video
from dlp_media.pts import build_pts_index
from dlp_schema.labels import LabelRecord, Provenance, Source
from dlp_schema.session import Stream, StreamKind, SyncMethod
from dlp_schema.testing import make_label

ROOT = Path(__file__).resolve().parents[3]
W, H = 64, 48


@pytest.fixture(scope="module")
def policy() -> ActivePolicy:
    """저장소의 config/policies/active.yaml."""
    return load_policy(ROOT)


@pytest.fixture(scope="module")
def video(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[int]]:
    """VFR 검은 영상(약 1초, 64x48)과 그 프레임 시각."""
    times = vfr_times(np.random.default_rng(3), 1000)
    path = tmp_path_factory.mktemp("v") / "blurred.mp4"
    write_video(path, ((t, np.zeros((H, W, 3), np.uint8)) for t in times), width=W, height=H)
    return path, times


def labels(times: list[int], offset: int) -> list[LabelRecord]:
    """정답 라벨: 박스(times[3]에 하나, 프레임 사이 하나, outside 하나), 관절(times[5]), 행동(마스터
    times[3]~times[5]), 블러(빠져야 함)."""
    m = Provenance(source=Source.MODEL, model_version="m1")
    # 공간 라벨 키프레임은 스트림 PTS 시각, 시간 구간 라벨은 마스터 시각 (= 스트림 시각 + 오프셋)
    t3, t5 = times[3], times[5]
    m3, m5 = t3 + offset, t5 + offset
    return [
        make_label(
            {"kind": "box_track", "entity_id": "cup_1", "class_id": "cup",
             "keyframes": [{"t_ms": t3, "x": 16, "y": 12, "w": 32, "h": 24},
                           {"t_ms": t3 + 7, "x": 0, "y": 0, "w": 8, "h": 8},  # 프레임 사이 → 버림
                           {"t_ms": t5, "x": 0, "y": 0, "w": 8, "h": 8, "outside": True}]},
            label_id="box", stream_id="bodycam", provenance=m, confidence=0.7,
            t_start_ms=t3, t_end_ms=t5,
        ),
        make_label(
            {"kind": "keypoint_track", "entity_id": "hand_r", "skeleton": "hand21", "hand": "right",
             "keyframes": [{"t_ms": t5, "points": [{"x": 32, "y": 24, "visibility": 2}]
                            + [{"x": 0, "y": 0, "visibility": 0}] * 20}]},
            label_id="kp", stream_id="bodycam", t_start_ms=t5, t_end_ms=t5,
        ),
        make_label(
            {"kind": "action", "action_id": "a1", "hand": "right", "verb": "wipe",
             "t_approach_ms": m3, "t_end_ms": m5},
            label_id="act", t_start_ms=m3, t_end_ms=m5,
        ),
        make_label(
            {"kind": "blur_track", "target": "face",
             "keyframes": [{"t_ms": t3, "x": 1, "y": 1, "w": 5, "h": 5}]},
            label_id="blur", stream_id="bodycam", t_start_ms=t3, t_end_ms=t3,
        ),
    ]  # fmt: skip


def stream(offset: int) -> Stream:
    """시각 변환 시험용 독립 Stream (마스터 시각과 offset ms 차이).

    실제 Session의 기준 바디캠은 오프셋 0이어야 한다. 이 객체를 Session으로 검증하지는 않는다.
    """
    return Stream(
        stream_id="bodycam",
        kind=StreamKind.BODYCAM,
        uri="local://dlp-labeling/x.mp4",
        sync_method=SyncMethod.REFERENCE,
        offset_ms=offset,
    )


def sample(video: tuple[Path, list[int]], policy: ActivePolicy, offset: int = 250) -> Any:
    """정답 라벨로 만든 `FoSample` (순위 1, 점수 1.5)."""
    path, times = video
    score = SessionScore("s1", 1.5, {"correction_rate": 1.5}, 2, [("box_track/cup", 1.5)])
    return stream_sample(
        "s1",
        path,
        build_pts_index(path),
        (W, H),
        stream(offset),
        labels(times, offset),
        policy,
        score_fields(1, score),
    )


def test_labels_map_to_blurred_video_frames(
    video: tuple[Path, list[int]], policy: ActivePolicy
) -> None:
    """라벨이 PTS로 맞춘 블러본 프레임에 들어간다.

    정답: 박스는 4번째 프레임(times[3])만 (프레임 사이·outside 버림), 관절은 6번째 프레임, 행동은
    오프셋을 빼 [4, 6] 구간, 블러는 없음. 좌표는 영상 크기로 정규화.
    """
    s = sample(video, policy)
    # VFR 영상에서도 PTS로 맞춘다: 4번째·6번째 프레임 (FiftyOne 번호는 1부터)
    assert sorted(s.frames) == [4, 6]
    (det,) = s.frames[4]
    assert det.box == (0.25, 0.25, 0.5, 0.5)
    assert det.label == "cup" and det.verification == "unreviewed" and det.confidence == 0.7
    (kp,) = s.frames[6]
    assert kp.points is not None and kp.points[0] == (0.5, 0.5) and math.isnan(kp.points[1][0])
    (seg,) = s.temporal
    assert seg.support == (4, 6) and seg.label == "wipe"
    # 블러 라벨은 넣지 않는다
    assert all(x.kind != "blur_track" for xs in s.frames.values() for x in xs)
    assert s.fields["active_top_classes"] == ["box_track/cup"]


# FiftyOne DB에 연결하지 못했다고 볼 예외의 클래스 이름 (pymongo·FiftyOne 서비스, 이름으로 본다: 그
# 모듈을 import하는 것 자체가 실패할 수 있다). 내장 ConnectionError·TimeoutError도 연결 실패다.
CONNECTION_ERROR_NAMES = frozenset(
    {"ConnectionFailure", "ServerSelectionTimeoutError", "AutoReconnect", "NetworkTimeout",
     "ServiceException", "ServiceListenTimeout", "ServiceExecutableNotFound",
     "FiftyOneConfigError"}
)  # fmt: skip


def is_connection_error(exc: BaseException) -> bool:
    """FiftyOne 데이터베이스 연결 실패인가 (건너뛸 환경 문제). 그 밖은 실패로 둔다 (F-COE-16).

    원인 사슬(`__cause__`)도 본다: FiftyOne은 pymongo 연결 시간 초과를 ConnectionError로 감싼다.
    """
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(seen, ConnectionError | TimeoutError) or any(
            c.__name__ in CONNECTION_ERROR_NAMES for c in type(seen).__mro__
        ):
            return True
        seen = seen.__cause__
    return False


def _fiftyone_with_db() -> Any:
    """fiftyone 모듈. 설치되지 않았거나 데이터베이스에 연결하지 못하면 이 테스트를 건너뛴다.

    FiftyOne은 내장 MongoDB(mongod)를 띄우거나 `FIFTYONE_DATABASE_URI`로 붙는다. 컨테이너·
    샌드박스에서 mongod를 띄울 수 없으면 import나 첫 연결이 연결 예외로 실패한다. 그것은 이 저장소
    코드의 결함이 아니라 환경 문제라 건너뛴다 (드라이버 왕복은 `services` 표시 테스트가 본다).
    설치돼 있는데 import가 실패하는 경우(의존성 비호환 등)와 연결 밖의 예외는 건너뛰지 않고 실패한다
    (F-COE-16).
    """
    if importlib.util.find_spec("fiftyone") is None:
        pytest.skip("fiftyone 미설치 (make install-curation)")
    try:
        fo: Any = importlib.import_module("fiftyone")
        importlib.import_module("fiftyone.core.odm.database").get_db_conn()
    except Exception as exc:
        if not is_connection_error(exc):
            raise
        pytest.skip(f"FiftyOne 데이터베이스에 연결하지 못함 ({type(exc).__name__})")
    return fo


def test_only_connection_errors_skip_the_round_trip() -> None:
    """F-COE-16: 연결 예외(내장 ConnectionError, pymongo·FiftyOne 서비스 예외, 그것을 원인으로 감싼
    예외)만 건너뛰기 사유다. 의존성 비호환(ImportError·AttributeError·TypeError)과 그 밖의 오류는
    실패로 드러난다."""

    class ServerSelectionTimeoutError(Exception):
        """pymongo의 같은 이름 예외를 흉내 낸다 (이름으로 판정)."""

    wrapped = ConnectionError("Could not connect to `mongod`")
    wrapped.__cause__ = ServerSelectionTimeoutError()
    for exc in (wrapped, ServerSelectionTimeoutError(), TimeoutError()):
        assert is_connection_error(exc), exc
    for exc in (ImportError("numpy ABI"), AttributeError("x"), TypeError("y"), RuntimeError("z")):
        assert not is_connection_error(exc), exc


def test_push_to_fiftyone_round_trip(video: tuple[Path, list[int]], policy: ActivePolicy) -> None:
    """FiftyOne에 넣고 다시 읽어 필드·프레임 라벨·구간·검증 상태 필터가 그대로인지.

    fiftyone이 없거나 그 데이터베이스(내장 또는 `FIFTYONE_DATABASE_URI`의 MongoDB)에 연결하지 못하면
    건너뛴다 (`make install-curation`).
    """
    fo = _fiftyone_with_db()
    rates = correction_rates([], policy)
    name = f"dlp-test-{uuid.uuid4().hex[:8]}"
    ds = push_to_fiftyone(name, [sample(video, policy)], rates_info(rates))
    try:
        loaded = fo.load_dataset(name)
        assert len(loaded) == 1 and loaded.info["overall_correction_rate"] == 0.0
        s = loaded.first()
        assert s.session_id == "s1" and s.active_rank == 1 and s.active_score == 1.5
        dets = s.frames[4].labels.detections
        assert [d.label for d in dets] == ["cup"] and dets[0].label_id == "box"
        assert s.frames[6].keypoints.keypoints[0].label == "hand21/right"
        (seg,) = s.segments.detections
        assert seg.label == "action/wipe" and list(seg.support) == [4, 6]
        # 수정률 분석: 미검수 모델 라벨만 거르기
        view = loaded.filter_labels("frames.labels", fo.ViewField("verification") == "unreviewed")
        assert view.count("frames.labels.detections") == 1
    finally:
        ds.delete()


@pytest.mark.services
def test_patched_async_mongo_driver_round_trip() -> None:
    """보안 패치 드라이버 조합의 비동기 삽입/집계를 실제 DB로 확인한다 (ADR 0035)."""
    motor: Any = pytest.importorskip("motor.motor_asyncio")
    uri = os.environ.get("FIFTYONE_DATABASE_URI")
    if not uri:
        pytest.skip("외부 큐레이션 MongoDB URI 없음")
    name = "dlp-driver-test-" + uuid.uuid4().hex

    async def query() -> None:
        client: Any = motor.AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        try:
            await client[name].values.insert_many([{"value": 1}, {"value": 2}])
            result = (
                await client[name]
                .values.aggregate([{"$group": {"_id": None, "total": {"$sum": "$value"}}}])
                .to_list(length=1)
            )
            assert result == [{"_id": None, "total": 3}]
        finally:
            await client.drop_database(name)
            client.close()

    asyncio.run(query())
