"""카메라 내부 파라미터·작업자 손 크기 등록과 결정 규칙 (ADR 0045, `dlp_schema.calibration`).

SQLite 테스트 DB(`sqlite_db`)로 저장소 함수까지 본다. 추가 전용 트리거는 PostgreSQL 서비스 테스트
(`test_db.py`)에서 본다. 정답은 테스트가 등록한 값과 모듈 설명의 규칙이다.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from pydantic import ValidationError

from dlp_schema.calibration import (
    CalibrationMethod,
    HandSizeSource,
    resolve_intrinsics,
    resolve_palm,
)
from dlp_schema.db.calibration_repository import (
    get_camera_calibrations,
    get_worker_hand_sizes,
    register_camera_calibration,
    register_hand_size,
    session_intrinsics,
    session_palm,
    worker_session_ids,
)
from dlp_schema.db.repository import insert_session, register_ontology
from dlp_schema.ontology import Ontology
from dlp_schema.session import Calibration, CameraIntrinsics
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

K1 = CameraIntrinsics(width=1920, height=1080, fx=1400.0, fy=1400.0, cx=960.0, cy=540.0)
K2 = K1.model_copy(update={"fx": 1300.0, "fy": 1300.0})
K3 = K1.model_copy(update={"fx": 1200.0, "fy": 1200.0})


def test_session_value_then_registered_camera_then_hfov() -> None:
    """세션에 직접 적힌 값 → 등록된 기기 값 → 없음. 유효 시작이 녹화 뒤인 등록은 쓰지 않고, 유효
    시작이 있는 것 중 가장 늦은 것(없는 것보다 앞섬), 같으면 늦게 등록한 것을 쓴다.
    """
    eng = sqlite_db()
    later, earlier = FIXED_TIME + timedelta(days=1), FIXED_TIME - timedelta(days=1)
    with eng.begin() as conn:
        a, _ = register_camera_calibration(conn, "cam-1", K1, method=CalibrationMethod.IMPORTED,
                                           now=FIXED_TIME)  # fmt: skip
        b, _ = register_camera_calibration(conn, "cam-1", K2, method=CalibrationMethod.IMPORTED,
                                           now=FIXED_TIME, valid_from=earlier)  # fmt: skip
        register_camera_calibration(conn, "cam-1", K3, method=CalibrationMethod.IMPORTED,
                                    now=FIXED_TIME, valid_from=later)  # fmt: skip
        records = get_camera_calibrations(conn, "cam-1")
    assert len(records) == 3
    tagged = make_session(calibration=Calibration(camera_id="cam-1"))
    got = resolve_intrinsics(tagged, records)
    assert got.intrinsics == K2 and got.origin == f"registry:{b.calibration_id}"
    # 유효 시작이 없는 등록만 있으면 그것 (a)
    only_a = [r for r in records if r.valid_from is None]
    assert resolve_intrinsics(tagged, only_a).origin == f"registry:{a.calibration_id}"
    direct = make_session(calibration=Calibration(camera_id="cam-1", intrinsics=K3))
    assert resolve_intrinsics(direct, records).origin == "session"
    none = resolve_intrinsics(make_session(), records)
    assert none.intrinsics is None and none.origin == "hfov"
    other = make_session(calibration=Calibration(camera_id="cam-2"))
    assert resolve_intrinsics(other, records).origin == "hfov"
    assert got.digest_input() != none.digest_input()


def test_registration_is_idempotent_and_newer_wins() -> None:
    """같은 기기·유효 시작에 같은 값을 다시 등록하면 쓰지 않는다. 다른 값을 늦게 등록하면 그것이
    앞선다. 손 크기도 최신 값·출처가 같으면 쓰지 않고, 바뀌면 새 값이 앞선다. 범위 밖 값은 거부.
    """
    eng = sqlite_db()
    with eng.begin() as conn:
        _, first = register_camera_calibration(
            conn, "cam-1", K1, method=CalibrationMethod.CHECKERBOARD, now=FIXED_TIME, rms_px=0.2,
            n_views=12, source_sha256="a" * 64,
        )  # fmt: skip
        _, again = register_camera_calibration(
            conn, "cam-1", K1, method=CalibrationMethod.CHECKERBOARD,
            now=FIXED_TIME + timedelta(hours=1),
        )  # fmt: skip
        newer, _ = register_camera_calibration(
            conn, "cam-1", K2, method=CalibrationMethod.CHECKERBOARD,
            now=FIXED_TIME + timedelta(hours=2),
        )  # fmt: skip
        records = get_camera_calibrations(conn, "cam-1")
        tagged = make_session(calibration=Calibration(camera_id="cam-1"))
        assert resolve_intrinsics(tagged, records).intrinsics == K2
        assert records[0].rms_px == 0.2 and records[0].created_at.tzinfo is not None

        _, w1 = register_hand_size(conn, "w01", 0.08, source=HandSizeSource.MEASURED,
                                   now=FIXED_TIME)  # fmt: skip
        _, w2 = register_hand_size(conn, "w01", 0.08, source=HandSizeSource.MEASURED,
                                   now=FIXED_TIME + timedelta(hours=1))  # fmt: skip
        register_hand_size(conn, "w01", 0.083, source=HandSizeSource.ESTIMATED,
                           now=FIXED_TIME + timedelta(hours=2), n_samples=50)  # fmt: skip
        sizes = get_worker_hand_sizes(conn, "w01")
    assert first and not again and newer.intrinsics == K2 and len(records) == 2
    assert w1 and not w2 and [h.palm_m for h in sizes] == [0.08, 0.083]
    palm = resolve_palm(sizes, "w01")
    assert palm.palm_m == 0.083 and palm.source == "w"
    assert resolve_palm(sizes, "w02").source == "p" and resolve_palm(sizes, "w02").palm_m is None
    assert "w01" not in palm.digest_input()  # 작업자 ID는 해시 입력에도 넣지 않는다
    with pytest.raises(ValidationError), eng.begin() as conn:
        register_hand_size(conn, "w01", 0.5, source=HandSizeSource.MEASURED, now=FIXED_TIME)


def test_session_lookup_reads_the_registry(ontology: Ontology) -> None:
    """DB 세션에서 내부 파라미터·손 크기를 찾는다 (작업자 세션 목록 포함)."""
    eng = sqlite_db()
    session = make_session("s001", calibration=Calibration(camera_id="cam-9"))
    with eng.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, session)
        insert_session(conn, make_session("s002", worker_id="w02"))
        assert session_intrinsics(conn, session).origin == "hfov"
        assert session_palm(conn, session).source == "p"
        register_camera_calibration(conn, "cam-9", K1, method=CalibrationMethod.IMPORTED,
                                    now=FIXED_TIME)  # fmt: skip
        register_hand_size(conn, "w01", 0.085, source=HandSizeSource.MEASURED, now=FIXED_TIME)
        assert session_intrinsics(conn, session).intrinsics == K1
        assert session_palm(conn, session).palm_m == 0.085
        assert worker_session_ids(conn, "w01") == ["s001"]
