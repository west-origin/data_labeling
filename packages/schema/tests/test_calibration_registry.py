"""카메라 내부 파라미터·작업자 손 크기 등록과 결정 규칙 (ADR 0045, `dlp_schema.calibration`).

SQLite 테스트 DB(`sqlite_db`)로 저장소 함수까지 본다. 추가 전용 트리거는 PostgreSQL 서비스 테스트
(`test_db.py`)에서 본다. 정답은 테스트가 등록한 값과 모듈 설명의 규칙이다.
"""

from __future__ import annotations

from datetime import timedelta, timezone

import pytest
from pydantic import ValidationError

from dlp_schema.calibration import (
    CalibrationMethod,
    HandSizeSource,
    resolve_intrinsics,
    resolve_palm,
)
from dlp_schema.db.calibration_repository import (
    CameraAssignmentConflictError,
    assign_session_camera,
    get_camera_calibrations,
    get_hand_size_invalidations,
    get_worker_hand_sizes,
    hand_size_records_from_session,
    invalidate_hand_sizes,
    register_camera_calibration,
    register_hand_size,
    session_intrinsics,
    session_palm,
    worker_session_ids,
)
from dlp_schema.db.repository import insert_session, insert_withdrawal, register_ontology
from dlp_schema.lineage import Withdrawal
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


def test_hand_size_invalidation_and_withdrawn_source_fall_back(ontology: Ontology) -> None:
    """무효 기록·출처 세션 사용 중지 뒤에는 그 등록을 건너뛰고 이전 유효 등록 → 정책으로 돌아간다.

    정답 (ADR 0052): 등록 하나 무효 → 그 이전 등록, 작업자 전체 무효 → 정책 기본값('p'), 무효 뒤 새
    등록은 다시 쓰인다. 추정의 출처 세션이 사용 중지되면 그 추정은 빠진다. 결정이 바뀌면 3D 입력
    해시(digest_input)가 바뀐다. 무효 기록은 멱등이고 잰 값에는 출처를 둘 수 없다.
    """
    eng = sqlite_db()
    s1, s2 = make_session("s001"), make_session("s002")
    t = [FIXED_TIME + timedelta(hours=i) for i in range(8)]
    with eng.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, s1)
        insert_session(conn, s2)
        measured, _ = register_hand_size(conn, "w01", 0.084, source=HandSizeSource.MEASURED,
                                         now=t[0])  # fmt: skip
        est, _ = register_hand_size(
            conn, "w01", 0.0712, source=HandSizeSource.ESTIMATED, now=t[1], n_samples=55,
            source_session_ids=["s002", "s001"], estimator="depth3d-x+pe1234",
        )  # fmt: skip
        [_, stored] = get_worker_hand_sizes(conn, "w01")
        assert stored.source_session_ids == ("s001", "s002") and stored.estimator
        assert hand_size_records_from_session(conn, "s001") == [est.record_id]
        before = session_palm(conn, s2)
        assert before.palm_m == 0.0712
        # 출처 세션 사용 중지 → 그 추정은 빠지고 잰 값으로
        insert_withdrawal(
            conn, Withdrawal(session_id="s001", reason="동의 철회", withdrawn_at=t[2])
        )
        after = session_palm(conn, s2)
        assert after.palm_m == 0.084 and after.digest_input() != before.digest_input()
        # 등록 하나 무효 → 정책 기본값 (남은 유효 등록 없음)
        inv, gone = invalidate_hand_sizes(conn, "w01", reason="잘못 잼", now=t[3],
                                          record_id=measured.record_id)  # fmt: skip
        assert inv is not None and [g.record_id for g in gone] == [measured.record_id]
        assert session_palm(conn, s2).source == "p"
        again, _ = invalidate_hand_sizes(conn, "w01", reason="중복", now=t[4],
                                         record_id=measured.record_id)  # fmt: skip
        assert again is None  # 멱등
        # 새로 잰 값은 쓰이고, 작업자 전체 무효 뒤에는 다시 정책 기본값
        register_hand_size(conn, "w01", 0.083, source=HandSizeSource.MEASURED, now=t[5])
        assert session_palm(conn, s2).palm_m == 0.083
        invalidate_hand_sizes(conn, "w01", reason="삭제 요청 #12", now=t[6])
        assert session_palm(conn, s2) == resolve_palm([], "w01")
        assert len(get_hand_size_invalidations(conn, "w01")) == 2
        # 무효가 된 값과 같은 값을 다시 등록하면 새 행을 쓴다 (무효 뒤 등록은 쓰인다)
        _, rewritten = register_hand_size(conn, "w01", 0.083, source=HandSizeSource.MEASURED,
                                          now=t[7])  # fmt: skip
        assert rewritten and session_palm(conn, s2).palm_m == 0.083
        with pytest.raises(KeyError):
            invalidate_hand_sizes(conn, "w01", reason="x", now=t[7], record_id="palm-nope")
    with pytest.raises(ValidationError), eng.begin() as conn:
        register_hand_size(conn, "w01", 0.08, source=HandSizeSource.MEASURED, now=t[7],
                           source_session_ids=["s001"])  # fmt: skip


def test_existing_session_without_camera_id_uses_assigned_camera(ontology: Ontology) -> None:
    """매니페스트에 camera_id가 없던 세션도 기기 연결 기록으로 등록된 값을 쓴다 (F-PMS-24).

    매니페스트에 다른 기기가 있으면 연결을 거부한다. 같은 기기 재연결은 멱등이다.
    """
    eng = sqlite_db()
    old = make_session("s001")
    tagged = make_session("s002", calibration=Calibration(camera_id="cam-x"))
    with eng.begin() as conn:
        register_ontology(conn, ontology)
        insert_session(conn, old)
        insert_session(conn, tagged)
        cal, _ = register_camera_calibration(conn, "gopro-1", K1, method=CalibrationMethod.IMPORTED,
                                             now=FIXED_TIME)  # fmt: skip
        assert session_intrinsics(conn, old).origin == "hfov"
        _, written = assign_session_camera(conn, old, "gopro-1", reason="촬영 기록부",
                                           now=FIXED_TIME)  # fmt: skip
        assert written and session_intrinsics(conn, old).origin == f"registry:{cal.calibration_id}"
        _, again = assign_session_camera(conn, old, "gopro-1", reason="재실행",
                                         now=FIXED_TIME + timedelta(hours=1))  # fmt: skip
        assert not again
        with pytest.raises(CameraAssignmentConflictError):
            assign_session_camera(conn, tagged, "gopro-1", reason="x", now=FIXED_TIME)


def test_undated_reregistration_does_not_replace_a_dated_one() -> None:
    """유효 시작 있는 잘못된 등록은 유효 시작 없이 다시 등록해도 대체되지 않고, 같은 유효 시작(다른
    시간대 표기 포함)으로 다시 등록해야 대체된다 (F-PMS-25, 규칙 고정. CLI는 가려짐을 알린다).
    """
    eng = sqlite_db()
    start = FIXED_TIME - timedelta(days=30)
    bad = K1.model_copy(update={"fx": 12000.0, "fy": 12000.0})
    session = make_session(calibration=Calibration(camera_id="cam-1"))
    with eng.begin() as conn:
        register_camera_calibration(conn, "cam-1", bad, method=CalibrationMethod.IMPORTED,
                                    now=start, valid_from=start)  # fmt: skip
        register_camera_calibration(conn, "cam-1", K1, method=CalibrationMethod.IMPORTED,
                                    now=start + timedelta(days=1))  # fmt: skip
        assert session_intrinsics(conn, session).intrinsics == bad
        kst = start.astimezone(timezone(timedelta(hours=9)))
        register_camera_calibration(conn, "cam-1", K1, method=CalibrationMethod.IMPORTED,
                                    now=start + timedelta(days=2), valid_from=kst)  # fmt: skip
        assert session_intrinsics(conn, session).intrinsics == K1
