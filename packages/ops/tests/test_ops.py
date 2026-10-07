"""완료 기준 (WP16): 합성 운영 이벤트로 지표 값이 기대와 일치한다.

DB 테스트는 PostgreSQL(make up)이 필요하다. 경고 규칙과 감사 리포트 판정은 순수 함수로 확인한다.

구성:
- 순수 함수 (DB 없음, `make check`에서 돈다): 주·달 경계, 경고 규칙, 원본 접근 감사 판정,
  블러 검수자 선택, 검증 완료 시각(`verified_at`) 추정.
- DB (`@pytest.mark.services`, `make test-services`): 테스트마다 임시 DB를 만들어
  마이그레이션·온톨로지 등록 뒤 합성 레코드를 넣고 `weekly_metrics`·CLI(`dlp ops
  log-work|privacy-audit`)·`retention_status`를 확인한다. 끝나면 DB를 지운다.

시간 기준: `WEEK` = 2026-W41(10-05 월 ~ 10-11 일), `IN` = 그 주 수요일 9시 UTC, `BEFORE` = 그 1주
전. 정답은 테스트가 직접 넣은 레코드 수에서 손으로 계산한 값이다 (각 단언 옆 주석).
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_media.pts import PtsIndex
from dlp_media.storage import LocalStore, sha256_file
from dlp_ops.audit import audit_report, blur_reviewers, month_range
from dlp_ops.metrics import (
    WeeklyMetrics,
    alerts,
    verified_at,
    week_of,
    week_range,
    weekly_metrics,
)
from dlp_ops.policy import OpsPolicy, load_policy
from dlp_ops.retention import retention_status
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_assignment,
    insert_golden_set,
    insert_labels,
    insert_privacy_audit,
    insert_retention_decision,
    insert_review_work,
    insert_session,
    insert_withdrawal,
    list_review_work,
    register_ontology,
    set_lifecycle,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.lineage import GoldenSet, Withdrawal
from dlp_schema.ontology import load_ontology
from dlp_schema.ops import PrivacyAuditRecord, RawAccessEvent, RetentionDecision, ReviewWork
from dlp_schema.review import AssignmentStatus, InjectedError, ReviewAssignment, ReviewMode
from dlp_schema.session import Domain, LifecycleState, Stream, StreamKind
from dlp_schema.testing import make_label, make_session

ROOT = Path(__file__).resolve().parents[3]
WEEK = "2026-W41"  # 2026-10-05(월) ~ 10-11
IN = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)  # WEEK 안의 시각 (수요일 9시 UTC)
BEFORE = IN - timedelta(days=7)  # 지난주(W40) 같은 시각 — "그 주가 아님"을 만들 때


@pytest.fixture(scope="module")
def policy() -> OpsPolicy:
    """저장소의 실제 `config/policies/ops.yaml` (모듈 범위에서 한 번 읽는다)."""
    return load_policy(ROOT)


def test_week_and_month_ranges() -> None:
    """ISO 주 → UTC [월요일, 다음 월요일), 시각 → 주 문자열, 달 경계(12월 → 다음 해 1월)를
    확인한다. 정책 시간대(Asia/Seoul)의 달 경계는 UTC로 바꾸면 전날 15시다.
    """
    start, end = week_range(WEEK)
    assert (start, end) == (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC))
    assert week_of(IN) == WEEK and week_of(BEFORE) == "2026-W40"
    assert month_range("2026-12") == (
        datetime(2026, 12, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )  # fmt: skip
    # 정책 시간대(서울, UTC+9)의 달: 11월 30일 15시 UTC부터
    assert month_range("2026-12", ZoneInfo("Asia/Seoul")) == (
        datetime(2026, 11, 30, 15, tzinfo=UTC), datetime(2026, 12, 31, 15, tzinfo=UTC)
    )  # fmt: skip


def test_quality_alerts(policy: OpsPolicy) -> None:
    """경고 규칙 세 가지를 합성 `WeeklyMetrics`로 확인한다.

    - 자동 승인율 +10%p, 발견율 -20%p → "검수 품질 저하" 하나만 (임계값 0.02/0.05 초과).
    - 검수 시간이 4주(`stagnation_weeks`) 동안 늘고 수정률이 그대로 → "가이드라인" 경고.
      둘 다 줄면 경고 없음.
    - 블러 검수 시간이 있는데 감사 0건 → "감사가 없다" 경고. 감사 1건이면 없음.
    """

    def w(week: str, minutes: float, corr: float, auto: float, det: float) -> WeeklyMetrics:
        """경고 판정에 필요한 네 지표만 채운 `WeeklyMetrics`를 만든다."""
        return WeeklyMetrics(week, review_minutes_per_video_hour=minutes, correction_rate=corr,
                             auto_approval_rate=auto, seeded_detection_rate=det)  # fmt: skip

    # 자동 승인율 상승 + 발견율 하락 → 검수 품질 저하 경고
    got = alerts([w("W1", 60, 0.3, 0.60, 0.9), w("W2", 50, 0.2, 0.70, 0.7)], policy)
    assert len(got) == 1 and "검수 품질 저하" in got[0]
    # 검수 시간·수정률이 4주 동안 줄지 않음 → 가이드라인·온톨로지 점검 경고
    flat = [w(f"W{i}", 60 + i, 0.3, 0.6, 0.9) for i in range(4)]
    assert any("가이드라인" in x for x in alerts(flat, policy))
    improving = [w(f"W{i}", 60 - 5 * i, 0.3 - 0.02 * i, 0.6, 0.9) for i in range(4)]
    assert alerts(improving, policy) == []
    # 블러 검수가 있었는데 그 주 감사가 없으면 경고, 감사가 있으면 없다
    no_audit = WeeklyMetrics("W9", privacy_review_minutes_per_video_hour=30.0)
    assert any("감사가 없다" in x for x in alerts([no_audit], policy))
    audited = WeeklyMetrics("W9", privacy_review_minutes_per_video_hour=30.0,
                            counts={"privacy_audits": 1})  # fmt: skip
    assert alerts([audited], policy) == []


def event(actor: str, action: str, hour: int, purpose: str = "privacy.detect") -> RawAccessEvent:
    """2026-10-07 `hour`시(UTC)의 원본 접근 이벤트 하나 (세션 `s1`, 바디캠 원본 키)."""
    return RawAccessEvent(
        event_id=uuid.uuid4().hex, at=datetime(2026, 10, 7, hour, tzinfo=UTC), actor=actor,
        purpose=purpose, action=action, bucket="dlp-raw",  # type: ignore[arg-type]
        key="sessions/s1/bodycam.mp4", session_id="s1",
    )  # fmt: skip


def test_audit_report_flags(policy: OpsPolicy) -> None:
    # 시각은 UTC. 서울(UTC+9) 기준 업무 시간 밖 = 22시~6시 = UTC 13시~21시
    """원본 접근 감사 판정 규칙을 이벤트 8건으로 확인한다 (각 이벤트 옆 주석이 기대 판정).

    권한자는 `rev-a` 하나, 서비스 계정은 정책의 `svc-pipeline`. 서울 기준 업무 시간 밖(22~6시)은
    UTC 13~21시다. 기대: 표시 5종이 각각 정확히 한 번 (정렬해 비교), 동작별 건수와 세션 수 1.
    """
    events = [
        event("svc-pipeline", "read", 15),  # 서비스 계정: 시간과 무관하게 정상
        event("rev-a", "grant", 2, "review.create"),  # 권한자에게 보여 줌: 정상
        event("rev-a", "presign", 3),  # 권한자 열람 (서울 12시): 정상
        event("intruder", "read", 4),  # 권한 없는 사람의 열람
        event("rev-b", "grant", 5, "review.create"),  # 권한 없는 사람에게 보여 줌
        event("unassigned", "grant", 5, "review.create"),  # 담당자 없는 원본 작업
        event("rev-a", "read", 16),  # 권한자지만 서울 새벽 1시
        event("svc-pipeline", "read", 4, "debug.view"),  # 서비스 계정을 파이프라인 밖 용도로
    ]
    r = audit_report(events, "2026-10", policy, privacy_reviewers=("rev-a",))
    assert r.events == 8 and r.sessions == 1
    assert r.by_action == {"grant": 3, "presign": 1, "read": 4}
    kinds = sorted(f.split(":")[0] for f in r.flags)
    assert kinds == [
        "담당자 없이 원본 검수 작업을 올림",
        "서비스 계정을 파이프라인 밖 용도로 씀",
        "업무 시간 밖 원본 접근",
        "원본 권한 없는 사람에게 보여 줌",
        "원본 권한 없는 사람의 열람",
    ]


def test_blur_reviewers_use_current_tracks_only() -> None:
    """감사자 검사는 지금 렌더에 쓰인(운영 현재) 블러 트랙의 검수자 모두와 비교한다.

    정답: 고쳐진 `b1`의 `old-rev`, 오류 삽입 사본의 `rev-c`, 다른 스트림의 `rev-d`는 빠지고,
    트랙 2개의 `rev-a`가 트랙 1개의 `rev-b`보다 앞선다.
    """

    def blur(lid: str, who: str, **kw: Any) -> LabelRecord:
        """바디캠 스트림의 얼굴 블러 트랙 하나 (`who`가 `IN`에 승인)."""
        return make_label(
            {"kind": "blur_track", "target": "face",
             "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
            label_id=lid, session_id="s1", stream_id="bodycam",
            verification=checked(VerificationState.HUMAN_APPROVED, IN, who), **kw,
        )  # fmt: skip

    history = [
        blur("b1", "old-rev"),  # 고쳐져서 지금은 쓰이지 않는다
        blur("b1-fix", "rev-b", parent_label_id="b1"),
        blur("b2", "rev-a"),
        blur("b3", "rev-a"),
        blur("b4", "rev-c", seeded_error=True),  # 오류 삽입 사본 (운영 라벨 아님)
        blur("b5", "rev-d").model_copy(update={"stream_id": "third"}),  # 다른 스트림
    ]
    assert blur_reviewers(history, "bodycam") == ["rev-a", "rev-b"]


def test_verified_at_is_stable() -> None:
    """검증 완료 시각은 그 뒤의 QA 수정·새 모델 버전·재검수로 바뀌지 않는다.

    시나리오: 모델 라벨 a(t2에 승인), b(미검수) → 미완료(None). b를 사람이 t2에 고침 → t2.
    그 뒤 a의 QA 수정과 새 모델 라벨이 생겨도 t2 그대로.
    """
    sid = "v"
    t1, t2, later = BEFORE, IN, IN + timedelta(days=14)
    base = [
        model(sid, "a", t1, verification=checked(VerificationState.HUMAN_APPROVED, t2)),
        model(sid, "b", t1),  # 미검수
    ]
    assert verified_at(base) is None
    done = [*base, human(sid, "b-fix", t2, parent_label_id=f"{sid}-b")]
    assert verified_at(done) == t2
    after = [
        *done,
        human(sid, "a-qa", later, parent_label_id=f"{sid}-a"),  # 나중 QA 수정
        model(sid, "new", later),  # 새 모델 버전의 미검수 라벨
    ]
    assert verified_at(after) == t2


def test_verified_at_waits_for_staged_labels() -> None:
    """단계별 파이프라인: 객체 박스를 먼저 검수하고 2주 뒤 행동 구간이 생겨 검수되면, 완료 시각은
    행동 구간의 검수 시각이다 (먼저 검수한 단계만 보고 일찍 세지 않는다).

    추가로 확인: 이미 검수한 박스의 QA 수정은 시각을 바꾸지 않는다. 완료 시각 전에 생긴 미검수 모델
    라벨이 있으면 None. 사람이 모델 라벨을 지운 것도 검수라 그 시각까지 늦춘다. 오류 삽입 레코드는
    무시한다.
    """
    sid = "st"
    box = {"kind": "box_track", "entity_id": "e1", "class_id": "towel",
           "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 2, "h": 2}]}  # fmt: skip
    m = Provenance(source=Source.MODEL, model_version="m1")
    t0 = BEFORE - timedelta(days=14)
    a = make_label(
        box, label_id=f"{sid}-A", session_id=sid, stream_id="bodycam", provenance=m,
        confidence=0.9, created_at=t0,
        verification=checked(VerificationState.HUMAN_APPROVED, t0 + timedelta(hours=1)),
    )  # fmt: skip
    b = model(sid, "B", BEFORE)  # 2주 뒤 행동 단계
    b_ok = b.model_copy(update={"verification": checked(VerificationState.HUMAN_APPROVED, IN)})
    assert verified_at([a, b_ok]) == IN and week_of(IN) == WEEK
    # 이미 검수한 박스를 나중에 고친 QA는 완료 시각을 바꾸지 않는다
    qa = make_label(
        box, label_id=f"{sid}-A-qa", session_id=sid, stream_id="bodycam",
        parent_label_id=f"{sid}-A", created_at=IN + timedelta(days=10),
        verification=checked(VerificationState.HUMAN_CORRECTED, IN + timedelta(days=10)),
    )  # fmt: skip
    assert verified_at([a, b_ok, qa]) == IN
    # 마지막 검수 시각 전에 있던 미검수 모델 라벨이 남아 있으면 완료가 아니다
    late = model(sid, "late", IN - timedelta(hours=1))
    assert verified_at([a, b_ok, late]) is None
    # 검수 전 모델 라벨을 사람이 지운 것은 검수다 (그 시각까지 늦춘다)
    fp = model(sid, "fp", BEFORE)
    gone = human(sid, "fp-del", IN + timedelta(hours=2), parent_label_id=f"{sid}-fp",
                 retracted=True)  # fmt: skip
    assert verified_at([a, b_ok, fp, gone]) == IN + timedelta(hours=2)
    # 블러·오류 삽입 레코드는 보지 않는다
    seed = model(sid, "seed", BEFORE, seeded_error=True)
    assert verified_at([a, b_ok, seed]) == IN


# ---------------------------------------------------------------- DB (make up)


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """임시 PostgreSQL 데이터베이스 (`make up` 필요).

    `DLP_DATABASE_URL`(없으면 개발 기본값) 서버에 `dlp_test_<임의>` DB를 만들고, Alembic 최신까지
    올린 뒤 온톨로지 v1을 등록한 엔진을 준다. 테스트가 끝나면 연결을 끊고 DB를 강제로 지운다.
    """
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def action(verb: str = "carry") -> dict[str, Any]:
    """오른손 `carry` 행동 구간 페이로드 (0~900 ms, 마스터 타임라인). `verb`로 동사를 바꾼다."""
    return {"kind": "action", "action_id": "a", "hand": "right", "verb": verb,
            "t_approach_ms": 0, "t_end_ms": 900}  # fmt: skip


def checked(state: VerificationState, at: datetime, who: str = "rev-1") -> Verification:
    """`who`가 `at`에 `state`로 검수한 `Verification`."""
    return Verification(state=state, reviewer_id=who, reviewed_at=at)


def model(sid: str, lid: str, at: datetime, **kw: Any) -> LabelRecord:
    """세션 `sid`의 모델 출처 행동 라벨 (`model_version=m1`, 신뢰도 0.8). 라벨 ID는
    `<sid>-<lid>`."""
    return make_label(
        action(), label_id=f"{sid}-{lid}", session_id=sid, t_end_ms=900, created_at=at,
        provenance=Provenance(source=Source.MODEL, model_version="m1"), confidence=0.8, **kw,
    )  # fmt: skip


def human(sid: str, lid: str, at: datetime, **kw: Any) -> LabelRecord:
    """세션 `sid`의 사람 출처 라벨. 기본 검수 상태는 `at`의 `human_corrected`, 페이로드는
    `action()`."""
    payload = kw.pop("payload", action())
    kw.setdefault("verification", checked(VerificationState.HUMAN_CORRECTED, at))
    return make_label(
        payload, label_id=f"{sid}-{lid}", session_id=sid, t_end_ms=900, created_at=at, **kw
    )


@pytest.mark.services
def test_weekly_metrics_from_synthetic_events(pg: sa.Engine, policy: OpsPolicy) -> None:
    """합성 운영 이벤트로 `weekly_metrics`의 모든 지표를 손계산 값과 맞춘다.

    정답 근거 (각 블록 옆 주석):
    - 그 주 검수: 승인 6, 수정 2, 삭제 1, 추가 1 → 수정률 4/10, 자동 승인율 6/9.
      지난주 승인·블러·오류 삽입·골든셋·사용 중지 세션의 검수는 세지 않는다.
    - 오류 삽입 2개 중 1개를 되돌림 → 발견율 0.5.
    - 검수 시간: 작업 라벨 50분 / 1.5시간 영상, 블러 10분 / 10분 영상 = 60분/시간. 지난주 기록 제외.
    - 잔여 블러 누락: 1시간짜리 감사 2건에서 누락 1 → 0.5/시간.
    - 검증 에피소드: ops-a(라벨 이력 추정), 골든 세션, ops-v(생애주기 기록, ADR 0028) = 3.
    - 원가: 검수 1시간 * 30000원 / 3 에피소드 = 10000원.
    마지막으로 `review_work`는 DB 트리거로 UPDATE가 막혀 있다(추가만, ADR 0020).
    """
    review_policy = load_review_policy(ROOT)
    sid, old = "ops-a", "ops-b"
    approved = VerificationState.HUMAN_APPROVED
    labels = (
        # 그 주: 승인 6, 수정 2, 삭제 1, 추가 1 → 수정률 4/10, 자동 승인율 6/9
        [model(sid, f"ok{i}", BEFORE, verification=checked(approved, IN)) for i in range(6)]
        + [model(sid, f"c{i}", BEFORE) for i in range(2)]
        + [human(sid, f"c{i}-fix", IN, parent_label_id=f"{sid}-c{i}") for i in range(2)]
        + [model(sid, "fp", BEFORE),
           human(sid, "fp-del", IN, parent_label_id=f"{sid}-fp", retracted=True),
           human(sid, "add", IN)]
        # 지난주 승인 (제외), 블러 승인 (제외), 오류 삽입 사본 (제외)
        + [model(sid, "last", BEFORE, verification=checked(approved, BEFORE)),
           make_label({"kind": "blur_track", "target": "face",
                       "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
                      label_id=f"{sid}-blur", session_id=sid, stream_id="bodycam",
                      provenance=Provenance(source=Source.MODEL, model_version="b"),
                      confidence=0.9, verification=checked(approved, IN), created_at=BEFORE)]
        # 오류 삽입: 원본(사람), 클래스를 바꾼 사본 2개, 검수자가 하나만 되돌림 → 발견율 1/2
        + [human(sid, "orig", BEFORE),
           human(sid, "seed1", BEFORE, seeded_error=True, payload=action("lift"),
                 verification=Verification()),
           human(sid, "seed2", BEFORE, seeded_error=True, payload=action("lift"),
                 verification=Verification()),
           human(sid, "seed1-fix", IN, seeded_error=True, parent_label_id=f"{sid}-seed1",
                 verification=checked(VerificationState.HUMAN_CORRECTED, IN, "rev-1"))]
    )  # fmt: skip
    gold, gone = "ops-g", "ops-w"
    with pg.begin() as conn:
        for s, state in (
            (sid, LifecycleState.HUMAN_VERIFIED),
            (old, LifecycleState.HUMAN_VERIFIED),
            (gold, LifecycleState.HUMAN_VERIFIED),
            (gone, LifecycleState.WITHDRAWN),
        ):
            insert_session(conn, make_session(s).model_copy(update={"lifecycle_state": state}))
        insert_labels(conn, labels)
        # 골든셋 정답(사람이 처음부터 만든 것)과 사용 중지 세션의 검수는 수정률에 넣지 않는다
        insert_labels(conn, [human(gold, f"g{i}", IN) for i in range(5)])
        insert_golden_set(
            conn,
            GoldenSet(version="g1", domain=Domain.CLEANING, session_ids=(gold,), created_at=IN),
        )
        insert_labels(conn, [model(gone, "x", BEFORE, verification=checked(approved, IN))])
        insert_withdrawal(conn, Withdrawal(session_id=gone, reason="동의 철회", withdrawn_at=IN))
        # 다른 세션: 마지막 검수가 지난주 → 이번 주 검증 에피소드 아님
        insert_labels(conn, [model(old, "x", BEFORE, verification=checked(approved, BEFORE))])
        errors = tuple(
            InjectedError(error_type="class_swap", original_label_id=f"{sid}-orig",
                          seeded_label_id=f"{sid}-seed{i}",
                          detail={"field": "verb", "original": "carry"})
            for i in (1, 2)
        )  # fmt: skip
        insert_assignment(
            conn,
            ReviewAssignment(
                assignment_id="seeded-1", session_id=sid, label_kinds=("action",),
                mode=ReviewMode.SEEDED_ERROR, priority=1.0, assignee="rev-1", injected=errors,
                status=AssignmentStatus.DONE, created_at=BEFORE, completed_at=IN,
            ),
        )  # fmt: skip
        # 검수 시간: 작업 라벨 (30분 + 20분) / 1.5시간 영상, 블러 10분 / 10분 영상
        # (지난주 기록은 제외)
        for wid, stage, sec, vms, at in [
            ("w1", "labeling", 1800, 3_600_000, IN), ("w2", "qa", 1200, 1_800_000, IN),
            ("w3", "privacy", 600, 600_000, IN), ("w0", "labeling", 9999, 1000, BEFORE),
        ]:  # fmt: skip
            insert_review_work(conn, ReviewWork(
                work_id=wid, session_id=sid, reviewer="rev-1", stage=stage,  # type: ignore[arg-type]
                seconds=sec, video_ms=vms, source="manual", recorded_at=at,
            ))  # fmt: skip
        # 잔여 블러 감사: 2시간 영상에서 누락 1 → 0.5 / 시간 (지난주 감사는 제외)
        for aid, misses, at in [("a1", 1, IN), ("a2", 0, IN), ("a0", 9, BEFORE)]:
            insert_privacy_audit(conn, PrivacyAuditRecord(
                audit_id=aid, session_id=sid, stream_id="bodycam", duration_ms=3_600_000,
                misses=misses, auditor="aud-1", blur_reviewer="rev-1", audited_at=at,
            ))  # fmt: skip

        # 생애주기 기록이 있으면 그 전이 시각이 검증 주다 (ADR 0028): 라벨 검수는 지난주였어도
        # 이번 주에 dlp review verify로 human_verified가 됐으면 이번 주 에피소드다
        insert_session(
            conn,
            make_session("ops-v").model_copy(update={"lifecycle_state": LifecycleState.PRELABELED}),
        )
        insert_labels(conn, [model("ops-v", "x", BEFORE, verification=checked(approved, BEFORE))])
        set_lifecycle(conn, "ops-v", LifecycleState.HUMAN_VERIFIED, at=IN, actor="lead")

    priced = policy.model_copy(
        update={"cost": policy.cost.model_copy(update={"hourly_cost": 30000.0})}
    )
    with pg.connect() as conn:
        m = weekly_metrics(conn, WEEK, review_policy, priced)
    assert m.counts["accepted"] == 6 and m.counts["corrected"] == 2
    assert (m.counts["deleted"], m.counts["added"]) == (1, 1)
    assert m.correction_rate == pytest.approx(4 / 10)
    assert m.auto_approval_rate == pytest.approx(6 / 9)
    assert m.review_minutes_per_video_hour == pytest.approx(50 / 1.5)
    assert m.privacy_review_minutes_per_video_hour == pytest.approx(60.0)
    assert m.seeded_detection_rate == pytest.approx(0.5)
    assert m.residual_blur_miss_per_hour == pytest.approx(0.5)
    # ops-a, 골든 세션, ops-v(생애주기 기록) — ops-b는 지난주, 사용 중지 제외
    assert m.verified_episodes == 3
    assert m.review_hours == pytest.approx(1.0)
    assert m.cost_per_episode == pytest.approx(10000.0)
    assert math.isnan(m.prelabel_bias)  # 이 주에 끝난 블라인드 배정 없음
    assert m.as_dict()["prelabel_bias"] is None

    # 감사·운영 기록은 고치거나 지울 수 없다
    with pytest.raises(sa.exc.DBAPIError, match="추가만"), pg.begin() as conn:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
        conn.execute(sa.text("UPDATE review_work SET seconds = 0"))


@pytest.mark.services
def test_cli_log_work_and_privacy_audit(
    pg: sa.Engine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """log-work --at은 검수한 주에 센다. 출처는 작업 키 접두사의 도구다.

    privacy-audit 감사자는 현재 블러 트랙의 모든 검수자와 다르다. 길이는 감사한 스트림의 길이다.

    정답: 서울 시각 10시는 UTC 1시라 W41에 기록된다. 소수 검수자 `rev-b`도 감사자가 될 수 없고,
    `DLP_ACTOR=aud-1`로 감사하면 기록의 `blur_reviewer`는 트랙이 많은 `rev-a`다.
    회귀: `label_studio:3` 작업 키의 출처가 `cvat`으로 기록됐다 → `label_studio`.
    모르는 접두사는 거부.
    회귀: 3인칭 감사의 길이가 세션 길이(60초)였다 → 3인칭 PTS 인덱스 길이(30초).
    """
    from dlp_cli.main import main
    from dlp_schema.db.repository import list_privacy_audits

    url = pg.url.render_as_string(hide_password=False)
    # 3인칭 PTS 인덱스(100 ms 간격 300프레임 → 30초)를 로컬 원본 저장소에 둔다
    raw = LocalStore(tmp_path, "dlp-raw")
    index = PtsIndex(
        Fraction(1, 1000), np.arange(0, 30_000, 100, dtype=np.int64), np.ones(300, dtype=bool)
    )
    index.write(tmp_path / "tp.pts.parquet")
    pts_key = "sessions/cli-1/derived/tp1.pts.parquet"
    raw.put_file(pts_key, tmp_path / "tp.pts.parquet", sha256_file(tmp_path / "tp.pts.parquet"))
    base = make_session("cli-1")
    third = Stream(stream_id="tp1", kind=StreamKind.THIRD_PERSON, uri=raw.uri("raw/tp1.mp4"),
                   pts_index_uri=raw.uri(pts_key))  # fmt: skip
    with pg.begin() as conn:
        insert_session(conn, base.model_copy(update={"streams": (*base.streams, third)}))
        insert_labels(conn, [
            make_label({"kind": "blur_track", "target": "face",
                        "keyframes": [{"t_ms": 0, "x": 1, "y": 1, "w": 5, "h": 5}]},
                       label_id=f"cli-1-b{i}", session_id="cli-1", stream_id=stream,
                       verification=checked(VerificationState.HUMAN_APPROVED, IN, who))
            for i, (who, stream) in enumerate(
                [("rev-a", "bodycam"), ("rev-a", "bodycam"), ("rev-b", "bodycam"),
                 ("rev-a", "tp1")]
            )
        ])  # fmt: skip
    seoul = "2026-10-07T10:00:00+09:00"
    assert main(["ops", "log-work", "cli-1", "--reviewer", "rev-a", "--stage", "privacy",
                 "--minutes", "10", "--at", seoul, "--url", url]) == 0  # fmt: skip
    with pg.connect() as conn:
        (w,) = list_review_work(conn)
    assert w.recorded_at == datetime(2026, 10, 7, 1, tzinfo=UTC) and week_of(w.recorded_at) == WEEK
    assert w.source == "manual"  # 작업 키 없음
    for key, source in (("label_studio:3", "label_studio"), ("cvat:7", "cvat")):
        assert main(["ops", "log-work", "cli-1", "--reviewer", "rev-a", "--stage", "labeling",
                     "--minutes", "5", "--task-key", key, "--url", url]) == 0  # fmt: skip
        with pg.connect() as conn:
            assert {x.source for x in list_review_work(conn) if x.task_key == key} == {source}
    with pytest.raises(SystemExit, match="도구를 알 수 없습니다"):
        main(["ops", "log-work", "cli-1", "--reviewer", "rev-a", "--stage", "labeling",
              "--minutes", "5", "--task-key", "jira:1", "--url", url])  # fmt: skip
    # 검수자 중 소수(rev-b)도 감사자가 될 수 없다
    with pytest.raises(SystemExit, match="블러 검수자"):
        main(["ops", "privacy-audit", "cli-1", "bodycam", "--misses", "0", "--auditor", "rev-b",
              "--url", url])  # fmt: skip
    monkeypatch.setenv("DLP_ACTOR", "aud-1")
    assert main(["ops", "privacy-audit", "cli-1", "bodycam", "--misses", "1", "--url", url]) == 0
    with pg.connect() as conn:
        (a,) = list_privacy_audits(conn)
    assert (a.auditor, a.blur_reviewer, a.misses) == ("aud-1", "rev-a", 1)
    assert a.duration_ms == 60_000  # 바디캠 = 세션 길이
    assert main(["ops", "privacy-audit", "cli-1", "tp1", "--misses", "0", "--url", url,
                 "--store", f"local:{tmp_path}"]) == 0  # fmt: skip
    with pg.connect() as conn:
        (tp,) = [x for x in list_privacy_audits(conn) if x.stream_id == "tp1"]
    assert tp.duration_ms == 30_000  # 그 스트림(3인칭)의 PTS 인덱스 길이
    with pytest.raises(SystemExit, match="영상 스트림이 아닙니다"):
        main(["ops", "privacy-audit", "cli-1", "imu", "--misses", "0", "--url", url])


@pytest.mark.services
def test_retention_alerts(pg: sa.Engine) -> None:
    """원본 보관 만료 상태를 세션별로 확인한다 (기준일 2027-10-01, 보관 365일, 30일 전부터 알림).

    정답 근거:
    - r-old: 2026-09-01 확정 → 2027-09-01 만료 → `expired`. 그 뒤의 모델·철회·오류 삽입 레코드는
      기산점을 늦추지 않는다.
    - r-soon: 2026-10-20 + 365일 = 2027-10-20 → 30일 안 → `due_soon`.
    - r-new: 2027-06-01 확정 → `ok`. r-ext: 2027-12-31까지 연장 → `extended`.
    - r-del·r-gone-del: 삭제 결정 → `delete_decided`. r-gone: 사용 중지, 결정 없음 → `withdrawn`.
    - r-wip: 검수 완료 전이라 목록에 없다. 보관 기간 미정이면 사용 중지 세션만 보인다.
    """
    today = date(2027, 10, 1)
    finished = {
        "r-old": datetime(2026, 9, 1, tzinfo=UTC),
        "r-soon": datetime(2026, 10, 20, tzinfo=UTC),
        "r-new": datetime(2027, 6, 1, tzinfo=UTC),
        "r-ext": datetime(2026, 1, 1, tzinfo=UTC),
        "r-del": datetime(2026, 1, 1, tzinfo=UTC),
    }
    late = datetime(2027, 9, 20, tzinfo=UTC)
    with pg.begin() as conn:
        for sid, at in finished.items():
            insert_session(
                conn,
                make_session(sid).model_copy(update={"lifecycle_state": LifecycleState.EXPORTED}),
            )
            insert_labels(
                conn,
                [model(sid, "x", at, verification=checked(VerificationState.HUMAN_APPROVED, at))],
            )
        # 나중의 모델 레코드(새 버전의 미검수 라벨, 버전 교체로 지운 레코드)와 오류 삽입 레코드는
        # 기산점을 늦추지 않는다
        insert_labels(conn, [
            model("r-old", "new", late),
            model("r-old", "x-retract", late, parent_label_id="r-old-new", retracted=True),
            human("r-old", "seed", late, seeded_error=True),
        ])  # fmt: skip
        for gone in ("r-gone", "r-gone-del"):
            insert_session(
                conn,
                make_session(gone).model_copy(update={"lifecycle_state": LifecycleState.WITHDRAWN}),
            )
        insert_session(
            conn,
            make_session("r-wip").model_copy(update={"lifecycle_state": LifecycleState.PRELABELED}),
        )
        for sid, kind, until in [
            ("r-ext", "extend", date(2027, 12, 31)),
            ("r-del", "delete", None),
            ("r-gone-del", "delete", None),
        ]:
            insert_retention_decision(conn, RetentionDecision(
                decision_id=f"d-{sid}", session_id=sid, decision=kind, until=until,  # type: ignore[arg-type]
                reason="재학습에 필요" if kind == "extend" else "보관 기간 만료",
                decided_by="ops-1", decided_at=datetime(2027, 9, 1, tzinfo=UTC),
            ))  # fmt: skip
    with pg.connect() as conn:
        items = {i.session_id: i for i in retention_status(conn, today, 365, 30)}
        undecided = retention_status(conn, today, None, 30)
    # 기간 미정이면 사용 중지 세션만 보인다
    assert {i.session_id: i.status for i in undecided} == {
        "r-gone": "withdrawn", "r-gone-del": "delete_decided",
    }  # fmt: skip
    assert {k: v.status for k, v in items.items()} == {
        "r-old": "expired", "r-soon": "due_soon", "r-new": "ok", "r-ext": "extended",
        "r-del": "delete_decided", "r-gone": "withdrawn", "r-gone-del": "delete_decided",
    }  # fmt: skip
    assert items["r-soon"].expires_on == date(2027, 10, 20)
    assert items["r-old"].finalized_at == finished["r-old"]
    assert {k for k, v in items.items() if v.alert} == {"r-old", "r-soon", "r-gone"}
