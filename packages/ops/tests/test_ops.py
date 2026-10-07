"""완료 기준 (WP16): 합성 운영 이벤트로 지표 값이 기대와 일치한다.

DB 테스트는 PostgreSQL(make up)이 필요하다. 경고 규칙과 감사 리포트 판정은 순수 함수로 확인한다.
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_ops.audit import audit_report, month_range
from dlp_ops.metrics import WeeklyMetrics, alerts, week_of, week_range, weekly_metrics
from dlp_ops.policy import OpsPolicy, load_policy
from dlp_ops.retention import retention_status
from dlp_review.ops.policy import load_policy as load_review_policy
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_assignment,
    insert_labels,
    insert_privacy_audit,
    insert_retention_decision,
    insert_review_work,
    insert_session,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.ops import PrivacyAuditRecord, RawAccessEvent, RetentionDecision, ReviewWork
from dlp_schema.review import AssignmentStatus, InjectedError, ReviewAssignment, ReviewMode
from dlp_schema.session import LifecycleState
from dlp_schema.testing import make_label, make_session

ROOT = Path(__file__).resolve().parents[3]
WEEK = "2026-W41"  # 2026-10-05(월) ~ 10-11
IN = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
BEFORE = IN - timedelta(days=7)


@pytest.fixture(scope="module")
def policy() -> OpsPolicy:
    return load_policy(ROOT)


def test_week_and_month_ranges() -> None:
    start, end = week_range(WEEK)
    assert (start, end) == (datetime(2026, 10, 5, tzinfo=UTC), datetime(2026, 10, 12, tzinfo=UTC))
    assert week_of(IN) == WEEK and week_of(BEFORE) == "2026-W40"
    assert month_range("2026-12") == (
        datetime(2026, 12, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC)
    )  # fmt: skip


def test_quality_alerts(policy: OpsPolicy) -> None:
    def w(week: str, minutes: float, corr: float, auto: float, det: float) -> WeeklyMetrics:
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


def event(actor: str, action: str, hour: int, purpose: str = "privacy.detect") -> RawAccessEvent:
    return RawAccessEvent(
        event_id=uuid.uuid4().hex, at=datetime(2026, 10, 7, hour, tzinfo=UTC), actor=actor,
        purpose=purpose, action=action, bucket="dlp-raw",  # type: ignore[arg-type]
        key="sessions/s1/bodycam.mp4", session_id="s1",
    )  # fmt: skip


def test_audit_report_flags(policy: OpsPolicy) -> None:
    # 시각은 UTC. 서울(UTC+9) 기준 업무 시간 밖 = 22시~6시 = UTC 13시~21시
    events = [
        event("svc-pipeline", "read", 15),  # 서비스 계정: 시간과 무관하게 정상
        event("rev-a", "grant", 2, "review.create"),  # 권한자에게 보여 줌: 정상
        event("rev-a", "presign", 3),  # 권한자 열람 (서울 12시): 정상
        event("intruder", "read", 4),  # 권한 없는 사람의 열람
        event("rev-b", "grant", 5, "review.create"),  # 권한 없는 사람에게 보여 줌
        event("unassigned", "grant", 5, "review.create"),  # 담당자 없는 원본 작업
        event("rev-a", "read", 16),  # 권한자지만 서울 새벽 1시
    ]
    r = audit_report(events, "2026-10", policy, privacy_reviewers=("rev-a",))
    assert r.events == 7 and r.sessions == 1
    assert r.by_action == {"grant": 3, "presign": 1, "read": 3}
    kinds = sorted(f.split(":")[0] for f in r.flags)
    assert kinds == [
        "담당자 없이 원본 검수 작업을 올림",
        "업무 시간 밖 원본 접근",
        "원본 권한 없는 사람에게 보여 줌",
        "원본 권한 없는 사람의 열람",
    ]


# ---------------------------------------------------------------- DB (make up)


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
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
    return {"kind": "action", "action_id": "a", "hand": "right", "verb": verb,
            "t_approach_ms": 0, "t_end_ms": 900}  # fmt: skip


def checked(state: VerificationState, at: datetime, who: str = "rev-1") -> Verification:
    return Verification(state=state, reviewer_id=who, reviewed_at=at)


def model(sid: str, lid: str, at: datetime, **kw: Any) -> LabelRecord:
    return make_label(
        action(), label_id=f"{sid}-{lid}", session_id=sid, t_end_ms=900, created_at=at,
        provenance=Provenance(source=Source.MODEL, model_version="m1"), confidence=0.8, **kw,
    )  # fmt: skip


def human(sid: str, lid: str, at: datetime, **kw: Any) -> LabelRecord:
    payload = kw.pop("payload", action())
    kw.setdefault("verification", checked(VerificationState.HUMAN_CORRECTED, at))
    return make_label(
        payload, label_id=f"{sid}-{lid}", session_id=sid, t_end_ms=900, created_at=at, **kw
    )


@pytest.mark.services
def test_weekly_metrics_from_synthetic_events(pg: sa.Engine, policy: OpsPolicy) -> None:
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
    with pg.begin() as conn:
        for s, state in (
            (sid, LifecycleState.HUMAN_VERIFIED),
            (old, LifecycleState.HUMAN_VERIFIED),
        ):
            insert_session(conn, make_session(s).model_copy(update={"lifecycle_state": state}))
        insert_labels(conn, labels)
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
    assert m.verified_episodes == 1
    assert m.review_hours == pytest.approx(1.0)
    assert m.cost_per_episode == pytest.approx(30000.0)
    assert math.isnan(m.prelabel_bias)  # 이 주에 끝난 블라인드 배정 없음
    assert m.as_dict()["prelabel_bias"] is None

    # 감사·운영 기록은 고치거나 지울 수 없다
    with pytest.raises(sa.exc.DBAPIError, match="추가만"), pg.begin() as conn:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownArgumentType]
        conn.execute(sa.text("UPDATE review_work SET seconds = 0"))


@pytest.mark.services
def test_retention_alerts(pg: sa.Engine) -> None:
    today = date(2027, 10, 1)
    finished = {
        "r-old": datetime(2026, 9, 1, tzinfo=UTC),
        "r-soon": datetime(2026, 10, 20, tzinfo=UTC),
        "r-new": datetime(2027, 6, 1, tzinfo=UTC),
        "r-ext": datetime(2026, 1, 1, tzinfo=UTC),
        "r-del": datetime(2026, 1, 1, tzinfo=UTC),
    }
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
        insert_session(
            conn,
            make_session("r-gone").model_copy(update={"lifecycle_state": LifecycleState.WITHDRAWN}),
        )
        insert_session(
            conn,
            make_session("r-wip").model_copy(update={"lifecycle_state": LifecycleState.PRELABELED}),
        )
        for sid, kind, until in [
            ("r-ext", "extend", date(2027, 12, 31)),
            ("r-del", "delete", None),
        ]:
            insert_retention_decision(conn, RetentionDecision(
                decision_id=f"d-{sid}", session_id=sid, decision=kind, until=until,  # type: ignore[arg-type]
                reason="재학습에 필요" if kind == "extend" else "보관 기간 만료",
                decided_by="ops-1", decided_at=datetime(2027, 9, 1, tzinfo=UTC),
            ))  # fmt: skip
    with pg.connect() as conn:
        items = {i.session_id: i for i in retention_status(conn, today, 365, 30)}
        assert retention_status(conn, today, None, 30) == [
            i for i in retention_status(conn, today, None, 30) if i.status == "withdrawn"
        ]  # 기간 미정이면 사용 중지만 보인다
    assert {k: v.status for k, v in items.items()} == {
        "r-old": "expired", "r-soon": "due_soon", "r-new": "ok", "r-ext": "extended",
        "r-del": "delete_decided", "r-gone": "withdrawn",
    }  # fmt: skip
    assert items["r-soon"].expires_on == date(2027, 10, 20)
    assert {k for k, v in items.items() if v.alert} == {"r-old", "r-soon", "r-gone"}
