"""`run_actions`의 쓰기 직전 확인과 재계획 (F-PAR-80·81). 시간대 보존 SQLite와 `OracleVlm`으로 돈다.

정답 근거: 합성 행동 시나리오(seed 1)의 입력 라벨(손 키포인트·손 상태)과 정답 행동·사이 구간.
- F-PAR-80: VLM에 묻는 동안 세션이 승인 보류·사용 중지되거나 승인이 풀리면(블러본 렌더 해시를 다시
  계산할 수 없음) 아무것도 쓰지 않고 `blocked`로 알린다 (설명 재생성·번역과 같은 확인).
- F-PAR-81: 버전 교체 재실행이 묻는 동안 검수자가 이전 버전 행동을 고치면(사람 레코드), 쓰기
  직전 이력을 다시 읽어 그 레코드를 보호 구간으로 보고 새 결과와 겹치지 않게 쓴다 (손마다 현재
  행동·공백이 겹치지 않고 빈틈이 없다).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

from dlp_actions.clients import OracleVlm
from dlp_actions.policy import load_policy
from dlp_actions.runner import run_actions
from dlp_actions.vlm import SegmentRequest
from dlp_fixtures.actions import generate_action_scenario
from dlp_media.storage import LocalStore, blurred_key, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderMeta, render_hash, write_render_meta
from dlp_schema.db.repository import (
    get_labels,
    insert_labels,
    insert_session,
    insert_withdrawal,
    record_privacy_hold,
    register_ontology,
    set_privacy_state,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ActionPayload,
    GapPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.lineage import Withdrawal
from dlp_schema.ontology import load_ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

ROOT = Path(__file__).resolve().parents[3]
SID = "act-wc-0001"
# 입력·정답을 주는 합성 행동 시나리오 (seed 1)
SCENARIO = generate_action_scenario(1, session_id=SID)
TRUTH = [x for x in SCENARIO.labels if isinstance(x.payload, ActionPayload | GapPayload)]
T1 = FIXED_TIME + timedelta(hours=1)


class _Hooked(OracleVlm):
    """첫 요청 직전에 `hook`을 한 번 부르는 stub (묻는 동안 일어난 다른 쓰기를 흉내 낸다)."""

    hook: Callable[[], object] | None = None

    def complete(self, request: SegmentRequest, prompt: str, schema: dict[str, Any]) -> str:
        if self.calls == 0 and self.hook is not None:
            self.hook()
        return super().complete(request, prompt, schema)


@pytest.fixture
def db() -> Iterator[sa.Engine]:
    """온톨로지 v1·세션·입력 라벨(손 키포인트·손 상태)만 넣은 SQLite."""
    engine = sqlite_db()
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        insert_session(conn, make_session(SID, duration_ms=SCENARIO.duration_ms))
        insert_labels(
            conn, [x for x in SCENARIO.labels if x.kind in ("keypoint_track", "hand_state")]
        )
    yield engine
    engine.dispose()


def _approved_store(db: sa.Engine, tmp: Path) -> LocalStore:
    """승인 상태와 지금 렌더 기록이 맞는 블러본(합성 바이트)을 둔 라벨링 저장소."""
    store = LocalStore(tmp / "store", "dlp-labeling")
    video = tmp / "v.bin"
    video.write_bytes(b"synthetic")
    sha = sha256_file(video)
    store.put_file(blurred_key(SID, "bodycam"), video, sha)
    meta = RenderMeta(render_hash([], load_privacy_policy(ROOT)), sha)
    write_render_meta(store, SID, "bodycam", meta, tmp)
    with db.begin() as conn:
        set_privacy_state(conn, SID, PrivacyState.APPROVED)
    return store


def _written(db: sa.Engine) -> list[LabelRecord]:
    with db.connect() as conn:
        return [x for x in get_labels(conn, SID) if x.kind in ("action", "gap", "description")]


@pytest.mark.parametrize("change", ["hold", "withdraw", "approval"])
def test_run_writes_nothing_when_held_withdrawn_or_unapproved_while_asking(
    db: sa.Engine, tmp_path: Path, change: str
) -> None:
    """F-PAR-80: 묻는 동안 승인 보류·사용 중지·승인 해제되면 아무것도 쓰지 않고 이유를 알린다."""
    labeling = _approved_store(db, tmp_path)
    ontology = load_ontology(ROOT / "config/ontology/v1")
    vlm = _Hooked(TRUTH)
    with db.begin() as conn:
        hooks: dict[str, Callable[[], object]] = {
            "hold": lambda: record_privacy_hold(
                conn, SID, "hold", reason="report", actor="ops01", at=T1
            ),
            "withdraw": lambda: insert_withdrawal(
                conn, Withdrawal(session_id=SID, reason="consent", withdrawn_at=T1)
            ),
            "approval": lambda: set_privacy_state(conn, SID, PrivacyState.AUTO_BLURRED),
        }
        vlm.hook = hooks[change]
        s = run_actions(conn, SID, vlm, ontology, load_policy(ROOT), T1, labeling)
    assert vlm.calls > 0
    assert s.blocked is not None and not s.hands
    assert _written(db) == []


def test_run_skips_asking_for_held_session(db: sa.Engine, tmp_path: Path) -> None:
    """이미 승인 보류 중이면 VLM에 묻지도 않는다."""
    labeling = _approved_store(db, tmp_path)
    vlm = OracleVlm(TRUTH)
    with db.begin() as conn:
        record_privacy_hold(conn, SID, "hold", reason="report", actor="ops01", at=T1)
        s = run_actions(
            conn, SID, vlm, load_ontology(ROOT / "config/ontology/v1"), load_policy(ROOT), T1,
            labeling,
        )  # fmt: skip
    assert s.blocked is not None and "승인 보류" in s.blocked
    assert vlm.calls == 0 and _written(db) == []


def test_run_writes_when_nothing_changed(db: sa.Engine, tmp_path: Path) -> None:
    """쓰기 직전 확인은 아무 변화가 없으면 막지 않는다 (블러본 프레임을 보낸 경로)."""
    labeling = _approved_store(db, tmp_path)
    with db.begin() as conn:
        s = run_actions(
            conn, SID, OracleVlm(TRUTH), load_ontology(ROOT / "config/ontology/v1"),
            load_policy(ROOT), T1, labeling,
        )  # fmt: skip
    assert s.blocked is None and s.hands
    assert any(x.kind == "action" for x in _written(db))


def test_human_fix_while_rerun_asks_is_protected_without_overlap(db: sa.Engine) -> None:
    """F-PAR-81: 버전 교체 재실행이 묻는 동안 검수자가 이전 버전 행동을 고치면, 새 결과는 그 사람
    레코드와 겹치는 구간을 버리고 빈 곳은 미상으로 채운다 (같은 손에서 겹침 없음·공백 0)."""
    ontology = load_ontology(ROOT / "config/ontology/v1")
    policy = load_policy(ROOT)
    with db.begin() as conn:
        run_actions(conn, SID, OracleVlm(TRUTH), ontology, policy, FIXED_TIME)
        first = [
            x for x in current_labels(get_labels(conn, SID)) if isinstance(x.payload, ActionPayload)
        ]
    target = first[0]
    assert isinstance(target.payload, ActionPayload)
    fix = target.model_copy(
        update={
            "label_id": target.label_id + "-fix",
            "parent_label_id": target.label_id,
            "provenance": Provenance(source=Source.HUMAN),
            "confidence": None,
            "created_at": T1,
            "verification": Verification(
                state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=T1
            ),
            "payload": target.payload.model_copy(update={"verb": "lift"}),
        }
    )
    # 정책 값을 바꿔 버전 교체 재실행을 만든다
    changed = policy.model_copy(
        update={"boundaries": policy.boundaries.model_copy(update={"merge_ms": 90})}
    )
    vlm = _Hooked(TRUTH)
    hand = target.payload.hand
    with db.begin() as conn:
        vlm.hook = lambda: insert_labels(conn, [fix])
        s = run_actions(conn, SID, vlm, ontology, changed, T1)
        timeline = sorted(
            (
                x
                for x in current_labels(get_labels(conn, SID, kinds=["action", "gap"]))
                if isinstance(x.payload, ActionPayload | GapPayload) and x.payload.hand is hand
            ),
            key=lambda x: (x.t_start_ms, x.label_id),
        )
    assert s.blocked is None and hand.value in s.hands
    assert fix.label_id in {x.label_id for x in timeline}
    overlaps = [
        (a.label_id, b.label_id)
        for a, b in itertools.combinations(timeline, 2)
        if a.t_start_ms < b.t_end_ms and b.t_start_ms < a.t_end_ms
    ]
    assert overlaps == []
    # 공백 0: 손 타임라인이 세션 전체를 빈틈없이 덮는다
    assert timeline[0].t_start_ms == 0 and timeline[-1].t_end_ms == SCENARIO.duration_ms
    assert all(a.t_end_ms == b.t_start_ms for a, b in itertools.pairwise(timeline))
