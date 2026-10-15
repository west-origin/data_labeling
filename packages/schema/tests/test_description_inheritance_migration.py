"""Alembic 0021의 물려받기 기록 백필 (F-PAR-20, ADR 0043 3절 2026-10-15 개정).

서비스 없이 SQLite 파일에 0020까지 올리고, 이 기록 전 코드가 남긴 것 같은 이력을 넣은 뒤 head로
올린다.

정답 근거 (손으로 만든 장면, 세션 s001)
- 행동 a1(a001)을 r1이 T1에 승인. 모델 설명 d1(a001)·d1en(번역)을 예전 규칙이 물려줬다 (r1, T1 /
  번역은 생성 시각 T2). → 백필 기록 2개 (준 주인 a1, 출처 backfill-heuristic).
- 행동 a2(a002)를 r1이 T1에 승인했지만 설명 d2는 lead가 T2에 직접 검수 → 기록 없음.
- 설명 d3(a003, 주인 없음)은 미검수 → 기록 없음. 사람 설명 dh(a001)도 기록 없음.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from dlp_schema import load_ontology, repo_root
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.migrate import downgrade, upgrade
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.description_follow import FOLLOW_VERSION, inheritance_only
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.review import InheritanceMechanism
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

T1 = FIXED_TIME + timedelta(hours=1)
T2 = FIXED_TIME + timedelta(hours=2)
MODEL = Provenance(source=Source.MODEL, model_version="actions-1")


def _desc(label_id: str, segment: str, *, source: str | None = None, **kw: Any) -> LabelRecord:
    """모델 설명 (source가 있으면 영어 번역)."""
    payload: dict[str, Any] = {"kind": "description", "segment_id": segment, "text": "설명"}
    if source is not None:
        payload |= {"language": "en", "source_label_id": source, "text": "desc"}
    fields: dict[str, Any] = {"provenance": MODEL, "confidence": 0.9, **kw}
    return make_label(payload, label_id=label_id, **fields)


def test_migration_backfills_old_inherited_descriptions(tmp_path: Path) -> None:
    """0021은 예전 추정(검수자·시각 일치)으로 이미 물려받은 설명을 한 번 기록한다 (docstring)."""
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url, "0020")
    engine = sqlite_db(url, create=False)
    a1 = make_label(action_payload(), label_id="a1", provenance=MODEL, confidence=0.9)
    a2 = make_label(action_payload(action_id="a002"), label_id="a2", provenance=MODEL,
                    confidence=0.9)  # fmt: skip
    human = _desc("dh", "a001", provenance=Provenance(source=Source.HUMAN), confidence=None)
    labels = [a1, a2, _desc("d1", "a001"), _desc("d1en", "a001", source="d1", created_at=T2),
              _desc("d2", "a002"), _desc("d3", "a003"), human]  # fmt: skip
    approved = VerificationState.HUMAN_APPROVED
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session())
        insert_labels(conn, labels)
        for lid in ("a1", "a2", "d1"):
            record_review(conn, lid, approved, "r1", T1)
        record_review(conn, "d1en", approved, "r1", T2)
        record_review(conn, "d2", approved, "lead", T2)
        record_review(conn, "dh", approved, "r1", T1)
    engine.dispose()
    upgrade(url)
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        recs = get_description_inheritance(conn, "s001")
        rows = {x.label_id: x for x in get_labels(conn, "s001")}
    engine.dispose()
    got = {(r.description_label_id, r.giver_label_id, r.reviewed_at) for r in recs}
    assert got == {("d1", "a1", T1), ("d1en", "a1", T2)}
    for r in recs:
        assert r.mechanism is InheritanceMechanism.BACKFILL and r.rule_version == FOLLOW_VERSION
        assert r.owner_label_ids == ("a1",) and r.state is approved
        assert inheritance_only(rows[r.description_label_id], recs)
    assert not inheritance_only(rows["d2"], recs)
    # 이관 백필만 있으면 되돌릴 수 있다
    downgrade(url, "0020")


def test_downgrade_refuses_when_live_records_exist(tmp_path: Path) -> None:
    """이관 백필이 아닌 기록(설명 따르기가 쓴 것)이 있으면 0021을 되돌리지 않는다."""
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    engine = sqlite_db(url, create=False)
    a1 = make_label(action_payload(), label_id="a1", provenance=MODEL, confidence=0.9)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session())
        insert_labels(conn, [a1, _desc("d1", "a001")])
        record_review(conn, "a1", VerificationState.HUMAN_APPROVED, "r1", T1)
        apply_description_follow(conn, "s001", T2, InheritanceMechanism.VERIFY)
    engine.dispose()
    with pytest.raises(RuntimeError, match="0021"):
        downgrade(url, "0020")
