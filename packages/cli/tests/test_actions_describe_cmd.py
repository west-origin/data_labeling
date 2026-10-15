"""`dlp actions describe` 명령 (ADR 0043 3절). 시간대 보존 SQLite 파일 DB와 `OracleVlm`으로 돈다.

정답 근거: 검수자가 동사를 고친 행동 하나(s001). `--pending`은 그 세션만 고르고, 실행 뒤 설명이
하나 생기며(고친 행동 검수를 물려받음) 다시 고르면 대상이 없다. VLM 주소가 없으면 종료 코드 2.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_actions.clients import OracleVlm
from dlp_cli import actions_cmds, ontology_access
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    DescriptionPayload,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

T1 = FIXED_TIME + timedelta(hours=1)


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


def _no_store(*_: object) -> None:
    """블러본 저장소 대역: 없음 (프레임 없이 묻는다)."""


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """동사를 고친 행동이 있는 세션 s001과 손대지 않은 세션 s002."""
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(db)
    model = Provenance(source=Source.MODEL, model_version="actions-abc+oracle-vlm-1+i0")
    act = make_label(action_payload(), label_id="act", provenance=model, confidence=0.9)
    fix = act.model_copy(
        update={"label_id": "fix", "parent_label_id": "act", "created_at": T1,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": Verification(state=VerificationState.HUMAN_CORRECTED,
                                             reviewer_id="r1", reviewed_at=T1),
                "payload": act.payload.model_copy(update={"verb": "lift"})}
    )  # fmt: skip
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session("s001"))
        insert_session(conn, make_session("s002"))
        insert_labels(conn, [act, fix])
    engine.dispose()
    for module in (actions_cmds, ontology_access):
        monkeypatch.setattr(module, "sa", _AwareSqlalchemy())
    # 블러본 없이 묻는다 (렌더 게이트는 dlp_actions 테스트가 본다)
    monkeypatch.setattr(actions_cmds, "store_from_spec", _no_store)
    return db


def _args(url: str, **kw: Any) -> argparse.Namespace:
    values: dict[str, Any] = dict(
        session_id=None, pending=False, vlm_url="http://vlm.internal", store="s3", url=url,
        translate_url=None, no_translate=True,
    )  # fmt: skip
    values.update(kw)
    return argparse.Namespace(**values)


def test_describe_pending_regenerates_once(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--pending은 고친 행동이 있는 세션만 처리하고, 다시 돌리면 대상이 없다."""
    truth = [
        make_label(action_payload(verb="lift"), label_id="t"),
        make_label({"kind": "description", "segment_id": "a001", "text": "걸레를 든다"},
                   label_id="td"),
    ]  # fmt: skip

    def oracle(*_: object) -> OracleVlm:
        return OracleVlm(truth)

    monkeypatch.setattr(actions_cmds, "_client", oracle)
    assert actions_cmds.cmd_describe(_args(url, pending=True)) == 0
    assert "s001: 고친 행동 설명 1개 다시 만듦" in capsys.readouterr().out
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        current = current_labels(get_labels(conn, "s001"))
    engine.dispose()
    (d,) = [x for x in current if isinstance(x.payload, DescriptionPayload)]
    assert d.verification.state is VerificationState.HUMAN_APPROVED
    assert actions_cmds.cmd_describe(_args(url, pending=True)) == 0
    assert "설명을 다시 만들 세션 없음" in capsys.readouterr().out


def test_describe_needs_vlm_url_and_one_target(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """VLM 주소가 없으면 2, 세션과 --pending을 함께 주거나 둘 다 없으면 5."""
    monkeypatch.delenv("DLP_VLM_URL", raising=False)
    assert actions_cmds.cmd_describe(_args(url, session_id="s001", vlm_url=None)) == 2
    assert actions_cmds.cmd_describe(_args(url)) == 5
