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
from dlp_actions.policy import load_policy
from dlp_cli import actions_cmds, ontology_access
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.description_repository import apply_description_follow
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    insert_labels,
    insert_session,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    DescriptionPayload,
    LabelRecord,
    Provenance,
    Source,
    Verification,
    VerificationState,
)
from dlp_schema.review import InheritanceMechanism
from dlp_schema.testing import FIXED_TIME, action_payload, make_label, make_session, sqlite_db

T1 = FIXED_TIME + timedelta(hours=1)
T2 = FIXED_TIME + timedelta(hours=2)


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

    def oracle(*_: object) -> tuple[OracleVlm, int]:
        return OracleVlm(truth), 0

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


def _truth() -> OracleVlm:
    """고친 동사(lift)와 정답 설명으로 답하는 stub."""
    return OracleVlm([
        make_label(action_payload(verb="lift"), label_id="t"),
        make_label({"kind": "description", "segment_id": "a001", "text": "걸레를 든다"},
                   label_id="td"),
    ])  # fmt: skip


def _inherited_then_changed(url: str) -> None:
    """s003: 승인한 모델 행동 → 모델 설명이 물려받기 기록과 함께 승인을 물려받음 → 그 뒤 행동의
    의미(동사)를 고친 수정 레코드만 DB에 넣는다 (설명 따르기는 아직 돌지 않았다)."""
    model = Provenance(source=Source.MODEL, model_version="actions-abc+oracle-vlm-1+i0")
    act = make_label(action_payload(), label_id="act3", session_id="s003", provenance=model,
                     confidence=0.9)  # fmt: skip
    desc = make_label(
        {"kind": "description", "segment_id": "a001", "text": "걸레를 잡는다"},
        label_id="d3", session_id="s003", provenance=model, confidence=0.9,
    )  # fmt: skip
    fix: LabelRecord = act.model_copy(
        update={"label_id": "fix3", "parent_label_id": "act3", "created_at": T2,
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
                "verification": Verification(state=VerificationState.HUMAN_CORRECTED,
                                             reviewer_id="r1", reviewed_at=T2),
                "payload": act.payload.model_copy(update={"verb": "lift"})}
    )  # fmt: skip
    engine = sqlite_db(url, create=False)
    with engine.begin() as conn:
        insert_session(conn, make_session("s003"))
        insert_labels(conn, [act, desc])
        record_review(conn, "act3", VerificationState.HUMAN_APPROVED, "r1", T1)
        follow = apply_description_follow(conn, "s003", T1, InheritanceMechanism.COLLECT)
        assert [r.label_id for r in follow.approve] == ["d3"]
        assert len(get_description_inheritance(conn, "s003")) == 1
        insert_labels(conn, [fix])
    engine.dispose()


def test_describe_pending_sees_inherited_descriptions_of_changed_actions(
    url: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-PAR-24: 물려받기 기록이 있는 설명의 행동이 다시 바뀌면 `--pending`도 그 세션을 고른다.

    정답: 설명 d3의 승인은 물려받은 것뿐이다(기록 있음). 행동 의미가 바뀌었으므로 설명 따르기가
    d3를 지우고 재생성 대상이 된다. 예전 `describe_pending`은 물려받기 기록을 넘기지 않아 d3를
    직접 검수로 보고 s003을 건너뛰었다 (`describe_session`은 처리한다).
    """
    _inherited_then_changed(url)
    policy = load_policy(repo_root())
    assert actions_cmds.describe_pending(url, policy, "oracle-vlm-1") == ["s001", "s003"]

    def oracle(*_: object) -> tuple[OracleVlm, int]:
        return _truth(), 0

    monkeypatch.setattr(actions_cmds, "_client", oracle)
    assert actions_cmds.cmd_describe(_args(url, pending=True)) == 0
    assert "s003: 고친 행동 설명 1개 다시 만듦" in capsys.readouterr().out
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        current = current_labels(get_labels(conn, "s003"))
    engine.dispose()
    (d,) = [x for x in current if isinstance(x.payload, DescriptionPayload)]
    assert d.label_id != "d3" and isinstance(d.payload, DescriptionPayload)
    assert d.payload.text == "걸레를 든다"
    assert actions_cmds.describe_pending(url, policy, "oracle-vlm-1") == []
