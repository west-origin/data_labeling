"""골든셋 CLI의 정답 고정 (`dlp dataset golden --create|--from`, `dlp eval golden`, ADR 0047).

시간대 보존 SQLite 파일 DB로 돈다 (서비스 없음). 정답은 합성 행동 시나리오
(`dlp_fixtures.actions.generate_action_scenario`)의 사람 행동 라벨이다.

정답 근거: `--create`는 그 순간의 정답 라벨 ID를 고정하고, 뒤에 사람이 라벨을 고쳐도 그 버전의
정답과 평가 리포트의 정답 해시는 그대로다. `--from`으로 만든 새 버전은 고친 정답과 사용 중지되지
않은 세션만 담는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_cli import dataset_cmds, eval_cmds
from dlp_cli.main import main
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_golden_set,
    insert_labels,
    insert_session,
    insert_withdrawal,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.lineage import Withdrawal
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session, sqlite_db

SESSIONS = ("g001", "g002")


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    @staticmethod
    def create_engine(url: Any, **_: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


def _actions(sid: str, seed: int) -> list[LabelRecord]:
    """세션 `sid`의 사람 행동 라벨 (합성 시나리오 `seed`)."""
    return [x for x in generate_action_scenario(seed, session_id=sid).labels if x.kind == "action"]


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """골든 후보 세션 둘(작업자·장소가 다르고 프라이버시 승인)과 사람 행동 라벨, 그 모델 예측."""
    db = f"sqlite:///{tmp_path / 'golden.db'}"
    engine = sqlite_db(db)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for i, sid in enumerate(SESSIONS):
            session = make_session(
                sid, worker_id=f"w{i}", site_id=f"site{i}", privacy_state=PrivacyState.APPROVED
            )
            insert_session(conn, session)
            truth = _actions(sid, i)
            model = [
                x.model_copy(
                    update={
                        "label_id": f"{x.label_id}-m",
                        "provenance": Provenance(source=Source.MODEL, model_version="actions-m1"),
                        "confidence": 0.8,
                    }
                )
                for x in truth
            ]
            insert_labels(conn, [*truth, *model])
    engine.dispose()
    for module in (dataset_cmds, eval_cmds):
        monkeypatch.setattr(module, "sa", _AwareSqlalchemy())
    return db


def test_golden_versions_freeze_truth_and_from_refreezes(url: str, tmp_path: Path) -> None:
    """`--create`는 정답을 고정하고, 정답을 고친 뒤에는 `--from`으로 새 버전에 다시 고정한다."""
    golden = ["dataset", "golden", "--domain", "cleaning", "--url", url]
    assert main([*golden, "--count", "2", "--create", "golden-v1"]) == 0
    engine = sqlite_db(url, create=False)
    first = _actions(SESSIONS[0], 0)
    with engine.connect() as conn:
        v1 = get_golden_set(conn, "golden-v1")
    assert sorted(v1.session_ids) == sorted(SESSIONS)
    expected = {x.label_id for i, sid in enumerate(SESSIONS) for x in _actions(sid, i)}
    assert set(v1.truth_label_ids or ()) == expected  # 모델 예측은 정답이 아니다

    def evaluate(version: str) -> dict[str, Any]:
        """골든 버전을 actions-m1로 평가한 JSON 리포트."""
        out = tmp_path / f"{version}.json"
        main(["eval", "golden", version, "--model", "actions=actions-m1", "--out", str(out),
              "--url", url])  # fmt: skip
        return json.loads(out.read_text("utf-8"))

    before = evaluate("golden-v1")["golden_truth"]
    assert before["truth_frozen"] is True and before["truth_labels"] == len(expected)

    # 사람이 정답 하나를 고치고 두 번째 세션은 동의 철회
    corrected = first[0].model_copy(
        update={
            "label_id": "fix-1",
            "parent_label_id": first[0].label_id,
            "verification": Verification(
                state=VerificationState.HUMAN_CORRECTED, reviewer_id="r1", reviewed_at=FIXED_TIME
            ),
        }
    )
    with engine.begin() as conn:
        insert_labels(conn, [corrected])
        insert_withdrawal(
            conn, Withdrawal(session_id=SESSIONS[1], reason="동의 철회", withdrawn_at=FIXED_TIME)
        )
    # 같은 버전: 고정한 정답 그대로 (사용 중지 세션은 평가에서만 빠진다)
    after = evaluate("golden-v1")["golden_truth"]
    assert after["sessions"] == [SESSIONS[0]]
    assert after["truth_labels"] == len(first)

    # --from은 --create 없이 쓸 수 없다
    with pytest.raises(SystemExit, match="--create"):
        main([*golden, "--from", "golden-v1"])
    assert main([*golden, "--from", "golden-v1", "--create", "golden-v2"]) == 0
    with engine.connect() as conn:
        v1_again = get_golden_set(conn, "golden-v1")
        v2 = get_golden_set(conn, "golden-v2")
    engine.dispose()
    assert v1_again.truth_label_ids == v1.truth_label_ids
    assert v2.session_ids == (SESSIONS[0],)
    assert set(v2.truth_label_ids or ()) == {x.label_id for x in first[1:]} | {"fix-1"}
    v2_anchor = evaluate("golden-v2")["golden_truth"]
    assert v2_anchor["truth_label_ids_sha256"] != after["truth_label_ids_sha256"]
