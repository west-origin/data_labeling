"""골든셋 CLI의 정답 고정 (`dlp dataset golden --create|--from`, `dlp dataset build --golden`,
`dlp eval golden`, ADR 0041).

시간대 보존 SQLite 파일 DB로 돈다 (서비스 없음). 데이터셋 빌드는 lakeFS 대신 로컬 스냅샷
저장소(`LocalSnapshotStore`)에 쓴다. 정답은 합성 행동 시나리오
(`dlp_fixtures.actions.generate_action_scenario`)의 사람 행동 라벨이다.

정답 근거: `--create`는 그 순간의 정답 라벨 ID를 고정하고, 뒤에 사람이 라벨을 고쳐도 그 버전의
정답과 평가 리포트의 정답 해시는 그대로다 (고친 직후 같은 버전을 다시 평가해 근거 전체가 같은지
본다). 고정한 정답이 지금 정답과 다르거나 비었으면 경고와 `--from` 안내를 낸다. `--from`으로 만든
새 버전은 고친 정답과 사용 중지되지 않은 세션만 담고, 세션이 모두 사용 중지되면 만들지 않는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_cli import dataset_cmds, eval_cmds
from dlp_cli.main import main
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_eval import runner
from dlp_eval.runner import GoldenSession
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema import load_ontology, repo_root
from dlp_schema.db.repository import (
    get_golden_set,
    insert_golden_set,
    insert_labels,
    insert_session,
    insert_withdrawal,
    register_ontology,
)
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.lineage import GoldenSet, Withdrawal
from dlp_schema.session import Domain, PrivacyState
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


def _insert_labels(db: str) -> None:
    """골든 후보 세션마다 사람 행동 라벨(정답)과 그 모델 예측(actions-m1, 미검수)을 넣는다."""
    engine = sqlite_db(db, create=False)
    with engine.begin() as conn:
        for i, sid in enumerate(SESSIONS):
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


def _withdraw(db: str, sid: str) -> None:
    """세션 `sid`를 사용 중지(동의 철회)로 기록한다."""
    engine = sqlite_db(db, create=False)
    with engine.begin() as conn:
        insert_withdrawal(
            conn, Withdrawal(session_id=sid, reason="동의 철회", withdrawn_at=FIXED_TIME)
        )
    engine.dispose()


class _LocalSnapshots:
    """`dataset_cmds.LakeFSSnapshotStore` 대신: 환경 변수 대신 로컬 디렉터리 스냅샷을 준다."""

    root = Path()

    @classmethod
    def from_env(cls, **_: Any) -> LocalSnapshotStore:
        return LocalSnapshotStore(cls.root)


@pytest.fixture
def bare_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """골든 후보 세션 둘 (작업자·장소가 다르고 프라이버시 승인, 라벨 없음)."""
    db = f"sqlite:///{tmp_path / 'golden.db'}"
    engine = sqlite_db(db)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        for i, sid in enumerate(SESSIONS):
            insert_session(
                conn,
                make_session(
                    sid, worker_id=f"w{i}", site_id=f"site{i}", privacy_state=PrivacyState.APPROVED
                ),
            )
    engine.dispose()
    for module in (dataset_cmds, eval_cmds):
        monkeypatch.setattr(module, "sa", _AwareSqlalchemy())
    snapshots = type("_Snapshots", (_LocalSnapshots,), {"root": tmp_path / "snapshots"})
    monkeypatch.setattr(dataset_cmds, "LakeFSSnapshotStore", snapshots)
    return db


@pytest.fixture
def url(bare_url: str) -> str:
    """`bare_url`에 사람 행동 라벨(정답)과 그 모델 예측을 더한 DB."""
    _insert_labels(bare_url)
    return bare_url


def _evaluate(url: str, version: str, out_dir: Path) -> dict[str, Any]:
    """골든 버전을 actions-m1로 평가한 JSON 리포트의 정답 근거(`golden_truth`)."""
    out = out_dir / f"{version}.json"
    main(["eval", "golden", version, "--model", "actions=actions-m1", "--out", str(out),
          "--url", url])  # fmt: skip
    return json.loads(out.read_text("utf-8"))["golden_truth"]


def test_golden_versions_freeze_truth_and_from_refreezes(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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

    before = _evaluate(url, "golden-v1", tmp_path)
    assert before["truth_frozen"] is True and before["truth_labels"] == len(expected)
    assert "경고" not in capsys.readouterr().out  # 고정한 정답 = 지금 정답

    # 사람이 정답 하나를 고친다 (새 레코드 + parent)
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
    # 같은 버전: 고친 직후에도 정답 근거(세션·정답 수·정답 ID 해시)가 그대로다
    assert _evaluate(url, "golden-v1", tmp_path) == before
    # 고정한 정답이 지금 정답과 달라졌다고 알리고 다시 고정하는 명령을 안내한다
    out = capsys.readouterr().out
    assert f"고정한 정답 {len(expected)}개가 지금 정답과 다릅니다" in out
    assert "새 정답 1개, 고쳐지거나 지워진 정답 1개" in out
    assert "--from golden-v1 --create" in out

    # 두 번째 세션은 동의 철회: 같은 버전에서 평가만 빠진다
    _withdraw(url, SESSIONS[1])
    after = _evaluate(url, "golden-v1", tmp_path)
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
    capsys.readouterr()
    v2_anchor = _evaluate(url, "golden-v2", tmp_path)
    assert v2_anchor["truth_label_ids_sha256"] != after["truth_label_ids_sha256"]
    assert "경고" not in capsys.readouterr().out  # 다시 고정한 버전은 지금 정답과 같다

    # 원본 버전의 세션이 모두 사용 중지되면 --from으로 새 버전을 만들 수 없다
    _withdraw(url, SESSIONS[0])
    with pytest.raises(SystemExit, match="모두 사용 중지"):
        main([*golden, "--from", "golden-v2", "--create", "golden-v3"])


def test_truth_frozen_before_labeling_is_flagged_until_refrozen(
    bare_url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """라벨링 전에 만든 골든 버전(정답 0개)은 빌드·평가에서 경고하고 `--from` 안내를 낸다.

    시나리오: 라벨 없는 세션으로 golden-v0을 만들고(정답 0개 고정), 라벨링한 뒤 그 버전으로 데이터셋
    버전을 만들고 평가한다. `--from`으로 다시 고정한 golden-v1은 경고가 없다. 고정하지 않은 이전
    형식 골든셋으로 빌드하면 재학습 루프가 쓰지 않는다고 알린다.
    """
    golden = ["dataset", "golden", "--domain", "cleaning", "--url", bare_url]
    assert main([*golden, "--count", "2", "--create", "golden-v0"]) == 0
    out = capsys.readouterr().out
    assert "고정한 정답 라벨 0개" in out and "--from golden-v0 --create" in out
    _insert_labels(bare_url)

    def build(version: str, golden_version: str) -> str:
        """`dataset build --golden`을 돌리고 출력을 돌려준다."""
        assert main(["dataset", "build", version, "--golden", golden_version,
                     "--url", bare_url]) == 0  # fmt: skip
        return capsys.readouterr().out

    out = build("dv0", "golden-v0")
    assert "골든셋 golden-v0에 고정한 정답이 없습니다" in out
    assert "--from golden-v0 --create" in out
    anchor = _evaluate(bare_url, "golden-v0", tmp_path)
    assert anchor["truth_frozen"] is True and anchor["truth_labels"] == 0
    assert "골든셋 golden-v0에 고정한 정답이 없습니다" in capsys.readouterr().out

    assert main([*golden, "--from", "golden-v0", "--create", "golden-v1"]) == 0
    capsys.readouterr()
    assert "경고" not in build("dv1", "golden-v1")
    assert _evaluate(bare_url, "golden-v1", tmp_path)["truth_labels"] > 0
    assert "경고" not in capsys.readouterr().out

    engine = sqlite_db(bare_url, create=False)
    with engine.begin() as conn:
        insert_golden_set(
            conn,
            GoldenSet(
                version="golden-legacy",
                domain=Domain.CLEANING,
                session_ids=SESSIONS,
                created_at=FIXED_TIME,
            ),
        )
    engine.dispose()
    out = build("dv-legacy", "golden-legacy")
    assert "이전 형식" in out and "--from golden-legacy --create" in out


def test_eval_builds_anchor_and_inputs_from_one_golden_read(
    url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`eval golden`은 골든셋을 한 번만 읽어 정답 근거와 후보·기존 평가 입력을 모두 만든다.

    두 번 읽으면 그 사이 커밋된 사용 중지·라벨로 리포트의 근거(세션·정답 해시)와 실제로 평가한
    세션·정답이 달라질 수 있다. 정답 근거: `--baseline`까지 준 평가에서 골든셋 읽기가 한 번이고,
    근거에 두 세션이 모두 있다.
    """
    assert main(["dataset", "golden", "--domain", "cleaning", "--count", "2",
                 "--create", "golden-v1", "--url", url]) == 0  # fmt: skip
    calls: list[str] = []
    original = runner.golden_sessions

    def counted(conn: sqlalchemy.Connection, version: str) -> list[GoldenSession]:
        calls.append(version)
        return original(conn, version)

    monkeypatch.setattr(runner, "golden_sessions", counted)
    monkeypatch.setattr(eval_cmds, "golden_sessions", counted)
    out = tmp_path / "eval.json"
    main(["eval", "golden", "golden-v1", "--model", "actions=actions-m1",
          "--baseline", "actions=actions-m1", "--out", str(out), "--url", url])  # fmt: skip
    assert calls == ["golden-v1"]
    report = json.loads(out.read_text("utf-8"))
    assert sorted(report["golden_truth"]["sessions"]) == sorted(SESSIONS)


def _hint_argv(hint: str, new_version: str) -> list[str]:
    """안내 문장의 첫 `dlp …` 명령을 argv로 바꾼다 (`<새 버전>`은 `new_version`으로)."""
    found = re.search(r"`dlp ([^`]+)`", hint)
    assert found is not None, hint
    return found.group(1).replace("<새 버전>", new_version).split()


def test_refreeze_hint_runs_as_printed(url: str) -> None:
    """다시 고정하는 안내 명령(`refreeze_hint`)은 그대로 실행된다 (`--from`이면 `--domain` 생략).

    정답 근거: 안내 명령에는 `--domain`이 없고, 실행하면 원본 골든셋(청소)의 도메인·세션으로 새
    버전을 만든다. `--domain`을 주면 원본과 같아야 하고, `--from`이 없으면 `--domain`이 필요하다.
    """
    assert main(["dataset", "golden", "--domain", "cleaning", "--count", "2",
                 "--create", "golden-v1", "--url", url]) == 0  # fmt: skip
    argv = _hint_argv(runner.refreeze_hint("golden-v1"), "golden-v2")
    assert "--domain" not in argv
    assert main([*argv, "--url", url]) == 0
    engine = sqlite_db(url, create=False)
    with engine.connect() as conn:
        v1, v2 = get_golden_set(conn, "golden-v1"), get_golden_set(conn, "golden-v2")
    engine.dispose()
    assert v2.domain is Domain.CLEANING and v2.session_ids == v1.session_ids
    assert v2.truth_label_ids == v1.truth_label_ids
    with pytest.raises(SystemExit, match="도메인은 cleaning"):
        main(["dataset", "golden", "--domain", "nursing", "--from", "golden-v1",
              "--create", "golden-x", "--url", url])  # fmt: skip
    with pytest.raises(SystemExit, match="--domain이 필요"):
        main(["dataset", "golden", "--count", "2", "--url", url])


def test_withdrawn_golden_points_to_new_proposal_not_from(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """고정한 세션이 모두 사용 중지되면 빌드·평가 경고는 "라벨링 전에 고정"이 아니라 "모두 사용
    중지"로 알리고, 거부될 `--from` 대신 새 골든셋 제안 명령을 안내한다.

    정답 근거: golden-v1은 두 세션의 정답을 고정했다. 두 세션을 사용 중지하면 `--from golden-v1`은
    거부되고(`_sessions_from`), 경고에는 고정 정답 수와 `--domain cleaning --create` 안내가 나온다.
    """
    assert main(["dataset", "golden", "--domain", "cleaning", "--count", "2",
                 "--create", "golden-v1", "--url", url]) == 0  # fmt: skip
    for sid in SESSIONS:
        _withdraw(url, sid)
    # 데이터셋 버전에 넣을 골든 밖 세션 하나 (빌드는 후보 세션이 있어야 한다)
    engine = sqlite_db(url, create=False)
    with engine.begin() as conn:
        insert_session(conn, make_session("t001", worker_id="w9", site_id="site9",
                                          privacy_state=PrivacyState.APPROVED))  # fmt: skip
    engine.dispose()
    frozen = sum(len(_actions(sid, i)) for i, sid in enumerate(SESSIONS))
    capsys.readouterr()
    _evaluate(url, "golden-v1", tmp_path)
    assert main(["dataset", "build", "dv1", "--golden", "golden-v1", "--url", url]) == 0
    out = capsys.readouterr().out
    for warning in [ln for ln in out.splitlines() if ln.startswith("경고")]:
        assert "모두 사용 중지" in warning and f"고정 정답 {frozen}개" in warning
        assert "라벨링·검수 전에 고정한 버전" not in warning
        assert "--from golden-v1" not in warning
    assert out.count("`dlp dataset golden --domain cleaning --create <새 버전>`") == 2
    # 안내하지 않은 --from은 실제로 거부된다
    with pytest.raises(SystemExit, match="모두 사용 중지"):
        main([*_hint_argv(runner.refreeze_hint("golden-v1"), "golden-v2"), "--url", url])
