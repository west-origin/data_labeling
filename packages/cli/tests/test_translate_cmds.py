"""`dlp translate run|pending`와 `dlp actions run` 끝의 자동 번역 (ADR 0043).

서비스 없이 시간대 보존 SQLite(CLI 엔진을 바꿔 끼운다)와 stub 번역기로 돈다.

정답 근거: 세션 둘(t1: 한국어 설명 2개, t2: 설명 없음). 번역 전에는 t1만 대기 2개, 번역 뒤 대기 0.
서버 주소가 없으면 종료 2, 사내 대역 밖 주소면 5, 서버 장애로 남으면 3. 자동 번역은 어떤 실패도
올리지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy

from dlp_actions.translate_policy import load_policy
from dlp_actions.translators import GlossaryStubTranslator, Translator
from dlp_cli import translate_cmds
from dlp_cli.main import main
from dlp_schema import repo_root
from dlp_schema.db.repository import get_labels, insert_labels, insert_session, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import DescriptionPayload, Provenance, Source
from dlp_schema.ontology import load_ontology
from dlp_schema.testing import FIXED_TIME, make_label, make_session, sqlite_db

ROOT = repo_root()


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, sqlalchemy.Engine]:
    """SQLite 파일 DB: 온톨로지 v1, 세션 t1(한국어 설명 2개)·t2(설명 없음)."""
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    engine = sqlite_db(url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config/ontology/v1"))
        for sid in ("t1", "t2"):
            insert_session(conn, make_session(sid), at=FIXED_TIME)
        insert_labels(
            conn,
            [
                make_label(
                    {"kind": "description", "segment_id": f"t1-a{i}", "text": text},
                    label_id=f"t1-d{i}",
                    session_id="t1",
                    confidence=0.8,
                    provenance=Provenance(source=Source.MODEL, model_version="actions-x"),
                )
                for i, text in enumerate(["걸레로 바닥을 닦는다", "양동이를 든다"])
            ],
        )
    monkeypatch.setattr(translate_cmds, "sa", _AwareSqlalchemy())
    monkeypatch.delenv(translate_cmds.URL_ENV, raising=False)
    return url, engine


def _use(monkeypatch: pytest.MonkeyPatch, translator: Translator) -> None:
    """make_translator가 주소 검사 없이 stub을 돌려주게 한다."""

    def make(url: str | None, policy: Any) -> tuple[Translator, int]:
        return translator, 0

    monkeypatch.setattr(translate_cmds, "make_translator", make)


def test_pending_then_run_translates_and_clears(
    db: tuple[str, sqlalchemy.Engine], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """대기 목록 → --pending 실행 → 영어 형제 2개, 대기 없음 (종료 0). 다시 돌려도 그대로."""
    url, engine = db
    assert translate_cmds.pending_sessions(url, load_policy(ROOT)) == {"t1": 2}
    assert main(["translate", "pending", "--url", url]) == 3
    assert "t1: 번역 대기 한국어 설명 2개" in capsys.readouterr().out
    _use(monkeypatch, GlossaryStubTranslator())
    assert main(["translate", "run", "--pending", "--url", url]) == 0
    with engine.connect() as conn:
        labels = current_labels(get_labels(conn, "t1", kinds=["description"]))
    en = [
        x
        for x in labels
        if isinstance(x.payload, DescriptionPayload) and x.payload.language == "en"
    ]
    assert len(en) == 2 and len(labels) == 4
    assert main(["translate", "pending", "--url", url]) == 0
    assert main(["translate", "run", "t1", "--url", url]) == 0
    assert "번역 0, 이미 있음 2" in capsys.readouterr().out


def test_run_exit_codes(
    db: tuple[str, sqlalchemy.Engine], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """주소 없음 2, 사내 대역 밖 5, 서버 장애로 남음 3, 인자 둘 다·둘 다 없음 5."""
    url, _ = db
    assert main(["translate", "run", "t1", "--url", url]) == 2
    assert "[번역 미연동]" in capsys.readouterr().out
    code = main(["translate", "run", "t1", "--url", url, "--translate-url", "http://8.8.8.8:8000"])
    assert code == 5 and "사내 대역 밖" in capsys.readouterr().out
    assert main(["translate", "run", "--url", url]) == 5
    assert main(["translate", "run", "t1", "--pending", "--url", url]) == 5
    _use(monkeypatch, GlossaryStubTranslator(fail_unavailable=["걸레로 바닥을 닦는다"]))
    assert main(["translate", "run", "t1", "--url", url]) == 3
    assert "[서버 장애]" in capsys.readouterr().out


def test_internal_url_builds_server_translator(monkeypatch: pytest.MonkeyPatch) -> None:
    """사내 주소(환경 변수)면 사내 서버 번역기를 만든다 (요청은 보내지 않는다)."""
    monkeypatch.setenv(translate_cmds.URL_ENV, "http://10.0.0.11:8000")
    translator, code = translate_cmds.make_translator(None, load_policy(ROOT))
    assert code == 0 and translator is not None
    assert translator.version == f"llm-{load_policy(ROOT).backend.model}"


def test_after_actions_never_raises(
    db: tuple[str, sqlalchemy.Engine], monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """행동 단계 뒤 자동 번역: 주소가 없으면 안내만, 내부 오류도 올리지 않고, 되면 번역한다."""
    url, engine = db
    translate_cmds.translate_after_actions(url, "t1", None)
    assert "[번역 미연동]" in capsys.readouterr().out

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("db gone")

    _use(monkeypatch, GlossaryStubTranslator())
    monkeypatch.setattr(translate_cmds, "translate_session", boom)
    translate_cmds.translate_after_actions(url, "t1", None)
    assert "[번역 실패] t1" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setattr(translate_cmds, "sa", _AwareSqlalchemy())
    _use(monkeypatch, GlossaryStubTranslator())
    translate_cmds.translate_after_actions(url, "t1", None)
    assert "t1: 번역 2" in capsys.readouterr().out
    with engine.connect() as conn:
        assert len(current_labels(get_labels(conn, "t1", kinds=["description"]))) == 4


def test_actions_run_has_translate_options() -> None:
    """`dlp actions run`이 번역 서버·끄기 옵션을 받는다 (주소 없음 경로는 행동 전에 끝난다)."""
    from dlp_cli.main import build_parser

    args = build_parser().parse_args(
        ["actions", "run", "s1", "--translate-url", "http://10.0.0.11:8000", "--no-translate"]
    )
    assert args.translate_url == "http://10.0.0.11:8000" and args.no_translate
