"""설명 라벨 영어 번역 하위 명령 (ADR 0043).

등록하는 명령:
- `dlp translate run <세션> [--translate-url] [--url]` — 세션의 현재 한국어 설명마다 영어 번역
  레코드를
  만든다 (멱등, `dlp_actions.translate.run_translate`).
- `dlp translate run --pending` — 번역이 하나도 없는 한국어 설명이 남은 세션을 모두 처리한다 (서버
  장애로
  밀린 것, 사람이 쓰거나 고친 설명 포함).
- `dlp translate pending [--url]` — 번역 대기 세션과 원문 수만 보인다 (서버에 묻지 않는다).

자동 실행: `dlp actions run`이 행동 트랜잭션을 커밋한 뒤 `translate_after_actions`로 그 세션을
번역한다
(실패해도 행동 결과·종료 코드에 영향이 없다). 검수 도구(Label Studio·CVAT)는 설명 라벨을 다루지
않으므로
검수 수집 뒤 자동 번역 고리는 없다. 사람이 DB에 쓴 설명은 `--pending`이 채운다.

번역 서버: 사내 OpenAI 호환 서버만 (`--translate-url` 또는 `DLP_TRANSLATE_URL`, 키
`DLP_VLM_API_KEY`).
주소는 `translate.yaml server.internal_hosts` 대역 안이어야 한다
(`dlp_aireview.backend.check_hosts`).

종료 코드 (`run`):
- 0: 남은 대기 없음.
- 2: 번역 서버 주소가 없음 (아무것도 쓰지 않았다).
- 3: 서버 장애·응답 위반으로 번역하지 못한 설명이 남음 (번역한 것은 썼다. 다시 실행한다).
- 5: 설정 오류 (사내 대역 밖 주소, 4xx 응답).
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_actions.translate import TranslateSummary, run_translate, translation_version, untranslated
from dlp_actions.translate_policy import TranslatePolicy, load_policy
from dlp_actions.translators import (
    OpenAICompatibleTranslator,
    Translator,
    server_translator_version,
)
from dlp_aireview.backend import check_hosts
from dlp_cli.schema_cmds import database_url
from dlp_schema import repo_root
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import get_labels, list_session_ids

# 번역 서버 주소 환경 변수
URL_ENV = "DLP_TRANSLATE_URL"
# 종료 코드
OK, NO_URL, PENDING, CONFIG = 0, 2, 3, 5


def make_translator(url: str | None, policy: TranslatePolicy) -> tuple[Translator | None, int]:
    """번역기와 (만들지 못했으면) 종료 코드. 주소가 없거나 사내 대역 밖이면 번역기 없이 알린다."""
    url = url or os.environ.get(URL_ENV)
    if not url:
        # TODO(real-model): 사내 번역 서버(services/vlm vlm-aireview)를 연결하면
        #   DLP_TRANSLATE_URL을 둔다
        print(
            f"[번역 미연동] 번역 서버 주소가 없습니다 (--translate-url 또는 {URL_ENV}). "
            "영어 번역을 만들지 않았습니다. 연결 뒤 `dlp translate run --pending`"
        )
        return None, NO_URL
    outside = check_hosts({"translate": url}, policy.server.internal_hosts)
    if outside:
        print(f"[설정 오류] 사내 대역 밖 주소에는 보내지 않습니다: {'; '.join(outside)}")
        return None, CONFIG
    # 서버 API 키는 행동·판정 서버와 같다. 출력·기록하지 않는다
    key = os.environ.get("DLP_VLM_API_KEY") or None
    return OpenAICompatibleTranslator(url, policy, api_key=key), OK


def _report(session_id: str, s: TranslateSummary) -> int:
    """요약을 출력하고 종료 코드를 돌려준다."""
    print(
        f"{session_id}: 번역 {s.translated}, 이미 있음 {s.covered}, 삭제 표시 {s.retracted}, "
        f"대기 {len(s.pending)}"
    )
    if s.inherited:
        print(f"  행동 검수를 물려받은 설명 {s.inherited}개 (ADR 0043 3절)")
    if s.kept_reviewed:
        print(
            f"  검수된 영어 설명이 있어 새로 만들지 않은 원문 {len(s.kept_reviewed)}개 "
            f"(원문이 바뀌었으면 검수자가 영어도 고친다): {', '.join(s.kept_reviewed[:5])}"
        )
    if s.invalid:
        print(f"  응답 위반으로 남은 원문 {len(s.invalid)}개: {', '.join(s.invalid[:5])}")
    if s.blocked is not None:
        print(f"  [쓰지 않음] 쓰기 직전 확인이 막았습니다 ({s.blocked}, F-PAR-23)")
    if s.stopped == "config":
        print(
            f"  [설정 오류] 서버가 요청을 거부했습니다. 서버 주소·모델·키를 확인하세요 ({s.error})"
        )
        return CONFIG
    if s.stopped == "unavailable":
        print(
            "  [서버 장애] 남은 원문은 다음 실행(`dlp translate run --pending`)이 채웁니다 "
            f"({s.error})"
        )
    return PENDING if s.pending else OK


def translate_session(
    db_url: str | None, session_id: str, translator: Translator, policy: TranslatePolicy
) -> TranslateSummary:
    """세션 하나를 한 트랜잭션에서 번역한다 (세션 온톨로지로 용어집을 만든다)."""
    engine = sa.create_engine(database_url(db_url))
    try:
        with engine.begin() as conn:
            ontology = get_session_ontology(conn, session_id)
            return run_translate(conn, session_id, translator, ontology, policy, datetime.now(UTC))
    finally:
        engine.dispose()


def pending_sessions(db_url: str | None, policy: TranslatePolicy) -> dict[str, int]:
    """번역이 하나도 없는 현재 한국어 설명이 남은 세션 → 원문 수 (서버에 묻지 않는다)."""
    engine = sa.create_engine(database_url(db_url))
    out: dict[str, int] = {}
    try:
        with engine.connect() as conn:
            for sid in list_session_ids(conn):
                history = get_labels(conn, sid, kinds=["description"])
                if not history:
                    continue
                version = translation_version(
                    policy, server_translator_version(policy), get_session_ontology(conn, sid)
                )
                if n := len(untranslated(history, policy, version)):
                    out[sid] = n
    finally:
        engine.dispose()
    return out


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp translate run <세션> | --pending` (모듈 docstring의 종료 코드)."""
    if bool(args.session_id) == bool(args.pending):
        print("세션 ID 하나 또는 --pending 중 하나를 준다")
        return CONFIG
    policy = load_policy(repo_root())
    translator, code = make_translator(args.translate_url, policy)
    if translator is None:
        return code
    sessions = [args.session_id] if args.session_id else sorted(pending_sessions(args.url, policy))
    if not sessions:
        print("번역 대기 세션 없음")
    worst = OK
    for sid in sessions:
        worst = max(worst, _report(sid, translate_session(args.url, sid, translator, policy)))
    return worst


def cmd_pending(args: argparse.Namespace) -> int:
    """`dlp translate pending`: 번역 대기 세션과 원문 수. 대기가 있으면 3."""
    found = pending_sessions(args.url, load_policy(repo_root()))
    for sid, n in sorted(found.items()):
        print(f"{sid}: 번역 대기 한국어 설명 {n}개")
    if not found:
        print("번역 대기 세션 없음")
    return PENDING if found else OK


def translate_after_actions(db_url: str | None, session_id: str, url: str | None) -> None:
    """`dlp actions run` 커밋 뒤 자동 번역 (최선 노력). 어떤 실패도 올리지 않는다.

    행동 결과는 이미 커밋됐다. 번역하지 못한 설명은 `dlp translate run --pending`이 채운다.
    """
    try:
        policy = load_policy(repo_root())
        translator, _ = make_translator(url, policy)
        if translator is None:
            return
        _report(session_id, translate_session(db_url, session_id, translator, policy))
    except Exception as exc:  # 번역 실패가 행동 단계를 실패시키지 않게 모두 받는다
        print(
            f"[번역 실패] {session_id}: 행동 결과는 저장됐습니다. "
            f"`dlp translate run {session_id}`로 다시 시도하세요 ({type(exc).__name__}: {exc})"
        )


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`translate run|pending` 하위 명령을 등록한다."""
    tr = sub.add_parser("translate", help="한국어 설명 라벨의 영어 번역 (ADR 0043)")
    tsub = tr.add_subparsers(dest="translate_command", required=True)
    run = tsub.add_parser("run", help="세션의 한국어 설명마다 영어 번역 레코드 (멱등)")
    run.add_argument("session_id", nargs="?")
    run.add_argument("--pending", action="store_true", help="번역 대기 세션을 모두 처리")
    run.add_argument("--translate-url", help=f"사내 번역 서버 (기본: {URL_ENV})")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_run)
    pend = tsub.add_parser("pending", help="번역 대기 세션과 원문 수 (서버에 묻지 않음)")
    pend.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    pend.set_defaults(func=cmd_pending)
