"""행동 구간 하위 명령 (WP10, ADR 0012·0026).

등록하는 명령:
- `dlp actions run <세션> [--vlm-url] [--store] [--url] [--translate-url] [--no-translate]` — 바디캠
  손마다 2단 구조로 행동 구간을 만든다. 1단: 손목 속도·접촉 신호로 경계 후보. 2단: OpenAI 호환 VLM이
  온톨로지 안에서 분류·설명(JSON Schema 강제, 재시도 뒤 실패하면 미상). 마지막으로 인접 동일 분류를
  병합하고 빈 시간을 미상 공백으로 채운다(타임라인 공백 0).
  행동 트랜잭션을 커밋한 뒤 한국어 설명의 영어 번역을 최선 노력으로 만든다
  (`dlp_cli.translate_cmds.translate_after_actions`, ADR 0043). 번역 실패·서버 미연동은 행동 결과와
  종료 코드에 영향이 없고, 남은 설명은 `dlp translate run --pending`이 채운다.

- `dlp actions describe <세션> | --pending [--vlm-url] [--store] [--url] [--translate-url]
  [--no-translate]` — 사람이 의미를 고친 행동(동사·대상·도구·손, 양손 중 한 손 삭제)의 한국어 설명을
  같은 VLM으로 다시 만든다 (`dlp_actions.describe`, ADR 0043 3절). 고친 동사·대상으로 묶은 설명만
  묻고, 고친 행동의 검수를 물려받는다. 커밋 뒤 그 세션의 영어 번역을 최선 노력으로 만든다.
  `--pending`은 다시 만들 설명이 있는 세션을 모두 처리한다 (검수 수집 뒤 cron). `dlp actions run`도
  행동 커밋 뒤 같은 세션에 최선 노력으로 돌린다 (실패해도 run의 결과·종료 코드 불변).

순서: `dlp prelabel run`·`dlp relations run` 뒤, 검수 배정(`dlp review plan`) 전.
`dlp actions describe --pending`은 검수 수집(`dlp review collect`·웹훅) 뒤.
VLM에는 블러본만 보낸다 (라벨링 버킷에서 읽고 원본 버킷에는 접근하지 않는다).
시간 구간은 마스터 타임라인 정수 ms다 (ADR 0019).

종료 코드:
- 0: 성공 (같은 버전 결과가 있으면 그 손은 건너뜀, 멱등).
- 2: VLM 서버 주소가 없음 (`--vlm-url`/`DLP_VLM_URL`). 모든 구간을 미상으로 채우면 검수 부담만
  늘어서 아무것도 만들지 않는다.
- 3: VLM 서버 장애. 트랜잭션이 되돌려져 아무것도 쓰지 않았으므로 서버 복구 뒤 다시 실행한다.
- 5: VLM 설정 오류 (4xx: 인증 실패, 없는 모델, 잘못된 요청 등. 다시 물어도 같아 재시도하지 않는다).
  트랜잭션이 되돌려져 아무것도 쓰지 않았다. 서버 주소·모델·키를 고친 뒤 다시 실행한다 (WP17.1).

정책 출처: `config/policies/actions.yaml` (`vlm` 절: 모델, 구간당 프레임 수, 최대 변 px, 제한 시간).
CI·CPU 테스트는 `OracleVlm` stub을 직접 쓴다 (이 명령은 실제 서버용).
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_actions.clients import OpenAICompatibleVlm
from dlp_actions.describe import DescribeSummary, describe_session, describe_version, plan_describe
from dlp_actions.policy import ActionsPolicy, load_policy
from dlp_actions.runner import run_actions
from dlp_actions.vlm import VlmRequestError, VlmUnavailableError
from dlp_cli.ontology_access import session_ontology
from dlp_cli.schema_cmds import database_url
from dlp_cli.translate_cmds import translate_after_actions
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_privacy.errors import RenderNotCurrentError
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import get_labels, list_session_ids

# 종료 코드 (모듈 docstring)
OK, NO_URL, UNAVAILABLE, CONFIG = 0, 2, 3, 5


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp actions run <세션>`: 행동 구간을 만들고 손별 결과를 출력한다.

    인자:
        args.vlm_url: OpenAI 호환 VLM 서버 기본 URL. 없으면 환경 변수 `DLP_VLM_URL`.
            서버 API 키는 환경 변수 `DLP_VLM_API_KEY`(있을 때만 Authorization 헤더).
        args.store: 라벨링 버킷 저장소 지정 (`s3` 또는 `local:<디렉터리>`), 블러본을 읽는다.
        args.url: DB URL.

    반환: 0 성공 / 2 VLM 주소 없음 / 3 VLM 장애 / 5 VLM 설정 오류 (모듈 docstring 참고).
    부작용: 한 트랜잭션에서 `label_records`에 action·gap·description 레코드와 이전 버전 철회
    레코드 INSERT, VLM 서버 HTTP 호출, 라벨링 버킷 읽기.
    온톨로지는 세션 버전의 DB 등록 사전을 읽는다.
    """
    root = repo_root()
    policy = load_policy(root)
    client = _client(args.vlm_url, policy)
    if client is None:
        return NO_URL
    ontology = session_ontology(args.url, args.session_id)
    # 블러본만 읽는다 (라벨링 버킷)
    labeling = store_from_spec(
        args.store, load_config(root / "config/defaults.yaml").buckets.labeling
    )
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            now = datetime.now(UTC)
            s = run_actions(conn, args.session_id, client, ontology, policy, now, labeling)
    except VlmUnavailableError as exc:
        # 트랜잭션이 되돌려져 아무것도 쓰지 않았다. 서버가 돌아오면 다시 실행한다
        print(
            f"[VLM 서버 장애] {args.session_id}: 아무것도 쓰지 않았습니다. 다시 실행하세요 ({exc})"
        )
        return 3
    except VlmRequestError as exc:
        # 4xx 설정 오류: 다시 물어도 같다. 트랜잭션이 되돌려져 아무것도 쓰지 않았다
        print(
            f"[VLM 설정 오류] {args.session_id}: 아무것도 쓰지 않았습니다. "
            f"서버 주소·모델·키를 확인하세요 ({exc})"
        )
        return 5
    finally:
        engine.dispose()
    for hand, counts in s.hands.items():
        print(f"{hand}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for hand in s.skipped:
        print(f"{hand}: 같은 버전 결과가 있어 건너뜀")
    if s.retracted:
        print(f"이전 버전 레코드 {s.retracted}개 삭제 표시")
    # 행동 결과는 커밋됐다. 사람이 고친 행동의 설명 재생성과 번역은 따로 돌고 실패해도 이 명령을
    # 실패시키지 않는다 (ADR 0043)
    _describe_after_run(args, client, policy, labeling)
    if not args.no_translate:
        translate_after_actions(args.url, args.session_id, args.translate_url)
    return 0


def _client(vlm_url: str | None, policy: ActionsPolicy) -> OpenAICompatibleVlm | None:
    """행동 VLM 클라이언트 (사내 OpenAI 호환 서버). 주소가 없으면 안내를 출력하고 None."""
    url = vlm_url or os.environ.get("DLP_VLM_URL")
    if not url:
        # TODO(real-model): VLM 서버(OpenAI 호환, GPU 또는 CPU용 llama.cpp)가 없으면 행동 구간·
        #   설명을 만들지 않는다. 모든 구간을 미상으로 채우면 검수 부담만 늘기 때문이다.
        print(
            "[미연동] VLM 서버 주소가 없습니다 (--vlm-url 또는 DLP_VLM_URL). "
            "행동 구간·설명을 만들지 않습니다."
        )
        return None
    vlm = policy.vlm
    return OpenAICompatibleVlm(
        url,
        vlm.model,
        frames=vlm.frames_per_segment,
        max_side=vlm.max_side_px,
        timeout_s=vlm.timeout_s,
        # 서버 API 키 (services/vlm, docs/handover/vlm-server.md). 출력·기록하지 않는다
        api_key=os.environ.get("DLP_VLM_API_KEY") or None,
    )


def describe_one(
    db_url: str | None,
    session_id: str,
    client: OpenAICompatibleVlm,
    policy: ActionsPolicy,
    labeling: ObjectStore,
) -> DescribeSummary:
    """세션 하나의 설명 재생성을 한 트랜잭션에서 한다 (예외는 호출자가 받는다)."""
    ontology = session_ontology(db_url, session_id)
    engine = sa.create_engine(database_url(db_url))
    try:
        with engine.begin() as conn:
            return describe_session(
                conn, session_id, client, ontology, policy, datetime.now(UTC), labeling
            )
    finally:
        engine.dispose()


def _print_describe(session_id: str, s: DescribeSummary) -> None:
    """설명 재생성 요약 한 줄과 남은 일."""
    if s.blocked is not None:
        print(f"[쓰지 않음] {session_id}: 쓰기 직전 확인이 막았습니다 ({s.blocked})")
        return
    print(
        f"{session_id}: 고친 행동 설명 {len(s.described)}개 다시 만듦, "
        f"검수 물려받음 {s.inherited}, 사람 확인 대기 {s.waiting}, "
        f"위반으로 못 만듦 {len(s.fallbacks)}"
    )
    if s.fallbacks:
        print(f"  다음 실행이 다시 묻는다: {', '.join(s.fallbacks[:5])}")


def _describe_after_run(
    args: argparse.Namespace,
    client: OpenAICompatibleVlm,
    policy: ActionsPolicy,
    labeling: ObjectStore,
) -> None:
    """`dlp actions run` 커밋 뒤 설명 재생성 (최선 노력, 어떤 실패도 올리지 않는다)."""
    try:
        s = describe_one(args.url, args.session_id, client, policy, labeling)
    except Exception as exc:  # 재생성 실패가 행동 단계를 실패시키지 않게 모두 받는다
        print(
            f"[설명 재생성 실패] {args.session_id}: 행동 결과는 저장됐습니다. "
            f"`dlp actions describe {args.session_id}`로 다시 시도하세요 ({type(exc).__name__})"
        )
        return
    if s.described or s.fallbacks:
        _print_describe(args.session_id, s)


def describe_pending(db_url: str | None, policy: ActionsPolicy, vlm_version: str) -> list[str]:
    """설명을 다시 만들 고친 행동이 있는 세션 (서버에 묻지 않는다, ID 순)."""
    version = describe_version(policy, vlm_version)
    engine = sa.create_engine(database_url(db_url))
    try:
        with engine.connect() as conn:
            return [
                sid
                for sid in list_session_ids(conn)
                if plan_describe(get_labels(conn, sid), version).targets
            ]
    finally:
        engine.dispose()


def cmd_describe(args: argparse.Namespace) -> int:
    """`dlp actions describe <세션> | --pending`: 고친 행동의 설명 재생성 (모듈 docstring).

    반환: 0 성공(위반으로 못 만든 설명은 다음 실행이 다시 묻는다) / 2 VLM 주소 없음 / 3 VLM 장애·
    블러본이 지금 승인된 렌더가 아님·쓰기 직전 확인(사용 중지·승인 보류·렌더 변경, F-PAR-23)이 막음
    (그 세션은 아무것도 쓰지 않았다) / 5 VLM 설정 오류.
    부작용: 세션마다 한 트랜잭션에서 `label_records` 설명·삭제 레코드 추가와 검수 상태 기록, VLM
    호출, 라벨링 버킷 블러본 읽기. 커밋 뒤 번역(최선 노력, `--no-translate`면 하지 않는다).
    """
    if bool(args.session_id) == bool(args.pending):
        print("세션 ID 하나 또는 --pending 중 하나를 준다")
        return CONFIG
    root = repo_root()
    policy = load_policy(root)
    client = _client(args.vlm_url, policy)
    if client is None:
        return NO_URL
    labeling = store_from_spec(
        args.store, load_config(root / "config/defaults.yaml").buckets.labeling
    )
    sessions = (
        [args.session_id] if args.session_id else describe_pending(args.url, policy, client.version)
    )
    if not sessions:
        print("설명을 다시 만들 세션 없음")
    worst = OK
    for sid in sessions:
        try:
            s = describe_one(args.url, sid, client, policy, labeling)
        except (VlmUnavailableError, RenderNotCurrentError) as exc:
            print(f"[다시 실행] {sid}: 아무것도 쓰지 않았습니다 ({type(exc).__name__}: {exc})")
            worst = max(worst, UNAVAILABLE)
            continue
        except VlmRequestError as exc:
            print(f"[VLM 설정 오류] {sid}: 서버 주소·모델·키를 확인하세요 ({exc})")
            return CONFIG
        _print_describe(sid, s)
        if s.blocked is not None:
            worst = max(worst, UNAVAILABLE)
            continue
        if s.described and not args.no_translate:
            translate_after_actions(args.url, sid, args.translate_url)
    return worst


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`actions run` 하위 명령을 등록한다."""
    act = sub.add_parser("actions", help="행동 구간 (경계 후보 + VLM 분류)")
    asub = act.add_subparsers(dest="actions_command", required=True)
    run = asub.add_parser("run", help="손별 행동·사이 구간과 설명 초안 (멱등)")
    run.add_argument("session_id")
    run.add_argument("--vlm-url", help="OpenAI 호환 VLM 서버 (기본: DLP_VLM_URL)")
    run.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.add_argument("--translate-url", help="사내 번역 서버 (기본: DLP_TRANSLATE_URL)")
    run.add_argument(
        "--no-translate", action="store_true", help="끝에 설명 영어 번역을 돌리지 않는다"
    )
    run.set_defaults(func=cmd_run)
    desc = asub.add_parser(
        "describe", help="사람이 의미를 고친 행동의 한국어 설명을 다시 만든다 (멱등, ADR 0043)"
    )
    desc.add_argument("session_id", nargs="?")
    desc.add_argument("--pending", action="store_true", help="다시 만들 설명이 있는 세션 모두")
    desc.add_argument("--vlm-url", help="OpenAI 호환 VLM 서버 (기본: DLP_VLM_URL)")
    desc.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    desc.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    desc.add_argument("--translate-url", help="사내 번역 서버 (기본: DLP_TRANSLATE_URL)")
    desc.add_argument("--no-translate", action="store_true", help="끝에 영어 번역을 돌리지 않는다")
    desc.set_defaults(func=cmd_describe)
