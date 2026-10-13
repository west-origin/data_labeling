"""행동 구간 하위 명령 (WP10, ADR 0012·0026).

등록하는 명령:
- `dlp actions run <세션> [--vlm-url] [--store] [--url] [--translate-url] [--no-translate]` — 바디캠
  손마다 2단 구조로 행동 구간을 만든다. 1단: 손목 속도·접촉 신호로 경계 후보. 2단: OpenAI 호환 VLM이
  온톨로지 안에서 분류·설명(JSON Schema 강제, 재시도 뒤 실패하면 미상). 마지막으로 인접 동일 분류를
  병합하고 빈 시간을 미상 공백으로 채운다(타임라인 공백 0).
  행동 트랜잭션을 커밋한 뒤 한국어 설명의 영어 번역을 최선 노력으로 만든다
  (`dlp_cli.translate_cmds.translate_after_actions`, ADR 0043). 번역 실패·서버 미연동은 행동 결과와
  종료 코드에 영향이 없고, 남은 설명은 `dlp translate run --pending`이 채운다.

순서: `dlp prelabel run`·`dlp relations run` 뒤, 검수 배정(`dlp review plan`) 전.
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
from dlp_actions.policy import load_policy
from dlp_actions.runner import run_actions
from dlp_actions.vlm import VlmRequestError, VlmUnavailableError
from dlp_cli.ontology_access import session_ontology
from dlp_cli.schema_cmds import database_url
from dlp_cli.translate_cmds import translate_after_actions
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root


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
    url = args.vlm_url or os.environ.get("DLP_VLM_URL")
    if not url:
        # TODO(real-model): VLM 서버(OpenAI 호환, GPU 또는 CPU용 llama.cpp)가 없으면 행동 구간을
        #   만들지 않는다.
        #   모든 구간을 미상으로 채우면 검수 부담만 늘기 때문이다.
        print(
            "[미연동] VLM 서버 주소가 없습니다 (--vlm-url 또는 DLP_VLM_URL). "
            "행동 구간을 만들지 않습니다."
        )
        return 2
    vlm = policy.vlm
    client = OpenAICompatibleVlm(
        url,
        vlm.model,
        frames=vlm.frames_per_segment,
        max_side=vlm.max_side_px,
        timeout_s=vlm.timeout_s,
        # 서버 API 키 (services/vlm, docs/handover/vlm-server.md). 출력·기록하지 않는다
        api_key=os.environ.get("DLP_VLM_API_KEY") or None,
    )
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
    # 행동 결과는 커밋됐다. 설명 번역은 따로 돌고 실패해도 이 명령을 실패시키지 않는다 (ADR 0043)
    if not args.no_translate:
        translate_after_actions(args.url, args.session_id, args.translate_url)
    return 0


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
