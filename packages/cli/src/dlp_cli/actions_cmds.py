"""행동 구간 하위 명령 (WP10, ADR 0012·0026).

등록하는 명령:
- `dlp actions run <세션> [--vlm-url] [--store] [--url]` — 바디캠 손마다 2단 구조로 행동 구간을
  만든다. 1단: 손목 속도·접촉 신호로 경계 후보. 2단: OpenAI 호환 VLM이 온톨로지 안에서
  분류·설명(JSON Schema 강제, 재시도 뒤 실패하면 미상). 마지막으로 인접 동일 분류를 병합하고
  빈 시간을 미상 공백으로 채운다(타임라인 공백 0).

순서: `dlp prelabel run`·`dlp relations run` 뒤, 검수 배정(`dlp review plan`) 전.
VLM에는 블러본만 보낸다 (라벨링 버킷에서 읽고 원본 버킷에는 접근하지 않는다).
시간 구간은 마스터 타임라인 정수 ms다 (ADR 0019).

종료 코드:
- 0: 성공 (같은 버전 결과가 있으면 그 손은 건너뜀, 멱등).
- 2: VLM 서버 주소가 없음 (`--vlm-url`/`DLP_VLM_URL`). 모든 구간을 미상으로 채우면 검수 부담만
  늘어서 아무것도 만들지 않는다.
- 3: VLM 서버 장애. 트랜잭션이 되돌려져 아무것도 쓰지 않았으므로 서버 복구 뒤 다시 실행한다.

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
from dlp_actions.vlm import VlmUnavailableError
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, load_ontology, repo_root


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp actions run <세션>`: 행동 구간을 만들고 손별 결과를 출력한다.

    인자:
        args.vlm_url: OpenAI 호환 VLM 서버 기본 URL. 없으면 환경 변수 `DLP_VLM_URL`.
        args.store: 라벨링 버킷 저장소 지정 (`s3` 또는 `local:<디렉터리>`), 블러본을 읽는다.
        args.url: DB URL.

    반환: 0 성공 / 2 VLM 주소 없음 / 3 VLM 장애 (모듈 docstring 참고).
    부작용: 한 트랜잭션에서 `label_records`에 action·gap·description 레코드와 이전 버전 철회
    레코드 INSERT, VLM 서버 HTTP 호출, 라벨링 버킷 읽기.
    온톨로지는 `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보).
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
    )
    ontology = load_ontology(root / "config/ontology/v1")
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
    finally:
        engine.dispose()
    for hand, counts in s.hands.items():
        print(f"{hand}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for hand in s.skipped:
        print(f"{hand}: 같은 버전 결과가 있어 건너뜀")
    if s.retracted:
        print(f"이전 버전 레코드 {s.retracted}개 삭제 표시")
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
    run.set_defaults(func=cmd_run)
