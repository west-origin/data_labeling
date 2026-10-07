"""행동 구간 하위 명령."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_actions.clients import OpenAICompatibleVlm
from dlp_actions.policy import load_policy
from dlp_actions.runner import run_actions
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, load_ontology, repo_root


def cmd_run(args: argparse.Namespace) -> int:
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
    raw = store_from_spec(args.store, load_config(root / "config/defaults.yaml").buckets.raw)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        s = run_actions(conn, args.session_id, client, ontology, policy, datetime.now(UTC), raw)
    engine.dispose()
    for hand, counts in s.hands.items():
        print(f"{hand}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    for hand in s.skipped:
        print(f"{hand}: 같은 버전 결과가 있어 건너뜀")
    if s.retracted:
        print(f"이전 버전 레코드 {s.retracted}개 삭제 표시")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    act = sub.add_parser("actions", help="행동 구간 (경계 후보 + VLM 분류)")
    asub = act.add_subparsers(dest="actions_command", required=True)
    run = asub.add_parser("run", help="손별 행동·사이 구간과 설명 초안 (멱등)")
    run.add_argument("session_id")
    run.add_argument("--vlm-url", help="OpenAI 호환 VLM 서버 (기본: DLP_VLM_URL)")
    run.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_run)
