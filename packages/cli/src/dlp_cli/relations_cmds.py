"""관계 도출·커버리지 하위 명령 (WP9, ADR 0011·0026).

등록하는 명령:
- `dlp relations run <세션>` — YAML 규칙(`config/policies/relations.yaml`)으로 현재 라벨에서
  관계(손-객체 접촉, 도구-표면 접촉 등)를 만들고, 도구 작용부 3D 궤적을 표면 평면에 투영해
  표면 커버리지를 계산한다. 결과는 `label_records`에 쓴다.

순서: `dlp prelabel run` 뒤(접촉·3D 궤적이 입력), `dlp actions run`·검수 전.
검수에서 접촉이 고쳐지면 다시 돌린다.

멱등성: 모델 버전은 `relations-<정책 해시>`이고 각 관계의 `derived_by`에는 규칙 ID가 들어간다.
다시 돌리면 같은 결과는 유지하고 달라진 것만 새로 쓰거나 삭제 표시한다(규칙을 바꿔도 차이만
반영). 검수자가 고치거나 지운 관계는 다시 넣지 않는다.
DB만 읽고 쓰며 원본 버킷에는 접근하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_relations.policy import load_policy
from dlp_relations.runner import run_relations
from dlp_schema import load_ontology, repo_root


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp relations run <세션>`: 관계·커버리지를 도출해 DB에 반영하고 요약을 출력한다.

    출력: 정책 버전과 관계 수, 레코드 변화(새로/유지/삭제 표시), 검수 때문에 건너뛴 수,
    (도구→표면)별 커버리지 비율.
    부작용: 한 트랜잭션에서 `label_records` INSERT(새 관계·철회 레코드).
    온톨로지는 `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보).
    """
    root = repo_root()
    policy = load_policy(root)
    ontology = load_ontology(root / "config/ontology/v1")
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        s = run_relations(conn, args.session_id, ontology, policy, datetime.now(UTC))
    engine.dispose()
    print(f"정책 {s.version}: 관계 {s.relations}개")
    print(f"레코드: 새로 {s.inserted}, 유지 {s.kept}, 삭제 표시 {s.retracted}")
    if s.skipped_by_review:
        print(f"검수자가 고치거나 지운 관계 {s.skipped_by_review}개는 다시 넣지 않음")
    for pair, ratio in s.coverage.items():
        print(f"커버리지 {pair}: {ratio:.1%}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`relations run` 하위 명령을 등록한다."""
    rel = sub.add_parser("relations", help="관계 도출과 표면 커버리지")
    rsub = rel.add_subparsers(dest="relations_command", required=True)
    run = rsub.add_parser("run", help="규칙으로 관계를 만들고 커버리지를 계산 (멱등)")
    run.add_argument("session_id")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.set_defaults(func=cmd_run)
