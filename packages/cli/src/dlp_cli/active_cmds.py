"""액티브 러닝 하위 명령 (WP14, ADR 0017).

등록하는 명령:
- `dlp active rank [--limit] [--json]` — 검수 결과에서 클래스별 수정률(검수자가 고친 비율)을
  계산하고, 그 클래스가 많이 나오는 검수 대기 세션에 높은 점수를 준다. 다음에 검수할 세션
  순위를 출력한다. 점수 항목은 `dlp_active.select.register_term`으로 플러그인처럼 붙인다.
- `dlp active fiftyone [--limit] [--name] [--cache]` — 고른 세션을 FiftyOne 데이터셋으로 올려
  눈으로 큐레이션한다. 블러본만 쓴다 (라벨링 버킷). FiftyOne은 선택 설치(`make install-curation`).

순서: 검수가 어느 정도 쌓인 뒤 주기적으로 돌려 `dlp review plan` 대상 세션을 고른다.
정책 출처: `config/policies/active.yaml` (`select` 개수, 점수 항목 가중치,
`fiftyone.dataset_prefix`). 두 명령 모두 DB를 읽기만 한다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import sqlalchemy as sa

from dlp_active.curation import build_samples, push_to_fiftyone, rates_info
from dlp_active.policy import load_policy
from dlp_active.select import rank_sessions
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_schema import load_config, repo_root


def cmd_rank(args: argparse.Namespace) -> int:
    """`dlp active rank`: 세션 점수와 순위를 출력한다.

    인자:
        args.limit: 고를 세션 수. None이면 정책 `select`.
        args.json: 참이면 순위 목록만 JSON으로 출력한다 (다른 도구에 넘길 때).

    출력(기본): 전체 수정률, 수정률 높은 클래스 상위 10개, 순위별 세션(점수, 검수 대기 라벨 수,
    점수에 기여한 상위 클래스). 읽기 전용.
    """
    policy = load_policy(repo_root())
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        ranked, rates = rank_sessions(conn, policy, args.limit)
    engine.dispose()
    if args.json:
        print(json.dumps([asdict(s) for s in ranked], ensure_ascii=False, indent=2))
        return 0
    print(f"전체 수정률 {rates.overall:.3f}, 클래스 {len(rates.classes)}개")
    for k, c in sorted(rates.classes.items(), key=lambda kv: -kv[1].rate)[:10]:
        print(f"  {k:32} 수정률 {c.rate:.3f} (고침 {c.changed}/{c.reviewed})")
    for i, s in enumerate(ranked, start=1):
        top = ", ".join(f"{k} {v:.2f}" for k, v in s.top_classes)
        print(f"{i:3}. {s.session_id}  점수 {s.score:.2f}  검수 대기 {s.pending}  [{top}]")
    return 0


def cmd_fiftyone(args: argparse.Namespace) -> int:
    """`dlp active fiftyone`: 순위 상위 세션의 블러본 프레임·라벨을 FiftyOne 데이터셋으로 만든다.

    인자:
        args.limit: 세션 수 (None이면 정책 `select`).
        args.name: 데이터셋 이름. 없으면 `<정책 접두>top<N>`.
        args.cache: 블러본을 받아 둘 로컬 디렉터리 (기본 `data/fiftyone`).
        args.store: 라벨링 버킷 저장소 지정.

    예외: 설정에서 라벨링 버킷과 원본 버킷이 같으면 `SystemExit` (원본을 큐레이션 도구에 노출하지
    않기 위한 방어). 블러본이 없는 세션은 `[건너뜀]`으로 알린다.
    부작용: 라벨링 버킷 읽기, 로컬 캐시 쓰기, FiftyOne 로컬 DB에 데이터셋 생성.
    """
    root = repo_root()
    policy = load_policy(root)
    # 블러본만 쓴다: 라벨링 버킷 (원본 버킷은 읽지 않는다)
    buckets = load_config(root / "config/defaults.yaml").buckets
    if buckets.labeling == buckets.raw:
        raise SystemExit("큐레이션은 원본 버킷을 읽지 않습니다 (buckets.labeling == buckets.raw)")
    labeling = store_from_spec(args.store, buckets.labeling)
    engine = sa.create_engine(database_url(args.url))
    with engine.connect() as conn:
        ranked, rates = rank_sessions(conn, policy, args.limit)
        samples, notes = build_samples(conn, labeling, Path(args.cache), ranked, policy)
    engine.dispose()
    for note in notes:
        print(f"[건너뜀] {note}")
    name = args.name or f"{policy.fiftyone.dataset_prefix}top{len(ranked)}"
    ds = push_to_fiftyone(name, samples, rates_info(rates))
    print(f"FiftyOne 데이터셋 {name}: 샘플 {len(samples)}개 (fiftyone app launch {name})")
    del ds
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`active rank|fiftyone` 하위 명령을 등록한다."""
    act = sub.add_parser("active", help="액티브 러닝 (다음에 검수할 세션 고르기)")
    asub = act.add_subparsers(dest="active_command", required=True)
    rank = asub.add_parser("rank", help="클래스별 수정률 기반 세션 점수와 순위")
    rank.add_argument("--limit", type=int, help="고를 세션 수 (기본: 정책 select)")
    rank.add_argument("--json", action="store_true")
    rank.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    rank.set_defaults(func=cmd_rank)
    fo = asub.add_parser(
        "fiftyone", help="고른 세션을 FiftyOne 데이터셋으로 (make install-curation)"
    )
    fo.add_argument("--limit", type=int)
    fo.add_argument("--name", help="데이터셋 이름 (기본: 정책 접두 + top<N>)")
    fo.add_argument("--cache", default="data/fiftyone", help="블러본을 받아 둘 디렉터리")
    fo.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    fo.add_argument("--url")
    fo.set_defaults(func=cmd_fiftyone)
