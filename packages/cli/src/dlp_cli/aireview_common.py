"""AI 1차 검수 명령의 공용 도우미 (`aireview_cmds`·`aireview_run_cmds`, WP17).

- 서버 주소: `vlm_endpoints`(`--vlm-url` 여러 번 또는 `DLP_VLM_URLS`, `이름=URL` 가능),
  `vlm_api_key`(`DLP_VLM_API_KEY`, 출력·기록하지 않는다), `no_url`(주소 없음 안내, 종료 2).
  `DLP_VLM_URL`(행동 분류 서버)은 쓰지 않는다 (판정 모델이 달라 늘 서버 모델 불일치가 난다).
- 인자 형식: `parse_ints`(`1,8,16`), `parse_groups`(`temporal,spatial`), `add_url_arg`.
- `flagged_spans`: flagged 선택 모드의 우선순위 사유 구간 (검수 운영 정책의 `flag_unit`, 검수
  단위마다, `DLP_REVIEW_POLICY_FILE` 반영).
- 종료 코드 상수: `NO_URL`(2), `UNAVAILABLE`(3), `CONFIG`(5).
"""

from __future__ import annotations

import argparse
import os

import sqlalchemy as sa

from dlp_aireview.backend import parse_endpoints
from dlp_cli.review_config import load_review_policy
from dlp_review.ops.priority import flag_unit, units_for
from dlp_review.ops.runner import known_classes
from dlp_schema import repo_root
from dlp_schema.aireview import AiGroup
from dlp_schema.labels import LabelRecord
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import Session, StreamKind

NO_URL = 2
UNAVAILABLE = 3
CONFIG = 5


def vlm_endpoints(args: argparse.Namespace) -> dict[str, str] | None:
    """`--vlm-url` 또는 `DLP_VLM_URLS` → {이름: URL} (없으면 None).

    Raises:
        ValueError: 잘못된 지정.
    """
    specs: list[str] = list(args.vlm_url or [])
    if not specs and os.environ.get("DLP_VLM_URLS"):
        specs = [os.environ["DLP_VLM_URLS"]]
    return parse_endpoints(specs) if specs else None


def vlm_api_key() -> str | None:
    """서버 API 키 (출력·기록하지 않는다)."""
    return os.environ.get("DLP_VLM_API_KEY") or None


def no_url() -> int:
    """주소가 없을 때 안내 (종료 2)."""
    print(
        "[미연동] AI 검수 VLM 서버 주소가 없습니다 (--vlm-url 또는 DLP_VLM_URLS). "
        "DLP_VLM_URL(행동 분류 서버)은 쓰지 않습니다."
    )
    return NO_URL


def parse_ints(text: str) -> tuple[int, ...]:
    """`1,8,16` → (1, 8, 16) (argparse 형식 함수)."""
    try:
        values = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"쉼표로 나눈 정수가 아닙니다: {text}") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"1 이상의 정수가 필요합니다: {text}")
    return values


def parse_groups(text: str) -> tuple[AiGroup, ...]:
    """`temporal,spatial` → 묶음 (argparse 형식 함수)."""
    try:
        return tuple(AiGroup(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"모르는 묶음: {text}") from exc


def add_url_arg(p: argparse.ArgumentParser) -> None:
    """서버 주소 인자."""
    p.add_argument(
        "--vlm-url", action="append",
        help="AI 검수 VLM 서버 (여러 번, 이름=URL 가능. 기본: DLP_VLM_URLS)",
    )  # fmt: skip


def flagged_spans(
    conn: sa.Connection, session: Session, current: list[LabelRecord]
) -> list[FlaggedSpan]:
    """flagged 모드의 우선순위 사유 구간 (검수 운영 정책의 `flag_unit`, 검수 단위마다)."""
    review_policy = load_review_policy(repo_root())
    known = known_classes(conn, session.session_id)
    glove = any(s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams)
    spans: list[FlaggedSpan] = []
    for unit in units_for(session, review_policy):
        spans += flag_unit(
            unit.select(current), review_policy, known_classes=known, glove_session=glove
        )
    return spans
