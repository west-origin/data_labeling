"""CLI의 검수 정책·설정 선택과 매체 회수 실행 (ADR 0039).

계정/원본 권한 설정은 기본 정책과 별도 파일로 관리할 수 있다 (`load_review_policy`,
`DLP_REVIEW_POLICY_FILE`). CLI에서 검수 정책을 읽는 곳은 모두 이 함수를 거친다: 검수 명령
(`dlp review …`), 운영 명령(`dlp ops …`), 파일럿 검사(`dlp pilot check`).
`review_setup`은 검수 명령이 쓰는 저장소·도구 클라이언트 묶음을 만들고, 고른 정책의 `cvat`(계정
연결)·`label_studio`(계정 연결)·`media`(검수 화면 매체)를 `ReviewSetup`에 싣는다.
라이브러리 `ReviewSetup`의 지연 로더는
환경 변수를 보지 않고 Git의 기본 `review.yaml`을 읽으므로 CLI가 비워 두면 안 된다 (F-REV-01).
`run_revocations`는 블러를 바꾼 트랜잭션이 커밋된 뒤 회수 대기를 실행한다 (`dlp review …`,
`dlp privacy detect|render`). 회수 실행기는 검수 정책 값을 쓰지 않으므로 회수가 스스로 만드는
설정(`revocation_only`)은 검수 정책 파일을 읽지 않는다. 정책 파일이 없거나 깨져도 낡은 라벨링
매체는 지운다 (노출을 줄이는 경로를 설정 오류에 묶지 않는다).

공개 이름:
- `REVOCATION_PENDING_EXIT`: 회수 대기가 남았을 때의 종료 코드 (4, argparse 사용 오류 2와 다르다).
- `load_review_policy`, `ai_display_config`(AI 1차 검수 판정 사용 설정, WP17.6), `review_setup`,
  `run_revocations`, `print_revocations`.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import sqlalchemy as sa

from dlp_aireview.policy import load_policy as load_aireview_policy
from dlp_cli.raw_access import raw_store
from dlp_media.storage import STORE_ERRORS, S3Store
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import request_media_revocation
from dlp_review.ai_display import AiDisplayConfig
from dlp_review.clients import CvatClient, LabelStudioClient, ToolError
from dlp_review.ops.policy import ReviewOpsPolicy, load_policy
from dlp_review.revoke import RevocationReport, execute_media_revocations, timeseries_sessions
from dlp_review.tasks import ReviewSetup
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import pending_revocations

# 회수 대기가 남았을 때 CLI 종료 코드. cron(`dlp review revoke --pending`)은 이 값을 경보로 쓴다
REVOCATION_PENDING_EXIT = 4


def load_review_policy(root: Path) -> ReviewOpsPolicy:
    """DLP_REVIEW_POLICY_FILE이 있으면 그 파일을 엄격히 읽는다. 오류 시 기본값으로 되돌리지 않는다.

    기본 YAML은 권한자가 없는 상태를 유지한다. 로컬 파일럿·운영의 실제 계정 연결은 별도 파일에
    설정한다. 라이브러리의 load_policy(root)는 환경 변수와 무관하게 기본 정책을 읽는다. 그래서
    CLI는 이 함수로 읽은 정책을 라이브러리 함수와 `ReviewSetup`에 명시적으로 넘긴다.
    배정 비율(`ratios`)은 고른 파일과 무관하게 `config/defaults.yaml`의 `review` 절이다.

    예외: 파일이 없으면 OSError(FileNotFoundError), 값·키가 맞지 않으면 pydantic ValidationError.
    """
    selected = os.environ.get("DLP_REVIEW_POLICY_FILE")
    return load_policy(root, Path(selected) if selected else None)


def ai_display_config(policy: ReviewOpsPolicy) -> AiDisplayConfig | None:
    """검수 정책의 `ai_review`로 AI 1차 검수 판정 사용 설정을 만든다 (끔이면 None, WP17.6).

    허용 판정 버전은 정책 `ai_review.versions`, 비면 `config/policies/aireview.yaml`의 지금 판정
    버전(`AiReviewPolicy.verdict_version`) 하나다. 판정 대상 조건의 정책 값
    (`AiReviewPolicy.target_rule`: 종류·출처·긴 상위 구간 상한)도 같은 파일에서 넘겨, 검수 화면
    판정률의 분모를 판정 실행과 같은 조건으로 센다. 검수 패키지는 판정 정책을 읽지 않으므로 CLI가
    계산해 넘긴다 (의존 표, 설계 C11·C23). 예외: 판정 정책 파일이 없거나 깨졌으면 OSError·
    ValidationError (끔이면 읽지 않는다).
    """
    ai = policy.ai_review
    if ai.mode == "off":
        return None
    judge = load_aireview_policy(repo_root())
    versions = ai.versions or (judge.verdict_version,)
    return AiDisplayConfig(policy=ai, versions=tuple(versions), targets=judge.target_rule)


def review_setup(
    purpose: str,
    url: str | None,
    *,
    cvat: bool = True,
    label_studio: bool = True,
    revocation_only: bool = False,
    policy: ReviewOpsPolicy | None = None,
) -> tuple[ReviewSetup, list[str]]:
    """검수 연동에 필요한 저장소·도구 클라이언트를 묶는다. (설정, 경고).

    인자:
        purpose: 원본 감사 기록의 목적 문자열 (예: `review.create`). 회수 실행기는 원본을 읽지
            않고 원본 버킷 이름만 쓴다.
        url: 감사 기록용 DB URL (None이면 `DLP_DATABASE_URL`/기본값).
        cvat / label_studio: False면 그 도구 클라이언트를 만들지 않는다 (None).
        revocation_only: True면 매체 회수 전용 설정이다 (`run_revocations`만 쓴다).
            CVAT 로그인 실패(`ToolError`·`httpx.HTTPError`)를 None과 경고로 바꾸고, 검수 정책을
            읽지 않는다(`policy`가 없으면 `cvat_config`·`media`가 None). 회수 실행기는 이 값을 쓰지
            않으므로 정책 파일이 없거나 깨져도 매체 삭제를 막지 않는다. 이 설정을 작업 생성·수집·
            웹훅에 쓰면 `ReviewSetup` 지연 로더가 Git 기본값을 읽으므로 그 용도로 쓰지 않는다.
        policy: 명령이 이미 읽은 검수 정책. None이면 `load_review_policy`로 읽는다 (고른 파일,
            `revocation_only`면 읽지 않는다). 명령이 정책을 따로 쓰면(원본 검수자·수집·웹훅) 같은
            객체를 넘겨 한 출처로 맞춘다.

    반환: `ReviewSetup`과 경고 문장 목록.
    - `raw`: 원본 버킷 감사 저장소 (항상 S3).
    - `labeling`: 라벨링 버킷 (서비스 자격 증명: 블러본·작업 매체 쓰기, 회수 삭제).
    - `labeling_reader`: 라벨러 자격 증명(읽기 전용) — 검수 화면 서명 URL용.
    - `ontology`: None. 작업 생성 시 DB에서 각 세션 버전의 사전을 읽는다.
    - `cvat_config`·`label_studio_config`·`media`: 고른 정책의 `cvat`·`label_studio`·`media`
      (라이브러리 지연 로더를 쓰지 않는다). `revocation_only`이고 `policy`가 없으면 None.
    - `ai`: 고른 정책의 `ai_review`로 만든 AI 1차 검수 판정 사용 설정 (`ai_display_config`, 끔이거나
      `revocation_only`면 None).
    부작용: 설정 파일을 읽고 저장소·HTTP 클라이언트를 만든다. 정책은 클라이언트보다 먼저 읽어
    파일 오류면 도구에 접속하지 않고 실패한다 (`revocation_only`면 읽지 않는다). CVAT 클라이언트는
    생성 중 로그인 API를 호출한다. `revocation_only`가 아니면 로그인 실패·통신 오류는 호출자에게
    전파된다.
    """
    root = repo_root()
    selected = policy
    if selected is None and not revocation_only:
        selected = load_review_policy(root)
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    warnings: list[str] = []
    client: CvatClient | None = None
    if cvat:
        try:
            client = CvatClient.from_env()
        except (ToolError, httpx.HTTPError) as exc:
            if not revocation_only:
                raise
            warnings.append(f"CVAT에 접속하지 못했습니다 ({type(exc).__name__}): CVAT 회수는 대기")
    setup = ReviewSetup(
        raw=raw_store("s3", url, purpose),
        labeling=S3Store.from_env(buckets.labeling),
        labeling_reader=S3Store.labeler_from_env(buckets.labeling),
        ontology=None,
        cvat=client,
        label_studio=LabelStudioClient.from_env() if label_studio else None,
        media=selected.media if selected is not None else None,
        cvat_config=selected.cvat if selected is not None else None,
        label_studio_config=selected.label_studio if selected is not None else None,
        ai=ai_display_config(selected) if selected is not None and not revocation_only else None,
    )
    return setup, warnings


def run_revocations(
    engine: sa.Engine,
    session_id: str | None,
    setup: ReviewSetup | None = None,
    *,
    sweep: bool = False,
    url: str | None = None,
    purpose: str = "review.revoke",
) -> RevocationReport:
    """커밋된 회수 대기를 실행한다 (호출자 트랜잭션 밖에서 부른다).

    1. sweep이고 세션이 있으면 짧은 트랜잭션에서 렌더 해시가 지금과 다른 작업에 회수를 요청한다
       (`reason="sweep"`, 구 작업 제외). 알려지지 않은 경로로 블러본이 바뀐 경우의 백스톱이다.
    2. 대기도 시계열 CSV를 확인할 사용 중지 세션(`revoke.timeseries_sessions`: Label Studio 작업이
       있던 사용 중지 세션)도 없으면 빈 보고서를 돌려준다 (설정을 만들지 않는다). 사용 중지 전에
       모든 작업의 회수가 끝났으면 대기는 없지만 CSV는 남아 있을 수 있어 실행기를 돌린다.
    3. setup이 없으면 `review_setup(revocation_only=True)`로 만든다. CVAT에 접속하지 못해도 매체
       삭제는 하고, 검수 정책 파일은 읽지 않는다 (회수는 정책 값을 쓰지 않는다. 파일이 없거나
       깨져도 `dlp privacy detect|render` 뒤 회수와 `dlp review revoke [--pending]`이 돈다). 작업
       대기가 없으면(CSV만 확인) 도구 클라이언트를 만들지 않는다.
    4. `execute_media_revocations`로 매체(사본, 사용 중지 세션 CSV) → CVAT → Label Studio 순으로
       끊는다.

    인자: session_id가 None이면 모든 세션의 대기 (`--pending`). url: 설정을 만들 때의 DB URL.
    반환: `RevocationReport` (setup을 만들며 생긴 경고는 warnings 앞에 붙는다).
    """
    if sweep and session_id is not None:
        policy = setup.privacy_policy() if setup is not None else load_privacy_policy(repo_root())
        with engine.begin() as conn:
            request_media_revocation(
                conn, session_id, policy, datetime.now(UTC), reason="sweep", legacy=False
            )
    with engine.connect() as conn:
        pending = bool(pending_revocations(conn, session_id))
        if not pending and not timeseries_sessions(conn, session_id):
            return RevocationReport()
    warnings: list[str] = []
    if setup is None:
        try:
            setup, warnings = review_setup(
                purpose, url, revocation_only=True, cvat=pending, label_studio=pending
            )
        except SETUP_ERRORS as exc:
            return _setup_failed(engine, session_id, exc)
    report = execute_media_revocations(
        engine,
        session_id,
        labeling=setup.labeling,
        raw_bucket=setup.raw.bucket,
        cvat=setup.cvat,
        label_studio=setup.label_studio,
    )
    report.warnings[:0] = warnings
    return report


# 회수 전용 설정을 만들 때 날 수 있는 오류 (저장소 지정·AWS 프로필·자격 증명·설정, Label Studio
# 설정). CVAT 접속 실패는 `review_setup`이 경고로 바꾼다
SETUP_ERRORS: tuple[type[Exception], ...] = (*STORE_ERRORS, ToolError, ValueError, KeyError)


def _setup_failed(engine: sa.Engine, session_id: str | None, exc: Exception) -> RevocationReport:
    """설정을 만들지 못한 회수 보고: 아무것도 끊지 않았고 DB의 대기를 그대로 보인다 (F-PMS-17)."""
    with engine.connect() as conn:
        pending = [t.task_key for t in pending_revocations(conn, session_id)]
    reason = f"{type(exc).__name__}: {exc}"
    return RevocationReport(
        errors=[f"[회수 실행 준비 실패] {reason}. 회수 요청은 커밋됐고 대기로 남았습니다"],
        pending=pending,
        setup_error=reason,
    )


def print_revocations(report: RevocationReport, session_id: str | None) -> int:
    """회수 실행 보고를 출력하고 종료 코드를 돌려준다 (대기가 남으면 `REVOCATION_PENDING_EXIT`).

    실패가 먼저 나온다. 매체 실패(`[매체 노출 남음]`)는 Label Studio 영상·시계열이 계속 열린다는
    뜻이다. 대기는 작업 키와 지우지 못한 사용 중지 세션 시계열 CSV 키다
    (`RevocationReport.remaining`).
    """
    for line in report.errors:
        print(f"  ! {line}")
    if report.media_done or report.tools_done or report.snapshots:
        print(
            f"  회수: 매체 {len(report.media_done)}개, 도구 작업 {len(report.tools_done)}개, "
            f"스냅숏 {len(report.snapshots)}개"
        )
    for line in report.warnings:
        print(f"  경고: {line}")
    if report.complete:
        return 0
    target = session_id or "--pending"
    left = report.remaining
    if not left:  # 설정 실패, 시계열 CSV 확인만 남음
        print(f"  회수 확인 대기: 설정을 고친 뒤 dlp review revoke {target}")
        return REVOCATION_PENDING_EXIT
    print(f"  회수 대기 {len(left)}개: {', '.join(left[:5])} (dlp review revoke {target})")
    return REVOCATION_PENDING_EXIT
