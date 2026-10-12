"""프라이버시 게이트 하위 명령 (WP5, ADR 0005·0023·0024).

등록하는 명령 (정상 순서):
1. `dlp privacy detect <세션>` — 블러 대상(얼굴, 문서·화면, 반사면, QR·바코드 등)을 자동 탐지해
   블러 트랙 라벨(`blur_track`, `LabelRecord`)과 검수 우선 구간을 만든다. 쓸 수 있는 탐지기가 없는
   대상은 "전수 검수 필요"로 표시한다.
2. (검수) `dlp review create --stage privacy` → 원본 접근 권한자가 블러 검수 → `dlp review collect`.
3. `dlp privacy approve <세션>` — 모든 운영 블러 라벨의 사람 검수와 영상 스트림별 최신 탐지
   세대의 검수 수집을 확인한 뒤 승인한다. 블러가 0개여도 스트림 검수는 필요하다.
4. `dlp privacy render <세션>` — 승인된 세션의 블러본(모자이크, 오디오 제거)을 라벨링 버킷에
   렌더한다. 이후 단계(프리라벨 검수 화면, 내보내기)는 이 블러본만 쓴다.
- `dlp privacy audit-sample [--week]` — 매주 승인된 블러본에서 잔여 누락 감사 표본을 뽑고, 최근 주
  누락률로 블러 검수 방식(전수/표본)을 판정한다. 감사 결과 입력은 `dlp ops privacy-audit`.
- `dlp privacy hold <세션> --reason [--actor]` — 사고 대응(노출 신고) 중 승인을 보류한다: 승인 해제,
  렌더 기록 무효, 배포된 라벨링 작업 전부 회수 요청(커밋 뒤 실행). 보류 중에는 승인이 거부된다.
  `--release`는 해제 기록만 남기고, 재승인(approve → render)을 다시 거친다 (ADR 0042, F-REV-18).

정책 출처: `config/policies/privacy.yaml`(탐지기·대상·렌더), `config/defaults.yaml`의
`privacy.full_review_exit`(표본 감사 비율, 연속 주 수)와
`success_criteria.residual_blur_miss_per_hour_max`.

주의:
- `detect`·`render`는 원본 영상을 읽으므로 `raw_store` 감사 저장소를 쓴다 (ADR 0020). 블러본은
  반드시 라벨링 버킷에 쓴다 (원본 버킷이면 `render_session`이 거부).
- 프라이버시 배포는 사람 승인 후에만 한다. `detect`는 게이트를 통과하고 승인된 재학습 블러 모델이
  있으면 `strict=True`로 불러 탐지기 결과와 합집합으로 쓴다 (못 부르면 조용히 넘어가지 않고 실패).
- 블러가 바뀌면(탐지 결과가 바뀐 스트림) 같은 트랜잭션에서 이미 배포한 라벨링 작업 매체에 회수를
  요청하고, 커밋 뒤 `review_config.run_revocations`로 라벨링 버킷 사본·도구 작업을 지운다
  (ADR 0039). `render`도 같은 트랜잭션에서 렌더 해시가 지금과 다른 작업에 회수를 요청한다
  (렌더 정책 변경 등 알려지지 않은 경로의 백스톱). 대기가 남으면 종료 코드 4.
- 결과가 0개여도 DB에 완료 탐지 세대를 남긴다. 최신 완료 세대의 활성 버전 집합이 같으면
  건너뛴다. 과거 버전으로 복귀하면 새 세대다. DB 이력 없이 원본 버킷의 0건 탐지 표시만 있으면
  완료 세대를 복구하고 새 검수를 요구한다 (ADR 0033).
  블러 라벨의 키프레임 시각은 그 스트림의 PTS 시각(정수 ms)이다 (ADR 0019).
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.review_config import print_revocations, run_revocations
from dlp_cli.schema_cmds import database_url
from dlp_media.audit import current_actor
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_privacy.audit import (
    AuditResult,
    audit_candidates,
    iso_week,
    iso_week_bounds,
    previous_weeks,
    review_mode,
    select_audit_sample,
    stream_duration_ms,
    weekly_miss_rates,
)
from dlp_privacy.detectors import build_detectors
from dlp_privacy.hold import hold_session, release_hold
from dlp_privacy.policy import load_policy
from dlp_privacy.runner import (
    approve_session,
    detect_session,
    render_session,
    request_media_revocation,
)
from dlp_schema import load_config, repo_root
from dlp_schema.db.repository import get_session, list_privacy_audits
from dlp_schema.session import Session, Stream, StreamKind
from dlp_train.deployed import deployed_predictors
from dlp_train.policy import load_policy as load_training_policy
from dlp_train.trainers import LoadContext


def _engine(args: argparse.Namespace) -> sa.Engine:
    """`--url`(없으면 `DLP_DATABASE_URL`/기본값)로 SQLAlchemy 엔진을 만든다. 호출자가
    `dispose`한다."""
    return sa.create_engine(database_url(args.url))


def cmd_detect(args: argparse.Namespace) -> int:
    """`dlp privacy detect <세션>`: 블러 대상 자동 탐지.

    흐름:
    1. 정책(`privacy.yaml`)으로 탐지기를 만든다. 가중치가 없는 등 만들 수 없는 탐지기는 `missing`에
       이유와 함께 모이고 화면에 `[탐지기 없음]`으로 알린다.
    2. 배포된 재학습 블러 모델(`stage=privacy`)을 MLflow 산출물 버킷에서 불러온다 (`strict=True`).
    3. `detect_session`이 영상 스트림마다 블러 트랙 라벨·완료 탐지 세대를 DB에 쓰고,
       검수 우선 구간 JSON을 원본 버킷에 쓴다.

    인자:
        args.session_id: 대상 세션 (온톨로지 버전이 있어야 한다).
        args.store: `s3` 또는 `local:<디렉터리>`. 원본·라벨링·MLflow 버킷 모두 같은 지정으로 연다.
        args.url: DB URL.

    반환: 0. 탐지기가 없는 대상은 실패가 아니라 "전수 검수 필요"로 출력만 한다. 커밋 뒤 회수
    대기가 남으면 `REVOCATION_PENDING_EXIT`(4).
    부작용: 한 트랜잭션에서 `label_records`에 `blur_track` 추가(이전 모델 버전의 미검수 라벨은
    철회), 세션 프라이버시 상태 갱신, 블러가 바뀐 스트림의 라벨링 작업 회수 요청(ADR 0039),
    라벨링 버킷의 이전 블러본 렌더 기록 무효화(ADR 0024), 원본 버킷 읽기와 쓰기(검수 우선 구간
    JSON, 결과 0개 버전의 탐지 표시; 감사 기록 `privacy.detect`). 커밋 뒤 회수 대기 실행
    (CVAT에 접속하지 못해도 매체 삭제는 하고 도구 작업은 대기로 남긴다. 검수 정책 파일
    `DLP_REVIEW_POLICY_FILE`은 읽지 않으므로 그 파일 오류도 매체 삭제를 막지 않는다).
    """
    root = repo_root()
    policy = load_policy(root)
    raw = raw_store(args.store, args.url, "privacy.detect")
    detectors, missing = build_detectors(policy, root)
    for name, reason in missing.items():
        print(f"[탐지기 없음] {name}: {reason}")
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    # 같은 실행 안의 모든 라벨이 같은 생성 시각을 갖도록 한 번만 잰다 (UTC, 시간대 포함)
    now = datetime.now(UTC)
    engine = _engine(args)
    with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
        # 게이트를 통과하고 사람이 승인한 재학습 블러 모델이 있으면 탐지기 결과와 합집합으로 쓴다
        ontology_version = get_session(conn, args.session_id).ontology_version or ""
        deployed = deployed_predictors(
            conn,
            store_from_spec(args.store, buckets.mlflow),
            load_training_policy(root),
            "privacy",
            LoadContext(now=now, ontology_version=ontology_version),
            Path(tmp),
            strict=True,
        )
        for note in deployed.notes:
            print(f"[재학습 모델] {note}")
        s = detect_session(
            conn, args.session_id, raw, detectors, missing, policy, now,
            extra=deployed.predictors,
            # 블러가 바뀐 스트림의 이전 블러본 렌더 기록을 무효로 둔다 (ADR 0024)
            labeling=store_from_spec(args.store, buckets.labeling),
        )  # fmt: skip
    # 블러가 바뀐 스트림의 회수 요청은 위 커밋에 있다. 외부 채널은 커밋 뒤에 끊는다
    report = run_revocations(engine, args.session_id, url=args.url, purpose="privacy.detect")
    engine.dispose()
    for stream, n in s.detected.items():
        print(f"{stream}: 블러 트랙 {n}개")
    for stream in s.skipped:
        print(f"{stream}: 같은 모델 버전 결과가 있어 건너뜀")
    for target, reason in s.missing.items():
        print(f"[전수 검수 필요] {target}: 쓸 수 있는 탐지기가 없음 ({reason})")
    if s.revocations:
        print(f"블러가 바뀌어 라벨링 작업 {len(s.revocations)}개 회수: {', '.join(s.revocations)}")
    return print_revocations(report, args.session_id)


def cmd_approve(args: argparse.Namespace) -> int:
    """`dlp privacy approve <세션>`: 세션을 프라이버시 승인한다.

    조건(`approve_session`): 자동 탐지가 실행되었고, 영상 스트림마다 마지막 탐지 뒤의 블러 검수
    작업이 수집되었으며, 현재 블러 라벨이 모두 사람 검수(승인·수정)를 거쳤다(표본 검증만으로는 안
    된다). 조건을 못 맞추면 `PrivacyGateError`로 끝난다.

    부작용: 한 트랜잭션에서 세션의 `privacy_state`·`lifecycle_state` 갱신. 원본은 읽지 않는다.
    """
    engine = _engine(args)
    with engine.begin() as conn:
        session = approve_session(conn, args.session_id)
    engine.dispose()
    print(f"{session.session_id}: 프라이버시 승인 ({session.lifecycle_state.value})")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """`dlp privacy render <세션>`: 승인된 세션의 블러본을 라벨링 버킷에 렌더한다.

    같은 블러 라벨 집합·렌더 정책의 블러본이 이미 있으면 건너뛴다 (멱등). 출력은 스트림 → 블러본
    URI.

    렌더한 트랜잭션에서 렌더 해시가 지금 블러본과 다른 라벨링 작업에 회수를 요청한다 (백스톱,
    ADR 0039). 렌더 해시에는 `privacy.yaml render` 절 전체가 들어가므로 블러 강도뿐 아니라 블러와
    무관한 인코더 설정(crf, encoder_rate)만 바꿔도 그 세션의 해시 있는 라벨링 작업이 모두 회수되고
    도구 작업이 지워진다. 커밋 뒤 회수 대기를 실행하고(`detect`와 같이 검수 정책 파일은 읽지
    않는다), 남으면 `REVOCATION_PENDING_EXIT`(4).

    예외: 승인 전 세션, 라벨링 버킷이 원본 버킷과 같을 때, 오디오를 남기는 정책이면
    `PrivacyGateError`.
    부작용: 원본 영상 읽기(감사 기록 `privacy.render`), 라벨링 버킷에 블러본·렌더 기록 쓰기,
    `review_tasks` 회수 요청, 커밋 뒤 라벨링 버킷 사본·도구 작업 삭제.
    """
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    raw = raw_store(args.store, args.url, "privacy.render")
    labeling = store_from_spec(args.store, buckets.labeling)
    policy = load_policy(root)
    engine = _engine(args)
    with engine.begin() as conn:
        uris = render_session(conn, args.session_id, raw, labeling, policy)
        requested = request_media_revocation(
            conn, args.session_id, policy, datetime.now(UTC), reason="render", legacy=False
        )
    report = run_revocations(engine, args.session_id, url=args.url, purpose="privacy.render")
    engine.dispose()
    for stream, uri in uris.items():
        print(f"{stream}: {uri}")
    if requested:
        keys = ", ".join(t.task_key for t in requested)
        print(f"렌더 해시가 바뀐 라벨링 작업 {len(requested)}개 회수: {keys}")
    return print_revocations(report, args.session_id)


def cmd_hold(args: argparse.Namespace) -> int:
    """`dlp privacy hold <세션> --reason R [--release] [--actor A]`: 승인 보류·해제 (ADR 0042).

    보류: 한 트랜잭션에서 보류 기록(추가만), 승인 해제(AUTO_BLURRED), 모든 영상 스트림의 배포된
    라벨링 작업 회수 요청(사유 `hold:<행위자>:<사유>`), 렌더 기록 무효(`--store`의 라벨링 버킷).
    커밋 뒤 회수 실행기를 돌린다 (검수 정책 파일은 읽지 않는다). 보류 중에는 블러본 열기·렌더·
    라벨링 작업 생성·승인이 거부된다. 블러 검수 작업은 만들 수 있다. 다시 보류하면 기록은 더하지
    않고 회수 요청·무효화만 다시 한다.
    해제(`--release`): 해제 기록만 남긴다. 승인은 풀린 채라 재승인 절차를 다시 거친다.

    행위자는 `--actor`, 없으면 `DLP_ACTOR`, 그것도 없으면 OS 사용자다.
    반환: 0, 보류 뒤 회수 대기가 남으면 `REVOCATION_PENDING_EXIT`(4).
    예외: 보류 중이 아닌데 해제하면 `PrivacyGateError`, 사유가 비면 ValueError.
    """
    actor = args.actor or current_actor()
    now = datetime.now(UTC)
    engine = _engine(args)
    if args.release:
        with engine.begin() as conn:
            event = release_hold(conn, args.session_id, now, reason=args.reason, actor=actor)
        engine.dispose()
        print(f"{args.session_id}: 승인 보류 해제 ({event.actor}: {event.reason})")
        print(
            "  승인은 풀린 채입니다. 재승인: (필요하면 블러 검수 → dlp review collect) → "
            f"dlp privacy approve {args.session_id} → dlp privacy render {args.session_id} → "
            f"다시 만들기 (dlp review verify {args.session_id}가 명령을 보인다)"
        )
        return 0
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    labeling = store_from_spec(args.store, buckets.labeling)
    with engine.begin() as conn:
        result = hold_session(
            conn, args.session_id, load_policy(root), now, reason=args.reason, actor=actor,
            labeling=labeling,
        )  # fmt: skip
    # 회수 요청은 위 커밋에 있다. 외부 채널은 커밋 뒤에 끊는다
    report = run_revocations(engine, args.session_id, url=args.url, purpose="privacy.hold")
    engine.dispose()
    held = result.held_since
    state = "새 보류" if result.event is not None else "이미 보류 중"
    print(f"{args.session_id}: 승인 보류 ({state}, {held.actor}: {held.reason}, {held.at})")
    print(f"  이전 프라이버시 상태 {result.previous_state.value} → 승인 해제, 렌더 기록 무효 "
          f"{result.invalidated or '-'}")  # fmt: skip
    if result.revocations:
        keys = ", ".join(t.task_key for t in result.revocations)
        print(f"  라벨링 작업 {len(result.revocations)}개 회수: {keys}")
    return print_revocations(report, args.session_id)


def cmd_audit_sample(args: argparse.Namespace) -> int:
    """그 주 승인된 블러본에서 잔여 누락 감사 표본을 뽑고, 블러 검수 방식(전수/표본)을 판정한다.

    인자:
        args.week: ISO 주 문자열(예: `2026-W41`). None이면 지금(UTC) 기준 이번 주.
        args.url: DB URL.
        args.store: 원본 저장소 ('s3' 또는 'local:<디렉터리>'). 바디캠이 아닌 후보 스트림의 길이를
            PTS 인덱스로 읽을 때만 쓴다 (감사 저장소, 읽기 기록이 남는다).

    흐름:
    1. 판정 창 = 이번 주를 포함해 거슬러 `full_review_exit.weeks_below_target`주.
    2. 이번 주 후보(그 주 블러 검수를 수집했고 지금 승인 상태인 영상 스트림) 중
       `audit_sample_ratio`만큼(최소 1개)을 결정적으로 뽑는다 (같은 주·후보면 같은 표본).
    3. 판정 창의 `privacy_audits` 기록으로 주별 잔여 누락률(누락 수/시간)을 계산한다.
       감사가 없던 주는 None이며 통과로 보지 않는다.
    4. 모든 주가 목표 이하면 "표본", 아니면 "전수".

    반환: 항상 0 (판정 결과는 출력으로만 알린다). DB는 읽기만 한다. 바디캠이 아닌 후보가 있으면
    원본 버킷에서 PTS 인덱스를 읽으므로 `raw_access_log`에 접근 기록이 남는다.
    감사자는 원 검수자와 다른 사람이어야 한다 (출력에 원 검수자를 함께 보여 준다).
    """
    root = repo_root()
    config = load_config(root / "config" / "defaults.yaml")
    exit_policy = config.privacy.full_review_exit
    week = args.week or iso_week(datetime.now(UTC))
    weeks = previous_weeks(week, exit_policy.weeks_below_target)
    engine = _engine(args)
    # 원본 저장소는 바디캠이 아닌 후보가 있을 때 처음 한 번만 만든다 (바디캠은 세션 길이를 쓴다)
    raw: list[ObjectStore] = []
    with engine.connect() as conn, tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        def duration(session: Session, stream: Stream) -> int:
            """후보 스트림 자신의 길이 (`stream_duration_ms`)."""
            if stream.kind is not StreamKind.BODYCAM and not raw:
                raw.append(raw_store(args.store, args.url, "privacy.audit-sample"))
            return stream_duration_ms(session, stream, raw[0] if raw else None, work)

        candidates = audit_candidates(conn, week, duration)
        # 판정 창 전체(가장 오래된 주 시작 ~ 이번 주 끝)의 감사 기록을 한 번에 읽는다
        start, _ = iso_week_bounds(weeks[0])
        _, end = iso_week_bounds(week)
        audits = [
            (a.audited_at, AuditResult(a.session_id, a.stream_id, a.duration_ms, a.misses,
                                       a.auditor, a.blur_reviewer))
            for a in list_privacy_audits(conn, start, end)
        ]  # fmt: skip
    engine.dispose()
    sample = select_audit_sample(candidates, exit_policy.audit_sample_ratio, week)
    print(f"{week}: 승인된 블러본 {len(candidates)}개 중 감사 표본 {len(sample)}개")
    for c in sample:
        print(f"  {c.session_id}/{c.stream_id} (원 검수자 {c.blur_reviewer}, 감사자는 다른 사람)")
    rates = weekly_miss_rates(audits, weeks)
    for w, r in zip(weeks, rates, strict=True):
        print(f"  {w}: " + ("감사 없음 (통과로 보지 않음)" if r is None else f"{r:.3f}/시간"))
    # 목표(시간당 잔여 누락 최대치)가 아직 정해지지 않았으면(None) review_mode가 전수로 판정한다
    target = config.success_criteria.residual_blur_miss_per_hour_max
    mode = review_mode(rates, target, exit_policy.weeks_below_target)
    print(f"블러 검수 방식: {'표본' if mode == 'sampled' else '전수'} (목표 {target})")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`privacy detect|approve|render|audit-sample|hold` 하위 명령을 등록한다.

    `approve`는 원본·라벨링 저장소를 쓰지 않으므로 `--store` 인자가 없다.
    """
    privacy = sub.add_parser("privacy", help="프라이버시 게이트")
    psub = privacy.add_subparsers(dest="privacy_command", required=True)
    for name, func, help_text in (
        ("detect", cmd_detect, "블러 대상 자동 탐지 → 블러 트랙 라벨 + 검수 우선 구간"),
        ("approve", cmd_approve, "모든 블러 라벨이 사람 검수를 거쳤으면 프라이버시 승인"),
        ("render", cmd_render, "승인된 세션의 블러본을 라벨링 버킷에 렌더"),
    ):
        p = psub.add_parser(name, help=help_text)
        p.add_argument("session_id")
        p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
        if name != "approve":
            p.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
        p.set_defaults(func=func)
    audit = psub.add_parser(
        "audit-sample", help="그 주 승인된 블러본의 잔여 누락 감사 표본과 블러 검수 방식(전수/표본)"
    )
    audit.add_argument("--week", help="ISO 주 (예: 2026-W41, 기본: 이번 주)")
    audit.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    # 3인칭 등 바디캠이 아닌 스트림의 길이를 PTS 인덱스(원본 버킷)에서 읽는다
    audit.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    audit.set_defaults(func=cmd_audit_sample)
    hold = psub.add_parser(
        "hold", help="사고 대응 승인 보류 (배포된 라벨링 작업 회수, --release로 해제 기록)"
    )
    hold.add_argument("session_id")
    hold.add_argument("--reason", required=True, help="보류·해제 사유 (감사 기록)")
    hold.add_argument("--release", action="store_true", help="보류 해제 (재승인은 따로 한다)")
    hold.add_argument("--actor", help="보류·해제한 사람 (기본: DLP_ACTOR 또는 OS 사용자)")
    hold.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    hold.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>' (렌더 기록 무효)")
    hold.set_defaults(func=cmd_hold)
