"""AI 1차 검수 하위 명령 (WP17, ADR 0040). 이번 단계(WP17.3): 서버 점검·처리량 벤치·작업량 추정.

판정 실행(`run`·`status`·`show`·`waive`)은 WP17.5, 골든 평가(`eval`)는 WP17.8에서 더한다. 세 명령
모두 DB에 쓰지 않는다. 절차와 기록 양식은 `docs/handover/vlm-server.md`의 "처리량 재기와 대수
정하기"다.

등록하는 명령
- `dlp aireview probe [--vlm-url URL …] [--ontology-version V] [--out FILE]`: 서버 점검
  (`dlp_aireview.probe`). 엔드포인트마다 점검 결과·권고를 출력한다.
  종료: 0 통과 / 2 주소 없음 / 3 서버 장애 / 5 설정 불일치(사내 주소 아님, 모델 없음·불일치, 키,
  구조화 출력 미강제, 이미지 수 한도 등). 여러 엔드포인트면 가장 심각한 코드(5 > 3).
- `dlp aireview bench [--vlm-url URL] [--endpoint NAME] [--source synthetic] [--concurrency 1,8,…]
  [--frames N,…] [--side PX,…] [--groups …] [--requests 200] [--warmup 10] [--ttft 10]
  [--cache-probe] [--no-capacity] [--decode-tokens 256] [--decode] [--decode-seconds 20]
  [--scene-units 40] [--frame-size WxH] [--seed 0] [--device NAME] [--out FILE]`: 처리량 벤치
  (`dlp_aireview.bench`). 입력은 합성 장면(`dlp_fixtures.aireview`, 영상·사람·글자 없음)이다.
  `--frames`·`--side`는 지금 정책 값에 더할 후보다 (정책 값은 늘 잰다, 주지 않으면 정책 값만).
  무릎 동시성에서 프리필·생성 용량도 따로 잰다 (`--no-capacity`로 끈다). 보고서는
  `reports/aireview-bench/`(Git 제외). `--decode`는 같은 합성 장면을 MP4로 써서 CPU 순차 디코드
  fps를 잰다.
  종료: 0 / 2 주소 없음 / 3 서버 장애(서버가 멈춰 벤치를 멈춤 포함) / 5 설정 불일치.
- `dlp aireview estimate [<세션>… | --prelabeled] [--mode all|flagged] [--bench REPORT …]
  [--hours-per-day H] [--rerun-margin R] [--utilization U] [--decode-workers N] [--include-actions]
  [--actions-req-s X] [--json] [--url]`: 작업량·장비 대수 추정 (`dlp_aireview.estimate`). DB를
  읽기만 한다 (서버 불필요). 가정 값의 기본은 `aireview.yaml estimate`. flagged 모드의 우선순위
  사유는 검수 운영 정책(`DLP_REVIEW_POLICY_FILE` 반영)의 `flag_unit`으로 계산한다. 벤치 보고서의
  셈법 값(align_px·text_bytes_per_token)이 지금 정책과 다르면 그 보고서용 작업량을 보고서 셈법으로
  다시 센다 (`estimate.counting_policy`). 모델·fingerprint가 다르면 경고한다.
  종료: 0 (세션이 없어도 0).

서버 주소: `--vlm-url`(여러 번, `이름=URL` 가능) 또는 `DLP_VLM_URLS`(`spark-01=http://…,…`).
`DLP_VLM_URL`(행동 분류 서버)은 쓰지 않는다. API 키는 `DLP_VLM_API_KEY` (Authorization 헤더로만
보내고 출력·기록하지 않는다). 주소는 사내 대역(`aireview.yaml server.internal_hosts`)이어야
한다 (I7).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_actions.runner import vlm_request_count
from dlp_aireview.backend import OpenAICompatibleJudge, check_hosts, parse_endpoints
from dlp_aireview.bench import (
    BenchConfig,
    BenchGrid,
    BenchInput,
    BenchReport,
    DecodeMeasure,
    run_bench,
)
from dlp_aireview.estimate import (
    Capacity,
    Workload,
    capacity,
    counting_policy,
    session_load,
    to_json,
)
from dlp_aireview.plan import VIDEO_KINDS, StreamMedia
from dlp_aireview.policy import AiReviewPolicy, load_policy
from dlp_aireview.probe import ProbeReport, probe_server, server_info
from dlp_aireview.sensors import SensorContext
from dlp_cli.review_config import load_review_policy
from dlp_cli.schema_cmds import database_url
from dlp_fixtures.aireview import AiReviewScene, aireview_scene, scene_frame
from dlp_fixtures.video import write_video
from dlp_media.frames import decode_frames
from dlp_media.pts import PtsIndex, build_pts_index
from dlp_models.vlm import VlmRequestError, VlmUnavailableError
from dlp_review.ops.priority import flag_unit, units_for
from dlp_review.ops.runner import known_classes
from dlp_schema import repo_root
from dlp_schema.aireview import AiGroup
from dlp_schema.db.ontology_repository import get_session_ontology
from dlp_schema.db.repository import get_labels, get_session, list_session_ids
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology, find_ontology_dir, load_ontology
from dlp_schema.review import FlaggedSpan
from dlp_schema.session import LifecycleState, Session, StreamKind

# 합성 장면의 온톨로지 버전 (dlp_fixtures 라벨이 쓰는 값)
_SYNTHETIC_ONTOLOGY = "1.0.0"
_NO_URL = 2
_UNAVAILABLE = 3
_CONFIG = 5


# ---------------------------------------------------------------- 공용


def _endpoints(args: argparse.Namespace) -> dict[str, str] | None:
    """`--vlm-url` 또는 `DLP_VLM_URLS` → {이름: URL} (없으면 None).

    Raises:
        ValueError: 잘못된 지정.
    """
    specs: list[str] = list(args.vlm_url or [])
    if not specs and os.environ.get("DLP_VLM_URLS"):
        specs = [os.environ["DLP_VLM_URLS"]]
    return parse_endpoints(specs) if specs else None


def _api_key() -> str | None:
    """서버 API 키 (출력·기록하지 않는다)."""
    return os.environ.get("DLP_VLM_API_KEY") or None


def _no_url() -> int:
    """주소가 없을 때 안내 (종료 2)."""
    print(
        "[미연동] AI 검수 VLM 서버 주소가 없습니다 (--vlm-url 또는 DLP_VLM_URLS). "
        "DLP_VLM_URL(행동 분류 서버)은 쓰지 않습니다."
    )
    return _NO_URL


def _ints(text: str) -> tuple[int, ...]:
    """`1,8,16` → (1, 8, 16) (argparse 형식 함수)."""
    try:
        values = tuple(int(x) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"쉼표로 나눈 정수가 아닙니다: {text}") from exc
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError(f"1 이상의 정수가 필요합니다: {text}")
    return values


def _groups(text: str) -> tuple[AiGroup, ...]:
    """`temporal,spatial` → 묶음 (argparse 형식 함수)."""
    try:
        return tuple(AiGroup(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"모르는 묶음: {text}") from exc


def _size(text: str) -> tuple[int, int]:
    """`1920x1080` → (1920, 1080) (짝수 크기, argparse 형식 함수)."""
    try:
        w, h = (int(x) for x in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"WxH 형식이 아닙니다: {text}") from exc
    if w < 32 or h < 32 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError(f"32 이상의 짝수 크기가 필요합니다: {text}")
    return w, h


def _write_json(path: Path, data: Any) -> None:
    """보고서 JSON 쓰기 (상위 디렉터리를 만든다)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stamp() -> str:
    """보고서 파일 이름의 UTC 시각."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------- probe


def _print_probe(report: ProbeReport) -> None:
    """점검 결과 출력."""
    s = report.server
    print(f"[probe] {s.endpoint} ({report.host}) 서버 모델 {s.served_model}")
    for c in report.checks:
        print(f"  [{c.status:<4}] {c.name}: {c.detail}")
    for r in report.recommendations:
        print(f"  권고: {r}")


def cmd_probe(args: argparse.Namespace) -> int:
    """`dlp aireview probe`: 엔드포인트마다 서버 점검 (모듈 docstring의 종료 코드).

    부작용: VLM 서버에 점검 요청 (사내 주소일 때만), `--out`이면 보고서 JSON 파일.
    """
    root = repo_root()
    policy = load_policy(root)
    try:
        endpoints = _endpoints(args)
    except ValueError as exc:
        print(f"[설정 오류] {exc}")
        return _CONFIG
    if not endpoints:
        return _no_url()
    ontology = load_ontology(find_ontology_dir(root / "config/ontology", args.ontology_version))
    reports = [
        probe_server(name, url, policy, ontology, api_key=_api_key())
        for name, url in endpoints.items()
    ]
    for report in reports:
        _print_probe(report)
    if args.out:
        _write_json(Path(args.out), [r.model_dump(mode="json") for r in reports])
    if any(r.failed for r in reports):  # 설정 불일치가 하나라도 있으면 서버 장애보다 먼저 (5 > 3)
        return _CONFIG
    return _UNAVAILABLE if any(r.unavailable for r in reports) else 0


# ---------------------------------------------------------------- bench


def _scene_input(scene: AiReviewScene, ontology: Ontology) -> BenchInput:
    """합성 장면 → 벤치 입력 (영상 스트림마다 같은 프레임 시각, 프레임은 그때그때 그린다)."""
    times = np.asarray(scene.frame_times, dtype=np.int64)
    pts = PtsIndex(Fraction(1, 1000), times, np.ones(len(times), dtype=bool))
    media = {
        s.stream_id: StreamMedia(pts, "synthetic")
        for s in scene.session.streams
        if s.kind in VIDEO_KINDS
    }

    def frames(stream: str, idx: Sequence[int]) -> list[Any]:
        return [scene_frame(scene, stream, scene.frame_times[i]) for i in idx]

    return BenchInput(
        scene.session, scene.labels, media, ontology, SensorContext.absent(), frames, "synthetic"
    )


def _measure_decode(scene: AiReviewScene, policy: AiReviewPolicy, seconds: float) -> DecodeMeasure:
    """합성 장면 앞 `seconds`초를 MP4로 써서 CPU 순차 디코드 fps를 잰다 (임시 디렉터리)."""
    times = [t for t in scene.frame_times if t < seconds * 1000] or [scene.frame_times[0]]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decode.mp4"
        write_video(
            path,
            ((t, scene_frame(scene, "bodycam", t)) for t in times),
            width=scene.width,
            height=scene.height,
        )
        pts = build_pts_index(path)
        start = time.monotonic()
        count = sum(
            1
            for _ in decode_frames(
                path,
                pts,
                range(len(pts)),
                max_side=policy.temporal.max_side_px,
                align=policy.image.align_px,
            )
        )
        elapsed = max(time.monotonic() - start, 1e-9)
    return DecodeMeasure(
        fps=count / elapsed,
        frames=count,
        width=scene.width,
        height=scene.height,
        max_side=policy.temporal.max_side_px,
    )


def _capacity_text(c: BenchConfig) -> str:
    """따로 잰 프리필·생성 용량 한 줄 조각 (없으면 빈 문자열)."""
    pre, dec = c.prefill, c.decode
    if pre is None or dec is None:
        return ""
    fresh = pre.fresh_tokens / pre.wall_s if pre.wall_s > 0 else 0.0
    if dec.error:
        state = f" (생성 측정 실패: {dec.error})"
    else:
        state = "" if dec.honored else " (생성 길이 고정 안 됨, 추정에 쓰지 않음)"
    return f"; 용량 프리필 근사 {fresh:.0f} tok/s, 생성 {dec.decode_tok_s:.0f} tok/s{state}"


def _print_bench(report: BenchReport, out: Path) -> None:
    """벤치 요약 출력 (설정마다 무릎 지점과 따로 잰 용량)."""
    print(f"[bench] {report.device} 서버 모델 {report.server.served_model} 엔진 "
          f"{report.server.engine_version or '알 수 없음'}")  # fmt: skip
    for c in report.configs:
        head = f"  {c.group.value} 프레임 {c.frames} 긴 변 {c.side}: 단위 {c.planned_units}"
        k = c.knee
        if k is None:
            print(f"{head} — 무릎 지점 없음{f' ({c.stopped})' if c.stopped else ''}")
            continue
        cache = f"{k.cache_hit_ratio:.0%}" if k.cache_hit_ratio is not None else "모름"
        print(
            f"{head} — 동시성 {k.concurrency}: {k.units_per_s:.2f} 단위/s, "
            f"p95 {k.latency_p95_ms} ms, 관측 프리필 {k.prefill_tok_s:.0f} tok/s, "
            f"관측 생성 {k.decode_tok_s:.0f} tok/s, 캐시 {cache}, 첫 토큰 {c.ttft_p50_ms} ms"
            f"{_capacity_text(c)}"
        )
    if report.decode is not None:
        print(f"  디코드(CPU): {report.decode.fps:.0f} fps ({report.decode.width}x"
              f"{report.decode.height} → 긴 변 {report.decode.max_side})")  # fmt: skip
    print(f"보고서: {out}")


def _grid(args: argparse.Namespace) -> BenchGrid:
    """CLI 인자 → 설정 격자."""
    return BenchGrid(
        concurrency=args.concurrency,
        frames=args.frames or (),
        sides=args.side or (),
        groups=args.groups,
        requests=args.requests,
        warmup=args.warmup,
        ttft=args.ttft,
        cache_probe=args.cache_probe,
        capacity=not args.no_capacity,
        decode_tokens=args.decode_tokens,
    )


def _pick_endpoint(args: argparse.Namespace) -> tuple[str, str] | int:
    """벤치할 엔드포인트 하나 (이름, URL) 또는 종료 코드."""
    try:
        endpoints = _endpoints(args)
    except ValueError as exc:
        print(f"[설정 오류] {exc}")
        return _CONFIG
    if not endpoints:
        return _no_url()
    if args.endpoint:
        if args.endpoint not in endpoints:
            print(f"[설정 오류] 엔드포인트 {args.endpoint}가 없습니다 ({', '.join(endpoints)})")
            return _CONFIG
        return args.endpoint, endpoints[args.endpoint]
    if len(endpoints) > 1:
        print(
            f"[설정 오류] 벤치는 장비 하나만 잽니다. --endpoint로 고르세요 ({', '.join(endpoints)})"
        )
        return _CONFIG
    return next(iter(endpoints.items()))


def cmd_bench(args: argparse.Namespace) -> int:
    """`dlp aireview bench`: 장비 1대의 처리량 벤치 (모듈 docstring의 종료 코드).

    부작용: VLM 서버에 판정 요청 (합성 입력), 보고서 JSON 파일, `--decode`면 임시 MP4.
    """
    root = repo_root()
    policy = load_policy(root)
    picked = _pick_endpoint(args)
    if isinstance(picked, int):
        return picked
    name, url = picked
    outside = check_hosts({name: url}, policy.server.internal_hosts, resolve=socket.getaddrinfo)
    if outside:
        print(f"[설정 오류] 사내 대역 밖 주소에는 보내지 않습니다: {'; '.join(outside)}")
        return _CONFIG
    width, height = args.frame_size or policy.estimate.frame_px
    scene = aireview_scene(
        args.seed, session_id="bench", n_units=args.scene_units, width=width, height=height
    )
    ontology = load_ontology(find_ontology_dir(root / "config/ontology", _SYNTHETIC_ONTOLOGY))
    judge = OpenAICompatibleJudge(name, url, policy, api_key=_api_key())
    try:
        info = server_info(name, judge.chat, policy.served_model)
        if policy.served_model not in info.models:
            print(f"[설정 오류] 서버에 {policy.served_model}이 없습니다 ({', '.join(info.models)})")
            return _CONFIG
        report = run_bench(
            judge, _scene_input(scene, ontology), policy, _grid(args), server=info,
            device=args.device or name, progress=print,
        )  # fmt: skip
    except VlmUnavailableError as exc:
        print(f"[VLM 서버 장애] {name}: {exc}")
        return _UNAVAILABLE
    except VlmRequestError as exc:
        print(f"[VLM 설정 오류] {name}: 서버 주소·모델·키를 확인하세요 ({exc})")
        return _CONFIG
    if args.decode:
        report = report.model_copy(
            update={"decode": _measure_decode(scene, policy, args.decode_seconds)}
        )
    out = Path(args.out) if args.out else (
        root / "reports" / "aireview-bench" / f"{report.device}-{_stamp()}.json"
    )  # fmt: skip
    _write_json(out, report.model_dump(mode="json"))
    _print_bench(report, out)
    return _UNAVAILABLE if report.server_down else 0


# ---------------------------------------------------------------- estimate


def _flagged(
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


def _session_ids(conn: sa.Connection, args: argparse.Namespace) -> list[str]:
    """추정할 세션 (지정 또는 prelabeled 전부, 사용 중지 제외)."""
    if args.sessions:
        return list(args.sessions)
    if not args.prelabeled:
        return []
    return [
        sid
        for sid in list_session_ids(conn)
        if get_session(conn, sid).lifecycle_state is LifecycleState.PRELABELED
    ]


@dataclass(frozen=True)
class _SessionInput:
    """작업량을 셀 세션 하나의 DB 입력 (셈법 값이 다른 보고서마다 다시 센다)."""

    session: Session
    current: list[LabelRecord]
    ontology: Ontology
    flagged: list[FlaggedSpan]
    actions_requests: int | None


def _session_inputs(args: argparse.Namespace, policy: AiReviewPolicy) -> list[_SessionInput]:
    """DB에서 세션·현재 운영 라벨·온톨로지·사유 구간·행동 요청 수를 읽는다 (읽기 전용 연결)."""
    actions = load_actions_policy(repo_root()) if args.include_actions else None
    engine = sa.create_engine(database_url(args.url))
    inputs: list[_SessionInput] = []
    try:
        with engine.connect() as conn:
            for sid in _session_ids(conn, args):
                session = get_session(conn, sid)
                if session.lifecycle_state is LifecycleState.WITHDRAWN:
                    print(f"{sid}: 사용 중지된 세션이라 뺍니다")
                    continue
                current = current_labels(get_labels(conn, sid))
                flagged = (
                    _flagged(conn, session, current) if policy.selection.mode == "flagged" else []
                )
                requests = (
                    sum(vlm_request_count(session, current, actions).values()) if actions else None
                )
                o = get_session_ontology(conn, sid)
                inputs.append(_SessionInput(session, current, o, flagged, requests))
    finally:
        engine.dispose()
    return inputs


def _workload(inputs: Sequence[_SessionInput], policy: AiReviewPolicy) -> Workload:
    """세션 입력 → 작업량 (`policy`의 셈법으로 센다)."""
    return Workload(
        tuple(
            session_load(
                x.session,
                x.current,
                policy,
                x.ontology,
                flagged=x.flagged,
                actions_requests=x.actions_requests,
            )
            for x in inputs
        )
    )


def _print_workload(w: Workload) -> None:
    """작업량 표 출력."""
    print(f"세션 {len(w.sessions)}개 · 영상 {w.video_hours:.2f}시간")
    print("묶음            단위    대상   프레임  이미지tok    글tok   출력tok  영상1시간당 단위")
    for g in AiGroup:
        t = w.total(g)
        print(
            f"{g.value:<14}{t.units:>6}{t.targets:>8}{t.frames:>8}{t.image_tokens:>11}"
            f"{t.text_tokens:>9}{t.output_tokens:>10.0f}{w.per_hour(t.units):>15.1f}"
        )
    if w.actions_requests is not None:
        per_hour = w.per_hour(w.actions_requests)
        print(f"행동 VLM 요청 {w.actions_requests} (영상 1시간당 {per_hour:.1f})")


def _print_capacity(c: Capacity) -> None:
    """대수 출력."""
    p = c.params
    print(f"[대수] {c.label} (H {p['hours_per_day']:g}, 여유 {p['rerun_margin']:g}, "
          f"가동률 {p['utilization']:g})")  # fmt: skip
    for s in c.groups:
        t = f"{s.units_per_s:.2f}/s ({s.limit})" if s.units_per_s else "측정 없음"
        dev = f"{s.device_s_per_hour:.0f}" if s.device_s_per_hour is not None else "-"
        print(f"  {s.group}: 영상 1시간당 {s.units_per_hour:.1f} · 장비 1대 {t} · 장비초 {dev}")
    if c.missing:
        print(f"  대수 없음: 측정이 없는 몫 {', '.join(c.missing)}")
    else:
        print(f"  영상 1시간당 장비초 D {c.device_s_per_hour:.0f} → 필요 대수 {c.devices}")
    for w in c.warnings:
        print(f"  경고: {w}")
    if c.decode_hours_per_day is not None:
        ok = "충분" if c.decode_hours_per_day <= 24 else "부족 (디코드 작업 수를 늘린다)"
        print(f"  디코드 하루 {c.decode_hours_per_day:.1f}시간 ({ok})")


def cmd_estimate(args: argparse.Namespace) -> int:
    """`dlp aireview estimate`: 작업량·장비 대수 추정 (DB 읽기만, 서버 불필요).

    부작용: 없음 (DB 읽기, 표준 출력).
    """
    root = repo_root()
    policy = load_policy(root)
    if args.mode:
        data = policy.model_dump()
        data["selection"]["mode"] = args.mode
        policy = AiReviewPolicy.model_validate(data)
    est = policy.estimate
    inputs = _session_inputs(args, policy)
    workload = _workload(inputs, policy)
    capacities: list[Capacity] = []
    paths: list[str] = list(args.bench or [])
    for path in paths:
        report = BenchReport.model_validate_json(Path(path).read_text("utf-8"))
        counted = counting_policy(policy, report)  # 보고서와 같은 셈법으로 작업량을 센다
        capacities.append(
            capacity(
                workload if counted is policy else _workload(inputs, counted),
                report,
                counted,
                hours_per_day=args.hours_per_day or est.hours_per_day,
                rerun_margin=(
                    args.rerun_margin if args.rerun_margin is not None else est.rerun_margin
                ),
                utilization=args.utilization or est.utilization,
                decode_workers=args.decode_workers or est.decode_workers,
                actions_req_s=args.actions_req_s,
            )
        )
    if args.json:
        data = {"workload": to_json(workload), "capacity": [to_json(c) for c in capacities]}
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    if not workload.sessions:
        print("추정할 세션이 없습니다 (<세션>… 또는 --prelabeled)")
    _print_workload(workload)
    for c in capacities:
        _print_capacity(c)
    return 0


# ---------------------------------------------------------------- 등록


def _url_arg(p: argparse.ArgumentParser) -> None:
    """서버 주소 인자."""
    p.add_argument(
        "--vlm-url", action="append",
        help="AI 검수 VLM 서버 (여러 번, 이름=URL 가능. 기본: DLP_VLM_URLS)",
    )  # fmt: skip


def _bench_parser(asub: Any) -> None:
    """`aireview bench` 인자."""
    b = asub.add_parser("bench", help="장비 1대 처리량 벤치 (합성 입력, DB에 쓰지 않음)")
    _url_arg(b)
    b.add_argument("--endpoint", help="엔드포인트가 여럿이면 잴 이름")
    b.add_argument("--source", choices=["synthetic"], default="synthetic",
                   help="입력 (실제 세션 블러본 입력은 WP17.5 open_blurred 뒤)")  # fmt: skip
    b.add_argument("--concurrency", type=_ints, default=(1, 8, 16, 32, 64))
    b.add_argument("--frames", type=_ints, help="temporal 프레임 수 후보 (정책 값에 더한다)")
    b.add_argument("--side", type=_ints, help="긴 변 후보 px (묶음마다 정책 값에 더한다)")
    b.add_argument("--groups", type=_groups, default=tuple(AiGroup))
    b.add_argument("--requests", type=int, default=200, help="점마다 판정할 단위 수")
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--ttft", type=int, default=10, help="첫 토큰 시간 표본 수 (0이면 생략)")
    b.add_argument("--cache-probe", action="store_true", help="캐시 적중·깨기 비교")
    b.add_argument(
        "--no-capacity",
        action="store_true",
        help="무릎 동시성의 프리필·생성 용량 측정을 건너뛴다 (추정이 보정하지 못한다)",
    )
    b.add_argument("--decode-tokens", type=int, default=256,
                   help="생성 용량 측정의 요청당 생성 길이 (vLLM 확장)")  # fmt: skip
    b.add_argument("--decode", action="store_true", help="CPU 순차 디코드 fps도 잰다")
    b.add_argument("--decode-seconds", type=float, default=20.0)
    b.add_argument("--scene-units", type=int, default=40, help="합성 장면 행동 단위 수")
    b.add_argument("--frame-size", type=_size, help="합성 프레임 WxH (기본 estimate.frame_px)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", help="장비 이름 (기본: 엔드포인트 이름)")
    b.add_argument("--out", help="보고서 JSON (기본 reports/aireview-bench/<장비>-<시각>.json)")
    b.set_defaults(func=cmd_bench)


def _estimate_parser(asub: Any) -> None:
    """`aireview estimate` 인자."""
    e = asub.add_parser("estimate", help="작업량·장비 대수 추정 (DB 읽기만)")
    e.add_argument("sessions", nargs="*")
    e.add_argument("--prelabeled", action="store_true", help="prelabeled 세션 전부")
    e.add_argument("--mode", choices=["all", "flagged"], help="선택 모드 (기본: 정책)")
    e.add_argument("--bench", action="append", help="벤치 보고서 JSON (여러 번)")
    e.add_argument("--hours-per-day", type=float, help="하루 영상 시간 H (기본: 정책)")
    e.add_argument("--rerun-margin", type=float, help="재실행 여유 (기본: 정책)")
    e.add_argument("--utilization", type=float, help="가동률 (기본: 정책)")
    e.add_argument("--decode-workers", type=int, help="디코드 동시 작업 수 (기본: 정책)")
    e.add_argument("--include-actions", action="store_true", help="행동 VLM 요청 수도 센다")
    e.add_argument("--actions-req-s", type=float, help="행동 VLM 장비 1대 지속 요청/s (실측)")
    e.add_argument("--json", action="store_true")
    e.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    e.set_defaults(func=cmd_estimate)


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`aireview probe|bench|estimate` 하위 명령을 등록한다."""
    ai = sub.add_parser("aireview", help="AI 1차 검수 (로컬 VLM, 참고용)")
    asub = ai.add_subparsers(dest="aireview_command", required=True)
    p = asub.add_parser("probe", help="VLM 서버 점검 (모델·이미지 한도·구조화 출력·캐시)")
    _url_arg(p)
    p.add_argument("--ontology-version", default=_SYNTHETIC_ONTOLOGY)
    p.add_argument("--out", help="보고서 JSON 파일")
    p.set_defaults(func=cmd_probe)
    _bench_parser(asub)
    _estimate_parser(asub)
