"""청소 전후 다각도 사진의 3D 복원·물체 이동 비교 하위 명령 (ADR 0066).

순서: ingest(before, after) → privacy detect → privacy review → privacy approve → privacy render
→ reconstruct(before, after) → compare → report. 단계마다 멱등이다 (같은 버전 태그가 있으면
건너뛴다).

공통 인자: `--store`는 `s3`(기본) 또는 `local:<디렉터리>`, `--url`은 DB URL, `--no-db`는 DB 없이
(감사 기록은 로컬 JSON Lines, `local:`만) 돈다. DB가 있으면 산출물마다 `scene3d_records`에 한 줄을
남긴다. `--oracle <truth.json>`은 합성 픽스처(`dlp scene3d fixture`)의 정답 stub을 끼운다 (로컬
시연·시험용, 정책으로는 고를 수 없다).

`dlp scene3d fixture --out DIR [--seed N]`
    정답을 아는 합성 방의 전후 사진(DIR/before, DIR/after)과 DIR/truth.json을 쓴다.
`dlp scene3d ingest <세션> <before|after> <사진 디렉터리>`
    원본 사진을 원본 버킷에 불변으로 올린다 (감사 저장소 `scene3d.ingest`).
`dlp scene3d privacy detect <세션> <단계>`
    `privacy.yaml` 탐지기로 사진마다 블러 상자를 찾는다 (`scene3d.privacy-detect`).
    새 세대면 렌더가 무효가 된다.
`dlp scene3d privacy review <세션> <단계> --reviewer ID --out DIR`
    원본 접근 권한자(`review.yaml reviewers.privacy`)에게 원본 사진과 boxes.json을 준다 (감사 기록의
    행위자는 검수자).
`dlp scene3d privacy approve <세션> <단계> --reviewer ID --boxes FILE`
    검수자가 고친 boxes.json(모든 사진)으로 승인한다.
`dlp scene3d privacy render <세션> <단계>`
    승인된 상자로 블러 PNG를 라벨링 버킷에 쓴다 (`scene3d.privacy-render`).
`dlp scene3d reconstruct <세션> <단계>`
    블러 사진으로 SfM 카메라·점과 스플랫 PLY를 만든다 (라벨링 버킷만).
`dlp scene3d compare <세션>`
    정렬·물체 3D 위치·전후 변화. 정렬 확신이 낮으면 종료 코드 3으로 거부한다.
`dlp scene3d report <세션> [--format json|csv]`
    지금 비교 결과를 출력한다 (위치·이동 벡터는 before 장면 좌표, 미터).

종료 코드: 0 성공, 3 게이트·정렬 거부(승인 없음, 낡은 입력, 정렬 확신 부족 등), 4 실제 모델·격리
환경을 쓸 수 없음. 인자 오류는 argparse가 2로 끝낸다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store, raw_store_offline
from dlp_cli.review_config import load_review_policy
from dlp_cli.schema_cmds import database_url
from dlp_fixtures.scene3d import generate_scene_pair, write_scene_pair
from dlp_media.audit import AuditedStore, current_actor
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_models.registry import resolve
from dlp_privacy.detectors import build_detectors
from dlp_privacy.errors import PrivacyGateError as VideoGateError
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_review.roles import AccessError, check_privacy_reviewers
from dlp_scene3d.backends import (
    GsplatTrainer,
    InitPointsSplat,
    IsolatedSfm,
    ObjectDetector,
    OwlObjectDetector,
    SfmBackend,
    SplatTrainer,
)
from dlp_scene3d.oracle import load_oracle, oracle_objects
from dlp_scene3d.photos import CaptureError, capture_key, ingest_capture
from dlp_scene3d.pipeline import (
    CSV_COLUMNS,
    Scene3dError,
    compare,
    current_report,
    reconstruct,
    report_rows,
)
from dlp_scene3d.policy import Scene3dPolicy, load_policy
from dlp_scene3d.privacy_gate import (
    FramePhotoDetector,
    PhotoPrivacyDetector,
    PrivacyGateError,
    approval_key,
    approve_capture,
    detect_capture,
    detection_key,
    render_capture,
    render_key,
    write_review_pack,
)
from dlp_schema import load_config, repo_root
from dlp_schema.db.scene3d_repository import record_scene3d
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.scene3d import PhotoBox, Scene3dRecord, Scene3dRecordKind, ScenePhase

REFUSED_EXIT = 3
UNAVAILABLE_EXIT = 4
GSPLAT_ENV = Path("scripts") / "gsplat-env"


@dataclass
class _Ctx:
    """명령 하나의 저장소·DB·정책."""

    args: argparse.Namespace
    root: Path
    policy: Scene3dPolicy
    labeling: ObjectStore
    engine: sa.Engine | None

    def raw(self, purpose: str) -> AuditedStore:
        """감사 원본 저장소 (DB 없으면 로컬 JSON Lines 기록)."""
        if self.engine is None:
            return raw_store_offline(self.args.store, purpose)
        return raw_store(self.args.store, self.args.url, purpose)

    def record(
        self,
        kind: Scene3dRecordKind,
        phase: ScenePhase | None,
        version: str | None,
        tag: str,
        bucket: str,
        key: str,
        actor: str | None = None,
    ) -> None:
        """DB가 있으면 산출물 기록을 남긴다."""
        if self.engine is None:
            return
        rec = Scene3dRecord(
            session_id=self.args.session_id,
            phase=phase,
            kind=kind,
            model_version=version,
            tag=tag,
            bucket="raw" if bucket == "raw" else "labeling",
            storage_key=key,
            actor=actor or current_actor(),
            created_at=datetime.now(UTC),
        )
        with self.engine.begin() as conn:
            record_scene3d(conn, rec)


def _ctx(args: argparse.Namespace) -> _Ctx:
    root = repo_root()
    buckets = load_config(root / "config" / "defaults.yaml").buckets
    engine = None if args.no_db else sa.create_engine(database_url(args.url))
    return _Ctx(
        args, root, load_policy(root), store_from_spec(args.store, buckets.labeling), engine
    )


def _refused(exc: Exception) -> int:
    print(f"거부: {exc}", file=sys.stderr)
    return REFUSED_EXIT


def cmd_fixture(args: argparse.Namespace) -> int:
    """합성 전후 사진과 정답을 쓴다."""
    path = write_scene_pair(generate_scene_pair(seed=args.seed), Path(args.out))
    print(f"정답: {path}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """원본 사진 수집."""
    c = _ctx(args)
    try:
        cap, n = ingest_capture(
            c.raw("scene3d.ingest"),
            c.labeling,
            args.session_id,
            args.phase,
            Path(args.photo_dir),
            c.policy.capture,
            datetime.now(UTC),
        )
    except CaptureError as exc:
        return _refused(exc)
    key = capture_key(args.session_id, args.phase)
    c.record("capture", args.phase, None, cap.photos[0].photo_id, "raw", key)
    print(f"{args.session_id} {args.phase}: 사진 {len(cap.photos)}장 (새로 올림 {n})")
    return 0


def _privacy_detectors(c: _Ctx) -> tuple[dict[str, PhotoPrivacyDetector], list[str]]:
    """(사진 탐지기, 탐지기가 없는 대상). --oracle이면 정답 stub만."""
    policy = load_privacy_policy(c.root)
    if c.args.oracle:
        return {"oracle": load_oracle(Path(c.args.oracle), c.args.phase).privacy}, []
    ready, missing = build_detectors(policy, c.root)
    for name, why in sorted(missing.items()):
        print(f"탐지기 없음 {name}: {why}", file=sys.stderr)
    no_detector: list[str] = [
        t for t, tp in policy.targets.items() if not any(d in ready for d in tp.detectors)
    ]
    wrapped: dict[str, PhotoPrivacyDetector] = {n: FramePhotoDetector(d) for n, d in ready.items()}
    return wrapped, no_detector


def cmd_privacy_detect(args: argparse.Namespace) -> int:
    """사진 블러 탐지."""
    c = _ctx(args)
    detectors, no_detector = _privacy_detectors(c)
    with tempfile.TemporaryDirectory() as tmp:
        det, new = detect_capture(
            c.raw("scene3d.privacy-detect"),
            c.labeling,
            args.session_id,
            args.phase,
            detectors,
            no_detector,
            load_privacy_policy(c.root),
            Path(tmp),
            datetime.now(UTC),
        )
    c.record(
        "privacy_detection",
        args.phase,
        det.model_version,
        det.model_version,
        "raw",
        detection_key(args.session_id, args.phase, det.model_version),
    )
    n = sum(len(b) for b in det.boxes.values())
    print(
        f"{'새 탐지' if new else '같은 세대(건너뜀)'}: 상자 {n}개, "
        f"탐지기 없는 대상 {list(det.no_detector)}"
    )
    print("다음: dlp scene3d privacy review (원본 접근 권한자가 모든 사진을 본다)")
    return 0


def _check_reviewer(c: _Ctx, reviewer: str) -> None:
    check_privacy_reviewers([reviewer], load_review_policy(c.root).reviewers.privacy)


def cmd_privacy_review(args: argparse.Namespace) -> int:
    """검수 묶음 (원본 접근 권한자만)."""
    c = _ctx(args)
    try:
        _check_reviewer(c, args.reviewer)
        raw = c.raw("scene3d.privacy-review")
        raw.actor = args.reviewer
        path = write_review_pack(raw, args.session_id, args.phase, Path(args.out))
    except (AccessError, PrivacyGateError) as exc:
        return _refused(exc)
    print(f"검수 묶음: {path} (상자를 고친 뒤 privacy approve --boxes)")
    return 0


def cmd_privacy_approve(args: argparse.Namespace) -> int:
    """검수자 승인."""
    c = _ctx(args)
    data = cast(dict[str, Any], json.loads(Path(args.boxes).read_text("utf-8")))
    boxes = {
        k: [PhotoBox.model_validate(b) for b in v]
        for k, v in cast(dict[str, list[Any]], data["boxes"]).items()
    }
    try:
        _check_reviewer(c, args.reviewer)
        raw = c.raw("scene3d.privacy-approve")
        approval = approve_capture(
            raw,
            args.session_id,
            args.phase,
            args.reviewer,
            boxes,
            str(data["detection_version"]),
            datetime.now(UTC),
        )
    except (AccessError, PrivacyGateError) as exc:
        return _refused(exc)
    c.record(
        "privacy_approval",
        args.phase,
        approval.detection_version,
        approval.approval_id,
        "raw",
        approval_key(args.session_id, args.phase, approval.approval_id),
        args.reviewer,
    )
    print(f"승인 {approval.approval_id}: 사진 {len(approval.boxes)}장")
    return 0


def cmd_privacy_render(args: argparse.Namespace) -> int:
    """블러 렌더."""
    c = _ctx(args)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rec, new = render_capture(
                c.raw("scene3d.privacy-render"),
                c.labeling,
                args.session_id,
                args.phase,
                load_privacy_policy(c.root),
                Path(tmp),
                datetime.now(UTC),
            )
    except PrivacyGateError as exc:
        return _refused(exc)
    c.record(
        "render",
        args.phase,
        rec.approval_id,
        str(rec.render_tag),
        "labeling",
        render_key(args.session_id, args.phase),
    )
    print(f"{'렌더' if new else '같은 렌더(건너뜀)'} {rec.render_tag}: 사진 {len(rec.files)}장")
    return 0


def _backends(c: _Ctx, phase: ScenePhase) -> tuple[SfmBackend, SplatTrainer]:
    rp = c.policy.reconstruction
    if c.args.oracle:
        return load_oracle(Path(c.args.oracle), phase).sfm, InitPointsSplat(rp.init_scale)
    if rp.sfm != "pycolmap":
        raise ModelUnavailableError("정책 reconstruction.sfm이 unavailable입니다")
    sfm = IsolatedSfm(c.root / GSPLAT_ENV)
    if rp.splat == "init_points":
        return sfm, InitPointsSplat(rp.init_scale)
    return sfm, GsplatTrainer(c.root / GSPLAT_ENV, rp.splat_iterations)


def cmd_reconstruct(args: argparse.Namespace) -> int:
    """복원."""
    c = _ctx(args)
    try:
        sfm, splat = _backends(c, args.phase)
        with tempfile.TemporaryDirectory() as tmp:
            rec, new = reconstruct(
                c.labeling,
                args.session_id,
                args.phase,
                sfm,
                splat,
                c.policy,
                Path(tmp),
                datetime.now(UTC),
            )
    except ModelUnavailableError as exc:
        print(f"모델 없음: {exc}", file=sys.stderr)
        return UNAVAILABLE_EXIT
    except (Scene3dError, VideoGateError) as exc:
        return _refused(exc)
    c.record(
        "reconstruction",
        args.phase,
        rec.model_version,
        rec.splat_key.split("/")[-2],
        "labeling",
        rec.splat_key,
    )
    print(
        f"{'복원' if new else '같은 복원(건너뜀)'}: 사진 {rec.registered}/{rec.photo_count}, "
        f"점 {rec.point_count}, metric={rec.metric}"
    )
    print(f"스플랫: {rec.splat_key}")
    return 0


def _detector(c: _Ctx) -> ObjectDetector:
    op = c.policy.objects
    if c.args.oracle:
        return oracle_objects(Path(c.args.oracle))
    if op.detector != "owlv2":
        raise ModelUnavailableError("정책 objects.detector가 unavailable입니다")
    model, version = resolve(c.root, op.model)
    tokenizer, _ = resolve(c.root, op.tokenizer)
    return OwlObjectDetector(model, tokenizer, version, op.queries, op.score_threshold, op.nms_iou)


def cmd_compare(args: argparse.Namespace) -> int:
    """정렬·물체·변화."""
    c = _ctx(args)
    try:
        detector = _detector(c)
        with tempfile.TemporaryDirectory() as tmp:
            report, new = compare(
                c.labeling, args.session_id, detector, c.policy, Path(tmp), datetime.now(UTC)
            )
    except ModelUnavailableError as exc:
        print(f"모델 없음: {exc}", file=sys.stderr)
        return UNAVAILABLE_EXIT
    except (Scene3dError, VideoGateError) as exc:
        return _refused(exc)
    c.record(
        "change_report",
        None,
        report.model_version,
        report.model_version,
        "labeling",
        f"sessions/{args.session_id}/scene3d/changes/current.json",
    )
    al = report.alignment
    print(
        f"{'비교' if new else '같은 비교(건너뜀)'}: "
        f"정렬 내점 {al.inliers}/{al.matches}, RMS {al.rms_m}"
    )
    for ch in report.changes:
        d = "" if ch.distance_m is None else f" {ch.distance_m:.3f} m"
        print(f"  {ch.kind:9s} {ch.label}{d} (신뢰도 {ch.confidence:.2f})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """지금 비교 결과 출력."""
    c = _ctx(args)
    try:
        report = current_report(c.labeling, args.session_id)
    except Scene3dError as exc:
        return _refused(exc)
    if args.format == "json":
        print(report.model_dump_json(indent=1))
        return 0
    w = csv.DictWriter(sys.stdout, fieldnames=list(CSV_COLUMNS))
    w.writeheader()
    w.writerows(report_rows(report))
    return 0


def _common(p: argparse.ArgumentParser, phase: bool = True) -> None:
    p.add_argument("session_id")
    if phase:
        p.add_argument("phase", choices=["before", "after"])
    p.add_argument("--store", default="s3", help="s3 또는 local:<디렉터리>")
    p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    p.add_argument("--no-db", action="store_true", help="DB 없이 (local: 저장소만)")
    p.add_argument("--oracle", help="합성 픽스처 truth.json (stub, 시연·시험용)")


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`scene3d …` 하위 명령을 등록한다."""
    top = sub.add_parser("scene3d", help="청소 전후 사진 3D 복원·물체 이동 비교 (ADR 0066)")
    s = top.add_subparsers(dest="scene3d_command", required=True)
    fx = s.add_parser("fixture", help="합성 전후 사진과 정답")
    fx.add_argument("--out", required=True)
    fx.add_argument("--seed", type=int, default=0)
    fx.set_defaults(func=cmd_fixture)
    ing = s.add_parser("ingest", help="원본 사진 수집 (불변, 감사)")
    _common(ing)
    ing.add_argument("photo_dir")
    ing.set_defaults(func=cmd_ingest)
    pv = s.add_parser("privacy", help="사진 블러 게이트").add_subparsers(
        dest="privacy_command", required=True
    )
    for name, func, help_ in (
        ("detect", cmd_privacy_detect, "블러 상자 탐지"),
        ("review", cmd_privacy_review, "원본 접근 권한자 검수 묶음"),
        ("approve", cmd_privacy_approve, "검수자 승인 (모든 사진)"),
        ("render", cmd_privacy_render, "승인된 상자로 블러 PNG"),
    ):
        p = pv.add_parser(name, help=help_)
        _common(p)
        if name in ("review", "approve"):
            p.add_argument("--reviewer", required=True)
        if name == "review":
            p.add_argument("--out", required=True)
        if name == "approve":
            p.add_argument("--boxes", required=True)
        p.set_defaults(func=func)
    rc = s.add_parser("reconstruct", help="SfM 카메라·점과 가우시안 스플랫")
    _common(rc)
    rc.set_defaults(func=cmd_reconstruct)
    cp = s.add_parser("compare", help="정렬·물체 3D 위치·전후 변화")
    _common(cp, phase=False)
    cp.set_defaults(func=cmd_compare)
    rp = s.add_parser("report", help="지금 비교 결과 (JSON·CSV)")
    _common(rp, phase=False)
    rp.add_argument("--format", choices=["json", "csv"], default="json")
    rp.set_defaults(func=cmd_report)
