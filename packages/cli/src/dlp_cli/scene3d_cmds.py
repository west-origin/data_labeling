"""청소 전후 다각도 사진의 3D 복원·물체 이동 비교와 장면 안 바디캠 위치 추정 하위 명령 (ADR 0066,
0067).

순서: ingest(before, after) → privacy detect → privacy review → privacy approve → privacy render
→ reconstruct(before, after) → compare → report. 바디캠 영상이 있는 세션은 두 복원 뒤에 localize.
어디까지 됐는지는 status. 단계마다 멱등이다 (같은 버전 태그가 있으면 건너뛴다).

공통 인자: `--store`는 `s3`(기본) 또는 `local:<디렉터리>`, `--url`은 DB URL, `--no-db`는 DB 없이
(감사 기록은 로컬 JSON Lines, `local:`만) 돈다. DB가 있으면 산출물마다 `scene3d_records`에 한 줄을
남긴다. `--oracle <truth.json>`은 합성 픽스처(`dlp scene3d fixture`)의 정답 stub을 끼운다 (로컬
시연·시험용, 정책으로는 고를 수 없다).

`dlp scene3d fixture --out DIR [--seed N] [--marker-side-m M] [--unit-scale S] [--ego]`
    정답을 아는 합성 방의 전후 사진(DIR/before, DIR/after)과 DIR/truth.json을 쓴다.
    `--marker-side-m`은 한 변이 M 미터인 척도 마커를 바닥에 그린다 (사전은 정책 값. 복원이 그 마커로
    척도를 맞추려면 정책 `reconstruction.marker.side_m`이 M이어야 한다). `--unit-scale`은 정답 stub
    SfM이 좌표에 S를 곱해 미터가 아닌 복원을 내게 한다. `--ego`는 방을 지나는 바디캠 영상
    (DIR/ego/bodycam.mp4)과 프레임별 정답 자세를 더한다.
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
    블러 사진으로 SfM 카메라·점과 스플랫 PLY를 만든다 (라벨링 버킷만). 크기를 아는 마커가 보이면
    미터 척도를 정하고, 아니면 metric=False로 두고 이유를 알린다 (ADR 0067).
`dlp scene3d compare <세션>`
    정렬·물체 3D 위치·전후 변화. 정렬 확신이 낮으면 종료 코드 3으로 거부한다.
`dlp scene3d report <세션> [--format json|csv]`
    지금 비교 결과를 출력한다 (위치·이동 벡터는 before 장면 좌표, 미터).
`dlp scene3d localize <세션>`
    그 세션 바디캠의 지금 승인된 블러본에서 프레임마다 카메라 자세를 before 장면 좌표로 구한다
    (ADR 0067). 전후 복원과 받아들여진 정렬, 등록된 바디캠 내부 파라미터(`dlp calib`)가 모두 있어야
    한다. 세션·블러 승인 상태·캘리브레이션을 DB에서 읽으므로 `--no-db`가 없다. 원본 버킷은 열지
    않는다.
`dlp scene3d status <세션>`
    before·after의 수집·블러 승인·렌더·복원(미터 여부, 척도 출처)과 정렬·비교·위치 추정이 지금
    있는지 표로 낸다. 읽기 전용이다 (DB를 쓰지 않고, 원본 버킷은 표시 파일이 있는지만 본다).

종료 코드: 0 성공, 1 status에서 전후 공간이 갖춰지지 않음(전후 복원 중 빠진 것이 있거나 정렬이
없거나 거부됨), 3 게이트·정렬 거부(승인 없음, 낡은 입력, 정렬 확신 부족, 빠진 전제 등), 4 실제
모델·격리 환경을 쓸 수 없음. 인자 오류는 argparse가 2로 끝낸다.
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
from dlp_fixtures.scene3d import (
    SceneMarker,
    floor_marker,
    generate_ego_scene,
    generate_scene_pair,
    write_scene_pair,
)
from dlp_media.audit import AuditedStore, current_actor
from dlp_media.storage import ObjectStore, store_from_spec
from dlp_models.registry import resolve
from dlp_privacy.detectors import build_detectors
from dlp_privacy.errors import PrivacyGateError as VideoGateError
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_review.roles import AccessError, check_privacy_reviewers
from dlp_scene3d.backends import (
    FrameFeatureExtractor,
    GsplatTrainer,
    InitPointsSplat,
    IsolatedFrameFeatures,
    IsolatedSfm,
    ObjectDetector,
    OwlObjectDetector,
    SfmBackend,
    SplatTrainer,
)
from dlp_scene3d.ego_video import VideoBoundaryError
from dlp_scene3d.layout import get_model, short_hash
from dlp_scene3d.localize import localization_key, localize, stored_localization
from dlp_scene3d.oracle import OracleError, load_oracle, oracle_features, oracle_objects
from dlp_scene3d.photos import CaptureError, capture_key, ingest_capture
from dlp_scene3d.pipeline import (
    CSV_COLUMNS,
    Scene3dError,
    alignment_key,
    alignment_tag,
    change_key,
    compare,
    current_report,
    reconstruct,
    report_rows,
    report_tag,
    stored_reconstruction,
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
from dlp_scene3d.status import format_status, scene_status
from dlp_schema import load_config, repo_root
from dlp_schema.db.scene3d_repository import record_scene3d
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.scene3d import (
    Alignment,
    PhotoBox,
    Reconstruction,
    Scene3dRecord,
    Scene3dRecordKind,
    ScenePhase,
)

# status: 전후 공간이 갖춰지지 않음
INCOMPLETE_EXIT = 1
REFUSED_EXIT = 3
UNAVAILABLE_EXIT = 4
GSPLAT_ENV = Path("scripts") / "gsplat-env"
# 합성 픽스처의 척도 마커: ID, 바닥 위 중심 (x, y) 미터, z축 둘레 회전 (도)
_FIXTURE_MARKER = (7, (0.15, 0.2), 25.0)


@dataclass
class _Ctx:
    """명령 하나의 저장소·DB·정책."""

    args: argparse.Namespace
    root: Path
    policy: Scene3dPolicy
    labeling: ObjectStore
    engine: sa.Engine | None
    raw_bucket: str

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
    # localize에는 --no-db 인자가 없다 (늘 DB를 읽는다)
    no_db = bool(getattr(args, "no_db", False))
    engine = None if no_db else sa.create_engine(database_url(args.url))
    labeling = store_from_spec(args.store, buckets.labeling)
    return _Ctx(args, root, load_policy(root), labeling, engine, raw_bucket=buckets.raw)


def _refused(exc: Exception) -> int:
    print(f"거부: {exc}", file=sys.stderr)
    return REFUSED_EXIT


def cmd_fixture(args: argparse.Namespace) -> int:
    """합성 전후 사진과 정답을 쓴다 (선택: 척도 마커, 미터가 아닌 stub SfM, 바디캠 영상)."""
    markers: list[SceneMarker] = []
    if args.marker_side_m is not None:
        marker_id, center, yaw = _FIXTURE_MARKER
        dictionary = load_policy(repo_root()).reconstruction.marker.dictionary
        markers = [floor_marker(marker_id, center, args.marker_side_m, yaw, dictionary)]
    if args.ego:
        pair, ego = generate_ego_scene(args.seed, markers=markers)
    else:
        pair, ego = generate_scene_pair(seed=args.seed, markers=markers), None
    path = write_scene_pair(pair, Path(args.out), ego=ego, sfm_unit_scale=args.unit_scale)
    print(f"정답: {path}")
    if ego is not None:
        print(f"바디캠: {path.parent / 'ego' / 'bodycam.mp4'} (프레임 {len(ego.frames)}장)")
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
    # 태그는 탐지 세대 버전의 해시다 (키에 쓴 값). 버전 문자열은 탐지기가 늘면 태그 열(64자)을
    # 넘으므로 model_version 열에만 둔다 (F-PAR-97)
    c.record(
        "privacy_detection",
        args.phase,
        det.model_version,
        short_hash(det.model_version),
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
    print(_scale_line(rec))
    print(f"스플랫: {rec.splat_key}")
    return 0


def _scale_line(rec: Reconstruction) -> str:
    """복원의 척도 출처 한 줄 (마커로 정했는지, 못 정했으면 왜)."""
    s = rec.scale
    if s is None:
        return "척도: 출처 기록 없음"
    if s.source == "marker":
        return (
            f"척도: 마커 {list(s.marker_ids)} (한 변 {s.marker_side_m} m, "
            f"사진 {s.marker_photos}장), 배율 {s.factor:.5f}, 변 길이 흩어짐 {s.side_spread:.4f}"
        )
    head = "척도: SfM 결과가 미터" if s.source == "sfm" else "척도: 정하지 못함 (미터 아님)"
    check = "" if s.marker_factor is None else f", 마커 확인 배율 {s.marker_factor:.5f}"
    return f"{head}{check}" + (f" ({s.reason})" if s.reason else "")


def _detector(c: _Ctx) -> ObjectDetector:
    op = c.policy.objects
    if c.args.oracle:
        return oracle_objects(Path(c.args.oracle))
    if op.detector != "owlv2":
        raise ModelUnavailableError("정책 objects.detector가 unavailable입니다")
    model, version = resolve(c.root, op.model)
    tokenizer, _ = resolve(c.root, op.tokenizer)
    return OwlObjectDetector(model, tokenizer, version, op.queries, op.score_threshold, op.nms_iou)


def _record_alignment(c: _Ctx) -> None:
    """지금 두 복원의 정렬 기록이 버킷에 있으면 DB에도 한 줄 남긴다 (받아들인 것·거부된 것 모두).

    정렬은 compare와 localize 가운데 먼저 돈 쪽이 만든다. 같은 (종류, 태그, 키)는 한 번만 남는다.
    """
    sid = c.args.session_id
    before = stored_reconstruction(c.labeling, sid, "before")
    after = stored_reconstruction(c.labeling, sid, "after")
    if before is None or after is None:
        return
    tag = alignment_tag(c.policy, before[0], after[0])
    found = get_model(c.labeling, alignment_key(sid, tag), Alignment)
    if found is not None:
        c.record("alignment", None, found.model_version, tag, "labeling", alignment_key(sid, tag))


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
        _record_alignment(c)
        return _refused(exc)
    _record_alignment(c)
    # 태그는 비교 결과 키의 해시다. 모델 버전 문자열(80자 이상)은 태그 열(64자)에 들어가지 않는다
    # (PostgreSQL에서 기록이 실패한다, F-PAR-97)
    tag = report_tag(report)
    c.record(
        "change_report",
        None,
        report.model_version,
        tag,
        "labeling",
        change_key(args.session_id, tag),
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


def _features(c: _Ctx) -> FrameFeatureExtractor:
    """바디캠 프레임 특징 추출기. --oracle이면 정답 stub."""
    lp = c.policy.localization
    if c.args.oracle:
        return oracle_features(Path(c.args.oracle))
    if lp.features != "pycolmap":
        raise ModelUnavailableError("정책 localization.features가 unavailable입니다")
    return IsolatedFrameFeatures(c.root / GSPLAT_ENV, lp.max_features)


def cmd_localize(args: argparse.Namespace) -> int:
    """장면 안 바디캠 위치 추정."""
    c = _ctx(args)
    if c.engine is None:  # 인자에 --no-db가 없어 늘 있다 (타입 좁히기)
        raise SystemExit("localize에는 DB가 필요합니다")
    sid = args.session_id
    try:
        extractor = _features(c)
        with tempfile.TemporaryDirectory() as tmp:
            record, new = localize(
                c.engine,
                c.labeling,
                sid,
                extractor,
                c.policy,
                load_privacy_policy(c.root),
                Path(tmp),
                datetime.now(UTC),
                raw_bucket=c.raw_bucket,
            )
    except ModelUnavailableError as exc:
        print(f"모델 없음: {exc}", file=sys.stderr)
        return UNAVAILABLE_EXIT
    except (Scene3dError, VideoGateError, VideoBoundaryError, OracleError) as exc:
        _record_alignment(c)
        return _refused(exc)
    _record_alignment(c)
    found = stored_localization(c.labeling, sid)
    tag = found[0] if found else ""
    c.record(
        "localization", None, record.model_version, tag, "labeling", localization_key(sid, tag)
    )
    after = sum(1 for p in record.poses if p.reference == "after")
    print(
        f"{'위치 추정' if new else '같은 위치 추정(건너뜀)'}: 프레임 {record.localized}/"
        f"{record.frame_count} (before 기준 {record.localized - after}, after 기준 {after}), "
        f"metric={record.metric}, 내부 파라미터 {record.intrinsics_origin}"
    )
    print(f"위치 기록: {localization_key(sid, tag)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """전후 3D 공간이 갖춰졌는지 (읽기 전용)."""
    c = _ctx(args)
    # 원본 저장소는 표시 파일이 있는지 head로만 본다 (내용을 읽지 않아 감사 기록도 쓰지 않는다)
    status = scene_status(c.labeling, args.session_id, c.policy, c.raw("scene3d.status"))
    print("\n".join(format_status(status)))
    return 0 if status.ready else INCOMPLETE_EXIT


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


def _common(p: argparse.ArgumentParser, phase: bool = True, no_db: bool = True) -> None:
    p.add_argument("session_id")
    if phase:
        p.add_argument("phase", choices=["before", "after"])
    p.add_argument("--store", default="s3", help="s3 또는 local:<디렉터리>")
    p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    if no_db:
        p.add_argument("--no-db", action="store_true", help="DB 없이 (local: 저장소만)")
    p.add_argument("--oracle", help="합성 픽스처 truth.json (stub, 시연·시험용)")


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`scene3d …` 하위 명령을 등록한다."""
    top = sub.add_parser(
        "scene3d", help="청소 전후 사진 3D 복원·물체 이동 비교, 바디캠 위치 추정 (ADR 0066·0067)"
    )
    s = top.add_subparsers(dest="scene3d_command", required=True)
    fx = s.add_parser("fixture", help="합성 전후 사진과 정답")
    fx.add_argument("--out", required=True)
    fx.add_argument("--seed", type=int, default=0)
    fx.add_argument("--marker-side-m", type=float, help="바닥에 그릴 척도 마커의 한 변 (m)")
    fx.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="stub SfM이 좌표에 곱할 배율 (미터가 아닌 복원)",
    )
    fx.add_argument("--ego", action="store_true", help="방을 지나는 바디캠 영상과 정답 자세")
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
    lc = s.add_parser("localize", help="바디캠 프레임별 카메라 자세 (before 장면 좌표)")
    _common(lc, phase=False, no_db=False)
    lc.set_defaults(func=cmd_localize)
    st = s.add_parser("status", help="전후 3D 공간이 갖춰졌는지 (읽기 전용)")
    _common(st, phase=False)
    st.set_defaults(func=cmd_status)
