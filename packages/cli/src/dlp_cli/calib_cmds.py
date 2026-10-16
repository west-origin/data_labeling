"""카메라 내부 파라미터·작업자 손 크기 등록 하위 명령 (ADR 0045).

등록하는 명령:
- `dlp calib intrinsics <기기> <체커보드 영상> --board 8x6 --square-mm 25 [--valid-from ISO]`
  — 그 기기로 체커보드만 찍은 로컬 영상에서 OpenCV로 내부 파라미터를 구해 DB `camera_calibrations`에
  등록한다 (`config/policies/calibration.yaml`). `--dry-run`이면 결과만 보이고 쓰지 않는다.
- `dlp calib import <기기> <파일> [--valid-from ISO]` — 알려진 캘리브레이션 파일(이 저장소 형식,
  OpenCV FileStorage, ROS camera_info)을 등록한다.
- `dlp calib palm <작업자> <손바닥 m>` — 잰 손바닥 길이(손목 → 중지 MCP)를 등록한다.
- `dlp calib palm-estimate <작업자> [--apply]` — 그 작업자의 캘리브레이션된 세션 바디캠에서
  깊이 모델로 손바닥 길이를 추정한다 (원본 영상, 감사 기록 `calib.palm_estimate`). `--apply`일
  때만 등록한다.
- `dlp calib show [--camera 기기] [--worker 작업자]` — 등록 이력과 지금 쓰는 값.

멱등성: 같은 값을 다시 등록하면 쓰지 않는다 (기기는 같은 유효 시작, 작업자는 최신 값·출처 비교).
세션은 수집 매니페스트 `calibration.camera_id`로 기기를 가리킨다. 등록 뒤
`dlp prelabel run <세션>`을 다시 돌리면 3D 단계 입력 해시가 바뀌어 검수 전 3D만 새 값으로 다시
만든다 (검수된 3D는 남는다).

체커보드 영상·캘리브레이션 파일은 원본 버킷이 아닌 로컬 파일이다 (보드만 찍는다, 절차는
`docs/handover/camera-calibration.md`). 작업자 ID는 DB에만 남고 내보내기에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.calibration import (
    BoardSpec,
    CalibrationError,
    calibrate_video,
    load_intrinsics_file,
)
from dlp_media.calibration import load_policy as load_calibration_policy
from dlp_media.storage import ObjectStore, sha256_file
from dlp_prelabel.common import iter_frames
from dlp_prelabel.lift3d import DepthLifter, DepthModel
from dlp_prelabel.palm import estimate_palm, palm_samples
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import pick_lift_tracks
from dlp_schema import repo_root
from dlp_schema.calibration import (
    CalibrationMethod,
    HandSizeSource,
    effective_calibration,
    latest_hand_size,
)
from dlp_schema.db.calibration_repository import (
    get_camera_calibrations,
    get_worker_hand_sizes,
    register_camera_calibration,
    register_hand_size,
    session_intrinsics,
    worker_session_ids,
)
from dlp_schema.db.repository import get_labels, get_session, withdrawn_session_ids
from dlp_schema.episode import current_labels
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.session import CameraIntrinsics


def _board(text: str) -> tuple[int, int]:
    """`8x6` → (8, 6) (안쪽 코너 열, 행)."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"보드는 '<열>x<행>' 안쪽 코너 수입니다: {text}") from exc
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError("안쪽 코너는 3 이상이어야 합니다")
    return cols, rows


def _when(text: str) -> datetime:
    """ISO 시각 (시간대 필수)."""
    t = datetime.fromisoformat(text)
    if t.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"시간대가 있는 ISO 시각이어야 합니다: {text}")
    return t


def _describe(k: CameraIntrinsics) -> str:
    return (
        f"{k.width}x{k.height} fx {k.fx:.2f} fy {k.fy:.2f} cx {k.cx:.2f} cy {k.cy:.2f} "
        f"왜곡 {list(k.distortion)}"
    )


def _register(
    args: argparse.Namespace,
    intrinsics: CameraIntrinsics,
    method: CalibrationMethod,
    *,
    rms_px: float | None = None,
    n_views: int | None = None,
) -> int:
    """기기 캘리브레이션을 등록하고 결과를 출력한다 (`--dry-run`이면 쓰지 않는다)."""
    if getattr(args, "dry_run", False):
        print("[미등록] --dry-run")
        return 0
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            record, written = register_camera_calibration(
                conn,
                args.camera_id,
                intrinsics,
                method=method,
                now=datetime.now(UTC),
                rms_px=rms_px,
                n_views=n_views,
                source_sha256=sha256_file(args.path),
                valid_from=args.valid_from,
            )
    finally:
        engine.dispose()
    state = "등록" if written else "같은 값이 이미 있어 건너뜀"
    print(f"{args.camera_id}: {state} ({record.calibration_id})")
    return 0


def cmd_intrinsics(args: argparse.Namespace) -> int:
    """`dlp calib intrinsics`: 체커보드 영상으로 내부 파라미터를 구해 등록한다.

    반환: 성공 0, 캘리브레이션 실패(장면 부족, 재투영 오차 초과) 2.
    """
    policy = load_calibration_policy(repo_root())
    cols, rows = args.board
    try:
        result = calibrate_video(args.path, BoardSpec(cols, rows, args.square_mm / 1000), policy)
    except CalibrationError as exc:
        print(f"[실패] {exc}")
        return 2
    print(f"{_describe(result.intrinsics)}")
    print(f"재투영 RMS {result.rms_px:.3f} px, 장면 {result.n_views}개")
    return _register(
        args,
        result.intrinsics,
        CalibrationMethod.CHECKERBOARD,
        rms_px=result.rms_px,
        n_views=result.n_views,
    )


def cmd_import(args: argparse.Namespace) -> int:
    """`dlp calib import`: 캘리브레이션 파일을 읽어 등록한다. 반환: 성공 0, 형식 오류 2."""
    try:
        intrinsics = load_intrinsics_file(args.path)
    except CalibrationError as exc:
        print(f"[실패] {exc}")
        return 2
    print(_describe(intrinsics))
    return _register(args, intrinsics, CalibrationMethod.IMPORTED)


def cmd_palm(args: argparse.Namespace) -> int:
    """`dlp calib palm <작업자> <m>`: 잰 손바닥 길이를 등록한다."""
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            record, written = register_hand_size(
                conn,
                args.worker_id,
                args.palm_m,
                source=HandSizeSource.MEASURED,
                now=datetime.now(UTC),
            )
    finally:
        engine.dispose()
    state = "등록" if written else "같은 값이 이미 있어 건너뜀"
    print(f"{args.worker_id}: 손바닥 {record.palm_m:.4f} m {state}")
    return 0


def _session_samples(
    conn: sa.Connection,
    session_id: str,
    raw: ObjectStore,
    depth: DepthModel,
    policy: PrelabelPolicy,
    work: Path,
) -> list[float] | None:
    """세션 하나의 손바닥 길이 표본. 캘리브레이션이 없거나 손 트랙이 없으면 None (건너뜀)."""
    session = get_session(conn, session_id)
    intr = session_intrinsics(conn, session)
    if intr.intrinsics is None:
        print(f"{session_id}: 캘리브레이션이 없어 건너뜀")
        return None
    body = session.reference_stream
    tracks = pick_lift_tracks(current_labels(get_labels(conn, session_id)), body.stream_id)
    if not tracks:
        print(f"{session_id}: 손 트랙이 없어 건너뜀")
        return None
    key = body.uri.removeprefix(raw.uri(""))
    video = work / key.replace("/", "__")
    raw.get_file(key, video)
    try:
        samples = palm_samples(iter_frames(video), depth, tracks, intr.intrinsics, policy)
    finally:
        video.unlink(missing_ok=True)
    print(f"{session_id}: 표본 {len(samples)}개 ({intr.origin})")
    return samples


def cmd_palm_estimate(args: argparse.Namespace) -> int:
    """`dlp calib palm-estimate <작업자>`: 캘리브레이션된 세션에서 손바닥 길이를 추정한다.

    반환: 추정(·등록) 0, 깊이 모델 없음 1, 표본 부족 3.
    부작용: 원본 버킷 바디캠 읽기(감사 `calib.palm_estimate`), `--apply`면 `worker_hand_sizes`
    한 행.
    """
    root = repo_root()
    policy = load_policy(root)
    now = datetime.now(UTC)
    try:
        depth = DepthLifter(root, policy, now=now).depth_model()
    except ModelUnavailableError as exc:
        print(f"[모델 없음] depth3d: {exc}")
        return 1
    raw = raw_store(args.store, args.url, "calib.palm_estimate")
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
            withdrawn = withdrawn_session_ids(conn)
            per_session = [
                _session_samples(conn, sid, raw, depth, policy, Path(tmp))
                for sid in worker_session_ids(conn, args.worker_id)
                if sid not in withdrawn
            ]
            found = estimate_palm((s for s in per_session if s), policy)
            if found is None:
                print(f"[표본 부족] 최소 {policy.palm_estimate.min_samples}개가 필요합니다")
                return 3
            print(
                f"{args.worker_id}: 손바닥 {found.palm_m:.4f} m "
                f"(표본 {found.n_samples}개, 세션 {found.n_sessions}개)"
            )
            if not args.apply:
                print("[미등록] --apply로 등록합니다")
                return 0
            _, written = register_hand_size(
                conn,
                args.worker_id,
                found.palm_m,
                source=HandSizeSource.ESTIMATED,
                now=now,
                n_samples=found.n_samples,
            )
            print("등록" if written else "같은 값이 이미 있어 건너뜀")
    finally:
        engine.dispose()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """`dlp calib show`: 기기·작업자 등록 이력과 지금 쓰는 값을 출력한다."""
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            if args.camera:
                records = get_camera_calibrations(conn, args.camera)
                now_used = effective_calibration(records, args.camera, datetime.now(UTC))
                for r in records:
                    mark = "*" if now_used and r.calibration_id == now_used.calibration_id else " "
                    since = r.valid_from.isoformat() if r.valid_from else "-"
                    print(f"{mark} {r.calibration_id} {r.method.value} from {since}: "
                          f"{_describe(r.intrinsics)}")  # fmt: skip
            if args.worker:
                sizes = get_worker_hand_sizes(conn, args.worker)
                latest = latest_hand_size(sizes, args.worker)
                for h in sizes:
                    mark = "*" if latest and h.record_id == latest.record_id else " "
                    print(f"{mark} {h.record_id} {h.source.value} {h.palm_m:.4f} m "
                          f"({h.created_at.isoformat()})")  # fmt: skip
    finally:
        engine.dispose()
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`calib intrinsics|import|palm|palm-estimate|show` 하위 명령을 등록한다."""
    calib = sub.add_parser("calib", help="카메라 내부 파라미터·작업자 손 크기 등록 (ADR 0045)")
    csub = calib.add_subparsers(dest="calib_command", required=True)
    intr = csub.add_parser("intrinsics", help="체커보드 영상으로 내부 파라미터를 구해 등록")
    intr.add_argument("camera_id")
    intr.add_argument("path", type=Path, help="그 기기로 체커보드만 찍은 로컬 영상")
    intr.add_argument("--board", type=_board, required=True, help="안쪽 코너 '<열>x<행>' (예: 8x6)")
    intr.add_argument("--square-mm", type=float, required=True, help="칸 한 변 (mm)")
    intr.add_argument("--valid-from", type=_when, help="이 시각 이후 녹화 세션에만 (ISO, 시간대)")
    intr.add_argument("--dry-run", action="store_true", help="결과만 보이고 등록하지 않는다")
    intr.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    intr.set_defaults(func=cmd_intrinsics)
    imp = csub.add_parser("import", help="알려진 캘리브레이션 파일 등록")
    imp.add_argument("camera_id")
    imp.add_argument("path", type=Path, help="YAML·JSON(이 저장소 형식), OpenCV FileStorage, ROS")
    imp.add_argument("--valid-from", type=_when, help="이 시각 이후 녹화 세션에만 (ISO, 시간대)")
    imp.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    imp.set_defaults(func=cmd_import)
    palm = csub.add_parser("palm", help="잰 손바닥 길이(손목→중지 MCP, m) 등록")
    palm.add_argument("worker_id")
    palm.add_argument("palm_m", type=float)
    palm.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    palm.set_defaults(func=cmd_palm)
    est = csub.add_parser("palm-estimate", help="캘리브레이션된 세션에서 손바닥 길이 추정")
    est.add_argument("worker_id")
    est.add_argument("--apply", action="store_true", help="추정값을 등록한다 (기본은 보고만)")
    est.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    est.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    est.set_defaults(func=cmd_palm_estimate)
    show = csub.add_parser("show", help="등록 이력과 지금 쓰는 값 (*)")
    show.add_argument("--camera")
    show.add_argument("--worker")
    show.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    show.set_defaults(func=cmd_show)
