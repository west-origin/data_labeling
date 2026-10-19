"""카메라 내부 파라미터·작업자 손 크기 등록 하위 명령 (ADR 0045, 0052).

기기 캘리브레이션:
- `dlp calib intrinsics <기기> <체커보드 영상> --board 8x6 --square-mm 25 [--valid-from ISO |
  --supersede <등록 ID>]` — 그 기기로 체커보드만 찍은 로컬 영상에서 OpenCV로 내부 파라미터를 구해
  DB `camera_calibrations`에 등록한다 (`config/policies/calibration.yaml`: 장면 수·RMS·보드 기울기·
  타당 범위). `--dry-run`이면 결과만 보이고 쓰지 않는다.
- `dlp calib import <기기> <파일> [--valid-from ISO | --supersede <등록 ID>]` — 알려진 캘리브레이션
  파일(이 저장소 형식, OpenCV FileStorage, ROS camera_info)을 같은 값 검사 뒤 등록한다.
- `dlp calib assign <기기> [<세션>...] [--worker W [--since ISO] [--until ISO]] --reason R` —
  매니페스트에 camera_id가 없던(이미 수집한) 세션을 기기에 잇는다 (`session_camera_assignments`).

작업자 손 크기:
- `dlp calib palm <작업자> <손바닥 m> [--allow-unknown-worker]` — 잰 손바닥 길이(손목 → 중지 MCP).
  세션이 없는 작업자 ID는 오타로 보고 거부한다 (첫 녹화 전 등록은 옵션).
- `dlp calib palm-estimate <작업자> [--apply [--override-measured]] [--reviewed-only]` — 그 작업자의
  캘리브레이션된 세션 바디캠에서 깊이 모델로 손바닥 길이를 추정한다 (원본 영상, 감사 기록
  `calib.palm_estimate`). `--apply`일 때만 등록하고, 지금 쓰이는 값이 잰 값이면 명시 옵션 없이는
  덮지 않는다. 표본을 낸 세션 ID와 추정기 판을 함께 남긴다.
- `dlp calib palm-revoke <작업자> --reason R [--record <등록 ID>]` — 손 크기 등록을 무효로 한다
  (동의 철회·삭제 요구·잘못 잰 값). 행을 지우지 않고 무효 기록을 더한다 (추가 전용).
- `dlp calib show [--camera 기기] [--worker 작업자] [--session 세션]` — 등록 이력과 지금 쓰는 값.

멱등성: 같은 값을 다시 등록하면 쓰지 않는다 (기기는 같은 유효 시작, 작업자는 지금 쓰이는 값·출처·
출처 세션, 기기 연결은 지금 기기 비교). 등록·무효·연결 뒤 `dlp prelabel run <세션>`을 다시 돌리면 3D
단계 입력 해시가 바뀌어 검수 전 3D만 새 값으로 다시 만든다 (검수된 3D는 남는다).

체커보드 영상·캘리브레이션 파일은 원본 버킷이 아닌 로컬 파일이다 (보드만 찍는다, 절차는
`docs/handover/camera-calibration.md`). 작업자 ID는 DB에만 남고 내보내기에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.exc import NoResultFound

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
from dlp_prelabel.palm import PalmEstimate, estimate_palm, palm_samples, usable_hand_times
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import pick_lift_tracks
from dlp_schema import repo_root
from dlp_schema.calibration import (
    CalibrationMethod,
    CameraCalibration,
    HandSizeSource,
    WorkerHandSize,
    effective_calibration,
)
from dlp_schema.db.calibration_repository import (
    CameraAssignmentConflictError,
    assign_session_camera,
    get_camera_assignments,
    get_camera_calibrations,
    hand_size_state,
    invalidate_hand_sizes,
    register_camera_calibration,
    register_hand_size,
    session_intrinsics,
    session_palm,
    worker_session_ids,
)
from dlp_schema.db.locks import share_lock_sessions
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    list_retention_decisions,
    withdrawn_session_ids,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, VerificationState
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.session import CameraIntrinsics, Session

# palm-estimate 종료 코드 (명령 설명·운영 문서와 같다)
EXIT_MODEL_MISSING = 1
EXIT_NO_SAMPLES = 3
EXIT_MEASURED_EXISTS = 4
EXIT_WITHDRAWN_DURING_RUN = 5


def _board(text: str) -> tuple[int, int]:
    """`8x6` → (8, 6) (안쪽 코너 열, 행. 칸 수가 아니라 칸 사이 꼭짓점 수다)."""
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"보드는 '<열>x<행>' 안쪽 코너 수입니다: {text}") from exc
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError("안쪽 코너는 3 이상이어야 합니다")
    return cols, rows


def _when(text: str) -> datetime:
    """ISO 시각 (시간대 필수: 유효 시작을 녹화 시각과 비교하므로 지역 시각을 추정하지 않는다)."""
    t = datetime.fromisoformat(text)
    if t.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"시간대가 있는 ISO 시각이어야 합니다: {text}")
    return t


def _describe(k: CameraIntrinsics) -> str:
    return (
        f"{k.width}x{k.height} fx {k.fx:.2f} fy {k.fy:.2f} cx {k.cx:.2f} cy {k.cy:.2f} "
        f"왜곡 {k.distortion_model} {list(k.distortion)}"
    )


# ── 기기 캘리브레이션 ──


def _supersede_start(
    conn: sa.Connection, camera_id: str, calibration_id: str
) -> tuple[bool, datetime | None]:
    """`--supersede`: 대체할 등록의 유효 시작. (찾았는지, 유효 시작)."""
    for r in get_camera_calibrations(conn, camera_id):
        if r.calibration_id == calibration_id:
            return True, r.valid_from
    return False, None


def _shadowed_by(
    records: list[CameraCalibration], record: CameraCalibration, now: datetime
) -> CameraCalibration | None:
    """새 등록이 자기 유효 시작(없으면 지금) 시점에 쓰이지 않으면 대신 쓰이는 등록.

    유효 시작 없는 등록은 유효 시작 있는 등록을 이기지 못한다 (`effective_calibration`). 운영자가
    잘못된 값을 유효 시작 없이 다시 등록하면 아무 효과가 없으므로 알린다 (F-PMS-25).
    """
    at = record.valid_from or now
    used = effective_calibration(records, record.camera_id, at)
    return used if used is not None and used.calibration_id != record.calibration_id else None


def _register(
    args: argparse.Namespace,
    intrinsics: CameraIntrinsics,
    method: CalibrationMethod,
    *,
    rms_px: float | None = None,
    n_views: int | None = None,
) -> int:
    """기기 캘리브레이션을 등록하고 결과를 출력한다 (`--dry-run`이면 쓰지 않는다).

    반환: 성공 0, `--supersede` 대상 없음 2.
    """
    if getattr(args, "dry_run", False):
        print("[미등록] --dry-run")
        return 0
    now = datetime.now(UTC)
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            valid_from = args.valid_from
            if args.supersede:
                found, valid_from = _supersede_start(conn, args.camera_id, args.supersede)
                if not found:
                    print(f"[실패] {args.camera_id}의 등록이 아닙니다: {args.supersede}")
                    return 2
            record, written = register_camera_calibration(
                conn,
                args.camera_id,
                intrinsics,
                method=method,
                now=now,
                rms_px=rms_px,
                n_views=n_views,
                source_sha256=sha256_file(args.path),
                valid_from=valid_from,
            )
            shadow = _shadowed_by(get_camera_calibrations(conn, args.camera_id), record, now)
    finally:
        engine.dispose()
    state = "등록" if written else "같은 값이 이미 있어 건너뜀"
    since = valid_from.isoformat() if valid_from else "-"
    print(f"{args.camera_id}: {state} ({record.calibration_id}, from {since})")
    if shadow is not None:
        shadow_since = shadow.valid_from.isoformat() if shadow.valid_from else "-"
        print(
            f"[주의] 이 등록은 {shadow.calibration_id}(from {shadow_since})에 가려 쓰이지 "
            f"않습니다. 그 값을 바로잡으려면 --supersede {shadow.calibration_id}로 다시 등록하세요"
        )
    return 0


def cmd_intrinsics(args: argparse.Namespace) -> int:
    """`dlp calib intrinsics`: 체커보드 영상으로 내부 파라미터를 구해 등록한다.

    반환: 성공 0, 캘리브레이션 실패(장면 부족, 재투영 오차 초과, 보드 기울기 부족, 타당 범위 밖) 2.
    """
    policy = load_calibration_policy(repo_root())
    cols, rows = args.board
    try:
        result = calibrate_video(args.path, BoardSpec(cols, rows, args.square_mm / 1000), policy)
    except CalibrationError as exc:
        print(f"[실패] {exc}")
        return 2
    print(f"{_describe(result.intrinsics)}")
    print(
        f"재투영 RMS {result.rms_px:.3f} px, 장면 {result.n_views}개, "
        f"최대 보드 기울기 {result.max_tilt_deg:.1f}도"
    )
    return _register(
        args,
        result.intrinsics,
        CalibrationMethod.CHECKERBOARD,
        rms_px=result.rms_px,
        n_views=result.n_views,
    )


def cmd_import(args: argparse.Namespace) -> int:
    """`dlp calib import`: 캘리브레이션 파일을 읽어 등록한다.

    반환: 성공 0, 형식·값 오류(파싱 오류, 모르는 키, 미보정·어안, 타당 범위 밖 포함) 2.
    """
    try:
        intrinsics = load_intrinsics_file(args.path, load_calibration_policy(repo_root()))
    except CalibrationError as exc:
        print(f"[실패] {exc}")
        return 2
    print(_describe(intrinsics))
    return _register(args, intrinsics, CalibrationMethod.IMPORTED)


def _assign_targets(conn: sa.Connection, args: argparse.Namespace) -> list[Session]:
    """`assign` 대상 세션: 위치 인자 세션과 `--worker`(녹화 시각 `--since`·`--until` 범위)."""
    ids = list(args.session_ids)
    if args.worker:
        ids += worker_session_ids(conn, args.worker)
    out: list[Session] = []
    for sid in dict.fromkeys(ids):
        s = get_session(conn, sid)
        if args.since and s.recorded_at < args.since:
            continue
        if args.until and s.recorded_at >= args.until:
            continue
        out.append(s)
    return out


def _assign_all(conn: sa.Connection, targets: list[Session], args: argparse.Namespace) -> int:
    """대상 세션을 모두 잇는다. 반환: 새로 쓴 연결 수.

    Raises:
        CameraAssignmentConflictError: 매니페스트에 다른 기기가 있는 세션.
    """
    now = datetime.now(UTC)
    written = 0
    for s in targets:
        _, new = assign_session_camera(conn, s, args.camera_id, reason=args.reason, now=now)
        written += new
        if s.calibration.intrinsics is not None:
            print(
                f"[주의] {s.session_id}: 매니페스트 내부 파라미터가 있어 기기 값이 쓰이지 않습니다"
            )
    return written


def cmd_assign(args: argparse.Namespace) -> int:
    """`dlp calib assign <기기> ...`: 이미 수집한 세션을 바디캠 기기에 잇는다 (추가만).

    반환: 성공 0, 대상 없음·없는 세션·매니페스트의 다른 기기와 충돌 2 (충돌이 하나라도 있으면 아무
    것도 쓰지 않는다).
    """
    if not args.session_ids and not args.worker:
        print("[실패] 세션 ID나 --worker를 주세요")
        return 2
    engine = sa.create_engine(database_url(args.url))
    try:
        # 충돌이 하나라도 있으면 예외로 트랜잭션을 빠져나가 앞서 쓴 연결도 되돌린다
        with engine.begin() as conn:
            targets = _assign_targets(conn, args)
            written = _assign_all(conn, targets, args)
    except NoResultFound:
        print("[실패] 없는 세션이 있습니다")
        return 2
    except CameraAssignmentConflictError as exc:
        print(f"[충돌] {exc} (아무것도 쓰지 않음)")
        return 2
    finally:
        engine.dispose()
    if not targets:
        print("[실패] 대상 세션이 없습니다")
        return 2
    print(f"{args.camera_id}: 세션 {len(targets)}개 중 {written}개 연결 (나머지는 이미 같은 기기)")
    print("다음: dlp prelabel run <세션> (검수 전 3D만 새 내부 파라미터로 다시 만든다)")
    return 0


# ── 작업자 손 크기 ──


def cmd_palm(args: argparse.Namespace) -> int:
    """`dlp calib palm <작업자> <m>`: 잰 손바닥 길이를 등록한다.

    반환: 성공 0, 세션이 없는 작업자 ID(오타 의심, `--allow-unknown-worker`로 허용) 2.
    """
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            if not args.allow_unknown_worker and not worker_session_ids(conn, args.worker_id):
                print(
                    f"[거부] 세션이 없는 작업자 ID: {args.worker_id} "
                    "(오타가 아니고 첫 녹화 전 등록이면 --allow-unknown-worker)"
                )
                return 2
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


def _reviewed(track: LabelRecord) -> bool:
    """검수자가 승인·표본 검증한 모델 트랙인가 (사람이 고친 트랙은 world가 없어 어차피 못 쓴다)."""
    return track.verification.state in (
        VerificationState.HUMAN_APPROVED,
        VerificationState.SAMPLE_VERIFIED,
    )


@dataclass(frozen=True)
class _EstimateInputs:
    """palm-estimate 한 번의 공통 입력."""

    raw: ObjectStore
    depth: DepthModel
    policy: PrelabelPolicy
    work: Path
    reviewed_only: bool


def _session_tracks(
    conn: sa.Connection, session: Session, reviewed_only: bool
) -> list[LabelRecord]:
    """3D 단계와 같은 입력 트랙 (`pick_lift_tracks`). reviewed_only면 검수된 트랙만."""
    picked = pick_lift_tracks(
        current_labels(get_labels(conn, session.session_id)), session.reference_stream.stream_id
    )
    return [t for t in picked if _reviewed(t)] if reviewed_only else picked


def _session_samples(conn: sa.Connection, session_id: str, inp: _EstimateInputs) -> list[float]:
    """세션 하나의 손바닥 길이 표본 (건너뛰면 빈 목록).

    원본(감사 대상)은 표본을 낼 수 있을 때만 받는다: 캘리브레이션이 있고, 손 3D 점(world)이 있는 손
    키프레임이 있고, 원본 객체가 있을 때. 존재 확인(head)은 내용을 읽지 않아 감사 기록이 없다.
    """
    session = get_session(conn, session_id)
    intr = session_intrinsics(conn, session)
    if intr.intrinsics is None:
        print(f"{session_id}: 캘리브레이션이 없어 건너뜀")
        return []
    tracks = _session_tracks(conn, session, inp.reviewed_only)
    if not usable_hand_times(tracks):
        print(f"{session_id}: 손 3D 점(world)이 있는 손 트랙이 없어 건너뜀")
        return []
    key = session.reference_stream.uri.removeprefix(inp.raw.uri(""))
    if inp.raw.head(key) is None:
        print(f"{session_id}: 원본이 없어 건너뜀 (보관 만료 삭제 등)")
        return []
    video = inp.work / key.replace("/", "__")
    inp.raw.get_file(key, video)
    try:
        samples = palm_samples(iter_frames(video), inp.depth, tracks, intr.intrinsics, inp.policy)
    finally:
        video.unlink(missing_ok=True)
    print(f"{session_id}: 표본 {len(samples)}개 ({intr.origin})")
    return samples


def _delete_decided(conn: sa.Connection, session_id: str) -> bool:
    """원본 보관 결정의 최신이 삭제인가 (지운 원본에 접근을 시도하지 않는다)."""
    decisions = list_retention_decisions(conn, session_id)
    return bool(decisions) and decisions[-1].decision == "delete"


def _candidate_sessions(conn: sa.Connection, worker_id: str) -> list[str] | None:
    """추정에 쓸 세션 (사용 중지·원본 삭제 결정 제외). 작업자 세션이 하나도 없으면 None."""
    ids = worker_session_ids(conn, worker_id)
    if not ids:
        return None
    withdrawn = withdrawn_session_ids(conn)
    return [s for s in ids if s not in withdrawn and not _delete_decided(conn, s)]


def _describe_hand_size(h: WorkerHandSize | None) -> str:
    if h is None:
        return "정책 기본값"
    return f"{h.palm_m:.4f} m ({h.source.value}, {h.record_id})"


def _apply_estimate(
    conn: sa.Connection,
    args: argparse.Namespace,
    samples: dict[str, list[float]],
    found: PalmEstimate,
    estimator: str,
) -> int:
    """추정값을 등록한다 (잰 값 보호, 사용 중지 재확인). 반환: 종료 코드."""
    current = hand_size_state(conn, args.worker_id).effective(args.worker_id)
    if (
        current is not None
        and current.source is HandSizeSource.MEASURED
        and not args.override_measured
    ):
        print(
            f"[거부] 잰 값 {_describe_hand_size(current)}이 쓰이고 있습니다. 추정값으로 바꾸려면 "
            "--override-measured"
        )
        return EXIT_MEASURED_EXISTS
    # 표본을 낸 세션을 잠근 뒤 사용 중지를 다시 읽는다: 추정하는 동안 커밋된 사용 중지도 본다
    share_lock_sessions(conn, samples)
    gone = sorted(set(samples) & withdrawn_session_ids(conn))
    if gone:
        print(f"[막힘] 추정 중 사용 중지된 세션 {gone}. 다시 실행하세요 (아무것도 쓰지 않음)")
        return EXIT_WITHDRAWN_DURING_RUN
    _, written = register_hand_size(
        conn,
        args.worker_id,
        found.palm_m,
        source=HandSizeSource.ESTIMATED,
        now=datetime.now(UTC),
        n_samples=found.n_samples,
        source_session_ids=samples,
        estimator=estimator,
    )
    print("등록" if written else "같은 값이 이미 있어 건너뜀")
    return 0


def _estimate(
    conn: sa.Connection, args: argparse.Namespace, inp: _EstimateInputs, estimator: str
) -> int:
    """palm-estimate 본문 (한 트랜잭션). 반환: 종료 코드."""
    candidates = _candidate_sessions(conn, args.worker_id)
    if candidates is None:
        print(f"[세션 없음] 작업자 {args.worker_id}의 세션이 없습니다 (ID 오타?)")
        return EXIT_NO_SAMPLES
    if not candidates:
        print(f"[세션 없음] 작업자 {args.worker_id}의 세션이 모두 사용 중지·원본 삭제 결정입니다")
        return EXIT_NO_SAMPLES
    per_session = {sid: _session_samples(conn, sid, inp) for sid in candidates}
    samples = {sid: s for sid, s in per_session.items() if s}
    found = estimate_palm(samples.values(), inp.policy)
    if found is None:
        print(f"[표본 부족] 최소 {inp.policy.palm_estimate.min_samples}개가 필요합니다")
        return EXIT_NO_SAMPLES
    current = hand_size_state(conn, args.worker_id).effective(args.worker_id)
    print(
        f"{args.worker_id}: 손바닥 {found.palm_m:.4f} m "
        f"(표본 {found.n_samples}개, 세션 {found.n_sessions}개). 지금 값: "
        f"{_describe_hand_size(current)}"
    )
    if not args.apply:
        print("[미등록] --apply로 등록합니다")
        return 0
    return _apply_estimate(conn, args, samples, found, estimator)


def cmd_palm_estimate(args: argparse.Namespace) -> int:
    """`dlp calib palm-estimate <작업자>`: 캘리브레이션된 세션에서 손바닥 길이를 추정한다.

    반환: 추정(·등록) 0, 깊이 모델 없음 1, 세션 없음·표본 부족 3, 잰 값이 쓰이고 있어 거부 4,
    추정 중 출처 세션 사용 중지 5.
    부작용: 원본 버킷 바디캠 읽기(감사 `calib.palm_estimate`, 표본을 낼 수 있는 세션만),
    `--apply`면 `worker_hand_sizes` 한 행과 출처 세션 `worker_hand_size_sources`.
    """
    root = repo_root()
    policy = load_policy(root)
    try:
        lifter = DepthLifter(root, policy, now=datetime.now(UTC))
        depth = lifter.depth_model()
    except ModelUnavailableError as exc:
        print(f"[모델 없음] depth3d: {exc}")
        return EXIT_MODEL_MISSING
    # 추정기 판: 깊이 3D 판(가중치·depth 절 해시) + palm_estimate 절 해시
    estimator = f"{lifter.version}+pe{policy.digest('palm_estimate')}"
    raw = raw_store(args.store, args.url, "calib.palm_estimate")
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
            # 쓰기는 끝(_apply_estimate)에서만 하므로 실패 코드로 끝나도 쓴 것이 없다
            inp = _EstimateInputs(raw, depth, policy, Path(tmp), args.reviewed_only)
            return _estimate(conn, args, inp, estimator)
    finally:
        engine.dispose()


def cmd_palm_revoke(args: argparse.Namespace) -> int:
    """`dlp calib palm-revoke <작업자> --reason R [--record ID]`: 손 크기 등록을 무효로 한다.

    반환: 성공(무효로 할 유효한 등록이 없어 쓰지 않은 경우 포함) 0, 그 작업자의 등록이 아닌 ID 2.
    부작용: `worker_hand_size_invalidations` 한 행 (추가만). 행을 지우지 않는다.
    """
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.begin() as conn:
            try:
                record, targets = invalidate_hand_sizes(
                    conn,
                    args.worker_id,
                    reason=args.reason,
                    now=datetime.now(UTC),
                    record_id=args.record,
                )
            except KeyError as exc:
                print(f"[실패] {exc.args[0]}")
                return 2
            current = hand_size_state(conn, args.worker_id).effective(args.worker_id)
    finally:
        engine.dispose()
    if record is None:
        print(f"{args.worker_id}: 무효로 할 유효한 손 크기 등록이 없습니다 (건너뜀)")
    else:
        ids = ", ".join(t.record_id for t in targets)
        print(f"{args.worker_id}: 무효 {record.invalidation_id} ({ids})")
    print(f"지금 값: {_describe_hand_size(current)}")
    print("다음: dlp prelabel run <세션> (검수 전 3D만 다시 만든다)")
    return 0


# ── 보기 ──


def _show_camera(conn: sa.Connection, camera: str) -> None:
    records = get_camera_calibrations(conn, camera)
    now_used = effective_calibration(records, camera, datetime.now(UTC))
    for r in records:
        mark = "*" if now_used and r.calibration_id == now_used.calibration_id else " "
        since = r.valid_from.isoformat() if r.valid_from else "-"
        print(f"{mark} {r.calibration_id} {r.method.value} from {since}: {_describe(r.intrinsics)}")


def _show_worker(conn: sa.Connection, worker: str) -> None:
    state = hand_size_state(conn, worker)
    current = state.effective(worker)
    for h in state.records:
        mark = "*" if current and h.record_id == current.record_id else " "
        why = state.invalid_reason(h)
        sources = f" 출처 {list(h.source_session_ids)}" if h.source_session_ids else ""
        note = f" [쓰지 않음: {why}]" if why else ""
        print(
            f"{mark} {h.record_id} {h.source.value} {h.palm_m:.4f} m "
            f"({h.created_at.isoformat()}){sources}{note}"
        )
    for inv in state.invalidations:
        print(
            f"  무효 {inv.invalidation_id} {inv.record_id or '전체'} "
            f"({inv.created_at.isoformat()}): {inv.reason}"
        )


def _show_session(conn: sa.Connection, session_id: str) -> None:
    session = get_session(conn, session_id)
    for a in get_camera_assignments(conn, session_id):
        print(f"  연결 {a.assignment_id} → {a.camera_id} ({a.created_at.isoformat()}): {a.reason}")
    intr = session_intrinsics(conn, session)
    palm = session_palm(conn, session)
    print(f"{session_id}: 내부 파라미터 {intr.origin}, 손 크기 {palm.source} {palm.palm_m}")


def cmd_show(args: argparse.Namespace) -> int:
    """`dlp calib show`: 기기·작업자·세션 등록 이력과 지금 쓰는 값을 출력한다."""
    engine = sa.create_engine(database_url(args.url))
    try:
        with engine.connect() as conn:
            if args.camera:
                _show_camera(conn, args.camera)
            if args.worker:
                _show_worker(conn, args.worker)
            if args.session:
                _show_session(conn, args.session)
    finally:
        engine.dispose()
    return 0


def _add_calibration_target(p: argparse.ArgumentParser) -> None:
    """기기 등록 공통 인자: 유효 시작 또는 대체할 등록."""
    group = p.add_mutually_exclusive_group()
    group.add_argument("--valid-from", type=_when, help="이 시각 이후 녹화 세션에만 (ISO, 시간대)")
    group.add_argument(
        "--supersede", help="이 등록(ID)을 바로잡는다: 그 등록의 유효 시작을 그대로 쓴다"
    )
    p.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")


def _add_camera_commands(csub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    intr = csub.add_parser("intrinsics", help="체커보드 영상으로 내부 파라미터를 구해 등록")
    intr.add_argument("camera_id")
    intr.add_argument("path", type=Path, help="그 기기로 체커보드만 찍은 로컬 영상")
    intr.add_argument("--board", type=_board, required=True, help="안쪽 코너 '<열>x<행>' (예: 8x6)")
    intr.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help="칸 한 변 (mm). 보드 좌표 단위로만 쓰고 내부 파라미터에는 영향 없음",
    )
    intr.add_argument("--dry-run", action="store_true", help="결과만 보이고 등록하지 않는다")
    _add_calibration_target(intr)
    intr.set_defaults(func=cmd_intrinsics)
    imp = csub.add_parser("import", help="알려진 캘리브레이션 파일 등록")
    imp.add_argument("camera_id")
    imp.add_argument("path", type=Path, help="YAML·JSON(이 저장소 형식), OpenCV FileStorage, ROS")
    _add_calibration_target(imp)
    imp.set_defaults(func=cmd_import)
    asg = csub.add_parser("assign", help="이미 수집한(매니페스트에 기기 없는) 세션을 기기에 잇기")
    asg.add_argument("camera_id")
    asg.add_argument("session_ids", nargs="*", help="세션 ID")
    asg.add_argument("--worker", help="이 작업자의 세션 전부 (--since·--until로 녹화 시각 범위)")
    asg.add_argument("--since", type=_when, help="녹화 시각 하한 (포함, ISO, 시간대)")
    asg.add_argument("--until", type=_when, help="녹화 시각 상한 (제외, ISO, 시간대)")
    asg.add_argument("--reason", required=True, help="근거 (예: 촬영 기록부)")
    asg.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    asg.set_defaults(func=cmd_assign)


def _add_palm_commands(csub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    palm = csub.add_parser("palm", help="잰 손바닥 길이(손목→중지 MCP, m) 등록")
    palm.add_argument("worker_id")
    palm.add_argument("palm_m", type=float)
    palm.add_argument(
        "--allow-unknown-worker", action="store_true", help="세션이 아직 없는 작업자도 등록"
    )
    palm.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    palm.set_defaults(func=cmd_palm)
    est = csub.add_parser("palm-estimate", help="캘리브레이션된 세션에서 손바닥 길이 추정")
    est.add_argument("worker_id")
    est.add_argument("--apply", action="store_true", help="추정값을 등록한다 (기본은 보고만)")
    est.add_argument(
        "--override-measured", action="store_true", help="지금 쓰이는 잰 값을 추정값으로 바꾼다"
    )
    est.add_argument("--reviewed-only", action="store_true", help="승인·표본 검증한 손 트랙만 쓴다")
    est.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    est.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    est.set_defaults(func=cmd_palm_estimate)
    rev = csub.add_parser("palm-revoke", help="손 크기 등록 무효 (동의 철회·삭제 요구·잘못 잰 값)")
    rev.add_argument("worker_id")
    rev.add_argument("--reason", required=True, help="근거 (예: 동의 철회, 요청 번호)")
    rev.add_argument("--record", help="이 등록 ID만 (기본: 그 작업자의 지금까지 모든 등록)")
    rev.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    rev.set_defaults(func=cmd_palm_revoke)


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`calib intrinsics|import|assign|palm|palm-estimate|palm-revoke|show` 하위 명령을 등록한다."""
    calib = sub.add_parser("calib", help="카메라 내부 파라미터·작업자 손 크기 등록 (ADR 0045)")
    csub = calib.add_subparsers(dest="calib_command", required=True)
    _add_camera_commands(csub)
    _add_palm_commands(csub)
    show = csub.add_parser("show", help="등록 이력과 지금 쓰는 값 (*)")
    show.add_argument("--camera")
    show.add_argument("--worker")
    show.add_argument("--session", help="세션이 지금 쓰는 내부 파라미터·손 크기와 기기 연결")
    show.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    show.set_defaults(func=cmd_show)
