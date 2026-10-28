"""카메라 내부 파라미터 캘리브레이션: 체커보드 영상 → OpenCV calibrateCamera, 또는 파일 가져오기
(ADR 0045·0052, `dlp calib intrinsics|import`).

입력은 운영자가 그 카메라로 체커보드만 찍은 로컬 영상(또는 이미 아는 캘리브레이션 파일)이다. 세션
영상(원본 버킷)을 읽지 않으므로 원본 접근이 아니다 (보드 외의 사람·장소를 찍지 않는다, 절차 문서
`docs/handover/camera-calibration.md`).

용어와 단위
    - 내부 파라미터 (픽셀 단위): fx·fy는 초점 거리(px), cx·cy는 주점(광축이 영상면과 만나는 점, px,
      왼쪽 위 화소 중심이 (0, 0)). 핀홀 투영은 u = fx·X/Z + cx, v = fy·Y/Z + cy.
    - 왜곡 계수: OpenCV 방사·접선 모델 순서 (k1, k2, p1, p2[, k3[, k4, k5, k6[, ...]]]),
      단위 없음. 어안(equidistant) 모델은 계수 뜻이 달라 받지 않는다 (`cv2.fisheye`가
      필요, F-PAR-28).
    - 보드 칸 한 변(m)은 보드 좌표의 단위일 뿐이다. 보드 전체를 같은 비율로 키워도 fx·fy·cx·cy·
      왜곡은 같고 외부 파라미터(보드 위치)의 이동량만 바뀐다. 결과를 좌우하는 것은 칸이
      정사각형인지, 그리고 보드 자세의 다양성(특히 기울기)이다.

체커보드 캘리브레이션 (`calibrate_video`)
    PTS 간격(`sample_every_ms`)마다 프레임을 검사해 보드 코너를 찾고(`cv2.findChessboardCornersSB`),
    앞서 채택한 장면과 거의 같은 장면은 건너뛴 뒤 `cv2.calibrateCamera`(Zhang 방법)로 푼다. 이
    방법은 보드 평면이 장면마다 다른 방향을 향한다고 가정한다. 보드가 늘 영상면과 평행하면(기울기
    없음) 초점 거리와 보드 거리를 구분할 수 없어 퇴화(degenerate)한다: 해가 정해지지 않아 최적화가
    큰 왜곡 계수로 보상한 엉뚱한 값으로 수렴하는데도 재투영 RMS는 작다. 그래서 장면 수·RMS뿐 아니라
    장면별 보드 기울기의 최대값(`min_max_view_tilt_deg`)과 결과의 타당 범위(`check_intrinsics`)를
    본다. 어느 하나라도 못 맞추면 `CalibrationError`이고 등록하지 않는다.

파일 가져오기 (`load_intrinsics_file`)
    이 저장소 형식(YAML/JSON, `CameraIntrinsics` 필드), OpenCV FileStorage(`%YAML:1.0` 또는 XML:
    `camera_matrix`, `distortion_coefficients`, `image_width`, `image_height`), ROS camera_info
    YAML(같은 키, 행렬은 `data` 목록, `distortion_model`)을 읽는다. 파싱 오류·모르는 키·
    지원하지 않는 왜곡 모델·타당하지 않은 값은 모두 `CalibrationError`다 (CLI 종료 코드 2).

한계: 핀홀 + OpenCV 방사·접선 왜곡 모델만 다룬다. 3D 단계는 아직 왜곡 계수를 쓰지 않는다
(fx·fy·cx·cy만, `dlp_prelabel.lift3d.intrinsics_for`, F-PAR-28).
"""

# PyAV·OpenCV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pydantic
import yaml
from numpy.typing import NDArray
from pydantic import Field

from dlp_media.probe import probe, require_upright
from dlp_schema.common import Contract
from dlp_schema.session import CameraIntrinsics

# 받는 왜곡 모델 이름. 계약 `CameraIntrinsics.distortion_model`의 값과 같은 "opencv"만 등록한다.
SUPPORTED_DISTORTION_MODEL = "opencv"
# ROS camera_info의 distortion_model 중 OpenCV 방사·접선 모델과 같은 뜻인 것
# (plumb_bob: k1 k2 p1 p2 k3, rational_polynomial: + k4 k5 k6)
ROS_OPENCV_MODELS = frozenset({"plumb_bob", "rational_polynomial"})
# OpenCV가 받는 왜곡 계수 개수 (0은 왜곡 없음)
DISTORTION_LENGTHS = frozenset({0, 4, 5, 8, 12, 14})


class CalibrationError(RuntimeError):
    """캘리브레이션을 받아들일 수 없다 (장면 부족·퇴화, 재투영 오차 초과, 파일 형식·값)."""


class IntrinsicsCalibrationPolicy(Contract):
    """`calibration.yaml intrinsics` 절 (값의 뜻은 YAML 주석)."""

    sample_every_ms: int = Field(ge=1)
    min_views: int = Field(ge=3)
    max_views: int = Field(ge=3)
    min_motion_px: float = Field(ge=0)
    max_rms_px: float = Field(gt=0)
    fix_k3: bool
    zero_tangent: bool
    min_max_view_tilt_deg: float = Field(ge=0, lt=90)
    hfov_range_deg: tuple[float, float]
    max_principal_offset_frac: float = Field(gt=0, le=0.5)
    max_fx_fy_ratio: float = Field(ge=1)
    max_abs_distortion: float = Field(gt=0)


class CalibrationPolicy(Contract):
    version: int
    intrinsics: IntrinsicsCalibrationPolicy


def load_policy(root: Path) -> IntrinsicsCalibrationPolicy:
    """`<root>/config/policies/calibration.yaml`의 intrinsics 절."""
    data: Any = yaml.safe_load((root / "config/policies/calibration.yaml").read_text("utf-8"))
    return CalibrationPolicy.model_validate(data).intrinsics


@dataclass(frozen=True)
class BoardSpec:
    """체커보드: 안쪽 코너 (열, 행)과 칸 한 변 (m).

    칸 한 변은 보드 좌표의 단위로만 쓴다 (내부 파라미터에는 영향이 없다, 모듈 설명).
    """

    cols: int
    rows: int
    square_m: float

    def object_points(self) -> NDArray[np.float32]:
        """보드 좌표 코너 (cols*rows, 3), OpenCV 검출 순서 (행마다 왼쪽 → 오른쪽), 평면 Z=0."""
        grid = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        pts = np.zeros((self.cols * self.rows, 3), np.float32)
        pts[:, :2] = grid * self.square_m
        return pts


@dataclass(frozen=True)
class CalibrationResult:
    """캘리브레이션 결과: 내부 파라미터, 재투영 RMS(px), 쓴 장면 수·PTS ms, 최대 보드 기울기(도)."""

    intrinsics: CameraIntrinsics
    rms_px: float
    n_views: int
    view_times_ms: tuple[int, ...]
    max_tilt_deg: float


def _gray_frames(path: Path, every_ms: int) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """(PTS ms, 회색조 프레임). 마지막으로 낸 프레임에서 every_ms가 지난 프레임만 낸다.

    시각은 PTS * time_base로만 구한다 (프레임 번호 * 간격은 가변 프레임률에서 틀린다).
    """
    last: int | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = round(float(frame.pts * frame.time_base) * 1000)
            if last is not None and t - last < every_ms:
                continue
            last = t
            yield t, np.asarray(frame.to_ndarray(format="gray"), np.uint8)


def _require_upright(path: Path) -> None:
    """표시 회전 태그가 있는 영상은 거부한다 (F-PMS-41).

    디코더는 태그를 적용하지 않은 저장 픽셀을 낸다. 수집(`require_upright`)은 그런 세션 영상을
    거부하고 `dlp media prepare`가 표시 방향으로 돌린 사본을 만들므로, 세션의 공간 라벨은 늘 표시
    방향 픽셀이다. 캘리브레이션만 저장 픽셀로 풀면 90°에서는 가로·세로와 fx·fy, 180°에서는 주점이
    세션 영상과 맞지 않는다. 같은 정규화를 거친 사본으로 캘리브레이션하게 한다.

    Raises:
        CalibrationError: 회전 태그가 있을 때 (prepare 안내).
    """
    try:
        require_upright(probe(path))
    except ValueError as exc:
        raise CalibrationError(
            f"{exc} 캘리브레이션 영상도 세션 영상과 같은 방향이어야 합니다: "
            f"`dlp media prepare {path} --out <똑바른 사본.mp4>`로 정규화한 사본을 쓰세요."
        ) from exc


def find_views(
    path: Path, board: BoardSpec, policy: IntrinsicsCalibrationPolicy
) -> tuple[list[tuple[int, NDArray[np.float32]]], tuple[int, int]]:
    """영상에서 보드 코너를 찾은 장면 (PTS ms, 코너 (n, 1, 2))과 프레임 크기 (가로, 세로).

    앞서 채택한 장면과 코너 평균 이동이 `min_motion_px`보다 작으면 건너뛴다 (보드를 멈춘 동안의 같은
    자세를 여러 번 넣으면 최소제곱이 그 자세에 치우친다).

    Raises:
        CalibrationError: 표시 회전 태그가 있을 때 (`_require_upright`).
    """
    _require_upright(path)
    views: list[tuple[int, NDArray[np.float32]]] = []
    size = (0, 0)
    for t, gray in _gray_frames(path, policy.sample_every_ms):
        size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCornersSB(gray, (board.cols, board.rows))
        if not found:
            continue
        pts = np.asarray(corners, np.float32).reshape(-1, 1, 2)
        if views:
            moved = float(np.mean(np.linalg.norm(pts - views[-1][1], axis=2)))
            if moved < policy.min_motion_px:
                continue
        views.append((t, pts))
    return views, size


def _spread(items: list[Any], k: int) -> list[Any]:
    """목록에서 고르게 k개 (k 이상이면 그대로)."""
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def board_tilt_deg(rvec: Any) -> float:
    """보드 법선과 카메라 광축 사이 각도 (도, 0이면 보드가 영상면과 평행).

    rvec는 calibrateCamera가 장면마다 돌려준 회전(로드리게스 벡터, 보드 → 카메라)이다. 보드 법선은
    보드 좌표 Z축이므로 카메라 좌표에서 R의 셋째 열이고, 광축(카메라 Z)과의 각도는 arccos(|R[2,
    2]|)다.
    앞뒤 어느 면이 보이든 같은 기울기로 본다 (절댓값).
    """
    rot, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return math.degrees(math.acos(min(1.0, abs(float(rot[2, 2])))))


def check_intrinsics(
    k: CameraIntrinsics, policy: IntrinsicsCalibrationPolicy | None, where: str
) -> CameraIntrinsics:
    """내부 파라미터가 3D 단계에 쓸 수 있는 값인지 본다. 통과하면 k를 그대로 돌려준다.

    기본 검사(정책 없이도): 모든 값이 유한하고, fx·fy > 0, 주점이 영상 안(0 ≤ cx < width,
    0 ≤ cy < height), 왜곡 모델이 opencv이고 계수 개수가 OpenCV가 받는 길이. fx가 0이면 3D 역투영이
    0으로 나누어 실패하고, 음수면 좌우가 뒤집힌 3D가 예외 없이 나온다.
    정책 검사(policy가 있으면): 가로 화각, 주점의 중앙 이탈, fx/fy 비, 왜곡 계수 절댓값 상한.

    Raises:
        CalibrationError: 하나라도 어기면 (where는 메시지 앞머리, 파일 경로 등).
    """
    problem = _basic_problem(k) or (_policy_problem(k, policy) if policy else None)
    if problem:
        raise CalibrationError(f"{where}: {problem}")
    return k


def _basic_problem(k: CameraIntrinsics) -> str | None:
    """정책과 무관하게 틀린 값의 이유 (없으면 None)."""
    values = (k.fx, k.fy, k.cx, k.cy, *k.distortion)
    if not all(math.isfinite(v) for v in values):
        return "내부 파라미터·왜곡 계수에 유한하지 않은 값(nan·inf)이 있습니다"
    if k.fx <= 0 or k.fy <= 0:
        return f"초점 거리는 양수여야 합니다 (fx {k.fx}, fy {k.fy})"
    if not (0 <= k.cx < k.width and 0 <= k.cy < k.height):
        return f"주점 ({k.cx}, {k.cy})이 영상 {k.width}x{k.height} 밖입니다"
    if k.distortion_model != SUPPORTED_DISTORTION_MODEL:
        return (
            f"왜곡 모델 {k.distortion_model!r}은 지원하지 않습니다 "
            "(핀홀 + OpenCV 방사·접선만, 어안은 F-PAR-28)"
        )
    if len(k.distortion) not in DISTORTION_LENGTHS:
        return f"왜곡 계수 {len(k.distortion)}개는 OpenCV 모델 길이가 아닙니다 (0·4·5·8·12·14)"
    return None


def _policy_problem(k: CameraIntrinsics, policy: IntrinsicsCalibrationPolicy) -> str | None:
    """정책 타당 범위를 벗어난 이유 (없으면 None)."""
    hfov = math.degrees(2 * math.atan(k.width / (2 * k.fx)))
    lo, hi = policy.hfov_range_deg
    if not lo <= hfov <= hi:
        return f"가로 화각 {hfov:.1f}도가 범위 [{lo}, {hi}] 밖입니다 (fx {k.fx})"
    off = max(abs(k.cx - k.width / 2) / k.width, abs(k.cy - k.height / 2) / k.height)
    if off > policy.max_principal_offset_frac:
        return (
            f"주점 ({k.cx:.1f}, {k.cy:.1f})이 화면 중앙에서 {off:.0%} 벗어났습니다 "
            f"(상한 {policy.max_principal_offset_frac:.0%})"
        )
    ratio = max(k.fx / k.fy, k.fy / k.fx)
    if ratio > policy.max_fx_fy_ratio:
        return f"fx/fy 비 {ratio:.3f}가 상한 {policy.max_fx_fy_ratio}를 넘습니다"
    worst = max((abs(d) for d in k.distortion), default=0.0)
    if worst > policy.max_abs_distortion:
        return f"왜곡 계수 절댓값 {worst:.3g}가 상한 {policy.max_abs_distortion}를 넘습니다"
    return None


def _solve(
    views: list[tuple[int, NDArray[np.float32]]],
    board: BoardSpec,
    size: tuple[int, int],
    policy: IntrinsicsCalibrationPolicy,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64], Sequence[Any]]:
    """calibrateCamera를 푼다: (재투영 RMS px, K 3x3, 왜곡 계수, 장면별 rvec).

    모든 장면이 같은 보드라 보드 좌표는 하나를 되풀이한다. 초기값 없이(None) 풀면 OpenCV가 장면별
    호모그래피로 초기값을 잡는다(Zhang).
    """
    obj = board.object_points()
    flags = (cv2.CALIB_FIX_K3 if policy.fix_k3 else 0) | (
        cv2.CALIB_ZERO_TANGENT_DIST if policy.zero_tangent else 0
    )
    rms, kmat, dist, rvecs, _ = cv2.calibrateCamera(
        [obj] * len(views), [v for _, v in views], size, None, None, flags=flags
    )
    return float(rms), np.asarray(kmat, np.float64), np.asarray(dist, np.float64), rvecs


def calibrate_video(
    path: Path, board: BoardSpec, policy: IntrinsicsCalibrationPolicy
) -> CalibrationResult:
    """체커보드 영상으로 내부 파라미터를 구한다 (같은 영상·정책이면 같은 결과, 결정적).

    Raises:
        CalibrationError: 보드 장면이 `min_views`보다 적을 때, 재투영 RMS가 `max_rms_px`를 넘을 때,
            장면별 보드 기울기의 최대값이 `min_max_view_tilt_deg`보다 작을 때(퇴화), 결과가 타당
            범위(`check_intrinsics`) 밖일 때.
    """
    views, (width, height) = find_views(path, board, policy)
    views = _spread(views, policy.max_views)
    if len(views) < policy.min_views:
        raise CalibrationError(
            f"보드를 찾은 장면 {len(views)}개 < 최소 {policy.min_views}개 "
            f"(보드 {board.cols}x{board.rows} 안쪽 코너가 맞는지, 여러 각도로 찍었는지 확인)"
        )
    rms, k, dist, rvecs = _solve(views, board, (width, height), policy)
    if rms > policy.max_rms_px:
        raise CalibrationError(f"재투영 RMS {rms:.3f} px > 상한 {policy.max_rms_px} px")
    # RMS가 작아도 보드가 늘 정면이면 해가 정해지지 않는다 (모듈 설명). 값 검사보다 먼저 보아
    # 운영자에게 "다시 찍는 법"을 알려 준다
    max_tilt = max(board_tilt_deg(r) for r in rvecs)
    if max_tilt < policy.min_max_view_tilt_deg:
        raise CalibrationError(
            f"보드를 앞뒤·좌우로 기울인 자세가 부족합니다 (최대 기울기 {max_tilt:.1f}도 < "
            f"{policy.min_max_view_tilt_deg}도). 보드를 20~40도 기울인 자세를 더 찍으세요"
        )
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=round(float(k[0, 0]), 3),
        fy=round(float(k[1, 1]), 3),
        cx=round(float(k[0, 2]), 3),
        cy=round(float(k[1, 2]), 3),
        distortion_model=SUPPORTED_DISTORTION_MODEL,
        distortion=tuple(round(float(d), 6) for d in dist.reshape(-1)),
    )
    check_intrinsics(intrinsics, policy, str(path))
    return CalibrationResult(
        intrinsics,
        round(rms, 4),
        len(views),
        tuple(t for t, _ in views),
        round(max_tilt, 2),
    )


def _intrinsics(path: Path, **values: Any) -> CameraIntrinsics:
    """계약을 만든다. 값이 계약에 맞지 않으면(모르는 키, width ≤ 0 등) CalibrationError."""
    try:
        return CameraIntrinsics.model_validate(values)
    except pydantic.ValidationError as exc:
        raise CalibrationError(f"{path}: 값이 계약에 맞지 않습니다 ({exc})") from exc


def _storage_model(fs: Any) -> str:
    """FileStorage의 distortion_model (없으면 opencv). ROS 이름(plumb_bob 등)은 opencv로 읽는다."""
    node = fs.getNode("distortion_model")
    name = "" if node.empty() else str(node.string() or "")
    if not name or name in ROS_OPENCV_MODELS:
        return SUPPORTED_DISTORTION_MODEL
    return name


def _from_opencv_storage(path: Path) -> CameraIntrinsics:
    """OpenCV FileStorage(YAML/XML)의 내부 파라미터.

    키: camera_matrix, distortion_coefficients, image_width, image_height, (선택) distortion_model.
    FileStorage가 아닌 XML·깨진 YAML은 OpenCV가 cv2.error 또는 SystemError를 낸다.
    """
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            k_node, d_node = fs.getNode("camera_matrix"), fs.getNode("distortion_coefficients")
            kmat = None if k_node.empty() else k_node.mat()
            dist = None if d_node.empty() else d_node.mat()
            width = int(fs.getNode("image_width").real())
            height = int(fs.getNode("image_height").real())
            model = _storage_model(fs)
        finally:
            fs.release()
    except (cv2.error, SystemError) as exc:
        raise CalibrationError(f"{path}: OpenCV FileStorage 형식이 아닙니다 ({exc})") from exc
    if kmat is None or width <= 0 or height <= 0:
        raise CalibrationError(
            f"{path}: camera_matrix·image_width·image_height가 없습니다 "
            "(이 저장소 형식이면 width·height·fx·fy·cx·cy 키를 씁니다)"
        )
    k = np.asarray(kmat, np.float64)
    coeffs = () if dist is None else tuple(float(d) for d in np.asarray(dist).reshape(-1))
    return _intrinsics(
        path, width=width, height=height, fx=float(k[0, 0]), fy=float(k[1, 1]),
        cx=float(k[0, 2]), cy=float(k[1, 2]), distortion_model=model, distortion=coeffs,
    )  # fmt: skip


def _from_camera_info(data: dict[str, Any], path: Path) -> CameraIntrinsics:
    """ROS camera_info 형식 YAML (camera_matrix.data 9개, distortion_coefficients.data).

    ROS는 미보정 카메라의 camera_info를 K·D 모두 0으로 낸다 (관례). 그런 파일은 값이 아니라 "아직
    재지 않음"이라는 뜻이므로 따로 알린다. distortion_model은 plumb_bob·rational_polynomial(없으면
    plumb_bob으로 본다)만 OpenCV 모델로 받는다. equidistant(어안) 계수는 OpenCV 방사·접선 계수와
    뜻이 달라 받으면 조용히 틀린 모델이 영구 기록된다.
    """
    try:
        k = [float(v) for v in data["camera_matrix"]["data"]]
        dist = [float(v) for v in (data.get("distortion_coefficients") or {}).get("data", [])]
        width, height = int(data["image_width"]), int(data["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{path}: camera_info 형식이 아닙니다 ({exc})") from exc
    if len(k) != 9:
        raise CalibrationError(f"{path}: camera_matrix.data는 9개여야 합니다")
    if k[0] == 0 and k[4] == 0:
        raise CalibrationError(f"{path}: 미보정(uncalibrated) camera_info입니다 (K가 모두 0)")
    model = str(data.get("distortion_model") or "plumb_bob")
    if model not in ROS_OPENCV_MODELS:
        raise CalibrationError(
            f"{path}: 왜곡 모델 {model}은 지원하지 않습니다 (plumb_bob·rational_polynomial만, "
            "어안은 F-PAR-28)"
        )
    return _intrinsics(
        path, width=width, height=height, fx=k[0], fy=k[4], cx=k[2], cy=k[5],
        distortion=tuple(dist),
    )  # fmt: skip


def _parse_text(path: Path, text: str) -> Any:
    """YAML·JSON 본문을 파싱한다 (파싱 오류는 CalibrationError)."""
    try:
        return json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CalibrationError(f"{path}: 파싱 오류 ({exc})") from exc


def _read_any(path: Path) -> CameraIntrinsics:
    """형식을 가려 읽는다 (값 검사 전)."""
    # BOM(U+FEFF)이 붙은 파일도 있다. 벗기지 않으면 `%YAML` 머리 검사를 지나쳐 일반 YAML로
    # 파싱하다 OpenCV 전용 태그(!!opencv-matrix)에서 실패한다
    text = path.read_text("utf-8-sig")
    if text.lstrip().startswith("%YAML") or path.suffix.lower() == ".xml":
        return _from_opencv_storage(path)
    data = _parse_text(path, text)
    if isinstance(data, dict) and "fx" in data:
        return _intrinsics(path, **{str(key): v for key, v in data.items()})
    if isinstance(data, dict) and "camera_matrix" in data:
        return _from_camera_info(data, path)
    raise CalibrationError(f"{path}: 알 수 없는 캘리브레이션 파일 형식")


def load_intrinsics_file(
    path: Path, policy: IntrinsicsCalibrationPolicy | None = None
) -> CameraIntrinsics:
    """캘리브레이션 파일을 읽고 값을 검사한다 (모듈 설명의 세 형식).

    Args:
        path: 파일 경로.
        policy: 주면 타당 범위(화각·주점·fx/fy 비·왜곡 상한)도 본다 (`dlp calib import`는 늘 준다).

    Raises:
        CalibrationError: 어느 형식도 아닐 때, 파싱 오류, 계약에 맞지 않는 값(모르는 키 포함),
            지원하지 않는 왜곡 모델, `check_intrinsics`를 통과하지 못하는 값.
    """
    return check_intrinsics(_read_any(path), policy, str(path))
