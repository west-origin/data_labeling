"""장면 안 바디캠 위치 추정과 위치 기록 읽기 (`dlp scene3d localize`, ADR 0067).

바디캠 영상의 프레임마다 카메라 자세를 **before 장면 좌표계**로 구한다 (`x_cam = R x_before + t`).
손 3D·촉각 같은 세션 자료를 전후 사진으로 만든 공간 안에 놓는 기반이다.

전제 (하나라도 없으면 무엇이 빠졌는지 알리고 실패한다. 한쪽 복원만으로 진행하지 않는다):
- before·after 복원이 둘 다 지금 것이고(`pipeline.current_reconstruction`), 둘의 정렬이
  받아들여졌다.
- 세션이 `sessions` 표에 있고 사용 중지되지 않았다.
- 바디캠 내부 파라미터가 등록돼 있다 (세션 매니페스트 또는 `dlp calib`의 기기 등록,
  `dlp_schema.db.calibration_repository.session_intrinsics`). 없으면 화각을 지어내지 않는다.
- 바디캠의 지금 승인된 블러본이 있다 (`ego_video.open_blurred_video`, 원본 버킷을 열지 않는다).

방법 (`solve_frame`): 프레임 특징의 기술자를 복원 희소 점의 기술자와 대응시키고(상호 최근접 +
비율 검사, `geometry.DescriptorMatcher`), 그 2D-3D 대응으로 PnP RANSAC을 푼다(`pnp.solve_pnp`).
작업 중에는 장면이 바뀌므로 프레임마다 before 복원과 after 복원 둘 다에 풀어, 문턱을 넘은 것 중
내점이 많은 쪽을 쓴다. after 기준 자세는 정렬의 닮음 변환으로 before 좌표의 강체 자세로 옮긴다
(`geometry.rigid_through_similarity`). 문턱(최소 내점, 재투영 오차, 내점이 퍼진 정도)을 못 넘은
프레임은 기록하지 않는다 (자세를 지어내지 않는다). 특징 추출만 백엔드다
(`backends.FrameFeatureExtractor`).

결과 `EgoLocalization`은 라벨이 아니다. 라벨링 버킷 `…/scene3d/localization/<태그>.json`에 한 번만
쓰고 `localization/current.json` 표시만 옮긴다. 태그 = (특징 추출기 버전, `localization`·
`alignment` 절 해시, 두 복원 태그, 스트림, 블러본 렌더 해시, 내부 파라미터)의 해시라 같은 입력이면
다시 계산하지 않는다.

읽기: `current_localization`(지금 복원과 맞는 기록만), `PoseTrack.at`·`pose_at`(이웃한 두 프레임
사이 보간: 카메라 중심은 선형, 회전은 구면. 이웃 간격이 한계를 넘거나 기록 밖 시각이면 None).
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlalchemy as sa
from sqlalchemy.exc import NoResultFound

from dlp_media.calibration import CalibrationError, check_intrinsics
from dlp_media.probe import probe
from dlp_media.storage import ObjectStore
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.rendering import expected_render_hash
from dlp_scene3d.backends import FrameFeatureExtractor, FrameFeatures, FrameImage
from dlp_scene3d.ego_video import (
    BlurredVideo,
    check_blurred_video,
    open_blurred_video,
    read_frames,
    sample_frames,
)
from dlp_scene3d.geometry import (
    F64,
    DescriptorMatcher,
    Similarity,
    interpolate_pose,
    rigid_through_similarity,
)
from dlp_scene3d.layout import (
    get_model,
    get_pointer,
    put_model_immutable,
    put_pointer,
    session_prefix,
    short_hash,
)
from dlp_scene3d.pipeline import (
    AlignmentRejectedError,
    Scene3dError,
    SceneAlignment,
    alignment_version,
    current_reconstruction,
    current_reconstruction_record,
    ensure_alignment,
)
from dlp_scene3d.pnp import MIN_POINTS, PnpSolution, solve_pnp
from dlp_scene3d.policy import LocalizationPolicy, Scene3dPolicy, section_digest
from dlp_schema.calibration import ResolvedIntrinsics
from dlp_schema.db.calibration_repository import session_intrinsics
from dlp_schema.db.repository import get_session, withdrawn_session_ids
from dlp_schema.scene3d import EgoLocalization, FramePose, Reconstruction, ScenePhase
from dlp_schema.session import CameraIntrinsics, LifecycleState, Session

LOCALIZE_VERSION = "scene3d-localize-1"
# 특징 추출기에 한 번에 넘기는 프레임 수 (메모리·임시 파일 상한. 결과와 무관하다)
_FRAME_BATCH = 16
# 블러본이 홀수 변에 덧댄 검은 줄 (화소, `dlp_privacy.render.render_blurred`)
_EVEN_PAD = 1


@dataclass(frozen=True)
class SceneModel:
    """프레임을 대응시킬 복원 하나: 단계, 희소 점 (N, 3), 그 기술자의 대응기."""

    phase: ScenePhase
    points: F64
    matcher: DescriptorMatcher


@dataclass(frozen=True)
class EgoCamera:
    """프레임 크기에 맞춘 바디캠 카메라: 3x3 내부 파라미터, OpenCV 왜곡 계수, 프레임 크기(화소)."""

    k: F64
    dist: F64
    width: int
    height: int

    @classmethod
    def of(cls, intrinsics: CameraIntrinsics) -> EgoCamera:
        """계약 `CameraIntrinsics`에서 만든다."""
        k = np.array(
            [[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]],
            dtype=np.float64,
        )
        dist = np.asarray(intrinsics.distortion, dtype=np.float64)
        return cls(k, dist, intrinsics.width, intrinsics.height)


def fit_intrinsics(
    calib: CameraIntrinsics, width: int, height: int, max_aspect_mismatch: float
) -> CameraIntrinsics:
    """등록된 내부 파라미터를 실제 프레임 크기에 맞춘다.

    - 크기가 같거나 블러본이 홀수 변에 검은 줄 하나를 덧댄 것이면(가로·세로가 0~1 화소 크다) 값은
      그대로다 (덧댄 줄은 화소 좌표를 옮기지 않는다).
    - 화면비가 `max_aspect_mismatch` 안에서 같으면 축소·확대로 보고 fx·cx는 가로 비, fy·cy는
      세로 비를 곱한다 (프리라벨 3D 단계 `dlp_prelabel.lift3d.intrinsics_for`와 같은 식). 왜곡
      계수는 정규화 좌표의 값이라 그대로다.

    Raises:
        Scene3dError: 화면비가 달라 그 캘리브레이션을 이 영상에 쓸 수 없을 때 (화각을 지어내지
            않는다).
    """
    if 0 <= width - calib.width <= _EVEN_PAD and 0 <= height - calib.height <= _EVEN_PAD:
        return calib.model_copy(update={"width": width, "height": height})
    ratio = (width / height) / (calib.width / calib.height)
    if abs(ratio - 1.0) > max_aspect_mismatch:
        raise Scene3dError(
            f"바디캠 캘리브레이션({calib.width}x{calib.height})과 영상({width}x{height})의 "
            "화면비가 다릅니다. 이 촬영 모드로 캘리브레이션을 등록하세요 (dlp calib intrinsics)"
        )
    sx, sy = width / calib.width, height / calib.height
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=calib.fx * sx,
        fy=calib.fy * sy,
        cx=calib.cx * sx,
        cy=calib.cy * sy,
        distortion_model=calib.distortion_model,
        distortion=calib.distortion,
    )


def _spread(pixels: F64, camera: EgoCamera) -> float:
    """화소들을 둘러싼 상자의 대각선 / 화면 대각선."""
    extent = pixels.max(axis=0) - pixels.min(axis=0)
    return float(np.hypot(extent[0], extent[1]) / np.hypot(camera.width, camera.height))


def _solve_against(
    features: FrameFeatures, scene: SceneModel, camera: EgoCamera, policy: LocalizationPolicy
) -> PnpSolution | None:
    """프레임 하나를 복원 하나에 푼다 (그 복원의 좌표). 문턱을 못 넘으면 None."""
    pairs = scene.matcher.match(features.descriptors, policy.descriptor_ratio)
    if len(pairs) < max(policy.min_inliers, MIN_POINTS):
        return None
    pixels = features.keypoints[pairs[:, 0]]
    sol = solve_pnp(
        scene.points[pairs[:, 1]],
        pixels,
        camera.k,
        camera.dist,
        threshold_px=policy.ransac_reproj_ratio * camera.width,
        iterations=policy.ransac_iterations,
        confidence=policy.ransac_confidence,
    )
    if (
        sol is None
        or len(sol.inliers) < policy.min_inliers
        or sol.reproj_px > policy.max_reproj_ratio * camera.width
        or _spread(pixels[sol.inliers], camera) < policy.min_inlier_spread
    ):
        return None
    return sol


def solve_frame(
    features: FrameFeatures,
    scenes: Sequence[SceneModel],
    after_to_before: Similarity,
    camera: EgoCamera,
    policy: LocalizationPolicy,
) -> tuple[ScenePhase, PnpSolution] | None:
    """프레임 하나의 자세: (기준 복원, before 좌표의 PnP 결과). 어느 복원으로도 못 풀면 None.

    scenes의 각 복원에 풀어 문턱을 넘은 것 중 내점이 가장 많은 것을 고른다 (같으면 앞에 온 복원).
    after 복원으로 푼 자세는 `after_to_before`(`x_b = s R x_a + t`)로 before 좌표의 강체 자세로
    옮긴다.
    """
    best: tuple[ScenePhase, PnpSolution] | None = None
    for scene in scenes:
        sol = _solve_against(features, scene, camera, policy)
        if sol is not None and (best is None or len(sol.inliers) > len(best[1].inliers)):
            best = (scene.phase, sol)
    if best is None or best[0] == "before":
        return best
    r, t = rigid_through_similarity(best[1].r, best[1].t, after_to_before)
    return best[0], PnpSolution(r, t, best[1].inliers, best[1].reproj_px, best[1].matches)


def _frame_pose(t_ms: int, phase: ScenePhase, sol: PnpSolution) -> FramePose:
    """PnP 결과 → 계약."""
    r = [float(v) for v in sol.r.reshape(-1)]
    t = [float(v) for v in sol.t]
    return FramePose(
        t_ms=t_ms,
        reference=phase,
        rotation=(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]),
        translation=(t[0], t[1], t[2]),
        matches=sol.matches,
        inliers=len(sol.inliers),
        reproj_px=sol.reproj_px,
    )


def localize_frames(
    features: Mapping[int, FrameFeatures],
    scenes: Sequence[SceneModel],
    after_to_before: Similarity,
    camera: EgoCamera,
    policy: LocalizationPolicy,
) -> list[FramePose]:
    """프레임 시각 → 특징 묶음에서 자세를 푼다 (시각 순). 문턱을 못 넘은 프레임은 빠진다."""
    poses: list[FramePose] = []
    for t_ms in sorted(features):
        found = solve_frame(features[t_ms], scenes, after_to_before, camera, policy)
        if found is not None:
            poses.append(_frame_pose(t_ms, found[0], found[1]))
    return poses


def localization_version(extractor: FrameFeatureExtractor, policy: Scene3dPolicy) -> str:
    """위치 추정의 모델 버전: 판, 특징 추출기 버전, `localization` 절 해시, 정렬 버전."""
    digest = section_digest(policy.localization)
    return f"{LOCALIZE_VERSION}+feat:{extractor.version}+p{digest}+{alignment_version(policy)}"


def localization_key(sid: str, tag: str) -> str:
    """라벨링 버킷의 위치 기록 키."""
    return f"{session_prefix(sid)}/localization/{tag}.json"


def _pointer_key(sid: str) -> str:
    return f"{session_prefix(sid)}/localization/current.json"


@dataclass(frozen=True)
class _Scenes:
    """전후 복원과 정렬 (위치 추정의 장면 쪽 입력)."""

    before_tag: str
    after_tag: str
    before: Reconstruction
    models: tuple[SceneModel, SceneModel]
    alignment: SceneAlignment
    transform: Similarity


def _load_scenes(labeling: ObjectStore, sid: str, policy: Scene3dPolicy) -> _Scenes:
    """전후 복원을 읽고 정렬을 얻는다.

    Raises:
        Scene3dError: 복원이 하나라도 없거나 낡았을 때 (빠진 것을 모두 적는다).
        AlignmentRejectedError: 정렬이 정책 문턱을 못 넘었을 때.
    """
    loaded: dict[ScenePhase, tuple[str, Reconstruction, F64, F64]] = {}
    missing: list[str] = []
    phases: tuple[ScenePhase, ScenePhase] = ("before", "after")
    for phase in phases:
        try:
            tag, rec, _, points, desc = current_reconstruction(labeling, sid, phase)
        except Scene3dError as exc:
            missing.append(str(exc))
            continue
        loaded[phase] = (tag, rec, points, desc)
    if missing:
        raise Scene3dError("위치 추정에는 전후 복원이 모두 필요합니다: " + "; ".join(missing))
    (tb, rb, pts_b, desc_b), (ta, _, pts_a, desc_a) = loaded["before"], loaded["after"]
    aligned = ensure_alignment(labeling, sid, policy, (tb, pts_b, desc_b), (ta, pts_a, desc_a))
    if aligned.transform is None:
        raise AlignmentRejectedError(aligned.record)
    models = (
        SceneModel("before", pts_b, DescriptorMatcher.build(desc_b)),
        SceneModel("after", pts_a, DescriptorMatcher.build(desc_a)),
    )
    return _Scenes(tb, ta, rb, models, aligned, aligned.transform)


def _usable_session(conn: sa.Connection, sid: str) -> Session:
    """세션을 읽는다. 없거나 사용 중지됐으면 Scene3dError."""
    try:
        session = get_session(conn, sid)
    except NoResultFound as exc:
        raise Scene3dError(
            f"{sid}: 세션이 등록돼 있지 않습니다 "
            "(바디캠 영상을 dlp ingest로 수집한 세션이어야 합니다)"
        ) from exc
    if session.lifecycle_state is LifecycleState.WITHDRAWN or sid in withdrawn_session_ids(conn):
        raise Scene3dError(f"{sid}: 사용 중지된 세션입니다")
    return session


def _bodycam(engine: sa.Engine, sid: str) -> tuple[str, ResolvedIntrinsics, CameraIntrinsics]:
    """(바디캠 스트림 ID, 내부 파라미터 출처, 등록된 내부 파라미터).

    Raises:
        Scene3dError: 세션이 없거나 사용 중지됐을 때, 등록된 내부 파라미터가 없거나 쓸 수 없는
            값일 때.
    """
    with engine.connect() as conn:
        session = _usable_session(conn, sid)
        resolved = session_intrinsics(conn, session)
    if resolved.intrinsics is None:
        raise Scene3dError(
            f"{sid}: 바디캠 내부 파라미터가 등록돼 있지 않습니다. 화각을 지어내지 않습니다 "
            "(dlp calib intrinsics|import로 기기를 등록하고, 세션에 기기가 없으면 dlp calib assign)"
        )
    try:
        check_intrinsics(resolved.intrinsics, None, f"{sid} 바디캠 내부 파라미터")
    except CalibrationError as exc:
        raise Scene3dError(str(exc)) from exc
    return session.reference_stream.stream_id, resolved, resolved.intrinsics


def _frame_size(video: BlurredVideo) -> tuple[int, int]:
    """블러본의 프레임 크기 (가로, 세로)."""
    info = probe(video.path).video
    if info is None:
        raise Scene3dError(f"{video.session_id}/{video.stream_id}: 블러본에 영상 트랙이 없습니다")
    return info.width, info.height


def _solve_video(
    video: BlurredVideo,
    extractor: FrameFeatureExtractor,
    scenes: _Scenes,
    camera: EgoCamera,
    policy: LocalizationPolicy,
    workdir: Path,
) -> tuple[list[FramePose], int]:
    """(프레임 자세, 풀어 본 프레임 수). 프레임을 묶음으로 꺼내 특징을 뽑고 자세를 푼다."""
    indices = sample_frames(video.pts, policy.frame_interval_ms)
    poses: list[FramePose] = []
    for batch in itertools.batched(read_frames(video, indices), _FRAME_BATCH):
        frames: Sequence[FrameImage] = batch
        features = extractor.extract(frames, workdir / "features")
        wanted = {f.t_ms: features[f.t_ms] for f in frames if f.t_ms in features}
        poses += localize_frames(wanted, scenes.models, scenes.transform, camera, policy)
    return poses, len(indices)


def _check_before_write(
    engine: sa.Engine, sid: str, stream_id: str, render: str, privacy_policy: PrivacyPolicy
) -> None:
    """쓰기 직전 다시 본다: 계산하는 사이 세션이 사용 중지됐거나 블러가 바뀌었으면 쓰지 않는다."""
    with engine.connect() as conn:
        _usable_session(conn, sid)
        if expected_render_hash(conn, sid, stream_id, privacy_policy) != render:
            raise RenderNotCurrentError(
                f"{sid}/{stream_id}: 위치를 추정하는 사이 블러본 렌더가 바뀌었습니다. "
                "다시 실행하세요"
            )


def localize(
    engine: sa.Engine,
    labeling: ObjectStore,
    sid: str,
    extractor: FrameFeatureExtractor,
    policy: Scene3dPolicy,
    privacy_policy: PrivacyPolicy,
    workdir: Path,
    now: datetime,
    *,
    raw_bucket: str,
) -> tuple[EgoLocalization, bool]:
    """(위치 기록, 새로 계산했는가). 세션 바디캠의 프레임별 자세를 before 장면 좌표로 구한다.

    Args:
        engine: DB 엔진 (세션·캘리브레이션·블러 승인 상태를 짧은 읽기 트랜잭션으로 본다. 쓰지
            않는다).
        labeling: 라벨링 버킷 저장소 (복원·블러본을 읽고 위치 기록을 쓴다).
        sid: 세션 ID.
        extractor: 프레임 특징 추출기.
        policy: scene3d 정책 (`localization`·`alignment` 절).
        privacy_policy: 프라이버시 정책 (바디캠 블러본 렌더 해시 계산).
        workdir: 임시 작업 디렉터리 (블러본과 프레임을 둔다, 호출자가 지운다).
        now: 기록 시각.
        raw_bucket: 원본 버킷 이름 (이 이름의 저장소를 `labeling`으로 받으면 거부한다).

    Raises:
        Scene3dError: 전후 복원 중 빠진 것이 있거나 낡았을 때, 세션이 없거나 사용 중지됐을 때,
            바디캠 내부 파라미터가 없거나 영상과 맞지 않을 때.
        AlignmentRejectedError: 정렬이 거부됐을 때.
        RenderNotCurrentError: 바디캠의 지금 승인된 블러본이 없거나 계산하는 사이 바뀌었을 때.
    """
    scenes = _load_scenes(labeling, sid, policy)
    stream_id, resolved, calib = _bodycam(engine, sid)
    render, _ = check_blurred_video(
        engine, labeling, sid, stream_id, privacy_policy, raw_bucket=raw_bucket
    )
    version = localization_version(extractor, policy)
    tag = short_hash(
        version, scenes.before_tag, scenes.after_tag, stream_id, render, resolved.digest_input()
    )
    key = localization_key(sid, tag)
    existing = get_model(labeling, key, EgoLocalization)
    if existing is not None:
        put_pointer(labeling, _pointer_key(sid), {"tag": tag})
        return existing, False
    video = open_blurred_video(
        engine, labeling, sid, stream_id, privacy_policy, workdir / "video", raw_bucket=raw_bucket
    )
    if video.render_hash != render:
        raise RenderNotCurrentError(
            f"{sid}/{stream_id}: 블러본 렌더가 바뀌었습니다. 다시 실행하세요"
        )
    width, height = _frame_size(video)
    lp = policy.localization
    intrinsics = fit_intrinsics(calib, width, height, lp.max_intrinsics_aspect_mismatch)
    poses, tried = _solve_video(video, extractor, scenes, EgoCamera.of(intrinsics), lp, workdir)
    _check_before_write(engine, sid, stream_id, render, privacy_policy)
    record = EgoLocalization(
        session_id=sid,
        stream_id=stream_id,
        model_version=version,
        before_reconstruction=scenes.before_tag,
        after_reconstruction=scenes.after_tag,
        alignment_tag=scenes.alignment.tag,
        alignment=scenes.alignment.record,
        render_hash=render,
        blurred_sha256=video.blurred_sha256,
        intrinsics=intrinsics,
        intrinsics_origin=resolved.origin,
        metric=scenes.before.metric,
        frame_count=tried,
        localized=len(poses),
        poses=tuple(poses),
        created_at=now,
    )
    put_model_immutable(labeling, key, record)
    put_pointer(labeling, _pointer_key(sid), {"tag": tag})
    return record, True


def stored_localization(labeling: ObjectStore, sid: str) -> tuple[str, EgoLocalization] | None:
    """현재 표시가 가리키는 (태그, 위치 기록). 없으면 None. 지금 복원과 맞는지는 보지 않는다."""
    ptr = get_pointer(labeling, _pointer_key(sid))
    if not ptr:
        return None
    record = get_model(labeling, localization_key(sid, ptr["tag"]), EgoLocalization)
    return None if record is None else (ptr["tag"], record)


def current_localization(labeling: ObjectStore, sid: str) -> EgoLocalization:
    """지금 위치 기록 (자세는 before 장면 좌표).

    기록이 지금 복원으로 만든 것일 때만 돌려준다. before 복원이 바뀌면 장면 좌표계 자체가
    달라지므로 예전 기록의 자세를 쓰면 안 된다. 바디캠 블러본이 그 뒤 바뀌었는지는 여기서 보지
    않는다 (DB가 필요하다): 필요하면 기록의 `render_hash`를
    `dlp_privacy.rendering.expected_render_hash`와 견준다.

    Raises:
        Scene3dError: 위치 기록이 없을 때, 복원이 없거나 낡았을 때, 기록이 지금 복원과 다를 때.
    """
    found = stored_localization(labeling, sid)
    if found is None:
        raise Scene3dError(f"{sid}: 위치 기록이 없습니다 (dlp scene3d localize)")
    record = found[1]
    tb, _ = current_reconstruction_record(labeling, sid, "before")
    ta, _ = current_reconstruction_record(labeling, sid, "after")
    if (record.before_reconstruction, record.after_reconstruction) != (tb, ta):
        raise Scene3dError(
            f"{sid}: 위치 기록이 지금 복원과 다릅니다. 다시 추정하세요 (dlp scene3d localize)"
        )
    return record


@dataclass(frozen=True)
class EgoPose:
    """한 시각의 바디캠 자세 (before 장면 좌표, `x_cam = R x + t`).

    t_ms는 바디캠 PTS 시각이다 (바디캠이 기준 스트림이라 마스터 타임라인 시각과 같다).
    """

    t_ms: float
    r: F64
    t: F64

    @property
    def center(self) -> F64:
        """before 장면 좌표의 카메라 중심 `-Rᵀ t`."""
        return -self.r.T @ self.t


@dataclass(frozen=True)
class PoseTrack:
    """위치 기록의 자세를 시각으로 찾는 표 (여러 시각을 물을 때 한 번만 만든다)."""

    times: tuple[int, ...]
    rotations: F64
    translations: F64

    @classmethod
    def of(cls, record: EgoLocalization) -> PoseTrack:
        """위치 기록에서 만든다 (자세는 시각 순이다, 계약 검증)."""
        n = len(record.poses)
        rot = np.asarray([p.rotation for p in record.poses], dtype=np.float64).reshape(n, 3, 3)
        trans = np.asarray([p.translation for p in record.poses], dtype=np.float64).reshape(n, 3)
        return cls(tuple(p.t_ms for p in record.poses), rot, trans)

    def at(self, t_ms: float, max_gap_ms: int) -> EgoPose | None:
        """시각 t_ms의 자세. 기록된 프레임 시각이면 그 자세, 아니면 이웃한 두 프레임 사이 보간.

        카메라 중심은 선형, 회전은 구면으로 보간한다 (`geometry.interpolate_pose`). 다음이면
        None이다 (자세를 지어내지 않는다): 첫 자세보다 앞이거나 마지막 자세보다 뒤(밖으로 늘리지
        않는다), 이웃한 두 자세의 간격이 `max_gap_ms`보다 큼 (그 사이 프레임을 풀지 못했다).

        Args:
            t_ms: 바디캠 PTS 시각 (ms, 실수도 된다).
            max_gap_ms: 보간을 허용하는 이웃 간격 상한 (`scene3d.yaml interpolation.max_gap_ms`).
        """
        i = bisect.bisect_left(self.times, t_ms)
        if i < len(self.times) and self.times[i] == t_ms:
            return EgoPose(float(t_ms), self.rotations[i], self.translations[i])
        if i == 0 or i == len(self.times):
            return None
        t0, t1 = self.times[i - 1], self.times[i]
        if t1 - t0 > max_gap_ms:
            return None
        r, t = interpolate_pose(
            self.rotations[i - 1],
            self.translations[i - 1],
            self.rotations[i],
            self.translations[i],
            (t_ms - t0) / (t1 - t0),
        )
        return EgoPose(float(t_ms), r, t)


def pose_at(record: EgoLocalization, t_ms: float, max_gap_ms: int) -> EgoPose | None:
    """위치 기록에서 시각 t_ms의 자세 (`PoseTrack.at`과 같다).

    여러 시각을 물을 때는 `PoseTrack.of`를 한 번 만들어 쓴다.
    """
    return PoseTrack.of(record).at(t_ms, max_gap_ms)
