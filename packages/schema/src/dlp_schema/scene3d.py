"""청소 전후 3D 비교와 장면 안 바디캠 위치의 계약 타입 (ADR 0066, 0067).

`dlp_scene3d`(파이프라인)와 `dlp_cli.scene3d_cmds`가 주고받는다. 산출물 본문은 라벨링·원본 버킷에
JSON으로 두고, 각 산출물의 존재·버전·위치는 추가 전용 표 `scene3d_records`에 남긴다
(`Scene3dRecord`, Alembic 0027).

단위와 좌표계 (모든 타입 공통):
- 길이는 미터. 단 `Reconstruction.metric`이 False인 복원의 좌표는 SfM 임의 척도다. 척도를 어떻게
  정했는지는 `Reconstruction.scale`(`SceneScale`)에 남는다.
- 카메라는 COLMAP 규약: 카메라 좌표 x 오른쪽, y 아래, z 앞. `x_cam = R · x_world + t`.
- 장면 좌표계는 각 단계(before/after) 복원의 세계 좌표계다. 비교 결과(`ObjectChange`)의 위치와
  이동 벡터, 바디캠 자세(`FramePose`)는 모두 **청소 전(before) 장면 좌표계**로 적는다. 중력 방향은
  모른다(축 의미 없음).
- 픽셀 박스는 EXIF 방향을 적용한 뒤의 이미지 기준 (x, y, w, h), 왼쪽 위 원점.

이름:
- `ScenePhase`: before | after.
- `PhotoEntry`, `PhotoCapture`: 수집한 사진 한 장, 한 단계의 사진 묶음 (원본 위치는 담지 않는다).
- `PhotoBox`, `PhotoPrivacyDetection`, `PhotoPrivacyApproval`, `PhotoRender`: 사진 블러 게이트 기록.
- `CameraPose`, `SceneScale`, `Reconstruction`: 사진별 카메라 자세, 척도 출처, 복원 결과.
- `Alignment`: after 장면 → before 장면 닮음 변환과 잔차.
- `ObjectInstance`, `ObjectChange`, `ChangeReport`: 물체 3D 위치와 전후 변화 (모델 출처, 미검수).
- `FramePose`, `EgoLocalization`: 바디캠 프레임별 카메라 자세 (before 장면 좌표, 모델 출처). 라벨이
  아니다 (`label_records`에 쓰지 않는다).
- `Scene3dRecord`: 산출물 하나의 DB 기록.
"""

from __future__ import annotations

import itertools
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, FiniteFloat, Ms, OntologyId
from dlp_schema.session import CameraIntrinsics

ScenePhase = Literal["before", "after"]
ChangeKind = Literal["moved", "unchanged", "removed", "added"]
# 모델 결과의 검수 상태. 지금은 모두 미검수로 낸다 (검수 연동은 후속, ADR 0066).
ReviewState = Literal["unreviewed"]
# 복원 좌표 척도의 출처 (ADR 0067). marker: 크기를 아는 마커로 정함, sfm: SfM 결과가 이미 미터
# (정답 stub), none: 정하지 못함 (그 복원은 metric이 아니다).
ScaleSource = Literal["marker", "sfm", "none"]

Vec3 = tuple[FiniteFloat, FiniteFloat, FiniteFloat]
Mat3 = tuple[
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
    FiniteFloat,
]


class PhotoEntry(Contract):
    """수집한 사진 한 장. photo_id는 원본 바이트 sha256에서 만든다 (같은 사진은 같은 ID).

    width/height는 EXIF 방향(orientation)을 적용한 뒤의 크기다. orientation은 원본 EXIF 값(1~8,
    없으면 1). 원본 위치·파일 이름은 담지 않는다 (라벨링 쪽 기록에도 쓰이므로).
    """

    photo_id: str = Field(pattern=r"^p[0-9a-f]{16}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ext: Literal[".jpg", ".jpeg", ".png"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    orientation: int = Field(ge=1, le=8)


class PhotoCapture(Contract):
    """한 세션의 한 단계(청소 전 또는 후) 사진 묶음. 사진은 photo_id 순으로 정렬한다."""

    session_id: str
    phase: ScenePhase
    photos: tuple[PhotoEntry, ...] = Field(min_length=1)
    ingested_at: AwareDatetime


class PhotoBox(Contract):
    """사진 한 장의 블러 상자 (픽셀, 여유를 더하고 화면 안으로 자른 값)."""

    target: OntologyId
    x: FiniteFloat
    y: FiniteFloat
    w: FiniteFloat = Field(gt=0)
    h: FiniteFloat = Field(gt=0)
    score: FiniteFloat | None = Field(default=None, ge=0, le=1)  # 사람이 그린 상자는 None


class PhotoPrivacyDetection(Contract):
    """사진 묶음의 자동 블러 탐지 한 세대 (원본 버킷, 덮어쓰지 않는다).

    model_version: 쓴 탐지기 버전과 탐지 정책 해시 (`privacy_gate.detection_version`).
    no_detector: 쓸 수 있는 탐지기가 없던 대상 (사람이 사진 전체를 보고 그려야 한다).
    """

    session_id: str
    phase: ScenePhase
    model_version: str
    boxes: dict[str, tuple[PhotoBox, ...]]
    no_detector: tuple[str, ...]
    created_at: AwareDatetime


class PhotoPrivacyApproval(Contract):
    """원본 접근 권한자의 블러 승인 (원본 버킷, 추가만).

    detection_version이 지금 최신 탐지 세대와 같을 때만 렌더할 수 있다. boxes는 검수자가 확정한
    상자 전부(자동 상자를 지우거나 새로 그린 결과)이고 모든 사진이 키로 있어야 한다.
    """

    approval_id: str
    session_id: str
    phase: ScenePhase
    detection_version: str
    reviewer: str
    boxes: dict[str, tuple[PhotoBox, ...]]
    approved_at: AwareDatetime


class PhotoRender(Contract):
    """승인된 블러 사진 묶음의 렌더 기록 (라벨링 버킷).

    render_tag가 None이면 무효(재탐지로 승인이 낡음)다. 하류 단계는 이 기록이 유효하고 파일 해시가
    맞는 블러본만 읽는다 (`privacy_gate.open_blurred`).
    """

    session_id: str
    phase: ScenePhase
    render_tag: str | None
    approval_id: str | None
    files: dict[str, str]  # photo_id → 블러본 sha256
    rendered_at: AwareDatetime


class CameraPose(Contract):
    """사진 한 장의 카메라 (핀홀). rotation은 행 우선 3x3 R, translation은 t (`x_cam = R x + t`)."""

    photo_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fx: FiniteFloat = Field(gt=0)
    fy: FiniteFloat = Field(gt=0)
    cx: FiniteFloat
    cy: FiniteFloat
    rotation: Mat3
    translation: Vec3


class SceneScale(Contract):
    """복원 좌표의 척도를 어떻게 정했는가 (ADR 0067).

    source: `ScaleSource`. marker(크기를 아는 마커로 정함), sfm(SfM 결과가 이미 미터: 정답 stub),
      none(정하지 못함: 그 복원은 metric이 아니다).
    factor: SfM 좌표(카메라 이동·희소 점)에 곱한 배율. marker일 때만 1이 아니다.
    marker_side_m: 정책이 준 마커 한 변의 실제 길이(m). 마커 척도를 꺼 두었으면 None.
    marker_factor: 마커가 말하는 배율 = marker_side_m / 복원 단위 변 길이(중앙값). 마커 추정이
      안정적일 때만 있다. source가 marker면 factor와 같고, sfm이면 확인값이다 (1에 가까워야 한다).
    marker_ids: 척도 계산·확인에 쓴 마커 ID (오름차순). marker_photos: 그 마커들을 삼각측량하는 데
      쓴 사진 수 (자세를 찾았고 모서리 재투영이 맞은 사진, 중복 없이).
    side_spread: 복원 단위 변 길이 추정들의 상대 흩어짐 ((최대 - 최소) / 중앙값).
    reason: 마커로 척도를 정하지 못한 이유 (마커 없음, 사진 수 미달, 흩어짐 초과 등). 정했으면 None.
    """

    source: ScaleSource
    factor: FiniteFloat = Field(gt=0)
    marker_side_m: FiniteFloat | None = Field(default=None, gt=0)
    marker_factor: FiniteFloat | None = Field(default=None, gt=0)
    marker_ids: tuple[int, ...] = ()
    marker_photos: int = Field(default=0, ge=0)
    side_spread: FiniteFloat | None = Field(default=None, ge=0)
    reason: str | None = None

    @model_validator(mode="after")
    def _check_factor(self) -> SceneScale:
        """마커로 정했으면 곱한 배율이 마커 배율이고, 아니면 좌표를 건드리지 않았어야 한다."""
        if self.source == "marker" and self.marker_factor != self.factor:
            raise ValueError("source가 marker면 factor는 marker_factor와 같아야 합니다")
        if self.source != "marker" and self.factor != 1.0:
            raise ValueError("마커로 정한 척도가 아니면 factor는 1이어야 합니다")
        return self


class Reconstruction(Contract):
    """한 단계의 복원 결과 (라벨링 버킷 `recon/<tag>/`).

    metric: 좌표가 미터인가. SfM만으로는 척도를 모르므로 실제 경로는 척도 기준(크기를 아는 마커)이
      있어야 True다. 비교 단계는 정책 `change.require_metric`이면 before가 metric이어야 한다.
    registered: 자세를 찾은 사진 수 / 전체 사진 수 (`photo_count`).
    splat_key·points_key·cameras_key: 같은 버킷의 산출물 키.
    scale: 척도 출처 (ADR 0067). 그 전에 만든 기록에는 없다 (None).
    """

    session_id: str
    phase: ScenePhase
    model_version: str
    render_tag: str
    metric: bool
    photo_count: int = Field(gt=0)
    registered: int = Field(ge=0)
    point_count: int = Field(ge=0)
    cameras_key: str
    points_key: str
    splat_key: str
    created_at: AwareDatetime
    scale: SceneScale | None = None

    @model_validator(mode="after")
    def _check_scale(self) -> Reconstruction:
        """척도 출처가 있으면 metric과 맞아야 한다 (출처 없이 미터라고 적지 않는다)."""
        if self.scale is not None and (self.scale.source == "none") == self.metric:
            raise ValueError("metric은 척도 출처(scale.source)가 marker·sfm일 때만 True입니다")
        return self


class Alignment(Contract):
    """after 장면 좌표 → before 장면 좌표 닮음 변환 `x_b = scale · R · x_a + t`.

    rms_m·inlier_threshold_m는 before 좌표 단위(metric이면 미터). accepted가 False면 비교하지
    않는다 (reason에 이유).
    """

    model_version: str
    scale: FiniteFloat = Field(gt=0)
    rotation: Mat3
    translation: Vec3
    matches: int = Field(ge=0)
    inliers: int = Field(ge=0)
    rms_m: FiniteFloat | None
    accepted: bool
    reason: str | None = None


class ObjectInstance(Contract):
    """여러 사진의 탐지로 삼각측량한 물체 하나 (그 단계 장면 좌표).

    size: 상자 대각선 크기 추정 (장면 단위). appearance: 색 히스토그램 (합 1).
    views: 받쳐 준 사진 수. reproj_px: 받친 사진들의 재투영 오차 RMS.
    """

    instance_id: str
    phase: ScenePhase
    label: OntologyId
    position: Vec3
    size: FiniteFloat = Field(gt=0)
    appearance: tuple[FiniteFloat, ...]
    views: int = Field(ge=2)
    photo_ids: tuple[str, ...]
    reproj_px: FiniteFloat = Field(ge=0)
    confidence: FiniteFloat = Field(ge=0, le=1)


class ObjectChange(Contract):
    """물체 하나의 전후 변화 (모델 출처, 미검수).

    위치·이동 벡터는 before 장면 좌표(metric이면 미터). moved·unchanged는 before·after 둘 다,
    removed는 before만, added는 after만 있다. displacement = after(before 좌표로 옮긴 값) - before.
    """

    change_id: str
    kind: ChangeKind
    label: OntologyId
    before_instance_id: str | None
    after_instance_id: str | None
    before_position: Vec3 | None
    after_position: Vec3 | None
    displacement: Vec3 | None
    distance_m: FiniteFloat | None = Field(default=None, ge=0)
    match_cost: FiniteFloat | None = None
    confidence: FiniteFloat = Field(ge=0, le=1)
    source: Literal["model"] = "model"
    review_state: ReviewState = "unreviewed"


class ChangeReport(Contract):
    """세션 하나의 전후 비교 결과 (라벨링 버킷 `changes/<tag>.json`, 덮어쓰지 않는다)."""

    session_id: str
    model_version: str
    before_reconstruction: str
    after_reconstruction: str
    alignment: Alignment
    metric: bool
    instances: tuple[ObjectInstance, ...]
    changes: tuple[ObjectChange, ...]
    created_at: AwareDatetime


class FramePose(Contract):
    """바디캠 프레임 한 장의 카메라 자세 (before 장면 좌표, ADR 0067).

    t_ms: 그 프레임의 바디캠 PTS 시각 (정수 ms, PTS 인덱스에서 얻는다). 바디캠이 기준 스트림이라
      마스터 타임라인 시각과 같다.
    reference: 자세를 푼 복원 (before | after). after면 정렬의 닮음 변환으로 before 좌표의 강체
      자세로 옮긴 값이다.
    rotation·translation: `x_cam = R · x_before + t` (`CameraPose`와 같은 규약, 행 우선 3x3 R).
      translation의 단위는 before 장면 단위다 (`EgoLocalization.metric`이면 미터). 카메라 중심은
      `-Rᵀ t`.
    matches: 그 복원과 맺은 2D-3D 대응 수. inliers: 그중 PnP 내점 수.
    reproj_px: 내점 재투영 오차 RMS (픽셀).
    """

    t_ms: Ms
    reference: ScenePhase
    rotation: Mat3
    translation: Vec3
    matches: int = Field(ge=4)
    inliers: int = Field(ge=4)
    reproj_px: FiniteFloat = Field(ge=0)


class EgoLocalization(Contract):
    """세션 바디캠 영상의 프레임별 카메라 자세 (라벨링 버킷 `localization/<tag>.json`, ADR 0067).

    한 번 쓰면 덮어쓰지 않는다. 모델 출처이고 라벨이 아니다. 입력은 그 세션의 지금 승인된 블러본
    바디캠 영상과 전후 복원이다.
    stream_id: 바디캠 스트림. model_version: 특징 추출기 버전과 위치·정렬 정책 절 해시.
    before_reconstruction·after_reconstruction: 쓴 복원의 태그.
    alignment_tag·alignment: 쓴 after → before 정렬의 태그와 내용 (받아들인 것만).
    render_hash: 쓴 블러본 영상의 렌더 해시 (`dlp_privacy.rendering.render_hash`).
      blurred_sha256: 그 블러본 파일 해시 (이전 형식 렌더 기록이면 None).
    intrinsics: 자세를 풀 때 쓴 바디캠 내부 파라미터 (영상 프레임 크기에 맞춘 값, 왜곡 계수 포함).
      intrinsics_origin: 그 출처 (`session` 또는 `registry:<calibration_id>`).
    metric: 자세의 길이 단위가 미터인가 (before 복원의 metric).
    frame_count: 자세를 풀어 본 프레임 수. localized: 문턱을 넘어 자세를 남긴 프레임 수
      (= poses 수).
    poses: 시각 순 프레임 자세. 문턱을 못 넘은 프레임은 없다 (자세를 지어내지 않는다).
    """

    session_id: str
    stream_id: str
    model_version: str
    before_reconstruction: str
    after_reconstruction: str
    alignment_tag: str
    alignment: Alignment
    render_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    blurred_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    intrinsics: CameraIntrinsics
    intrinsics_origin: str
    metric: bool
    frame_count: int = Field(ge=0)
    localized: int = Field(ge=0)
    poses: tuple[FramePose, ...]
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check_poses(self) -> EgoLocalization:
        """프레임 수가 맞고 자세가 시각 순(중복 없음)이며 쓴 정렬이 받아들인 것인지 본다."""
        if self.localized != len(self.poses) or self.localized > self.frame_count:
            raise ValueError("localized는 poses 수와 같고 frame_count 이하여야 합니다")
        times = [p.t_ms for p in self.poses]
        if any(b <= a for a, b in itertools.pairwise(times)):
            raise ValueError("poses는 시각 순이고 같은 시각이 없어야 합니다")
        if not self.alignment.accepted:
            raise ValueError("거부된 정렬로 만든 위치 기록은 둘 수 없습니다")
        return self


Scene3dRecordKind = Literal[
    "capture",
    "privacy_detection",
    "privacy_approval",
    "render",
    "reconstruction",
    "alignment",
    "change_report",
    "localization",
]


class Scene3dRecord(Contract):
    """scene3d 산출물 하나의 기록 (추가 전용 표 `scene3d_records`, Alembic 0027).

    본문은 bucket·storage_key의 JSON이고 이 기록은 무엇을 어떤 버전으로 만들었는지만 남긴다.
    phase는 단계별 산출물에만 있다 (alignment·change_report·localization은 None). kind 열은 문자열
    (VARCHAR(32), 값 제약 없음)이라 종류를 더해도 마이그레이션이 없다. bucket은 `raw`(원본 버킷:
    수집·블러 탐지·승인) 또는 `labeling`. 원본 버킷 키는 원본 접근 권한자만 보는 이 표에만 둔다.
    actor는 사람 승인(privacy_approval)의 검수자, 그 밖은 실행자.
    """

    session_id: str
    phase: ScenePhase | None
    kind: Scene3dRecordKind
    model_version: str | None
    tag: str
    bucket: Literal["raw", "labeling"]
    storage_key: str
    actor: str
    created_at: AwareDatetime
