"""청소 전후 3D 비교의 계약 타입 (ADR 0066).

`dlp_scene3d`(파이프라인)와 `dlp_cli.scene3d_cmds`가 주고받는다. 산출물 본문은 라벨링·원본 버킷에
JSON으로 두고, 각 산출물의 존재·버전·위치는 추가 전용 표 `scene3d_records`에 남긴다
(`Scene3dRecord`, Alembic 0027).

단위와 좌표계 (모든 타입 공통):
- 길이는 미터. 단 `Reconstruction.metric`이 False인 복원의 좌표는 SfM 임의 척도다.
- 카메라는 COLMAP 규약: 카메라 좌표 x 오른쪽, y 아래, z 앞. `x_cam = R · x_world + t`.
- 장면 좌표계는 각 단계(before/after) 복원의 세계 좌표계다. 비교 결과(`ObjectChange`)의 위치와
  이동 벡터는 모두 **청소 전(before) 장면 좌표계**로 적는다. 중력 방향은 모른다(축 의미 없음).
- 픽셀 박스는 EXIF 방향을 적용한 뒤의 이미지 기준 (x, y, w, h), 왼쪽 위 원점.

이름:
- `ScenePhase`: before | after.
- `PhotoEntry`, `PhotoCapture`: 수집한 사진 한 장, 한 단계의 사진 묶음 (원본 위치는 담지 않는다).
- `PhotoBox`, `PhotoPrivacyDetection`, `PhotoPrivacyApproval`, `PhotoRender`: 사진 블러 게이트 기록.
- `CameraPose`, `Reconstruction`: 사진별 카메라 자세와 복원 결과.
- `Alignment`: after 장면 → before 장면 닮음 변환과 잔차.
- `ObjectInstance`, `ObjectChange`, `ChangeReport`: 물체 3D 위치와 전후 변화 (모델 출처, 미검수).
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from dlp_schema.common import Contract, FiniteFloat, OntologyId

ScenePhase = Literal["before", "after"]
ChangeKind = Literal["moved", "unchanged", "removed", "added"]
# 모델 결과의 검수 상태. 지금은 모두 미검수로 낸다 (검수 연동은 후속, ADR 0066).
ReviewState = Literal["unreviewed"]

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


class Reconstruction(Contract):
    """한 단계의 복원 결과 (라벨링 버킷 `recon/<tag>/`).

    metric: 좌표가 미터인가. SfM만으로는 척도를 모르므로 실제 경로는 척도 기준(크기를 아는 마커 등)
      없이는 False다. 비교 단계는 정책 `change.require_metric`이면 before가 metric이어야 한다.
    registered: 자세를 찾은 사진 수 / 전체 사진 수 (`photo_count`).
    splat_key·points_key·cameras_key: 같은 버킷의 산출물 키.
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


Scene3dRecordKind = Literal[
    "capture",
    "privacy_detection",
    "privacy_approval",
    "render",
    "reconstruction",
    "alignment",
    "change_report",
]


class Scene3dRecord(Contract):
    """scene3d 산출물 하나의 기록 (추가 전용 표 `scene3d_records`, Alembic 0027).

    본문은 bucket·storage_key의 JSON이고 이 기록은 무엇을 어떤 버전으로 만들었는지만 남긴다.
    phase는 단계별 산출물에만 있다 (alignment·change_report는 None). bucket은 `raw`(원본 버킷:
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
