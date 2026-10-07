"""내보내기 정책 (config/policies/export.yaml) 로더와 검증 (WP15, ADR 0018·0021·0027).

`load_policy(root)`가 YAML을 읽어 `ExportPolicy`로 검증한다. 값은 코드에 하드코딩하지 않고 모두
이 정책에서 읽는다. 검증기 두 개가 판매용 내보내기의 안전 규칙을 강제한다.
- `label_states`에 `unreviewed`를 넣을 수 없다 (미검수는 `--include-unreviewed`로만).
- `excluded_kinds`에 `blur_track`이 반드시 있어야 한다 (블러 라벨은 어떤 형식에도 내보내지 않는다).

주요 클래스: `ExportPolicy`(최상위), `CocoPolicy`, `IntervalsPolicy`, `LeRobotPolicy`,
`LeRobotEnv`(격리 환경), `IdsPolicy`(가명).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from dlp_schema.common import Contract
from dlp_schema.dataset import Split
from dlp_schema.labels import VerificationState


class CocoPolicy(Contract):
    """export.yaml `coco` 절."""

    # 이미지(블러본 프레임) JPEG 품질 (1~100)
    jpeg_quality: int = Field(ge=1, le=100)
    # 공간 라벨 키프레임 시각(스트림 PTS, ms)과 블러본 프레임 시각의 허용 차이 (ms).
    # 반올림 오차만 허용한다. 밖이면 그 주석은 "keyframe_between_frames"로 버린다
    frame_tolerance_ms: int = Field(ge=0)


class IntervalsPolicy(Contract):
    """export.yaml `intervals` 절."""

    # 구간 JSON에 넣는 라벨 종류 (`LabelRecord.kind`). 공간 라벨은 COCO·LeRobot으로 나간다
    kinds: tuple[str, ...] = Field(min_length=1)


class LeRobotEnv(Contract):
    """LeRobot 공식 API를 돌리는 격리된 일회용 환경 (export.yaml `lerobot.env`, ADR 0021).

    LeRobot은 PyTorch를 끌어와 작업공간 의존성(av·numpy 버전)과 충돌하므로 작업공간 밖의
    별도 프로젝트(`scripts/lerobot-env`, pyproject + uv.lock)에서 `uv run --locked --isolated`로
    돈다.
    """

    python: str  # 격리 환경의 Python 버전 (예: "3.12")
    project: str  # 저장소 루트 기준, pyproject.toml과 uv.lock이 있는 디렉터리


class LeRobotPolicy(Contract):
    """export.yaml `lerobot` 절 (에피소드 구성과 격리 환경)."""

    fps: int = Field(ge=1)  # 에피소드 고정 프레임률. 가변 프레임 블러본에서 PTS로 프레임을 고른다
    video_stream: str  # 에피소드 영상으로 쓰는 스트림 종류 (`StreamKind` 값, 기본 bodycam)
    max_width: int = Field(ge=0)  # 이보다 넓으면 줄인다 (0이면 원본 크기)
    # 키포인트·3D 궤적을 선형 보간하는 최대 표본 간격 (ms). 더 벌어지면 값 없음
    interp_max_gap_ms: int = Field(ge=0)
    # observation.state에 넣는 손 3D 점 (`Trajectory3DPayload.part`, 개체는 "<손>_hand")
    hand_points_3d: tuple[str, ...]
    repo_id: str  # LeRobot 데이터셋 repo_id (허브에는 올리지 않는다, 오프라인)
    robot_type: str  # LeRobot 메타데이터 robot_type
    vcodec: str  # 에피소드 영상 코덱
    aspect_tolerance: float = Field(ge=0)  # 한 데이터셋에 넣을 수 있는 화면비(가로/세로) 차이
    env: LeRobotEnv


class IdsPolicy(Contract):
    """export.yaml `ids` 절: 작업자·장소·세션·라벨 ID 가명 (ADR 0021·0027)."""

    pseudonymize: bool  # 거짓이면 내부 ID를 그대로 내보낸다 (내부용 내보내기에만)
    secret_env: str  # 가명 비밀값을 담은 환경 변수 (없으면 실행마다 임의 값)
    # 개발용 비밀값 (.env.example). env_var가 dev_envs 밖이면 이 값으로 내보내지 않는다
    dev_secrets: tuple[str, ...]
    env_var: str  # 실행 환경을 알려 주는 환경 변수 이름 (예: DLP_ENV)
    dev_envs: tuple[str, ...] = Field(min_length=1)  # 개발용 비밀값을 허용하는 env_var 값들


class ExportPolicy(Contract):
    """export.yaml 전체."""

    version: int  # 정책 파일 형식 버전
    # 모델 라벨 중 기본으로 넣는 검증 상태. 사람이 만든 라벨은 상태와 관계없이 늘 넣는다
    label_states: tuple[VerificationState, ...] = Field(min_length=1)
    # 기본으로 내보내는 분할 (골든셋·holdout은 판매용 기본 내보내기에 넣지 않는다)
    splits: tuple[Split, ...] = Field(min_length=1)
    excluded_kinds: tuple[str, ...]  # 어떤 형식으로도 내보내지 않는 라벨 종류 (blur_track 필수)
    coco: CocoPolicy
    intervals: IntervalsPolicy
    lerobot: LeRobotPolicy
    ids: IdsPolicy

    @field_validator("label_states")
    @classmethod
    def _reviewed(cls, v: tuple[VerificationState, ...]) -> tuple[VerificationState, ...]:
        """기본 검증 정책에 `unreviewed`가 있으면 거부한다.

        Raises:
            ValueError: `unreviewed`가 들어 있을 때 (미검수는 `--include-unreviewed`로만).
        """
        if VerificationState.UNREVIEWED in v:
            raise ValueError("미검수 라벨은 기본 정책에 넣지 않는다 (--include-unreviewed로만)")
        return v

    @field_validator("excluded_kinds")
    @classmethod
    def _no_blur(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """`blur_track`이 제외 종류에 없으면 거부한다.

        블러 라벨은 얼굴 등 개인정보 위치 자체이므로 어떤 형식으로도 판매하지 않는다.

        Raises:
            ValueError: `blur_track`이 없을 때.
        """
        if "blur_track" not in v:
            raise ValueError("블러 라벨은 내보내지 않는다 (excluded_kinds에 blur_track 필요)")
        return v


def load_policy(root: Path) -> ExportPolicy:
    """`<root>/config/policies/export.yaml`을 읽어 검증한다.

    Args:
        root: 저장소 루트.

    Returns:
        검증된 `ExportPolicy`.

    Raises:
        pydantic.ValidationError: 키가 빠졌거나 범위·검증기 규칙을 어길 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "export.yaml").read_text("utf-8"))
    return ExportPolicy.model_validate(data)
