"""내보내기 정책 (config/policies/export.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from dlp_schema.common import Contract
from dlp_schema.dataset import Split
from dlp_schema.labels import VerificationState


class CocoPolicy(Contract):
    jpeg_quality: int = Field(ge=1, le=100)
    frame_tolerance_ms: int = Field(ge=0)


class IntervalsPolicy(Contract):
    kinds: tuple[str, ...] = Field(min_length=1)


class LeRobotEnv(Contract):
    python: str
    project: str  # 저장소 루트 기준, pyproject.toml과 uv.lock이 있는 디렉터리


class LeRobotPolicy(Contract):
    fps: int = Field(ge=1)
    video_stream: str
    max_width: int = Field(ge=0)
    interp_max_gap_ms: int = Field(ge=0)
    hand_points_3d: tuple[str, ...]
    repo_id: str
    robot_type: str
    vcodec: str
    env: LeRobotEnv


class IdsPolicy(Contract):
    pseudonymize: bool
    secret_env: str  # 가명 비밀값을 담은 환경 변수 (없으면 실행마다 임의 값)


class ExportPolicy(Contract):
    version: int
    label_states: tuple[VerificationState, ...] = Field(min_length=1)
    splits: tuple[Split, ...] = Field(min_length=1)
    excluded_kinds: tuple[str, ...]
    coco: CocoPolicy
    intervals: IntervalsPolicy
    lerobot: LeRobotPolicy
    ids: IdsPolicy

    @field_validator("label_states")
    @classmethod
    def _reviewed(cls, v: tuple[VerificationState, ...]) -> tuple[VerificationState, ...]:
        if VerificationState.UNREVIEWED in v:
            raise ValueError("미검수 라벨은 기본 정책에 넣지 않는다 (--include-unreviewed로만)")
        return v

    @field_validator("excluded_kinds")
    @classmethod
    def _no_blur(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if "blur_track" not in v:
            raise ValueError("블러 라벨은 내보내지 않는다 (excluded_kinds에 blur_track 필요)")
        return v


def load_policy(root: Path) -> ExportPolicy:
    data: Any = yaml.safe_load((root / "config" / "policies" / "export.yaml").read_text("utf-8"))
    return ExportPolicy.model_validate(data)
