"""리타기팅 정책 (`config/policies/retarget.yaml`, ADR 0047).

로봇마다 다른 값(허용 오차, 배치, 그리퍼)은 로봇 설정(`dlp_retarget.robot`)에 두고, 여기에는 모든
로봇에 같은 운영 규칙만 둔다: 풀이기 선택, 도달하지 못한 프레임 처리, 손 점 이름, mink 격리 환경.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from dlp_schema.common import Contract, OntologyId


class SolverName(StrEnum):
    DLS = "dls"  # 내장 감쇠 최소제곱 (numpy, CPU, `dlp_retarget.ik.DlsIkSolver`)
    MINK = "mink"  # mink(MuJoCo) 격리 환경 (`dlp_retarget.mink.MinkIkSolver`)


class UnreachableAction(StrEnum):
    # 도달하지 못한 프레임도 가장 가까운 해를 내고 retarget.reachable=0으로 표시한다
    # (값을 지어내지 않는다)
    MARK = "mark"
    # 도달하지 못한 비율이 max_unreachable_fraction을 넘는 에피소드는 내보내지 않는다 (나머지는
    # mark)
    DROP_EPISODE = "drop_episode"


class HandParts(Contract):
    """손 점 역할 → 3D 궤적 부위 이름 (온톨로지 hand_joints)."""

    wrist: OntologyId
    thumb: OntologyId
    index: OntologyId


class MinkEnv(Contract):
    """mink 격리 환경 (`uv run --project <project> --locked --isolated`)."""

    python: str
    project: str


class RetargetPolicy(Contract):
    """retarget.yaml 전체.

    solver: IK 풀이기. unreachable: 도달하지 못한 프레임 처리. max_unreachable_fraction: 목표가 있는
    프레임 중 도달하지 못한 비율 상한 (drop_episode). min_target_fraction: 에피소드 프레임 중 손
    목표가 있어야 하는 비율 하한 (밑돌면 drop_episode에서 버린다, mark에서는 표시만).
    """

    version: int
    solver: SolverName
    unreachable: UnreachableAction
    max_unreachable_fraction: float = Field(ge=0, le=1)
    min_target_fraction: float = Field(ge=0, le=1)
    hand_parts: HandParts
    mink: MinkEnv

    def digest(self) -> str:
        """정책 값 해시 (sha256 앞 8자, 리타기팅 버전에 들어간다). mink 환경 위치는 빼지 않는다."""
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:8]


def load_policy(root: Path) -> RetargetPolicy:
    """`<root>/config/policies/retarget.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때. pydantic.ValidationError: 값이 틀렸을 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "retarget.yaml").read_text("utf-8"))
    return RetargetPolicy.model_validate(data)
