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
    """`retarget.yaml solver`: 쓸 IK 풀이기.

    풀이기 이름과 판은 리타기팅 판(`retarget_version`)에 들어간다. 두 풀이기는 오차·도달·속도 한계를
    같은 규칙(`ik.frame_errors`, `ik.limit_step`)으로 정하므로 바꿔도 판정 기준은 같다.

    - `dls`: 내장 감쇠 최소제곱 (numpy, CPU, `dlp_retarget.ik.DlsIkSolver`). 저장소
      `retarget.yaml`이 쓰는 값이다 (필드 기본값은 없다).
    - `mink`: mink(MuJoCo) 격리 환경 (`dlp_retarget.mink.MinkIkSolver`). `retarget.yaml mink`의
      환경이 있어야 한다.
    """

    DLS = "dls"
    MINK = "mink"


class UnreachableAction(StrEnum):
    """`retarget.yaml unreachable`: 로봇이 닿지 못한 프레임을 어떻게 다룰지.

    어느 쪽이든 닿지 못한 프레임은 가장 가까운 해를 내고 `retarget.reachable=0`과 오차를 남긴다.
    목표를 작업 공간 안으로 잘라 "도달"로 만들지 않는다. 손 목표가 하나도 없는 에피소드는 이 값과
    관계없이 버린다 (`retarget._decide`).

    - `mark`: 표시만 하고 에피소드는 낸다. 비율 기준을 보지 않는다.
    - `drop_episode`: 손 목표 비율이 `min_target_fraction` 미만이거나, 목표가 있는 프레임 중 닿지
      못한 비율이 `max_unreachable_fraction`을 넘는 에피소드를 버린다. 남는 에피소드는 `mark`와
      같다.
    """

    MARK = "mark"
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
    프레임 중 도달하지 못한 비율 상한 (0~1, drop_episode에서만 본다). min_target_fraction: 에피소드
    프레임 중 손 목표가 있어야 하는 비율 하한 (0~1, 밑돌면 drop_episode에서 버린다. mark는 이 값을
    보지 않고 보고서 `target_fraction`에만 비율이 남는다). hand_parts: 손 점 역할별 3D 궤적 부위.
    mink: mink 풀이기 격리 환경.
    """

    version: int
    solver: SolverName
    unreachable: UnreachableAction
    max_unreachable_fraction: float = Field(ge=0, le=1)
    min_target_fraction: float = Field(ge=0, le=1)
    hand_parts: HandParts
    mink: MinkEnv

    def digest(self) -> str:
        """정책 값 해시 (sha256 앞 8자, 리타기팅 판의 `+p` 부분).

        mink 환경 위치도 해시에 들어간다. 그래서 dls 풀이기를 써도 `mink` 절을 바꾸면 판이 바뀐다.
        """
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:8]


def load_policy(root: Path) -> RetargetPolicy:
    """`<root>/config/policies/retarget.yaml`을 읽어 검증한다.

    Args:
        root: 저장소 루트.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        pydantic.ValidationError: 키가 없거나 모르는 키가 있거나 값이 범위를 벗어날 때.
    """
    data: Any = yaml.safe_load((root / "config" / "policies" / "retarget.yaml").read_text("utf-8"))
    return RetargetPolicy.model_validate(data)
