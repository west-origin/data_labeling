"""학습 정책과 다른 단계 정책·코드의 대응 (서비스 없이).

정답 근거: 재학습 루프는 게이트 규칙이 없는 과제를 거절한다(F-ETA-26). replaces는 프리라벨 기본
Predictor 이름과 접촉 단계만 대신할 수 있다(`dlp_train.deployed.Deployed.apply`,
`dlp_prelabel.runner.CONTACT_STEP`, F-ETA-27).
"""

from __future__ import annotations

from dlp_eval.policy import load_policy as load_eval_policy
from dlp_prelabel.adapters.mediapipe_models import MediaPipeHands, MediaPipeObjects
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.adapters.rtmpose import RtmPose
from dlp_prelabel.adapters.yolox_tools import YoloxTools
from dlp_prelabel.runner import CONTACT_STEP
from dlp_schema import repo_root
from dlp_train.policy import PRELABEL_REPLACEABLE, load_policy

ROOT = repo_root()


def test_every_training_task_has_a_gate_rule() -> None:
    """training.yaml tasks의 과제는 모두 evaluation.yaml gate 규칙이 있다 (없으면 학습 거절)."""
    tasks = set(load_policy(ROOT).tasks)
    assert sorted(tasks - set(load_eval_policy(ROOT).gate)) == []


def test_replaceable_names_match_prelabel_adapters() -> None:
    """replaces에 쓸 수 있는 이름 = 프리라벨 기본 어댑터 이름 + 접촉 단계."""
    adapters = {c.name for c in (MediaPipeHands, RtmPose, MediaPipeObjects, OwlObjects, YoloxTools)}
    assert adapters | {CONTACT_STEP} == PRELABEL_REPLACEABLE
