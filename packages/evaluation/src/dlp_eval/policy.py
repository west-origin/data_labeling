"""평가 정책 (config/policies/evaluation.yaml).

WP11, ADR 0013·0025·0031. `load_policy(root)`가 YAML을 읽어 `EvaluationPolicy`로 검증한다.
하네스(지표 계산의 허용 오차·IoU 문턱·보간 간격), 게이트(과제별 배포 규칙), 재학습 루프가 같은
객체를 쓴다.

공개 이름:
- `Task`·`TASKS` — 평가 과제 이름과 평가 순서. 학습 정책(`dlp_train.policy`)과 러너의 과제 키도
  이것이다.
- `TASK_METRICS`·`task_metrics`·`CLASS_METRIC_PREFIXES`·`is_class_metric` — 과제별 평가기가 내는
  지표 이름 (게이트 규칙 검사와 하네스 출력 검사에 쓴다).
- `Tolerances`, `PrivacyEval`, `InterpGaps`, `GateRule`, `EvaluationPolicy` — YAML 절별 모델.
- `load_policy` — 저장소 루트에서 정책을 읽는다.

주의: 이 모델들은 `dlp_schema.common.Contract`(엄격 검증, 모르는 키 거부)를 상속하지만 `schemas/`의
JSON Schema 생성 대상은 아니다. 그래도 `Field(description=...)` 문자열은 바꾸지 않는다 (다른 계약과
같은 규칙). 이 정책은 모델 버전 해시에 들어가지 않는다 (평가 결과는 리포트로만 남는다).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_schema.common import Contract

# 평가 과제. 라벨 종류와의 대응은 `dlp_eval.runner.TASK_KINDS`
Task = Literal[
    "objects", "hands", "body", "contact", "actions", "relations", "states", "coverage", "privacy"
]
# 평가·리포트 순서 (`harness.evaluate`가 이 순서로 돈다)
TASKS: tuple[Task, ...] = (
    "objects",
    "hands",
    "body",
    "contact",
    "actions",
    "relations",
    "states",
    "coverage",
    "privacy",
)

# 과제 → 그 과제 평가기(`dlp_eval.harness.EVALUATORS`)가 내는 지표 이름. 행동 과제의 구간 F1
# (`segment_f1_<문턱>`)은 정책 `segment_iou`에 따라 달라지므로 여기에 없고 `task_metrics`가 붙인다.
# 쓰임 (ADR 0031): (1) 정책을 읽을 때 게이트 규칙의 지표 이름 검사
# (`EvaluationPolicy._check_gate_metrics`): 오타가 있으면 후보·기존 모두 NaN이 되어 게이트가 그
# 비교를 조용히 건너뛰기 때문이다. (2) `harness.evaluate`가 평가기 출력이 이 목록과 같은지 확인해,
# 평가기에 지표를 더하고 여기를 잊으면 바로 실패한다.
TASK_METRICS: dict[Task, frozenset[str]] = {
    "objects": frozenset({"map", "ap50", "ap75", "hota", "idf1", "mota", "ece"}),
    "hands": frozenset({"pck"}),
    "body": frozenset({"pck"}),
    "contact": frozenset(
        {
            "contact_start_f1",
            "contact_end_f1",
            "contact_start_error_ms",
            "contact_end_error_ms",
            "grasp_macro_f1",
        }
    ),
    "actions": frozenset({"temporal_map", "boundary_f1", "verb_macro_f1"}),
    "relations": frozenset({"relation_f1"}),
    "states": frozenset({"transition_accuracy", "transition_precision"}),
    "coverage": frozenset({"coverage_abs_error"}),
    "privacy": frozenset({"blur_recall", "blur_precision"}),
}


# 과제 → 클래스별 지표의 접두사. 평가기는 "<접두사>/<클래스>" 이름으로 클래스마다 값을 더 낸다
# (objects: ap/<클래스>, privacy: blur_recall/<블러 대상>). 클래스 목록은 데이터(온톨로지)에 따라
# 달라지므로 이름 검사는 접두사까지만 한다.
CLASS_METRIC_PREFIXES: dict[Task, frozenset[str]] = {
    "objects": frozenset({"ap"}),
    "privacy": frozenset({"blur_recall"}),
}


def is_class_metric(task: Task, name: str) -> bool:
    """`name`이 그 과제의 클래스별 지표("<접두사>/<클래스>", 클래스는 비지 않음)인가."""
    prefix, sep, cls = name.partition("/")
    return bool(sep and cls) and prefix in CLASS_METRIC_PREFIXES.get(task, frozenset())


def task_metrics(task: Task, segment_iou: tuple[float, ...]) -> frozenset[str]:
    """과제 평가기가 늘 내는 지표 이름 전체 (클래스별 지표 `is_class_metric`은 빼고).

    Args:
        task: 평가 과제.
        segment_iou: 정책의 행동 구간 F1 IoU 문턱들. actions면 `segment_f1_<문턱>`을 더한다
            (이름 형식은 `harness.eval_actions`의 f-string과 같다: 0.5 → "segment_f1_0.5").
    """
    names = TASK_METRICS[task]
    if task == "actions":
        names = names | {f"segment_f1_{thr}" for thr in segment_iou}
    return names


class Tolerances(Contract):
    """시점 허용 오차 (`tolerance_ms`, 모두 정수 ms, 마스터 타임라인 시각)."""

    contact: int  # 영상 기준 접촉 시작·종료 허용 오차 (맨손 세션)
    contact_glove: int  # 장갑 스트림이 있는 세션의 접촉 허용 오차 (장갑 압력이 더 정확하다)
    boundary: int  # 행동 경계 일치 허용 오차
    state: int  # 상태 전이 시각 허용 오차


class PrivacyEval(Contract):
    """블러 평가 기준 (`privacy`). 둘 다 면적 비율 0~1."""

    coverage: float = Field(
        gt=0, le=1, description="정답 블러 박스 면적 중 예측 블러가 덮어야 하는 비율"
    )
    precision_overlap: float = Field(
        gt=0, le=1, description="예측 블러 박스 면적 중 정답 대상 위에 있어야 맞은 것으로 보는 비율"
    )


class InterpGaps(Contract):
    """예측 트랙을 정답 시각에 맞출 때 보간할 최대 키프레임 간격 (과제별).

    프리라벨 트래커는 max_gap_ms보다 짧은 끊김을 한 트랙으로 잇되 그 사이에 키프레임을 만들지 않는다
    (OWLv2 도구는 frame_stride_ms마다만 추론한다). 그래서 과제의 값은 그 과제 예측을 내는 어댑터의
    트랙 간격 이상이어야 한다. 짧으면 같은 박스도 정답 시각에서 예측이 없는 것으로 세어 기존 지표가
    0에 가까워지고, 약한 후보도 게이트를 통과한다.

    단위는 ms (스트림 PTS 시각 차). 정답 트랙은 이 값과 상관없이 보간한다 (`harness.TRUTH_GAP`).
    `prelabel.yaml`과의 관계는 test_harness.py `test_interp_gaps_cover_prelabel_tracker_gaps`가
    검사한다.
    """

    default: int = Field(ge=0)  # tasks에 없는 과제의 간격
    tasks: dict[Task, int] = Field(default_factory=dict[Task, int])  # 과제 → 간격 (ms)

    def for_task(self, task: Task) -> int:
        """과제의 최대 보간 간격 (ms). 과제 값이 없으면 `default`."""
        return self.tasks.get(task, self.default)


class GateRule(Contract):
    """과제 하나의 배포 게이트 규칙 (`gate.<과제>`). 판정 로직은 `dlp_eval.gate.decide`."""

    primary: str  # 주 지표 이름 (하네스 지표 키, 예: "hota", "segment_f1_0.5")
    min_gain: float  # 기존 대비 주 지표가 최소 이만큼 좋아져야 한다 (방향은 LOWER_IS_BETTER로 보정)
    max_drop: float = Field(ge=0)  # 지키는 지표의 허용 하락 (0~1 지표 기준)
    max_drop_by: dict[str, float] = Field(
        default_factory=dict[str, float],
        description="지표별 허용 하락 (단위가 다른 지표용, 예: 오차 ms). 없으면 max_drop",
    )
    guards: tuple[str, ...]  # 나빠지면 안 되는 지표 (주 지표는 자동 포함)

    def allowed_drop(self, metric: str) -> float:
        """지표의 허용 하락량. `max_drop_by`에 있으면 그 값(그 지표 단위), 없으면 `max_drop`."""
        return self.max_drop_by.get(metric, self.max_drop)

    # 기존 모델이 없을 때(첫 배포) 주 지표가 넘어야 할 기준 (오차 지표는 이하여야 한다)
    first_deploy: float


class EvaluationPolicy(Contract):
    """`config/policies/evaluation.yaml` 전체. 각 키의 의미는 YAML 주석을 본다."""

    version: int  # 정책 형식 버전
    tolerance_ms: Tolerances
    max_interp_ms: InterpGaps
    track_iou: float = Field(gt=0, le=1)  # IDF1·MOTA와 ECE 정답 판정의 IoU 문턱
    segment_iou: tuple[float, ...]  # 행동 구간 F1을 낼 IoU 문턱들 (지표 이름 segment_f1_<문턱>)
    relation_iou: float = Field(gt=0, le=1)  # 관계 구간 F1의 IoU 문턱
    match_iou: float = Field(gt=0, le=1)  # 파지 유형·동사 비교 때 정답·예측 구간 짝 IoU 문턱
    pck_alpha: float = Field(gt=0)  # PCK 기준 길이 비율
    ece_bins: int = Field(ge=1)  # ECE 신뢰도 구간 수
    min_samples_per_class: int = Field(ge=1)  # 이보다 정답 표본이 적은 클래스는 "표본 부족" 표시
    privacy: PrivacyEval
    # 하위 집단 리포트 축 (값은 `runner.golden_sessions`가 정한다)
    subgroups: tuple[Literal["glove", "site"], ...]
    gate: dict[Task, GateRule]  # 과제 → 배포 규칙 (없는 과제는 판정하지 않고 경고)

    @model_validator(mode="after")
    def _check_gate_metrics(self) -> EvaluationPolicy:
        """게이트 규칙의 지표 이름(primary·guards·max_drop_by 키)이 그 과제 평가기의 지표인지 본다.

        게이트(`gate._compare`)는 리포트에 없는 지표를 NaN으로 보고, 후보·기존이 모두 NaN이면 그
        비교를 건너뛴다. 그래서 YAML에 지표 이름을 잘못 적으면(예: "hotaa", segment_iou를 바꾸고
        primary를 그대로 둔 "segment_f1_0.5") 그 지표의 하락이 조용히 통과된다. 읽을 때 막는다.

        Raises:
            ValueError: 모르는 지표 이름이 있을 때 (pydantic이 ValidationError로 감싼다).
        """
        problems: list[str] = []
        for task, rule in self.gate.items():
            known = task_metrics(task, self.segment_iou)
            named = [rule.primary, *rule.guards, *rule.max_drop_by]
            unknown = sorted({m for m in named if m not in known and not is_class_metric(task, m)})
            if unknown:
                problems.append(f"gate.{task}: {unknown} (가능한 지표: {sorted(known)})")
        if problems:
            raise ValueError("게이트 규칙에 모르는 지표 이름: " + "; ".join(problems))
        return self


def load_policy(root: Path) -> EvaluationPolicy:
    """저장소 루트 `root`의 `config/policies/evaluation.yaml`을 읽어 검증한다.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        pydantic.ValidationError: 키가 빠졌거나 모르는 키·범위 밖 값이 있을 때, 게이트 규칙에
            그 과제 평가기가 내지 않는 지표 이름이 있을 때.
    """
    data: Any = yaml.safe_load(
        (root / "config" / "policies" / "evaluation.yaml").read_text("utf-8")
    )
    return EvaluationPolicy.model_validate(data)
