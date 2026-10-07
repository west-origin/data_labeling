"""골든셋 평가 실행 (`dlp eval golden`).

DB에서 정답·예측을 모아 지표·하위 집단·게이트 리포트를 만든다.

- 정답: 골든셋 세션의 현재 라벨 중 사람이 만든 것과 사람이 승인·수정한 것.
  (표본 검증만 된 모델 라벨은 정답으로 보지 않는다.)
- 예측: 과제별로 지정한 model_version의 레코드 (삭제 레코드 제외). 나중에 검수자가 고쳤어도 모델이
  낸 원래 레코드로 평가한다. 버전이 `*`로 끝나면 그 앞부분으로 시작하는 버전의 레코드 중 새
  모델 버전이 지우지(retractions) 않은 것을 평가한다. 검수자가 지우거나 고친 모델 레코드도
  그대로 예측이다 (검수자가 지운 오탐이 사라지면 안 된다).
- 오류 삽입 사본·측정 레코드와 그 후손은 예측에서 뺀다.
- 하위 집단: glove(장갑 스트림 유무), site(장소). evaluation.yaml subgroups에 적은 것만 리포트한다.
- 사용 중지(동의 철회 등)된 세션은 골든셋에 있어도 평가하지 않는다.
- 한 과제에 모델 버전을 여럿 주면 (예: objects를 대신할 기본 어댑터 objects·tools) 예측을 세션별로
  합친다 (`load_golden_merged`).
- 세션마다 버전이 다른 단계(접촉·행동·3D 궤적은 버전에 세션 데이터 해시가 들어간다)는 버전
  앞부분+`*`로 고른다 (예: contact=contact-heuristic-1*, ADR 0013).

WP11, ADR 0013·0015·0025. 진입점: `dlp_cli.eval_cmds`(`dlp eval golden`)와 재학습 루프
(`dlp_train.loop`). DB는 읽기만 한다 (sessions·streams, label_records, golden_sets, withdrawals
테이블). 원본 버킷에는 접근하지 않는다. 리포트 파일은 `write_report`가 로컬 경로에 쓴다.

공개 이름: `TASK_KINDS`, `TIMELINE_TASKS`, `is_truth`, `GoldenSession`, `golden_sessions`,
`load_golden`, `load_golden_merged`, `MissingPredictionsError`, `report_dict`, `markdown`,
`write_report`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from dlp_eval.gate import GateDecision
from dlp_eval.harness import LOWER_IS_BETTER, EvalReport, SessionData
from dlp_eval.policy import Task
from dlp_schema.db.repository import (
    get_golden_set,
    get_labels,
    get_session,
    withdrawn_session_ids,
)
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.session import LifecycleState, Session, StreamKind

# 과제 → 평가에 쓰는 라벨 종류 (LabelRecord.kind). 손·전신은 같은 종류를 골격으로 나눈다 (하네스가
# 거른다). 학습 예제 추출(`dlp_train.extract.matches`)도 이 표를 쓴다.
TASK_KINDS: dict[Task, tuple[str, ...]] = {
    "objects": ("box_track",),
    "hands": ("keypoint_track",),
    "body": ("keypoint_track",),
    "contact": ("hand_state",),
    "actions": ("action",),
    "relations": ("relation",),
    "states": ("object_state",),
    "coverage": ("coverage",),
    "privacy": ("blur_track",),
}
# 마스터 타임라인 구간 라벨(stream_id 없음)로 평가하는 과제. 세션마다 한 번만 예측한다
# (영상 스트림마다 돌리면 같은 타임라인 구간이 스트림 수만큼 겹친다, ADR 0019)
TIMELINE_TASKS: frozenset[Task] = frozenset(
    {"contact", "actions", "relations", "states", "coverage"}
)
# 모델 라벨이 정답이 되는 검증 상태 (표본 검증 sample_verified는 들지 않는다)
TRUSTED = {VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED}


def is_truth(x: LabelRecord) -> bool:
    """정답으로 쓰는 라벨인가: 사람이 만들었거나 사람이 승인·수정한 라벨.

    호출자는 이미 `current_labels`로 운영 라벨만 골랐다고 가정한다 (오류 삽입·측정 레코드 제외).
    """
    return x.provenance.source is Source.HUMAN or x.verification.state in TRUSTED


@dataclass(frozen=True)
class GoldenSession:
    """골든셋 세션 하나의 평가 재료 (과제와 무관)."""

    session: Session
    labels: list[LabelRecord]  # 전체 이력
    truth: list[LabelRecord]  # 정답 (현재 라벨 중 사람이 만들거나 승인·수정한 것)
    groups: dict[str, str]  # 하위 집단: {"glove": "glove"|"bare", "site": 장소 ID}


def golden_sessions(conn: sa.Connection, golden_version: str) -> list[GoldenSession]:
    """골든셋 버전의 평가할 세션들 (골든셋에 적힌 순서).

    사용 중지 기록(withdrawals)이 있거나 생애주기가 withdrawn인 세션은 뺀다. 세션마다 라벨 전체
    이력을 한 번 읽는다 (읽기 전용).

    Raises:
        sqlalchemy.exc.NoResultFound: 골든셋 버전이나 세션이 DB에 없을 때 (저장소 함수가 던진다).
    """
    golden = get_golden_set(conn, golden_version)
    withdrawn = withdrawn_session_ids(conn)
    out: list[GoldenSession] = []
    for sid in golden.session_ids:
        if sid in withdrawn:
            continue
        session = get_session(conn, sid)
        if session.lifecycle_state is LifecycleState.WITHDRAWN:
            continue
        labels = get_labels(conn, sid)
        truth = [x for x in current_labels(labels) if is_truth(x)]
        # 장갑 스트림(좌·우 어느 쪽이든)이 있으면 장갑 세션 → 접촉 허용 오차가 다르다
        glove = any(
            s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams
        )
        groups = {"glove": "glove" if glove else "bare", "site": session.site_id}
        out.append(GoldenSession(session, labels, truth, groups))
    return out


def _predictions(
    labels: list[LabelRecord],
    kinds: tuple[str, ...],
    version: str,
    excluded: set[str],
    model_retracted: set[str],
) -> list[LabelRecord]:
    """전체 이력에서 그 모델 버전이 낸 레코드 (삭제 레코드·비운영 레코드 제외).

    버전이 `*`로 끝나면 그 앞부분으로 시작하는 버전의 레코드 중 새 모델 버전이 지우지 않은 것
    전부다. 관계·커버리지처럼 정책이 바뀌어도 내용이 같은 레코드는 옛 버전을 그대로 두는 모듈용
    (예: relations-*). 검수자가 지우거나 고친 레코드는 그대로 예측이다.

    Args:
        labels: 세션의 전체 라벨 이력 (`get_labels`).
        kinds: 과제의 라벨 종류 (`TASK_KINDS`).
        version: 정확한 모델 버전, 또는 앞부분 + `*`.
        excluded: 비운영 레코드 ID (`non_operational_ids`: 오류 삽입·측정 레코드와 그 후손).
        model_retracted: 모델 단계가 지운(retractions) 레코드 ID. 앞부분 고르기에서만 쓴다 — 정확한
            버전을 주면 그 버전이 낸 레코드를 나중에 다른 버전이 지웠어도 그대로 평가한다.

    Returns:
        사람 출처·삭제 레코드를 뺀, 버전이 맞는 레코드 (이력 순서).
    """
    prefix = version[:-1] if version.endswith("*") else None
    out: list[LabelRecord] = []
    for x in labels:
        if (
            x.kind not in kinds
            or x.provenance.source is Source.HUMAN
            or x.retracted
            or x.label_id in excluded
        ):
            continue
        mv = x.provenance.model_version or ""
        if prefix is None:
            if mv == version:
                out.append(x)
        elif mv.startswith(prefix) and x.label_id not in model_retracted:
            out.append(x)
    return out


def load_golden(
    conn: sa.Connection, golden_version: str, models: dict[Task, str]
) -> dict[Task, list[SessionData]]:
    """골든셋 세션마다 과제별 정답·예측을 모은다 (과제마다 모델 버전 하나).

    Args:
        conn: DB 연결 (읽기만 한다).
        golden_version: 골든셋 버전.
        models: 과제 → 모델 버전 (끝이 `*`이면 앞부분 고르기).

    Returns:
        과제 → 세션별 `SessionData` (평가할 세션 모두, 예측이 없어도 넣는다). 정답은 과제의 라벨
        종류로만 거르고, 골격·접촉 여부는 하네스가 거른다.
    """
    out: dict[Task, list[SessionData]] = {task: [] for task in models}
    for g in golden_sessions(conn, golden_version):
        sid, labels, truth, groups = g.session.session_id, g.labels, g.truth, g.groups
        # 오류 삽입 사본·측정 레코드와 그 후손은 모델 버전을 달고 있어도 예측이 아니다
        excluded = non_operational_ids(labels)
        # 모델 버전이 바뀌어 지운 레코드 (retractions, 출처 model). 사람이 지운 것은 들지 않는다
        model_retracted = {
            x.parent_label_id
            for x in labels
            if x.retracted and x.provenance.source is not Source.HUMAN and x.parent_label_id
        }
        for task, version in models.items():
            kinds = TASK_KINDS[task]
            pred = _predictions(labels, kinds, version, excluded, model_retracted)
            out[task].append(SessionData(sid, [x for x in truth if x.kind in kinds], pred, groups))
    return out


class MissingPredictionsError(ValueError):
    """골든셋에 예측이 하나도 없는 모델 버전.

    버전을 잘못 적었거나 골든셋에 그 단계를 돌리지 않았다.
    """


def load_golden_merged(
    conn: sa.Connection,
    golden_version: str,
    models: dict[Task, list[str]],
    *,
    require_predictions: bool = False,
) -> dict[Task, list[SessionData]]:
    """과제마다 여러 모델 버전의 예측을 세션별로 합친다 (대신할 기본 어댑터 여럿과 비교할 때).

    require_predictions: 골든셋에 예측이 하나도 없는 버전이 있으면 MissingPredictionsError.
    기존(비교) 모델에 쓴다. 잘못 적은 버전은 예측이 비어 기존 지표가 0에 가까워지고 어떤 후보든
    통과하기 때문이다.

    버전마다 `load_golden`을 따로 불러 골든셋을 다시 읽는다 (버전 수만큼 DB 조회). 정답·하위 집단은
    처음 읽은 버전의 것을 쓴다 (모두 같다). 같은 레코드가 두 버전 패턴에 모두 걸리면 두 번 들어간다.
    """
    out: dict[Task, list[SessionData]] = {}
    for task, versions in models.items():
        merged: dict[str, SessionData] = {}
        for v in versions:
            sessions = load_golden(conn, golden_version, {task: v})[task]
            if require_predictions and not any(s.pred for s in sessions):
                raise MissingPredictionsError(
                    f"{task}: 모델 버전 {v}의 예측이 골든셋 {golden_version}에 없습니다 "
                    "(버전을 확인하세요. 버전 앞부분으로 고르려면 끝에 *)"
                )
            for s in sessions:
                if s.session_id in merged:
                    merged[s.session_id].pred.extend(s.pred)
                else:
                    merged[s.session_id] = SessionData(
                        s.session_id, s.truth, list(s.pred), s.groups
                    )
        out[task] = list(merged.values())
    return out


def _clean(value: Any) -> Any:
    """JSON으로 쓸 수 있게 NaN을 None(null)으로 바꾼다 (dict·list는 재귀). 표준 JSON에는 NaN이
    없다."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_clean(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def report_dict(
    report: EvalReport,
    decision: GateDecision | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """리포트를 JSON으로 쓸 수 있는 dict로 만든다.

    extra는 리포트에 함께 남길 값이다 (예: 재학습 루프가 비교한 배포 모델 버전). 키:
    golden_version, model_versions, overall, subgroups, lower_is_better, (판정이 있으면) gate,
    그리고 extra의 키 (같은 키면 extra가 덮어쓴다). 재학습 루프는 extra에 `deployed_baseline`을
    넣고, 승인 배포(`dlp_train.loop.deploy`)가 다시 읽는다. NaN은 null이 된다.
    """
    data: dict[str, Any] = {
        "golden_version": report.golden_version,
        "model_versions": report.model_versions,
        "overall": {t: asdict(r) for t, r in report.overall.items()},
        "subgroups": {
            g: {t: asdict(r) for t, r in rs.items()} for g, rs in report.subgroups.items()
        },
        "lower_is_better": sorted(LOWER_IS_BETTER),
    }
    if decision is not None:
        data["gate"] = {
            "passed": decision.passed,
            "tasks": {t: asdict(d) for t, d in decision.tasks.items()},
        }
    data |= dict(extra or {})
    return _clean(data)


def markdown(report: EvalReport, decision: GateDecision | None) -> str:
    """사람이 읽는 Markdown 리포트: 과제마다 지표 x (전체, 하위 집단들) 표, 표본 부족, 게이트 판정.

    하위 집단에 그 과제 리포트가 없으면 "-", 지표 값이 NaN이면 "nan"으로 찍힌다.
    """
    lines = [f"# 골든셋 평가 {report.golden_version}", ""]
    for task, r in report.overall.items():
        lines += [f"## {task} (모델 {report.model_versions.get(task, '-')}, 세션 {r.sessions})", ""]
        lines += ["| 지표 | 전체 | " + " | ".join(report.subgroups) + " |"]
        lines += ["| --- | --- | " + " | ".join("---" for _ in report.subgroups) + " |"]
        for name, value in r.metrics.items():
            subs = [report.subgroups[g].get(task) for g in report.subgroups]
            cells = [f"{s.metrics.get(name, math.nan):.4f}" if s else "-" for s in subs]
            lines.append(f"| {name} | {value:.4f} | " + " | ".join(cells) + " |")
        if r.under_sampled:
            lines += ["", f"표본 부족 클래스: {', '.join(r.under_sampled)}"]
        if decision is not None and task in decision.tasks:
            d = decision.tasks[task]
            lines += ["", f"게이트: {'통과' if d.passed else '실패'}"]
            lines += [f"- {x}" for x in d.reasons] + [f"- (경고) {x}" for x in d.warnings]
        lines.append("")
    if decision is not None:
        lines.append(f"**배포 게이트: {'통과' if decision.passed else '실패'}**")
    return "\n".join(lines) + "\n"


def write_report(
    path: Path,
    report: EvalReport,
    decision: GateDecision | None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """리포트를 `path`(JSON)와 같은 이름의 `.md`(Markdown)로 쓴다. 상위 디렉터리를 만들고 덮어쓴다.

    로컬 파일만 쓴다. 저장소 업로드는 호출자(재학습 루프·CLI)가 한다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_dict(report, decision, extra), ensure_ascii=False, indent=2), "utf-8"
    )
    path.with_suffix(".md").write_text(markdown(report, decision), "utf-8")
