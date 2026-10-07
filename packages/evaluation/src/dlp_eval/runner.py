"""골든셋 평가 실행 (`dlp eval golden`).

DB에서 정답·예측을 모아 지표·하위 집단·게이트 리포트를 만든다.

- 정답: 골든셋 세션의 현재 라벨 중 사람이 만든 것과 사람이 승인·수정한 것.
  (표본 검증만 된 모델 라벨은 정답으로 보지 않는다.)
- 예측: 과제별로 지정한 model_version의 레코드 (삭제 레코드 제외). 나중에 검수자가 고쳤어도 모델이
  낸 원래 레코드로 평가한다. 버전이 `*`로 끝나면 그 앞부분으로 시작하는 현재 레코드를 평가한다.
- 오류 삽입 사본·측정 레코드와 그 후손은 예측에서 뺀다.
- 하위 집단: glove(장갑 스트림 유무), site(장소).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from dlp_eval.gate import GateDecision
from dlp_eval.harness import LOWER_IS_BETTER, EvalReport, SessionData
from dlp_eval.policy import Task
from dlp_schema.db.repository import get_golden_set, get_labels, get_session
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.session import StreamKind

TASK_KINDS: dict[Task, tuple[str, ...]] = {
    "objects": ("box_track",),
    "hands": ("keypoint_track",),
    "body": ("keypoint_track",),
    "contact": ("hand_state",),
    "actions": ("action",),
    "relations": ("relation",),
    "states": ("object_state",),
    "coverage": ("coverage",),
}
TRUSTED = {VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED}


def is_truth(x: LabelRecord) -> bool:
    return x.provenance.source is Source.HUMAN or x.verification.state in TRUSTED


def load_golden(
    conn: sa.Connection, golden_version: str, models: dict[Task, str]
) -> dict[Task, list[SessionData]]:
    golden = get_golden_set(conn, golden_version)
    out: dict[Task, list[SessionData]] = {task: [] for task in models}
    for sid in golden.session_ids:
        session = get_session(conn, sid)
        labels = get_labels(conn, sid)
        truth = [x for x in current_labels(labels) if is_truth(x)]
        # 오류 삽입 사본·측정 레코드와 그 후손은 모델 버전을 달고 있어도 예측이 아니다
        excluded = non_operational_ids(labels)
        glove = any(
            s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT) for s in session.streams
        )
        groups = {"glove": "glove" if glove else "bare", "site": session.site_id}
        current = current_labels(labels)
        for task, version in models.items():
            kinds = TASK_KINDS[task]
            if version.endswith("*"):
                # 버전 앞부분으로 고르면 그 모듈의 현재 레코드를 평가한다. 관계·커버리지처럼 정책이
                # 바뀌어도 내용이 같은 레코드는 옛 버전을 그대로 두는 모듈용 (예: relations-*)
                prefix = version[:-1]
                pred = [
                    x
                    for x in current
                    if x.kind in kinds and (x.provenance.model_version or "").startswith(prefix)
                ]
                out[task].append(
                    SessionData(sid, [x for x in truth if x.kind in kinds], pred, groups)
                )
                continue
            out[task].append(
                SessionData(
                    sid,
                    [x for x in truth if x.kind in kinds],
                    [
                        x
                        for x in labels
                        if x.kind in kinds
                        and x.provenance.model_version == version
                        and not x.retracted
                        and x.label_id not in excluded
                    ],
                    groups,
                )
            )
    return out


def _clean(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_clean(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def report_dict(report: EvalReport, decision: GateDecision | None) -> dict[str, Any]:
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
    return _clean(data)


def markdown(report: EvalReport, decision: GateDecision | None) -> str:
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


def write_report(path: Path, report: EvalReport, decision: GateDecision | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_dict(report, decision), ensure_ascii=False, indent=2), "utf-8"
    )
    path.with_suffix(".md").write_text(markdown(report, decision), "utf-8")
