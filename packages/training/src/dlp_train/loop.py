"""재학습 루프 한 번 (`dlp train run`).

1. 데이터셋 버전에서 과제 학습 예제를 뽑는다
   (학습·검증 분할만, 운영 라벨만, 자동 원본과의 차이 포함). 버전을 만든 뒤 사용 중지(동의 철회)된
   세션은 뺀다 (골든셋 평가도 golden_sessions가 뺀다).
2. 누적 확인: 예제 수가 min_examples 미만이거나 직전 학습보다 min_new_examples만큼 늘지 않았으면
   건너뛴다.
3. 학습 → 산출물을 학습 산출물 버킷에 올리고(sha256) MLflow 실행에 파라미터·지표·산출물을 남긴다.
   학습 실행(training_runs)과 모델 버전(model_versions, candidate)을 DB에 쓴다.
4. 골든셋 평가: 후보와 기존 모델(배포 중인 재학습 모델, 없으면 baseline_versions의 DB 예측)을
   골든셋 세션에 메모리에서 돌려 같은 코드로 평가한다. 골든셋 세션에는 아무것도 쓰지 않는다.
5. 게이트(config/policies/evaluation.yaml) 통과 → 배포(정책 deploy: auto) 또는 승인 대기(approve).
   실패 → rejected. 배포하면 이전 배포 모델은 retired가 된다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import sqlalchemy as sa

from dlp_datasets.lineage import register_training_run
from dlp_datasets.snapshot import SnapshotStore
from dlp_eval.gate import GateDecision, TaskDecision, decide
from dlp_eval.harness import EvalReport, SessionData, evaluate
from dlp_eval.policy import EvaluationPolicy, Task
from dlp_eval.runner import (
    TIMELINE_TASKS,
    GoldenSession,
    MissingPredictionsError,
    golden_sessions,
    load_golden_merged,
    write_report,
)
from dlp_media.storage import ObjectStore, sha256_file
from dlp_schema.dataset import DatasetVersion
from dlp_schema.db.repository import (
    get_dataset_version,
    get_golden_set,
    get_session,
    get_training_run,
    insert_model_version,
    list_model_versions,
    set_model_status,
    withdrawn_session_ids,
)
from dlp_schema.labels import LabelRecord
from dlp_schema.lineage import ModelStatus, ModelVersion, TrainingRun
from dlp_schema.predictor import Clip, Predictor
from dlp_schema.session import LifecycleState, Session, Stream, StreamKind
from dlp_train.extract import load_training_data, matches
from dlp_train.policy import TrainingPolicy
from dlp_train.tracking import Tracker
from dlp_train.trainers import LOADERS, TRAINERS, LoadContext, ModelLoader, Trainer

VIDEO = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 게이트에서 비교한 배포 모델 버전 (없으면 null).
# 평가 리포트(golden.json)와 MLflow 파라미터에 남긴다
DEPLOYED_BASELINE = "deployed_baseline"

# (세션, 스트림, 작업 디렉터리) → 모델에 넣을 영상 파일
ClipSource = Callable[[Session, Stream, Path], Path]


class TrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingJob:
    task: Task
    dataset_version_id: str
    trainer: str | None = None  # 없으면 정책의 과제 템플릿
    params: dict[str, Any] = field(default_factory=dict[str, Any])  # 템플릿 파라미터 위에 덮어쓴다
    # 배포된 재학습 모델이 없을 때 비교할 DB 예측의 모델 버전들 (대신할 기본 어댑터들의 버전).
    # 정책 replaces가 있는 과제는 필수다 (기본 어댑터보다 나쁜 모델이 첫 배포 기준만 넘고
    # 대신하지 않게).
    baseline_versions: tuple[str, ...] = ()
    force: bool = False  # 누적 조건을 무시한다


@dataclass
class LoopResult:
    status: Literal["skipped", "deployed", "passed", "rejected"]
    reason: str
    model_version: str | None = None
    examples: dict[str, int] = field(default_factory=dict[str, int])
    candidate: EvalReport | None = None
    baseline: EvalReport | None = None
    decision: GateDecision | None = None
    # 배포했으면 MLflow 레지스트리에 올릴 것 (이름, 실행, 출처, 별칭). DB 커밋 뒤에 올린다
    registration: tuple[str, str, str, str] | None = None
    mlflow_run_id: str | None = None


def raw_clips(raw: ObjectStore) -> ClipSource:
    """원본 저장소에서 영상을 받는다 (학습·평가 서버는 원본 권한자다)."""

    def fetch(session: Session, stream: Stream, work: Path) -> Path:
        dest = work / f"{session.session_id}__{stream.stream_id}"
        if not dest.exists():
            raw.get_file(stream.uri.removeprefix(raw.uri("")), dest)
        return dest

    return fetch


def _short(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:10]


def _key(store: ObjectStore, uri: str) -> str:
    return uri.removeprefix(store.uri(""))


def _download(store: ObjectStore, mv: ModelVersion, work: Path) -> Path:
    """레지스트리 모델의 산출물을 받고 해시를 확인한다."""
    key = _key(store, mv.artifact_uri)
    dest = work / mv.model_version / Path(key).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    store.get_file(key, dest)
    digest = sha256_file(dest)
    if digest != mv.sha256:
        raise TrainingError(f"{mv.model_version} 산출물 해시가 다릅니다: {digest}")
    return dest


def predict_golden(
    predictor: Predictor,
    task: Task,
    golden: list[GoldenSession],
    clips: ClipSource,
    work: Path,
) -> list[SessionData]:
    """후보·기존 모델을 골든셋 세션 영상에 돌린다 (DB에 쓰지 않는다).

    공간 과제는 영상 스트림마다, 타임라인 과제(접촉·행동 등, stream_id 없는 라벨)는 세션마다 기준
    스트림(바디캠)에 한 번 돌린다. 스트림마다 돌리면 같은 타임라인 구간이 겹쳐 오탐이 된다.
    """
    out: list[SessionData] = []
    for g in golden:
        pred: list[LabelRecord] = []
        streams = (
            [g.session.reference_stream]
            if task in TIMELINE_TASKS
            else [s for s in g.session.streams if s.kind in VIDEO]
        )
        for stream in streams:
            labels = predictor.run(
                Clip(g.session.session_id, stream.stream_id, clips(g.session, stream, work))
            )
            pred += [x for x in labels if matches(task, x)]
        truth = [x for x in g.truth if matches(task, x)]
        out.append(SessionData(g.session.session_id, truth, pred, g.groups))
    return out


def _truth_lookup(golden: list[GoldenSession]) -> Callable[[str, str], list[LabelRecord]]:
    """(세션, 스트림) → 그 스트림의 정답과 세션의 타임라인 정답(stream_id 없음).

    타임라인 과제는 predict_golden이 세션마다 한 번만 부르므로 겹치지 않는다.
    """
    by_key: dict[tuple[str, str | None], list[LabelRecord]] = {}
    for g in golden:
        for x in g.truth:
            by_key.setdefault((x.session_id, x.stream_id), []).append(x)
    return lambda sid, stream: by_key.get((sid, stream), []) + by_key.get((sid, None), [])


def run_training_job(
    conn: sa.Connection,
    job: TrainingJob,
    *,
    snapshots: SnapshotStore,
    artifacts: ObjectStore,
    tracker: Tracker,
    clips: ClipSource,
    policy: TrainingPolicy,
    eval_policy: EvaluationPolicy,
    now: datetime,
    trainers: dict[str, Trainer] | None = None,
    loaders: dict[str, ModelLoader] | None = None,
    stub_truth: bool = False,
) -> LoopResult:
    """stub_truth: oracle-stub처럼 정답이 필요한 로더에 골든 정답을 넘긴다 (CI·테스트 전용)."""
    trainers = TRAINERS if trainers is None else trainers
    loaders = LOADERS if loaders is None else loaders
    spec = policy.tasks.get(job.task)
    if spec is None:
        raise TrainingError(f"{job.task}: 학습 정책이 없습니다 (config/policies/training.yaml)")
    trainer_name = job.trainer or spec.trainer
    trainer, loader = trainers.get(trainer_name), loaders.get(trainer_name)
    if trainer is None or loader is None:
        raise TrainingError(f"학습기·로더가 없습니다: {trainer_name}")
    params = {**spec.params, **job.params}

    version = get_dataset_version(conn, job.dataset_version_id)
    if version.golden_set_version is None:
        raise TrainingError(
            f"{version.version_id}: 골든셋이 없는 데이터셋 버전은 평가할 수 없습니다"
        )
    golden_ids = set(get_golden_set(conn, version.golden_set_version).session_ids)
    data = load_training_data(
        snapshots, version, job.task, policy, excluded=withdrawn_sessions(conn, version, policy)
    )
    leaks = data.session_ids & golden_ids
    if leaks:
        raise TrainingError(f"골든셋 세션이 학습 데이터에 있습니다: {sorted(leaks)[:5]}")
    counts = data.counts()
    result = LoopResult("skipped", "", examples=counts)

    # 누적 확인
    previous = list_model_versions(conn, job.task)
    n = len(data.examples)
    if not job.force:
        if n < spec.min_examples:
            result.reason = f"학습 예제 {n}개 < min_examples {spec.min_examples}"
            return result
        if previous and n - previous[-1].train_examples < spec.min_new_examples:
            result.reason = (
                f"직전 학습({previous[-1].train_examples}개) 대비 새 예제 "
                f"{n - previous[-1].train_examples}개 < min_new_examples {spec.min_new_examples}"
            )
            return result
    golden = golden_sessions(conn, version.golden_set_version)
    if not any(matches(job.task, x) for g in golden for x in g.truth):
        result.reason = (
            f"골든셋 {version.golden_set_version}에 {job.task} 정답이 없어 평가할 수 없습니다"
        )
        return result
    has_deployed = bool(list_model_versions(conn, job.task, ModelStatus.DEPLOYED))
    if spec.replaces and not job.baseline_versions and not has_deployed:
        raise TrainingError(
            f"{job.task}: 배포되면 기본 어댑터 {list(spec.replaces)}를 대신하므로 비교할 "
            "기본 어댑터 예측 버전이 필요합니다 (--baseline-version, 골든셋에 그 예측이 있어야 함)"
        )
    # 배포 모델이 없으면 기본 어댑터의 DB 예측과 비교한다. 잘못 적은 버전은 예측이 비어 기존 지표가
    # 0에 가까워지고 어떤 후보든 통과하므로, 골든셋에 예측이 없는 버전이 하나라도 있으면 멈춘다.
    baseline_golden = (
        _merged_golden(conn, version.golden_set_version, job)
        if job.baseline_versions and not has_deployed
        else None
    )

    tag = _short(
        job.task,
        version.version_id,
        trainer_name,
        json.dumps(params, sort_keys=True),
        now.isoformat(),
    )
    run_id = f"train-{job.task}-{tag}"
    model_version = f"{job.task}-{trainer_name}-{tag}"
    mlflow_run = tracker.start(
        f"{policy.mlflow.experiment_prefix}{job.task}",
        run_id,
        {
            "dlp.task": job.task,
            "dlp.dataset_version": version.version_id,
            "dlp.golden_set": version.golden_set_version,
            "dlp.trainer": trainer_name,
            "dlp.model_version": model_version,
        },
    )
    result.mlflow_run_id = mlflow_run
    try:
        tracker.log_params(
            mlflow_run,
            {k: json.dumps(v) for k, v in sorted(params.items())}
            | {"dataset_version": version.version_id, "examples": str(n)},
        )
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out = trainer.train(data, params, work)
            digest = sha256_file(out.artifact)
            key = f"{policy.artifact_prefix}/{job.task}/{model_version}/{out.artifact.name}"
            artifacts.put_file(key, out.artifact, digest)
            tracker.log_artifact(mlflow_run, out.artifact, "model")
            tracker.log_metrics(mlflow_run, {f"train/{k}": v for k, v in out.metrics.items()})
            register_training_run(
                conn,
                TrainingRun(
                    run_id=run_id,
                    dataset_version_id=version.version_id,
                    model_name=job.task,
                    model_version=model_version,
                    mlflow_run_id=mlflow_run,
                    created_at=now,
                ),
            )
            candidate_mv = ModelVersion(
                model_version=model_version,
                task=job.task,
                run_id=run_id,
                trainer=trainer_name,
                artifact_uri=artifacts.uri(key),
                sha256=digest,
                train_examples=n,
                status=ModelStatus.CANDIDATE,
                created_at=now,
            )
            insert_model_version(conn, candidate_mv)
            result.model_version = model_version

            # 골든셋 평가 (후보와 기존 모델을 같은 코드로)
            ctx = LoadContext(
                now=now,
                ontology_version=version.ontology_version,
                truth=_truth_lookup(golden) if stub_truth else None,
            )
            cand_pred = loader.load(out.artifact, version=model_version, ctx=ctx)
            cand_data = predict_golden(cand_pred, job.task, golden, clips, work)
            candidate = evaluate(
                {job.task: cand_data},
                eval_policy,
                golden_version=version.golden_set_version,
                model_versions={job.task: model_version},
            )
            deployed = list_model_versions(conn, job.task, ModelStatus.DEPLOYED)
            # 비교한 배포 모델 (없으면 None). 승인 배포 때 지금 배포 모델과 같은지 본다
            compared_with = deployed[-1].model_version if deployed else None
            baseline: EvalReport | None = None
            if deployed:
                cur = deployed[-1]
                base_loader = loaders.get(cur.trainer)
                if base_loader is None:
                    raise TrainingError(
                        f"배포 모델 {cur.model_version}의 로더가 없습니다: {cur.trainer}"
                    )
                base_pred = base_loader.load(
                    _download(artifacts, cur, work), version=cur.model_version, ctx=ctx
                )
                base_data = predict_golden(base_pred, job.task, golden, clips, work)
                baseline = evaluate(
                    {job.task: base_data},
                    eval_policy,
                    golden_version=version.golden_set_version,
                    model_versions={job.task: cur.model_version},
                )
            elif baseline_golden is not None:
                baseline = evaluate(
                    {job.task: baseline_golden},
                    eval_policy,
                    golden_version=version.golden_set_version,
                    model_versions={job.task: "+".join(job.baseline_versions)},
                )
            decision = decide(candidate, baseline, eval_policy)
            if job.task not in candidate.overall:
                # 정답이 있어도 평가기가 지표를 못 내면(예: 키프레임이 모두 화면 밖) 배포하지 않는다
                decision = GateDecision(
                    False,
                    {job.task: TaskDecision(job.task, False, ["후보 지표를 계산하지 못했습니다"])},
                )
            result.candidate, result.baseline, result.decision = candidate, baseline, decision

            report = work / "report" / "golden.json"
            write_report(report, candidate, decision, {DEPLOYED_BASELINE: compared_with})
            report_key = f"{policy.artifact_prefix}/{job.task}/{model_version}/golden.json"
            artifacts.put_file(report_key, report, sha256_file(report))
            md = report.with_suffix(".md")
            artifacts.put_file(report_key.removesuffix(".json") + ".md", md, sha256_file(md))
            tracker.log_params(mlflow_run, {DEPLOYED_BASELINE: compared_with or "none"})
            tracker.log_artifact(mlflow_run, report, "golden")
            tracker.log_artifact(mlflow_run, md, "golden")
            metrics = candidate.overall[job.task].metrics if job.task in candidate.overall else {}
            tracker.log_metrics(
                mlflow_run,
                {f"golden/{k}": v for k, v in metrics.items()}
                | {"gate/passed": float(decision.passed)},
            )

        report_uri = artifacts.uri(report_key)
        if not decision.passed:
            set_model_status(conn, model_version, ModelStatus.REJECTED, now, report_uri)
            result.status = "rejected"
            result.reason = "; ".join(r for d in decision.tasks.values() for r in d.reasons)
        elif spec.deploy == "approve":
            set_model_status(conn, model_version, ModelStatus.PASSED, now, report_uri)
            result.status, result.reason = "passed", "게이트 통과, 사람 배포 승인 대기"
        else:
            mv = deploy(conn, model_version, now=now, report_uri=report_uri)
            result.registration = registration(conn, mv, policy)
            result.status, result.reason = "deployed", "게이트 통과, 배포"
        tracker.finish(mlflow_run, "FINISHED")
    except Exception:
        tracker.finish(mlflow_run, "FAILED")
        raise
    return result


def _merged_golden(conn: sa.Connection, golden_version: str, job: TrainingJob) -> list[SessionData]:
    """여러 기본 어댑터 버전의 예측을 세션별로 합친다 (대신할 어댑터 모두와 비교).

    골든셋에 예측이 하나도 없는 버전이 있으면 TrainingError (버전을 잘못 적었거나 골든셋에
    프리라벨을 돌리지 않았다).
    """
    try:
        merged = load_golden_merged(
            conn, golden_version, {job.task: list(job.baseline_versions)}, require_predictions=True
        )
    except MissingPredictionsError as exc:
        raise TrainingError(f"{exc} (--baseline-version)") from exc
    return merged[job.task]


def withdrawn_sessions(
    conn: sa.Connection, version: DatasetVersion, policy: TrainingPolicy
) -> set[str]:
    """학습 분할 세션 중 사용 중지(동의 철회 등)된 세션. 데이터셋 버전을 만든 뒤 철회됐어도 뺀다."""
    withdrawn = withdrawn_session_ids(conn)
    out: set[str] = set()
    for sid, split in version.splits.items():
        if split not in policy.splits:
            continue
        if sid in withdrawn or get_session(conn, sid).lifecycle_state is LifecycleState.WITHDRAWN:
            out.add(sid)
    return out


def deploy(
    conn: sa.Connection,
    model_version: str,
    *,
    now: datetime,
    report_uri: str | None = None,
    artifacts: ObjectStore | None = None,
) -> ModelVersion:
    """게이트를 통과한 모델(candidate 판정 직후 또는 passed)을 배포한다. 과제마다 배포는 하나다.

    passed 모델은 평가 리포트(report_uri, artifacts 저장소)에 적힌 비교 대상 배포 모델이 지금 배포
    모델과 다르면 배포하지 않는다 (지금 모델과 비교하지 않았으므로 다시 평가해야 한다). 판정 시각은
    학습 실행 시작 시각이라 비교에 쓰지 않는다. MLflow 등록은 DB 커밋 뒤에 register()로 한다.
    """
    mv = next((m for m in list_model_versions(conn) if m.model_version == model_version), None)
    if mv is None:
        raise TrainingError(f"모델 버전이 없습니다: {model_version}")
    if mv.status not in (ModelStatus.CANDIDATE, ModelStatus.PASSED):
        raise TrainingError(f"{model_version}은 배포할 수 없는 상태입니다: {mv.status.value}")
    if mv.status is ModelStatus.CANDIDATE and report_uri is None:
        raise TrainingError(f"{model_version}은 게이트 판정 전입니다")
    current = list_model_versions(conn, mv.task, ModelStatus.DEPLOYED)
    if mv.status is ModelStatus.PASSED:
        if artifacts is None:
            raise TrainingError(f"{model_version}: 승인 배포에는 평가 리포트 저장소가 필요합니다")
        compared = _compared_with(artifacts, mv)
        now_deployed = current[-1].model_version if current else None
        if compared != now_deployed:
            raise TrainingError(
                f"{model_version}은 게이트에서 {compared or '배포 모델 없음'}과 비교했지만 "
                f"지금 배포 모델은 {now_deployed or '없음'}입니다. "
                "그 모델과 비교하도록 다시 학습·평가하세요"
            )
    for cur in current:
        set_model_status(conn, cur.model_version, ModelStatus.RETIRED, now)
    set_model_status(conn, model_version, ModelStatus.DEPLOYED, now, report_uri)
    return mv


def _compared_with(artifacts: ObjectStore, mv: ModelVersion) -> str | None:
    """passed 모델의 평가 리포트에 적힌, 게이트에서 비교한 배포 모델 버전 (없었으면 None)."""
    if mv.report_uri is None:
        raise TrainingError(f"{mv.model_version}: 평가 리포트가 없습니다")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "golden.json"
        artifacts.get_file(_key(artifacts, mv.report_uri), path)
        data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if DEPLOYED_BASELINE not in data:
        raise TrainingError(
            f"{mv.model_version}: 평가 리포트에 비교한 배포 모델이 없습니다. 다시 학습·평가하세요"
        )
    value = data[DEPLOYED_BASELINE]
    return str(value) if value is not None else None


def registration(
    conn: sa.Connection, mv: ModelVersion, policy: TrainingPolicy
) -> tuple[str, str, str, str] | None:
    """MLflow 모델 레지스트리에 올릴 것 (이름, 실행, 출처, 별칭). MLflow 실행이 없으면 None."""
    run = get_training_run(conn, mv.run_id).mlflow_run_id
    if run is None:
        return None
    name = f"{policy.mlflow.registered_model_prefix}{mv.task}"
    return name, run, f"runs:/{run}/model", policy.mlflow.deployed_alias
