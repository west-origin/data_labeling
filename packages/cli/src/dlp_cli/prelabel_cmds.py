"""자동 프리라벨 하위 명령."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_prelabel.adapters.mediapipe_models import MediaPipeHands, MediaPipeObjects
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.adapters.rtmpose import RtmPose
from dlp_prelabel.adapters.stubs import UNAVAILABLE
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import load_policy
from dlp_prelabel.runner import run_prelabel
from dlp_schema import load_config, load_ontology, repo_root
from dlp_schema.predictor import ModelUnavailableError, Predictor


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    policy = load_policy(root)
    ontology = load_ontology(root / "config/ontology/v1")
    now = datetime.now(UTC)
    predictors: list[Predictor] = []
    for cls in (MediaPipeHands, RtmPose, MediaPipeObjects, OwlObjects):
        if cls.name in args.skip:
            continue
        try:
            predictors.append(cls(root, policy, ontology_version=ontology.version, now=now))
        except ModelUnavailableError as exc:
            print(f"[모델 없음] {cls.name}: {exc}")
    lifter: DepthLifter | None = None
    if DepthLifter.name not in args.skip:
        try:
            lifter = DepthLifter(root, policy, now=now)
        except ModelUnavailableError as exc:
            print(f"[모델 없음] {DepthLifter.name}: {exc}")
    for p in UNAVAILABLE:
        print(f"[미연동] {p.name}: {p.reason}")  # TODO(real-model) 표시가 붙은 기능
    raw = store_from_spec(args.store, load_config(root / "config/defaults.yaml").buckets.raw)
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn:
        s = run_prelabel(
            conn, args.session_id, raw, predictors, policy, ontology, now, lifter=lifter
        )
    engine.dispose()
    for key, n in s.produced.items():
        print(f"{key}: 라벨 {n}개")
    for key in s.skipped:
        print(f"{key}: 같은 모델 버전 결과가 있어 건너뜀")
    print(f"3D 궤적 {s.lifted}개, 접촉 구간 {s.contacts}개, 3인칭 착용자 {s.wearer or '-'}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    pre = sub.add_parser("prelabel", help="자동 프리라벨")
    psub = pre.add_subparsers(dest="prelabel_command", required=True)
    run = psub.add_parser("run", help="손·전신·객체·도구 모델, 3D 궤적, 접촉, 착용자 매칭")
    run.add_argument("session_id")
    run.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    run.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    run.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["hands", "body", "objects", "tools", "depth3d"],
        help="건너뛸 단계 (여러 번 쓸 수 있다). CPU에서 tools(OWLv2)는 프레임당 수 초가 걸린다",
    )
    run.set_defaults(func=cmd_run)
