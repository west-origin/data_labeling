"""자동 프리라벨 하위 명령 (WP8, ADR 0008·0009·0026).

등록하는 명령:
- `dlp prelabel run <세션> [--store] [--url] [--skip 단계]` — 세션의 영상 스트림마다
  손(MediaPipe 21관절), 전신(RTMPose), COCO 객체(MediaPipe), 도구(OWLv2) 모델을 돌린다.
  이어 깊이 기반 3D 궤적, 장갑·영상 접촉 구간, 3인칭 착용자 매칭을 만들어
  `label_records`에 쓴다.

순서: `dlp privacy approve`(가능하면 `render`도) 뒤, `dlp relations run`·`dlp actions run` 전.
`privacy_approved` 세션이면 끝에 생애주기를 `prelabeled`로 옮긴다.

입력 → 출력: 원본 버킷 영상·장갑 데이터(감사 기록 `prelabel.run`) → 모델 출처 라벨.
라벨 ID에는 `version_tag(모델 버전)`이 들어가고, 모델 버전에는 정책 절 해시가 들어간다.

멱등성 (ADR 0015·0019):
- 같은 모델 버전 결과가 전체 이력(`get_labels`)에 있으면 그 (스트림, 모델)은 건너뛴다.
- 버전이 바뀌면 검수 전인 이전 버전 라벨만 `retractions()`로 지우고 새로 쓴다.
- 검수자가 승인·표본 검증한 라벨은 지우지 않는다.

재학습 모델: 게이트를 통과해 배포된 모델(`stage=prelabel`)이 있으면 `training.yaml`의
`replaces`에 적힌 기본 어댑터 대신 쓴다. 배포된 접촉 모델을 실제로 불러왔으면 기본 접촉 단계
(`CONTACT_STEP`)도 건너뛴다.

정책 출처: `config/policies/prelabel.yaml`(모델·임계값), `sync.yaml`의 `glove.pressure_prefixes`,
`training.yaml`(배포 과제).
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store
from dlp_cli.schema_cmds import database_url
from dlp_media.storage import store_from_spec
from dlp_prelabel.adapters.mediapipe_models import MediaPipeHands, MediaPipeObjects
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.adapters.rtmpose import RtmPose
from dlp_prelabel.adapters.stubs import UNAVAILABLE
from dlp_prelabel.lift3d import DepthLifter
from dlp_prelabel.policy import load_policy
from dlp_prelabel.runner import CONTACT_STEP, run_prelabel
from dlp_schema import load_config, load_ontology, repo_root
from dlp_schema.db.repository import list_model_versions
from dlp_schema.lineage import ModelStatus
from dlp_schema.predictor import ModelUnavailableError, Predictor
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_train.deployed import deployed_predictors
from dlp_train.policy import load_policy as load_training_policy
from dlp_train.trainers import LoadContext

CONTACT_TASK = "contact"  # config/policies/training.yaml tasks의 접촉 과제 이름


def cmd_run(args: argparse.Namespace) -> int:
    """`dlp prelabel run <세션>`: 자동 프리라벨을 실행한다.

    인자:
        args.session_id: 대상 세션 (온톨로지 버전이 있어야 한다).
        args.store: `s3` 또는 `local:<디렉터리>`. 원본·MLflow 버킷을 같은 지정으로 연다.
        args.url: DB URL.
        args.skip: 건너뛸 단계 이름 목록 (`hands`, `body`, `objects`, `tools`, `depth3d`).
            각 어댑터 클래스의 `name`과 비교한다.

    흐름:
    1. 기본 어댑터를 만든다. 가중치가 없으면(`ModelUnavailableError`) `[모델 없음]`만 출력하고 뺀다.
       아직 실제 모델이 없는 기능(`UNAVAILABLE`)은 `[미연동]`으로 알린다.
    2. 배포된 재학습 모델을 불러 `deployed.apply`로 기본 어댑터를 바꾼다.
    3. `run_prelabel`이 스트림·모델마다 라벨을 쓰고 요약을 돌려준다.

    반환: 0. 부작용: 한 트랜잭션에서 `label_records` INSERT(철회 레코드 포함)·세션 생애주기 갱신,
    원본 버킷 읽기(감사 기록). 온톨로지는 `config/ontology/v1`을 고정으로 읽는다 (리팩토링 후보).
    """
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
    buckets = load_config(root / "config/defaults.yaml").buckets
    raw = raw_store(args.store, args.url, "prelabel.run")
    engine = sa.create_engine(database_url(args.url))
    with engine.begin() as conn, tempfile.TemporaryDirectory() as tmp:
        # 게이트를 통과해 배포된 재학습 모델이 있으면 정책의 replaces 기본 어댑터 대신 쓴다
        deployed = deployed_predictors(
            conn,
            store_from_spec(args.store, buckets.mlflow),
            load_training_policy(root),
            "prelabel",
            LoadContext(now=now, ontology_version=ontology.version),
            Path(tmp),
        )
        for note in deployed.notes:
            print(f"[재학습 모델] {note}")
        replaced = set(deployed.replaces)
        # 배포된 재학습 접촉 모델(hand_state를 낸다)을 실제로 불러왔으면 기본 접촉 단계를 대신한다.
        # 둘 다 돌면 같은 접촉이 두 번 남는다 (training.yaml contact.replaces에 contact가 있으면
        # deployed.replaces로도 들어온다)
        contact_models = list_model_versions(conn, CONTACT_TASK, ModelStatus.DEPLOYED)
        loaded = {p.version for p in deployed.predictors}
        if contact_models and contact_models[-1].model_version in loaded:
            replaced.add(CONTACT_STEP)
            print(f"[재학습 모델] {CONTACT_TASK}: 기본 접촉 단계(장갑·영상 휴리스틱)를 대신합니다")
        s = run_prelabel(
            conn,
            args.session_id,
            raw,
            deployed.apply(predictors),
            policy,
            ontology,
            now,
            lifter=lifter,
            replaced=replaced,
            # 접촉 단계가 쓰는 장갑 압력 채널 (접촉 모델 버전에 들어간다)
            pressure_prefixes=load_sync_policy(
                root / "config/policies/sync.yaml"
            ).glove.pressure_prefixes,
        )
    engine.dispose()
    for key, n in s.produced.items():
        print(f"{key}: 라벨 {n}개")
    for key in s.skipped:
        print(f"{key}: 같은 모델 버전 결과가 있어 건너뜀")
    print(f"3D 궤적 {s.lifted}개, 접촉 구간 {s.contacts}개, 3인칭 착용자 {s.wearer or '-'}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`prelabel run` 하위 명령을 등록한다."""
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
