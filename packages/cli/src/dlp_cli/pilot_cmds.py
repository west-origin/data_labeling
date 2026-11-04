"""실제 영상 파일럿의 실행 전 준비 검사 (`dlp pilot check`).

실제 영상을 처음 돌리기 전에 모델·서비스·검수 계정이 갖춰졌는지 한 번에 본다. 상태를 승인하거나
라벨·세션을 DB에 쓰지 않는다. 실제 영상의 정확도, 사람 검수 완료, 재학습 성능 향상은 이 검사로
입증되지 않는다.

하위 명령 (인자 오류는 argparse가 종료 코드 2로 끝낸다):

`dlp pilot check [--profile hands|cleaning] [--models-only] [--video PATH] [--url URL]`
    세 묶음을 차례로 검사하고 `[OK]`·`[FAIL]` 한 줄씩 출력한다.
    1. 모델 (`model_checks`). `pilot.yaml profiles.<profile>`(기본 `cleaning`)의 어댑터와 모든
       프라이버시 탐지기가 쓰는 모델의 라이선스 위반(`dlp_models.registry.load_registry`), 가중치
       해시 (`dlp_models.registry.resolve`)를 본다. 모두 통과하면 100 ms 합성 영상으로 어댑터 추론과
       계약 검사(`check_label`), 탐지기 초기화(`dlp_privacy.detectors.build_detectors`)까지 돌린다.
    2. 서비스 (`service_checks`, `--models-only`면 생략). 개발 서비스 응답(CVAT 포함), DB
       Alembic 리비전이 head인지, `review.yaml reviewers.privacy`가 비어 있지 않은지, 원본 검수자와
       `pilot.yaml labeler`의 CVAT 일반 계정 연결, Label Studio 서비스 인증과 검수 계정 연결을 본다.
    3. 입력 영상 (`--video`가 있을 때). 로컬 파일을 `dlp_media.probe.probe`로 열어 비디오 트랙,
       회전 메타데이터(`require_upright`), PTS 인덱스를 확인한다. 저장소 감사 경로를 거치지 않는
       로컬 진단이다 (`dlp media probe`와 같은 용도). 업로드·세션 등록은 없다.
    읽는 정책: `config/policies/pilot.yaml`, `prelabel.yaml`, `privacy.yaml`, `review.yaml`,
    `config/models.yaml`. 원본 버킷: 접근하지 않는다.
    종료 코드: 0 모두 OK, 1 하나라도 FAIL. 프로필의 어댑터 이름이 잘못되면 `ValueError`로
    종료 코드 1이다.

공개 이름:
- `PilotPolicy`, `load_policy(root)` — `pilot.yaml` 계약과 로더.
- `model_checks(root, profile)`, `service_checks(root, url)` — 위 1·2단계 검사 목록.
- `cmd_check`, `add_commands` — 실행 함수와 등록.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
import yaml
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from pydantic import Field

from dlp_cli.health import CheckResult, default_checks, run_checks
from dlp_cli.models_cmds import used_models
from dlp_cli.review_config import load_review_policy
from dlp_cli.schema_cmds import database_url
from dlp_fixtures.video import generate_blur_scenario
from dlp_media.probe import probe, require_upright
from dlp_media.pts import build_pts_index
from dlp_models.registry import load_registry, resolve
from dlp_prelabel.adapters.mediapipe_models import MediaPipeHands, MediaPipeObjects
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy.detectors import build_detectors
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.tasks import TaskError, label_studio_user_id
from dlp_schema import load_ontology, repo_root
from dlp_schema.common import Contract, Identifier
from dlp_schema.db.migrate import alembic_config
from dlp_schema.predictor import Clip
from dlp_schema.validation import check_label


class PilotPolicy(Contract):
    """pilot.yaml: 요청할 어댑터와 일반 라벨 검수자. 실제 권한은 검수 정책/도구에서 확인한다."""

    version: int
    profiles: dict[str, tuple[str, ...]]
    labeler: Identifier = Field(description="일반 라벨 검수자 ID")


def load_policy(root: Path) -> PilotPolicy:
    """`config/policies/pilot.yaml`을 읽어 검증한다."""
    data: Any = yaml.safe_load((root / "config/policies/pilot.yaml").read_text("utf-8"))
    return PilotPolicy.model_validate(data)


def model_checks(root: Path, profile: str) -> list[CheckResult]:
    """선택한 어댑터와 모든 프라이버시 탐지기에 필요한 모델을 검사한다. 실패는 구분해서 반환한다."""
    selected = load_policy(root).profiles[profile]
    adapters = (MediaPipeHands, MediaPipeObjects, OwlObjects)
    available = {cls.name for cls in adapters}
    if set(selected) - available or not selected:
        raise ValueError(f"pilot 프로필의 어댑터가 올바르지 않습니다: {selected}")
    roles = {
        "hands": ("hands",),
        "objects": ("coco_objects",),
        "tools": ("open_vocab", "open_vocab_tokenizer"),
    }
    where = used_models(root)
    names = {n for w, n in where.items() if w.startswith("privacy.")}
    for step in selected:
        names.update(where[f"prelabel.models.{role}"] for role in roles[step])
    registry = load_registry(root)
    results: list[CheckResult] = []
    for problem in registry.violations({n: n for n in names}):
        results.append(CheckResult("license", False, problem))
    for name in sorted(names):
        try:
            _, version = resolve(root, name)
            results.append(CheckResult(name, True, version))
        except Exception as exc:
            results.append(CheckResult(name, False, str(exc)))
    if any(not item.ok for item in results):
        return results
    policy = load_prelabel_policy(root)
    ontology = load_ontology(root / "config/ontology/v1")
    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "synthetic.mp4"
        generate_blur_scenario(0, duration_ms=100).write(video)
        for cls in adapters:
            if cls.name not in selected:
                continue
            try:
                adapter = cls(
                    root, policy, ontology_version=ontology.version, now=datetime.now(UTC)
                )
                labels = adapter.run(Clip("pilot-check", "bodycam", video))
                errors = [err for label in labels for err in check_label(label, ontology)]
                if errors:
                    raise ValueError(str(errors))
                results.append(
                    CheckResult(cls.name + "-runtime", True, "합성 영상 추론/계약 검사 통과")
                )
            except Exception as exc:
                results.append(CheckResult(cls.name + "-runtime", False, str(exc)))
        try:
            _, missing = build_detectors(load_privacy_policy(root), root)
            results.append(
                CheckResult(
                    "privacy-runtime",
                    not missing,
                    "탐지기 초기화 통과" if not missing else str(missing),
                )
            )
        except Exception as exc:
            results.append(CheckResult("privacy-runtime", False, str(exc)))
    return results


def service_checks(root: Path, url: str | None) -> list[CheckResult]:
    """서비스 접속만으로 준비됐다고 판정하지 않는다. DB 리비전과 검수 도구 인증도 확인한다."""
    results = run_checks(default_checks(include_cvat=True))
    engine = sa.create_engine(database_url(url))
    try:
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
            current = set(MigrationContext.configure(conn).get_current_heads())
            expected = set(
                ScriptDirectory.from_config(alembic_config(database_url(url))).get_heads()
            )
            results.append(
                CheckResult("db-schema", current == expected, f"리비전 {sorted(current)}")
            )
    except Exception as exc:
        results.append(CheckResult("db-schema", False, type(exc).__name__))
    finally:
        engine.dispose()
    review = load_review_policy(root)
    if not review.reviewers.privacy:
        results.append(CheckResult("privacy-reviewer", False, "원본 검수자를 설정해야 합니다"))
    try:
        cvat = CvatClient.from_env()
        try:
            for reviewer in {*review.reviewers.privacy, load_policy(root).labeler}:
                username = review.cvat.users.get(reviewer)
                uid = cvat.find_user_id(username) if username is not None else None
                ready = uid is not None and cvat.is_non_admin_reviewer(uid)
                results.append(CheckResult("cvat-account", ready, reviewer + " (활성 일반 계정)"))
        finally:
            cvat.http.close()
    except Exception as exc:
        results.append(CheckResult("cvat-auth", False, type(exc).__name__))
    ls = LabelStudioClient.from_env()
    try:
        ls.service_user_id()
        results.append(CheckResult("label-studio-auth", True, "서비스 계정 인증 통과"))
        results.append(_ls_account_check(ls, review, load_policy(root).labeler))
    except Exception as exc:
        results.append(CheckResult("label-studio-auth", False, type(exc).__name__))
    finally:
        ls.http.close()
    return results


def _ls_account_check(ls: LabelStudioClient, review: ReviewOpsPolicy, labeler: str) -> CheckResult:
    """일반 라벨 검수자의 Label Studio 계정 연결(`label_studio.users`)이 서비스 계정이 아닌 실제
    계정을 가리키는지 본다. 없으면 Label Studio 시간 검수 작업을 만들 수 없다 (F-REV-06)."""
    try:
        ready = label_studio_user_id(ls, review.label_studio.users, labeler) is not None
    except TaskError:
        ready = False
    return CheckResult("label-studio-account", ready, labeler + " (서비스 계정이 아닌 검수 계정)")


def cmd_check(args: argparse.Namespace) -> int:
    """`dlp pilot check`: 모델(→ 서비스 → --video) 검사 결과를 한 줄씩 출력한다.

    종료 코드: 모두 OK면 0, 하나라도 FAIL이면 1.
    """
    root = repo_root()
    results = model_checks(root, args.profile)
    if not args.models_only:
        results += service_checks(root, args.url)
    if args.video:
        try:
            path = Path(args.video)
            info = probe(path)
            if info.video is None:
                raise ValueError("비디오 트랙이 없습니다")
            require_upright(info)
            index = build_pts_index(path)
            results.append(
                CheckResult("video", True, f"{len(index)} 프레임, {index.duration_ms:.1f} ms")
            )
        except Exception as exc:
            results.append(CheckResult("video", False, str(exc)))
    for result in results:
        print(f"[{'OK' if result.ok else 'FAIL'}] {result.name}: {result.detail}")
    return 0 if all(item.ok for item in results) else 1


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`dlp pilot check` 하위 명령을 등록한다."""
    pilot = sub.add_parser("pilot", help="실제 영상 파일럿 준비 검사")
    commands = pilot.add_subparsers(dest="pilot_command", required=True)
    check = commands.add_parser("check", help="모델·도구·검수 계정 준비 검사")
    check.add_argument("--profile", choices=["hands", "cleaning"], default="cleaning")
    check.add_argument(
        "--models-only", action="store_true", help="모델 진단만 수행 (서비스 준비 판정 아님)"
    )
    check.add_argument("--video", help="입력 영상 진단 (로컬 파일, 업로드/세션 등록 없음)")
    check.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    check.set_defaults(func=cmd_check)
