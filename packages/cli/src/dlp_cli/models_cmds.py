"""모델 가중치 내려받기·변환과 라이선스 검사 (config/models.yaml, ADR 0009·0010). 가중치는 저장소에
넣지 않는다.

등록하는 명령:
- `dlp models fetch [이름...]` — `config/models.yaml`의 가중치를 `data/models/`에 받고 sha256을
  확인한다 (`make models`). 변환형 모델(`export` 지정)은 받지 않고 "변환 필요"만 알린다.
- `dlp models export` — 공개 ONNX가 없는 모델(메트릭 깊이 등)을 공식 가중치에서 ONNX로 변환한다
  (`make export-models`). 무거운 PyTorch가 필요해 일회용 `uv run --no-project` 환경에서 돈다.
- `dlp models licenses` — 정책이 쓰는 모델의 가중치 라이선스·학습 데이터·상업 사용 분류
  표(Markdown)를 출력하고, 허용 분류가 아닌 모델이 있으면 실패한다 (`make licenses`, `make check`에
  포함).

공개 함수:
- `used_models(root)` — 정책 YAML(privacy·prelabel·actions)이 실제로 쓰는 모델을 "쓰는 곳 → 모델
  이름"으로 모은다. `tests/test_model_licenses.py`도 이 함수를 쓴다.

주의: 라벨링 데이터를 판매하므로 가중치가 비상업(`forbidden`)이면 쓰지 않는다 (ADR 0010). 새 모델을
정책에 넣으면 `config/models.yaml`에 라이선스·학습 데이터·`commercial`을 함께 적어야 `licenses`가
통과한다.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from dlp_actions.policy import load_policy as load_actions_policy
from dlp_models.registry import fetch, load_registry
from dlp_prelabel.policy import load_policy as load_prelabel_policy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_schema import repo_root

# 변환 스크립트는 무거운 PyTorch가 필요해 프로젝트 환경과 따로 일회용 환경에서 돌린다.
# CPU판 PyTorch 색인을 추가하고(`unsafe-best-match`로 PyPI와 섞어 고른다), 버전은 재현을 위해
# 고정한다.
# 이 목록 뒤에 변환 스크립트 경로(`scripts/export_*.py`)를 붙여 실행한다.
EXPORT_ENV = [
    "uv", "run", "--no-project", "--python", "3.12",
    "--with", "torch==2.9.1", "--with", "transformers==4.57.1", "--with", "onnx",
    "--index", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match",
    "python",
]  # fmt: skip


def used_models(root: Path) -> dict[str, str]:
    """정책이 쓰는 모델: 쓰는 곳 → config/models.yaml 이름.

    키 형식:
    - `privacy.detectors.<탐지기>.model|tokenizer` — 프라이버시 대상(`targets`)이 쓰는 탐지기와, 그
      탐지기가 다시 쓰는 하위 탐지기(`region_detector`, `face_detector`)까지 따라간다.
    - `prelabel.models.<역할>` — 프리라벨 정책의 `models` 절 전부.
    - `actions.vlm.model` — 행동 분류 VLM.

    인자:
        root: 저장소 루트 (`config/policies/*.yaml`을 읽는다).
    반환: 쓰는 곳 → 모델 이름 사전. 정책에 없는 탐지기 이름은 조용히 건너뛴다.
    """
    used: dict[str, str] = {}
    privacy = load_privacy_policy(root)
    names = {n for tp in privacy.targets.values() for n in tp.detectors}
    while names:  # reflection처럼 다른 탐지기를 쓰는 탐지기를 따라간다
        name = names.pop()
        spec = privacy.detectors.get(name)
        if spec is None:
            continue
        for field in ("model", "tokenizer"):
            if (model := getattr(spec, field)) is not None:
                used[f"privacy.detectors.{name}.{field}"] = model
        # 하위 탐지기를 작업 목록에 넣는다. 이미 처리한 이름이 다시 들어와도 결과 키가 같아
        # 덮어쓸 뿐이며, 순환 참조가 있으면 끝나지 않는다 (현재 정책에는 순환이 없다)
        names |= {n for n in (spec.region_detector, spec.face_detector) if n is not None}
    for role, model in load_prelabel_policy(root).models.model_dump().items():
        used[f"prelabel.models.{role}"] = model
    used["actions.vlm.model"] = load_actions_policy(root).vlm.model
    return used


def cmd_licenses(args: argparse.Namespace) -> int:
    """정책이 쓰는 모델의 라이선스·학습 데이터·상업 사용 분류 (판매 실사 자료). 위반이 있으면 1.

    표에는 레지스트리의 모든 모델(`models`: 가중치 파일, `external`: 외부 서비스·바이너리)을 싣고,
    정책이 쓰지 않는 모델은 "쓰는 곳"이 `-`다. 위반 = 정책이 쓰는데 목록에 없거나 `accept`에 없는
    분류.
    """
    root = repo_root()
    registry = load_registry(root)
    used = used_models(root)
    print("| 모델 | 쓰는 곳 | 가중치 라이선스 | 학습 데이터 | 상업 사용 |")
    print("| --- | --- | --- | --- | --- |")
    for name, spec in [*registry.models.items(), *registry.external.items()]:
        where = ", ".join(sorted(w for w, n in used.items() if n == name)) or "-"
        data = "; ".join(spec.training_data) or "-"
        print(f"| {name} | {where} | {spec.license} | {data} | {spec.commercial} |")
    problems = registry.violations(used)
    for p in problems:
        print(f"[위반] {p}")
    print(f"허용 분류: {', '.join(registry.accept)}")
    return 1 if problems else 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """`dlp models fetch [이름...]`: 가중치를 내려받고 sha256을 확인한다.

    인자:
        args.names: 받을 모델 이름 목록. 비어 있으면 레지스트리 `models` 전부.

    반환: 하나라도 `[실패]`(해시 불일치)면 1, 아니면 0. 네트워크 오류는 예외로 끝난다.
    부작용: `<저장소>/<spec.path>`(보통 `data/models/…`)에 파일 쓰기. 이미 있고 해시가 맞으면
    건너뛴다(멱등).
    """
    root = repo_root()
    failed = 0
    for name, spec in load_registry(root).models.items():
        if args.names and name not in args.names:
            continue
        message = fetch(root, name, spec)
        # fetch는 "[받음] / [있음] / [변환 필요] / [실패]" 접두어 메시지를 돌려준다
        failed += message.startswith("[실패]")
        print(message)
    return 1 if failed else 0


def cmd_export(args: argparse.Namespace) -> int:
    """`dlp models export`: 변환형 모델(`spec.export` 스크립트 지정) 중 결과 파일이 없는 것을
        변환한다.

    부작용: 일회용 PyTorch 환경(`EXPORT_ENV`)에서 `scripts/export_*.py`를 실행해 `spec.path`에
    ONNX를 쓴다. 이미 파일이 있으면 건너뛴다. 변환 결과의 해시는 여기서 확인하지 않는다.
    예외: 변환 스크립트가 실패하면 `subprocess.CalledProcessError`.
    """
    root = repo_root()
    for name, spec in load_registry(root).models.items():
        if spec.export is None or (root / spec.path).is_file():
            continue
        print(f"[변환] {name}: {spec.export}")
        subprocess.run([*EXPORT_ENV, str(root / spec.export)], cwd=root, check=True)
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`models fetch|licenses|export` 하위 명령을 등록한다."""
    models = sub.add_parser("models", help="모델 가중치")
    msub = models.add_subparsers(dest="models_command", required=True)
    fetch_p = msub.add_parser("fetch", help="config/models.yaml의 가중치를 받고 해시를 확인")
    fetch_p.add_argument("names", nargs="*", help="받을 모델 이름 (기본: 전부)")
    fetch_p.set_defaults(func=cmd_fetch)
    lic = msub.add_parser("licenses", help="모델 라이선스 표와 상업 사용 검사 (위반 시 실패)")
    lic.set_defaults(func=cmd_licenses)
    export = msub.add_parser("export", help="공개 ONNX가 없는 모델을 공식 가중치에서 변환")
    export.set_defaults(func=cmd_export)
