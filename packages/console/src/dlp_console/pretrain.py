"""외부에서 학습한 가중치 가져오기 (ADR 0060 1단계 산출물, ADR 0061 6절).

Colab·Kaggle(scripts/pretrain/objects_pretrain.ipynb)에서 공개 데이터만으로 학습한 YOLOX ONNX와
`model_manifest.json`·`data_manifest.json`을 받아:

1. 검사한다 (`check_bundle`): manifest 종류, 데이터 manifest 해시 연결, ONNX sha256,
   `decode_in_inference` false, 클래스 순서(= `config/pretrain/objects.yaml` classes 키 순서 =
   `prelabel.yaml closed_vocab_tools.classes`), 입력 크기, 학습 데이터셋이 공개 데이터셋 목록에 있고
   forbidden이 아님.
2. `config/models.yaml` 항목을 만든다 (`plan_entries`): path·sha256·license·training_data·
   training_datasets·base_model·commercial. 분류는 데이터셋·출발 가중치 중 가장 엄한 값이고,
   `dlp_models.datasets.dataset_violations`를 메모리 안의 새 목록으로 돌려 통과해야 한다.
3. 콘솔은 Git이 관리하는 설정 파일을 고치지 않는다 (ADR 0061 6절). 적용할 내용 미리보기와 패치 파일
   (작업 디렉터리), `git apply`·`make licenses` 명령을 안내한다. 운영자가 확인하면 가중치를
   `data/models/pretrain/`(Git 밖)에 두고 가져오기 기록(`pretrain_imports.jsonl`, 추가만)을 남긴다.

우리 데이터(블러본·라벨·골든셋)를 밖으로 내보내는 기능은 없다 (ADR 0060 6절: 2단계는 사내 전용).
가중치를 읽는 로더·학습기는 만들지 않는다 (dlp_prelabel·dlp_train의 YOLOX 어댑터를 쓴다).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from dlp_models.datasets import (
    SEVERITY,
    DatasetRegistry,
    dataset_violations,
    load_datasets,
    load_pretrain_objects,
)
from dlp_models.registry import Commercial, ModelRegistry

MODEL_KIND = "dlp-pretrain-objects-model"
DATA_KIND = "dlp-pretrain-objects-data"
# 출발 가중치(YOLOX COCO)의 라이선스·학습 데이터 (config/models.yaml yolox_m_coco와 같은 근거)
BASE_LICENSE = "Apache-2.0"
BASE_TRAINING_DATA = (
    "COCO 2017 (주석 CC BY 4.0, 이미지는 Flickr 개별 CC 라이선스로 CC BY-NC 계열 포함)"
)
BASE_COMMERCIAL: Commercial = "review"
# manifest `train_settings.model`(구조 이름, 예: yolox_s). models.yaml 키·가중치 파일 이름에 그대로
# 들어가므로 영어 snake_case 식별자만 받는다 (F-COE-77: 경로 이탈·YAML 주입 방지).
ARCH = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
# models.yaml 키 (가져온 모델 이름 = `<구조>_objects_pretrain_<해시 8자>`)
MODEL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
# 출발 가중치 url: 공개 https 주소만 (공백·제어 문자 없음)
BASE_URL = re.compile(r"^https://[!-~]{1,2000}$")
DEFAULT_ARCH = "yolox_s"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass
class ImportPlan:
    """검사 결과와 적용할 내용 (화면 미리보기)."""

    import_id: str
    checks: list[Check]
    onnx_sha256: str
    model_name: str
    dest: str  # 저장소 루트 기준 가중치 경로
    entries: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    commercial: str | None = None
    license_problems: list[str] = field(default_factory=list[str])
    models_patch: str = ""
    activation_patch: str = ""
    notes: list[str] = field(default_factory=list[str])

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks) and not self.license_problems

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text("utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    """dict이면 그대로, 아니면 빈 dict (JSON·YAML의 빠진 절)."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def manifest_arch(model: dict[str, Any]) -> str | None:
    """`train_settings.model` 값. 식별자 모양이 아니면 None (빠지면 기본 yolox_s)."""
    value = _dict(model.get("train_settings")).get("model", DEFAULT_ARCH)
    return value if isinstance(value, str) and ARCH.match(value) else None


def _manifest_checks(model: dict[str, Any]) -> list[Check]:
    """설정 파일·경로에 들어가는 manifest 문자열의 모양 (F-COE-77)."""
    start = _dict(model.get("start_weights"))
    url = start.get("url")
    return [
        Check(
            "구조 이름 (train_settings.model)",
            manifest_arch(model) is not None,
            f"{ARCH.pattern} 모양만 받는다 (경로·줄바꿈 금지)",
        ),
        Check(
            "출발 가중치 형식",
            isinstance(start.get("sha256"), str)
            and bool(SHA256.match(str(start.get("sha256"))))
            and (url is None or (isinstance(url, str) and bool(BASE_URL.match(url)))),
            "start_weights.sha256은 소문자 16진수 64자, url은 공백 없는 https 주소",
        ),
    ]


def _closed_tools_policy(root: Path) -> dict[str, Any]:
    return _dict(
        _dict(_yaml(root / "config" / "policies" / "prelabel.yaml")).get("closed_vocab_tools")
    )


def _kind_checks(model: dict[str, Any], data: dict[str, Any]) -> list[Check]:
    return [
        Check(
            "모델 manifest 종류",
            model.get("kind") == MODEL_KIND and model.get("task") == "objects",
            f"kind={model.get('kind')}, task={model.get('task')} (기대 {MODEL_KIND}, objects)",
        ),
        Check(
            "데이터 manifest 종류",
            data.get("kind") == DATA_KIND and data.get("dry_run") is False,
            f"kind={data.get('kind')}, dry_run={data.get('dry_run')} (dry-run 결과는 받지 않는다)",
        ),
    ]


def _hash_checks(model: dict[str, Any], onnx_sha: str, data_sha: str) -> list[Check]:
    onnx = _dict(model.get("onnx"))
    return [
        Check(
            "ONNX sha256",
            onnx.get("sha256") == onnx_sha,
            f"올린 파일 {onnx_sha[:12]}…, manifest {str(onnx.get('sha256'))[:12]}…",
        ),
        Check(
            "데이터 manifest 연결",
            model.get("data_manifest_sha256") == data_sha,
            "모델 manifest의 data_manifest_sha256과 올린 data_manifest.json 해시 비교",
        ),
        Check(
            "decode_in_inference 끔",
            onnx.get("decode_in_inference") is False,
            f"decode_in_inference={onnx.get('decode_in_inference')} "
            "(prelabel.yaml decoded_output: false와 맞춤)",
        ),
    ]


def _class_checks(root: Path, model: dict[str, Any], data: dict[str, Any]) -> list[Check]:
    expected = list(load_pretrain_objects(root).classes)
    policy = _closed_tools_policy(root)
    onnx_classes = _list(_dict(model.get("onnx")).get("classes"))
    settings = _dict(model.get("train_settings"))
    return [
        Check(
            "클래스 순서 (ONNX = objects.yaml)",
            onnx_classes == expected,
            f"ONNX {len(onnx_classes)}개, objects.yaml {len(expected)}개 (순서까지 같아야 한다)",
        ),
        Check(
            "클래스 순서 (objects.yaml = prelabel.yaml)",
            _list(policy.get("classes")) == expected,
            "prelabel.yaml closed_vocab_tools.classes와 비교",
        ),
        Check(
            "데이터 manifest 클래스",
            _list(data.get("classes")) == expected,
            "data_manifest.json classes와 비교",
        ),
        Check(
            "입력 크기",
            settings.get("input_size") == policy.get("input_size"),
            f"학습 {settings.get('input_size')}, 정책 {policy.get('input_size')}",
        ),
        Check(
            "정책 decoded_output",
            policy.get("decoded_output") is False,
            f"prelabel.yaml closed_vocab_tools.decoded_output={policy.get('decoded_output')}",
        ),
    ]


def dataset_names(data: dict[str, Any]) -> list[str]:
    return sorted(str(k) for k in _dict(data.get("datasets")))


def _dataset_checks(root: Path, data: dict[str, Any]) -> list[Check]:
    registry = load_datasets(root)
    names = dataset_names(data)
    unknown = [n for n in names if n not in registry.datasets]
    forbidden = [
        n
        for n in names
        if n in registry.datasets and registry.datasets[n].commercial == "forbidden"
    ]
    return [
        Check("학습 데이터셋 목록", bool(names) and not unknown, f"{names}, 목록에 없음 {unknown}"),
        Check("forbidden 데이터셋 없음", not forbidden, f"forbidden {forbidden}"),
        Check(
            "주석 전체 받음",
            data.get("partial_annotations") is False,
            f"partial_annotations={data.get('partial_annotations')}",
        ),
    ]


def check_bundle(
    root: Path, onnx_path: Path, model_manifest: Path, data_manifest: Path
) -> tuple[list[Check], dict[str, Any], dict[str, Any], str]:
    """세 파일 검사. 반환: (검사 목록, 모델 manifest, 데이터 manifest, ONNX sha256)."""
    try:
        model: dict[str, Any] = json.loads(model_manifest.read_text("utf-8"))
        data: dict[str, Any] = json.loads(data_manifest.read_text("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return [Check("manifest JSON", False, f"읽지 못함: {exc}")], {}, {}, sha256_file(onnx_path)
    onnx_sha = sha256_file(onnx_path)
    checks = [
        *_kind_checks(model, data),
        *_hash_checks(model, onnx_sha, sha256_file(data_manifest)),
        *_class_checks(root, model, data),
        *_dataset_checks(root, data),
        *_manifest_checks(model),
        base_check(root, model),
    ]
    return checks, model, data, onnx_sha


def _base_entry(model: dict[str, Any], base_name: str, arch: str) -> dict[str, Any]:
    start = _dict(model.get("start_weights"))
    return {
        "path": f"data/models/yolox/{arch}_coco.pth",
        "url": start.get("url"),
        "sha256": start.get("sha256"),
        "license": BASE_LICENSE,
        "training_data": [BASE_TRAINING_DATA],
        "commercial": BASE_COMMERCIAL,
        "_note": f"{base_name}: 1차 학습의 출발 가중치 (계보 기록용, 프리라벨 정책은 쓰지 않는다)",
    }


def _base_name(known: dict[str, Any], model: dict[str, Any], arch: str) -> str:
    """출발 가중치의 models.yaml 이름: 같은 sha256으로 이미 등록된 항목, 없으면 `<arch>_coco_pth`.

    `<arch>_coco_pth`는 yolox 학습기의 출발 가중치 이름 규칙(training.yaml yolox.base_model)과 같다.
    """
    sha = _dict(model.get("start_weights")).get("sha256")
    for name, entry in known.items():
        if sha and _dict(entry).get("sha256") == sha:
            return str(name)
    return f"{arch}_coco_pth"


def base_check(root: Path, model: dict[str, Any]) -> Check:
    """출발 가중치가 등록된 항목과 다른 파일이 아닌가 (같은 이름에 다른 sha256이면 실패)."""
    known = _dict(_dict(_yaml(root / "config" / "models.yaml")).get("models"))
    arch = manifest_arch(model)
    if arch is None:
        return Check("출발 가중치 등록", False, "구조 이름이 올바르지 않아 확인하지 못함")
    name = _base_name(known, model, arch)
    sha = _dict(model.get("start_weights")).get("sha256")
    registered = _dict(known.get(name)).get("sha256")
    ok = registered is None or registered == sha
    return Check(
        "출발 가중치 등록", ok, f"{name}: manifest {str(sha)[:12]}…, 등록 {str(registered)[:12]}…"
    )


def plan_entries(
    root: Path, model: dict[str, Any], data: dict[str, Any], onnx_sha: str, models_dir: str
) -> tuple[str, str, dict[str, dict[str, Any]], str, list[str]]:
    """(모델 이름, 가중치 경로, 새 models.yaml 항목들, 분류, 위반 메시지).

    구조 이름이 식별자 모양이 아니거나 가중치 경로가 `models_dir` 밖이면 항목 없이 위반만 돌려준다
    (F-COE-77).
    """
    raw = _dict(_yaml(root / "config" / "models.yaml"))
    known = _dict(raw.get("models"))
    arch = manifest_arch(model)
    if arch is None or not SHA256.match(onnx_sha):
        return "", "", {}, BASE_COMMERCIAL, ["train_settings.model이 식별자 모양이 아닙니다"]
    base_name = _base_name(known, model, arch)
    name = f"{arch}_objects_pretrain_{onnx_sha[:8]}"
    dest = f"{models_dir.rstrip('/')}/{name}.onnx"
    if not dest_inside(root, models_dir, dest):
        return "", "", {}, BASE_COMMERCIAL, [f"가중치 경로가 {models_dir} 밖입니다"]
    entries: dict[str, dict[str, Any]] = {}
    if base_name not in known:
        entries[base_name] = _base_entry(model, base_name, arch)
    registry_ds = load_datasets(root)
    names = dataset_names(data)
    floor: list[Commercial] = [
        registry_ds.datasets[n].commercial for n in names if n in registry_ds.datasets
    ]
    base = _dict(known.get(base_name)) or entries.get(base_name, {})
    floor.append(cast(Commercial, base.get("commercial", BASE_COMMERCIAL)))
    commercial: Commercial = max(floor, key=SEVERITY.__getitem__)
    entries[name] = {
        "path": dest,
        # models.yaml에는 '로컬에서 가져온 파일' 출처 형식이 없다 (F-COE-75). 가져온 사본 위치를
        # url로 적어 make models가 해시만 확인하게 한다. 다른 컴퓨터는 이 파일을 사내 보관소에서
        # 같은 경로에 둔다.
        "url": f"file://{(root / dest).resolve()}",
        "sha256": onnx_sha,
        "license": BASE_LICENSE,
        "training_data": [_dataset_label(data, n) for n in names],
        "training_datasets": names,
        "base_model": base_name,
        "commercial": commercial,
    }
    problems = _license_problems(raw, entries, registry_ds)
    return name, dest, entries, commercial, problems


def dest_inside(root: Path, models_dir: str, dest: str) -> bool:
    """`dest`(저장소 루트 기준)가 resolve 뒤 `models_dir` 안이고 그 디렉터리가 저장소 안인가."""
    if Path(dest).is_absolute() or Path(models_dir).is_absolute():
        return False
    base = root.resolve()
    folder = (base / models_dir).resolve()
    target = (base / dest).resolve()
    return folder.is_relative_to(base) and target.is_relative_to(folder) and target != folder


def _dataset_label(data: dict[str, Any], name: str) -> str:
    """models.yaml training_data 한 줄 (이름·판, 라이선스는 external_datasets.yaml)."""
    version = str(_dict(_dict(data.get("datasets")).get(name)).get("version", "")).strip()
    return f"{name} {version} (라이선스: config/external_datasets.yaml)".replace("  ", " ")


def _clean(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def _license_problems(
    raw: Any, entries: dict[str, dict[str, Any]], datasets: DatasetRegistry
) -> list[str]:
    """새 항목을 더한 목록을 메모리에서 검증하고 데이터셋 분류를 검사한다 (파일 쓰기 없음)."""
    merged = _dict(raw).copy()
    merged["models"] = {**_dict(merged.get("models")), **{k: _clean(v) for k, v in entries.items()}}
    try:
        registry = ModelRegistry.model_validate(merged)
    except ValueError as exc:
        return [f"models.yaml 형식: {exc}"]
    # 새 항목에 대한 위반만 (기존 항목의 위반은 make licenses가 따로 알린다)
    return [p for p in dataset_violations(registry, datasets) if p.split(":")[0] in entries]


def entries_yaml(entries: dict[str, dict[str, Any]]) -> str:
    """models.yaml `models:` 아래에 넣을 YAML 조각 (두 칸 들여쓰기)."""
    lines: list[str] = []
    for name, entry in entries.items():
        note = entry.get("_note")
        if note:
            # 주석 한 줄에 줄바꿈·제어 문자가 섞이면 YAML 본문이 된다 (F-COE-77): 공백으로 바꾼다
            lines.append("  # " + "".join(c if c.isprintable() else " " for c in str(note)))
        body = yaml.safe_dump({name: _clean(entry)}, allow_unicode=True, sort_keys=False)
        lines += ["  " + x for x in body.splitlines()]
    return "\n".join(lines) + "\n"


def models_patch(root: Path, entries: dict[str, dict[str, Any]]) -> str:
    """config/models.yaml에 항목을 넣는 unified diff (`external:` 절 앞, 없으면 끝)."""
    path = root / "config" / "models.yaml"
    old = path.read_text("utf-8").splitlines(keepends=True)
    snippet = entries_yaml(entries).splitlines(keepends=True)
    at = next((i for i, x in enumerate(old) if x.startswith("external:")), len(old))
    # external 절 앞의 주석·빈 줄 묶음 앞에 넣는다
    while at > 0 and (old[at - 1].startswith("#") or not old[at - 1].strip()):
        at -= 1
    new = [*old[:at], *snippet, *old[at:]]
    return "".join(difflib.unified_diff(old, new, "a/config/models.yaml", "b/config/models.yaml"))


def activation_patch(root: Path, model_name: str) -> str:
    """closed_tools 단계를 켜는 diff (prelabel.yaml models.closed_vocab_tools). 켜졌으면 빈 글."""
    path = root / "config" / "policies" / "prelabel.yaml"
    old = path.read_text("utf-8").splitlines(keepends=True)
    new = [
        x.replace("closed_vocab_tools: null", f"closed_vocab_tools: {model_name}")
        if x.strip().startswith("closed_vocab_tools: null")
        else x
        for x in old
    ]
    if new == old:
        return ""
    return "".join(
        difflib.unified_diff(
            old, new, "a/config/policies/prelabel.yaml", "b/config/policies/prelabel.yaml"
        )
    )


def training_replaces_closed_tools(root: Path) -> bool:
    """training.yaml objects.replaces에 closed_tools가 있는가 (재학습 모델이 대신하는지)."""
    tasks = _dict(_dict(_yaml(root / "config" / "policies" / "training.yaml")).get("tasks"))
    return "closed_tools" in _list(_dict(tasks.get("objects")).get("replaces"))


def build_plan(
    root: Path,
    import_id: str,
    onnx: Path,
    model_manifest: Path,
    data_manifest: Path,
    models_dir: str,
) -> ImportPlan:
    """검사 + 적용 미리보기."""
    checks, model, data, onnx_sha = check_bundle(root, onnx, model_manifest, data_manifest)
    plan = ImportPlan(
        import_id=import_id, checks=checks, onnx_sha256=onnx_sha, model_name="", dest=""
    )
    if not model or not data:
        return plan
    name, dest, entries, commercial, problems = plan_entries(
        root, model, data, onnx_sha, models_dir
    )
    plan.model_name, plan.dest, plan.entries = name, dest, entries
    plan.commercial, plan.license_problems = commercial, problems
    if not entries:
        return plan
    plan.models_patch = models_patch(root, entries)
    plan.activation_patch = activation_patch(root, name)
    if not training_replaces_closed_tools(root):
        plan.notes.append(
            "training.yaml objects.replaces에 closed_tools가 없습니다. "
            "재학습 모델이 배포되면 이 단계와 겹칠 수 있습니다."
        )
    plan.notes.append(
        "공개 데이터 검증 점수는 참고값입니다. 정책에 켜기 전에 우리 골든셋 비교(게이트)를 "
        "통과해야 합니다 (ADR 0060 5절)."
    )
    return plan
