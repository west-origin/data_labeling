"""AI 1차 검수 패키지의 경계 정적 검사 (WP17, ADR 0040 I1·I3·I4).

두 가지 방식으로 훑는다.

- 줄 단위 텍스트 (주석·docstring 포함, 기존 원본 접근 검사와 같은 방식): 판정은 라벨도 검수 상태도
  아니므로 `packages/aireview/src`는 라벨·검수·생애주기 기록 함수와 라벨 이력 표 이름, 원본
  버킷·프록시 이름을 쓰지 않는다. 픽셀을 얻는 호출(저장소 파일 받기, 블러본 프레임 객체, 영상
  디코드·파일 열기)은 블러본 경계 모듈(`media.py`, WP17.5)에서만 한다.
- import 문 (ast, 모듈 이름과 가져오는 이름을 함께 본다): `from dlp_media import audit`처럼 점 없이
  가져오는 형태와 `from dlp_schema.db import aireview_repository`처럼 이름으로 가져오는 형태도
  잡는다.
  - aireview는 CLI·검수 패키지·감사 원본 저장소(`dlp_media.audit`)를 import하지 않고, 영상 디코드
    (`av`, `dlp_media.frames.decode_frames`)는 `media.py`에서만 import한다.
  - 내보내기·학습·데이터셋·액티브 러닝은 AI 판정 계약·저장소·패키지(이름에 aireview가 든 모듈이나
    이름)를 import하지 않는다 (판정이 학습·판매 데이터에 섞이지 않게).

WP17.5에서 더한 검사
- 블러본 경계 모듈(`media.py`)은 원본 버킷 이름 비교(`labeling.bucket == raw_bucket`)와 렌더 게이트
  (`expected_render_hash`·`assert_render_current`·`check_fetched`)를 거친다 (지우면 실패).
- 실행기(`runner.py`)는 프레임을 `media.frames_for`로만 얻고 영상 파일을 직접 열지 않는다.
- CLI의 AI 검수 명령(`dlp_cli/aireview*.py`)은 원본 저장소를 용도 `aireview.run`으로만 만든다
  (장갑 Parquet, 월간 감사가 이상 접근으로 표시하지 않게 `ops.yaml audit.service_purposes`에
  등록된 값).

정답은 저장소 소스 자체다. 서비스 없이 돈다.
"""

from __future__ import annotations

import ast
from pathlib import Path

from dlp_schema import repo_root

ROOT = repo_root()
SRC = ROOT / "packages" / "aireview" / "src" / "dlp_aireview"
# 판정 패키지에 나오면 안 되는 이름 (쓰기 함수, 라벨 이력 표, 원본·프록시 경로)
FORBIDDEN = (
    "insert_labels", "record_review", "set_lifecycle", "update_assignment", "label_records",
    "proxy_key", "raw_store", '"dlp-raw"',
)  # fmt: skip
# 블러본 경계 모듈 밖에서 쓰면 안 되는 것 (저장소에서 파일 받기, 픽셀 객체 만들기, 영상 디코드·열기)
MEDIA_ONLY = (
    "get_file(", "BlurredFrame(", "decode_frames(", "av.open(", "cv2.imread", "VideoCapture",
    "imageio", "Image.open(",
)  # fmt: skip
# AI 판정을 import하면 안 되는 패키지 (학습·판매 데이터 경로)
CLEAN_PACKAGES = ("export", "training", "datasets", "active")


def _lines(path: Path) -> list[tuple[int, str]]:
    """(줄 번호, 내용)."""
    return list(enumerate(path.read_text("utf-8").splitlines(), start=1))


def imports_in(source: str) -> list[tuple[int, str, tuple[str, ...]]]:
    """소스의 import 문 → (줄 번호, 모듈 이름, 가져오는 이름들). `import a.b`는 (a.b, ()).

    상대 import(`from . import x`)는 모듈 이름을 빈 문자열로 둔다. 함수 안 import도 센다.
    """
    out: list[tuple[int, str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module if node.level == 0 and node.module else ""
            out.append((node.lineno, module, tuple(a.name for a in node.names)))
        elif isinstance(node, ast.Import):
            out += [(node.lineno, a.name, ()) for a in node.names]
    return out


def aireview_forbidden_import(module: str, names: tuple[str, ...], in_media: bool) -> bool:
    """aireview 소스에서 금지한 import인가.

    CLI·검수 패키지, 감사 원본 저장소(`dlp_media.audit`, `from dlp_media import audit` 포함), 그리고
    `media.py` 밖의 영상 디코드(`av`, `decode_frames`).
    """
    top = module.split(".")[0]
    full = {module, *(f"{module}.{n}" for n in names)}
    if top in ("dlp_cli", "dlp_review"):
        return True
    if any(m == "dlp_media.audit" or m.startswith("dlp_media.audit.") for m in full):
        return True
    if in_media:
        return False
    return top == "av" or "decode_frames" in names


def mentions_aireview(module: str, names: tuple[str, ...]) -> bool:
    """AI 판정 계약·저장소·패키지를 가리키는 import인가 (모듈이나 가져오는 이름에 aireview)."""
    return "aireview" in module or any("aireview" in n for n in names)


def test_aireview_does_not_write_labels_or_touch_raw() -> None:
    """판정 패키지 소스에 쓰기 함수·라벨 이력 표·원본·프록시 이름과 금지 import가 없다."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(ROOT)
        offenders += [
            f"{rel}:{n}: {line.strip()}"
            for n, line in _lines(path)
            if any(name in line for name in FORBIDDEN)
        ]
        offenders += [
            f"{rel}:{n}: import {module} {names}"
            for n, module, names in imports_in(path.read_text("utf-8"))
            if aireview_forbidden_import(module, names, path.name == "media.py")
        ]
    assert offenders == [], "\n".join(offenders)


def test_pixels_only_through_blurred_media_module() -> None:
    """저장소 파일 받기, 블러본 프레임 생성, 영상 디코드·열기는 `media.py`(블러본 경계)에서만
    한다."""
    offenders = [
        f"{path.relative_to(ROOT)}:{n}"
        for path in sorted(SRC.rglob("*.py"))
        if path.name != "media.py"
        for n, line in _lines(path)
        if any(name in line for name in MEDIA_ONLY)
    ]
    assert offenders == []


def test_training_and_export_never_import_ai_verdicts() -> None:
    """내보내기·학습·데이터셋·액티브 러닝은 AI 판정 계약·저장소·패키지를 import하지 않는다."""
    offenders = [
        f"{path.relative_to(ROOT)}:{n}: {module} {names}"
        for pkg in CLEAN_PACKAGES
        for path in sorted((ROOT / "packages" / pkg / "src").rglob("*.py"))
        for n, module, names in imports_in(path.read_text("utf-8"))
        if mentions_aireview(module, names)
    ]
    assert offenders == []


def _one(source: str) -> tuple[str, tuple[str, ...]]:
    """한 줄 import → (모듈, 이름)."""
    [(_, module, names)] = imports_in(source)
    return module, names


def test_scan_catches_bypass_forms() -> None:
    """검사의 회귀 테스트: 점 없는 형태·이름으로 가져오는 형태·함수 안 import도 잡는다."""
    for source in (
        "from dlp_review.tasks import x",
        "import dlp_cli.main",
        "from dlp_media.audit import AuditedStore",
        "from dlp_media import audit",
        "import dlp_media.audit as a",
        "from dlp_media.frames import decode_frames",
        "import av",
    ):
        assert aireview_forbidden_import(*_one(source), in_media=False), source
    assert not aireview_forbidden_import(*_one("from dlp_media.frames import pick_frames"), False)
    assert not aireview_forbidden_import(*_one("import av"), in_media=True)
    for source in (
        "from dlp_schema.db import aireview_repository",
        "from dlp_schema import aireview",
        "from dlp_schema.aireview import AiVerdict",
        "import dlp_aireview.plan",
    ):
        assert mentions_aireview(*_one(source)), source
    assert not mentions_aireview(*_one("from dlp_schema.labels import LabelRecord"))
    nested = imports_in("def f():\n    from dlp_media import audit\n")
    assert [(m, n) for _, m, n in nested] == [("dlp_media", ("audit",))]
    for line in ("frames = decode_frames(path, pts, idx)", "with av.open(path) as c:"):
        assert any(name in line for name in MEDIA_ONLY), line


CLI = ROOT / "packages" / "cli" / "src" / "dlp_cli"


def test_blurred_media_module_keeps_its_gates() -> None:
    """`media.py`가 원본 버킷 이름 거부와 렌더 게이트 네 단계를 모두 부른다 (I1)."""
    text = (SRC / "media.py").read_text("utf-8")
    for needle in (
        "labeling.bucket == raw_bucket", "expected_render_hash(", "assert_render_current(",
        "check_fetched(", "render_hashes_now(",
    ):  # fmt: skip
        assert needle in text, needle


def test_runner_gets_frames_only_through_blurred_media() -> None:
    """실행기는 프레임을 `frames_for`(블러본 경계)로만 얻고 프레임 공급 형은 `BlurredFrame`이다."""
    text = (SRC / "runner.py").read_text("utf-8")
    assert "frames_for(" in text and "Iterable[BlurredFrame]" in text
    names = {n for _, module, ns in imports_in(text) if module == "dlp_aireview.media" for n in ns}
    assert {"frames_for", "open_blurred", "BlurredFrame"} <= names


def _raw_store_purposes(source: str) -> list[object]:
    """소스의 `raw_store(...)` 호출마다 세 번째 인자(용도) 값 (상수가 아니면 그 노드 이름)."""
    out: list[object] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "raw_store"
        ):
            arg = node.args[2] if len(node.args) > 2 else None
            out.append(arg.value if isinstance(arg, ast.Constant) else type(arg).__name__)
    return out


def test_aireview_cli_reads_raw_only_for_aireview_run() -> None:
    """AI 검수 CLI의 원본 저장소는 용도 `aireview.run` 상수로만 만든다 (ops.yaml에 등록된 용도)."""
    purposes = [
        p
        for path in sorted(CLI.glob("aireview*.py"))
        for p in _raw_store_purposes(path.read_text())
    ]
    assert purposes == ["aireview.run"]
    assert _raw_store_purposes('raw_store(a, b, "x.y")\nraw_store(a, b, name)') == ["x.y", "Name"]
