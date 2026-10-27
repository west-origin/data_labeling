"""가중치 가져오기 진행 기록 (작업 디렉터리, ADR 0061 6절).

- `pretrain/<가져오기 ID>/`: 올린 세 파일과 검사 결과(plan.json), 확인 뒤 패치 파일.
- `pretrain_imports.jsonl`: 확인한 가져오기 기록 (추가만). 누가 언제 어떤 해시의 가중치를 어디에
  두었는지.
확인(`confirm`)은 검사를 모두 통과한 계획만 받고, 복사한 가중치의 sha256을 다시 확인한다.
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dlp_console.pretrain import ImportPlan, build_plan, sha256_file
from dlp_console.steps import IDENT, ParamError

FILES = ("model.onnx", "model_manifest.json", "data_manifest.json")


class ImportStore:
    def __init__(self, root: Path, work_dir: Path, models_dir: str) -> None:
        self.root = root
        self.dir = work_dir / "pretrain"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = work_dir / "pretrain_imports.jsonl"
        self.models_dir = models_dir

    def new_dir(self) -> tuple[str, Path]:
        import_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        path = self.dir / import_id
        path.mkdir(parents=True)
        return import_id, path

    def _path(self, import_id: str) -> Path:
        if not IDENT.match(import_id):
            raise ParamError("가져오기 ID가 올바르지 않습니다")
        path = self.dir / import_id
        if not path.is_dir():
            raise ParamError(f"모르는 가져오기: {import_id}")
        return path

    def check(self, import_id: str) -> ImportPlan:
        """올린 세 파일을 검사하고 계획을 plan.json에 남긴다."""
        path = self._path(import_id)
        onnx, model, data = (path / f for f in FILES)
        plan = build_plan(self.root, import_id, onnx, model, data, self.models_dir)
        (path / "plan.json").write_text(json.dumps(plan.public(), ensure_ascii=False), "utf-8")
        return plan

    def plan(self, import_id: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self._path(import_id) / "plan.json").read_text("utf-8"))
        return data

    def confirm(self, import_id: str, actor: str) -> dict[str, Any]:
        """확인: 가중치를 data/models 아래에 두고 패치 파일·기록을 남긴다. 예외: ParamError."""
        path = self._path(import_id)
        plan = self.plan(import_id)
        if not plan.get("ok"):
            raise ParamError("검사를 모두 통과한 가져오기만 확인할 수 있습니다")
        dest = self.root / str(plan["dest"])
        if dest.exists() and sha256_file(dest) != plan["onnx_sha256"]:
            raise ParamError(f"같은 자리에 다른 파일이 있습니다: {plan['dest']}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        shutil.copyfile(path / "model.onnx", tmp)
        if sha256_file(tmp) != plan["onnx_sha256"]:
            tmp.unlink(missing_ok=True)
            raise ParamError("복사한 가중치의 sha256이 다릅니다")
        tmp.replace(dest)
        (path / "models.patch").write_text(plan["models_patch"], "utf-8")
        if plan["activation_patch"]:
            (path / "activation.patch").write_text(plan["activation_patch"], "utf-8")
        record = {
            "import_id": import_id,
            "at": datetime.now(UTC).isoformat(),
            "actor": actor,
            "model_name": plan["model_name"],
            "onnx_sha256": plan["onnx_sha256"],
            "dest": plan["dest"],
            "commercial": plan["commercial"],
            "entries": plan["entries"],
        }
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {**record, "next": self.next_commands(import_id)}

    def next_commands(self, import_id: str) -> list[str]:
        """확인 뒤 운영자가 터미널에서 할 일 (설정 파일은 콘솔이 고치지 않는다)."""
        patch = self._path(import_id) / "models.patch"
        return [
            f"git apply {patch}",
            "make licenses",
            "uv run dlp models fetch   # 해시만 확인 ([있음])",
        ]

    def records(self) -> list[dict[str, Any]]:
        """확인한 가져오기 (새것 먼저)."""
        if not self.log.is_file():
            return []
        out: list[dict[str, Any]] = []
        for line in self.log.read_text("utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(reversed(out))
