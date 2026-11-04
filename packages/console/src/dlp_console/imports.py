"""가중치 가져오기 진행 기록 (작업 디렉터리, ADR 0061 6절).

- `pretrain/<가져오기 ID>/`: 올린 세 파일과 검사 결과(plan.json), 확인 뒤 패치 파일.
- `pretrain_imports.jsonl`: 확인한 가져오기 기록 (추가만). 누가 언제 어떤 해시의 가중치를 어디에
  두었는지.

확인(`confirm`)은 검사를 모두 통과한 계획만 받고, 복사한 가중치의 sha256을 다시 확인한다.

흐름 (`/api/pretrain/*`, `dlp_console.app`): `new_dir`로 디렉터리를 만들고 세 파일(`FILES`)을
저장 → `check`(검사·미리보기, plan.json) → `confirm`(가중치 복사·기록) → 운영자가 `next_commands`의
명령을 터미널에서 돌린다. 콘솔은 Git이 관리하는 설정 파일(`config/models.yaml`,
`config/policies/prelabel.yaml`)을 직접 고치지 않는다 (ADR 0061 6절).
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dlp_console.pretrain import (
    MODEL_NAME,
    SHA256,
    ImportPlan,
    build_plan,
    dest_inside,
    sha256_file,
)
from dlp_console.steps import IDENT, ParamError

FILES = ("model.onnx", "model_manifest.json", "data_manifest.json")


class ImportStore:
    """가져오기 디렉터리·기록 다루기 (`<작업 디렉터리>/pretrain/`, `pretrain_imports.jsonl`)."""

    def __init__(self, root: Path, work_dir: Path, models_dir: str) -> None:
        """`models_dir`: 가중치를 둘 저장소 루트 기준 디렉터리 (`pretrain_import.models_dir`)."""
        self.root = root
        self.dir = work_dir / "pretrain"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = work_dir / "pretrain_imports.jsonl"
        self.models_dir = models_dir

    def new_dir(self) -> tuple[str, Path]:
        """새 가져오기 ID(`<UTC 시각>-<무작위 6자>`)와 그 빈 디렉터리를 만든다."""
        import_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        path = self.dir / import_id
        path.mkdir(parents=True)
        return import_id, path

    def _path(self, import_id: str) -> Path:
        """ID 모양을 검사하고 그 가져오기 디렉터리를 돌려준다. 모양 오류·없는 ID는 ParamError."""
        if not IDENT.match(import_id):
            raise ParamError("가져오기 ID가 올바르지 않습니다")
        path = self.dir / import_id
        if not path.is_dir():
            raise ParamError(f"모르는 가져오기: {import_id}")
        return path

    def check(self, import_id: str) -> ImportPlan:
        """올린 세 파일을 검사하고 계획을 plan.json에 남긴다 (다시 부르면 덮어쓴다).

        검사가 실패해도 예외가 아니다. 반환한 계획의 `ok`가 False다.
        """
        path = self._path(import_id)
        onnx, model, data = (path / f for f in FILES)
        plan = build_plan(self.root, import_id, onnx, model, data, self.models_dir)
        (path / "plan.json").write_text(json.dumps(plan.public(), ensure_ascii=False), "utf-8")
        return plan

    def plan(self, import_id: str) -> dict[str, Any]:
        """검사 계획. 검사가 끝나지 않아 plan.json이 없으면 ParamError(400, F-COE-86)."""
        path = self._path(import_id) / "plan.json"
        if not path.is_file():
            raise ParamError(f"검사 결과가 없는 가져오기입니다 (다시 검사하세요): {import_id}")
        data: dict[str, Any] = json.loads(path.read_text("utf-8"))
        return data

    def confirm(self, import_id: str, actor: str) -> dict[str, Any]:
        """확인: 가중치를 `models_dir` 아래에 두고 패치 파일·기록을 남긴다. 예외: ParamError.

        plan.json은 작업 디렉터리 파일이라 손으로 바뀌었을 수 있다. 그래서 이름·경로·해시 모양을
        다시 본다(`_check_target`). 가중치는 `.part`로 복사해 sha256을 확인한 뒤 제자리로 옮긴다.
        같은 자리에 같은 해시 파일이 이미 있으면 덮어써도 내용이 같다 (다시 확인해도 안전하다).
        기록은 확인할 때마다 한 줄씩 추가된다.

        Returns:
            기록한 값과 `next`(운영자가 터미널에서 할 명령).
        """
        path = self._path(import_id)
        plan = self.plan(import_id)
        if not plan.get("ok"):
            raise ParamError("검사를 모두 통과한 가져오기만 확인할 수 있습니다")
        self._check_target(plan)
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

    def _check_target(self, plan: dict[str, Any]) -> None:
        """계획의 이름·경로를 다시 본다 (F-COE-77): 식별자 모양 이름, models_dir 안의 경로."""
        name, dest, sha = plan.get("model_name"), plan.get("dest"), plan.get("onnx_sha256")
        if not (isinstance(name, str) and MODEL_NAME.match(name)):
            raise ParamError("가져오기 계획의 모델 이름이 올바르지 않습니다")
        if not (isinstance(sha, str) and SHA256.match(sha)):
            raise ParamError("가져오기 계획의 sha256이 올바르지 않습니다")
        if not (
            isinstance(dest, str)
            and dest == f"{self.models_dir.rstrip('/')}/{name}.onnx"
            and dest_inside(self.root, self.models_dir, dest)
        ):
            raise ParamError("가중치 경로가 가져오기 디렉터리 밖입니다")

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
