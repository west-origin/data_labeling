"""`dlp aireview calibrate`와 검수 CLI의 보정 고정 읽기 (ADR 0062).

DB 읽기(`load_ai_rows`)는 합성 결과 행으로 바꾼다 (서비스 불필요). 정답: 같은 기간·행이면 같은 파일
(멱등), 표본 부족이면 파일을 쓰지 않음, 고정한 sha256으로만 읽고 판정 버전이 다르면 쓰지 않음.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from dlp_aireview.policy import AiReviewPolicy, load_policy, select_model
from dlp_cli import aireview_cmds
from dlp_cli.main import main
from dlp_cli.review_config import display_calibration, load_review_policy
from dlp_review.ai_calibration import AiCalibrationError
from dlp_review.ops.ai_quality import AiRow
from dlp_review.ops.policy import AiCalibrationPolicy, AiCalibrationRef, ReviewOpsPolicy
from dlp_schema import repo_root
from dlp_schema.aireview import AiArm, AiDecision, AiHumanAction, AiVerdictValue
from dlp_schema.testing import AI_VERSION

ARGS = ["--until", "2026-10-01T00:00:00+00:00", "--version", AI_VERSION, "--url", "sqlite://"]


def _rows(n_bad: int = 60) -> list[AiRow]:
    """mop 의심 n_bad건 중 3건만 사람이 고친 대조군 행 (쓸모없는 칸)."""
    return [
        AiRow(
            task_key=f"cvat:{i % 5}",
            label_id=f"m{i}",
            kind="box_track",
            label_class="mop",
            arm=AiArm.HOLDOUT,
            verdict=AiVerdictValue.SUSPECT,
            confidence=None,
            has_suggestion=False,
            shown=False,
            human=AiHumanAction.CORRECTED if i < 3 else AiHumanAction.APPROVED,
            decision=AiDecision.NONE,
            apply_result=None,
            match="na",
            version=AI_VERSION,
        )
        for i in range(n_bad)
    ]


def _patch(monkeypatch: pytest.MonkeyPatch, rows: list[AiRow]) -> None:
    def fake(conn: Any, since: Any, until: Any) -> list[AiRow]:
        return rows

    monkeypatch.setattr(aireview_cmds, "load_ai_rows", fake)


def _pinned(path: Path) -> ReviewOpsPolicy:
    """저장소 검수 정책 + 그 파일을 고정한 보정 절."""
    policy = load_review_policy(repo_root())
    ref = AiCalibrationRef(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ai = policy.ai_review.model_copy(update={"calibration": AiCalibrationPolicy(artifact=ref)})
    return policy.model_copy(update={"ai_review": ai})


def test_calibrate_writes_once_and_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """첫 실행은 쓰고 두 번째는 '있음'. 출력된 sha256으로 고정하면 검수 CLI가 읽는다."""
    _patch(monkeypatch, _rows())
    assert main(["aireview", "calibrate", *ARGS, "--out", str(tmp_path)]) == 0
    first = capsys.readouterr().out
    files = list(tmp_path.glob("aireview-calibration-*.json"))
    assert len(files) == 1 and "씀:" in first and "숨김" in first
    assert main(["aireview", "calibrate", *ARGS, "--out", str(tmp_path)]) == 0
    assert "있음:" in capsys.readouterr().out
    assert main(["aireview", "calibrate", *ARGS, "--out", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sha256"] == hashlib.sha256(files[0].read_bytes()).hexdigest()
    assert report["effect_in_sample"]["flags_after"] == 0
    cal = display_calibration(repo_root(), _pinned(files[0]), (AI_VERSION,))
    assert cal is not None and not cal.allows("suspect", "box_track", "mop", None)
    # 다른 판정 버전에는 쓰지 않는다 (표시 = 보정 없음)
    assert display_calibration(repo_root(), _pinned(files[0]), ("other-version",)) is None
    assert "쓰지 않습니다" in capsys.readouterr().err
    # 고정 뒤 파일이 바뀌면 멈춘다
    policy = _pinned(files[0])
    files[0].write_bytes(files[0].read_bytes() + b" ")
    with pytest.raises(AiCalibrationError):
        display_calibration(repo_root(), policy, (AI_VERSION,))


def test_calibrate_without_enough_samples_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """표본 부족이면 칸이 없어 파일을 쓰지 않는다 (지금 동작 그대로)."""
    _patch(monkeypatch, _rows(5))
    assert main(["aireview", "calibrate", *ARGS, "--out", str(tmp_path)]) == 0
    assert "표본 부족" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------- 판정단 --model (ADR 0063)

D3 = "vllm-sr/d3-turbo"


def _with_panel(monkeypatch: pytest.MonkeyPatch) -> str:
    """정책에 판정단 모델 d3를 더하고 그 모델의 지금 판정 버전을 돌려준다."""
    original = load_policy

    def load(root: Path) -> AiReviewPolicy:
        data = original(root).model_dump()
        data["backend"]["panel"] = [
            {"model": D3, "served_name": "d3-turbo", "fingerprint": "vllm-bf16"}
        ]
        return AiReviewPolicy.model_validate(data)

    monkeypatch.setattr(aireview_cmds, "load_policy", load)
    return select_model(load(repo_root()), D3).verdict_version


def test_calibrate_model_learns_only_that_models_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--model`이면 그 모델의 판정만 배운다: review.yaml 허용 버전 중 그 모델 것, 없으면 그 모델의
    지금 버전. `--version`이 앞선다. 판정단에 없는 모델은 DB를 읽기 전에 종료 5."""
    d3_now = _with_panel(monkeypatch)
    _patch(monkeypatch, _rows())
    window = ["--until", "2026-10-01T00:00:00+00:00", "--url", "sqlite://", "--json"]
    out = ["--out", str(tmp_path)]

    def judged(*extra: str) -> list[str]:
        assert main(["aireview", "calibrate", *window, *out, *extra]) == 0
        return list(json.loads(capsys.readouterr().out)["calibration"]["judge_versions"])

    assert judged("--model", D3) == [d3_now]  # review.yaml versions가 비어 지금 버전
    assert judged("--model", D3, "--version", AI_VERSION) == [AI_VERSION]
    old_d3 = f"aireview-ffffffffffff+vlm-{D3}"
    review = load_review_policy(repo_root())
    ai = review.ai_review.model_copy(update={"versions": (AI_VERSION, old_d3)})
    pinned = review.model_copy(update={"ai_review": ai})

    def review_policy(root: Path) -> ReviewOpsPolicy:
        return pinned

    monkeypatch.setattr(aireview_cmds, "load_review_policy", review_policy)
    assert judged("--model", D3) == [old_d3]
    assert judged() == [AI_VERSION, old_d3]
    with pytest.raises(SystemExit) as exc:
        main(["aireview", "calibrate", *window, *out, "--model", "other/model"])
    assert exc.value.code == 5
