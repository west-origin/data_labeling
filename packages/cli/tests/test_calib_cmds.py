"""`dlp calib intrinsics|import|palm|show` 명령 테스트 (ADR 0045, SQLite 파일 DB, 서비스 불필요).

정답: 합성 체커보드 영상(`dlp_fixtures.calibration`)의 렌더 내부 파라미터, 테스트가 쓴 캘리브레이션
파일과 손 크기 값. 같은 명령을 다시 돌리면 쓰지 않는다 (멱등).
`palm-estimate`는 깊이 모델 가중치와 원본 저장소가 필요해 여기서 돌리지 않는다 (추정 함수는
`packages/prelabel/tests/test_handfit_quality.py`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
import yaml

from dlp_cli import calib_cmds
from dlp_cli.main import main
from dlp_fixtures.calibration import generate_checkerboard_video
from dlp_schema.db.calibration_repository import get_camera_calibrations, get_worker_hand_sizes
from dlp_schema.testing import sqlite_db


class _AwareSqlalchemy:
    """CLI 모듈의 `sa`를 대신한다: create_engine만 시간대 보존 SQLite 엔진으로 바꾼다."""

    def __getattr__(self, name: str) -> Any:
        return getattr(sqlalchemy, name)

    def create_engine(self, url: Any, **kwargs: Any) -> sqlalchemy.Engine:
        return sqlite_db(str(url), create=False)


@pytest.fixture
def url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """표를 만든 SQLite 파일 DB URL."""
    db = f"sqlite:///{tmp_path / 'cli.db'}"
    sqlite_db(db).dispose()
    monkeypatch.setattr(calib_cmds, "sa", _AwareSqlalchemy())
    return db


def test_intrinsics_from_checkerboard_video_is_registered_once(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """체커보드 영상 → 정답과 1% 안의 fx로 등록되고, 다시 돌리면 같은 값이라 쓰지 않는다.
    --dry-run은 쓰지 않는다. 안쪽 코너 수를 틀리면 종료 코드 2.
    """
    s = generate_checkerboard_video(tmp_path / "board.mp4", 3)
    cols, rows = s.inner_corners
    args = ["calib", "intrinsics", "cam-7", str(tmp_path / "board.mp4"), "--board",
            f"{cols}x{rows}", "--square-mm", str(s.square_m * 1000), "--url", url]  # fmt: skip
    assert main([*args, "--dry-run"]) == 0
    assert main(args) == 0
    assert main(args) == 0
    assert "건너뜀" in capsys.readouterr().out
    with sqlite_db(url, create=False).connect() as conn:
        [record] = get_camera_calibrations(conn, "cam-7")
    assert record.intrinsics.fx == pytest.approx(s.intrinsics.fx, rel=0.01)
    assert record.method.value == "checkerboard" and record.n_views and record.source_sha256
    wrong = [*args[:5], "7x7", *args[6:]]
    assert main(wrong) == 2


def test_import_palm_and_show(url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """캘리브레이션 파일 가져오기(유효 시작 포함)·잰 손 크기 등록·보기. 범위 밖 손 크기는 거부."""
    path = tmp_path / "cam.yaml"
    k = {"width": 640, "height": 480, "fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0}
    path.write_text(yaml.safe_dump(k), "utf-8")
    since = "2026-10-01T00:00:00+09:00"
    assert main(["calib", "import", "cam-1", str(path), "--valid-from", since, "--url", url]) == 0
    assert main(["calib", "palm", "w01", "0.081", "--url", url]) == 0
    assert main(["calib", "palm", "w01", "0.081", "--url", url]) == 0
    assert main(["calib", "show", "--camera", "cam-1", "--worker", "w01", "--url", url]) == 0
    out = capsys.readouterr().out
    assert "fx 500.00" in out and "0.0810 m" in out and "건너뜀" in out
    with sqlite_db(url, create=False).connect() as conn:
        [cal] = get_camera_calibrations(conn, "cam-1")
        sizes = get_worker_hand_sizes(conn, "w01")
    assert cal.valid_from is not None and cal.valid_from.utcoffset() is not None
    assert len(sizes) == 1 and sizes[0].source.value == "measured"
    with pytest.raises(Exception, match="palm_m"):
        main(["calib", "palm", "w01", "0.5", "--url", url])
    with pytest.raises(SystemExit):
        main(["calib", "import", "cam-1", str(path), "--valid-from", "2026-10-01", "--url", url])
