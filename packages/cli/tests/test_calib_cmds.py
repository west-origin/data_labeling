"""`dlp calib intrinsics|import|assign|palm|show` 명령 테스트 (ADR 0045·0052, SQLite 파일 DB).

정답: 합성 체커보드 영상(`dlp_fixtures.calibration`)의 렌더 내부 파라미터, 테스트가 쓴 캘리브레이션
파일과 손 크기 값. 같은 명령을 다시 돌리면 쓰지 않는다 (멱등). `palm-estimate`·`palm-revoke`는
`test_calib_palm_estimate.py`(가짜 깊이·로컬 원본 저장소).
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
from dlp_schema import repo_root
from dlp_schema.db.calibration_repository import (
    get_camera_calibrations,
    get_worker_hand_sizes,
    session_intrinsics,
)
from dlp_schema.db.repository import get_session, insert_session, register_ontology
from dlp_schema.ontology import load_ontology
from dlp_schema.session import Calibration
from dlp_schema.testing import make_session, sqlite_db


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
    # 세션이 없는 작업자 ID는 오타로 보고 거부한다 (F-COE-72). 첫 녹화 전 등록은 옵션으로
    assert main(["calib", "palm", "w01", "0.081", "--url", url]) == 2
    assert "[거부]" in capsys.readouterr().out
    allow = "--allow-unknown-worker"
    assert main(["calib", "palm", "w01", "0.081", allow, "--url", url]) == 0
    assert main(["calib", "palm", "w01", "0.081", allow, "--url", url]) == 0
    assert main(["calib", "show", "--camera", "cam-1", "--worker", "w01", "--url", url]) == 0
    out = capsys.readouterr().out
    assert "fx 500.00" in out and "0.0810 m" in out and "건너뜀" in out
    with sqlite_db(url, create=False).connect() as conn:
        [cal] = get_camera_calibrations(conn, "cam-1")
        sizes = get_worker_hand_sizes(conn, "w01")
    assert cal.valid_from is not None and cal.valid_from.utcoffset() is not None
    assert len(sizes) == 1 and sizes[0].source.value == "measured"
    with pytest.raises(Exception, match="palm_m"):
        main(["calib", "palm", "w01", "0.5", allow, "--url", url])
    with pytest.raises(SystemExit):
        main(["calib", "import", "cam-1", str(path), "--valid-from", "2026-10-01", "--url", url])


def test_import_rejects_bad_files_with_exit_2(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """파싱 오류·모르는 키·ROS 미보정·타당 범위 밖 파일은 처리되지 않은 예외가 아니라 종료
    코드 2이고 DB에 쓰지 않는다 (F-PMS-21·22).
    """
    files = {
        "broken.json": '{"fx": 1',
        "extra.yaml": "width: 640\nheight: 480\nfx: 500\nfy: 500\ncx: 320\ncy: 240\nname: x\n",
        "uncal.yaml": yaml.safe_dump({
            "image_width": 640, "image_height": 480,
            "camera_matrix": {"rows": 3, "cols": 3, "data": [0.0] * 9},
            "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0] * 5},
        }),
        "narrow.yaml": "width: 640\nheight: 480\nfx: 9000\nfy: 9000\ncx: 320\ncy: 240\n",
    }  # fmt: skip
    for name, text in files.items():
        (tmp_path / name).write_text(text, "utf-8")
        assert main(["calib", "import", "cam-1", str(tmp_path / name), "--url", url]) == 2, name
    assert capsys.readouterr().out.count("[실패]") == len(files)
    with sqlite_db(url, create=False).connect() as conn:
        assert get_camera_calibrations(conn, "cam-1") == []


def test_undated_registration_shadowed_by_dated_warns_and_supersede_fixes(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """유효 시작 있는 등록에 가려 쓰이지 않는 새 등록은 경고하고, --supersede로 같은 유효 시작에
    다시 등록하면 그 값이 쓰인다 (F-PMS-25).
    """
    bad, good = tmp_path / "bad.yaml", tmp_path / "good.yaml"
    bad.write_text("width: 640\nheight: 480\nfx: 520\nfy: 520\ncx: 320\ncy: 240\n", "utf-8")
    good.write_text("width: 640\nheight: 480\nfx: 500\nfy: 500\ncx: 320\ncy: 240\n", "utf-8")
    since = "2026-01-01T00:00:00+09:00"
    assert main(["calib", "import", "cam-1", str(bad), "--valid-from", since, "--url", url]) == 0
    capsys.readouterr()
    assert main(["calib", "import", "cam-1", str(good), "--url", url]) == 0
    out = capsys.readouterr().out
    assert "[주의]" in out and "--supersede" in out
    with sqlite_db(url, create=False).connect() as conn:
        dated = next(r for r in get_camera_calibrations(conn, "cam-1") if r.valid_from)
    args = ["calib", "import", "cam-1", str(good), "--supersede", dated.calibration_id]
    assert main([*args, "--url", url]) == 0
    assert "[주의]" not in capsys.readouterr().out
    with sqlite_db(url, create=False).connect() as conn:
        records = get_camera_calibrations(conn, "cam-1")
    assert records[-1].valid_from == dated.valid_from and records[-1].intrinsics.fx == 500.0
    assert main(["calib", "import", "cam-1", str(good), "--supersede", "cal-x", "--url", url]) == 2


def test_assign_links_existing_sessions_to_a_camera(
    url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """매니페스트에 기기가 없던 세션을 `dlp calib assign`으로 이으면 등록된 값이 쓰인다 (F-PMS-24).

    매니페스트에 다른 기기가 있는 세션이 섞이면 아무것도 쓰지 않고 종료 코드 2.
    """
    eng = sqlite_db(url, create=False)
    with eng.begin() as conn:
        register_ontology(conn, load_ontology(repo_root() / "config/ontology/v1"))
        insert_session(conn, make_session("s001"))
        insert_session(conn, make_session("s002"))
        insert_session(conn, make_session("s003", calibration=Calibration(camera_id="cam-x")))
    path = tmp_path / "cam.yaml"
    path.write_text("width: 1920\nheight: 1080\nfx: 1400\nfy: 1400\ncx: 960\ncy: 540\n", "utf-8")
    assert main(["calib", "import", "gopro-1", str(path), "--url", url]) == 0
    base = ["calib", "assign", "gopro-1", "--reason", "촬영 기록부", "--url", url]
    assert main([*base, "s001", "s003"]) == 2
    assert "[충돌]" in capsys.readouterr().out
    with eng.connect() as conn:
        assert session_intrinsics(conn, get_session(conn, "s001")).origin == "hfov"
    assert main([*base, "--worker", "w01", "--until", "2100-01-01T00:00:00+00:00"]) == 2
    assert main([*base, "s001", "s002"]) == 0
    assert main([*base, "s001"]) == 0
    assert "0개 연결" in capsys.readouterr().out
    assert main([*base[:3], "--reason", "x", "--url", url]) == 2
    assert main(["calib", "show", "--session", "s001", "--url", url]) == 0
    assert "registry:" in capsys.readouterr().out
    with eng.connect() as conn:
        assert session_intrinsics(conn, get_session(conn, "s002")).origin.startswith("registry:")
    eng.dispose()
