"""`dlp sync adjust` 명령줄 인자 검사 (F-SCH-27).

NaN·±Inf 조정값은 DB를 열기 전에 argparse 인자 오류(종료 코드 2)로 끝나야 한다.
"""

from __future__ import annotations

import pytest

from dlp_cli.main import build_parser


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN", "abc"])
def test_sync_adjust_rejects_non_finite(bad: str) -> None:
    """유한 실수가 아니면 SystemExit(2)로 끝나고 실행 함수까지 가지 않는다."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["sync", "adjust", "s001", "imu", bad])
    assert exc.value.code == 2


def test_sync_adjust_accepts_finite() -> None:
    """보통 실수(음수 포함)는 그대로 받는다."""
    args = build_parser().parse_args(["sync", "adjust", "s001", "imu", "--", "-4.5"])
    assert args.adjustment_ms == -4.5
