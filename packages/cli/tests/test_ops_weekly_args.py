"""`dlp ops weekly --weeks`는 1~52만 받는다 (F-COE-88).

정답 근거: 0·음수·53은 빈 표를 성공(0)으로 내던 값이라 argparse 인자 오류(종료 코드 2)로 끝나야
한다. 범위 안 값은 파서를 통과해 실행 함수에 정수로 넘어간다.
"""

from __future__ import annotations

import pytest

from dlp_cli.main import build_parser


@pytest.mark.parametrize("weeks", ["0", "-1", "53", "x"])
def test_weeks_out_of_range_is_an_argument_error(
    weeks: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """F-COE-88: 범위 밖·정수 아닌 --weeks는 종료 코드 2와 --weeks 오류 문구."""
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["ops", "weekly", "--weeks", weeks])
    assert exc.value.code == 2
    assert "--weeks" in capsys.readouterr().err


@pytest.mark.parametrize("weeks", [1, 52])
def test_weeks_in_range_is_accepted(weeks: int) -> None:
    """F-COE-88: 경계 1·52는 받는다."""
    assert build_parser().parse_args(["ops", "weekly", "--weeks", str(weeks)]).weeks == weeks
