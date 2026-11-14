"""`dlp` 진입점의 인자 해석 (`dlp_cli.main.parse_cli_args`).

같은 명령줄은 파이썬 패치 버전과 관계없이 같게 읽혀야 한다. 파이썬 3.12.7·3.13.1 전의 argparse는
`<위치 인자> --옵션 값 <nargs="*" 위치 인자…>` 꼴의 뒤쪽 위치 인자를 받지 못한다 (우분투 24.04의
시스템 파이썬 3.12.3 포함). 정답은 argparse 문서의 동작(고친 버전의 동작)이다.
"""

# argparse는 하위 파서 목록을 공개 API로 주지 않아, 모든 하위 명령을 훑는 시험만 내부 속성을 읽는다.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from typing import cast

import pytest

from dlp_cli.main import LATE_POSITIONAL, build_parser, parse_cli_args


def test_calib_assign_reads_session_ids_before_or_after_options() -> None:
    """세션 ID를 기기 바로 뒤에 적든 옵션 뒤에 적든 같은 결과다 (옛 argparse가 거부하던 꼴 포함)."""
    parser = build_parser()
    first = parse_cli_args(parser, ["calib", "assign", "cam", "s001", "s003", "--reason", "r"])
    last = parse_cli_args(parser, ["calib", "assign", "cam", "--reason", "r", "s001", "s003"])
    assert first.session_ids == last.session_ids == ["s001", "s003"]
    assert first.camera_id == last.camera_id == "cam"
    assert parse_cli_args(parser, ["calib", "assign", "cam", "--reason", "r"]).session_ids == []


def _old_argparse(
    monkeypatch: pytest.MonkeyPatch, parsed: argparse.Namespace, extra: list[str]
) -> argparse.ArgumentParser:
    """옛 argparse처럼 위치 인자를 남기는 파서 (어느 파이썬에서든 남은 값 처리를 시험한다)."""

    def leftover(_argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
        return parsed, extra

    parser = argparse.ArgumentParser(prog="dlp")
    monkeypatch.setattr(parser, "parse_known_args", leftover)
    return parser


def test_leftover_values_fill_the_marked_empty_positional(monkeypatch: pytest.MonkeyPatch) -> None:
    """남은 값은 하위 명령이 표시한 빈 위치 인자에 순서대로 들어간다."""
    parsed = argparse.Namespace(camera_id="cam", session_ids=[], **{LATE_POSITIONAL: "session_ids"})
    args = parse_cli_args(_old_argparse(monkeypatch, parsed, ["s001", "s003"]), [])
    assert args.session_ids == ["s001", "s003"]


@pytest.mark.parametrize(
    ("parsed", "extra"),
    [
        # 표시가 없는 하위 명령: 남은 값은 그대로 인자 오류
        (argparse.Namespace(session_id="s001"), ["s002"]),
        # 이미 값이 있는 위치 인자: 옵션 사이에 끼워 적은 두 번째 묶음은 받지 않는다
        (argparse.Namespace(session_ids=["s001"], **{LATE_POSITIONAL: "session_ids"}), ["s002"]),
        # 모르는 옵션은 세션 ID로 삼키지 않는다
        (argparse.Namespace(session_ids=[], **{LATE_POSITIONAL: "session_ids"}), ["--oops", "s1"]),
    ],
)
def test_other_leftover_values_stay_argument_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    parsed: argparse.Namespace,
    extra: list[str],
) -> None:
    """표시한 빈 위치 인자에 넣을 수 없는 남은 값은 `parse_args`와 같은 인자 오류(종료 코드 2)다."""
    with pytest.raises(SystemExit) as stop:
        parse_cli_args(_old_argparse(monkeypatch, parsed, extra), [])
    assert stop.value.code == 2
    assert "unrecognized arguments: " + " ".join(extra) in capsys.readouterr().err


def _leaf_parsers(
    parser: argparse.ArgumentParser, path: tuple[str, ...] = ()
) -> Iterator[tuple[str, argparse.ArgumentParser]]:
    """하위 명령이 더 없는 파서를 (명령 이름, 파서)로 낸다."""
    groups = [
        cast("argparse._SubParsersAction[argparse.ArgumentParser]", a)
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction)
    ]
    if not groups:
        yield " ".join(path), parser
    for group in groups:
        for name, child in group.choices.items():
            yield from _leaf_parsers(child, (*path, name))


def _positionals(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in parser._actions if not a.option_strings]


def test_every_late_optional_positional_is_marked() -> None:
    """다른 위치 인자 뒤의 `nargs="*"` 위치 인자는 모두 `late_positional` 표시가 있다.

    이 자리의 `nargs="*"`·`"?"`가 옛 argparse에서 옵션 뒤 값을 놓치는 꼴이다. 새 하위 명령이 그 꼴을
    표시 없이 더하면 파이썬 패치 버전에 따라 명령줄 해석이 갈린다. 표시한 인자는 남은 값을 그대로
    받으므로 마지막 위치 인자이고 `type`·`choices`가 없어야 한다. `"?"`는 받는 길이 없어 이 자리에
    두지 않는다.
    """
    seen = 0
    for name, parser in _leaf_parsers(build_parser()):
        positionals = _positionals(parser)
        late = [a for i, a in enumerate(positionals) if i > 0 and a.nargs in ("*", "?")]
        marked = parser.get_default(LATE_POSITIONAL)
        assert [a.dest for a in late] == ([marked] if marked else []), name
        for action in late:
            seen += 1
            assert action.nargs == "*", name
            assert action is positionals[-1], name
            assert action.type is None and action.choices is None, name
    assert seen >= 1  # `calib assign`이 그 꼴이다 (훑기가 하위 명령에 닿았다는 확인)
