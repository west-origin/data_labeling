"""`dlp console serve`가 찍는 콘솔 주소의 호스트 표기."""

from __future__ import annotations

import pytest

from dlp_console.server import url_host


@pytest.mark.parametrize(
    ("bind", "shown"),
    [
        ("127.0.0.1", "localhost"),
        ("::1", "localhost"),
        ("0.0.0.0", "0.0.0.0"),
        ("10.0.0.5", "10.0.0.5"),
        ("::", "[::]"),
        ("fe80::1", "[fe80::1]"),
        ("console.internal", "console.internal"),
    ],
)
def test_ipv6_literal_is_bracketed_f_coe_87(bind: str, shown: str) -> None:
    """F-COE-87: IPv6 리터럴은 대괄호로 감싸야 포트와 구분되는 주소가 된다."""
    assert url_host(bind) == shown
