"""우리 코드는 FiftyOne 웹 서버(GraphQL)를 띄우지 않는다 (S3 완화, ADR 0036).

FiftyOne 1.22.1(2026-11-19 확인한 최신판)은 Strawberry GraphQL 0.317.0 미만을 요구한다. 잠금
파일의 0.316.0에는 GraphQL 서버 쪽 취약점 2건(CVE-2026-107727 구독 한도 소진, CVE-2026-107728 권한
검사 우회)이 남아 있고, 패치판(0.327.2 이상)은 그 FiftyOne 서버와 맞지 않는다
(`docs/handover/known-issues.md` S3). 취약한 코드는 FiftyOne 앱 서버를 띄울 때만 쓰인다. 그래서
패키지 소스와 스크립트가 앱·서버를 띄우는 호출(`launch_app`, `fiftyone.server`, `fiftyone.app`)을
쓰지 않는지 줄 단위로 본다 (주석 포함).
호환되는 패치가 나와 잠금 파일을 올리면 이 검사와 S3를 함께 정리한다.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# FiftyOne 앱·GraphQL 서버를 띄우거나 가져오는 표현
SERVER = re.compile(r"\blaunch_app\b|\bfiftyone\.server\b|\bfiftyone\.app\b|\bfo\.app\b")


def test_sources_never_start_the_fiftyone_server() -> None:
    """`packages/*/src`와 `scripts/`의 파이썬 파일에 FiftyOne 서버를 띄우는 줄이 없다."""
    paths = [*ROOT.glob("packages/*/src/**/*.py"), *ROOT.glob("scripts/**/*.py")]
    assert len(paths) > 100  # 훑기가 저장소에 닿았다는 확인
    offenders = [
        f"{path.relative_to(ROOT)}:{n}: {line.strip()}"
        for path in sorted(paths)
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if SERVER.search(line)
    ]
    listing = "\n".join(offenders)
    assert not offenders, f"FiftyOne 서버를 띄우는 코드 (S3, 패치 전 취약점):\n{listing}"
