"""`dlp` 명령줄 진입점 (WP0).

`pyproject.toml`의 `[project.scripts]`가 `dlp = dlp_cli.main:main`으로 이 모듈을 가리킨다.
최상위 파서를 만들고, 각 단계 모듈(`<단계>_cmds.py`)의 `add_commands(sub)`를 불러 하위 명령을
등록한다. 하위 명령은 `set_defaults(func=...)`로 실행 함수를 달아 두고, `main`은 파싱 뒤 그 함수를
불러 반환값(종료 코드)을 그대로 돌려준다.

공개 함수:
- `build_parser()` — 모든 하위 명령이 등록된 `ArgumentParser`.
- `main(argv)` — 진입점. 0이면 성공, 그 밖의 값은 실패(명령마다 의미가 다르다).

이 모듈이 직접 구현하는 명령은 `dlp services check`(개발 서비스 헬스체크, `make health`) 하나다.
등록 순서는 `--help` 출력 순서일 뿐 실행 순서와는 무관하다. 세션 하나의 정상 운영 순서는 ingest →
sync run → privacy detect → (블러 검수) → privacy approve → privacy render → prelabel run →
relations run → actions run → review create/collect/verify → dataset build → train/eval →
export이며, 자세한 선후 관계는 각 `<단계>_cmds.py` 모듈 docstring에 적었다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from dlp_cli import (
    __version__,
    actions_cmds,
    active_cmds,
    dataset_cmds,
    eval_cmds,
    export_cmds,
    fixtures_cmds,
    media_cmds,
    models_cmds,
    ops_cmds,
    prelabel_cmds,
    privacy_cmds,
    relations_cmds,
    review_cmds,
    schema_cmds,
    sync_cmds,
    train_cmds,
)
from dlp_cli.health import default_checks, run_checks


def _cmd_services_check(args: argparse.Namespace) -> int:
    """`dlp services check`: docker compose 개발 서비스가 응답하는지 점검한다.

    인자:
        args.include_cvat: 참이면 CVAT(`make cvat-up`으로 따로 띄움)도 점검한다.
        args.timeout: 서비스별 연결 제한 시간(초).

    반환: 모든 점검이 성공하면 0, 하나라도 실패하면 1.
    부작용: 표준 출력에 서비스마다 `[OK  ]`/`[FAIL]` 한 줄. 외부 서비스에 TCP·HTTP 요청만 한다.
    """
    results = run_checks(default_checks(include_cvat=args.include_cvat), timeout=args.timeout)
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        print(f"[{mark}] {r.name:<14} {r.detail}")
    return 0 if all(r.ok for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    """모든 하위 명령을 등록한 최상위 `dlp` 파서를 만든다.

    하위 명령(`command`)은 필수다. 단계 모듈마다 `add_commands(sub)`가 자기 하위 명령 묶음과
    실행 함수(`func`)를 등록한다. 테스트는 이 함수로 파서를 받아 인자 해석만 검사하기도 한다.
    """
    parser = argparse.ArgumentParser(prog="dlp", description="돌봄 영상 라벨링 플랫폼 CLI")
    parser.add_argument("--version", action="version", version=f"dlp {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # `dlp services check` — 개발 서비스 헬스체크 (이 모듈이 직접 구현)
    services = sub.add_parser("services", help="개발 서비스 관리")
    services_sub = services.add_subparsers(dest="services_command", required=True)
    check = services_sub.add_parser("check", help="docker compose 서비스 헬스체크")
    check.add_argument("--include-cvat", action="store_true", help="CVAT도 점검")
    check.add_argument("--timeout", type=float, default=3.0, help="서비스별 제한 시간(초)")
    check.set_defaults(func=_cmd_services_check)

    # 단계별 하위 명령. 대략 파이프라인 순서로 나열한다 (도움말 출력 순서에만 영향).
    schema_cmds.add_commands(sub)  # dlp schema export, ontology validate|register, db upgrade (WP1)
    fixtures_cmds.add_commands(sub)  # dlp fixtures generate — 합성 픽스처 (WP2)
    media_cmds.add_commands(sub)  # dlp ingest — 세션 수집 (WP3)
    sync_cmds.add_commands(sub)  # dlp sync run|adjust — 멀티스트림 동기화 (WP4)
    privacy_cmds.add_commands(sub)  # dlp privacy detect|approve|render — 프라이버시 게이트 (WP5)
    models_cmds.add_commands(sub)  # dlp models fetch|export|licenses (ADR 0009·0010)
    review_cmds.add_commands(sub)  # dlp review … — 검수 도구 연동·운영 (WP6, WP12)
    dataset_cmds.add_commands(sub)  # dlp dataset … / dlp lineage — 데이터셋·계보 (WP7)
    prelabel_cmds.add_commands(sub)  # dlp prelabel run — 자동 프리라벨 (WP8)
    relations_cmds.add_commands(sub)  # dlp relations run — 관계 도출 (WP9)
    actions_cmds.add_commands(sub)  # dlp actions run — 행동 구간 (WP10)
    eval_cmds.add_commands(sub)  # dlp eval golden — 골든셋 평가 (WP11)
    train_cmds.add_commands(sub)  # dlp train run|models|approve — 재학습 (WP13)
    active_cmds.add_commands(sub)  # dlp active rank|fiftyone — 액티브 러닝 (WP14)
    export_cmds.add_commands(sub)  # dlp export coco|intervals|lerobot — 내보내기 (WP15)
    ops_cmds.add_commands(sub)  # dlp ops … — 운영 지표·감사·보관 (WP16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """`dlp` 진입점. `argv`가 None이면 `sys.argv[1:]`을 쓴다.

    반환: 선택된 하위 명령 실행 함수의 반환값(종료 코드). 인자 오류는 `argparse`가
    `SystemExit(2)`로 끝낸다.
    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
