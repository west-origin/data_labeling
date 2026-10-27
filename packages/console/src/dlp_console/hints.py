"""실패한 작업 로그 → 마지막 오류 줄과 다음 할 일 안내 (한국어, ADR 0061 3절).

안내는 로그 글귀를 보고 고르는 참고문이다. 원인을 단정하지 않으며 맞는 것이 없으면 단계의 기본
안내를 보인다. 순서대로 먼저 맞는 것을 쓴다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# (로그 글귀 정규식, 안내)
_RULES: tuple[tuple[str, str], ...] = (
    (
        r"localhost:8080|cvat.*(Connection refused|ConnectError|연결)",
        "CVAT가 꺼져 있습니다. 터미널에서 `make cvat-up`을 실행하고 "
        "(처음이면 `make cvat-superuser`) "
        "다시 누르세요. Label Studio만 쓰려면 'Label Studio만' 선택으로 다시 실행합니다.",
    ),
    (
        r"localhost:8081|label.?studio.*(Connection refused|ConnectError)",
        "Label Studio가 응답하지 않습니다. `make up`으로 개발 서비스를 띄운 뒤 다시 누르세요.",
    ),
    (
        r"(psycopg|OperationalError).*(Connection refused|could not connect|connection failed)",
        "DB(PostgreSQL)에 연결하지 못했습니다. `make up` 뒤 `make health`로 확인하세요.",
    ),
    (
        r"InternalError|SlowDown|No space left|디스크",
        "저장 공간이 부족할 수 있습니다. 디스크 여유를 늘리고(원본·DB 볼륨은 지우지 않는다) "
        "`docker restart dlp-seaweedfs-1` 뒤 다시 누르세요.",
    ),
    (
        r"원본 접근 권한자|reviewers\.privacy",
        "블러 검수 담당자는 검수 정책의 reviewers.privacy에 있는 원본 접근 권한자여야 합니다. "
        "DLP_REVIEW_POLICY_FILE 정책 파일에 담당자와 CVAT 계정 연결을 적으세요.",
    ),
    (
        r"SessionConflictError",
        "같은 세션 ID가 다른 내용으로 이미 있습니다. 새 세션 ID로 다시 올리세요 "
        "(이미 수집한 세션의 "
        "영상은 바꾸지 않습니다).",
    ),
    (
        r"ModelUnavailableError|모델 파일|make models",
        "모델 가중치가 없습니다. 터미널에서 `make models`를 실행한 뒤 다시 누르세요.",
    ),
    (
        r"VLM|vlm-url|DLP_VLM",
        "사내 VLM 서버 주소가 없거나 응답하지 않습니다. 서버 없이는 이 단계를 건너뛰어도 됩니다.",
    ),
    (
        r"Field required",
        "정책 파일에 필수 값이 빠졌습니다. 기본 config/policies의 값을 옮겨 적으세요.",
    ),
    (
        r"lakefs|LakeFS",
        "lakeFS가 응답하지 않습니다. `make up`·`make health`로 확인하세요.",
    ),
    (r"mlflow|MLflow", "MLflow가 응답하지 않습니다. `make up`·`make health`로 확인하세요."),
)

# 오류 줄로 보이는 글귀
_ERROR_LINE = re.compile(r"(Error|Exception|오류|실패|거부|FAIL|아직 완료가 아니다|없습니다)")


def last_error_line(lines: Sequence[str]) -> str | None:
    """로그 꼬리에서 오류로 보이는 마지막 줄 (없으면 마지막 비어 있지 않은 줄)."""
    body = [x for x in lines if x.strip() and not x.startswith("$ dlp ") and "[종료 코드" not in x]
    for line in reversed(body):
        if _ERROR_LINE.search(line):
            return line.strip()
    return body[-1].strip() if body else None


def next_action(lines: Sequence[str], default: str) -> str:
    """로그 꼬리에 맞는 안내. 맞는 규칙이 없으면 default."""
    text = "\n".join(lines)
    for pattern, hint in _RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return hint
    return default
