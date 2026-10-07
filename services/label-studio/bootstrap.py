"""개발용 Label Studio 조직에서 레거시 API 토큰을 켠다.

Label Studio 1.23은 새 조직의 레거시 토큰 인증을 기본으로 끈다. 관리자 계정으로 세션 로그인한 뒤
조직 JWT 설정에서 legacy_api_tokens_enabled를 켜고, 토큰이 동작하는지 확인한다. 여러 번 실행해도
된다. 표준 라이브러리만 쓴다 (`python services/label-studio/bootstrap.py`).

`make up`이 서비스 기동 뒤 .env를 불러와 실행한다. `dlp review`의 Label Studio 클라이언트는
`Authorization: Token <DLP_LABEL_STUDIO_TOKEN>`으로 접속하므로 이 설정이 필요하다.
읽는 환경 변수: `DLP_LABEL_STUDIO_PORT`, `DLP_LABEL_STUDIO_USER`, `DLP_LABEL_STUDIO_PASSWORD`,
`DLP_LABEL_STUDIO_TOKEN` (없으면 .env.example과 같은 개발 기본값).
종료 코드: 0 성공(이미 동작 포함), 1 로그인 페이지 형식이 다르거나 켠 뒤에도 인증 실패.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import re
import sys
import urllib.parse
import urllib.request


def env(name: str, default: str) -> str:
    """환경 변수 `name`의 값, 없으면 `default`."""
    return os.environ.get(name, default)


def main() -> int:
    """토큰이 이미 동작하면 끝내고, 아니면 로그인 → JWT 설정 변경 → 재확인한다.

    흐름:
    1. `GET /api/projects`를 토큰으로 호출해 이미 동작하는지 본다.
    2. 로그인 페이지에서 CSRF 토큰을 긁어 이메일·비밀번호로 세션 로그인한다 (쿠키 저장).
    3. `POST /api/jwt/settings`로 레거시·API 토큰을 모두 켠다 (CSRF 쿠키를 헤더로).
    4. 다시 1번으로 확인한다.
    연결 실패(서버 미기동)는 예외로 끝난다.
    """
    base = f"http://localhost:{env('DLP_LABEL_STUDIO_PORT', '8081')}"
    email = env("DLP_LABEL_STUDIO_USER", "admin@dlp.local")
    password = env("DLP_LABEL_STUDIO_PASSWORD", "dlp-dev-password")
    token = env("DLP_LABEL_STUDIO_TOKEN", "dlpdevlabelstudiotoken0000000000000000")

    def token_ok() -> bool:
        """토큰으로 프로젝트 목록 API를 불러 2xx면 참. HTTP 오류(401 등)면 거짓, 연결 오류는
        예외."""
        req = urllib.request.Request(
            f"{base}/api/projects", headers={"Authorization": f"Token {token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                return True
        except urllib.error.HTTPError:
            return False

    if token_ok():
        print("Label Studio 레거시 토큰: 이미 동작")
        return 0

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    page = opener.open(f"{base}/user/login/", timeout=10).read().decode()
    match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page)
    if match is None:
        print("로그인 페이지에서 CSRF 토큰을 찾지 못했습니다", file=sys.stderr)
        return 1
    form = urllib.parse.urlencode(
        {"csrfmiddlewaretoken": match.group(1), "email": email, "password": password}
    ).encode()
    opener.open(
        urllib.request.Request(
            f"{base}/user/login/", data=form, headers={"Referer": f"{base}/user/login/"}
        ),
        timeout=10,
    )
    csrf = next(c.value for c in jar if c.name == "csrftoken")
    req = urllib.request.Request(
        f"{base}/api/jwt/settings",
        data=json.dumps({"legacy_api_tokens_enabled": True, "api_tokens_enabled": True}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-CSRFToken": csrf, "Referer": base},
    )
    opener.open(req, timeout=10)
    if not token_ok():
        print("레거시 토큰을 켰지만 인증에 실패했습니다", file=sys.stderr)
        return 1
    print("Label Studio 레거시 토큰: 켬")
    return 0


if __name__ == "__main__":
    sys.exit(main())
