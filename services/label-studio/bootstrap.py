"""개발용 Label Studio 조직에서 레거시 API 토큰을 켠다.

Label Studio 1.23은 새 조직의 레거시 토큰 인증을 기본으로 끈다.
관리자 계정으로 세션 로그인한 뒤
조직 JWT 설정에서 legacy_api_tokens_enabled를 켜고, 토큰이 동작하는지 확인한다.
여러 번 실행해도 된다.
표준 라이브러리만 쓴다 (`python services/label-studio/bootstrap.py`).
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
    return os.environ.get(name, default)


def main() -> int:
    base = f"http://localhost:{env('DLP_LABEL_STUDIO_PORT', '8081')}"
    email = env("DLP_LABEL_STUDIO_USER", "admin@dlp.local")
    password = env("DLP_LABEL_STUDIO_PASSWORD", "dlp-dev-password")
    token = env("DLP_LABEL_STUDIO_TOKEN", "dlpdevlabelstudiotoken0000000000000000")

    def token_ok() -> bool:
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
