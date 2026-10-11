"""검수 패키지 테스트 공용 픽스처·도우미.

CVAT 계정 연결(`review.yaml cvat.users` 자리)을 시험용으로 만든다. CVAT가 닿으면 시험 검수자마다
일반 사용자 계정을 만들어 두고, 닿지 않으면 연결 정보만 돌려준다 (CVAT가 필요한 테스트는 따로
건너뛴다). Label Studio 계정 연결(`review.yaml label_studio.users` 자리)도 같다: Label Studio가
닿으면 시험 검수자마다 조직 초대 링크로 일반 계정을 만들고, 검수자 제출은 그 계정의 토큰으로 낸다
(F-REV-06: 수집은 담당자 계정의 주석만 받는다). 실제 사용자·개인정보는 쓰지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import httpx
import pytest

from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.ops.policy import CvatPolicy, LabelStudioPolicy

# 테스트가 쓰는 dlp 검수자 ID (블러 검수 담당자와 작업 라벨 담당자)
TEST_REVIEWERS = ("rev01", "privacy01", "p1", "p2", "labeler01")
# Label Studio 시간 검수를 맡는 시험 검수자 (labeler02는 담당자가 아닌 다른 라벨러 역할)
LS_TEST_REVIEWERS = ("labeler01", "labeler02", "r1", "r2")
CVAT_TEST_PASSWORD = "Kq7zr-Wv2mN8-xT4"  # 개발용 CVAT의 시험 계정 (일반 사용자, 관리자 아님)


def cvat_username(reviewer: str) -> str:
    """시험 검수자 ID → 시험용 CVAT 사용자 이름 (`dlp-test-<검수자>`)."""
    return f"dlp-test-{reviewer}"


def ensure_cvat_user(cvat: CvatClient, username: str) -> int:
    """개발용 CVAT에 일반 사용자 계정을 만든다 (있으면 그대로). 사용자 ID."""
    uid = cvat.find_user_id(username)
    if uid is not None:
        return uid
    with httpx.Client(base_url=str(cvat.http.base_url), timeout=60) as anon:
        resp = anon.post(
            "/api/auth/register",
            json={
                "username": username, "email": f"{username}@example.com",
                "password1": CVAT_TEST_PASSWORD, "password2": CVAT_TEST_PASSWORD,
                "first_name": "Review", "last_name": "Account",
            },
        )  # fmt: skip
        if resp.status_code >= 400:
            raise RuntimeError(f"CVAT 시험 계정 {username}을 만들 수 없습니다: {resp.text[:300]}")
    uid = cvat.find_user_id(username)
    assert uid is not None
    return uid


@pytest.fixture(scope="session")
def cvat_config() -> CvatPolicy:
    """검수자 → CVAT 계정 연결 (review.yaml cvat.users 자리). CVAT가 닿으면 계정을 만든다."""
    users = {r: cvat_username(r) for r in TEST_REVIEWERS}
    try:
        cvat = CvatClient.from_env()
    except httpx.HTTPError:
        pass  # CVAT가 필요한 테스트는 따로 건너뛴다
    else:
        for name in users.values():
            ensure_cvat_user(cvat, name)
    return CvatPolicy(users=users, privacy_issue_limit=50)


@pytest.fixture
def cvat_as() -> Callable[[CvatClient, str], CvatClient]:
    """시험 검수자의 일반 CVAT 계정으로 로그인한다 (라벨러 화면에서 보이는지 본다)."""

    def login(admin: CvatClient, reviewer: str) -> CvatClient:
        return CvatClient(str(admin.http.base_url), cvat_username(reviewer), CVAT_TEST_PASSWORD)

    return login


def ls_email(reviewer: str) -> str:
    """시험 검수자 ID → 시험용 Label Studio 계정 이메일 (`dlp-test-<검수자>@example.com`)."""
    return f"dlp-test-{reviewer}@example.com"


def _csrf(page: str) -> str:
    """Label Studio 로그인·가입 화면의 CSRF 토큰 (없으면 AssertionError)."""
    match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', page)
    assert match is not None, "Label Studio 화면에서 CSRF 토큰을 찾지 못했습니다"
    return match.group(1)


def ensure_ls_user(admin: LabelStudioClient, email: str) -> str:
    """개발용 Label Studio에 일반 계정을 만들고(있으면 그대로) 그 계정의 API 토큰을 돌려준다.

    가입은 조직 초대 링크(`/api/invite`의 토큰)로 한다 (compose가 링크 없는 가입을 막는다). 토큰은
    그 계정으로 로그인한 세션에서 `/api/current-user/token`으로 받는다.
    """
    base = str(admin.http.base_url).rstrip("/")
    with httpx.Client(base_url=base, timeout=60) as web:
        if admin.find_user_id(email) is None:
            invite = admin.http.get("/api/invite").json()["token"]
            page = f"/user/signup/?token={invite}"
            form = {
                "csrfmiddlewaretoken": _csrf(web.get(page).text), "email": email,
                "password": CVAT_TEST_PASSWORD, "how_find_us": "Other", "elaborate": "test",
            }  # fmt: skip
            resp = web.post(page, data=form, headers={"Referer": base + page})
            if resp.status_code >= 400 or admin.find_user_id(email) is None:
                raise RuntimeError(f"Label Studio 시험 계정 {email}을 만들 수 없습니다")
        else:
            form = {"csrfmiddlewaretoken": _csrf(web.get("/user/login/").text), "email": email,
                    "password": CVAT_TEST_PASSWORD}  # fmt: skip
            web.post("/user/login/", data=form, headers={"Referer": base + "/user/login/"})
        resp = web.get("/api/current-user/token")
        resp.raise_for_status()
        return str(resp.json()["token"])


# Label Studio 시험 계정의 토큰 (검수자 ID → 토큰). 세션 픽스처가 채운다
_LS_TOKENS: dict[str, str] = {}


@pytest.fixture(scope="session")
def ls_config() -> LabelStudioPolicy:
    """검수자 → Label Studio 계정 연결 (review.yaml label_studio.users 자리).

    Label Studio가 닿으면 계정을 만들고 토큰을 받아 둔다 (`ls_as`가 쓴다).
    """
    users = {r: ls_email(r) for r in LS_TEST_REVIEWERS}
    try:
        admin = LabelStudioClient.from_env()
        admin.service_user_id()
    except httpx.HTTPError:
        return LabelStudioPolicy(users=users)  # Label Studio가 필요한 테스트는 따로 실패·건너뛴다
    for reviewer, email in users.items():
        _LS_TOKENS[reviewer] = ensure_ls_user(admin, email)
    return LabelStudioPolicy(users=users)


@pytest.fixture
def ls_as(ls_config: LabelStudioPolicy) -> Callable[[LabelStudioClient, str], LabelStudioClient]:
    """시험 검수자의 Label Studio 계정 토큰으로 접속한다 (검수자의 제출을 그 계정으로 낸다)."""

    def login(admin: LabelStudioClient, reviewer: str) -> LabelStudioClient:
        return LabelStudioClient(str(admin.http.base_url).rstrip("/"), _LS_TOKENS[reviewer])

    return login
