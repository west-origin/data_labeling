from __future__ import annotations

import httpx
import pytest

from dlp_review.clients import CvatClient
from dlp_review.ops.policy import CvatPolicy

# 테스트가 쓰는 dlp 검수자 ID (블러 검수 담당자와 작업 라벨 담당자)
TEST_REVIEWERS = ("rev01", "privacy01", "p1", "p2", "labeler01")
CVAT_TEST_PASSWORD = "Kq7zr-Wv2mN8-xT4"  # 개발용 CVAT의 시험 계정 (일반 사용자, 관리자 아님)


def cvat_username(reviewer: str) -> str:
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
