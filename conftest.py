"""저장소 전체 테스트 공용 설정.

주변 AWS 설정과 테스트를 떼어 놓는다: 개발자 셸의 `AWS_PROFILE`이 없는 프로필을 가리키거나
`~/.aws/config`의 기본 프로필에 다른 끝점·지역이 있으면, 자격 증명을 명시해 만든 boto3
클라이언트도 프로필을 읽다가 실패하거나(`ProfileNotFound`) 설정이 섞인다. 저장소 코드
(`dlp_media.storage.S3Store`)는 끝점·키를 늘 명시하므로 프로필·설정 파일이 필요 없다. 서비스
테스트도 `.env`의 `DLP_S3_*`만 쓴다.
"""

from __future__ import annotations

import os

import pytest

# 테스트가 읽지 않게 지우는 AWS 프로필·설정 변수 (키 자체는 저장소 코드가 쓰지 않는다)
_AWS_PROFILE_VARS = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_S3",
)


@pytest.fixture(autouse=True)
def _isolate_aws_config(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """AWS 프로필 변수를 지우고 설정·자격 증명 파일을 빈 파일로 돌린다 (모든 테스트)."""
    for name in _AWS_PROFILE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", os.devnull)
