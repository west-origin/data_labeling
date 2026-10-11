"""CLI의 검수 정책 선택. 계정/원본 권한 설정은 기본 정책과 별도 파일로 관리할 수 있다."""

from __future__ import annotations

import os
from pathlib import Path

from dlp_review.ops.policy import ReviewOpsPolicy, load_policy


def load_review_policy(root: Path) -> ReviewOpsPolicy:
    """DLP_REVIEW_POLICY_FILE이 있으면 그 파일을 엄격히 읽는다. 오류 시 기본값으로 되돌리지 않는다.

    기본 YAML은 권한자가 없는 상태를 유지한다. 로컬 파일럿·운영의 실제 계정 연결은 별도 파일에
    설정한다. 라이브러리의 load_policy(root)는 환경 변수와 무관하게 기본 정책을 읽는다.
    """
    selected = os.environ.get("DLP_REVIEW_POLICY_FILE")
    return load_policy(root, Path(selected) if selected else None)
