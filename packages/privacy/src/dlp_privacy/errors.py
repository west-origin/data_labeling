"""프라이버시 게이트·렌더 검증의 공용 예외."""


class PrivacyGateError(RuntimeError):
    """게이트 조건을 만족하지 않아 단계를 진행할 수 없다 (승인 전 렌더, 미검수 블러 등).

    메시지에 다음에 할 명령을 적는다.
    """


class RenderNotCurrentError(PrivacyGateError):
    """블러본이 지금 승인된 블러 라벨·렌더 정책으로 만든 것이 아니다 (쓰면 안 된다)."""
