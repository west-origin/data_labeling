"""고정한 정답과 지금 정답의 차이 경고 (`dlp_eval.runner.truth_lag`, ADR 0041).

DB 없이 `GoldenSession`을 직접 만든다. 정답은 합성 행동 시나리오
(`dlp_fixtures.actions.generate_action_scenario`)의 사람 행동 라벨이다 (사람 출처 = 정답).

정답 근거:
- 고정한 세션이 모두 사용 중지되면 평가 세션이 비지만 정답은 고정돼 있었다. "라벨링·검수 전에
  고정한 버전"이 아니라 "세션이 모두 사용 중지"로 알리고, 실패할 `--from` 대신 새 골든셋 제안
  (`--domain <도메인> --create`)을 안내한다 (`--from`은 세션이 모두 사용 중지면 거부된다).
- 일부 세션만 사용 중지되어 남은 세션에 고정 정답이 없으면 빠진 고정 정답 수를 따로 보인다.
- 정말로 라벨링 전에 고정한 버전(고정 정답 0개)은 이전과 같이 안내한다.
"""

from __future__ import annotations

from dlp_eval.runner import GoldenSession, truth_lag
from dlp_fixtures.actions import generate_action_scenario
from dlp_schema.labels import LabelRecord
from dlp_schema.lineage import GoldenSet
from dlp_schema.session import Domain
from dlp_schema.testing import FIXED_TIME, make_session


def _truth(sid: str, seed: int) -> list[LabelRecord]:
    """세션 `sid`의 사람 행동 라벨 (합성 시나리오 `seed`)."""
    labels = generate_action_scenario(seed, session_id=sid).labels
    return [x for x in labels if x.kind == "action"]


def _golden(truth_ids: tuple[str, ...], sessions: tuple[str, ...]) -> GoldenSet:
    """청소 도메인 골든셋 golden-v2 (정답 `truth_ids`를 고정)."""
    return GoldenSet(
        version="golden-v2",
        domain=Domain.CLEANING,
        session_ids=sessions,
        created_at=FIXED_TIME,
        truth_label_ids=truth_ids,
    )


def _evaluated(sid: str, labels: list[LabelRecord], frozen: set[str]) -> GoldenSession:
    """평가하는 골든 세션 (`golden_sessions`가 만드는 것과 같이 고정 ID로 정답을 고른다)."""
    truth = [x for x in labels if x.label_id in frozen]
    return GoldenSession(make_session(sid), labels, truth, {"glove": "bare", "site": "site01"})


def test_all_sessions_withdrawn_is_not_reported_as_frozen_before_labeling() -> None:
    """세션이 모두 사용 중지: 고정 정답 수를 보이고 `--from`이 아닌 새 골든셋 제안을 안내한다."""
    s1 = _truth("s1", 0)
    golden = _golden(tuple(sorted(x.label_id for x in s1)), ("s1",))
    lag = truth_lag(golden, [])  # golden_sessions는 사용 중지 세션을 뺀다
    assert lag is not None
    assert lag.all_withdrawn and lag.withdrawn == len(s1) and lag.frozen == 0
    msg = lag.message()
    assert "모두 사용 중지" in msg and f"사용 중지로 빠진 고정 정답 {len(s1)}개" in msg
    assert "라벨링·검수 전에 고정한 버전" not in msg
    assert "--from golden-v2" not in msg  # 세션이 모두 사용 중지면 --from은 거부된다
    assert "`dlp dataset golden --domain cleaning --create <새 버전>`" in msg
    assert lag.remedy() in msg


def test_partly_withdrawn_golden_counts_dropped_frozen_truth() -> None:
    """남은 세션에 고정 정답이 없으면 사용 중지로 빠진 고정 정답 수를 따로 보인다 (`--from` 안내).

    s1(정답 고정)은 사용 중지, s2는 고정할 때 라벨이 없었고 지금은 사람 라벨이 있다.
    """
    s1, s2 = _truth("s1", 0), _truth("s2", 1)
    golden = _golden(tuple(sorted(x.label_id for x in s1)), ("s1", "s2"))
    lag = truth_lag(golden, [_evaluated("s2", s2, set(golden.truth_label_ids or ()))])
    assert lag is not None and not lag.all_withdrawn
    assert (lag.frozen, lag.added, lag.withdrawn) == (0, len(s2), len(s1))
    msg = lag.message()
    assert f"사용 중지로 빠진 고정 정답 {len(s1)}개" in msg
    assert "라벨링·검수 전에 고정한 버전" not in msg
    assert "--from golden-v2 --create" in msg  # s2가 남아 --from으로 다시 고정할 수 있다


def test_truth_frozen_before_labeling_and_consistent_truth() -> None:
    """라벨링 전에 고정(정답 0개)은 그대로 안내하고, 남은 세션의 정답이 같으면 경고가 없다."""
    s1, s2 = _truth("s1", 0), _truth("s2", 1)
    empty = truth_lag(_golden((), ("s1",)), [_evaluated("s1", s1, set())])
    assert empty is not None and (empty.frozen, empty.withdrawn) == (0, 0)
    assert "라벨링·검수 전에 고정한 버전" in empty.message()
    assert "--from golden-v2 --create" in empty.message()
    # s2가 사용 중지됐어도 남은 s1의 고정 정답 = 지금 정답이면 평가에 문제가 없다
    frozen = {x.label_id for x in [*s1, *s2]}
    golden = _golden(tuple(sorted(frozen)), ("s1", "s2"))
    assert truth_lag(golden, [_evaluated("s1", s1, frozen)]) is None
