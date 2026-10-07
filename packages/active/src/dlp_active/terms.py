"""세션 점수 항목 (플러그인).

항목은 `register_term`으로 이름을 붙여 등록하고, 정책 score.terms에 이름과 가중치를 적어 켠다.
기준 문서대로 처음에는 correction_rate 하나만 쓰고, 운영이 안정되면 하나씩 추가한다
(후보: 불확실성, 모델 간 불일치, 신규성, 희소 클래스, 임베딩 다양성).

항목은 세션 문맥(SessionContext)을 받아 0 이상 값을 낸다. 세션이 클수록 커지는 합계여야 정책의
normalize(per_minute)가 의미가 있다.

새 항목 추가 방법:
    class MyTerm:
        name = "my_term"
        def score(self, ctx: SessionContext) -> float: ...
        def explain(self, ctx: SessionContext) -> dict[str, float]: ...

    register_term("my_term")(lambda policy: MyTerm())

그리고 active.yaml `score.terms`에 `my_term: <가중치>`를 적는다. 등록은 import 때 일어나므로 항목을
정의한 모듈이 점수 계산 전에 import되어야 한다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from dlp_active.policy import ActivePolicy
from dlp_active.rates import RateTable, class_key
from dlp_schema.labels import LabelRecord
from dlp_schema.session import Session


@dataclass(frozen=True)
class SessionContext:
    """점수 항목이 받는 세션 문맥."""

    session: Session
    pending: list[LabelRecord]  # 아직 검수하지 않은 운영 모델 라벨 (블러 등 제외 종류 빼고)
    rates: RateTable  # 전체 세션에서 센 클래스별 수정률


class ScoreTerm(Protocol):
    """점수 항목 인터페이스."""

    name: str  # 등록 이름과 같아야 한다 (`SessionScore.terms`의 키)

    def score(self, ctx: SessionContext) -> float:
        """세션 점수 (0 이상, 세션이 클수록 커지는 합계)."""
        ...

    def explain(self, ctx: SessionContext) -> dict[str, float]:
        """점수에 크게 기여한 것 (클래스 → 기여). 리포트용."""
        ...


# 정책을 받아 항목 객체를 만드는 함수
TermFactory = Callable[[ActivePolicy], ScoreTerm]
# 등록된 항목: 이름 → 팩토리 (모듈 전역, import 때 채워진다)
TERMS: dict[str, TermFactory] = {}


def register_term(name: str) -> Callable[[TermFactory], TermFactory]:
    """점수 항목 팩토리를 이름으로 등록하는 데코레이터.

    Raises:
        ValueError: 같은 이름이 이미 등록돼 있을 때.
    """

    def deco(factory: TermFactory) -> TermFactory:
        """`factory`를 `TERMS[name]`에 넣고 그대로 돌려준다."""
        if name in TERMS:
            raise ValueError(f"점수 항목 이름이 겹칩니다: {name}")
        TERMS[name] = factory
        return factory

    return deco


def build_terms(policy: ActivePolicy) -> list[tuple[ScoreTerm, float]]:
    """정책 `score.terms`에 켠 항목을 만든다.

    Returns:
        (항목, 가중치) 목록 (정책에 적힌 순서).

    Raises:
        ValueError: 등록되지 않은 항목 이름이 있을 때.
    """
    unknown = sorted(set(policy.score.terms) - set(TERMS))
    if unknown:
        raise ValueError(f"등록되지 않은 점수 항목: {unknown} (있는 것: {sorted(TERMS)})")
    return [(TERMS[name](policy), w) for name, w in policy.score.terms.items()]


class CorrectionRateTerm:
    """예상 수정 수 = Σ (검수 대기 라벨 클래스의 수정률).

    수정률 높은 작업·객체를 많이 담은 세션이 앞선다.
    """

    name = "correction_rate"

    def score(self, ctx: SessionContext) -> float:
        """검수 대기 라벨마다 그 클래스 수정률을 더한다."""
        return sum(ctx.rates.rate(class_key(x)) for x in ctx.pending)

    def explain(self, ctx: SessionContext) -> dict[str, float]:
        """클래스별 예상 수정 수."""
        out: dict[str, float] = {}
        for x in ctx.pending:
            k = class_key(x)
            out[k] = out.get(k, 0.0) + ctx.rates.rate(k)
        return out


class UncertaintyTerm:
    """Σ (1 - 모델 신뢰도). 신뢰도 보정(ECE)이 확인된 뒤에 켠다."""

    name = "uncertainty"

    def score(self, ctx: SessionContext) -> float:
        """클래스별 불확실성의 합."""
        return sum(self.explain(ctx).values())

    def explain(self, ctx: SessionContext) -> dict[str, float]:
        """클래스별 Σ(1 - 신뢰도). 신뢰도가 없는 라벨은 1로 보아 0을 더한다."""
        out: dict[str, float] = {}
        for x in ctx.pending:
            k = class_key(x)
            out[k] = out.get(k, 0.0) + 1.0 - (x.confidence if x.confidence is not None else 1.0)
        return out


# 기본 항목 등록 (이 모듈 import 때)
register_term("correction_rate")(lambda policy: CorrectionRateTerm())
register_term("uncertainty")(lambda policy: UncertaintyTerm())
