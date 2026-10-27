"""AI 1차 검수 판정의 표시 보정 산출물 (ADR 0062, ADR 0040 6절 보강).

보정은 사람 판정 기록(`ai_review_outcomes`의 대조군 행)으로 "이 칸의 의심·틀림 지적이 실제로 사람이
고치게 만든 비율(지적 정밀도)"을 추정한 표다. 학습은 `dlp_review.ops.ai_calibration.fit_calibration`
(`dlp aireview calibrate`)이 하고, 결과는 정책이 아니라 버전 붙은 파일(`AiCalibration`, 내용 해시
`calibration_id` + 파일 sha256)이다. review.yaml `ai_review.calibration.artifact`가 경로·sha256으로
고정하면 CLI가 읽어(`load_calibration`) `AiDisplayConfig.calibration`으로 넘기고, `ai_display`가
검수 화면에 따로 표시할 지적만 거른다.

칸과 찾기 (`AiCalibration.cell`): (판정, 라벨 종류, 클래스) → (판정, 라벨 종류) → (판정) 순으로 가장
좁은 칸을 쓴다. 칸이 없으면(대조군 유효 표본 부족) 보정하지 않는다 (지금 동작 그대로 표시).

칸이 하는 일 (표시만):
- show 거짓: 그 칸의 지적은 따로 표시하지 않는다 (노출 기록 hidden). 정밀도 95% 상한이 정책 문턱보다
  낮을 때만 (확실히 쓸모없는 지적).
- min_confidence: 확신도가 이보다 낮은 지적은 따로 표시하지 않는다 (그 아래 묶음이 확실히 쓸모없을
  때만 정한다). 확신도가 없는 판정은 이 문턱으로 숨기지 않는다.
- weight: 같은 심각도 안에서 표시 순서 (10분당 상한·CVAT 이슈 상한이 걸릴 때 정밀도 높은 칸을 먼저).

바꾸지 않는 것 (I9): AI의 ok, unsure, 증가 프로필(우선순위 사유·표본 묶음 제외·마무리 보호), 판정
자체. 숨긴 지적의 라벨도 사람이 그대로 검수한다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from dlp_schema.common import Contract

KIND = "dlp-aireview-calibration"
FORMAT = 1


class AiCalibrationError(ValueError):
    """보정 산출물을 쓸 수 없음 (파일 없음·sha256 불일치·내용 해시 불일치·형식)."""


class CalibrationCell(Contract):
    """보정 칸 하나. kind·label_class가 None이면 그만큼 넓은 칸이다.

    n: 대조군 행 수, n_eff: 유효 표본 수 (표본 가중), precision / upper95: 지적 정밀도(행 가중)와
    한쪽 95% 상한, show: 따로 표시하는가, min_confidence: 표시 최소 확신도 (None이면 없음), weight:
    표시 순서 가중 (사전 정밀도로 수축한 정밀도).
    """

    verdict: Literal["wrong", "suspect"]
    kind: str | None = None
    label_class: str | None = None
    n: int = Field(ge=0)
    n_eff: float = Field(ge=0)
    precision: float = Field(ge=0, le=1)
    upper95: float = Field(ge=0, le=1)
    show: bool
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    weight: float = Field(ge=0, le=1)


class AiCalibration(Contract):
    """보정 산출물 (파일 하나). 모듈 docstring.

    calibration_id: `calibration_id` 필드를 뺀 내용의 정렬 JSON sha256 앞 12자. judge_versions:
    학습에 쓴 판정 버전 (표시 허용 버전이 모두 여기 들어야 쓴다). since·until: 결과 기록 기간
    [since, until).
    fit: 학습 설정 (review.yaml `ai_review.calibration.fit` 값). source: 행 수·대조군 행 수·행 해시.
    prior: 판정 값 → 대조군 전체 지적 정밀도. cells: 칸 (판정·종류·클래스 순 정렬).
    """

    kind: Literal["dlp-aireview-calibration"] = KIND
    format: Literal[1] = FORMAT
    calibration_id: str = Field(pattern=r"^[0-9a-f]{12}$")
    judge_versions: tuple[str, ...]
    since: datetime
    until: datetime
    fit: dict[str, Any]
    source: dict[str, Any]
    prior: dict[str, float]
    cells: tuple[CalibrationCell, ...]

    def cell(
        self, verdict: str, kind: str | None, label_class: str | None
    ) -> CalibrationCell | None:
        """가장 좁은 칸 (판정·종류·클래스 → 판정·종류 → 판정). 없으면 None (보정 없음)."""
        index = {(c.verdict, c.kind, c.label_class): c for c in self.cells}
        keys: list[tuple[str, str | None, str | None]] = []
        if kind is not None and label_class is not None:
            keys.append((verdict, kind, label_class))
        if kind is not None:
            keys.append((verdict, kind, None))
        keys.append((verdict, None, None))
        return next((index[k] for k in keys if k in index), None)

    def allows(
        self, verdict: str, kind: str | None, label_class: str | None, confidence: float | None
    ) -> bool:
        """그 지적을 화면에 따로 표시해도 되는가 (칸이 없으면 참, ok·unsure는 늘 참)."""
        c = self.cell(verdict, kind, label_class)
        if c is None:
            return True
        if not c.show:
            return False
        return c.min_confidence is None or confidence is None or confidence >= c.min_confidence

    def weight(self, verdict: str, kind: str | None, label_class: str | None) -> float:
        """표시 순서 가중 (칸이 없으면 그 판정의 사전 정밀도, 그것도 없으면 0)."""
        c = self.cell(verdict, kind, label_class)
        return c.weight if c is not None else self.prior.get(verdict, 0.0)


def content_id(data: dict[str, Any]) -> str:
    """`calibration_id`를 뺀 내용의 정렬 JSON sha256 앞 12자."""
    body = {k: v for k, v in data.items() if k != "calibration_id"}
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def to_bytes(calibration: AiCalibration) -> bytes:
    """파일 바이트 (정렬 JSON, 같은 내용이면 같은 바이트)."""
    data = calibration.model_dump(mode="json")
    return (json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def load_calibration(path: Path, sha256: str) -> AiCalibration:
    """고정한 보정 산출물을 읽는다 (파일 sha256·내용 해시 확인).

    Raises:
        AiCalibrationError: 파일 없음, sha256이 고정 값과 다름, 형식 오류, 내용 해시가
            calibration_id와 다름.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AiCalibrationError(f"보정 산출물을 읽을 수 없습니다: {path} ({exc})") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        raise AiCalibrationError(f"보정 산출물 sha256이 고정 값과 다릅니다: {path} ({digest})")
    try:
        data: dict[str, Any] = json.loads(raw)
        calibration = AiCalibration.model_validate(data)
    except ValueError as exc:
        raise AiCalibrationError(f"보정 산출물 형식이 맞지 않습니다: {path} ({exc})") from exc
    if content_id(data) != calibration.calibration_id:
        raise AiCalibrationError(f"보정 산출물 내용 해시가 calibration_id와 다릅니다: {path}")
    return calibration
