"""탐지기 구현과 정책에서 탐지기를 만드는 공장."""

from __future__ import annotations

from pathlib import Path

from dlp_models.owlv2 import Owlv2
from dlp_models.registry import resolve
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors.codes import CodeDetector
from dlp_privacy.detectors.open_vocab import OpenVocabDetector
from dlp_privacy.detectors.reflection import ReflectionDetector
from dlp_privacy.detectors.yunet import YuNetFaceDetector
from dlp_privacy.policy import DetectorSpec, PrivacyPolicy
from dlp_schema.predictor import ModelUnavailableError


def build_detectors(
    policy: PrivacyPolicy, root: Path, extra: dict[str, FrameDetector] | None = None
) -> tuple[dict[str, FrameDetector], dict[str, str]]:
    """(쓸 수 있는 탐지기, 쓸 수 없는 탐지기 → 이유).

    extra는 같은 이름의 탐지기를 덮어쓴다 (테스트·stub용).
    """
    ready: dict[str, FrameDetector] = dict(extra or {})
    missing: dict[str, str] = {}

    def make(name: str) -> FrameDetector | None:
        if name in ready:
            return ready[name]
        if name in missing:
            return None
        spec = policy.detectors.get(name)
        try:
            if spec is None:
                raise ModelUnavailableError(f"정책에 탐지기 {name}이 없습니다")
            if spec.kind == "opencv_codes":
                det: FrameDetector = CodeDetector(name, _required(spec.score, name, "score"))
            elif spec.kind == "yunet":
                det = YuNetFaceDetector(name, *resolve(root, spec.model or "yunet"))
            elif spec.kind == "open_vocab":
                det = _open_vocab(name, spec, root)
            elif spec.kind == "reflection":
                region = make(spec.region_detector or "")
                face = make(spec.face_detector or "")
                if region is None or face is None:
                    absent = [
                        n
                        for n, d in ((spec.region_detector, region), (spec.face_detector, face))
                        if d is None
                    ]
                    raise ModelUnavailableError(
                        f"반사면 탐지에 필요한 탐지기가 없습니다: {', '.join(map(str, absent))}"
                    )
                scale = _required(spec.threshold_scale, name, "threshold_scale")
                det = ReflectionDetector(name, region, face, scale)
            else:
                raise ModelUnavailableError(f"{name}: 이 환경에서 쓸 수 없는 탐지기 ({spec.kind})")
        except ModelUnavailableError as exc:
            missing[name] = str(exc)
            return None
        ready[name] = det
        return det

    for target in policy.targets.values():
        for name in target.detectors:
            make(name)
    return ready, missing


def _required(value: float | None, name: str, field: str) -> float:
    """정책 값은 코드 기본값으로 채우지 않는다 (config/policies/privacy.yaml)."""
    if value is None:
        raise ValueError(f"privacy.yaml detectors.{name}.{field}가 없습니다")
    return value


def _open_vocab(name: str, spec: DetectorSpec, root: Path) -> FrameDetector:
    if not spec.queries or spec.model is None or spec.tokenizer is None:
        raise ModelUnavailableError(f"{name}: 모델·토크나이저·질의가 정책에 없습니다")
    model, version = resolve(root, spec.model)
    tokenizer, _ = resolve(root, spec.tokenizer)
    queries = list(spec.queries)
    return OpenVocabDetector(
        name,
        Owlv2(model, tokenizer, queries),
        [spec.queries[q] for q in queries],
        version=version,
        frame_stride_ms=spec.frame_stride_ms,
        score_threshold=_required(spec.score_threshold, name, "score_threshold"),
        score_full=_required(spec.score_full, name, "score_full"),
    )
