"""탐지기 구현과 정책에서 탐지기를 만드는 공장."""

from __future__ import annotations

from pathlib import Path

from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors.codes import CodeDetector
from dlp_privacy.detectors.reflection import ReflectionDetector
from dlp_privacy.detectors.yunet import YuNetFaceDetector
from dlp_privacy.policy import PrivacyPolicy
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
                det: FrameDetector = CodeDetector(name)
            elif spec.kind == "yunet":
                det = YuNetFaceDetector(name, root / (spec.model_path or ""), spec.model_sha256)
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
                det = ReflectionDetector(name, region, face, spec.threshold_scale or 0.5)
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
