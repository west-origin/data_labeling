"""탐지기 구현과 정책에서 탐지기를 만드는 공장.

TODO(real-model): 글자 탐지(OCR) 탐지기가 없다 (송장·문서·주소 표지는 OWLv2와 QR·바코드만).
  후보는 RapidOCR(PaddleOCR 검출 ONNX, Apache-2.0, CPU). config/policies/privacy.yaml 참조.

WP5, ADR 0005·0009. `dlp privacy detect`(`dlp_cli.privacy_cmds.cmd_detect`)가 `build_detectors`로
정책의 탐지기를 만들고, 만들 수 없는 탐지기(가중치 없음 등)는 이유와 함께 따로 돌려준다. 쓸 수 있는
탐지기가 하나도 없는 대상은 파이프라인이 영상 전체를 no_detector 검수 구간으로 낸다 (사람이 전부
본다).

구현 (`kind` → 클래스):
- `yunet` → `yunet.YuNetFaceDetector` (얼굴, OpenCV FaceDetectorYN, MIT 가중치)
- `opencv_codes` → `codes.CodeDetector` (QR·바코드 → shipping_label, 가중치 없음)
- `open_vocab` → `open_vocab.OpenVocabDetector` (OWLv2 ONNX, 문서·화면·사진물·문패·송장·반사면 영역)
- `reflection` → `reflection.ReflectionDetector` (반사면 영역 안에서 낮은 문턱으로 얼굴)
- `oracle` → `oracle.OracleDetector` (정답을 아는 CPU stub, 테스트·CI가 `extra`로 끼운다)
가중치 경로·버전은 `dlp_models.registry.resolve`가 config/models.yaml에서 찾고 sha256을 확인한다.
"""

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

    정책 대상(`targets.*.detectors`)이 쓰는 이름만 만든다. reflection은
    region_detector·face_detector를
    먼저 만들어 (재귀) 같은 인스턴스를 공유한다 (OWLv2 캐시 공유로 같은 프레임 추론이 한 번이다).

    Args:
        policy: 프라이버시 정책.
        root: 저장소 루트 (config/models.yaml, data/models/ 기준).
        extra: 미리 만든 탐지기 (이름 → 탐지기).

    Returns:
        (이름 → 탐지기, 이름 → 쓸 수 없는 이유). 이유에는 사람이 할 일(예: `make models`)이
        들어간다.

    Raises:
        ValueError: 정책에 필요한 값(score, nms_threshold 등)이 비어 있을 때 (`_required`).
            가중치가 없는 것은 오류가 아니라 missing으로 보고한다.
    """
    ready: dict[str, FrameDetector] = dict(extra or {})
    missing: dict[str, str] = {}

    def make(name: str) -> FrameDetector | None:
        """이름의 탐지기를 만들어 ready에 넣는다 (이미 있으면 그대로). 못 만들면 missing에 넣고
        None."""
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
                det = YuNetFaceDetector(
                    name,
                    *resolve(root, spec.model or "yunet"),  # (가중치 경로, 버전)
                    nms_threshold=_required(spec.nms_threshold, name, "nms_threshold"),
                    top_k=int(_required(spec.top_k, name, "top_k")),
                )
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
                # "unavailable"·"oracle" 등: 정책 파일만으로는 만들 수 없다
                # (oracle은 extra로 넣는다)
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
    """정책 값은 코드 기본값으로 채우지 않는다 (config/policies/privacy.yaml).

    Raises:
        ValueError: value가 None일 때 (어느 키가 빠졌는지 메시지에 적는다).
    """
    if value is None:
        raise ValueError(f"privacy.yaml detectors.{name}.{field}가 없습니다")
    return value


def _open_vocab(name: str, spec: DetectorSpec, root: Path) -> FrameDetector:
    """OWLv2 오픈 보캐뷸러리 탐지기를 만든다.

    질의 목록 순서와 대상 목록 순서를 맞춰 넘긴다 (질의 i의 탐지 → 대상 i).
    버전은 가중치 버전에 frame_stride_ms를 붙인 값이다 (`OpenVocabDetector.version`).

    Raises:
        ModelUnavailableError: 정책에 모델·토크나이저·질의가 없거나 가중치를 찾을 수 없을 때.
        ValueError: score_threshold·score_full이 없을 때.
    """
    if not spec.queries or spec.model is None or spec.tokenizer is None:
        raise ModelUnavailableError(f"{name}: 모델·토크나이저·질의가 정책에 없습니다")
    model, version = resolve(root, spec.model)
    tokenizer, _ = resolve(root, spec.tokenizer)
    queries = list(spec.queries)  # dict 삽입 순서 = YAML 순서
    return OpenVocabDetector(
        name,
        Owlv2(model, tokenizer, queries),
        [spec.queries[q] for q in queries],
        version=version,
        frame_stride_ms=spec.frame_stride_ms,
        score_threshold=_required(spec.score_threshold, name, "score_threshold"),
        score_full=_required(spec.score_full, name, "score_full"),
    )
