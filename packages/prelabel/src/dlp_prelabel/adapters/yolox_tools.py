"""고정 클래스 도구·가사 객체 탐지 (YOLOX, 공개 데이터 미세조정, ADR 0060): 도구 박스 트랙.

OWLv2(`owl_objects.OwlObjects`, 단계 `tools`)는 질의 문장으로 아무 도구나 찾지만 오탐이 많다
(F-PAR-60). 이 어댑터는 공개 데이터로 미세조정한 YOLOX가 정해진 클래스(`prelabel.yaml
closed_vocab_tools.classes`, 모델 출력 열 순서)만 낸다. 구조는 `OwlObjects`와 같다: frame_stride_ms
간격 추론 → 클래스별 IoU 추적 → 트랙 거르기(ADR 0054) → 착용자 몸 억제(ADR 0058) → box_track 라벨.

단계 이름 `closed_tools` (라벨 ID `<세션>-<스트림>-closed_tools-...`, entity_id
`cv_<클래스>_<순번>`):
- 러너는 `<세션>-<스트림>-<name>-` 접두사로 같은 단계의 이전 버전을 지운다. `tools`를 같이 쓰면 두
  도구 단계가 서로의 결과를 이전 버전으로 보고 지운다. `closed_tools-`는 `tools-`로 시작하지 않아
  접두사가 겹치지 않는다.
- entity_id `cv_`는 COCO 객체(접두사 없음)·OWLv2(`ov_`) 트랙과 겹치지 않게 한다 (같은 클래스일 수
  있다).
- 이름은 모델(yolox)이 아니라 역할이다. 가중치·구조가 바뀌어도 단계 이름(`--skip closed_tools`,
  training.yaml replaces)은 그대로다.

OWLv2와의 관계: 함께 돈다. `replace_open_vocab_classes`가 참이면 이 단계가 실제로 만들어졌을 때만
(`link_open_vocab`) 이 단계의 클래스를 OWLv2 질의에서 뺀다. `--skip closed_tools`나 가중치가 없으면
OWLv2가 지금처럼 모든 질의를 묻는다.

켜고 끄기: `prelabel.yaml models.closed_vocab_tools`가 null(기본)이면 `from_policy`가 None을 돌려줘
단계가 없다 (버전·라벨·OWLv2 질의 모두 지금 그대로).

version: `yolox-<가중치 버전>+p<closed_vocab_tools 절 해시><+w<wearer_suppression 절 해시>>`. 착용자
억제는 `wearer_suppression.stages`의 `tools`를 따른다 (도구 단계 둘을 함께 뜻한다, 그 절의 해시와
다른 단계 버전이 바뀌지 않게).

모델 입출력·전후처리는 `dlp_models.yolox` 참고. 출력 좌표계: 바디캠 스트림 픽셀 (x, y, w, h), 왼쪽
위 원점. 키프레임 시각은 스트림 PTS ms. CPU stub: `stubs.ScriptedToolDetector`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from dlp_models.registry import resolve
from dlp_models.yolox import Yolox, YoloxDetection
from dlp_prelabel.adapters.mediapipe_models import boxes_to_labels
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.common import iter_frames, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_prelabel.temporal import filter_tracks
from dlp_prelabel.wearer_suppress import WearerAware, suppress_box_tracks, wearer_tag
from dlp_schema.labels import LabelRecord
from dlp_schema.predictor import Clip, Predictor

STAGE = "closed_tools"  # 단계 이름 (라벨 ID 접두사, --skip)
WEARER_STAGE = "tools"  # wearer_suppression.stages에서 이 단계를 뜻하는 이름 (도구 단계 공통)
ENTITY_PREFIX = "cv_"  # entity_id 접두사 (closed vocabulary)


class ToolDetector(Protocol):
    """프레임 하나의 고정 클래스 탐지 (`dlp_models.yolox.Yolox`, CPU stub
    `ScriptedToolDetector`)."""

    def detect(self, rgb: NDArray[np.uint8], thresholds: Sequence[float]) -> list[YoloxDetection]:
        """(H, W, 3) RGB uint8 → 원래 픽셀 탐지 (class_index는 정책 classes 번호)."""
        ...


class YoloxTools(WearerAware):
    """고정 클래스 도구 박스 어댑터 (name="closed_tools"). 위 모듈 설명 참고."""

    name = STAGE

    def __init__(
        self,
        policy: PrelabelPolicy,
        detector: Callable[[], ToolDetector],
        weights_version: str,
        *,
        ontology_version: str,
        now: datetime,
    ) -> None:
        """버전을 정한다 (탐지기는 `run`에서 연다).

        Args:
            policy: 프리라벨 정책 (`closed_vocab_tools`, `wearer_suppression`).
            detector: 탐지기를 여는 함수 (스트림마다 한 번 부른다). 실제는 `from_policy`가 만든다.
            weights_version: 가중치 버전 (`registry.resolve`, stub이면 그 이름).
            ontology_version, now: 새 라벨에 넣을 값.
        """
        self.policy, self.detector, self.ontology_version, self.now = (
            policy,
            detector,
            ontology_version,
            now,
        )
        self.version = (
            f"yolox-{weights_version}+p{policy.digest('closed_vocab_tools')}"
            f"{wearer_tag(policy, WEARER_STAGE)}"
        )

    @classmethod
    def from_policy(
        cls, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> YoloxTools | None:
        """정책이 켠 경우 가중치를 확인해 어댑터를 만든다. `models.closed_vocab_tools`가 None이면
        None.

        Raises:
            ModelUnavailableError: 가중치가 없거나 해시가 다를 때 (`make models`).
        """
        name = policy.models.closed_vocab_tools
        if name is None:
            return None
        path, version = resolve(root, name)
        sec = policy.closed_vocab_tools

        def open_model() -> ToolDetector:
            return Yolox(
                path,
                len(sec.classes),
                input_size=sec.input_size,
                decoded=sec.decoded_output,
                nms_iou=sec.nms_iou,
            )

        return cls(policy, open_model, version, ontology_version=ontology_version, now=now)

    @property
    def open_vocab_exclusions(self) -> frozenset[str]:
        """OWLv2 질의에서 뺄 클래스 (replace_open_vocab_classes가 거짓이면 빈 집합)."""
        sec = self.policy.closed_vocab_tools
        return frozenset(sec.classes) if sec.replace_open_vocab_classes else frozenset()

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 하나를 frame_stride_ms 간격으로 탐지해 박스 트랙 라벨로 돌려준다.

        탐지의 클래스 번호 → 정책 classes의 온톨로지 ID (트랙 키). 클래스별로 IoU 추적한다.
        """
        sec = self.policy.closed_vocab_tools
        detector = self.detector()
        thresholds = sec.thresholds()
        detections: list[
            tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]
        ] = []
        last: int | None = None
        size = (0, 0)
        for t, img in iter_frames(clip.video):
            size = (int(img.shape[1]), int(img.shape[0]))
            # 마지막 추론 프레임에서 frame_stride_ms가 지나기 전 프레임은 건너뛴다 (디코드는 한다)
            if last is not None and t - last < sec.frame_stride_ms:
                continue
            last = t
            found = detector.detect(img, thresholds)
            detections.append((t, [(sec.classes[d.class_index], d.box, d.score) for d in found]))
        tracks = filter_tracks(
            track_boxes(detections, iou_match=sec.track_iou, max_gap_ms=sec.max_gap_ms),
            sec.track_filter,
        )
        # 착용자 자신의 팔·몸통을 덮은 큰 박스 트랙은 라벨로 만들지 않는다 (ADR 0058). 버전 태그와
        # 같은 조건(stages에 tools)일 때만 억제한다
        ws = self.policy.wearer_suppression
        self.last_wearer = None
        if WEARER_STAGE in ws.stages:
            tracks, self.last_wearer = suppress_box_tracks(tracks, self.wearer, ws, size)
        return boxes_to_labels(
            tracks,
            clip,
            self.version,
            self.ontology_version,
            self.now,
            prefix=self.name,
            entity_prefix=ENTITY_PREFIX,
        )


def link_open_vocab(predictors: Sequence[Predictor]) -> frozenset[str]:
    """고정 클래스 단계가 목록에 있으면 그 클래스를 OWLv2 단계 질의에서 뺀다.

    CLI가 어댑터를 다 만든 뒤(--skip·가중치 없음으로 빠진 단계는 목록에 없다) 부른다. 고정 클래스
    단계가 없거나 replace_open_vocab_classes가 거짓이면 아무것도 바꾸지 않는다.

    Returns:
        OWLv2 질의에서 실제로 뺀 클래스 (OWLv2 질의 대상인 것만, 없으면 빈 집합).
    """
    wanted = frozenset[str]().union(
        *(p.open_vocab_exclusions for p in predictors if isinstance(p, YoloxTools))
    )
    owls = [p for p in predictors if isinstance(p, OwlObjects)]
    if wanted:
        for p in owls:
            p.set_excluded_classes(wanted)
    return frozenset[str]().union(*(p.excluded for p in owls))
