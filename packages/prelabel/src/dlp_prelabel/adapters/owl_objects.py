"""OWLv2 오픈 보캐뷸러리 객체 탐지 (CPU): COCO에 없는 청소·돌봄 도구 박스 트랙.

CPU에서 프레임당 수 초라 frame_stride_ms 간격으로만 추론하고, 트랙 사이 빈 시각은 남겨 둔다
(검수 도구가 키프레임 사이를 보간한다). 질의 문장과 온톨로지 객체의 대응은
config/policies/prelabel.yaml open_vocab_objects에 있다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from dlp_models.owlv2 import Owlv2
from dlp_models.registry import resolve
from dlp_prelabel.adapters.mediapipe_models import boxes_to_labels
from dlp_prelabel.common import iter_frames, track_boxes
from dlp_prelabel.policy import PrelabelPolicy
from dlp_schema.labels import LabelRecord
from dlp_schema.predictor import Clip


class OwlObjects:
    name = "tools"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        self.model_path, version = resolve(root, policy.models.open_vocab)
        self.tokenizer_path, _ = resolve(root, policy.models.open_vocab_tokenizer)
        self.version = f"owlv2-{version}+p{policy.digest('open_vocab_objects')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        op = self.policy.open_vocab_objects
        queries = list(op.queries)
        model = Owlv2(self.model_path, self.tokenizer_path, queries, nms_iou=op.nms_iou)
        thresholds = [op.min_score] * len(queries)
        detections: list[
            tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]
        ] = []
        last: int | None = None
        for t, img in iter_frames(clip.video):
            if last is not None and t - last < op.frame_stride_ms:
                continue
            last = t
            found = model.detect(img, thresholds)
            detections.append(
                (t, [(op.queries[queries[d.query_index]], d.box, d.score) for d in found])
            )
        return boxes_to_labels(
            track_boxes(detections, iou_match=op.track_iou, max_gap_ms=op.max_gap_ms),
            clip,
            self.version,
            self.ontology_version,
            self.now,
            prefix="tools",
            entity_prefix="ov_",  # COCO 탐지 트랙(같은 클래스일 수 있다)과 ID가 겹치지 않게
        )
