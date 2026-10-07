"""OWLv2 오픈 보캐뷸러리 객체 탐지 (CPU): COCO에 없는 청소·돌봄 도구 박스 트랙.

CPU에서 프레임당 수 초라 frame_stride_ms 간격으로만 추론하고, 트랙 사이 빈 시각은 남겨 둔다 (검수
도구가 키프레임 사이를 보간한다). 질의 문장과 온톨로지 객체의 대응은 config/policies/prelabel.yaml
open_vocab_objects에 있다.

모델 입출력·전처리는 `dlp_models.owlv2` 참고 (960 정사각 패딩, CLIP 정규화, 픽셀 박스로 복원).
출력 좌표계: 바디캠 스트림 픽셀 (x, y, w, h), 왼쪽 위 원점. 키프레임 시각은 스트림 PTS ms.
라이선스: `config/models.yaml owlv2_onnx`(review), `owlv2_tokenizer`(allowed), ADR 0010.
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
    """OWLv2 도구 박스 어댑터 (name="tools").

    version: `owlv2-<가중치 버전>+p<open_vocab_objects 절 해시>`. 라벨 ID는
    `<세션>-<스트림>-tools-<version_tag>-<클래스>-<순번>`, entity_id는 `ov_<클래스>_<순번>` (COCO
    탐지 트랙과 겹치지 않게 `ov_` 접두사).
    """

    name = "tools"

    def __init__(
        self, root: Path, policy: PrelabelPolicy, *, ontology_version: str, now: datetime
    ) -> None:
        """가중치·토크나이저를 확인하고 버전을 정한다 (모델은 `run`에서 연다).

        Raises:
            ModelUnavailableError: 가중치가 없거나 해시가 다를 때 (`make models`).
        """
        self.model_path, version = resolve(root, policy.models.open_vocab)
        self.tokenizer_path, _ = resolve(root, policy.models.open_vocab_tokenizer)
        self.version = f"owlv2-{version}+p{policy.digest('open_vocab_objects')}"
        self.policy, self.ontology_version, self.now = policy, ontology_version, now

    def run(self, clip: Clip) -> list[LabelRecord]:
        """영상 하나를 frame_stride_ms 간격으로 탐지해 박스 트랙 라벨로 돌려준다.

        질의 순서는 정책 `queries`의 키 순서다. 탐지마다 질의 → 온톨로지 클래스로 바꾸고,
        클래스별로 IoU 추적한다. 호출마다 ONNX 세션을 새로 연다 (스트림 하나에 한 번).
        """
        op = self.policy.open_vocab_objects
        queries = list(op.queries)
        model = Owlv2(self.model_path, self.tokenizer_path, queries, nms_iou=op.nms_iou)
        thresholds = [op.min_score] * len(queries)
        detections: list[
            tuple[int, list[tuple[str, tuple[float, float, float, float], float]]]
        ] = []
        last: int | None = None
        for t, img in iter_frames(clip.video):
            # 마지막 추론 프레임에서 frame_stride_ms가 지나기 전 프레임은 건너뛴다 (디코드는 한다)
            if last is not None and t - last < op.frame_stride_ms:
                continue
            last = t
            found = model.detect(img, thresholds)
            detections.append(
                # 질의 번호 → 질의 문장 → 온톨로지 클래스 (트랙 키)
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
