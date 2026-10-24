"""객체·도구 오탐 의심 힌트 (ADR 0055). 라벨을 바꾸지 않고 검수 화면에 표시만 한다.

규칙 (트랙마다, 정책 `review.yaml object_hints`):
- `wearer_body` (COCO person 박스): 키프레임 중 박스(여유 margin) 안에 MediaPipe 손목(착용자 손)이
  있고 보이는 RTMPose 머리 관절은 없는 비율이 wearer_share 이상. 1인칭 화면에서 COCO 사람 박스가
  착용자 자신의 팔·몸통을 덮는 경우다 (2026-10-22 파일럿). 판정 규칙은 프리라벨 억제(ADR 0058)와
  같은 `dlp_schema.wearer_body.wearer_box_frame`이다 (여기서는 프레임 높이를 몰라 아래 가장자리
  조건은 보지 않는다). 프리라벨이 이미 착용자 박스를 억제하므로 남은 것은 억제 문턱 아래 트랙이다.
- `ai_crop_no` (ADR 0056): 사내 VLM이 블러본 조각에 그 클래스가 없다고 답함.
- `tta_unstable` (TTA를 돌린 트랙): 좌우 뒤집기·배율 변환 화면에서 같은 자리 탐지가 다시 나온 비율의
  평균이 tta_min_support 미만 (`dlp_media.geometry.tta_support`).

저장: 라벨링 버킷 `sessions/<세션>/hints/<스트림>.objects.json` (라벨 ID·규칙·비율만, 원본 화소
없음). 라벨링 CVAT 작업 생성이 읽어 이슈로 단다 (`dlp_review.tasks`).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from dlp_media.storage import ObjectStore, sha256_file
from dlp_review.ops.policy import ObjectHintsPolicy
from dlp_schema.labels import BoxTrackPayload, LabelRecord
from dlp_schema.wearer_body import WearerEvidence, wearer_box_frame


@dataclass(frozen=True)
class ObjectHint:
    """오탐 의심 하나 (트랙 단위)."""

    label_id: str
    stream_id: str
    class_id: str
    reason: str  # wearer_body | tta_unstable
    share: float
    t_start_ms: int
    t_end_ms: int


def object_hints(
    labels: Sequence[LabelRecord],
    stream_id: str,
    policy: ObjectHintsPolicy,
    tta: Mapping[str, float] | None = None,
    ai: Mapping[str, float] | None = None,
) -> list[ObjectHint]:
    """한 스트림 운영 라벨에서 박스 트랙의 오탐 의심 힌트.

    Args:
        labels: 그 세션의 운영 라벨 (박스·손·COCO 트랙). 다른 스트림은 무시한다.
        tta: 박스 라벨 ID → TTA 지지 비율 평균. 없으면 그 규칙을 보지 않는다.
        ai: 박스 라벨 ID → AI 조각 확인의 의심 정도 (ADR 0056). 있으면 `ai_crop_no` 힌트.
    """
    own = [x for x in labels if x.stream_id == stream_id]
    evidence = WearerEvidence.from_labels(own, stream_id)
    tol, m = policy.time_tolerance_ms, policy.margin
    out: list[ObjectHint] = []
    for x in own:
        p = x.payload
        if not isinstance(p, BoxTrackPayload):
            continue
        found: list[tuple[str, float]] = []
        frames = [k for k in p.keyframes if not k.outside]
        if p.class_id == "person" and frames:
            own_body = sum(
                wearer_box_frame(
                    (k.x, k.y, k.w, k.h),
                    evidence.hands_near(k.t_ms, tol),
                    evidence.heads_near(k.t_ms, tol),
                    margin=m,
                    frame_h=None,
                    bottom_band=1.0,
                )
                for k in frames
            ) / len(frames)
            if own_body >= policy.wearer_share:
                found.append(("wearer_body", round(own_body, 4)))
        if tta is not None and x.label_id in tta and tta[x.label_id] < policy.tta_min_support:
            found.append(("tta_unstable", round(1 - tta[x.label_id], 4)))
        if ai is not None and x.label_id in ai:
            found.append(("ai_crop_no", ai[x.label_id]))
        out += [
            ObjectHint(x.label_id, stream_id, p.class_id, r, s, x.t_start_ms, x.t_end_ms)
            for r, s in found
        ]
    return out


def hints_key(session_id: str, stream_id: str) -> str:
    """라벨링 버킷의 객체 오탐 의심 파일."""
    return f"sessions/{session_id}/hints/{stream_id}.objects.json"


def write_object_hints(
    store: ObjectStore, session_id: str, stream_id: str, hints: Sequence[ObjectHint], work: Path
) -> str:
    """힌트를 통째로 덮어쓴다 (파생물). 키를 돌려준다. 부작용: 라벨링 버킷 쓰기."""
    out = work / f"{stream_id}-object-hints.json"
    out.write_text(json.dumps([asdict(h) for h in hints]), encoding="utf-8")
    key = hints_key(session_id, stream_id)
    store.put_file(key, out, sha256_file(out))
    return key


def read_object_hints(
    store: ObjectStore, session_id: str, stream_id: str, work: Path
) -> list[ObjectHint]:
    """저장한 힌트. 없으면 빈 목록."""
    key = hints_key(session_id, stream_id)
    if store.head(key) is None:
        return []
    dest = work / f"{stream_id}-object-hints.in.json"
    store.get_file(key, dest)
    data: list[dict[str, object]] = json.loads(dest.read_text(encoding="utf-8"))
    return [ObjectHint(**d) for d in data]  # type: ignore[arg-type]
