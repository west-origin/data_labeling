"""COCO (객체 박스·키포인트): 정답 키프레임 시각의 블러본 프레임을 이미지로 낸다.

`dlp export coco <버전>`이 쓴다. 출력: `out/coco/annotations.json`과 `out/coco/images/*.jpg`.
pycocotools로 읽고 평가할 수 있어야 한다 (테스트가 확인).

- 이미지: 세션·스트림·시각(t_ms)마다 하나. 파일은 블러본에서 그 시각에 정확히 있는 프레임
  (JPEG). 키프레임이 블러본 프레임 시각에 없으면(허용 오차 밖) 그 주석은 버리고 센다.
  남은 주석이 하나도 없는 프레임은 이미지도 내지 않는다 (내보낸 세션 = 주석이 들어간 세션).
- 세션·라벨 ID는 내보내기마다 다른 가명이다 (dlp_export.pseudonym).
- 범주: 온톨로지 객체(이름순) + 키포인트 범주 hand(hand21), person(coco17; 객체 person이 있으면
  그 범주에 키포인트를 붙인다). 키포인트 범주에 든 박스만의 주석(예: person box_track)은
  num_keypoints 0과 0으로 채운 keypoints(3*K)를 가진다 (COCO 키포인트 평가가 모든 주석에서 읽는다).
- 주석마다 track_id(개체), label_id, 검증 상태, 출처, 모델 버전, 신뢰도를 붙인다
  (보간한 값은 넣지 않는다).

시각 규약 (ADR 0019): 박스·키포인트 키프레임 시각은 그 스트림 영상의 PTS 시각이므로 마스터 시각으로
바꾸지 않고 그대로 블러본 PTS 인덱스에서 찾는다 (3인칭 스트림도 같다). 이미지 `t_ms`도 스트림 시각.
버린 주석 사유(`CocoResult.dropped`): keyframe_between_frames, unknown_class,
no_visible_keypoints, unsupported_skeleton.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from dlp_export.frames import decode_frames, exact_frame
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.source import ExportSource, fetch_blurred
from dlp_media.probe import probe
from dlp_media.pts import build_pts_index
from dlp_media.storage import ObjectStore
from dlp_schema.labels import BoxTrackPayload, KeypointTrackPayload, LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.session import StreamKind

# hand21 관절 이름 (MediaPipe 손 21점 순서). LeRobot state 열 이름에도 쓴다
HAND21 = (
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)  # fmt: skip
# hand21 뼈대 (0부터 번호). COCO는 1부터라 categories에서 +1 한다
HAND21_SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
                   [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                   [18, 19], [19, 20], [5, 9], [9, 13], [13, 17]]  # fmt: skip
# COCO person 17관절 이름 (공식 순서)
COCO17 = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)  # fmt: skip
# COCO 공식 person 뼈대 (이미 1부터 번호)
COCO17_SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7],
                   [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
                   [4, 6], [5, 7]]  # fmt: skip
# 이미지를 내는 영상 스트림 종류
VIDEO = (StreamKind.BODYCAM, StreamKind.THIRD_PERSON)


def categories(ontology: Ontology) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """(범주 목록, 객체 클래스 → ID, 골격 → ID). ID는 1부터, 같은 온톨로지면 같다.

    순서: 온톨로지 객체(이름순) → (객체 person이 없으면) person 범주 → hand 범주.
    person 범주(객체든 새로 만든 것이든)에 coco17 keypoints·skeleton을 붙인다.
    """
    cats: list[dict[str, Any]] = []
    obj: dict[str, int] = {}
    for i, name in enumerate(sorted(ontology.objects), start=1):
        cats.append({"id": i, "name": name, "supercategory": "object"})
        obj[name] = i
    skel: dict[str, int] = {}
    if "person" in obj:
        # 객체 person 범주를 그대로 키포인트 범주로 쓴다 (박스와 관절이 같은 범주)
        cat = cats[obj["person"] - 1]
        skel["coco17"] = obj["person"]
    else:
        skel["coco17"] = len(cats) + 1
        cat: dict[str, Any] = {"id": skel["coco17"], "name": "person", "supercategory": "person"}
        cats.append(cat)
    cat |= {"keypoints": list(COCO17), "skeleton": COCO17_SKELETON}
    hand: dict[str, Any] = {
        "id": len(cats) + 1,
        "name": "hand",
        "supercategory": "person",
        "keypoints": list(HAND21),
        "skeleton": [[a + 1, b + 1] for a, b in HAND21_SKELETON],
    }
    cats.append(hand)
    skel["hand21"] = hand["id"]
    return cats, obj, skel


def _meta(x: LabelRecord, ids: Pseudonymizer) -> dict[str, Any]:
    """주석에 붙이는 출처 정보 (라벨 ID는 가명, 검수자 ID는 넣지 않는다)."""
    return {
        "label_id": ids.label(x.label_id),
        "verification": x.verification.state.value,
        "source": x.provenance.source.value,
        "model_version": x.provenance.model_version,
        "score": x.confidence,
    }


@dataclass
class CocoResult:
    """`write_coco` 결과. sessions·labels는 내부 ID (이력·계보용)."""

    images: int = 0  # 낸 이미지 수
    annotations: int = 0  # 낸 주석 수
    dropped: dict[str, int] = field(default_factory=dict[str, int])  # 사유 → 버린 주석 수
    sessions: set[str] = field(default_factory=set[str])  # 주석이 하나라도 들어간 세션
    labels: dict[str, LabelRecord] = field(default_factory=dict[str, LabelRecord])  # 쓴 라벨

    def written(self, x: LabelRecord, session_id: str) -> None:
        """라벨 x의 주석을 하나 이상 결과에 넣었다고 기록한다."""
        self.sessions.add(session_id)
        self.labels[x.label_id] = x


def write_coco(
    src: ExportSource,
    policy: ExportPolicy,
    ontology: Ontology,
    labeling: ObjectStore,
    out: Path,
    work: Path,
    *,
    export_id: str,
    now: datetime,
    ids: Pseudonymizer,
) -> CocoResult:
    """out/coco/annotations.json과 images/. 세션·라벨 ID와 트랙 ID 속 세션 ID는 ids로 가명 처리한다.

    결과의 sessions·labels는 내부 ID다 (내보내기 이력·사용 중지 계보용).

    Args:
        src: `load_source` 결과.
        policy: 내보내기 정책 (`coco.jpeg_quality`, `coco.frame_tolerance_ms`).
        ontology: 범주를 만들 온톨로지.
        labeling: 라벨링 버킷 (블러본을 받는다, `fetch_blurred`).
        out: 결과 루트. work: 블러본 캐시 디렉터리.
        export_id, now: info에 기록.
        ids: 가명 함수.

    Raises:
        ExportError: 블러본을 받을 수 없을 때 (`fetch_blurred`).

    이미지 파일 이름: "<세션 가명>__<스트림>__<t_ms 9자리>.jpg". 이미지·주석 ID는 1부터 차례로.
    """
    cats, obj_ids, skel_ids = categories(ontology)
    # 키포인트 범주 ID → 관절 수 (박스만의 주석도 그 범주면 키포인트 필드를 0으로 채운다)
    n_points = {c["id"]: len(c["keypoints"]) for c in cats if "keypoints" in c}
    images: list[dict[str, Any]] = []
    anns: list[dict[str, Any]] = []
    result = CocoResult()
    img_dir = out / "coco" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    def drop(reason: str) -> None:
        """버린 주석을 사유별로 센다."""
        result.dropped[reason] = result.dropped.get(reason, 0) + 1

    for es in src.sessions:
        s = es.session
        for stream in (st for st in s.streams if st.kind in VIDEO):
            spatial = [
                x
                for x in es.labels
                if x.stream_id == stream.stream_id
                and isinstance(x.payload, BoxTrackPayload | KeypointTrackPayload)
            ]
            # 공간 라벨이 없는 스트림은 블러본을 받지도 않는다
            if not spatial:
                continue
            video = fetch_blurred(labeling, es, stream, work)
            index = build_pts_index(video)
            info = probe(video).video
            assert info is not None
            # 키프레임(스트림 시각) → 블러본 프레임. 이미지는 프레임마다 하나다
            items: list[tuple[int, LabelRecord, Any]] = []  # (프레임, 라벨, 키프레임)
            for x in spatial:
                p = x.payload
                assert isinstance(p, BoxTrackPayload | KeypointTrackPayload)
                for k in p.keyframes:
                    # 화면 밖 표시(outside) 키프레임은 주석이 아니다 (박스만 가진 필드)
                    if getattr(k, "outside", False):
                        continue
                    f = exact_frame(index, k.t_ms, policy.coco.frame_tolerance_ms)
                    if f is None:
                        drop("keyframe_between_frames")
                        continue
                    items.append((f, x, k))
            # 주석을 먼저 만들고 걸러낸 뒤, 남은 주석이 하나라도 있는 프레임만 이미지로 낸다
            # (주석이 모두 버려진 프레임·세션의 이미지는 내보내지 않는다)
            kept: list[tuple[int, LabelRecord, dict[str, Any]]] = []  # (프레임, 라벨, 주석)
            for f, x, k in items:
                p = x.payload
                base = {"iscrowd": 0} | _meta(x, ids)
                if isinstance(p, BoxTrackPayload):
                    if p.class_id not in obj_ids:
                        drop("unknown_class")
                        continue
                    cid = obj_ids[p.class_id]
                    ann = base | {
                        "category_id": cid, "track_id": ids.ref(s.session_id, p.entity_id),
                        "bbox": [k.x, k.y, k.w, k.h], "area": k.w * k.h,
                    }  # fmt: skip
                    if cid in n_points:
                        ann |= {"keypoints": [0] * (3 * n_points[cid]), "num_keypoints": 0}
                    kept.append((f, x, ann))
                elif isinstance(p, KeypointTrackPayload) and p.skeleton in skel_ids:
                    # COCO keypoints: [x1, y1, v1, x2, …]. 보이지 않는(v=0) 관절은 (0, 0, 0)
                    flat: list[float] = []
                    xs: list[float] = []
                    ys: list[float] = []
                    for q in k.points:
                        flat += [q.x, q.y, q.visibility] if q.visibility else [0.0, 0.0, 0]
                        if q.visibility:
                            xs.append(q.x)
                            ys.append(q.y)
                    if not xs:
                        drop("no_visible_keypoints")
                        continue
                    # 박스 = 보이는 관절을 감싸는 최소 사각형 (COCO 키포인트 평가의 면적 기준)
                    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
                    kept.append((f, x, base | {
                        "category_id": skel_ids[p.skeleton],
                        "track_id": ids.ref(s.session_id, p.entity_id),
                        "hand": p.hand.value if p.hand else None,
                        "keypoints": flat, "num_keypoints": len(xs),
                        "bbox": [min(xs), min(ys), bw, bh], "area": bw * bh,
                    }))  # fmt: skip
                else:
                    drop("unsupported_skeleton")
            if not kept:
                continue
            pseudo = ids.session(s.session_id)
            image_id: dict[int, int] = {}
            names: dict[int, str] = {}
            for f in sorted({f for f, _, _ in kept}):
                # 이미지 시각 = 그 프레임의 PTS 시각 (반올림한 정수 ms)
                t = round(float(index.ms[f]))
                iid = len(images) + 1
                image_id[f] = iid
                names[f] = f"{pseudo}__{stream.stream_id}__{t:09d}.jpg"
                images.append({
                    "id": iid,
                    "file_name": f"images/{names[f]}",
                    "width": info.width, "height": info.height,
                    "session_id": pseudo, "stream_id": stream.stream_id,
                    "t_ms": t, "split": es.split.value,
                })  # fmt: skip
            # 필요한 프레임만 한 번 순서대로 디코딩해 JPEG로 쓴다
            for f, rgb in decode_frames(video, set(image_id)):
                ok, buf = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, policy.coco.jpeg_quality],
                )
                assert ok
                (img_dir / names[f]).write_bytes(buf.tobytes())
            for f, x, ann in kept:
                anns.append({"id": len(anns) + 1, "image_id": image_id[f]} | ann)
                result.written(x, s.session_id)
    coco = {
        "info": {
            "description": "dlp export", "version": src.version.version_id,
            "export_id": export_id, "ontology_version": src.version.ontology_version,
            "label_states": [s.value for s in src.label_states],
            "date_created": now.isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": cats,
    }  # fmt: skip
    (out / "coco" / "annotations.json").write_text(json.dumps(coco, ensure_ascii=False), "utf-8")
    result.images, result.annotations = len(images), len(anns)
    return result
