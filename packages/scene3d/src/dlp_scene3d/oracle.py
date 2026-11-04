"""합성 픽스처 정답 파일(truth.json, `dlp_fixtures.scene3d.write_scene_pair`)에서 stub을 만든다.

`dlp scene3d ... --oracle truth.json`(로컬 시연)과 테스트가 쓴다. 정답의 사진은 파일 sha256으로
가리키므로 수집한 photo_id(`photos.photo_id_for`)로 바로 이어진다. 정책 파일로는 만들 수 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dlp_scene3d.backends import OracleObjectDetector, OracleSfm
from dlp_scene3d.geometry import Camera
from dlp_scene3d.photos import photo_id_for
from dlp_scene3d.privacy_gate import OraclePhotoDetector
from dlp_schema.scene3d import ScenePhase

# 픽스처 얼굴 상자의 프라이버시 대상 ID (privacy.yaml targets)
FACE_TARGET = "face"


@dataclass(frozen=True)
class OracleSet:
    """한 단계의 stub 묶음."""

    privacy: OraclePhotoDetector
    sfm: OracleSfm
    objects: OracleObjectDetector


def load_oracle(truth_path: Path, phase: ScenePhase) -> OracleSet:
    """truth.json의 한 단계 → (프라이버시·SfM·물체) stub."""
    truth = cast(dict[str, Any], json.loads(truth_path.read_text("utf-8")))
    data = cast(dict[str, Any], truth[phase])
    cams: dict[str, Camera] = {}
    faces: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    boxes: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
    for p in cast(list[dict[str, Any]], data["photos"]):
        pid = photo_id_for(str(p["sha256"]))
        cams[pid] = Camera(
            pid,
            np.asarray(p["k"], dtype=np.float64),
            np.asarray(p["r"], dtype=np.float64),
            np.asarray(p["t"], dtype=np.float64),
            int(p["width"]),
            int(p["height"]),
        )
        faces[pid] = [(FACE_TARGET, _box(b)) for b in p["faces"]]
        boxes[pid] = [(str(d["label"]), _box(d["box"])) for d in p["detections"]]
    sfm = OracleSfm(
        cams,
        np.asarray(data["points"], dtype=np.float64),
        np.asarray(data["descriptors"], dtype=np.float64),
        metric=bool(truth["metric"]) if phase == "before" else False,
    )
    return OracleSet(OraclePhotoDetector(faces), sfm, OracleObjectDetector(boxes))


def _box(b: list[float]) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def oracle_objects(truth_path: Path) -> OracleObjectDetector:
    """두 단계 사진 모두의 정답 물체 상자를 내는 stub (비교 단계는 탐지기 하나로 두 단계를 본다)."""
    b, a = load_oracle(truth_path, "before").objects, load_oracle(truth_path, "after").objects
    return OracleObjectDetector({**b.boxes, **a.boxes})
