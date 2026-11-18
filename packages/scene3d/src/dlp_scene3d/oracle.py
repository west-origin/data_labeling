"""합성 픽스처 정답 파일(truth.json, `dlp_fixtures.scene3d.write_scene_pair`)에서 stub을 만든다.

`dlp scene3d ... --oracle truth.json`(로컬 시연)과 테스트가 쓴다. 정답의 사진은 파일 sha256으로
가리키므로 수집한 photo_id(`photos.photo_id_for`)로 바로 이어진다. 정책 파일로는 만들 수 없다.

- `load_oracle`: 한 단계의 프라이버시·SfM·물체 stub. truth에 `sfm_unit_scale`이 있으면 SfM
  stub이 좌표에 그 배율을 곱해 미터가 아닌 복원을 낸다 (마커 척도 시험, ADR 0067).
- `oracle_objects`: 두 단계 사진 모두의 정답 물체 상자 stub.
- `oracle_features`: truth의 `ego` 절(바디캠 프레임별 정답 자세, 물체·벽이 바뀐 시각)로 프레임
  특징 stub을 만든다. 그 시각에 방에 있는 점은 전 상태면 before 정답 점, 바뀐 뒤면 after 정답
  점(정답 닮음 변환으로 before 좌표에 옮긴 것)이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from dlp_scene3d.backends import (
    OracleFrameFeatures,
    OracleLayer,
    OracleObjectDetector,
    OracleSfm,
)
from dlp_scene3d.geometry import F64, Camera, Similarity
from dlp_scene3d.photos import photo_id_for
from dlp_scene3d.privacy_gate import OraclePhotoDetector
from dlp_schema.scene3d import ScenePhase

# 픽스처 얼굴 상자의 프라이버시 대상 ID (privacy.yaml targets)
FACE_TARGET = "face"


class OracleError(ValueError):
    """truth.json에 필요한 정답이 없다 (예: 바디캠 경로 없이 만든 픽스처로 위치 추정)."""


@dataclass(frozen=True)
class OracleSet:
    """한 단계의 stub 묶음."""

    privacy: OraclePhotoDetector
    sfm: OracleSfm
    objects: OracleObjectDetector


def _truth(truth_path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(truth_path.read_text("utf-8")))


def load_oracle(truth_path: Path, phase: ScenePhase) -> OracleSet:
    """truth.json의 한 단계 → (프라이버시·SfM·물체) stub."""
    truth = _truth(truth_path)
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
        unit_scale=float(truth.get("sfm_unit_scale", 1.0)),
    )
    return OracleSet(OraclePhotoDetector(faces), sfm, OracleObjectDetector(boxes))


def _box(b: list[float]) -> tuple[float, float, float, float]:
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))


def oracle_objects(truth_path: Path) -> OracleObjectDetector:
    """두 단계 사진 모두의 정답 물체 상자를 내는 stub (비교 단계는 탐지기 하나로 두 단계를 본다)."""
    b, a = load_oracle(truth_path, "before").objects, load_oracle(truth_path, "after").objects
    return OracleObjectDetector({**b.boxes, **a.boxes})


def _by_key(data: dict[str, Any], to_before: Similarity | None) -> dict[str, tuple[F64, F64]]:
    """한 단계 정답 점을 점 key별로 묶는다: key → (before 좌표 점, 기술자)."""
    points = np.asarray(data["points"], dtype=np.float64)
    if to_before is not None:
        points = to_before.apply(points)
    desc = np.asarray(data["descriptors"], dtype=np.float64)
    keys = np.asarray([str(k) for k in data["point_keys"]])
    return {str(k): (points[keys == k], desc[keys == k]) for k in sorted(set(keys.tolist()))}


def _layers(truth: dict[str, Any], change_ms: dict[str, int]) -> list[OracleLayer]:
    """프레임 시각별로 방에 있는 점 묶음 (모듈 설명)."""
    a2b = cast(dict[str, Any], truth["after_to_before"])
    to_before = Similarity(
        float(a2b["s"]),
        np.asarray(a2b["r"], dtype=np.float64),
        np.asarray(a2b["t"], dtype=np.float64),
    )
    before = _by_key(cast(dict[str, Any], truth["before"]), None)
    after = _by_key(cast(dict[str, Any], truth["after"]), to_before)
    layers: list[OracleLayer] = []
    for key, (points, desc) in before.items():
        # 고정 구조("")와 바뀐 시각이 없는 것은 끝까지 전 상태다
        layers.append(OracleLayer(points, desc, 0, change_ms.get(key) if key else None))
    for key, (points, desc) in after.items():
        if key and key in change_ms:
            layers.append(OracleLayer(points, desc, change_ms[key], None))
    return layers


def oracle_features(truth_path: Path) -> OracleFrameFeatures:
    """truth.json의 바디캠 정답 → 프레임 특징 stub.

    Raises:
        OracleError: truth에 바디캠 경로(`ego`)나 점 key(`point_keys`)가 없을 때.
    """
    truth = _truth(truth_path)
    ego = cast(dict[str, Any] | None, truth.get("ego"))
    if ego is None or "point_keys" not in truth["before"]:
        raise OracleError(f"{truth_path}: 바디캠 정답(ego)이 없습니다 (dlp scene3d fixture --ego)")
    poses = {
        int(f["t_ms"]): (np.asarray(f["r"], dtype=np.float64), np.asarray(f["t"], dtype=np.float64))
        for f in cast(list[dict[str, Any]], ego["frames"])
    }
    change_ms = {str(k): int(v) for k, v in cast(dict[str, Any], ego["change_ms"]).items()}
    return OracleFrameFeatures(
        np.asarray(ego["k"], dtype=np.float64),
        np.asarray(ego.get("distortion", []), dtype=np.float64),
        int(ego["width"]),
        int(ego["height"]),
        poses,
        _layers(truth, change_ms),
    )
