"""블러 사진 → pycolmap SfM → cameras.json, points.npz (격리 환경, ADR 0066).

TODO(real-model): CPU판 pycolmap 4.2.1로 무늬가 있는 합성 방 그림에서만 돌려 봤다 (README.md).
GPU 서버의 잠금 환경과 실제 사진에서는 돌려 보지 않았다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycolmap


def _pose(image: pycolmap.Image) -> pycolmap.Rigid3d:
    """cam_from_world (판에 따라 속성 또는 메서드)."""
    cfw = image.cam_from_world
    return cfw() if callable(cfw) else cfw


def main() -> None:
    """특징 추출 → 전수 대응 → 증분 SfM → 가장 큰 복원을 파일로 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    db = a.out / "database.db"
    reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
    pycolmap.extract_features(db, a.images, reader_options=reader)
    pycolmap.match_exhaustive(db)
    maps = pycolmap.incremental_mapping(db, a.images, a.out)
    rec = max(maps.values(), key=lambda r: r.num_reg_images())
    cams = []
    for image in rec.images.values():
        cam = rec.cameras[image.camera_id]
        pose = _pose(image)
        fx, fy, cx, cy = cam.params[:4]
        cams.append(
            {
                "photo_id": Path(image.name).stem,
                "width": cam.width,
                "height": cam.height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "rotation": np.asarray(pose.rotation.matrix()).reshape(-1).tolist(),
                "translation": np.asarray(pose.translation).tolist(),
            }
        )
    (a.out / "cameras.json").write_text(json.dumps(cams), "utf-8")
    desc_cache: dict[int, np.ndarray] = {}
    pts, descs, cols = [], [], []
    # pycolmap 4.x: Database는 생성자가 없고 open()으로 연다. read_descriptors는 기술자 묶음
    # (FeatureDescriptors)을 돌려주며 배열은 그 .data다 (CPU판 4.2.1로 확인, README.md)
    with pycolmap.Database.open(db) as database:
        for p in rec.points3D.values():
            track = []
            for el in p.track.elements:
                if el.image_id not in desc_cache:
                    desc_cache[el.image_id] = np.asarray(
                        database.read_descriptors(el.image_id).data, dtype=np.float64
                    )
                track.append(desc_cache[el.image_id][el.point2D_idx])
            pts.append(np.asarray(p.xyz))
            descs.append(np.mean(track, axis=0))
            cols.append(np.asarray(p.color))
    np.savez(
        a.out / "points.npz",
        points=np.asarray(pts),
        descriptors=np.asarray(descs),
        colors=np.asarray(cols, dtype=np.uint8),
    )


if __name__ == "__main__":
    main()
