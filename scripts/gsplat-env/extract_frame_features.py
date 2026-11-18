"""바디캠 블러본 프레임 → pycolmap SIFT 특징 → frame_features.npz (격리 환경, ADR 0067).

`dlp scene3d localize`의 실제 특징 추출기(`dlp_scene3d.backends.IsolatedFrameFeatures`)가 부른다.
`--images` 디렉터리의 `<프레임 시각 ms>.png`마다 SIFT 특징을 뽑아 `--out` NPZ에 `kp_<시각>`
((M, 2) 화소, COLMAP 규약: 왼쪽 위 화소의 중심이 (0.5, 0.5))과 `desc_<시각>`((M, 128) uint8)을
쓴다. 특징이 없는 프레임은 빈 배열이다. 화소 규약은 어댑터가 작업공간 규약으로 바꾼다.

SfM(`run_sfm.py`)과 같은 추출기·같은 기술자 정규화를 써야 복원 점 기술자(트랙의 SIFT 기술자
평균)와 대응된다. 그래서 `pycolmap.extract_features`를 그대로 쓰고 특징 수 상한만 받는다 (방향이
둘인 특징은 둘로 세어 상한의 두 배쯤까지 나온다).

TODO(real-model): CPU판 pycolmap 4.2.1로 무늬가 있는 합성 방 그림에서만 돌려 봤다 (README.md).
GPU 서버의 잠금 환경과 실제 바디캠 프레임에서는 돌려 보지 않았다.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pycolmap

# SIFT 기술자 차원
_DIM = 128


def _features(db: pycolmap.Database, image_id: int) -> tuple[np.ndarray, np.ndarray]:
    """그림 한 장의 (특징 위치 (M, 2) float32, 기술자 (M, 128) uint8). 특징이 없으면 빈 배열."""
    keypoints = np.asarray(db.read_keypoints(image_id), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0, _DIM), np.uint8)
    descriptors = np.asarray(db.read_descriptors(image_id).data, dtype=np.uint8)
    return keypoints[:, :2], descriptors.reshape(len(keypoints), -1)


def main() -> None:
    """프레임 디렉터리의 특징을 뽑아 NPZ 하나로 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-features", type=int, required=True)
    a = ap.parse_args()
    options = pycolmap.FeatureExtractionOptions()
    options.sift.max_num_features = a.max_features
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    arrays: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "frames.db"
        pycolmap.extract_features(
            db_path, a.images, reader_options=reader, extraction_options=options
        )
        with pycolmap.Database.open(db_path) as db:
            for image in db.read_all_images():
                stem = Path(image.name).stem
                arrays[f"kp_{stem}"], arrays[f"desc_{stem}"] = _features(db, image.image_id)
    np.savez(a.out, **arrays)


if __name__ == "__main__":
    main()
