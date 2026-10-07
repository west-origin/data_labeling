from __future__ import annotations

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from dlp_fixtures.sync import SyncScenario
from dlp_media.glove import read_glove
from dlp_media.imu import read_imu_table
from dlp_media.storage import ImmutableObjectError, LocalStore, put_immutable
from dlp_media.tables import read_parquet


def test_imu_sidecar_parquet_and_csv(sync: tuple[SyncScenario, Path], tmp_path: Path) -> None:
    scenario, out = sync
    imu = read_imu_table(out / "imu.parquet")
    assert np.allclose(imu.acc[:, 2], scenario.imu["az"])
    assert imu.sample_rate_hz == pytest.approx(200.0)

    csv = tmp_path / "imu.csv"
    cols = ["t_ms", "ax", "ay", "az", "gx", "gy", "gz"]
    data = np.stack([scenario.imu[c] for c in cols], axis=1)[:100]
    np.savetxt(csv, data, delimiter=",", header=",".join(cols), comments="")
    from_csv = read_imu_table(csv)
    assert np.allclose(from_csv.gyro, data[:, 4:])

    imu.write(tmp_path / "norm.parquet")
    cols_back, meta = read_parquet(tmp_path / "norm.parquet")
    assert set(cols_back) == set(cols) and meta["source"] == "sidecar:parquet"

    bad = tmp_path / "bad.csv"
    bad.write_text("t_ms,ax\n0,1\n1,2\n")
    with pytest.raises(ValueError, match="IMU 열"):
        read_imu_table(bad)
    single = tmp_path / "single.csv"
    single.write_text("t_ms,ax,ay,az,gx,gy,gz\n0,0,0,9.8,0,0,0\n")
    with pytest.raises(ValueError, match="2개 미만"):
        read_imu_table(single)


def test_glove_parquet_and_hdf5(sync: tuple[SyncScenario, Path], tmp_path: Path) -> None:
    scenario, out = sync
    glove = read_glove(out / "glove_right.parquet")
    assert set(glove.channels) == {f"pressure_{i}" for i in range(5)}
    assert glove.sample_rate_hz == pytest.approx(100.0)

    h5 = tmp_path / "glove.h5"
    t = scenario.glove_right["t_ms"]
    pressure = np.stack([scenario.glove_right[f"pressure_{i}"] for i in range(5)], axis=1)
    with h5py.File(h5, "w") as f:
        f["timestamp_ms"] = t
        f["pressure"] = pressure
        f["temperature"] = np.full(t.size, 31.5)
    from_h5 = read_glove(h5)
    assert np.array_equal(from_h5.t_ms, t)
    assert np.array_equal(from_h5.channels["pressure_3"], pressure[:, 3])
    assert "temperature" in from_h5.channels
    assert "timestamp_ms" not in from_h5.channels

    # 시각 열이 둘이면 남은 시각 열도 채널이 아니다 (압력 합에 섞이지 않게)
    both = tmp_path / "both.h5"
    with h5py.File(both, "w") as f:
        f["t_ms"] = t
        f["timestamp_ms"] = t + 5.0
        f["pressure"] = pressure
    assert set(read_glove(both).channels) == {f"pressure_{i}" for i in range(5)}

    with h5py.File(tmp_path / "bad.h5", "w") as f:
        f["t_ms"] = np.array([0.0, 10.0, 5.0])
        f["pressure"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="증가"):
        read_glove(tmp_path / "bad.h5")
    with h5py.File(tmp_path / "notime.h5", "w") as f:
        f["pressure"] = np.zeros((3, 2))
    with pytest.raises(ValueError, match="시각 열"):
        read_glove(tmp_path / "notime.h5")


def test_local_store_is_immutable_and_idempotent(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store", "dlp-raw")
    src = tmp_path / "a.bin"
    src.write_bytes(b"first")
    assert put_immutable(store, "sessions/s/raw/a.bin", src) is True
    assert put_immutable(store, "sessions/s/raw/a.bin", src) is False
    src.write_bytes(b"changed")
    with pytest.raises(ImmutableObjectError):
        put_immutable(store, "sessions/s/raw/a.bin", src)
    with pytest.raises(ValueError, match="버킷 밖"):
        store.head("../../etc/passwd")
