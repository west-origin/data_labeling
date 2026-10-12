"""센서(IMU 사이드카·장갑) 정규화와 로컬 저장소 불변성 테스트 (dlp_media.imu·glove·storage, WP3).

정답 근거: 동기화 픽스처의 IMU(200 Hz)·장갑(100 Hz, pressure_0~4) 배열을 그대로 비교한다.
"""

from __future__ import annotations

from pathlib import Path

import h5py  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import pytest

from dlp_fixtures.sync import SyncScenario
from dlp_media.audit import AuditedStore, MemorySink
from dlp_media.glove import read_glove
from dlp_media.imu import read_imu_table
from dlp_media.storage import (
    DeletableStore,
    ImmutableObjectError,
    ListableStore,
    LocalStore,
    put_immutable,
    sha256_file,
)
from dlp_media.tables import read_parquet


def test_imu_sidecar_parquet_and_csv(sync: tuple[SyncScenario, Path], tmp_path: Path) -> None:
    """IMU 사이드카 Parquet·CSV를 읽고 정규화 Parquet으로 쓴다.

    값이 픽스처와 같고 200 Hz이며, 정규화 파일의 열·출처 메타데이터가 맞다. 열이 빠지거나
    샘플이 하나뿐이면 ValueError.
    """
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
    """장갑 Parquet·HDF5를 같은 형태로 정규화한다.

    HDF5의 2차원 pressure(N, 5)는 pressure_0~4 채널로 펼치고, 1차원 temperature는 채널, 시각 열
    (timestamp_ms)은 채널이 아니다. 시각 열이 둘이면 남은 시각 열도 채널에서 뺀다. 시각이
    증가하지 않거나 시각 열이 없으면 ValueError.
    """
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
    """로컬 저장소 불변 업로드: 같은 내용은 건너뛰고(False), 다른 내용은 ImmutableObjectError.

    버킷 밖을 가리키는 키("../")는 ValueError.
    """
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


def test_local_store_delete_is_idempotent_and_raw_audit_store_cannot_delete(tmp_path: Path) -> None:
    """라벨링 버킷 사본 삭제(ADR 0039): 객체와 해시 옆 파일을 함께 지우고, 다시 지워도 성공한다.

    원본 감사 저장소(`AuditedStore`)는 삭제가 없어 `DeletableStore`가 아니다 (원본 불변 유지).
    버킷 밖을 가리키는 키는 지우기 전에 ValueError.
    """
    store = LocalStore(tmp_path / "store", "dlp-labeling")
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    key = "sessions/s/review/a/bodycam.mp4"
    store.put_file(key, src, sha256_file(src))
    assert isinstance(store, DeletableStore)
    store.delete(key)
    assert store.head(key) is None
    assert not (store.root / (key + ".sha256")).exists()
    store.delete(key)  # 없어도 성공 (멱등)
    with pytest.raises(ValueError, match="버킷 밖"):
        store.delete("../outside.mp4")
    audited = AuditedStore(
        LocalStore(tmp_path / "store", "dlp-raw"), MemorySink(), actor="t", purpose="test"
    )
    assert not isinstance(audited, DeletableStore)


def test_local_store_lists_objects_by_prefix_without_sidecars(tmp_path: Path) -> None:
    """고아 사본 찾기(F-REV-15)용 나열: 접두사로 고르고 키 순, 해시 옆 파일은 객체가 아니다.

    원본 감사 저장소(`AuditedStore`)는 나열이 없어 `ListableStore`가 아니다 (원본 목록을 감사 경로
    밖으로 내지 않는다). 버킷이 비었으면 빈 목록이다.
    """
    store = LocalStore(tmp_path / "store", "dlp-labeling")
    assert store.list_objects("sessions/") == []
    src = tmp_path / "v.mp4"
    src.write_bytes(b"video")
    keys = [
        "sessions/s2/review/a/bodycam.mp4",
        "sessions/s1/review/b/bodycam.mp4",
        "sessions/s1/blurred/bodycam.mp4",
    ]
    for key in keys:
        store.put_file(key, src, sha256_file(src))
    assert isinstance(store, ListableStore)
    got = store.list_objects("sessions/s1/review/")
    assert [o.key for o in got] == ["sessions/s1/review/b/bodycam.mp4"]
    assert got[0].size == 5 and got[0].modified_at.utcoffset() is not None
    assert [o.key for o in store.list_objects("sessions/")] == sorted(keys)
    audited = AuditedStore(
        LocalStore(tmp_path / "store", "dlp-raw"), MemorySink(), actor="t", purpose="test"
    )
    assert not isinstance(audited, ListableStore)
