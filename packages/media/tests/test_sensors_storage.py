"""센서(IMU 사이드카·장갑) 정규화와 로컬 저장소 불변성 테스트 (dlp_media.imu·glove·storage, WP3).

정답 근거: 동기화 픽스처의 IMU(200 Hz)·장갑(100 Hz, pressure_0~4) 배열을 그대로 비교한다.
"""

from __future__ import annotations

import threading
import uuid
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
    S3Store,
    StoredObject,
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


def _race(store: object, key: str, tmp_path: Path, n: int = 4) -> dict[str, object]:
    """서로 다른 내용 n개를 같은 키로 동시에 put_immutable한다 → 이름 → True 또는 예외 이름."""
    gate = threading.Barrier(n)
    out: dict[str, object] = {}

    def go(name: str) -> None:
        src = tmp_path / f"{name}.bin"
        src.write_bytes(name.encode() * 1000)
        gate.wait()
        try:
            out[name] = put_immutable(store, key, src)  # pyright: ignore[reportArgumentType]
        except ImmutableObjectError as exc:
            out[name] = type(exc).__name__

    threads = [threading.Thread(target=go, args=(chr(65 + i),)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_local_store_concurrent_different_contents_have_one_winner(tmp_path: Path) -> None:
    """같은 원본 키에 다른 내용이 동시에 와도 하나만 올라가고 나머지는 ImmutableObjectError.

    F-PMS-40.

    예전에는 head → 복사 사이에 다른 쓰기가 끼어 둘 다 성공하고 나중 것이 원본을 덮었다. 정답:
    True는 하나, 저장된 내용·해시 옆 파일은 그 승자의 것이다.
    """
    store = LocalStore(tmp_path / "store", "dlp-raw")
    for round_ in range(5):
        key = f"sessions/s/raw/race{round_}.bin"
        got = _race(store, key, tmp_path)
        winners = [n for n, r in got.items() if r is True]
        assert len(winners) == 1, got
        assert set(got.values()) == {True, "ImmutableObjectError"}
        head = store.head(key)
        assert head is not None and head.sha256 == sha256_file(tmp_path / f"{winners[0]}.bin")
        assert (store.root / key).read_bytes() == (tmp_path / f"{winners[0]}.bin").read_bytes()


class _RacyStore:
    """조건부 쓰기가 없는 저장소: 두 쓰기가 모두 head를 지나고, 둘 다 쓴 뒤에 다시 본다."""

    bucket = "dlp-raw"

    def __init__(self) -> None:
        """head를 두 번 부를 때까지 기다리게 한다."""
        self.objects: dict[str, StoredObject] = {}
        self.gate = threading.Barrier(2)
        self.written = threading.Barrier(2)
        self.lock = threading.Lock()
        self.heads = 0

    def uri(self, key: str) -> str:
        """메모리 URI."""
        return f"mem://{key}"

    def head(self, key: str) -> StoredObject | None:
        """처음 두 번은 서로를 기다린 뒤 (둘 다 '없음'을 본 상태로) 답한다."""
        with self.lock:
            self.heads += 1
            first_two = self.heads <= 2
            got = self.objects.get(key)
        if first_two:
            self.gate.wait()
        return got

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """덮어쓴다 (조건 없음). 두 쓰기가 모두 끝날 때까지 기다린다."""
        with self.lock:
            self.objects[key] = StoredObject(key, path.stat().st_size, sha256)
        self.written.wait()

    def get_file(self, key: str, dest: Path) -> None:
        """쓰지 않는다."""
        raise NotImplementedError


def test_store_without_conditional_write_detects_overwrite_by_reading_back(
    tmp_path: Path,
) -> None:
    """조건부 쓰기가 없는 저장소는 쓰기 뒤 해시를 다시 보아, 덮인 쪽이 ImmutableObjectError를 낸다.

    정답: 둘 다 '없음'을 보고 썼어도 성공(True)은 마지막에 쓴 하나뿐이다. 한계: 앞 쓰기의 대조가
    뒤 쓰기보다 먼저 끝나면 둘 다 성공한다. 그래서 원본 저장소(LocalStore·S3Store)는 조건부 쓰기를
    하고 이 대조는 보조 수단이다.
    """
    got = _race(_RacyStore(), "sessions/s/raw/a.bin", tmp_path, n=2)
    assert sorted(map(str, got.values())) == ["ImmutableObjectError", "True"], got


def test_s3_create_only_sends_if_none_match_and_maps_412_to_false(tmp_path: Path) -> None:
    """S3Store.put_file_new는 단일 PUT·분할 업로드 완료 모두 `If-None-Match: *`를 건다 (F-PMS-40).

    네트워크 없이 botocore Stubber로 요청 인자를 대조한다. 412는 False(이미 있음), 분할 업로드가
    조건에 걸리면 업로드를 버린다(abort). 실제 SeaweedFS 동작은 아래 services 테스트가 본다.
    """
    from botocore.stub import ANY, Stubber

    store = S3Store("dlp-raw", endpoint_url="http://s3.invalid", access_key="a", secret_key="b")
    small = tmp_path / "small.bin"
    small.write_bytes(b"s" * 10)
    big = tmp_path / "big.bin"
    big.write_bytes(b"b" * 25)
    meta = {"sha256": "h"}
    with Stubber(store.client) as stub:
        put = {"Bucket": "dlp-raw", "Key": "k", "Body": ANY, "Metadata": meta, "IfNoneMatch": "*"}
        stub.add_response("put_object", {}, put)
        stub.add_client_error("put_object", "PreconditionFailed", http_status_code=412,
                              expected_params=put)  # fmt: skip
        stub.add_response(
            "create_multipart_upload",
            {"UploadId": "u"},
            {"Bucket": "dlp-raw", "Key": "m", "Metadata": meta},
        )
        for n in (1, 2, 3):
            stub.add_response(
                "upload_part",
                {"ETag": f"e{n}"},
                {"Bucket": "dlp-raw", "Key": "m", "UploadId": "u", "PartNumber": n, "Body": ANY},
            )
        stub.add_client_error(
            "complete_multipart_upload",
            "PreconditionFailed",
            http_status_code=412,
            expected_params={
                "Bucket": "dlp-raw",
                "Key": "m",
                "UploadId": "u",
                "MultipartUpload": {
                    "Parts": [{"ETag": f"e{n}", "PartNumber": n} for n in (1, 2, 3)]
                },
                "IfNoneMatch": "*",
            },
        )
        stub.add_response(
            "abort_multipart_upload", {}, {"Bucket": "dlp-raw", "Key": "m", "UploadId": "u"}
        )
        assert store.put_file_new("k", small, "h") is True
        assert store.put_file_new("k", small, "h") is False
        assert store.put_file_new("m", big, "h", part_size=10) is False
        stub.assert_no_pending_responses()


@pytest.mark.services
def test_s3_concurrent_different_contents_have_one_winner(tmp_path: Path) -> None:
    """SeaweedFS(dlp-raw)에서도 조건부 쓰기(If-None-Match: *)로 동시 쓰기의 승자가 하나다.

    F-PMS-40.

    작은 파일(단일 PUT)과 분할 업로드(완료 요청에 조건) 둘 다 본다. 쓴 키는 지운다.
    """
    store = S3Store.from_env("dlp-raw")
    keys = [f"sessions/test-race-{uuid.uuid4().hex[:8]}/raw/{n}.bin" for n in ("one", "multi")]
    try:
        got = _race(store, keys[0], tmp_path)
        winners = [n for n, r in got.items() if r is True]
        assert len(winners) == 1, got
        head = store.head(keys[0])
        assert head is not None and head.sha256 == sha256_file(tmp_path / f"{winners[0]}.bin")

        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (6 << 20))
        other = tmp_path / "other.bin"
        other.write_bytes(b"y" * (6 << 20))
        assert store.put_file_new(keys[1], big, sha256_file(big), part_size=5 << 20) is True
        assert store.put_file_new(keys[1], other, sha256_file(other), part_size=5 << 20) is False
        assert put_immutable(store, keys[1], big) is False
        with pytest.raises(ImmutableObjectError):
            put_immutable(store, keys[1], other)
        head = store.head(keys[1])
        assert head is not None and head.sha256 == sha256_file(big) and head.size == 6 << 20
    finally:
        for key in keys:
            store.delete(key)


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
