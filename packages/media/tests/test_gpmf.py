"""GoPro GPMF 해석 단위 테스트 (dlp_media.imu, WP3).

실제 GoPro 파일 대신 이 파일의 도우미로 GPMF KLV 바이트를 직접 만든다 (값을 아는 합성 데이터).
가속도는 SCAL 100, 자이로는 SCAL 1000으로 저장하므로 정수 981 → 9.81 m/s², 1000 → 1.0 rad/s다.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from dlp_media.imu import imu_from_gpmf_payloads, klv_values, parse_klv, sensor_samples


def klv(key: str, typ: str, size: int, repeat: int, data: bytes) -> bytes:
    """GPMF KLV 하나 (4바이트 경계로 채움)."""
    header = key.encode() + typ.encode("latin-1") + bytes([size]) + struct.pack(">H", repeat)
    return header + data + b"\x00" * (-len(data) % 4)


def nested(key: str, *children: bytes) -> bytes:
    """자식 KLV들을 담은 중첩 KLV (타입 0, 크기 1, 반복 = 본문 바이트 수)."""
    body = b"".join(children)
    return klv(key, "\x00", 1, len(body), body)


def sensor(fourcc: str, samples: np.ndarray, scal: int, orin: str | None) -> bytes:
    """samples: (N, 3) 파일 저장 순서의 정수 값.

    STRM 하나: STNM(이름), SCAL(배율), ORIN(축 순서, None이면 생략), 센서 데이터("s" 16비트 x 3축).
    """
    parts = [
        klv("STNM", "c", 4, 1, b"test"),
        klv("SCAL", "s", 2, 1, struct.pack(">h", scal)),
    ]
    if orin is not None:
        parts.append(klv("ORIN", "c", 3, 1, orin.encode()))
    data = struct.pack(f">{samples.size}h", *samples.astype(int).ravel())
    parts.append(klv(fourcc, "s", 6, samples.shape[0], data))
    return nested("STRM", *parts)


def payload(acc: np.ndarray, gyro: np.ndarray, orin: str | None = "ZXY") -> bytes:
    """DEVC 하나에 ACCL(SCAL 100)·GYRO(SCAL 1000) 스트림을 담은 GPMF 페이로드."""
    return nested(
        "DEVC",
        klv("DVID", "L", 4, 1, struct.pack(">I", 1)),
        sensor("ACCL", acc, 100, orin),
        sensor("GYRO", gyro, 1000, orin),
    )


def test_parse_nested_klv_with_padding() -> None:
    """중첩 KLV와 4바이트 패딩을 해석해 DEVC → DVID·STRM·STRM 구조를 얻는다."""
    [devc] = parse_klv(payload(np.ones((3, 3)), np.ones((3, 3))))
    assert devc.key == "DEVC"
    assert [c.key for c in devc.children] == ["DVID", "STRM", "STRM"]
    strm = devc.children[1]
    assert strm.find("STNM") is not None and strm.find("ORIN") is not None


def test_scal_and_orin_reorder_to_xyz() -> None:
    """SCAL로 나누고 ORIN 순서를 XYZ로 바꾼다.

    파일 열 순서 ZXY → 출력 XYZ, 소문자 z는 부호 반전, ORIN이 없으면 기본 ZXY와 같다.
    """
    zxy = np.array([[981, 10, 20], [982, 11, 21]])
    acc = sensor_samples(payload(zxy, zxy), "ACCL")
    # 파일 순서 Z, X, Y → X, Y, Z, SCAL 100으로 나눔
    assert np.allclose(acc, [[0.10, 0.20, 9.81], [0.11, 0.21, 9.82]])
    flipped = sensor_samples(payload(zxy, zxy, orin="zXY"), "ACCL")
    assert np.allclose(flipped[:, 2], [-9.81, -9.82])
    default = sensor_samples(payload(zxy, zxy, orin=None), "ACCL")
    assert np.allclose(default, acc)


def test_fixed_point_and_truncation() -> None:
    """고정소수점 q(Q15.16)를 실수로 바꾸고, 잘린 페이로드는 ValueError."""
    [q] = parse_klv(klv("TEST", "q", 4, 2, struct.pack(">2i", 3 << 16, 1 << 15)))
    assert klv_values(q).ravel().tolist() == [3.0, 0.5]
    with pytest.raises(ValueError, match="잘렸"):
        parse_klv(klv("ACCL", "s", 6, 10, b"\x00" * 60)[:30])


def test_payload_samples_are_spread_over_packet_duration() -> None:
    """패킷 안 샘플을 패킷 길이에 균등 배치하고 자이로를 가속도 시각으로 보간한다.

    1초 패킷 두 개에 가속도 4개씩 → 250 ms 간격(4 Hz). 자이로는 2개씩(500 ms 간격)이라
    가속도 시각으로 선형 보간한 값 [1, 1, 1, 2, 3, 3, 3, 3]이 정답이다 (끝은 양 끝 값 유지).
    """
    acc1 = np.tile([[981, 0, 0]], (4, 1))
    acc2 = np.tile([[981, 100, 0]], (4, 1))
    gyro1 = np.tile([[0, 1000, 0]], (2, 1))  # 자이로는 절반 속도
    gyro2 = np.tile([[0, 3000, 0]], (2, 1))
    imu = imu_from_gpmf_payloads(
        [(1_000.0, 1_000.0, payload(acc1, gyro1)), (2_000.0, 1_000.0, payload(acc2, gyro2))]
    )
    assert imu is not None
    assert imu.t_ms.tolist() == [1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750]
    assert np.allclose(imu.acc[:4, 0], 0) and np.allclose(imu.acc[4:, 0], 1.0)
    assert imu.sample_rate_hz == pytest.approx(4.0)
    # 자이로 시각 1000, 1500, 2000, 2500에 X값 1, 1, 3, 3 → 가속도 시각으로 보간
    assert np.allclose(imu.gyro[:, 0], [1, 1, 1, 2, 3, 3, 3, 3])


def test_packets_without_duration_use_next_packet_start() -> None:
    """회귀: packet.duration이 없으면(0) 샘플이 패킷 시작에 몰려 ImuData 검증이 실패했다.

    정답: 1초 간격 패킷 세 개(가속도 4개씩), 길이 모두 0 → 앞 두 패킷은 다음 시작까지 1000 ms,
    마지막은 중앙값 1000 ms → 250 ms 간격 12개. 패킷 하나뿐이고 길이가 없으면 None (수집은 계속).
    """
    acc = np.tile([[981, 0, 0]], (4, 1))
    gyro = np.tile([[0, 1000, 0]], (2, 1))
    imu = imu_from_gpmf_payloads([(t, 0.0, payload(acc, gyro)) for t in (0.0, 1_000.0, 2_000.0)])
    assert imu is not None
    assert imu.t_ms.tolist() == [250.0 * i for i in range(12)]
    assert imu.sample_rate_hz == pytest.approx(4.0)
    assert imu_from_gpmf_payloads([(0.0, 0.0, payload(acc, gyro))]) is None


def accel_only(acc: np.ndarray) -> bytes:
    """가속도 스트림만 있는 GPMF 페이로드."""
    return nested(
        "DEVC", klv("DVID", "L", 4, 1, struct.pack(">I", 1)), sensor("ACCL", acc, 100, "ZXY")
    )


def gyro_only(gyro: np.ndarray) -> bytes:
    """자이로 스트림만 있는 GPMF 페이로드."""
    return nested(
        "DEVC", klv("DVID", "L", 4, 1, struct.pack(">I", 1)), sensor("GYRO", gyro, 1000, "ZXY")
    )


def test_gpmf_without_accelerometer_gives_no_imu() -> None:
    """ACCL이 없으면 샘플레이트 0인 스트림을 만들지 않고 None을 돌려준다."""
    gyro = np.tile([[0, 1000, 0]], (4, 1))
    assert imu_from_gpmf_payloads([(0.0, 1_000.0, gyro_only(gyro))]) is None
    assert imu_from_gpmf_payloads([]) is None
    one = np.array([[981, 0, 0]])
    assert imu_from_gpmf_payloads([(0.0, 1_000.0, accel_only(one))]) is None


def test_gpmf_without_gyro_keeps_accelerometer() -> None:
    """자이로가 없으면 가속도는 그대로 두고 자이로 열을 NaN으로 채운다 (0이면 정지로 오해)."""
    acc = np.tile([[981, 0, 0]], (4, 1))
    imu = imu_from_gpmf_payloads([(0.0, 1_000.0, accel_only(acc))])
    assert imu is not None
    assert imu.t_ms.tolist() == [0, 250, 500, 750]
    assert np.allclose(imu.acc[:, 2], 9.81)
    assert np.isnan(imu.gyro).all()
    assert imu.sample_rate_hz == pytest.approx(4.0)
