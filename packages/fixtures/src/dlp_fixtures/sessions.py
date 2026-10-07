"""작업자·장소·도메인 메타데이터를 가진 가짜 세션 (WP2, WP7 분할 시험용).

작업자는 소수의 장소에서 반복 촬영하는 현실적인 구조를 흉내 낸다. 작업자·장소 단위 분할(WP7)
테스트에서 교집합이 생기도록 일부 장소는 여러 작업자가 공유한다.

세션 메타데이터만 만든다. 스트림 URI(`s3://dlp-raw/...`)는 형식만 맞춘 가짜이고 실제 파일은 없다.
데이터셋 분할·골든셋·액티브 러닝·내보내기 테스트가 세션 목록이 필요할 때 쓴다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from dlp_schema.session import Domain, Session, Stream, StreamKind, SyncMethod

# 가짜 세션 녹화 시각의 기준 (UTC)
EPOCH = datetime(2026, 11, 1, 9, 0, tzinfo=UTC)


def generate_sessions(
    n: int,
    seed: int = 0,
    *,
    n_workers: int = 24,
    n_sites: int = 16,
    domain_weights: tuple[float, float, float] = (0.6, 0.3, 0.1),
    ontology_version: str = "1.0.0",
) -> list[Session]:
    """가짜 세션 `n`개를 만든다.

    Args:
        n: 세션 수. 세션 ID는 `syn-s0000`부터 순서대로.
        seed: 난수 seed (같으면 같은 결과).
        n_workers: 작업자 수 (ID `syn-w000`…). 작업자마다 도메인 하나와 주 장소 1~3곳을 정한다.
        n_sites: 장소 수 (ID `syn-site000`…). 장소는 작업자 사이에 겹칠 수 있다.
        domain_weights: 작업자 도메인 비율 (청소, 돌봄, 간호 순, `Domain` 정의 순서). 합이 1이
            아니어도 된다 (정규화한다). 원소는 정확히 3개여야 한다.
        ontology_version: 세션에 적을 온톨로지 버전.

    Returns:
        세션 목록. 스트림 구성: 바디캠(reference)은 항상, 내장 IMU(shared_clock, 200 Hz) 70%,
        3인칭 영상 30%, 좌우 장갑(100 Hz) 한 쌍 25%. 길이 5~39분(분 단위 정수).
        녹화 시각은 `EPOCH`부터 0~119일 뒤 (+ 세션 번호만큼 분).
    """
    rng = np.random.default_rng(seed)
    domains = list(Domain)
    weights = np.asarray(domain_weights, dtype=float) / sum(domain_weights)
    # 작업자마다 도메인 하나. `rng.choice(3, ...)`는 Domain 3개를 전제한다
    worker_domain = [domains[int(rng.choice(3, p=weights))] for _ in range(n_workers)]
    # 작업자마다 주로 가는 장소 1~3곳. 장소는 작업자 사이에 겹칠 수 있다.
    worker_sites: list[list[int]] = [
        [int(x) for x in rng.choice(n_sites, size=int(rng.integers(1, 4)), replace=False)]
        for _ in range(n_workers)
    ]

    sessions: list[Session] = []
    for i in range(n):
        # 세션마다 작업자를 고르고, 그 작업자의 주 장소 중 하나에서 찍는다
        w = int(rng.integers(n_workers))
        site = int(rng.choice(worker_sites[w]))
        sid = f"syn-s{i:04d}"
        duration_ms = int(rng.integers(5, 40)) * 60_000
        streams = [
            Stream(
                stream_id="bodycam",
                kind=StreamKind.BODYCAM,
                uri=f"s3://dlp-raw/{sid}/bodycam.mp4",
                sync_method=SyncMethod.REFERENCE,
            )
        ]
        if rng.random() < 0.7:
            streams.append(_stream(sid, "imu", StreamKind.IMU, "parquet", 200.0, shared=True))
        if rng.random() < 0.3:
            streams.append(_stream(sid, "third_person", StreamKind.THIRD_PERSON, "mp4", None))
        if rng.random() < 0.25:
            streams.append(_stream(sid, "glove_left", StreamKind.GLOVE_LEFT, "parquet", 100.0))
            streams.append(_stream(sid, "glove_right", StreamKind.GLOVE_RIGHT, "parquet", 100.0))
        sessions.append(
            Session(
                session_id=sid,
                domain=worker_domain[w],
                worker_id=f"syn-w{w:03d}",
                site_id=f"syn-site{site:03d}",
                consent_version="c1",
                recorded_at=EPOCH + timedelta(days=int(rng.integers(0, 120)), minutes=i),
                duration_ms=duration_ms,
                streams=tuple(streams),
                ontology_version=ontology_version,
            )
        )
    return sessions


def _stream(
    sid: str, name: str, kind: StreamKind, ext: str, rate: float | None, *, shared: bool = False
) -> Stream:
    """가짜 스트림 계약 하나. URI = `s3://dlp-raw/<세션>/<name>.<ext>` (실제 파일 없음).

    `shared=True`면 바디캠과 같은 시계(shared_clock), 아니면 unsynced로 시작한다.
    """
    return Stream(
        stream_id=name,
        kind=kind,
        uri=f"s3://dlp-raw/{sid}/{name}.{ext}",
        sample_rate_hz=rate,
        sync_method=SyncMethod.SHARED_CLOCK if shared else SyncMethod.UNSYNCED,
    )
