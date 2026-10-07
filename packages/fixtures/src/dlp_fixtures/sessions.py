"""작업자·장소·도메인 메타데이터를 가진 가짜 세션.

작업자는 소수의 장소에서 반복 촬영하는 현실적인 구조를 흉내 낸다. 작업자·장소 단위 분할(WP7)
테스트에서 교집합이 생기도록 일부 장소는 여러 작업자가 공유한다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from dlp_schema.session import Domain, Session, Stream, StreamKind, SyncMethod

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
    rng = np.random.default_rng(seed)
    domains = list(Domain)
    weights = np.asarray(domain_weights, dtype=float) / sum(domain_weights)
    worker_domain = [domains[int(rng.choice(3, p=weights))] for _ in range(n_workers)]
    # 작업자마다 주로 가는 장소 1~3곳. 장소는 작업자 사이에 겹칠 수 있다.
    worker_sites: list[list[int]] = [
        [int(x) for x in rng.choice(n_sites, size=int(rng.integers(1, 4)), replace=False)]
        for _ in range(n_workers)
    ]

    sessions: list[Session] = []
    for i in range(n):
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
    return Stream(
        stream_id=name,
        kind=kind,
        uri=f"s3://dlp-raw/{sid}/{name}.{ext}",
        sample_rate_hz=rate,
        sync_method=SyncMethod.SHARED_CLOCK if shared else SyncMethod.UNSYNCED,
    )
