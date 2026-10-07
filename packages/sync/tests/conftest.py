from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from dlp_fixtures.sync import SyncScenario, generate_sync_scenario
from dlp_schema.session import Domain, Session, Stream, StreamKind, SyncMethod
from dlp_schema.testing import FIXED_TIME
from dlp_sync.loader import load_session_media
from dlp_sync.pipeline import StreamMedia
from dlp_sync.policy import SyncPolicy, load_policy

ROOT = Path(__file__).resolve().parents[3]
FPS = 30.0
FRAME_MS = 1000 / FPS


@pytest.fixture(scope="session")
def policy() -> SyncPolicy:
    return load_policy(ROOT / "config" / "policies" / "sync.yaml")


def session_for(scenario: SyncScenario, d: Path) -> Session:
    def stream(sid: str, kind: StreamKind, name: str, method: SyncMethod) -> Stream:
        return Stream(stream_id=sid, kind=kind, uri=str(d / name), sync_method=method)

    return Session(
        session_id=scenario.session_id,
        domain=Domain.CLEANING,
        worker_id="w01",
        site_id="site01",
        consent_version="c1",
        recorded_at=FIXED_TIME,
        duration_ms=round(scenario.duration_ms),
        streams=(
            stream("bodycam", StreamKind.BODYCAM, "bodycam.mp4", SyncMethod.REFERENCE),
            stream("imu", StreamKind.IMU, "imu.parquet", SyncMethod.SHARED_CLOCK),
            stream(
                "third_person", StreamKind.THIRD_PERSON, "third_person.mp4", SyncMethod.UNSYNCED
            ),
            stream(
                "glove_right", StreamKind.GLOVE_RIGHT, "glove_right.parquet", SyncMethod.UNSYNCED
            ),
        ),
    )


Built = tuple[SyncScenario, Session, dict[str, StreamMedia]]


@pytest.fixture(scope="session")
def build(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Built]:
    """시나리오를 30 fps 영상으로 쓰고 세션과 동기화 입력을 만든다 (같은 인자는 한 번만)."""
    cache: dict[tuple[object, ...], Built] = {}

    def make(
        seed: int = 4,
        duration_ms: float = 30_000.0,
        with_slates: bool = True,
        audible_taps: bool = True,
    ) -> Built:
        key = (seed, duration_ms, with_slates, audible_taps)
        if key not in cache:
            scenario = generate_sync_scenario(
                seed,
                recorded_at=FIXED_TIME,
                duration_ms=duration_ms,
                with_slates=with_slates,
                audible_taps=audible_taps,
            )
            d = tmp_path_factory.mktemp("sync")
            scenario.write(d, videos=False)
            for name in ("bodycam", "third_person"):
                scenario.write_video(d / f"{name}.mp4", name, fps=FPS)
            session = session_for(scenario, d)
            cache[key] = (scenario, session, load_session_media(session, Path))
        return cache[key]

    return make
