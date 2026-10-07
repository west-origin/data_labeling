"""동기화 테스트 공용 픽스처 (WP4).

정답을 아는 합성 시나리오(`dlp_fixtures.sync.generate_sync_scenario`)를 30 fps 영상·WAV·Parquet으로
쓰고, 그 파일을 가리키는 세션과 동기화 입력(`StreamMedia`)을 만든다. 시나리오의 `clocks`(스트림별
정답 오프셋·배율)가 판정 기준이다.

- `policy`: 저장소의 `config/policies/sync.yaml`
- `session_for`: 시나리오 → 세션 (bodycam=reference, imu=shared_clock, 3인칭·장갑=unsynced)
- `build`: (seed, 길이, 슬레이트 여부, 두드림 들림 여부)별로 한 번만 만드는 캐시 팩토리
"""

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

# 저장소 루트 (packages/sync/tests/conftest.py에서 세 단계 위)
ROOT = Path(__file__).resolve().parents[3]
# 테스트 영상 프레임레이트. 완료 기준(1·2프레임)의 프레임 시간이 여기서 나온다
FPS = 30.0
FRAME_MS = 1000 / FPS


@pytest.fixture(scope="session")
def policy() -> SyncPolicy:
    """저장소의 동기화 정책 (세션 범위에서 한 번 읽는다)."""
    return load_policy(ROOT / "config" / "policies" / "sync.yaml")


def session_for(scenario: SyncScenario, d: Path) -> Session:
    """시나리오 파일이 든 디렉터리 `d`를 가리키는 세션을 만든다.

    스트림 URI는 로컬 경로 문자열이라 `load_session_media(session, Path, policy)`처럼 `fetch=Path`로
    읽는다. 바디캠은 reference, 내장 IMU는 shared_clock, 3인칭·오른손 장갑은 unsynced로 시작한다.
    """

    def stream(sid: str, kind: StreamKind, name: str, method: SyncMethod) -> Stream:
        """스트림 계약 하나 (URI = `d / name`)."""
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


# (정답 시나리오, 세션, stream_id → 동기화 입력)
Built = tuple[SyncScenario, Session, dict[str, StreamMedia]]


@pytest.fixture(scope="session")
def build(tmp_path_factory: pytest.TempPathFactory, policy: SyncPolicy) -> Callable[..., Built]:
    """시나리오를 30 fps 영상으로 쓰고 세션과 동기화 입력을 만든다 (같은 인자는 한 번만)."""
    cache: dict[tuple[object, ...], Built] = {}

    def make(
        seed: int = 4,
        duration_ms: float = 30_000.0,
        with_slates: bool = True,
        audible_taps: bool = True,
    ) -> Built:
        """인자 조합별 시나리오·세션·입력을 만들거나 캐시에서 꺼낸다."""
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
            cache[key] = (scenario, session, load_session_media(session, Path, policy))
        return cache[key]

    return make
