"""프라이버시 게이트 통합 테스트: 수집 → 탐지 → 검수 → 승인 → 렌더 → 렌더 확인 (WP5).

실행 중인 서비스(PostgreSQL, SeaweedFS S3)가 필요하다 (`@pytest.mark.services`, `make up` 뒤
`make test-services`). 테스트마다 임시 DB를 만들고 지운다.

정답 근거: 합성 블러 시나리오의 대상 6개(얼굴·반사·문서·화면·사진·송장)를 그대로 내는
`OracleDetector`를 쓰므로 바디캠 스트림의 트랙 수는 6이다. 검수 도구(CVAT) 대신
`reviewed_task`로 수집된 검수 작업 기록만 만들고, `record_review`로 라벨 검증 상태를 바꾼다.
관련 회귀: ADR 0024 감사 4-2(승인 취소 뒤 이전 블러본 사용), 4-8(재학습 모델 구간 병합).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import av
import pytest
import sqlalchemy as sa
import yaml

from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import BlurScenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.storage import S3Store
from dlp_privacy.detection import Detection, Image
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.policy import PrivacyPolicy, TargetPolicy
from dlp_privacy.runner import (
    PrivacyGateError,
    RenderNotCurrentError,
    approve_session,
    assert_render_current,
    detect_marker_key,
    detect_session,
    render_session,
)
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    get_labels,
    get_session,
    insert_labels,
    insert_review_task,
    record_review,
    register_ontology,
)
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord, Provenance, Source, VerificationState
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.review import ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.testing import FIXED_TIME

pytestmark = pytest.mark.services
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pg() -> Iterator[sa.Engine]:
    """빈 임시 PostgreSQL DB (마이그레이션 + 온톨로지 v1 등록). 끝나면 DB를 지운다.

    DLP_DATABASE_URL(없으면 개발 기본값)의 서버에 `dlp_test_<임의>` DB를 만든다.
    """
    url = sa.make_url(
        os.environ.get(
            "DLP_DATABASE_URL", "postgresql+psycopg://dlp:dlp-dev-password@localhost:5432/dlp"
        )
    )
    name = f"dlp_test_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
    test_url = url.set(database=name).render_as_string(hide_password=False)
    upgrade(test_url)
    engine = sa.create_engine(test_url)
    with engine.begin() as conn:
        register_ontology(conn, load_ontology(ROOT / "config" / "ontology" / "v1"))
    try:
        yield engine
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_detect_review_approve_render(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """게이트 전체 흐름과 승인 취소·재승인.

    시나리오:
    1. 바디캠(블러 픽스처)과 3인칭(오디오가 있는 동기화 픽스처 영상)으로 세션을 수집한다.
    2. 탐지 전 승인은 실패, 탐지 뒤 AUTO_BLURRED, 같은 버전 재탐지는 건너뜀.
    3. 승인 전 렌더 실패 → 검수 작업 없으면 승인 실패 → 미검수 라벨 있으면 실패 → 모두 검수 후 승인.
    4. 렌더는 멱등이고, 블러본은 라벨링 버킷에만 있으며 오디오가 없다. 같은 버킷 렌더는 거부.
    5. 탐지기 버전이 바뀌어 블러가 바뀌면 승인이 풀리고 이전 블러본은 어떤 단계도 쓰지 못한다.
    6. 다시 검수·승인해도 다시 렌더하기 전에는 무효, 렌더 뒤에는 현재 것. 렌더 기록을 거치지 않고
       블러 라벨이 늘면 해시가 달라 막힌다. 새 블러본 파일 해시는 이전과 다르다.
    """
    scenario, video = blur
    # 3인칭 영상 자리에는 오디오가 있는 다른 영상을 둔다 (블러본에서 오디오가 빠지는지 본다)
    sync = generate_sync_scenario(2, recorded_at=FIXED_TIME, duration_ms=3_000)
    sync.write_video(tmp_path / "third.mp4", "bodycam")
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [
            {"stream_id": "bodycam", "kind": "bodycam", "path": str(video)},
            {"stream_id": "third_person", "kind": "third_person", "path": "third.mp4"},
        ],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw, labeling = S3Store.from_env("dlp-raw"), S3Store.from_env("dlp-labeling")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)

    oracle = OracleDetector("oracle", scenario.labels)
    oracle_policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    with pg.begin() as conn:
        with pytest.raises(PrivacyGateError, match="자동 탐지"):
            approve_session(conn, sid)
        summary = detect_session(conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME)
    assert summary.detected["bodycam"] == 6 and set(summary.detected) == {"bodycam", "third_person"}
    with pg.connect() as conn:
        assert get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED

    # 같은 모델 버전으로 다시 실행하면 건너뛴다
    with pg.begin() as conn:
        again = detect_session(conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME)
    assert sorted(again.skipped) == ["bodycam", "third_person"]

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        raw.get_file(f"sessions/{sid}/derived/privacy_review/bodycam.json", dest)
        segments = json.loads(dest.read_text(encoding="utf-8"))
    assert any(s["reason"] == "reflection" for s in segments)

    with pg.begin() as conn:
        with pytest.raises(PrivacyGateError, match="승인 전"):
            render_session(conn, sid, raw, labeling, oracle_policy)
        with pytest.raises(PrivacyGateError, match="블러 검수 작업"):
            approve_session(conn, sid)
        for stream in ("bodycam", "third_person"):
            reviewed_task(conn, sid, stream, FIXED_TIME)
        with pytest.raises(PrivacyGateError, match="사람 검수"):
            approve_session(conn, sid)
        for label in get_labels(conn, sid, kinds=["blur_track"]):
            record_review(
                conn, label.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME
            )
        approved = approve_session(conn, sid)
    assert approved.privacy_state is PrivacyState.APPROVED
    assert approved.lifecycle_state is LifecycleState.PRIVACY_APPROVED

    with pg.begin() as conn:
        uris = render_session(conn, sid, raw, labeling, oracle_policy)
        assert render_session(conn, sid, raw, labeling, oracle_policy) == uris  # 멱등
        rendered = assert_render_current(conn, labeling, sid, "bodycam", oracle_policy)
        head = labeling.head(f"sessions/{sid}/blurred/bodycam.mp4")
        assert head is not None and rendered == head.sha256
        with pytest.raises(PrivacyGateError, match="라벨링 버킷"):
            render_session(conn, sid, raw, raw, oracle_policy)
    assert uris["bodycam"] == f"s3://dlp-labeling/sessions/{sid}/blurred/bodycam.mp4"
    assert all("dlp-raw" not in u for u in uris.values())
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "b.mp4"
        labeling.get_file(f"sessions/{sid}/blurred/third_person.mp4", dest)
        with av.open(str(dest)) as c:
            assert c.streams.video and not c.streams.audio

    # 회귀: 승인 뒤 다시 탐지해 블러가 바뀌면 승인이 풀리고, 다시 검수·승인하면 블러본을 새로 만든다
    later = FIXED_TIME + timedelta(hours=1)
    before = labeling.head(f"sessions/{sid}/blurred/bodycam.mp4")
    face_only = OracleDetector("oracle", [x for x in scenario.labels if x.kind == "blur_track"][:2])
    face_only.version = "oracle-2"  # 탐지기 버전이 바뀌었다
    with pg.begin() as conn:
        changed = detect_session(
            conn, sid, raw, {"oracle": face_only}, {}, oracle_policy, later, labeling=labeling
        )
        assert changed.changed
        assert get_session(conn, sid).privacy_state is PrivacyState.AUTO_BLURRED
        # 회귀(감사 4-2): 승인이 풀리면 이전 블러본을 어떤 단계도 쓰지 못한다
        with pytest.raises(RenderNotCurrentError, match="승인 상태"):
            assert_render_current(conn, labeling, sid, "bodycam", oracle_policy)
        with pytest.raises(PrivacyGateError, match="승인 전"):
            render_session(conn, sid, raw, labeling, oracle_policy)
        with pytest.raises(PrivacyGateError, match="블러 검수 작업"):
            approve_session(conn, sid)  # 이전 검수 작업은 새 탐지보다 앞이다
        for stream in ("bodycam", "third_person"):
            reviewed_task(conn, sid, stream, later + timedelta(minutes=5), "2")
        for label in current_labels(get_labels(conn, sid, kinds=["blur_track"])):
            record_review(conn, label.label_id, VerificationState.HUMAN_APPROVED, "rev01", later)
        again_approved = approve_session(conn, sid)
        assert again_approved.lifecycle_state is LifecycleState.PRIVACY_APPROVED
        # 다시 승인했어도 다시 렌더하기 전에는 이전 블러본이 현재 것이 아니다 (기록 무효·해시 다름)
        for stream in ("bodycam", "third_person"):
            with pytest.raises(RenderNotCurrentError, match="무효"):
                assert_render_current(conn, labeling, sid, stream, oracle_policy)
        render_session(conn, sid, raw, labeling, oracle_policy)
        assert_render_current(conn, labeling, sid, "bodycam", oracle_policy)
        # 렌더 기록을 무효로 두지 못한 경로가 있어도 블러 라벨 집합이 다르면 해시로 막힌다
        extra = current_labels(get_labels(conn, sid, kinds=["blur_track"]))[0].model_copy(
            update={
                "label_id": f"{sid}-human-extra", "stream_id": "third_person",
                "provenance": Provenance(source=Source.HUMAN), "confidence": None,
            }
        )  # fmt: skip
        insert_labels(conn, [extra])
        with pytest.raises(RenderNotCurrentError, match="승인된 블러 라벨"):
            assert_render_current(conn, labeling, sid, "third_person", oracle_policy)
    after = labeling.head(f"sessions/{sid}/blurred/bodycam.mp4")
    assert before is not None and after is not None and before.sha256 != after.sha256


def reviewed_task(conn: sa.Connection, sid: str, stream: str, at: datetime, tag: str = "1") -> None:
    """수집까지 끝난 블러 검수 작업 기록 (검수 도구 없이 승인 조건만 맞춘다)."""
    insert_review_task(
        conn,
        ReviewTask(
            task_key=f"cvat:{sid}-{stream}-{tag}", tool=ReviewTool.CVAT, external_id="0",
            session_id=sid, stream_id=stream, stage=ReviewStage.PRIVACY, assignee="rev01",
            media_uri=f"s3://dlp-raw/sessions/{sid}/derived/{stream}.proxy.mp4",
            label_kinds=("blur_track",), created_at=at, status=ReviewTaskStatus.COLLECTED,
            collected_at=at,
        ),
    )  # fmt: skip


def test_approval_needs_a_review_task_even_without_detections(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """회귀: 탐지가 하나도 없으면 미검수 라벨이 없어 사람 검수 없이 승인됐다.

    정답: 탐지 0개여도 수집한 블러 검수 작업이 생기기 전에는 승인이 실패한다.
    """
    _, video = blur
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": str(video)}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    nothing = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
        summary = detect_session(
            conn, sid, raw, {"oracle": OracleDetector("oracle", [])}, {}, nothing, FIXED_TIME
        )
        assert summary.detected == {"bodycam": 0}
        with pytest.raises(PrivacyGateError, match="블러 검수 작업"):
            approve_session(conn, sid)
        reviewed_task(conn, sid, "bodycam", FIXED_TIME)
        assert approve_session(conn, sid).privacy_state is PrivacyState.APPROVED


class CountingOracle(OracleDetector):
    """프레임 탐지 호출 수를 세는 오라클 (영상을 다시 탐지했는지 확인용)."""

    calls = 0

    def detect(self, image: Image, t_ms: int, threshold: float) -> list[Detection]:
        """호출 수를 늘리고 오라클 결과를 그대로 낸다."""
        self.calls += 1
        return super().detect(image, t_ms, threshold)


def test_zero_detection_stream_is_not_redetected(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """회귀: 탐지 0개인 스트림은 DB 이력에 모델 버전이 남지 않아 실행마다 영상 전체를 다시 탐지했다.

    정답: 첫 실행은 탐지기를 부르고 원본 버킷에 탐지 표시(결과 0개 버전)를 남긴다. 같은 버전으로
    다시 돌리면 스트림을 건너뛰고 탐지기를 한 번도 부르지 않는다. 탐지기 버전이 바뀌면 다시 돈다.
    """
    _, video = blur
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": str(video)}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    nothing = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    first = CountingOracle("oracle", [])
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
        summary = detect_session(conn, sid, raw, {"oracle": first}, {}, nothing, FIXED_TIME)
    assert summary.detected == {"bodycam": 0} and first.calls > 0
    assert raw.head(detect_marker_key(sid, "bodycam")) is not None

    second = CountingOracle("oracle", [])
    with pg.begin() as conn:
        again = detect_session(conn, sid, raw, {"oracle": second}, {}, nothing, FIXED_TIME)
    assert again.skipped == ["bodycam"] and again.detected == {} and second.calls == 0

    # 탐지기 버전이 바뀌면(모델 버전 해시가 달라진다) 다시 탐지한다
    bumped = CountingOracle("oracle", [])
    bumped.version = "oracle-2"
    with pg.begin() as conn:
        rerun = detect_session(conn, sid, raw, {"oracle": bumped}, {}, nothing, FIXED_TIME)
    assert rerun.skipped == [] and bumped.calls > 0


class TrainedBlur:
    """배포된 재학습 블러 모델 자리 (정답 블러를 그대로 낸다)."""

    name = "trained-privacy"

    def __init__(self, labels: list[LabelRecord], version: str) -> None:
        """labels: 낼 블러 라벨(정답), version: 모델 버전 (provenance.model_version)."""
        self.labels, self.version = labels, version

    def run(self, clip: Clip) -> list[LabelRecord]:
        """클립의 세션·스트림으로 ID·출처를 바꾼 정답 블러 라벨 (모델 출처, 신뢰도 0.9)."""
        tag = self.version.replace(".", "_")
        return [
            x.model_copy(
                update={
                    "label_id": f"{clip.session_id}-{clip.stream_id}-{self.name}-{tag}-{i}",
                    "session_id": clip.session_id,
                    "stream_id": clip.stream_id,
                    "provenance": Provenance(source=Source.MODEL, model_version=self.version),
                    "confidence": 0.9,
                }
            )
            for i, x in enumerate(self.labels)
        ]


def test_deployed_blur_model_is_unioned_and_versioned(
    pg: sa.Engine, policy: PrivacyPolicy, blur: tuple[BlurScenario, Path], tmp_path: Path
) -> None:
    """배포된 재학습 블러 모델은 탐지기 결과와 합집합으로 들어가고 버전별로 바뀐다.

    정답: 첫 실행 트랙 수 = 탐지기 6 + 모델 블러 수, trained_model 검수 구간이 생긴다. 같은 모델이면
    건너뛰고, 모델이 v1→v2로 바뀌면 v1 블러만 지워지고(검수 전) v2가 들어오며 탐지기 블러 6개는
    그대로다. 검수 우선 구간에는 탐지기 구간이 남고 trained_model 구간은 v2만 남는다 (감사 4-8).
    """
    scenario, video = blur
    sid = f"priv-{uuid.uuid4().hex[:8]}"
    manifest = {
        "session_id": sid, "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": str(video)}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = S3Store.from_env("dlp-raw")
    with pg.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    oracle = OracleDetector("oracle", scenario.labels)
    oracle_policy = policy.model_copy(
        update={
            "targets": {
                t: TargetPolicy(margin=tp.margin, detectors=("oracle",))
                for t, tp in policy.targets.items()
            }
        }
    )
    blurs = [x for x in scenario.labels if x.kind == "blur_track"]
    v1, v2 = TrainedBlur(blurs, "trained-v1"), TrainedBlur(blurs, "trained-v2")

    def current_versions(conn: sa.Connection) -> dict[str, int]:
        """현재 블러 라벨의 모델 버전별 개수."""
        out: dict[str, int] = {}
        for x in current_labels(get_labels(conn, sid, kinds=["blur_track"])):
            v = x.provenance.model_version or ""
            out[v] = out.get(v, 0) + 1
        return out

    with pg.begin() as conn:
        first = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v1]
        )
        before = current_versions(conn)
    assert first.detected["bodycam"] == 6 + len(blurs)  # 탐지기 + 재학습 모델 (합집합)
    assert before["trained-v1"] == len(blurs)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        raw.get_file(f"sessions/{sid}/derived/privacy_review/bodycam.json", dest)
        reasons = {s["reason"] for s in json.loads(dest.read_text(encoding="utf-8"))}
    assert "trained_model" in reasons

    with pg.begin() as conn:
        again = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v1]
        )
        assert again.skipped == ["bodycam"]
        # 모델이 바뀌면 이전 모델의 검수 전 블러만 바뀌고 탐지기 결과는 그대로다
        swapped = detect_session(
            conn, sid, raw, {"oracle": oracle}, {}, oracle_policy, FIXED_TIME, extra=[v2]
        )
        after = current_versions(conn)
    assert swapped.detected["bodycam"] == len(blurs)
    # 회귀(감사 4-8): 재학습 모델만 다시 돌아도 탐지기 검수 우선 구간은 남고 이전 모델 구간은 빠진다
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "r.json"
        raw.get_file(f"sessions/{sid}/derived/privacy_review/bodycam.json", dest)
        merged = json.loads(dest.read_text(encoding="utf-8"))
    assert {s["reason"] for s in merged} >= reasons
    assert {s["detail"] for s in merged if s["reason"] == "trained_model"} == {"trained-v2"}
    assert "trained-v1" not in after and after["trained-v2"] == len(blurs)
    assert sum(n for v, n in after.items() if not v.startswith("trained")) == 6
