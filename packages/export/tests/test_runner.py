"""데이터셋 버전 → 내보내기 → 이력 (PostgreSQL, make up).

완료 기준: 기본 정책에서 미검수 라벨 0건, 사용 중지 세션 0건. 결과에 원본 위치가 없다.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa

from dlp_datasets.build import build_dataset_version
from dlp_datasets.lineage import session_lineage, withdraw_session
from dlp_datasets.policy import load_policy as load_dataset_policy
from dlp_datasets.snapshot import LocalSnapshotStore
from dlp_export.policy import ExportPolicy
from dlp_export.pseudonym import Pseudonymizer
from dlp_export.runner import id_map_key, run_export
from dlp_export.source import ExportError
from dlp_fixtures.video import vfr_times
from dlp_media.storage import LocalStore, sha256_file
from dlp_schema.db.migrate import upgrade
from dlp_schema.db.repository import (
    insert_labels,
    insert_session,
    register_ontology,
    set_privacy_state,
)
from dlp_schema.labels import VerificationState
from dlp_schema.ontology import Ontology
from dlp_schema.session import PrivacyState
from dlp_schema.testing import FIXED_TIME, make_session

from .conftest import ROOT, record_render, scenario_labels, write_blurred

pytestmark = pytest.mark.services


@pytest.fixture
def engine() -> Iterator[sa.Engine]:
    """테스트마다 새 PostgreSQL 데이터베이스(dlp_test_<임의>)를 만들고 마이그레이션한다.

    끝나면 지운다.

    DLP_DATABASE_URL(없으면 개발 compose 기본값)의 서버를 쓴다 (`make up` 필요).
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
    e = sa.create_engine(test_url)
    try:
        yield e
    finally:
        e.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_export_applies_policy_and_records_history(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """데이터셋 버전 → 구간 JSON·COCO 내보내기가 검증 정책·사용 중지·가명을 지키고 이력을 남긴다.

    시나리오: 작업자·장소가 모두 다른 세션 4개로 버전 dv1을 만든 뒤 s2를 사용 중지한다.
    정답: 결과·이력에 s2가 없고(s0·s1·s3만), 미검수·블러·오류 삽입·측정 라벨 0건, 원본 위치·검수자
    ID·내부 ID 없음, manifest 파일 목록 = 올린 파일과 sha256, 가명 대응표는 내부 경로에만.
    미검수 포함 옵션은 이력의 검증 정책에 남는다. 계보: s0은 내보내기 3건, s2는 0건.
    """
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    sids = [f"s{i}" for i in range(4)]
    times = vfr_times(np.random.default_rng(4), 1000)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(sids):
            insert_session(
                conn,
                make_session(
                    sid,
                    worker_id=f"w{i}",
                    site_id=f"site{i}",
                    streams=[
                        {
                            "stream_id": "bodycam",
                            "kind": "bodycam",
                            "sync_method": "reference",
                            "uri": f"s3://dlp-raw/sessions/{sid}/bodycam.mp4",
                        },
                    ],
                ).model_copy(update={"privacy_state": PrivacyState.APPROVED}),
            )
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip
        withdraw_session(conn, "s2", "동의 철회", FIXED_TIME + timedelta(days=1))

    def export(fmt: str, include_unreviewed: bool = False):
        """dv1을 buyer-a에게 내보낸다.

        고정 시각·고정 비밀값이라 가명을 테스트에서 다시 계산할 수 있다.
        """
        return run_export(
            engine, root=ROOT, version_id="dv1", fmt=fmt, target="buyer-a",  # type: ignore[arg-type]
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=policy, ontology=ontology, include_unreviewed=include_unreviewed,
            splits=None, now=FIXED_TIME + timedelta(days=2), id_secret=b"test-secret",
        )  # fmt: skip

    kept = {"s0", "s1", "s3"}  # s2는 사용 중지
    for fmt in ("intervals", "coco"):
        r = export(fmt)
        out = tmp_path / "store" / "dlp-datasets" / "exports" / r.record.export_id
        manifest = json.loads((out / "manifest.json").read_text())
        exported = {s["session_id"] for s in manifest["sessions"]}
        # 결과 파일의 세션·라벨 ID는 이 내보내기의 가명, 이력에는 내부 세션 ID
        ids = Pseudonymizer.for_export(r.record.export_id, b"test-secret", enabled=True)
        pseudo = {sid: ids.session(sid) for sid in kept}
        assert r.session_pseudonyms == pseudo
        # 사용 중지 세션 0건 (데이터셋 버전에는 있었다)
        assert exported == set(pseudo.values()) and set(r.record.session_ids) == kept
        # 세션 가명 대응표는 내보내기 폴더 밖 내부 경로에만 있다
        id_map = tmp_path / "store" / "dlp-datasets" / id_map_key(r.record.export_id)
        assert json.loads(id_map.read_text()) == pseudo
        # manifest의 파일 목록 = 올린 파일 전부(manifest 제외)와 그 sha256
        # (LocalStore가 옆에 두는 .sha256 파일은 저장소 내부 기록이라 뺀다)
        files = {
            p.relative_to(out).as_posix(): sha256_file(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "manifest.json" and p.suffix != ".sha256"
        }
        assert manifest["files"] == files and files and r.files == len(files) + 1
        # 미검수 라벨 0건
        assert not [k for k in r.label_counts if k.endswith("/unreviewed")]
        assert manifest["verification_policy"]["include_unreviewed"] is False
        assert r.record.label_states == (
            VerificationState.HUMAN_APPROVED,
            VerificationState.HUMAN_CORRECTED,
            VerificationState.SAMPLE_VERIFIED,
        )
        text = "".join(p.read_text(errors="ignore") for p in out.rglob("*.json"))
        assert "dlp-raw" not in text and "reviewer-7" not in text
        # 내부 세션·라벨 ID는 결과 어디에도 없다 (파일 이름 포함)
        assert not any(f'"s{i}' in text or f"s{i}-" in text for i in range(4))
        assert not [p for p in out.rglob("*") if any(f"s{i}" in p.name for i in range(4))]
        if fmt == "coco":
            coco = json.loads((out / "coco" / "annotations.json").read_text())
            assert {a["verification"] for a in coco["annotations"]} <= {
                "human_approved", "human_corrected", "sample_verified"
            }  # fmt: skip
            assert {i["session_id"] for i in coco["images"]} == exported
        else:
            # 작업자·장소 ID는 이 내보내기의 가명 (원래 ID는 결과에 없다)
            assert manifest["pseudonymized_ids"] == [
                "worker_id", "site_id", "session_id", "label_id"
            ]  # fmt: skip
            workers = set[str]()
            for sid in exported:
                f = json.loads((out / "intervals" / f"{sid}.json").read_text())
                assert f["worker_id"].startswith("worker-") and f["site_id"].startswith("site-")
                workers.add(f["worker_id"])
            assert len(workers) == len(exported)
            assert not any(f'"w{i}"' in text or f'"site{i}"' in text for i in range(4))
            for sid in kept:
                f = json.loads((out / "intervals" / f"{pseudo[sid]}.json").read_text())
                assert "unreviewed" not in {x["verification"] for x in f["labels"]}
                assert not {x["label_id"] for x in f["labels"]} & {
                    ids.label(f"{sid}-blind"),
                    ids.label(f"{sid}-seed"),
                    ids.label(f"{sid}-seedfix"),
                }
                assert all(x["payload"]["kind"] != "blur_track" for x in f["labels"])

    # 미검수 포함은 명시적 옵션으로만, 이력에 정책이 남는다
    r = export("coco", include_unreviewed=True)
    assert any(k.endswith("/unreviewed") for k in r.label_counts)
    assert VerificationState.UNREVIEWED in r.record.label_states
    # 계보: 세션 → 내보내기 (사용 중지 세션은 이후 내보내기에 없다)
    with engine.connect() as conn:
        assert len(session_lineage(conn, "s0").exports) == 3
        assert session_lineage(conn, "s2").exports == []


@pytest.mark.isolated_env
def test_lerobot_export_end_to_end(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """LeRobot 내보내기 종단 (격리 환경 사용, `make test-isolated`).

    정답: 세션 a·b가 에피소드 2개로 쓰이고 공식 로더도 2개를 읽는다. meta/dlp_episodes.json의
    세션 ID는 에피소드 순서대로 a·b의 가명이고, 영상(mp4)과 데이터(parquet)가 있다.
    """
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    times = vfr_times(np.random.default_rng(5), 800)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(("a", "b")):
            insert_session(
                conn,
                make_session(sid, worker_id=f"w{i}", site_id=f"x{i}").model_copy(
                    update={"privacy_state": PrivacyState.APPROVED}
                ),
            )
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip
    r = run_export(
        engine, root=ROOT, version_id="dv1", fmt="lerobot", target="internal",
        snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
        policy=policy, ontology=ontology, include_unreviewed=False, splits=None,
        now=FIXED_TIME,
    )  # fmt: skip
    out = tmp_path / "store" / "dlp-datasets" / "exports" / r.record.export_id / "lerobot"
    assert r.details["episodes"] == 2 and r.details["loader_check"]["episodes"] == 2
    assert (out / "meta" / "info.json").exists() and (out / "meta" / "dlp_vocab.json").exists()
    episodes = json.loads((out / "meta" / "dlp_episodes.json").read_text())
    assert [e["session_id"] for e in episodes] == [
        r.session_pseudonyms["a"], r.session_pseudonyms["b"]
    ]  # fmt: skip
    assert all(e["session_id"].startswith("session-") for e in episodes)
    assert list(out.rglob("*.mp4")) and list(out.rglob("data/**/*.parquet"))


def test_lerobot_refuses_mixed_aspect_ratios(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """화면비가 다른 블러본(64x48과 64x36)은 한 LeRobot 데이터셋에 넣지 않는다.

    격리 환경을 부르기 전에 실패하므로 기본 서비스 테스트로 돈다. 정답: ExportError("화면비")이고
    데이터셋 버킷에 아무것도 올라가지 않았다.
    """
    from dlp_fixtures.video import write_video
    from dlp_media.storage import sha256_file

    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    times = vfr_times(np.random.default_rng(6), 500)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, (sid, (w, h)) in enumerate([("a", (64, 48)), ("b", (64, 36))]):
            insert_session(
                conn,
                make_session(sid, worker_id=f"w{i}", site_id=f"x{i}").model_copy(
                    update={"privacy_state": PrivacyState.APPROVED}
                ),
            )
            insert_labels(conn, scenario_labels(sid, times))
            video = tmp_path / f"{sid}.mp4"
            write_video(
                video, ((t, np.zeros((h, w, 3), np.uint8)) for t in times), width=w, height=h
            )
            sha = sha256_file(video)
            labeling.put_file(f"sessions/{sid}/blurred/bodycam.mp4", video, sha)
            record_render(labeling, sid, "bodycam", scenario_labels(sid, times), sha, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip
    with pytest.raises(ExportError, match="화면비"):
        run_export(
            engine, root=ROOT, version_id="dv1", fmt="lerobot", target="internal",
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=policy, ontology=ontology, include_unreviewed=False, splits=None,
            now=FIXED_TIME,
        )  # fmt: skip
    # 아무것도 올리지 않았다
    assert not (tmp_path / "store" / "dlp-datasets").exists()


class _FailingStore(LocalStore):
    """n번째 올리기에서 실패하거나(fail_at), 첫 올리기 직후 세션을 사용 중지한다(withdraw)."""

    def __init__(self, root: Path, bucket: str, *, fail_at: int | None = None,
                 withdraw: tuple[sa.Engine, str] | None = None) -> None:  # fmt: skip
        """fail_at: 이 번째(1부터) put_file에서 OSError.

        withdraw: (엔진, 세션) — 첫 올리기 직후 그 세션을 사용 중지한다.
        """
        super().__init__(root, bucket)
        self.puts, self.fail_at, self.withdraw = 0, fail_at, withdraw

    def put_file(self, key: str, path: Path, sha256: str) -> None:
        """올리기 횟수를 세고, 설정에 따라 실패하거나 다른 트랜잭션에서 세션을 사용 중지한다."""
        self.puts += 1
        if self.fail_at is not None and self.puts == self.fail_at:
            raise OSError("올리기 실패 (시험)")
        super().put_file(key, path, sha256)
        if self.withdraw is not None and self.puts == 1:
            engine, sid = self.withdraw
            with engine.begin() as conn:
                withdraw_session(conn, sid, "동의 철회", FIXED_TIME + timedelta(days=1))


def test_export_history_is_committed_before_upload(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """올리다 실패해도 이력은 남는다. 올리는 사이 사용 중지되면 manifest를 올리지 않고 실패하며,
    사용 중지의 계보 목록에 그 내보내기가 보인다."""
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    times = vfr_times(np.random.default_rng(7), 300)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(("s0", "s1")):
            insert_session(
                conn,
                make_session(sid, worker_id=f"w{i}", site_id=f"site{i}").model_copy(
                    update={"privacy_state": PrivacyState.APPROVED}
                ),
            )
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip

    def export(datasets: LocalStore, minutes: int) -> None:
        """dv1 구간 JSON 내보내기 (분마다 다른 시각 → 다른 내보내기 ID)."""
        run_export(
            engine, root=ROOT, version_id="dv1", fmt="intervals", target="buyer-a",
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=policy, ontology=ontology, include_unreviewed=False, splits=None,
            now=FIXED_TIME + timedelta(minutes=minutes),
        )  # fmt: skip

    # 1) 두 번째 파일을 올리다 실패: 이력은 남고 manifest는 없다
    crashing = _FailingStore(tmp_path / "store", "dlp-datasets", fail_at=2)
    with pytest.raises(OSError, match="올리기 실패"):
        export(crashing, 1)
    with engine.connect() as conn:
        (first,) = session_lineage(conn, "s0").exports
    folder = tmp_path / "store" / "dlp-datasets" / "exports" / first.export_id
    assert folder.exists() and not (folder / "manifest.json").exists()

    # 2) 올리는 사이 s1 사용 중지: 실패, manifest 없음, 계보에 이 내보내기가 보인다
    racing = _FailingStore(tmp_path / "store", "dlp-datasets", withdraw=(engine, "s1"))
    with pytest.raises(ExportError, match="올리는 동안 사용 중지"):
        export(racing, 2)
    with engine.connect() as conn:
        exports = session_lineage(conn, "s1").exports
    assert len(exports) == 2
    late = next(e for e in exports if e.export_id != first.export_id)
    assert not (
        tmp_path / "store" / "dlp-datasets" / "exports" / late.export_id / "manifest.json"
    ).exists()

    # 3) 다시 내보내면 s1은 빠지고 manifest까지 올라간다
    export(LocalStore(tmp_path / "store", "dlp-datasets"), 3)
    with engine.connect() as conn:
        s0 = session_lineage(conn, "s0").exports
        assert len(s0) == 3 and len(session_lineage(conn, "s1").exports) == 2
    done = next(e for e in s0 if e.session_ids == ("s0",))
    assert (
        tmp_path / "store" / "dlp-datasets" / "exports" / done.export_id / "manifest.json"
    ).exists()


def test_export_refuses_blurred_video_after_privacy_reopened(
    engine: sa.Engine, policy: ExportPolicy, ontology: Ontology, tmp_path: Path
) -> None:
    """회귀(감사 4-2): 스냅샷에는 승인 상태였어도 그 뒤 블러 QA로 프라이버시가 다시 열린 세션의
    이전 블러본을 내보냈다. 지금 DB 상태로 승인을 확인하고, 다시 승인해도 다시 렌더하기 전에는
    렌더 기록 해시가 지금 블러 라벨과 달라 내보내지 않는다."""
    snapshots = LocalSnapshotStore(tmp_path / "snap")
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    datasets = LocalStore(tmp_path / "store", "dlp-datasets")
    times = vfr_times(np.random.default_rng(4), 1000)
    with engine.begin() as conn:
        register_ontology(conn, ontology)
        for i, sid in enumerate(["s0", "s1"]):
            insert_session(
                conn,
                make_session(
                    sid, worker_id=f"w{i}", site_id=f"site{i}",
                    streams=[{"stream_id": "bodycam", "kind": "bodycam",
                              "sync_method": "reference",
                              "uri": f"s3://dlp-raw/sessions/{sid}/bodycam.mp4"}],
                ).model_copy(update={"privacy_state": PrivacyState.APPROVED}),
            )  # fmt: skip
            insert_labels(conn, scenario_labels(sid, times))
            write_blurred(labeling, sid, times, tmp_path)
        build_dataset_version(
            conn, snapshots, load_dataset_policy(ROOT), version_id="dv1",
            ontology_version="1.0.0", golden_set_version=None, now=FIXED_TIME,
        )  # fmt: skip

    calls = iter(range(100))

    def export(fmt: str = "coco"):
        """dv1 내보내기 (호출마다 1분씩 늦은 시각 → 다른 내보내기 ID)."""
        return run_export(
            engine, root=ROOT, version_id="dv1", fmt=fmt, target="buyer",  # type: ignore[arg-type]
            snapshots=snapshots, labeling=labeling, datasets=datasets, raw_bucket="dlp-raw",
            policy=policy, ontology=ontology, include_unreviewed=False, splits=None,
            now=FIXED_TIME + timedelta(days=2, minutes=next(calls)),
        )  # fmt: skip

    assert set(export().record.session_ids) == {"s0", "s1"}
    with engine.begin() as conn:
        # 블러 QA에서 누락이 발견되어 프라이버시가 다시 열림 (dlp review collect와 같은 전이)
        set_privacy_state(conn, "s0", PrivacyState.AUTO_BLURRED)
    for fmt in ("coco", "intervals"):
        with pytest.raises(ExportError, match="s0: 지금 프라이버시 승인 상태가 아닙니다"):
            export(fmt)
    # 검수자가 블러를 더 그려 다시 승인했지만 다시 렌더하지 않았다
    blur = next(x for x in scenario_labels("s0", times) if x.kind == "blur_track")
    extra = blur.model_copy(update={"label_id": "s0-blur-added"})
    with engine.begin() as conn:
        insert_labels(conn, [extra])
        set_privacy_state(conn, "s0", PrivacyState.APPROVED)
    with pytest.raises(ExportError, match="s0/bodycam: 블러본이 지금 승인된 블러 라벨"):
        export()
    # 다시 렌더하면 내보낸다
    write_blurred(labeling, "s0", times, tmp_path, [*scenario_labels("s0", times), extra])
    assert set(export().record.session_ids) == {"s0", "s1"}
    # 블러본 파일만 바뀌고 렌더 기록이 그대로면 (렌더 밖에서 덮어씀) 받지 않는다
    other = tmp_path / "other.mp4"
    labeling.get_file("sessions/s1/blurred/bodycam.mp4", other)
    labeling.put_file("sessions/s0/blurred/bodycam.mp4", other, "0" * 64)
    with pytest.raises(ExportError, match="렌더 기록과 다릅니다"):
        export()
