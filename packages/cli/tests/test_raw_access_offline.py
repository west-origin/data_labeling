"""DB 없이(--no-db, 로컬 저장소) 수집해도 원본 접근이 감사 기록(JSON Lines)에 남는다 (WP16, ADR
0021).

`dlp ingest --no-db`의 두 경로를 확인한다.
- 로컬 저장소(`local:<디렉터리>`)에 수집하면 원본 쓰기마다 `<디렉터리>/raw_access.jsonl`에 기록이
  남는다.
- S3(실제 원본 버킷)에는 DB 감사 없이 올릴 수 없어 `SystemExit`로 거부한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dlp_cli.main import main
from dlp_cli.raw_access import OFFLINE_LOG
from dlp_fixtures.video import generate_blur_scenario
from dlp_schema.ops import RawAccessEvent
from dlp_schema.testing import FIXED_TIME


def test_no_db_ingest_is_audited_to_local_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """합성 블러 시나리오 영상(3초, `generate_blur_scenario`) 하나짜리 매니페스트를 `--no-db`로
    수집한다.

    정답: 모든 기록은 `RawAccessEvent`로 검증되고, 동작은 쓰기(`write`)뿐이며, 실행자는 `DLP_ACTOR`
    (`dev-1`), 용도는 `media.ingest`, 세션은 매니페스트의 `off-1`이다 (원본 키
    `sessions/off-1/...`에서 추출).
    """
    monkeypatch.setenv("DLP_ACTOR", "dev-1")
    generate_blur_scenario(3, session_id="off-1").write(tmp_path / "bodycam.mp4")
    manifest = {
        "session_id": "off-1", "domain": "cleaning", "worker_id": "w", "site_id": "s",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(), "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    store = tmp_path / "store"
    assert main(["ingest", str(tmp_path / "m.yaml"), "--store", f"local:{store}", "--no-db"]) == 0
    lines = (store / OFFLINE_LOG).read_text(encoding="utf-8").splitlines()
    events = [RawAccessEvent.model_validate(json.loads(x)) for x in lines]
    assert events and {e.action for e in events} == {"write"}
    assert {(e.actor, e.purpose, e.session_id) for e in events} == {
        ("dev-1", "media.ingest", "off-1")
    }


def test_no_db_refuses_real_raw_bucket(tmp_path: Path) -> None:
    """`--store s3 --no-db`는 매니페스트를 읽기 전에 "DB가 필요" 메시지로 거부된다 (빈
    매니페스트로도 확인)."""
    (tmp_path / "m.yaml").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="DB가 필요"):
        main(["ingest", str(tmp_path / "m.yaml"), "--store", "s3", "--no-db"])
