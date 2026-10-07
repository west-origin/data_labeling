"""원본 버킷 저장소는 이 모듈로만 만든다 (모든 접근이 감사 기록으로 남는다, WP16, ADR 0020·0021).

정적 검사(tests/test_raw_access_audited.py)가 다른 곳에서 원본 버킷 이름으로 저장소를 만드는 것
(설정의 원본 버킷 키, 이 모듈의 버킷 이름 도우미)을 막는다. 이 파일만 검사 예외다.

공개 함수:
- `raw_store(spec, url, purpose)` — DB 감사 기록(`raw_access_log`)을 남기는 원본 저장소.
  파이프라인의 기본 경로 (`ingest`, `sync run`, `privacy detect|render|audit-sample`, `prelabel
  run`, `train run`, `review …`, `ops privacy-audit`).
- `raw_store_offline(spec, purpose)` — DB 없이 로컬 개발할 때만 (`dlp ingest --no-db`). 기록은 로컬
  JSON Lines 파일(`OFFLINE_LOG`)에 남는다.
- `raw_bucket()` — 설정(`config/defaults.yaml`)의 원본 버킷 이름.

감사 규약 (`dlp_media.audit.AuditedStore`):
- 접근 **전에** 기록한다. 기록이 실패하면 접근하지 않는다.
- 내용을 보지 않는 `head`(존재·해시 확인)는 기록하지 않는다.
- 실행자(`actor`)는 `DLP_ACTOR` 환경 변수, 없으면 OS 사용자. 목적(`purpose`)은 명령 이름
  (예: `privacy.detect`)이며 `ops.yaml audit.service_purposes`와 대조된다.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from dlp_cli.schema_cmds import database_url
from dlp_media.audit import AuditedStore, DbAccessSink, FileAccessSink, current_actor
from dlp_media.storage import S3Store, store_from_spec
from dlp_schema import load_config, repo_root

# DB 없이 돌 때 감사 기록 파일 (로컬 저장소 루트 아래)
OFFLINE_LOG = "raw_access.jsonl"


def raw_bucket() -> str:
    """`config/defaults.yaml`의 원본 버킷 이름 (개발 기본값 `dlp-raw`). 이 모듈 밖에서 부르면 정적
    검사가 실패한다."""
    return load_config(repo_root() / "config" / "defaults.yaml").buckets.raw


def raw_store(spec: str, url: str | None, purpose: str) -> AuditedStore:
    """감사되는 원본 저장소. purpose는 명령 이름 (예: privacy.detect).

    인자:
        spec: `s3`(환경 변수 `DLP_S3_*`의 서비스 자격 증명) 또는 `local:<디렉터리>`.
        url: 감사 기록을 쓸 DB URL (`database_url` 규칙). 저장소 자체와는 무관하다.
        purpose: 감사 기록의 용도 문자열 (`<단계>.<명령>`).

    반환: `AuditedStore`. 읽기·서명 URL·쓰기·grant마다 `raw_access_log`에 한 건씩 **별도
    커밋**한다 (호출자의 트랜잭션이 롤백되어도 감사 기록은 남는다).
    부작용: DB 엔진 생성 (연결은 첫 기록 때).
    """
    sink = DbAccessSink(sa.create_engine(database_url(url)))
    inner = S3Store.from_env(raw_bucket()) if spec == "s3" else store_from_spec(spec, raw_bucket())
    return AuditedStore(inner, sink, actor=current_actor(), purpose=purpose)


def raw_store_offline(spec: str, purpose: str) -> AuditedStore:
    """DB 없이 쓰는 감사 원본 저장소 (로컬 저장소 'local:<디렉터리>'만).

    접근 기록은 <디렉터리>/raw_access.jsonl에 덧붙인다. 실제 원본 버킷(S3)은 DB 감사 기록이
    있어야 하므로 여기서 만들지 않는다.

    예외: `spec`이 `local:`로 시작하지 않으면 `SystemExit`.
    """
    if not spec.startswith("local:"):
        raise SystemExit(
            "원본 버킷에 올릴 때는 DB가 필요합니다 (접근 감사 기록). --no-db는 local: 저장소에만"
        )
    root = Path(spec.removeprefix("local:"))
    inner = store_from_spec(spec, raw_bucket())
    sink = FileAccessSink(root / OFFLINE_LOG)
    return AuditedStore(inner, sink, actor=current_actor(), purpose=purpose)
