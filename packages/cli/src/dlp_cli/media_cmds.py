"""미디어·수집 하위 명령 (WP3, ADR 0003).

등록하는 명령:
- `dlp media probe <파일>` — ffprobe로 컨테이너·스트림 정보를 출력하는 진단 도구.
- `dlp media pts-index <파일> --out <parquet>` — 영상의 PTS 인덱스를 만든다 (진단·디버그용).
- `dlp ingest <매니페스트> [--store] [--url] [--no-db]` — 세션 수집: 원본을 원본 버킷에
  불변·멱등으로 올리고, PTS 인덱스·프록시·IMU·장갑 정규화본 같은 파생 파일을 만들고, `sessions`
  테이블에 세션을 등록한다. 파이프라인의 첫 단계다.

입력 → 출력: 세션 매니페스트(YAML/JSON, 스트림 파일 경로 포함) → 원본 버킷 `sessions/<세션>/raw/…`,
`sessions/<세션>/derived/…` + DB `sessions`(스트림 포함).

주의:
- 원본 저장소는 항상 `dlp_cli.raw_access`의 감사 저장소로 만든다 (ADR 0020). 수집도 원본 쓰기라
  `raw_access_log`(또는 `--no-db`일 때 로컬 `raw_access.jsonl`)에 기록이 남는다.
- 같은 매니페스트를 다시 수집하면 이미 있는 원본·파생 파일은 건너뛰고, DB의 세션이 같은 내용이면
  `unchanged`, 다르면 `SessionConflictError`로 실패한다 (멱등).
- 영상 시간은 PTS 인덱스로만 계산한다 (프레임 번호 * 간격 금지).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store, raw_store_offline
from dlp_cli.schema_cmds import database_url
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.probe import probe
from dlp_media.pts import build_pts_index
from dlp_schema import repo_root
from dlp_schema.config import load_config


def cmd_probe(args: argparse.Namespace) -> int:
    """`dlp media probe <파일>`: 컨테이너 형식, 길이(ms), 생성 시각, 비디오·오디오·데이터 트랙을
    출력한다.

    데이터 트랙(`tag=gpmd` 등)은 GoPro GPMF 같은 내장 IMU가 있는지 볼 때 쓴다. 파일은 로컬 경로이며
    저장소를 거치지 않으므로 감사 기록이 남지 않는다. 반환: 항상 0 (ffprobe 실패는 예외로 끝난다).
    """
    info = probe(Path(args.file))
    print(f"형식 {info.format_name}, 길이 {info.duration_ms} ms, 생성 시각 {info.creation_time}")
    if info.video:
        v = info.video
        print(f"비디오 #{v.index} {v.codec} {v.width}x{v.height} time_base={v.time_base}")
    if info.audio:
        a = info.audio
        print(f"오디오 #{a.index} {a.codec} {a.sample_rate} Hz {a.channels}ch")
    for d in info.data_streams:
        print(f"데이터 #{d.index} tag={d.codec_tag}")
    return 0


def cmd_pts_index(args: argparse.Namespace) -> int:
    """`dlp media pts-index <파일> --out <경로>`: 영상의 프레임별 PTS 인덱스를 Parquet로 쓴다.

    출력에는 프레임 수, 가변/고정 프레임레이트 여부(VFR), 길이(ms, 소수 표시)를 찍는다.
    수집(`dlp ingest`)도 같은 함수로 `derived/<스트림>.pts.parquet`를 만들므로, 이 명령은 로컬
    확인용이다.
    """
    index = build_pts_index(Path(args.file))
    index.write(Path(args.out))
    vfr = "가변" if index.is_vfr else "고정"
    print(
        f"프레임 {len(index)}개, {vfr} 프레임레이트, 길이 {index.duration_ms:.1f} ms → {args.out}"
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """`dlp ingest <매니페스트>`: 세션 하나를 수집한다.

    인자:
        args.manifest: 세션 매니페스트 경로. 스트림 상대 경로는 매니페스트 위치 기준이다.
        args.store: 원본 저장소 지정. `s3`(환경 변수의 S3 자격 증명) 또는 `local:<디렉터리>`.
        args.url: DB URL (`database_url` 규칙).
        args.no_db: 참이면 DB에 세션을 등록하지 않는다. 이때 `--store`는 `local:`만 허용된다
            (`raw_store_offline`이 S3면 `SystemExit`).

    반환: 0. 매니페스트 오류·파일 누락·세션 충돌은 예외로 끝난다.
    부작용: 원본 버킷 쓰기(감사 기록), DB 모드면 한 트랜잭션에서 `sessions`(스트림 포함) INSERT.
    프록시 인코딩 설정은 `config/defaults.yaml`의 `media.proxy`에서 읽는다.
    """
    # 원본 저장소는 늘 감사 저장소다. --no-db(로컬 저장소만)면 기록을 로컬 JSON Lines 파일에 남긴다
    raw = (
        raw_store_offline(args.store, "media.ingest")
        if args.no_db
        else raw_store(args.store, args.url, "media.ingest")
    )
    manifest, base = load_manifest(Path(args.manifest))
    proxy = load_config(repo_root() / "config" / "defaults.yaml").media.proxy
    if args.no_db:
        result = ingest_session(manifest, base, raw, proxy=proxy)
    else:
        # 업로드와 DB 등록을 한 트랜잭션 블록 안에서 한다. 업로드 뒤 등록이 실패해도 원본은
        # 불변 키로 남으므로 다시 돌리면 업로드는 건너뛰고 등록만 다시 시도한다
        engine = sa.create_engine(database_url(args.url))
        with engine.begin() as conn:
            result = ingest_session(manifest, base, raw, conn, proxy=proxy)
        engine.dispose()
    print(f"세션 {result.session.session_id}: 스트림 {len(result.session.streams)}개")
    # result.db: "inserted"(새 세션) / "unchanged"(같은 내용으로 이미 있음) / "skipped"(--no-db)
    print(f"업로드 {len(result.uploaded)}, 건너뜀 {len(result.skipped)}, DB {result.db}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
    """`media probe|pts-index`와 최상위 `ingest` 명령을 등록한다."""
    media = sub.add_parser("media", help="미디어 파일 도구")
    media_sub = media.add_subparsers(dest="media_command", required=True)
    p = media_sub.add_parser("probe", help="컨테이너·스트림 정보")
    p.add_argument("file")
    p.set_defaults(func=cmd_probe)
    p = media_sub.add_parser("pts-index", help="PTS 인덱스 Parquet 생성")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_pts_index)

    ingest = sub.add_parser("ingest", help="세션 매니페스트 수집")
    ingest.add_argument("manifest")
    ingest.add_argument("--store", default="s3", help="'s3' 또는 'local:<디렉터리>'")
    ingest.add_argument("--url", help="DB URL (기본: DLP_DATABASE_URL)")
    ingest.add_argument("--no-db", action="store_true", help="DB에 등록하지 않는다")
    ingest.set_defaults(func=cmd_ingest)
