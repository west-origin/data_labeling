"""미디어·수집 하위 명령."""

from __future__ import annotations

import argparse
from pathlib import Path

import sqlalchemy as sa

from dlp_cli.raw_access import raw_store, raw_store_offline
from dlp_cli.schema_cmds import database_url
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.probe import probe
from dlp_media.pts import build_pts_index


def cmd_probe(args: argparse.Namespace) -> int:
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
    index = build_pts_index(Path(args.file))
    index.write(Path(args.out))
    vfr = "가변" if index.is_vfr else "고정"
    print(
        f"프레임 {len(index)}개, {vfr} 프레임레이트, 길이 {index.duration_ms:.1f} ms → {args.out}"
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    # 원본 저장소는 늘 감사 저장소다. --no-db(로컬 저장소만)면 기록을 로컬 JSON Lines 파일에 남긴다
    raw = (
        raw_store_offline(args.store, "media.ingest")
        if args.no_db
        else raw_store(args.store, args.url, "media.ingest")
    )
    manifest, base = load_manifest(Path(args.manifest))
    if args.no_db:
        result = ingest_session(manifest, base, raw)
    else:
        engine = sa.create_engine(database_url(args.url))
        with engine.begin() as conn:
            result = ingest_session(manifest, base, raw, conn)
        engine.dispose()
    print(f"세션 {result.session.session_id}: 스트림 {len(result.session.streams)}개")
    print(f"업로드 {len(result.uploaded)}, 건너뜀 {len(result.skipped)}, DB {result.db}")
    return 0


def add_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:  # pyright: ignore[reportPrivateUsage]
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
