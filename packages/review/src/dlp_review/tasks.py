# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""세션에서 검수 작업을 만든다.

- 프라이버시 검수 (CVAT, 원본 접근 권한자): 원본 프록시 영상 + 현재 블러 트랙.
- 작업 라벨 검수 (일반 라벨러): 라벨러 워터마크를 입힌 블러본.
  - CVAT: 객체·도구 박스 트랙과 키포인트 트랙
  - Label Studio: 시간 구간 라벨, 영상과 장갑·IMU 시계열 동기 재생
  일반 라벨러 경로의 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
- 블러본은 지금 승인된 블러 라벨로 렌더한 것만 쓴다 (dlp_privacy.runner.assert_render_current).
- CVAT 작업은 담당자의 CVAT 계정(review.yaml cvat.users)에 배정한다. 블러 검수는 연결이 없으면
  만들지 않고, 검수 우선 구간을 CVAT 이슈로 남겨 검수자가 먼저 보게 한다 (ADR 0024).

WP6, ADR 0006·0020·0023·0024. 진입점: `dlp review create <세션>`(직접 만들기)과
`dlp review assign <배정>`(`ops.runner.create_assignment_tasks` 경유).

공개 이름:
- `VIDEO_KINDS`, `PRIVACY_PROJECT`, `SPATIAL_PROJECT`, `TEMPORAL_PROJECT`:
  영상 종류와 도구 프로젝트 이름.
- `TaskError`: 작업을 만들거나 수집할 수 없다 (전제 조건 위반).
- `ReviewSetup`: 저장소·도구 클라이언트·정책 묶음 (정책은 없으면 저장소에서 늦게 읽는다).
- `frame_times`, `video_size`, `proxy_key`, `raw_video_size`, `label_scale`, `object_key`:
  매체 도우미.
- `current_for_review`, `Selector`, `default_selector`: 작업에 넣을 라벨 고르기.
- `cvat_user_id`: dlp 검수자 → CVAT 사용자 ID.
- `create_privacy_tasks`: 블러 검수 작업 (스트림별 CVAT).
- `lock_assignment`: 배정 행 잠금 (작업 중복 생성 방지).
- `create_labeling_tasks`: 작업 라벨 검수 작업 (CVAT 공간 + Label Studio 시간).

저장소 키:
- 원본 버킷: `sessions/<세션>/derived/<스트림>.proxy.mp4`(검수 프록시), 장갑·IMU 원본 Parquet.
- 라벨링 버킷: `sessions/<세션>/review/<담당자>/<스트림>.mp4`(워터마크 영상),
  `sessions/<세션>/review/timeseries.csv`(시계열).
원본 접근은 `setup.raw`가 감사 저장소(`dlp_cli.raw_access.raw_store`)일 때 기록된다 (ADR 0020).
"""

from __future__ import annotations

import bisect
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import sqlalchemy as sa
from av.error import FFmpegError

from dlp_media.audit import grant
from dlp_media.pts import build_pts_index
from dlp_media.storage import ObjectStore, S3Store, blurred_key, sha256_file
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.review import ReviewSegment
from dlp_privacy.runner import (
    RenderNotCurrentError,
    assert_render_current,
    check_fetched,
    read_review_segments,
)
from dlp_review.clients import CvatClient, LabelStudioClient
from dlp_review.cvat import CvatSchema, Scale, label_spec, to_cvat_tracks
from dlp_review.labelstudio import LS_KINDS, label_config, to_ls_results
from dlp_review.ops.policy import CvatPolicy, MediaPolicy
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.roles import check_privacy_reviewers, check_stage_uris
from dlp_review.timeseries import write_timeseries_csv
from dlp_review.watermark import burn_watermark
from dlp_schema import repo_root
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_session,
    insert_review_task,
    list_review_tasks,
)
from dlp_schema.db.tables import review_assignments
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTool
from dlp_schema.session import PrivacyState, Session, Stream, StreamKind
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import Series, glove_series, imu_series

# 검수 작업을 만드는 영상 스트림 종류
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 검수 도구 프로젝트 이름 (없으면 처음 쓸 때 만든다)
PRIVACY_PROJECT = "dlp-privacy"
SPATIAL_PROJECT = "dlp-labeling-spatial"
TEMPORAL_PROJECT = "dlp-labeling-temporal"


class TaskError(RuntimeError):
    """검수 작업을 만들거나 수집할 수 없다 (승인 전, 계정 연결 없음, 주석 없음 등)."""


@dataclass
class ReviewSetup:
    """작업 만들기·수집에 필요한 저장소, 도구 클라이언트, 정책 묶음.

    정책 필드가 None이면 처음 쓸 때 저장소 루트의 YAML을 읽어 채운다 (그래서 frozen이 아니다).
    """

    # 원본 버킷 저장소 (운영에서는 감사 저장소 `dlp_cli.raw_access.raw_store`)
    raw: ObjectStore
    # 라벨링 버킷 저장소 (블러본·워터마크 영상·시계열 CSV)
    labeling: ObjectStore
    labeling_reader: S3Store  # 라벨러 자격 증명. 서명 URL용
    # 라벨 이름·블러 대상·객체 클래스 목록의 출처
    ontology: Ontology
    # CVAT 클라이언트 (없으면 CVAT 작업을 만들지 않는다)
    cvat: CvatClient | None = None
    # Label Studio 클라이언트 (없으면 Label Studio 작업을 만들지 않는다)
    label_studio: LabelStudioClient | None = None
    media: MediaPolicy | None = None  # 없으면 config/policies/review.yaml media를 읽는다
    cvat_config: CvatPolicy | None = None  # 없으면 config/policies/review.yaml cvat
    privacy: PrivacyPolicy | None = None  # 없으면 config/policies/privacy.yaml (블러본 해시)
    glove_prefixes: tuple[str, ...] | None = None  # 없으면 config/policies/sync.yaml glove

    def media_policy(self) -> MediaPolicy:
        """검수 화면 매체 정책 (`review.yaml media`). 처음 부를 때 읽어 둔다."""
        if self.media is None:
            self.media = load_ops_policy(repo_root()).media
        return self.media

    def cvat_policy(self) -> CvatPolicy:
        """CVAT 계정 연결 정책 (`review.yaml cvat`). 처음 부를 때 읽어 둔다."""
        if self.cvat_config is None:
            self.cvat_config = load_ops_policy(repo_root()).cvat
        return self.cvat_config

    def privacy_policy(self) -> PrivacyPolicy:
        """프라이버시 정책 (`privacy.yaml`). 블러본 렌더 해시 확인에 쓴다."""
        if self.privacy is None:
            self.privacy = load_privacy_policy(repo_root())
        return self.privacy

    def pressure_prefixes(self) -> tuple[str, ...]:
        """장갑 압력 채널 접두사 (`sync.yaml glove.pressure_prefixes`). 처음 부를 때 읽어 둔다."""
        if self.glove_prefixes is None:
            path = repo_root() / "config" / "policies" / "sync.yaml"
            self.glove_prefixes = load_sync_policy(path).glove.pressure_prefixes
        return self.glove_prefixes


def frame_times(video: Path) -> list[int]:
    """영상의 프레임 시각 목록 (PTS 인덱스, 정수 ms로 반올림). 인덱스 = CVAT 프레임 번호.

    프레임 번호 * 간격으로 계산하지 않고 PTS 인덱스만 쓴다 (VFR 영상도 정확, CLAUDE.md 규칙).
    """
    return [round(t) for t in build_pts_index(video).ms]


def video_size(source: Path | str) -> tuple[int, int]:
    """영상 (너비, 높이). source는 파일 경로 또는 서명 URL (헤더만 읽는다)."""
    with av.open(str(source)) as c:
        ctx = c.streams.video[0].codec_context
        return int(ctx.width), int(ctx.height)


def proxy_key(session_id: str, stream_id: str) -> str:
    """원본 버킷의 검수용 프록시 (수집 때 dlp_media.proxy.make_proxy로 만든다, 축소될 수 있다)."""
    return f"sessions/{session_id}/derived/{stream_id}.proxy.mp4"


def raw_video_size(raw: ObjectStore, stream: Stream, work: Path) -> tuple[int, int]:
    """원본 영상의 해상도. 서명 URL을 만들 수 있으면 헤더만 읽고, 아니면 내려받아 본다.

    둘 다 감사 저장소를 거치므로 원본 접근 기록이 남는다.

    인자: raw(원본 저장소), stream(원본 영상 스트림, `uri`가 원본 버킷), work(내려받을 임시 폴더).
    같은 work에 이미 받아 둔 파일이 있으면 다시 받지 않는다.
    """
    key = object_key(raw, stream.uri)
    # presign이 없는 저장소(LocalStore 등)는 내려받는다
    presign: Callable[[str], str] | None = getattr(raw, "presign", None)
    if presign is not None:
        try:
            return video_size(presign(key))
        except (FFmpegError, OSError):
            pass  # 서명 URL을 열 수 없는 환경이면 내려받는다
    dest = work / f"{stream.stream_id}.raw{Path(key).suffix}"
    if not dest.exists():
        raw.get_file(key, dest)
    return video_size(dest)


def label_scale(raw: ObjectStore, stream: Stream, media: Path, work: Path) -> Scale:
    """라벨(원본 화소) → 검수 화면 영상(media) 화소 비율.

    블러 라벨은 원본 영상 좌표인데 프라이버시 검수 화면은 축소한 프록시라서 좌표를 맞춘다.
    반환: (화면 너비 / 원본 너비, 화면 높이 / 원본 높이). 원본 해상도를 읽으므로 원본 접근이
    기록된다.
    """
    mw, mh = video_size(media)
    rw, rh = raw_video_size(raw, stream, work)
    return (mw / rw, mh / rh)


def object_key(store: ObjectStore, uri: str) -> str:
    """저장소 URI → 버킷 안 키. uri가 이 저장소의 버킷을 가리키지 않으면 `TaskError`."""
    prefix = store.uri("")
    if not uri.startswith(prefix):
        raise TaskError(f"{uri}는 {store.bucket}에 있지 않습니다")
    return uri.removeprefix(prefix)


def current_for_review(
    conn: sa.Connection, session_id: str, stream_id: str | None, kinds: tuple[str, ...]
) -> list[LabelRecord]:
    """검수에 보낼 현재 운영 라벨.

    인자: stream_id가 None이면 세션의 모든 스트림, 아니면 그 스트림 라벨과 스트림이 없는 라벨.
    kinds: 라벨 종류.
    `current_labels`가 측정 레코드·오류 삽입 계보를 빼지만, 오류 삽입 사본을 한 번 더 걸러 낸다.
    """
    labels = current_labels(get_labels(conn, session_id, kinds=list(kinds)))
    return [
        x
        for x in labels
        if not x.seeded_error and (stream_id is None or x.stream_id in (stream_id, None))
    ]


# 작업에 넣을 라벨을 고르는 함수: (스트림 또는 None, 라벨 종류) → 라벨
Selector = Callable[[str | None, tuple[str, ...]], list[LabelRecord]]


def default_selector(conn: sa.Connection, session_id: str) -> Selector:
    """현재 운영 라벨 (오류 삽입·측정 레코드 제외)."""

    def pick(stream_id: str | None, kinds: tuple[str, ...]) -> list[LabelRecord]:
        """`current_for_review`를 그대로 부른다."""
        return current_for_review(conn, session_id, stream_id, kinds)

    return pick


def cvat_user_id(cvat: CvatClient, users: Mapping[str, str], reviewer: str) -> int | None:
    """dlp 검수자 → CVAT 사용자 ID (review.yaml cvat.users). 연결이 없으면 None.

    예외: 연결은 있는데 CVAT에 그 사용자가 없으면 `TaskError`.
    """
    username = users.get(reviewer)
    if username is None:
        return None
    uid = cvat.find_user_id(username)
    if uid is None:
        raise TaskError(f"CVAT에 사용자 {username}(검수자 {reviewer})가 없습니다")
    return uid


def _segment_issues(
    cvat: CvatClient,
    task_id: int,
    segments: Sequence[ReviewSegment],
    times: list[int],
    limit: int,
) -> int:
    """검수 우선 구간을 CVAT 이슈로 남긴다 (우선순위 순으로 limit개). 남긴 수.

    인자:
    - segments: 프라이버시 단계가 남긴 검수 우선 구간 (`priority`가 작을수록 먼저).
    - times: 화면 영상 프레임 시각. 구간 시작 시각 이상인 첫 프레임에 이슈를 단다.
    - limit: 최대 이슈 수 (`review.yaml cvat.privacy_issue_limit`).

    이슈는 그 프레임을 담은 job에 단다 (못 찾으면 첫 job). 위치는 화면 왼쪽 위 16*16 화소.
    부작용: CVAT 이슈 생성 (외부 서비스 호출).
    """
    jobs = cvat.jobs(task_id)
    if not jobs or not times:
        return 0
    made = 0
    for seg in sorted(segments, key=lambda x: (x.priority, x.t_start_ms, x.target))[:limit]:
        # 구간 시작(ms) → 그 시각 이상인 첫 프레임 번호 (영상 끝을 넘으면 마지막 프레임)
        frame = min(bisect.bisect_left(times, seg.t_start_ms), len(times) - 1)
        job = next(
            (j for j in jobs if int(j["start_frame"]) <= frame <= int(j["stop_frame"])), jobs[0]
        )
        detail = f" ({seg.detail})" if seg.detail else ""
        message = (
            f"[먼저 볼 구간] {seg.reason} · {seg.target} · "
            f"{seg.t_start_ms}-{seg.t_end_ms} ms{detail}"
        )
        cvat.add_issue(int(job["id"]), frame, [0.0, 0.0, 16.0, 16.0], message)
        made += 1
    return made


def _cvat_project(cvat: CvatClient, name: str, labels: list[str]) -> tuple[int, CvatSchema]:
    """이름으로 CVAT 프로젝트를 찾거나 만든다. (프로젝트 ID, 라벨·속성 ID 표).

    주의: 이미 있는 프로젝트의 라벨 목록은 바꾸지 않는다 (온톨로지에 새 클래스가 생겨도 그대로).
    """
    project = cvat.find_project(name)
    pid = int(project["id"]) if project else cvat.create_project(name, label_spec(labels))
    return pid, CvatSchema.from_labels(cvat.project_labels(pid))


def create_privacy_tasks(
    conn: sa.Connection,
    session_id: str,
    setup: ReviewSetup,
    now: datetime,
    *,
    assignee: str,
    privacy_reviewers: Sequence[str],
    select: Selector | None = None,
    mode: ReviewMode = ReviewMode.STANDARD,
    assignment_id: str | None = None,
    streams: set[str] | None = None,
) -> list[ReviewTask]:
    """select가 없으면 현재 블러 트랙을 넣는다 (배정 방식에 따라 다른 라벨을 넣을 때 쓴다).

    블러 검수는 원본 영상을 열므로 담당자는 원본 접근 권한자(privacy_reviewers,
    config/policies/review.yaml reviewers.privacy)여야 한다. 원본을 검수 도구에 올리기 **전에**
    grant 감사 기록을 남긴다 (ADR 0020: 접근 전에 기록).

    인자:
    - assignee: 담당자 (원본 접근 권한자이고 `cvat.users`에 CVAT 계정 연결이 있어야 한다).
    - mode·assignment_id: 배정에서 만들 때 작업에 남길 방식·배정 ID.
    - streams: 이 스트림만 (None이면 세션의 모든 영상 스트림).

    반환: 스트림마다 하나씩 만든 `ReviewTask` (DB `review_tasks`에도 삽입).
    예외: `AccessError`(권한자 아님), `TaskError`(CVAT 없음, 계정 연결 없음, 자동 탐지 전),
    `clients.ToolError`(CVAT API 오류).
    부작용: 원본 프록시 내려받기, grant 감사 기록, CVAT 작업 생성·배정·트랙·이슈, DB 삽입.
    순서: 권한·계정 검사 → (스트림마다) 프록시 받기 → grant 기록 → CVAT 업로드.
    """
    # 원본을 읽기 전에 권한부터 확인한다 (권한이 없으면 감사 기록도 남지 않는다)
    check_privacy_reviewers([assignee], privacy_reviewers)
    pick = select or default_selector(conn, session_id)
    if setup.cvat is None:
        raise TaskError("CVAT 클라이언트가 없습니다")
    cvat_conf = setup.cvat_policy()
    # 담당자의 CVAT 계정에 배정해야 그 사람만 본다. 연결이 없으면 원본을 올리기 전에 멈춘다
    cvat_user = cvat_user_id(setup.cvat, cvat_conf.users, assignee)
    if cvat_user is None:
        raise TaskError(
            f"블러 검수 담당자 {assignee}의 CVAT 계정이 없습니다 "
            "(config/policies/review.yaml cvat.users)"
        )
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise TaskError(f"{session_id}: 블러 자동 탐지를 먼저 실행하세요 (dlp privacy detect)")
    pid, schema = _cvat_project(setup.cvat, PRIVACY_PROJECT, list(setup.ontology.privacy_targets))
    out: list[ReviewTask] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in (s for s in session.streams if s.kind in VIDEO_KINDS):
            if streams is not None and stream.stream_id not in streams:
                continue
            key = proxy_key(session_id, stream.stream_id)
            video = work / f"{stream.stream_id}.mp4"
            setup.raw.get_file(key, video)
            # 블러 라벨(원본 화소) → 프록시 화면 화소 비율
            scale = label_scale(setup.raw, stream, video, work)
            labels = pick(stream.stream_id, ("blur_track",))
            segments = read_review_segments(setup.raw, session_id, stream.stream_id, work)
            grant(setup.raw, key, assignee)  # 원본을 검수자에게 보여 준다 (올리기 전에 기록)
            tid = setup.cvat.create_task(f"{session_id}/{stream.stream_id}/privacy", pid, video)
            setup.cvat.assign(tid, cvat_user)
            # 키프레임 시각 ↔ 프록시 프레임 번호 (프록시는 원본과 같은 PTS를 유지한다고 가정)
            times = frame_times(video)
            setup.cvat.put_tracks(tid, to_cvat_tracks(labels, times, schema, scale=scale))
            _segment_issues(setup.cvat, tid, segments, times, cvat_conf.privacy_issue_limit)
            task = ReviewTask(
                task_key=f"cvat:{tid}", tool=ReviewTool.CVAT, external_id=str(tid),
                session_id=session_id, stream_id=stream.stream_id, stage=ReviewStage.PRIVACY,
                media_uri=setup.raw.uri(key), label_kinds=("blur_track",), created_at=now,
                mode=mode, assignment_id=assignment_id, assignee=assignee,
                sent_label_ids=tuple(x.label_id for x in labels),
            )  # fmt: skip
            insert_review_task(conn, task)
            out.append(task)
    return out


def lock_assignment(conn: sa.Connection, assignment_id: str) -> list[ReviewTask] | None:
    """배정 행을 잠근다 (SELECT … FOR UPDATE). 이미 작업이 있으면 그 작업들을 돌려준다.

    같은 배정으로 동시에 작업을 만들면 두 번째 호출은 첫 트랜잭션이 끝날 때까지 기다렸다가
    이미 만든 작업을 받는다 (검수 도구에 작업이 두 번 생기지 않는다).

    반환: 작업이 아직 없으면 None (잠금은 호출자 트랜잭션 끝까지 유지된다).
    예외: 배정이 없으면 `scalar_one`이 예외를 던진다.
    """
    task_key = conn.execute(
        sa.select(review_assignments.c.task_key)
        .where(review_assignments.c.assignment_id == assignment_id)
        .with_for_update()
    ).scalar_one()
    if task_key is None:
        return None
    a = get_assignment(conn, assignment_id)
    return [t for t in list_review_tasks(conn, a.session_id) if t.assignment_id == assignment_id]


def _series(
    session: Session, raw: ObjectStore, work: Path, prefixes: Sequence[str]
) -> dict[str, Series]:
    """prefixes: 장갑 압력 채널 접두사 (sync.yaml glove.pressure_prefixes).

    장갑·IMU 스트림의 원본 Parquet을 원본 버킷에서 내려받아(원본 접근 기록) 시계열로 바꾼다.
    반환: 스트림 ID → `Series` (그 스트림 시계 ms). 장갑은 압력 합, IMU는 가속도 크기.
    """
    out: dict[str, Series] = {}
    for s in session.streams:
        if s.kind in (StreamKind.GLOVE_LEFT, StreamKind.GLOVE_RIGHT, StreamKind.IMU):
            path = work / f"{s.stream_id}.parquet"
            raw.get_file(object_key(raw, s.uri), path)
            out[s.stream_id] = (
                imu_series(path)
                if s.kind is StreamKind.IMU
                else glove_series(path, pressure_prefixes=prefixes)
            )
    return out


def create_labeling_tasks(
    conn: sa.Connection,
    session_id: str,
    setup: ReviewSetup,
    assignee: str,
    now: datetime,
    *,
    select: Selector | None = None,
    mode: ReviewMode = ReviewMode.STANDARD,
    assignment_id: str | None = None,
    streams: set[str] | None = None,
    tools: set[ReviewTool] | None = None,
) -> list[ReviewTask]:
    """select가 없으면 현재 운영 라벨을 넣는다. tools·streams로 만들 작업을 좁힌다.

    영상 스트림마다: 현재 블러본을 받아 확인 → 담당자 워터마크 → 라벨링 버킷에 올림 →
    CVAT 공간 작업(박스·키포인트)과, 바디캠이면 Label Studio 시간 작업(시계열 CSV 포함)을 만든다.

    인자:
    - assignee: 담당자 (워터마크 글씨와 매체 경로에 들어간다. CVAT 계정 연결이 있으면 배정한다).
    - tools: 만들 도구 (None이면 setup에 있는 클라이언트 모두).
    - streams: 이 스트림만 (None이면 모든 영상 스트림).

    반환: 만든 `ReviewTask` 목록 (모두 만든 뒤 한꺼번에 DB `review_tasks`에 삽입).
    예외: `TaskError`(프라이버시 승인 전, 블러본이 현재 블러 라벨과 다름·무효),
    `RawAccessError`(작업 URI에 원본 버킷이 섞임), `clients.ToolError`.
    부작용: 라벨링 버킷 쓰기(워터마크 영상, 시계열 CSV), 원본 버킷 읽기(장갑·IMU Parquet,
    감사 기록), 도구 작업 생성, DB 삽입.
    주의: Label Studio 작업은 바디캠 스트림마다 하나씩 생긴다. 바디캠이 여럿이면 같은 시간 라벨로
    작업이 여러 개 생기고 시계열 CSV 키가 같아 덮어쓴다 (배정 경로는 기준 스트림 하나만 넘긴다).
    """
    pick = select or default_selector(conn, session_id)
    use_cvat = setup.cvat is not None and (tools is None or ReviewTool.CVAT in tools)
    use_ls = setup.label_studio is not None and (tools is None or ReviewTool.LABEL_STUDIO in tools)
    session = get_session(conn, session_id)
    media = setup.media_policy()
    if session.privacy_state is not PrivacyState.APPROVED:
        raise TaskError(f"{session_id}: 프라이버시 승인 전에는 작업 라벨 검수를 만들 수 없습니다")
    raw_bucket = setup.raw.bucket
    out: list[ReviewTask] = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for stream in (s for s in session.streams if s.kind in VIDEO_KINDS):
            if streams is not None and stream.stream_id not in streams:
                continue
            # 이 스트림에 만들 작업이 없으면 블러본도 받지 않는다 (Label Studio는 바디캠만)
            if not use_cvat and not (use_ls and stream.kind is StreamKind.BODYCAM):
                continue
            blurred = work / f"{stream.stream_id}-blurred.mp4"
            # 지금 승인된 블러 라벨로 렌더한 블러본만 쓴다 (승인 취소·재승인 뒤 이전 블러본 금지)
            try:
                rendered = assert_render_current(
                    conn, setup.labeling, session_id, stream.stream_id, setup.privacy_policy()
                )
                setup.labeling.get_file(blurred_key(session_id, stream.stream_id), blurred)
                # 받은 파일이 렌더 기록의 해시와 같은지 확인한다 (그 사이 바뀐 파일 금지)
                check_fetched(blurred, rendered, session_id, stream.stream_id)
            except RenderNotCurrentError as exc:
                raise TaskError(str(exc)) from exc
            marked = work / f"{stream.stream_id}-{assignee}.mp4"
            burn_watermark(
                blurred, marked, f"{assignee} {session_id}", opacity=media.watermark_opacity,
                crf=media.watermark_crf, encoder_rate=media.encoder_rate,
            )  # fmt: skip
            vkey = f"sessions/{session_id}/review/{assignee}/{stream.stream_id}.mp4"
            setup.labeling.put_file(vkey, marked, sha256_file(marked))
            media_uri = setup.labeling.uri(vkey)

            if use_cvat:
                assert setup.cvat is not None
                labels = pick(stream.stream_id, ("box_track", "keypoint_track"))
                # 프로젝트 라벨: 온톨로지 객체 클래스 + 키포인트 골격
                # (kp_<골격>, cvat.cvat_label_name)
                names = [*setup.ontology.objects, "kp_hand21", "kp_coco17", "kp_wholebody133"]
                pid, schema = _cvat_project(setup.cvat, SPATIAL_PROJECT, names)
                cvat_user = cvat_user_id(setup.cvat, setup.cvat_policy().users, assignee)
                tid = setup.cvat.create_task(
                    f"{session_id}/{stream.stream_id}/{assignee}", pid, marked
                )
                # 작업 라벨 작업은 계정 연결이 없어도 만든다 (배정 없이 관리자가 나눠 줄 수 있다)
                if cvat_user is not None:
                    setup.cvat.assign(tid, cvat_user)
                # 워터마크 영상은 블러본의 PTS·해상도를 유지하므로
                # 라벨 좌표를 그대로(UNIT_SCALE) 보낸다
                setup.cvat.put_tracks(tid, to_cvat_tracks(labels, frame_times(marked), schema))
                spatial = ("box_track", "keypoint_track")
                out.append(
                    _task(f"cvat:{tid}", ReviewTool.CVAT, tid, session_id, stream.stream_id,
                          assignee, media_uri, spatial, now, mode, assignment_id, labels)
                )  # fmt: skip

            if use_ls and stream.kind is StreamKind.BODYCAM:
                assert setup.label_studio is not None
                csv = work / "timeseries.csv"
                write_timeseries_csv(
                    session, _series(session, setup.raw, work, setup.pressure_prefixes()), csv,
                    rate_hz=media.timeseries_rate_hz,
                )  # fmt: skip
                ckey = f"sessions/{session_id}/review/timeseries.csv"
                setup.labeling.put_file(ckey, csv, sha256_file(csv))
                # 시간 라벨은 세션 단위 (스트림 None)
                labels = pick(None, LS_KINDS)
                ls = setup.label_studio
                project = ls.find_project(TEMPORAL_PROJECT)
                if project:
                    lpid = int(project["id"])
                    ls.ensure_prelabel_settings(lpid)  # 예측 방식 이전에 만든 프로젝트
                else:
                    lpid = ls.create_project(TEMPORAL_PROJECT, label_config(setup.ontology))
                # 라벨러가 여는 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다
                data = {
                    "video": setup.labeling_reader.presign(vkey),
                    "timeseries": setup.labeling_reader.presign(ckey),
                    "session_id": session_id,
                    "assignee": assignee,
                }
                check_stage_uris(ReviewStage.LABELING, [str(v) for v in data.values()], raw_bucket)
                tid = ls.create_task(lpid, data, to_ls_results(labels))
                out.append(_task(f"label_studio:{tid}", ReviewTool.LABEL_STUDIO, tid, session_id,
                                 stream.stream_id, assignee, media_uri, LS_KINDS, now, mode,
                                 assignment_id, labels))  # fmt: skip
    # 마지막 방어선: 일반 라벨러 작업의 매체 URI에 원본 버킷이 없어야 한다
    check_stage_uris(ReviewStage.LABELING, [t.media_uri for t in out], raw_bucket)
    for task in out:
        insert_review_task(conn, task)
    return out


def _task(
    key: str, tool: ReviewTool, tid: int, session_id: str, stream_id: str, assignee: str,
    media_uri: str, kinds: tuple[str, ...], now: datetime,
    mode: ReviewMode = ReviewMode.STANDARD, assignment_id: str | None = None,
    sent: Sequence[LabelRecord] = (),
) -> ReviewTask:  # fmt: skip
    """작업 라벨 단계(LABELING) `ReviewTask`를 만든다. sent: 작업에 실제로 보낸 라벨."""
    return ReviewTask(
        task_key=key, tool=tool, external_id=str(tid), session_id=session_id, stream_id=stream_id,
        stage=ReviewStage.LABELING, assignee=assignee, media_uri=media_uri, label_kinds=kinds,
        created_at=now, mode=mode, assignment_id=assignment_id,
        sent_label_ids=tuple(x.label_id for x in sent),
    )  # fmt: skip
