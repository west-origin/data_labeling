# PyAV 타입 스텁이 일부 반환 타입을 비워 두어 이 모듈에서만 해당 경고를 끈다.
# pyright: reportUnknownMemberType=false

"""세션에서 검수 작업을 만든다.

- 프라이버시 검수 (CVAT, 원본 접근 권한자): 원본 프록시 영상 + 현재 블러 트랙.
- 작업 라벨 검수 (일반 라벨러): 라벨러 워터마크를 입힌 블러본.
  - CVAT: 객체·도구 박스 트랙과 키포인트 트랙
  - Label Studio: 시간 구간 라벨, 영상과 장갑·IMU 시계열 동기 재생
  일반 라벨러 경로의 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
- 블러본은 지금 승인된 블러 라벨로 렌더한 것만 쓴다 (dlp_privacy.runner.assert_render_current).
  작업에는 그 블러본의 렌더 해시(`ReviewTask.render_hash`)를 남긴다. 블러가 바뀌면 이 해시로 낡은
  매체를 찾아 회수한다 (ADR 0039). 회수가 끝나지 않은 스트림에는 새 라벨링 작업을 만들지 않는다.
- CVAT 작업은 담당자의 CVAT 계정(review.yaml cvat.users)에 배정한다. 블러 검수는 연결이 없으면
  만들지 않고, 검수 우선 구간을 CVAT 이슈로 남겨 검수자가 먼저 보게 한다 (ADR 0024).
- Label Studio 시간 검수 작업은 담당자의 Label Studio 계정 연결(review.yaml label_studio.users)이
  있어야 만든다. 수집은 그 계정의 주석만 받는다 (F-REV-06). Label Studio Community에는 작업 배정
  API가 없어(Enterprise 기능) 작업을 그 계정에만 보이게 할 수는 없다.

WP6, ADR 0006·0020·0023·0024. 진입점: `dlp review create <세션>`(직접 만들기)과
`dlp review assign <배정>`(`ops.runner.create_assignment_tasks` 경유).

공개 이름:
- `VIDEO_KINDS`, `PRIVACY_PROJECT`, `SPATIAL_PROJECT`, `TEMPORAL_PROJECT`:
  영상 종류와 도구 프로젝트 이름.
- `TaskError`: 작업을 만들거나 수집할 수 없다 (전제 조건 위반).
- `ReviewSetup`: 저장소·도구 클라이언트·정책 묶음 (정책은 없으면 저장소의 기본 YAML에서 늦게
  읽는다. CLI는 `DLP_REVIEW_POLICY_FILE`로 고른 검수 정책을 넘긴다).
- `frame_times`, `video_size`, `proxy_key`, `raw_video_size`, `label_scale`, `object_key`:
  매체 도우미.
- `current_for_review`, `Selector`, `default_selector`: 작업에 넣을 라벨 고르기.
- `cvat_user_id`: dlp 검수자 → CVAT 사용자 ID.
- `label_studio_user_id`, `require_label_studio_user`: dlp 검수자 → Label Studio 사용자 ID
  (서비스 계정 거부, 뒤쪽은 연결이 없어도 `TaskError`).
- `create_privacy_tasks`: 블러 검수 작업 (스트림별 CVAT).
- `lock_assignment`: 배정 행 잠금 (작업 중복 생성 방지). 회수한 작업은 "작업 없음"으로 본다.
- `create_labeling_tasks`: 작업 라벨 검수 작업 (CVAT 공간 + Label Studio 시간).
- `MEDIA_HASH_CHARS`, `labeling_media_key`: 워터마크 사본 키 (회수 실행기가 같은 값으로 키
  모양을 본다).

저장소 키:
- 원본 버킷: `sessions/<세션>/derived/<스트림>.proxy.mp4`(검수 프록시), 장갑·IMU 원본 Parquet.
- 라벨링 버킷: `sessions/<세션>/review/<담당자>/<스트림>-<렌더 해시 앞 16자>.mp4`
  (워터마크 영상, 내용 주소: 블러본이 바뀌면 키도 바뀐다. 회수 실행기가 지운다),
  `sessions/<세션>/review/timeseries.csv`(시계열, 얼굴이 없어 회수하지 않는다).
  구 작업은 `<담당자>/<스트림>.mp4` 키를 쓴다.
원본 접근은 `setup.raw`가 감사 저장소(`dlp_cli.raw_access.raw_store`)일 때 기록된다 (ADR 0020).

DB 트랜잭션은 호출자가 관리한다. DB 롤백으로 CVAT·Label Studio 작업이나 저장소 사본이 삭제되지는
않는다. 작업 생성(블러 검수·라벨링 모두)은 세션 행을 잠근 상태로 원본 접근·업로드·도구 호출까지
수행한다. 라벨링 작업 생성의 잠금은 회수 요청(같은 세션 잠금 아래에서 남긴다)과 겹치지 않게 한다.
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
    expected_render_hash,
    read_review_segments,
    stale_media_tasks,
)
from dlp_review.clients import CvatClient, LabelStudioClient, cvat_task_name
from dlp_review.cvat import CvatSchema, Scale, label_spec, to_cvat_tracks
from dlp_review.labelstudio import LS_KINDS, label_config, to_ls_results
from dlp_review.ontology_projects import ontology_project
from dlp_review.ops.policy import CvatPolicy, LabelStudioPolicy, MediaPolicy
from dlp_review.ops.policy import load_policy as load_ops_policy
from dlp_review.roles import check_privacy_reviewers, check_stage_uris
from dlp_review.timeseries import write_timeseries_csv
from dlp_review.ui_theme import UiTheme, cvat_label_color, load_ui_theme, logo_data_uri
from dlp_review.watermark import burn_watermark
from dlp_schema import repo_root
from dlp_schema.db.repository import (
    get_assignment,
    get_labels,
    get_session,
    get_session_ontology,
    insert_review_task,
    latest_privacy_detection,
    list_review_tasks,
    lock_session,
    pending_revocations,
)
from dlp_schema.db.tables import review_assignments
from dlp_schema.episode import current_labels
from dlp_schema.labels import LabelRecord
from dlp_schema.ontology import Ontology
from dlp_schema.review import ReviewMode, ReviewStage, ReviewTask, ReviewTaskStatus, ReviewTool
from dlp_schema.session import PrivacyState, Session, Stream, StreamKind
from dlp_sync.policy import load_policy as load_sync_policy
from dlp_sync.signals import Series, glove_series, imu_series

# 검수 작업을 만드는 영상 스트림 종류
VIDEO_KINDS = {StreamKind.BODYCAM, StreamKind.THIRD_PERSON}
# 검수 도구 프로젝트 이름 (없으면 처음 쓸 때 만든다)
PRIVACY_PROJECT = "dlp-privacy"
SPATIAL_PROJECT = "dlp-labeling-spatial"
TEMPORAL_PROJECT = "dlp-labeling-temporal"
# 워터마크 사본 키에 넣는 렌더 해시 앞 글자 수 (내용 주소). 회수 실행기(`dlp_review.revoke`)가 이
# 값으로 "지워도 되는 키" 모양을 만든다. 한 곳에만 두어 두 모듈이 어긋나지 않게 한다
MEDIA_HASH_CHARS = 16


class TaskError(RuntimeError):
    """검수 작업을 만들거나 수집할 수 없다 (승인 전, 계정 연결 없음, 주석 없음 등)."""


@dataclass
class ReviewSetup:
    """작업 만들기·수집에 필요한 저장소, 도구 클라이언트, 정책 묶음.

    정책 필드가 None이면 처음 쓸 때 저장소 루트의 YAML을 읽어 채운다 (그래서 frozen이 아니다).
    이 지연 로더는 기본 파일(`config/policies/review.yaml` 등)만 읽고 `DLP_REVIEW_POLICY_FILE`을
    보지 않는다. CLI(`dlp_cli.review_config.review_setup`)는 고른 검수 정책의 `media`·`cvat`를
    `media`·`cvat_config`로 넘기므로 `dlp review …` 명령에서는 지연 로더가 돌지 않는다. 예외는 매체
    회수 전용 설정(`review_setup(revocation_only=True)`)으로, 회수 실행기가 두 값을 쓰지 않아 정책을
    읽지 않고 None으로 둔다 (그 설정을 작업 생성·수집에 쓰면 안 된다). 라이브러리를
    직접 쓰는 호출자도 별도 정책 파일을 쓰려면 두 필드를 채워서 넘긴다 (F-REV-01).
    Label Studio 계정 연결(`label_studio_config`)도 같다 (F-REV-06).
    """

    # 원본 버킷 저장소 (운영에서는 감사 저장소 `dlp_cli.raw_access.raw_store`)
    raw: ObjectStore
    # 라벨링 버킷 저장소 (블러본·워터마크 영상·시계열 CSV)
    labeling: ObjectStore
    labeling_reader: S3Store  # 라벨러 자격 증명. 서명 URL용
    # 구 호출자 호환용 필드. 라벨 이름·블러 대상·객체 클래스의 출처는 DB 등록 사전이다.
    ontology: Ontology | None  # 구 호출자 호환용. 실제 작업의 사전은 DB 세션 버전으로 읽는다.
    # CVAT 클라이언트 (없으면 CVAT 작업을 만들지 않는다)
    cvat: CvatClient | None = None
    # Label Studio 클라이언트 (없으면 Label Studio 작업을 만들지 않는다)
    label_studio: LabelStudioClient | None = None
    # 검수 정책의 두 절. 없으면 Git의 config/policies/review.yaml을 읽는다
    # (CLI는 고른 정책을 넘긴다. 회수 전용 설정만 None)
    media: MediaPolicy | None = None  # review.yaml media (워터마크·시계열 격자)
    cvat_config: CvatPolicy | None = None  # review.yaml cvat (검수자 → CVAT 계정 연결)
    # review.yaml label_studio (검수자 → Label Studio 계정 연결, F-REV-06)
    label_studio_config: LabelStudioPolicy | None = None
    privacy: PrivacyPolicy | None = None  # 없으면 config/policies/privacy.yaml (블러본 해시)
    glove_prefixes: tuple[str, ...] | None = None  # 없으면 config/policies/sync.yaml glove
    ui: UiTheme | None = None  # 없으면 config/review_ui.yaml (검수 화면 색·단축키·로고)

    def media_policy(self) -> MediaPolicy:
        """검수 화면 매체 정책 (`review.yaml media`).

        `media`가 없으면 처음 부를 때 기본 로더(`dlp_review.ops.policy.load_policy(repo_root())`,
        환경 변수 무관)로 읽어 둔다. CLI는 고른 정책을 넘기므로 이 경로를 쓰지 않는다.
        """
        if self.media is None:
            self.media = load_ops_policy(repo_root()).media
        return self.media

    def cvat_policy(self) -> CvatPolicy:
        """CVAT 계정 연결 정책 (`review.yaml cvat`).

        `cvat_config`가 없으면 처음 부를 때 기본 로더(환경 변수 무관, Git 기본값은 연결이 비어
        있다)로 읽어 둔다. CLI는 `DLP_REVIEW_POLICY_FILE`로 고른 정책의 `cvat`를 넘긴다.
        """
        if self.cvat_config is None:
            self.cvat_config = load_ops_policy(repo_root()).cvat
        return self.cvat_config

    def label_studio_policy(self) -> LabelStudioPolicy:
        """Label Studio 계정 연결 정책 (`review.yaml label_studio`).

        `label_studio_config`가 없으면 처음 부를 때 기본 로더(환경 변수 무관, Git 기본값은 연결이
        비어 있다)로 읽어 둔다. CLI는 `DLP_REVIEW_POLICY_FILE`로 고른 정책의 `label_studio`를
        넘긴다.
        """
        if self.label_studio_config is None:
            self.label_studio_config = load_ops_policy(repo_root()).label_studio
        return self.label_studio_config

    def privacy_policy(self) -> PrivacyPolicy:
        """프라이버시 정책 (`privacy.yaml`). 블러본 렌더 해시 확인에 쓴다."""
        if self.privacy is None:
            self.privacy = load_privacy_policy(repo_root())
        return self.privacy

    def ui_theme(self) -> UiTheme:
        """검수 화면 테마 (`config/review_ui.yaml`). 처음 부를 때 읽어 둔다."""
        if self.ui is None:
            self.ui = load_ui_theme(repo_root())
        return self.ui

    def ls_label_config(self, ontology: Ontology) -> str:
        """테마를 입힌 Label Studio 프로젝트 설정 XML (로고는 저장소 루트의 `brand.logo`)."""
        theme = self.ui_theme()
        return label_config(ontology, theme, logo_uri=logo_data_uri(theme, repo_root()))

    def pressure_prefixes(self) -> tuple[str, ...]:
        """장갑 압력 채널 접두사 (`sync.yaml glove.pressure_prefixes`). 처음 부를 때 읽어 둔다."""
        if self.glove_prefixes is None:
            path = repo_root() / "config" / "policies" / "sync.yaml"
            self.glove_prefixes = load_sync_policy(path).glove.pressure_prefixes
        return self.glove_prefixes


def labeling_media_key(session_id: str, assignee: str, stream_id: str, render_hash: str) -> str:
    """라벨러 워터마크 사본 키 `sessions/<세션>/review/<담당자>/<스트림>-<렌더 해시 앞부분>.mp4`.

    해시 글자 수는 `MEDIA_HASH_CHARS`다. 블러본이 바뀌면 렌더 해시가 바뀌어 키도 바뀐다 (회수할
    사본과 새 사본이 섞이지 않는다, ADR 0039).
    """
    digest = render_hash[:MEDIA_HASH_CHARS]
    return f"sessions/{session_id}/review/{assignee}/{stream_id}-{digest}.mp4"


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


def label_studio_user_id(
    ls: LabelStudioClient, users: Mapping[str, str], reviewer: str
) -> int | None:
    """dlp 검수자 → Label Studio 사용자 숫자 ID (review.yaml label_studio.users의 이메일로 찾는다).

    연결이 없으면 None (호출자가 거부한다). 수집·웹훅은 이 계정이 만들고 마지막으로 고친 주석만
    받는다 (F-REV-06).
    예외: 연결은 있는데 Label Studio에 그 이메일의 사용자가 없거나, 그 사용자가 이 클라이언트의
    서비스 계정이면 `TaskError` (서비스 계정이 올린 주석을 사람 검수로 받지 않는다).
    """
    email = users.get(reviewer)
    if email is None:
        return None
    uid = ls.find_user_id(email)
    if uid is None:
        raise TaskError(f"Label Studio에 사용자 {email}(검수자 {reviewer})가 없습니다")
    if uid == ls.service_user_id():
        raise TaskError(
            f"검수자 {reviewer}의 Label Studio 계정 {email}이 서비스 계정입니다 "
            "(검수자마다 서비스 계정과 다른 계정을 연결하세요)"
        )
    return uid


def require_label_studio_user(ls: LabelStudioClient, setup: ReviewSetup, reviewer: str) -> int:
    """`label_studio_user_id`와 같되 연결이 없으면 `TaskError` (작업 생성·수집 공통 안내)."""
    uid = label_studio_user_id(ls, setup.label_studio_policy().users, reviewer)
    if uid is None:
        raise TaskError(
            f"검수자 {reviewer}의 Label Studio 계정 연결이 없습니다 (검수 정책 label_studio.users: "
            "DLP_REVIEW_POLICY_FILE 또는 config/policies/review.yaml)"
        )
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


def _cvat_project(
    cvat: CvatClient, name: str, labels: list[str], theme: UiTheme
) -> tuple[int, CvatSchema]:
    """이름으로 CVAT 프로젝트를 찾거나 만든다. (프로젝트 ID, 라벨·속성 ID 표).

    새 프로젝트는 라벨마다 테마 색(`ui_theme.cvat_label_color`)을 넣어 만든다. 이미 있는 프로젝트는
    색이 테마와 다른 라벨만 색을 바꾼다 (화면 모양만, 라벨 ID·속성은 그대로).
    주의: 이미 있는 프로젝트의 라벨 목록은 바꾸지 않는다 (온톨로지에 새 클래스가 생겨도 그대로).
    """
    project = cvat.find_project(name)
    if project is None:
        colors = {n: cvat_label_color(theme, n) for n in labels}
        pid = cvat.create_project(name, label_spec(labels, colors))
        return pid, CvatSchema.from_labels(cvat.project_labels(pid))
    pid = int(project["id"])
    existing = cvat.project_labels(pid)
    for lb in existing:
        want = cvat_label_color(theme, lb["name"])
        if str(lb.get("color", "")).lower() != want.lower():
            cvat.set_label_color(int(lb["id"]), want)
    return pid, CvatSchema.from_labels(existing)


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
    # 탐지·수집과 같은 순서로 세션을 먼저 잠가, 보낸 라벨과 완료 세대를 함께 고정한다.
    lock_session(conn, session_id)
    pick = select or default_selector(conn, session_id)
    if setup.cvat is None:
        raise TaskError("CVAT 클라이언트가 없습니다")
    cvat_conf = setup.cvat_policy()
    # 담당자 계정에 배정한다. 원본 열람 범위는 CVAT 프로젝트 역할·관리자 권한에도 좌우된다.
    # 계정 연결이 없으면 원본을 올리기 전에 멈춘다.
    cvat_user = cvat_user_id(setup.cvat, cvat_conf.users, assignee)
    if cvat_user is None:
        raise TaskError(
            f"블러 검수 담당자 {assignee}의 CVAT 계정이 없습니다 "
            "(검수 정책 cvat.users: DLP_REVIEW_POLICY_FILE 또는 config/policies/review.yaml)"
        )
    session = get_session(conn, session_id)
    if session.privacy_state is PrivacyState.PENDING:
        raise TaskError(f"{session_id}: 블러 자동 탐지를 먼저 실행하세요 (dlp privacy detect)")
    ontology = get_session_ontology(conn, session_id)
    pid, schema = _cvat_project(
        setup.cvat,
        ontology_project(PRIVACY_PROJECT, ontology),
        list(ontology.privacy_targets),
        setup.ui_theme(),
    )
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
            run = latest_privacy_detection(conn, session_id, stream.stream_id)
            segments = read_review_segments(setup.raw, session_id, stream.stream_id, work)
            grant(setup.raw, key, assignee)  # 원본을 검수자에게 보여 준다 (올리기 전에 기록)
            tid = setup.cvat.create_task(
                cvat_task_name(session_id, stream.stream_id, "privacy"), pid, video
            )
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
                privacy_detection_id=run.event_id if run else None,
            )  # fmt: skip
            insert_review_task(conn, task)
            out.append(task)
    return out


def lock_assignment(conn: sa.Connection, assignment_id: str) -> list[ReviewTask] | None:
    """배정 행을 잠근다 (SELECT … FOR UPDATE). 이미 작업이 있으면 그 작업들을 돌려준다.

    같은 배정으로 동시에 작업을 만들면 두 번째 호출은 첫 트랜잭션이 끝날 때까지 기다렸다가
    이미 만든 작업을 받는다 (검수 도구에 작업이 두 번 생기지 않는다).
    회수한(revoked) 작업은 빼고 본다. 배정의 작업이 모두 회수됐으면 None이라 호출자가 지금
    블러본으로 다시 만든다 (`ops.runner.create_assignment_tasks`, ADR 0039).

    반환: 작업이 아직 없거나 모두 회수됐으면 None (잠금은 호출자 트랜잭션 끝까지 유지된다).
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
    live = [
        t
        for t in list_review_tasks(conn, a.session_id)
        if t.assignment_id == assignment_id and t.status is not ReviewTaskStatus.REVOKED
    ]
    return live or None


def _check_no_pending_revocation(
    conn: sa.Connection, session_id: str, streams: Sequence[str], policy: PrivacyPolicy
) -> None:
    """대상 스트림에 끝나지 않은 매체 회수나 회수 요청 없는 낡은 작업이 있으면 `TaskError`.

    회수 대기 중인 스트림에 새 사본을 올리지 않는다 (같은 키에 올리면 실행기가 새 사본을 지운다).
    렌더 해시가 지금과 다른데 요청이 없는 작업은 알려지지 않은 경로(렌더 정책 변경 등)의 백스톱이다.
    도구·매체 작업 전에 부른다. 해결: `dlp review revoke <세션>`.
    """
    targets = set(streams)
    pending = [t for t in pending_revocations(conn, session_id) if t.stream_id in targets]
    stale = stale_media_tasks(conn, session_id, policy, streams=targets)
    if pending or stale:
        raise TaskError(
            f"{session_id}: 매체 회수가 끝나지 않은 작업 {len(pending) + len(stale)}개 "
            f"(dlp review revoke {session_id})"
        )


def _check_ls_account(
    setup: ReviewSetup, session: Session, assignee: str, streams: set[str] | None
) -> None:
    """Label Studio 시간 작업을 만들 바디캠 스트림이 있으면 담당자 계정 연결을 확인한다.

    매체를 올리거나 도구 작업을 만들기 전에 부른다. 연결 없이 만든 작업은 누가 제출했는지 확인할
    수 없어 수집할 수 없다 (F-REV-06). 예외: `TaskError` (`require_label_studio_user`).
    """
    if setup.label_studio is None:
        return
    targets = [
        s
        for s in session.streams
        if s.kind is StreamKind.BODYCAM and (streams is None or s.stream_id in streams)
    ]
    if targets:
        require_label_studio_user(setup.label_studio, setup, assignee)


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
    예외: `TaskError`(프라이버시 승인 전, 블러본이 현재 블러 라벨과 다름·무효, Label Studio 작업을
    만들 담당자의 Label Studio 계정 연결이 없음), `RawAccessError`(작업 URI에 원본 버킷이 섞임),
    `clients.ToolError`.
    부작용: 세션 행 잠금, 라벨링 버킷 쓰기(워터마크 영상, 시계열 CSV), 원본 버킷 읽기(장갑·IMU
    Parquet, 감사 기록), 도구 작업 생성, DB 삽입.
    주의: Label Studio 시간 작업은 세션 계약의 단일 바디캠 스트림에 만든다.
    직접 호출로 반복 생성하면 기존 도구 작업을 재사용하지 않으며
    같은 시계열 CSV 키를 다시 쓸 수 있다.
    회수(ADR 0039): 세션을 먼저 잠가 블러 검수 수집·탐지·렌더의 회수 요청과 직렬화한다. 대상
    스트림에 회수 대기(또는 요청 없는 낡은 작업)가 있으면 `TaskError` (도구·매체 작업 전).
    계정(F-REV-06): Label Studio 작업을 만들 바디캠 스트림이 있으면 그다음에 담당자의 Label Studio
    계정 연결을 확인한다 (역시 도구·매체 작업 전).
    """
    # 회수 요청과 같은 세션 잠금. 없으면 수집이 대상을 고른 뒤 이전 렌더로 만든 작업이 커밋될
    # 수 있다 (READ COMMITTED)
    lock_session(conn, session_id)
    pick = select or default_selector(conn, session_id)
    use_cvat = setup.cvat is not None and (tools is None or ReviewTool.CVAT in tools)
    use_ls = setup.label_studio is not None and (tools is None or ReviewTool.LABEL_STUDIO in tools)
    session = get_session(conn, session_id)
    media = setup.media_policy()
    ontology = get_session_ontology(conn, session_id)
    if session.privacy_state is not PrivacyState.APPROVED:
        raise TaskError(f"{session_id}: 프라이버시 승인 전에는 작업 라벨 검수를 만들 수 없습니다")
    videos = [s.stream_id for s in session.streams if s.kind in VIDEO_KINDS]
    _check_no_pending_revocation(
        conn,
        session_id,
        [s for s in videos if streams is None or s in streams],
        setup.privacy_policy(),
    )
    if use_ls:
        _check_ls_account(setup, session, assignee, streams)
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
                # 작업에 남길 렌더 해시 (블러가 바뀌면 이 값이 달라져 낡은 매체로 판정된다)
                rhash = expected_render_hash(
                    conn, session_id, stream.stream_id, setup.privacy_policy()
                )
            except RenderNotCurrentError as exc:
                raise TaskError(str(exc)) from exc
            marked = work / f"{stream.stream_id}-{assignee}.mp4"
            burn_watermark(
                blurred, marked, f"{assignee} {session_id}", opacity=media.watermark_opacity,
                crf=media.watermark_crf, encoder_rate=media.encoder_rate,
            )  # fmt: skip
            # 내용 주소 키: 블러본이 바뀌면 키도 바뀌어 회수 대상 사본과 새 사본이 섞이지 않는다
            vkey = labeling_media_key(session_id, assignee, stream.stream_id, rhash)
            setup.labeling.put_file(vkey, marked, sha256_file(marked))
            media_uri = setup.labeling.uri(vkey)

            if use_cvat:
                assert setup.cvat is not None
                labels = pick(stream.stream_id, ("box_track", "keypoint_track"))
                # 프로젝트 라벨: 온톨로지 객체 클래스 + 키포인트 골격
                # (kp_<골격>, cvat.cvat_label_name)
                names = [*ontology.objects, "kp_hand21", "kp_coco17", "kp_wholebody133"]
                pid, schema = _cvat_project(
                    setup.cvat, ontology_project(SPATIAL_PROJECT, ontology), names, setup.ui_theme()
                )
                cvat_user = cvat_user_id(setup.cvat, setup.cvat_policy().users, assignee)
                # 이름은 회수 실행기가 지우기 전에 대조한다 (`CvatClient.task_owner`)
                tid = setup.cvat.create_task(
                    cvat_task_name(session_id, stream.stream_id, assignee), pid, marked
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
                          assignee, media_uri, spatial, now, mode, assignment_id, labels,
                          render_hash=rhash)
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
                project_name = ontology_project(TEMPORAL_PROJECT, ontology)
                project = ls.find_project(project_name)
                if project:
                    lpid = int(project["id"])
                    ls.ensure_prelabel_settings(lpid)  # 예측 방식 이전에 만든 프로젝트
                    # 이벤트 속성 입력란처럼 설정이 바뀌었으면 기존 프로젝트에도 반영한다
                    ls.ensure_label_config(lpid, setup.ls_label_config(ontology))
                else:
                    lpid = ls.create_project(project_name, setup.ls_label_config(ontology))
                # 라벨러가 여는 URL은 라벨러 자격 증명(라벨링 버킷 읽기 전용)으로 서명한다.
                # session_id·assignee는 회수 실행기가 지우기 전에 대조한다 (`task_owner`)
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
                                 assignment_id, labels, render_hash=rhash))  # fmt: skip
    # 마지막 방어선: 일반 라벨러 작업의 매체 URI에 원본 버킷이 없어야 한다
    check_stage_uris(ReviewStage.LABELING, [t.media_uri for t in out], raw_bucket)
    for task in out:
        insert_review_task(conn, task)
    return out


def _task(
    key: str, tool: ReviewTool, tid: int, session_id: str, stream_id: str, assignee: str,
    media_uri: str, kinds: tuple[str, ...], now: datetime,
    mode: ReviewMode = ReviewMode.STANDARD, assignment_id: str | None = None,
    sent: Sequence[LabelRecord] = (), *, render_hash: str,
) -> ReviewTask:  # fmt: skip
    """작업 라벨 단계(LABELING) `ReviewTask`를 만든다.

    sent: 작업에 실제로 보낸 라벨.
    render_hash: 매체를 만든 블러본의 렌더 해시 (회수 판정, ADR 0039).
    """
    return ReviewTask(
        task_key=key, tool=tool, external_id=str(tid), session_id=session_id, stream_id=stream_id,
        stage=ReviewStage.LABELING, assignee=assignee, media_uri=media_uri, label_kinds=kinds,
        created_at=now, mode=mode, assignment_id=assignment_id,
        sent_label_ids=tuple(x.label_id for x in sent), render_hash=render_hash,
    )  # fmt: skip
