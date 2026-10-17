"""주간 운영 지표 (`dlp ops weekly`, WP16, ADR 0014·0020·0027·0028·0029).

주는 ISO 주(월요일 0시 UTC부터 7일)다. 지표마다 그 주에 일어난 일만 센다.

- 검수 시간: 영상 1시간당 검수 분 (작업 라벨 검수·QA, 블러 검수는 따로). review_work 기록에서.
- 수정률: 그 주에 검수한 라벨 중 고침·지움·추가 비율 (개별 검수만, 블러 제외).
  골든셋 세션(정답을 사람이 처음부터 만들어 모두 "추가"로 보인다)과 사용 중지 세션은 뺀다
  (dlp_active와 같은 기준).
- 자동 승인율: 그 주에 검수한 모델 라벨 중 수정 없이 승인된 비율 (같은 세션 기준).
- 프리라벨 편향: 그 주에 끝난 블라인드 배정의 편향 평균 (ADR 0014).
- 오류 삽입 발견율: 그 주에 끝난 오류 삽입 배정에서 발견한 오류 비율.
- 잔여 블러 누락: 그 주 감사의 영상 1시간당 잔여 누락 수.
- 검증 에피소드: 검증 완료 시각이 그 주인 세션.
  검증 완료 시각 = 생애주기 기록(session_lifecycle_events)에서 human_verified로 처음 옮긴 시각
  (`dlp review verify`가 남긴다, ADR 0028). 한 번 정해지면 바뀌지 않아 지난 주의 수가 그대로다.
  그 기록이 없는 세션(0011 이전에 검증 단계를 지난 세션)만 라벨 이력에서 추정한다
  (verified_at: 운영 라벨 항목마다 처음 검수한 시각 중 가장 늦은 것, ADR 0027).
  블러(프라이버시 검수)는 따로 센다.
  생산원가 = 검수 시간 * 인건비 / 그 수.
- AI 1차 검수 (WP17.8, ADR 0040 7.3, `AiWeekly`): 판정 영상 시간·백로그·판정률, 대조군 기준 지적
  정밀도·재현율·층화 ok 놓침률과 상한, 제안 적용률, fallback율, 캐시 적중률, 장비 시간, 집단별 검수
  시간, 오류 삽입 누출. 그 주에 수집한 작업(결과 기록)·만든 작업(노출)·기록한 단위·끝난 실행으로
  센다.

값이 없는 지표(분모 0)는 `math.nan`이고, JSON(`as_dict`)에서는 null, Markdown에서는 `-`다.

공개 함수:
- `week_range(week)` / `week_of(t)` — ISO 주 문자열 ↔ UTC 구간.
- `work_source(task_key)` — 손으로 기록한 검수 시간의 출처 (작업 키 접두사의 도구, 없으면 manual).
- `verified_time(conn, …)` / `verified_at(history)` — 세션의 검증 완료 시각.
- `weekly_metrics(conn, week, review_policy, policy, ai)` — 한 주의 `WeeklyMetrics` (읽기 전용).
- `ai_weekly(conn, start, end, ai)` — 한 주의 AI 1차 검수 지표 `AiWeekly` (WP17.8, AI 끔이면 None).
- `alerts(history, policy)` — 여러 주 추이에서 경고 문장.
- `markdown(history, warnings, currency)` — 주별 열의 Markdown 표 (AI 지표가 있으면 AI 절).

성능 주의: `weekly_metrics`는 주마다 모든 세션의 라벨 이력을 다시 읽는다 (주 수 * 세션 수).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import sqlalchemy as sa

from dlp_ops.policy import AlertPolicy, OpsPolicy
from dlp_privacy.audit import AuditResult, residual_miss_rate
from dlp_review.ai_display import AiDisplayConfig
from dlp_review.ops.ai_quality import ai_quality
from dlp_review.ops.policy import ReviewOpsPolicy
from dlp_review.ops.runner import quality_report
from dlp_review.ops.seeding import detected
from dlp_schema.aireview import AiRunEventKind, is_full_completion
from dlp_schema.db.aireview_repository import (
    list_ai_run_events,
    list_ai_run_events_between,
    list_ai_runs,
    list_ai_units_created,
)
from dlp_schema.db.repository import (
    get_description_inheritance,
    get_labels,
    get_session,
    get_trajectory_inheritance,
    list_assignments,
    list_golden_sets,
    list_lifecycle_events,
    list_privacy_audits,
    list_review_work,
    list_session_ids,
    withdrawn_session_ids,
)
from dlp_schema.description_follow import active_inheritance
from dlp_schema.episode import current_labels, non_operational_ids
from dlp_schema.history import ReviewChange, review_changes
from dlp_schema.labels import LabelRecord, Source, VerificationState
from dlp_schema.review import AssignmentStatus, ReviewMode, ReviewTool
from dlp_schema.session import LifecycleState, PrivacyState
from dlp_schema.trajectory_follow import active_trajectory_inheritance

# 개별 검수로 보는 상태 (수정률·자동 승인율 계산). 표본 검증은 개별로 본 것이 아니라 뺀다
REVIEWED = (VerificationState.HUMAN_APPROVED, VerificationState.HUMAN_CORRECTED)
# 검증 완료로 보는 모델 라벨 상태 (표본 검증 묶음의 나머지도 검증된 것으로 본다)
VERIFIED_LABEL = (*REVIEWED, VerificationState.SAMPLE_VERIFIED)
# 검증 완료 이후의 생애주기 상태 (옛 세션의 검증 시각을 라벨 이력에서 추정할지 정할 때)
VERIFIED = (LifecycleState.HUMAN_VERIFIED, LifecycleState.SPLIT_ASSIGNED, LifecycleState.EXPORTED)


def work_source(task_key: str | None) -> Literal["label_studio", "cvat", "manual"]:
    """`dlp ops log-work`가 남길 `ReviewWork.source` (검수 작업이 있던 도구).

    회귀: 작업 키가 있으면 무조건 `cvat`으로 기록해, Label Studio 작업(`label_studio:<id>`)의
    검수 시간이 CVAT 시간으로 잘못 집계됐다.

    Args:
        task_key: 검수 작업 키 (`<도구>:<도구 작업 ID>`, 예: `cvat:7`, `label_studio:3`).
            없으면 None.

    Returns:
        작업 키 접두사의 도구(`ReviewTool` 값). 작업 키가 없으면 `manual`.

    Raises:
        ValueError: 접두사가 알려진 도구가 아닐 때 (잘못된 키를 조용히 한 도구로 세지 않는다).
    """
    if task_key is None:
        return "manual"
    tool, sep, _ = task_key.partition(":")
    if sep and tool == ReviewTool.CVAT:
        return "cvat"
    if sep and tool == ReviewTool.LABEL_STUDIO:
        return "label_studio"
    tools = ", ".join(f"{t.value}:<id>" for t in ReviewTool)
    raise ValueError(f"작업 키 {task_key!r}의 도구를 알 수 없습니다 (형식: {tools})")


def week_range(week: str) -> tuple[datetime, datetime]:
    """'2026-W41' → (월요일 0시 UTC, 다음 월요일 0시 UTC).

    반열림 구간 [시작, 끝)이다. 형식이 틀리면 `ValueError`.
    """
    year, w = week.split("-W")
    start = datetime.combine(date.fromisocalendar(int(year), int(w), 1), datetime.min.time(), UTC)
    return start, start + timedelta(days=7)


def week_of(t: datetime) -> str:
    """시각이 속한 ISO 주 문자열(`YYYY-Www`, UTC 기준). 연말·연초는 ISO 연도를 따른다 (예:
    2026-12-31 → 2026-W53)."""
    y, w, _ = t.astimezone(UTC).isocalendar()
    return f"{y}-W{w:02d}"


def verified_time(
    conn: sa.Connection, session_id: str, state: LifecycleState, history: list[LabelRecord]
) -> datetime | None:
    """세션의 검증 완료 시각.

    생애주기 기록에 human_verified로 옮긴 전이(이전 상태가 있는 것)가 있으면 그 첫 시각이다.
    0011 이관의 보충 기록(이전 상태 없음)뿐인 옛 세션은 지금 상태가 검증 이후일 때만 라벨 이력에서
    추정한다 (verified_at).

    인자:
        conn: DB 연결 (`session_lifecycle_events`를 읽는다).
        session_id: 세션 ID.
        state: 세션의 지금 생애주기 상태.
        history: 세션의 전체 라벨 이력 (추정에만 쓴다).
    반환: 시간대 있는 datetime, 아직 검증 전이면 None.
    """
    for event in list_lifecycle_events(conn, session_id):
        if event.to_state is LifecycleState.HUMAN_VERIFIED and event.from_state is not None:
            return event.at
    return verified_at(history) if state in VERIFIED else None


def verified_at(history: list[LabelRecord]) -> datetime | None:
    """세션의 검증 완료 시각 (블러 제외, 운영 라벨만).

    단위는 "항목"(수정 이력 사슬)이다. 항목이 처음 사람 손을 거친 시각 = 사슬(자기와 조상)에서
    가장 이른 검수 시각 (모델 레코드는 VERIFIED_LABEL 상태의 reviewed_at, 사람 레코드는 작성 시각).
    - 완료 시각 T = 현재 운영 라벨과 사람이 지운 항목마다의 "처음 검수 시각" 중 가장 늦은 것.
      단계별로 늦게 생긴 라벨(예: 2주 뒤의 행동 구간)은 자기 검수 시각까지 T를 늦춘다.
    - 이미 검수한 항목을 나중에 고치거나 지우는 QA는 사슬의 처음 검수 시각을 바꾸지 않아 T도
      그대로다.
    - T 이전(같은 시각 포함)에 있던 현재 모델 라벨 중 미검수가 있으면 완료가 아니다 (None).
      T 뒤에 생긴 미검수 모델 라벨(검증 뒤의 새 모델 버전)은 보지 않는다.
    한계: 이 추정은 생애주기 기록(ADR 0028)이 없는 옛 세션에만 쓴다. 전이 시각이 없으므로(ADR 0027)
    검증 뒤에 생긴 라벨을 나중에 검수하면 그 검수 시각으로 T가 늦춰지고, 수명 주기를 검수보다 늦게
    바꾸면 지난 주의 수가 늘 수 있다.
    """
    labels = [x for x in history if x.kind != "blur_track"]
    excluded = non_operational_ids(labels)
    ops = [x for x in labels if x.label_id not in excluded]
    by_id = {x.label_id: x for x in ops}

    def own_review(x: LabelRecord) -> datetime | None:
        """레코드 자체의 검수 시각: 사람 레코드는 작성 시각, 모델 레코드는 검수된 상태일 때
        `reviewed_at`, 아니면 None."""
        if x.provenance.source is Source.HUMAN:
            return x.created_at
        v = x.verification
        return v.reviewed_at if v.state in VERIFIED_LABEL else None

    def first_review(x: LabelRecord) -> datetime | None:
        """사슬(자기 → 부모 → …)에서 가장 이른 검수 시각. 순환 참조는 `seen`으로 끊는다. 없으면
        None."""
        times: list[datetime] = []
        seen: set[str] = set()
        cur: LabelRecord | None = x
        while cur is not None and cur.label_id not in seen:
            seen.add(cur.label_id)
            t = own_review(cur)
            if t is not None:
                times.append(t)
            cur = by_id.get(cur.parent_label_id) if cur.parent_label_id else None
        return min(times) if times else None

    current = current_labels(ops, operational=False)
    pending = [x for x in current if x.provenance.source is Source.MODEL and own_review(x) is None]
    waiting = {x.label_id for x in pending}
    items = [x for x in current if x.label_id not in waiting]
    # 사람이 지운 항목 (모델 버전 교체로 지운 삭제 레코드는 검수가 아니다)
    items += [x for x in ops if x.retracted and x.provenance.source is Source.HUMAN]
    reviewed = [t for t in (first_review(x) for x in items) if t is not None]
    if not reviewed:
        return None
    done = max(reviewed)
    if any(x.created_at <= done for x in pending):
        return None
    return done


def _ratio(a: float, b: float) -> float:
    """`a / b`. 분모가 0이면 `math.nan`(지표 없음)을 돌려준다 (0으로 보지 않는다)."""
    return a / b if b else math.nan


def _clean(value: Any) -> Any:
    """JSON용: NaN은 None, 사전은 안까지."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    return value


@dataclass
class AiWeekly:
    """한 주의 AI 1차 검수 운영 지표 (WP17.8, ADR 0040 7.3). `math.nan`은 자료 없음.

    필드:
        video_hours: 그 주에 판정 실행이 끝난(완료·중단·실패) 세션의 영상 시간 합(시간).
        backlog_hours: 주 끝 시점에 prelabeled이고 승인됐는데 허용 판정 버전의 전체 완료 실행도
            건너뜀 기록도 없는 세션의 영상 시간 합 (사용 중지 제외, 좁힌 실행의 `partial:` 완료만
            있으면 백로그다).
        coverage: 그 주에 만든 작업의 판정률 평균 (제외 집단 빼고).
        flag_precision / flag_recall / ok_miss_rate / ok_miss_upper95: 그 주에 수집한 작업의
            대조군(holdout) 지적 정밀도·재현율, 층화 ok 놓침률과 보수적 95% 상한 (표본 가중,
            `dlp_review.ops.ai_quality`).
        suggestion_accept_rate: 보인 제안 중 실제로 적용된 비율.
        fallback_rate: 그 주에 기록한 판정 단위 중 형식에 맞는 응답을 끝내 받지 못한 단위 비율.
        cache_hit_ratio: 그 주 판정 단위의 접두사 캐시 적중 토큰 / 프롬프트 토큰. 서버가 적중 토큰을
            보고하지 않은 단위(`cached_tokens` None, vLLM `--enable-prompt-tokens-details` 꺼짐)는
            분자·분모에서 모두 뺀다 (모름을 적중 0으로 세지 않는다. bench·estimate·probe와 같다).
            보고한 단위가 없으면 nan(모름, 경고 없음).
        device_seconds_per_video_hour: 그 주에 끝난 실행의 걸린 시간 합(초) / `video_hours`
            (세션을 함께 돌리면 장비 시간보다 크게 나온다: 상한으로 본다).
        review_minutes_per_video_hour_shown / _holdout: 집단별 영상 1시간당 검수 분.
        seeded_leak: 블라인드가 아닌 "AI 없음" 화면 작업 중 오류 삽입 작업 비율 (F-AIR-03).
        counts: 원자료 건수 (`units`, `fallback_units`, `cache_unreported_units`(프롬프트 토큰은
            있는데 캐시 적중을 보고하지 않은 단위), `runs`, `outcome_rows`).
    """

    video_hours: float = math.nan
    backlog_hours: float = math.nan
    coverage: float = math.nan
    flag_precision: float = math.nan
    flag_recall: float = math.nan
    ok_miss_rate: float = math.nan
    ok_miss_upper95: float = math.nan
    suggestion_accept_rate: float = math.nan
    fallback_rate: float = math.nan
    cache_hit_ratio: float = math.nan
    device_seconds_per_video_hour: float = math.nan
    review_minutes_per_video_hour_shown: float = math.nan
    review_minutes_per_video_hour_holdout: float = math.nan
    seeded_leak: float = math.nan
    counts: dict[str, int] = field(default_factory=dict[str, int])


def _ai_backlog_hours(conn: sa.Connection, end: datetime, versions: frozenset[str]) -> float:
    """주 끝(end) 시점의 AI 판정 백로그(영상 시간). 생애주기·실행 기록을 end 앞까지만 본다.

    완료는 허용 버전 실행의 전체 완료(`is_full_completion`: 좁힌 실행의 `partial:` 마감은 아니다,
    required 게이트의 완료 버전과 같은 기준) 또는 건너뛰기(waived)다. 프라이버시 승인 상태는 지금
    값이다 (승인 이력은 따로 남지 않는다).
    """
    withdrawn = withdrawn_session_ids(conn)
    total_ms = 0
    for sid in list_session_ids(conn):
        if sid in withdrawn:
            continue
        events = [e for e in list_lifecycle_events(conn, sid) if e.at < end]
        if not events or events[-1].to_state is not LifecycleState.PRELABELED:
            continue
        session = get_session(conn, sid)
        if session.privacy_state is not PrivacyState.APPROVED:
            continue
        runs = {r.run_id: r.version for r in list_ai_runs(conn, sid)}
        done = any(
            e.at < end
            and (
                e.event is AiRunEventKind.WAIVED
                or (is_full_completion(e) and runs.get(e.run_id or "") in versions)
            )
            for e in list_ai_run_events(conn, sid)
        )
        if not done:
            total_ms += session.duration_ms
    return total_ms / 3_600_000


def ai_weekly(
    conn: sa.Connection, start: datetime, end: datetime, ai: AiDisplayConfig | None
) -> AiWeekly | None:
    """한 주 [start, end)의 AI 지표 (AI 끔이면 None). 읽기 전용.

    ai: `ReviewSetup.ai`와 같은 설정 (허용 판정 버전 → 백로그). CLI가
    `review_config.ai_display_config`로 만든다.
    """
    if ai is None:
        return None
    q = ai_quality(conn, start, end)
    out = AiWeekly()
    holdout = q.agreement.arms.get("holdout")
    if holdout is not None:
        out.flag_precision = holdout.flag_precision.value
        out.flag_recall = holdout.flag_recall.value
    out.ok_miss_rate, out.ok_miss_upper95 = q.agreement.ok_miss_rate, q.agreement.ok_miss_upper95
    if q.agreement.apply is not None:
        out.suggestion_accept_rate = q.agreement.apply.accept_rate.value
    out.coverage = q.exposure.coverage
    out.seeded_leak = q.exposure.seeded_share_no_ai.value
    out.review_minutes_per_video_hour_shown = q.review_minutes.get("shown", math.nan)
    out.review_minutes_per_video_hour_holdout = q.review_minutes.get("holdout", math.nan)
    units = list_ai_units_created(conn, start, end)
    fallback = sum(1 for u in units if u.outcome == "fallback")
    out.fallback_rate = _ratio(fallback, len(units))
    # 캐시 적중을 보고하지 않은 단위(None)는 모름이다: 적중 0으로 분모에 넣지 않는다
    prompted = [u for u in units if u.prompt_tokens]
    reported = [u for u in prompted if u.cached_tokens is not None]
    out.cache_hit_ratio = _ratio(
        sum(u.cached_tokens or 0 for u in reported), sum(u.prompt_tokens or 0 for u in reported)
    )
    ends = (AiRunEventKind.COMPLETED, AiRunEventKind.INTERRUPTED, AiRunEventKind.ABORTED)
    finished = [e for e in list_ai_run_events_between(conn, start, end) if e.event in ends]
    sessions = {e.session_id for e in finished}
    out.video_hours = sum(get_session(conn, sid).duration_ms for sid in sessions) / 3_600_000
    seconds = sum(e.counts.get("elapsed_s", 0) for e in finished)
    out.device_seconds_per_video_hour = _ratio(seconds, out.video_hours)
    out.backlog_hours = _ai_backlog_hours(conn, end, frozenset(ai.versions))
    out.counts = {
        "units": len(units), "fallback_units": fallback,
        "cache_unreported_units": len(prompted) - len(reported), "runs": len(finished),
        "outcome_rows": len(q.rows),
    }  # fmt: skip
    return out


@dataclass
class WeeklyMetrics:
    """한 주의 운영 지표. `math.nan`은 "그 주에 계산할 자료가 없음"이다.

    필드:
        week: ISO 주 `YYYY-Www`.
        review_minutes_per_video_hour: 작업 라벨·QA 검수 분 / 검수한 영상 시간(시간).
        privacy_review_minutes_per_video_hour: 블러 검수 분 / 영상 시간.
        correction_rate: (고침+지움+추가) / (승인+고침+지움+추가). 블러·골든셋·사용 중지 세션
            제외.
        auto_approval_rate: 승인 / (승인+고침+지움). 사람이 추가한 것은 분모에 넣지 않는다.
        prelabel_bias: 그 주에 끝난 블라인드 배정 편향의 평균 (양수 = 프리라벨에 끌려감).
        seeded_detection_rate: 발견한 삽입 오류 / 삽입한 오류 (그 주에 끝난 오류 삽입 배정).
        residual_blur_miss_per_hour: 그 주 감사의 누락 수 / 감사한 영상 시간.
        verified_episodes: 검증 완료 시각이 그 주인 세션 수 (골든셋 포함, 사용 중지 제외).
        review_hours: 그 주 모든 단계(블러 포함)의 검수 시간 합(시간).
        cost_per_episode: `review_hours` * `hourly_cost` / `verified_episodes` (인건비 미정이면
            nan).
        counts: 원자료 건수 (`accepted`, `corrected`, `deleted`, `added`, `seeded_errors`,
            `seeded_found`, `privacy_audits`).
        ai: AI 1차 검수 지표 (`ai_weekly`, AI 끔이면 None).
    """

    week: str
    review_minutes_per_video_hour: float = math.nan
    privacy_review_minutes_per_video_hour: float = math.nan
    correction_rate: float = math.nan
    auto_approval_rate: float = math.nan
    prelabel_bias: float = math.nan
    seeded_detection_rate: float = math.nan
    residual_blur_miss_per_hour: float = math.nan
    verified_episodes: int = 0
    review_hours: float = 0.0
    cost_per_episode: float = math.nan
    counts: dict[str, int] = field(default_factory=dict[str, int])
    ai: AiWeekly | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON용 사전. `nan`은 `None`(null)으로 바꾼다 (JSON 표준에 NaN이 없다, AI 절 안까지)."""
        return {k: _clean(v) for k, v in asdict(self).items()}


def _reviewer_changes(
    conn: sa.Connection, sid: str, history: list[LabelRecord]
) -> list[ReviewChange]:
    """검수자가 직접 낸 검수 변화 (수정률·자동 승인율 분자·분모).

    블러 트랙은 따로 본다. 검수를 물려받기만 한 라벨(행동 검수를 물려받은 설명 ADR 0043 3절,
    2D 손 트랙 검수를 물려받은 3D 궤적 ADR 0048)은 검수자가 본 것이 아니므로 뺀다. 행동·손 트랙
    하나 승인에 물려받은 라벨 여러 개가 "그대로 승인"으로 세어져 자동 승인율이 부풀지 않게 한다
    (F-COE-23). 물려받음은 명시 기록으로만 판정한다 (지금 검수 상태가 마지막 기록과 같을 때).
    """
    descriptions = get_description_inheritance(conn, sid)
    trajectories = get_trajectory_inheritance(conn, sid)
    return [
        c
        for c in review_changes(history, REVIEWED)
        if c.label.kind != "blur_track"
        and active_inheritance(c.label, descriptions) is None
        and active_trajectory_inheritance(c.label, trajectories) is None
    ]


def weekly_metrics(  # noqa: C901
    conn: sa.Connection,
    week: str,
    review_policy: ReviewOpsPolicy,
    policy: OpsPolicy,
    ai: AiDisplayConfig | None = None,
) -> WeeklyMetrics:
    """한 주의 운영 지표를 DB에서 계산한다. 읽기 전용.

    인자:
        conn: DB 연결.
        week: ISO 주 `YYYY-Www`.
        review_policy: `review.yaml` 운영 정책 (오류 삽입 발견 허용 오차
            `seeding.detect_tolerance_ms`,
            `seeding.blur_overlap`, 블라인드 편향 계산).
        policy: `ops.yaml` (인건비).
        ai: AI 1차 검수 판정 사용 설정 (`review_config.ai_display_config`, None이면 AI 절 없음).

    읽는 테이블: `review_work`, `sessions`, `label_records`, `golden_sets`, `withdrawals`,
    `review_assignments`, `privacy_audits`, `session_lifecycle_events`, (AI) `ai_review_*`.
    """
    start, end = week_range(week)
    m = WeeklyMetrics(week)

    def inside(t: datetime | None) -> bool:
        """시각이 이 주 [start, end) 안인지. None이면 거짓."""
        return t is not None and start <= t < end

    # 검수 시간
    work = list_review_work(conn, start, end)
    labeling = [w for w in work if w.stage != "privacy"]
    privacy = [w for w in work if w.stage == "privacy"]
    m.review_minutes_per_video_hour = _ratio(
        sum(w.seconds for w in labeling) / 60, sum(w.video_ms for w in labeling) / 3_600_000
    )
    m.privacy_review_minutes_per_video_hour = _ratio(
        sum(w.seconds for w in privacy) / 60, sum(w.video_ms for w in privacy) / 3_600_000
    )
    m.review_hours = sum(w.seconds for w in work) / 3600

    # 수정률·자동 승인율 (골든셋·사용 중지 세션 제외), 검증 에피소드
    accepted = corrected = deleted = added = 0
    withdrawn = withdrawn_session_ids(conn)
    golden = {sid for g in list_golden_sets(conn) for sid in g.session_ids}
    for sid in list_session_ids(conn):
        if sid in withdrawn:
            continue
        history = get_labels(conn, sid)
        session = get_session(conn, sid)
        if inside(verified_time(conn, sid, session.lifecycle_state, history)):
            m.verified_episodes += 1
        if sid in golden:
            continue
        for c in _reviewer_changes(conn, sid, history):
            if not inside(c.at):
                continue
            if c.change == "accepted":
                accepted += 1
            elif c.change == "corrected":
                corrected += 1
            elif c.change == "deleted":
                deleted += 1
            else:
                added += 1
    m.counts = {"accepted": accepted, "corrected": corrected, "deleted": deleted, "added": added}
    m.correction_rate = _ratio(corrected + deleted + added, accepted + corrected + deleted + added)
    m.auto_approval_rate = _ratio(accepted, accepted + corrected + deleted)

    # 블라인드 편향·오류 삽입 발견율 (그 주에 끝난 배정)
    done = [
        a for a in list_assignments(conn, status=AssignmentStatus.DONE) if inside(a.completed_at)
    ]
    total = found = 0
    for a in done:
        if a.mode is ReviewMode.SEEDED_ERROR:
            labels = get_labels(conn, a.session_id)
            for err in a.injected:
                total += 1
                found += detected(
                    err, labels, a.assignee, review_policy.seeding.detect_tolerance_ms,
                    review_policy.seeding.blur_overlap,
                )  # fmt: skip
    m.seeded_detection_rate = _ratio(found, total)
    m.counts |= {"seeded_errors": total, "seeded_found": found}
    blind = {a.assignment_id for a in done if a.mode is ReviewMode.BLIND}
    if blind:
        biases = [
            v
            for k, v in quality_report(conn, review_policy).blind_bias.items()
            if k in blind and not math.isnan(v)
        ]
        m.prelabel_bias = sum(biases) / len(biases) if biases else math.nan

    # 잔여 블러 누락
    audits = list_privacy_audits(conn, start, end)
    m.counts["privacy_audits"] = len(audits)
    if audits:
        m.residual_blur_miss_per_hour = residual_miss_rate(
            [
                AuditResult(
                    a.session_id, a.stream_id, a.duration_ms, a.misses, a.auditor, a.blur_reviewer
                )
                for a in audits
            ]
        )

    if policy.cost.hourly_cost is not None and m.verified_episodes:
        m.cost_per_episode = m.review_hours * policy.cost.hourly_cost / m.verified_episodes
    m.ai = ai_weekly(conn, start, end, ai)  # 분기 없이 합친다 (AI 끔이면 None)
    return m


def _ai_alerts(cur: WeeklyMetrics, ap: AlertPolicy) -> list[str]:
    """AI 1차 검수 경고 (마지막 주, `ops.yaml alerts.ai_*`). nan이면 내지 않는다."""
    ai = cur.ai
    if ai is None:
        return []
    out: list[str] = []
    if ai.ok_miss_upper95 > ap.ai_ok_miss_upper95_max:
        out.append(
            f"{cur.week}: AI ok 놓침률 95% 상한(대조군) {ai.ok_miss_upper95:.1%}이 기준 "
            f"{ap.ai_ok_miss_upper95_max:.1%}보다 높다 — AI의 맞음을 믿을 근거가 없다 (줄이는 데 "
            "쓰지 않는다, 판정 모델·프롬프트 점검)"
        )
    if ai.backlog_hours > ap.ai_backlog_hours_max:
        out.append(
            f"{cur.week}: AI 판정 백로그 {ai.backlog_hours:.0f}시간이 기준 "
            f"{ap.ai_backlog_hours_max:g}시간보다 많다 (장비·dlp aireview run --pending 확인)"
        )
    if ai.fallback_rate > ap.ai_fallback_rate_max:
        out.append(
            f"{cur.week}: AI 응답 형식 실패(fallback) {ai.fallback_rate:.1%}가 기준 "
            f"{ap.ai_fallback_rate_max:.1%}보다 높다 (dlp aireview probe로 구조화 출력 확인)"
        )
    if ai.cache_hit_ratio < ap.ai_cache_hit_min:
        out.append(
            f"{cur.week}: AI 접두사 캐시 적중 {ai.cache_hit_ratio:.1%}가 기준 "
            f"{ap.ai_cache_hit_min:.1%}보다 낮다 (서버 접두사 캐시 설정 확인)"
        )
    return out


def alerts(history: list[WeeklyMetrics], policy: OpsPolicy) -> list[str]:
    """주간 지표 추이에서 경고 (오래된 주부터 정렬된 목록).

    규칙:
    1. 마지막 주에 블러 검수나 검증 에피소드가 있었는데 잔여 누락 감사가 0건이면 경고.
    2. 직전 주 대비 자동 승인율이 `auto_approval_rise` 이상 오르고 오류 삽입 발견율이
       `detection_drop` 이상 떨어지면 "검수 품질 저하" 경고.
    3. 최근 `stagnation_weeks`주의 처음과 끝을 비교해 검수 시간과 수정률이 둘 다 줄지 않았으면
       "가이드라인·온톨로지 점검" 경고.
    4. AI 1차 검수(마지막 주, `ai_*`): ok 놓침률 95% 상한 초과, 백로그 시간 초과, fallback율 초과,
       캐시 적중률 미달 (`_ai_alerts`).
    `nan`과의 비교는 항상 거짓이라 자료가 없는 주는 2·3·4번 경고를 내지 않는다.
    반환: 경고 문장 목록 (없으면 빈 목록).
    """
    out: list[str] = []
    if history:
        cur = history[-1]
        privacy_work = not math.isnan(cur.privacy_review_minutes_per_video_hour)
        if cur.counts.get("privacy_audits", 0) == 0 and (privacy_work or cur.verified_episodes):
            out.append(
                f"{cur.week}: 블러 잔여 누락 감사가 없다 — 감사 없는 주는 통과로 보지 않는다 "
                "(dlp privacy audit-sample로 표본을 뽑아 감사한다)"
            )
    ap = policy.alerts
    if len(history) >= 2:
        prev, cur = history[-2], history[-1]
        rise = cur.auto_approval_rate - prev.auto_approval_rate
        drop = prev.seeded_detection_rate - cur.seeded_detection_rate
        if rise >= ap.auto_approval_rise and drop >= ap.detection_drop:
            out.append(
                f"{cur.week}: 자동 승인율이 {rise:+.1%} 올랐는데 오류 삽입 발견율이 {-drop:+.1%} "
                "떨어졌다 — 모델 개선이 아니라 검수 품질 저하로 본다"
            )
    if history:
        out += _ai_alerts(history[-1], ap)
    n = ap.stagnation_weeks
    if len(history) >= n:
        first, last = history[-n], history[-1]
        if (
            last.review_minutes_per_video_hour >= first.review_minutes_per_video_hour
            and last.correction_rate >= first.correction_rate
        ):
            out.append(
                f"{first.week}~{last.week}: 검수 시간과 수정률이 {n}주 동안 줄지 않았다 — "
                "모델보다 가이드라인·온톨로지를 점검한다"
            )
    return out


def markdown(history: list[WeeklyMetrics], warnings: list[str], currency: str) -> str:
    """주간 지표 목록 → 주를 열로 둔 Markdown 표 (+ 경고 절). `nan`은 `-`로 표시한다."""
    rows = [
        ("검수 분 / 영상 1시간 (작업 라벨)", "review_minutes_per_video_hour", "{:.1f}"),
        ("검수 분 / 영상 1시간 (블러)", "privacy_review_minutes_per_video_hour", "{:.1f}"),
        ("수정률", "correction_rate", "{:.1%}"),
        ("자동 승인율", "auto_approval_rate", "{:.1%}"),
        ("프리라벨 편향 (블라인드)", "prelabel_bias", "{:.3f}"),
        ("오류 삽입 발견율", "seeded_detection_rate", "{:.1%}"),
        ("잔여 블러 누락 / 영상 1시간", "residual_blur_miss_per_hour", "{:.2f}"),
        ("검증 에피소드", "verified_episodes", "{}"),
        (f"에피소드당 생산원가 ({currency})", "cost_per_episode", "{:,.0f}"),
    ]
    lines = ["# 주간 운영 지표", "", "| 지표 | " + " | ".join(m.week for m in history) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in history) + " |")
    for name, key, fmt in rows:
        cells: list[str] = []
        for m in history:
            v = getattr(m, key)
            cells.append("-" if isinstance(v, float) and math.isnan(v) else fmt.format(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += _ai_markdown(history)
    if warnings:
        lines += ["", "## 경고", ""] + [f"- {w}" for w in warnings]
    return "\n".join(lines) + "\n"


# AI 1차 검수 절의 행 (이름, AiWeekly 필드, 형식)
_AI_ROWS = (
    ("판정한 영상 시간", "video_hours", "{:.1f}"),
    ("판정 백로그 (영상 시간)", "backlog_hours", "{:.1f}"),
    ("판정률", "coverage", "{:.1%}"),
    ("지적 정밀도 (대조군)", "flag_precision", "{:.1%}"),
    ("지적 재현율 (대조군)", "flag_recall", "{:.1%}"),
    ("ok 놓침률 (대조군, 층화)", "ok_miss_rate", "{:.1%}"),
    ("ok 놓침률 95% 상한", "ok_miss_upper95", "{:.1%}"),
    ("제안 적용률", "suggestion_accept_rate", "{:.1%}"),
    ("fallback율", "fallback_rate", "{:.1%}"),
    ("캐시 적중률", "cache_hit_ratio", "{:.1%}"),
    ("장비 초 / 영상 1시간", "device_seconds_per_video_hour", "{:.0f}"),
    ("검수 분 / 영상 1시간 (AI 보임)", "review_minutes_per_video_hour_shown", "{:.1f}"),
    ("검수 분 / 영상 1시간 (대조군)", "review_minutes_per_video_hour_holdout", "{:.1f}"),
    ("오류 삽입 누출 (AI 없음 화면)", "seeded_leak", "{:.1%}"),
)


def _ai_markdown(history: list[WeeklyMetrics]) -> list[str]:
    """AI 1차 검수 절 (어느 주에도 AI 지표가 없으면 빈 목록)."""
    if all(m.ai is None for m in history):
        return []
    lines = ["", "## AI 1차 검수 (참고용 판정, 대조군 기준)", ""]
    lines.append("| 지표 | " + " | ".join(m.week for m in history) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in history) + " |")
    for name, key, fmt in _AI_ROWS:
        cells: list[str] = []
        for m in history:
            v = getattr(m.ai, key) if m.ai is not None else math.nan
            cells.append("-" if isinstance(v, float) and math.isnan(v) else fmt.format(v))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines
