"""검수 작업: 외부 검수 도구(CVAT, Label Studio)의 작업과 세션·스트림의 대응.

역할
    두 층의 계약을 둔다 (WP6, WP12, ADR 0006·0014).
    - `ReviewAssignment`(배정): 검수 운영 로직(`dlp review plan|assign`)이 정하는 "누가, 어떤
      방식으로, 얼마나 먼저" 검수할지. DB `review_assignments`.
    - `ReviewTask`(작업): 외부 검수 도구에 실제로 만든 작업 하나 (`dlp review create`). 수집
      (`dlp review collect`, 웹훅)이 결과를 라벨 이력으로 옮긴다(reconcile). DB `review_tasks`.

주요 이름
    - 열거형: `ReviewTool`, `ReviewStage`, `ReviewTaskStatus`, `ReviewMode`, `ReviewReason`,
      `AssignmentStatus`.
    - `FlaggedSpan`: 먼저 볼 구간. `InjectedError`: 오류 삽입 기록.
    - `ReviewAssignment`, `ReviewTask`.
    - `InheritanceMechanism`, `DescriptionInheritance`: 설명이 행동 검수를 물려받은 기록 (DB
      `description_inheritance`, 추가만, ADR 0043 3절).

주의
    - 프라이버시(블러) 검수 작업은 원본 영상을 보여 주므로 `config/policies/review.yaml
      reviewers.privacy`의 원본 접근 권한자에게만 배정한다. 작업 라벨 검수의 `media_uri`는 블러본
      (라벨링 버킷)이어야 하며 원본 버킷 URI가 들어가면 안 된다
      (`dlp_review.roles.check_stage_uris`).
    - 블라인드·이중 측정 결과와 오류 삽입 결과는 운영 라벨이 아니다 (`LabelRecord.measurement`,
      `seeded_error`).
    - 시각 필드의 `t_*_ms`는 라벨 시각 규약을 따른다 (시간 구간은 마스터 ms, 공간은 스트림 PTS ms).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from dlp_schema.common import Contract, Identifier, Ms
from dlp_schema.labels import VerificationState


# 외부 검수 도구 종류. 작업 키 접두사(`cvat:42`, `label_studio:7`)로도 쓰인다.
class ReviewTool(StrEnum):
    CVAT = "cvat"
    LABEL_STUDIO = "label_studio"


class ReviewStage(StrEnum):
    """프라이버시 검수는 원본 접근 권한자만, 작업 라벨 검수는 블러본만 본다."""

    PRIVACY = "privacy"
    LABELING = "labeling"
    QA = "qa"


# 검수 작업 상태. open=도구에서 검수 중, collected=결과를 라벨 이력으로 옮김 (collected_at 기록),
# revoked=수집 전에 회수한 종료 상태 (블러가 바뀌어 매체가 낡음, ADR 0039). 회수한 작업은 수집하지
# 않는다. 수집 뒤 회수 요청은 상태를 collected로 두고 `revoke_requested_at`만 남긴다.
class ReviewTaskStatus(StrEnum):
    OPEN = "open"
    COLLECTED = "collected"
    REVOKED = "revoked"


class ReviewMode(StrEnum):
    """검수 방식. 블라인드·이중의 결과는 측정용 레코드, 오류 삽입 과제의 결과는 학습에서 빠진다."""

    STANDARD = "standard"  # 프리라벨을 고친다 (운영 라벨)
    BLIND = "blind"  # 프리라벨 없이 처음부터 (프리라벨 편향 측정)
    SEEDED_ERROR = "seeded_error"  # 정답에 오류를 넣은 사본을 고친다 (검수자 발견율 측정)
    DOUBLE = "double"  # 같은 구간을 다른 라벨러가 한 번 더 (일치도 측정)
    QA = "qa"  # 선임 재검수 (운영 라벨)


# 검수 우선순위 사유 (`FlaggedSpan.reason`). 정책 `config/policies/review.yaml`의
# 우선순위 가중치와 대응한다.
class ReviewReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"  # 모델 신뢰도가 임계값보다 낮음
    MODEL_DISAGREEMENT = "model_disagreement"  # 모델끼리 결과가 다름
    NEW_OBJECT = "new_object"  # 처음 보는(드문) 객체 클래스
    CONTACT_MISMATCH = "contact_mismatch"  # 장갑 센서 접촉과 영상 기반 접촉이 다름
    SAMPLE = "sample"  # 높은 신뢰도 묶음의 표본 검수
    ROUTINE = "routine"  # 사유 없는 일반 검수
    # AI 1차 검수 판정이 지적한 구간 (참고용, 사람 검수를 늘리는 쪽으로만 쓴다. WP17, ADR 0040)
    AI_WRONG = "ai_wrong"  # AI 판정: 틀림
    AI_SUSPECT = "ai_suspect"  # AI 판정: 의심
    AI_UNSURE = "ai_unsure"  # AI 판정: 판단 불가
    AI_MISSING = "ai_missing"  # AI 누락 후보 (v1 기본 끔)


class FlaggedSpan(Contract):
    """검수 단위 안에서 먼저 볼 구간."""

    # reason: 먼저 볼 사유. t_start_ms / t_end_ms: 구간 (ms).
    # label_ids: 이 구간을 플래그하게 만든 라벨 (없으면 빈 튜플).

    reason: ReviewReason
    t_start_ms: Ms
    t_end_ms: Ms
    label_ids: tuple[Identifier, ...] = ()


# 오류 삽입 과제(seeded_error)에서 넣은 오류 하나. 검수자가 이것을 고쳤는지로 발견율을 잰다.
#   error_type: boundary_shift(경계 이동) | class_swap(클래스
#     바꾸기) | blur_deletion(블러 트랙 삭제).
#   original_label_id: 오류를 넣은 원래 라벨.
#   seeded_label_id: 오류를 넣은 사본 라벨 (seeded_error=True).
#     블러 삭제는 사본 없이 원래를 빼므로 None.
#   detail: 오류 세부 (예: 이동량 ms, 바꾼 클래스). 값은 문자열·정수·실수만.
class InjectedError(Contract):
    error_type: Literal["boundary_shift", "class_swap", "blur_deletion"]
    original_label_id: Identifier
    seeded_label_id: Identifier | None = Field(
        default=None, description="오류를 넣은 사본. 블러 삭제는 사본이 없다"
    )
    detail: dict[str, str | int | float] = Field(default_factory=dict[str, str | int | float])


# 배정 상태. open=아직 검수 중(또는 작업 생성 전), done=검수 결과 수집 완료 (completed_at 기록).
class AssignmentStatus(StrEnum):
    OPEN = "open"
    DONE = "done"


class ReviewAssignment(Contract):
    """검수 단위(세션, 스트림, 라벨 종류 묶음)의 배정.

    우선순위·방식·담당자와 오류 삽입 기록을 남긴다.
    """

    # assignment_id: 배정 ID.
    # session_id / stream_id: 검수 대상 세션과 스트림 (시간 구간 라벨만이면 stream_id None 가능).
    # label_kinds: 검수할 라벨 종류 (예: ["action", "gap"]). 1개 이상.
    # mode: 검수 방식 (ReviewMode).
    # priority: 우선순위 점수 (클수록 먼저. `list_assignments`가 이 순으로 정렬).
    # flagged: 먼저 볼 구간 목록.
    # assignee: 담당 검수자 (배정 전이면 None).
    # pair_id / sample_label_ids / withheld_label_ids / only_label_ids: Field description 참고.
    # injected: 오류 삽입 기록 (mode=seeded_error일 때).
    # task_key: 이 배정으로 만든 검수 작업 키 (작업 생성 전이면 None).
    # status / created_at / completed_at: 배정 상태와 시각.
    # DB에서 바꿀 수 있는 필드는 assignee·task_key·status·completed_at뿐이다
    # (`db.repository.update_assignment`).

    assignment_id: Identifier
    session_id: Identifier
    stream_id: Identifier | None = None
    label_kinds: tuple[str, ...] = Field(min_length=1)
    mode: ReviewMode
    priority: float = Field(description="클수록 먼저")
    flagged: tuple[FlaggedSpan, ...] = ()
    assignee: Identifier | None = None
    pair_id: Identifier | None = Field(
        default=None, description="블라인드·이중·QA 배정이 짝을 이루는 표준 배정"
    )
    injected: tuple[InjectedError, ...] = ()
    sample_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="높은 신뢰도 묶음의 표본. 이 배정의 결과로 묶음 합격 여부를 정한다"
    )
    withheld_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="작업에 넣지 않은 묶음의 나머지. 표본이 합격하면 표본 검증이 된다"
    )
    only_label_ids: tuple[Identifier, ...] = Field(
        default=(), description="비어 있지 않으면 작업에 이 라벨만 넣는다 (불합격 묶음 재검수)"
    )
    task_key: Identifier | None = None
    status: AssignmentStatus = AssignmentStatus.OPEN
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None


# 외부 검수 도구에 만든 작업 하나.
#   task_key: `<도구>:<외부 작업 ID>` (예: cvat:42). DB 기본 키.
#   tool / external_id: 도구 종류와 그 도구 안의 작업 ID.
#   session_id / stream_id: 검수 대상.
#   stage: 검수 단계 (privacy면 원본 권한자만).
#   assignee: 담당 검수자.
#   media_uri: 도구에 올린 영상 URI. 작업 라벨 검수는 블러본(라벨링 버킷)이어야 한다.
#   label_kinds: 작업에 넣은 라벨 종류.
#   mode / assignment_id: 검수 방식과 출처 배정 (배정 없이 만든 작업이면 None).
#   sent_label_ids: Field description 참고. None이면 0006 이전 작업이라, 수집
#     (`dlp_review.collect`)이 배정이 있으면 배정의 라벨 선택 함수로, 배정도 없으면 현재 운영
#     라벨로 비교 기준을 다시 고른다.
#   privacy_detection_id: 작업에 보낸 블러의 완료 탐지 세대. 구 작업은 None (ADR 0033).
#   status / created_at / collected_at: 작업 상태와 시각.
#   render_hash: 작업 라벨 단계 매체(워터마크 사본)를 만든 블러본의 렌더 해시
#     (`dlp_privacy.rendering.render_hash`). 블러 검수·구 작업은 None. 지금 운영 블러의 해시와
#     다르면 매체가 낡았다 (ADR 0039).
#   revoke_requested_at / revoke_reason: 회수 요청 시각과 사유 (예: `blur_changed:cvat:42`,
#     `privacy_detect`, `render`, `sweep`, `manual:<행위자>:<사유>`, 같은 매체를 쓰는 작업이면
#     `…:shared-media`). 블러를 바꾼 트랜잭션과 같은 커밋에 남는다.
#     블러 검수 작업은 회수하지 않는다.
#   media_revoked_at: 이 작업 매체 객체가 라벨링 버킷에서 지워졌음을 확인한 시각.
#   tool_revoked_at: 도구 작업이 없음을 확인한 시각 (이미 없던 404 포함).
#   revoke_note: 수동 완료 표시(도구 폐기 등)의 `<행위자>: <사유>`.
#   요청 뒤 media·tool 중 하나라도 비었으면 회수 대기다 (`pending_revocations`).
class ReviewTask(Contract):
    task_key: Identifier  # 도구:작업 ID (예: cvat:42)
    tool: ReviewTool
    external_id: str
    session_id: Identifier
    stream_id: Identifier
    stage: ReviewStage
    assignee: Identifier | None = None
    media_uri: str
    label_kinds: tuple[str, ...]
    mode: ReviewMode = ReviewMode.STANDARD
    assignment_id: Identifier | None = None
    sent_label_ids: tuple[Identifier, ...] | None = Field(
        default=None,
        description="작업에 보낸 라벨. 수집은 이 라벨과 비교한다 (그 사이 라벨이 바뀌어도)",
    )
    privacy_detection_id: int | None = Field(
        default=None, ge=1, description="작업 생성 시 완료된 탐지 세대. 구 작업·일반 검수는 None"
    )
    status: ReviewTaskStatus = ReviewTaskStatus.OPEN
    created_at: AwareDatetime
    collected_at: AwareDatetime | None = None
    render_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description="작업 매체를 만든 블러본 렌더 해시. 구 작업·블러 검수는 None",
    )
    revoke_requested_at: AwareDatetime | None = None
    revoke_reason: str | None = None
    media_revoked_at: AwareDatetime | None = None
    tool_revoked_at: AwareDatetime | None = None
    revoke_note: str | None = None

    @model_validator(mode="after")
    def _check_revocation(self) -> ReviewTask:
        """회수 필드의 상태 규칙 (ADR 0039).

        Raises:
            ValueError: 회수 상태인데 요청·사유가 없거나 수집 시각이 있음, 열린 작업에 요청이 있음,
                요청 없이 완료 표시가 있음, 사유 없는 요청, 블러 검수 작업에 렌더 해시·회수 필드.
        """
        requested = self.revoke_requested_at is not None
        rules = (
            (
                self.status is ReviewTaskStatus.REVOKED
                and (not requested or self.revoke_reason is None or self.collected_at is not None),
                "revoked 작업은 회수 요청 시각·사유가 있고 수집 시각이 없어야 합니다",
            ),
            (
                self.status is ReviewTaskStatus.OPEN and requested,
                "열린 작업에는 회수 요청이 없어야 합니다",
            ),
            (
                not requested
                and (self.media_revoked_at is not None or self.tool_revoked_at is not None),
                "회수 완료 표시에는 회수 요청이 있어야 합니다",
            ),
            (requested and not self.revoke_reason, "회수 요청에는 사유가 있어야 합니다"),
            (
                self.stage is ReviewStage.PRIVACY
                and (
                    self.render_hash is not None
                    or requested
                    or self.revoke_reason is not None
                    or self.revoke_note is not None
                ),
                "블러 검수 작업에는 렌더 해시·회수 필드가 없어야 합니다",
            ),
        )
        for broken, message in rules:
            if broken:
                raise ValueError(message)
        return self


# ---------------------------------------------------------------- 설명의 검수 물려받기 기록


class InheritanceMechanism(StrEnum):
    """설명이 주인의 검수를 물려받은 경로 (`DescriptionInheritance.mechanism`, ADR 0043 3절).

    collect: 검수 수집(`dlp review collect`·웹훅). verify: 완료 판정(`dlp review verify`).
    translate: 번역 단계 뒤(`dlp translate run`). describe: 설명 재생성 뒤(`dlp actions describe`).
    backfill-heuristic: Alembic 0021이 이 기록 전의 물려받음을 예전 추정(검수자·시각 일치)으로
    한 번 메운 것.
    """

    COLLECT = "collect"
    VERIFY = "verify"
    TRANSLATE = "translate"
    DESCRIBE = "describe"
    BACKFILL = "backfill-heuristic"


class DescriptionInheritance(Contract):
    """설명 라벨이 주인(행동·상위 구간)의 검수를 물려받은 기록 하나 (DB description_inheritance).

    추가만 한다 (PostgreSQL 트리거). 물려받는 `record_review`와 같은 트랜잭션에 쓴다. 설명의 지금
    검수 상태(state·reviewer_id·reviewed_at)가 그 설명의 마지막 기록과 같을 때만 "물려받은 검수만
    있음"이다 (`dlp_schema.description_follow.inheritance_only`). 그 뒤 직접 검수로 상태가 바뀌면
    더는 물려받은 것이 아니다. 검수자·시각이 우연히 주인과 같은 직접 검수는 기록이 없으므로
    물려받은 것으로 보지 않는다 (F-PAR-20, ADR 0043 3절).
    """

    event_id: int = Field(ge=1, description="DB가 매기는 일련번호 (같은 설명 안에서 시간 순)")
    session_id: Identifier
    description_label_id: Identifier
    owner_label_ids: tuple[Identifier, ...] = Field(
        min_length=1, description="물려줄 때 현재였던 주인 레코드 (이관 기록은 물려준 주인 하나)"
    )
    giver_label_id: Identifier = Field(description="상태·검수자·시각을 준 주인 레코드")
    state: VerificationState
    reviewer_id: Identifier
    reviewed_at: AwareDatetime
    mechanism: InheritanceMechanism
    rule_version: str = Field(min_length=1, description="설명 따르기 규칙 버전 (FOLLOW_VERSION)")
    recorded_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> DescriptionInheritance:
        """물려받을 수 있는 상태(승인·표본 검증)이고 준 주인이 주인 목록에 있는지.

        Raises:
            ValueError: 그 밖의 상태, 목록에 없는 준 주인.
        """
        if self.state not in (VerificationState.HUMAN_APPROVED, VerificationState.SAMPLE_VERIFIED):
            raise ValueError("물려받는 검수 상태는 human_approved·sample_verified뿐입니다")
        if self.giver_label_id not in self.owner_label_ids:
            raise ValueError("검수를 준 주인은 주인 목록에 있어야 합니다")
        return self
