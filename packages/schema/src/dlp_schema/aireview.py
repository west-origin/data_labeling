"""AI 1차 검수(로컬 VLM) 계약: 판정·제안·요청 단위·실행·노출·결과 (WP17, ADR 0040).

역할
    사내 OpenAI 호환 서버의 VLM이 승인된 블러본 프레임만 보고 검수 전 자동 라벨마다 남긴 판정
    (맞음/의심/틀림/판단 불가, 한국어 이유, 온톨로지 안의 수정 제안)의 형식이다. 판정은 라벨이
    아니고 검수 상태도 아니다. `label_records` 밖의 추가 전용 표(ai_review_*)에만 저장하며, 제안은
    사람이 검수 도구에서 명시적으로 적용할 때만 사람 레코드가 된다 (ADR 0040 I3·I4·I6).

시각 규약 (ADR 0019)
    - `AiVerdict.evidence_t_ms`, `AiReviewUnit.t_*_ms`: 시간 묶음(temporal, temporal_long)은 마스터
      타임라인 ms(= 기준 스트림 바디캠 PTS ms), 공간 묶음(spatial)은 그 스트림의 PTS ms다.

주요 이름
    - 열거형: `AiVerdictValue`, `AiProblem`, `AiGroup`, `AiRunEventKind`, `AiArm`, `AiDecision`,
      `AiHumanAction`, `AiApplyResult`.
    - 판정: `AiSuggestion`, `AiVerdict`, `AiFinding`, `AiFrameRef`.
    - 기록(표와 1:1): `AiReviewUnit`, `AiReviewRun`, `AiRunEvent`, `AiExposure`, `AiOutcomeItem`,
      `AiOutcome`. 표는 `db.tables`의 ai_review_* (Alembic 0016, 추가 전용), 저장소는
      `db.aireview_repository`.
    - `AI_REVIEWER_PREFIX`: AI 판정 실행자 접두사. 이 검수자 ID로는 검수 상태를 기록할 수 없다.
    - 순수 함수: `worst`(한 라벨의 여러 판정 중 가장 나쁜 것), `valid_verdicts`(지금 유효한 판정).

주의
    - 클래스 docstring·Field description은 `schemas/ai_*.schema.json`에 들어간다 (`make schemas`).
    - 응답 원문은 저장하지 않는다(sha256만). 블러본은 글자 OCR 블러가 없어 원문에 명찰·번호가 옮겨
      적힐 수 있기 때문이다.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from dlp_schema.common import Confidence, Contract, Identifier, Ms, OntologyId, SemVer
from dlp_schema.labels import LabelPayload, LabelRecord, Source, VerificationState

# sha256 16진수 64자
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
# 판정 단위 ID: "air-" + 해시 24자 (dlp_aireview.plan.unit_id)
UnitId = Annotated[str, StringConstraints(pattern=r"^air-[0-9a-f]{24}$")]
# 판정 버전: "aireview-<정책 해시 12자>+vlm-<모델>"
# (dlp_aireview.policy.AiReviewPolicy.verdict_version)
AiVersion = Annotated[str, StringConstraints(pattern=r"^aireview-[0-9a-f]{12}\+vlm-\S+$")]
# 판정 대상이 될 수 있는 라벨 종류. blur_track은 어떤 경우에도 판정하지 않는다
# (I2: 블러 검수에 AI를 쓰지 않는다).
AiLabelKind = Literal[
    "action", "gap", "hand_state", "event", "segment", "object_state", "box_track", "keypoint_track"
]  # fmt: skip
# 이유 최대 글자 수 (정책 response.reason_max_chars의 상한)
REASON_MAX = 60
# AI 판정 실행자 표시 접두사. 이 접두사의 검수자 ID로는 검수 상태를 기록하지 않는다 (I3: AI 판정은
# 검수가 아니다. `db.repository.record_review`·`insert_labels`가 거부한다).
AI_REVIEWER_PREFIX = "aireview:"


class AiVerdictValue(StrEnum):
    """판정 값. unsure는 맞음이 아니다 (가려짐·화면 밖·블러·프레임 부족)."""

    OK = "ok"  # 맞음
    SUSPECT = "suspect"  # 의심 (사람 확인 필요)
    WRONG = "wrong"  # 틀림 (확신)
    UNSURE = "unsure"  # 판단 불가


# 판정 순위 (클수록 나쁨). worst가 쓴다.
VERDICT_RANK = {
    AiVerdictValue.OK: 0,
    AiVerdictValue.UNSURE: 1,
    AiVerdictValue.SUSPECT: 2,
    AiVerdictValue.WRONG: 3,
}


class AiProblem(StrEnum):
    """판정이 지적한 문제 유형. 라벨 종류별 허용 목록은 dlp_aireview.schema.PROBLEMS_BY_KIND."""

    # 행동
    WRONG_VERB = "wrong_verb"
    WRONG_TARGET = "wrong_target"
    WRONG_TOOL = "wrong_tool"
    WRONG_HAND = "wrong_hand"
    SHOULD_BE_GAP = "should_be_gap"
    BOUNDARY_EARLY = "boundary_early"  # 경계가 이르다 (표시만, 제안하지 않는다)
    BOUNDARY_LATE = "boundary_late"  # 경계가 늦다 (표시만)
    # 사이 구간
    SHOULD_BE_ACTION = "should_be_action"
    WRONG_GAP_TYPE = "wrong_gap_type"
    # 손 상태
    CONTACT_MISMATCH_SENSOR = "contact_mismatch_sensor"
    WRONG_CONTACT_KIND = "wrong_contact_kind"
    WRONG_CONTACT_TARGET = "wrong_contact_target"
    WRONG_GRASP = "wrong_grasp"
    # 이벤트
    FALSE_EVENT = "false_event"
    WRONG_EVENT_TYPE = "wrong_event_type"
    WRONG_SEVERITY = "wrong_severity"
    # 상위 구간
    WRONG_REF = "wrong_ref"
    HIERARCHY_INCONSISTENT = "hierarchy_inconsistent"
    # 객체 상태
    WRONG_STATE_VALUE = "wrong_state_value"
    # 박스·키포인트 트랙
    WRONG_CLASS = "wrong_class"
    FALSE_POSITIVE = "false_positive"
    BOX_MISALIGNED = "box_misaligned"
    IDENTITY_SWITCH = "identity_switch"
    TRACK_DRIFT = "track_drift"
    HAND_SIDE_SWAPPED = "hand_side_swapped"
    KEYPOINT_MISPLACED = "keypoint_misplaced"
    # 누락 발견 (v1 기본 끔)
    MISSING_EVENT = "missing_event"
    MISSING_OBJECT = "missing_object"
    # 공통
    NOT_VISIBLE = "not_visible"
    INSUFFICIENT_FRAMES = "insufficient_frames"
    OTHER = "other"
    # 시스템 전용: 응답에서 빠졌거나 형식이 맞지 않는 항목 (모델 판단이 아니다)
    RESPONSE_INVALID = "response_invalid"


# unsure에 쓸 수 있는 문제 유형 (response_invalid는 시스템만 쓴다)
UNSURE_PROBLEMS = frozenset(
    {
        AiProblem.NOT_VISIBLE,
        AiProblem.INSUFFICIENT_FRAMES,
        AiProblem.OTHER,
        AiProblem.RESPONSE_INVALID,
    }
)


class AiGroup(StrEnum):
    """요청 묶음. 시간 라벨은 기준 스트림(바디캠) 블러본, 공간 라벨은 그 스트림 블러본을 본다."""

    TEMPORAL = "temporal"  # 고정 길이 창 (마스터 ms)
    TEMPORAL_LONG = "temporal_long"  # 긴 라벨 하나 = 한 단위 (상위 구간·긴 객체 상태 등)
    SPATIAL = "spatial"  # 스냅샷 창 (그 스트림 PTS ms)


class AiRunEventKind(StrEnum):
    """실행 이벤트. waived는 required 모드에서 AI 단계를 건너뛰기로 한 결정."""

    STARTED = "started"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"
    WAIVED = "waived"


class AiArm(StrEnum):
    """검수 작업에 판정을 보였는지 (대조군 holdout은 편향 없는 품질 측정에 쓴다)."""

    SHOWN = "shown"
    HOLDOUT = "holdout"
    NO_VERDICTS = "no_verdicts"
    MODE_EXCLUDED = "mode_excluded"  # 블라인드·오류 삽입·QA 작업
    OFF = "off"  # review.yaml ai_review.mode = off


class AiDecision(StrEnum):
    """검수자가 AI 지적에 대해 명시적으로 한 결정."""

    APPLY = "apply"
    DISMISS = "dismiss"
    NONE = "none"


class AiHumanAction(StrEnum):
    """사람 검수 결과 (원 라벨 기준)."""

    APPROVED = "approved"
    CORRECTED = "corrected"
    DELETED = "deleted"
    REPLACED = "replaced"  # 종류 변경 (action↔gap)
    STALE = "stale"  # 보낸 뒤 다른 단계가 바꾼 라벨
    NA = "na"  # 이중 작업 등 사람 결과 비교 대상이 아님


class AiApplyResult(StrEnum):
    """제안 적용 결정의 처리 결과."""

    APPLIED = "applied"
    CONFLICT_EDITED = "conflict_edited"  # 적용했지만 검수자가 구간·값도 고쳐 검수자 결과를 씀
    NO_SUGGESTION = "no_suggestion"
    INVALID_NOW = "invalid_now"  # 지금 온톨로지·세션 검증을 통과하지 못해 적용하지 않음
    NOT_SHOWN = "not_shown"  # 화면에 보이지 않은 판정의 결정 (무시)


class AiSuggestion(Contract):
    """수정 제안. 원 라벨에 적용한 완성 페이로드를 담는다 (온톨로지 검사를 통과한 것)."""

    # op: set(같은 종류의 값 바꾸기) | change_kind(action↔gap) | delete(오탐 삭제).
    # fields: 응답이 바꾸자고 한 키 (예: ("verb",), ("kind", "gap_type")).
    # payload: 적용 결과 페이로드. delete면 None. 경계(시각)는 바꾸지 않는다.

    op: Literal["set", "change_kind", "delete"]
    fields: tuple[str, ...] = Field(min_length=1)
    payload: LabelPayload | None = None

    @model_validator(mode="after")
    def _check(self) -> AiSuggestion:
        """delete는 페이로드 없음, 나머지는 페이로드 필수, change_kind는 action·gap만, 블러 금지.

        Raises:
            ValueError: 위반.
        """
        if self.op == "delete":
            if self.payload is not None:
                raise ValueError("삭제 제안에는 페이로드가 없어야 합니다")
            return self
        if self.payload is None:
            raise ValueError(f"{self.op} 제안에는 페이로드가 필요합니다")
        if self.payload.kind == "blur_track":
            raise ValueError("블러 트랙에는 제안하지 않습니다")
        if self.op == "change_kind" and self.payload.kind not in ("action", "gap"):
            raise ValueError("종류 변경은 action↔gap만 허용합니다")
        return self


class AiVerdict(Contract):
    """자동 라벨 하나에 대한 AI 판정 (참고용, 사람 검수가 아니다)."""

    # label_id / kind: 판정한 라벨과 그 종류 (blur_track 불가).
    # verdict / problems: 판정 값과 문제 유형 (ok이면 문제 없음, 최대 3개).
    # reason_ko: 한국어 이유 (60자 이하, 개인 특징·글자·번호·URL·프레임 번호를 쓰지 않는다).
    #   ok면 고정 문구 "AI: 문제 없음".
    # confidence: 모델이 스스로 보고한 확신도 (보정 전에는 결정에 쓰지 않는다).
    # evidence_t_ms: 근거 프레임 시각 (temporal 마스터 ms, spatial 스트림 PTS ms).
    # answered: 모델이 이 라벨에 형식에 맞게 답했는가.
    #   False면 시스템이 채운 unsure(response_invalid).
    # suggestion / suggestion_error: 검증을 통과한 제안, 또는 버린 제안의 사유.

    label_id: Identifier
    kind: AiLabelKind
    verdict: AiVerdictValue
    problems: tuple[AiProblem, ...] = Field(default=(), max_length=3)
    reason_ko: str = Field(min_length=1, max_length=REASON_MAX)
    confidence: Confidence | None = None
    evidence_t_ms: Ms | None = None
    answered: bool = True
    suggestion: AiSuggestion | None = None
    suggestion_error: str | None = None

    @model_validator(mode="after")
    def _check(self) -> AiVerdict:
        """판정 값별 규칙 (모듈 docstring 참고).

        - ok: 문제·제안 없음. unsure: 문제는 not_visible·insufficient_frames·other·response_invalid
          안이고 제안 없음. suspect·wrong: 문제 1개 이상, response_invalid 불가.
        - answered=False면 unsure + (response_invalid,) 하나.
        - 제안 종류: set은 같은 종류, change_kind는 action↔gap.

        Raises:
            ValueError: 위반.
        """
        v, problems = self.verdict, set(self.problems)
        if len(problems) != len(self.problems):
            raise ValueError("문제 유형이 중복됩니다")
        if v is AiVerdictValue.OK and (problems or self.suggestion is not None):
            raise ValueError("ok 판정에는 문제 유형과 제안이 없어야 합니다")
        if v is AiVerdictValue.UNSURE and (
            not problems <= UNSURE_PROBLEMS or self.suggestion is not None
        ):
            raise ValueError("unsure 판정은 판단 불가 사유만 쓰고 제안이 없어야 합니다")
        if v in (AiVerdictValue.SUSPECT, AiVerdictValue.WRONG) and (
            not problems or AiProblem.RESPONSE_INVALID in problems
        ):
            raise ValueError("suspect·wrong 판정에는 문제 유형이 필요합니다")
        if (AiProblem.RESPONSE_INVALID in problems) != (not self.answered) or (
            not self.answered and self.problems != (AiProblem.RESPONSE_INVALID,)
        ):
            raise ValueError(
                "response_invalid는 답하지 않은 항목(answered=False)에만 하나로 씁니다"
            )
        s = self.suggestion
        if s is not None and s.payload is not None:
            if s.op == "set" and s.payload.kind != self.kind:
                raise ValueError("set 제안은 원 라벨과 같은 종류여야 합니다")
            if s.op == "change_kind" and {s.payload.kind, self.kind} != {"action", "gap"}:
                raise ValueError("종류 변경은 action↔gap 사이만 허용합니다")
        return self


class AiFinding(Contract):
    """누락 후보 (v1 기본 끔). 좌표·라벨을 만들지 않고 표시만 한다."""

    # kind: 빠진 라벨 종류 (event 또는 box_track). stream_id: 공간 묶음이면 그 스트림.
    # t_start_ms / t_end_ms: 후보 구간 (시각 규약은 모듈 docstring).
    # value: 이벤트 유형 또는 객체 클래스 (온톨로지 ID). box: 정규화 상자 (0~1000 정수 x, y, w, h).

    kind: Literal["event", "box_track"]
    stream_id: Identifier | None = None
    t_start_ms: Ms
    t_end_ms: Ms
    value: OntologyId | None = None
    box: tuple[int, int, int, int] | None = None
    reason_ko: str = Field(min_length=1, max_length=REASON_MAX)

    @model_validator(mode="after")
    def _check(self) -> AiFinding:
        """구간 순서와 상자 범위(0~1000). Raises: ValueError."""
        if self.t_start_ms > self.t_end_ms:
            raise ValueError("t_start_ms가 t_end_ms보다 늦습니다")
        if self.box is not None and any(not 0 <= v <= 1000 for v in self.box):
            raise ValueError("상자는 0~1000 정규화 정수여야 합니다")
        return self


class AiFrameRef(Contract):
    """VLM에 보낸 이미지 하나의 감사 기록 (픽셀은 저장하지 않는다)."""

    # i: 단위 이미지 번호 (프레임은 단위 프레임 번호 F#, 확대 조각은 그 뒤 순번. 문맥 초과로 줄여
    #   보낸 요청에서도 같은 번호라 건너뛸 수 있다). 확대 조각은 crop_of에 원 프레임 번호.
    # t_ms: 그 프레임의 PTS ms (블러본 PTS 인덱스). sha256: 보낸 JPEG의 해시. w, h: 보낸 크기 px.

    i: int = Field(ge=0)
    t_ms: Ms
    sha256: Sha256
    w: int = Field(gt=0)
    h: int = Field(gt=0)
    crop_of: int | None = Field(default=None, ge=0, description="확대 조각이면 원 프레임 번호")


class AiReviewUnit(Contract):
    """요청 단위 하나의 판정 기록 (계획 단위 하나 = 행 하나, 재개·감사의 기준)."""

    # unit_id: 입력 전체(버전·세션·묶음·스트림·구간·분할 순번·대상·맥락·프레임·렌더 해시·
    #   센서·온톨로지)의 해시 (dlp_aireview.plan.unit_id). run_id: 처음 쓴 실행.
    # version: 판정 버전. group / stream_id: 묶음과 프레임을 뽑은 블러본 스트림.
    # t_start_ms / t_end_ms: 증거 구간. target_label_ids: 판정 대상 (순서 = 응답 번호 i).
    # context_label_ids: 텍스트로만 준 맥락 라벨 (정렬).
    # render_hash / blurred_sha256: 그 스트림 블러본의 렌더 해시와 파일 sha (이전 형식이면 None).
    # sensor_sha256: 장갑 접촉 구간 해시 (이 단위가 센서 문장을 쓰지 않으면 "absent").
    # frames: 실제로 보낸 이미지 (문맥 초과로 나눈 요청들의 합). prompt_digest: 실제로 보낸 요청의
    #   다이제스트 (정적 접두사·글·이미지 해시의 해시, 재시도 대화가 있으면 그 해시까지. 나눈 단위는
    #   요청별 다이제스트 목록의 해시, dlp_aireview.judge.request_digest).
    # outcome: answered(응답을 받음, 일부 항목이 response_invalid일 수 있음) | fallback(형식에 맞는
    #   응답을 끝내 받지 못해 모든 대상이 unsure).
    # attempts / sub_requests: 모델 호출 수와 문맥 초과로 나눈 뒤 결과를 낸 요청 수.
    # errors: 위반·장애 기록 문장 (번호·위반 종류만, 모델이 쓴 값·서버 응답 본문 없음).
    # verdicts: 대상마다 정확히 하나 (ok 포함). findings: 누락 후보.
    # response_sha256: 받아들인 응답 원문의 해시 (원문은 저장하지 않는다).
    # served_model / endpoint: 서버가 보고한 모델과 엔드포인트 이름 (URL·키 아님).

    unit_id: UnitId
    run_id: Identifier
    session_id: Identifier
    version: AiVersion
    group: AiGroup
    stream_id: Identifier
    t_start_ms: Ms
    t_end_ms: Ms
    target_label_ids: tuple[Identifier, ...] = Field(min_length=1)
    context_label_ids: tuple[Identifier, ...] = ()
    render_hash: Sha256
    blurred_sha256: Sha256 | None = None
    sensor_sha256: Sha256 | Literal["absent"]
    ontology_version: SemVer
    frames: tuple[AiFrameRef, ...] = ()
    prompt_digest: Sha256
    outcome: Literal["answered", "fallback"]
    attempts: int = Field(ge=0)
    sub_requests: int = Field(ge=1)
    errors: tuple[str, ...] = ()
    verdicts: tuple[AiVerdict, ...]
    findings: tuple[AiFinding, ...] = ()
    response_sha256: Sha256 | None = None
    served_model: str | None = None
    endpoint: str | None = Field(default=None, max_length=64)
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> AiReviewUnit:
        """구간 순서, 대상마다 판정 정확히 하나, 대상·맥락 중복 없음, fallback이면 모두 미응답.

        Raises:
            ValueError: 위반.
        """
        if self.t_start_ms > self.t_end_ms:
            raise ValueError("t_start_ms가 t_end_ms보다 늦습니다")
        targets = self.target_label_ids
        if len(set(targets)) != len(targets) or set(targets) & set(self.context_label_ids):
            raise ValueError("대상 라벨이 중복되거나 맥락과 겹칩니다")
        if tuple(v.label_id for v in self.verdicts) != targets:
            raise ValueError("판정은 대상마다 정확히 하나, 대상 순서대로여야 합니다")
        if self.outcome == "fallback" and any(v.answered for v in self.verdicts):
            raise ValueError("fallback 단위의 판정은 모두 미응답(unsure)이어야 합니다")
        return self


class AiBackendRecord(Contract):
    """실행이 쓴 판정 백엔드. 엔드포인트는 이름만 남긴다 (URL·키 아님)."""

    # kind: 백엔드 종류 (지금은 사내 OpenAI 호환 서버뿐). model: 정책 모델.
    # served_model: 서버가 보고한 모델. fingerprint: 엔진·양자화·가중치 revision.
    # external: 외부 서비스 여부 (지금은 늘 False, 외부 백엔드는 별도 ADR).
    # same_model_as_actions: 행동 분류 VLM(actions.yaml vlm.model)과 같은 모델인가 (자기 검수 경고).

    kind: Literal["openai_compatible"]
    model: str
    served_model: str | None = None
    fingerprint: str
    endpoints: tuple[str, ...] = ()
    external: Literal[False] = False
    same_model_as_actions: bool


class AiSelectionRecord(Contract):
    """실행의 단위 선택 방식 (판정 버전에는 들어가지 않는다)."""

    mode: Literal["all", "flagged"]
    reasons: tuple[str, ...] = ()
    audit_ratio: float = Field(ge=0, le=1)


class AiReviewRun(Contract):
    """실행 세대 하나 (계획을 고정한 기록)."""

    # run_id: "air-run-<해시 24자>". version / policy_digest / prompt_version: 판정 버전과 그 구성.
    # render_hashes / blurred_sha256: 스트림 → 계획 시점 렌더 해시·블러본 sha.
    # plan_digest: 계획한 단위 ID 집합의 해시 16자. planned_units: 그 수.
    # requested_by: 실행자 (DLP_ACTOR).

    run_id: Annotated[str, StringConstraints(pattern=r"^air-run-[0-9a-f]{24}$")]
    session_id: Identifier
    version: AiVersion
    policy_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]
    prompt_version: int = Field(ge=1)
    backend: AiBackendRecord
    ontology_version: SemVer
    render_hashes: dict[Identifier, Sha256]
    blurred_sha256: dict[Identifier, Sha256 | None] = Field(default_factory=dict[str, str | None])
    sensor_sha256: Sha256 | Literal["absent"]
    selection: AiSelectionRecord
    plan_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$")]
    planned_units: int = Field(ge=0)
    requested_by: str | None = Field(default=None, max_length=128)
    created_at: AwareDatetime


class AiRunEvent(Contract):
    """실행 이벤트 하나 (시작·완료·중단·건너뜀)."""

    # event_id: DB 자동 증가 (쓰기 전 None). run_id: 실행 (waived처럼 실행이 없으면 None).
    # reason: 예) aborted:config, aborted:render_changed, waived:timeout.
    # counts: 단위 수·판정 분포·토큰 합 등 정수 집계.

    event_id: int | None = None
    session_id: Identifier
    run_id: str | None = None
    event: AiRunEventKind
    reason: str | None = None
    counts: dict[str, int] = Field(default_factory=dict[str, int])
    actor: str | None = Field(default=None, max_length=128)
    at: AwareDatetime


class AiExposure(Contract):
    """검수 작업 하나에 AI 판정을 무엇을 보였는가 (작업 생성 때 고정)."""

    # task_key: 검수 작업. arm / arm_key: 노출 집단과 그 판정에 쓴 루트 키.
    # versions: 허용한 판정 버전. render_hash: 작업 매체의 렌더 해시.
    # coverage: 작업 대상 라벨 중 유효한 판정이 있는 비율.
    # shown: 화면에 보인 판정 {라벨 ID: 단위 ID}. hidden: 대조군·표시 상한으로 숨긴 판정.

    task_key: Identifier
    session_id: Identifier
    assignment_id: Identifier | None = None
    arm: AiArm
    arm_key: str
    versions: tuple[str, ...] = ()
    render_hash: Sha256 | None = None
    coverage: float = Field(ge=0, le=1)
    shown: dict[Identifier, UnitId] = Field(default_factory=dict[str, str])
    hidden: dict[Identifier, UnitId] = Field(default_factory=dict[str, str])
    created_at: AwareDatetime

    @model_validator(mode="after")
    def _check(self) -> AiExposure:
        """보인 판정과 숨긴 판정은 겹치지 않고, shown이 아닌 집단은 보인 판정이 없다.

        Raises:
            ValueError: 위반.
        """
        if set(self.shown) & set(self.hidden):
            raise ValueError("같은 라벨이 shown과 hidden에 모두 있습니다")
        if self.arm is not AiArm.SHOWN and self.shown:
            raise ValueError(f"{self.arm} 작업에는 보인 판정이 없어야 합니다")
        return self


class AiOutcomeItem(Contract):
    """작업 안 라벨 하나의 사람 결과와 AI 판정의 연결."""

    # unit_id / verdict: 그 라벨의 판정 (없으면 None). arm: 작업의 노출 집단.
    # human: 사람 결과. decision / apply_result: 검수자의 명시 결정과 그 처리.
    # match: 사람 결과가 제안과 일치하는 정도 (exact | partial | none | na).
    # result_label_ids: 이 라벨에서 생긴 결과 레코드.

    label_id: Identifier
    unit_id: UnitId | None = None
    verdict: AiVerdictValue | None = None
    arm: AiArm
    human: AiHumanAction
    decision: AiDecision = AiDecision.NONE
    apply_result: AiApplyResult | None = None
    match: Literal["exact", "partial", "none", "na"] = "na"
    result_label_ids: tuple[Identifier, ...] = ()


class AiOutcome(Contract):
    """검수 작업 하나의 수집 결과와 AI 판정의 연결 (작업마다 한 번 쓴다)."""

    task_key: Identifier
    session_id: Identifier
    reviewer_id: Identifier | None = None
    collected_at: AwareDatetime
    items: tuple[AiOutcomeItem, ...] = ()
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------- 순수 함수


def worst(candidates: Iterable[tuple[AiVerdict, AiReviewUnit]]) -> tuple[AiVerdict, AiReviewUnit]:
    """가장 나쁜 판정 (wrong > suspect > unsure > ok). 같으면 최신 단위, 그다음 단위 ID 순.

    Raises:
        ValueError: 후보가 없을 때.
    """
    items = list(candidates)
    if not items:
        raise ValueError("판정 후보가 없습니다")
    return max(items, key=lambda c: (VERDICT_RANK[c[0].verdict], c[1].created_at, c[1].unit_id))


def _is_open_target(label: LabelRecord) -> bool:
    """검수 전 자동 운영 라벨인가 (판정이 아직 의미 있는 라벨)."""
    return (
        label.provenance.source is not Source.HUMAN
        and label.verification.state is VerificationState.UNREVIEWED
        and not label.seeded_error
        and label.measurement is None
        and label.kind != "blur_track"
    )


def valid_verdicts(
    units: Iterable[AiReviewUnit],
    current: Sequence[LabelRecord],
    *,
    versions: Collection[str],
    render_hashes: Mapping[str, str] | None,
    strict_context: bool = False,
) -> dict[str, tuple[AiVerdict, str]]:
    """지금 유효한 판정: 라벨 ID → (가장 나쁜 판정, 그 단위 ID). ADR 0040 4.4.

    조건:
    1. 단위 버전이 `versions` 안 (CLI가 review.yaml ai_review.versions 또는 현재 버전을 넘긴다).
    2. 표시 프로필(`render_hashes`가 있음): 단위 렌더 해시가 지금 그 스트림의 렌더 해시와 같다.
       세션이 승인 상태가 아니면 호출자가 표시 판정을 비운다. 증가 프로필(None)은 렌더를 보지 않는다
       (표본 제외·마무리 보호처럼 사람 검수를 늘리는 쪽은 낡은 블러본의 지적이어도 안전하다).
    3. 대상 라벨이 지금도 `current`(운영 라벨) 안에 있고 검수 전 자동 라벨이다.
    4. `strict_context`면 단위의 대상·맥락 라벨이 모두 지금도 `current` 안에 있다.
    5. 한 라벨에 판정이 여럿이면 `worst`.

    Args:
        units: 판정 단위 (보통 그 세션의 `list_ai_units`).
        current: 지금 운영 라벨 (`current_labels(get_labels(...))`).
        versions: 허용 판정 버전.
        render_hashes: 스트림 → 지금 렌더 해시 (표시 프로필), None이면 증가 프로필.
        strict_context: 맥락 라벨까지 현재여야 하는지 (기본 False).
    """
    by_id = {x.label_id: x for x in current}
    allowed = set(versions)
    found: dict[str, list[tuple[AiVerdict, AiReviewUnit]]] = {}
    for unit in units:
        if unit.version not in allowed:
            continue
        if render_hashes is not None and render_hashes.get(unit.stream_id) != unit.render_hash:
            continue
        if strict_context and any(
            i not in by_id for i in (*unit.target_label_ids, *unit.context_label_ids)
        ):
            continue
        for verdict in unit.verdicts:
            label = by_id.get(verdict.label_id)
            if label is None or not _is_open_target(label):
                continue
            found.setdefault(verdict.label_id, []).append((verdict, unit))
    out: dict[str, tuple[AiVerdict, str]] = {}
    for label_id, candidates in found.items():
        verdict, unit = worst(candidates)
        out[label_id] = (verdict, unit.unit_id)
    return out
