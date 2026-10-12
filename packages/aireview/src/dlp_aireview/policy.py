"""AI 1차 검수 정책 (`config/policies/aireview.yaml`, WP17.2, ADR 0040).

- `AiReviewPolicy`: 최상위. `digest`(파싱값 해시 12자)와 `verdict_version`
  (`aireview-<digest>+vlm-<model>`)이 판정 단위 키와 기록에 들어간다. `selection`·`kinds`·
  `pending`·`server`·`gate`·`estimate`는 "어느 단위를 묻나"나 "어떻게 돌리나"만 바꾸므로 digest에서
  뺀다 (대상 집합은 단위 키가 담는다). 그래서 이 절들은 요청 본문을 바꾸면 안 된다: 프롬프트 고정
  블록·응답 스키마·맥락은 `kinds`가 아니라 판정할 수 있는 종류 전체(`TEMPORAL_KINDS`,
  `SPATIAL_KINDS`) 기준이고, 결과를 바꾸는 문맥 초과 분할 횟수는
  `response.max_context_splits`(해시)에 둔다. 프롬프트 틀 문장은 정책 값이 아니라
  `PROMPT_VERSION`으로 버전에 들어간다 (틀을 바꾸면 올린다, `packages/aireview/tests/golden/`의
  골든 파일이 틀 바이트를 고정한다).
- 절별 모델: `BackendPolicy`, `KindsPolicy`, `SelectionPolicy`, `PendingPolicy`, `TemporalPolicy`,
  `SpatialPolicy`, `CropPolicy`, `SensorPolicy`, `ImagePolicy`, `ResponsePolicy`, `SuggestPolicy`,
  `EstimatePolicy`, `GatePolicy`, `ServerPolicy`.
- `load_policy`: 저장소 루트 → `AiReviewPolicy`.

`Contract`라 알 수 없는 키는 적재 오류다. 결정으로 고정한 값은 검증기로 막는다: 블러 트랙 판정 금지
(I2), 장갑 센서 출처 판정 제외, 외부 백엔드 금지(`server.allow_external`), 행동·사이 구간·손 상태
삭제 제안 금지(손별 타임라인 공백 0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from dlp_aireview.perturb import KINDS as PERTURB_KINDS
from dlp_schema.aireview import REASON_MAX, AiTargetRule
from dlp_schema.common import Contract
from dlp_schema.review import ReviewReason

# 프롬프트 틀 버전. dlp_aireview.prompt·schema의 고정 문장·스키마 모양을 바꾸면 올리고 aireview.yaml
# prompt_version도 같이 올린다 (값이 digest에 들어가 판정 버전이 바뀐다). 틀 바이트는
# packages/aireview/tests/golden/의 골든 파일이 고정한다 (바뀌면 테스트가 이 절차를 알린다).
PROMPT_VERSION = 1
# 판정할 수 있는 종류 (검수 도구 화면이 있는 종류). 블러 트랙은 넣을 수 없다 (I2).
TEMPORAL_KINDS = ("action", "gap", "hand_state", "event", "segment", "object_state")
SPATIAL_KINDS = ("box_track", "keypoint_track")
# 삭제 제안을 받을 수 있는 종류 (행동·사이 구간·손 상태는 손별 타임라인 공백 0이라 삭제 불가)
DELETABLE_KINDS = ("event", "box_track", "keypoint_track")
# digest에서 빼는 절 (요청 본문·결과를 바꾸지 않는 절만)
_NOT_HASHED = {"selection", "kinds", "pending", "server", "gate", "estimate"}
# flagged 선택에 쓸 수 없는 사유: AI 판정 자신이 만든 사유 (판정이 다음 판정의 대상을 고르는 순환)
_AI_REASONS = frozenset(
    {
        ReviewReason.AI_WRONG,
        ReviewReason.AI_SUSPECT,
        ReviewReason.AI_UNSURE,
        ReviewReason.AI_MISSING,
    }
)


class BackendPolicy(Contract):
    """`backend`: 판정 모델과 요청 형식 (모두 digest에 들어간다)."""

    # 지금 허용하는 유일한 백엔드 (사내 OpenAI 호환 서버). 외부 백엔드는 별도 ADR + 코드 변경.
    kind: Literal["openai_compatible"]
    # 정책 모델 (config/models.yaml external 이름, make licenses가 검사)
    model: str = Field(min_length=1)
    # 서버에 등록된 모델 이름이 model과 다르면 그 이름 (vLLM --served-model-name 등). 있으면 요청
    # 본문 model과 응답 model 대조 모두 이 이름을 쓴다 (없으면 model)
    served_name: str | None = None
    # 엔진·양자화·가중치 revision. 다르면 버전이 달라진다
    fingerprint: str = Field(min_length=1)
    # 프레임 전달 방식 (video 입력은 WP17.3 bench에서 비교한 뒤 추가)
    frames_as: Literal["images"]
    # server: response_format json_schema로 서버가 강제 / prompt: 스키마를 프롬프트에 싣는다
    schema_enforcement: Literal["server", "prompt"]
    # portable: 서버마다 지원이 같은 키워드만 / full: 숫자 범위·문자열 길이까지 (probe 확인 뒤)
    schema_profile: Literal["portable", "full"]
    # 생성 토큰 상한
    max_tokens: int = Field(ge=64)


class KindsPolicy(Contract):
    """`kinds`: 판정할 라벨 종류와 출처 (digest 제외).

    대상만 고른다. 프롬프트·응답 스키마·맥락은 이 값과 무관하다 (모듈 docstring).
    """

    temporal: tuple[str, ...]
    spatial: tuple[str, ...]
    # 판정 대상 출처. 사람 라벨은 판정하지 않는다. 장갑 센서에서 나온 라벨(센서 출처, 그리고 자동
    # 손 상태 중 장갑 근거 OBSERVED)은 이 값과 별개로 늘 뺀다
    # (결정, `dlp_schema.aireview.glove_derived`, ADR 0040)
    sources: tuple[Literal["model"], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> KindsPolicy:
        """검수 화면이 있는 종류만, 블러 트랙 금지. Raises: ValueError."""
        if "blur_track" in (*self.temporal, *self.spatial):
            raise ValueError("블러 트랙은 AI로 판정하지 않습니다 (I2)")
        extra = (set(self.temporal) - set(TEMPORAL_KINDS)) | (
            set(self.spatial) - set(SPATIAL_KINDS)
        )
        if extra:
            raise ValueError(f"판정할 수 없는 라벨 종류: {sorted(extra)}")
        return self


class SelectionPolicy(Contract):
    """`selection`: 어느 단위를 묻나 (digest 제외)."""

    mode: Literal["all", "flagged"]
    # flagged에서 고르는 우선순위 사유 (`ReviewReason` 값, 모르는 값은 로드 오류). AI 판정 사유
    # (ai_*)는 쓸 수 없다 (판정이 다음 판정의 대상을 고르는 순환)
    reasons: tuple[ReviewReason, ...] = ()
    # flagged: 미상 사이 구간이 든 단위도 고른다
    unknown_gaps: bool = True
    # flagged: 지적 없는 단위 중 감사 표본 비율
    audit_ratio: float = Field(ge=0, le=1)
    seed: int = 0

    @model_validator(mode="after")
    def _check(self) -> SelectionPolicy:
        """AI 판정 사유는 선택 사유가 될 수 없다. Raises: ValueError."""
        bad = sorted(r.value for r in set(self.reasons) & _AI_REASONS)
        if bad:
            raise ValueError(f"AI 판정 사유는 selection.reasons에 쓸 수 없습니다: {bad}")
        return self


class PendingPolicy(Contract):
    """`pending`: --pending이 세션을 고르는 조건 (digest 제외)."""

    # 마지막 라벨 기록 뒤 이만큼(분) 조용해야 고른다
    quiet_minutes: int = Field(ge=0)


class TemporalPolicy(Contract):
    """`temporal`: 시간 라벨 창·프레임 (기준 스트림 블러본, 마스터 ms)."""

    window_ms: int = Field(gt=0)
    context_ms: int = Field(ge=0)
    frames: int = Field(ge=1)
    boundary_frames: bool
    min_frame_gap_ms: int = Field(ge=0)
    max_frames: int = Field(ge=1)
    max_side_px: int = Field(ge=64)
    max_targets: int = Field(ge=1)
    point_event_focus_ms: int = Field(ge=0)
    long_label_ms: int = Field(gt=0)
    long_object_state_ms: int = Field(gt=0)
    long_frames: int = Field(ge=1)
    long_max_side_px: int = Field(ge=64)
    long_max_span_ms: int = Field(gt=0)
    timestamp_overlay: bool

    @model_validator(mode="after")
    def _check(self) -> TemporalPolicy:
        """프레임 수가 상한 안이어야 한다. Raises: ValueError."""
        if self.frames > self.max_frames or self.long_frames > self.max_frames:
            raise ValueError("frames·long_frames는 max_frames 이하여야 합니다")
        return self


class CropPolicy(Contract):
    """`spatial.crops`: 작은 박스 확대 조각."""

    enabled: bool
    # 프레임에서 이보다 작은 변의 박스만 조각을 만든다 (보낸 프레임 픽셀 기준)
    min_box_side_px: int = Field(ge=1)
    side_px: int = Field(ge=32)
    # 요청당 조각 상한
    max: int = Field(ge=0)


class SpatialPolicy(Contract):
    """`spatial`: 공간 라벨 스냅샷 창 (그 스트림 블러본, 스트림 PTS ms)."""

    stride_ms: int = Field(gt=0)
    snapshot_ms: int = Field(gt=0)
    frames: int = Field(ge=1)
    max_side_px: int = Field(ge=64)
    track_start_snapshot: bool
    max_windows_per_track: int = Field(ge=1)
    max_targets: int = Field(ge=1)
    marks: bool
    crops: CropPolicy


class SensorPolicy(Contract):
    """`sensor`: 장갑 접촉 문장을 넣을지 (구간 계산은 CLI가 prelabel 함수로 한다)."""

    enabled: bool


class ImagePolicy(Contract):
    """`image`: JPEG 품질과 크기 정렬 단위."""

    jpeg_quality: int = Field(ge=30, le=100)
    # 크기를 이 배수로 맞춘다 (Qwen3-VL 32, Qwen2.5-VL 28)
    align_px: int = Field(ge=1)


class SuggestPolicy(Contract):
    """`response.suggest`: 받는 제안. 경계 이동은 제안하지 않는다 (VLM은 경계를 정하지 않는다)."""

    kind_change: bool
    delete_kinds: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> SuggestPolicy:
        """삭제 제안은 이벤트·공간 트랙만. Raises: ValueError."""
        bad = set(self.delete_kinds) - set(DELETABLE_KINDS)
        if bad:
            raise ValueError(f"삭제 제안을 받을 수 없는 종류 (타임라인 공백 0): {sorted(bad)}")
        return self


class ResponsePolicy(Contract):
    """`response`: 응답 형식과 재검사 (모두 digest에 들어간다)."""

    reason_max_chars: int = Field(ge=2, le=REASON_MAX)
    max_problems: int = Field(ge=1, le=3)
    # 누락 발견 상한 (0 = 끔)
    max_findings: int = Field(ge=0)
    # 공간 누락 후보 상자가 같은 클래스 트랙 상자와 이 IoU 이상 겹치면 이미 라벨이 있는 것으로 보고
    # 버린다
    finding_overlap_iou: float = Field(gt=0, le=1)
    suggest: SuggestPolicy
    max_retries: int = Field(ge=0)
    # (번호 위반 + 항목 위반) / 대상 수가 이보다 크면 위반 목록을 붙여 다시 묻는다
    retry_invalid_ratio: float = Field(ge=0, le=1)
    # 이유에 쓰면 고정 문장으로 바꾸는 낱말 (사람의 외모·신원)
    reason_banned_terms: tuple[str, ...] = ()
    # 문맥 초과 때 단위 안에서 나누는 횟수 (프레임 절반 → 대상 절반). 끝내 실패한 대상은 미응답이라
    # 결과를 바꾸므로 digest에 들어간다 (바꾸면 그 단위들을 다시 판정한다)
    max_context_splits: int = Field(ge=0, le=2)


class EstimatePolicy(Contract):
    """`estimate`: 작업량·장비 대수 추정 가정 값 (digest 제외, `dlp aireview estimate`·`probe`).

    모두 가정 값이다. 파일럿 실측(벤치·지적 비율)으로 바꾼다. 요청 본문·결과를 바꾸지 않는다.
    """

    # 맞는 대상 하나의 출력 토큰 (ok 목록 번호)
    ok_tokens: int = Field(ge=0)
    # 지적 항목 하나의 출력 토큰
    item_tokens: int = Field(ge=0)
    # 지적(wrong·suspect·unsure) 비율 가정. 예상 출력 = 대상 수 * (ok_tokens + 비율 * item_tokens)
    flag_ratio: float = Field(ge=0, le=1)
    # 글 토큰 근사 = UTF-8 바이트 / 이 값 (서버 토크나이저 없이 셀 때)
    text_bytes_per_token: float = Field(gt=0)
    # 세션 캘리브레이션에 영상 크기가 없을 때 가정하는 프레임 크기 (너비, 높이) px
    frame_px: tuple[int, int]
    # 블러본 프레임레이트 가정 (순차 디코드가 지나는 프레임 수)
    video_fps: float = Field(gt=0)
    # 하루 영상 시간 H (시간)
    hours_per_day: float = Field(gt=0)
    # 재실행 여유 (정책·모델 변경, 블러 변경 뒤 다시 판정)
    rerun_margin: float = Field(ge=0)
    # 장비 가동률 u (0~1]
    utilization: float = Field(gt=0, le=1)
    # 디코드 동시 작업 수 (블러본 순차 디코드)
    decode_workers: int = Field(ge=1)


class GatePolicy(Contract):
    """`gate`: 골든 게이트 (digest 제외, `dlp aireview eval --gate`, `dlp_aireview.evaluate`).

    값은 골든 평가 지표의 기준이다: 틀린 라벨 탐지 재현율 하한, 깨끗한 라벨 오지적률 상한, ok 놓침률
    상한 (ok 놓침률 = P(틀린 라벨 | AI ok), AI가 맞다고 한 라벨 중 틀린 라벨 비율. 품질 보고의 ok
    놓침률과 같은 정의, `evaluate` 모듈 docstring). block이 false(결정)면 실패해도 경고만 하고 종료
    코드 0이다. 차단(종료 1)은 파일럿(WP17.9) 뒤 정한다.
    """

    min_wrong_recall: float = Field(ge=0, le=1)
    max_false_flag_rate: float = Field(ge=0, le=1)
    max_ok_miss_rate: float = Field(ge=0, le=1)
    block: bool = False
    # perturb 방식의 기본 교란 비율 (교란 종류 → 라벨마다 적용 확률, `perturb.KINDS`).
    # CLI `--inject`가 바꾼다. 모르는 종류는 정책 로드 오류다
    inject: dict[str, float] = Field(default_factory=dict[str, float])

    @model_validator(mode="after")
    def _rates(self) -> GatePolicy:
        """교란 종류는 `perturb.KINDS` 안, 비율은 0~1 (평가 도중 traceback 대신 로드 때 막는다)."""
        unknown = sorted(set(self.inject) - set(PERTURB_KINDS))
        if unknown:
            raise ValueError(
                f"gate.inject: 모르는 교란 종류 {unknown} (가능: {list(PERTURB_KINDS)})"
            )
        bad = sorted(k for k, v in self.inject.items() if not 0 <= v <= 1)
        if bad:
            raise ValueError(f"gate.inject: 0~1 밖의 비율 {bad}")
        return self


class ServerPolicy(Contract):
    """`server`: 서버 연결·실행 (digest 제외, 결과에 영향 없음)."""

    internal_hosts: tuple[str, ...] = Field(min_length=1)
    # 외부 API는 쓰지 않는다 (결정, ADR 0040). true는 로드 오류
    allow_external: Literal[False]
    concurrency_per_endpoint: int = Field(ge=1)
    timeout_s: float = Field(gt=0)
    backoff_s: tuple[float, ...] = ()
    breaker_failures: int = Field(ge=1)
    breaker_cooldown_s: float = Field(ge=0)
    abort_after_s: float = Field(gt=0)
    commit_every: int = Field(ge=1)
    commit_interval_s: float = Field(gt=0)
    render_check_interval_s: float = Field(gt=0)
    progress_interval_s: float = Field(gt=0)


class AiReviewPolicy(Contract):
    """`config/policies/aireview.yaml` 전체."""

    version: int
    prompt_version: int
    unresolved_entity_ids: tuple[str, ...] = ()
    backend: BackendPolicy
    kinds: KindsPolicy
    selection: SelectionPolicy
    pending: PendingPolicy
    temporal: TemporalPolicy
    spatial: SpatialPolicy
    sensor: SensorPolicy
    image: ImagePolicy
    response: ResponsePolicy
    estimate: EstimatePolicy
    gate: GatePolicy
    server: ServerPolicy

    @model_validator(mode="after")
    def _check(self) -> AiReviewPolicy:
        """정책의 프롬프트 버전은 코드 틀 버전과 같아야 한다. Raises: ValueError."""
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"prompt_version {self.prompt_version}이 코드 프롬프트 버전 "
                f"{PROMPT_VERSION}과 다릅니다"
            )
        return self

    @property
    def digest(self) -> str:
        """판정 버전에 들어가는 해시 (sha256 앞 12자).

        파싱값(`model_dump(mode="json")`)에서 digest 제외 절을 빼고 키 정렬 JSON으로 해시하므로
        YAML 주석·순서·서식은 영향이 없다.
        """
        data = self.model_dump(mode="json", exclude=_NOT_HASHED)
        content = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def verdict_version(self) -> str:
        """판정 버전 `aireview-<digest>+vlm-<model>` (행동 단계 `vlm-<model>`과 같은 관례).

        `version`(정책 형식 버전, 정수)과 다르다.
        """
        return f"aireview-{self.digest}+vlm-{self.backend.model}"

    @property
    def target_rule(self) -> AiTargetRule:
        """판정 대상 조건의 정책 값 (`dlp_schema.aireview.is_judge_target`).

        판정 실행(`plan.targets`)과 검수 화면 판정률 분모(CLI가 `AiDisplayConfig.targets`로
        넘긴다)가 같은 값을 쓴다.
        """
        return AiTargetRule(
            kinds=frozenset((*self.kinds.temporal, *self.kinds.spatial)),
            sources=frozenset(self.kinds.sources),
            long_max_span_ms=self.temporal.long_max_span_ms,
        )

    @property
    def served_model(self) -> str:
        """서버에 등록된 모델 이름 (`served_name`, 없으면 `model`). 요청 본문 `model`로 보내고 응답
        `model`과 대조한다."""
        return self.backend.served_name or self.backend.model


def load_policy(root: Path, path: Path | None = None) -> AiReviewPolicy:
    """`<root>/config/policies/aireview.yaml`(또는 `path`)을 읽어 검증한다.

    Raises:
        OSError: 파일이 없을 때.
        pydantic.ValidationError: 키가 없거나, 모르는 키가 있거나, 값이 범위·결정을 벗어날 때.
    """
    source = path if path is not None else root / "config" / "policies" / "aireview.yaml"
    data: Any = yaml.safe_load(source.read_text("utf-8"))
    return AiReviewPolicy.model_validate(data)
