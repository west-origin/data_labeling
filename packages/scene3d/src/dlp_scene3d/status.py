"""세션의 전후 3D 공간이 다 갖춰졌는지 보기 (`dlp scene3d status`, ADR 0067).

작업 전후 공간 정보가 **둘 다** 있어야 한다는 요구를 한 번에 확인한다. before·after 각각의 수집,
블러 탐지·승인, 블러 렌더, 복원(미터 여부와 척도 출처)과, 세션의 정렬·비교·위치 추정이 지금
있는지를 본다. 전후 복원이 모두 지금 것이고 정렬이 받아들여졌으면 갖춰진 것이다
(`SessionStatus.ready`).

읽기 전용이다. 저장소에 쓰지 않고 DB를 쓰지 않는다.
- 라벨링 버킷: 수집 기록, 렌더 표시, 복원·정렬·비교·위치 기록을 읽는다.
- 원본 버킷: 블러 탐지·승인 표시 파일이 **있는지만** 본다 (`privacy_gate.gate_marks`, head만 써서
  감사 기록도 남지 않는다). 내용을 읽지 않으므로 그 승인이 최신 탐지 세대의 것인지는 판정하지
  못한다. 지금 유효한 승인인지는 렌더 표시가 말한다 (유효한 렌더 = 최신 탐지 세대의 승인으로 만든
  블러본).

"지금 것"의 뜻: 복원은 지금 렌더 표시의 블러 사진으로 만든 것, 정렬은 지금 두 복원과 정렬 정책의
것, 비교·위치 기록은 지금 두 복원으로 만든 것이다. 정책을 바꾸고 다시 돌리지 않은 것까지는 가리지
않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dlp_media.storage import ObjectStore
from dlp_scene3d.layout import get_model
from dlp_scene3d.localize import stored_localization
from dlp_scene3d.photos import capture_key
from dlp_scene3d.pipeline import (
    alignment_key,
    alignment_tag,
    reconstruction_is_current,
    stored_reconstruction,
    stored_report,
)
from dlp_scene3d.policy import Scene3dPolicy
from dlp_scene3d.privacy_gate import gate_marks, render_key
from dlp_schema.scene3d import (
    Alignment,
    ChangeReport,
    EgoLocalization,
    PhotoCapture,
    PhotoRender,
    Reconstruction,
    ScenePhase,
)

# 산출물 상태: 지금 것 | 낡음(입력이 바뀜) | 없음
State = Literal["current", "stale", "missing"]
# 렌더 표시 상태: 유효 | 무효(재탐지로 승인이 낡음) | 없음
RenderState = Literal["current", "invalid", "missing"]


@dataclass(frozen=True)
class PhaseStatus:
    """한 단계(before 또는 after)의 상태.

    photos: 수집한 사진 수 (수집 전이면 None). detected·approved: 원본 버킷에 블러 탐지·승인 표시가
    있는가 (원본 저장소를 주지 않았으면 None). approval_id: 지금 렌더를 만든 승인. reconstruction:
    현재 표시가 가리키는 복원 기록 (없으면 None).
    """

    phase: ScenePhase
    photos: int | None
    detected: bool | None
    approved: bool | None
    render: RenderState
    render_tag: str | None
    approval_id: str | None
    recon: State
    recon_tag: str | None
    reconstruction: Reconstruction | None


@dataclass(frozen=True)
class SessionStatus:
    """세션 하나의 전후 3D 공간 상태.

    alignment: 지금 두 복원·정렬 정책의 정렬 기록 (두 복원이 지금 것이 아니거나 아직 정렬하지
    않았으면 None). change·localization: 비교·위치 기록의 상태와 그 기록.
    """

    session_id: str
    before: PhaseStatus
    after: PhaseStatus
    alignment: Alignment | None
    change: State
    change_report: ChangeReport | None
    localization: State
    localization_record: EgoLocalization | None

    @property
    def missing(self) -> list[str]:
        """전후 공간이 갖춰지려면 빠진 것 (다음에 할 명령 포함). 비어 있으면 갖춰졌다."""
        out = [m for p in (self.before, self.after) if (m := _phase_gap(self.session_id, p))]
        if out:
            return out
        if self.alignment is None:
            return [
                f"정렬 없음: dlp scene3d compare {self.session_id} "
                f"(또는 dlp scene3d localize {self.session_id})"
            ]
        if not self.alignment.accepted:
            return [f"정렬 거부: {self.alignment.reason}"]
        return []

    @property
    def ready(self) -> bool:
        """전후 복원이 모두 지금 것이고 정렬이 받아들여졌는가."""
        return not self.missing

    @property
    def notes(self) -> list[str]:
        """갖춤 판정과는 별개로 알려 둘 것 (미터가 아닌 before 복원)."""
        rec = self.before.reconstruction
        if self.before.recon != "current" or rec is None or rec.metric:
            return []
        why = rec.scale.reason if rec.scale and rec.scale.reason else "척도 기준 없음"
        return [
            f"before 복원이 미터가 아닙니다 ({why}). 위치·이동 벡터의 단위가 미터가 아니고, "
            "scene3d.yaml change.require_metric이 켜져 있으면 비교가 거부됩니다"
        ]


def _phase_gap(sid: str, p: PhaseStatus) -> str | None:
    """그 단계의 복원이 지금 것이 아니면 무엇이 빠졌는지 (가장 앞 단계부터)."""
    if p.recon == "current":
        return None
    if p.photos is None:
        return f"{p.phase} 사진 수집 없음: dlp scene3d ingest {sid} {p.phase} <사진 디렉터리>"
    if p.render != "current":
        why = "무효 (재탐지로 승인이 낡음)" if p.render == "invalid" else "없음"
        return (
            f"{p.phase} 승인된 블러 사진 {why}: dlp scene3d privacy detect → review → approve → "
            f"render {sid} {p.phase}"
        )
    what = "낡음 (블러 사진이 바뀜)" if p.recon == "stale" else "없음"
    return f"{p.phase} 복원 {what}: dlp scene3d reconstruct {sid} {p.phase}"


def _phase_status(
    labeling: ObjectStore, raw: ObjectStore | None, sid: str, phase: ScenePhase
) -> PhaseStatus:
    """한 단계의 상태를 읽는다."""
    capture = get_model(labeling, capture_key(sid, phase), PhotoCapture)
    render = get_model(labeling, render_key(sid, phase), PhotoRender)
    render_state: RenderState = "missing"
    if render is not None:
        render_state = "invalid" if render.render_tag is None else "current"
    found = stored_reconstruction(labeling, sid, phase)
    recon: State = "missing"
    if found is not None:
        recon = "current" if reconstruction_is_current(labeling, sid, phase, found[1]) else "stale"
    detected, approved = gate_marks(raw, sid, phase) if raw is not None else (None, None)
    return PhaseStatus(
        phase=phase,
        photos=None if capture is None else len(capture.photos),
        detected=detected,
        approved=approved,
        render=render_state,
        render_tag=None if render is None else render.render_tag,
        approval_id=None if render is None else render.approval_id,
        recon=recon,
        recon_tag=None if found is None else found[0],
        reconstruction=None if found is None else found[1],
    )


def _derived_state(made_from: tuple[str, str] | None, now: tuple[str, str] | None) -> State:
    """두 복원 태그로 만든 산출물의 상태 (지금 두 복원 태그와 같으면 지금 것)."""
    if made_from is None:
        return "missing"
    return "current" if made_from == now else "stale"


def scene_status(
    labeling: ObjectStore, sid: str, policy: Scene3dPolicy, raw: ObjectStore | None = None
) -> SessionStatus:
    """세션의 전후 3D 공간 상태를 읽는다 (모듈 설명: 쓰지 않는다).

    Args:
        labeling: 라벨링 버킷 저장소.
        sid: 세션 ID.
        policy: scene3d 정책 (지금 정렬 태그를 정하는 `alignment` 절).
        raw: 원본 버킷 저장소. 주면 블러 탐지·승인 표시가 있는지 head로만 본다. None이면 보지
            않는다.
    """
    before = _phase_status(labeling, raw, sid, "before")
    after = _phase_status(labeling, raw, sid, "after")
    tags: tuple[str, str] | None = None
    alignment: Alignment | None = None
    if before.recon == "current" and after.recon == "current":
        tags = (str(before.recon_tag), str(after.recon_tag))
        key = alignment_key(sid, alignment_tag(policy, tags[0], tags[1]))
        alignment = get_model(labeling, key, Alignment)
    report = stored_report(labeling, sid)
    located = stored_localization(labeling, sid)
    loc = None if located is None else located[1]
    return SessionStatus(
        session_id=sid,
        before=before,
        after=after,
        alignment=alignment,
        change=_derived_state(
            None if report is None else (report.before_reconstruction, report.after_reconstruction),
            tags,
        ),
        change_report=report,
        localization=_derived_state(
            None if loc is None else (loc.before_reconstruction, loc.after_reconstruction), tags
        ),
        localization_record=loc,
    )


_STATE_KO: dict[str, str] = {"current": "있음", "stale": "낡음", "missing": "없음"}


def _mark(flag: bool | None) -> str:
    if flag is None:
        return "확인 안 함"
    return "표시 있음" if flag else "없음"


def _approval_cell(p: PhaseStatus) -> str:
    """블러 승인 칸: 유효한 렌더가 있으면 그 승인, 아니면 원본 버킷 표시 유무."""
    if p.render == "current":
        return f"승인 {p.approval_id} (렌더에 쓰임)"
    if p.approved:
        return "기록 있음 (최신 탐지 세대의 것인지는 render가 확인)"
    return _mark(p.approved)


def _render_cell(p: PhaseStatus) -> str:
    if p.render == "current":
        return f"유효 {p.render_tag}"
    return "무효 (재탐지로 승인이 낡음)" if p.render == "invalid" else "없음"


def _recon_cell(p: PhaseStatus) -> str:
    rec = p.reconstruction
    if rec is None:
        return "없음"
    note = "" if p.recon == "current" else " (낡음: 블러 사진이 바뀜)"
    return f"{p.recon_tag} 사진 {rec.registered}/{rec.photo_count}, 점 {rec.point_count}{note}"


def _scale_cell(p: PhaseStatus) -> str:
    """척도 칸: 미터 여부와 출처."""
    rec = p.reconstruction
    if rec is None:
        return "-"
    unit = "미터" if rec.metric else "미터 아님"
    s = rec.scale
    if s is None:
        return f"{unit} (출처 기록 없음: ADR 0067 이전 복원)"
    if s.source == "marker":
        return (
            f"{unit} (마커 {list(s.marker_ids)}, 사진 {s.marker_photos}장, 배율 {s.factor:.4f}, "
            f"흩어짐 {s.side_spread:.3f})"
        )
    why = f": {s.reason}" if s.reason else ""
    return f"{unit} ({'SfM 결과' if s.source == 'sfm' else '척도 없음'}{why})"


def _alignment_cell(a: Alignment | None) -> str:
    if a is None:
        return "없음"
    head = "받아들임" if a.accepted else f"거부 ({a.reason})"
    rms = "-" if a.rms_m is None else f"{a.rms_m:.4f}"
    return f"{head}: 내점 {a.inliers}/{a.matches}, RMS {rms}, 척도 {a.scale:.4f}"


def _change_cell(s: SessionStatus) -> str:
    if s.change_report is None:
        return "없음"
    note = "" if s.change == "current" else " (낡음: 복원이 바뀜)"
    return f"{_STATE_KO[s.change]}: 변화 {len(s.change_report.changes)}개{note}"


def _localization_cell(s: SessionStatus) -> str:
    loc = s.localization_record
    if loc is None:
        return "없음"
    after = sum(1 for p in loc.poses if p.reference == "after")
    note = "" if s.localization == "current" else " (낡음: 복원이 바뀜)"
    unit = "미터" if loc.metric else "미터 아님"
    return (
        f"{_STATE_KO[s.localization]}: 프레임 {loc.localized}/{loc.frame_count} "
        f"(before 기준 {loc.localized - after}, after 기준 {after}), {unit}{note}"
    )


def format_status(status: SessionStatus) -> list[str]:
    """상태를 사람이 읽는 표(Markdown 표 두 개)와 빠진 것 목록으로 만든다."""
    b, a = status.before, status.after
    photos = ["없음" if p.photos is None else f"사진 {p.photos}장" for p in (b, a)]
    rows = [
        ("수집", photos[0], photos[1]),
        ("블러 탐지", _mark(b.detected), _mark(a.detected)),
        ("블러 승인", _approval_cell(b), _approval_cell(a)),
        ("블러 렌더", _render_cell(b), _render_cell(a)),
        ("복원", _recon_cell(b), _recon_cell(a)),
        ("척도", _scale_cell(b), _scale_cell(a)),
    ]
    lines = [f"세션 {status.session_id}: 전후 3D 공간 상태", "", "| 단계 | before | after |"]
    lines.append("| --- | --- | --- |")
    lines += [f"| {name} | {x} | {y} |" for name, x, y in rows]
    lines += ["", "| 세션 | 상태 |", "| --- | --- |"]
    lines.append(f"| 정렬 (after → before) | {_alignment_cell(status.alignment)} |")
    lines.append(f"| 비교 | {_change_cell(status)} |")
    lines.append(f"| 위치 추정 | {_localization_cell(status)} |")
    lines.append("")
    if status.ready:
        lines.append("갖춰짐: 전후 복원이 모두 있고 정렬이 받아들여졌습니다")
    else:
        lines += [f"빠짐: {m}" for m in status.missing]
    lines += [f"주의: {n}" for n in status.notes]
    return lines
