"""복원과 전후 비교 실행 (`dlp scene3d reconstruct|compare|report`, ADR 0066·0067).

두 단계 모두 입력은 `privacy_gate.open_blurred`의 승인된 블러 사진뿐이고 산출물은 라벨링 버킷에
쓴다. 모델 버전은 (백엔드 버전, 쓰는 정책 절 해시)이고 버전 태그는 그 버전과 입력(렌더 태그·복원
태그)의 해시다. 같은 태그가 이미 있으면 다시 계산하지 않는다 (멱등). 결과는 버전 키에 한 번만
쓰고 덮어쓰지 않으며 `current.json` 표시만 옮긴다.

복원은 SfM 뒤, 스플랫 학습 전에 마커로 미터 척도를 정한다 (`markers`, ADR 0067): 척도를 곱한 카메라·
점으로 스플랫을 학습하고 척도의 출처를 복원 기록(`Reconstruction.scale`)에 남긴다. 마커가 없거나
추정이 불안정하면 좌표를 건드리지 않고 `metric=False`로 둔다.

정렬(after → before)은 비교와 위치 추정(`localize`)이 함께 쓴다. `ensure_alignment`가 두 복원 태그와
정렬 정책으로 정한 키에 한 번만 계산해 쓰고, 이미 있으면 그것을 읽는다.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from dlp_media.storage import ObjectStore
from dlp_scene3d.alignment import align_scenes
from dlp_scene3d.backends import ObjectDetector, SfmBackend, SfmResult, SplatTrainer
from dlp_scene3d.change import compare_objects, to_before_frame
from dlp_scene3d.geometry import F64, Camera, Similarity
from dlp_scene3d.layout import (
    get_bytes,
    get_model,
    get_pointer,
    phase_prefix,
    put_bytes_immutable,
    put_model_immutable,
    put_pointer,
    session_prefix,
    short_hash,
)
from dlp_scene3d.markers import MarkerConfigError, estimate_scale, find_markers, scale_sfm
from dlp_scene3d.objects import Detection2D, appearance, localize_objects
from dlp_scene3d.policy import MarkerPolicy, Scene3dPolicy, section_digest
from dlp_scene3d.privacy_gate import BlurredPhoto, open_blurred, render_key
from dlp_schema.scene3d import (
    Alignment,
    CameraPose,
    ChangeReport,
    ObjectInstance,
    PhotoRender,
    Reconstruction,
    ScenePhase,
    SceneScale,
)

# 복원 단계 판 (2: SfM 뒤 마커 척도, ADR 0067)
RECON_VERSION = "scene3d-recon-2"
ALIGN_VERSION = "scene3d-align-1"


class Scene3dError(RuntimeError):
    """복원·비교·위치 추정을 진행할 수 없다 (등록 사진 부족, 척도 없음, 낡은 입력, 빠진 전제)."""


class AlignmentRejectedError(Scene3dError):
    """정렬 확신이 낮아 비교·위치 추정을 거부했다 (정렬 기록은 남긴다)."""

    def __init__(self, alignment: Alignment) -> None:
        super().__init__(f"정렬 거부: {alignment.reason}")
        self.alignment = alignment


def _recon_prefix(sid: str, phase: ScenePhase, tag: str) -> str:
    return f"{phase_prefix(sid, phase)}/recon/{tag}"


def scale_by_markers(
    result: SfmResult, photos: Sequence[BlurredPhoto], policy: MarkerPolicy
) -> tuple[SfmResult, SceneScale]:
    """SfM 결과에 마커 척도를 적용한다: (스플랫 학습에 넘길 SfM 결과, 척도 출처 기록).

    - 마커 척도가 꺼져 있으면 그대로 둔다 (출처는 SfM이 미터면 sfm, 아니면 none).
    - SfM 결과가 이미 미터(정답 stub)면 좌표를 건드리지 않는다. 마커가 안정적으로 보이면 그
      배율이 1과 `metric_check_tolerance` 안에서 맞는지만 확인한다 (다르면 거부: 마커 설정이나
      SfM이 틀렸다).
    - 미터가 아닌 결과는 마커가 안정적으로 보일 때만 그 배율을 곱한다. 아니면 척도를 지어내지
      않고 metric=False로 둔 채 이유를 남긴다.

    Raises:
        Scene3dError: 정책의 마커 사전 이름이 틀렸을 때, 미터인 SfM 결과와 마커 배율이 다를 때.
    """
    default = "sfm" if result.metric else "none"
    if not policy.enabled:
        why = "마커 척도를 쓰지 않습니다 (reconstruction.marker.enabled)"
        return result, SceneScale(source=default, factor=1.0, reason=why)
    try:
        est = estimate_scale(find_markers(photos, result.cameras, policy), result.cameras, policy)
    except MarkerConfigError as exc:
        raise Scene3dError(str(exc)) from exc
    found = SceneScale(
        source=default,
        factor=1.0,
        marker_side_m=policy.side_m,
        marker_factor=est.factor,
        marker_ids=est.marker_ids,
        marker_photos=est.photos,
        side_spread=est.side_spread,
        reason=est.reason,
    )
    if est.factor is None:
        return result, found
    if result.metric:
        if abs(est.factor - 1.0) > policy.metric_check_tolerance:
            raise Scene3dError(
                f"SfM 결과는 미터인데 마커 척도 배율이 {est.factor:.4f}입니다 "
                f"(허용 1 ± {policy.metric_check_tolerance}). "
                "마커 한 변 길이(side_m)·사전을 확인하세요"
            )
        return result, found
    scaled = found.model_copy(update={"source": "marker", "factor": est.factor})
    return scale_sfm(result, est.factor), SceneScale.model_validate(scaled.model_dump())


def reconstruct(
    labeling: ObjectStore,
    sid: str,
    phase: ScenePhase,
    sfm: SfmBackend,
    splat: SplatTrainer,
    policy: Scene3dPolicy,
    workdir: Path,
    now: datetime,
) -> tuple[Reconstruction, bool]:
    """(복원 기록, 새로 계산했는가).

    Raises:
        RenderNotCurrentError: 승인된 블러 사진이 없을 때.
        Scene3dError: 자세를 찾은 사진 비율이 정책 미만일 때, `scale_by_markers`의 오류.
    """
    render_tag, photos = open_blurred(labeling, sid, phase, workdir / "blurred")
    digest = section_digest(policy.reconstruction)
    version = f"{RECON_VERSION}+sfm:{sfm.version}+splat:{splat.version}+p{digest}"
    tag = short_hash(version, render_tag)
    prefix = _recon_prefix(sid, phase, tag)
    existing = get_model(labeling, f"{prefix}/reconstruction.json", Reconstruction)
    if existing is not None:
        put_pointer(labeling, f"{phase_prefix(sid, phase)}/recon/current.json", {"tag": tag})
        return existing, False
    result = sfm.reconstruct(photos, workdir / "sfm")
    ratio = len(result.cameras) / len(photos)
    if ratio < policy.reconstruction.min_registered_ratio:
        raise Scene3dError(
            f"자세를 찾은 사진 {len(result.cameras)}/{len(photos)}장: 정책 최소 비율 "
            f"{policy.reconstruction.min_registered_ratio} 미만"
        )
    result, scale = scale_by_markers(result, photos, policy.reconstruction.marker)
    splat_path = workdir / "splat.ply"
    splat.train(photos, result, splat_path)
    poses = [result.cameras[k].to_pose().model_dump(mode="json") for k in sorted(result.cameras)]
    buf = io.BytesIO()
    np.savez(buf, points=result.points, descriptors=result.descriptors, colors=result.colors)
    put_bytes_immutable(labeling, f"{prefix}/cameras.json", json.dumps(poses).encode(), ".json")
    put_bytes_immutable(labeling, f"{prefix}/points.npz", buf.getvalue(), ".npz")
    put_bytes_immutable(labeling, f"{prefix}/splat.ply", splat_path.read_bytes(), ".ply")
    rec = Reconstruction(
        session_id=sid,
        phase=phase,
        model_version=version,
        render_tag=render_tag,
        metric=result.metric,
        photo_count=len(photos),
        registered=len(result.cameras),
        point_count=len(result.points),
        cameras_key=f"{prefix}/cameras.json",
        points_key=f"{prefix}/points.npz",
        splat_key=f"{prefix}/splat.ply",
        created_at=now,
        scale=scale,
    )
    put_model_immutable(labeling, f"{prefix}/reconstruction.json", rec)
    put_pointer(labeling, f"{phase_prefix(sid, phase)}/recon/current.json", {"tag": tag})
    return rec, True


def stored_reconstruction(
    labeling: ObjectStore, sid: str, phase: ScenePhase
) -> tuple[str, Reconstruction] | None:
    """현재 표시가 가리키는 (태그, 복원 기록). 없으면 None.

    지금 블러 사진과 맞는지는 보지 않는다 (`reconstruction_is_current`).
    """
    ptr = get_pointer(labeling, f"{phase_prefix(sid, phase)}/recon/current.json")
    if not ptr:
        return None
    tag = ptr["tag"]
    rec = get_model(
        labeling, f"{_recon_prefix(sid, phase, tag)}/reconstruction.json", Reconstruction
    )
    return None if rec is None else (tag, rec)


def reconstruction_is_current(
    labeling: ObjectStore, sid: str, phase: ScenePhase, rec: Reconstruction
) -> bool:
    """복원이 지금 승인된 블러 사진(렌더 표시)으로 만든 것인가."""
    render = get_model(labeling, render_key(sid, phase), PhotoRender)
    return render is not None and render.render_tag == rec.render_tag


def current_reconstruction_record(
    labeling: ObjectStore, sid: str, phase: ScenePhase
) -> tuple[str, Reconstruction]:
    """(태그, 복원 기록). 카메라·점은 읽지 않는다.

    Raises:
        Scene3dError: 복원이 없거나, 지금 블러 사진과 다를 때 (블러 렌더가 바뀜).
    """
    found = stored_reconstruction(labeling, sid, phase)
    if found is None:
        raise Scene3dError(f"{sid} {phase}: 복원이 없습니다 (dlp scene3d reconstruct)")
    if not reconstruction_is_current(labeling, sid, phase, found[1]):
        raise Scene3dError(f"{sid} {phase}: 복원이 지금 블러 사진과 다릅니다. 다시 복원하세요")
    return found


def current_reconstruction(
    labeling: ObjectStore, sid: str, phase: ScenePhase
) -> tuple[str, Reconstruction, dict[str, Camera], F64, F64]:
    """(태그, 복원 기록, 카메라, 점, 기술자).

    Raises:
        Scene3dError: 복원이 없거나 낡았을 때 (블러 렌더가 바뀜).
    """
    tag, rec = current_reconstruction_record(labeling, sid, phase)
    poses = json.loads(get_bytes(labeling, rec.cameras_key) or b"[]")
    cams = {p["photo_id"]: Camera.from_pose(CameraPose.model_validate(p)) for p in poses}
    npz = np.load(io.BytesIO(get_bytes(labeling, rec.points_key) or b""))
    pts = np.asarray(npz["points"], dtype=np.float64)
    return tag, rec, cams, pts, np.asarray(npz["descriptors"], dtype=np.float64)


def alignment_version(policy: Scene3dPolicy) -> str:
    """정렬 단계의 모델 버전 (판 + `alignment` 절 해시)."""
    return f"{ALIGN_VERSION}+p{section_digest(policy.alignment)}"


def alignment_tag(policy: Scene3dPolicy, before_tag: str, after_tag: str) -> str:
    """두 복원과 정렬 정책으로 정해지는 정렬 태그."""
    return short_hash(alignment_version(policy), before_tag, after_tag)


def alignment_key(sid: str, tag: str) -> str:
    """라벨링 버킷의 정렬 기록 키."""
    return f"{session_prefix(sid)}/alignment/{tag}.json"


def similarity_of(alignment: Alignment) -> Similarity:
    """정렬 계약 → 닮음 변환 `x_b = s R x_a + t`."""
    return Similarity(
        alignment.scale,
        np.asarray(alignment.rotation, dtype=np.float64).reshape(3, 3),
        np.asarray(alignment.translation, dtype=np.float64),
    )


@dataclass(frozen=True)
class SceneAlignment:
    """`ensure_alignment` 결과: 태그, 기록, 받아들인 변환 (거부됐으면 None)."""

    tag: str
    record: Alignment
    transform: Similarity | None


def ensure_alignment(
    labeling: ObjectStore,
    sid: str,
    policy: Scene3dPolicy,
    before: tuple[str, F64, F64],
    after: tuple[str, F64, F64],
) -> SceneAlignment:
    """두 복원의 after → before 정렬. 같은 태그의 기록이 있으면 읽고, 없으면 계산해 한 번만 쓴다.

    Args:
        before, after: (복원 태그, 점 (N, 3), 기술자 (N, D)).

    Returns:
        거부된 정렬도 기록과 함께 돌려준다 (transform이 None). 비교·위치 추정은 거부된 정렬로
        진행하지 않는다.
    """
    tag = alignment_tag(policy, before[0], after[0])
    key = alignment_key(sid, tag)
    record = get_model(labeling, key, Alignment)
    if record is None:
        record, _ = align_scenes(
            before[1], before[2], after[1], after[2], policy.alignment, alignment_version(policy)
        )
        put_model_immutable(labeling, key, record)
    return SceneAlignment(tag, record, similarity_of(record) if record.accepted else None)


def _phase_objects(
    labeling: ObjectStore,
    sid: str,
    phase: ScenePhase,
    cams: Mapping[str, Camera],
    detector: ObjectDetector,
    policy: Scene3dPolicy,
    workdir: Path,
) -> list[ObjectInstance]:
    """한 단계의 블러 사진에서 물체를 찾아 3D로 올린다."""
    _, photos = open_blurred(labeling, sid, phase, workdir / f"objects-{phase}")
    op = policy.objects
    dets: list[Detection2D] = []
    for p in photos:
        img = p.load()
        for label, box, score in detector.detect(p.photo_id, img):
            if label in op.queries and score >= op.score_threshold:
                app = appearance(img, box, op.appearance_bins, op.appearance_crop)
                dets.append(Detection2D(p.photo_id, label, box, score, app))
    return localize_objects(phase, cams, dets, op)


def compare(
    labeling: ObjectStore,
    sid: str,
    detector: ObjectDetector,
    policy: Scene3dPolicy,
    workdir: Path,
    now: datetime,
) -> tuple[ChangeReport, bool]:
    """(비교 결과, 새로 계산했는가). 두 단계의 지금 복원으로 정렬·물체·변화를 계산한다.

    Raises:
        Scene3dError: 복원이 없거나 낡았을 때, 정책상 before가 미터 단위여야 하는데 아닐 때.
        AlignmentRejectedError: 정렬이 정책 문턱을 못 넘었을 때 (정렬 기록은 남긴다).
    """
    tb, rb, cams_b, pts_b, desc_b = current_reconstruction(labeling, sid, "before")
    ta, _, cams_a, pts_a, desc_a = current_reconstruction(labeling, sid, "after")
    if policy.change.require_metric and not rb.metric:
        why = rb.scale.reason if rb.scale and rb.scale.reason else "척도 기준 없음"
        raise Scene3dError(
            f"before 복원이 미터 단위가 아닙니다 ({why}). 크기를 아는 마커가 보이게 찍어 다시 "
            "복원하세요 (scene3d.yaml reconstruction.marker, ADR 0067)"
        )
    version = (
        f"scene3d-change-1+det:{detector.version}+p{section_digest(policy.objects)}"
        f"{section_digest(policy.change)}+{alignment_version(policy)}"
    )
    tag = short_hash(version, tb, ta)
    key = change_key(sid, tag)
    existing = get_model(labeling, key, ChangeReport)
    if existing is not None:
        put_pointer(labeling, f"{session_prefix(sid)}/changes/current.json", {"tag": tag})
        return existing, False
    aligned = ensure_alignment(labeling, sid, policy, (tb, pts_b, desc_b), (ta, pts_a, desc_a))
    if aligned.transform is None:
        raise AlignmentRejectedError(aligned.record)
    before = _phase_objects(labeling, sid, "before", cams_b, detector, policy, workdir)
    after = _phase_objects(labeling, sid, "after", cams_a, detector, policy, workdir)
    moved = [to_before_frame(o, aligned.transform) for o in after]
    report = ChangeReport(
        session_id=sid,
        model_version=version,
        before_reconstruction=tb,
        after_reconstruction=ta,
        alignment=aligned.record,
        metric=rb.metric,
        instances=(*before, *moved),
        changes=tuple(compare_objects(before, moved, policy.change)),
        created_at=now,
    )
    put_model_immutable(labeling, key, report)
    put_pointer(labeling, f"{session_prefix(sid)}/changes/current.json", {"tag": tag})
    return report, True


def change_key(sid: str, tag: str) -> str:
    """라벨링 버킷의 비교 결과 키."""
    return f"{session_prefix(sid)}/changes/{tag}.json"


def report_tag(report: ChangeReport) -> str:
    """비교 결과의 버전 태그 (`compare`가 키에 쓴 값: 모델 버전과 두 복원 태그의 해시)."""
    return short_hash(
        report.model_version, report.before_reconstruction, report.after_reconstruction
    )


def stored_report(labeling: ObjectStore, sid: str) -> ChangeReport | None:
    """현재 표시가 가리키는 비교 결과. 없으면 None."""
    ptr = get_pointer(labeling, f"{session_prefix(sid)}/changes/current.json")
    return get_model(labeling, change_key(sid, ptr["tag"]), ChangeReport) if ptr else None


def current_report(labeling: ObjectStore, sid: str) -> ChangeReport:
    """지금 비교 결과. 없으면 Scene3dError."""
    report = stored_report(labeling, sid)
    if report is None:
        raise Scene3dError(f"{sid}: 비교 결과가 없습니다 (dlp scene3d compare)")
    return report


CSV_COLUMNS: tuple[str, ...] = (
    "change_id",
    "kind",
    "label",
    "before_instance_id",
    "after_instance_id",
    "before_x",
    "before_y",
    "before_z",
    "after_x",
    "after_y",
    "after_z",
    "dx_m",
    "dy_m",
    "dz_m",
    "distance_m",
    "confidence",
    "review_state",
)


def report_rows(report: ChangeReport) -> list[dict[str, str]]:
    """CSV 행 (좌표는 before 장면 좌표, 비어 있으면 빈 칸)."""

    def xyz(v: tuple[float, float, float] | None) -> list[str]:
        return ["", "", ""] if v is None else [f"{c:.4f}" for c in v]

    rows: list[dict[str, str]] = []
    for c in report.changes:
        vals = [c.change_id, c.kind, c.label, c.before_instance_id or "", c.after_instance_id or ""]
        vals += xyz(c.before_position) + xyz(c.after_position) + xyz(c.displacement)
        vals += [
            "" if c.distance_m is None else f"{c.distance_m:.4f}",
            f"{c.confidence:.3f}",
            c.review_state,
        ]
        rows.append(dict(zip(CSV_COLUMNS, vals, strict=True)))
    return rows
