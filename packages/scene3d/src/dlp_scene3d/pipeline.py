"""복원과 전후 비교 실행 (`dlp scene3d reconstruct|compare|report`, ADR 0066).

두 단계 모두 입력은 `privacy_gate.open_blurred`의 승인된 블러 사진뿐이고 산출물은 라벨링 버킷에
쓴다. 모델 버전은 (백엔드 버전, 쓰는 정책 절 해시)이고 버전 태그는 그 버전과 입력(렌더 태그·복원
태그)의 해시다. 같은 태그가 이미 있으면 다시 계산하지 않는다 (멱등). 결과는 버전 키에 한 번만
쓰고 덮어쓰지 않으며 `current.json` 표시만 옮긴다.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np

from dlp_media.storage import ObjectStore
from dlp_scene3d.alignment import align_scenes
from dlp_scene3d.backends import ObjectDetector, SfmBackend, SplatTrainer
from dlp_scene3d.change import compare_objects, to_before_frame
from dlp_scene3d.geometry import F64, Camera
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
from dlp_scene3d.objects import Detection2D, appearance, localize_objects
from dlp_scene3d.policy import Scene3dPolicy, section_digest
from dlp_scene3d.privacy_gate import open_blurred, render_key
from dlp_schema.scene3d import (
    Alignment,
    CameraPose,
    ChangeReport,
    ObjectInstance,
    PhotoRender,
    Reconstruction,
    ScenePhase,
)


class Scene3dError(RuntimeError):
    """복원·비교를 진행할 수 없다 (등록 사진 부족, 척도 없음, 낡은 입력)."""


class AlignmentRejectedError(Scene3dError):
    """정렬 확신이 낮아 비교를 거부했다 (정렬 기록은 남긴다)."""

    def __init__(self, alignment: Alignment) -> None:
        super().__init__(f"정렬 거부: {alignment.reason}")
        self.alignment = alignment


def _recon_prefix(sid: str, phase: ScenePhase, tag: str) -> str:
    return f"{phase_prefix(sid, phase)}/recon/{tag}"


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
        Scene3dError: 자세를 찾은 사진 비율이 정책 미만일 때.
    """
    render_tag, photos = open_blurred(labeling, sid, phase, workdir / "blurred")
    digest = section_digest(policy.reconstruction)
    version = f"scene3d-recon-1+sfm:{sfm.version}+splat:{splat.version}+p{digest}"
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
    )
    put_model_immutable(labeling, f"{prefix}/reconstruction.json", rec)
    put_pointer(labeling, f"{phase_prefix(sid, phase)}/recon/current.json", {"tag": tag})
    return rec, True


def current_reconstruction(
    labeling: ObjectStore, sid: str, phase: ScenePhase
) -> tuple[str, Reconstruction, dict[str, Camera], F64, F64]:
    """(태그, 복원 기록, 카메라, 점, 기술자). 복원이 낡았으면(블러 렌더가 바뀜) Scene3dError."""
    ptr = get_pointer(labeling, f"{phase_prefix(sid, phase)}/recon/current.json")
    tag = ptr["tag"] if ptr else ""
    rec = get_model(
        labeling, f"{_recon_prefix(sid, phase, tag)}/reconstruction.json", Reconstruction
    )
    if rec is None:
        raise Scene3dError(f"{sid} {phase}: 복원이 없습니다 (dlp scene3d reconstruct)")
    render = get_model(labeling, render_key(sid, phase), PhotoRender)
    if render is None or render.render_tag != rec.render_tag:
        raise Scene3dError(f"{sid} {phase}: 복원이 지금 블러 사진과 다릅니다. 다시 복원하세요")
    poses = json.loads(get_bytes(labeling, rec.cameras_key) or b"[]")
    cams = {p["photo_id"]: Camera.from_pose(CameraPose.model_validate(p)) for p in poses}
    npz = np.load(io.BytesIO(get_bytes(labeling, rec.points_key) or b""))
    pts = np.asarray(npz["points"], dtype=np.float64)
    return tag, rec, cams, pts, np.asarray(npz["descriptors"], dtype=np.float64)


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
        raise Scene3dError("before 복원이 미터 단위가 아닙니다 (척도 기준 필요, ADR 0066)")
    align_version = f"scene3d-align-1+p{section_digest(policy.alignment)}"
    version = (
        f"scene3d-change-1+det:{detector.version}+p{section_digest(policy.objects)}"
        f"{section_digest(policy.change)}+{align_version}"
    )
    tag = short_hash(version, tb, ta)
    key = f"{session_prefix(sid)}/changes/{tag}.json"
    existing = get_model(labeling, key, ChangeReport)
    if existing is not None:
        put_pointer(labeling, f"{session_prefix(sid)}/changes/current.json", {"tag": tag})
        return existing, False
    alignment, transform = align_scenes(
        pts_b, desc_b, pts_a, desc_a, policy.alignment, align_version
    )
    put_model_immutable(
        labeling,
        f"{session_prefix(sid)}/alignment/{short_hash(align_version, tb, ta)}.json",
        alignment,
    )
    if transform is None:
        raise AlignmentRejectedError(alignment)
    before = _phase_objects(labeling, sid, "before", cams_b, detector, policy, workdir)
    after = _phase_objects(labeling, sid, "after", cams_a, detector, policy, workdir)
    moved = [to_before_frame(o, transform) for o in after]
    report = ChangeReport(
        session_id=sid,
        model_version=version,
        before_reconstruction=tb,
        after_reconstruction=ta,
        alignment=alignment,
        metric=rb.metric,
        instances=(*before, *moved),
        changes=tuple(compare_objects(before, moved, policy.change)),
        created_at=now,
    )
    put_model_immutable(labeling, key, report)
    put_pointer(labeling, f"{session_prefix(sid)}/changes/current.json", {"tag": tag})
    return report, True


def current_report(labeling: ObjectStore, sid: str) -> ChangeReport:
    """지금 비교 결과. 없으면 Scene3dError."""
    ptr = get_pointer(labeling, f"{session_prefix(sid)}/changes/current.json")
    report = None
    if ptr:
        report = get_model(
            labeling, f"{session_prefix(sid)}/changes/{ptr['tag']}.json", ChangeReport
        )
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
