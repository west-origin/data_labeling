"""after 장면 → before 장면 정렬 (고정 구조 점의 닮음 변환, 척도 포함).

두 복원의 희소 3D 점을 기술자로 대응하고(`geometry.match_descriptors`), RANSAC Umeyama로 닮음
변환을 찾는다. 움직인 물체 위의 점은 이상점으로 빠진다. 대응 수·내점 수·내점 비율·잔차 RMS 중
하나라도 정책 문턱에 못 미치면 정렬을 거부한다: 잘못된 정렬은 모든 물체를 "움직였다"로 만들기
때문에 낮은 확신의 정렬로 비교하지 않는다.

실제 경로의 대안 (ADR 0066): before·after 블러 사진을 한 SfM으로 함께 복원하면 두 단계가 같은
좌표계에 놓여 이 변환은 항등에 가깝다. 이 모듈은 두 경우 모두에 쓴다 (항등도 같은 검사를 받는다).
"""

from __future__ import annotations

import numpy as np

from dlp_scene3d.geometry import F64, Similarity, match_descriptors, ransac_similarity
from dlp_scene3d.policy import AlignmentPolicy
from dlp_schema.scene3d import Alignment


def _contract(
    version: str,
    fit: Similarity | None,
    matches: int,
    inliers: int,
    rms: float | None,
    reason: str | None,
) -> Alignment:
    """정렬 결과 계약 (fit이 없으면 항등 자리값으로 둔다)."""
    s, r, t = (fit.s, fit.r, fit.t) if fit else (1.0, np.eye(3), np.zeros(3))
    rv = [float(v) for v in np.asarray(r).reshape(-1)]
    tv = [float(v) for v in np.asarray(t)]
    return Alignment(
        model_version=version,
        scale=float(s),
        rotation=(rv[0], rv[1], rv[2], rv[3], rv[4], rv[5], rv[6], rv[7], rv[8]),
        translation=(tv[0], tv[1], tv[2]),
        matches=matches,
        inliers=inliers,
        rms_m=rms,
        accepted=reason is None,
        reason=reason,
    )


def align_scenes(
    before_points: F64,
    before_desc: F64,
    after_points: F64,
    after_desc: F64,
    policy: AlignmentPolicy,
    version: str,
) -> tuple[Alignment, Similarity | None]:
    """(정렬 계약, 받아들인 변환 또는 None). 거부 이유는 계약의 reason에 적는다."""
    pairs = match_descriptors(after_desc, before_desc, policy.descriptor_ratio)
    src, dst = after_points[pairs[:, 0]], before_points[pairs[:, 1]]
    res = ransac_similarity(
        src,
        dst,
        threshold=policy.inlier_threshold_m,
        iterations=policy.ransac_iterations,
        seed=policy.seed,
    )
    m, k = len(pairs), len(res.inliers)
    reason: str | None = None
    if res.transform is None or k < policy.min_inliers:
        reason = f"정렬 내점 {k}개 < 최소 {policy.min_inliers}개 (대응 {m}개)"
    elif k / m < policy.min_inlier_ratio:
        reason = f"정렬 내점 비율 {k / m:.2f} < 최소 {policy.min_inlier_ratio}"
    elif res.rms is not None and res.rms > policy.max_rms_m:
        reason = f"정렬 잔차 RMS {res.rms:.4f} > 최대 {policy.max_rms_m}"
    accepted = res.transform if reason is None else None
    return _contract(version, res.transform, m, k, res.rms, reason), accepted
