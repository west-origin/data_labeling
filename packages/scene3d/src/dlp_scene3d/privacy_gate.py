"""사진 프라이버시 게이트: 탐지 → 사람 승인 → 블러 렌더 → 하류의 블러본 확인 (ADR 0066).

영상 게이트(`dlp_privacy`)와 같은 약속을 사진에 적용한다.
- 탐지는 `dlp_privacy`의 탐지기(`FrameDetector`)를 사진마다 한 번 부른다. 대상·문턱·여유는
  `privacy.yaml`을 그대로 쓴다. 탐지를 거르거나 지우지 않는다 (재현율 우선).
- 승인은 원본 접근 권한자(`review.yaml reviewers.privacy`, CLI가 확인)가 사진 **전부**의 확정 상자를
  내야 한다 (표본 승인 없음). 승인은 최신 탐지 세대에 묶인다.
- 렌더는 승인된 상자로만 한다 (`dlp_privacy.render.apply_blur`, 정책의 렌더 방식). 블러본은 PNG라
  EXIF가 없다. 다시 탐지해 세대가 바뀌면 렌더 표시를 무효로 바꾼다.
- 하류(SfM·스플랫·물체 탐지·비교·뷰어)는 `open_blurred`만으로 사진을 받는다. 렌더 표시가 유효하고
  파일 sha256이 표시와 같아야 한다. 이 모듈 밖에서 원본 사진을 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from dlp_media.storage import ObjectStore, sha256_file
from dlp_privacy.detection import Detection, FrameDetector
from dlp_privacy.errors import RenderNotCurrentError
from dlp_privacy.geometry import Box
from dlp_privacy.pipeline import detection_policy_digest
from dlp_privacy.policy import PrivacyPolicy
from dlp_privacy.render import apply_blur
from dlp_scene3d.layout import (
    get_model,
    get_pointer,
    phase_prefix,
    put_bytes_immutable,
    put_model_immutable,
    put_pointer,
    short_hash,
)
from dlp_scene3d.photos import capture_key, decode_photo, raw_photo_key
from dlp_schema.scene3d import (
    PhotoBox,
    PhotoCapture,
    PhotoPrivacyApproval,
    PhotoPrivacyDetection,
    PhotoRender,
    ScenePhase,
)

DETECTION_VERSION = "photo-privacy-1"


class PrivacyGateError(RuntimeError):
    """승인·렌더 전제가 맞지 않는다 (탐지 없음, 승인 없음·낡음, 상자 누락)."""


class PhotoPrivacyDetector(Protocol):
    """사진 한 장의 블러 대상 탐지기. 실제는 `FramePhotoDetector`, stub은 `OraclePhotoDetector`."""

    name: str
    version: str

    def detect(self, photo_id: str, image: NDArray[np.uint8], threshold: float) -> list[Detection]:
        """(H, W, 3) RGB 사진 한 장의 탐지 (문턱 미만은 내지 않는다)."""
        ...


@dataclass
class FramePhotoDetector:
    """`dlp_privacy`의 프레임 탐지기를 사진에 쓰는 어댑터 (사진마다 시각 0으로 부른다)."""

    inner: FrameDetector
    name: str = field(init=False)
    version: str = field(init=False)

    def __post_init__(self) -> None:
        """이름·버전은 감싼 탐지기 값 (privacy.yaml detectors 키, 가중치 버전)."""
        self.name, self.version = self.inner.name, self.inner.version

    def detect(self, photo_id: str, image: NDArray[np.uint8], threshold: float) -> list[Detection]:
        """사진 사이에 상태(간격 캐시)를 지우고 한 번 탐지한다."""
        reset = getattr(self.inner, "reset", None)
        if callable(reset):
            reset()
        return self.inner.detect(image, 0, threshold)


@dataclass
class OraclePhotoDetector:
    """정답 상자를 그대로 내는 CPU stub (합성 픽스처·테스트용). boxes: photo_id → (대상, 상자)."""

    boxes: Mapping[str, list[tuple[str, tuple[float, float, float, float]]]]
    name: str = "oracle"
    version: str = "oracle-photo-1"

    def detect(self, photo_id: str, image: NDArray[np.uint8], threshold: float) -> list[Detection]:
        """그 사진의 정답 상자 (점수 0.9)."""
        return [Detection(t, Box(*b), 0.9, self.name) for t, b in self.boxes.get(photo_id, [])]


def detection_version(detectors: Mapping[str, PhotoPrivacyDetector], policy: PrivacyPolicy) -> str:
    """탐지 세대 버전: 탐지기 이름·버전(이름순)과 탐지 정책 해시."""
    parts = ",".join(f"{n}:{d.version}" for n, d in sorted(detectors.items()))
    return f"{DETECTION_VERSION}+{parts}+p{detection_policy_digest(policy)}"


def _latest_key(sid: str, phase: ScenePhase) -> str:
    return f"{phase_prefix(sid, phase)}/privacy/latest.json"


def _approved_key(sid: str, phase: ScenePhase) -> str:
    return f"{phase_prefix(sid, phase)}/privacy/approved.json"


def detection_key(sid: str, phase: ScenePhase, version: str) -> str:
    """원본 버킷의 탐지 세대 키."""
    return f"{phase_prefix(sid, phase)}/privacy/detections/{short_hash(version)}.json"


def approval_key(sid: str, phase: ScenePhase, approval_id: str) -> str:
    """원본 버킷의 승인 키."""
    return f"{phase_prefix(sid, phase)}/privacy/approvals/{approval_id}.json"


def render_key(sid: str, phase: ScenePhase) -> str:
    """라벨링 버킷의 렌더 표시 키."""
    return f"{phase_prefix(sid, phase)}/blurred/render.json"


def load_capture(store: ObjectStore, sid: str, phase: ScenePhase) -> PhotoCapture:
    """수집 기록. 없으면 PrivacyGateError."""
    cap = get_model(store, capture_key(sid, phase), PhotoCapture)
    if cap is None:
        raise PrivacyGateError(f"{sid} {phase}: 수집 기록이 없습니다 (dlp scene3d ingest)")
    return cap


def _fetch_raw(raw: ObjectStore, sid: str, cap: PhotoCapture, workdir: Path) -> dict[str, Path]:
    """원본 사진을 workdir로 받는다 (감사 저장소로 받으므로 접근이 기록된다)."""
    out: dict[str, Path] = {}
    workdir.mkdir(parents=True, exist_ok=True)
    for e in cap.photos:
        dest = workdir / f"{e.photo_id}{e.ext}"
        raw.get_file(raw_photo_key(sid, cap.phase, e), dest)
        out[e.photo_id] = dest
    return out


def _detect_photo(
    img: NDArray[np.uint8],
    photo_id: str,
    detectors: Mapping[str, PhotoPrivacyDetector],
    policy: PrivacyPolicy,
) -> tuple[PhotoBox, ...]:
    """사진 한 장의 블러 상자 (정책 대상만, 대상 여유를 더하고 화면 안으로 자른다)."""
    h, w = img.shape[:2]
    boxes: list[PhotoBox] = []
    for name in sorted(detectors):
        for d in detectors[name].detect(photo_id, img, policy.detection_threshold):
            target = policy.targets.get(d.target)
            if target is None:
                continue
            box = d.box.padded(target.margin).clipped(w, h)
            if box is not None and box.w > 0 and box.h > 0:
                boxes.append(
                    PhotoBox(target=d.target, x=box.x, y=box.y, w=box.w, h=box.h, score=d.score)
                )
    return tuple(sorted(boxes, key=lambda b: (b.target, b.x, b.y)))


def detect_capture(
    raw: ObjectStore,
    labeling: ObjectStore,
    sid: str,
    phase: ScenePhase,
    detectors: Mapping[str, PhotoPrivacyDetector],
    no_detector: list[str],
    policy: PrivacyPolicy,
    workdir: Path,
    now: datetime,
) -> tuple[PhotoPrivacyDetection, bool]:
    """(탐지 세대, 새로 탐지했는가). 최신 세대 버전이 같으면 건너뛴다 (멱등).

    새 세대를 쓰면 원본 버킷의 최신 표시를 옮기고, 라벨링 버킷 렌더 표시를 무효로 바꾼다
    (이전 승인으로 만든 블러본을 하류가 더는 쓰지 않게).
    """
    version = detection_version(detectors, policy)
    key = detection_key(sid, phase, version)
    latest = get_pointer(raw, _latest_key(sid, phase))
    current = get_model(raw, key, PhotoPrivacyDetection)
    if latest and latest.get("detection_version") == version and current:
        return current, False
    cap = load_capture(raw, sid, phase)
    files = _fetch_raw(raw, sid, cap, workdir)
    boxes = {
        pid: _detect_photo(decode_photo(path)[0], pid, detectors, policy)
        for pid, path in files.items()
    }
    det = current or PhotoPrivacyDetection(
        session_id=sid,
        phase=phase,
        model_version=version,
        boxes=boxes,
        no_detector=tuple(sorted(no_detector)),
        created_at=now,
    )
    put_model_immutable(raw, key, det)
    put_pointer(raw, _latest_key(sid, phase), {"detection_version": version, "key": key})
    if labeling.head(render_key(sid, phase)) is not None:
        put_pointer(
            labeling,
            render_key(sid, phase),
            PhotoRender(
                session_id=sid,
                phase=phase,
                render_tag=None,
                approval_id=None,
                files={},
                rendered_at=now,
            ),
        )
    return det, True


def latest_detection(raw: ObjectStore, sid: str, phase: ScenePhase) -> PhotoPrivacyDetection:
    """최신 탐지 세대. 없으면 PrivacyGateError."""
    latest = get_pointer(raw, _latest_key(sid, phase))
    det = get_model(raw, latest["key"], PhotoPrivacyDetection) if latest else None
    if det is None:
        raise PrivacyGateError(f"{sid} {phase}: 블러 탐지가 없습니다 (dlp scene3d privacy detect)")
    return det


def write_review_pack(raw: ObjectStore, sid: str, phase: ScenePhase, out: Path) -> Path:
    """원본 접근 권한자용 검수 묶음: 원본 사진(감사 기록)과 최신 탐지 상자 `boxes.json`.

    검수자는 boxes.json에서 상자를 고치거나(지우기·그리기) 그대로 두고 `approve`에 넘긴다.
    """
    det = latest_detection(raw, sid, phase)
    cap = load_capture(raw, sid, phase)
    out.mkdir(parents=True, exist_ok=True)
    _fetch_raw(raw, sid, cap, out)
    data = {
        "detection_version": det.model_version,
        "no_detector": list(det.no_detector),
        "boxes": {pid: [b.model_dump(mode="json") for b in bs] for pid, bs in det.boxes.items()},
    }
    path = out / "boxes.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def approve_capture(
    raw: ObjectStore,
    sid: str,
    phase: ScenePhase,
    reviewer: str,
    boxes: Mapping[str, list[PhotoBox]],
    detection_version_seen: str,
    now: datetime,
) -> PhotoPrivacyApproval:
    """검수자가 확정한 상자로 승인을 남긴다 (추가만). 검수자 권한은 호출자(CLI)가 먼저 확인한다.

    Raises:
        PrivacyGateError: 검수자가 본 탐지 세대가 최신이 아니거나, 상자 목록의 사진이 수집 기록과
            다를 때 (모든 사진을 다 봐야 한다).
    """
    det = latest_detection(raw, sid, phase)
    if detection_version_seen != det.model_version:
        raise PrivacyGateError("검수한 탐지 세대가 최신이 아닙니다. 검수 묶음을 다시 받으세요")
    cap = load_capture(raw, sid, phase)
    expected = {e.photo_id for e in cap.photos}
    if set(boxes) != expected:
        raise PrivacyGateError(
            f"모든 사진의 상자를 내야 합니다 (빠짐 {sorted(expected - set(boxes))}, "
            f"모르는 사진 {sorted(set(boxes) - expected)})"
        )
    final = {pid: tuple(boxes[pid]) for pid in sorted(boxes)}
    body = json.dumps(
        {k: [b.model_dump(mode="json") for b in v] for k, v in final.items()}, sort_keys=True
    )
    approval = PhotoPrivacyApproval(
        approval_id=short_hash(det.model_version, reviewer, body, now.isoformat()),
        session_id=sid,
        phase=phase,
        detection_version=det.model_version,
        reviewer=reviewer,
        boxes=final,
        approved_at=now,
    )
    put_model_immutable(raw, approval_key(sid, phase, approval.approval_id), approval)
    put_pointer(raw, _approved_key(sid, phase), {"approval_id": approval.approval_id})
    return approval


def current_approval(raw: ObjectStore, sid: str, phase: ScenePhase) -> PhotoPrivacyApproval:
    """최신 탐지 세대에 묶인 지금 승인. 없거나 낡았으면 PrivacyGateError."""
    det = latest_detection(raw, sid, phase)
    ptr = get_pointer(raw, _approved_key(sid, phase))
    approval = None
    if ptr:
        key = approval_key(sid, phase, ptr["approval_id"])
        approval = get_model(raw, key, PhotoPrivacyApproval)
    if approval is None or approval.detection_version != det.model_version:
        raise PrivacyGateError(f"{sid} {phase}: 최신 탐지 세대의 사람 승인이 없습니다")
    return approval


def render_capture(
    raw: ObjectStore,
    labeling: ObjectStore,
    sid: str,
    phase: ScenePhase,
    policy: PrivacyPolicy,
    workdir: Path,
    now: datetime,
) -> tuple[PhotoRender, bool]:
    """(렌더 표시, 새로 렌더했는가). 지금 승인으로만 블러하고 라벨링 버킷에 PNG로 쓴다 (멱등)."""
    approval = current_approval(raw, sid, phase)
    mode, r = policy.platform.render_mode, policy.render
    tag = short_hash(approval.approval_id, mode, str(r.min_block_px), str(r.blocks_per_box))
    existing = get_model(labeling, render_key(sid, phase), PhotoRender)
    if existing is not None and existing.render_tag == tag:
        return existing, False
    cap = load_capture(raw, sid, phase)
    files: dict[str, str] = {}
    for pid, path in _fetch_raw(raw, sid, cap, workdir).items():
        img = decode_photo(path)[0]
        for b in approval.boxes[pid]:
            apply_blur(img, Box(b.x, b.y, b.w, b.h), mode, r.min_block_px, r.blocks_per_box)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"PNG 인코딩 실패: {pid}")
        data = buf.tobytes()
        put_bytes_immutable(
            labeling, f"{phase_prefix(sid, phase)}/blurred/{tag}/{pid}.png", data, ".png"
        )
        files[pid] = hashlib.sha256(data).hexdigest()
    record = PhotoRender(
        session_id=sid,
        phase=phase,
        render_tag=tag,
        approval_id=approval.approval_id,
        files=files,
        rendered_at=now,
    )
    put_pointer(labeling, render_key(sid, phase), record)
    return record, True


@dataclass(frozen=True)
class BlurredPhoto:
    """하류 단계 입력: 승인된 블러 사진 한 장의 로컬 파일 (EXIF 방향 적용 후 크기)."""

    photo_id: str
    path: Path
    width: int
    height: int

    def load(self) -> NDArray[np.uint8]:
        """(H, W, 3) RGB 배열."""
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RenderNotCurrentError(f"블러 사진을 읽지 못했습니다: {self.photo_id}")
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def open_blurred(
    labeling: ObjectStore, sid: str, phase: ScenePhase, workdir: Path
) -> tuple[str, list[BlurredPhoto]]:
    """(렌더 태그, 블러 사진 목록). 하류 단계가 사진을 받는 유일한 길.

    Raises:
        RenderNotCurrentError: 렌더 표시가 없거나 무효(재탐지로 승인이 낡음)이거나, 받은 파일 해시가
            표시와 다를 때.
    """
    record = get_model(labeling, render_key(sid, phase), PhotoRender)
    if record is None or record.render_tag is None:
        raise RenderNotCurrentError(f"{sid} {phase}: 지금 승인된 블러 사진이 없습니다")
    cap = load_capture(labeling, sid, phase)
    out: list[BlurredPhoto] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for e in cap.photos:
        dest = workdir / f"{e.photo_id}.png"
        labeling.get_file(
            f"{phase_prefix(sid, phase)}/blurred/{record.render_tag}/{e.photo_id}.png", dest
        )
        if sha256_file(dest) != record.files.get(e.photo_id):
            raise RenderNotCurrentError(
                f"{sid} {phase}: 블러 사진 해시가 렌더 기록과 다릅니다 ({e.photo_id})"
            )
        out.append(BlurredPhoto(e.photo_id, dest, e.width, e.height))
    return record.render_tag, out
