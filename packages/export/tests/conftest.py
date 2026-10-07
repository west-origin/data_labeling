"""정답을 아는 내보내기 시나리오: 블러본(VFR) 하나와 검증 상태가 섞인 라벨.

dlp_export 테스트 공용 픽스처. 블러본은 `dlp_fixtures.video.vfr_times`로 만든 가변 프레임률 영상이라
프레임 시각(`times`)을 정확히 안다. 라벨(`scenario_labels`)은 그 프레임 시각에 키프레임을 두므로
"어느 프레임에 어떤 주석이 들어가야 하는지"가 정답으로 정해진다.

정답 요약 (세션 sid 기준, 라벨 ID 접미사):
- box(사람 승인 박스, 키프레임 3개 중 t[3]+5는 프레임 사이라 COCO에서 버림),
  unrev(미검수 박스), kp(사람 hand21 트랙, t[2]·t[6]), hs(표본 검증 손 상태 100~600 ms, 대걸레 쥠),
  tip(사람 대걸레 작용부 3D 궤적), tsc(사람 도구-표면 접촉 300~500 ms),
  act(사람 승인 행동 200~700 ms), task(사람 작업 구간 0~500 ms) — 기본 검증 정책으로 내보내는 것.
- blur(블러), seed(오류 삽입), seedfix(오류 삽입의 후손), blind(블라인드 측정) — 어떤 옵션으로도
  내보내면 안 되는 것.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlp_export.policy import ExportPolicy, load_policy
from dlp_fixtures.video import vfr_times, write_video
from dlp_media.storage import LocalStore, sha256_file
from dlp_privacy.policy import load_policy as load_privacy_policy
from dlp_privacy.runner import RenderMeta, operational_blur, render_hash, write_render_meta
from dlp_schema.labels import LabelRecord, Provenance, Source, Verification, VerificationState
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.session import Session
from dlp_schema.testing import FIXED_TIME, make_label, make_session

ROOT = Path(__file__).resolve().parents[3]
W, H = 64, 48


@pytest.fixture(scope="session")
def policy() -> ExportPolicy:
    """저장소의 config/policies/export.yaml."""
    return load_policy(ROOT)


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    """온톨로지 v1 (COCO 범주·LeRobot 어휘의 기준)."""
    return load_ontology(ROOT / "config/ontology/v1")


def model(state: VerificationState) -> dict[str, Any]:
    """모델 라벨용 `make_label` 인자 (출처 m1, 신뢰도 0.8, 주어진 검증 상태; 검수자 reviewer-7)."""
    v = (
        Verification(state=state, reviewer_id="reviewer-7", reviewed_at=FIXED_TIME)
        if state is not VerificationState.UNREVIEWED
        else Verification()
    )
    return {
        "provenance": Provenance(source=Source.MODEL, model_version="m1"),
        "confidence": 0.8,
        "verification": v,
    }


def human() -> dict[str, Any]:
    """사람이 만든(수정한) 라벨용 `make_label` 인자. 검수자 ID가 결과에 새지 않는지 볼 때 쓴다."""
    return {
        "verification": Verification(
            state=VerificationState.HUMAN_CORRECTED,
            reviewer_id="reviewer-7",
            reviewed_at=FIXED_TIME,
        )
    }


def hand21(x: float, y: float) -> list[dict[str, Any]]:
    """hand21 관절 21개: i번째 관절은 (x + i, y). 마지막 관절(pinky_tip)만 가려짐(v=1)."""
    return [{"x": x + i, "y": y, "visibility": 2 if i < 20 else 1} for i in range(21)]


@dataclass
class Scenario:
    """내보내기 시나리오 하나 (세션, 라벨 이력, 블러본 프레임 시각, 라벨링 저장소, 렌더 해시)."""

    session: Session
    labels: list[LabelRecord]
    times: list[int]  # 블러본 프레임 시각
    labeling: LocalStore
    render_hashes: dict[str, str]  # 스트림 → 블러본 렌더 해시 (load_source가 DB로 계산하는 값)


def scenario_labels(sid: str, times: list[int]) -> list[LabelRecord]:
    """세션 sid의 정답 라벨 이력 (모듈 docstring의 정답 요약 참고).

    공간 라벨 키프레임은 블러본 프레임 시각 `times[i]`(스트림 PTS)에 둔다. 구간 라벨은 마스터 시각
    (바디캠은 오프셋 0이라 같다).
    """
    t = times
    box = {"kind": "box_track", "entity_id": "cup_1", "class_id": "cup", "keyframes": [
        {"t_ms": t[2], "x": 8, "y": 8, "w": 16, "h": 12},
        {"t_ms": t[3] + 5, "x": 9, "y": 8, "w": 16, "h": 12},  # 프레임 사이 → COCO에서 버림
        {"t_ms": t[4], "x": 10, "y": 8, "w": 16, "h": 12},
    ]}  # fmt: skip
    kp = {"kind": "keypoint_track", "entity_id": "right_hand", "skeleton": "hand21",
          "hand": "right", "keyframes": [{"t_ms": t[2], "points": hand21(10, 20)},
                                         {"t_ms": t[6], "points": hand21(30, 20)}]}  # fmt: skip
    base = {"session_id": sid}
    return [
        make_label(box, label_id=f"{sid}-box", stream_id="bodycam", t_start_ms=t[2], t_end_ms=t[4],
                   **base, **model(VerificationState.HUMAN_APPROVED)),
        make_label({**box, "entity_id": "sponge_1", "class_id": "sponge"}, label_id=f"{sid}-unrev",
                   stream_id="bodycam", t_start_ms=t[2], t_end_ms=t[4], **base,
                   **model(VerificationState.UNREVIEWED)),
        make_label(kp, label_id=f"{sid}-kp", stream_id="bodycam", t_start_ms=t[2], t_end_ms=t[6],
                   **base, **human()),
        make_label({"kind": "hand_state", "hand": "right", "contact_target_kind": "tool",
                    "target_id": "mop_1", "grasp_type": "tool_grip", "role": "active"},
                   label_id=f"{sid}-hs", t_start_ms=100, t_end_ms=600, **base,
                   **model(VerificationState.SAMPLE_VERIFIED)),
        make_label({"kind": "trajectory3d", "entity_id": "mop_1", "part": "mop_head",
                    "frame": "camera", "source_3d": "mono_depth",
                    "samples": [{"t_ms": ms, "x": ms / 1000, "y": 0.0, "z": 1.0}
                                for ms in range(100, 700, 100)]},
                   label_id=f"{sid}-tip", stream_id="bodycam", t_start_ms=100, t_end_ms=600,
                   **base, **human()),
        make_label({"kind": "relation", "subject_id": "mop_1", "subject_part": "mop_head",
                    "predicate": "contact", "object_id": "floor_1"},
                   label_id=f"{sid}-tsc", t_start_ms=300, t_end_ms=500, **base, **human()),
        make_label({"kind": "action", "action_id": "a1", "hand": "right", "verb": "carry",
                    "t_approach_ms": 200, "t_end_ms": 700},
                   label_id=f"{sid}-act", t_start_ms=200, t_end_ms=700, **base,
                   **model(VerificationState.HUMAN_APPROVED)),
        make_label({"kind": "segment", "segment_id": "task1", "level": "task",
                    "ref_id": "floor_sweep_mop"},
                   label_id=f"{sid}-task", t_start_ms=0, t_end_ms=500, **base, **human()),
        make_label({"kind": "blur_track", "target": "face",
                    "keyframes": [{"t_ms": t[2], "x": 1, "y": 1, "w": 5, "h": 5}]},
                   label_id=f"{sid}-blur", stream_id="bodycam", t_start_ms=t[2], t_end_ms=t[2],
                   **base, **human()),
        make_label({**box, "entity_id": "cup_9"}, label_id=f"{sid}-seed", stream_id="bodycam",
                   t_start_ms=t[2], t_end_ms=t[4], seeded_error=True, **base, **human()),
        # 오류 삽입 사본을 검수자가 고친 후손 (seeded_error 표시는 없지만 운영 라벨이 아니다)
        make_label({**box, "entity_id": "cup_9", "class_id": "bucket"}, label_id=f"{sid}-seedfix",
                   stream_id="bodycam", t_start_ms=t[2], t_end_ms=t[4],
                   parent_label_id=f"{sid}-seed", **base, **human()),
        # 블라인드 측정 레코드
        make_label({"kind": "action", "action_id": "m1", "hand": "left", "verb": "carry",
                    "t_approach_ms": 0, "t_end_ms": 900},
                   label_id=f"{sid}-blind", t_start_ms=0, t_end_ms=900, measurement="blind",
                   **base, **human()),
    ]  # fmt: skip


def write_blurred(
    labeling: LocalStore,
    sid: str,
    times: list[int],
    tmp: Path,
    labels: list[LabelRecord] | None = None,
) -> str:
    """블러본과 렌더 기록(dlp privacy render가 남기는 것)을 쓴다. 기록한 렌더 해시를 돌려준다.

    labels: 렌더에 쓴 블러 라벨 이력 (없으면 scenario_labels(sid, times)).
    """
    video = tmp / f"{sid}.mp4"
    rng = np.random.default_rng(1)
    frames = ((t, rng.integers(0, 255, (H, W, 3), dtype=np.uint8)) for t in times)
    write_video(video, frames, width=W, height=H)
    sha = sha256_file(video)
    labeling.put_file(f"sessions/{sid}/blurred/bodycam.mp4", video, sha)
    history = scenario_labels(sid, times) if labels is None else labels
    return record_render(labeling, sid, "bodycam", history, sha, tmp)


def record_render(
    labeling: LocalStore, sid: str, stream: str, history: list[LabelRecord], sha: str, tmp: Path
) -> str:
    """렌더 기록 (블러 라벨 이력·프라이버시 정책 해시 + 블러본 파일 해시). 해시를 돌려준다."""
    digest = render_hash(operational_blur(history, stream), load_privacy_policy(ROOT))
    write_render_meta(labeling, sid, stream, RenderMeta(digest, sha), tmp)
    return digest


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    """세션 s1: VFR 블러본(약 1초)·렌더 기록을 로컬 라벨링 저장소에 쓴다.

    정답 라벨(`scenario_labels`)과 프레임 시각을 함께 돌려준다.
    """
    times = vfr_times(np.random.default_rng(4), 1000)
    labeling = LocalStore(tmp_path / "store", "dlp-labeling")
    digest = write_blurred(labeling, "s1", times, tmp_path)
    return Scenario(
        make_session("s1"), scenario_labels("s1", times), times, labeling, {"bodycam": digest}
    )
