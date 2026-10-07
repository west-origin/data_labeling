from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import pytest

from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import BlurScenario, render_qr, target_boxes_at
from dlp_media.pts import build_pts_index
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors import build_detectors
from dlp_privacy.detectors.codes import CodeDetector
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.pipeline import StreamResult, detect_video
from dlp_privacy.policy import PrivacyPolicy, TargetPolicy
from dlp_privacy.render import render_blurred
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME
from dlp_schema.validation import check_label

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ["face", "reflection", "document", "screen", "photo", "shipping_label"]


def oracle_policy(policy: PrivacyPolicy, detectors: dict[str, list[str]]) -> PrivacyPolicy:
    """대상마다 지정한 탐지기만 쓰는 정책 (margin은 실제 정책 값)."""
    targets = {
        t: TargetPolicy(margin=policy.targets[t].margin, detectors=tuple(names))
        for t, names in detectors.items()
    }
    return policy.model_copy(update={"targets": targets})


def run(
    blur: tuple[BlurScenario, Path],
    policy: PrivacyPolicy,
    detectors: dict[str, FrameDetector],
    assignment: dict[str, list[str]] | None = None,
    missing: dict[str, str] | None = None,
) -> StreamResult:
    assignment = assignment or {t: ["oracle"] for t in TARGETS}
    return detect_video(
        blur[1],
        session_id="s1",
        stream_id="bodycam",
        detectors=detectors,
        missing_detectors=missing or {},
        policy=oracle_policy(policy, assignment),
        ontology_version="1.0.0",
        now=FIXED_TIME,
    )


def label_box(labels: list[LabelRecord], target: str, t_ms: int) -> list[BoxKeyframe]:
    out: list[BoxKeyframe] = []
    for x in labels:
        p = x.payload
        if isinstance(p, BlurTrackPayload) and p.target == target:
            out += [k for k in p.keyframes if k.t_ms == t_ms and not k.outside]
    return out


def covered(labels: list[LabelRecord], scenario: BlurScenario) -> tuple[int, int]:
    """(정답 박스가 라벨 박스 안에 완전히 들어간 (대상, 프레임) 수, 전체 수)."""
    hit = total = 0
    for t in scenario.frame_times:
        for target, gt in target_boxes_at(scenario.labels, t).items():
            total += 1
            if any(
                k.x <= gt.x + 1e-6
                and k.y <= gt.y + 1e-6
                and gt.x + gt.w <= k.x + k.w + 1e-6
                and gt.y + gt.h <= k.y + k.h + 1e-6
                for k in label_box(labels, target, t)
            ):
                hit += 1
    return hit, total


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    return load_ontology(ROOT / "config" / "ontology" / "v1")


def test_perfect_stub_gives_full_recall(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy, ontology: Ontology
) -> None:
    result = run(blur, policy, {"oracle": OracleDetector("oracle", blur[0].labels)})
    hit, total = covered(result.labels, blur[0])
    assert total > 300 and hit == total  # 완료 기준: 재현율 100%
    assert {
        x.payload.target for x in result.labels if isinstance(x.payload, BlurTrackPayload)
    } == set(TARGETS)
    for label in result.labels:
        assert check_label(label, ontology) == []
        assert label.provenance.model_version and label.confidence is not None


def test_misses_are_filled_by_interpolation_and_flagged(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy
) -> None:
    oracle = OracleDetector("oracle", blur[0].labels, miss_rate=0.25, seed=3)
    result = run(blur, policy, {"oracle": oracle})
    hit, total = covered(result.labels, blur[0])
    assert hit == total
    gaps = [s for s in result.segments if s.reason == "track_gap"]
    assert gaps and all(s.priority == policy.review_priority.index("track_gap") for s in gaps)


def test_jittered_boxes_still_cover_nearly_all_area(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy
) -> None:
    """위치가 흔들린 탐지에서도 정답 박스 면적 대부분이 블러 안에 든다."""
    scenario = blur[0]
    labels = run(
        blur, policy, {"oracle": OracleDetector("oracle", scenario.labels, jitter_px=1.0, seed=4)}
    ).labels
    fractions: list[float] = []
    for t in scenario.frame_times:
        for target, gt in target_boxes_at(scenario.labels, t).items():
            mask = np.zeros((scenario.height, scenario.width), dtype=bool)
            for k in label_box(labels, target, t):
                mask[int(k.y) : int(np.ceil(k.y + k.h)), int(k.x) : int(np.ceil(k.x + k.w))] = True
            region = mask[int(gt.y) : int(gt.y + gt.h), int(gt.x) : int(gt.x + gt.w)]
            fractions.append(float(region.mean()))
    assert np.mean(fractions) > 0.97 and min(fractions) > 0.8


def test_blur_is_held_after_track_loss(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy
) -> None:
    scenario = blur[0]
    hold = policy.platform.blur_hold_ms
    # 얼굴을 1초 동안 놓친다 (max_gap_ms보다 길어 트랙이 끊긴다)
    oracle = OracleDetector("oracle", scenario.labels, miss_spans_ms=[("face", 1_000, 2_000)])
    result = run(blur, policy, {"oracle": oracle})
    face_times = [t for t in scenario.frame_times if label_box(result.labels, "face", t)]
    last_seen = max(t for t in scenario.frame_times if t < 1_000)
    seen_again = min(t for t in scenario.frame_times if t > 2_000)
    after_loss = [t for t in face_times if last_seen < t < seen_again]
    held = [t for t in after_loss if t <= last_seen + hold]
    pre_roll = [t for t in after_loss if t >= seen_again - hold]
    # 잃은 뒤 hold_ms까지 유지하고, 다시 찾기 hold_ms 전부터 미리 가린다. 그 사이는 비어 있다.
    assert held and max(held) > last_seen + hold - 50
    assert pre_roll and min(pre_roll) < seen_again - hold + 50
    assert sorted(held + pre_roll) == after_loss
    assert any(s.reason == "track_gap" and s.target == "face" for s in result.segments)

    # 화면 밖으로 나간 송장도 유지 시간만큼 블러가 남는다
    exit_t = max(
        t for t in scenario.frame_times if "shipping_label" in target_boxes_at(scenario.labels, t)
    )
    after = [
        t
        for t in scenario.frame_times
        if t > exit_t and label_box(result.labels, "shipping_label", t)
    ]
    assert after and max(after) <= exit_t + hold


def test_review_segments_cover_low_confidence_disagreement_reflection_and_missing(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy
) -> None:
    labels = blur[0].labels
    detectors: dict[str, FrameDetector] = {
        "strong": OracleDetector("strong", labels),
        "weak": OracleDetector(
            "weak", labels, targets={"face"}, miss_rate=0.5, score_range=(0.35, 0.5)
        ),
    }
    assignment = {"face": ["strong", "weak"], "reflection": ["strong"], "document": ["open_vocab"]}
    result = run(blur, policy, detectors, assignment, missing={"open_vocab": "GPU 없음"})
    reasons = {(s.target, s.reason) for s in result.segments}
    assert ("face", "disagreement") in reasons
    assert ("reflection", "reflection") in reasons
    assert ("document", "no_detector") in reasons
    [nd] = [s for s in result.segments if s.reason == "no_detector"]
    assert (nd.t_start_ms, nd.t_end_ms) == (blur[0].frame_times[0], blur[0].frame_times[-1])
    assert "GPU 없음" in nd.detail
    assert result.missing == {"document": "GPU 없음"}
    assert result.segments[0].reason == "no_detector"  # 우선순위가 가장 높다

    low = run(blur, policy, {"o": OracleDetector("o", labels, score_range=(0.31, 0.5))},
              {t: ["o"] for t in TARGETS})  # fmt: skip
    assert any(s.reason == "low_confidence" for s in low.segments)


def test_render_mosaics_targets_keeps_pts_and_leaves_rest(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy, tmp_path: Path
) -> None:
    scenario, src = blur
    result = run(blur, policy, {"oracle": OracleDetector("oracle", scenario.labels)})
    dst = tmp_path / "blurred.mp4"
    applied = render_blurred(
        src, dst, result.labels, mode="mosaic", min_block_px=6, blocks_per_box=5
    )
    assert applied > 0
    assert build_pts_index(dst).ms.tolist() == build_pts_index(src).ms.tolist()

    def frames(path: Path) -> dict[int, np.ndarray]:
        with av.open(str(path)) as c:
            return {
                round(float(f.time or 0) * 1000): f.to_ndarray(format="rgb24")
                for f in c.decode(video=0)
            }

    original, blurred = frames(src), frames(dst)
    checked = 0
    for t in scenario.frame_times[5::15]:
        for target, gt in target_boxes_at(scenario.labels, t).items():
            if target not in ("face", "reflection", "document") or gt.w < 20 or gt.h < 20:
                continue
            x, y, w, h = int(gt.x), int(gt.y), int(gt.w), int(gt.h)
            before = original[t][y : y + h, x : x + w].mean(axis=2)
            after = blurred[t][y : y + h, x : x + w].mean(axis=2)
            # 원본에는 어두운 눈·글줄 픽셀이 있고, 블러본에서는 블록 평균에 묻혀 사라진다
            # (배경도 블록에 섞이므로 배경 최솟값 약 55보다 조금 낮은 45를 기준으로 본다)
            assert before.min() <= 40 and after.min() > 45, (target, t, before.min(), after.min())
            checked += 1
    assert checked >= 5
    # 블러 대상이 없는 영역은 그대로 (코덱 손실 범위)
    t = scenario.frame_times[10]
    assert np.abs(original[t][200:240, 0:30].astype(int) - blurred[t][200:240, 0:30]).mean() < 4


def test_render_strips_audio(tmp_path: Path, policy: PrivacyPolicy) -> None:
    sync = generate_sync_scenario(1, recorded_at=FIXED_TIME, duration_ms=3_000)
    src = tmp_path / "with_audio.mp4"
    sync.write_video(src, "bodycam")
    dst = tmp_path / "blurred.mp4"
    render_blurred(src, dst, [], mode="solid", min_block_px=6, blocks_per_box=5)
    with av.open(str(src)) as c:
        assert c.streams.audio
    with av.open(str(dst)) as c:
        assert not c.streams.audio


def test_detector_factory_reports_unavailable_models(policy: PrivacyPolicy, tmp_path: Path) -> None:
    ready, missing = build_detectors(policy, tmp_path)  # 가중치가 없는 루트
    assert "codes" in ready
    assert "yunet" in missing and "make models" in missing["yunet"]
    assert "open_vocab" in missing and "reflection" in missing
    assert missing["reflection"].endswith("open_vocab, yunet")


def test_code_detector_finds_qr_as_shipping_label() -> None:
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    frame[40:200, 80:240] = render_qr("SHIP|1234-5678", 160)
    [det] = CodeDetector("codes").detect(frame, 0, 0.3)
    assert det.target == "shipping_label"
    assert 70 <= det.box.x <= 110 and 30 <= det.box.y <= 70 and det.box.w > 60


@pytest.mark.skipif(
    not (ROOT / "data/models/face_detection_yunet_2023mar.onnx").is_file(),
    reason="YuNet 가중치 없음 (make models)",
)
def test_yunet_loads_and_runs(policy: PrivacyPolicy, blur: tuple[BlurScenario, Path]) -> None:
    ready, _ = build_detectors(policy, ROOT)
    yunet = ready["yunet"]
    assert yunet.version.startswith("yunet-")
    assert yunet.detect(np.zeros((240, 320, 3), dtype=np.uint8), 0, 0.3) == []
    # 합성 얼굴은 실제 얼굴이 아니므로 결과 개수는 보지 않고, 박스 형식만 확인한다
    for d in yunet.detect(blur[0].frames[0], 0, 0.3):
        assert d.target == "face" and d.box.w > 0
