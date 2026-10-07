from __future__ import annotations

import shutil
from datetime import timedelta
from pathlib import Path

import av
import numpy as np
import pytest

from dlp_fixtures.sync import generate_sync_scenario
from dlp_fixtures.video import BlurScenario, render_qr, target_boxes_at, vfr_times, write_video
from dlp_media.pts import build_pts_index
from dlp_models.owlv2 import OwlDetection
from dlp_privacy.detection import FrameDetector
from dlp_privacy.detectors import build_detectors
from dlp_privacy.detectors.codes import CodeDetector
from dlp_privacy.detectors.open_vocab import OpenVocabDetector
from dlp_privacy.detectors.oracle import OracleDetector
from dlp_privacy.geometry import Box
from dlp_privacy.pipeline import StreamResult, detect_video, model_version
from dlp_privacy.policy import PrivacyPolicy, ReviewReason, TargetPolicy
from dlp_privacy.render import blur_tracks, render_blurred
from dlp_privacy.review import ReviewSegment
from dlp_privacy.runner import last_detection, merge_segments
from dlp_schema.labels import BlurTrackPayload, BoxKeyframe, LabelRecord, Provenance, Source
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.testing import FIXED_TIME, make_label
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


def test_unblurred_gap_between_split_tracks_is_flagged(
    blur: tuple[BlurScenario, Path], policy: PrivacyPolicy
) -> None:
    """회귀: 오래 끊겨 트랙이 나뉘면 그 사이 블러 없는 프레임을 검수 구간으로 낸다."""
    scenario = blur[0]
    oracle = OracleDetector("oracle", scenario.labels, miss_spans_ms=[("face", 1_000, 2_000)])
    result = run(blur, policy, {"oracle": oracle})
    unblurred = [
        t
        for t in scenario.frame_times
        if 1_000 < t < 2_000 and not label_box(result.labels, "face", t)
    ]
    assert unblurred  # 유지 시간 사이에 블러가 빠진 프레임이 있다
    splits = [
        s
        for s in result.segments
        if s.reason == "track_gap" and s.target == "face" and s.detail == "split"
    ]
    assert splits and all(any(s.t_start_ms <= t <= s.t_end_ms for s in splits) for t in unblurred)
    # 떨어진 거리가 split_review_ms보다 길면 내지 않는다 (다른 등장으로 본다)
    far = policy.model_copy(
        update={"tracker": policy.tracker.model_copy(update={"split_review_ms": 100})}
    )
    again = run(blur, far, {"oracle": oracle})
    assert not [s for s in again.segments if s.detail == "split"]


def test_render_interpolates_between_sparse_keyframes() -> None:
    """회귀: 검수자가 CVAT에서 고친 트랙은 키프레임만 남는다. 블러본은 CVAT처럼 보간해야 한다."""
    kfs = (
        BoxKeyframe(t_ms=0, x=0, y=0, w=10, h=10),
        BoxKeyframe(t_ms=100, x=100, y=50, w=30, h=10),
        BoxKeyframe(t_ms=200, x=0, y=0, w=0, h=0, outside=True),
        BoxKeyframe(t_ms=300, x=5, y=5, w=10, h=10),
    )
    label = make_label(
        BlurTrackPayload(target="face", keyframes=kfs), t_end_ms=300, stream_id="bodycam"
    )
    [track] = blur_tracks([label], list(range(0, 401, 50)))  # 고른 간격: 시각 비율 = 프레임 비율
    assert track.box_at(50) == Box(50, 25, 20, 10)  # 선형 보간
    assert track.box_at(150) == Box(100, 50, 30, 10)  # 다음이 화면 밖이면 직전 박스 유지
    assert track.box_at(250) is None  # 화면 밖
    assert track.box_at(400) == Box(5, 5, 10, 10)  # 마지막 뒤는 유지
    assert track.box_at(-1) is None


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
        src,
        dst,
        result.labels,
        mode="mosaic",
        min_block_px=6,
        blocks_per_box=5,
        encoder_rate=policy.render.encoder_rate,
        crf=policy.render.crf,
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
    render_blurred(
        src, dst, [], mode="solid", min_block_px=6, blocks_per_box=5,
        encoder_rate=policy.render.encoder_rate, crf=policy.render.crf,
    )  # fmt: skip
    with av.open(str(src)) as c:
        assert c.streams.audio
    with av.open(str(dst)) as c:
        assert not c.streams.audio


def test_detector_factory_reports_unavailable_models(policy: PrivacyPolicy, tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    shutil.copy(ROOT / "config/models.yaml", tmp_path / "config/models.yaml")
    ready, missing = build_detectors(policy, tmp_path)  # 가중치가 없는 루트
    assert "codes" in ready
    assert "yunet" in missing and "make models" in missing["yunet"]
    assert "open_vocab" in missing and "reflection" in missing
    assert missing["reflection"].endswith("open_vocab, yunet")


def test_code_detector_finds_qr_as_shipping_label() -> None:
    frame = np.full((240, 320, 3), 200, dtype=np.uint8)
    frame[40:200, 80:240] = render_qr("SHIP|1234-5678", 160)
    [det] = CodeDetector("codes", 0.9).detect(frame, 0, 0.3)
    assert det.target == "shipping_label"
    assert 70 <= det.box.x <= 110 and 30 <= det.box.y <= 70 and det.box.w > 60


class FakeOwl:
    def __init__(self) -> None:
        self.queries = ["a printed document", "a mirror"]
        self.calls = 0

    def detect(self, image: np.ndarray, thresholds: list[float]) -> list[OwlDetection]:
        self.calls += 1
        return [
            OwlDetection(0, (10.0, 10.0, 50.0, 40.0), 0.2),
            OwlDetection(1, (100.0, 20.0, 60.0, 80.0), 0.05),
        ]


def test_open_vocab_detector_runs_every_stride_and_holds_results() -> None:
    fake = FakeOwl()
    det = OpenVocabDetector(
        "open_vocab", fake, ["document", "reflective_surface"], version="owl-x",
        frame_stride_ms=500, score_threshold=0.05, score_full=0.4,
    )  # fmt: skip
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    per_frame = [det.detect(img, t, 0.3) for t in range(0, 1_000, 33)]
    assert fake.calls == 2  # 0 ms, 528 ms
    assert all(len(found) == 1 for found in per_frame)  # 0.05/0.4 < 0.3은 빠진다
    [doc] = per_frame[0]
    assert doc.target == "document" and doc.score == pytest.approx(0.5)
    assert det.detect(img, 990, 0.1)[1].target == "reflective_surface"  # 같은 시각: 추론 없음
    assert fake.calls == 2
    det.detect(img, 0, 0.3)  # 시간이 거꾸로 가면 새 영상
    assert fake.calls == 3
    # 짧은 영상(한 번만 추론, 마지막 추론 시각 0) 다음 영상도 0 ms부터 시작한다.
    # 초기화해야 새로 추론한다
    det.reset()
    det.detect(img, 0, 0.3)
    assert fake.calls == 4


def test_pipeline_resets_open_vocab_cache_per_video(
    policy: PrivacyPolicy, blur: tuple[BlurScenario, Path]
) -> None:
    fake = FakeOwl()
    det = OpenVocabDetector(
        "open_vocab", fake, ["document", "reflective_surface"], version="owl-x",
        frame_stride_ms=10_000, score_threshold=0.05, score_full=0.4,
    )  # fmt: skip
    p = oracle_policy(policy, {"document": ["open_vocab"]})
    for _ in range(2):  # 같은 탐지기로 영상 두 개 (세션의 두 스트림과 같다)
        detect_video(
            blur[1], session_id="s", stream_id="v", detectors={"open_vocab": det},
            missing_detectors={}, policy=p, ontology_version="1.0.0", now=FIXED_TIME,
        )  # fmt: skip
    assert fake.calls == 2  # 영상마다 한 번씩


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


def test_render_interpolates_by_frame_index_on_vfr_like_cvat(tmp_path: Path) -> None:
    """회귀(감사 4-7): CVAT는 프레임 번호로 보간한다. VFR 영상에서 시각 비율로 보간하면 검수
    화면과 블러본의 박스가 다르다. 블러본도 PTS 프레임 순서로 보간해야 한다."""
    times = [0, 10, 20, 30, 300]  # 고르지 않은 간격 (VFR)
    w, h = 200, 40
    src = tmp_path / "vfr.mp4"
    write_video(src, ((t, np.zeros((h, w, 3), dtype=np.uint8)) for t in times), width=w, height=h)
    assert [round(t) for t in build_pts_index(src).ms] == times
    kfs = (
        BoxKeyframe(t_ms=0, x=0, y=10, w=20, h=20),
        BoxKeyframe(t_ms=300, x=160, y=10, w=20, h=20),
    )
    label = make_label(
        BlurTrackPayload(target="face", keyframes=kfs), t_end_ms=300, stream_id="bodycam"
    )
    [track] = blur_tracks([label], times)
    # t=20 ms는 다섯 프레임 중 세 번째 (번호 2/4) → x = 160 * 0.5 (시각 비율이면 160 * 20/300)
    assert track.box_at(20) == Box(80, 10, 20, 20)
    dst = tmp_path / "blurred.mp4"
    render_blurred(
        src, dst, [label], mode="solid", min_block_px=6, blocks_per_box=5, encoder_rate=30, crf=0
    )
    with av.open(str(dst)) as c:
        frames = {
            round(float(f.time or 0) * 1000): f.to_ndarray(format="rgb24")
            for f in c.decode(video=0)
        }
    third = frames[20]
    assert third[20, 85:95].mean() > 100  # 프레임 번호 보간 위치에 블러
    assert third[20, 12:20].mean() < 30  # 시각 비율 위치(x≈10.7)에는 없다


def test_vfr_fixture_positions_match_frame_order() -> None:
    """VFR 픽스처의 모든 프레임에서 보간 위치 = 프레임 번호 비율."""
    times = vfr_times(np.random.default_rng(3), 2_000)
    kfs = (
        BoxKeyframe(t_ms=times[0], x=0, y=0, w=10, h=10),
        BoxKeyframe(t_ms=times[-1], x=1000, y=0, w=10, h=10),
    )
    label = make_label(
        BlurTrackPayload(target="face", keyframes=kfs), t_end_ms=times[-1], stream_id="bodycam"
    )
    [track] = blur_tracks([label], times)
    n = len(times) - 1
    for i, t in enumerate(times):
        box = track.box_at(t)
        assert box is not None and abs(box.x - 1000 * i / n) < 1e-6


def test_model_version_changes_with_detection_policy(policy: PrivacyPolicy) -> None:
    """회귀(감사 4-3): 문턱·여유·트래커·유지 시간·탐지기 설정이 바뀌면 다시 탐지해야 한다."""
    oracle: dict[str, FrameDetector] = {"oracle": OracleDetector("oracle", [])}
    base = oracle_policy(policy, {t: ["oracle"] for t in TARGETS})
    v0 = model_version(oracle, base)
    assert v0 == model_version(oracle, base)  # 같은 정책이면 같은 버전 (멱등)
    face = base.targets["face"]
    changed = [
        base.model_copy(update={"detection_threshold": 0.05}),
        base.model_copy(
            update={"targets": {**base.targets, "face": face.model_copy(update={"margin": 0.8})}}
        ),
        base.model_copy(update={"tracker": base.tracker.model_copy(update={"max_gap_ms": 1})}),
        base.model_copy(update={"platform": base.platform.model_copy(update={"blur_hold_ms": 2})}),
    ]
    yunet = policy.detectors["yunet"]
    changed.append(
        policy.model_copy(
            update={
                "detectors": {**policy.detectors, "yunet": yunet.model_copy(update={"top_k": 7})}
            }
        )
    )
    ov = policy.detectors["open_vocab"]
    changed.append(
        policy.model_copy(
            update={
                "detectors": {
                    **policy.detectors,
                    "open_vocab": ov.model_copy(
                        update={"queries": {"a mirror": "reflective_surface"}}
                    ),
                }
            }
        )
    )
    versions = {model_version(oracle, p) for p in changed[:4]}
    assert v0 not in versions and len(versions) == 4
    real = model_version({}, policy)
    assert all(model_version({}, p) != real for p in changed[4:])
    # 렌더 설정은 탐지 버전에 넣지 않는다 (블러본 해시가 맡는다)
    assert (
        model_version(
            oracle, base.model_copy(update={"render": base.render.model_copy(update={"crf": 30})})
        )
        == v0
    )


def test_last_detection_ignores_seeded_and_measurement_records() -> None:
    """회귀(감사 4-6): 오류 삽입 계획이 만든 모델 출처 블러 사본(지금 시각)이 마지막 탐지 시각을
    옮겨 승인을 막았다. 운영 블러가 아닌 레코드와 그 후손은 세지 않는다."""
    kfs = (BoxKeyframe(t_ms=0, x=0, y=0, w=10, h=10),)
    model = Provenance(source=Source.MODEL, model_version="det-1")
    payload = BlurTrackPayload(target="face", keyframes=kfs)
    base = make_label(payload, "b1", 0, 0, stream_id="bodycam", provenance=model, confidence=0.9)
    later = FIXED_TIME + timedelta(days=1)
    seeded = base.model_copy(
        update={"label_id": "seed-1", "seeded_error": True, "created_at": later}
    )
    seeded_fix = base.model_copy(
        update={"label_id": "seed-1-fix", "parent_label_id": "seed-1", "created_at": later}
    )
    measured = base.model_copy(
        update={"label_id": "m-1", "measurement": "blind", "created_at": later}
    )
    assert last_detection([base, seeded, seeded_fix, measured], "bodycam") == FIXED_TIME
    assert (
        last_detection(
            [base, seeded, base.model_copy(update={"label_id": "b2", "created_at": later})],
            "bodycam",
        )
        == later
    )


def _seg(reason: ReviewReason, detail: str = "", t: int = 0) -> ReviewSegment:
    return ReviewSegment(
        stream_id="bodycam", target="face", reason=reason, t_start_ms=t, t_end_ms=t + 10,
        priority=0, detail=detail,
    )  # fmt: skip


def test_merge_segments_keeps_detector_segments_when_only_model_reruns() -> None:
    """회귀(감사 4-8): 재학습 모델만 다시 돌면 검수 우선 구간 파일을 덮어써 탐지기 구간이
    사라졌다."""
    old = [_seg("track_gap"), _seg("trained_model", "m1"), _seg("trained_model", "m0", 5)]
    new = [_seg("trained_model", "m1", 20)]
    merged = merge_segments(old, new, detector_rerun=False, rerun_models={"m1"}, live_models={"m1"})
    assert {(s.reason, s.detail, s.t_start_ms) for s in merged} == {
        ("track_gap", "", 0), ("trained_model", "m1", 20),
    }  # fmt: skip
    rerun = merge_segments(
        old, [_seg("low_confidence")], detector_rerun=True, rerun_models=set(), live_models={"m1"}
    )
    assert {(s.reason, s.detail) for s in rerun} == {
        ("low_confidence", ""),
        ("trained_model", "m1"),
    }
