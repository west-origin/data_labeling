"""고정 클래스 도구 단계(`adapters.yolox_tools`, ADR 0060) 테스트. 가중치 없이 CPU stub 탐지기로
돈다.

정답: stub(`ScriptedToolDetector`)이 내는 박스를 정해 두고, 합성 영상(`dlp_fixtures.video`)의 PTS로
추론 시각을 계산한다. 보는 것: 박스 트랙 라벨 ID·entity_id·클래스, 버전(가중치·정책 절 해시·착용자
억제 태그), 클래스별 문턱, 착용자 몸 억제, 기본 정책에서 단계 미생성, OWLv2 질의에서 클래스 빼기,
정책 classes의 온톨로지·학습 설정 일치.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

import dlp_prelabel.adapters.owl_objects as owl_module
from dlp_fixtures.video import generate_blur_scenario
from dlp_models.yolox import YoloxDetection
from dlp_prelabel.adapters.owl_objects import OwlObjects
from dlp_prelabel.adapters.stubs import ScriptedToolDetector
from dlp_prelabel.adapters.yolox_tools import YoloxTools, link_open_vocab
from dlp_prelabel.common import iter_frames
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_schema.labels import BoxTrackPayload, Hand, Keypoint, KeypointFrame
from dlp_schema.ontology import load_ontology
from dlp_schema.predictor import Clip
from dlp_schema.validation import check_label
from dlp_schema.wearer_body import HandAt, WearerEvidence, hand_at

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 26, tzinfo=UTC)
MOP, TOWEL = 0, 9  # 정책 classes 번호 (mop, ..., towel)


@pytest.fixture
def policy() -> PrelabelPolicy:
    """저장소 정책에서 고정 클래스 단계를 켜고(가짜 이름) towel 문턱만 0.9로 올린 것."""
    p = load_policy(ROOT)
    sec = p.closed_vocab_tools.model_copy(update={"class_min_score": {"towel": 0.9}})
    return p.model_copy(
        update={
            "models": p.models.model_copy(update={"closed_vocab_tools": "yolox_tools_test"}),
            "closed_vocab_tools": sec,
        }
    )


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """1초 합성 VFR 영상 (320x240)."""
    path = tmp_path / "v.mp4"
    generate_blur_scenario(1, duration_ms=1_000).write(path)
    return path


def _times(video: Path, stride: int) -> list[int]:
    """어댑터가 추론할 시각: 첫 프레임부터 stride ms가 지난 첫 프레임마다."""
    out: list[int] = []
    for t, _ in iter_frames(video):
        if not out or t - out[-1] >= stride:
            out.append(t)
    return out


def _adapter(policy: PrelabelPolicy, detector: ScriptedToolDetector) -> YoloxTools:
    """stub 탐지기를 쓰는 어댑터 (가중치 버전 "stub")."""
    return YoloxTools(policy, lambda: detector, "stub", ontology_version="1.0.0", now=NOW)


def test_default_policy_creates_no_stage_and_validates_classes() -> None:
    """기본 정책(models.closed_vocab_tools: null)은 단계를 만들지 않는다 (가중치를 찾지도 않는다).

    classes는 온톨로지 객체이고 `config/pretrain/objects.yaml classes` 키 순서와 같다 (출력 열
    순서).
    """
    p = load_policy(ROOT)
    assert p.models.closed_vocab_tools is None
    assert YoloxTools.from_policy(ROOT, p, ontology_version="1.0.0", now=NOW) is None
    pretrain = yaml.safe_load((ROOT / "config/pretrain/objects.yaml").read_text("utf-8"))
    assert list(p.closed_vocab_tools.classes) == list(pretrain["classes"])
    ontology = load_ontology(ROOT / "config/ontology/v1")
    assert all(c in ontology.objects for c in p.closed_vocab_tools.classes)


def test_unknown_class_id_fails_at_policy_load(tmp_path: Path) -> None:
    """온톨로지에 없는 ID를 classes에 넣으면 `load_policy`가 ValueError를 낸다."""
    shutil.copytree(ROOT / "config/ontology", tmp_path / "config/ontology")
    (tmp_path / "config/policies").mkdir(parents=True)
    text = (ROOT / "config/policies/prelabel.yaml").read_text("utf-8")
    bad = text.replace("classes: [mop, rag,", "classes: [mop, laser_sword,", 1)
    assert bad != text
    (tmp_path / "config/policies/prelabel.yaml").write_text(bad, "utf-8")
    with pytest.raises(ValueError, match="laser_sword"):
        load_policy(tmp_path)


def test_box_tracks_version_and_class_thresholds(policy: PrelabelPolicy, video: Path) -> None:
    """추론 시각마다 같은 자리 mop 박스(0.8)와 towel 박스(0.6 < 문턱 0.9) → mop 트랙 하나.

    라벨 ID `<세션>-<스트림>-closed_tools-<tag>-mop-01`, entity `cv_mop_01`, 키프레임 = 추론 시각,
    신뢰도 = 점수 평균. 버전 = yolox-<가중치>+p<절 해시>+w<억제 절 해시>. 문턱은 classes 순서로
    넘긴다.
    """
    times = _times(video, policy.closed_vocab_tools.frame_stride_ms)
    frame = [
        YoloxDetection(MOP, (40.0, 50.0, 60.0, 30.0), 0.8),
        YoloxDetection(TOWEL, (200.0, 50.0, 40.0, 40.0), 0.6),
    ]
    detector = ScriptedToolDetector([frame] * len(times))
    adapter = _adapter(policy, detector)
    assert adapter.name == "closed_tools"
    assert adapter.version == (
        f"yolox-stub+p{policy.digest('closed_vocab_tools')}+w{policy.digest('wearer_suppression')}"
    )
    labels = adapter.run(Clip("s1", "bodycam", video))
    ontology = load_ontology(ROOT / "config/ontology/v1")
    [label] = labels
    p = label.payload
    assert isinstance(p, BoxTrackPayload) and p.class_id == "mop" and p.entity_id == "cv_mop_01"
    assert label.label_id.startswith("s1-bodycam-closed_tools-") and label.label_id.endswith(
        "-mop-01"
    )
    assert [k.t_ms for k in p.keyframes] == times and len(times) >= 3
    assert (p.keyframes[0].x, p.keyframes[0].w) == (40.0, 60.0)
    assert label.confidence == pytest.approx(0.8)
    assert label.provenance.model_version == adapter.version
    assert check_label(label, ontology) == []
    thresholds = detector.calls[0][1]
    assert len(thresholds) == len(policy.closed_vocab_tools.classes)
    assert thresholds[TOWEL] == 0.9 and thresholds[MOP] == policy.closed_vocab_tools.min_score
    assert detector.calls[0][0] == (320, 240) and len(detector.calls) == len(times)


def test_policy_section_change_changes_version(policy: PrelabelPolicy) -> None:
    """closed_vocab_tools 절 값이 바뀌면 버전이 바뀐다 (러너가 다시 돌고 검수 전 결과를 지운다)."""
    a = _adapter(policy, ScriptedToolDetector([]))
    sec = policy.closed_vocab_tools.model_copy(update={"min_score": 0.5})
    b = _adapter(policy.model_copy(update={"closed_vocab_tools": sec}), ScriptedToolDetector([]))
    assert a.version != b.version and b.version.startswith("yolox-stub+p")


def _wearer_hands(video: Path) -> WearerEvidence:
    """프레임마다 화면 아래쪽 (160, 200)에 착용자 왼손 손목이 있는 증거 (21관절 모두 그 근처)."""
    hands: list[tuple[int, HandAt]] = []
    for t, _ in iter_frames(video):
        points = tuple(Keypoint(x=160.0 + i, y=200.0, visibility=2) for i in range(21))
        at = hand_at(Hand.LEFT, KeypointFrame(t_ms=t, points=points))
        assert at is not None
        hands.append((t, at))
    return WearerEvidence(hands)


def test_wearer_body_box_is_suppressed_and_held_tool_kept(
    policy: PrelabelPolicy, video: Path
) -> None:
    """착용자 손목을 덮고 화면 아래까지 내려온 큰 towel 박스(화면 37%)는 억제, 작은 mop 박스는
    남는다.

    증거가 없으면(3인칭 등) 둘 다 남는다. stages에 tools가 없으면 억제하지 않고 버전에 +w가 없다.
    """
    sec = policy.closed_vocab_tools.model_copy(update={"class_min_score": {}})
    policy = policy.model_copy(update={"closed_vocab_tools": sec})
    times = _times(video, sec.frame_stride_ms)
    frame = [
        YoloxDetection(MOP, (10.0, 10.0, 40.0, 30.0), 0.8),
        YoloxDetection(TOWEL, (40.0, 120.0, 240.0, 120.0), 0.8),
    ]
    adapter = _adapter(policy, ScriptedToolDetector([frame] * len(times)))
    adapter.set_wearer(_wearer_hands(video))
    labels = adapter.run(Clip("s1", "bodycam", video))
    assert [x.payload.class_id for x in labels if isinstance(x.payload, BoxTrackPayload)] == ["mop"]
    report = adapter.last_wearer
    assert report is not None and report.tracks == 1 and report.by_class == {"towel": 1}
    plain = _adapter(policy, ScriptedToolDetector([frame] * len(times)))
    plain.set_wearer(None)
    assert len(plain.run(Clip("s1", "bodycam", video))) == 2 and plain.last_wearer is None
    ws = policy.wearer_suppression.model_copy(update={"stages": ("objects", "body")})
    no_tools = policy.model_copy(update={"wearer_suppression": ws})
    off = _adapter(no_tools, ScriptedToolDetector([frame] * len(times)))
    off.set_wearer(_wearer_hands(video))
    assert "+w" not in off.version and len(off.run(Clip("s1", "bodycam", video))) == 2


def test_open_vocab_queries_drop_closed_classes_only_when_stage_exists(
    policy: PrelabelPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """고정 클래스 단계가 목록에 있고 replace_open_vocab_classes면 OWLv2가 그 클래스 질의를 뺀다.

    OWLv2 질의 대상 중 classes에 든 것(mop, rag, ...)만 빠지고 toilet_brush·spray_bottle·trash_bag은
    남는다. 버전에 +xc<해시>가 붙는다. 단계가 없거나 옵션이 거짓이면 버전·질의가 그대로다.
    """

    def fake_resolve(root: Path, name: str) -> tuple[Path, str]:
        return Path(name), "w1"

    monkeypatch.setattr(owl_module, "resolve", fake_resolve)
    owl = OwlObjects(ROOT, policy, ontology_version="1.0.0", now=NOW)
    before = owl.version
    assert link_open_vocab([owl]) == frozenset() and owl.version == before
    closed = _adapter(policy, ScriptedToolDetector([]))
    excluded = link_open_vocab([owl, closed])
    targets = set(policy.open_vocab_objects.queries.values())
    assert excluded == targets & set(policy.closed_vocab_tools.classes)
    assert {"toilet_brush", "spray_bottle", "trash_bag"} <= targets - excluded
    assert owl.version.startswith(before + "+xc") and owl.excluded == excluded
    keep = policy.closed_vocab_tools.model_copy(update={"replace_open_vocab_classes": False})
    no_replace = policy.model_copy(update={"closed_vocab_tools": keep})
    owl2 = OwlObjects(ROOT, no_replace, ontology_version="1.0.0", now=NOW)
    assert link_open_vocab([owl2, _adapter(no_replace, ScriptedToolDetector([]))]) == frozenset()
    assert owl2.version == before
    # 모든 질의 대상을 빼면 모델을 열지 않고 빈 결과
    owl2.set_excluded_classes(targets)
    assert owl2.run(Clip("s1", "bodycam", Path("missing.mp4"))) == []
