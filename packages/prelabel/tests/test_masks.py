"""객체·손 모양 마스크 단계 (ADR 0059): 정답 모양 합성 영상(`dlp_fixtures.shapes`)으로 본다.

- stub(`BoxEllipseSegmenter`)으로 마스크 IoU·출처·착용자 팔 제외·간격.
- 검사 규칙(새어 나감·낮은 점수·작음·손 빼기)과 RLE 왕복(pycocotools와 같은 문자열).
- 실행기: SQLite + 로컬 저장소로 멱등·입력 변경 시 검수 전 것만 지우기·검수된 마스크 보호.
- 실제 SAM 2.1 가중치가 있으면(`make models`) 정답 모양과 IoU (없으면 건너뛴다).
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
import yaml
from numpy.typing import NDArray

from dlp_fixtures.shapes import ShapeScenario, generate_shape_scenario
from dlp_media.ingest import ingest_session, load_manifest
from dlp_media.rle import decode_rle, encode_rle
from dlp_media.storage import LocalStore
from dlp_models.sam2 import MaskResult
from dlp_prelabel.adapters.stubs import OraclePredictor
from dlp_prelabel.masks import BoxEllipseSegmenter, MaskMaker, check_mask
from dlp_prelabel.policy import PrelabelPolicy, load_policy
from dlp_prelabel.runner import pick_mask_tracks, run_prelabel
from dlp_schema.db.repository import get_labels, record_review, register_ontology
from dlp_schema.episode import current_labels
from dlp_schema.labels import (
    ARM3_JOINTS,
    BoxTrackPayload,
    Hand,
    Keypoint,
    KeypointFrame,
    KeypointTrackPayload,
    LabelRecord,
    MaskTrackPayload,
    VerificationState,
)
from dlp_schema.ontology import Ontology, load_ontology
from dlp_schema.predictor import ModelUnavailableError
from dlp_schema.testing import FIXED_TIME, make_label, sqlite_db
from dlp_schema.validation import check_label
from dlp_schema.wearer_body import WearerEvidence

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def policy() -> PrelabelPolicy:
    """저장소 정책에서 마스크 간격만 100 ms로 줄인 것 (짧은 합성 영상에 키프레임 넷)."""
    p = load_policy(ROOT)
    return p.model_copy(update={"masks": p.masks.model_copy(update={"frame_stride_ms": 100})})


@pytest.fixture
def ontology() -> Ontology:
    """저장소 온톨로지 v1."""
    return load_ontology(ROOT / "config/ontology/v1")


def _iou(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> float:
    """두 마스크 IoU."""
    return float((a & b).sum() / max(1, (a | b).sum()))


def _ious(labels: list[LabelRecord], sc: ShapeScenario) -> dict[str, float]:
    """개체 → 키프레임 마스크와 정답 마스크의 평균 IoU."""
    out: dict[str, float] = {}
    for x in labels:
        p = x.payload
        assert isinstance(p, MaskTrackPayload)
        vals = [
            _iou(decode_rle(k.rle), sc.masks[p.entity_id][k.t_ms])
            for k in p.keyframes
            if k.rle is not None
        ]
        out[p.entity_id] = float(np.mean(vals))
    return out


def _run(maker: MaskMaker, sc: ShapeScenario) -> list[LabelRecord]:
    """합성 프레임으로 마스크 단계를 돌린다 (착용자 증거 = 손 트랙)."""
    return maker.run(
        Path("unused.mp4"), session_id="shapes", stream_id="bodycam", tracks=sc.labels,
        ontology_version="1.0.0", wearer=WearerEvidence.from_labels(sc.labels, "bodycam"),
        frames=iter(sc.frames),
    )  # fmt: skip


def test_stub_masks_follow_prompts_and_skip_the_wearer_arm(
    policy: PrelabelPolicy, ontology: Ontology
) -> None:
    """stub 타원 마스크: 컵(타원)은 정답과 IoU 0.9 이상, 착용자 팔(person 박스)은 마스크가 없고,
    손은 class hand·entity left_hand, derived_from이 프롬프트 트랙, 키프레임 간격이 정책
    간격이다."""
    sc = generate_shape_scenario(0)
    maker = MaskMaker(policy, BoxEllipseSegmenter(), "stub", now=FIXED_TIME)
    labels = _run(maker, sc)
    by_entity = {x.payload.entity_id: x for x in labels if isinstance(x.payload, MaskTrackPayload)}
    assert set(by_entity) == {"cup_01", "bucket_01", "left_hand"}
    report = maker.last_report
    assert report is not None and report.wearer_skipped > 0 and report.hand_masks > 0
    ious = _ious(labels, sc)
    assert ious["cup_01"] >= 0.9
    hand = by_entity["left_hand"].payload
    assert isinstance(hand, MaskTrackPayload) and hand.class_id == "hand"
    assert hand.derived_from == ("shapes-hand-left",)
    times = [k.t_ms for k in hand.keyframes]
    assert all(b - a >= 100 for a, b in pairwise(times)) and len(times) >= 3
    assert all(k.score == 0.9 for k in hand.keyframes)
    assert all(check_label(x, ontology) == [] for x in labels)
    assert all(x.label_id.startswith("shapes-bodycam-masks-") for x in labels)


def test_object_masks_exclude_hand_pixels(policy: PrelabelPolicy) -> None:
    """손 마스크와 겹치는 화소는 객체 마스크에서 빠진다 (subtract_hands)."""
    box = (10.0, 10.0, 40.0, 40.0)
    obj = np.zeros((60, 60), bool)
    obj[10:50, 10:50] = True
    hands = np.zeros((60, 60), bool)
    hands[10:30, 10:50] = True
    mask, reason = check_mask(MaskResult(obj, 0.9), box, policy.masks, hands)
    assert reason is None and mask is not None
    assert int(mask.sum()) == 20 * 40 and not (mask & hands).any()


@pytest.mark.parametrize(
    ("score", "region", "reason"),
    [
        (0.2, (slice(10, 50), slice(10, 50)), "low_score"),
        (0.9, (slice(0, 60), slice(0, 60)), "leak"),
        (0.9, (slice(20, 21), slice(20, 21)), "small"),
    ],
)
def test_check_mask_drops_by_policy(
    policy: PrelabelPolicy, score: float, region: tuple[slice, slice], reason: str
) -> None:
    """낮은 점수·박스 밖으로 새어 나감·거의 빈 마스크는 사유와 함께 버린다."""
    m = np.zeros((60, 60), bool)
    m[region] = True
    assert check_mask(MaskResult(m, score), (10.0, 10.0, 40.0, 40.0), policy.masks) == (
        None,
        reason,
    )
    assert check_mask(None, (10.0, 10.0, 40.0, 40.0), policy.masks) == (None, "no_object")


def test_rle_round_trip_matches_pycocotools() -> None:
    """우리 COCO RLE 부호화가 pycocotools와 같은 문자열이고 왕복이 정확하다 (정답 손 마스크)."""
    mask_util: Any = pytest.importorskip("pycocotools.mask")
    sc = generate_shape_scenario(1)
    for m in (sc.masks["left_hand"][0], sc.masks["bucket_01"][33], np.ones((5, 7), bool)):
        rle = encode_rle(m)
        ref = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        assert rle.counts == ref["counts"].decode() and list(rle.size) == list(ref["size"])
        assert np.array_equal(decode_rle(rle), m)


def test_pick_mask_tracks_ignores_arm_and_body_skeletons() -> None:
    """프롬프트는 박스·hand21 트랙만 (착용자 팔 arm3·전신 자세는 아니다, ADR 0058)."""
    sc = generate_shape_scenario(0)
    arm = make_label(
        KeypointTrackPayload(
            entity_id="left_arm", skeleton="arm3", hand=Hand.LEFT,
            keyframes=(KeypointFrame(t_ms=0, points=tuple(
                Keypoint(x=1, y=1, visibility=2) for _ in ARM3_JOINTS)),),
        ),
        label_id="arm", stream_id="bodycam", t_start_ms=0, t_end_ms=0,
    )  # fmt: skip
    picked = pick_mask_tracks([*sc.labels, arm], "bodycam")
    assert {x.label_id for x in picked} == {x.label_id for x in sc.labels}


def _session(tmp_path: Path, ontology: Ontology, sc: ShapeScenario) -> tuple[sa.Engine, LocalStore]:
    """정답 모양 영상 하나짜리 세션(shapes)을 SQLite·로컬 원본 저장소에 수집한다."""
    sc.write(tmp_path / "bodycam.mp4")
    engine = sqlite_db(f"sqlite:///{tmp_path / 'db.sqlite'}")
    with engine.begin() as conn:
        register_ontology(conn, ontology)
    manifest = {
        "session_id": "shapes", "domain": "cleaning", "worker_id": "w01", "site_id": "site01",
        "consent_version": "c1", "recorded_at": FIXED_TIME.isoformat(),
        "ontology_version": "1.0.0",
        "streams": [{"stream_id": "bodycam", "kind": "bodycam", "path": "bodycam.mp4"}],
    }  # fmt: skip
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    raw = LocalStore(tmp_path / "store", "dlp-raw")
    with engine.begin() as conn:
        ingest_session(*load_manifest(tmp_path / "m.yaml"), raw, conn)
    return engine, raw


def _masks(conn: sa.Connection) -> list[LabelRecord]:
    """현재 mask_track."""
    return current_labels(get_labels(conn, "shapes", kinds=["mask_track"]))


def test_runner_mask_stage_is_idempotent_and_protects_reviewed_masks(
    tmp_path: Path, policy: PrelabelPolicy, ontology: Ontology
) -> None:
    """실행기 마스크 단계: 첫 실행에 셋, 같은 입력 재실행은 아무것도 넣지 않는다. 박스 입력이 바뀌면
    (새 박스 모델 버전) 검수 전 마스크만 새 버전으로 바뀌고, 승인한 컵 마스크는 남고 컵은 새로
    만들지 않는다."""
    sc = generate_shape_scenario(0)
    engine, raw = _session(tmp_path, ontology, sc)
    maker = MaskMaker(policy, BoxEllipseSegmenter(), "stub", now=FIXED_TIME)

    def predictors(jitter: float) -> list[Any]:
        """손·객체 Oracle (jitter가 바뀌면 박스 모델 버전이 바뀐다)."""
        return [
            OraclePredictor("hands", sc.labels, ("keypoint_track",), now=FIXED_TIME),
            OraclePredictor("objects", sc.labels, ("box_track",), jitter_px=jitter, now=FIXED_TIME),
        ]

    with engine.begin() as conn:
        first = run_prelabel(
            conn, "shapes", raw, predictors(0.0), policy, ontology, FIXED_TIME, masker=maker
        )
        masks = _masks(conn)
    assert first.masks == len(masks) == 3 and first.mask_report is not None
    assert {x.payload.entity_id for x in masks if isinstance(x.payload, MaskTrackPayload)} == {
        "cup_01", "bucket_01", "left_hand",
    }  # fmt: skip
    with engine.begin() as conn:
        again = run_prelabel(
            conn, "shapes", raw, predictors(0.0), policy, ontology, FIXED_TIME, masker=maker
        )
    assert again.masks == 0 and again.mask_report is None
    cup = next(
        x
        for x in masks
        if isinstance(x.payload, MaskTrackPayload) and x.payload.entity_id == "cup_01"
    )
    with engine.begin() as conn:
        record_review(conn, cup.label_id, VerificationState.HUMAN_APPROVED, "rev01", FIXED_TIME)
        third = run_prelabel(
            conn, "shapes", raw, predictors(0.5), policy, ontology, FIXED_TIME, masker=maker
        )
        now = _masks(conn)
    assert third.masks == 2 and third.masks_blocked == 1
    ids = {x.label_id for x in now}
    assert cup.label_id in ids and not ({x.label_id for x in masks} - {cup.label_id}) & ids


def test_real_sam2_masks_match_known_shapes(policy: PrelabelPolicy) -> None:
    """실제 SAM 2.1 tiny(가중치가 있을 때만): 컵·양동이·손 마스크가 정답과 IoU 0.8 이상."""
    try:
        maker = MaskMaker.from_registry(ROOT, policy, now=FIXED_TIME)
    except ModelUnavailableError as exc:
        pytest.skip(f"SAM 2.1 가중치 없음 (make models): {exc}")
    sc = generate_shape_scenario(0, n_frames=4)
    ious = _ious(_run(maker, sc), sc)
    assert set(ious) == {"cup_01", "bucket_01", "left_hand"}
    assert min(ious.values()) >= 0.8, ious


def test_person_box_at_the_bottom_edge_is_skipped_without_hand_evidence(
    policy: PrelabelPolicy,
) -> None:
    """손이 안 보여 착용자 증거가 없어도 화면 아래에 닿은 사람 박스(착용자 몸통)는 마스크를 만들지
    않는다. 아래에 닿지 않는 사람 박스(다른 사람)는 만든다 (person_bottom_band)."""
    sc = generate_shape_scenario(0)
    arm = next(x for x in sc.labels if x.label_id == "shapes-box-wearer_arm")
    p = arm.payload
    assert isinstance(p, BoxTrackPayload)
    raised = arm.model_copy(
        update={
            "label_id": "shapes-box-other",
            "payload": p.model_copy(
                update={
                    "entity_id": "person_02",
                    "keyframes": tuple(
                        k.model_copy(update={"y": 5.0, "h": 60.0}) for k in p.keyframes
                    ),
                }
            ),
        }
    )
    maker = MaskMaker(policy, BoxEllipseSegmenter(), "stub", now=FIXED_TIME)
    labels = maker.run(
        Path("unused.mp4"), session_id="shapes", stream_id="bodycam", tracks=[arm, raised],
        ontology_version="1.0.0", frames=iter(sc.frames),
    )  # fmt: skip
    entities = {x.payload.entity_id for x in labels if isinstance(x.payload, MaskTrackPayload)}
    assert entities == {"person_02"}
    assert maker.last_report is not None and maker.last_report.wearer_skipped > 0
