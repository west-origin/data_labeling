"""참조 구현과의 일치 테스트.

pycocotools(mAP), TrackEval(HOTA·IDF1·CLEAR), scikit-learn(macro F1·카파),
scipy(Clopper-Pearson 상한).

WP11 완료 기준("가능한 지표는 기존 참조 구현과 결과 일치")의 테스트. 시드를 고정한 무작위 입력을
우리 구현과 참조 라이브러리에 똑같이 넣고, 참조 라이브러리의 출력을 정답으로 본다. 참조
라이브러리는 개발 의존성이다 (pycocotools, trackeval, scikit-learn). TrackEval은 numpy 2에서 없어진
`np.float`·`np.int` 별칭을 써서 테스트 안에서 별칭을 되살린다.
"""

# 참조 라이브러리에는 타입 정보가 없어 이 테스트에서만 관련 경고를 끈다.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np
import pytest

from dlp_eval.metrics.classification import cohen_kappa, macro_f1
from dlp_eval.metrics.detection import DetBox, GtBox, average_precision, box_iou
from dlp_eval.metrics.tracking import TrackingData, clear, hota, identity


def _random_detection(seed: int) -> tuple[list[GtBox], list[DetBox]]:
    """이미지 12장의 무작위 정답 박스와, 정답을 흔든 예측(80%)과 오탐을 섞은 예측을 만든다."""
    rng = np.random.default_rng(seed)
    gts: list[GtBox] = []
    dets: list[DetBox] = []
    for img in range(12):
        for _ in range(int(rng.integers(0, 6))):
            cat = str(rng.choice(["cup", "mop", "sink"]))
            x, y = rng.uniform(0, 500, 2)
            w, h = rng.uniform(10, 120, 2)
            gts.append(GtBox(img, cat, (float(x), float(y), float(w), float(h))))
            if rng.random() < 0.8:  # 흔들린 예측
                j = rng.normal(0, 0.15, 4) * [w, h, w, h]
                box = (
                    float(x + j[0]),
                    float(y + j[1]),
                    float(max(1, w + j[2])),
                    float(max(1, h + j[3])),
                )
                dets.append(DetBox(img, cat, box, float(rng.uniform(0.3, 1))))
        for _ in range(int(rng.integers(0, 3))):  # 오탐
            x, y = rng.uniform(0, 500, 2)
            cat = str(rng.choice(["cup", "mop", "sink"]))
            dets.append(
                DetBox(img, cat, (float(x), float(y), 40.0, 40.0), float(rng.uniform(0, 0.8)))
            )
    return gts, dets


@pytest.mark.parametrize("seed", range(4))
def test_average_precision_matches_pycocotools(seed: int) -> None:
    """COCO mAP·AP50·AP75가 pycocotools COCOeval(bbox) stats[0..2]와 1e-9 안에서 같다."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gts, dets = _random_detection(seed)
    cats = {
        c: i + 1
        for i, c in enumerate(sorted({g.category for g in gts} | {d.category for d in dets}))
    }
    data: dict[str, Any] = {
        "images": [{"id": i, "width": 640, "height": 640} for i in range(12)],
        "categories": [{"id": i, "name": c} for c, i in cats.items()],
        "annotations": [
            {"id": k + 1, "image_id": g.image, "category_id": cats[g.category], "bbox": list(g.box),
             "area": g.box[2] * g.box[3], "iscrowd": 0}
            for k, g in enumerate(gts)
        ],
    }  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = data
        coco.createIndex()
        results = coco.loadRes(
            [
                {
                    "image_id": d.image,
                    "category_id": cats[d.category],
                    "bbox": list(d.box),
                    "score": d.score,
                }
                for d in dets
            ]
        )
        ev = COCOeval(coco, results, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ours = average_precision(gts, dets)
    assert ours.map == pytest.approx(ev.stats[0], abs=1e-9)
    assert ours.ap50 == pytest.approx(ev.stats[1], abs=1e-9)
    assert ours.ap75 == pytest.approx(ev.stats[2], abs=1e-9)


def _random_tracking(seed: int) -> TrackingData:
    """개체 5개가 40 시각 동안 움직이는 시퀀스. 놓침·오탐과 ID 전환 하나(p2 → p_switch)를 넣는다."""
    rng = np.random.default_rng(seed)
    frames = []
    n_obj = 5
    pos = rng.uniform(0, 400, (n_obj, 2))
    for t in range(40):
        pos += rng.normal(0, 5, pos.shape)
        visible = [i for i in range(n_obj) if rng.random() < 0.9]
        gt_boxes = [(float(pos[i, 0]), float(pos[i, 1]), 50.0, 50.0) for i in visible]
        pred_keys: list[str] = []
        pred_boxes = []
        for i in visible:
            if rng.random() < 0.85:
                key = f"p{i}" if not (i == 2 and t > 20) else "p_switch"  # ID 전환 하나
                pred_keys.append(key)
                pred_boxes.append(
                    (
                        float(pos[i, 0] + rng.normal(0, 8)),
                        float(pos[i, 1] + rng.normal(0, 8)),
                        50.0,
                        50.0,
                    )
                )
        if rng.random() < 0.3:
            pred_keys.append(f"fp{t}")
            pred_boxes.append((float(rng.uniform(0, 400)), float(rng.uniform(0, 400)), 50.0, 50.0))
        frames.append(([f"g{i}" for i in visible], pred_keys, box_iou(gt_boxes, pred_boxes)))
    return TrackingData.from_frames(frames)


def _trackeval(data: TrackingData) -> dict[str, Any]:
    """TrackEval의 HOTA·Identity·CLEAR `eval_sequence`로 같은 시퀀스를 계산한다 (참조 값)."""
    np.float = float  # type: ignore[attr-defined]  # TrackEval은 numpy 2에서 없어진 별칭을 쓴다
    np.int = int  # type: ignore[attr-defined]
    with contextlib.redirect_stdout(io.StringIO()):
        from trackeval.metrics.clear import CLEAR
        from trackeval.metrics.hota import HOTA
        from trackeval.metrics.identity import Identity

    raw = {
        "num_gt_ids": data.num_gt_ids, "num_tracker_ids": data.num_tracker_ids,
        "num_gt_dets": data.num_gt_dets, "num_tracker_dets": data.num_tracker_dets,
        "gt_ids": data.gt_ids, "tracker_ids": data.tracker_ids,
        "similarity_scores": data.similarity, "num_timesteps": len(data.gt_ids),
    }  # fmt: skip
    h = HOTA().eval_sequence(raw)
    i = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(raw)
    c = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(raw)
    return {
        "HOTA": h["HOTA"].mean(), "DetA": h["DetA"].mean(), "AssA": h["AssA"].mean(),
        "LocA": h["LocA"].mean(), "IDF1": i["IDF1"], "MOTA": c["MOTA"], "IDSW": c["IDSW"],
    }  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_tracking_metrics_match_trackeval(seed: int) -> None:
    """HOTA·DetA·AssA·LocA·IDF1·MOTA·IDSW가 TrackEval과 1e-12 안에서 같다."""
    data = _random_tracking(seed)
    ref = _trackeval(data)
    h, i, c = hota(data), identity(data), clear(data)
    assert h.hota == pytest.approx(ref["HOTA"], abs=1e-12)
    assert (h.det_a, h.ass_a, h.loc_a) == pytest.approx(
        (ref["DetA"], ref["AssA"], ref["LocA"]), abs=1e-12
    )
    assert i.idf1 == pytest.approx(ref["IDF1"], abs=1e-12)
    assert c.mota == pytest.approx(ref["MOTA"], abs=1e-12) and c.idsw == ref["IDSW"]


def test_perfect_tracking_scores_one() -> None:
    """예측이 정답과 똑같으면(유사도 단위 행렬) HOTA·IDF1·MOTA가 모두 1이다."""
    frames = [(["a", "b"], ["x", "y"], np.eye(2)) for _ in range(10)]
    data = TrackingData.from_frames(frames)
    assert hota(data).hota == pytest.approx(1.0) and identity(data).idf1 == 1.0
    assert clear(data).mota == 1.0


@pytest.mark.parametrize("seed", range(3))
def test_classification_metrics_match_sklearn(seed: int) -> None:
    """macro F1과 Cohen 카파가 scikit-learn f1_score(macro)·cohen_kappa_score와 같다."""
    from sklearn.metrics import cohen_kappa_score, f1_score

    rng = np.random.default_rng(seed)
    labels = ["rub", "push", "pull", "press"]
    truth = [str(x) for x in rng.choice(labels, 200)]
    pred = [t if rng.random() < 0.7 else str(rng.choice(labels)) for t in truth]
    assert macro_f1(truth, pred) == pytest.approx(f1_score(truth, pred, average="macro"), abs=1e-12)
    assert cohen_kappa(truth, pred) == pytest.approx(cohen_kappa_score(truth, pred), abs=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_weighted_kappa_and_ece_match_references(seed: int) -> None:
    """표본 가중 카파는 scikit-learn `cohen_kappa_score(sample_weight=…)`와 같고, 정수 가중치의
    카파·ECE는 표본을 그만큼 반복한 값과 같다. 가중치를 주지 않으면 예전 값과 같다."""
    from sklearn.metrics import cohen_kappa_score

    from dlp_eval.metrics.classification import ece

    rng = np.random.default_rng(seed)
    a = [bool(x) for x in rng.random(120) < 0.4]
    b = [x if rng.random() < 0.8 else not x for x in a]
    w = [float(x) for x in rng.integers(1, 6, 120)]
    ref = cohen_kappa_score(a, b, sample_weight=w)
    assert cohen_kappa(a, b, w) == pytest.approx(ref, abs=1e-12)
    rep_a = [x for x, k in zip(a, w, strict=True) for _ in range(int(k))]
    rep_b = [x for x, k in zip(b, w, strict=True) for _ in range(int(k))]
    assert cohen_kappa(a, b, w) == pytest.approx(cohen_kappa(rep_a, rep_b), abs=1e-12)
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b), abs=1e-12)
    conf = [float(x) for x in rng.random(120)]
    rep_conf = [c for c, k in zip(conf, w, strict=True) for _ in range(int(k))]
    rep_ok = [x for x, k in zip(a, w, strict=True) for _ in range(int(k))]
    assert ece(conf, a, weights=w) == pytest.approx(ece(rep_conf, rep_ok), abs=1e-12)
    assert ece(conf, a, weights=[1.0] * 120) == pytest.approx(ece(conf, a), abs=1e-12)


@pytest.mark.parametrize(("k", "n", "w"), [(0, 5, 3.0), (1, 5, 20.0), (4, 9, 0.5), (7, 7, 2.0)])
def test_weighted_upper_equals_clopper_pearson_for_equal_weights(k: int, n: int, w: float) -> None:
    """가중치가 모두 같으면 유효 표본 수 = n이라 가중 상한이 Clopper-Pearson 상한과 같다."""
    from dlp_eval.metrics.proportion import clopper_pearson_upper, weighted_upper

    assert weighted_upper([w] * k, [w] * n) == pytest.approx(clopper_pearson_upper(k, n), abs=1e-12)


def test_weighted_upper_hand_example() -> None:
    """손 계산: 가중치 4인 표본 2개(사건 없음) + 가중치 20인 표본 5개(사건 1개) → 비율 20/108,
    유효 표본 수 n* = 108²/(2·16 + 5·400) = 11664/2032, x* = (20/108)·n*. 상한 =
    Beta(0.95; x*+1, n*-x*). 빈 표본은 NaN, 잘못된 입력은 ValueError."""
    import math

    from scipy.stats import beta

    from dlp_eval.metrics.proportion import weighted_upper

    n_eff = 108**2 / (2 * 16 + 5 * 400)
    x = 20 / 108 * n_eff
    want = float(beta.ppf(0.95, x + 1, n_eff - x))
    assert weighted_upper([20.0], [4.0, 4.0, *[20.0] * 5]) == pytest.approx(want, abs=1e-12)
    assert math.isnan(weighted_upper([], []))
    with pytest.raises(ValueError):
        weighted_upper([5.0], [1.0])
    with pytest.raises(ValueError):
        weighted_upper([], [-1.0])


@pytest.mark.parametrize(("k", "n"), [(0, 1), (0, 10), (1, 10), (3, 40), (12, 12), (7, 250)])
def test_clopper_pearson_upper_matches_scipy_binomtest(k: int, n: int) -> None:
    """한쪽 Clopper-Pearson 상한이 scipy `binomtest(alternative="less")`의 정확 구간 상한과 같다.
    손 계산: k=0이면 1 - (1-신뢰)^(1/n) (n=10, 95% → 0.2589)."""
    from scipy.stats import binomtest

    from dlp_eval.metrics.proportion import clopper_pearson_upper

    ref = binomtest(k, n, alternative="less").proportion_ci(confidence_level=0.95, method="exact")
    assert clopper_pearson_upper(k, n) == pytest.approx(ref.high, abs=1e-12)
    if k == 0:
        assert clopper_pearson_upper(k, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-12)


def test_clopper_pearson_upper_edges() -> None:
    """시행이 없으면 NaN, 범위 밖 입력은 ValueError."""
    import math

    from dlp_eval.metrics.proportion import clopper_pearson_upper

    assert math.isnan(clopper_pearson_upper(0, 0))
    with pytest.raises(ValueError):
        clopper_pearson_upper(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 3, confidence=1.0)
