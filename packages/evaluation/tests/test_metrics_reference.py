"""참조 구현과의 일치 테스트.

pycocotools(mAP), TrackEval(HOTA·IDF1·CLEAR), scikit-learn(macro F1·카파).

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
