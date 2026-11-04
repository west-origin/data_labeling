"""골든 평가의 예측은 모델 출처 레코드만이다. 센서 출처는 정답도 예측도 아니다 (F-ETA-32)."""

from __future__ import annotations

from dlp_eval import runner
from dlp_schema.labels import Provenance, Source
from dlp_schema.testing import action_payload, make_label


def test_sensor_records_are_not_predictions_f_eta_32() -> None:
    model = make_label(
        action_payload(),
        label_id="m",
        provenance=Provenance(source=Source.MODEL, model_version="v1"),
        confidence=0.5,
    )
    sensor = make_label(
        action_payload(),
        label_id="s",
        provenance=Provenance(source=Source.SENSOR, sensor_id="glove1", model_version="v1"),
    )
    bare = make_label(
        action_payload(),
        label_id="s2",
        provenance=Provenance(source=Source.SENSOR, sensor_id="glove1"),
    )
    labels = [model, sensor, bare]
    pick = runner._predictions  # pyright: ignore[reportPrivateUsage]
    assert [x.label_id for x in pick(labels, ("action",), "v1", set(), set())] == ["m"]
    assert [x.label_id for x in pick(labels, ("action",), "*", set(), set())] == ["m"]
