from __future__ import annotations

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL

from stopcontroller_surprise_helpers import evaluate_rows


def test_prediction_mismatch_does_not_create_rerun_required(tmp_path):
    result = evaluate_rows(
        tmp_path,
        [{"pred": PHASE_NORMAL, "exact": PHASE_FFLO}],
        mode="trusted",
        min_denominator=1,
    )
    assert result["surprise_details"]["rerun_required_count"] == 0
    assert result["surprise_details"]["trusted"]["n_denominator"] == 1
    assert result["metrics"]["label_surprise_trusted"] == 1.0
    assert not result["conditions"]["C4_label_surprise_rate"]
