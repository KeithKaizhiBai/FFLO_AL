from __future__ import annotations

from ml_phase.labels import PHASE_FFLO

from stopcontroller_surprise_helpers import evaluate_rows


def test_trusted_denominator_floor_blocks_clean_rate(tmp_path):
    rows = [{"pred": PHASE_FFLO, "exact": PHASE_FFLO}]
    result = evaluate_rows(tmp_path, rows, mode="trusted", min_denominator=2)
    assert result["metrics"]["label_surprise_trusted"] == 0.0
    assert not result["metric_availability"]["trusted_surprise_denominator_valid"]
    assert not result["conditions"]["C4_label_surprise_rate"]


def test_trusted_denominator_fraction_blocks_clean_rate(tmp_path):
    rows = [
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO},
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO, "rerun_required": True},
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO, "rerun_required": True},
    ]
    result = evaluate_rows(tmp_path, rows, mode="trusted", min_denominator=1, min_fraction=0.5)
    assert result["surprise_details"]["trusted"]["denominator_fraction_selected"] == 1 / 3
    assert not result["metric_availability"]["trusted_surprise_denominator_valid"]
    assert not result["conditions"]["C4_label_surprise_rate"]
