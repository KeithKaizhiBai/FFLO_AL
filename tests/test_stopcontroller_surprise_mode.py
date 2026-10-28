from __future__ import annotations

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL

from stopcontroller_surprise_helpers import evaluate_rows


def test_trusted_mode_uses_trusted_metric_for_c4(tmp_path):
    rows = [
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "rerun_required": True},
    ]
    result = evaluate_rows(tmp_path, rows, mode="trusted", min_denominator=1)
    assert result["metrics"]["label_surprise_all_selected"] == 0.5
    assert result["metrics"]["label_surprise_trusted"] == 0.0
    assert result["metrics"]["label_surprise_selected_for_gate"] == 0.0
    assert result["surprise_details"]["selected_gate_metric"] == "label_surprise_trusted"
    assert result["conditions"]["C4_label_surprise_rate"]


def test_all_selected_mode_uses_all_selected_metric_for_c4(tmp_path):
    rows = [
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "rerun_required": True},
    ]
    result = evaluate_rows(tmp_path, rows, mode="all_selected")
    assert result["metrics"]["label_surprise_selected_for_gate"] == 0.5
    assert result["surprise_details"]["selected_gate_metric"] == "label_surprise_all_selected"
    assert not result["conditions"]["C4_label_surprise_rate"]
