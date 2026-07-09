from __future__ import annotations

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL

from stopcontroller_surprise_helpers import evaluate_rows


def test_trusted_mask_excludes_only_untrusted_numerical_frontier(tmp_path):
    rows = [
        {"pred": PHASE_FFLO, "exact": PHASE_FFLO, "q_expanded": True},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "rerun_required": True},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "trusted_exact": False},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "training_eligible_exact": False},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "q_unresolved": True},
        {"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "delta_unresolved": True},
    ]
    result = evaluate_rows(tmp_path, rows, mode="trusted", min_denominator=1)
    assert result["surprise_details"]["trusted"]["n_denominator"] == 1
    assert result["surprise_details"]["trusted"]["n_surprise"] == 0
    assert result["surprise_details"]["hard_risk"]["n_denominator"] == 5
    assert result["metrics"]["label_surprise_all_selected"] == 5 / 6
    assert result["metrics"]["label_surprise_trusted"] == 0.0
    assert result["conditions"]["C4_label_surprise_rate"]
