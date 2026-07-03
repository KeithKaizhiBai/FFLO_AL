from __future__ import annotations

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL

from stopcontroller_surprise_helpers import evaluate_rows


def test_label_surprise_rate_alias_preserves_all_selected_behavior(tmp_path):
    result = evaluate_rows(
        tmp_path,
        [{"pred": PHASE_NORMAL, "exact": PHASE_FFLO, "rerun_required": True}],
        mode="all_selected",
    )
    assert result["metrics"]["label_surprise_rate"] == 1.0
    assert result["metrics"]["label_surprise_all_selected"] == 1.0
    assert result["metric_availability"]["label_surprise_rate"]
    assert result["surprise_details"]["stop_surprise_mode"] == "all_selected"
    assert not result["conditions"]["C4_label_surprise_rate"]
