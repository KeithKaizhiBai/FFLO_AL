from __future__ import annotations

from ml_phase.labels import PHASE_FFLO

from stopcontroller_surprise_helpers import evaluate_rows


def test_numerical_frontier_unresolved_when_q_or_delta_unresolved(tmp_path):
    result = evaluate_rows(
        tmp_path,
        [{"pred": PHASE_FFLO, "exact": PHASE_FFLO, "q_unresolved": True}],
        mode="trusted",
        min_denominator=1,
    )
    assert result["numerical_frontier_status"] == "unresolved"


def test_numerical_frontier_active_for_rerun_without_unresolved(tmp_path):
    result = evaluate_rows(
        tmp_path,
        [{"pred": PHASE_FFLO, "exact": PHASE_FFLO, "rerun_required": True}],
        mode="trusted",
        min_denominator=1,
    )
    assert result["numerical_frontier_status"] == "active"


def test_numerical_frontier_closed_for_all_trusted_clean(tmp_path):
    result = evaluate_rows(
        tmp_path,
        [{"pred": PHASE_FFLO, "exact": PHASE_FFLO}],
        mode="trusted",
        min_denominator=1,
    )
    assert result["numerical_frontier_status"] == "closed"
