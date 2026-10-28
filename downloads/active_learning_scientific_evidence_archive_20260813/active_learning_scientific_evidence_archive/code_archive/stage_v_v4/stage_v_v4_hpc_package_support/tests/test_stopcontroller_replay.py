from __future__ import annotations

from dataclasses import asdict

from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL

from scripts.replay_stopcontroller_surprise_modes import discrepancy_rows, replay
from stopcontroller_surprise_helpers import stop_config, toy_config, write_iteration


def test_all_selected_replay_matches_saved_history(tmp_path):
    cfg = toy_config()
    rows = [{"pred": PHASE_NORMAL, "exact": PHASE_FFLO}]
    write_iteration(tmp_path, 0, cfg, rows)
    saved = {
        "iteration": 0,
        "completed_iterations": 1,
        "conditions": {
            "C1_phase_map_change": True,
            "C2_boundary_shift_normal_sc": True,
            "C3_boundary_shift_uniform_fflo": True,
            "C4_label_surprise_rate": False,
            "C5_boundary_coverage_p95": True,
        },
        "metrics": {"label_surprise_rate": 1.0},
        "passed_condition_count": 4,
        "required_pass_count": 4,
        "convergence_pass": True,
        "patience_counter": 1,
        "stop": True,
        "stop_reason": "converged_main_phase_boundaries",
        "hard_stop": False,
        "stop_config": asdict(stop_config()),
    }
    replayed = replay([saved], tmp_path, cfg, "all_selected", 64, 0.25)
    discrepancies = discrepancy_rows(replayed)
    assert len(discrepancies) == 1
    assert discrepancies[0]["surprise_match"]
    assert discrepancies[0]["passed_count_match"]
    assert discrepancies[0]["convergence_pass_match"]
    assert discrepancies[0]["patience_counter_match"]
    assert discrepancies[0]["stop_match"]
