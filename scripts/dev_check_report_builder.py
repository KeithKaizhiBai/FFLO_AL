from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml_phase.report_builder import build_report

TEMPLATE = ROOT / "report" / "active_learning_phase_boundary_report.tex"


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_fake_run(tmp: Path) -> Path:
    run_root = tmp / "ml_phase_test" / "active_runs"
    run_id = "active_boundary_discovery_test"
    run_dir = run_root / run_id
    iter_dir = run_dir / "iter065"
    iter_dir.mkdir(parents=True)

    cfg = {
        "args": {
            "run_id": run_id,
            "mode": "hpc",
            "world_size": 8,
            "partition_strategy": "contiguous",
            "points_per_iter": 256,
        },
        "active_learning_config": {
            "run_mode": "discovery",
            "candidate_domain_mode": "full",
            "selection_mode": "stochastic",
            "initialization": "random_grid",
            "initial_seed_size": 512,
            "batch_size_max": 256,
            "finite_t_band_width": None,
            "sampling_power": 1.0,
            "n_ensemble": 3,
            "val_fraction": 0.15,
            "delta_eps": 1e-3,
            "q_eps": 1e-2,
        },
    }
    _write_json(run_dir / "run_config.json", cfg)
    _write_json(
        run_dir / "metrics_history.json",
        [
            {
                "eta_rmse": 0.1,
                "phase_accuracy": 0.9,
                "estimated_reduction": 4.0,
            }
        ],
    )
    _write_json(
        iter_dir / "stop_metrics_iter065.json",
        {
            "stop": True,
            "stop_reason": "converged_main_phase_boundaries",
            "hard_stop": False,
            "convergence_pass": True,
            "passed_condition_count": 4,
            "patience_counter": 4,
            "metrics": {
                "phase_map_change": 0.001,
                "boundary_shift_normal_sc": 0.002,
                "boundary_shift_uniform_fflo": 0.0,
                "label_surprise_rate": 0.0,
                "boundary_coverage_p95": 0.004,
            },
            "conditions": {
                "C1_phase_map_change": True,
                "C2_boundary_shift_normal_sc": True,
                "C3_boundary_shift_uniform_fflo": True,
                "C4_label_surprise_rate": True,
                "C5_boundary_coverage_p95": False,
            },
            "boundary_details": {
                "normal_sc": {"value": 0.002, "p95": 0.002},
                "uniform_fflo": {"value": 0.0, "p95": 0.0},
            },
        },
    )
    _write_json(
        iter_dir / "merge_summary_iter065.json",
        {
            "merged_points": 256,
            "training_eligible_points": 250,
            "rerun_required_points": 6,
        },
    )
    (iter_dir / "selected_points.csv").write_text("kT,JA\n0.1,0.2\n", encoding="utf-8")
    (iter_dir / "selected_points_by_pool.csv").write_text(
        "\n".join(
            [
                "selection_rank,selection_source,grid_index,kT,JA,A0_main,A_phase,A_numerical,A_explore,A_response,R_obs,R_batch,Aselect,cls_uncertainty_mix,cls_entropy,cls_margin_uncertainty,U_delta,U_q,U_reg_phase,delta_boundary_score,B_q_SC,E_q_SC,E_ext_uncertain,sampling_probability_before_pick",
                "1,acquisition_stochastic,0,0.0,0.0,1.2,1.1,0.02,0.08,0.4,0.8,1.0,0.96,0.2,0.1,0.3,0.4,0.2,0.3,0.7,0.0,0.01,0.2,0.5",
                "2,acquisition_stochastic,3,0.56,2.12,0.9,0.85,0.01,0.04,0.2,0.7,1.0,0.63,0.1,0.05,0.2,0.1,0.2,0.15,0.2,0.1,0.02,0.1,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    grid_points = np.array([[0.0, 0.0], [0.56, 0.0], [0.0, 2.12], [0.56, 2.12]], dtype=np.float64)
    np.savez(
        iter_dir / "monitor_predictions_iter065.npz",
        grid_points=grid_points,
        kt_values=np.array([0.0, 0.56], dtype=np.float64),
        ja_values=np.array([0.0, 2.12], dtype=np.float64),
        full_shape=np.array([2, 2], dtype=np.int64),
        candidate_mask=np.ones(4, dtype=np.int8),
        phase_pred=np.array([0, 1, 0, 2], dtype=np.int64),
        A0_main=np.array([1.2, 1.0, 0.8, 0.9], dtype=np.float64),
        A_phase=np.array([1.1, 0.9, 0.75, 0.85], dtype=np.float64),
        A_numerical=np.array([0.02, 0.02, 0.01, 0.01], dtype=np.float64),
        A_explore=np.array([0.08, 0.08, 0.04, 0.04], dtype=np.float64),
        A_response=np.array([0.4, 0.3, 0.2, 0.2], dtype=np.float64),
        cls_uncertainty_mix=np.array([0.2, 0.2, 0.1, 0.1], dtype=np.float64),
        cls_entropy=np.array([0.1, 0.1, 0.05, 0.05], dtype=np.float64),
        cls_margin_uncertainty=np.array([0.3, 0.3, 0.2, 0.2], dtype=np.float64),
        B_q_SC=np.array([0.0, 0.2, 0.0, 0.1], dtype=np.float64),
        E_q_SC=np.array([0.01, 0.02, 0.01, 0.02], dtype=np.float64),
        E_ext_uncertain=np.array([0.2, 0.2, 0.1, 0.1], dtype=np.float64),
        score=np.array([0.96, 0.9, -np.inf, 0.63], dtype=np.float64),
        active_pool_mask=np.array([1, 1, 0, 1], dtype=np.int8),
        Aselect_initial=np.array([0.96, 0.9, -np.inf, 0.63], dtype=np.float64),
        R_obs=np.array([0.8, 0.9, 0.8, 0.7], dtype=np.float64),
    )

    x = np.column_stack([np.linspace(0.0, 0.56, 4), np.linspace(0.0, 2.1, 4)])
    y_reg = np.zeros((4, 5), dtype=np.float64)
    y_reg[:, 0] = [0.0, 0.02, 0.03, 0.0]
    y_reg[:, 1] = [0.0, 0.0, 0.05, 0.0]
    y_phase = np.array([0, 1, 2, 0], dtype=np.int64)
    np.savez(run_dir / "dataset_iter066.npz", x=x, y_reg=y_reg, y_phase=y_phase)
    np.savez(run_dir / "dataset_iter000.npz", x=x[:1], y_reg=y_reg[:1], y_phase=y_phase[:1])
    return run_dir


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        run_dir = _make_fake_run(tmp)
        out = tmp / "report.tex"
        build_report(run_dir, TEMPLATE, out)
        tex = out.read_text(encoding="utf-8")

        assert "final greedy selection score" not in tex
        assert r"p_i =" in tex and r"\gamma = 1" in tex
        assert "Latest completed iteration:} 65" in tex
        assert "Completed active-learning iterations:} 66" in tex
        assert r"Final dataset:} dataset\_iter066.npz" in tex
        assert "missing_cumulative" not in tex
        assert "cumulative_progress.png" in tex or "Figure unavailable:" in tex
        assert "Boundary F1 & N/A" in tex
        assert "Boundary F1 is not available" in tex
        assert "diagnostic maximum displacement" in tex
        assert "StopController boundary-shift metric" in tex
        assert "Selection Region and Component Diagnostics" in tex
        assert "normal interior" in tex
        assert "SC interior" in tex
        assert "Boundary band" in tex
        assert "component attribution" in tex
        assert "selection_region_fractions.png" in tex or "Figure unavailable:" in tex
        assert "exact_eta_revised_boundaries" in tex or "Final exact diode-efficiency data" in tex or "Figure unavailable:" in tex

        diag = run_dir / "iter065" / "selection_region_diagnostics_iter065.json"
        assert diag.exists()
        raw = json.loads(diag.read_text(encoding="utf-8"))
        assert raw["groups"]
        assert raw["component_breakdown"]

    print("report_builder smoke checks passed")


if __name__ == "__main__":
    main()
