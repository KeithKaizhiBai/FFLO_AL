from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_phase.acquisition import build_candidate_grid, compute_acquisition_scores, select_acquisition_batch
from ml_phase.acquisition import observation_repulsion
from ml_phase.active_refine import _empty_flat_dataset, _select_random_seed_points
from ml_phase.config import ActiveLearningConfig, validate_active_learning_config
from ml_phase.stop_controller import StopConfig, evaluate_stop


def test_discovery_mode_disables_bandwidth() -> None:
    bad = ActiveLearningConfig(
        run_mode="discovery",
        candidate_domain_mode="full",
        finite_t_band_width=0.08,
    )
    try:
        validate_active_learning_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("discovery mode accepted finite_t_band_width")

    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=5,
            n_ja_candidates=5,
        )
    )
    grid = build_candidate_grid(cfg)
    high_ja = grid.points[:, 1] > 1.2
    assert np.any(high_ja)
    assert np.all(grid.candidate_mask[high_ja])


def test_random_seed_no_warmup() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            initial_seed_size=12,
            n_kt_candidates=5,
            n_ja_candidates=5,
            random_seed=7,
        )
    )
    dataset = _empty_flat_dataset()
    assert dataset.x.shape == (0, 2)
    grid = build_candidate_grid(cfg)
    selected, meta = _select_random_seed_points(cfg, grid.points, grid.candidate_mask, iteration=0)
    assert selected.shape == (12, 2)
    assert set(meta["selection_source"]) == {"random_seed"}


def test_stochastic_acquisition_selection() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)], axis=1)
    score = np.linspace(0.01, 1.0, 10)
    scores = {
        "score": score.copy(),
        "A0_main": score.copy(),
        "R_obs": np.ones_like(score),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            selection_mode="stochastic",
            batch_size_max=4,
            sampling_power=2.0,
            active_pool_quantile_schedule="constant",
            active_pool_quantile=0.0,
            active_pool_min_quantile=0.0,
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
        )
    )
    selected_a, rows_a = select_acquisition_batch(points, scores, k=4, cfg=cfg, rng_seed=1)
    selected_b, rows_b = select_acquisition_batch(points, scores, k=4, cfg=cfg, rng_seed=2)
    idx_a = [int(r["grid_index"]) for r in rows_a]
    idx_b = [int(r["grid_index"]) for r in rows_b]
    assert selected_a.shape == (4, 2)
    assert len(set(idx_a)) == 4
    assert idx_a != idx_b
    assert np.mean(idx_a + idx_b) > 4.0


def test_active_pool_uses_A0_not_Aselect() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)], axis=1)
    a0 = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05])
    r_obs = np.ones_like(a0)
    r_obs[5] = 0.1
    scores = {
        "score": a0 * r_obs,
        "A0_main": a0,
        "R_obs": r_obs,
        "candidate_excluded": np.zeros_like(a0, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            selection_mode="topk",
            active_pool_quantile=0.90,
            active_pool_quantile_schedule="constant",
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
            batch_size_max=1,
        )
    )
    _, rows = select_acquisition_batch(points, scores, k=1, cfg=cfg, iteration=10)
    assert int(rows[0]["grid_index"]) == 5
    assert bool(scores["active_pool_mask"][5])


def test_stochastic_sampling_restricted_to_active_pool() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 20)], axis=1)
    a0 = np.concatenate([np.full(10, 0.01), np.linspace(0.7, 1.0, 10)])
    r_obs = np.concatenate([np.full(10, 100.0), np.ones(10)])
    scores = {
        "score": a0 * r_obs,
        "A0_main": a0,
        "R_obs": r_obs,
        "candidate_excluded": np.zeros_like(a0, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            selection_mode="stochastic",
            active_pool_quantile=0.80,
            active_pool_quantile_schedule="constant",
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
            batch_size_max=5,
            sampling_power=2.0,
        )
    )
    _, rows = select_acquisition_batch(points, scores, k=5, cfg=cfg, rng_seed=3, iteration=10)
    assert rows
    assert all(int(r["grid_index"]) >= 10 for r in rows)


def test_adaptive_batch_not_forced_full() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100)], axis=1)
    a0 = np.concatenate([np.full(80, 1.0), np.full(20, 0.01)])
    scores = {
        "score": a0.copy(),
        "A0_main": a0.copy(),
        "R_obs": np.ones_like(a0),
        "candidate_excluded": np.zeros_like(a0, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            selection_mode="stochastic",
            active_pool_quantile=0.20,
            active_pool_quantile_schedule="constant",
            active_pool_min_quantile=0.20,
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
            batch_size_max=256,
        )
    )
    _, rows = select_acquisition_batch(points, scores, k=256, cfg=cfg, rng_seed=5, iteration=10)
    assert len(rows) <= 80
    assert scores["_selection_summary"]["selected_batch_size"] <= 80


def test_min_iter_relaxes_threshold() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100)], axis=1)
    a0 = np.arange(100, dtype=np.float64)
    scores = {
        "score": a0.copy(),
        "A0_main": a0.copy(),
        "R_obs": np.ones_like(a0),
        "candidate_excluded": np.zeros_like(a0, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            selection_mode="topk",
            active_pool_quantile=0.95,
            active_pool_quantile_schedule="constant",
            active_pool_min_quantile=0.70,
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
            batch_size_max=64,
            batch_size_min_before_min_iter=30,
        )
    )
    _, rows = select_acquisition_batch(points, scores, k=64, cfg=cfg, iteration=1)
    assert len(rows) >= 30
    assert scores["_selection_summary"]["active_pool_threshold_relaxed"] is True


def test_Robs_mild_floor() -> None:
    d = np.array([0.0, 0.01, 1.0])
    r = observation_repulsion(d, ell=0.02, floor=0.5)
    assert np.min(r) >= 0.5
    assert r[0] == 0.5


def test_effective_sample_size_diagnostic() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)], axis=1)
    uniform = np.ones(10)
    scores = {
        "score": uniform.copy(),
        "A0_main": uniform.copy(),
        "R_obs": np.ones_like(uniform),
        "candidate_excluded": np.zeros_like(uniform, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(run_mode="discovery", candidate_domain_mode="full", finite_t_band_width=None, selection_mode="stochastic", active_pool_quantile_schedule="constant", active_pool_quantile=0.0, active_pool_min_quantile=0.0, active_pool_rel_to_p95=0.0, active_pool_max_fraction_start=1.0, active_pool_max_fraction_end=1.0)
    )
    select_acquisition_batch(points, scores, k=1, cfg=cfg, rng_seed=1, iteration=10)
    assert abs(float(scores["_selection_summary"]["N_eff"]) - 10.0) < 1.0e-9
    concentrated = uniform.copy()
    concentrated[-1] = 100.0
    scores2 = {
        "score": concentrated.copy(),
        "A0_main": concentrated.copy(),
        "R_obs": np.ones_like(concentrated),
        "candidate_excluded": np.zeros_like(concentrated, dtype=np.int8),
    }
    select_acquisition_batch(points, scores2, k=1, cfg=cfg, rng_seed=1, iteration=10)
    assert float(scores2["_selection_summary"]["N_eff"]) < 3.0


def test_bdelta_gated_suppresses_deep_normal() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=2,
            n_ja_candidates=2,
        )
    )
    grid = build_candidate_grid(cfg)
    n = grid.points.shape[0]
    predictions = {
        "reg_mean": np.tile(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1)),
        "reg_std": np.zeros((n, 5), dtype=np.float64),
        "phase_proba": np.tile(np.array([[0.99, 0.01, 0.0]], dtype=np.float64), (n, 1)),
        "cls_uncertainty": np.zeros(n, dtype=np.float64),
    }
    scores = compute_acquisition_scores(cfg, grid, predictions, existing_points=np.empty((0, 2)), iteration=20)
    assert 0.0 < float(scores["U_NS"][0]) < 0.05
    assert float(scores["B_delta_raw"][0]) > 0.95
    assert float(scores["B_delta_gated"][0]) < 0.05


def test_bdelta_gated_preserves_normal_sc_boundary() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=2,
            n_ja_candidates=2,
        )
    )
    grid = build_candidate_grid(cfg)
    n = grid.points.shape[0]
    predictions = {
        "reg_mean": np.tile(np.array([[cfg.delta_eps, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1)),
        "reg_std": np.zeros((n, 5), dtype=np.float64),
        "phase_proba": np.tile(np.array([[0.5, 0.5, 0.0]], dtype=np.float64), (n, 1)),
        "cls_uncertainty": np.zeros(n, dtype=np.float64),
    }
    scores = compute_acquisition_scores(cfg, grid, predictions, existing_points=np.empty((0, 2)), iteration=20)
    assert abs(float(scores["U_NS"][0]) - 1.0) < 1.0e-12
    assert abs(float(scores["B_delta_gated"][0]) - float(scores["B_delta_raw"][0])) < 1.0e-12


def test_active_pool_max_threshold_not_or() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)], axis=1)
    a0_pool = np.linspace(0.1, 1.0, 10)
    scores = {
        "score": a0_pool.copy(),
        "A0_main": a0_pool.copy(),
        "A0_for_pool": a0_pool.copy(),
        "R_obs": np.ones_like(a0_pool),
        "candidate_excluded": np.zeros_like(a0_pool, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            active_pool_rule="max_threshold",
            active_pool_quantile_schedule="constant",
            active_pool_quantile=0.5,
            active_pool_min_quantile=0.5,
            active_pool_rel_to_p95=1.05,
            active_pool_max_fraction_start=1.0,
            active_pool_max_fraction_end=1.0,
            selection_mode="topk",
        )
    )
    select_acquisition_batch(points, scores, k=10, cfg=cfg, iteration=10)
    summary = scores["_selection_summary"]
    assert summary["active_pool_threshold_final"] == max(
        summary["active_pool_threshold_quantile"],
        summary["active_pool_threshold_rel_p95"],
    )
    assert summary["active_pool_size"] <= 1


def test_active_pool_fraction_cap() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100)], axis=1)
    a0 = np.linspace(0.0, 1.0, 100)
    scores = {
        "score": a0.copy(),
        "A0_main": a0.copy(),
        "A0_for_pool": a0.copy(),
        "R_obs": np.ones_like(a0),
        "candidate_excluded": np.zeros_like(a0, dtype=np.int8),
    }
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            active_pool_quantile=0.5,
            active_pool_quantile_schedule="constant",
            active_pool_min_quantile=0.5,
            active_pool_rel_to_p95=0.0,
            active_pool_max_fraction_start=0.1,
            active_pool_max_fraction_end=0.1,
            selection_mode="topk",
        )
    )
    select_acquisition_batch(points, scores, k=100, cfg=cfg, iteration=20)
    assert scores["_selection_summary"]["active_pool_fraction_cap_tightened"] is True
    assert scores["_selection_summary"]["active_pool_fraction"] <= 0.11


def test_high_confidence_interior_penalty() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=2,
            n_ja_candidates=2,
        )
    )
    grid = build_candidate_grid(cfg)
    n = grid.points.shape[0]
    predictions = {
        "reg_mean": np.tile(np.array([[0.2, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1)),
        "reg_std": np.zeros((n, 5), dtype=np.float64),
        "phase_proba": np.tile(np.array([[0.99, 0.01, 0.0]], dtype=np.float64), (n, 1)),
        "cls_uncertainty": np.zeros(n, dtype=np.float64),
    }
    scores = compute_acquisition_scores(cfg, grid, predictions, existing_points=np.empty((0, 2)), iteration=20)
    assert int(scores["high_confidence_interior"][0]) == 1
    assert abs(float(scores["A0_for_pool"][0]) - 0.1 * float(scores["A0_main"][0])) < 1.0e-12


def test_boundary_candidate_not_penalized() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=2,
            n_ja_candidates=2,
        )
    )
    grid = build_candidate_grid(cfg)
    n = grid.points.shape[0]
    predictions = {
        "reg_mean": np.tile(np.array([[cfg.delta_eps, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (n, 1)),
        "reg_std": np.zeros((n, 5), dtype=np.float64),
        "phase_proba": np.tile(np.array([[0.5, 0.5, 0.0]], dtype=np.float64), (n, 1)),
        "cls_uncertainty": np.ones(n, dtype=np.float64),
    }
    scores = compute_acquisition_scores(cfg, grid, predictions, existing_points=np.empty((0, 2)), iteration=20)
    assert int(scores["high_confidence_interior"][0]) == 0
    assert abs(float(scores["A0_for_pool"][0]) - float(scores["A0_main"][0])) < 1.0e-12


def test_sampling_power_schedule() -> None:
    points = np.stack([np.linspace(0.0, 1.0, 10), np.linspace(0.0, 1.0, 10)], axis=1)
    a0 = np.ones(10)
    cfg = validate_active_learning_config(
        ActiveLearningConfig(run_mode="discovery", candidate_domain_mode="full", finite_t_band_width=None, active_pool_quantile_schedule="constant", active_pool_quantile=0.0, active_pool_min_quantile=0.0, active_pool_rel_to_p95=0.0, active_pool_max_fraction_start=1.0, active_pool_max_fraction_end=1.0)
    )
    for iteration, expected in [(0, 1.5), (10, 2.5), (30, 4.0)]:
        scores = {"score": a0.copy(), "A0_main": a0.copy(), "A0_for_pool": a0.copy(), "R_obs": np.ones_like(a0)}
        _, rows = select_acquisition_batch(points, scores, k=1, cfg=cfg, iteration=iteration)
        assert rows and abs(float(rows[0]["sampling_power"]) - expected) < 1.0e-12


def test_w_ext_schedule() -> None:
    cfg = validate_active_learning_config(
        ActiveLearningConfig(
            run_mode="discovery",
            candidate_domain_mode="full",
            finite_t_band_width=None,
            n_kt_candidates=2,
            n_ja_candidates=2,
        )
    )
    grid = build_candidate_grid(cfg)
    n = grid.points.shape[0]
    predictions = {
        "reg_mean": np.zeros((n, 5), dtype=np.float64),
        "reg_std": np.ones((n, 5), dtype=np.float64),
        "phase_proba": np.tile(np.array([[0.5, 0.5, 0.0]], dtype=np.float64), (n, 1)),
        "cls_uncertainty": np.ones(n, dtype=np.float64),
    }
    expected = [(0, 0.15), (10, 0.08), (30, 0.03)]
    for iteration, w_ext in expected:
        scores = compute_acquisition_scores(cfg, grid, predictions, existing_points=np.empty((0, 2)), iteration=iteration)
        assert abs(float(scores["w_ext_current"][0]) - w_ext) < 1.0e-12


def test_no_fixed_quota() -> None:
    text = (
        Path("ml_phase/acquisition.py").read_text(encoding="utf-8")
        + Path("ml_phase/active_refine.py").read_text(encoding="utf-8")
    ).lower()
    forbidden = ["boundary_quota", "exploration_quota", "70%", "20%", "10%"]
    assert not any(token in text for token in forbidden)


def _write_monitor(path: Path, phase: np.ndarray, cfg: ActiveLearningConfig) -> None:
    kt_vals = np.linspace(cfg.kt_min, cfg.kt_max, 5)
    ja_vals = np.linspace(cfg.ja_min, cfg.ja_max, 5)
    kt, ja = np.meshgrid(kt_vals, ja_vals, indexing="xy")
    np.savez(
        path,
        grid_points=np.stack([kt.ravel(), ja.ravel()], axis=1),
        phase_pred=phase.ravel(),
        full_shape=np.asarray(phase.shape, dtype=np.int64),
    )


def test_stop_without_old_c5_c6_mandatory() -> None:
    cfg = ActiveLearningConfig(
        run_mode="discovery",
        candidate_domain_mode="full",
        finite_t_band_width=None,
        n_kt_candidates=5,
        n_ja_candidates=5,
    )
    stop_cfg = StopConfig(min_iterations=1, patience=1, max_iterations=10, required_pass_count=4)
    phase = np.zeros((5, 5), dtype=np.int64)
    phase[:, 2:4] = 1
    phase[:, 4:] = 2
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        dataset = run_dir / "dataset_iter001.npz"
        points = build_candidate_grid(cfg).points
        np.savez(dataset, x=points, y_reg=np.zeros((points.shape[0], 5)), y_phase=np.zeros(points.shape[0]), y_eta_sign=np.zeros(points.shape[0]), y_strong_diode=np.zeros(points.shape[0]))
        for it in range(2):
            iter_dir = run_dir / f"iter{it:03d}"
            iter_dir.mkdir()
            _write_monitor(iter_dir / f"monitor_predictions_iter{it:03d}.npz", phase, cfg)
            pd.DataFrame({"kT": points[:4, 0], "JA": points[:4, 1], "predicted_phase_before_exact": [0, 0, 0, 0], "A0_main": [10, 10, 10, 10]}).to_csv(iter_dir / "selected_points_by_pool.csv", index=False)
            np.savez(
                iter_dir / f"exact_merged_iter{it:03d}.npz",
                kT=points[:4, 0],
                JA=points[:4, 1],
                delta_opt=np.zeros(4),
                q_opt=np.zeros(4),
                q_edge_hit=np.ones(4),
                training_eligible_exact=np.zeros(4),
            )
            result = evaluate_stop(run_dir, it, dataset, cfg, stop_cfg)
        assert result["stop"] is True
        assert result["stop_reason"] == "converged_main_phase_boundaries"
        assert result["numerical_cleanup_warning"] is True


def test_hidden_ground_truth_not_used_online() -> None:
    text = Path("ml_phase/acquisition.py").read_text(encoding="utf-8") + Path("ml_phase/stop_controller.py").read_text(encoding="utf-8")
    assert "hidden_ground_truth" not in text


def main() -> None:
    test_discovery_mode_disables_bandwidth()
    test_random_seed_no_warmup()
    test_stochastic_acquisition_selection()
    test_active_pool_uses_A0_not_Aselect()
    test_stochastic_sampling_restricted_to_active_pool()
    test_adaptive_batch_not_forced_full()
    test_min_iter_relaxes_threshold()
    test_Robs_mild_floor()
    test_effective_sample_size_diagnostic()
    test_bdelta_gated_suppresses_deep_normal()
    test_bdelta_gated_preserves_normal_sc_boundary()
    test_active_pool_max_threshold_not_or()
    test_active_pool_fraction_cap()
    test_high_confidence_interior_penalty()
    test_boundary_candidate_not_penalized()
    test_sampling_power_schedule()
    test_w_ext_schedule()
    test_no_fixed_quota()
    test_stop_without_old_c5_c6_mandatory()
    test_hidden_ground_truth_not_used_online()
    print("discovery-mode checks passed")


if __name__ == "__main__":
    main()
