from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from .acquisition import (
    build_candidate_grid,
    compute_acquisition_scores,
    normalized_min_distance,
    observation_repulsion,
    select_acquisition_batch,
)
from .config import ActiveLearningConfig, ensure_output_dirs, validate_active_learning_config
from .dataset_builder import (
    OPTIONAL_RECORD_DEFAULTS,
    FlatDataset,
    _with_optional_record_defaults,
    build_warm_start_dataset,
    load_flat_dataset,
)
from .evaluate import evaluate_predictions
from .exact_oracle import evaluate_points
from .extract_phase_boundaries import extract_phase_boundaries
from .hpc import write_point_shards
from .labels import PHASE_NAMES, eta_sign_label, phase_label, strong_diode_label
from .models import ModelBundle, predict_models, train_models
from .plot_active_learning import write_iteration_figures, write_learning_curve


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Active-learning phase-boundary refinement.")
    p.add_argument("--warm-start", type=Path, default=None, help="Path to eta_phase_diagram_*.npz")
    p.add_argument("--resume-dataset", type=Path, default=None, help="Existing dataset_iterXXX.npz to continue from.")
    p.add_argument("--run-mode", type=str, default="discovery", choices=["discovery", "refinement"])
    p.add_argument("--candidate-domain-mode", type=str, default="full", choices=["full", "prior_band"])
    p.add_argument("--initialization", type=str, default="random_grid", choices=["random_grid", "sobol_scrambled"])
    p.add_argument("--initial-seed-size", type=int, default=512)
    p.add_argument("--batch-size-max", type=int, default=256)
    p.add_argument("--batch-size-min", type=int, default=0)
    p.add_argument("--batch-size-min-before-min-iter", type=int, default=64)
    p.add_argument("--batch-size-min-after-min-iter", type=int, default=0)
    p.add_argument("--selection-mode", type=str, default="stochastic", choices=["topk", "stochastic"])
    p.add_argument("--sampling-power", type=float, default=2.0)
    p.add_argument("--sampling-power-start", type=float, default=1.5)
    p.add_argument("--sampling-power-mid", type=float, default=2.5)
    p.add_argument("--sampling-power-end", type=float, default=4.0)
    p.add_argument("--sampling-power-mid-iter", type=int, default=10)
    p.add_argument("--sampling-power-end-iter", type=int, default=30)
    p.add_argument("--sampling-power-schedule", type=str, default="piecewise", choices=["constant", "linear", "piecewise"])
    p.add_argument("--score-threshold-abs", type=float, default=0.0)
    p.add_argument("--score-threshold-rel", type=float, default=0.0)
    p.add_argument("--acquisition-profile", type=str, default="full", choices=["full", "simple_phase", "surprise_cleanup", "topo_trivial"])
    p.add_argument("--active-pool-rule", type=str, default="max_threshold", choices=["legacy_or", "max_threshold"])
    p.add_argument("--active-pool-quantile", type=float, default=0.90)
    p.add_argument("--active-pool-quantile-schedule", type=str, default="piecewise", choices=["constant", "piecewise"])
    p.add_argument("--active-pool-quantile-start", type=float, default=0.90)
    p.add_argument("--active-pool-quantile-mid", type=float, default=0.95)
    p.add_argument("--active-pool-quantile-end", type=float, default=0.98)
    p.add_argument("--active-pool-quantile-mid-iter", type=int, default=10)
    p.add_argument("--active-pool-quantile-end-iter", type=int, default=30)
    p.add_argument("--active-pool-rel-to-p95", type=float, default=0.7)
    p.add_argument("--active-pool-min-quantile", type=float, default=0.70)
    p.add_argument("--active-pool-max-fraction-start", type=float, default=0.20)
    p.add_argument("--active-pool-max-fraction-end", type=float, default=0.05)
    p.add_argument("--active-pool-max-fraction-end-iter", type=int, default=30)
    p.add_argument("--active-selection-min-iterations", type=int, default=5)
    p.add_argument("--no-underfilled-batch-after-min-iter", action="store_true")
    p.add_argument("--b-delta-gate-mode", type=str, default="normal_sc_competition", choices=["none", "normal_sc_competition"])
    p.add_argument("--q-boundary-gate-mode", type=str, default="psc", choices=["psc", "uf_competition"])
    p.add_argument("--interior-filter-mode", type=str, default="soft_penalty", choices=["off", "soft_penalty", "hard_exclude"])
    p.add_argument("--interior-penalty-start-iter", type=int, default=10)
    p.add_argument("--interior-penalty-early", type=float, default=0.5)
    p.add_argument("--interior-penalty-late", type=float, default=0.1)
    p.add_argument("--p-conf-threshold", type=float, default=0.98)
    p.add_argument("--u-ns-low", type=float, default=0.05)
    p.add_argument("--u-uf-low", type=float, default=0.05)
    p.add_argument("--g-phase-low", type=float, default=0.05)
    p.add_argument("--e-q-low", type=float, default=0.05)
    p.add_argument("--e-ext-low", type=float, default=0.05)
    p.add_argument("--w-ext-schedule", type=str, default="piecewise", choices=["constant", "piecewise"])
    p.add_argument("--w-ext-start", type=float, default=0.15)
    p.add_argument("--w-ext-mid", type=float, default=0.08)
    p.add_argument("--w-ext-end", type=float, default=0.03)
    p.add_argument("--w-ext-mid-iter", type=int, default=10)
    p.add_argument("--w-ext-end-iter", type=int, default=30)
    p.add_argument("--w-cls-simple", type=float, default=1.0)
    p.add_argument("--w-ns-simple", type=float, default=1.0)
    p.add_argument("--w-uf-simple", type=float, default=0.5)
    p.add_argument("--w-grad-simple", type=float, default=0.2)
    p.add_argument("--w-reg-simple", type=float, default=0.1)
    p.add_argument("--w-ext-simple-schedule", type=str, default="piecewise", choices=["constant", "piecewise"])
    p.add_argument("--w-ext-simple-start", type=float, default=0.02)
    p.add_argument("--w-ext-simple-mid", type=float, default=0.01)
    p.add_argument("--w-ext-simple-end", type=float, default=0.0)
    p.add_argument("--w-ext-simple-mid-iter", type=int, default=10)
    p.add_argument("--w-ext-simple-end-iter", type=int, default=30)
    p.add_argument("--surprise-cleanup-qedge-penalty", type=float, default=0.85)
    p.add_argument("--surprise-cleanup-qedge-floor", type=float, default=0.05)
    p.add_argument("--surprise-cleanup-response-weight", type=float, default=0.25)
    p.add_argument("--surprise-cleanup-explore-scale", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--finite-t-band-width", type=float, default=None)
    p.add_argument("--hidden-ground-truth", type=Path, default=None)
    p.add_argument("--start-iteration", type=int, default=0, help="Iteration number used for output paths.")
    p.add_argument("--run-id", type=str, required=True, help="Run identifier")
    p.add_argument("--iterations", type=int, default=100, help="Number of active-learning iterations")
    p.add_argument("--points-per-iter", type=int, default=256, help="Selected exact points per iteration")
    p.add_argument("--mode", type=str, default="local", choices=["local", "hpc"], help="Execution mode")
    p.add_argument("--world-size", type=int, default=1, help="Number of H100 ranks/tasks for hpc mode")
    p.add_argument("--partition-strategy", type=str, default="round_robin", help="round_robin|contiguous|cost_aware")
    p.add_argument("--dry-run", action="store_true", help="Select points but skip exact oracle")
    p.add_argument("--device", type=str, default=None, help="Torch device for local exact oracle, e.g. cuda:0")
    p.add_argument(
        "--oracle-mode",
        type=str,
        default="robust_al",
        choices=["legacy", "robust_al", "robust_incremental"],
        help="Exact-oracle mode used for pointwise BdG labels.",
    )
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"), help="Output root")
    p.add_argument("--n-ensemble", type=int, default=5, help="Model ensemble size")
    p.add_argument("--reg-epochs", type=int, default=240, help="Regression epochs per ensemble member")
    p.add_argument("--cls-epochs", type=int, default=240, help="Classification epochs per ensemble member")
    p.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    p.add_argument("--submit", action="store_true", help="In hpc mode, submit slurm array job")
    p.add_argument("--disable-early-stop", action="store_true", help="Disable low-new-point early stopping.")
    p.add_argument(
        "--boundary-refinement-mode",
        type=str,
        default="off",
        choices=["off", "diagnostic", "hybrid", "local"],
        help="Boundary diagnostic mode. hybrid/local are legacy midpoint modes and now raise an error.",
    )
    p.add_argument("--boundary-kt-bin-width", type=float, default=0.005, help="kT bin width for boundary extraction.")
    p.add_argument(
        "--boundary-max-local-spacing",
        type=float,
        default=0.035,
        help="Normalized bracket spacing above which extracted boundary segments are low confidence.",
    )
    p.add_argument(
        "--boundary-position-tol",
        type=float,
        default=0.00375,
        help="Normalized boundary displacement tolerance used for diagnostics.",
    )
    p.add_argument("--boundary-stable-stages", type=int, default=2, help="Stable boundary stages needed for convergence.")
    p.add_argument(
        "--min-new-points-per-iter",
        type=int,
        default=8,
        help="Stop after repeated iterations adding fewer than this many unique samples.",
    )
    p.add_argument(
        "--max-low-append-iters",
        type=int,
        default=2,
        help="Consecutive low-append iterations allowed before stopping.",
    )
    p.add_argument(
        "--slurm-script",
        type=Path,
        default=Path("scripts/slurm_exact_oracle_array.sh"),
        help="SLURM array script path for --mode hpc --submit",
    )
    return p.parse_args()


def _dataset_from_result(dataset: FlatDataset, result: Dict[str, np.ndarray], cfg: ActiveLearningConfig) -> FlatDataset:
    use_mu = ("mu" in result) or int(dataset.x.shape[1]) >= 3
    if use_mu:
        mu_vals = result.get("mu", np.full(np.asarray(result["kT"]).shape[0], 0.55, dtype=np.float64))
        new_x = np.stack([result["kT"], result["JA"], mu_vals], axis=1).astype(np.float64)
        old_x = dataset.x
        if old_x.shape[1] == 2:
            old_mu = dataset.records.get("mu", np.full(old_x.shape[0], 0.55, dtype=np.float64))
            old_x = np.column_stack([old_x, old_mu]).astype(np.float64)
    else:
        new_x = np.stack([result["kT"], result["JA"]], axis=1).astype(np.float64)
        old_x = dataset.x
    new_y_reg = np.stack(
        [result["delta_opt"], result["q_opt"], result["eta"], result["ic_plus"], result["ic_minus"]],
        axis=1,
    ).astype(np.float64)

    x_all = np.vstack([old_x, new_x])
    y_reg_all = np.vstack([dataset.y_reg, new_y_reg])

    # Deduplicate by (kT, JA), keep latest entry.
    keys = np.round(x_all, decimals=12)
    _, rev_idx = np.unique(keys[::-1], axis=0, return_index=True)
    uniq_idx = np.sort(keys.shape[0] - 1 - rev_idx)
    x_all = x_all[uniq_idx]
    y_reg_all = y_reg_all[uniq_idx]

    y_phase = phase_label(y_reg_all[:, 0], y_reg_all[:, 1], cfg.delta_eps, cfg.q_eps)
    y_eta_sign = eta_sign_label(y_reg_all[:, 2])
    y_strong = strong_diode_label(y_reg_all[:, 2], cfg.eta_strong)

    records: Dict[str, np.ndarray] = {
        "kT": x_all[:, 0],
        "JA": x_all[:, 1],
        "mu": x_all[:, 2] if x_all.shape[1] >= 3 else np.full(x_all.shape[0], 0.55, dtype=np.float64),
        "delta_opt": y_reg_all[:, 0],
        "q_opt": y_reg_all[:, 1],
        "eta": y_reg_all[:, 2],
        "ic_plus": y_reg_all[:, 3],
        "ic_minus": y_reg_all[:, 4],
        "phase_label": y_phase,
        "phase_name": np.vectorize(PHASE_NAMES.get)(y_phase),
        "eta_sign_label": y_eta_sign,
        "strong_diode_label": y_strong,
    }
    for key, default in OPTIONAL_RECORD_DEFAULTS.items():
        old_vals = dataset.records.get(key)
        if old_vals is None:
            old_vals = np.full(dataset.x.shape[0], default)
        new_vals = result.get(key)
        if new_vals is None:
            new_vals = np.full(new_x.shape[0], default)
        records[key] = np.concatenate([old_vals, new_vals], axis=0)[uniq_idx]
    records = _with_optional_record_defaults(records, x_all.shape[0])

    return FlatDataset(
        x=x_all,
        y_reg=y_reg_all,
        y_phase=y_phase.astype(np.int64),
        y_eta_sign=y_eta_sign.astype(np.int64),
        y_strong_diode=y_strong.astype(np.int64),
        records=records,
    )


def _empty_flat_dataset() -> FlatDataset:
    x = np.empty((0, 2), dtype=np.float64)
    y_reg = np.empty((0, 5), dtype=np.float64)
    y_phase = np.empty((0,), dtype=np.int64)
    y_eta_sign = np.empty((0,), dtype=np.int64)
    y_strong = np.empty((0,), dtype=np.int64)
    records: Dict[str, np.ndarray] = {
        "kT": x[:, 0],
        "JA": x[:, 1],
        "delta_opt": y_reg[:, 0],
        "q_opt": y_reg[:, 1],
        "eta": y_reg[:, 2],
        "ic_plus": y_reg[:, 3],
        "ic_minus": y_reg[:, 4],
        "phase_label": y_phase,
        "phase_name": np.empty((0,), dtype="<U16"),
        "eta_sign_label": y_eta_sign,
        "strong_diode_label": y_strong,
    }
    records = _with_optional_record_defaults(records, 0)
    return FlatDataset(
        x=x,
        y_reg=y_reg,
        y_phase=y_phase,
        y_eta_sign=y_eta_sign,
        y_strong_diode=y_strong,
        records=records,
    )


def _save_dataset(iter_dir: Path, iteration: int, dataset: FlatDataset) -> Tuple[Path, Path]:
    npz_path = iter_dir.parent / f"dataset_iter{iteration:03d}.npz"
    csv_path = iter_dir.parent / f"dataset_iter{iteration:03d}.csv"
    np.savez(
        npz_path,
        x=dataset.x,
        y_reg=dataset.y_reg,
        y_phase=dataset.y_phase,
        y_eta_sign=dataset.y_eta_sign,
        y_strong_diode=dataset.y_strong_diode,
        **{k: v for k, v in dataset.records.items() if k in OPTIONAL_RECORD_DEFAULTS},
    )
    pd.DataFrame(dataset.records).to_csv(csv_path, index=False)
    return npz_path, csv_path


def _topology_context_from_dataset(dataset: FlatDataset) -> Dict[str, np.ndarray]:
    records = dataset.records
    n = int(dataset.x.shape[0])
    empty = np.empty((0, 2), dtype=np.float64)
    if n == 0:
        return {
            "trivial_points": empty,
            "topological_points": empty,
            "gapless_points": empty,
            "trusted_topology_points": empty,
        }
    topo_enabled = np.asarray(records.get("topology_enabled", np.zeros(n, dtype=np.int8))).astype(bool)
    topo_trusted = np.asarray(records.get("topology_trusted", np.zeros(n, dtype=np.int8))).astype(bool)
    topo_label = np.asarray(records.get("topology_label_code", np.full(n, -1, dtype=np.int64))).astype(np.int64)
    finite_xy = np.all(np.isfinite(dataset.x), axis=1)
    base = topo_enabled & topo_trusted & finite_xy
    trivial = base & (topo_label == 0)
    topological = base & (topo_label == 1)
    gapless = base & (topo_label == 2)
    trusted = trivial | topological | gapless
    return {
        "trivial_points": dataset.x[trivial],
        "topological_points": dataset.x[topological],
        "gapless_points": dataset.x[gapless],
        "trusted_topology_points": dataset.x[trusted],
    }


def _save_candidate_csv(
    iter_dir: Path,
    grid_points: np.ndarray,
    scores: Dict[str, np.ndarray],
    filename: str = "candidate_scores.csv",
) -> Path:
    df = pd.DataFrame(
        {
            "kT": grid_points[:, 0],
            "JA": grid_points[:, 1],
            "score": scores["score"],
            "A0_main": scores.get("A0_main", scores["score"]),
            "A0_main_raw": scores.get("A0_main_raw", scores.get("A0_main", scores["score"])),
            "A0_for_pool": scores.get("A0_for_pool", scores.get("A0_main", scores["score"])),
            "A_phase": scores.get("A_phase", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_numerical": scores.get("A_numerical", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_explore": scores.get("A_explore", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_response": scores.get("A_response", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_spectral": scores.get("A_spectral", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_topology": scores.get("A_topology", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_topology_pf_margin": scores.get("A_topology_pf_margin", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_topology_z2_edge": scores.get("A_topology_z2_edge", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_topology_gapless_edge": scores.get("A_topology_gapless_edge", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "A_coverage": scores.get("A_coverage", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "Aselect_initial": scores.get("Aselect_initial", scores["score"]),
            "R_obs": scores.get("R_obs", np.ones(grid_points.shape[0], dtype=np.float64)),
            "active_pool_mask": scores.get("active_pool_mask", np.zeros(grid_points.shape[0], dtype=np.int8)),
            "cls_uncertainty": scores["cls_uncertainty"],
            "cls_uncertainty_mix": scores.get("cls_uncertainty", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "cls_entropy": scores.get("cls_entropy", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "cls_margin_uncertainty": scores.get("cls_margin_uncertainty", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "P_normal": scores.get("P_normal", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "P_uniform": scores.get("P_uniform", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "P_FFLO": scores.get("P_FFLO", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "P_SC": scores.get("P_SC", np.ones(grid_points.shape[0], dtype=np.float64)),
            "U_NS": scores.get("U_NS", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_UF": scores.get("U_UF", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "reg_uncertainty": scores["reg_uncertainty"],
            "U_delta": scores.get("U_delta", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_q": scores.get("U_q", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_eta": scores.get("U_eta", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_ic_plus": scores.get("U_ic_plus", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_ic_minus": scores.get("U_ic_minus", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "U_reg_phase": scores.get("U_reg_phase", scores["reg_uncertainty"]),
            "U_reg_response": scores.get("U_reg_response", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "B_delta_raw": scores.get("B_delta_raw", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "B_delta_gated": scores.get("B_delta_gated", scores["delta_boundary_score"]),
            "delta_boundary_score": scores["delta_boundary_score"],
            "B_delta": scores["delta_boundary_score"],
            "q_boundary_score": scores["q_boundary_score"],
            "B_q_SC": scores["q_boundary_score"],
            "B_q_raw": scores.get("B_q_raw", scores.get("q_boundary_raw", np.zeros(grid_points.shape[0], dtype=np.float64))),
            "B_q_gated": scores.get("B_q_gated", scores["q_boundary_score"]),
            "q_boundary_raw": scores.get("q_boundary_raw", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "eta_zero_score": scores["eta_zero_score"],
            "eta_zero_raw": scores.get("eta_zero_raw", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "gradient_score": scores["gradient_score"],
            "gradient_response": scores.get("gradient_response", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "diversity_score": scores["diversity_score"],
            "q_edge_risk_score": scores["q_edge_risk_score"],
            "E_q_SC": scores["q_edge_risk_score"],
            "q_edge_risk_raw": scores.get("q_edge_risk_raw", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "surprise_cleanup_qedge_factor": scores.get(
                "surprise_cleanup_qedge_factor", np.ones(grid_points.shape[0], dtype=np.float64)
            ),
            "delta_refine_risk_score": scores["delta_refine_risk_score"],
            "extrapolation_risk_score": scores["extrapolation_risk_score"],
            "E_ext_uncertain": scores["extrapolation_risk_score"],
            "extrapolation_raw": scores.get("extrapolation_raw", np.zeros(grid_points.shape[0], dtype=np.float64)),
            "w_ext_current": scores.get("w_ext_current", np.full(grid_points.shape[0], np.nan, dtype=np.float64)),
            "topology_pfaffian_margin_pred": scores.get("topology_pfaffian_margin_pred", np.full(grid_points.shape[0], np.nan, dtype=np.float64)),
            "topology_distance_to_trivial": scores.get("topology_distance_to_trivial", np.full(grid_points.shape[0], np.inf, dtype=np.float64)),
            "topology_distance_to_topological": scores.get("topology_distance_to_topological", np.full(grid_points.shape[0], np.inf, dtype=np.float64)),
            "topology_distance_to_gapless": scores.get("topology_distance_to_gapless", np.full(grid_points.shape[0], np.inf, dtype=np.float64)),
            "topology_distance_to_trusted": scores.get("topology_distance_to_trusted", np.full(grid_points.shape[0], np.inf, dtype=np.float64)),
            "topology_trivial_count": scores.get("topology_trivial_count", np.zeros(grid_points.shape[0], dtype=np.int64)),
            "topology_topological_count": scores.get("topology_topological_count", np.zeros(grid_points.shape[0], dtype=np.int64)),
            "topology_gapless_count": scores.get("topology_gapless_count", np.zeros(grid_points.shape[0], dtype=np.int64)),
            "topology_trusted_count": scores.get("topology_trusted_count", np.zeros(grid_points.shape[0], dtype=np.int64)),
            "interior_penalty": scores.get("interior_penalty", np.ones(grid_points.shape[0], dtype=np.float64)),
            "interior_penalty_applied": scores.get("interior_penalty_applied", np.zeros(grid_points.shape[0], dtype=np.int8)),
            "high_confidence_interior": scores.get("high_confidence_interior", np.zeros(grid_points.shape[0], dtype=np.int8)),
            "active_pool_candidate_mask": scores.get("active_pool_candidate_mask", np.ones(grid_points.shape[0], dtype=np.int8)),
            "score_raw": scores.get("score_raw", scores["score"]),
            "existing_exact_exclusion": scores.get(
                "existing_exact_exclusion", np.zeros(grid_points.shape[0], dtype=np.int8)
            ),
            "existing_distance_exclusion": scores.get(
                "existing_distance_exclusion", np.zeros(grid_points.shape[0], dtype=np.int8)
            ),
            "existing_min_distance": scores.get(
                "existing_min_distance", np.full(grid_points.shape[0], np.inf, dtype=np.float64)
            ),
            "recent_selected_exclusion": scores.get(
                "recent_selected_exclusion", np.zeros(grid_points.shape[0], dtype=np.int8)
            ),
            "boundary_band_cooldown": scores.get("boundary_band_cooldown", np.zeros(grid_points.shape[0], dtype=np.int8)),
            "candidate_excluded": scores.get("candidate_excluded", np.zeros(grid_points.shape[0], dtype=np.int8)),
        }
    )
    out = iter_dir / filename
    df.to_csv(out, index=False)
    return out


def _save_monitor_predictions(
    iter_dir: Path,
    iteration: int,
    grid,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
) -> Path:
    out = iter_dir / f"monitor_predictions_iter{iteration:03d}.npz"
    np.savez(
        out,
        grid_points=np.asarray(grid.points, dtype=np.float64),
        kt_values=np.asarray(grid.kt_values, dtype=np.float64),
        ja_values=np.asarray(grid.ja_values, dtype=np.float64),
        full_shape=np.asarray(grid.full_shape, dtype=np.int64),
        candidate_mask=np.asarray(grid.candidate_mask, dtype=np.int8),
        phase_pred=np.asarray(predictions["phase_pred"], dtype=np.int64),
        A0_main=np.asarray(scores.get("A0_main", scores["score"]), dtype=np.float64),
        A0_main_raw=np.asarray(scores.get("A0_main_raw", scores.get("A0_main", scores["score"])), dtype=np.float64),
        A0_for_pool=np.asarray(scores.get("A0_for_pool", scores.get("A0_main", scores["score"])), dtype=np.float64),
        A_phase=np.asarray(scores.get("A_phase", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_numerical=np.asarray(scores.get("A_numerical", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_explore=np.asarray(scores.get("A_explore", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_response=np.asarray(scores.get("A_response", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_spectral=np.asarray(scores.get("A_spectral", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_topology=np.asarray(scores.get("A_topology", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_topology_pf_margin=np.asarray(scores.get("A_topology_pf_margin", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_topology_z2_edge=np.asarray(scores.get("A_topology_z2_edge", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_topology_gapless_edge=np.asarray(scores.get("A_topology_gapless_edge", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        A_coverage=np.asarray(scores.get("A_coverage", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        cls_uncertainty_mix=np.asarray(scores.get("cls_uncertainty", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        cls_entropy=np.asarray(scores.get("cls_entropy", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        cls_margin_uncertainty=np.asarray(scores.get("cls_margin_uncertainty", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        U_delta=np.asarray(scores.get("U_delta", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        U_q=np.asarray(scores.get("U_q", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        U_reg_phase=np.asarray(scores.get("U_reg_phase", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        P_normal=np.asarray(scores.get("P_normal", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        P_uniform=np.asarray(scores.get("P_uniform", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        P_FFLO=np.asarray(scores.get("P_FFLO", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        P_SC=np.asarray(scores.get("P_SC", np.ones(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        U_NS=np.asarray(scores.get("U_NS", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        U_UF=np.asarray(scores.get("U_UF", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        B_delta_raw=np.asarray(scores.get("B_delta_raw", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        B_delta_gated=np.asarray(scores.get("B_delta_gated", scores.get("delta_boundary_score", np.zeros(grid.points.shape[0], dtype=np.float64))), dtype=np.float64),
        delta_boundary_score=np.asarray(scores.get("delta_boundary_score", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        B_q_SC=np.asarray(scores.get("q_boundary_score", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        B_q_raw=np.asarray(scores.get("B_q_raw", scores.get("q_boundary_raw", np.zeros(grid.points.shape[0], dtype=np.float64))), dtype=np.float64),
        B_q_gated=np.asarray(scores.get("B_q_gated", scores.get("q_boundary_score", np.zeros(grid.points.shape[0], dtype=np.float64))), dtype=np.float64),
        gradient_score=np.asarray(scores.get("gradient_score", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        E_q_SC=np.asarray(scores.get("q_edge_risk_score", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        surprise_cleanup_qedge_factor=np.asarray(
            scores.get("surprise_cleanup_qedge_factor", np.ones(grid.points.shape[0], dtype=np.float64)),
            dtype=np.float64,
        ),
        E_ext_uncertain=np.asarray(scores.get("extrapolation_risk_score", np.zeros(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        w_ext_current=np.asarray(scores.get("w_ext_current", np.full(grid.points.shape[0], np.nan, dtype=np.float64)), dtype=np.float64),
        topology_pfaffian_margin_pred=np.asarray(scores.get("topology_pfaffian_margin_pred", np.full(grid.points.shape[0], np.nan, dtype=np.float64)), dtype=np.float64),
        topology_distance_to_trivial=np.asarray(scores.get("topology_distance_to_trivial", np.full(grid.points.shape[0], np.inf, dtype=np.float64)), dtype=np.float64),
        topology_distance_to_topological=np.asarray(scores.get("topology_distance_to_topological", np.full(grid.points.shape[0], np.inf, dtype=np.float64)), dtype=np.float64),
        topology_distance_to_gapless=np.asarray(scores.get("topology_distance_to_gapless", np.full(grid.points.shape[0], np.inf, dtype=np.float64)), dtype=np.float64),
        topology_distance_to_trusted=np.asarray(scores.get("topology_distance_to_trusted", np.full(grid.points.shape[0], np.inf, dtype=np.float64)), dtype=np.float64),
        topology_trivial_count=np.asarray(scores.get("topology_trivial_count", np.zeros(grid.points.shape[0], dtype=np.int64)), dtype=np.int64),
        topology_topological_count=np.asarray(scores.get("topology_topological_count", np.zeros(grid.points.shape[0], dtype=np.int64)), dtype=np.int64),
        topology_gapless_count=np.asarray(scores.get("topology_gapless_count", np.zeros(grid.points.shape[0], dtype=np.int64)), dtype=np.int64),
        topology_trusted_count=np.asarray(scores.get("topology_trusted_count", np.zeros(grid.points.shape[0], dtype=np.int64)), dtype=np.int64),
        interior_penalty=np.asarray(scores.get("interior_penalty", np.ones(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
        interior_penalty_applied=np.asarray(scores.get("interior_penalty_applied", np.zeros(grid.points.shape[0], dtype=np.int8)), dtype=np.int8),
        high_confidence_interior=np.asarray(scores.get("high_confidence_interior", np.zeros(grid.points.shape[0], dtype=np.int8)), dtype=np.int8),
        score=np.asarray(scores["score"], dtype=np.float64),
        active_pool_mask=np.asarray(scores.get("active_pool_mask", np.zeros(grid.points.shape[0], dtype=np.int8)), dtype=np.int8),
        Aselect_initial=np.asarray(scores.get("Aselect_initial", scores["score"]), dtype=np.float64),
        R_obs=np.asarray(scores.get("R_obs", np.ones(grid.points.shape[0], dtype=np.float64)), dtype=np.float64),
    )
    return out


def _attach_selected_prediction_metadata(
    selected_meta: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    if selected_meta.empty or "grid_index" not in selected_meta:
        return selected_meta
    out = selected_meta.copy()
    grid_idx = out["grid_index"].to_numpy(dtype=np.int64)
    phase_pred = np.asarray(predictions["phase_pred"], dtype=np.int64)
    valid = (grid_idx >= 0) & (grid_idx < phase_pred.shape[0])
    predicted_phase = np.full(grid_idx.shape[0], -1, dtype=np.int64)
    predicted_phase[valid] = phase_pred[grid_idx[valid]]
    out["predicted_phase_before_exact"] = predicted_phase
    return out


def _attach_selected_boundary_distance(
    selected_meta: pd.DataFrame,
    grid,
    predictions: Dict[str, np.ndarray],
    cfg: ActiveLearningConfig,
) -> pd.DataFrame:
    if selected_meta.empty or "grid_index" not in selected_meta:
        return selected_meta
    out = selected_meta.copy()
    boundary_points = _predicted_main_boundary_points(grid, predictions["phase_pred"])
    grid_idx = out["grid_index"].to_numpy(dtype=np.int64)
    valid = (grid_idx >= 0) & (grid_idx < grid.points.shape[0])
    dist = np.full(grid_idx.shape[0], np.nan, dtype=np.float64)
    if boundary_points.size and np.any(valid):
        dist[valid] = normalized_min_distance(
            grid.points[grid_idx[valid]],
            boundary_points,
            kt_range=(float(cfg.kt_min), float(cfg.kt_max)),
            ja_range=(float(cfg.ja_min), float(cfg.ja_max)),
        )
    out["selected_to_predicted_boundary_distance"] = dist
    return out


def _infer_boundary_band_mask(payload: Dict[str, np.ndarray], cfg: ActiveLearningConfig) -> np.ndarray:
    n = int(np.asarray(payload["kT"]).shape[0])
    if "delta_boundary_band_normal" in payload:
        return np.asarray(payload["delta_boundary_band_normal"]).astype(bool)
    delta_unresolved = np.asarray(payload.get("delta_unresolved", np.zeros(n, dtype=np.int8))).astype(bool)
    delta_opt = np.asarray(payload.get("delta_opt", np.full(n, np.nan)), dtype=np.float64)
    positive_delta_gap = np.asarray(payload.get("positive_delta_gap", np.full(n, np.nan)), dtype=np.float64)
    return (
        delta_unresolved
        & (delta_opt < float(cfg.delta_eps))
        & np.isfinite(positive_delta_gap)
        & (positive_delta_gap >= 0.0)
        & (positive_delta_gap <= float(cfg.positive_delta_gap_tol))
    )


def _load_boundary_band_points(run_dir: Path, cfg: ActiveLearningConfig) -> np.ndarray:
    points: list[np.ndarray] = []
    for p in sorted(run_dir.glob("iter*/exact_merged_iter*.npz")):
        try:
            with np.load(p, allow_pickle=False) as z:
                payload = {k: z[k].copy() for k in z.files}
        except Exception:
            continue
        if "kT" not in payload or "JA" not in payload:
            continue
        mask = _infer_boundary_band_mask(payload, cfg)
        if np.any(mask):
            points.append(np.stack([payload["kT"][mask], payload["JA"][mask]], axis=1).astype(np.float64))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(points)


def _rounded_key_set(points: np.ndarray, decimals: int) -> set[tuple[float, float]]:
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return set()
    rounded = np.round(points.reshape(-1, 2), decimals=int(decimals))
    return {tuple(x) for x in rounded}


def _rounded_membership_mask(points: np.ndarray, reference_points: np.ndarray, decimals: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    blocked = _rounded_key_set(reference_points, decimals=decimals)
    if not blocked:
        return np.zeros(points.shape[0], dtype=bool)
    rounded = np.round(points.reshape(-1, 2), decimals=int(decimals))
    return np.array([tuple(x) in blocked for x in rounded], dtype=bool)


def _load_recent_selected_points(run_dir: Path, current_iteration: int, cooldown_iters: int) -> np.ndarray:
    if cooldown_iters <= 0:
        return np.empty((0, 2), dtype=np.float64)
    start = max(0, int(current_iteration) - int(cooldown_iters))
    points: list[np.ndarray] = []
    for prev_iter in range(start, int(current_iteration)):
        p = run_dir / f"iter{prev_iter:03d}" / "selected_points.csv"
        if not p.exists():
            continue
        try:
            prev = pd.read_csv(p)[["kT", "JA"]].to_numpy(dtype=np.float64)
        except Exception:
            continue
        if prev.size:
            points.append(prev)
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(points)


def _finite_after_exclusion(raw_score: np.ndarray, mask: np.ndarray) -> int:
    return int(np.sum(np.isfinite(raw_score) & ~mask.astype(bool)))


def _normalized_min_distance_to_reference(
    points: np.ndarray,
    reference_points: np.ndarray,
    kt_range: tuple[float, float],
    ja_range: tuple[float, float],
) -> np.ndarray:
    return normalized_min_distance(points, reference_points, kt_range=kt_range, ja_range=ja_range)


def _extract_iteration_boundaries(
    cfg: ActiveLearningConfig,
    dataset_path: Path,
    iter_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    boundary_dir = iter_dir / "boundaries"
    args = argparse.Namespace(
        dataset=dataset_path,
        output_dir=boundary_dir,
        kt_bin_width=float(cfg.boundary_kt_bin_width),
        max_local_spacing=float(cfg.boundary_max_local_spacing),
        max_refinement_points=max(int(cfg.points_per_iter) * 4, int(cfg.points_per_iter)),
        output_root=Path(cfg.output_root),
    )
    summary = extract_phase_boundaries(args)
    path = boundary_dir / "all_boundary_segments.csv"
    if path.exists():
        boundaries = pd.read_csv(path)
    else:
        boundaries = pd.DataFrame()
    return boundaries, summary


def _validate_boundary_mode(mode: str) -> str:
    mode = str(mode).lower().strip()
    if mode in {"hybrid", "local"}:
        raise ValueError("Midpoint-based selection has been disabled. Use ML-guided acquisition instead.")
    if mode not in {"off", "diagnostic"}:
        raise ValueError(f"Unknown boundary_refinement_mode: {mode}")
    return mode


def _write_boundary_candidate_csv(iter_dir: Path) -> Path:
    out = iter_dir / "candidate_scores_boundary.csv"
    pd.DataFrame(
        {
            "selection_disabled": [1],
            "reason": ["Midpoint-based boundary candidate selection is disabled; see boundaries/all_boundary_segments.csv."],
        }
    ).to_csv(out, index=False)
    return out


def _write_selected_by_pool(iter_dir: Path, selected_rows: list[dict[str, object]]) -> Path:
    out = iter_dir / "selected_points_by_pool.csv"
    if selected_rows:
        pd.DataFrame(selected_rows).to_csv(out, index=False)
    else:
        pd.DataFrame(
            columns=[
                "kT",
                "JA",
                "selection_rank",
                "selection_source",
                "selection_pool",
                "boundary_type",
                "grid_index",
                "A0_main",
                "A0_main_raw",
                "A0_for_pool",
                "A_phase",
                "A_numerical",
                "A_explore",
                "A_response",
                "R_obs",
                "R_batch",
                "Aselect",
                "final_score",
                "selection_score",
                "sampling_probability_before_pick",
                "cls_uncertainty_mix",
                "cls_entropy",
                "cls_margin_uncertainty",
                "P_normal",
                "P_uniform",
                "P_FFLO",
                "P_SC",
                "U_NS",
                "U_UF",
                "U_delta",
                "U_q",
                "U_eta",
                "U_ic_plus",
                "U_ic_minus",
                "U_reg_phase",
                "U_reg_response",
                "B_delta_raw",
                "B_delta_gated",
                "B_delta",
                "B_q_SC",
                "B_q_raw",
                "B_q_gated",
                "gradient_score",
                "G_phase",
                "E_q_SC",
                "E_ext_uncertain",
                "interior_penalty",
                "interior_penalty_applied",
                "high_confidence_interior",
                "sampling_power",
                "w_ext_current",
            ]
        ).to_csv(out, index=False)
    return out


def _select_random_seed_points(
    cfg: ActiveLearningConfig,
    grid_points: np.ndarray,
    candidate_mask: np.ndarray,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    selectable = np.flatnonzero(np.asarray(candidate_mask, dtype=bool))
    n_select = min(int(cfg.initial_seed_size), int(selectable.size))
    rng = np.random.default_rng(int(cfg.random_seed) + int(iteration) * 1000003)
    selected_idx = (
        rng.choice(selectable, size=n_select, replace=False)
        if n_select > 0
        else np.empty((0,), dtype=np.int64)
    )
    selected_points = grid_points[selected_idx].astype(np.float64, copy=False)
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(selected_idx):
        rows.append(
            {
                "selection_rank": int(rank + 1),
                "selection_source": "random_seed",
                "selection_pool": "random_seed",
                "boundary_type": "",
                "grid_index": int(idx),
                "kT": float(grid_points[idx, 0]),
                "JA": float(grid_points[idx, 1]),
                "A0_main": np.nan,
                "A_phase": np.nan,
                "A_numerical": np.nan,
                "A_explore": np.nan,
                "A_response": np.nan,
                "R_obs": 1.0,
                "R_batch": 1.0,
                "final_score": np.nan,
                "selection_score": np.nan,
                "sampling_probability_before_pick": float(1.0 / max(int(selectable.size) - rank, 1)),
            }
        )
    return selected_points, pd.DataFrame(rows)


def _select_sobol_seed_points(
    cfg: ActiveLearningConfig,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if str(cfg.candidate_domain_mode) != "full":
        raise ValueError("sobol_scrambled initialization currently requires candidate_domain_mode='full'.")
    n_select = max(0, int(cfg.initial_seed_size))
    seed = int(cfg.random_seed) + int(iteration) * 1000003
    if n_select == 0:
        selected_points = np.empty((0, 2), dtype=np.float64)
    else:
        engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
        unit = engine.draw(n_select).cpu().numpy().astype(np.float64, copy=False)
        lo = np.array([float(cfg.kt_min), float(cfg.ja_min)], dtype=np.float64)
        hi = np.array([float(cfg.kt_max), float(cfg.ja_max)], dtype=np.float64)
        selected_points = lo + unit * (hi - lo)

    rows: list[dict[str, object]] = []
    for rank, point in enumerate(selected_points):
        rows.append(
            {
                "selection_rank": int(rank + 1),
                "selection_source": "sobol_scrambled_seed",
                "selection_pool": "sobol_scrambled_seed",
                "boundary_type": "",
                "grid_index": -1,
                "kT": float(point[0]),
                "JA": float(point[1]),
                "A0_main": np.nan,
                "A_phase": np.nan,
                "A_numerical": np.nan,
                "A_explore": np.nan,
                "A_response": np.nan,
                "R_obs": 1.0,
                "R_batch": 1.0,
                "final_score": np.nan,
                "selection_score": np.nan,
                "sampling_probability_before_pick": np.nan,
            }
        )
    return selected_points, pd.DataFrame(rows)


def _select_acquisition_points(
    cfg: ActiveLearningConfig,
    grid_points: np.ndarray,
    scores: Dict[str, np.ndarray],
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    selected, rows = select_acquisition_batch(
        points=grid_points,
        scores=scores,
        k=int(cfg.batch_size_max if str(cfg.run_mode) == "discovery" else cfg.points_per_iter),
        cfg=cfg,
        rng_seed=int(cfg.random_seed) + int(iteration) * 1000003,
        iteration=iteration,
    )
    selected_meta = pd.DataFrame(rows)
    summary = {
        "mode": _validate_boundary_mode(cfg.boundary_refinement_mode),
        "selection_policy": "acquisition_only",
        "selection_mode": str(cfg.selection_mode),
        "midpoint_selection": "disabled",
        "selected_by_pool": (
            {str(k): int(v) for k, v in selected_meta["selection_pool"].value_counts().sort_index().items()}
            if not selected_meta.empty
            else {}
        ),
        "selected_by_boundary_type": {},
    }
    return selected, selected_meta, summary


def _write_boundary_displacement(
    run_dir: Path,
    iter_dir: Path,
    iteration: int,
    current_boundaries: pd.DataFrame,
    cfg: ActiveLearningConfig,
) -> Path | None:
    if iteration <= 0 or current_boundaries.empty:
        return None
    prev_path = run_dir / f"iter{iteration - 1:03d}" / "boundaries" / "all_boundary_segments.csv"
    if not prev_path.exists():
        return None
    previous = pd.read_csv(prev_path)
    if previous.empty:
        return None

    rows: dict[str, dict[str, object]] = {}
    for boundary_type, cur_group in current_boundaries.groupby("boundary_type"):
        prev_group = previous[previous["boundary_type"] == boundary_type].copy()
        if prev_group.empty:
            continue
        distances: list[float] = []
        for _, cur in cur_group.iterrows():
            dk = np.abs(prev_group["kT_boundary"].to_numpy(dtype=np.float64) - float(cur["kT_boundary"]))
            j = int(np.argmin(dk))
            prev = prev_group.iloc[j]
            dkt = (float(cur["kT_boundary"]) - float(prev["kT_boundary"])) / max(float(cfg.kt_max - cfg.kt_min), 1e-12)
            dja = (float(cur["JA_boundary"]) - float(prev["JA_boundary"])) / max(float(cfg.ja_max - cfg.ja_min), 1e-12)
            distances.append(float(np.sqrt(dkt * dkt + dja * dja)))
        arr = np.asarray(distances, dtype=np.float64)
        rows[str(boundary_type)] = {
            "matched_segments": int(arr.shape[0]),
            "median_displacement_normalized": float(np.median(arr)) if arr.size else None,
            "max_displacement_normalized": float(np.max(arr)) if arr.size else None,
            "within_boundary_position_tol": bool(arr.size and np.median(arr) < float(cfg.boundary_position_tol)),
            "current_confidence_counts": {
                str(k): int(v) for k, v in cur_group["confidence"].value_counts().sort_index().items()
            }
            if "confidence" in cur_group
            else {},
        }

    out = iter_dir / f"boundary_displacement_iter{iteration:03d}.json"
    out.write_text(
        json.dumps(
            {
                "iteration": int(iteration),
                "boundary_position_tol": float(cfg.boundary_position_tol),
                "stable_stage_target": int(cfg.boundary_stable_stages),
                "by_boundary_type": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _apply_candidate_exclusions(
    cfg: ActiveLearningConfig,
    grid_points: np.ndarray,
    scores: Dict[str, np.ndarray],
    existing_points: np.ndarray,
    boundary_band_points: np.ndarray,
    run_dir: Path,
    current_iteration: int,
) -> Dict[str, np.ndarray]:
    updated = {k: np.asarray(v).copy() for k, v in scores.items()}
    n = int(grid_points.shape[0])
    if "score" not in updated:
        return updated

    raw_score = np.asarray(updated["score"], dtype=np.float64).copy()
    updated["score_raw"] = raw_score.copy()

    existing_mask = np.zeros(n, dtype=bool)
    existing_distance_mask = np.zeros(n, dtype=bool)
    existing_min_distance = np.full(n, np.inf, dtype=np.float64)
    if bool(cfg.exclude_existing_exact):
        existing_mask = _rounded_membership_mask(
            grid_points,
            existing_points,
            decimals=int(cfg.existing_exclusion_decimals),
        )
        if float(cfg.existing_min_dist) > 0.0:
            existing_min_distance = _normalized_min_distance_to_reference(
                grid_points,
                existing_points,
                kt_range=(float(cfg.kt_min), float(cfg.kt_max)),
                ja_range=(float(cfg.ja_min), float(cfg.ja_max)),
            )
            existing_distance_mask = existing_min_distance < float(cfg.existing_min_dist)

    recent_points = _load_recent_selected_points(
        run_dir,
        current_iteration=current_iteration,
        cooldown_iters=int(cfg.recent_selection_cooldown_iters),
    )
    recent_mask = _rounded_membership_mask(
        grid_points,
        recent_points,
        decimals=int(cfg.recent_selection_cooldown_decimals),
    )

    boundary_mask = np.zeros(n, dtype=bool)
    if bool(cfg.boundary_band_cooldown_enabled):
        boundary_mask = _rounded_membership_mask(
            grid_points,
            boundary_band_points,
            decimals=int(cfg.boundary_band_cooldown_decimals),
        )
    if str(cfg.run_mode) == "discovery":
        # Discovery mode should be governed by A0_main active-pool gating plus
        # hard exact-coordinate duplicate exclusion. Recent-selection and
        # boundary-band cooldowns are refinement-era safeguards, so keep them
        # out of the online discovery candidate mask.
        recent_mask = np.zeros(n, dtype=bool)
        boundary_mask = np.zeros(n, dtype=bool)

    r_obs = observation_repulsion(
        existing_min_distance,
        ell=float(cfg.observation_repulsion_length),
        floor=float(cfg.observation_repulsion_floor),
    )
    # Existing coordinates are still forbidden. Nearby already-computed points
    # now apply a soft observation-repulsion factor rather than a hard radius
    # cutoff, so ML-guided boundary candidates are not blocked by a rigid r.
    existing_combined_mask = existing_mask
    exclude_mask = existing_combined_mask | recent_mask | boundary_mask
    recent_relaxed = False
    boundary_relaxed = False
    target_batch_size = int(cfg.batch_size_max if str(cfg.run_mode) == "discovery" else cfg.points_per_iter)
    if _finite_after_exclusion(raw_score, exclude_mask) < target_batch_size and np.any(recent_mask):
        recent_mask = np.zeros(n, dtype=bool)
        recent_relaxed = True
        exclude_mask = existing_combined_mask | boundary_mask
    if _finite_after_exclusion(raw_score, exclude_mask) < target_batch_size and np.any(boundary_mask):
        boundary_mask = np.zeros(n, dtype=bool)
        boundary_relaxed = True
        exclude_mask = existing_combined_mask

    updated["score"] = np.where(exclude_mask, -np.inf, raw_score * r_obs)
    updated["existing_exact_exclusion"] = existing_mask.astype(np.int8)
    updated["existing_distance_exclusion"] = existing_distance_mask.astype(np.int8)
    updated["existing_min_distance"] = existing_min_distance
    updated["R_obs"] = r_obs
    updated["recent_selected_exclusion"] = recent_mask.astype(np.int8)
    updated["boundary_band_cooldown"] = boundary_mask.astype(np.int8)
    updated["candidate_excluded"] = exclude_mask.astype(np.int8)
    updated["recent_selection_exclusion_relaxed"] = np.full(n, int(recent_relaxed), dtype=np.int8)
    updated["boundary_band_exclusion_relaxed"] = np.full(n, int(boundary_relaxed), dtype=np.int8)
    return updated


def _previous_selected_key_set(run_dir: Path, iter_dir: Path, decimals: int) -> set[tuple[float, float]]:
    previous_keys: set[tuple[float, float]] = set()
    for p in run_dir.glob("iter*/selected_points.csv"):
        if p.parent == iter_dir:
            continue
        try:
            prev = pd.read_csv(p)[["kT", "JA"]].to_numpy(dtype=np.float64)
        except Exception:
            continue
        previous_keys.update(_rounded_key_set(prev, decimals=decimals))
    return previous_keys


def _candidate_pool_summary(scores: Dict[str, np.ndarray]) -> dict:
    if "score" not in scores:
        return {}
    score = np.asarray(scores["score"], dtype=np.float64)
    raw_score = np.asarray(scores.get("score_raw", score), dtype=np.float64)
    existing = np.asarray(scores.get("existing_exact_exclusion", np.zeros_like(score)), dtype=bool)
    existing_distance = np.asarray(scores.get("existing_distance_exclusion", np.zeros_like(score)), dtype=bool)
    recent = np.asarray(scores.get("recent_selected_exclusion", np.zeros_like(score)), dtype=bool)
    boundary = np.asarray(scores.get("boundary_band_cooldown", np.zeros_like(score)), dtype=bool)
    excluded = np.asarray(scores.get("candidate_excluded", np.zeros_like(score)), dtype=bool)
    high_conf = np.asarray(scores.get("high_confidence_interior", np.zeros_like(score)), dtype=bool)
    penalty_applied = np.asarray(scores.get("interior_penalty_applied", np.zeros_like(score)), dtype=bool)
    active_pool = np.asarray(scores.get("active_pool_mask", np.zeros_like(score)), dtype=bool)
    return {
        "candidate_pool_total": int(score.shape[0]),
        "candidate_pool_finite_before_exclusion": int(np.sum(np.isfinite(raw_score))),
        "candidate_pool_finite_after_exclusion": int(np.sum(np.isfinite(score))),
        "active_pool_size": int(np.sum(active_pool)),
        "high_confidence_interior_count": int(np.sum(high_conf)),
        "high_confidence_interior_fraction_unexcluded": float(np.sum(high_conf & ~excluded) / max(int(np.sum(~excluded)), 1)),
        "active_pool_high_confidence_interior_count": int(np.sum(active_pool & high_conf)),
        "active_pool_interior_penalty_applied_count": int(np.sum(penalty_applied & active_pool)),
        "excluded_existing_exact_candidates": int(np.sum(existing)),
        "soft_penalized_existing_distance_candidates": int(np.sum(existing_distance)),
        "excluded_recent_selected_candidates": int(np.sum(recent)),
        "excluded_boundary_band_candidates": int(np.sum(boundary)),
        "excluded_any_candidates": int(np.sum(excluded)),
        "recent_selection_exclusion_relaxed": bool(np.any(scores.get("recent_selection_exclusion_relaxed", False))),
        "boundary_band_exclusion_relaxed": bool(np.any(scores.get("boundary_band_exclusion_relaxed", False))),
        **{str(k): v for k, v in scores.get("_selection_summary", {}).items()},
    }


def _finite_quantiles(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    out: dict[str, float | None] = {}
    for p in (50, 75, 90, 95, 99):
        out[f"{prefix}_p{p}"] = float(np.percentile(vals, p)) if vals.size else None
    return out


def _predicted_main_boundary_points(grid, phase_pred: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_pred, dtype=np.int64).reshape(grid.full_shape)
    kt = np.asarray(grid.kt_values, dtype=np.float64)
    ja = np.asarray(grid.ja_values, dtype=np.float64)
    pts: list[tuple[float, float]] = []
    nja, nkt = phase.shape
    for j in range(nja):
        for i in range(nkt - 1):
            a = int(phase[j, i])
            b = int(phase[j, i + 1])
            if {a, b} in ({0, 1}, {0, 2}, {1, 2}):
                pts.append((0.5 * (kt[i] + kt[i + 1]), float(ja[j])))
    for j in range(nja - 1):
        for i in range(nkt):
            a = int(phase[j, i])
            b = int(phase[j + 1, i])
            if {a, b} in ({0, 1}, {0, 2}, {1, 2}):
                pts.append((float(kt[i]), 0.5 * (ja[j] + ja[j + 1])))
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _distance_stats(prefix: str, dist: np.ndarray) -> dict[str, float | None]:
    d = np.asarray(dist, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {f"{prefix}_{name}": None for name in ("mean", "median", "p75", "p90", "p95")}
    return {
        f"{prefix}_mean": float(np.mean(d)),
        f"{prefix}_median": float(np.median(d)),
        f"{prefix}_p75": float(np.percentile(d, 75)),
        f"{prefix}_p90": float(np.percentile(d, 90)),
        f"{prefix}_p95": float(np.percentile(d, 95)),
    }


def _boundary_focus_diagnostics(
    selected_points: np.ndarray,
    grid,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    cfg: ActiveLearningConfig,
    iteration: int,
) -> dict:
    selected_points = np.asarray(selected_points, dtype=np.float64).reshape(-1, 2)
    boundary_points = _predicted_main_boundary_points(grid, predictions["phase_pred"])
    spacing = max(
        1.0 / max(int(cfg.n_kt_candidates) - 1, 1),
        1.0 / max(int(cfg.n_ja_candidates) - 1, 1),
    )
    band_width = 2.0 * spacing
    out: dict[str, object] = {
        "predicted_main_boundary_points": int(boundary_points.shape[0]),
        "boundary_band_width_norm": float(band_width),
    }
    if selected_points.size == 0 or boundary_points.size == 0:
        out.update(_distance_stats("selected_to_predicted_boundary_distance", np.empty((0,), dtype=np.float64)))
        out["selected_boundary_band_fraction"] = None
    else:
        d_sel = normalized_min_distance(
            selected_points,
            boundary_points,
            kt_range=(float(cfg.kt_min), float(cfg.kt_max)),
            ja_range=(float(cfg.ja_min), float(cfg.ja_max)),
        )
        out.update(_distance_stats("selected_to_predicted_boundary_distance", d_sel))
        out["selected_boundary_band_fraction"] = float(np.mean(d_sel <= band_width))

    active_pool = np.asarray(scores.get("active_pool_mask", np.zeros(grid.points.shape[0], dtype=np.int8)), dtype=bool)
    available_idx = np.flatnonzero(active_pool)
    if selected_points.shape[0] and available_idx.size and boundary_points.size:
        rng = np.random.default_rng(int(cfg.random_seed) + int(iteration) * 1000003 + 777)
        size = min(int(selected_points.shape[0]), int(available_idx.size))
        sample_idx = rng.choice(available_idx, size=size, replace=False)
        random_points = grid.points[sample_idx]
        d_rand = normalized_min_distance(
            random_points,
            boundary_points,
            kt_range=(float(cfg.kt_min), float(cfg.kt_max)),
            ja_range=(float(cfg.ja_min), float(cfg.ja_max)),
        )
        out.update(_distance_stats("random_baseline_boundary_distance", d_rand))
        out["random_baseline_boundary_band_fraction"] = float(np.mean(d_rand <= band_width))
    else:
        out.update(_distance_stats("random_baseline_boundary_distance", np.empty((0,), dtype=np.float64)))
        out["random_baseline_boundary_band_fraction"] = None

    a0 = np.asarray(scores.get("A0_main", scores["score"]), dtype=np.float64)
    a0_pool = np.asarray(scores.get("A0_for_pool", a0), dtype=np.float64)
    unseen = np.isfinite(a0) & ~np.asarray(scores.get("candidate_excluded", np.zeros_like(a0)), dtype=bool)
    out.update(_finite_quantiles("unseen_A0_main", a0[unseen]))
    out.update(_finite_quantiles("unseen_A0_for_pool", a0_pool[unseen]))
    selected_idx = []
    if selected_points.size:
        # Reconstruct selected indices by rounded coordinates; selected metadata has
        # exact grid indices, but diagnostics should also work if only points exist.
        keys = {tuple(row) for row in np.round(selected_points, decimals=12)}
        for i, pt in enumerate(np.round(grid.points, decimals=12)):
            if tuple(pt) in keys:
                selected_idx.append(i)
    idx_arr = np.asarray(selected_idx, dtype=np.int64)
    if idx_arr.size:
        out.update(_finite_quantiles("selected_A0_main", a0[idx_arr]))
        out.update(_finite_quantiles("selected_A0_for_pool", a0_pool[idx_arr]))
        mean_unseen = float(np.nanmean(a0[unseen])) if np.any(unseen) else None
        mean_selected = float(np.nanmean(a0[idx_arr])) if idx_arr.size else None
        mean_unseen_pool = float(np.nanmean(a0_pool[unseen])) if np.any(unseen) else None
        mean_selected_pool = float(np.nanmean(a0_pool[idx_arr])) if idx_arr.size else None
        out["selected_A0_main_mean"] = mean_selected
        out["selected_A0_for_pool_mean"] = mean_selected_pool
        out["unseen_A0_main_mean"] = mean_unseen
        out["unseen_A0_for_pool_mean"] = mean_unseen_pool
        out["selected_A0_main_over_unseen_mean"] = (
            float(mean_selected / mean_unseen) if mean_unseen not in (None, 0.0) and mean_selected is not None else None
        )
        out["selected_A0_for_pool_over_unseen_mean"] = (
            float(mean_selected_pool / mean_unseen_pool)
            if mean_unseen_pool not in (None, 0.0) and mean_selected_pool is not None
            else None
        )
        for key in ("A_phase", "A_numerical", "A_explore"):
            vals = np.asarray(scores.get(key, np.full_like(a0, np.nan)), dtype=np.float64)
            out[f"selected_{key}_mean"] = float(np.nanmean(vals[idx_arr])) if idx_arr.size else None
        r_obs = np.asarray(scores.get("R_obs", np.ones_like(a0)), dtype=np.float64)
        out["selected_R_obs_mean"] = float(np.nanmean(r_obs[idx_arr]))
        out["selected_R_obs_min"] = float(np.nanmin(r_obs[idx_arr]))
        out["selected_R_obs_p10"] = float(np.nanpercentile(r_obs[idx_arr], 10))
    else:
        out.update(_finite_quantiles("selected_A0_main", np.empty((0,), dtype=np.float64)))
        out.update(_finite_quantiles("selected_A0_for_pool", np.empty((0,), dtype=np.float64)))
        out["selected_A0_main_mean"] = None
        out["selected_A0_for_pool_mean"] = None
        out["unseen_A0_main_mean"] = float(np.nanmean(a0[unseen])) if np.any(unseen) else None
        out["unseen_A0_for_pool_mean"] = float(np.nanmean(a0_pool[unseen])) if np.any(unseen) else None
        out["selected_A0_main_over_unseen_mean"] = None
        out["selected_A0_for_pool_over_unseen_mean"] = None
    r_obs_all = np.asarray(scores.get("R_obs", np.ones_like(a0)), dtype=np.float64)
    out["active_pool_R_obs_mean"] = float(np.nanmean(r_obs_all[active_pool])) if np.any(active_pool) else None
    return out


def _selection_distance_summary(selected_points: np.ndarray, existing_points: np.ndarray) -> dict:
    if selected_points.size == 0 or existing_points.size == 0:
        return {
            "min_distance_to_dataset_min": None,
            "min_distance_to_dataset_median": None,
            "min_distance_to_dataset_mean": None,
        }

    # Physical-coordinate distance is kept for continuity with earlier reports.
    diff = selected_points[:, None, :] - existing_points[None, :, :]
    min_dist = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    finite = min_dist[np.isfinite(min_dist)]
    return {
        "min_distance_to_dataset_min": float(finite.min()) if finite.size else None,
        "min_distance_to_dataset_median": float(np.median(finite)) if finite.size else None,
        "min_distance_to_dataset_mean": float(finite.mean()) if finite.size else None,
    }


def _selection_normalized_distance_summary(
    selected_points: np.ndarray,
    existing_points: np.ndarray,
    cfg: ActiveLearningConfig,
) -> dict:
    min_dist = _normalized_min_distance_to_reference(
        selected_points,
        existing_points,
        kt_range=(float(cfg.kt_min), float(cfg.kt_max)),
        ja_range=(float(cfg.ja_min), float(cfg.ja_max)),
    )
    finite = min_dist[np.isfinite(min_dist)]
    return {
        "selected_to_existing_normalized_min": float(finite.min()) if finite.size else None,
        "selected_to_existing_normalized_median": float(np.median(finite)) if finite.size else None,
        "selected_to_existing_normalized_mean": float(finite.mean()) if finite.size else None,
        "selected_within_existing_min_dist": int(np.sum(finite < float(cfg.existing_min_dist))) if finite.size else 0,
    }


def _write_selection_diagnostics(
    run_dir: Path,
    iter_dir: Path,
    selected_points: np.ndarray,
    existing_points: np.ndarray,
    scores: Dict[str, np.ndarray],
    cfg: ActiveLearningConfig,
    grid=None,
    predictions: Dict[str, np.ndarray] | None = None,
    iteration: int | None = None,
    boundary_selection_summary: dict | None = None,
) -> Path:
    decimals = int(cfg.existing_exclusion_decimals)
    selected_points = np.asarray(selected_points, dtype=np.float64)
    existing_points = np.asarray(existing_points, dtype=np.float64)
    summary = _candidate_pool_summary(scores)
    summary.update(
        {
            "existing_exclusion_decimals": int(cfg.existing_exclusion_decimals),
            "existing_min_dist": float(cfg.existing_min_dist),
            "recent_selection_cooldown_iters": int(cfg.recent_selection_cooldown_iters),
            "recent_selection_cooldown_decimals": int(cfg.recent_selection_cooldown_decimals),
            "boundary_band_cooldown_decimals": int(cfg.boundary_band_cooldown_decimals),
            "run_mode": str(cfg.run_mode),
            "candidate_domain_mode": str(cfg.candidate_domain_mode),
            "boundary_refinement_mode": str(cfg.boundary_refinement_mode),
            "selection_policy": "acquisition_only",
            "selection_mode": str(cfg.selection_mode),
            "midpoint_selection": "disabled",
            "observation_repulsion_length": float(cfg.observation_repulsion_length),
            "observation_repulsion_floor": float(cfg.observation_repulsion_floor),
            "batch_repulsion_length": float(cfg.batch_repulsion_length),
            "batch_repulsion_floor": float(cfg.batch_repulsion_floor),
            "boundary_position_tol": float(cfg.boundary_position_tol),
        }
    )
    if boundary_selection_summary:
        summary["boundary_selection"] = boundary_selection_summary
    if grid is not None and predictions is not None and iteration is not None:
        summary.update(_boundary_focus_diagnostics(selected_points, grid, predictions, scores, cfg, iteration))
    if selected_points.size == 0:
        summary.update(
            {
                "selected_points": 0,
                "unique_selected_rounded": 0,
                "already_in_dataset_rounded": 0,
                "previously_selected_rounded": 0,
                "min_distance_to_dataset_min": None,
                "min_distance_to_dataset_median": None,
                "min_distance_to_dataset_mean": None,
                "selected_to_existing_normalized_min": None,
                "selected_to_existing_normalized_median": None,
                "selected_to_existing_normalized_mean": None,
                "selected_within_existing_min_dist": 0,
            }
        )
    else:
        selected_keys = _rounded_key_set(selected_points, decimals=decimals)
        existing_keys = _rounded_key_set(existing_points, decimals=decimals)
        previous_keys = _previous_selected_key_set(run_dir, iter_dir, decimals=decimals)
        summary.update(
            {
                "selected_points": int(selected_points.shape[0]),
                "unique_selected_rounded": int(len(selected_keys)),
                "already_in_dataset_rounded": int(len(selected_keys & existing_keys)),
                "previously_selected_rounded": int(len(selected_keys & previous_keys)),
            }
        )
        summary.update(_selection_distance_summary(selected_points, existing_points))
        summary.update(_selection_normalized_distance_summary(selected_points, existing_points, cfg))

    out = iter_dir / "selection_diagnostics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _write_random_seed_diagnostics(
    iter_dir: Path,
    cfg: ActiveLearningConfig,
    candidate_mask: np.ndarray,
    selected_points: np.ndarray,
) -> Path:
    out = iter_dir / "selection_diagnostics.json"
    summary = {
        "run_mode": str(cfg.run_mode),
        "candidate_domain_mode": str(cfg.candidate_domain_mode),
        "selection_policy": "initial_seed",
        "selection_mode": str(cfg.initialization),
        "midpoint_selection": "disabled",
        "finite_t_band_width": None,
        "candidate_pool_total": int(candidate_mask.size),
        "candidate_pool_available": int(np.sum(np.asarray(candidate_mask, dtype=bool))),
        "initial_seed_size": int(cfg.initial_seed_size),
        "selected_points": int(selected_points.shape[0]),
        "selected_by_pool": {"random_seed": int(selected_points.shape[0])},
        "uses_warm_start_training_data": False,
        "uses_prior_finite_t_candidate_mask": False,
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _write_hpc_selection_artifacts(
    args: argparse.Namespace,
    run_dir: Path,
    iter_dir: Path,
    iteration: int,
    selected_points: np.ndarray,
) -> None:
    shard_paths = write_point_shards(
        run_dir=run_dir,
        iteration=iteration,
        points=selected_points,
        world_size=max(1, args.world_size),
        strategy=args.partition_strategy,
    )
    (iter_dir / "hpc_instructions.txt").write_text(
        "\n".join(
            [
                f"world_size={args.world_size}",
                f"partition_strategy={args.partition_strategy}",
                f"slurm_script={args.slurm_script}",
                f"selected_points={len(selected_points)}",
                "Run SLURM array for exact oracle shards, then merge with ml_phase.hpc --merge.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"HPC mode: wrote {len(shard_paths)} point-shard files.")
    print(f"Iteration artifacts: {iter_dir}")
    if args.submit:
        _maybe_submit_slurm(args.slurm_script, args.run_id, iteration, args.oracle_mode)


def _evaluate_on_validation(bundle: ModelBundle, dataset: FlatDataset, n_exact_calls: int, cfg: ActiveLearningConfig) -> dict:
    preds = predict_models(bundle, dataset.x)
    metrics = evaluate_predictions(
        x=dataset.x,
        y_reg_true=dataset.y_reg,
        y_phase_true=dataset.y_phase,
        y_reg_pred=preds["reg_mean"],
        y_phase_pred=preds["phase_pred"],
        n_exact_calls=n_exact_calls,
        dense_grid_points=cfg.n_kt_candidates * cfg.n_ja_candidates,
    )
    return metrics.to_dict()


def _maybe_submit_slurm(script: Path, run_id: str, iteration: int, oracle_mode: str) -> None:
    cmd = ["sbatch", str(script)]
    env = os.environ.copy()
    env["RUN_ID"] = run_id
    env["ITER"] = str(iteration)
    env["ORACLE_MODE"] = str(oracle_mode)
    print("Submitting SLURM array job:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def _jsonable_args(ns: argparse.Namespace) -> dict:
    out: dict = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def run_active_refinement(args: argparse.Namespace) -> None:
    boundary_mode = _validate_boundary_mode(args.boundary_refinement_mode)
    cfg = ActiveLearningConfig(
        run_mode=args.run_mode,
        candidate_domain_mode=args.candidate_domain_mode,
        initialization=args.initialization,
        initial_seed_size=args.initial_seed_size,
        batch_size_max=args.batch_size_max,
        batch_size_min=args.batch_size_min,
        batch_size_min_before_min_iter=args.batch_size_min_before_min_iter,
        batch_size_min_after_min_iter=args.batch_size_min_after_min_iter,
        selection_mode=args.selection_mode,
        sampling_power=args.sampling_power,
        sampling_power_start=args.sampling_power_start,
        sampling_power_mid=args.sampling_power_mid,
        sampling_power_end=args.sampling_power_end,
        sampling_power_mid_iter=args.sampling_power_mid_iter,
        sampling_power_end_iter=args.sampling_power_end_iter,
        sampling_power_schedule=args.sampling_power_schedule,
        score_threshold_abs=args.score_threshold_abs,
        score_threshold_rel=args.score_threshold_rel,
        acquisition_profile=args.acquisition_profile,
        active_pool_rule=args.active_pool_rule,
        active_pool_quantile=args.active_pool_quantile,
        active_pool_quantile_schedule=args.active_pool_quantile_schedule,
        active_pool_quantile_start=args.active_pool_quantile_start,
        active_pool_quantile_mid=args.active_pool_quantile_mid,
        active_pool_quantile_end=args.active_pool_quantile_end,
        active_pool_quantile_mid_iter=args.active_pool_quantile_mid_iter,
        active_pool_quantile_end_iter=args.active_pool_quantile_end_iter,
        active_pool_rel_to_p95=args.active_pool_rel_to_p95,
        active_pool_min_quantile=args.active_pool_min_quantile,
        active_pool_max_fraction_start=args.active_pool_max_fraction_start,
        active_pool_max_fraction_end=args.active_pool_max_fraction_end,
        active_pool_max_fraction_end_iter=args.active_pool_max_fraction_end_iter,
        active_selection_min_iterations=args.active_selection_min_iterations,
        allow_underfilled_batch_after_min_iter=not bool(args.no_underfilled_batch_after_min_iter),
        b_delta_gate_mode=args.b_delta_gate_mode,
        q_boundary_gate_mode=args.q_boundary_gate_mode,
        interior_filter_mode=args.interior_filter_mode,
        interior_penalty_start_iter=args.interior_penalty_start_iter,
        interior_penalty_early=args.interior_penalty_early,
        interior_penalty_late=args.interior_penalty_late,
        p_conf_threshold=args.p_conf_threshold,
        u_ns_low=args.u_ns_low,
        u_uf_low=args.u_uf_low,
        g_phase_low=args.g_phase_low,
        e_q_low=args.e_q_low,
        e_ext_low=args.e_ext_low,
        w_ext_schedule=args.w_ext_schedule,
        w_ext_start=args.w_ext_start,
        w_ext_mid=args.w_ext_mid,
        w_ext_end=args.w_ext_end,
        w_ext_mid_iter=args.w_ext_mid_iter,
        w_ext_end_iter=args.w_ext_end_iter,
        w_cls_simple=args.w_cls_simple,
        w_ns_simple=args.w_ns_simple,
        w_uf_simple=args.w_uf_simple,
        w_grad_simple=args.w_grad_simple,
        w_reg_simple=args.w_reg_simple,
        w_ext_simple_schedule=args.w_ext_simple_schedule,
        w_ext_simple_start=args.w_ext_simple_start,
        w_ext_simple_mid=args.w_ext_simple_mid,
        w_ext_simple_end=args.w_ext_simple_end,
        w_ext_simple_mid_iter=args.w_ext_simple_mid_iter,
        w_ext_simple_end_iter=args.w_ext_simple_end_iter,
        surprise_cleanup_qedge_penalty=args.surprise_cleanup_qedge_penalty,
        surprise_cleanup_qedge_floor=args.surprise_cleanup_qedge_floor,
        surprise_cleanup_response_weight=args.surprise_cleanup_response_weight,
        surprise_cleanup_explore_scale=args.surprise_cleanup_explore_scale,
        random_seed=args.random_seed,
        finite_t_band_width=args.finite_t_band_width,
        hidden_ground_truth=str(args.hidden_ground_truth or ""),
        iterations=args.iterations,
        points_per_iter=args.points_per_iter,
        dry_run=bool(args.dry_run),
        mode=args.mode,
        oracle_mode=args.oracle_mode,
        world_size=args.world_size,
        partition_strategy=args.partition_strategy,
        enable_early_stop=not bool(args.disable_early_stop),
        min_new_points_per_iter=args.min_new_points_per_iter,
        max_low_append_iters=args.max_low_append_iters,
        boundary_refinement_mode=boundary_mode,
        boundary_kt_bin_width=args.boundary_kt_bin_width,
        boundary_max_local_spacing=args.boundary_max_local_spacing,
        boundary_position_tol=args.boundary_position_tol,
        boundary_stable_stages=args.boundary_stable_stages,
        output_root=str(args.output_root),
        n_ensemble=args.n_ensemble,
        reg_epochs=args.reg_epochs,
        cls_epochs=args.cls_epochs,
        batch_size=args.batch_size,
    )
    cfg = validate_active_learning_config(cfg)
    ensure_output_dirs(cfg)

    if str(cfg.run_mode) == "discovery" and args.start_iteration == 0 and args.resume_dataset is None:
        flat = _empty_flat_dataset()
        warm_npz = None
        warm_csv = None
        print("Discovery mode: starting from an empty training dataset.")
    elif args.resume_dataset is not None:
        flat = load_flat_dataset(args.resume_dataset)
        warm_npz = args.resume_dataset
        warm_csv = args.resume_dataset.with_suffix(".csv")
        print(f"Resume dataset: {args.resume_dataset}")
        print(f"Resume samples: {flat.x.shape[0]}")
    else:
        if args.warm_start is None:
            raise ValueError("refinement mode requires --warm-start unless --resume-dataset is provided.")
        # Build warm-start flat dataset under ML_Phase/datasets.
        flat, warm_npz, warm_csv, _ = build_warm_start_dataset(args.warm_start, cfg, output_root=args.output_root)
        print(f"Warm-start dataset: {warm_npz}")
        print(f"Warm-start csv: {warm_csv}")

    run_dir = cfg.active_runs_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": _jsonable_args(args),
                "active_learning_config": cfg.to_dict(),
                "python": sys.version,
                "torch": torch.__version__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dataset = flat
    metrics_path = run_dir / "metrics_history.json"
    if metrics_path.exists():
        metrics_history = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics_history: list[dict[str, float]] = []
    n_exact_calls = int(dataset.x.shape[0])
    low_append_iters = 0

    for iteration in range(int(args.start_iteration), int(args.start_iteration) + cfg.iterations):
        iter_dir = run_dir / f"iter{iteration:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        current_dataset_npz, _ = _save_dataset(iter_dir, iteration, dataset)

        is_discovery_seed = (
            str(cfg.run_mode) == "discovery"
            and int(iteration) == 0
            and int(dataset.x.shape[0]) == 0
        )
        if is_discovery_seed:
            grid = build_candidate_grid(cfg)
            if str(cfg.initialization) == "sobol_scrambled":
                selected_points, selected_meta = _select_sobol_seed_points(cfg=cfg, iteration=iteration)
            else:
                selected_points, selected_meta = _select_random_seed_points(
                    cfg=cfg,
                    grid_points=grid.points,
                    candidate_mask=grid.candidate_mask,
                    iteration=iteration,
                )
            pd.DataFrame(selected_points, columns=["kT", "JA"]).to_csv(iter_dir / "selected_points.csv", index=False)
            _write_selected_by_pool(iter_dir, selected_meta.to_dict("records") if not selected_meta.empty else [])
            _write_boundary_candidate_csv(iter_dir)
            _write_random_seed_diagnostics(iter_dir, cfg, grid.candidate_mask, selected_points)

            if cfg.enable_early_stop and selected_points.shape[0] == 0:
                print(f"Early stop at iteration {iteration}: random seed selected no candidate points.")
                _save_dataset(iter_dir, iteration, dataset)
                break

            if args.mode == "hpc":
                _write_hpc_selection_artifacts(args, run_dir, iter_dir, iteration, selected_points)
                _save_dataset(iter_dir, iteration, dataset)
                break

            if args.dry_run:
                print(f"Dry-run discovery seed iteration {iteration}: selected {len(selected_points)} initial-seed points.")
                _save_dataset(iter_dir, iteration, dataset)
                continue

            input_samples = int(dataset.x.shape[0])
            oracle = evaluate_points(
                points=selected_points,
                device=args.device,
                output_file=iter_dir / "exact_local_partial.npz",
                save_every=1,
                enable_q_expansion=True,
                enable_delta_refinement=True,
                oracle_mode=str(args.oracle_mode),
                branch_dir=iter_dir / "branch_candidates",
            )
            result = oracle.to_dict()
            np.savez(iter_dir / "exact_local_iter.npz", **result)
            trusted_mask = np.asarray(result.get("trusted_exact", np.ones(selected_points.shape[0], dtype=np.int8))).astype(bool)
            if np.any(trusted_mask):
                trusted_result = {
                    k: (np.asarray(v)[trusted_mask] if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == trusted_mask.shape[0] else v)
                    for k, v in result.items()
                }
                dataset = _dataset_from_result(dataset, trusted_result, cfg)
            new_unique_samples = max(0, int(dataset.x.shape[0]) - input_samples)
            (iter_dir / "local_append_summary.json").write_text(
                json.dumps(
                    {
                        "iteration": int(iteration),
                        "input_samples": input_samples,
                        "output_samples": int(dataset.x.shape[0]),
                        "trusted_points_appended": int(np.sum(trusted_mask)),
                        "new_unique_samples_added": new_unique_samples,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            n_exact_calls += int(np.sum(trusted_mask))
            _save_dataset(iter_dir, iteration + 1, dataset)
            continue

        if int(dataset.x.shape[0]) == 0:
            print(f"Early stop at iteration {iteration}: no training samples available.")
            _save_dataset(iter_dir, iteration, dataset)
            break

        bundle = train_models(dataset.x, dataset.y_reg, dataset.y_phase, cfg)
        grid = build_candidate_grid(cfg)
        pred_grid = predict_models(bundle, grid.points)
        topology_context = _topology_context_from_dataset(dataset)
        score_pack = compute_acquisition_scores(
            cfg,
            grid,
            pred_grid,
            existing_points=dataset.x,
            iteration=iteration,
            topology_context=topology_context,
        )
        boundary_band_points = _load_boundary_band_points(run_dir, cfg)
        score_pack = _apply_candidate_exclusions(
            cfg=cfg,
            grid_points=grid.points,
            scores=score_pack,
            existing_points=dataset.x,
            boundary_band_points=boundary_band_points,
            run_dir=run_dir,
            current_iteration=iteration,
        )

        boundaries = pd.DataFrame()
        if _validate_boundary_mode(cfg.boundary_refinement_mode) == "diagnostic":
            boundaries, _ = _extract_iteration_boundaries(cfg, current_dataset_npz, iter_dir)
            _write_boundary_displacement(run_dir, iter_dir, iteration, boundaries, cfg)
        _write_boundary_candidate_csv(iter_dir)

        selected_points, selected_meta, boundary_selection_summary = _select_acquisition_points(
            cfg=cfg,
            grid_points=grid.points,
            scores=score_pack,
            iteration=iteration,
        )
        selected_meta = _attach_selected_prediction_metadata(selected_meta, pred_grid)
        selected_meta = _attach_selected_boundary_distance(selected_meta, grid, pred_grid, cfg)
        _save_monitor_predictions(iter_dir, iteration, grid, pred_grid, score_pack)

        _save_candidate_csv(iter_dir, grid.points, score_pack)
        _save_candidate_csv(iter_dir, grid.points, score_pack, filename="candidate_scores_global.csv")
        pd.DataFrame(selected_points, columns=["kT", "JA"]).to_csv(iter_dir / "selected_points.csv", index=False)
        _write_selected_by_pool(iter_dir, selected_meta.to_dict("records") if not selected_meta.empty else [])
        _write_selection_diagnostics(
            run_dir,
            iter_dir,
            selected_points,
            dataset.x,
            score_pack,
            cfg,
            grid=grid,
            predictions=pred_grid,
            iteration=iteration,
            boundary_selection_summary=boundary_selection_summary,
        )

        if cfg.enable_early_stop and selected_points.shape[0] == 0:
            print(f"Early stop at iteration {iteration}: no selected candidate points.")
            _save_dataset(iter_dir, iteration, dataset)
            break

        # diagnostics and learning-curve metrics against current exact data
        current_metrics = _evaluate_on_validation(bundle, dataset, n_exact_calls=n_exact_calls, cfg=cfg)
        current_metrics["topology_context"] = {
            "trivial_count": int(topology_context["trivial_points"].shape[0]),
            "topological_count": int(topology_context["topological_points"].shape[0]),
            "gapless_count": int(topology_context["gapless_points"].shape[0]),
            "trusted_topology_count": int(topology_context["trusted_topology_points"].shape[0]),
        }
        metrics_history.append(current_metrics)
        fig_paths = write_iteration_figures(
            figures_dir=cfg.figures_dir,
            run_id=args.run_id,
            iteration=iteration,
            grid=grid,
            predictions=pred_grid,
            scores=score_pack,
            selected_points=selected_points,
            existing_points=dataset.x,
        )
        (iter_dir / "metrics.json").write_text(json.dumps(current_metrics, indent=2), encoding="utf-8")
        (iter_dir / "figures.json").write_text(
            json.dumps({k: str(v) for k, v in fig_paths.items()}, indent=2),
            encoding="utf-8",
        )

        if args.mode == "hpc":
            _write_hpc_selection_artifacts(args, run_dir, iter_dir, iteration, selected_points)
            _save_dataset(iter_dir, iteration, dataset)
            # HPC mode exits after shard generation for this iteration.
            break

        if args.dry_run:
            print(f"Dry-run iteration {iteration}: selected {len(selected_points)} points; exact oracle skipped.")
            _save_dataset(iter_dir, iteration, dataset)
            continue

        # local exact evaluation
        input_samples = int(dataset.x.shape[0])
        oracle = evaluate_points(
            points=selected_points,
            device=args.device,
            output_file=iter_dir / "exact_local_partial.npz",
            save_every=1,
            enable_q_expansion=True,
            enable_delta_refinement=True,
            oracle_mode=str(args.oracle_mode),
            branch_dir=iter_dir / "branch_candidates",
        )
        result = oracle.to_dict()
        np.savez(iter_dir / "exact_local_iter.npz", **result)
        trusted_mask = np.asarray(result.get("trusted_exact", np.ones(selected_points.shape[0], dtype=np.int8))).astype(bool)
        if not np.any(trusted_mask):
            print("Local exact oracle produced no trusted points; dataset append skipped.")
        else:
            trusted_result = {
                k: (np.asarray(v)[trusted_mask] if np.asarray(v).ndim >= 1 and np.asarray(v).shape[0] == trusted_mask.shape[0] else v)
                for k, v in result.items()
            }
            dataset = _dataset_from_result(dataset, trusted_result, cfg)
        new_unique_samples = max(0, int(dataset.x.shape[0]) - input_samples)
        (iter_dir / "local_append_summary.json").write_text(
            json.dumps(
                {
                    "iteration": int(iteration),
                    "input_samples": input_samples,
                    "output_samples": int(dataset.x.shape[0]),
                    "trusted_points_appended": int(np.sum(trusted_mask)),
                    "new_unique_samples_added": new_unique_samples,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        n_exact_calls += int(np.sum(trusted_mask))
        _save_dataset(iter_dir, iteration, dataset)

        if cfg.enable_early_stop:
            if new_unique_samples <= 0:
                print(f"Early stop at iteration {iteration}: zero new unique samples appended.")
                break
            if new_unique_samples < int(cfg.min_new_points_per_iter):
                low_append_iters += 1
                print(
                    "Low new-sample append count "
                    f"{new_unique_samples} < {cfg.min_new_points_per_iter}; "
                    f"streak={low_append_iters}/{cfg.max_low_append_iters}"
                )
            else:
                low_append_iters = 0
            if low_append_iters >= int(cfg.max_low_append_iters):
                print(
                    f"Early stop at iteration {iteration}: new unique samples below "
                    f"{cfg.min_new_points_per_iter} for {low_append_iters} consecutive iterations."
                )
                break

    lc = write_learning_curve(cfg.figures_dir, args.run_id, metrics_history)
    metrics_path.write_text(json.dumps(metrics_history, indent=2), encoding="utf-8")
    print(f"Learning curve saved: {lc}")
    print(f"Run complete: {run_dir}")


def main() -> None:
    args = _parse_args()
    run_active_refinement(args)


if __name__ == "__main__":
    main()
