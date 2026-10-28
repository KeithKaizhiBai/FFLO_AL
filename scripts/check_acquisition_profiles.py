from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.acquisition import build_candidate_grid, compute_acquisition_scores, select_acquisition_batch
from ml_phase.config import ActiveLearningConfig


def _mock_predictions(n: int, seed: int = 1234) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    reg_mean = np.column_stack(
        [
            rng.uniform(0.0, 0.08, size=n),    # delta
            rng.uniform(-0.35, 0.35, size=n),  # q
            rng.uniform(-0.8, 0.8, size=n),    # eta
            rng.uniform(0.0, 0.02, size=n),    # ic+
            rng.uniform(0.0, 0.02, size=n),    # ic-
        ]
    )
    reg_std = np.column_stack(
        [
            rng.uniform(0.001, 0.02, size=n),
            rng.uniform(0.001, 0.03, size=n),
            rng.uniform(0.001, 0.08, size=n),
            rng.uniform(0.001, 0.01, size=n),
            rng.uniform(0.001, 0.01, size=n),
        ]
    )
    raw = rng.uniform(0.0, 1.0, size=(n, 3))
    phase_prob = raw / np.sum(raw, axis=1, keepdims=True)
    phase_pred = np.argmax(phase_prob, axis=1)
    cls_unc = 1.0 - np.max(phase_prob, axis=1)
    return {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "phase_prob": phase_prob,
        "phase_pred": phase_pred,
        "cls_uncertainty": cls_unc,
    }


def _manual_full_reference(cfg: ActiveLearningConfig, score_pack: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "A_phase_ref": (
            float(cfg.w_cls_mix) * score_pack["cls_uncertainty"]
            + float(cfg.w_reg_phase) * score_pack["U_reg_phase"]
            + float(cfg.w_delta_boundary) * score_pack["B_delta_gated"]
            + float(cfg.w_q_boundary_sc) * score_pack["B_q_gated"]
            + float(cfg.w_gradient_phase) * score_pack["gradient_score"]
        ),
        "A_numerical_ref": float(cfg.w_q_edge_risk) * score_pack["q_edge_risk_score"],
        "A_explore_ref": score_pack["w_ext_current"] * score_pack["extrapolation_risk_score"],
    }


def run_checks(output: Path) -> None:
    cfg = ActiveLearningConfig(
        run_mode="discovery",
        candidate_domain_mode="full",
        n_kt_candidates=48,
        n_ja_candidates=64,
        batch_size_max=64,
        batch_size_min_before_min_iter=16,
        batch_size_min_after_min_iter=8,
        active_pool_quantile_start=0.90,
        active_pool_quantile_mid=0.95,
        active_pool_quantile_end=0.98,
        active_pool_rel_to_p95=0.7,
        random_seed=42,
    )
    grid = build_candidate_grid(cfg)
    preds = _mock_predictions(grid.points.shape[0], seed=20260525)
    existing_points = grid.points[::31].copy()

    full_cfg = replace(cfg, acquisition_profile="full")
    full_scores = compute_acquisition_scores(full_cfg, grid, preds, existing_points=existing_points, iteration=12)
    ref = _manual_full_reference(full_cfg, full_scores)
    a0_ref = ref["A_phase_ref"] + ref["A_numerical_ref"] + ref["A_explore_ref"]
    assert np.allclose(full_scores["A_phase"], ref["A_phase_ref"], rtol=1e-12, atol=1e-12)
    assert np.allclose(full_scores["A_numerical"], ref["A_numerical_ref"], rtol=1e-12, atol=1e-12)
    assert np.allclose(full_scores["A_explore"], ref["A_explore_ref"], rtol=1e-12, atol=1e-12)
    assert np.allclose(full_scores["A0_main"], a0_ref, rtol=1e-12, atol=1e-12)

    simple_cfg = replace(
        cfg,
        acquisition_profile="simple_phase",
        w_ext_simple_start=0.02,
        w_ext_simple_mid=0.01,
        w_ext_simple_end=0.0,
    )
    simple_scores = compute_acquisition_scores(simple_cfg, grid, preds, existing_points=existing_points, iteration=12)
    assert np.allclose(simple_scores["A_numerical"], 0.0, rtol=0.0, atol=0.0)
    assert np.allclose(
        simple_scores["A0_main"],
        simple_scores["A_phase_simple"] + simple_scores["A_explore_simple"],
        rtol=1e-12,
        atol=1e-12,
    )

    cleanup_cfg = replace(cfg, acquisition_profile="surprise_cleanup")
    cleanup_scores = compute_acquisition_scores(cleanup_cfg, grid, preds, existing_points=existing_points, iteration=12)
    assert np.allclose(cleanup_scores["A_numerical"], 0.0, rtol=0.0, atol=0.0)
    assert np.all(cleanup_scores["surprise_cleanup_qedge_factor"] >= cleanup_cfg.surprise_cleanup_qedge_floor - 1e-12)
    assert np.all(cleanup_scores["surprise_cleanup_qedge_factor"] <= 1.0 + 1e-12)

    full_scores_for_select = dict(full_scores)
    selected_full, rows_full = select_acquisition_batch(
        grid.points,
        full_scores_for_select,
        k=64,
        cfg=full_cfg,
        rng_seed=123,
        iteration=12,
    )
    simple_scores_for_select = dict(simple_scores)
    selected_simple, rows_simple = select_acquisition_batch(
        grid.points,
        simple_scores_for_select,
        k=64,
        cfg=simple_cfg,
        rng_seed=123,
        iteration=12,
    )
    cleanup_scores_for_select = dict(cleanup_scores)
    selected_cleanup, rows_cleanup = select_acquisition_batch(
        grid.points,
        cleanup_scores_for_select,
        k=64,
        cfg=cleanup_cfg,
        rng_seed=123,
        iteration=12,
    )
    assert selected_full.shape[0] > 0
    assert selected_simple.shape[0] > 0
    assert selected_cleanup.shape[0] > 0
    assert all(str(r.get("acquisition_profile")) == "full" for r in rows_full)
    assert all(str(r.get("acquisition_profile")) == "simple_phase" for r in rows_simple)
    assert all(str(r.get("acquisition_profile")) == "surprise_cleanup" for r in rows_cleanup)

    high_conf_simple = float(np.mean([float(r.get("high_confidence_interior", 0)) for r in rows_simple])) if rows_simple else 0.0
    pool_frac_full = float(full_scores_for_select.get("_selection_summary", {}).get("active_pool_fraction", np.nan))
    pool_frac_simple = float(simple_scores_for_select.get("_selection_summary", {}).get("active_pool_fraction", np.nan))

    report = {
        "full_profile_formula_regression": "pass",
        "simple_profile_output_check": "pass",
        "selection_nonempty_full": int(selected_full.shape[0]),
        "selection_nonempty_simple": int(selected_simple.shape[0]),
        "selection_nonempty_surprise_cleanup": int(selected_cleanup.shape[0]),
        "selected_high_confidence_interior_fraction_simple": high_conf_simple,
        "active_pool_fraction_full": pool_frac_full,
        "active_pool_fraction_simple": pool_frac_simple,
        "active_pool_fraction_surprise_cleanup": float(cleanup_scores_for_select.get("_selection_summary", {}).get("active_pool_fraction", np.nan)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    out = Path("reports/acquisition_profile_smoke/acquisition_profile_smoke.json")
    run_checks(out)


if __name__ == "__main__":
    main()
