from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_phase.acquisition import build_candidate_grid, compute_acquisition_scores, select_acquisition_batch
from ml_phase.active_refine import _apply_candidate_exclusions, _validate_boundary_mode
from ml_phase.config import ActiveLearningConfig
from ml_phase.extract_phase_boundaries import extract_phase_boundaries


def _toy_predictions(n: int) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    delta = 1e-3 + 4e-3 * np.sin(2 * np.pi * x)
    q = 1e-2 + 3e-2 * np.cos(2 * np.pi * x)
    eta = 0.1 * np.sin(4 * np.pi * x)
    reg_mean = np.stack([delta, q, eta, eta + 0.2, eta - 0.2], axis=1)
    reg_std = np.stack([0.01 + x, 0.02 + x, 0.03 + x, 0.04 + x, 0.05 + x], axis=1) * 1e-2
    probs = np.stack([1.0 - x, 0.5 * x, 0.5 * x], axis=1)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "reg_mean": reg_mean,
        "reg_std": reg_std,
        "phase_proba": probs,
        "phase_pred": np.argmax(probs, axis=1),
        "cls_uncertainty": 1.0 - np.max(probs, axis=1),
    }


def check_acquisition_only_selection() -> None:
    cfg = ActiveLearningConfig(
        run_mode="refinement",
        candidate_domain_mode="full",
        selection_mode="topk",
        n_kt_candidates=16,
        n_ja_candidates=18,
        points_per_iter=12,
        boundary_refinement_mode="diagnostic",
    )
    grid = build_candidate_grid(cfg)
    pred = _toy_predictions(grid.points.shape[0])
    scores = compute_acquisition_scores(cfg, grid, pred, existing_points=np.empty((0, 2)))
    scores = _apply_candidate_exclusions(
        cfg=cfg,
        grid_points=grid.points,
        scores=scores,
        existing_points=grid.points[:5],
        boundary_band_points=np.empty((0, 2)),
        run_dir=Path("__missing_run_dir__"),
        current_iteration=0,
    )
    selected, rows = select_acquisition_batch(grid.points, scores, k=cfg.points_per_iter, cfg=cfg)
    assert selected.shape[0] == cfg.points_per_iter
    assert rows
    assert {r["selection_source"] for r in rows} == {"acquisition"}
    assert {r["selection_pool"] for r in rows} == {"acquisition"}
    assert all(r["boundary_type"] == "" for r in rows)
    assert np.all(np.isfinite([r["final_score"] for r in rows]))
    assert "R_obs" in scores
    assert np.all(scores["R_obs"] >= cfg.observation_repulsion_floor)


def check_midpoint_modes_rejected() -> None:
    assert _validate_boundary_mode("off") == "off"
    assert _validate_boundary_mode("diagnostic") == "diagnostic"
    for mode in ("hybrid", "local"):
        try:
            _validate_boundary_mode(mode)
        except ValueError as exc:
            assert "Midpoint-based selection has been disabled" in str(exc)
        else:
            raise AssertionError(f"legacy midpoint mode was not rejected: {mode}")


def check_boundary_extraction_diagnostic_only() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "toy_dataset.npz"
        x = np.array([[0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4]], dtype=np.float64)
        y_reg = np.array(
            [
                [0.0, 0.0, -0.1, 0.0, 0.0],
                [2e-3, 0.0, 0.1, 0.0, 0.0],
                [2e-3, 0.02, 0.2, 0.0, 0.0],
                [0.0, 0.0, -0.2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.savez(
            dataset,
            x=x,
            y_reg=y_reg,
            y_phase=np.array([0, 1, 2, 0], dtype=np.int64),
            y_eta_sign=np.array([0, 2, 2, 0], dtype=np.int64),
            y_strong_diode=np.array([0, 0, 0, 0], dtype=np.int64),
        )
        out = root / "boundaries"
        summary = extract_phase_boundaries(
            Namespace(
                dataset=dataset,
                output_dir=out,
                kt_bin_width=0.01,
                max_local_spacing=1.0,
                max_refinement_points=16,
                output_root=root,
            )
        )
        assert summary["n_boundary_segments"] > 0
        assert summary["n_targeted_refinement_points"] == 0
        assert summary["target_generation"] == "disabled"


def main() -> None:
    check_acquisition_only_selection()
    check_midpoint_modes_rejected()
    check_boundary_extraction_diagnostic_only()
    print("acquisition-only checks passed")


if __name__ == "__main__":
    main()
