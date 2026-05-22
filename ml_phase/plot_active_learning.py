from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .acquisition import CandidateGrid


def _reshape_on_grid(values: np.ndarray, grid: CandidateGrid) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(grid.full_shape)


def _save_map(
    out_path: Path,
    grid: CandidateGrid,
    z: np.ndarray,
    title: str,
    cmap: str = "viridis",
    scatter_points: np.ndarray | None = None,
    existing_points: np.ndarray | None = None,
    boundary_points: np.ndarray | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.pcolormesh(grid.kt_values, grid.ja_values, _reshape_on_grid(z, grid), cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax)
    if existing_points is not None and existing_points.size:
        ax.scatter(existing_points[:, 0], existing_points[:, 1], s=2.0, c="white", alpha=0.15, linewidths=0)
    if boundary_points is not None and boundary_points.size:
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], s=5.0, c="black", alpha=0.55, linewidths=0, label="predicted boundary")
    if scatter_points is not None and scatter_points.size:
        ax.scatter(scatter_points[:, 0], scatter_points[:, 1], s=18, c="white", marker="x", linewidths=1.0, label="selected")
    if boundary_points is not None and boundary_points.size:
        ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.set_xlabel(r"$k_B T / t$")
    ax.set_ylabel(r"$J_A / t$")
    ax.set_xlim(0.0, float(grid.kt_values.max()))
    ax.set_ylim(float(grid.ja_values.min()), float(grid.ja_values.max()))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _predicted_boundary_points(grid: CandidateGrid, phase_pred: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase_pred, dtype=np.int64).reshape(grid.full_shape)
    kt = np.asarray(grid.kt_values, dtype=np.float64)
    ja = np.asarray(grid.ja_values, dtype=np.float64)
    pts: list[tuple[float, float]] = []
    nja, nkt = phase.shape
    for j in range(nja):
        for i in range(nkt - 1):
            if {int(phase[j, i]), int(phase[j, i + 1])} in ({0, 1}, {0, 2}, {1, 2}):
                pts.append((0.5 * (kt[i] + kt[i + 1]), float(ja[j])))
    for j in range(nja - 1):
        for i in range(nkt):
            if {int(phase[j, i]), int(phase[j + 1, i])} in ({0, 1}, {0, 2}, {1, 2}):
                pts.append((float(kt[i]), 0.5 * (ja[j] + ja[j + 1])))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


def write_iteration_figures(
    figures_dir: Path,
    run_id: str,
    iteration: int,
    grid: CandidateGrid,
    predictions: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    selected_points: np.ndarray,
    existing_points: np.ndarray,
) -> Dict[str, Path]:
    prefix = f"{run_id}_iter{iteration:03d}"
    paths: Dict[str, Path] = {}

    eta_pred = predictions["reg_mean"][:, 2]
    phase_pred = predictions["phase_pred"]
    cls_unc = predictions["cls_uncertainty"]
    acq = scores.get("A0_main", scores["score"])
    boundary_points = _predicted_boundary_points(grid, phase_pred)

    p1 = figures_dir / f"{prefix}_phase_prediction.png"
    _save_map(p1, grid, phase_pred, "Phase Prediction", cmap="tab10", scatter_points=selected_points, existing_points=existing_points, boundary_points=boundary_points)
    paths["phase_prediction"] = p1

    p2 = figures_dir / f"{prefix}_eta_prediction.png"
    _save_map(p2, grid, eta_pred, "Eta Prediction", cmap="coolwarm", scatter_points=selected_points, existing_points=existing_points)
    paths["eta_prediction"] = p2

    p3 = figures_dir / f"{prefix}_uncertainty.png"
    _save_map(p3, grid, cls_unc, "Classification Uncertainty", cmap="magma", scatter_points=selected_points, existing_points=existing_points)
    paths["uncertainty"] = p3

    p4 = figures_dir / f"{prefix}_acquisition.png"
    _save_map(p4, grid, acq, "A0_main Acquisition Value", cmap="plasma", scatter_points=selected_points, existing_points=existing_points, boundary_points=boundary_points)
    paths["acquisition"] = p4

    p5 = figures_dir / f"{prefix}_selected_points.png"
    _save_map(p5, grid, phase_pred, "Selected Points on Predicted Phase Map", cmap="tab10", scatter_points=selected_points, existing_points=existing_points, boundary_points=boundary_points)
    paths["selected_points"] = p5

    if "q_edge_risk_score" in scores:
        p6 = figures_dir / f"{prefix}_q_edge_risk.png"
        _save_map(p6, grid, scores["q_edge_risk_score"], "q-Window Risk", cmap="inferno", scatter_points=selected_points, existing_points=existing_points)
        paths["q_edge_risk"] = p6

    if "delta_refine_risk_score" in scores:
        p7 = figures_dir / f"{prefix}_delta_refine_risk.png"
        _save_map(p7, grid, scores["delta_refine_risk_score"], "Delta Refinement Risk", cmap="magma", scatter_points=selected_points, existing_points=existing_points)
        paths["delta_refine_risk"] = p7
    return paths


def write_learning_curve(
    figures_dir: Path,
    run_id: str,
    metrics_history: list[dict[str, float]],
) -> Path:
    out = figures_dir / f"{run_id}_learning_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Learning Curve")
        fig.savefig(out, dpi=220)
        plt.close(fig)
        return out

    it = np.arange(len(metrics_history))
    eta_rmse = np.array([m.get("eta_rmse", np.nan) for m in metrics_history], dtype=np.float64)
    acc = np.array([m.get("phase_accuracy", np.nan) for m in metrics_history], dtype=np.float64)
    f1 = np.array([m.get("boundary_f1", np.nan) for m in metrics_history], dtype=np.float64)
    has_boundary_f1 = bool(np.any(np.isfinite(f1)))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax1.plot(it, eta_rmse, marker="o", label="eta RMSE")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("eta RMSE")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(it, acc, marker="s", color="tab:green", label="phase acc")
    if has_boundary_f1:
        ax2.plot(it, f1, marker="^", color="tab:red", label="boundary F1")
        ax2.set_ylabel("Accuracy / F1")
    else:
        ax2.set_ylabel("Accuracy")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out
