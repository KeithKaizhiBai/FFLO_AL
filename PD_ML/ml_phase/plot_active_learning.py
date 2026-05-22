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
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.pcolormesh(grid.kt_values, grid.ja_values, _reshape_on_grid(z, grid), cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax)
    if existing_points is not None and existing_points.size:
        ax.scatter(existing_points[:, 0], existing_points[:, 1], s=2.0, c="white", alpha=0.15, linewidths=0)
    if scatter_points is not None and scatter_points.size:
        ax.scatter(scatter_points[:, 0], scatter_points[:, 1], s=18, c="black", marker="x", linewidths=1.0)
    ax.set_title(title)
    ax.set_xlabel(r"$k_B T / t$")
    ax.set_ylabel(r"$J_A / t$")
    ax.set_xlim(0.0, float(grid.kt_values.max()))
    ax.set_ylim(float(grid.ja_values.min()), float(grid.ja_values.max()))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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
    acq = scores["score"]

    p1 = figures_dir / f"{prefix}_phase_prediction.png"
    _save_map(p1, grid, phase_pred, "Phase Prediction", cmap="tab10", scatter_points=selected_points, existing_points=existing_points)
    paths["phase_prediction"] = p1

    p2 = figures_dir / f"{prefix}_eta_prediction.png"
    _save_map(p2, grid, eta_pred, "Eta Prediction", cmap="coolwarm", scatter_points=selected_points, existing_points=existing_points)
    paths["eta_prediction"] = p2

    p3 = figures_dir / f"{prefix}_uncertainty.png"
    _save_map(p3, grid, cls_unc, "Classification Uncertainty", cmap="magma", scatter_points=selected_points, existing_points=existing_points)
    paths["uncertainty"] = p3

    p4 = figures_dir / f"{prefix}_acquisition.png"
    _save_map(p4, grid, acq, "Acquisition Score", cmap="plasma", scatter_points=selected_points, existing_points=existing_points)
    paths["acquisition"] = p4

    p5 = figures_dir / f"{prefix}_selected_points.png"
    selected_only = np.full(grid.points.shape[0], np.nan, dtype=np.float64)
    _save_map(p5, grid, np.nan_to_num(selected_only), "Selected Points", cmap="Greys", scatter_points=selected_points, existing_points=existing_points)
    paths["selected_points"] = p5
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

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax1.plot(it, eta_rmse, marker="o", label="eta RMSE")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("eta RMSE")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(it, acc, marker="s", color="tab:green", label="phase acc")
    ax2.plot(it, f1, marker="^", color="tab:red", label="boundary F1")
    ax2.set_ylabel("Accuracy / F1")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out

