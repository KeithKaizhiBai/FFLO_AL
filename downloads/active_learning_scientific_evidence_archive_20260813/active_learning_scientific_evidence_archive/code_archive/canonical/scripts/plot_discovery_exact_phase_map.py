from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PHASE_COLORS = {
    0: "#d9d9d9",
    1: "#2c7fb8",
    2: "#d95f02",
}

PHASE_LABELS = {
    0: "normal",
    1: "uniform SC",
    2: "FFLO",
}


def _load_boundary(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    df = pd.read_csv(path)
    if "kT_boundary" not in df or "JA_boundary" not in df:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    return df[np.isfinite(df["kT_boundary"]) & np.isfinite(df["JA_boundary"])].copy()


def _scatter_phase(ax: plt.Axes, x: np.ndarray, phase: np.ndarray) -> None:
    for label in sorted(PHASE_LABELS):
        mask = phase == label
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask, 0],
            x[mask, 1],
            s=7,
            c=PHASE_COLORS[label],
            alpha=0.58 if label else 0.34,
            edgecolors="none",
            rasterized=True,
        )


def plot_exact_phase_map(dataset: Path, boundary_dir: Path, output_stem: Path) -> dict[str, object]:
    data = np.load(dataset, allow_pickle=True)
    x = np.asarray(data["x"], dtype=np.float64)
    phase = np.asarray(data["y_phase"], dtype=np.int64)
    delta = np.asarray(data["y_reg"][:, 0], dtype=np.float64)
    q = np.asarray(data["y_reg"][:, 1], dtype=np.float64)

    normal_sc = _load_boundary(boundary_dir / "normal_sc_boundary_segments.csv")
    uniform_fflo = _load_boundary(boundary_dir / "uniform_fflo_boundary_segments.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    _scatter_phase(ax, x, phase)

    if not normal_sc.empty:
        ax.scatter(
            normal_sc["kT_boundary"],
            normal_sc["JA_boundary"],
            s=16,
            c="#111111",
            marker=".",
            linewidths=0,
            label="normal/SC boundary",
            zorder=4,
        )
    if not uniform_fflo.empty:
        ax.scatter(
            uniform_fflo["kT_boundary"],
            uniform_fflo["JA_boundary"],
            s=17,
            facecolors="none",
            edgecolors="#7b1fa2",
            marker="o",
            linewidths=0.8,
            label="uniform/FFLO boundary",
            zorder=5,
        )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[0], markeredgecolor="none", markersize=6, alpha=0.45, label="normal"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[1], markeredgecolor="none", markersize=6, alpha=0.75, label="uniform SC"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PHASE_COLORS[2], markeredgecolor="none", markersize=6, alpha=0.75, label="FFLO"),
        Line2D([0], [0], marker=".", color="#111111", linestyle="none", markersize=9, label="normal/SC boundary"),
        Line2D([0], [0], marker="o", color="#7b1fa2", markerfacecolor="none", linestyle="none", markersize=5, label="uniform/FFLO boundary"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92, fontsize=8)

    ax.set_xlim(-0.005, 0.565)
    ax.set_ylim(-0.02, 2.15)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Discovery exact-data phase map")
    ax.grid(True, color="white", linewidth=0.5, alpha=0.5)
    ax.set_facecolor("#f7f7f7")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = output_stem.with_suffix(".pdf")
    png = output_stem.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=220)
    plt.close(fig)

    phase_counts = {PHASE_LABELS[int(k)]: int(v) for k, v in zip(*np.unique(phase, return_counts=True))}
    summary = {
        "dataset": str(dataset),
        "n_points": int(x.shape[0]),
        "phase_counts": phase_counts,
        "delta_min": float(np.nanmin(delta)),
        "delta_max": float(np.nanmax(delta)),
        "q_min": float(np.nanmin(q)),
        "q_max": float(np.nanmax(q)),
        "normal_sc_boundary_points": int(len(normal_sc)),
        "uniform_fflo_boundary_points": int(len(uniform_fflo)),
        "pdf": str(pdf),
        "png": str(png),
    }
    output_stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an exact-data phase map for a discovery active-learning run.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--boundary-dir", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    args = parser.parse_args()
    summary = plot_exact_phase_map(args.dataset, args.boundary_dir, args.output_stem)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
