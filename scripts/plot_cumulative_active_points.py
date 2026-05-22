from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_COLORS = {
    0: "#d9d9d9",
    1: "#7fbf7b",
    2: "#6baed6",
}


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            raise
        df = pd.read_csv(csv_path)
        if not {"kT", "JA", "phase_label"}.issubset(df.columns):
            raise
        return {
            "x": df[["kT", "JA"]].to_numpy(dtype=np.float64),
            "y_phase": df["phase_label"].to_numpy(dtype=np.int64),
        }


def _phase_colors(labels: np.ndarray) -> list[str]:
    return [PHASE_COLORS.get(int(v), "#bdbdbd") for v in labels]


def _read_points_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    df = pd.read_csv(path)
    if not {"kT", "JA"}.issubset(df.columns):
        return np.empty((0, 2), dtype=np.float64)
    return df[["kT", "JA"]].to_numpy(dtype=np.float64)


def _iter_dirs(run_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(run_dir.glob("iter*")):
        if p.is_dir():
            try:
                int(p.name.replace("iter", ""))
            except ValueError:
                continue
            out.append(p)
    return out


def _iter_index(iter_dir: Path) -> int:
    return int(iter_dir.name.replace("iter", ""))


def _load_exact_points(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64), {}
    with np.load(path, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files}
    if "kT" not in data or "JA" not in data:
        return np.empty((0, 2), dtype=np.float64), data
    points = np.column_stack([data["kT"].astype(float), data["JA"].astype(float)])
    return points, data


def _style_axes(ax) -> None:
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_xlim(-0.01, 0.57)
    ax.set_ylim(-0.03, 2.15)
    ax.grid(alpha=0.18, linewidth=0.5)


def plot_cumulative_selected(run_dir: Path, output_dir: Path) -> dict:
    warm = _load_dataset(run_dir / "dataset_iter000.npz")
    x = warm["x"]
    phase = warm["y_phase"]
    iter_dirs = _iter_dirs(run_dir)

    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    ax.scatter(x[:, 0], x[:, 1], s=5, c=_phase_colors(phase), alpha=0.24, linewidths=0)

    selected_parts: list[np.ndarray] = []
    selected_iters: list[np.ndarray] = []
    selected_total = 0
    for iter_dir in iter_dirs:
        idx = _iter_index(iter_dir)
        pts = _read_points_csv(iter_dir / "selected_points.csv")
        selected_total += int(pts.shape[0])
        if pts.size:
            selected_parts.append(pts)
            selected_iters.append(np.full(pts.shape[0], idx, dtype=np.float64))
    if selected_parts:
        selected_all = np.vstack(selected_parts)
        iter_all = np.concatenate(selected_iters)
        sc = ax.scatter(
            selected_all[:, 0],
            selected_all[:, 1],
            s=16,
            c=iter_all,
            cmap="viridis",
            marker="x",
            linewidths=0.8,
            label=f"selected: {selected_total}",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label("iteration")

    _style_axes(ax)
    ax.set_title("Cumulative Active-Learning Selected Points")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{run_dir.name}_cumulative_selected_points.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return {"path": str(out), "selected_total": selected_total}


def plot_cumulative_accepted(run_dir: Path, output_dir: Path) -> dict:
    warm = _load_dataset(run_dir / "dataset_iter000.npz")
    x = warm["x"]
    phase = warm["y_phase"]
    iter_dirs = _iter_dirs(run_dir)

    accepted_parts: list[np.ndarray] = []
    boundary_parts: list[np.ndarray] = []
    rerun_parts: list[np.ndarray] = []
    accepted_total = 0
    boundary_total = 0
    rerun_total = 0

    for iter_dir in iter_dirs:
        idx = _iter_index(iter_dir)
        accepted, accepted_data = _load_exact_points(iter_dir / f"exact_training_iter{idx:03d}.npz")
        merged, merged_data = _load_exact_points(iter_dir / f"exact_merged_iter{idx:03d}.npz")

        if accepted.size:
            accepted_parts.append(accepted)
            accepted_total += int(accepted.shape[0])
            if "delta_boundary_band_normal" in accepted_data:
                mask = accepted_data["delta_boundary_band_normal"].astype(bool)
                if np.any(mask):
                    boundary_parts.append(accepted[mask])
                    boundary_total += int(np.sum(mask))

        if merged.size and "needs_rerun_exact" in merged_data:
            mask = merged_data["needs_rerun_exact"].astype(bool)
            if np.any(mask):
                rerun_parts.append(merged[mask])
                rerun_total += int(np.sum(mask))

    accepted_all = np.vstack(accepted_parts) if accepted_parts else np.empty((0, 2), dtype=np.float64)
    boundary_all = np.vstack(boundary_parts) if boundary_parts else np.empty((0, 2), dtype=np.float64)
    rerun_all = np.vstack(rerun_parts) if rerun_parts else np.empty((0, 2), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    ax.scatter(x[:, 0], x[:, 1], s=5, c=_phase_colors(phase), alpha=0.20, linewidths=0, label="warm-start")
    if accepted_all.size:
        ax.scatter(
            accepted_all[:, 0],
            accepted_all[:, 1],
            s=18,
            c="#1b9e77",
            alpha=0.78,
            linewidths=0,
            label=f"accepted: {accepted_total}",
        )
    if boundary_all.size:
        ax.scatter(
            boundary_all[:, 0],
            boundary_all[:, 1],
            s=34,
            facecolors="none",
            edgecolors="#e6ab02",
            linewidths=1.1,
            label=f"boundary-band normal: {boundary_total}",
        )
    if rerun_all.size:
        ax.scatter(
            rerun_all[:, 0],
            rerun_all[:, 1],
            s=34,
            c="#d95f02",
            marker="x",
            linewidths=1.1,
            label=f"rerun-required: {rerun_total}",
        )

    _style_axes(ax)
    ax.set_title("Cumulative Accepted and Rejected Exact Points")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{run_dir.name}_cumulative_accepted_points.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return {
        "path": str(out),
        "accepted_total": accepted_total,
        "boundary_band_total": boundary_total,
        "rerun_required_total": rerun_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot cumulative active-learning selected and accepted points.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = plot_cumulative_selected(args.run_dir, args.output_dir)
    accepted = plot_cumulative_accepted(args.run_dir, args.output_dir)
    summary = {
        "run_dir": str(args.run_dir),
        "selected": selected,
        "accepted": accepted,
    }
    summary_path = args.output_dir / f"{args.run_dir.name}_cumulative_points_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
