from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_phase.config import ActiveLearningConfig
from ml_phase.dataset_builder import load_flat_dataset, load_warm_start_npz
from ml_phase.labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC, phase_label


DEFAULT_WARM_START = Path(
    "eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_"
    "kk200_eb10000_fp64_libcusolver_cfg422bd68ce6/"
    "eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_"
    "kk200_eb10000_fp64_libcusolver_cfg422bd68ce6.npz"
)
DEFAULT_CURRENT_DATASET = Path(
    "hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/"
    "active_runs/active_boundary_loop_v1/dataset_iter042.npz"
)
DEFAULT_BOUNDARY_DIR = Path(
    "hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/"
    "boundaries/recheck_iter042_default"
)
DEFAULT_OUTPUT_DIR = Path(
    "hpc_upload_qdelta_20260512_131417/ML_Phase_128_39_42/"
    "figures/phase_boundary_overlay"
)


def cell_edges_from_centers(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def rounded_coord_keys(x: np.ndarray, decimals: int) -> set[tuple[float, float]]:
    rounded = np.round(np.asarray(x, dtype=np.float64), decimals=decimals)
    return {(float(row[0]), float(row[1])) for row in rounded}


def current_dataset_frame(path: Path) -> pd.DataFrame:
    dataset = load_flat_dataset(path)
    records = dataset.records
    return pd.DataFrame(
        {
            "kT": dataset.x[:, 0],
            "JA": dataset.x[:, 1],
            "delta_opt": dataset.y_reg[:, 0],
            "q_opt": dataset.y_reg[:, 1],
            "eta": dataset.y_reg[:, 2],
            "phase_label": dataset.y_phase,
            "q_expanded": np.asarray(records.get("q_expanded", np.zeros(dataset.x.shape[0])), dtype=bool),
            "delta_refined": np.asarray(records.get("delta_refined", np.zeros(dataset.x.shape[0])), dtype=bool),
            "delta_boundary_band_normal": np.asarray(
                records.get("delta_boundary_band_normal", np.zeros(dataset.x.shape[0])), dtype=bool
            ),
        }
    )


def load_boundary_csvs(boundary_dir: Path) -> dict[str, pd.DataFrame]:
    names = {
        "normal_sc": "normal_sc_boundary_segments.csv",
        "uniform_fflo": "uniform_fflo_boundary_segments.csv",
        "strong_diode": "strong_diode_boundary_segments.csv",
        "eta_zero": "eta_zero_boundary_segments.csv",
    }
    out = {}
    for key, name in names.items():
        path = boundary_dir / name
        out[key] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return out


def git_commit_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def plot_overlay(
    warm_start: Path,
    current_dataset: Path,
    boundary_dir: Path,
    output_dir: Path,
    coord_decimals: int,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = ActiveLearningConfig()
    warm = load_warm_start_npz(warm_start)
    kt_vec = np.asarray(warm["kT_vec"], dtype=np.float64)
    ja_vec = np.asarray(warm["JA_vec"], dtype=np.float64)
    delta = np.asarray(warm["delta_opt_matrix"], dtype=np.float64)
    q_opt = np.asarray(warm["q_opt_matrix"], dtype=np.float64)
    phase = phase_label(delta.ravel(), q_opt.ravel(), cfg.delta_eps, cfg.q_eps).reshape(delta.shape)

    kt_mask = kt_vec >= 0.0
    kt_plot = kt_vec[kt_mask]
    phase_plot = phase[:, kt_mask]
    kt_edges = cell_edges_from_centers(kt_plot)
    kt_edges[0] = max(0.0, kt_edges[0])
    ja_edges = cell_edges_from_centers(ja_vec)

    current = current_dataset_frame(current_dataset)
    kt_mesh, ja_mesh = np.meshgrid(kt_vec, ja_vec, indexing="xy")
    warm_keys = rounded_coord_keys(np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1), decimals=coord_decimals)
    current_keys = rounded_coord_keys(current[["kT", "JA"]].to_numpy(), decimals=coord_decimals)
    current["is_new_exact"] = [
        (round(float(row.kT), coord_decimals), round(float(row.JA), coord_decimals)) not in warm_keys
        for row in current.itertuples(index=False)
    ]
    new_points = current[current["is_new_exact"]].copy()
    boundary_band_new = new_points[new_points["delta_boundary_band_normal"]].copy()
    ordinary_new = new_points[~new_points["delta_boundary_band_normal"]].copy()
    boundaries = load_boundary_csvs(boundary_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    phase_cmap = ListedColormap(["#f2f2f2", "#8ecae6", "#f4a261"])
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    ax.pcolormesh(
        kt_edges,
        ja_edges,
        phase_plot,
        cmap=phase_cmap,
        vmin=-0.5,
        vmax=2.5,
        shading="flat",
        alpha=0.72,
        zorder=0,
    )

    if not boundaries["eta_zero"].empty:
        eta = boundaries["eta_zero"]
        ax.scatter(
            eta["kT_boundary"],
            eta["JA_boundary"],
            s=5,
            c="#2a9d8f",
            alpha=0.12,
            linewidths=0,
            zorder=2,
        )
    if not boundaries["strong_diode"].empty:
        strong = boundaries["strong_diode"]
        ax.scatter(
            strong["kT_boundary"],
            strong["JA_boundary"],
            s=13,
            c="#d62728",
            marker="x",
            alpha=0.75,
            linewidths=0.8,
            zorder=4,
        )
    if not boundaries["uniform_fflo"].empty:
        ufflo = boundaries["uniform_fflo"]
        ax.scatter(
            ufflo["kT_boundary"],
            ufflo["JA_boundary"],
            s=14,
            c="#6a3d9a",
            marker="s",
            alpha=0.9,
            linewidths=0,
            zorder=5,
        )
    if not boundaries["normal_sc"].empty:
        nsc = boundaries["normal_sc"]
        ax.scatter(
            nsc["kT_boundary"],
            nsc["JA_boundary"],
            s=20,
            c="black",
            marker="o",
            alpha=0.92,
            linewidths=0,
            zorder=6,
        )

    if not ordinary_new.empty:
        ax.scatter(
            ordinary_new["kT"],
            ordinary_new["JA"],
            s=17,
            facecolors="none",
            edgecolors="#005f73",
            linewidths=0.65,
            alpha=0.82,
            zorder=7,
        )
    if not boundary_band_new.empty:
        ax.scatter(
            boundary_band_new["kT"],
            boundary_band_new["JA"],
            s=22,
            facecolors="none",
            edgecolors="#e7298a",
            linewidths=0.85,
            alpha=0.9,
            zorder=8,
        )

    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Original Exact Phase Diagram with Rechecked Boundaries and New Exact Points")
    ax.set_xlim(0.0, max(0.56, float(current["kT"].max()) + 0.01))
    ax.set_ylim(0.0, max(2.12, float(current["JA"].max()) + 0.02))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    legend_items = [
        Patch(facecolor="#f2f2f2", edgecolor="0.65", label="original normal"),
        Patch(facecolor="#8ecae6", edgecolor="0.65", label="original uniform SC"),
        Patch(facecolor="#f4a261", edgecolor="0.65", label="original FFLO"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markersize=5, label="normal/SC boundary"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#6a3d9a", markersize=5, label="uniform/FFLO boundary"),
        Line2D([0], [0], marker="x", color="#d62728", linestyle="none", markersize=5, label="strong-diode boundary"),
        Line2D([0], [0], marker="^", color="#2a9d8f", linestyle="none", alpha=0.45, markersize=5, label=r"$\eta=0$ boundary"),
        Line2D([0], [0], marker="o", color="#005f73", markerfacecolor="none", linestyle="none", markersize=5, label="new exact points"),
        Line2D([0], [0], marker="o", color="#e7298a", markerfacecolor="none", linestyle="none", markersize=5, label="new boundary-band points"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.92)

    png_path = output_dir / "original_phase_with_rechecked_boundaries_and_new_data.png"
    pdf_path = output_dir / "original_phase_with_rechecked_boundaries_and_new_data.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "warm_start": str(warm_start),
        "current_dataset": str(current_dataset),
        "boundary_dir": str(boundary_dir),
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "git_commit": git_commit_or_none(),
        "coord_decimals_for_new_point_detection": int(coord_decimals),
        "warm_start_points": int(kt_vec.size * ja_vec.size),
        "current_dataset_points": int(current.shape[0]),
        "new_exact_points": int(new_points.shape[0]),
        "new_boundary_band_points": int(boundary_band_new.shape[0]),
        "boundary_counts": {key: int(value.shape[0]) for key, value in boundaries.items()},
        "phase_background_counts": {
            "normal": int((phase == PHASE_NORMAL).sum()),
            "uniform_SC": int((phase == PHASE_UNIFORM_SC).sum()),
            "FFLO": int((phase == PHASE_FFLO).sum()),
        },
    }
    (output_dir / "original_phase_boundary_overlay_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay rechecked boundary data and new exact points on the original phase map.")
    p.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    p.add_argument("--current-dataset", type=Path, default=DEFAULT_CURRENT_DATASET)
    p.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--coord-decimals", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = plot_overlay(
        warm_start=args.warm_start,
        current_dataset=args.current_dataset,
        boundary_dir=args.boundary_dir,
        output_dir=args.output_dir,
        coord_decimals=args.coord_decimals,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
