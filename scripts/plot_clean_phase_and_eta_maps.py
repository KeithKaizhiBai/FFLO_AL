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
    "figures/clean_phase_eta_maps"
)


def cell_edges_from_centers(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Need at least two grid centers.")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    edges[0] = values[0] - 0.5 * (values[1] - values[0])
    edges[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return edges


def finite_t_phase_boundary_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_fflo_ns = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56])
    ja_fflo_ns = np.array([2.12, 1.733, 1.5, 1.32, 1.16, 0.9, 0.78, 0.733, 0.7, 0.667, 0.59, 0.4, 0.178, 0.0])
    t_1st = np.array([0.01, 0.04, 0.05])
    ja_1st = np.array([0.6, 0.6, 0.6])
    t_2nd = np.array([0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4])
    ja_2nd = np.array([0.6, 0.6, 0.62, 0.6277, 0.63, 0.628, 0.617, 0.598, 0.565])
    return t_fflo_ns, ja_fflo_ns, t_1st, ja_1st, t_2nd, ja_2nd


def signed_power(values: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.power(np.abs(values), gamma)


def rounded_coord_keys(x: np.ndarray, decimals: int) -> set[tuple[float, float]]:
    rounded = np.round(np.asarray(x, dtype=np.float64), decimals=decimals)
    return {(float(row[0]), float(row[1])) for row in rounded}


def load_current_frame(path: Path) -> pd.DataFrame:
    dataset = load_flat_dataset(path)
    records = dataset.records
    return pd.DataFrame(
        {
            "kT": dataset.x[:, 0],
            "JA": dataset.x[:, 1],
            "phase_label": dataset.y_phase,
            "delta_opt": dataset.y_reg[:, 0],
            "q_opt": dataset.y_reg[:, 1],
            "eta": dataset.y_reg[:, 2],
            "delta_boundary_band_normal": np.asarray(
                records.get("delta_boundary_band_normal", np.zeros(dataset.x.shape[0])),
                dtype=bool,
            ),
        }
    )


def load_boundary_segments(boundary_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal_sc = pd.read_csv(boundary_dir / "normal_sc_boundary_segments.csv")
    uniform_fflo = pd.read_csv(boundary_dir / "uniform_fflo_boundary_segments.csv")
    return normal_sc, uniform_fflo


def normalized_min_dist_to_boundaries(points: pd.DataFrame, boundaries: pd.DataFrame, cfg: ActiveLearningConfig) -> np.ndarray:
    if points.empty:
        return np.array([], dtype=np.float64)
    if boundaries.empty:
        return np.full(points.shape[0], np.inf, dtype=np.float64)
    p = points[["kT", "JA"]].to_numpy(dtype=np.float64)
    b = boundaries[["kT_boundary", "JA_boundary"]].to_numpy(dtype=np.float64)
    scale = np.array([cfg.kt_max - cfg.kt_min, cfg.ja_max - cfg.ja_min], dtype=np.float64)
    scale = np.maximum(scale, 1.0e-12)
    out = np.empty(p.shape[0], dtype=np.float64)
    chunk = 512
    for start in range(0, p.shape[0], chunk):
        block = p[start : start + chunk]
        d = (block[:, None, :] - b[None, :, :]) / scale[None, None, :]
        out[start : start + chunk] = np.sqrt(np.sum(d * d, axis=2)).min(axis=1)
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


def setup_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_clean_phase_map(
    warm: dict[str, np.ndarray],
    current_df: pd.DataFrame,
    normal_sc: pd.DataFrame,
    uniform_fflo: pd.DataFrame,
    output_dir: Path,
    near_threshold: float,
    coord_decimals: int,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = ActiveLearningConfig()
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

    kt_mesh, ja_mesh = np.meshgrid(kt_vec, ja_vec, indexing="xy")
    warm_keys = rounded_coord_keys(np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1), coord_decimals)
    work = current_df.copy()
    work["is_new"] = [
        (round(float(row.kT), coord_decimals), round(float(row.JA), coord_decimals)) not in warm_keys
        for row in work.itertuples(index=False)
    ]
    new_points = work[work["is_new"]].copy()
    boundary_union = pd.concat([normal_sc, uniform_fflo], ignore_index=True)
    new_points["d_to_main_boundary"] = normalized_min_dist_to_boundaries(new_points, boundary_union, cfg)
    near_new = new_points[new_points["d_to_main_boundary"] <= near_threshold].copy()
    near_band = near_new[near_new["delta_boundary_band_normal"]].copy()
    near_regular = near_new[~near_new["delta_boundary_band_normal"]].copy()

    phase_cmap = ListedColormap(["#f1f1f1", "#54b6d3", "#f28e2b"])
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.pcolormesh(
        kt_edges,
        ja_edges,
        phase_plot,
        cmap=phase_cmap,
        vmin=-0.5,
        vmax=2.5,
        shading="flat",
        alpha=0.82,
        zorder=0,
    )
    ax.scatter(
        normal_sc["kT_boundary"],
        normal_sc["JA_boundary"],
        s=20,
        c="black",
        marker="o",
        linewidths=0,
        alpha=0.95,
        zorder=5,
    )
    ax.scatter(
        uniform_fflo["kT_boundary"],
        uniform_fflo["JA_boundary"],
        s=20,
        c="#5e3c99",
        marker="s",
        linewidths=0,
        alpha=0.95,
        zorder=5,
    )
    if not near_regular.empty:
        ax.scatter(
            near_regular["kT"],
            near_regular["JA"],
            s=24,
            facecolors="none",
            edgecolors="#006d77",
            linewidths=0.8,
            alpha=0.92,
            zorder=6,
        )
    if not near_band.empty:
        ax.scatter(
            near_band["kT"],
            near_band["JA"],
            s=28,
            facecolors="none",
            edgecolors="#d81b60",
            linewidths=1.0,
            alpha=0.95,
            zorder=7,
        )

    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Main Phase Boundaries with Nearby New Exact Data")
    ax.set_xlim(0.0, 0.56)
    ax.set_ylim(0.0, 2.12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    legend_items = [
        Patch(facecolor="#f1f1f1", edgecolor="0.55", label="original normal"),
        Patch(facecolor="#54b6d3", edgecolor="0.55", label="original uniform SC"),
        Patch(facecolor="#f28e2b", edgecolor="0.55", label="original FFLO"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black", markersize=5, label="normal/SC boundary"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#5e3c99", markersize=5, label="uniform/FFLO boundary"),
        Line2D([0], [0], marker="o", color="#006d77", markerfacecolor="none", linestyle="none", markersize=5.5, label="new exact near boundary"),
        Line2D([0], [0], marker="o", color="#d81b60", markerfacecolor="none", linestyle="none", markersize=5.5, label="new boundary-band near boundary"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=True, framealpha=0.95)

    png = output_dir / "clean_phase_main_boundaries_new_nearby.png"
    pdf = output_dir / "clean_phase_main_boundaries_new_nearby.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        "output_png": str(png),
        "output_pdf": str(pdf),
        "near_threshold_normalized": float(near_threshold),
        "new_exact_points_total": int(new_points.shape[0]),
        "new_exact_near_main_boundaries": int(near_new.shape[0]),
        "new_boundary_band_near_main_boundaries": int(near_band.shape[0]),
        "normal_sc_boundary_segments": int(normal_sc.shape[0]),
        "uniform_fflo_boundary_segments": int(uniform_fflo.shape[0]),
    }


def plot_enhanced_eta_map(
    warm: dict[str, np.ndarray],
    current_df: pd.DataFrame,
    normal_sc: pd.DataFrame,
    uniform_fflo: pd.DataFrame,
    warm_keys: set[tuple[float, float]],
    output_dir: Path,
    gamma: float,
    coord_decimals: int,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D

    kt_vec = np.asarray(warm["kT_vec"], dtype=np.float64)
    ja_vec = np.asarray(warm["JA_vec"], dtype=np.float64)
    eta = np.asarray(warm["eta_matrix"], dtype=np.float64)
    delta0 = float(np.asarray(warm["delta0"])) if "delta0" in warm else 1.0

    kt = kt_vec / delta0
    ja = ja_vec / delta0
    kt_mask = kt >= 0.0
    kt_plot = kt[kt_mask]
    eta_plot = eta[:, kt_mask]
    eta_enhanced = signed_power(eta_plot, gamma)
    kt_edges = cell_edges_from_centers(kt_plot)
    kt_edges[0] = max(0.0, kt_edges[0])
    ja_edges = cell_edges_from_centers(ja)

    work = current_df.copy()
    work["is_new"] = [
        (round(float(row.kT), coord_decimals), round(float(row.JA), coord_decimals)) not in warm_keys
        for row in work.itertuples(index=False)
    ]
    new_points = work[work["is_new"]].copy()
    new_points["eta_color_value"] = signed_power(new_points["eta"].to_numpy(dtype=np.float64), gamma)
    all_color_values = np.concatenate([eta_enhanced.ravel(), new_points["eta_color_value"].to_numpy(dtype=np.float64)])
    vmax = float(np.nanmax(np.abs(all_color_values[np.isfinite(all_color_values)])))

    cmap = LinearSegmentedColormap.from_list(
        "enhanced_bwr",
        ["#08306b", "#2b8cbe", "#f7fbff", "#fdd0a2", "#b30000"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        kt_edges,
        ja_edges,
        eta_enhanced,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        shading="flat",
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    mesh.set_norm(norm)
    if not new_points.empty:
        ax.scatter(
            new_points["kT"],
            new_points["JA"],
            c=new_points["eta_color_value"],
            cmap=cmap,
            norm=norm,
            s=20,
            marker="o",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.9,
            zorder=5,
            label="new exact eta",
        )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    eta_ticks = np.array([-1.0, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    eta_min = float(np.nanmin(eta_plot))
    eta_max = float(np.nanmax(eta_plot))
    eta_ticks = eta_ticks[(eta_ticks >= eta_min) & (eta_ticks <= eta_max)]
    cbar.set_ticks(signed_power(eta_ticks, gamma))
    cbar.set_ticklabels([f"{x:g}" for x in eta_ticks])
    cbar.set_label(r"$\eta$ (signed-power color scale)")

    _, _, t_1st, ja_1st, t_2nd, ja_2nd = finite_t_phase_boundary_arrays()
    ax.plot(
        t_1st / delta0,
        ja_1st / delta0,
        color="#c1121f",
        linewidth=1.0,
        linestyle=":",
        marker="D",
        markersize=3.2,
        markerfacecolor="white",
        zorder=7,
        label=r"old $c$FFLO-$t$FFLO, 1st",
    )
    ax.plot(
        t_2nd / delta0,
        ja_2nd / delta0,
        color="#2d6a4f",
        linewidth=1.0,
        linestyle="-.",
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        zorder=7,
        label=r"old $c$FFLO-$t$FFLO, 2nd",
    )
    nsc = normal_sc.sort_values(["kT_boundary", "JA_boundary"])
    ax.plot(
        nsc["kT_boundary"],
        nsc["JA_boundary"],
        color="black",
        linewidth=1.3,
        marker="o",
        markersize=2.8,
        markerfacecolor="black",
        markeredgewidth=0.0,
        zorder=8,
        label="active-learning normal/SC",
    )
    ufflo = uniform_fflo.sort_values(["kT_boundary", "JA_boundary"])
    ax.scatter(
        ufflo["kT_boundary"],
        ufflo["JA_boundary"],
        color="#5e3c99",
        s=14,
        marker="s",
        linewidths=0.0,
        zorder=8,
        label="active-learning uniform/FFLO",
    )

    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(r"Diode-Efficiency Map with Active-Learning Exact Data")
    ax.set_xlim(0.0, max(float(kt_edges[-1]), float(new_points["kT"].max()) if not new_points.empty else 0.56))
    ax.set_ylim(0.0, max(2.12 / delta0, float(new_points["JA"].max()) if not new_points.empty else 2.12 / delta0))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    eta_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="black",
        markerfacecolor="white",
        linestyle="none",
        markersize=5.0,
        label="new exact eta",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles = [eta_handle if label == "new exact eta" else handle for handle, label in zip(handles, labels)]
    ax.legend(handles=handles, labels=labels, loc="upper right", frameon=True, framealpha=0.92)

    png = output_dir / "enhanced_eta_active_learning_with_revised_boundaries.png"
    pdf = output_dir / "enhanced_eta_active_learning_with_revised_boundaries.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        "output_png": str(png),
        "output_pdf": str(pdf),
        "eta_min": eta_min,
        "eta_max": eta_max,
        "signed_power_gamma": float(gamma),
        "new_exact_points_colored": int(new_points.shape[0]),
        "active_learning_normal_sc_segments": int(normal_sc.shape[0]),
        "active_learning_uniform_fflo_segments": int(uniform_fflo.shape[0]),
        "color_scale_note": (
            "Background warm-start eta and new exact eta points use the same "
            "sign(eta) * abs(eta)**gamma color scale."
        ),
        "boundary_note": (
            "normal/SC and uniform/FFLO use active-learning extracted boundaries; "
            "old cFFLO-tFFLO reference curves are retained because topology was not revalidated."
        ),
    }


def plot_all_exact_eta_map(
    current_df: pd.DataFrame,
    normal_sc: pd.DataFrame,
    uniform_fflo: pd.DataFrame,
    output_dir: Path,
    gamma: float,
) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.lines import Line2D

    work = current_df.copy()
    work = work[np.isfinite(work["kT"]) & np.isfinite(work["JA"]) & np.isfinite(work["eta"])].copy()
    work["eta_color_value"] = signed_power(work["eta"].to_numpy(dtype=np.float64), gamma)
    finite_color = work["eta_color_value"].to_numpy(dtype=np.float64)
    vmax = float(np.nanmax(np.abs(finite_color[np.isfinite(finite_color)])))
    eta_min = float(np.nanmin(work["eta"].to_numpy(dtype=np.float64)))
    eta_max = float(np.nanmax(work["eta"].to_numpy(dtype=np.float64)))

    cmap = LinearSegmentedColormap.from_list(
        "enhanced_bwr",
        ["#08306b", "#2b8cbe", "#f7fbff", "#fdd0a2", "#b30000"],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    scatter = ax.scatter(
        work["kT"],
        work["JA"],
        c=work["eta_color_value"],
        cmap=cmap,
        norm=norm,
        s=7,
        marker="s",
        linewidths=0.0,
        alpha=0.92,
        rasterized=True,
        zorder=1,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.025)
    eta_ticks = np.array([-1.0, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    eta_ticks = eta_ticks[(eta_ticks >= eta_min) & (eta_ticks <= eta_max)]
    cbar.set_ticks(signed_power(eta_ticks, gamma))
    cbar.set_ticklabels([f"{x:g}" for x in eta_ticks])
    cbar.set_label(r"$\eta$ (signed-power color scale)")

    _, _, t_1st, ja_1st, t_2nd, ja_2nd = finite_t_phase_boundary_arrays()
    ax.plot(
        t_1st,
        ja_1st,
        color="#c1121f",
        linewidth=1.0,
        linestyle=":",
        marker="D",
        markersize=3.0,
        markerfacecolor="white",
        zorder=6,
        label=r"old $c$FFLO-$t$FFLO, 1st",
    )
    ax.plot(
        t_2nd,
        ja_2nd,
        color="#2d6a4f",
        linewidth=1.0,
        linestyle="-.",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        zorder=6,
        label=r"old $c$FFLO-$t$FFLO, 2nd",
    )
    nsc = normal_sc.sort_values(["kT_boundary", "JA_boundary"])
    ax.plot(
        nsc["kT_boundary"],
        nsc["JA_boundary"],
        color="black",
        linewidth=1.25,
        marker="o",
        markersize=2.5,
        markerfacecolor="black",
        markeredgewidth=0.0,
        zorder=7,
        label="active-learning normal/SC",
    )
    ufflo = uniform_fflo.sort_values(["kT_boundary", "JA_boundary"])
    ax.scatter(
        ufflo["kT_boundary"],
        ufflo["JA_boundary"],
        color="#5e3c99",
        s=14,
        marker="s",
        linewidths=0.0,
        zorder=7,
        label="active-learning uniform/FFLO",
    )

    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(r"All Exact Diode-Efficiency Data with Revised Boundaries")
    ax.set_xlim(0.0, max(0.56, float(work["kT"].max())))
    ax.set_ylim(0.0, max(2.12, float(work["JA"].max())))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    point_handle = Line2D(
        [0],
        [0],
        marker="s",
        color="0.25",
        markerfacecolor="0.75",
        linestyle="none",
        markersize=4.5,
        label="all exact eta data",
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=[point_handle, *handles], loc="upper right", frameon=True, framealpha=0.92)

    png = output_dir / "all_exact_eta_with_revised_boundaries.png"
    pdf = output_dir / "all_exact_eta_with_revised_boundaries.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    return {
        "output_png": str(png),
        "output_pdf": str(pdf),
        "exact_points_colored": int(work.shape[0]),
        "eta_min": eta_min,
        "eta_max": eta_max,
        "signed_power_gamma": float(gamma),
        "active_learning_normal_sc_segments": int(normal_sc.shape[0]),
        "active_learning_uniform_fflo_segments": int(uniform_fflo.shape[0]),
        "boundary_note": (
            "All exact points are colored by eta without distinguishing new points. "
            "normal/SC and uniform/FFLO use active-learning extracted boundaries; "
            "old cFFLO-tFFLO reference curves are retained because topology was not revalidated."
        ),
    }


def build_figures(args: argparse.Namespace) -> dict:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    warm = load_warm_start_npz(args.warm_start)
    current = load_current_frame(args.current_dataset)
    normal_sc, uniform_fflo = load_boundary_segments(args.boundary_dir)
    kt_vec = np.asarray(warm["kT_vec"], dtype=np.float64)
    ja_vec = np.asarray(warm["JA_vec"], dtype=np.float64)
    kt_mesh, ja_mesh = np.meshgrid(kt_vec, ja_vec, indexing="xy")
    warm_keys = rounded_coord_keys(np.stack([kt_mesh.ravel(), ja_mesh.ravel()], axis=1), args.coord_decimals)

    phase_summary = plot_clean_phase_map(
        warm=warm,
        current_df=current,
        normal_sc=normal_sc,
        uniform_fflo=uniform_fflo,
        output_dir=output_dir,
        near_threshold=args.near_boundary_distance,
        coord_decimals=args.coord_decimals,
    )
    eta_summary = plot_enhanced_eta_map(
        warm=warm,
        current_df=current,
        normal_sc=normal_sc,
        uniform_fflo=uniform_fflo,
        warm_keys=warm_keys,
        output_dir=output_dir,
        gamma=args.eta_gamma,
        coord_decimals=args.coord_decimals,
    )
    all_exact_eta_summary = plot_all_exact_eta_map(
        current_df=current,
        normal_sc=normal_sc,
        uniform_fflo=uniform_fflo,
        output_dir=output_dir,
        gamma=args.eta_gamma,
    )
    summary = {
        "warm_start": str(args.warm_start),
        "current_dataset": str(args.current_dataset),
        "boundary_dir": str(args.boundary_dir),
        "git_commit": git_commit_or_none(),
        "phase_map": phase_summary,
        "enhanced_eta_map": eta_summary,
        "all_exact_eta_map": all_exact_eta_summary,
    }
    (output_dir / "clean_phase_eta_maps_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make clean phase-boundary and enhanced eta maps.")
    p.add_argument("--warm-start", type=Path, default=DEFAULT_WARM_START)
    p.add_argument("--current-dataset", type=Path, default=DEFAULT_CURRENT_DATASET)
    p.add_argument("--boundary-dir", type=Path, default=DEFAULT_BOUNDARY_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--near-boundary-distance", type=float, default=0.025)
    p.add_argument("--coord-decimals", type=int, default=8)
    p.add_argument("--eta-gamma", type=float, default=0.45)
    return p.parse_args()


def main() -> None:
    summary = build_figures(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
