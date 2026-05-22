#!/usr/bin/env python3
"""Standalone plotting script for the eta phase diagram."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DEFAULT_INPUT = (
    "eta_phase_diagram_nkt138_nja156_nd400_nq400_nk800_kc8_jc8_dc4_qc100_"
    "kk200_eb10000_fp64_libcusolver_cfg422bd68ce6"
)


def finite_T_phase_boundary_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boundary values copied from finite_T_phase_diagram.m."""
    t_fflo_ns = np.array(
        [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.56],
        dtype=np.float64,
    )
    ja_fflo_ns = np.array(
        [2.12, 1.733, 1.5, 1.32, 1.16, 0.9, 0.78, 0.733, 0.7, 0.667, 0.59, 0.4, 0.178, 0.0],
        dtype=np.float64,
    )
    t_1st = np.array([0.01, 0.04, 0.05], dtype=np.float64)
    ja_1st = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    t_2nd = np.array([0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float64)
    ja_2nd = np.array([0.6, 0.6, 0.62, 0.6277, 0.63, 0.628, 0.617, 0.598, 0.565], dtype=np.float64)
    return t_fflo_ns, ja_fflo_ns, t_1st, ja_1st, t_2nd, ja_2nd


def bwr_cmap_adaptive(cmin: float, cmax: float, n_colors: int = 256) -> np.ndarray:
    """Blue-white-red colormap with white placed at eta=0."""
    if cmin >= 0:
        return np.stack(
            [np.ones(n_colors), np.linspace(1.0, 0.0, n_colors), np.linspace(1.0, 0.0, n_colors)],
            axis=1,
        )
    if cmax <= 0:
        return np.stack(
            [np.linspace(0.0, 1.0, n_colors), np.linspace(0.0, 1.0, n_colors), np.ones(n_colors)],
            axis=1,
        )

    zero_ratio = abs(cmin) / (cmax - cmin)
    n_blue = max(1, round(n_colors * zero_ratio))
    n_red = max(1, n_colors - n_blue)
    blue = np.stack(
        [np.linspace(0.0, 1.0, n_blue), np.linspace(0.0, 1.0, n_blue), np.ones(n_blue)],
        axis=1,
    )
    red = np.stack(
        [
            np.ones(n_red),
            np.linspace(1.0 - 1.0 / n_red, 0.0, n_red),
            np.linspace(1.0 - 1.0 / n_red, 0.0, n_red),
        ],
        axis=1,
    )
    return np.vstack([blue, red])


def cell_edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """Return pcolormesh edges for possibly non-uniform cell centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("Grid center array must be one-dimensional with at least two points.")

    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def configure_matplotlib(use_tex: bool) -> None:
    """Use LaTeX typography when available, with a Computer Modern fallback."""
    if use_tex and shutil.which("latex") is None:
        raise RuntimeError("LaTeX was requested, but 'latex' was not found on PATH.")

    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_input_npz(input_path: Path) -> Path:
    """Accept either a .npz file or a result directory."""
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be a .npz file: {input_path}")
        return input_path

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    candidates = sorted(input_path.glob("eta_phase_diagram_*.npz"))
    candidates = [path for path in candidates if "shard" not in path.name]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = "\n".join(str(path) for path in candidates)
        raise RuntimeError(f"Multiple phase-diagram .npz files found. Pass one explicitly:\n{names}")

    nested = sorted(input_path.glob("*/eta_phase_diagram_*.npz"))
    nested = [path for path in nested if "shard" not in path.name]
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        names = "\n".join(str(path) for path in nested)
        raise RuntimeError(f"Multiple nested phase-diagram .npz files found. Pass one explicitly:\n{names}")

    raise FileNotFoundError(f"No eta_phase_diagram_*.npz file found under: {input_path}")


def positive_kt_view(kt: np.ndarray, eta_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop any kT centers below zero."""
    mask = kt >= 0.0
    if not np.any(mask):
        raise ValueError("No kT >= 0 points remain after clipping.")
    return kt[mask], eta_matrix[:, mask]


def output_stem(npz_path: Path, suffix: str) -> Path:
    stem = npz_path.stem
    if stem.startswith("eta_phase_diagram_"):
        stem = stem.removeprefix("eta_phase_diagram_")
    return npz_path.with_name(f"eta_phase_diagram_{stem}{suffix}")


def plot_boundaries(ax: plt.Axes, delta0: float, linewidth: float, markersize: float) -> None:
    t_fflo_ns, ja_fflo_ns, t_1st, ja_1st, t_2nd, ja_2nd = finite_T_phase_boundary_arrays()
    ax.plot(
        t_fflo_ns / delta0,
        ja_fflo_ns / delta0,
        color="black",
        linestyle="-",
        marker="s",
        markersize=markersize,
        markerfacecolor="white",
        markeredgewidth=linewidth,
        linewidth=linewidth,
        label=r"$t$FFLO-normal",
        zorder=3,
    )
    ax.plot(
        t_1st / delta0,
        ja_1st / delta0,
        color="red",
        linestyle=":",
        marker="D",
        markersize=markersize,
        markerfacecolor="white",
        markeredgewidth=linewidth,
        linewidth=linewidth,
        label=r"$c$FFLO-$t$FFLO, 1st",
        zorder=3,
    )
    ax.plot(
        t_2nd / delta0,
        ja_2nd / delta0,
        color="green",
        linestyle="-.",
        marker="o",
        markersize=markersize,
        markerfacecolor="white",
        markeredgewidth=linewidth,
        linewidth=linewidth,
        label=r"$c$FFLO-$t$FFLO, 2nd",
        zorder=3,
    )


def boundary_plot_limits(delta0: float) -> Tuple[float, float, float]:
    t_fflo_ns, ja_fflo_ns, t_1st, ja_1st, t_2nd, ja_2nd = finite_T_phase_boundary_arrays()
    t_max = max(float(np.max(t_fflo_ns)), float(np.max(t_1st)), float(np.max(t_2nd))) / delta0
    ja_min = min(float(np.min(ja_fflo_ns)), float(np.min(ja_1st)), float(np.min(ja_2nd))) / delta0
    ja_max = max(float(np.max(ja_fflo_ns)), float(np.max(ja_1st)), float(np.max(ja_2nd))) / delta0
    return t_max, ja_min, ja_max


def plot_eta_phase_diagram(
    npz_path: Path,
    output_prefix: Path,
    use_tex: bool,
    boundary_linewidth: float,
    boundary_markersize: float,
    dpi: int,
    formats: Iterable[str],
) -> None:
    configure_matplotlib(use_tex)

    with np.load(npz_path) as data:
        kt_vec = np.asarray(data["kT_vec"], dtype=np.float64)
        ja_vec = np.asarray(data["JA_vec"], dtype=np.float64)
        eta_matrix = np.asarray(data["eta_matrix"], dtype=np.float64)
        delta0 = float(np.asarray(data["delta0"])) if "delta0" in data.files else 1.0

    kt_plot, eta_plot = positive_kt_view(kt_vec / delta0, eta_matrix)
    ja_plot = ja_vec / delta0
    kt_edges = cell_edges_from_centers(kt_plot)
    ja_edges = cell_edges_from_centers(ja_plot)
    kt_edges[0] = max(0.0, kt_edges[0])

    min_eta = float(np.nanmin(eta_plot))
    max_eta = float(np.nanmax(eta_plot))
    if min_eta == 0.0 and max_eta == 0.0:
        min_eta, max_eta = -1e-6, 1e-6

    cmap = ListedColormap(bwr_cmap_adaptive(min_eta, max_eta, 256))

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        kt_edges,
        ja_edges,
        eta_plot,
        cmap=cmap,
        vmin=min_eta,
        vmax=max_eta,
        shading="flat",
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.025)
    cbar.set_label(r"$\eta$")

    ax.set_xlabel(r"$k_{\mathrm{B}}T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(r"Superconducting diode efficiency $\eta$")
    boundary_t_max, boundary_ja_min, boundary_ja_max = boundary_plot_limits(delta0)
    x_max = max(float(kt_edges[-1]), boundary_t_max)
    y_min = min(float(ja_edges[0]), boundary_ja_min)
    y_max = max(float(ja_edges[-1]), boundary_ja_max)
    y_pad = 0.025 * (y_max - y_min)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(top=False, right=False)

    plot_boundaries(ax, delta0, boundary_linewidth, boundary_markersize)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        output_path = output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"format": fmt}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(output_path, **save_kwargs)

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot eta phase-diagram data from a saved .npz file.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help="Input result directory or eta_phase_diagram_*.npz file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without suffix. Defaults to the input .npz name plus '_replot'.",
    )
    parser.add_argument("--suffix", default="_replot", help="Suffix used when --output-prefix is not set.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], help="Output formats.")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI.")
    parser.add_argument("--boundary-linewidth", type=float, default=0.8, help="Phase-boundary line width.")
    parser.add_argument("--boundary-markersize", type=float, default=2.8, help="Phase-boundary marker size.")
    parser.add_argument(
        "--no-usetex",
        action="store_true",
        help="Use Computer Modern mathtext fallback instead of external LaTeX.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_input_npz(args.input)
    out_prefix = args.output_prefix if args.output_prefix is not None else output_stem(npz_path, args.suffix)
    plot_eta_phase_diagram(
        npz_path=npz_path,
        output_prefix=out_prefix,
        use_tex=not args.no_usetex,
        boundary_linewidth=args.boundary_linewidth,
        boundary_markersize=args.boundary_markersize,
        dpi=args.dpi,
        formats=args.formats,
    )
    outputs = ", ".join(str(out_prefix.with_suffix(f".{fmt.lower().lstrip('.')}")) for fmt in args.formats)
    print(f"Wrote: {outputs}")


if __name__ == "__main__":
    main()
