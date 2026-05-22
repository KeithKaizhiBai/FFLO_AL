from __future__ import annotations

import argparse
import base64
import json
import shutil
from pathlib import Path
from typing import Dict

import numpy as np


def _load_dataset_npz_or_csv(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            raise
        import pandas as pd

        df = pd.read_csv(csv_path)
        required = {"kT", "JA", "delta_opt", "q_opt"}
        if not required.issubset(df.columns):
            raise
        return {
            "x": df[["kT", "JA"]].to_numpy(dtype=np.float64),
            "y_reg": df[["delta_opt", "q_opt"]].to_numpy(dtype=np.float64),
        }


def _load_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float | int | str | None) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "N/A"
        return f"{float(v):.6g}"
    return str(v)


def _safe_float(v: object) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fmt_dict(raw: dict | None) -> str:
    if not raw:
        return "N/A"
    return ", ".join(f"{k}: {_fmt(v)}" for k, v in raw.items())


def _fmt_boundary_displacement(raw: dict | None) -> str:
    if not raw:
        return "N/A"
    by_type = raw.get("by_boundary_type", {})
    if not by_type:
        return "N/A"
    parts = []
    for boundary_type, row in by_type.items():
        parts.append(
            f"{boundary_type}: median {_fmt(row.get('median_displacement_normalized'))}, "
            f"max {_fmt(row.get('max_displacement_normalized'))}"
        )
    return "; ".join(parts)


def _fmt_bool(v: object) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return _fmt(v)


def _fmt_condition_summary(raw: dict | None) -> str:
    if not raw:
        return "N/A"
    labels = {
        "C1_phase_map_change": "phase map",
        "C2_boundary_shift_normal_sc": "normal/SC shift",
        "C3_boundary_shift_uniform_fflo": "uniform/FFLO shift",
        "C4_label_surprise_rate": "label surprise",
        "C5_selected_A0_ratio": "selected A0 ratio",
        "C5_boundary_coverage_p95": "boundary coverage p95",
        "C6_qedge_and_rerun_rates": "q-edge/rerun rates",
        "C7_boundary_coverage_p95": "boundary coverage p95",
    }
    parts = []
    for key in sorted(raw):
        parts.append(f"{labels.get(key, key)}: {_fmt_bool(raw.get(key))}")
    return "; ".join(parts)


def _fmt_stop_metric_summary(raw: dict | None) -> str:
    if not raw:
        return "N/A"
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else raw
    keys = [
        "phase_map_change",
        "boundary_shift_normal_sc",
        "boundary_shift_uniform_fflo",
        "label_surprise_rate",
        "selected_A0_ratio",
        "q_edge_trigger_rate",
        "rerun_required_rate",
        "boundary_coverage_p95",
    ]
    labels = {
        "phase_map_change": "phase map change",
        "boundary_shift_normal_sc": "normal/SC shift",
        "boundary_shift_uniform_fflo": "uniform/FFLO shift",
        "label_surprise_rate": "label surprise",
        "selected_A0_ratio": "selected A0 ratio",
        "q_edge_trigger_rate": "q-edge trigger",
        "rerun_required_rate": "rerun required",
        "boundary_coverage_p95": "boundary coverage p95",
    }
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{labels.get(key, key)}: {_fmt(metrics.get(key))}")
    return ", ".join(parts) if parts else "N/A"


def _fmt_table_value(v: object) -> str:
    return _tex_escape(_fmt(v))


def _fmt_acq_weight_summary(cfg_raw: dict) -> str:
    def math_text(value: object) -> str:
        return r"\texttt{" + _fmt(value).replace("_", r"\_") + "}"

    w_ext = _fmt(cfg_raw.get("w_extrapolation"))
    if str(cfg_raw.get("w_ext_schedule", "constant")) == "piecewise":
        w_ext = (
            f"{_fmt(cfg_raw.get('w_ext_start'))}"
            f"\\rightarrow{_fmt(cfg_raw.get('w_ext_mid'))}"
            f"\\rightarrow{_fmt(cfg_raw.get('w_ext_end'))}"
        )
    return (
        r"\begin{aligned}"
        + f"w_{{\\mathrm{{cls}}}} &= {_fmt(cfg_raw.get('w_cls_mix'))}, "
        + f"w_{{\\mathrm{{reg}}}} &= {_fmt(cfg_raw.get('w_reg_phase'))}, "
        + f"w_\\Delta &= {_fmt(cfg_raw.get('w_delta_boundary'))}, "
        + f"w_q &= {_fmt(cfg_raw.get('w_q_boundary_sc'))} \\\\ "
        + f"w_g &= {_fmt(cfg_raw.get('w_gradient_phase'))}, "
        + f"w_{{\\mathrm{{edge}}}} &= {_fmt(cfg_raw.get('w_q_edge_risk'))}, "
        + f"w_{{\\mathrm{{ext}}}} &= {w_ext}, "
        + f"w_\\eta &= {_fmt(cfg_raw.get('w_eta_response'))} \\\\ "
        + f"w_{{\\nabla \\eta}} &= {_fmt(cfg_raw.get('w_gradient_response'))}, "
        + f"w_{{\\mathrm{{resp}}}} &= {_fmt(cfg_raw.get('w_reg_response'))}, "
        + f"w_{{\\mathrm{{ent}}}} &= {_fmt(cfg_raw.get('w_cls_entropy_inner'))}, "
        + f"w_{{\\mathrm{{mar}}}} &= {_fmt(cfg_raw.get('w_cls_margin_inner'))} \\\\ "
        + f"B_\\Delta\\mathrm{{\\ gate}} &= {math_text(cfg_raw.get('b_delta_gate_mode'))}, "
        + f"B_q\\mathrm{{\\ gate}} &= {math_text(cfg_raw.get('q_boundary_gate_mode'))}, "
        + f"\\mathrm{{pool}} &= {math_text(cfg_raw.get('active_pool_rule'))}"
        + r"\end{aligned}"
    )


def _tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _latest_iter_dir(run_dir: Path) -> Path | None:
    iters = sorted([p for p in run_dir.glob("iter*") if p.is_dir()])
    return iters[-1] if iters else None


def _latest_dataset_path(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for p in run_dir.glob("dataset_iter*.npz"):
        stem = p.stem
        try:
            idx = int(stem.replace("dataset_iter", ""))
        except ValueError:
            continue
        candidates.append((idx, p))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _dataset_iter_index(path: Path | None) -> int | None:
    if path is None:
        return None
    try:
        return int(path.stem.replace("dataset_iter", ""))
    except ValueError:
        return None


def _dataset_iter_path(run_dir: Path, idx: int) -> Path:
    return run_dir / f"dataset_iter{idx:03d}.npz"


def _copy_if_exists(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    try:
        from PIL import Image

        with Image.open(dst) as im:
            if im.mode not in {"RGB", "L"}:
                im.convert("RGB").save(dst)
    except Exception:
        pass
    return dst.name


def _placeholder_png(dst: Path) -> str:
    """Write a tiny valid PNG so LaTeX can compile even when an optional plot is absent."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    png_8x8_rgb = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFUlEQVR4nGP8"
        "//8/AzbAhFV00EoAAFbUAw037MyjAAAAAElFTkSuQmCC"
    )
    dst.write_bytes(base64.b64decode(png_8x8_rgb))
    return dst.name


def _selection_description(selection_mode: str, cfg_raw: dict) -> str:
    sampling_power = _fmt(cfg_raw.get("sampling_power", 1.0))
    if str(cfg_raw.get("sampling_power_schedule", "constant")) == "piecewise":
        sampling_power = (
            f"{_fmt(cfg_raw.get('sampling_power_start'))}"
            f"\\rightarrow{_fmt(cfg_raw.get('sampling_power_mid'))}"
            f"\\rightarrow{_fmt(cfg_raw.get('sampling_power_end'))}"
        )
    if selection_mode == "stochastic":
        return rf"""
The stochastic acquisition sampler converts the corrected score into sampling
weights after the high-information pool has first been gated by
\(A_{{0,\mathrm{{pool}}}}\).  For each candidate \(x_i\) inside that pool,
\[
A_{{\mathrm{{select}}}}(x_i)
= A_{{0,\mathrm{{pool}}}}(x_i)\,R_{{\mathrm{{obs}}}}(x_i)\,R_{{\mathrm{{batch}}}}(x_i),
\]
and the sampling probability is
\[
p_i =
\frac{{[\max(A_{{\mathrm{{select}}}}(x_i),0)]^\gamma}}
{{\sum_j [\max(A_{{\mathrm{{select}}}}(x_j),0)]^\gamma}},
\qquad \gamma = {sampling_power}.
\]
After each point is sampled, \(R_{{\mathrm{{batch}}}}\) is updated around that
new point and the probabilities are renormalized before the next draw.  Thus
\(A_{{\mathrm{{select}}}}\) is the sampling-weight base, not a greedy
top-\(k\) score.
"""
    return r"""
The deterministic top-\(k\) selector ranks candidates by the corrected score
\[
A_{\mathrm{select}}(x)
= A_{0,\mathrm{main}}(x)\,R_{\mathrm{obs}}(x)\,R_{\mathrm{batch}}(x),
\]
where \(R_{\mathrm{obs}}\) is the soft repulsion from previously computed exact
points and \(R_{\mathrm{batch}}\) is the within-batch repulsion applied while
constructing the selected batch.
"""


def _b_delta_gate_description(cfg_raw: dict) -> str:
    mode = str(cfg_raw.get("b_delta_gate_mode", "not_recorded"))
    if mode == "normal_sc_competition":
        return r"""
For the normal/SC transition score the raw \(\Delta\)-proximity factor is gated
by phase competition:
\[
\begin{aligned}
B_{\Delta,\mathrm{raw}}(x)
&=\exp\!\left[-\frac{|\Delta_{\mathrm{pred}}(x)-\Delta_\epsilon|}
{\Delta_{\mathrm{scale}}}\right],\\
U_{\mathrm{NS}}(x)&=4P_{\mathrm{normal}}(x)P_{\mathrm{SC}}(x),\\
B_{\Delta,\mathrm{gated}}(x)
&=B_{\Delta,\mathrm{raw}}(x)U_{\mathrm{NS}}(x).
\end{aligned}
\]
The \(B_\Delta\) term in \(A_{\mathrm{phase}}\) denotes this gated quantity.
This prevents deep predicted-normal regions with
\(\Delta_{\mathrm{pred}}\approx 0\) from being treated as normal/SC boundary
points solely because \(\Delta_\epsilon\) is small.
"""
    return r"""
This run configuration did not record the normal/SC competition gate for
\(B_\Delta\).  Historical runs may therefore have used the older raw
\(\Delta\)-proximity score.  New discovery runs write
\(B_{\Delta,\mathrm{raw}}\), \(U_{\mathrm{NS}}\), and
\(B_{\Delta,\mathrm{gated}}\) separately so the report can distinguish the raw
and gated contributions.
"""


def _figure_unavailable_block(reason: str) -> str:
    return (
        r"\begin{center}" "\n"
        r"\fbox{\begin{minipage}{0.86\textwidth}" "\n"
        r"\textbf{Figure unavailable:} "
        + _tex_escape(reason)
        + "\n"
        r"\end{minipage}}" "\n"
        r"\end{center}"
    )


def _include_single_figure_block(filename: str, width: str, caption: str) -> str:
    return (
        r"\begin{figure}[H]" "\n"
        r"\centering" "\n"
        + rf"\includegraphics[width={width}]{{{filename}}}" "\n"
        + rf"\caption{{{caption}}}" "\n"
        r"\end{figure}"
    )


def _build_exact_phase_map_block(figures_dir: Path, run_id: str, latest_dataset: Path | None) -> str:
    pdf = figures_dir / f"{run_id}_exact_phase_map.pdf"
    png = figures_dir / f"{run_id}_exact_phase_map.png"
    figure = pdf if pdf.exists() else png
    if not figure.exists():
        return ""
    dataset_name = latest_dataset.name if latest_dataset is not None else "the latest dataset"
    caption = (
        "Exact-data phase map after the discovery active-learning loop.  "
        rf"The color of each point is assigned from the exact BdG label in \texttt{{{_tex_escape(dataset_name)}}}: "
        "normal, uniform superconducting, or FFLO.  Boundary markers show the extracted main thermodynamic "
        "boundaries from the final accepted exact dataset."
    )
    return _include_single_figure_block(figure.name, r"0.82\textwidth", caption)


def _finite_t_reference_boundaries() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_1st = np.array([0.01, 0.04, 0.05], dtype=np.float64)
    ja_1st = np.array([0.6, 0.6, 0.6], dtype=np.float64)
    t_2nd = np.array([0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float64)
    ja_2nd = np.array([0.6, 0.6, 0.62, 0.6277, 0.63, 0.628, 0.617, 0.598, 0.565], dtype=np.float64)
    return t_1st, ja_1st, t_2nd, ja_2nd


def _signed_power(values: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.power(np.abs(values), gamma)


def _load_boundary_csv(path: Path):
    import pandas as pd

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    df = pd.read_csv(path)
    if "kT_boundary" not in df.columns or "JA_boundary" not in df.columns:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    return df[np.isfinite(df["kT_boundary"]) & np.isfinite(df["JA_boundary"])].copy()


def _ensure_final_boundary_dir(run_dir: Path, figures_dir: Path, latest_dataset: Path) -> tuple[Path | None, str]:
    boundary_dir = figures_dir / f"{run_dir.name}_final_exact_boundaries"
    summary = boundary_dir / "boundary_summary.json"
    normal_sc = boundary_dir / "normal_sc_boundary_segments.csv"
    uniform_fflo = boundary_dir / "uniform_fflo_boundary_segments.csv"
    if summary.exists() and normal_sc.exists() and uniform_fflo.exists():
        return boundary_dir, ""
    try:
        from .extract_phase_boundaries import extract_phase_boundaries

        args = argparse.Namespace(
            dataset=latest_dataset,
            output_dir=boundary_dir,
            kt_bin_width=0.005,
            max_local_spacing=0.035,
            max_refinement_points=0,
            output_root=figures_dir.parent,
        )
        extract_phase_boundaries(args)
    except Exception as exc:
        return None, f"final boundary extraction failed ({exc})"
    if not summary.exists():
        return None, "final boundary extraction did not write boundary_summary.json"
    return boundary_dir, ""


def _build_exact_eta_revised_boundary_figure(
    run_dir: Path,
    figures_dir: Path,
    run_id: str,
    latest_dataset: Path | None,
) -> tuple[str, str]:
    if latest_dataset is None or not latest_dataset.exists():
        return "", "final dataset was not found"
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.lines import Line2D
    except Exception as exc:
        return "", f"matplotlib is unavailable ({exc})"

    try:
        data = _load_dataset_npz_or_csv(latest_dataset)
    except Exception as exc:
        return "", f"final dataset could not be loaded ({exc})"
    if "x" not in data or "y_reg" not in data:
        return "", "final dataset does not contain x and y_reg arrays"
    x = np.asarray(data["x"], dtype=np.float64)
    y_reg = np.asarray(data["y_reg"], dtype=np.float64)
    if y_reg.ndim != 2 or y_reg.shape[1] < 3:
        return "", "final dataset does not contain eta as y_reg[:, 2]"
    eta = y_reg[:, 2]
    finite = np.isfinite(x[:, 0]) & np.isfinite(x[:, 1]) & np.isfinite(eta) & (x[:, 0] >= 0.0)
    if not np.any(finite):
        return "", "final dataset has no finite eta points with kT >= 0"
    x = x[finite]
    eta = eta[finite]

    boundary_dir, reason = _ensure_final_boundary_dir(run_dir, figures_dir, latest_dataset)
    if boundary_dir is None:
        return "", reason
    normal_sc = _load_boundary_csv(boundary_dir / "normal_sc_boundary_segments.csv")
    uniform_fflo = _load_boundary_csv(boundary_dir / "uniform_fflo_boundary_segments.csv")

    gamma = 0.35
    eta_color = _signed_power(eta, gamma)
    finite_eta = eta[np.isfinite(eta)]
    max_abs_eta = float(np.nanmax(np.abs(finite_eta))) if finite_eta.size else 1.0
    max_abs_eta = max(max_abs_eta, 1e-12)
    color_limit = max(0.5, float(np.power(max_abs_eta, gamma)))

    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{run_id}_exact_eta_revised_boundaries.png"
    pdf = figures_dir / f"{run_id}_exact_eta_revised_boundaries.pdf"
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    sc = ax.scatter(
        x[:, 0],
        x[:, 1],
        c=eta_color,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-color_limit, vmax=color_limit),
        marker="s",
        s=11,
        linewidths=0.0,
        alpha=0.82,
        rasterized=True,
        label="all exact eta data",
    )

    t_1st, ja_1st, t_2nd, ja_2nd = _finite_t_reference_boundaries()
    ax.plot(
        t_1st,
        ja_1st,
        color="#d0001f",
        linewidth=1.0,
        linestyle=":",
        marker="D",
        markersize=4.0,
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
    if not normal_sc.empty:
        nsc = normal_sc.sort_values(["kT_boundary", "JA_boundary"])
        ax.plot(
            nsc["kT_boundary"],
            nsc["JA_boundary"],
            color="black",
            linewidth=1.25,
            marker="o",
            markersize=2.2,
            markerfacecolor="black",
            markeredgewidth=0.0,
            zorder=7,
            label="active-learning normal/SC",
        )
    if not uniform_fflo.empty:
        ufflo = uniform_fflo.sort_values(["kT_boundary", "JA_boundary"])
        ax.scatter(
            ufflo["kT_boundary"],
            ufflo["JA_boundary"],
            color="#5e3c99",
            s=12,
            marker="s",
            linewidths=0.0,
            zorder=7,
            label="active-learning uniform/FFLO",
        )

    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(r"Final Exact Diode-Efficiency Data with Revised Boundaries")
    ax.set_xlim(0.0, max(0.56, float(np.nanmax(x[:, 0]))))
    ax.set_ylim(0.0, max(2.12, float(np.nanmax(x[:, 1]))))
    for spine in ax.spines.values():
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
    handles, _ = ax.get_legend_handles_labels()
    handles = [point_handle, *[h for h in handles if getattr(h, "get_label", lambda: "")() != "all exact eta data"]]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\eta$ (signed-power color scale)")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)

    summary = {
        "dataset": str(latest_dataset),
        "boundary_dir": str(boundary_dir),
        "output_png": str(png),
        "output_pdf": str(pdf),
        "exact_points_colored": int(x.shape[0]),
        "eta_min": float(np.nanmin(eta)),
        "eta_max": float(np.nanmax(eta)),
        "signed_power_gamma": gamma,
        "active_learning_normal_sc_segments": int(normal_sc.shape[0]),
        "active_learning_uniform_fflo_segments": int(uniform_fflo.shape[0]),
        "boundary_note": (
            "normal/SC and uniform/FFLO use boundaries re-extracted from the final exact dataset; "
            "old cFFLO-tFFLO reference curves are retained because topology is not revalidated by the current pointwise oracle."
        ),
    }
    png.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return pdf.name if pdf.exists() else png.name, ""


def _build_exact_eta_revised_boundary_block(run_dir: Path, figures_dir: Path, run_id: str, latest_dataset: Path | None) -> str:
    existing_pdf = figures_dir / f"{run_id}_exact_eta_revised_boundaries.pdf"
    existing_png = figures_dir / f"{run_id}_exact_eta_revised_boundaries.png"
    if existing_pdf.exists() and existing_pdf.stat().st_size > 1024:
        filename, reason = existing_pdf.name, ""
    elif existing_png.exists() and existing_png.stat().st_size > 1024:
        filename, reason = existing_png.name, ""
    else:
        filename, reason = _build_exact_eta_revised_boundary_figure(run_dir, figures_dir, run_id, latest_dataset)
    if filename:
        return _include_single_figure_block(
            filename,
            r"0.84\textwidth",
            "Final exact diode-efficiency data colored by \\(\\eta\\) on a signed-power color scale. "
            "The normal/SC and uniform-SC/FFLO boundaries are re-extracted from the final exact dataset. "
            "The old \\(c\\)FFLO-\\(t\\)FFLO curves are retained only as topology-reference curves; "
            "the current pointwise exact oracle does not revalidate topology.",
        )
    return _figure_unavailable_block(reason or "final exact eta map could not be generated")


def _build_cumulative_progress_figure(run_dir: Path, figures_dir: Path, run_id: str) -> tuple[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return "", f"matplotlib is unavailable ({exc})"

    rows: list[dict[str, float]] = []
    for iter_dir in sorted([p for p in run_dir.glob("iter*") if p.is_dir()]):
        try:
            idx = int(iter_dir.name.replace("iter", ""))
        except ValueError:
            continue
        selected_path = iter_dir / "selected_points.csv"
        selected = 0
        if selected_path.exists():
            with selected_path.open("r", encoding="utf-8", errors="ignore") as f:
                selected = max(0, sum(1 for _ in f) - 1)
        merge = _load_json(iter_dir / f"merge_summary_iter{idx:03d}.json", default={})
        dataset_path = _dataset_iter_path(run_dir, idx + 1)
        dataset_samples = np.nan
        if dataset_path.exists():
            dataset_samples = float(_load_dataset_npz_or_csv(dataset_path)["x"].shape[0])
        rows.append(
            {
                "iter": float(idx),
                "selected": float(selected),
                "merged": _safe_float(merge.get("merged_points")),
                "training": _safe_float(merge.get("training_eligible_points")),
                "rerun": _safe_float(merge.get("rerun_required_points")),
                "dataset": dataset_samples,
            }
        )

    if not rows:
        return "", "no iteration directories were found"

    it = np.array([r["iter"] for r in rows], dtype=np.float64)
    selected = np.array([r["selected"] for r in rows], dtype=np.float64)
    merged = np.array([r["merged"] for r in rows], dtype=np.float64)
    training = np.array([r["training"] for r in rows], dtype=np.float64)
    rerun = np.array([r["rerun"] for r in rows], dtype=np.float64)
    dataset = np.array([r["dataset"] for r in rows], dtype=np.float64)

    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{run_id}_cumulative_progress.png"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True, constrained_layout=True)
    ax1.plot(it, selected, marker="o", ms=3.0, label="selected")
    ax1.plot(it, merged, marker="s", ms=3.0, label="merged exact")
    ax1.plot(it, training, marker="^", ms=3.0, label="training eligible")
    ax1.plot(it, rerun, marker="x", ms=3.0, label="rerun required")
    ax1.set_ylabel("points per iteration")
    ax1.grid(alpha=0.28)
    ax1.legend(loc="best", fontsize=8)

    ax2.plot(it, dataset, marker="o", ms=3.0, color="tab:purple", label="dataset samples")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("cumulative exact samples")
    ax2.grid(alpha=0.28)
    ax2.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out.name, ""


def _build_cumulative_figure_block(run_dir: Path, figures_dir: Path, run_id: str) -> str:
    existing = figures_dir / f"{run_id}_cumulative_progress.png"
    if existing.exists() and existing.stat().st_size > 1024:
        filename, reason = existing.name, ""
    else:
        filename, reason = _build_cumulative_progress_figure(run_dir, figures_dir, run_id)
    if filename:
        return _include_single_figure_block(
            filename,
            r"0.82\textwidth",
            "Cumulative active-learning selected points, merged exact outputs, training-eligible exact points, rerun-required points, and dataset samples across all recorded iterations.",
        )
    return _figure_unavailable_block(reason or "cumulative progress data could not be generated")


def _build_selection_source_map_block(run_dir: Path, figures_dir: Path, run_id: str) -> str:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        return _figure_unavailable_block(f"selection-source map could not be generated ({exc})")

    rows: list[pd.DataFrame] = []
    for iter_dir in sorted([p for p in run_dir.glob("iter*") if p.is_dir()]):
        p = iter_dir / "selected_points_by_pool.csv"
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if {"kT", "JA", "selection_source"}.issubset(df.columns):
            rows.append(df[["kT", "JA", "selection_source"]].copy())
    if not rows:
        return ""
    all_rows = pd.concat(rows, ignore_index=True)
    out = figures_dir / f"{run_id}_selection_sources.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    random_rows = all_rows[all_rows["selection_source"] == "random_seed"]
    acq_rows = all_rows[all_rows["selection_source"] != "random_seed"]
    if not random_rows.empty:
        ax.scatter(random_rows["kT"], random_rows["JA"], s=12, c="#8c8c8c", alpha=0.45, linewidths=0, label="random seed")
    if not acq_rows.empty:
        ax.scatter(acq_rows["kT"], acq_rows["JA"], s=10, c="#1f78b4", alpha=0.35, marker="x", linewidths=0.8, label="acquisition stochastic")
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title("Selected Points by Source")
    ax.set_xlim(0.0, 0.56)
    ax.set_ylim(0.0, 2.12)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return _include_single_figure_block(
        out.name,
        r"0.72\textwidth",
        "Cumulative selected points separated by source. Random initial seeds and later acquisition-stochastic selections are shown with different markers.",
    )


def _build_selection_focus_curve_block(run_dir: Path, figures_dir: Path, run_id: str) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return _figure_unavailable_block(f"selection-focus curve could not be generated ({exc})")

    records: list[dict[str, float]] = []
    for iter_dir in sorted([p for p in run_dir.glob("iter*") if p.is_dir()]):
        try:
            idx = int(iter_dir.name.replace("iter", ""))
        except ValueError:
            continue
        raw = _load_json(iter_dir / "selection_diagnostics.json", default={})
        if not raw:
            continue
        records.append(
            {
                "iter": float(idx),
                "band": _safe_float(raw.get("selected_boundary_band_fraction")),
                "neff": _safe_float(raw.get("N_eff_over_active_pool_size")),
                "a0_ratio": _safe_float(raw.get("selected_A0_main_over_unseen_mean")),
            }
        )
    if not records:
        return ""
    it = np.asarray([r["iter"] for r in records], dtype=np.float64)
    band = np.asarray([r["band"] for r in records], dtype=np.float64)
    neff = np.asarray([r["neff"] for r in records], dtype=np.float64)
    a0_ratio = np.asarray([r["a0_ratio"] for r in records], dtype=np.float64)
    if not (np.any(np.isfinite(band)) or np.any(np.isfinite(neff)) or np.any(np.isfinite(a0_ratio))):
        return ""
    out = figures_dir / f"{run_id}_selection_focus_curve.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    if np.any(np.isfinite(band)):
        ax.plot(it, band, marker="o", label="boundary-band fraction")
    if np.any(np.isfinite(neff)):
        ax.plot(it, neff, marker="s", label=r"$N_{\mathrm{eff}}/N_{\mathrm{pool}}$")
    if np.any(np.isfinite(a0_ratio)):
        ax.plot(it, a0_ratio, marker="^", label=r"selected/unseen $A_{0,\mathrm{main}}$ mean")
    ax.set_xlabel("iteration")
    ax.set_ylabel("diagnostic ratio")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return _include_single_figure_block(
        out.name,
        r"0.72\textwidth",
        "Selection-focus diagnostics. Boundary-band fraction measures how often selected points lie near predicted main phase boundaries, while the effective-sample-size ratio measures whether stochastic sampling is concentrated or nearly uniform.",
    )


def _iter_index(iter_dir: Path) -> int | None:
    try:
        return int(iter_dir.name.replace("iter", ""))
    except ValueError:
        return None


def _normalized_points(points: np.ndarray, cfg_raw: dict) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    kt_min = float(cfg_raw.get("kt_min", 0.0))
    kt_max = float(cfg_raw.get("kt_max", 0.56))
    ja_min = float(cfg_raw.get("ja_min", 0.0))
    ja_max = float(cfg_raw.get("ja_max", 2.12))
    offset = np.array([kt_min, ja_min], dtype=np.float64)
    scale = np.array([max(kt_max - kt_min, 1.0e-12), max(ja_max - ja_min, 1.0e-12)], dtype=np.float64)
    return (pts - offset) / scale


def _main_boundary_points_from_phase(
    grid_points: np.ndarray,
    kt_values: np.ndarray,
    ja_values: np.ndarray,
    full_shape: np.ndarray,
    phase_pred: np.ndarray,
) -> np.ndarray:
    phase = np.asarray(phase_pred, dtype=np.int64).reshape(tuple(np.asarray(full_shape, dtype=np.int64)))
    kt = np.asarray(kt_values, dtype=np.float64)
    ja = np.asarray(ja_values, dtype=np.float64)
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


def _nearest_distance_norm(points: np.ndarray, boundary_points: np.ndarray, cfg_raw: dict, chunk: int = 4096) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    bnd = np.asarray(boundary_points, dtype=np.float64).reshape(-1, 2)
    out = np.full(pts.shape[0], np.inf, dtype=np.float64)
    if pts.size == 0 or bnd.size == 0:
        return out
    pts_n = _normalized_points(pts, cfg_raw)
    bnd_n = _normalized_points(bnd, cfg_raw)
    for start in range(0, pts_n.shape[0], chunk):
        block = pts_n[start : start + chunk]
        diff = block[:, None, :] - bnd_n[None, :, :]
        out[start : start + chunk] = np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)
    return out


def _dense_grid_spacing_norm(monitor: dict[str, np.ndarray], cfg_raw: dict) -> float:
    n_kt = int(np.asarray(monitor.get("kt_values", [])).size or cfg_raw.get("n_kt_candidates", 1))
    n_ja = int(np.asarray(monitor.get("ja_values", [])).size or cfg_raw.get("n_ja_candidates", 1))
    return max(1.0 / max(n_kt - 1, 1), 1.0 / max(n_ja - 1, 1))


def _computed_grid_mask(run_dir: Path, iteration: int, grid_points: np.ndarray, decimals: int = 4) -> np.ndarray:
    dataset = _dataset_iter_path(run_dir, iteration)
    if not dataset.exists():
        return np.zeros(grid_points.shape[0], dtype=bool)
    try:
        data = _load_dataset_npz_or_csv(dataset)
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1, 2)
    except Exception:
        return np.zeros(grid_points.shape[0], dtype=bool)
    computed = {tuple(row) for row in np.round(x, decimals=decimals)}
    rounded = np.round(np.asarray(grid_points, dtype=np.float64).reshape(-1, 2), decimals=decimals)
    return np.asarray([tuple(row) in computed for row in rounded], dtype=bool)


def _region_labels(phase_pred: np.ndarray, dist: np.ndarray, band_width: float) -> np.ndarray:
    phase = np.asarray(phase_pred, dtype=np.int64)
    d = np.asarray(dist, dtype=np.float64)
    labels = np.full(phase.shape[0], "unknown", dtype=object)
    band = np.isfinite(d) & (d <= float(band_width))
    labels[(phase == 0) & band] = "normal_boundary_band"
    labels[np.isin(phase, [1, 2]) & band] = "sc_boundary_band"
    labels[(phase == 0) & ~band] = "normal_interior"
    labels[(phase == 1) & ~band] = "uniform_sc_interior"
    labels[(phase == 2) & ~band] = "fflo_interior"
    return labels


def _region_counts(labels: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    lab = np.asarray(labels, dtype=object)
    m = np.asarray(mask, dtype=bool)
    total = int(np.sum(m))
    keys = {
        "normal_interior": lab == "normal_interior",
        "uniform_sc_interior": lab == "uniform_sc_interior",
        "fflo_interior": lab == "fflo_interior",
        "sc_interior": (lab == "uniform_sc_interior") | (lab == "fflo_interior"),
        "boundary_band": (lab == "normal_boundary_band") | (lab == "sc_boundary_band"),
    }
    out: dict[str, float | int | None] = {"count_total": total}
    for name, region_mask in keys.items():
        count = int(np.sum(m & region_mask))
        out[f"count_{name}"] = count
        out[f"fraction_{name}"] = float(count / total) if total else None
    return out


def _component_stats(values: np.ndarray) -> dict[str, float | None]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
    }


def _monitor_component_arrays(monitor: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = np.asarray(monitor["grid_points"]).shape[0]
    zeros = np.zeros(n, dtype=np.float64)
    return {
        "A0_main": np.asarray(monitor.get("A0_main", zeros), dtype=np.float64),
        "A0_main_raw": np.asarray(monitor.get("A0_main_raw", monitor.get("A0_main", zeros)), dtype=np.float64),
        "A0_for_pool": np.asarray(monitor.get("A0_for_pool", monitor.get("A0_main", zeros)), dtype=np.float64),
        "A_phase": np.asarray(monitor.get("A_phase", zeros), dtype=np.float64),
        "A_numerical": np.asarray(monitor.get("A_numerical", zeros), dtype=np.float64),
        "A_explore": np.asarray(monitor.get("A_explore", zeros), dtype=np.float64),
        "A_response": np.asarray(monitor.get("A_response", zeros), dtype=np.float64),
        "U_cls_mix": np.asarray(monitor.get("cls_uncertainty_mix", zeros), dtype=np.float64),
        "cls_entropy": np.asarray(monitor.get("cls_entropy", zeros), dtype=np.float64),
        "cls_margin_uncertainty": np.asarray(monitor.get("cls_margin_uncertainty", zeros), dtype=np.float64),
        "U_reg_phase": np.asarray(monitor.get("U_reg_phase", zeros), dtype=np.float64),
        "U_delta": np.asarray(monitor.get("U_delta", zeros), dtype=np.float64),
        "U_q": np.asarray(monitor.get("U_q", zeros), dtype=np.float64),
        "B_delta_raw": np.asarray(monitor.get("B_delta_raw", zeros), dtype=np.float64),
        "U_NS": np.asarray(monitor.get("U_NS", zeros), dtype=np.float64),
        "B_delta_gated": np.asarray(monitor.get("B_delta_gated", monitor.get("delta_boundary_score", zeros)), dtype=np.float64),
        "B_delta": np.asarray(monitor.get("delta_boundary_score", monitor.get("B_delta_gated", zeros)), dtype=np.float64),
        "U_UF": np.asarray(monitor.get("U_UF", zeros), dtype=np.float64),
        "B_q_raw": np.asarray(monitor.get("B_q_raw", zeros), dtype=np.float64),
        "B_q_gated": np.asarray(monitor.get("B_q_gated", monitor.get("B_q_SC", zeros)), dtype=np.float64),
        "B_q_SC": np.asarray(monitor.get("B_q_SC", monitor.get("B_q_gated", zeros)), dtype=np.float64),
        "G_phase": np.asarray(monitor.get("gradient_score", zeros), dtype=np.float64),
        "E_q_SC": np.asarray(monitor.get("E_q_SC", zeros), dtype=np.float64),
        "E_ext_uncertain": np.asarray(monitor.get("E_ext_uncertain", zeros), dtype=np.float64),
        "R_obs": np.asarray(monitor.get("R_obs", np.ones(n, dtype=np.float64)), dtype=np.float64),
        "Aselect": np.asarray(monitor.get("Aselect_initial", monitor.get("score", zeros)), dtype=np.float64),
        "interior_penalty": np.asarray(monitor.get("interior_penalty", np.ones(n, dtype=np.float64)), dtype=np.float64),
        "interior_penalty_applied": np.asarray(monitor.get("interior_penalty_applied", zeros), dtype=np.float64),
        "high_confidence_interior": np.asarray(monitor.get("high_confidence_interior", zeros), dtype=np.float64),
        "w_ext_current": np.asarray(monitor.get("w_ext_current", np.full(n, np.nan, dtype=np.float64)), dtype=np.float64),
    }


def _selected_component_arrays(selected) -> dict[str, np.ndarray]:
    import pandas as pd

    if selected is None or selected.empty:
        return {}
    aliases = {
        "U_cls_mix": "cls_uncertainty_mix",
        "B_delta": "delta_boundary_score",
        "G_phase": "gradient_score",
        "E_q_SC": "q_edge_risk_score",
        "E_ext_uncertain": "extrapolation_risk_score",
    }
    keys = [
        "A0_main",
        "A0_main_raw",
        "A0_for_pool",
        "A_phase",
        "A_numerical",
        "A_explore",
        "A_response",
        "U_cls_mix",
        "cls_entropy",
        "cls_margin_uncertainty",
        "U_reg_phase",
        "U_delta",
        "U_q",
        "B_delta_raw",
        "U_NS",
        "B_delta_gated",
        "B_delta",
        "U_UF",
        "B_q_raw",
        "B_q_gated",
        "B_q_SC",
        "G_phase",
        "E_q_SC",
        "E_ext_uncertain",
        "R_obs",
        "R_batch",
        "Aselect",
        "interior_penalty",
        "interior_penalty_applied",
        "high_confidence_interior",
        "sampling_power",
        "w_ext_current",
        "sampling_probability_before_pick",
    ]
    out: dict[str, np.ndarray] = {}
    for key in keys:
        col = key if key in selected.columns else aliases.get(key, "")
        if col and col in selected.columns:
            out[key] = pd.to_numeric(selected[col], errors="coerce").to_numpy(dtype=np.float64)
    return out


def _component_breakdown(
    components: dict[str, np.ndarray],
    labels: np.ndarray,
    mask: np.ndarray,
    group: str,
    regions: tuple[str, ...] = ("normal_interior", "sc_interior", "boundary_band"),
) -> list[dict[str, object]]:
    lab = np.asarray(labels, dtype=object)
    base = np.asarray(mask, dtype=bool)
    region_masks = {
        "normal_interior": lab == "normal_interior",
        "sc_interior": (lab == "uniform_sc_interior") | (lab == "fflo_interior"),
        "boundary_band": (lab == "normal_boundary_band") | (lab == "sc_boundary_band"),
    }
    rows: list[dict[str, object]] = []
    for region in regions:
        m = base & region_masks[region]
        row: dict[str, object] = {"group": group, "region": region, "count": int(np.sum(m))}
        for key, arr in components.items():
            if arr.shape[0] != base.shape[0]:
                continue
            stats = _component_stats(arr[m])
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_median"] = stats["median"]
            row[f"{key}_p90"] = stats["p90"]
            row[f"{key}_p95"] = stats["p95"]
        a0 = row.get("A0_main_mean")
        if isinstance(a0, (float, int, np.floating, np.integer)) and np.isfinite(float(a0)) and abs(float(a0)) > 1.0e-12:
            for key in ("A_phase", "A_numerical", "A_explore"):
                val = row.get(f"{key}_mean")
                if isinstance(val, (float, int, np.floating, np.integer)) and np.isfinite(float(val)):
                    row[f"{key}_over_A0_main"] = float(val) / float(a0)
        rows.append(row)
    return rows


def _write_selection_region_diagnostics_for_iter(
    run_dir: Path,
    iter_dir: Path,
    cfg_raw: dict,
    rng_seed: int,
) -> dict[str, object] | None:
    import pandas as pd

    idx = _iter_index(iter_dir)
    if idx is None:
        return None
    monitor_path = iter_dir / f"monitor_predictions_iter{idx:03d}.npz"
    selected_path = iter_dir / "selected_points_by_pool.csv"
    if not monitor_path.exists() or not selected_path.exists():
        return None

    with np.load(monitor_path, allow_pickle=False) as z:
        monitor = {k: z[k] for k in z.files}
    grid_points = np.asarray(monitor["grid_points"], dtype=np.float64)
    phase_pred = np.asarray(monitor["phase_pred"], dtype=np.int64)
    boundary_points = _main_boundary_points_from_phase(
        grid_points,
        np.asarray(monitor["kt_values"], dtype=np.float64),
        np.asarray(monitor["ja_values"], dtype=np.float64),
        np.asarray(monitor["full_shape"], dtype=np.int64),
        phase_pred,
    )
    band_width = 2.0 * _dense_grid_spacing_norm(monitor, cfg_raw)
    dist = _nearest_distance_norm(grid_points, boundary_points, cfg_raw)
    labels = _region_labels(phase_pred, dist, band_width)

    candidate_mask = np.asarray(monitor.get("candidate_mask", np.ones(grid_points.shape[0])), dtype=bool)
    computed = _computed_grid_mask(run_dir, idx, grid_points, decimals=int(cfg_raw.get("existing_exclusion_decimals", 4)))
    a0 = np.asarray(monitor.get("A0_main", np.full(grid_points.shape[0], np.nan)), dtype=np.float64)
    hard_unseen = candidate_mask & np.isfinite(a0) & ~computed
    active_pool = np.asarray(monitor.get("active_pool_mask", np.zeros(grid_points.shape[0])), dtype=bool) & hard_unseen

    try:
        selected = pd.read_csv(selected_path)
    except Exception:
        selected = pd.DataFrame()
    selected_mask = np.zeros(grid_points.shape[0], dtype=bool)
    selected_indices = np.asarray([], dtype=np.int64)
    if not selected.empty and "grid_index" in selected.columns:
        selected_indices = pd.to_numeric(selected["grid_index"], errors="coerce").dropna().to_numpy(dtype=np.int64)
        selected_indices = selected_indices[(selected_indices >= 0) & (selected_indices < grid_points.shape[0])]
        selected_mask[selected_indices] = True

    rng = np.random.default_rng(int(rng_seed) + idx * 1000003 + 991)
    available_idx = np.flatnonzero(hard_unseen)
    random_mask = np.zeros(grid_points.shape[0], dtype=bool)
    random_indices = np.asarray([], dtype=np.int64)
    if selected_indices.size and available_idx.size:
        n_rand = min(int(selected_indices.size), int(available_idx.size))
        random_indices = rng.choice(available_idx, size=n_rand, replace=False)
        random_mask[random_indices] = True

    groups = {
        "all_unseen_candidates": hard_unseen,
        "active_pool_candidates": active_pool,
        "selected_points": selected_mask,
        "random_baseline": random_mask,
    }
    group_rows: list[dict[str, object]] = []
    for group, mask in groups.items():
        row: dict[str, object] = {"iteration": idx, "group": group}
        row.update(_region_counts(labels, mask))
        if np.any(mask):
            row["A0_main_mean"] = float(np.nanmean(a0[mask]))
            row["boundary_distance_median"] = float(np.nanmedian(dist[mask]))
        else:
            row["A0_main_mean"] = None
            row["boundary_distance_median"] = None
        group_rows.append(row)

    selection_diag = _load_json(iter_dir / "selection_diagnostics.json", default={})
    a0_for_pool = np.asarray(monitor.get("A0_for_pool", a0), dtype=np.float64)
    finite_unseen = a0_for_pool[hard_unseen]
    finite_unseen = finite_unseen[np.isfinite(finite_unseen)]
    p95 = float(np.percentile(finite_unseen, 95)) if finite_unseen.size else np.nan
    q = float(selection_diag.get("active_pool_quantile_used") or cfg_raw.get("active_pool_quantile", 0.9))
    q_threshold = float(np.quantile(finite_unseen, q)) if finite_unseen.size else np.nan
    rel_threshold = float(cfg_raw.get("active_pool_rel_to_p95", 0.0)) * p95 if np.isfinite(p95) else np.nan
    quantile_rule = hard_unseen & np.isfinite(a0_for_pool) & (a0_for_pool >= q_threshold) if np.isfinite(q_threshold) else np.zeros_like(hard_unseen)
    rel_rule = hard_unseen & np.isfinite(a0_for_pool) & (a0_for_pool >= rel_threshold) if np.isfinite(rel_threshold) else np.zeros_like(hard_unseen)
    selected_a0 = a0_for_pool[selected_mask]
    random_a0 = a0_for_pool[random_mask]
    high_conf = np.asarray(monitor.get("high_confidence_interior", np.zeros(grid_points.shape[0])), dtype=bool)
    penalty_applied = np.asarray(monitor.get("interior_penalty_applied", np.zeros(grid_points.shape[0])), dtype=bool)
    active_summary = {
        "iteration": idx,
        "boundary_band_width_norm": float(band_width),
        "boundary_point_count": int(boundary_points.shape[0]),
        "unseen_count": int(np.sum(hard_unseen)),
        "active_pool_count": int(np.sum(active_pool)),
        "active_pool_fraction": float(np.sum(active_pool) / max(int(np.sum(hard_unseen)), 1)),
        "active_pool_rule": selection_diag.get("active_pool_rule", cfg_raw.get("active_pool_rule")),
        "active_pool_quantile": q,
        "active_pool_quantile_requested": selection_diag.get("active_pool_quantile_requested"),
        "active_pool_rel_to_p95": float(cfg_raw.get("active_pool_rel_to_p95", 0.0)),
        "active_pool_threshold_quantile": selection_diag.get("active_pool_threshold_quantile", q_threshold if np.isfinite(q_threshold) else None),
        "active_pool_threshold_rel_p95": selection_diag.get("active_pool_threshold_rel_p95", rel_threshold if np.isfinite(rel_threshold) else None),
        "active_pool_threshold_final": selection_diag.get("active_pool_threshold_final"),
        "active_pool_fraction_cap": selection_diag.get("active_pool_fraction_cap"),
        "active_pool_fraction_cap_tightened": selection_diag.get("active_pool_fraction_cap_tightened"),
        "quantile_rule_count": int(selection_diag.get("active_pool_quantile_rule_count", int(np.sum(quantile_rule)))),
        "relative_p95_rule_count": int(selection_diag.get("active_pool_relative_p95_rule_count", int(np.sum(rel_rule)))),
        "active_pool_rule_overlap_count": int(selection_diag.get("active_pool_rule_overlap_count", int(np.sum(quantile_rule & rel_rule)))),
        "sampling_power_used": selection_diag.get("sampling_power_used"),
        "w_ext_current": float(np.nanmedian(monitor["w_ext_current"])) if "w_ext_current" in monitor else None,
        "high_confidence_interior_fraction_unseen": float(np.sum(high_conf & hard_unseen) / max(int(np.sum(hard_unseen)), 1)),
        "high_confidence_interior_fraction_active_pool": float(np.sum(high_conf & active_pool) / max(int(np.sum(active_pool)), 1)),
        "interior_penalty_applied_fraction_selected": float(np.sum(penalty_applied & selected_mask) / max(int(np.sum(selected_mask)), 1)),
        "N_eff_over_active_pool_size": selection_diag.get("N_eff_over_active_pool_size"),
        "selected_A0_mean": float(np.nanmean(selected_a0)) if selected_a0.size else None,
        "random_A0_mean": float(np.nanmean(random_a0)) if random_a0.size else None,
        "selected_A0_over_random_A0": (
            float(np.nanmean(selected_a0) / np.nanmean(random_a0))
            if selected_a0.size and random_a0.size and np.isfinite(np.nanmean(random_a0)) and abs(np.nanmean(random_a0)) > 1.0e-12
            else None
        ),
    }
    if finite_unseen.size:
        for p in (50, 75, 90, 95, 98, 99):
            active_summary[f"A0_for_pool_p{p}_unseen"] = float(np.percentile(finite_unseen, p))

    monitor_components = _monitor_component_arrays(monitor)
    component_rows = []
    component_rows.extend(_component_breakdown(monitor_components, labels, active_pool, "active_pool"))
    component_rows.extend(_component_breakdown(monitor_components, labels, hard_unseen, "all_unseen"))
    if selected_indices.size and not selected.empty:
        selected_labels = labels[selected_indices]
        selected_components = _selected_component_arrays(selected)
        selected_base = np.ones(selected_indices.shape[0], dtype=bool)
        component_rows.extend(_component_breakdown(selected_components, selected_labels, selected_base, "selected"))
    if random_indices.size:
        random_components = {k: v[random_indices] for k, v in monitor_components.items()}
        random_labels = labels[random_indices]
        random_base = np.ones(random_indices.shape[0], dtype=bool)
        component_rows.extend(_component_breakdown(random_components, random_labels, random_base, "random_baseline"))

    out_json = iter_dir / f"selection_region_diagnostics_iter{idx:03d}.json"
    payload = {
        "iteration": idx,
        "groups": group_rows,
        "active_pool": active_summary,
        "component_breakdown": component_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_rows = []
    for row in group_rows:
        r = {"table": "region_distribution", **row}
        csv_rows.append(r)
    for row in component_rows:
        r = {"table": "component_breakdown", "iteration": idx, **row}
        csv_rows.append(r)
    pd.DataFrame(csv_rows).to_csv(iter_dir / f"selection_region_diagnostics_iter{idx:03d}.csv", index=False)
    return payload


def _ensure_selection_region_diagnostics(run_dir: Path, cfg_raw: dict) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seed = int(cfg_raw.get("random_seed", cfg_raw.get("seed", 42)))
    for iter_dir in sorted([p for p in run_dir.glob("iter*") if p.is_dir()]):
        idx = _iter_index(iter_dir)
        if idx is None:
            continue
        payload = _write_selection_region_diagnostics_for_iter(run_dir, iter_dir, cfg_raw, seed)
        if payload is not None:
            records.append(payload)
    return records


def _rows_by_group(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload.get("groups", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    return {str(row.get("group")): row for row in rows if isinstance(row, dict)}


def _component_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows = payload.get("component_breakdown", []) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _region_distribution_table(payload: dict[str, object] | None) -> str:
    if not payload:
        return r"\multicolumn{7}{c}{N/A} \\"
    labels = {
        "all_unseen_candidates": "unseen candidates",
        "active_pool_candidates": "active pool",
        "selected_points": "selected points",
        "random_baseline": "random baseline",
    }
    rows = []
    by_group = _rows_by_group(payload)
    for key in ("all_unseen_candidates", "active_pool_candidates", "selected_points", "random_baseline"):
        row = by_group.get(key, {})
        rows.append(
            " & ".join(
                [
                    _tex_escape(labels[key]),
                    _fmt_table_value(row.get("count_total")),
                    _fmt_table_value(row.get("fraction_normal_interior")),
                    _fmt_table_value(row.get("fraction_uniform_sc_interior")),
                    _fmt_table_value(row.get("fraction_fflo_interior")),
                    _fmt_table_value(row.get("fraction_sc_interior")),
                    _fmt_table_value(row.get("fraction_boundary_band")),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _component_attribution_table(payload: dict[str, object] | None) -> str:
    if not payload:
        return r"\multicolumn{19}{c}{N/A} \\"
    wanted = [
        ("selected", "normal_interior"),
        ("selected", "sc_interior"),
        ("selected", "boundary_band"),
        ("active_pool", "normal_interior"),
        ("active_pool", "boundary_band"),
    ]
    rows = []
    components = {(str(r.get("group")), str(r.get("region"))): r for r in _component_rows(payload)}
    for group, region in wanted:
        row = components.get((group, region), {})
        rows.append(
            " & ".join(
                [
                    _tex_escape(group),
                    _tex_escape(region.replace("_", " ")),
                    _fmt_table_value(row.get("count")),
                    _fmt_table_value(row.get("A0_main_raw_mean", row.get("A0_main_mean"))),
                    _fmt_table_value(row.get("A0_for_pool_mean")),
                    _fmt_table_value(row.get("A0_main_mean")),
                    _fmt_table_value(row.get("A_phase_mean")),
                    _fmt_table_value(row.get("A_numerical_mean")),
                    _fmt_table_value(row.get("A_explore_mean")),
                    _fmt_table_value(row.get("B_delta_raw_mean")),
                    _fmt_table_value(row.get("U_NS_mean")),
                    _fmt_table_value(row.get("B_delta_gated_mean", row.get("B_delta_mean"))),
                    _fmt_table_value(row.get("U_cls_mix_mean")),
                    _fmt_table_value(row.get("U_reg_phase_mean")),
                    _fmt_table_value(row.get("B_q_gated_mean", row.get("B_q_SC_mean"))),
                    _fmt_table_value(row.get("G_phase_mean")),
                    _fmt_table_value(row.get("E_ext_uncertain_mean")),
                    _fmt_table_value(row.get("R_obs_mean")),
                    _fmt_table_value(row.get("Aselect_mean")),
                ]
            )
            + r" \\"
        )
    return "\n".join(rows)


def _selection_interpretation(payload: dict[str, object] | None) -> str:
    if not payload:
        return "Selection-region diagnostics are unavailable for this run."
    by_group = _rows_by_group(payload)
    comp = {(str(r.get("group")), str(r.get("region"))): r for r in _component_rows(payload)}
    selected = by_group.get("selected_points", {})
    active = by_group.get("active_pool_candidates", {})
    random = by_group.get("random_baseline", {})
    active_summary = payload.get("active_pool", {}) if isinstance(payload.get("active_pool", {}), dict) else {}
    parts: list[str] = []
    sel_norm = _safe_float(selected.get("fraction_normal_interior"))
    act_norm = _safe_float(active.get("fraction_normal_interior"))
    sel_band = _safe_float(selected.get("fraction_boundary_band"))
    rand_band = _safe_float(random.get("fraction_boundary_band"))
    sel_sc = _safe_float(selected.get("fraction_sc_interior"))
    if np.isfinite(sel_norm) and sel_norm > 0.5:
        parts.append("A significant fraction of acquisition-selected points lies in predicted normal-interior regions.")
    if np.isfinite(act_norm) and act_norm > 0.5:
        parts.append("The active-pool gate itself admits many normal-interior candidates, so the stochastic sampler is not operating on a boundary-only pool.")
    if np.isfinite(sel_band) and np.isfinite(rand_band) and sel_band <= 1.25 * max(rand_band, 1.0e-12):
        parts.append("The selected boundary-band fraction is close to the random baseline, indicating weak boundary focusing in this iteration.")
    elif np.isfinite(sel_band) and np.isfinite(rand_band) and sel_band > 1.5 * max(rand_band, 1.0e-12):
        parts.append("The selected boundary-band fraction is above the random baseline, indicating stronger boundary focusing than uniform random sampling.")
    active_fraction = _safe_float(active_summary.get("active_pool_fraction"))
    active_iter = active_summary.get("iteration")
    if isinstance(active_iter, (int, float, np.integer, np.floating)) and int(active_iter) >= 30 and np.isfinite(active_fraction) and active_fraction > 0.3:
        parts.append("The active pool still contains more than 30 percent of hard-unseen candidates after iter 30, so the pool remains broad.")
    selected_a0 = _safe_float(selected.get("A0_main_mean"))
    random_a0 = _safe_float(random.get("A0_main_mean"))
    if np.isfinite(selected_a0) and np.isfinite(random_a0) and random_a0 != 0.0 and selected_a0 / random_a0 > 1.1 and np.isfinite(sel_band) and sel_band < 0.15:
        parts.append("The selected points have elevated acquisition score relative to random candidates, but the high score is not dominated by proximity to the main phase boundaries.")
    row_norm = comp.get(("selected", "normal_interior"), {})
    a0 = _safe_float(row_norm.get("A0_main_mean"))
    aexplore = _safe_float(row_norm.get("A_explore_mean"))
    ureg = _safe_float(row_norm.get("U_reg_phase_mean"))
    bdelta = _safe_float(row_norm.get("B_delta_mean"))
    bdelta_raw = _safe_float(row_norm.get("B_delta_raw_mean"))
    u_ns = _safe_float(row_norm.get("U_NS_mean"))
    bdelta_gated = _safe_float(row_norm.get("B_delta_gated_mean", row_norm.get("B_delta_mean")))
    if np.isfinite(a0) and abs(a0) > 1.0e-12:
        if np.isfinite(aexplore) and aexplore / a0 > 0.25:
            parts.append("Normal-interior selections have a sizable exploration contribution, suggesting that exploration uncertainty is one driver of these points.")
        if np.isfinite(ureg) and ureg > 0.25:
            parts.append("Normal-interior selections show noticeable regression uncertainty, so they may be uncertainty-driven rather than boundary-proximity driven.")
        if np.isfinite(bdelta_raw) and np.isfinite(u_ns) and np.isfinite(bdelta_gated) and bdelta_raw > 0.5 and u_ns < 0.2 and bdelta_gated < 0.25 * bdelta_raw:
            parts.append("The raw Delta-boundary score is high in predicted normal interior, but the normal/SC competition gate suppresses that false boundary signal.")
        elif np.isfinite(bdelta) and bdelta / a0 > 0.35:
            parts.append("The Delta-boundary score contributes strongly inside the predicted normal region; inspect whether the predicted Delta transition tail is too broad.")
    if np.isfinite(sel_sc) and sel_sc < 0.1:
        parts.append("Only a small fraction of selected points lies in predicted superconducting interiors in this iteration.")
    if not parts:
        parts.append("The latest selection diagnostics do not show a single dominant pathology; compare the tables and time series before changing the acquisition policy.")
    return " ".join(parts)


def _build_selection_region_figures(
    diagnostics: list[dict[str, object]],
    run_dir: Path,
    figures_dir: Path,
    run_id: str,
    cfg_raw: dict,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import pandas as pd
    except Exception:
        return {}
    if not diagnostics:
        return {}
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    active_rows = []
    for payload in diagnostics:
        idx = int(payload.get("iteration", -1))
        by_group = _rows_by_group(payload)
        for group, row in by_group.items():
            rows.append({"iter": idx, "group": group, **row})
        active = payload.get("active_pool", {})
        if isinstance(active, dict):
            active_rows.append({"iter": idx, **active})
    df = pd.DataFrame(rows)
    adf = pd.DataFrame(active_rows)
    out: dict[str, str] = {}

    if not df.empty:
        p = figures_dir / f"{run_id}_selection_region_fractions.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        for col, label in [
            ("fraction_normal_interior", "selected normal interior"),
            ("fraction_sc_interior", "selected SC interior"),
            ("fraction_boundary_band", "selected boundary band"),
        ]:
            sub = df[df["group"] == "selected_points"]
            if col in sub:
                ax.plot(sub["iter"], sub[col], marker="o", ms=3, label=label)
        sub_rand = df[df["group"] == "random_baseline"]
        if "fraction_boundary_band" in sub_rand:
            ax.plot(sub_rand["iter"], sub_rand["fraction_boundary_band"], marker="x", ms=3, label="random boundary band")
        ax.set_xlabel("iteration")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.28)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(p, dpi=240)
        plt.close(fig)
        out["region_fractions"] = p.name

        p = figures_dir / f"{run_id}_active_pool_region_fractions.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        sub = df[df["group"] == "active_pool_candidates"]
        if not sub.empty:
            ax.plot(sub["iter"], sub["fraction_normal_interior"], marker="o", ms=3, label="active-pool normal interior")
            ax.plot(sub["iter"], sub["fraction_sc_interior"], marker="s", ms=3, label="active-pool SC interior")
            ax.plot(sub["iter"], sub["fraction_boundary_band"], marker="^", ms=3, label="active-pool boundary band")
        if not adf.empty and "active_pool_fraction" in adf:
            ax.plot(adf["iter"], adf["active_pool_fraction"], marker="x", ms=3, label="active-pool / unseen")
        ax.set_xlabel("iteration")
        ax.set_ylabel("fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.28)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(p, dpi=240)
        plt.close(fig)
        out["active_pool_regions"] = p.name

    if not adf.empty:
        p = figures_dir / f"{run_id}_selection_score_concentration.png"
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        if "N_eff_over_active_pool_size" in adf:
            ax.plot(adf["iter"], adf["N_eff_over_active_pool_size"], marker="o", ms=3, label=r"$N_{\mathrm{eff}}/N_{\mathrm{pool}}$")
        if "selected_A0_over_random_A0" in adf:
            ax.plot(adf["iter"], adf["selected_A0_over_random_A0"], marker="s", ms=3, label=r"selected/random $A_{0,\mathrm{main}}$")
        ax.set_xlabel("iteration")
        ax.set_ylabel("ratio")
        ax.grid(alpha=0.28)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(p, dpi=240)
        plt.close(fig)
        out["score_concentration"] = p.name

    latest = diagnostics[-1]
    idx = int(latest.get("iteration", -1))
    iter_dir = run_dir / f"iter{idx:03d}"
    monitor_path = iter_dir / f"monitor_predictions_iter{idx:03d}.npz"
    selected_path = iter_dir / "selected_points_by_pool.csv"
    if monitor_path.exists() and selected_path.exists():
        try:
            with np.load(monitor_path, allow_pickle=False) as z:
                monitor = {k: z[k] for k in z.files}
            selected = pd.read_csv(selected_path)
            grid_points = np.asarray(monitor["grid_points"], dtype=np.float64)
            phase = np.asarray(monitor["phase_pred"], dtype=np.int64)
            labels = _region_labels(
                phase,
                _nearest_distance_norm(
                    grid_points,
                    _main_boundary_points_from_phase(
                        grid_points,
                        np.asarray(monitor["kt_values"], dtype=np.float64),
                        np.asarray(monitor["ja_values"], dtype=np.float64),
                        np.asarray(monitor["full_shape"], dtype=np.int64),
                        phase,
                    ),
                    cfg_raw,
                ),
                float(latest.get("active_pool", {}).get("boundary_band_width_norm") or 2.0 * _dense_grid_spacing_norm(monitor, cfg_raw)),
            )
            phase_grid = phase.reshape(tuple(np.asarray(monitor["full_shape"], dtype=np.int64)))
            kt = np.asarray(monitor["kt_values"], dtype=np.float64)
            ja = np.asarray(monitor["ja_values"], dtype=np.float64)
            p = figures_dir / f"{run_id}_iter{idx:03d}_selected_regions.png"
            fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
            cmap = ListedColormap(["#4c78a8", "#72b7b2", "#f58518"])
            ax.imshow(
                phase_grid,
                origin="lower",
                aspect="auto",
                extent=[float(kt.min()), float(kt.max()), float(ja.min()), float(ja.max())],
                cmap=cmap,
                alpha=0.45,
                vmin=0,
                vmax=2,
            )
            boundary = _main_boundary_points_from_phase(grid_points, kt, ja, np.asarray(monitor["full_shape"], dtype=np.int64), phase)
            if boundary.size:
                ax.scatter(boundary[:, 0], boundary[:, 1], s=3, c="black", alpha=0.55, label="predicted boundary")
            if "grid_index" in selected:
                idxs = pd.to_numeric(selected["grid_index"], errors="coerce").dropna().to_numpy(dtype=np.int64)
                idxs = idxs[(idxs >= 0) & (idxs < grid_points.shape[0])]
                pts = grid_points[idxs]
                slabels = labels[idxs]
                masks = {
                    "normal interior": slabels == "normal_interior",
                    "SC interior": (slabels == "uniform_sc_interior") | (slabels == "fflo_interior"),
                    "boundary band": (slabels == "normal_boundary_band") | (slabels == "sc_boundary_band"),
                }
                colors = {"normal interior": "#1f77b4", "SC interior": "#2ca02c", "boundary band": "#d62728"}
                for name, mask in masks.items():
                    if np.any(mask):
                        ax.scatter(pts[mask, 0], pts[mask, 1], s=28, marker="x", c=colors[name], linewidths=1.0, label=name)
            ax.set_xlabel(r"$k_B T/t$")
            ax.set_ylabel(r"$J_A/t$")
            ax.set_title("Latest Selected Points by Predicted Region")
            ax.legend(loc="best", fontsize=8)
            fig.savefig(p, dpi=240)
            plt.close(fig)
            out["selected_regions"] = p.name
        except Exception:
            pass
    return out


def _selection_region_figure_block(figures: dict[str, str]) -> str:
    blocks = []
    for key, caption in [
        ("region_fractions", "Time series of selected region fractions compared with the random boundary-band baseline."),
        ("active_pool_regions", "Active-pool region fractions and active-pool size relative to hard-unseen candidates."),
        ("score_concentration", "Sampling concentration diagnostics and selected/random acquisition-score ratio."),
        ("selected_regions", "Latest selected points colored by predicted region type on the predicted phase map."),
    ]:
        filename = figures.get(key)
        if filename:
            blocks.append(_include_single_figure_block(filename, r"0.72\textwidth", caption))
    return "\n\n".join(blocks) if blocks else _figure_unavailable_block("selection region diagnostics could not be generated")


def _learning_curve_caption(latest_metrics: dict) -> str:
    f1 = latest_metrics.get("boundary_f1")
    if isinstance(f1, (float, int, np.floating, np.integer)) and np.isfinite(f1):
        return "Learning-curve summary including phase accuracy, response RMSE, and boundary F1."
    return "Learning-curve summary. Boundary F1 is omitted because hidden-ground-truth boundary evaluation is not available for this report."


def build_report(run_dir: Path, template_path: Path, output_tex: Path) -> Path:
    run_cfg = _load_json(run_dir / "run_config.json", default={})
    metrics_hist = _load_json(run_dir / "metrics_history.json", default=[])
    latest_metrics = metrics_hist[-1] if metrics_hist else {}
    latest_iter = _latest_iter_dir(run_dir)
    cfg_raw = run_cfg.get("active_learning_config", {})
    run_mode = str(cfg_raw.get("run_mode", "refinement"))
    candidate_domain_mode = str(cfg_raw.get("candidate_domain_mode", "prior_band"))
    selection_mode = str(cfg_raw.get("selection_mode", "topk"))
    initialization = str(cfg_raw.get("initialization", "random_grid"))
    report_title = (
        "Active-Learning Discovery from Random Exact Seeds"
        if run_mode == "discovery"
        else "Warm-Start ML-Guided Boundary Refinement"
    )
    if run_mode == "discovery":
        run_mode_note = (
            "Discovery mode starts from random exact seed points on the full rectangular candidate grid; "
            "no warm-up exact dataset or finite-T prior candidate mask is used for training initialization."
        )
    elif candidate_domain_mode == "prior_band":
        run_mode_note = "Refinement mode uses a prior-constrained candidate band; interpret unexplored regions outside the band accordingly."
    else:
        run_mode_note = "Refinement mode uses the configured candidate domain and starts from an existing exact dataset."
    delta_eps = float(cfg_raw.get("delta_eps", 1e-3))
    q_eps = float(cfg_raw.get("q_eps", 1e-2))
    positive_delta_gap_tol = float(cfg_raw.get("positive_delta_gap_tol", 1e-8))
    boundary_refinement_mode = cfg_raw.get("boundary_refinement_mode", "off")
    boundary_position_tol = cfg_raw.get("boundary_position_tol")
    observation_repulsion_length = cfg_raw.get("observation_repulsion_length")
    batch_repulsion_length = cfg_raw.get("batch_repulsion_length")
    selected_by_pool = "N/A"
    selected_by_boundary_type = "N/A"
    boundary_segment_counts = "N/A"
    boundary_displacement_summary = "N/A"
    stop_status = "N/A"
    stop_reason = "N/A"
    stop_convergence_pass = "N/A"
    stop_hard_stop = "N/A"
    stop_passed_count = "N/A"
    stop_patience_counter = "N/A"
    stop_conditions = "N/A"
    stop_metric_summary = "N/A"
    stop_boundary_metric_type = "N/A"
    acq_weight_summary = _fmt_acq_weight_summary(cfg_raw)

    warm_npz = run_cfg.get("args", {}).get("warm_start")
    n_warm = "N/A"
    if warm_npz:
        p = Path(warm_npz)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                n_warm = int(z["eta_matrix"].shape[0] * z["eta_matrix"].shape[1])

    n_exact = "N/A"
    n_normal = "N/A"
    n_uniform_sc = "N/A"
    n_fflo = "N/A"
    latest_dataset = _latest_dataset_path(run_dir)
    final_dataset_name = "N/A"
    completed_iterations = "N/A"
    latest_completed_iteration = "N/A"
    final_dataset_idx = _dataset_iter_index(latest_dataset)
    if latest_dataset is not None:
        final_dataset_name = latest_dataset.name
    if final_dataset_idx is not None:
        completed_iterations = final_dataset_idx
        latest_completed_iteration = final_dataset_idx - 1
    if latest_dataset is not None:
        data = _load_dataset_npz_or_csv(latest_dataset)
        n_exact = int(data["x"].shape[0])
        y_reg = data["y_reg"].astype(float)
        delta = y_reg[:, 0]
        q = y_reg[:, 1]
        normal = delta < delta_eps
        uniform_sc = (~normal) & (np.abs(q) < q_eps)
        fflo = (~normal) & ~uniform_sc
        n_normal = int(np.sum(normal))
        n_uniform_sc = int(np.sum(uniform_sc))
        n_fflo = int(np.sum(fflo))
    latest_append_status = "N/A"
    append_files = sorted(run_dir.glob("dataset_iter*.append.json"))
    if append_files:
        latest_append = _load_json(append_files[-1], default={})
        n_new = latest_append.get("new_unique_samples_added")
        n_train = latest_append.get("training_eligible_points_appended")
        latest_append_status = f"{_fmt(n_new)} new unique samples, {_fmt(n_train)} training-eligible appended"

    iteration_rows: list[str] = []
    for iter_dir in sorted([p for p in run_dir.glob("iter*") if p.is_dir()]):
        try:
            idx = int(iter_dir.name.replace("iter", ""))
        except ValueError:
            continue
        selected = "N/A"
        selected_path = iter_dir / "selected_points.csv"
        if selected_path.exists():
            with selected_path.open("r", encoding="utf-8", errors="ignore") as f:
                selected = max(0, sum(1 for _ in f) - 1)
        merge = _load_json(iter_dir / f"merge_summary_iter{idx:03d}.json", default={})
        merged = merge.get("merged_points", "N/A")
        training = merge.get("training_eligible_points", "N/A")
        rerun = merge.get("rerun_required_points", "N/A")
        dataset_path = _dataset_iter_path(run_dir, idx + 1)
        dataset_samples = "N/A"
        if dataset_path.exists():
            dataset_samples = int(_load_dataset_npz_or_csv(dataset_path)["x"].shape[0])
        iteration_rows.append(
            " & ".join(
                [
                    _fmt(idx),
                    _fmt(selected),
                    _fmt(merged),
                    _fmt(training),
                    _fmt(rerun),
                    _fmt(dataset_samples),
                ]
            )
            + r" \\"
        )
    if len(iteration_rows) > 18:
        omitted = len(iteration_rows) - 13
        iteration_rows = (
            iteration_rows[:5]
            + [rf"\multicolumn{{6}}{{c}}{{\ldots {omitted} intermediate iterations omitted; see JSON outputs for full history}} \\"]
            + iteration_rows[-8:]
        )
    iteration_summary_rows = "\n".join(iteration_rows) if iteration_rows else r"\multicolumn{6}{c}{N/A} \\"
    q_edge_rate = "N/A"
    delta_amb_rate = "N/A"
    n_merged_exact = "N/A"
    n_trusted_exact = "N/A"
    n_training_eligible_exact = "N/A"
    n_rerun_required = "N/A"
    n_boundary_band_normal = "N/A"
    n_delta_unresolved_requiring_rerun = "N/A"
    q_expanded_count = "N/A"
    q_unresolved_count = "N/A"
    delta_refined_count = "N/A"
    delta_unresolved_count = "N/A"
    positive_delta_checked_count = "N/A"
    normal_q_na_count = "N/A"
    if latest_iter:
        selection_diag = _load_json(latest_iter / "selection_diagnostics.json", default={})
        boundary_selection = selection_diag.get("boundary_selection", {})
        selected_by_pool = _fmt_dict(boundary_selection.get("selected_by_pool"))
        selected_by_boundary_type = _fmt_dict(boundary_selection.get("selected_by_boundary_type"))
        boundary_summary = _load_json(latest_iter / "boundaries" / "boundary_summary.json", default={})
        boundary_segment_counts = _fmt_dict(boundary_summary.get("boundary_segments_by_type"))
        try:
            latest_iter_idx = int(latest_iter.name.replace("iter", ""))
        except ValueError:
            latest_iter_idx = -1
        boundary_displacement = _load_json(
            latest_iter / f"boundary_displacement_iter{latest_iter_idx:03d}.json",
            default={},
        )
        boundary_displacement_summary = _fmt_boundary_displacement(boundary_displacement)
        merged_path = latest_iter / f"exact_merged_iter{latest_iter.name[-3:]}.npz"
        if merged_path.exists():
            with np.load(merged_path, allow_pickle=False) as z:
                n = int(z["kT"].shape[0]) if "kT" in z.files else 0
                n_merged_exact = n
                if "q_edge_hit" in z.files and z["q_edge_hit"].size:
                    q_edge_rate = float(np.mean(z["q_edge_hit"].astype(bool)))
                if "delta_boundary_ambiguous" in z.files and z["delta_boundary_ambiguous"].size:
                    delta_amb_rate = float(np.mean(z["delta_boundary_ambiguous"].astype(bool)))
                if "trusted_exact" in z.files:
                    status = z["exact_status_code"].astype(int) if "exact_status_code" in z.files else np.zeros(n, dtype=int)
                    trusted = z["trusted_exact"].astype(bool) & (status == 0)
                    n_trusted_exact = int(np.sum(trusted))
                if "delta_boundary_band_normal" in z.files:
                    boundary_band = z["delta_boundary_band_normal"].astype(bool)
                    n_boundary_band_normal = int(np.sum(boundary_band))
                else:
                    delta_unresolved_for_band = (
                        z["delta_unresolved"].astype(bool) if "delta_unresolved" in z.files else np.zeros(n, dtype=bool)
                    )
                    delta_opt_for_band = (
                        z["delta_opt"].astype(float) if "delta_opt" in z.files else np.full(n, np.nan)
                    )
                    positive_gap_for_band = (
                        z["positive_delta_gap"].astype(float)
                        if "positive_delta_gap" in z.files
                        else np.full(n, np.nan)
                    )
                    boundary_band = (
                        delta_unresolved_for_band
                        & (delta_opt_for_band < delta_eps)
                        & np.isfinite(positive_gap_for_band)
                        & (positive_gap_for_band >= 0.0)
                        & (positive_gap_for_band <= positive_delta_gap_tol)
                    )
                    n_boundary_band_normal = int(np.sum(boundary_band))
                if "training_eligible_exact" in z.files:
                    training_eligible = z["training_eligible_exact"].astype(bool)
                    n_training_eligible_exact = int(np.sum(training_eligible))
                    n_rerun_required = int(n - np.sum(training_eligible))
                elif "trusted_exact" in z.files:
                    training_eligible = z["trusted_exact"].astype(bool) | boundary_band
                    n_training_eligible_exact = int(np.sum(training_eligible))
                    n_rerun_required = int(n - np.sum(training_eligible))
                if "q_expanded" in z.files:
                    q_expanded_count = int(np.sum(z["q_expanded"].astype(bool)))
                if "q_unresolved" in z.files:
                    q_unresolved_count = int(np.sum(z["q_unresolved"].astype(bool)))
                if "delta_refined" in z.files:
                    delta_refined_count = int(np.sum(z["delta_refined"].astype(bool)))
                if "delta_unresolved" in z.files:
                    delta_unresolved_mask = z["delta_unresolved"].astype(bool)
                    delta_unresolved_count = int(np.sum(delta_unresolved_mask))
                    n_delta_unresolved_requiring_rerun = int(np.sum(delta_unresolved_mask & ~boundary_band))
                if "positive_delta_checked" in z.files:
                    positive_delta_checked_count = int(np.sum(z["positive_delta_checked"].astype(bool)))
                if "q_status" in z.files:
                    normal_q_na_count = int(np.sum(z["q_status"].astype(int) == 0))
        stop_metrics = _load_json(latest_iter / f"stop_metrics_iter{latest_iter_idx:03d}.json", default={})
        stop_state = _load_json(run_dir / "stop_state.json", default={})
        stop_status = _fmt_bool(stop_metrics.get("stop", stop_state.get("stop")))
        stop_reason = _fmt(stop_metrics.get("stop_reason", stop_state.get("stop_reason")))
        stop_convergence_pass = _fmt_bool(stop_metrics.get("convergence_pass"))
        stop_hard_stop = _fmt_bool(stop_metrics.get("hard_stop"))
        stop_passed_count = _fmt(stop_metrics.get("passed_condition_count"))
        stop_patience_counter = _fmt(stop_metrics.get("patience_counter", stop_state.get("patience_counter")))
        stop_conditions = _fmt_condition_summary(stop_metrics.get("conditions"))
        stop_metric_summary = _fmt_stop_metric_summary(stop_metrics)
        boundary_details = stop_metrics.get("boundary_details", {})
        if isinstance(boundary_details, dict) and boundary_details:
            stop_boundary_metric_type = (
                "normalized nearest-neighbor boundary shift; the reported stop value is the configured "
                "p95-like value stored in boundary_details.*.value"
            )

    boundary_f1_note = ""
    if _fmt(latest_metrics.get("boundary_f1")) == "N/A":
        boundary_f1_note = (
            "Boundary F1 is not available because hidden-ground-truth boundary evaluation has not been "
            "implemented for this report."
        )
    exact_reduction_note = (
        "The estimated exact-call reduction is a rough ratio of the full dense candidate grid size to the "
        "number of exact calls used by the run. It is not, by itself, proof of final physical accuracy; "
        "that requires hidden-ground-truth offline boundary evaluation and multi-seed aggregation."
    )

    fig_phase = ""
    fig_uncert = ""
    fig_acq = ""
    fig_selected = ""
    fig_lc = ""
    fig_cumulative_selected = ""
    fig_cumulative_accepted = ""

    output_root = run_dir.parent.parent
    figures_dir = output_root / "figures"
    run_id = run_cfg.get("args", {}).get("run_id", run_dir.name)
    selection_region_diagnostics = _ensure_selection_region_diagnostics(run_dir, cfg_raw)
    latest_selection_region_payload = selection_region_diagnostics[-1] if selection_region_diagnostics else None
    selection_region_distribution_rows = _region_distribution_table(latest_selection_region_payload)
    selection_component_rows = _component_attribution_table(latest_selection_region_payload)
    selection_interpretation = _selection_interpretation(latest_selection_region_payload)
    selection_region_figures = _build_selection_region_figures(
        selection_region_diagnostics,
        run_dir,
        figures_dir,
        str(run_id),
        cfg_raw,
    )
    selection_region_figure_block = _selection_region_figure_block(selection_region_figures)
    if latest_iter:
        i = int(latest_iter.name[-3:])
        fig_phase = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_phase_prediction.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_phase_prediction.png",
        )
        fig_uncert = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_uncertainty.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_uncertainty.png",
        )
        fig_acq = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_acquisition.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_acquisition.png",
        )
        fig_selected = _copy_if_exists(
            figures_dir / f"{run_id}_iter{i:03d}_selected_points.png",
            Path("report/figures") / f"{run_id}_iter{i:03d}_selected_points.png",
        )
    if metrics_hist:
        try:
            from .plot_active_learning import write_learning_curve

            write_learning_curve(figures_dir, str(run_id), metrics_hist)
        except Exception:
            pass
    fig_lc = _copy_if_exists(
        figures_dir / f"{run_id}_learning_curve.png",
        Path("report/figures") / f"{run_id}_learning_curve.png",
    )
    fig_cumulative_selected = _copy_if_exists(
        figures_dir / f"{run_id}_cumulative_selected_points.png",
        Path("report/figures") / f"{run_id}_cumulative_selected_points.png",
    )
    fig_cumulative_accepted = _copy_if_exists(
        figures_dir / f"{run_id}_cumulative_accepted_points.png",
        Path("report/figures") / f"{run_id}_cumulative_accepted_points.png",
    )
    cumulative_figure_block = _build_cumulative_figure_block(run_dir, figures_dir, str(run_id))
    exact_phase_map_block = _build_exact_phase_map_block(figures_dir, str(run_id), latest_dataset)
    exact_eta_revised_boundary_block = _build_exact_eta_revised_boundary_block(run_dir, figures_dir, str(run_id), latest_dataset)
    selection_source_map_block = _build_selection_source_map_block(run_dir, figures_dir, str(run_id))
    selection_focus_curve_block = _build_selection_focus_curve_block(run_dir, figures_dir, str(run_id))

    replacements: Dict[str, str] = {
        "{{REPORT_TITLE}}": _tex_escape(report_title),
        "{{RUN_ID}}": _tex_escape(str(run_id)),
        "{{RUN_MODE}}": _tex_escape(run_mode),
        "{{CANDIDATE_DOMAIN_MODE}}": _tex_escape(candidate_domain_mode),
        "{{SELECTION_MODE}}": _tex_escape(selection_mode),
        "{{INITIALIZATION}}": _tex_escape(initialization),
        "{{INITIAL_SEED_SIZE}}": _tex_escape(_fmt(cfg_raw.get("initial_seed_size"))),
        "{{BATCH_SIZE_MAX}}": _tex_escape(_fmt(cfg_raw.get("batch_size_max"))),
        "{{ACTIVE_POOL_QUANTILE}}": _tex_escape(_fmt(cfg_raw.get("active_pool_quantile"))),
        "{{ACTIVE_POOL_REL_TO_P95}}": _tex_escape(_fmt(cfg_raw.get("active_pool_rel_to_p95"))),
        "{{FINITE_T_BAND_WIDTH}}": _tex_escape("disabled" if cfg_raw.get("finite_t_band_width") is None else _fmt(cfg_raw.get("finite_t_band_width"))),
        "{{RUN_MODE_NOTE}}": _tex_escape(run_mode_note),
        "{{N_WARM_START}}": _tex_escape(_fmt(n_warm)),
        "{{N_EXACT}}": _tex_escape(_fmt(n_exact)),
        "{{LATEST_COMPLETED_ITERATION}}": _tex_escape(_fmt(latest_completed_iteration)),
        "{{COMPLETED_ITERATIONS}}": _tex_escape(_fmt(completed_iterations)),
        "{{FINAL_DATASET}}": _tex_escape(_fmt(final_dataset_name)),
        "{{N_ITERS}}": _tex_escape(_fmt(len(metrics_hist))),
        "{{N_NORMAL}}": _tex_escape(_fmt(n_normal)),
        "{{N_UNIFORM_SC}}": _tex_escape(_fmt(n_uniform_sc)),
        "{{N_FFLO}}": _tex_escape(_fmt(n_fflo)),
        "{{LATEST_APPEND_STATUS}}": _tex_escape(latest_append_status),
        "{{REG_MODEL}}": _tex_escape("Torch MLP Ensemble"),
        "{{CLS_MODEL}}": _tex_escape("Torch MLP Ensemble"),
        "{{N_ENSEMBLE}}": _tex_escape(_fmt(run_cfg.get("active_learning_config", {}).get("n_ensemble"))),
        "{{N_FEATURES}}": _tex_escape("2"),
        "{{N_REG_TARGETS}}": _tex_escape("5"),
        "{{VAL_FRACTION}}": _tex_escape(_fmt(run_cfg.get("active_learning_config", {}).get("val_fraction"))),
        "{{MODE}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("mode"))),
        "{{WORLD_SIZE}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("world_size"))),
        "{{PARTITION_STRATEGY}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("partition_strategy"))),
        "{{POINTS_PER_ITER}}": _tex_escape(_fmt(run_cfg.get("args", {}).get("points_per_iter"))),
        "{{BOUNDARY_REFINEMENT_MODE}}": _tex_escape(_fmt(boundary_refinement_mode)),
        "{{OBSERVATION_REPULSION_LENGTH}}": _tex_escape(_fmt(observation_repulsion_length)),
        "{{OBSERVATION_REPULSION_FLOOR}}": _tex_escape(_fmt(cfg_raw.get("observation_repulsion_floor"))),
        "{{BATCH_REPULSION_LENGTH}}": _tex_escape(_fmt(batch_repulsion_length)),
        "{{BATCH_REPULSION_FLOOR}}": _tex_escape(_fmt(cfg_raw.get("batch_repulsion_floor"))),
        "{{BOUNDARY_POSITION_TOL}}": _tex_escape(_fmt(boundary_position_tol)),
        "{{SELECTED_BY_POOL}}": _tex_escape(selected_by_pool),
        "{{SELECTED_BY_BOUNDARY_TYPE}}": _tex_escape(selected_by_boundary_type),
        "{{BOUNDARY_SEGMENT_COUNTS}}": _tex_escape(boundary_segment_counts),
        "{{BOUNDARY_DISPLACEMENT_SUMMARY}}": _tex_escape(boundary_displacement_summary),
        "{{ACQ_WEIGHT_SUMMARY}}": acq_weight_summary,
        "{{B_DELTA_GATE_DESCRIPTION}}": _b_delta_gate_description(cfg_raw),
        "{{SELECTION_DESCRIPTION}}": _selection_description(selection_mode, cfg_raw),
        "{{DELTA_RMSE}}": _tex_escape(_fmt(latest_metrics.get("delta_rmse"))),
        "{{Q_RMSE}}": _tex_escape(_fmt(latest_metrics.get("q_rmse"))),
        "{{ETA_RMSE}}": _tex_escape(_fmt(latest_metrics.get("eta_rmse"))),
        "{{ICP_RMSE}}": _tex_escape(_fmt(latest_metrics.get("ic_plus_rmse"))),
        "{{ICM_RMSE}}": _tex_escape(_fmt(latest_metrics.get("ic_minus_rmse"))),
        "{{PHASE_ACC}}": _tex_escape(_fmt(latest_metrics.get("phase_accuracy"))),
        "{{BOUNDARY_F1}}": _tex_escape(_fmt(latest_metrics.get("boundary_f1"))),
        "{{BOUNDARY_F1_NOTE}}": _tex_escape(boundary_f1_note),
        "{{EXACT_REDUCTION}}": _tex_escape(_fmt(latest_metrics.get("estimated_reduction"))),
        "{{EXACT_REDUCTION_NOTE}}": _tex_escape(exact_reduction_note),
        "{{Q_EDGE_HIT_RATE}}": _tex_escape(_fmt(q_edge_rate)),
        "{{DELTA_AMBIGUOUS_RATE}}": _tex_escape(_fmt(delta_amb_rate)),
        "{{N_MERGED_EXACT}}": _tex_escape(_fmt(n_merged_exact)),
        "{{N_TRUSTED_EXACT}}": _tex_escape(_fmt(n_trusted_exact)),
        "{{N_TRAINING_ELIGIBLE_EXACT}}": _tex_escape(_fmt(n_training_eligible_exact)),
        "{{N_RERUN_REQUIRED}}": _tex_escape(_fmt(n_rerun_required)),
        "{{N_BOUNDARY_BAND_NORMAL}}": _tex_escape(_fmt(n_boundary_band_normal)),
        "{{N_DELTA_UNRESOLVED_REQUIRING_RERUN}}": _tex_escape(_fmt(n_delta_unresolved_requiring_rerun)),
        "{{Q_EXPANDED_COUNT}}": _tex_escape(_fmt(q_expanded_count)),
        "{{Q_UNRESOLVED_COUNT}}": _tex_escape(_fmt(q_unresolved_count)),
        "{{DELTA_REFINED_COUNT}}": _tex_escape(_fmt(delta_refined_count)),
        "{{DELTA_UNRESOLVED_COUNT}}": _tex_escape(_fmt(delta_unresolved_count)),
        "{{POSITIVE_DELTA_CHECKED_COUNT}}": _tex_escape(_fmt(positive_delta_checked_count)),
        "{{NORMAL_Q_NA_COUNT}}": _tex_escape(_fmt(normal_q_na_count)),
        "{{STOP_STATUS}}": _tex_escape(stop_status),
        "{{STOP_REASON}}": _tex_escape(stop_reason),
        "{{STOP_CONVERGENCE_PASS}}": _tex_escape(stop_convergence_pass),
        "{{STOP_HARD_STOP}}": _tex_escape(stop_hard_stop),
        "{{STOP_PASSED_COUNT}}": _tex_escape(stop_passed_count),
        "{{STOP_PATIENCE_COUNTER}}": _tex_escape(stop_patience_counter),
        "{{STOP_CONDITIONS}}": _tex_escape(stop_conditions),
        "{{STOP_METRIC_SUMMARY}}": _tex_escape(stop_metric_summary),
        "{{STOP_BOUNDARY_METRIC_TYPE}}": _tex_escape(stop_boundary_metric_type),
        "{{SELECTION_REGION_DISTRIBUTION_ROWS}}": selection_region_distribution_rows,
        "{{SELECTION_COMPONENT_ATTRIBUTION_ROWS}}": selection_component_rows,
        "{{SELECTION_REGION_INTERPRETATION}}": _tex_escape(selection_interpretation),
        "{{ITERATION_SUMMARY_ROWS}}": iteration_summary_rows,
        "__FIG_EXACT_PHASE_MAP_BLOCK__": exact_phase_map_block,
        "__FIG_EXACT_ETA_REVISED_BOUNDARY_BLOCK__": exact_eta_revised_boundary_block,
        "__FIG_CUMULATIVE_BLOCK__": cumulative_figure_block,
        "__FIG_SELECTION_SOURCE_BLOCK__": selection_source_map_block,
        "__FIG_SELECTION_FOCUS_BLOCK__": selection_focus_curve_block,
        "__FIG_SELECTION_REGION_BLOCK__": selection_region_figure_block,
        "__FIG_PHASE__": fig_phase or _placeholder_png(Path("report/figures/missing_phase.png")),
        "__FIG_UNCERT__": fig_uncert or _placeholder_png(Path("report/figures/missing_uncertainty.png")),
        "__FIG_ACQ__": fig_acq or _placeholder_png(Path("report/figures/missing_acquisition.png")),
        "__FIG_SELECTED__": fig_selected or _placeholder_png(Path("report/figures/missing_selected.png")),
        "__FIG_LC__": fig_lc or _placeholder_png(Path("report/figures/missing_learning_curve.png")),
        "__FIG_CUMULATIVE_SELECTED__": fig_cumulative_selected,
        "__FIG_CUMULATIVE_ACCEPTED__": fig_cumulative_accepted,
        "{{LEARNING_CURVE_CAPTION}}": _tex_escape(_learning_curve_caption(latest_metrics)),
    }

    tex = template_path.read_text(encoding="utf-8")
    for k, v in replacements.items():
        tex = tex.replace(k, v)

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(tex, encoding="utf-8")
    return output_tex


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LaTeX report from active-learning outputs.")
    p.add_argument("--run-id", type=str, required=True, help="Run id under ML_Phase/active_runs")
    p.add_argument("--run-root", type=Path, default=Path("ML_Phase/active_runs"), help="Active runs root")
    p.add_argument(
        "--template",
        type=Path,
        default=Path("report/active_learning_phase_boundary_report.tex"),
        help="LaTeX template path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("ML_Phase/reports/active_learning_phase_boundary_report.tex"),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_root / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out = build_report(run_dir, args.template, args.output)
    print(f"Wrote report tex: {out}")


if __name__ == "__main__":
    main()
