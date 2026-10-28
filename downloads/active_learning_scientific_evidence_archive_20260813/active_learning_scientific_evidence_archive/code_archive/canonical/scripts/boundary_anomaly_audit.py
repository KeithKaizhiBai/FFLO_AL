from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PHASE_NAMES = {
    0: "normal",
    1: "uniform_SC",
    2: "FFLO",
}

KT_MIN = 0.0
KT_MAX = 0.56
JA_MIN = 0.0
JA_MAX = 2.12


def _signed_power(values: np.ndarray, gamma: float = 0.35) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.power(np.abs(values), gamma)


def _finite_t_reference_boundaries() -> dict[str, pd.DataFrame]:
    return {
        "old_cfflo_tfflo_1st": pd.DataFrame(
            {
                "kT_boundary": np.array([0.01, 0.04, 0.05], dtype=np.float64),
                "JA_boundary": np.array([0.6, 0.6, 0.6], dtype=np.float64),
            }
        ),
        "old_cfflo_tfflo_2nd": pd.DataFrame(
            {
                "kT_boundary": np.array([0.06, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4], dtype=np.float64),
                "JA_boundary": np.array([0.6, 0.6, 0.62, 0.6277, 0.63, 0.628, 0.617, 0.598, 0.565], dtype=np.float64),
            }
        ),
    }


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_boundary(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    df = pd.read_csv(path)
    if "kT_boundary" not in df.columns or "JA_boundary" not in df.columns:
        return pd.DataFrame(columns=["kT_boundary", "JA_boundary"])
    return df[np.isfinite(df["kT_boundary"]) & np.isfinite(df["JA_boundary"])].copy()


def _dataset_frame(dataset: Path) -> tuple[pd.DataFrame, list[str]]:
    missing: list[str] = []
    data = np.load(dataset, allow_pickle=True)
    x = np.asarray(data["x"], dtype=np.float64)
    y_reg = np.asarray(data["y_reg"], dtype=np.float64)
    y_phase = np.asarray(data["y_phase"], dtype=np.int64)
    df = pd.DataFrame(
        {
            "kBT": x[:, 0],
            "JA": x[:, 1],
            "phase_label": y_phase,
            "phase": [PHASE_NAMES.get(int(v), str(v)) for v in y_phase],
            "Delta_opt": y_reg[:, 0],
            "q_opt": y_reg[:, 1],
            "eta": y_reg[:, 2],
            "Ic_plus": y_reg[:, 3] if y_reg.shape[1] > 3 else np.nan,
            "Ic_minus": y_reg[:, 4] if y_reg.shape[1] > 4 else np.nan,
        }
    )
    optional_fields = [
        "q_min",
        "q_max",
        "n_q",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "delta_status",
        "delta_boundary_ambiguous",
        "delta_unresolved",
        "delta_boundary_band_normal",
        "free_energy_gap_to_normal",
        "positive_delta_gap",
        "trusted_exact",
        "training_eligible_exact",
        "needs_rerun_exact",
    ]
    for name in optional_fields:
        if name in data.files:
            df[name] = np.asarray(data[name])
        else:
            df[name] = np.nan
            missing.append(name)
    df["F_SC"] = np.nan
    df["F_normal"] = np.nan
    df["DeltaF"] = df["free_energy_gap_to_normal"]
    df["eta_denominator_diagnostic"] = np.abs(df["Ic_plus"]) + np.abs(df["Ic_minus"])
    df["abs_Ic_plus"] = np.abs(df["Ic_plus"])
    df["abs_Ic_minus"] = np.abs(df["Ic_minus"])
    df["min_abs_Ic"] = np.nanmin(np.column_stack([df["abs_Ic_plus"], df["abs_Ic_minus"]]), axis=1)
    df["coord_key"] = [f"{kt:.4f},{ja:.4f}" for kt, ja in zip(df["kBT"], df["JA"])]
    df["data_origin"] = "final_dataset"
    return df, missing


def _latest_oracle_frame(run_dir: Path) -> pd.DataFrame:
    iter_dirs = sorted([p for p in run_dir.glob("iter*") if p.is_dir()])
    if not iter_dirs:
        return pd.DataFrame()
    latest = iter_dirs[-1]
    try:
        iteration = int(latest.name.replace("iter", ""))
    except ValueError:
        iteration = -1
    merged = latest / f"exact_merged_iter{iteration:03d}.npz"
    if not merged.exists():
        return pd.DataFrame()
    data = np.load(merged, allow_pickle=True)
    required = ["kT", "JA", "delta_opt", "q_opt", "eta", "ic_plus", "ic_minus", "phase_candidate"]
    if any(k not in data.files for k in required):
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "kBT": np.asarray(data["kT"], dtype=np.float64),
            "JA": np.asarray(data["JA"], dtype=np.float64),
            "phase_label": np.asarray(data["phase_candidate"], dtype=np.int64),
            "Delta_opt": np.asarray(data["delta_opt"], dtype=np.float64),
            "q_opt": np.asarray(data["q_opt"], dtype=np.float64),
            "eta": np.asarray(data["eta"], dtype=np.float64),
            "Ic_plus": np.asarray(data["ic_plus"], dtype=np.float64),
            "Ic_minus": np.asarray(data["ic_minus"], dtype=np.float64),
        }
    )
    df["phase"] = [PHASE_NAMES.get(int(v), str(v)) for v in df["phase_label"]]
    optional = [
        "q_min",
        "q_max",
        "n_q",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "delta_status",
        "delta_boundary_ambiguous",
        "delta_unresolved",
        "delta_boundary_band_normal",
        "free_energy_gap_to_normal",
        "positive_delta_gap",
        "trusted_exact",
        "training_eligible_exact",
        "needs_rerun_exact",
    ]
    for name in optional:
        df[name] = np.asarray(data[name]) if name in data.files else np.nan
    df["F_SC"] = np.nan
    df["F_normal"] = np.nan
    df["DeltaF"] = df["free_energy_gap_to_normal"]
    df["eta_denominator_diagnostic"] = np.abs(df["Ic_plus"]) + np.abs(df["Ic_minus"])
    df["abs_Ic_plus"] = np.abs(df["Ic_plus"])
    df["abs_Ic_minus"] = np.abs(df["Ic_minus"])
    df["min_abs_Ic"] = np.nanmin(np.column_stack([df["abs_Ic_plus"], df["abs_Ic_minus"]]), axis=1)
    df["coord_key"] = [f"{kt:.4f},{ja:.4f}" for kt, ja in zip(df["kBT"], df["JA"])]
    df["source_iteration"] = iteration
    df["data_origin"] = "latest_exact_oracle"
    return df


def _normalized_distances(points: np.ndarray, boundary: pd.DataFrame) -> np.ndarray:
    if boundary.empty:
        return np.full(points.shape[0], np.nan, dtype=np.float64)
    b = boundary[["kT_boundary", "JA_boundary"]].to_numpy(dtype=np.float64)
    scale = np.array([KT_MAX - KT_MIN, JA_MAX - JA_MIN], dtype=np.float64)
    out = np.empty(points.shape[0], dtype=np.float64)
    for start in range(0, points.shape[0], 1024):
        block = points[start : start + 1024]
        d = (block[:, None, :] - b[None, :, :]) / scale[None, None, :]
        out[start : start + 1024] = np.sqrt(np.sum(d * d, axis=2)).min(axis=1)
    return out


def _polyline_distances(points: np.ndarray, curve: pd.DataFrame) -> np.ndarray:
    if curve.empty:
        return np.full(points.shape[0], np.nan, dtype=np.float64)
    c = curve[["kT_boundary", "JA_boundary"]].to_numpy(dtype=np.float64)
    if c.shape[0] == 1:
        return _normalized_distances(points, curve)
    scale = np.array([KT_MAX - KT_MIN, JA_MAX - JA_MIN], dtype=np.float64)
    p = points / scale[None, :]
    c = c / scale[None, :]
    best = np.full(points.shape[0], np.inf, dtype=np.float64)
    for i in range(c.shape[0] - 1):
        a = c[i]
        b = c[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-24:
            d = np.sqrt(np.sum((p - a[None, :]) ** 2, axis=1))
        else:
            t = np.clip(((p - a[None, :]) @ ab) / denom, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.sqrt(np.sum((p - proj) ** 2, axis=1))
        best = np.minimum(best, d)
    return best


def _selection_trace(run_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    keep = [
        "selection_source",
        "selection_pool",
        "kT",
        "JA",
        "A0_main",
        "A0_main_raw",
        "A0_for_pool",
        "Aselect",
        "final_score",
        "B_delta_raw",
        "U_NS",
        "B_delta_gated",
        "predicted_phase_before_exact",
    ]
    for iter_dir in sorted(run_dir.glob("iter*")):
        if not iter_dir.is_dir():
            continue
        try:
            iteration = int(iter_dir.name.replace("iter", ""))
        except ValueError:
            continue
        candidates = [iter_dir / "selected_points_by_pool.csv", iter_dir / "selected_points.csv"]
        for path in candidates:
            if not path.exists() or path.stat().st_size == 0:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            kt_col = "kT" if "kT" in df.columns else "kBT" if "kBT" in df.columns else None
            if kt_col is None or "JA" not in df.columns:
                continue
            out = pd.DataFrame({"kBT": df[kt_col].astype(float), "JA": df["JA"].astype(float)})
            for name in keep:
                out[name] = df[name] if name in df.columns else np.nan
            out["source_iteration"] = iteration
            rows.append(out)
            break
    if not rows:
        return pd.DataFrame(columns=["coord_key", "source_iteration"])
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows["coord_key"] = [f"{kt:.4f},{ja:.4f}" for kt, ja in zip(all_rows["kBT"], all_rows["JA"])]
    all_rows = all_rows.sort_values("source_iteration").drop_duplicates("coord_key", keep="first")
    return all_rows.drop(columns=["kBT", "JA"])


def _add_response_reliability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom_tol = 1.0e-5
    delta_response_min = 1.0e-3
    labels: list[str] = []
    for row in out.itertuples(index=False):
        trusted = bool(getattr(row, "trusted_exact", False))
        rerun = bool(getattr(row, "needs_rerun_exact", False))
        delta_amb = bool(getattr(row, "delta_boundary_ambiguous", False)) or bool(getattr(row, "delta_unresolved", False))
        q_bad = bool(getattr(row, "q_edge_hit", False)) or bool(getattr(row, "q_unresolved", False))
        near_normal = (
            float(getattr(row, "Delta_opt", np.nan)) < delta_response_min
            or int(getattr(row, "phase_label", -1)) == 0
            or bool(getattr(row, "delta_boundary_band_normal", False))
        )
        small_den = float(getattr(row, "eta_denominator_diagnostic", np.nan)) < denom_tol
        if trusted and not rerun and not delta_amb and not q_bad and not near_normal and not small_den:
            labels.append("clean")
        elif rerun:
            labels.append("rerun_required")
        elif q_bad:
            labels.append("q_window_suspect")
        elif delta_amb:
            labels.append("delta_ambiguous")
        elif near_normal:
            labels.append("near_normal_unreliable")
        elif small_den:
            labels.append("small_Ic_denominator")
        else:
            labels.append("uncategorized")
    out["response_reliability"] = labels
    out["eta_positive_strong"] = out["eta"] > 0.02
    return out


def _bool_sum(df: pd.DataFrame, col: str) -> int:
    if col not in df:
        return 0
    return int(np.nansum(df[col].astype(float).to_numpy() != 0))


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _plot_phase_background(ax, df: pd.DataFrame) -> None:
    colors = {0: "#cfcfcf", 1: "#2c7fb8", 2: "#d95f02"}
    for phase, color in colors.items():
        mask = df["phase_label"] == phase
        ax.scatter(df.loc[mask, "kBT"], df.loc[mask, "JA"], s=5, c=color, alpha=0.20, edgecolors="none", rasterized=True)


def _overlay_boundaries(ax, normal_sc: pd.DataFrame, uniform_fflo: pd.DataFrame) -> None:
    if not normal_sc.empty:
        nsc = normal_sc.sort_values(["kT_boundary", "JA_boundary"])
        ax.plot(nsc["kT_boundary"], nsc["JA_boundary"], c="black", lw=1.2, marker=".", ms=3, label="normal/SC")
    if not uniform_fflo.empty:
        ufflo = uniform_fflo.sort_values(["kT_boundary", "JA_boundary"])
        ax.scatter(ufflo["kT_boundary"], ufflo["JA_boundary"], c="#5e3c99", s=10, marker="s", label="uniform/FFLO")


def _finish_axes(ax, title: str) -> None:
    ax.set_xlim(0.0, 0.56)
    ax.set_ylim(0.0, 2.12)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)


def _write_figures(
    df: pd.DataFrame,
    high_kink: pd.DataFrame,
    eta_pos: pd.DataFrame,
    delta_amb: pd.DataFrame,
    rerun: pd.DataFrame,
    old_near: pd.DataFrame,
    normal_sc: pd.DataFrame,
    uniform_fflo: pd.DataFrame,
    fig_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    _plot_phase_background(ax, df)
    _overlay_boundaries(ax, normal_sc, uniform_fflo)
    classes = [
        ("clean", ~(high_kink["delta_boundary_ambiguous"].astype(bool) | high_kink["delta_unresolved"].astype(bool) | high_kink["needs_rerun_exact"].astype(bool) | high_kink["delta_boundary_band_normal"].astype(bool)), "tab:green", "o"),
        ("boundary-band normal", high_kink["delta_boundary_band_normal"].astype(bool), "tab:blue", "^"),
        ("delta ambiguous", high_kink["delta_boundary_ambiguous"].astype(bool) | high_kink["delta_unresolved"].astype(bool), "tab:orange", "s"),
        ("rerun required", high_kink["needs_rerun_exact"].astype(bool), "tab:red", "x"),
    ]
    for label, mask, color, marker in classes:
        sub = high_kink[mask]
        if not sub.empty:
            ax.scatter(sub["kBT"], sub["JA"], s=28, c=color, marker=marker, label=f"kink {label}", zorder=5)
    _finish_axes(ax, "High-JA normal/SC boundary kink audit")
    fig.savefig(fig_dir / "high_JA_boundary_kink_audit.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    eta_color = _signed_power(df["eta"].to_numpy(dtype=np.float64))
    lim = max(0.5, float(np.nanmax(np.abs(eta_color))) if eta_color.size else 0.5)
    sc = ax.scatter(
        df["kBT"],
        df["JA"],
        c=eta_color,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
        marker="s",
        s=7,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    markers = {
        "clean": "o",
        "near_normal_unreliable": "^",
        "small_Ic_denominator": "v",
        "delta_ambiguous": "s",
        "q_window_suspect": "P",
        "rerun_required": "x",
        "uncategorized": "D",
    }
    for reliability, marker in markers.items():
        sub = eta_pos[eta_pos["response_reliability"] == reliability]
        if not sub.empty:
            ax.scatter(sub["kBT"], sub["JA"], s=34, facecolors="none", edgecolors="black", marker=marker, label=reliability, zorder=5)
    _overlay_boundaries(ax, normal_sc, uniform_fflo)
    _finish_axes(ax, r"$J_A/t>1.25$, $\eta>0$ response audit")
    fig.colorbar(sc, ax=ax, pad=0.02, label=r"$\eta$ (signed-power color scale)")
    fig.savefig(fig_dir / "eta_positive_high_JA_audit.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.scatter(df["kBT"], df["JA"], s=5, c="0.80", alpha=0.35, edgecolors="none", rasterized=True, label="all exact")
    if not delta_amb.empty:
        ax.scatter(delta_amb["kBT"], delta_amb["JA"], s=16, c="tab:orange", alpha=0.85, label="Delta ambiguous")
    if not rerun.empty:
        ax.scatter(rerun["kBT"], rerun["JA"], s=22, c="tab:red", marker="x", label="rerun required")
    _overlay_boundaries(ax, normal_sc, uniform_fflo)
    _finish_axes(ax, "Delta ambiguity and rerun-required map")
    fig.savefig(fig_dir / "delta_ambiguous_map.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    if not delta_amb.empty:
        ax.hist(delta_amb["distance_to_normal_sc_boundary"].dropna(), bins=32, alpha=0.65, label="Delta ambiguous")
    if not rerun.empty:
        ax.hist(rerun["distance_to_normal_sc_boundary"].dropna(), bins=32, alpha=0.65, label="rerun required")
    ax.set_xlabel("normalized distance to normal/SC boundary")
    ax.set_ylabel("count")
    ax.set_title("Ambiguity distance to normal/SC boundary")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(fig_dir / "ambiguity_distance_hist.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.3), constrained_layout=True)
    sc = ax.scatter(
        df["kBT"],
        df["JA"],
        c=eta_color,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
        marker="s",
        s=7,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    curves = _finite_t_reference_boundaries()
    ax.plot(curves["old_cfflo_tfflo_1st"]["kT_boundary"], curves["old_cfflo_tfflo_1st"]["JA_boundary"], ":D", c="#d0001f", mfc="white", ms=4, label="old cFFLO-tFFLO 1st")
    ax.plot(curves["old_cfflo_tfflo_2nd"]["kT_boundary"], curves["old_cfflo_tfflo_2nd"]["JA_boundary"], "-.o", c="#2d6a4f", mfc="white", ms=3, label="old cFFLO-tFFLO 2nd")
    if not old_near.empty:
        ax.scatter(old_near["kBT"], old_near["JA"], s=26, facecolors="none", edgecolors="black", marker="o", label="near old curves")
    _finish_axes(ax, "Response near old topology-reference curves")
    fig.colorbar(sc, ax=ax, pad=0.02, label=r"$\eta$ (signed-power color scale)")
    fig.savefig(fig_dir / "old_topology_reference_response_audit.png", dpi=240)
    plt.close(fig)


def _fraction(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def _summary_counts(df: pd.DataFrame) -> dict[str, object]:
    n = int(df.shape[0])
    return {
        "count": n,
        "trusted": _bool_sum(df, "trusted_exact"),
        "trusted_fraction": _fraction(_bool_sum(df, "trusted_exact"), n),
        "delta_ambiguous": _bool_sum(df, "delta_boundary_ambiguous"),
        "delta_unresolved": _bool_sum(df, "delta_unresolved"),
        "boundary_band_normal": _bool_sum(df, "delta_boundary_band_normal"),
        "rerun_required": _bool_sum(df, "needs_rerun_exact"),
        "q_edge_hit": _bool_sum(df, "q_edge_hit"),
        "q_expanded": _bool_sum(df, "q_expanded"),
        "q_unresolved": _bool_sum(df, "q_unresolved"),
    }


def _add_distances(df: pd.DataFrame, normal_sc: pd.DataFrame, uniform_fflo: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    points = out[["kBT", "JA"]].to_numpy(dtype=np.float64)
    out["distance_to_normal_sc_boundary"] = _normalized_distances(points, normal_sc)
    out["distance_to_uniform_fflo_boundary"] = _normalized_distances(points, uniform_fflo)
    curves = _finite_t_reference_boundaries()
    out["distance_to_old_curve_1"] = _polyline_distances(points, curves["old_cfflo_tfflo_1st"])
    out["distance_to_old_curve_2"] = _polyline_distances(points, curves["old_cfflo_tfflo_2nd"])
    out["distance_to_old_cfflo_tfflo_reference"] = np.nanmin(
        out[["distance_to_old_curve_1", "distance_to_old_curve_2"]].to_numpy(dtype=np.float64), axis=1
    )
    return out


def _write_report(
    path: Path,
    summary: dict[str, object],
    high_kink: pd.DataFrame,
    eta_pos: pd.DataFrame,
    delta_amb: pd.DataFrame,
    rerun: pd.DataFrame,
    old_near: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Boundary and Anomaly Audit Report")
    lines.append("")
    lines.append(f"- Run ID: `{summary['run_id']}`")
    lines.append(f"- Final dataset: `{summary['final_dataset']}`")
    lines.append(f"- Current exact samples: {summary['n_exact_samples']}")
    lines.append(f"- Completed iterations: {summary['completed_iterations']}")
    lines.append(f"- Stop reason: `{summary['stop_reason']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report audits existing discovery active-learning outputs only. It does not modify the acquisition function, stochastic selection, StopController, neural networks, or exact BdG oracle.")
    lines.append("")
    lines.append("## Key Counts")
    lines.append("")
    for key in ["high_JA_boundary_kink_points", "eta_positive_high_JA_points", "delta_ambiguous_points", "rerun_required_points", "old_topology_reference_nearby_points"]:
        lines.append(f"- {key}: {summary[key]['count']}")
    lines.append(f"- latest exact-oracle rerun-required points: {summary['latest_exact_oracle']['rerun_required']}")
    lines.append(f"- latest exact-oracle Delta-unresolved points: {summary['latest_exact_oracle']['delta_unresolved']}")
    lines.append("")
    lines.append("## A. High-JA Normal/SC Boundary Kink Audit")
    lines.append("")
    lines.append(f"Selection rule: `JA > 1.1` and normalized distance to the final normal/SC boundary <= `{summary['thresholds']['high_ja_boundary_distance_tol']}`.")
    lines.append("")
    lines.append(json.dumps(summary["high_JA_boundary_kink_points"], indent=2))
    lines.append("")
    if high_kink.empty:
        lines.append("No high-JA kink audit points were found under the configured distance threshold.")
    else:
        clean = high_kink[
            high_kink["trusted_exact"].astype(bool)
            & ~high_kink["delta_boundary_ambiguous"].astype(bool)
            & ~high_kink["delta_unresolved"].astype(bool)
            & ~high_kink["needs_rerun_exact"].astype(bool)
        ]
        lines.append(f"Clean trusted non-ambiguous points in this subset: {clean.shape[0]} / {high_kink.shape[0]}.")
        if _bool_sum(high_kink, "delta_boundary_ambiguous") or _bool_sum(high_kink, "delta_unresolved"):
            lines.append("Delta ambiguity/unresolved metadata is present in the kink subset; the kink should be interpreted together with the finite-resolution normal/SC boundary-band logic.")
        if _bool_sum(high_kink, "q_edge_hit") or _bool_sum(high_kink, "q_unresolved"):
            lines.append("q-window failure metadata is present in this subset.")
        else:
            lines.append("q-edge or q-unresolved metadata is not common in this subset.")
    lines.append("")
    lines.append("## B. High-JA Positive-Eta Audit")
    lines.append("")
    lines.append("Selection rule: `JA > 1.25` and `eta > 0`; the `eta_positive_strong` flag marks `eta > 0.02`.")
    lines.append("")
    lines.append(json.dumps(summary["eta_positive_high_JA_points"], indent=2))
    lines.append("")
    if eta_pos.empty:
        lines.append("No high-JA positive-eta points were found.")
    else:
        reliability = eta_pos["response_reliability"].value_counts().to_dict()
        lines.append(f"Response reliability classification: `{reliability}`.")
        clean = int((eta_pos["response_reliability"] == "clean").sum())
        if clean == 0:
            lines.append("None of these points are classified as clean response signals under the audit criteria; they should not yet be interpreted as robust high-JA positive diode response.")
        else:
            lines.append(f"{clean} points are classified as clean; these deserve a focused physics check rather than being dismissed as numerical artifacts.")
    lines.append("")
    lines.append("## C. Delta Ambiguity and Rerun-Required Audit")
    lines.append("")
    lines.append(json.dumps({"delta_ambiguous": summary["delta_ambiguous_points"], "rerun_required": summary["rerun_required_points"]}, indent=2))
    lines.append("")
    if not delta_amb.empty:
        near = int((delta_amb["distance_to_normal_sc_boundary"] <= summary["thresholds"]["near_boundary_distance_tol"]).sum())
        lines.append(f"Delta-ambiguous/unresolved points within the near-boundary distance threshold: {near} / {delta_amb.shape[0]}.")
    if not rerun.empty:
        near = int((rerun["distance_to_normal_sc_boundary"] <= summary["thresholds"]["near_boundary_distance_tol"]).sum())
        lines.append(f"Rerun-required points within the near-boundary distance threshold: {near} / {rerun.shape[0]}.")
    lines.append("")
    lines.append("## D. Old cFFLO/tFFLO Reference-Curve Response Audit")
    lines.append("")
    lines.append("The old cFFLO/tFFLO curves are used only as reference curves. This is only a response correlation audit; no pointwise topology oracle was evaluated.")
    lines.append("")
    lines.append(json.dumps(summary["old_topology_reference_nearby_points"], indent=2))
    lines.append("")
    if old_near.empty:
        lines.append("No exact points fell within the configured old-curve distance tolerance.")
    else:
        pos = int((old_near["eta"] > 0).sum())
        strong = int((np.abs(old_near["eta"]) > 0.5).sum())
        clean = int((old_near["trusted_exact"].astype(bool) & ~old_near["needs_rerun_exact"].astype(bool) & ~old_near["delta_boundary_ambiguous"].astype(bool) & ~old_near["delta_unresolved"].astype(bool)).sum())
        lines.append(f"Near old reference curves: eta>0 count = {pos}, |eta|>0.5 count = {strong}, clean trusted non-ambiguous count = {clean}.")
    lines.append("")
    lines.append("## Numerical Cleanup Recommendation")
    lines.append("")
    if summary["eta_positive_high_JA_points"]["count"] and summary["eta_positive_high_JA_points"].get("clean_response", 0) == 0:
        lines.append("- High-JA positive-eta points are not clean under the current audit; prioritize numerical cleanup before interpreting them as physical response.")
    if summary["delta_ambiguous_points"]["count"] or summary["rerun_required_points"]["count"]:
        lines.append("- Rerun the listed Delta-ambiguous / rerun-required points with stricter near-zero Delta refinement, a finer positive-Delta scan, and explicit verification of `F_SC - F_normal`.")
        lines.append("- Mark eta as unreliable when `Delta_opt` is too close to zero or the current denominator diagnostic is too small.")
    if summary["q_window_issue_count"]:
        lines.append("- q-window metadata appears in suspicious subsets; use a larger q window and/or denser q grid for those points.")
    else:
        lines.append("- q-window edge failures do not appear to be the dominant explanation in the audited suspicious subsets.")
    lines.append("- If any high-JA positive-eta points remain clean after numerical cleanup, compare them against old topology-reference curves and consider adding a pointwise topology oracle.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for rel in summary["outputs"]:
        lines.append(f"- `{rel}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(ml_phase_root: Path, run_id: str) -> dict[str, object]:
    run_dir = ml_phase_root / "active_runs" / run_id
    reports_dir = ml_phase_root / "reports"
    tables_dir = reports_dir / "audit_tables"
    figures_dir = reports_dir / "audit_figures"
    final_dataset = run_dir / "dataset_iter020.npz"
    if not final_dataset.exists():
        candidates = sorted(run_dir.glob("dataset_iter*.npz"))
        if not candidates:
            raise FileNotFoundError(f"No dataset_iter*.npz found under {run_dir}")
        final_dataset = candidates[-1]

    df, missing_fields = _dataset_frame(final_dataset)
    selection = _selection_trace(run_dir)
    if not selection.empty:
        df = df.merge(selection, on="coord_key", how="left")
    else:
        for name in [
            "source_iteration",
            "selection_source",
            "A0_main",
            "A0_main_raw",
            "A0_for_pool",
            "Aselect",
            "B_delta_raw",
            "U_NS",
            "B_delta_gated",
            "predicted_phase_before_exact",
        ]:
            df[name] = np.nan
        missing_fields.append("selected point score trace")

    boundary_dir = ml_phase_root / "figures" / f"{run_id}_final_exact_boundaries"
    normal_sc = _load_boundary(boundary_dir / "normal_sc_boundary_segments.csv")
    uniform_fflo = _load_boundary(boundary_dir / "uniform_fflo_boundary_segments.csv")
    if normal_sc.empty or uniform_fflo.empty:
        latest_iter_dirs = sorted([p for p in run_dir.glob("iter*") if p.is_dir()])
        if latest_iter_dirs:
            fallback = latest_iter_dirs[-1] / "boundaries"
            if normal_sc.empty:
                normal_sc = _load_boundary(fallback / "normal_sc_boundary_segments.csv")
            if uniform_fflo.empty:
                uniform_fflo = _load_boundary(fallback / "uniform_fflo_boundary_segments.csv")

    df = _add_distances(df, normal_sc, uniform_fflo)
    df = _add_response_reliability(df)
    latest_oracle = _latest_oracle_frame(run_dir)
    if not latest_oracle.empty:
        latest_oracle = _add_distances(latest_oracle, normal_sc, uniform_fflo)
        latest_oracle = _add_response_reliability(latest_oracle)
        if not selection.empty:
            latest_oracle = latest_oracle.drop(
                columns=[c for c in selection.columns if c in latest_oracle.columns and c not in {"coord_key", "source_iteration"}],
                errors="ignore",
            ).merge(selection, on="coord_key", how="left", suffixes=("", "_selected"))
            if "source_iteration_selected" in latest_oracle:
                latest_oracle["source_iteration"] = latest_oracle["source_iteration"].fillna(
                    latest_oracle["source_iteration_selected"]
                )
                latest_oracle = latest_oracle.drop(columns=["source_iteration_selected"])

    high_ja_tol = 0.025
    near_boundary_tol = 0.01
    old_curve_tol = 0.012
    high_kink = df[(df["JA"] > 1.1) & (df["distance_to_normal_sc_boundary"] <= high_ja_tol)].copy()
    eta_pos = df[(df["JA"] > 1.25) & (df["eta"] > 0)].copy()
    delta_amb_sources = [df[df["delta_boundary_ambiguous"].astype(bool) | df["delta_unresolved"].astype(bool)].copy()]
    if not latest_oracle.empty:
        delta_amb_sources.append(
            latest_oracle[
                latest_oracle["delta_boundary_ambiguous"].astype(bool) | latest_oracle["delta_unresolved"].astype(bool)
            ].copy()
        )
    delta_amb = pd.concat(delta_amb_sources, ignore_index=True).drop_duplicates("coord_key", keep="last")
    rerun = (
        latest_oracle[latest_oracle["needs_rerun_exact"].astype(bool)].copy()
        if not latest_oracle.empty
        else df[df["needs_rerun_exact"].astype(bool)].copy()
    )
    old_near = df[df["distance_to_old_cfflo_tfflo_reference"] <= old_curve_tol].copy()

    field_order = [
        "kBT",
        "JA",
        "phase",
        "phase_label",
        "predicted_phase_before_exact",
        "distance_to_normal_sc_boundary",
        "distance_to_uniform_fflo_boundary",
        "distance_to_old_curve_1",
        "distance_to_old_curve_2",
        "Delta_opt",
        "q_opt",
        "eta",
        "Ic_plus",
        "Ic_minus",
        "eta_denominator_diagnostic",
        "abs_Ic_plus",
        "abs_Ic_minus",
        "min_abs_Ic",
        "F_SC",
        "F_normal",
        "DeltaF",
        "free_energy_gap_to_normal",
        "positive_delta_gap",
        "delta_status",
        "delta_boundary_ambiguous",
        "delta_unresolved",
        "delta_unresolved_requiring_rerun",
        "delta_boundary_band_normal",
        "needs_rerun_exact",
        "q_min",
        "q_max",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "trusted_exact",
        "training_eligible_exact",
        "source_iteration",
        "selection_source",
        "A0_main",
        "A0_main_raw",
        "A0_for_pool",
        "Aselect",
        "B_delta_raw",
        "U_NS",
        "B_delta_gated",
        "response_reliability",
        "eta_positive_strong",
    ]
    for table in [df, high_kink, eta_pos, delta_amb, rerun, old_near]:
        if "delta_unresolved_requiring_rerun" not in table.columns:
            table["delta_unresolved_requiring_rerun"] = (
                table.get("delta_unresolved", pd.Series(False, index=table.index)).astype(bool)
                & table.get("needs_rerun_exact", pd.Series(False, index=table.index)).astype(bool)
            )
    existing_order = [c for c in field_order if c in df.columns]
    _write_csv(high_kink[existing_order], tables_dir / "high_JA_boundary_kink_points.csv")
    _write_csv(eta_pos[existing_order], tables_dir / "eta_positive_high_JA_points.csv")
    _write_csv(delta_amb[existing_order], tables_dir / "delta_ambiguous_points.csv")
    _write_csv(rerun[existing_order], tables_dir / "rerun_required_points.csv")
    _write_csv(old_near[existing_order], tables_dir / "old_topology_reference_nearby_points.csv")

    _write_figures(df, high_kink, eta_pos, delta_amb, rerun, old_near, normal_sc, uniform_fflo, figures_dir)

    final_iter = int(final_dataset.stem.replace("dataset_iter", ""))
    stop_state = _load_json(run_dir / "stop_state.json", {})
    summary = {
        "run_id": run_id,
        "ml_phase_root": str(ml_phase_root),
        "final_dataset": str(final_dataset),
        "completed_iterations": final_iter,
        "n_exact_samples": int(df.shape[0]),
        "phase_counts": df["phase"].value_counts().to_dict(),
        "stop_reason": stop_state.get("stop_reason", "N/A"),
        "missing_fields": sorted(set(missing_fields)),
        "thresholds": {
            "high_ja_boundary_distance_tol": high_ja_tol,
            "near_boundary_distance_tol": near_boundary_tol,
            "old_curve_distance_tol": old_curve_tol,
            "eta_small_threshold": 0.02,
        },
        "boundary_summary": _load_json(boundary_dir / "boundary_summary.json", {}),
        "latest_exact_oracle": {
            "available": not latest_oracle.empty,
            "count": int(latest_oracle.shape[0]) if not latest_oracle.empty else 0,
            "delta_ambiguous": _bool_sum(latest_oracle, "delta_boundary_ambiguous") if not latest_oracle.empty else 0,
            "delta_unresolved": _bool_sum(latest_oracle, "delta_unresolved") if not latest_oracle.empty else 0,
            "delta_unresolved_requiring_rerun": int(
                (
                    latest_oracle["delta_unresolved"].astype(bool)
                    & latest_oracle["needs_rerun_exact"].astype(bool)
                ).sum()
            )
            if not latest_oracle.empty
            else 0,
            "rerun_required": _bool_sum(latest_oracle, "needs_rerun_exact") if not latest_oracle.empty else 0,
            "trusted": _bool_sum(latest_oracle, "trusted_exact") if not latest_oracle.empty else 0,
            "q_edge_hit": _bool_sum(latest_oracle, "q_edge_hit") if not latest_oracle.empty else 0,
            "q_expanded": _bool_sum(latest_oracle, "q_expanded") if not latest_oracle.empty else 0,
            "q_unresolved": _bool_sum(latest_oracle, "q_unresolved") if not latest_oracle.empty else 0,
        },
        "high_JA_boundary_kink_points": _summary_counts(high_kink),
        "eta_positive_high_JA_points": {
            **_summary_counts(eta_pos),
            "eta_gt_0p02": int((eta_pos["eta"] > 0.02).sum()) if not eta_pos.empty else 0,
            "clean_response": int((eta_pos["response_reliability"] == "clean").sum()) if not eta_pos.empty else 0,
            "response_reliability_counts": eta_pos["response_reliability"].value_counts().to_dict(),
        },
        "delta_ambiguous_points": _summary_counts(delta_amb),
        "rerun_required_points": _summary_counts(rerun),
        "old_topology_reference_nearby_points": _summary_counts(old_near),
        "q_window_issue_count": int(
            _bool_sum(high_kink, "q_edge_hit")
            + _bool_sum(high_kink, "q_unresolved")
            + _bool_sum(eta_pos, "q_edge_hit")
            + _bool_sum(eta_pos, "q_unresolved")
        ),
        "outputs": [
            "reports/boundary_and_anomaly_audit_report.md",
            "reports/audit_tables/high_JA_boundary_kink_points.csv",
            "reports/audit_tables/eta_positive_high_JA_points.csv",
            "reports/audit_tables/delta_ambiguous_points.csv",
            "reports/audit_tables/rerun_required_points.csv",
            "reports/audit_tables/old_topology_reference_nearby_points.csv",
            "reports/audit_tables/boundary_audit_summary.json",
            "reports/audit_figures/high_JA_boundary_kink_audit.png",
            "reports/audit_figures/eta_positive_high_JA_audit.png",
            "reports/audit_figures/delta_ambiguous_map.png",
            "reports/audit_figures/ambiguity_distance_hist.png",
            "reports/audit_figures/old_topology_reference_response_audit.png",
        ],
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "boundary_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(reports_dir / "boundary_and_anomaly_audit_report.md", summary, high_kink, eta_pos, delta_amb, rerun, old_near)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit boundary and response anomalies in a discovery active-learning run.")
    parser.add_argument("--ml-phase-root", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="active_boundary_discovery_512seed_256x50")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_audit(args.ml_phase_root, args.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
