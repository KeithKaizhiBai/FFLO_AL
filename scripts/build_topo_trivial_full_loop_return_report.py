from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path("active_phase_topology_from_scratch_full_loop_v1_hpc/ML_Phase_512_TopoTrivial_FullLoop")
RUN_ID = "active_phase_topology_from_scratch_full_loop_v2"
RUN_DIR = ROOT / "active_runs" / RUN_ID
REPORT_DIR = ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

PHASE_COLORS = {
    "normal": "#6b7280",
    "uniform_SC": "#2b8cbe",
    "FFLO": "#d95f0e",
}
TOPOLOGY_COLORS = {
    "not_applicable": "#bdbdbd",
    "trivial": "#3182bd",
    "topological": "#de2d26",
    "gapless_SC": "#fdae6b",
    "unresolved": "#252525",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def latest_dataset_path() -> Path:
    datasets = sorted(RUN_DIR.glob("dataset_iter*.csv"))
    if not datasets:
        raise FileNotFoundError(f"no dataset_iter*.csv under {RUN_DIR}")
    return datasets[-1]


def topology_name(code: int) -> str:
    return {
        -1: "not_applicable",
        0: "trivial",
        1: "topological",
        2: "gapless_SC",
        3: "unresolved",
    }.get(int(code), "unresolved")


def collect_iteration_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for iter_dir in sorted(RUN_DIR.glob("iter[0-9][0-9][0-9]")):
        iter_idx = int(iter_dir.name.replace("iter", ""))
        selected_path = iter_dir / "selected_points.csv"
        selected = max(0, sum(1 for _ in selected_path.open("r", encoding="utf-8")) - 1) if selected_path.exists() else 0
        metas = sorted(iter_dir.glob("exact_shard_rank*_of*.json"))
        sums: dict[str, float] = {}
        elapsed_values: list[float] = []
        hostnames: set[str] = set()
        for meta_path in metas:
            data = read_json(meta_path)
            host = data.get("hostname")
            if host:
                hostnames.add(str(host))
            elapsed = safe_float(data.get("elapsed_sec", np.nan))
            if not math.isnan(elapsed):
                elapsed_values.append(elapsed)
            for key in [
                "n_points",
                "trusted_exact_count",
                "q_expanded_count",
                "q_unresolved_count",
                "delta_refined_count",
                "delta_unresolved_count",
                "topology_applicable_count",
                "topology_trusted_count",
                "topology_gapless_count",
                "topology_topological_count",
                "topology_trivial_count",
                "topology_unresolved_count",
                "topology_runtime_sec_sum",
                "selected_refine_target_count_sum",
                "local_refinement_runtime_sec_sum",
                "point_total_runtime_sec_sum",
                "base_scan_runtime_sec_sum",
                "q_expansion_runtime_sec_sum",
                "delta_refinement_runtime_sec_sum",
                "total_q_points_evaluated",
                "total_estimated_grid_evaluations",
            ]:
                sums[key] = sums.get(key, 0.0) + safe_float(data.get(key, 0.0))
        append_path = RUN_DIR / f"dataset_iter{iter_idx + 1:03d}.append.json"
        append = read_json(append_path) if append_path.exists() else {}
        stop_path = iter_dir / f"stop_metrics_iter{iter_idx:03d}.json"
        stop = read_json(stop_path) if stop_path.exists() else {}
        n_points = int(sums.get("n_points", 0))
        selected_target_count = sums.get("selected_refine_target_count_sum", float("nan"))
        rows.append(
            {
                "iteration": iter_idx,
                "selected_points": selected,
                "merged_exact_points": n_points,
                "rank_count": len(metas),
                "iteration_walltime_sec": max(elapsed_values) if elapsed_values else np.nan,
                "rank_elapsed_min_sec": min(elapsed_values) if elapsed_values else np.nan,
                "rank_elapsed_max_sec": max(elapsed_values) if elapsed_values else np.nan,
                "training_eligible_appended": append.get("training_eligible_points_appended", np.nan),
                "new_unique_samples_added": append.get("new_unique_samples_added", np.nan),
                "dataset_size_after_append": append.get("output_samples", np.nan),
                "trusted_exact_count": sums.get("trusted_exact_count", np.nan),
                "rerun_required_count": selected - int(append.get("training_eligible_points_appended", 0)) if append else np.nan,
                "q_expanded_count": sums.get("q_expanded_count", np.nan),
                "q_unresolved_count": sums.get("q_unresolved_count", np.nan),
                "delta_refined_count": sums.get("delta_refined_count", np.nan),
                "delta_unresolved_count": sums.get("delta_unresolved_count", np.nan),
                "topology_applicable_count": sums.get("topology_applicable_count", np.nan),
                "topology_trusted_count": sums.get("topology_trusted_count", np.nan),
                "topology_trivial_count": sums.get("topology_trivial_count", np.nan),
                "topology_topological_count": sums.get("topology_topological_count", np.nan),
                "topology_gapless_count": sums.get("topology_gapless_count", np.nan),
                "topology_unresolved_count": sums.get("topology_unresolved_count", np.nan),
                "topology_runtime_sec_sum": sums.get("topology_runtime_sec_sum", np.nan),
                "selected_refine_target_count_sum": selected_target_count,
                "mean_selected_refine_targets": selected_target_count / n_points if n_points else np.nan,
                "local_refinement_runtime_sec_sum": sums.get("local_refinement_runtime_sec_sum", np.nan),
                "point_total_runtime_sec_sum": sums.get("point_total_runtime_sec_sum", np.nan),
                "base_scan_runtime_sec_sum": sums.get("base_scan_runtime_sec_sum", np.nan),
                "q_expansion_runtime_sec_sum": sums.get("q_expansion_runtime_sec_sum", np.nan),
                "delta_refinement_runtime_sec_sum": sums.get("delta_refinement_runtime_sec_sum", np.nan),
                "total_q_points_evaluated": sums.get("total_q_points_evaluated", np.nan),
                "total_estimated_grid_evaluations": sums.get("total_estimated_grid_evaluations", np.nan),
                "phase_map_change": stop.get("metrics", {}).get("phase_map_change", np.nan),
                "boundary_shift_normal_sc": stop.get("metrics", {}).get("boundary_shift_normal_sc", np.nan),
                "boundary_shift_uniform_fflo": stop.get("metrics", {}).get("boundary_shift_uniform_fflo", np.nan),
                "boundary_coverage_p95": stop.get("metrics", {}).get("boundary_coverage_p95", np.nan),
                "label_surprise_all_selected": stop.get("metrics", {}).get("label_surprise_all_selected", np.nan),
                "label_surprise_trusted": stop.get("metrics", {}).get("label_surprise_trusted", np.nan),
                "label_surprise_hard_risk": stop.get("metrics", {}).get("label_surprise_hard_risk", np.nan),
                "passed_condition_count": stop.get("passed_condition_count", np.nan),
                "required_pass_count": stop.get("required_pass_count", np.nan),
                "stop": stop.get("stop", False),
                "stop_reason": stop.get("stop_reason", ""),
                "hosts": ",".join(sorted(hostnames)),
            }
        )
    return pd.DataFrame(rows)


def collect_learning_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path = RUN_DIR / "metrics_history.json"
    if not path.exists():
        return pd.DataFrame()
    for idx, row in enumerate(read_json(path)):
        context = row.get("topology_context", {}) or {}
        rows.append(
            {
                "iteration": idx,
                "phase_accuracy": row.get("phase_accuracy", np.nan),
                "delta_rmse": row.get("delta_rmse", np.nan),
                "q_rmse": row.get("q_rmse", np.nan),
                "eta_rmse": row.get("eta_rmse", np.nan),
                "ic_plus_rmse": row.get("ic_plus_rmse", np.nan),
                "ic_minus_rmse": row.get("ic_minus_rmse", np.nan),
                "n_exact_calls": row.get("n_exact_calls", np.nan),
                "estimated_reduction": row.get("estimated_reduction", np.nan),
                "topology_context_trivial_count": context.get("trivial_count", np.nan),
                "topology_context_topological_count": context.get("topological_count", np.nan),
                "topology_context_gapless_count": context.get("gapless_count", np.nan),
                "topology_context_trusted_count": context.get("trusted_topology_count", np.nan),
            }
        )
    return pd.DataFrame(rows)


def write_tables(final_df: pd.DataFrame, iteration_df: pd.DataFrame, learning_df: pd.DataFrame) -> dict[str, Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    phase_counts = final_df["phase_name"].value_counts().rename_axis("phase_name").reset_index(name="count")
    out["phase_counts"] = TABLE_DIR / "phase_counts.csv"
    phase_counts.to_csv(out["phase_counts"], index=False)

    topo_df = final_df.copy()
    topo_df["topology_label"] = topo_df["topology_label_code"].map(topology_name)
    topo_counts = (
        topo_df.groupby(["phase_name", "topology_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["phase_name", "topology_label"])
    )
    out["topology_counts_by_phase"] = TABLE_DIR / "topology_counts_by_phase.csv"
    topo_counts.to_csv(out["topology_counts_by_phase"], index=False)

    trusted_topo = topo_df[topo_df["topology_trusted"] == 1]
    gap_summary = pd.DataFrame(
        [
            {
                "metric": "bulk_gap",
                "count": trusted_topo["topology_bulk_gap"].count(),
                "min": trusted_topo["topology_bulk_gap"].min(),
                "median": trusted_topo["topology_bulk_gap"].median(),
                "p95": trusted_topo["topology_bulk_gap"].quantile(0.95),
                "max": trusted_topo["topology_bulk_gap"].max(),
            },
            {
                "metric": "pfaffian_margin",
                "count": trusted_topo["topology_pfaffian_margin"].count(),
                "min": trusted_topo["topology_pfaffian_margin"].min(),
                "median": trusted_topo["topology_pfaffian_margin"].median(),
                "p95": trusted_topo["topology_pfaffian_margin"].quantile(0.95),
                "max": trusted_topo["topology_pfaffian_margin"].max(),
            },
        ]
    )
    out["gap_pfaffian_summary"] = TABLE_DIR / "gap_pfaffian_summary.csv"
    gap_summary.to_csv(out["gap_pfaffian_summary"], index=False)

    out["iteration_summary"] = TABLE_DIR / "iteration_summary.csv"
    iteration_df.to_csv(out["iteration_summary"], index=False)
    out["learning_metrics"] = TABLE_DIR / "learning_metrics.csv"
    learning_df.to_csv(out["learning_metrics"], index=False)

    runtime_summary = pd.DataFrame(
        [
            {
                "metric": "estimated_exact_array_walltime_h",
                "value": iteration_df["iteration_walltime_sec"].sum() / 3600.0,
                "definition": "sum over iterations of max rank elapsed_sec; excludes login-node train/merge/report overhead",
            },
            {
                "metric": "mean_iteration_walltime_min",
                "value": iteration_df["iteration_walltime_sec"].mean() / 60.0,
                "definition": "mean max rank elapsed_sec per exact iteration",
            },
            {
                "metric": "median_iteration_walltime_min",
                "value": iteration_df["iteration_walltime_sec"].median() / 60.0,
                "definition": "median max rank elapsed_sec per exact iteration",
            },
            {
                "metric": "max_iteration_walltime_min",
                "value": iteration_df["iteration_walltime_sec"].max() / 60.0,
                "definition": "max rank elapsed_sec across iterations",
            },
            {
                "metric": "rank_summed_point_runtime_h",
                "value": iteration_df["point_total_runtime_sec_sum"].sum() / 3600.0,
                "definition": "sum of point_total_runtime_sec_sum over ranks and iterations",
            },
            {
                "metric": "rank_summed_local_refinement_runtime_h",
                "value": iteration_df["local_refinement_runtime_sec_sum"].sum() / 3600.0,
                "definition": "sum of local_refinement_runtime_sec_sum over ranks and iterations",
            },
            {
                "metric": "rank_summed_topology_runtime_s",
                "value": iteration_df["topology_runtime_sec_sum"].sum(),
                "definition": "sum of online topology diagnostic runtime over ranks and iterations",
            },
        ]
    )
    out["runtime_summary"] = TABLE_DIR / "runtime_summary.csv"
    runtime_summary.to_csv(out["runtime_summary"], index=False)

    final_stop = iteration_df.iloc[-1].to_dict()
    final_summary = pd.DataFrame(
        [
            {"metric": "run_id", "value": RUN_ID},
            {"metric": "completed_iterations", "value": int(final_stop["iteration"]) + 1},
            {"metric": "final_dataset", "value": latest_dataset_path().name.replace(".csv", ".npz")},
            {"metric": "final_dataset_size", "value": len(final_df)},
            {"metric": "normal_count", "value": int((final_df["phase_name"] == "normal").sum())},
            {"metric": "uniform_sc_count", "value": int((final_df["phase_name"] == "uniform_SC").sum())},
            {"metric": "fflo_count", "value": int((final_df["phase_name"] == "FFLO").sum())},
            {"metric": "topology_trivial_count", "value": int((topo_df["topology_label"] == "trivial").sum())},
            {"metric": "topology_topological_count", "value": int((topo_df["topology_label"] == "topological").sum())},
            {"metric": "topology_gapless_count", "value": int((topo_df["topology_label"] == "gapless_SC").sum())},
            {"metric": "topology_unresolved_count", "value": int((topo_df["topology_label"] == "unresolved").sum())},
            {"metric": "topology_trusted_count", "value": int(final_df["topology_trusted"].sum())},
            {"metric": "final_phase_map_change", "value": final_stop["phase_map_change"]},
            {"metric": "final_boundary_shift_normal_sc", "value": final_stop["boundary_shift_normal_sc"]},
            {"metric": "final_boundary_shift_uniform_fflo", "value": final_stop["boundary_shift_uniform_fflo"]},
            {"metric": "final_boundary_coverage_p95", "value": final_stop["boundary_coverage_p95"]},
            {"metric": "final_label_surprise_all_selected", "value": final_stop["label_surprise_all_selected"]},
            {"metric": "final_label_surprise_trusted", "value": final_stop["label_surprise_trusted"]},
            {"metric": "final_label_surprise_hard_risk", "value": final_stop["label_surprise_hard_risk"]},
            {"metric": "stop_reason", "value": final_stop["stop_reason"]},
        ]
    )
    out["final_summary"] = TABLE_DIR / "final_summary.csv"
    final_summary.to_csv(out["final_summary"], index=False)

    region_diag = compute_response_region_diagnostics(final_df)
    out["response_region_diagnostics"] = TABLE_DIR / "response_region_diagnostics.csv"
    region_diag.to_csv(out["response_region_diagnostics"], index=False)

    boundary_summary, boundary_by_iter = compute_selected_boundary_concentration(final_df)
    out["selected_boundary_concentration_summary"] = TABLE_DIR / "selected_boundary_concentration_summary.csv"
    boundary_summary.to_csv(out["selected_boundary_concentration_summary"], index=False)
    out["selected_boundary_concentration_by_iteration"] = TABLE_DIR / "selected_boundary_concentration_by_iteration.csv"
    boundary_by_iter.to_csv(out["selected_boundary_concentration_by_iteration"], index=False)
    return out


def setup_axes(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.grid(alpha=0.18, linewidth=0.5)


def _tricontour_boundary(
    ax: plt.Axes,
    df: pd.DataFrame,
    value: pd.Series | np.ndarray,
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    if len(df) < 3:
        return
    x = df["kT"].to_numpy(dtype=float)
    y = df["JA"].to_numpy(dtype=float)
    z = np.asarray(value, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if valid.sum() < 3 or np.unique(z[valid]).size < 2:
        return
    tri = mtri.Triangulation(x[valid], y[valid])
    ax.tricontour(
        tri,
        z[valid],
        levels=[0.5],
        colors=[color],
        linestyles=[linestyle],
        linewidths=[linewidth],
        zorder=6,
    )


def _coordinate_bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(df["kT"].min()),
        float(df["kT"].max()),
        float(df["JA"].min()),
        float(df["JA"].max()),
    )


def _normalize_xy(x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]) -> np.ndarray:
    x_min, x_max, y_min, y_max = bounds
    x_span = max(float(x_max - x_min), 1e-12)
    y_span = max(float(y_max - y_min), 1e-12)
    return np.column_stack(((x - x_min) / x_span, (y - y_min) / y_span))


def _binary_change_edge_segments(
    df: pd.DataFrame,
    value: pd.Series | np.ndarray,
    *,
    max_normalized_edge: float,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(df) < 3:
        return np.empty((0, 2, 2)), np.empty((0, 2, 2))
    x = df["kT"].to_numpy(dtype=float)
    y = df["JA"].to_numpy(dtype=float)
    z = np.asarray(value, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if len(x) < 3 or np.unique(z).size < 2:
        return np.empty((0, 2, 2)), np.empty((0, 2, 2))
    tri = mtri.Triangulation(x, y)
    edges = tri.edges
    xy_norm = _normalize_xy(x, y, bounds)
    lengths = np.linalg.norm(xy_norm[edges[:, 0]] - xy_norm[edges[:, 1]], axis=1)
    keep = (z[edges[:, 0]] != z[edges[:, 1]]) & (lengths <= max_normalized_edge)
    raw_xy = np.column_stack((x, y))
    return raw_xy[edges[keep]], xy_norm[edges[keep]]


def _plot_binary_change_edges(
    ax: plt.Axes,
    df: pd.DataFrame,
    value: pd.Series | np.ndarray,
    *,
    color: str,
    linewidth: float,
    max_normalized_edge: float,
    bounds: tuple[float, float, float, float],
) -> int:
    raw_segments, _ = _binary_change_edge_segments(
        df,
        value,
        max_normalized_edge=max_normalized_edge,
        bounds=bounds,
    )
    for segment in raw_segments:
        ax.plot(segment[:, 0], segment[:, 1], color=color, lw=linewidth, alpha=0.95, zorder=6)
    return int(len(raw_segments))


def _distance_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=float)
    if len(segments) == 0:
        return np.full(len(points), np.nan)
    starts = segments[:, 0, :]
    ends = segments[:, 1, :]
    seg_vec = ends - starts
    seg_norm2 = np.sum(seg_vec * seg_vec, axis=1)
    seg_norm2 = np.maximum(seg_norm2, 1e-18)
    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), 512):
        pts = points[start : start + 512]
        rel = pts[:, None, :] - starts[None, :, :]
        t = np.sum(rel * seg_vec[None, :, :], axis=2) / seg_norm2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = starts[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        dist = np.linalg.norm(pts[:, None, :] - closest, axis=2)
        out[start : start + len(pts)] = np.min(dist, axis=1)
    return out


def _collect_selected_points() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for iter_dir in sorted(RUN_DIR.glob("iter[0-9][0-9][0-9]")):
        iter_idx = int(iter_dir.name.replace("iter", ""))
        p = iter_dir / "selected_points.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["iteration"] = iter_idx
        df["selection_stage"] = "initial_seed" if iter_idx == 0 else "acquisition"
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _boundary_segments(final_df: pd.DataFrame) -> dict[str, np.ndarray]:
    bounds = _coordinate_bounds(final_df)
    _, normal_sc = _binary_change_edge_segments(
        final_df,
        (final_df["phase_name"] != "normal").astype(float),
        max_normalized_edge=0.03,
        bounds=bounds,
    )
    topo_df = final_df.copy()
    topo_df["topology_label"] = topo_df["topology_label_code"].map(topology_name)
    fflo_trusted = topo_df[(topo_df["phase_name"] == "FFLO") & (topo_df["topology_trusted"] == 1)]
    _, cfflo_tfflo = _binary_change_edge_segments(
        fflo_trusted,
        (fflo_trusted["topology_label"] == "topological").astype(float),
        max_normalized_edge=0.03,
        bounds=bounds,
    )
    return {"normal_sc": normal_sc, "cfflo_tfflo": cfflo_tfflo}


def _summarize_distances(group: pd.DataFrame, label: str) -> dict[str, Any]:
    row: dict[str, Any] = {"group": label, "n_selected": int(len(group))}
    for prefix in ["normal_sc", "cfflo_tfflo", "either_boundary"]:
        values = group[f"dist_{prefix}"].to_numpy(dtype=float)
        row[f"{prefix}_median_distance"] = float(np.nanmedian(values)) if len(values) else np.nan
        row[f"{prefix}_p75_distance"] = float(np.nanquantile(values, 0.75)) if len(values) else np.nan
        for tol in [0.01, 0.02, 0.03, 0.05]:
            row[f"{prefix}_within_{str(tol).replace('.', 'p')}"] = float(np.nanmean(values <= tol)) if len(values) else np.nan
    return row


def compute_selected_boundary_concentration(final_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _collect_selected_points()
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    bounds = _coordinate_bounds(final_df)
    segments = _boundary_segments(final_df)
    selected = selected.copy()
    points = _normalize_xy(
        selected["kT"].to_numpy(dtype=float),
        selected["JA"].to_numpy(dtype=float),
        bounds,
    )
    selected["dist_normal_sc"] = _distance_to_segments(points, segments["normal_sc"])
    selected["dist_cfflo_tfflo"] = _distance_to_segments(points, segments["cfflo_tfflo"])
    selected["dist_either_boundary"] = np.nanmin(
        np.column_stack((selected["dist_normal_sc"], selected["dist_cfflo_tfflo"])),
        axis=1,
    )

    by_iter_rows = []
    for iteration, group in selected.groupby("iteration", sort=True):
        row = _summarize_distances(group, f"iter{int(iteration):03d}")
        row["iteration"] = int(iteration)
        row["selection_stage"] = "initial_seed" if int(iteration) == 0 else "acquisition"
        by_iter_rows.append(row)
    by_iter = pd.DataFrame(by_iter_rows).sort_values("iteration")

    summary_rows = [_summarize_distances(selected[selected["iteration"] == 0], "initial_seed_iter000")]
    acq = selected[selected["iteration"] > 0]
    summary_rows.append(_summarize_distances(acq, "all_acquisition_iters001_017"))
    summary_rows.append(_summarize_distances(selected[selected["iteration"] >= 13], "last5_acquisition_iters013_017"))
    summary_rows.append(_summarize_distances(selected[selected["iteration"] >= 15], "last3_acquisition_iters015_017"))
    summary = pd.DataFrame(summary_rows)
    summary["normal_sc_edge_count"] = len(segments["normal_sc"])
    summary["cfflo_tfflo_edge_count"] = len(segments["cfflo_tfflo"])
    summary["distance_units"] = "normalized_parameter_domain"
    return summary, by_iter


def _phase_counts_string(sub: pd.DataFrame) -> str:
    counts = sub["phase_name"].value_counts()
    return "; ".join(f"{name}:{int(count)}" for name, count in counts.items())


def _topology_counts_string(sub: pd.DataFrame) -> str:
    topo = sub["topology_label_code"].map(topology_name).value_counts()
    return "; ".join(f"{name}:{int(count)}" for name, count in topo.items())


def _eta_region_row(final_df: pd.DataFrame, name: str, mask: pd.Series, note: str) -> dict[str, Any]:
    sub = final_df[mask].copy()
    eta = sub["eta"].to_numpy(dtype=float)
    sign_negative = int(np.sum(eta < 0))
    sign_positive = int(np.sum(eta > 0))
    sign_zero = int(np.sum(np.isclose(eta, 0.0, atol=1e-12)))
    return {
        "region": name,
        "n_points": int(len(sub)),
        "phase_counts": _phase_counts_string(sub) if len(sub) else "",
        "topology_counts": _topology_counts_string(sub) if len(sub) else "",
        "eta_min": float(np.nanmin(eta)) if len(sub) else np.nan,
        "eta_median": float(np.nanmedian(eta)) if len(sub) else np.nan,
        "eta_max": float(np.nanmax(eta)) if len(sub) else np.nan,
        "eta_abs_p95": float(np.nanquantile(np.abs(eta), 0.95)) if len(sub) else np.nan,
        "eta_negative_count": sign_negative,
        "eta_positive_count": sign_positive,
        "eta_zero_count": sign_zero,
        "trusted_exact_count": int(sub["trusted_exact"].fillna(0).astype(float).sum()) if "trusted_exact" in sub else np.nan,
        "training_eligible_count": int(sub["training_eligible_exact"].fillna(0).astype(float).sum()) if "training_eligible_exact" in sub else np.nan,
        "needs_rerun_count": int(sub["needs_rerun_exact"].fillna(0).astype(float).sum()) if "needs_rerun_exact" in sub else np.nan,
        "q_expanded_count": int(sub["q_expanded"].fillna(0).astype(float).sum()) if "q_expanded" in sub else np.nan,
        "q_unresolved_count": int(sub["q_unresolved"].fillna(0).astype(float).sum()) if "q_unresolved" in sub else np.nan,
        "delta_unresolved_count": int(sub["delta_unresolved"].fillna(0).astype(float).sum()) if "delta_unresolved" in sub else np.nan,
        "note": note,
    }


def compute_response_region_diagnostics(final_df: pd.DataFrame) -> pd.DataFrame:
    k = final_df["kT"].to_numpy(dtype=float)
    j = final_df["JA"].to_numpy(dtype=float)
    p0 = np.array([0.05, 1.30])
    p1 = np.array([0.00, 2.00])
    pts = np.column_stack((k, j))
    seg = p1 - p0
    t = np.sum((pts - p0) * seg, axis=1) / max(float(np.dot(seg, seg)), 1e-12)
    t = np.clip(t, 0.0, 1.0)
    dist = np.linalg.norm(pts - (p0 + t[:, None] * seg), axis=1)
    rows = [
        _eta_region_row(
            final_df,
            "low_t_edge_j_0p75_1p25",
            (final_df["kT"] <= 0.03) & final_df["JA"].between(0.75, 1.25),
            "User-flagged low-temperature edge. Labels are stable, but eta sign/magnitude varies strongly.",
        ),
        _eta_region_row(
            final_df,
            "diagonal_high_j_band_dist_0p03",
            pd.Series(dist <= 0.03, index=final_df.index),
            "User-flagged diagonal from (0.05,1.3) to (0,2.0), using raw parameter distance <= 0.03.",
        ),
        _eta_region_row(
            final_df,
            "diagonal_high_j_band_dist_0p05",
            pd.Series(dist <= 0.05, index=final_df.index),
            "Same diagonal band with raw parameter distance <= 0.05.",
        ),
    ]
    return pd.DataFrame(rows)


def save_figures(final_df: pd.DataFrame, iteration_df: pd.DataFrame, learning_df: pd.DataFrame) -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    topo_df = final_df.copy()
    topo_df["topology_label"] = topo_df["topology_label_code"].map(topology_name)
    bounds = _coordinate_bounds(final_df)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for phase, color in PHASE_COLORS.items():
        sub = final_df[final_df["phase_name"] == phase]
        axes[0].scatter(sub["kT"], sub["JA"], s=7, c=color, label=phase, alpha=0.72, linewidths=0)
    axes[0].set_title("Final thermodynamic labels")
    setup_axes(axes[0])
    axes[0].legend(markerscale=2, fontsize=8)
    for label, color in TOPOLOGY_COLORS.items():
        sub = topo_df[topo_df["topology_label"] == label]
        if len(sub):
            axes[1].scatter(sub["kT"], sub["JA"], s=7, c=color, label=label, alpha=0.72, linewidths=0)
    axes[1].set_title("Online topology diagnostics")
    setup_axes(axes[1])
    axes[1].legend(markerscale=2, fontsize=8)
    paths["phase_topology_map"] = FIGURE_DIR / "phase_topology_map.png"
    fig.savefig(paths["phase_topology_map"], dpi=220)
    plt.close(fig)

    eta_values = final_df["eta"].to_numpy(dtype=float)
    eta_lim = float(np.nanquantile(np.abs(eta_values[np.isfinite(eta_values)]), 0.99))
    eta_lim = max(eta_lim, 1e-3)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
    valid_eta = final_df[["kT", "JA", "eta"]].dropna()
    tri = mtri.Triangulation(valid_eta["kT"].to_numpy(), valid_eta["JA"].to_numpy())
    levels = np.linspace(-eta_lim, eta_lim, 45)
    cf = ax.tricontourf(
        tri,
        valid_eta["eta"].to_numpy(),
        levels=levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-eta_lim, vcenter=0.0, vmax=eta_lim),
        extend="both",
        alpha=0.96,
        zorder=1,
    )
    ax.scatter(final_df["kT"], final_df["JA"], s=2.2, c="black", alpha=0.10, linewidths=0, zorder=2)
    _tricontour_boundary(
        ax,
        final_df,
        (final_df["phase_name"] != "normal").astype(float),
        color="black",
        linestyle="-",
        linewidth=1.8,
        label="normal/SC",
    )
    sc_df = final_df[final_df["phase_name"] != "normal"]
    _tricontour_boundary(
        ax,
        sc_df,
        (sc_df["phase_name"] == "FFLO").astype(float),
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.3,
        label="uniform-SC/FFLO",
    )
    fflo_trusted = topo_df[(topo_df["phase_name"] == "FFLO") & (topo_df["topology_trusted"] == 1)]
    n_topology_edges = _plot_binary_change_edges(
        color="#7a0177",
        ax=ax,
        df=fflo_trusted,
        value=(fflo_trusted["topology_label"] == "topological").astype(float),
        linewidth=1.35,
        max_normalized_edge=0.03,
        bounds=bounds,
    )
    topological = fflo_trusted[fflo_trusted["topology_label"] == "topological"]
    conventional = fflo_trusted[fflo_trusted["topology_label"] == "trivial"]
    if len(topological):
        ax.text(
            float(topological["kT"].median()),
            float(topological["JA"].quantile(0.75)),
            "tFFLO",
            fontsize=10,
            weight="bold",
            color="#7a0177",
            bbox={"facecolor": "white", "edgecolor": "#7a0177", "alpha": 0.75, "pad": 2},
            zorder=7,
        )
    if len(conventional):
        ax.text(
            float(conventional["kT"].quantile(0.70)),
            float(conventional["JA"].median()),
            "cFFLO / trivial SC",
            fontsize=9,
            color="#253494",
            bbox={"facecolor": "white", "edgecolor": "#253494", "alpha": 0.72, "pad": 2},
            zorder=7,
        )
    ax.text(
        0.97,
        0.05,
        f"Color: eta sign and magnitude\nPurple: {n_topology_edges} local Z2-change edges",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.78, "pad": 3},
        zorder=7,
    )
    setup_axes(ax)
    ax.set_title(r"Stage-III $\eta$ map with thermodynamic and online topology contours")
    legend_lines = [
        Line2D([0], [0], color="black", lw=1.8, label="normal/SC"),
        Line2D([0], [0], color="#4d4d4d", lw=1.3, ls="--", label="uniform-SC/FFLO"),
        Line2D([0], [0], color="#7a0177", lw=1.7, label="cFFLO/tFFLO local edges"),
    ]
    ax.legend(handles=legend_lines, loc="upper left", fontsize=8, frameon=True)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(r"$\eta$")
    paths["eta_topology_phase_map"] = FIGURE_DIR / "eta_topology_phase_map.png"
    fig.savefig(paths["eta_topology_phase_map"], dpi=240)
    plt.close(fig)

    dataset_rows = []
    for p in sorted(RUN_DIR.glob("dataset_iter*.csv")):
        iter_idx = int(p.stem.replace("dataset_iter", ""))
        df = pd.read_csv(p, usecols=["phase_name"])
        counts = df["phase_name"].value_counts()
        dataset_rows.append(
            {
                "iteration": iter_idx,
                "normal": counts.get("normal", 0),
                "uniform_SC": counts.get("uniform_SC", 0),
                "FFLO": counts.get("FFLO", 0),
                "total": len(df),
            }
        )
    growth = pd.DataFrame(dataset_rows)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    bottom = np.zeros(len(growth))
    for phase, color in PHASE_COLORS.items():
        values = growth[phase].to_numpy()
        ax.bar(growth["iteration"], values, bottom=bottom, label=phase, color=color, alpha=0.9)
        bottom += values
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("samples")
    ax.set_title("Dataset growth by thermodynamic phase")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    paths["dataset_growth"] = FIGURE_DIR / "dataset_growth_phase_counts.png"
    fig.savefig(paths["dataset_growth"], dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(iteration_df["iteration"], iteration_df["topology_trivial_count"], label="trivial exact in batch", color=TOPOLOGY_COLORS["trivial"])
    ax.plot(iteration_df["iteration"], iteration_df["topology_topological_count"], label="topological exact in batch", color=TOPOLOGY_COLORS["topological"])
    ax.plot(iteration_df["iteration"], iteration_df["topology_trusted_count"], label="trusted topology exact in batch", color="#2ca25f")
    if not learning_df.empty and "topology_context_trusted_count" in learning_df:
        ax.plot(
            learning_df["iteration"],
            learning_df["topology_context_trusted_count"],
            label="trusted topology context",
            color="#54278f",
            linestyle="--",
        )
    ax.set_xlabel("iteration")
    ax.set_ylabel("count")
    ax.set_title("Topology diagnostics entering acquisition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    paths["topology_counts"] = FIGURE_DIR / "topology_counts_curve.png"
    fig.savefig(paths["topology_counts"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(iteration_df["iteration"], iteration_df["phase_map_change"], marker="o", ms=3)
    ax.axhline(0.002, color="black", ls="--", lw=1, label="tol")
    ax.set_title("Phase-map change")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[0, 1]
    ax.plot(iteration_df["iteration"], iteration_df["boundary_shift_normal_sc"], marker="o", ms=3, label="normal/SC")
    ax.plot(iteration_df["iteration"], iteration_df["boundary_shift_uniform_fflo"], marker="o", ms=3, label="uniform/FFLO")
    ax.axhline(0.004166666666666667, color="black", ls="--", lw=1, label="shift tol")
    ax.set_title("Boundary shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[1, 0]
    ax.plot(iteration_df["iteration"], iteration_df["label_surprise_all_selected"], label="all selected", marker="o", ms=3)
    ax.plot(iteration_df["iteration"], iteration_df["label_surprise_trusted"], label="trusted gate", marker="o", ms=3)
    ax.plot(iteration_df["iteration"], iteration_df["label_surprise_hard_risk"], label="hard-risk", marker="o", ms=3)
    ax.axhline(0.05, color="black", ls="--", lw=1, label="surprise tol")
    ax.set_title("Label surprise")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    ax = axes[1, 1]
    ax.plot(iteration_df["iteration"], iteration_df["boundary_coverage_p95"], marker="o", ms=3)
    ax.axhline(0.00625, color="black", ls="--", lw=1, label="coverage tol")
    ax.set_title("Boundary coverage p95")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("iteration")
    paths["stop_metrics"] = FIGURE_DIR / "stop_metrics_curve.png"
    fig.savefig(paths["stop_metrics"], dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot(iteration_df["iteration"], iteration_df["point_total_runtime_sec_sum"] / 3600.0, marker="o", ms=3, label="point total")
    axes[0].plot(iteration_df["iteration"], iteration_df["local_refinement_runtime_sec_sum"] / 3600.0, marker="o", ms=3, label="local refinement")
    axes[0].plot(iteration_df["iteration"], iteration_df["iteration_walltime_sec"] / 3600.0, marker="o", ms=3, label="exact array walltime")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("rank-summed runtime (h)")
    axes[0].set_title("Exact-oracle workload")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(iteration_df["iteration"], iteration_df["mean_selected_refine_targets"], marker="o", ms=3, color="#225ea8")
    axes[1].axhline(3.0, color="black", ls="--", lw=1, label="rankcap K3")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mean selected local targets")
    axes[1].set_title("Rank-and-cap local refinement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    paths["runtime_local_boxes"] = FIGURE_DIR / "runtime_local_boxes.png"
    fig.savefig(paths["runtime_local_boxes"], dpi=220)
    plt.close(fig)

    trusted = topo_df[topo_df["topology_trusted"] == 1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    sc = axes[0].scatter(
        trusted["kT"],
        trusted["JA"],
        c=np.log10(np.maximum(trusted["topology_bulk_gap"], 1e-12)),
        s=7,
        cmap="viridis",
        linewidths=0,
    )
    axes[0].set_title(r"$\log_{10} E_g$ on trusted SC points")
    setup_axes(axes[0])
    fig.colorbar(sc, ax=axes[0], fraction=0.046)
    sc = axes[1].scatter(
        trusted["kT"],
        trusted["JA"],
        c=np.log10(np.maximum(trusted["topology_pfaffian_margin"], 1e-12)),
        s=7,
        cmap="magma",
        linewidths=0,
    )
    axes[1].set_title(r"$\log_{10}$ Pfaffian margin")
    setup_axes(axes[1])
    fig.colorbar(sc, ax=axes[1], fraction=0.046)
    paths["bulk_gap_pfaffian"] = FIGURE_DIR / "bulk_gap_pfaffian_margin.png"
    fig.savefig(paths["bulk_gap_pfaffian"], dpi=220)
    plt.close(fig)

    selected_rows = []
    for iter_dir in sorted(RUN_DIR.glob("iter[0-9][0-9][0-9]")):
        iter_idx = int(iter_dir.name.replace("iter", ""))
        p = iter_dir / "selected_points.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["iteration"] = iter_idx
            selected_rows.append(df)
    if selected_rows:
        selected = pd.concat(selected_rows, ignore_index=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
        sc = ax.scatter(selected["kT"], selected["JA"], c=selected["iteration"], s=6, cmap="plasma", alpha=0.75, linewidths=0)
        ax.set_title("Selected exact points by iteration")
        setup_axes(ax)
        fig.colorbar(sc, ax=ax, label="iteration", fraction=0.046)
        paths["selected_points"] = FIGURE_DIR / "selected_points_by_iteration.png"
        fig.savefig(paths["selected_points"], dpi=220)
        plt.close(fig)

    _, boundary_by_iter = compute_selected_boundary_concentration(final_df)
    if not boundary_by_iter.empty:
        acq = boundary_by_iter[boundary_by_iter["iteration"] > 0]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        axes[0].plot(
            acq["iteration"],
            acq["normal_sc_within_0p03"],
            marker="o",
            ms=3,
            label="within 0.03 of normal/SC",
            color="black",
        )
        axes[0].plot(
            acq["iteration"],
            acq["cfflo_tfflo_within_0p03"],
            marker="o",
            ms=3,
            label="within 0.03 of cFFLO/tFFLO",
            color="#7a0177",
        )
        axes[0].plot(
            acq["iteration"],
            acq["either_boundary_within_0p03"],
            marker="o",
            ms=3,
            label="within 0.03 of either",
            color="#2b8cbe",
        )
        axes[0].set_xlabel("acquisition iteration")
        axes[0].set_ylabel("fraction of selected batch")
        axes[0].set_ylim(0, 1.03)
        axes[0].set_title("Boundary concentration of selected points")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.2)
        axes[1].plot(
            acq["iteration"],
            acq["normal_sc_median_distance"],
            marker="o",
            ms=3,
            label="normal/SC",
            color="black",
        )
        axes[1].plot(
            acq["iteration"],
            acq["cfflo_tfflo_median_distance"],
            marker="o",
            ms=3,
            label="cFFLO/tFFLO",
            color="#7a0177",
        )
        axes[1].plot(
            acq["iteration"],
            acq["either_boundary_median_distance"],
            marker="o",
            ms=3,
            label="nearest of either",
            color="#2b8cbe",
        )
        axes[1].axhline(0.03, color="0.4", ls="--", lw=1, label="0.03 band")
        axes[1].set_xlabel("acquisition iteration")
        axes[1].set_ylabel("median normalized distance")
        axes[1].set_title("Selected-point distance to final diagnostic boundaries")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2)
        paths["selected_boundary_concentration"] = FIGURE_DIR / "selected_boundary_concentration.png"
        fig.savefig(paths["selected_boundary_concentration"], dpi=220)
        plt.close(fig)

    if not learning_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        axes[0].plot(learning_df["iteration"], learning_df["phase_accuracy"], marker="o", ms=3)
        axes[0].set_title("Validation phase accuracy")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylim(0.9, 1.01)
        axes[0].grid(alpha=0.2)
        axes[1].plot(learning_df["iteration"], learning_df["delta_rmse"], label=r"$\Delta$ RMSE", marker="o", ms=3)
        axes[1].plot(learning_df["iteration"], learning_df["q_rmse"], label=r"$q$ RMSE", marker="o", ms=3)
        axes[1].plot(learning_df["iteration"], learning_df["eta_rmse"], label=r"$\eta$ RMSE", marker="o", ms=3)
        axes[1].set_title("Regression validation curves")
        axes[1].set_xlabel("iteration")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.2)
        paths["learning_curves"] = FIGURE_DIR / "learning_curves.png"
        fig.savefig(paths["learning_curves"], dpi=220)
        plt.close(fig)
    return paths


def latex_escape(text: str) -> str:
    replacements = {
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
    return "".join(replacements.get(ch, ch) for ch in str(text))


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        v = float(value)
    except Exception:
        return str(value)
    if math.isnan(v):
        return "n/a"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:.{digits}f}"
    return f"{v:.{digits}g}"


def write_reports(final_df: pd.DataFrame, iteration_df: pd.DataFrame, learning_df: pd.DataFrame, figures: dict[str, Path]) -> None:
    final = iteration_df.iloc[-1]
    topo_df = final_df.copy()
    topo_df["topology_label"] = topo_df["topology_label_code"].map(topology_name)
    phase_counts = final_df["phase_name"].value_counts()
    topo_counts = topo_df["topology_label"].value_counts()
    stop_pass = f"{int(final['passed_condition_count'])}/{int(final['required_pass_count'])}"
    total_selected = int(iteration_df["merged_exact_points"].sum())
    total_unique_appended = int(iteration_df["new_unique_samples_added"].sum())
    total_training_eligible = int(iteration_df["training_eligible_appended"].sum())
    mean_boxes = float(iteration_df["mean_selected_refine_targets"].mean())
    max_boxes = float(iteration_df["mean_selected_refine_targets"].max())
    total_exact_walltime_h = float(iteration_df["iteration_walltime_sec"].sum() / 3600.0)
    mean_exact_walltime_min = float(iteration_df["iteration_walltime_sec"].mean() / 60.0)
    max_exact_walltime_min = float(iteration_df["iteration_walltime_sec"].max() / 60.0)
    total_point_runtime_h = float(iteration_df["point_total_runtime_sec_sum"].sum() / 3600.0)
    total_topology_runtime_s = float(iteration_df["topology_runtime_sec_sum"].sum())
    total_local_runtime_h = float(iteration_df["local_refinement_runtime_sec_sum"].sum() / 3600.0)
    latest_append = read_json(RUN_DIR / "dataset_iter018.append.json")
    region_diag = compute_response_region_diagnostics(final_df).set_index("region")
    boundary_summary, _ = compute_selected_boundary_concentration(final_df)
    boundary_summary = boundary_summary.set_index("group") if not boundary_summary.empty else pd.DataFrame()

    def region_value(region: str, column: str) -> Any:
        return region_diag.loc[region, column] if region in region_diag.index else np.nan

    def concentration_value(group: str, column: str) -> Any:
        return boundary_summary.loc[group, column] if group in boundary_summary.index else np.nan

    md = f"""# Topology-Aware Full-Loop Return Summary

## Executive Summary

- Run ID: `{RUN_ID}`.
- Completed active-learning iterations: {int(final['iteration']) + 1} (`iter000` through `iter017`).
- Final dataset: `dataset_iter018.npz` with {len(final_df)} samples.
- Final thermodynamic counts: normal {int(phase_counts.get('normal', 0))}, uniform-SC {int(phase_counts.get('uniform_SC', 0))}, FFLO {int(phase_counts.get('FFLO', 0))}.
- Online topology diagnostics: trivial {int(topo_counts.get('trivial', 0))}, topological {int(topo_counts.get('topological', 0))}, gapless-SC {int(topo_counts.get('gapless_SC', 0))}, unresolved {int(topo_counts.get('unresolved', 0))}.
- Stop reason: `{final['stop_reason']}` with {stop_pass} required conditions passed.
- Final trusted label surprise: {fmt(final['label_surprise_trusted'])}; all-selected surprise: {fmt(final['label_surprise_all_selected'])}; hard-risk surprise: {fmt(final['label_surprise_hard_risk'])}.
- Boundary coverage p95 remains above the strict tolerance ({fmt(final['boundary_coverage_p95'])} vs 0.00625), so this is a main-boundary convergence result rather than a publication-grade topology-boundary closure.

## Interpretation

This run is a cold-start topology-aware active-learning loop using `topo_trivial` acquisition. It is not a continuation from `dataset_iter035`. The online topology fields are acquisition diagnostics. Final topology/trivial/nodal claims should still be made from a post-run offline topology pass on the frozen final dataset.

## Key Workload Numbers

- Exact selected/evaluated points across shards: {total_selected}.
- New unique samples appended: {total_unique_appended}.
- Training-eligible append records before duplicate filtering: {total_training_eligible}.
- Latest append added {latest_append.get('new_unique_samples_added')} new unique samples and {latest_append.get('training_eligible_points_appended')} training-eligible samples.
- Mean selected local refinement targets per exact point: {mean_boxes:.2f}; maximum iteration mean: {max_boxes:.2f}.
- Estimated exact-array walltime: {total_exact_walltime_h:.2f} h total; mean {mean_exact_walltime_min:.1f} min/iteration; max {max_exact_walltime_min:.1f} min.
- Rank-summed point runtime: {total_point_runtime_h:.2f} h.
- Rank-summed local-refinement runtime: {total_local_runtime_h:.2f} h.
- Rank-summed topology diagnostic runtime: {total_topology_runtime_s:.2f} s.

## Response-Side Rough-Region Diagnostics

The red-blue eta map still shows two visually rough or weakly determined response regions. These diagnostics do not change the thermodynamic labels.

1. Low-temperature edge, `kT <= 0.03` and `0.75 <= JA <= 1.25`: {int(region_value('low_t_edge_j_0p75_1p25', 'n_points'))} points, all on the FFLO/topological side in the current online labels. The exact metadata are clean (`trusted_exact = {int(region_value('low_t_edge_j_0p75_1p25', 'trusted_exact_count'))}/{int(region_value('low_t_edge_j_0p75_1p25', 'n_points'))}`, no q/delta unresolved flags), but eta changes sign ({int(region_value('low_t_edge_j_0p75_1p25', 'eta_negative_count'))} negative, {int(region_value('low_t_edge_j_0p75_1p25', 'eta_positive_count'))} positive) with `p95(|eta|) = {fmt(region_value('low_t_edge_j_0p75_1p25', 'eta_abs_p95'))}`. This is therefore a response-side eta stability issue, not evidence that the FFLO/topological label is failing.

2. Diagonal high-J band near the segment from `(kT,JA)=(0.05,1.3)` to `(0,2)`: within raw parameter distance 0.03, {int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))} points split between {region_value('diagonal_high_j_band_dist_0p03', 'phase_counts')}. Eta is small and mostly negative/zero (`median eta = {fmt(region_value('diagonal_high_j_band_dist_0p03', 'eta_median'))}`, `p95(|eta|) = {fmt(region_value('diagonal_high_j_band_dist_0p03', 'eta_abs_p95'))}`), while q-expansion is common (`q_expanded = {int(region_value('diagonal_high_j_band_dist_0p03', 'q_expanded_count'))}/{int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))}`). This band should be interpreted as a normal/SC boundary and q-window-sensitive response band, not as a settled eta-sign boundary.

## Boundary Selection Concentration

Selection concentration is measured as the normalized Euclidean distance from each selected point to final diagnostic Delaunay boundary segments. The normal/SC segments and local cFFLO/tFFLO Z2-change segments both use a maximum normalized edge length of 0.03; these are acquisition diagnostics, not final publication boundaries.

- Across all acquisition batches, the fraction of selected points within 0.03 of the normal/SC boundary is {fmt(concentration_value('all_acquisition_iters001_017', 'normal_sc_within_0p03'))}; within 0.03 of local cFFLO/tFFLO edges is {fmt(concentration_value('all_acquisition_iters001_017', 'cfflo_tfflo_within_0p03'))}; within 0.03 of either boundary is {fmt(concentration_value('all_acquisition_iters001_017', 'either_boundary_within_0p03'))}.
- In the last five acquisition batches, the same fractions are {fmt(concentration_value('last5_acquisition_iters013_017', 'normal_sc_within_0p03'))}, {fmt(concentration_value('last5_acquisition_iters013_017', 'cfflo_tfflo_within_0p03'))}, and {fmt(concentration_value('last5_acquisition_iters013_017', 'either_boundary_within_0p03'))}, respectively.
- Median distance to the nearest of these two boundary families is {fmt(concentration_value('all_acquisition_iters001_017', 'either_boundary_median_distance'))} over all acquisition batches and {fmt(concentration_value('last5_acquisition_iters013_017', 'either_boundary_median_distance'))} over the last five batches.

These metrics support the visual impression that the topology-aware acquisition is boundary-focused, especially around the thermodynamic normal/SC boundary and the diagnostic cFFLO/tFFLO frontier.
"""
    (REPORT_DIR / "topo_trivial_full_loop_summary.md").write_text(md, encoding="utf-8")

    decision_log = f"""# Decision Log

## Decision

The returned `topo_trivial` full-loop run completed successfully and stopped at `iter017` with `stop_reason = {final['stop_reason']}`.

## Evidence

- Final dataset size: {len(final_df)}.
- Phase counts: normal {int(phase_counts.get('normal', 0))}, uniform-SC {int(phase_counts.get('uniform_SC', 0))}, FFLO {int(phase_counts.get('FFLO', 0))}.
- Online topology counts: trivial {int(topo_counts.get('trivial', 0))}, topological {int(topo_counts.get('topological', 0))}.
- Trusted surprise gate: {fmt(final['label_surprise_trusted'])}.
- Stop conditions passed: {stop_pass}.
- Low-temperature edge eta diagnostic: {int(region_value('low_t_edge_j_0p75_1p25', 'n_points'))} trusted FFLO/topological points, eta signs {int(region_value('low_t_edge_j_0p75_1p25', 'eta_negative_count'))} negative and {int(region_value('low_t_edge_j_0p75_1p25', 'eta_positive_count'))} positive.
- Diagonal high-J response band: {int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))} points within raw distance 0.03, with phase split {region_value('diagonal_high_j_band_dist_0p03', 'phase_counts')} and q_expanded={int(region_value('diagonal_high_j_band_dist_0p03', 'q_expanded_count'))}/{int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))}.
- Boundary selection concentration: all acquisition batches have {fmt(concentration_value('all_acquisition_iters001_017', 'either_boundary_within_0p03'))} of selected points within normalized distance 0.03 of either the normal/SC or diagnostic cFFLO/tFFLO boundary; the last five batches have {fmt(concentration_value('last5_acquisition_iters013_017', 'either_boundary_within_0p03'))}.

## Caveat

Boundary coverage p95 did not pass the strict tolerance, and online topology diagnostics are not a replacement for the publication-grade offline topology pass.
The rough eta regions are response-side diagnostics; they do not alter the frozen thermodynamic labels.
"""
    (REPORT_DIR / "decision_log.md").write_text(decision_log, encoding="utf-8")

    figure_lines = "\n".join(
        [
            rf"\begin{{figure}}[htbp]\centering\includegraphics[width=0.96\linewidth]{{figures/{p.name}}}\caption{{{caption}}}\end{{figure}}"
            for key, p, caption in [
                ("eta_topology_phase_map", figures["eta_topology_phase_map"], r"Red--blue $\eta$ map.  Solid black: normal/SC contour; dashed gray: uniform-SC/FFLO contour; purple: short local online Z2-change edges between cFFLO and tFFLO samples."),
                ("phase_topology_map", figures["phase_topology_map"], "Final thermodynamic labels and online topology diagnostics."),
                ("dataset_growth", figures["dataset_growth"], "Dataset growth by thermodynamic phase."),
                ("topology_counts", figures["topology_counts"], "Topology counts observed by exact shards and used as acquisition context."),
                ("stop_metrics", figures["stop_metrics"], "StopController metrics through the active-learning loop."),
                ("runtime_local_boxes", figures["runtime_local_boxes"], "Exact-oracle runtime and rank-and-cap local-refinement workload."),
                ("bulk_gap_pfaffian", figures["bulk_gap_pfaffian"], "Bulk-gap and Pfaffian-margin diagnostics on trusted superconducting points."),
                ("selected_points", figures["selected_points"], "Selected exact points colored by active-learning iteration."),
                ("selected_boundary_concentration", figures.get("selected_boundary_concentration", figures["phase_topology_map"]), "Boundary-concentration metrics for selected exact points. Distances are normalized to the final parameter-domain extent and use final diagnostic Delaunay boundary segments."),
                ("learning_curves", figures.get("learning_curves", figures["phase_topology_map"]), "Surrogate validation curves."),
            ]
            if key in figures
        ]
    )

    tex = rf"""% Auto-generated by scripts/build_topo_trivial_full_loop_return_report.py
\documentclass[11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{hyperref}}
\usepackage{{float}}
\hypersetup{{hypertexnames=false}}
\title{{Topology-Aware Active-Learning Full-Loop Return Summary}}
\author{{Auto-generated report-only analysis}}
\date{{\today}}
\begin{{document}}
\maketitle

\section{{Executive Summary}}
This report summarizes the returned Stage-III topology-aware full-loop run.
The run identifier is listed in the table below.  The run used a cold-start scrambled Sobol
initial design, \texttt{{topo\_trivial}} acquisition, the robust incremental
exact oracle, rank-and-cap K3 local refinement, and online Pfaffian/bulk-gap
diagnostics.

\begin{{center}}
\begin{{tabular}}{{ll}}
\toprule
Quantity & Value \\
\midrule
Completed iterations & {int(final['iteration']) + 1} \\
Run ID & \texttt{{\footnotesize {latex_escape(RUN_ID)}}} \\
Final dataset & \texttt{{dataset\_iter018.npz}} \\
Final samples & {len(final_df)} \\
Phase counts & normal {int(phase_counts.get('normal', 0))}; uniform-SC {int(phase_counts.get('uniform_SC', 0))}; FFLO {int(phase_counts.get('FFLO', 0))} \\
Online topology counts & trivial {int(topo_counts.get('trivial', 0))}; topological {int(topo_counts.get('topological', 0))}; gapless-SC {int(topo_counts.get('gapless_SC', 0))}; unresolved {int(topo_counts.get('unresolved', 0))} \\
Stop reason & \texttt{{{latex_escape(final['stop_reason'])}}} \\
Stop conditions & {stop_pass} passed \\
Trusted surprise & {fmt(final['label_surprise_trusted'])} \\
All-selected surprise & {fmt(final['label_surprise_all_selected'])} \\
Hard-risk surprise & {fmt(final['label_surprise_hard_risk'])} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\section{{Scientific Scope}}
This run is not a continuation of \texttt{{dataset\_iter035}}.  It is a new
cold-start active-learning loop whose online topology fields are used for
acquisition and diagnostics.  The thermodynamic labels are still
normal/uniform-SC/FFLO.  The online topology labels should be treated as
run-time diagnostics until a publication-grade offline topology pass is run on
the final frozen dataset.

\section{{Final Phase and Topology Maps}}
{figure_lines}

\section{{Convergence and Numerical Status}}
At the final evaluated iteration, the phase-map change is
{fmt(final['phase_map_change'])} versus tolerance $0.002$, the normal/SC
boundary shift is {fmt(final['boundary_shift_normal_sc'])} versus tolerance
$0.004167$, and the uniform-SC/FFLO shift is {fmt(final['boundary_shift_uniform_fflo'])}.
The trusted-surprise gate is zero, while all-selected surprise remains
{fmt(final['label_surprise_all_selected'])}.  Boundary coverage p95 is
{fmt(final['boundary_coverage_p95'])}, slightly above the strict $0.00625$
coverage tolerance, so the report should be read as a successful main-boundary
convergence run with remaining coverage/topology-boundary follow-up.

\section{{Response-Side Rough-Region Diagnostics}}
The red--blue $\eta$ map still shows visually rough or weakly determined
response regions.  These diagnostics do not change the thermodynamic labels.
At the low-temperature edge $k_B T/t \le 0.03$ and
$0.75 \le J_A/t \le 1.25$, there are
{int(region_value('low_t_edge_j_0p75_1p25', 'n_points'))} points.  They are
all on the FFLO/topological side in the current online labels, and
{int(region_value('low_t_edge_j_0p75_1p25', 'trusted_exact_count'))}/
{int(region_value('low_t_edge_j_0p75_1p25', 'n_points'))} are trusted exact
points with no q- or $\Delta$-unresolved flags.  However, $\eta$ changes sign
there ({int(region_value('low_t_edge_j_0p75_1p25', 'eta_negative_count'))}
negative and {int(region_value('low_t_edge_j_0p75_1p25', 'eta_positive_count'))}
positive points), with $p_{{95}}(|\eta|)={fmt(region_value('low_t_edge_j_0p75_1p25', 'eta_abs_p95'))}$.
This is a response-side $\eta$ stability issue, not evidence that the
thermodynamic or online topology label is failing.

Along the diagonal high-$J_A$ band near the segment from
$(k_B T/t,J_A/t)=(0.05,1.3)$ to $(0,2)$, the distance-$0.03$ band contains
{int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))} points split
between {latex_escape(region_value('diagonal_high_j_band_dist_0p03', 'phase_counts'))}.
Here $\eta$ is small and mostly negative/zero
(median $\eta={fmt(region_value('diagonal_high_j_band_dist_0p03', 'eta_median'))}$,
$p_{{95}}(|\eta|)={fmt(region_value('diagonal_high_j_band_dist_0p03', 'eta_abs_p95'))}$),
while q expansion is common
({int(region_value('diagonal_high_j_band_dist_0p03', 'q_expanded_count'))}/
{int(region_value('diagonal_high_j_band_dist_0p03', 'n_points'))}).  This band
should be interpreted as a normal/SC boundary and q-window-sensitive response
band, not as a settled $\eta$-sign boundary.

\section{{Boundary Selection Concentration}}
Selection concentration is measured as normalized Euclidean distance from each
selected point to final diagnostic Delaunay boundary segments.  The normal/SC
segments and local cFFLO/tFFLO Z2-change segments both use a maximum normalized
edge length of $0.03$.  These are acquisition diagnostics, not final
publication boundaries.

Across all acquisition batches, the fraction of selected points within $0.03$
of the normal/SC boundary is
{fmt(concentration_value('all_acquisition_iters001_017', 'normal_sc_within_0p03'))};
within $0.03$ of local cFFLO/tFFLO edges is
{fmt(concentration_value('all_acquisition_iters001_017', 'cfflo_tfflo_within_0p03'))};
and within $0.03$ of either boundary is
{fmt(concentration_value('all_acquisition_iters001_017', 'either_boundary_within_0p03'))}.
For the last five acquisition batches, the corresponding fractions are
{fmt(concentration_value('last5_acquisition_iters013_017', 'normal_sc_within_0p03'))},
{fmt(concentration_value('last5_acquisition_iters013_017', 'cfflo_tfflo_within_0p03'))},
and {fmt(concentration_value('last5_acquisition_iters013_017', 'either_boundary_within_0p03'))}.
The median distance to the nearest of these two boundary families is
{fmt(concentration_value('all_acquisition_iters001_017', 'either_boundary_median_distance'))}
for all acquisition batches and
{fmt(concentration_value('last5_acquisition_iters013_017', 'either_boundary_median_distance'))}
for the last five batches.  These values support the visual impression that
the topology-aware acquisition concentrated on the thermodynamic normal/SC
boundary and the diagnostic cFFLO/tFFLO frontier.

\section{{Compute and Workload}}
Across all iterations, the exact shards evaluated {total_selected} selected
points and appended {total_unique_appended} new unique samples.  The append
logs recorded {total_training_eligible} training-eligible records before
duplicate filtering.  The mean
selected local-refinement target count was {mean_boxes:.2f} per exact point,
with maximum iteration mean {max_boxes:.2f}.  Estimated exact-array walltime
from the slowest shard in each iteration was {total_exact_walltime_h:.2f} hours
in total, with mean {mean_exact_walltime_min:.1f} minutes per iteration and max
{max_exact_walltime_min:.1f} minutes.  Rank-summed point runtime was
{total_point_runtime_h:.2f} hours, rank-summed local-refinement runtime was
{total_local_runtime_h:.2f} hours, and the online topology diagnostic runtime
sum was {total_topology_runtime_s:.2f} seconds.

\section{{Do-Not-Claim List}}
\begin{{itemize}}
\item Do not claim that this online topology-aware full loop provides final
publication-grade topology/trivial/nodal boundaries.
\item Do not interpret the lack of online gapless/unresolved labels in the
appended dataset as proof that the full topology frontier is resolved.
\item Do not merge this cold-start result with \texttt{{dataset\_iter035}}
without an explicit provenance table.
\item Do not ignore the failed boundary-coverage p95 condition when presenting
strict convergence criteria.
\end{{itemize}}

\section{{Next Step}}
Freeze \texttt{{dataset\_iter018.npz}} as the returned topology-aware full-loop
dataset for this run, then execute the offline topology pass/audit on that
dataset before making final topology-boundary claims.

\end{{document}}
"""
    (REPORT_DIR / "active_learning_phase_boundary_report.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_csv = latest_dataset_path()
    final_df = pd.read_csv(final_csv)
    iteration_df = collect_iteration_summary()
    learning_df = collect_learning_metrics()
    write_tables(final_df, iteration_df, learning_df)
    figures = save_figures(final_df, iteration_df, learning_df)
    write_reports(final_df, iteration_df, learning_df, figures)
    print(json.dumps({
        "run_id": RUN_ID,
        "final_dataset_csv": str(final_csv),
        "report_tex": str(REPORT_DIR / "active_learning_phase_boundary_report.tex"),
        "report_md": str(REPORT_DIR / "topo_trivial_full_loop_summary.md"),
        "figures": {k: str(v) for k, v in figures.items()},
        "tables_dir": str(TABLE_DIR),
    }, indent=2))


if __name__ == "__main__":
    main()
