from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_ROOT = Path(
    "rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/"
    "active_runs/active_boundary_discovery_rankcap_k3_full_loop_v1"
)
PACKAGE_REPORT_ROOT = Path("rankcap_k3_full_loop/reports/rankcap_k3_last5_stop_audit")
MIRROR_REPORT_ROOT = Path(
    "rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/"
    "reports/last5_selection_stop_audit"
)
LAST_N = 5

PHASE_NAMES = {
    0: "normal",
    1: "uniform_SC",
    2: "FFLO",
}


def label_eps() -> tuple[float, float]:
    cfg_path = RUN_ROOT / "run_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    active_cfg = cfg.get("active_learning_config", {})
    delta_eps = float(active_cfg.get("delta_eps", 0.001))
    q_eps = float(active_cfg.get("q_eps", 0.01))
    return delta_eps, q_eps


def stop_phase_label(delta_opt: pd.Series | np.ndarray, q_opt: pd.Series | np.ndarray) -> np.ndarray:
    delta_eps, q_eps = label_eps()
    delta = np.asarray(delta_opt, dtype=np.float64)
    q = np.asarray(q_opt, dtype=np.float64)
    out = np.full(delta.shape, 2, dtype=np.int64)
    out[delta < delta_eps] = 0
    uniform = (delta >= delta_eps) & (np.abs(q) < q_eps)
    out[uniform] = 1
    return out


def ensure_dirs(report_root: Path) -> tuple[Path, Path]:
    tables = report_root / "tables"
    figures = report_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def npz_to_frame(path: Path) -> pd.DataFrame:
    data = np.load(path, allow_pickle=True)
    cols = {}
    n = None
    for key in data.files:
        arr = data[key]
        if arr.ndim != 1:
            continue
        if n is None:
            n = len(arr)
        if len(arr) == n:
            cols[key] = arr
    return pd.DataFrame(cols)


def key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_kT_key"] = out["kT"].round(12)
    out["_JA_key"] = out["JA"].round(12)
    return out


def safe_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return math.nan
    return float(pd.to_numeric(df[column], errors="coerce").mean())


def safe_p90(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or len(df) == 0:
        return math.nan
    return float(pd.to_numeric(df[column], errors="coerce").quantile(0.90))


def value_counts_prefixed(series: pd.Series, prefix: str) -> dict[str, int]:
    counts = series.value_counts(dropna=False).to_dict()
    result = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "missing"
        else:
            try:
                label = PHASE_NAMES.get(int(key), str(int(key)))
            except Exception:
                label = str(key)
        result[f"{prefix}_{label}"] = int(value)
    return result


def load_iteration(iteration: int) -> dict[str, object]:
    iter_dir = RUN_ROOT / f"iter{iteration:03d}"
    selected = pd.read_csv(iter_dir / "selected_points_by_pool.csv")
    selected = key_columns(selected)

    exact = npz_to_frame(iter_dir / f"exact_merged_iter{iteration:03d}.npz")
    exact = key_columns(exact)
    exact["stop_exact_phase_label"] = stop_phase_label(exact["delta_opt"], exact["q_opt"])
    exact_cols = [
        "kT",
        "JA",
        "_kT_key",
        "_JA_key",
        "phase_candidate",
        "stop_exact_phase_label",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "q_expanded",
        "q_unresolved",
        "delta_boundary_ambiguous",
        "delta_refined",
        "delta_unresolved",
        "q_expansion_count",
        "q_expansion_trigger",
        "local_boxes_refined_count",
        "selected_refine_target_count",
        "point_total_runtime_sec",
        "local_refinement_runtime_sec",
        "rank",
    ]
    exact_cols = [c for c in exact_cols if c in exact.columns]
    joined = selected.merge(
        exact[exact_cols],
        on=["_kT_key", "_JA_key"],
        how="left",
        suffixes=("", "_exact"),
        validate="one_to_one",
    )

    dataset_before = key_columns(pd.read_csv(RUN_ROOT / f"dataset_iter{iteration:03d}.csv"))
    dataset_after = key_columns(pd.read_csv(RUN_ROOT / f"dataset_iter{iteration + 1:03d}.csv"))
    before_keys = set(zip(dataset_before["_kT_key"], dataset_before["_JA_key"]))
    dataset_after["new_unique_appended"] = [
        (kt, ja) not in before_keys for kt, ja in zip(dataset_after["_kT_key"], dataset_after["_JA_key"])
    ]
    label_cols = [
        "_kT_key",
        "_JA_key",
        "phase_label",
        "phase_name",
        "new_unique_appended",
    ]
    joined = joined.merge(
        dataset_after[label_cols],
        on=["_kT_key", "_JA_key"],
        how="left",
        validate="one_to_one",
    )
    joined = joined.rename(columns={"phase_label": "phase_label_after", "phase_name": "phase_name_after"})

    joined["raw_phase_candidate_name"] = joined["phase_candidate"].map(
        lambda x: PHASE_NAMES.get(int(x), "missing") if pd.notna(x) else "missing"
    )
    joined["stop_exact_phase_name"] = joined["stop_exact_phase_label"].map(
        lambda x: PHASE_NAMES.get(int(x), "missing") if pd.notna(x) else "missing"
    )
    joined["actual_phase_name"] = joined["stop_exact_phase_name"]
    joined["dataset_phase_name_after"] = joined["phase_label_after"].map(
        lambda x: PHASE_NAMES.get(int(x), "missing") if pd.notna(x) else "missing"
    )
    joined["predicted_phase_name"] = joined["predicted_phase_before_exact"].map(
        lambda x: PHASE_NAMES.get(int(x), "missing") if pd.notna(x) else "missing"
    )
    joined["phase_prediction_mismatch"] = (
        joined["stop_exact_phase_label"].notna()
        & joined["predicted_phase_before_exact"].notna()
        & (joined["stop_exact_phase_label"].astype(float) != joined["predicted_phase_before_exact"].astype(float))
    )

    stop = read_json(iter_dir / f"stop_metrics_iter{iteration:03d}.json")
    merge_summary = read_json(iter_dir / f"merge_summary_iter{iteration:03d}.json")
    selection_diag = read_json(iter_dir / "selection_diagnostics.json")
    region_diag_path = iter_dir / f"selection_region_diagnostics_iter{iteration:03d}.csv"
    region_diag = pd.read_csv(region_diag_path)

    return {
        "iteration": iteration,
        "joined": joined,
        "stop": stop,
        "merge_summary": merge_summary,
        "selection_diag": selection_diag,
        "region_diag": region_diag,
    }


def summarize_iteration(data: dict[str, object]) -> dict[str, object]:
    iteration = int(data["iteration"])
    joined = data["joined"]
    stop = data["stop"]
    merge_summary = data["merge_summary"]
    selection_diag = data["selection_diag"]
    region_diag = data["region_diag"]
    metrics = stop.get("metrics", {})
    conditions = stop.get("conditions", {})
    diagnostics = stop.get("diagnostic_conditions", {})
    thresholds = stop.get("thresholds", {})

    selected_region = region_diag[
        (region_diag["table"] == "region_distribution")
        & (region_diag["group"] == "selected_points")
    ]
    active_region = region_diag[
        (region_diag["table"] == "region_distribution")
        & (region_diag["group"] == "active_pool_candidates")
    ]
    selected_region_row = selected_region.iloc[0].to_dict() if len(selected_region) else {}
    active_region_row = active_region.iloc[0].to_dict() if len(active_region) else {}
    appended = joined[joined.get("new_unique_appended", False) == True].copy()
    selected_with_dataset_label = joined[joined["phase_label_after"].notna()].copy()
    selected_with_stop_label = joined[joined["stop_exact_phase_label"].notna()].copy()

    row: dict[str, object] = {
        "iteration": iteration,
        "selected_count": int(len(joined)),
        "exact_points": int(merge_summary.get("exact_points", merge_summary.get("merged_points", 0))),
        "training_eligible_points": int(merge_summary.get("training_eligible_points", 0)),
        "clean_trusted_points": int(merge_summary.get("clean_trusted_points", 0)),
        "boundary_band_points": int(merge_summary.get("boundary_band_normal_points", 0)),
        "rerun_required_points": int(merge_summary.get("rerun_required_points", 0)),
        "q_expanded_points": int(merge_summary.get("q_expanded_points", 0)),
        "delta_refined_points": int(merge_summary.get("delta_refined_points", 0)),
        "q_unresolved_points": int(merge_summary.get("q_unresolved_points", 0)),
        "delta_unresolved_points": int(merge_summary.get("delta_unresolved_points", 0)),
        "label_surprise_rate_stop": float(metrics.get("label_surprise_rate", math.nan)),
        "boundary_coverage_p95": float(metrics.get("boundary_coverage_p95", math.nan)),
        "phase_map_change": float(metrics.get("phase_map_change", math.nan)),
        "boundary_shift_normal_sc": float(metrics.get("boundary_shift_normal_sc", math.nan)),
        "boundary_shift_uniform_fflo": float(metrics.get("boundary_shift_uniform_fflo", math.nan)),
        "selected_A0_ratio": float(metrics.get("selected_A0_ratio", math.nan)),
        "q_edge_trigger_rate": float(metrics.get("q_edge_trigger_rate", math.nan)),
        "rerun_required_rate_stop": float(metrics.get("rerun_required_rate", math.nan)),
        "passed_condition_count": int(stop.get("passed_condition_count", 0)),
        "required_pass_count": int(stop.get("required_pass_count", 0)),
        "convergence_pass": bool(stop.get("convergence_pass", False)),
        "stop_reason": stop.get("stop_reason", ""),
        "C1_phase_map_change": bool(conditions.get("C1_phase_map_change", False)),
        "C2_boundary_shift_normal_sc": bool(conditions.get("C2_boundary_shift_normal_sc", False)),
        "C3_boundary_shift_uniform_fflo": bool(conditions.get("C3_boundary_shift_uniform_fflo", False)),
        "C4_label_surprise_rate": bool(conditions.get("C4_label_surprise_rate", False)),
        "C5_boundary_coverage_p95": bool(conditions.get("C5_boundary_coverage_p95", False)),
        "diag_selected_A0_ratio_below_tol": bool(diagnostics.get("selected_A0_ratio_below_tol", False)),
        "diag_q_edge_trigger_rate_below_tol": bool(diagnostics.get("q_edge_trigger_rate_below_tol", False)),
        "diag_rerun_required_rate_below_tol": bool(diagnostics.get("rerun_required_rate_below_tol", False)),
        "surprise_tol": thresholds.get("surprise_tol", math.nan),
        "coverage_tol": thresholds.get("coverage_tol", math.nan),
        "active_pool_size": int(selection_diag.get("active_pool_size", 0)),
        "active_pool_fraction": float(selection_diag.get("active_pool_fraction", math.nan)),
        "N_eff": float(selection_diag.get("N_eff", math.nan)),
        "N_eff_over_active_pool_size": float(selection_diag.get("N_eff_over_active_pool_size", math.nan)),
        "sampling_power_used": float(selection_diag.get("sampling_power_used", math.nan)),
        "active_pool_quantile_used": float(selection_diag.get("active_pool_quantile_used", math.nan)),
        "selected_boundary_band_fraction_region": selected_region_row.get("fraction_boundary_band", math.nan),
        "selected_fflo_interior_fraction_region": selected_region_row.get("fraction_fflo_interior", math.nan),
        "active_pool_boundary_band_fraction_region": active_region_row.get("fraction_boundary_band", math.nan),
        "active_pool_fflo_interior_fraction_region": active_region_row.get("fraction_fflo_interior", math.nan),
        "selected_A0_main_mean": safe_mean(joined, "A0_main"),
        "selected_A_phase_mean": safe_mean(joined, "A_phase"),
        "selected_A_numerical_mean": safe_mean(joined, "A_numerical"),
        "selected_A_explore_mean": safe_mean(joined, "A_explore"),
        "selected_A_response_mean": safe_mean(joined, "A_response"),
        "selected_q_edge_risk_score_mean": safe_mean(joined, "q_edge_risk_score"),
        "selected_E_q_SC_mean": safe_mean(joined, "E_q_SC"),
        "selected_B_delta_mean": safe_mean(joined, "B_delta"),
        "selected_B_q_SC_mean": safe_mean(joined, "B_q_SC"),
        "selected_boundary_distance_median": float(
            pd.to_numeric(joined.get("selected_to_predicted_boundary_distance"), errors="coerce").median()
        )
        if "selected_to_predicted_boundary_distance" in joined.columns
        else math.nan,
        "prediction_mismatch_count": int(joined["phase_prediction_mismatch"].sum()),
        "prediction_mismatch_rate_stop_definition": float(
            joined["phase_prediction_mismatch"].sum() / max(len(selected_with_stop_label), 1)
        ),
        "selected_points_with_stop_exact_label": int(len(selected_with_stop_label)),
        "selected_points_with_dataset_label": int(len(selected_with_dataset_label)),
        "new_unique_appended_points_matched": int(len(appended)),
    }
    row.update(value_counts_prefixed(appended["phase_label_after"], "appended_phase"))
    row.update(value_counts_prefixed(selected_with_dataset_label["phase_label_after"], "selected_dataset_phase"))
    row.update(value_counts_prefixed(joined["predicted_phase_before_exact"], "predicted_phase"))
    return row


def build_tables(iterations: list[int], tables_dir: Path) -> dict[str, pd.DataFrame]:
    loaded = [load_iteration(it) for it in iterations]
    selected_rows = []
    for item in loaded:
        joined = item["joined"].copy()
        joined.insert(0, "iteration", item["iteration"])
        selected_rows.append(joined)
    selected_points = pd.concat(selected_rows, ignore_index=True)
    selected_points.to_csv(tables_dir / "last5_selected_point_audit.csv", index=False)

    summary = pd.DataFrame([summarize_iteration(item) for item in loaded])
    summary.to_csv(tables_dir / "last5_selection_decomposition.csv", index=False)

    stop_rows = []
    for item in loaded:
        stop = item["stop"]
        thresholds = stop.get("thresholds", {})
        metrics = stop.get("metrics", {})
        conditions = stop.get("conditions", {})
        mapping = [
            ("C1_phase_map_change", "phase_map_change", "map_tol", "<"),
            ("C2_boundary_shift_normal_sc", "boundary_shift_normal_sc", "boundary_shift_tol", "<="),
            ("C3_boundary_shift_uniform_fflo", "boundary_shift_uniform_fflo", "boundary_shift_tol", "<="),
            ("C4_label_surprise_rate", "label_surprise_rate", "surprise_tol", "<="),
            ("C5_boundary_coverage_p95", "boundary_coverage_p95", "coverage_tol", "<="),
        ]
        for condition, metric, threshold, relation in mapping:
            stop_rows.append(
                {
                    "iteration": item["iteration"],
                    "condition": condition,
                    "metric": metric,
                    "value": metrics.get(metric, math.nan),
                    "threshold": thresholds.get(threshold, math.nan),
                    "relation": relation,
                    "passed": conditions.get(condition, False),
                }
            )
    stop_table = pd.DataFrame(stop_rows)
    stop_table.to_csv(tables_dir / "last5_stop_condition_audit.csv", index=False)

    component_cols = [
        "A0_main",
        "A_phase",
        "A_numerical",
        "A_explore",
        "A_response",
        "B_delta",
        "B_q_SC",
        "q_edge_risk_score",
        "E_q_SC",
        "extrapolation_risk_score",
        "selected_to_predicted_boundary_distance",
    ]
    component_rows = []
    for it, group in selected_points.groupby("iteration"):
        for col in component_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce")
                component_rows.append(
                    {
                        "iteration": it,
                        "component": col,
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "p90": float(values.quantile(0.9)),
                        "p95": float(values.quantile(0.95)),
                    }
                )
    component_summary = pd.DataFrame(component_rows)
    component_summary.to_csv(tables_dir / "last5_acquisition_component_summary.csv", index=False)

    confusion_source = selected_points[selected_points["stop_exact_phase_label"].notna()].copy()
    confusion = (
        confusion_source.groupby(["iteration", "predicted_phase_name", "actual_phase_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["iteration", "predicted_phase_name", "actual_phase_name"])
    )
    confusion.to_csv(tables_dir / "last5_label_surprise_confusion.csv", index=False)

    root_causes = pd.DataFrame(
        [
            {
                "hypothesis": "H1 phase map is still moving",
                "status": "not supported",
                "evidence": "C1-C3 pass in the final five iterations except no final failures; phase_map_change and boundary shifts are below tolerance at iteration 30.",
            },
            {
                "hypothesis": "H2 label surprise keeps convergence from passing",
                "status": "confirmed",
                "evidence": "C4_label_surprise_rate is false in every last-five iteration; final label_surprise_rate is 0.18359375 > 0.05.",
            },
            {
                "hypothesis": "H3 boundary coverage slightly misses tolerance",
                "status": "confirmed",
                "evidence": "C5_boundary_coverage_p95 is false in the last-five iterations; final 0.006588078458684216 > 0.00625.",
            },
            {
                "hypothesis": "H4 full acquisition is still selecting q-risk / rerun-heavy points",
                "status": "supported",
                "evidence": "Final q_edge_trigger_rate is 0.66015625 and rerun_required_rate is 0.36328125; exact batch has 169 q-expanded and 93 rerun-required points.",
            },
            {
                "hypothesis": "H5 rankcap_k3 local refinement caused stop failure",
                "status": "not supported",
                "evidence": "Corrected local-refinement gate passes with max local boxes = 3; stop failure is from AL metrics, not local-box cap violation.",
            },
        ]
    )
    root_causes.to_csv(tables_dir / "last5_stop_failure_root_cause.csv", index=False)

    return {
        "selected_points": selected_points,
        "summary": summary,
        "stop_table": stop_table,
        "component_summary": component_summary,
        "confusion": confusion,
        "root_causes": root_causes,
    }


def plot_figures(tables: dict[str, pd.DataFrame], figures_dir: Path) -> list[str]:
    paths: list[str] = []
    summary = tables["summary"]
    stop_table = tables["stop_table"]
    selected = tables["selected_points"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for metric in ["phase_map_change", "boundary_shift_normal_sc", "boundary_shift_uniform_fflo"]:
        ax.plot(summary["iteration"], summary[metric], marker="o", label=metric)
    ax.axhline(0.002, color="tab:gray", linestyle="--", linewidth=1, label="map_tol")
    ax.axhline(0.004166666666666667, color="tab:brown", linestyle=":", linewidth=1, label="boundary_shift_tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Stability metric")
    ax.set_title("Last-five phase-map and boundary-shift stability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "last5_phase_stability_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["iteration"], summary["label_surprise_rate_stop"], marker="o", label="label_surprise_rate")
    ax.axhline(float(summary["surprise_tol"].iloc[-1]), color="tab:red", linestyle="--", label="surprise_tol")
    ax2 = ax.twinx()
    ax2.plot(summary["iteration"], summary["boundary_coverage_p95"], marker="s", color="tab:orange", label="boundary_coverage_p95")
    ax2.axhline(float(summary["coverage_tol"].iloc[-1]), color="tab:orange", linestyle=":", label="coverage_tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Label surprise")
    ax2.set_ylabel("Boundary coverage p95")
    ax.set_title("Final stop failures: label surprise and coverage")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "last5_failed_stop_metrics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(len(summary))
    for offset, col, label in [
        (-width, "appended_phase_normal", "normal"),
        (0, "appended_phase_uniform_SC", "uniform_SC"),
        (width, "appended_phase_FFLO", "FFLO"),
    ]:
        values = summary[col] if col in summary.columns else 0
        ax.bar(x + offset, values, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["iteration"].astype(str))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("New dataset labels appended")
    ax.set_title("Last-five newly appended phase labels")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "last5_appended_phase_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["iteration"], summary["q_edge_trigger_rate"], marker="o", label="q_edge_trigger_rate")
    ax.plot(summary["iteration"], summary["rerun_required_rate_stop"], marker="s", label="rerun_required_rate")
    ax.plot(summary["iteration"], summary["selected_A0_ratio"], marker="^", label="selected_A0_ratio")
    ax.axhline(0.01, color="tab:gray", linestyle="--", linewidth=1, label="q/rerun tol")
    ax.axhline(0.15, color="tab:purple", linestyle=":", linewidth=1, label="A0 ratio tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Diagnostic rate / ratio")
    ax.set_title("Diagnostics show acquisition remains active")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "last5_diagnostic_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in ["selected_A_phase_mean", "selected_A_numerical_mean", "selected_A_explore_mean", "selected_A_response_mean"]:
        ax.plot(summary["iteration"], summary[col], marker="o", label=col.replace("selected_", ""))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean selected score component")
    ax.set_title("Selected-point acquisition component means")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "last5_acquisition_component_means.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        selected["kT"],
        selected["JA"],
        c=selected["iteration"],
        s=12,
        cmap="viridis",
        alpha=0.75,
        linewidths=0,
    )
    ax.set_xlabel("kBT")
    ax.set_ylabel("JA")
    ax.set_title("Last-five selected points in parameter space")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Iteration")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "last5_selected_points_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    return paths


def markdown_report(tables: dict[str, pd.DataFrame], figure_paths: list[str], report_root: Path) -> str:
    summary = tables["summary"]
    final = summary.iloc[-1]
    root_causes = tables["root_causes"]
    confusion = tables["confusion"]
    last_iters = ", ".join(str(int(x)) for x in summary["iteration"])
    final_confusion = confusion[confusion["iteration"] == final["iteration"]]
    top_confusion = final_confusion.sort_values("count", ascending=False).head(8)

    rel_figs = [Path(p).relative_to(report_root).as_posix() for p in figure_paths]

    lines = [
        "# Last-5-Iteration Selection Decomposition and Stop-Failure Audit",
        "",
        "## Executive Summary",
        "",
        f"Audited acquisition iterations `{last_iters}` from `active_boundary_discovery_rankcap_k3_full_loop_v1`.",
        "This is a report-only audit.  No acquisition, exact-oracle, rankcap, StopController, or tolerance code was changed.",
        "",
        "Main conclusion: the full loop did not formally converge because late-stage selection still produced too much label surprise and slightly insufficient boundary coverage, while the phase map itself was already stable.",
        "",
        "Final StopController state:",
        "",
        "```text",
        f"stop_reason = {final['stop_reason']}",
        f"convergence_pass = {final['convergence_pass']}",
        f"passed_condition_count = {final['passed_condition_count']}",
        f"required_pass_count = {final['required_pass_count']}",
        "```",
        "",
        "Final failed conditions:",
        "",
        "```text",
        f"label_surprise_rate = {final['label_surprise_rate_stop']:.6f} > surprise_tol = {final['surprise_tol']}",
        f"boundary_coverage_p95 = {final['boundary_coverage_p95']:.6f} > coverage_tol = {final['coverage_tol']}",
        "```",
        "",
        "Final passed stability conditions:",
        "",
        "```text",
        f"phase_map_change = {final['phase_map_change']:.6f}",
        f"boundary_shift_normal_sc = {final['boundary_shift_normal_sc']:.6f}",
        f"boundary_shift_uniform_fflo = {final['boundary_shift_uniform_fflo']:.6f}",
        "```",
        "",
        "## Last-Five Selection Decomposition",
        "",
        "Key per-iteration table:",
        "",
        "| iteration | exact | appended normal | appended uniform_SC | appended FFLO | label surprise | coverage p95 | q-edge rate | rerun rate | boundary-band fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {iteration:d} | {exact:d} | {normal:d} | {uniform:d} | {fflo:d} | {surprise:.4f} | {coverage:.6f} | {qedge:.4f} | {rerun:.4f} | {bb:.4f} |".format(
                iteration=int(row["iteration"]),
                exact=int(row["exact_points"]),
                normal=int_count(row.get("appended_phase_normal", 0)),
                uniform=int_count(row.get("appended_phase_uniform_SC", 0)),
                fflo=int_count(row.get("appended_phase_FFLO", 0)),
                surprise=float(row["label_surprise_rate_stop"]),
                coverage=float(row["boundary_coverage_p95"]),
                qedge=float(row["q_edge_trigger_rate"]),
                rerun=float(row["rerun_required_rate_stop"]),
                bb=float(row["selected_boundary_band_fraction_region"]),
            )
        )
    lines += [
        "",
        "The final acquisition batch remained highly active:",
        "",
        "```text",
        f"active_pool_size = {int(final['active_pool_size'])}",
        f"N_eff = {final['N_eff']:.3f}",
        f"N_eff / active_pool = {final['N_eff_over_active_pool_size']:.5f}",
        f"selected_A0_ratio = {final['selected_A0_ratio']:.6f}",
        f"q_edge_trigger_rate = {final['q_edge_trigger_rate']:.6f}",
        f"rerun_required_rate = {final['rerun_required_rate_stop']:.6f}",
        "```",
        "",
        "## Label Surprise Trace",
        "",
        "Top final-iteration predicted/actual phase pairs:",
        "",
        "| predicted | actual | count |",
        "|---|---|---:|",
    ]
    for _, row in top_confusion.iterrows():
        lines.append(f"| {row['predicted_phase_name']} | {row['actual_phase_name']} | {int(row['count'])} |")
    lines += [
        "",
        "The mismatch table uses the same phase-label definition as StopController: `phase_label(delta_opt, q_opt, delta_eps, q_eps)` on the exact merged batch, compared with `predicted_phase_before_exact`.  It audits C4 label surprise and does not redefine the thermodynamic criterion.",
        "",
        "## Root-Cause Candidates",
        "",
        "| hypothesis | status | evidence |",
        "|---|---|---|",
    ]
    for _, row in root_causes.iterrows():
        lines.append(f"| {row['hypothesis']} | {row['status']} | {row['evidence']} |")

    lines += [
        "",
        "## Figures",
        "",
    ]
    for fig in rel_figs:
        lines.append(f"![{Path(fig).stem}]({fig})")
        lines.append("")

    lines += [
        "## Output Tables",
        "",
        "```text",
        "tables/last5_selection_decomposition.csv",
        "tables/last5_stop_condition_audit.csv",
        "tables/last5_acquisition_component_summary.csv",
        "tables/last5_label_surprise_confusion.csv",
        "tables/last5_selected_point_audit.csv",
        "tables/last5_stop_failure_root_cause.csv",
        "```",
        "",
        "## Do-Not-Claim List",
        "",
        "1. Do not claim the active-learning loop formally converged.",
        "2. Do not blame rankcap_k3 local refinement for the stop failure; corrected local-box validation passes.",
        "3. Do not change acquisition, oracle, StopController, or thresholds based on this audit alone.",
        "4. Do not interpret label surprise as a change in thermodynamic phase criterion.",
        "5. Do not claim the old 20-iteration run is a strict A/B baseline; it used an earlier q-delta discovery workflow.",
        "",
        "## Next Step",
        "",
        "If formal convergence is required, the next calculation should be a deliberately designed report-backed late-stage cleanup validation that targets label surprise and boundary coverage.  It should be planned separately and should not silently change the production full-acquisition result.",
        "",
    ]
    return "\n".join(lines)


def latex_escape(text: object) -> str:
    s = str(text)
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
    return "".join(repl.get(ch, ch) for ch in s)


def int_count(value: object) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except Exception:
        pass
    return int(value)


def write_latex_pdf(markdown_text: str, figure_paths: list[str], report_root: Path) -> tuple[Path, Path | None, str]:
    tex_path = report_root / "last5_selection_stop_audit.tex"
    pdf_path = report_root / "last5_selection_stop_audit.pdf"
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\usepackage{xcolor}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\begin{document}",
        r"\title{Last-5-Iteration Selection Decomposition and Stop-Failure Audit}",
        r"\author{report-only analysis}",
        r"\date{2026-06-17}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        "The rankcap\\_k3 full loop passed corrected oracle validation, but it did not formally converge under the StopController.  The final stop reason is \\texttt{max\\_iterations}.  The phase map and boundary shifts are stable; the remaining failures are label surprise and boundary coverage.",
        r"\section*{Final Stop State}",
        r"\begin{verbatim}",
    ]
    # Extract compact verbatim snippets from the Markdown to avoid fragile parsing.
    snippets = []
    capture = False
    for line in markdown_text.splitlines():
        if line.strip() == "```text":
            capture = True
            snippets.append([])
            continue
        if line.strip() == "```" and capture:
            capture = False
            continue
        if capture and len(snippets) <= 3:
            snippets[-1].append(line)
    compact = "\n\n".join("\n".join(s) for s in snippets[:3])
    lines += [compact, r"\end{verbatim}"]
    lines += [
        r"\section*{Figures}",
    ]
    for path in figure_paths:
        rel = Path(path).relative_to(report_root).as_posix()
        lines += [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{{latex_escape(rel)}}}",
            rf"\caption{{{latex_escape(Path(path).stem.replace('_', ' '))}}}",
            r"\end{figure}",
        ]
    lines += [
        r"\section*{Conclusion}",
        "The last five iterations show a stable phase map but continuing acquisition activity in hard-risk regions.  This audit is evidence for a late-stage selection/coverage issue, not a local-refinement failure.",
        r"\end{document}",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    status = "not_run"
    if shutil.which("pdflatex"):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=report_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_path.name],
                cwd=report_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
            status = "pass"
        except Exception as exc:
            status = f"failed: {exc}"
    else:
        status = "pdflatex_missing"
    return tex_path, pdf_path if pdf_path.exists() else None, status


def write_decision_log(tables: dict[str, pd.DataFrame], report_root: Path, pdf_status: str) -> None:
    summary = tables["summary"]
    final = summary.iloc[-1]
    text = f"""# Decision Log

Date: 2026-06-17

Decision-level conclusion:

```text
rankcap_k3 full-loop local-refinement validation remains accepted.
formal active-learning convergence is not established.
stop failure is attributed to late-stage selection / stop metrics, not local-box target explosion.
```

Final evidence:

```text
stop_reason = {final['stop_reason']}
convergence_pass = {final['convergence_pass']}
label_surprise_rate = {final['label_surprise_rate_stop']}
boundary_coverage_p95 = {final['boundary_coverage_p95']}
q_edge_trigger_rate = {final['q_edge_trigger_rate']}
rerun_required_rate = {final['rerun_required_rate_stop']}
```

PDF generation status:

```text
{pdf_status}
```

Next recommended action:

```text
Plan a separate late-stage cleanup validation only if formal convergence is required.
Do not silently alter acquisition, StopController, thresholds, exact oracle, or rankcap_k3.
```
"""
    (report_root / "decision_log.md").write_text(text, encoding="utf-8")


def mirror_report(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    if not RUN_ROOT.exists():
        raise FileNotFoundError(RUN_ROOT)
    if PACKAGE_REPORT_ROOT.exists():
        shutil.rmtree(PACKAGE_REPORT_ROOT)
    tables_dir, figures_dir = ensure_dirs(PACKAGE_REPORT_ROOT)
    stop_history = read_json(RUN_ROOT / "stop_metrics_history.json")
    iterations = [int(row["iteration"]) for row in stop_history[-LAST_N:]]
    tables = build_tables(iterations, tables_dir)
    figure_paths = plot_figures(tables, figures_dir)
    md = markdown_report(tables, figure_paths, PACKAGE_REPORT_ROOT)
    md_path = PACKAGE_REPORT_ROOT / "last5_selection_stop_audit.md"
    md_path.write_text(md, encoding="utf-8")
    tex_path, pdf_path, pdf_status = write_latex_pdf(md, figure_paths, PACKAGE_REPORT_ROOT)
    write_decision_log(tables, PACKAGE_REPORT_ROOT, pdf_status)
    summary = {
        "run_root": str(RUN_ROOT),
        "iterations": iterations,
        "markdown": str(md_path),
        "tex": str(tex_path),
        "pdf": str(pdf_path) if pdf_path else "",
        "pdf_status": pdf_status,
        "tables_dir": str(tables_dir),
        "figures_dir": str(figures_dir),
        "final_stop_reason": str(tables["summary"].iloc[-1]["stop_reason"]),
        "final_convergence_pass": bool(tables["summary"].iloc[-1]["convergence_pass"]),
    }
    (PACKAGE_REPORT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mirror_report(PACKAGE_REPORT_ROOT, MIRROR_REPORT_ROOT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
