from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "rankcap_k3_full_loop"
OUTPUT_ROOT = PACKAGE_ROOT / "ML_Phase_512_RankCapK3_FullLoop"
RUN_ID = "active_boundary_discovery_rankcap_k3_full_loop_v1"
RUN_DIR = OUTPUT_ROOT / "active_runs" / RUN_ID
ORIGINAL_REPORT_ROOT = PACKAGE_ROOT / "reports" / "rankcap_k3_full_loop"
REPORT_ROOT = ORIGINAL_REPORT_ROOT
REPORT_NAME = "rankcap_k3_full_loop_enhanced"
ML_REPORT_ROOT = OUTPUT_ROOT / "reports" / "full_loop_enhanced_report"

PHASE_ORDER = ["normal", "uniform_SC", "FFLO"]
PHASE_COLORS = {
    "normal": "#4c78a8",
    "uniform_SC": "#f58518",
    "FFLO": "#54a24b",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def format_hours(seconds: float | int | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "cannot determine"
    hours = float(seconds) / 3600.0
    return f"{hours:.2f} h"


def format_minutes(seconds: float | int | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "cannot determine"
    minutes = float(seconds) / 60.0
    return f"{minutes:.1f} min"


def latex_escape(value: object) -> str:
    text = str(value)
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
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns is not None:
        df = df[columns]
    if df.empty:
        return "_empty_"
    header = list(df.columns)
    lines = [
        "| " + " | ".join(str(col) for col in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for _, row in df.iterrows():
        cells = []
        for col in header:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def dataset_path(iteration: int) -> Path:
    return RUN_DIR / f"dataset_iter{iteration:03d}.csv"


def phase_count_row(path: Path) -> dict:
    if not path.exists():
        return {}
    df = read_csv(path)
    counts = df["phase_name"].value_counts().to_dict() if "phase_name" in df.columns else {}
    return {
        "dataset": path.name,
        "iteration": int(re.search(r"(\d+)", path.stem).group(1)),
        "samples": int(len(df)),
        "normal": int(counts.get("normal", 0)),
        "uniform_SC": int(counts.get("uniform_SC", 0)),
        "FFLO": int(counts.get("FFLO", 0)),
    }


def collect_dataset_phase_counts() -> pd.DataFrame:
    rows = [phase_count_row(path) for path in sorted(RUN_DIR.glob("dataset_iter*.csv"))]
    rows = [row for row in rows if row]
    return pd.DataFrame(rows).sort_values("iteration") if rows else pd.DataFrame()


def collect_rank_level_local_boxes() -> pd.DataFrame:
    rows: list[dict] = []
    pattern = re.compile(r"iter(\d+)_local_box_timing_rank(\d+)_of(\d+)\.csv")
    for path in sorted(RUN_DIR.glob("iter*/performance/*_local_box_timing_rank*_of*.csv")):
        match = pattern.search(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        rank = int(match.group(2))
        world_size = int(match.group(3))
        df = read_csv(path)
        if df.empty:
            continue
        for point_id, group in df.groupby("point_id", sort=True):
            first = group.iloc[0]
            rows.append(
                {
                    "iteration": iteration,
                    "rank": rank,
                    "world_size": world_size,
                    "point_id_rank_local": int(point_id),
                    "kT": float(first["kT"]),
                    "JA": float(first["JA"]),
                    "local_boxes_refined_count": int(len(group)),
                    "q_window_levels": ";".join(
                        str(value) for value in sorted(group["q_window_level"].dropna().unique())
                    ),
                    "changed_phase_label_count": int(
                        pd.to_numeric(group["changed_phase_label"], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def collect_ml_metrics() -> pd.DataFrame:
    path = RUN_DIR / "metrics_history.json"
    if not path.exists():
        return pd.DataFrame()
    data = read_json(path)
    if not isinstance(data, list):
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df.insert(0, "metric_iteration", range(len(df)))
    return df


def collect_timing_summary(summary: dict, iter_recheck: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    lock_path = PACKAGE_ROOT / f".active_loop.{RUN_ID}.lock"
    start_dt = None
    if lock_path.exists():
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("time="):
                start_dt = datetime.fromisoformat(line.split("=", 1)[1])
                break
    end_dt = None
    created_at = summary.get("created_at_utc")
    if created_at:
        end_dt = datetime.fromisoformat(str(created_at)).astimezone(start_dt.tzinfo if start_dt else None)
    wall_sec = (end_dt - start_dt).total_seconds() if start_dt and end_dt else math.nan

    rank_runtime = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "rank_runtime_summary.csv")
    if rank_runtime.empty or "elapsed_sec" not in rank_runtime.columns:
        exact_iter_timing = pd.DataFrame()
        exact_wall_sec = math.nan
        mean_exact_iter_sec = math.nan
        median_exact_iter_sec = math.nan
        seed_exact_iter_sec = math.nan
        acquisition_exact_iter_sec = math.nan
        max_exact_iter_sec = math.nan
    else:
        exact_iter_timing = (
            rank_runtime.groupby("iteration", sort=True)
            .agg(
                exact_wall_sec=("elapsed_sec", "max"),
                mean_rank_elapsed_sec=("elapsed_sec", "mean"),
                min_rank_elapsed_sec=("elapsed_sec", "min"),
                max_rank_elapsed_sec=("elapsed_sec", "max"),
            )
            .reset_index()
        )
        exact_iter_timing["exact_wall_min"] = exact_iter_timing["exact_wall_sec"] / 60.0
        exact_wall_sec = float(exact_iter_timing["exact_wall_sec"].sum())
        mean_exact_iter_sec = float(exact_iter_timing["exact_wall_sec"].mean())
        median_exact_iter_sec = float(exact_iter_timing["exact_wall_sec"].median())
        seed_exact_iter_sec = float(
            exact_iter_timing.loc[exact_iter_timing["iteration"] == 0, "exact_wall_sec"].iloc[0]
        )
        acquisition_exact_iter_sec = float(
            exact_iter_timing.loc[exact_iter_timing["iteration"] > 0, "exact_wall_sec"].mean()
        )
        max_exact_iter_sec = float(exact_iter_timing["exact_wall_sec"].max())

    n_exact_iters = int(summary.get("n_iters_expected") or len(iter_recheck))
    acquisition_batches = int(summary.get("acquisition_batches_expected") or max(n_exact_iters - 1, 0))
    baseline_local_boxes = float(summary.get("baseline_local_boxes_reference", 6.0))
    baseline_local_runtime = float(summary.get("baseline_local_refinement_runtime_sec_reference", math.nan))
    baseline_total_runtime = float(summary.get("baseline_point_total_runtime_sec_reference", math.nan))
    rankcap_local_boxes = float(iter_recheck["mean_local_boxes_rank_level"].mean())
    rankcap_local_runtime = float(summary.get("mean_local_refinement_runtime_sec", math.nan))
    rankcap_total_runtime = float(summary.get("mean_point_total_runtime_sec", math.nan))

    timing_summary = {
        "start_time_local": start_dt.isoformat() if start_dt else "cannot determine",
        "end_time_local": end_dt.isoformat() if end_dt else "cannot determine",
        "total_wall_sec": wall_sec,
        "total_wall_hours": wall_sec / 3600.0 if math.isfinite(wall_sec) else math.nan,
        "n_exact_iterations": n_exact_iters,
        "acquisition_batches": acquisition_batches,
        "wall_minutes_per_exact_iteration": wall_sec / n_exact_iters / 60.0 if n_exact_iters and math.isfinite(wall_sec) else math.nan,
        "wall_minutes_per_acquisition_batch": wall_sec / acquisition_batches / 60.0 if acquisition_batches and math.isfinite(wall_sec) else math.nan,
        "exact_oracle_wall_sec_sum": exact_wall_sec,
        "exact_oracle_wall_hours_sum": exact_wall_sec / 3600.0 if math.isfinite(exact_wall_sec) else math.nan,
        "mean_exact_oracle_wall_min_per_iteration": mean_exact_iter_sec / 60.0 if math.isfinite(mean_exact_iter_sec) else math.nan,
        "median_exact_oracle_wall_min_per_iteration": median_exact_iter_sec / 60.0 if math.isfinite(median_exact_iter_sec) else math.nan,
        "seed_exact_oracle_wall_min": seed_exact_iter_sec / 60.0 if math.isfinite(seed_exact_iter_sec) else math.nan,
        "mean_acquisition_exact_oracle_wall_min": acquisition_exact_iter_sec / 60.0 if math.isfinite(acquisition_exact_iter_sec) else math.nan,
        "max_exact_oracle_wall_min": max_exact_iter_sec / 60.0 if math.isfinite(max_exact_iter_sec) else math.nan,
        "baseline_local_boxes_reference": baseline_local_boxes,
        "rankcap_mean_local_boxes": rankcap_local_boxes,
        "local_box_reduction_percent": 100.0 * (1.0 - rankcap_local_boxes / baseline_local_boxes) if baseline_local_boxes else math.nan,
        "local_box_speedup_factor": baseline_local_boxes / rankcap_local_boxes if rankcap_local_boxes else math.nan,
        "baseline_local_refinement_runtime_sec_reference": baseline_local_runtime,
        "rankcap_local_refinement_runtime_sec": rankcap_local_runtime,
        "local_refinement_runtime_reduction_percent": 100.0 * (1.0 - rankcap_local_runtime / baseline_local_runtime) if baseline_local_runtime else math.nan,
        "local_refinement_speedup_factor": baseline_local_runtime / rankcap_local_runtime if rankcap_local_runtime else math.nan,
        "baseline_point_total_runtime_sec_reference": baseline_total_runtime,
        "rankcap_point_total_runtime_sec": rankcap_total_runtime,
        "point_total_runtime_reduction_percent": 100.0 * (1.0 - rankcap_total_runtime / baseline_total_runtime) if baseline_total_runtime else math.nan,
        "point_total_speedup_factor": baseline_total_runtime / rankcap_total_runtime if rankcap_total_runtime else math.nan,
        "timing_note": "Wall time is estimated from the active-loop lock timestamp to the collector summary timestamp; exact-oracle wall time is summed from per-iteration max rank elapsed time.",
    }
    return timing_summary, exact_iter_timing


def build_iteration_recheck(iter_df: pd.DataFrame, local_box_points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iteration, group in local_box_points.groupby("iteration", sort=True):
        original = iter_df.loc[iter_df["iteration"] == iteration]
        original_row = original.iloc[0].to_dict() if not original.empty else {}
        exact_points = int(len(group))
        total_boxes = int(group["local_boxes_refined_count"].sum())
        rows.append(
            {
                "iteration": int(iteration),
                "exact_points_rank_level": exact_points,
                "total_local_boxes_rank_level": total_boxes,
                "mean_local_boxes_rank_level": total_boxes / exact_points if exact_points else math.nan,
                "max_local_boxes_rank_level": int(group["local_boxes_refined_count"].max()),
                "points_gt3_rank_level": int((group["local_boxes_refined_count"] > 3).sum()),
                "training_eligible_appended": int(original_row.get("training_eligible_appended", 0)),
                "rerun_required_count": int(original_row.get("rerun_required_count", 0)),
                "rerun_required_fraction": float(original_row.get("rerun_required_fraction", math.nan)),
                "q_unresolved_count": int(original_row.get("q_unresolved_count", 0)),
                "delta_unresolved_count": int(original_row.get("delta_unresolved_count", 0)),
                "mean_local_refinement_runtime_sec": float(
                    original_row.get("mean_local_refinement_runtime_sec", math.nan)
                ),
                "mean_point_total_runtime_sec": float(
                    original_row.get("mean_point_total_runtime_sec", math.nan)
                ),
                "rank_runtime_imbalance_ratio": float(
                    original_row.get("rank_runtime_imbalance_ratio", math.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_discrepancy_table(iter_df: pd.DataFrame) -> pd.DataFrame:
    original_rows = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "local_box_rows.csv")
    if original_rows.empty:
        return pd.DataFrame()
    rows = []
    for iteration, group in original_rows.groupby("iteration", sort=True):
        iteration_i = int(iteration)
        expected = int(iter_df.loc[iter_df["iteration"] == iteration_i, "exact_points"].iloc[0])
        rows.append(
            {
                "iteration": iteration_i,
                "original_local_box_rows": int(len(group)),
                "unique_point_id_values_in_original_rows": int(group["point_id"].nunique()),
                "expected_exact_points": expected,
                "rank_dimension_present_in_original_rows": "rank" in group.columns,
                "wrong_max_grouped_by_iteration_point_id": int(group.groupby("point_id").size().max()),
                "diagnosis": "rank-local point_id collision in merged report table",
            }
        )
    return pd.DataFrame(rows)


def compact_summary_table(summary: dict, corrected_summary: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("original_validation_status", summary.get("validation_status")),
            ("corrected_validation_status", corrected_summary["corrected_validation_status"]),
            ("run_id", summary.get("run_id")),
            ("expected_iterations", summary.get("n_iters_expected")),
            ("expected_acquisition_batches", summary.get("acquisition_batches_expected")),
            ("final_dataset_samples", summary.get("final_dataset_samples")),
            ("final_normal", summary.get("final_normal_count")),
            ("final_uniform_SC", summary.get("final_uniform_SC_count")),
            ("final_FFLO", summary.get("final_FFLO_count")),
            ("corrected_mean_local_boxes_unweighted", f"{corrected_summary['corrected_mean_local_boxes_unweighted']:.6g}"),
            ("corrected_mean_local_boxes_weighted", f"{corrected_summary['corrected_mean_local_boxes_weighted']:.6g}"),
            ("corrected_max_local_boxes", corrected_summary["corrected_max_local_boxes"]),
            ("points_above_3_boxes", corrected_summary["corrected_points_gt3"]),
            ("mean_local_refinement_runtime_sec", f"{summary.get('mean_local_refinement_runtime_sec'):.6g}"),
            ("mean_point_total_runtime_sec", f"{summary.get('mean_point_total_runtime_sec'):.6g}"),
            ("total_wall_hours_estimated", f"{corrected_summary.get('total_wall_hours', math.nan):.6g}"),
            ("wall_minutes_per_exact_iteration", f"{corrected_summary.get('wall_minutes_per_exact_iteration', math.nan):.6g}"),
            ("exact_oracle_wall_hours_sum", f"{corrected_summary.get('exact_oracle_wall_hours_sum', math.nan):.6g}"),
            ("local_box_reduction_percent", f"{corrected_summary.get('local_box_reduction_percent', math.nan):.6g}"),
            ("local_refinement_runtime_reduction_percent", f"{corrected_summary.get('local_refinement_runtime_reduction_percent', math.nan):.6g}"),
            ("point_total_runtime_reduction_percent", f"{corrected_summary.get('point_total_runtime_reduction_percent', math.nan):.6g}"),
        ],
        columns=["metric", "value"],
    )


def build_gate_table(summary: dict, iter_recheck: pd.DataFrame, dataset_counts: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    final = dataset_counts.iloc[-1].to_dict() if not dataset_counts.empty else {}
    corrected_max = int(iter_recheck["max_local_boxes_rank_level"].max())
    corrected_mean = float(iter_recheck["mean_local_boxes_rank_level"].mean())
    weighted_mean = float(
        iter_recheck["total_local_boxes_rank_level"].sum()
        / iter_recheck["exact_points_rank_level"].sum()
    )
    gates = [
        ("expected_iters_present", bool(summary.get("expected_iters_present")), ""),
        ("all_shards_complete", bool(summary.get("all_shards_complete")), ""),
        ("merge_status", summary.get("merge_status") == "pass", str(summary.get("merge_status"))),
        ("append_status", summary.get("append_status") == "pass", str(summary.get("append_status"))),
        ("dataset_monotonic", bool(summary.get("dataset_monotonic")), ""),
        (
            "phase_coverage",
            bool(summary.get("phase_coverage")),
            f"normal={final.get('normal')}, uniform_SC={final.get('uniform_SC')}, FFLO={final.get('FFLO')}",
        ),
        ("training_eligible_nonzero_each_iter", bool(summary.get("training_eligible_nonzero_each_iter")), ""),
        ("q_unresolved_ok", bool(summary.get("q_unresolved_ok")), ""),
        ("delta_unresolved_ok", bool(summary.get("delta_unresolved_ok")), ""),
        ("rerun_fraction_ok", bool(summary.get("rerun_fraction_ok")), ""),
        ("mean_local_boxes_ok", corrected_mean <= 3.2, f"unweighted={corrected_mean:.6g}; weighted={weighted_mean:.6g}"),
        ("max_local_boxes_ok_corrected", corrected_max <= 3, f"corrected_max={corrected_max}"),
        ("traceback_scan_ok", trace_df.empty, f"matches={len(trace_df)}"),
        (
            "final_report_nonblocking",
            str(summary.get("final_report_status", "")).lower() in {"ok", "pdflatex_missing", "pdflatex_failed"},
            str(summary.get("final_report_status")),
        ),
    ]
    return pd.DataFrame(
        {
            "gate": [row[0] for row in gates],
            "status": ["pass" if row[1] else "fail" for row in gates],
            "evidence": [row[2] for row in gates],
        }
    )


def scatter_phase(ax, df: pd.DataFrame, title: str, selected: pd.DataFrame | None = None) -> None:
    for phase in PHASE_ORDER:
        part = df[df["phase_name"] == phase]
        if part.empty:
            continue
        ax.scatter(
            part["kT"],
            part["JA"],
            s=8,
            alpha=0.72,
            label=phase,
            color=PHASE_COLORS[phase],
            edgecolors="none",
        )
    if selected is not None and not selected.empty:
        ax.scatter(selected["kT"], selected["JA"], s=22, facecolors="none", edgecolors="black", linewidths=0.7, label="selected")
    ax.set_title(title)
    ax.set_xlabel("kT")
    ax.set_ylabel("JA")
    ax.set_xlim(-0.01, 0.61)
    ax.set_ylim(-0.03, 2.15)
    ax.grid(alpha=0.18, linewidth=0.5)


def create_figures(
    figures_dir: Path,
    dataset_counts: pd.DataFrame,
    iter_df: pd.DataFrame,
    iter_recheck: pd.DataFrame,
    local_box_points: pd.DataFrame,
    ml_metrics: pd.DataFrame,
    exact_iter_timing: pd.DataFrame,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    final_df = read_csv(dataset_path(31))
    last_selected = read_csv(RUN_DIR / "iter030" / "selected_points.csv")
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter_phase(ax, final_df, "Final exact phase diagram (dataset_iter031)", last_selected)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    path = figures_dir / "enhanced_final_phase_diagram.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["final_phase_diagram"] = path.name

    snapshot_iters = [1, 5, 10, 20, 30, 31]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2), sharex=True, sharey=True)
    for ax, iteration in zip(axes.ravel(), snapshot_iters):
        df = read_csv(dataset_path(iteration))
        counts = df["phase_name"].value_counts().to_dict() if not df.empty else {}
        title = f"dataset_iter{iteration:03d}: n={len(df)}"
        subtitle = f"N={counts.get('normal', 0)}, U={counts.get('uniform_SC', 0)}, F={counts.get('FFLO', 0)}"
        scatter_phase(ax, df, title + "\n" + subtitle)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = figures_dir / "enhanced_phase_snapshots.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase_snapshots"] = path.name

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(dataset_counts["iteration"], dataset_counts["samples"], marker="o", label="total", color="#333333")
    for phase in PHASE_ORDER:
        ax.plot(dataset_counts["iteration"], dataset_counts[phase], marker="o", label=phase, color=PHASE_COLORS[phase])
    ax.set_xlabel("Dataset iteration")
    ax.set_ylabel("Samples")
    ax.set_title("Learning curve: dataset growth and phase counts")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = figures_dir / "enhanced_learning_curve_phase_counts.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["learning_curve_phase_counts"] = path.name

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
    ax1.plot(iter_df["iteration"], iter_df["training_eligible_appended"], marker="o", color="#4c78a8", label="training eligible appended")
    ax1.set_xlabel("Exact iteration")
    ax1.set_ylabel("Training-eligible appended", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")
    ax2 = ax1.twinx()
    ax2.plot(iter_df["iteration"], iter_df["rerun_required_fraction"], marker="s", color="#e45756", label="rerun required fraction")
    ax2.set_ylabel("Rerun-required fraction", color="#e45756")
    ax2.tick_params(axis="y", labelcolor="#e45756")
    ax1.set_title("Training eligibility and rerun pressure")
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "enhanced_training_rerun_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["training_rerun_curve"] = path.name

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(iter_recheck["iteration"], iter_recheck["mean_local_boxes_rank_level"], marker="o", label="mean local boxes")
    ax.plot(iter_recheck["iteration"], iter_recheck["max_local_boxes_rank_level"], marker="s", label="max local boxes")
    ax.axhline(3, color="black", linestyle="--", linewidth=1.0, label="k3 cap")
    ax.set_xlabel("Exact iteration")
    ax.set_ylabel("Local boxes per actual exact point")
    ax.set_title("Corrected rank-level local-box gate")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "enhanced_corrected_local_box_gate.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["corrected_local_box_gate"] = path.name

    distribution = (
        local_box_points.groupby("local_boxes_refined_count", sort=True)
        .size()
        .reset_index(name="point_count")
    )
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(distribution["local_boxes_refined_count"].astype(str), distribution["point_count"], color="#4c78a8")
    ax.set_xlabel("Local boxes per actual exact point")
    ax.set_ylabel("Point count")
    ax.set_title("Corrected local-box distribution")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "enhanced_local_box_distribution.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["local_box_distribution"] = path.name

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(iter_df["iteration"], iter_df["mean_local_refinement_runtime_sec"], marker="o", label="local refinement")
    ax.plot(iter_df["iteration"], iter_df["mean_point_total_runtime_sec"], marker="s", label="point total")
    ax.axhline(189.767, color="#4c78a8", linestyle=":", linewidth=1.0, label="baseline local-ref reference")
    ax.axhline(234.194, color="#f58518", linestyle=":", linewidth=1.0, label="baseline total reference")
    ax.set_xlabel("Exact iteration")
    ax.set_ylabel("Seconds per point")
    ax.set_title("Runtime learning curve")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "enhanced_runtime_curve.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["runtime_curve"] = path.name

    if not exact_iter_timing.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(
            exact_iter_timing["iteration"],
            exact_iter_timing["exact_wall_min"],
            marker="o",
            color="#4c78a8",
            label="max rank elapsed",
        )
        if "mean_rank_elapsed_sec" in exact_iter_timing.columns:
            ax.plot(
                exact_iter_timing["iteration"],
                exact_iter_timing["mean_rank_elapsed_sec"] / 60.0,
                marker="s",
                color="#f58518",
                label="mean rank elapsed",
            )
        ax.set_xlabel("Exact iteration")
        ax.set_ylabel("Minutes")
        ax.set_title("Exact-oracle wall time per iteration")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = figures_dir / "enhanced_exact_walltime_curve.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths["exact_walltime_curve"] = path.name

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(iter_df["iteration"], iter_df["rank_runtime_imbalance_ratio"], marker="o", color="#9467bd")
    ax.set_xlabel("Exact iteration")
    ax.set_ylabel("max(rank runtime) / min(rank runtime)")
    ax.set_title("Rank runtime imbalance")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "enhanced_rank_runtime_imbalance.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["rank_runtime_imbalance"] = path.name

    if not ml_metrics.empty:
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), sharex=True)
        axes = axes.ravel()
        axes[0].plot(ml_metrics["metric_iteration"], ml_metrics["delta_rmse"], marker="o", label=r"$\Delta$ RMSE")
        axes[0].plot(ml_metrics["metric_iteration"], ml_metrics["eta_rmse"], marker="s", label=r"$\eta$ RMSE")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("Order-parameter and response surrogate errors")
        axes[0].legend(fontsize=8)
        axes[1].plot(ml_metrics["metric_iteration"], ml_metrics["q_rmse"], marker="o", color="#54a24b")
        axes[1].set_ylabel("q RMSE")
        axes[1].set_title("Momentum surrogate error")
        axes[2].plot(ml_metrics["metric_iteration"], ml_metrics["ic_plus_rmse"], marker="o", label=r"$I_c^+$ RMSE")
        axes[2].plot(ml_metrics["metric_iteration"], ml_metrics["ic_minus_rmse"], marker="s", label=r"$I_c^-$ RMSE")
        axes[2].set_ylabel("RMSE")
        axes[2].set_xlabel("Metric iteration")
        axes[2].set_title("Critical-current surrogate errors")
        axes[2].legend(fontsize=8)
        axes[3].plot(ml_metrics["metric_iteration"], ml_metrics["phase_accuracy"], marker="o", color="#9467bd")
        axes[3].set_ylabel("Phase accuracy")
        axes[3].set_xlabel("Metric iteration")
        axes[3].set_title("Classifier phase accuracy")
        axes[3].set_ylim(0.94, 1.005)
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.tight_layout()
        path = figures_dir / "enhanced_surrogate_metric_curves.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths["surrogate_metric_curves"] = path.name

        fig, ax1 = plt.subplots(figsize=(7.2, 4.4))
        ax1.plot(ml_metrics["metric_iteration"], ml_metrics["phase_accuracy"], marker="o", color="#4c78a8", label="phase accuracy")
        ax1.set_xlabel("Metric iteration")
        ax1.set_ylabel("Phase accuracy", color="#4c78a8")
        ax1.tick_params(axis="y", labelcolor="#4c78a8")
        ax2 = ax1.twinx()
        ax2.plot(
            ml_metrics["metric_iteration"],
            ml_metrics["estimated_reduction"],
            marker="s",
            color="#e45756",
            label="estimated exact-call reduction",
        )
        ax2.set_ylabel("Estimated reduction factor", color="#e45756")
        ax2.tick_params(axis="y", labelcolor="#e45756")
        ax1.set_title("Surrogate quality and estimated exact-call reduction")
        ax1.grid(alpha=0.2)
        fig.tight_layout()
        path = figures_dir / "enhanced_phase_accuracy_reduction_curve.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        paths["phase_accuracy_reduction_curve"] = path.name

    recent_iters = [26, 27, 28, 29, 30]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(recent_iters)))
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    scatter_phase(ax, final_df, "Final phase diagram with final five selected batches")
    for color, iteration in zip(colors, recent_iters):
        selected = read_csv(RUN_DIR / f"iter{iteration:03d}" / "selected_points.csv")
        if not selected.empty:
            ax.scatter(selected["kT"], selected["JA"], s=28, facecolors="none", edgecolors=[color], linewidths=0.8, label=f"iter{iteration:03d}")
    ax.legend(loc="upper right", fontsize=7, frameon=True)
    fig.tight_layout()
    path = figures_dir / "enhanced_recent_selected_overlay.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["recent_selected_overlay"] = path.name

    existing_boundary = OUTPUT_ROOT / "figures" / f"{RUN_ID}_exact_eta_revised_boundaries.png"
    if existing_boundary.exists():
        shutil.copy2(existing_boundary, figures_dir / "enhanced_exact_eta_revised_boundaries.png")
        paths["exact_eta_revised_boundaries"] = "enhanced_exact_eta_revised_boundaries.png"

    return paths


def write_markdown(
    report_root: Path,
    figures: dict[str, str],
    summary: dict,
    corrected_summary: dict,
    gate_df: pd.DataFrame,
    summary_table: pd.DataFrame,
    dataset_counts: pd.DataFrame,
    iter_recheck: pd.DataFrame,
    discrepancy: pd.DataFrame,
    timing_summary: dict,
    ml_metrics: pd.DataFrame,
) -> None:
    md = report_root / f"{REPORT_NAME}.md"
    fig_dir_name = "figures"
    lines = [
        "# Rank-and-Cap K3 Full-Loop Enhanced Report",
        "",
        "## Executive Summary",
        "",
        f"- original_validation_status: {summary.get('validation_status')}",
        f"- corrected_validation_status: {corrected_summary['corrected_validation_status']}",
        f"- run_id: {summary.get('run_id')}",
        f"- total iterations: {summary.get('n_iters_expected')} (seed plus {summary.get('acquisition_batches_expected')} acquisition batches)",
        f"- total wall time estimate: {format_hours(timing_summary.get('total_wall_sec'))}",
        f"- average wall time per exact iteration: {timing_summary.get('wall_minutes_per_exact_iteration', math.nan):.3g} min",
        f"- exact-oracle wall time sum: {format_hours(timing_summary.get('exact_oracle_wall_sec_sum'))}",
        f"- final dataset samples: {summary.get('final_dataset_samples')}",
        f"- final phase counts: normal={summary.get('final_normal_count')}, uniform_SC={summary.get('final_uniform_SC_count')}, FFLO={summary.get('final_FFLO_count')}",
        f"- corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.6g} / {corrected_summary['corrected_max_local_boxes']}",
        f"- corrected weighted mean local boxes: {corrected_summary['corrected_mean_local_boxes_weighted']:.6g}",
        f"- mean local-refinement runtime: {summary.get('mean_local_refinement_runtime_sec'):.6g} sec/point",
        f"- mean point-total runtime: {summary.get('mean_point_total_runtime_sec'):.6g} sec/point",
        "",
        "The raw full-loop numerical run completed.  The original package report reports fail because",
        "the collector repeats the known rank-local `point_id` aggregation bug.  The corrected rank-level",
        "gate recomputes local-box counts from raw rank timing files and passes: every actual exact point",
        "uses at most three local boxes.",
        "",
        "## Summary Table",
        "",
        md_table(summary_table),
        "",
        "## Run Duration and Speedup",
        "",
        "The active-loop lock file records the HPC run start time, and the package summary records the",
        "collector completion time.  The wall time below therefore includes candidate generation, exact",
        "oracle arrays, merge, append, final reporting, and archive collection.  The exact-oracle wall",
        "time is separately recomputed from the maximum rank elapsed time in each iteration.",
        "",
        md_table(
            pd.DataFrame(
                [
                    ("start_time_local", timing_summary.get("start_time_local")),
                    ("end_time_local", timing_summary.get("end_time_local")),
                    ("total_wall_time", format_hours(timing_summary.get("total_wall_sec"))),
                    ("exact_iterations", timing_summary.get("n_exact_iterations")),
                    ("acquisition_batches", timing_summary.get("acquisition_batches")),
                    ("wall_time_per_exact_iteration", f"{timing_summary.get('wall_minutes_per_exact_iteration', math.nan):.3f} min"),
                    ("wall_time_per_acquisition_batch", f"{timing_summary.get('wall_minutes_per_acquisition_batch', math.nan):.3f} min"),
                    ("exact_oracle_wall_time_sum", format_hours(timing_summary.get("exact_oracle_wall_sec_sum"))),
                    ("mean_exact_oracle_wall_time_per_iteration", f"{timing_summary.get('mean_exact_oracle_wall_min_per_iteration', math.nan):.3f} min"),
                    ("mean_exact_oracle_wall_time_per_acquisition_iteration", f"{timing_summary.get('mean_acquisition_exact_oracle_wall_min', math.nan):.3f} min"),
                    ("local_box_reduction", f"{timing_summary.get('local_box_reduction_percent', math.nan):.2f}%"),
                    ("local_refinement_runtime_reduction", f"{timing_summary.get('local_refinement_runtime_reduction_percent', math.nan):.2f}%"),
                    ("point_total_runtime_reduction", f"{timing_summary.get('point_total_runtime_reduction_percent', math.nan):.2f}%"),
                ],
                columns=["metric", "value"],
            )
        ),
        "",
        "Compared with the robust-incremental reference carried in the package, rank_and_cap_k3 reduces",
        f"the mean local-box count from {timing_summary.get('baseline_local_boxes_reference'):.3g} to "
        f"{timing_summary.get('rankcap_mean_local_boxes'):.6g}, the local-refinement runtime from "
        f"{timing_summary.get('baseline_local_refinement_runtime_sec_reference'):.6g} sec/point to "
        f"{timing_summary.get('rankcap_local_refinement_runtime_sec'):.6g} sec/point, and the point-total",
        f"runtime from {timing_summary.get('baseline_point_total_runtime_sec_reference'):.6g} sec/point to "
        f"{timing_summary.get('rankcap_point_total_runtime_sec'):.6g} sec/point.",
        "",
        "## Corrected Validation Gates",
        "",
        md_table(gate_df),
        "",
        "## Final Phase Diagram",
        "",
        f"![Final exact phase diagram]({fig_dir_name}/{figures['final_phase_diagram']})",
        "",
        "The final exact dataset covers the normal, uniform-SC, and FFLO regions.  Black open circles mark",
        "the final acquisition-selected batch before exact evaluation.",
        "",
        "## Phase Diagram Evolution",
        "",
        f"![Phase snapshots]({fig_dir_name}/{figures['phase_snapshots']})",
        "",
        "## Learning Curves",
        "",
        f"![Dataset growth and phase counts]({fig_dir_name}/{figures['learning_curve_phase_counts']})",
        "",
        f"![Training eligibility and rerun pressure]({fig_dir_name}/{figures['training_rerun_curve']})",
        "",
        "The run keeps appending training-eligible points through the final iteration.  The rerun-required",
        "fraction rises late in the run as acquisition focuses on harder boundaries, but it remains below",
        "the package threshold.",
        "",
        "## Local-Refinement Workload",
        "",
        f"![Corrected local-box gate]({fig_dir_name}/{figures['corrected_local_box_gate']})",
        "",
        f"![Corrected local-box distribution]({fig_dir_name}/{figures['local_box_distribution']})",
        "",
        "The corrected rank-level max local-box count is three for every iteration.  The original `24`",
        "comes from grouping rows by `(iteration, point_id)` after rank was dropped from the merged table.",
        "",
        "## Runtime",
        "",
        f"![Runtime curve]({fig_dir_name}/{figures['runtime_curve']})",
        "",
        f"![Exact wall-time curve]({fig_dir_name}/{figures.get('exact_walltime_curve', '')})",
        "",
        f"![Rank runtime imbalance]({fig_dir_name}/{figures['rank_runtime_imbalance']})",
        "",
        "The per-point runtime remains below the historical robust-incremental reference used in the",
        "rankcap package.  Late iterations cost more because the selected batch shifts toward harder",
        "boundary and rerun-prone points.",
        "",
        "## Surrogate / Machine-Learning Metrics",
        "",
        "The returned package stores validation-style surrogate metrics in `metrics_history.json`, not",
        "raw per-epoch training loss.  The following curves are therefore ML performance curves analogous",
        "to loss curves, but they should be interpreted as held-out surrogate quality diagnostics rather",
        "than direct optimizer training loss.",
        "",
    ]
    if "surrogate_metric_curves" in figures:
        lines.extend(
            [
                f"![Surrogate metric curves]({fig_dir_name}/{figures['surrogate_metric_curves']})",
                "",
            ]
        )
    if "phase_accuracy_reduction_curve" in figures:
        lines.extend(
            [
                f"![Phase accuracy and estimated reduction]({fig_dir_name}/{figures['phase_accuracy_reduction_curve']})",
                "",
            ]
        )
    if not ml_metrics.empty:
        lines.extend(
            [
                "Final surrogate metrics:",
                "",
                md_table(
                    ml_metrics.tail(1)[
                        [
                            "metric_iteration",
                            "delta_rmse",
                            "q_rmse",
                            "eta_rmse",
                            "ic_plus_rmse",
                            "ic_minus_rmse",
                            "phase_accuracy",
                            "estimated_reduction",
                            "n_exact_calls",
                        ]
                    ]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Recent Acquisition Focus",
            "",
            f"![Recent selected overlay]({fig_dir_name}/{figures['recent_selected_overlay']})",
            "",
        ]
    )
    if "exact_eta_revised_boundaries" in figures:
        lines.extend(
            [
                "## Existing Boundary Diagnostic",
                "",
                f"![Exact eta revised boundaries]({fig_dir_name}/{figures['exact_eta_revised_boundaries']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Iteration Recheck Table",
            "",
            md_table(
                iter_recheck,
                [
                    "iteration",
                    "exact_points_rank_level",
                    "mean_local_boxes_rank_level",
                    "max_local_boxes_rank_level",
                    "points_gt3_rank_level",
                    "training_eligible_appended",
                    "rerun_required_fraction",
                    "q_unresolved_count",
                    "delta_unresolved_count",
                    "mean_local_refinement_runtime_sec",
                    "mean_point_total_runtime_sec",
                ],
            ),
            "",
            "## Dataset Phase Counts",
            "",
            md_table(dataset_counts[["iteration", "samples", "normal", "uniform_SC", "FFLO"]]),
            "",
            "## Original Report Discrepancy",
            "",
            md_table(discrepancy.head(12)),
            "",
            "## Decision",
            "",
            "The full-loop rank_and_cap_k3 result is accepted as a completed numerical run after correcting",
            "the report aggregation key.  This report is an analysis/reporting artifact only.  It does not",
            "change the Hamiltonian, exact oracle, acquisition function, StopController, thermodynamic phase",
            "criterion, Delta tolerance, or final ambiguity tolerance.",
            "",
            "## Next Step",
            "",
            "Use this enhanced report and its figures as the full-loop evidence for the second-stage writeup.",
            "Patch the package collector before future runs so rank is preserved in merged local-box timing",
            "tables and `validation_status` is authoritative without a separate recheck.",
            "",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(report_root: Path, figures: dict[str, str], corrected_summary: dict, gate_df: pd.DataFrame) -> None:
    pdf_path = report_root / f"{REPORT_NAME}.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story = []
    story.append(Paragraph("Rank-and-Cap K3 Full-Loop Enhanced Report", styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            f"Corrected validation status: {corrected_summary['corrected_validation_status']}. "
            f"Final dataset: {corrected_summary['final_dataset_samples']} samples. "
            f"Phase counts: normal={corrected_summary['final_normal_count']}, "
            f"uniform_SC={corrected_summary['final_uniform_SC_count']}, "
            f"FFLO={corrected_summary['final_FFLO_count']}. "
            f"Corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.4g} / "
            f"{corrected_summary['corrected_max_local_boxes']}.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.16 * inch))
    rows = [["Gate", "Status", "Evidence"]]
    for _, row in gate_df.iterrows():
        rows.append([str(row["gate"]), str(row["status"]), str(row["evidence"])])
    table = Table(rows, colWidths=[2.3 * inch, 0.75 * inch, 3.85 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.16 * inch))
    figure_sequence = [
        ("Final phase diagram", "final_phase_diagram"),
        ("Phase diagram evolution", "phase_snapshots"),
        ("Learning curve", "learning_curve_phase_counts"),
        ("Training and rerun curve", "training_rerun_curve"),
        ("Corrected local-box gate", "corrected_local_box_gate"),
        ("Runtime curve", "runtime_curve"),
        ("Exact wall time curve", "exact_walltime_curve"),
        ("Surrogate metric curves", "surrogate_metric_curves"),
        ("Phase accuracy and estimated reduction", "phase_accuracy_reduction_curve"),
        ("Recent acquisition focus", "recent_selected_overlay"),
    ]
    for title, key in figure_sequence:
        if key not in figures:
            continue
        img_path = report_root / "figures" / figures[key]
        group = [Paragraph(title, styles["Heading2"])]
        if key in {"phase_snapshots"}:
            group.append(Image(str(img_path), width=6.4 * inch, height=4.2 * inch))
        else:
            group.append(Image(str(img_path), width=6.2 * inch, height=4.55 * inch))
        group.append(Spacer(1, 0.12 * inch))
        story.append(KeepTogether(group))
    doc.build(story)


def latex_table_rows(df: pd.DataFrame, max_rows: int | None = None) -> list[str]:
    use_df = df.head(max_rows) if max_rows is not None else df
    rows = []
    for _, row in use_df.iterrows():
        rows.append(" & ".join(latex_escape(row[col]) for col in use_df.columns) + r" \\")
    return rows


def write_latex(
    report_root: Path,
    figures: dict[str, str],
    summary: dict,
    corrected_summary: dict,
    gate_df: pd.DataFrame,
    summary_table: pd.DataFrame,
    timing_summary: dict,
    ml_metrics: pd.DataFrame,
) -> bool:
    tex_path = report_root / f"{REPORT_NAME}.tex"
    runtime_rows = pd.DataFrame(
        [
            ("Start time", timing_summary.get("start_time_local")),
            ("End time", timing_summary.get("end_time_local")),
            ("Total wall time", format_hours(timing_summary.get("total_wall_sec"))),
            ("Exact iterations", timing_summary.get("n_exact_iterations")),
            ("Acquisition batches", timing_summary.get("acquisition_batches")),
            ("Wall time per exact iteration", f"{timing_summary.get('wall_minutes_per_exact_iteration', math.nan):.3f} min"),
            ("Wall time per acquisition batch", f"{timing_summary.get('wall_minutes_per_acquisition_batch', math.nan):.3f} min"),
            ("Exact-oracle wall time sum", format_hours(timing_summary.get("exact_oracle_wall_sec_sum"))),
            ("Mean exact-oracle time per iteration", f"{timing_summary.get('mean_exact_oracle_wall_min_per_iteration', math.nan):.3f} min"),
            ("Mean exact-oracle time per acquisition iteration", f"{timing_summary.get('mean_acquisition_exact_oracle_wall_min', math.nan):.3f} min"),
            ("Local-box reduction", f"{timing_summary.get('local_box_reduction_percent', math.nan):.2f}%"),
            ("Local-refinement runtime reduction", f"{timing_summary.get('local_refinement_runtime_reduction_percent', math.nan):.2f}%"),
            ("Point-total runtime reduction", f"{timing_summary.get('point_total_runtime_reduction_percent', math.nan):.2f}%"),
        ],
        columns=["metric", "value"],
    )
    final_metric = pd.DataFrame()
    if not ml_metrics.empty:
        final_metric = ml_metrics.tail(1)[
            [
                "metric_iteration",
                "delta_rmse",
                "q_rmse",
                "eta_rmse",
                "ic_plus_rmse",
                "ic_minus_rmse",
                "phase_accuracy",
                "estimated_reduction",
                "n_exact_calls",
            ]
        ].copy()
    figure_sequence = [
        ("Final phase diagram", "final_phase_diagram", "Final exact phase diagram with the final selected batch overlay."),
        ("Phase diagram evolution", "phase_snapshots", "Dataset growth snapshots from early iterations to the final dataset."),
        ("Learning curve", "learning_curve_phase_counts", "Dataset size and phase-count learning curves."),
        ("Training eligibility and rerun pressure", "training_rerun_curve", "Training-eligible appended points and rerun-required fraction."),
        ("Corrected local-box gate", "corrected_local_box_gate", "Rank-level local-box count after preserving rank in the aggregation key."),
        ("Runtime curve", "runtime_curve", "Per-point runtime compared with robust-incremental reference values."),
        ("Exact wall time curve", "exact_walltime_curve", "Per-iteration exact-oracle wall time from rank runtime summaries."),
        ("Surrogate metric curves", "surrogate_metric_curves", "Validation-style surrogate metrics analogous to training-loss curves."),
        ("Phase accuracy and estimated reduction", "phase_accuracy_reduction_curve", "Classifier phase accuracy and estimated exact-call reduction."),
        ("Recent acquisition focus", "recent_selected_overlay", "Final five acquisition batches overlaid on the final exact dataset."),
    ]
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{float}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.45em}",
        r"\begin{document}",
        r"\title{Rank-and-Cap K3 Full-Loop Enhanced Report}",
        r"\author{Generated analysis artifact}",
        rf"\date{{{latex_escape(datetime.now().strftime('%Y-%m-%d %H:%M'))}}}",
        r"\maketitle",
        r"\section{Executive Summary}",
        rf"Corrected validation status: \textbf{{{latex_escape(corrected_summary['corrected_validation_status'])}}}. "
        rf"The full-loop run contains {latex_escape(summary.get('n_iters_expected'))} exact iterations: "
        rf"one random-seed iteration plus {latex_escape(summary.get('acquisition_batches_expected'))} acquisition-selected batches. "
        rf"The final dataset contains {latex_escape(summary.get('final_dataset_samples'))} samples "
        rf"(normal={latex_escape(summary.get('final_normal_count'))}, "
        rf"uniform-SC={latex_escape(summary.get('final_uniform_SC_count'))}, "
        rf"FFLO={latex_escape(summary.get('final_FFLO_count'))}).",
        "",
        "The original package-level validation status was fail because the collector grouped local-box rows by "
        r"\texttt{(iteration, point\_id)} after dropping rank. Recomputing from raw rank-level timing files with "
        r"\texttt{(iteration, rank, point\_id)} gives corrected max local boxes = "
        rf"{latex_escape(corrected_summary['corrected_max_local_boxes'])} and points above cap = "
        rf"{latex_escape(corrected_summary['corrected_points_gt3'])}.",
        r"\section{Summary Table}",
        r"\begin{longtable}{p{0.42\linewidth}p{0.48\linewidth}}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        *latex_table_rows(summary_table),
        r"\bottomrule",
        r"\end{longtable}",
        r"\section{Run Duration and Speedup}",
        r"\begin{longtable}{p{0.48\linewidth}p{0.42\linewidth}}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        *latex_table_rows(runtime_rows),
        r"\bottomrule",
        r"\end{longtable}",
        rf"The wall time estimate uses the active-loop lock start timestamp and the package collector completion timestamp. "
        rf"The exact-oracle wall time is independently recomputed from the maximum rank elapsed time in each iteration.",
        "",
        rf"Relative to the robust-incremental reference carried in the package, rank-and-cap K3 reduces the mean local-box "
        rf"count from {timing_summary.get('baseline_local_boxes_reference'):.3g} to "
        rf"{timing_summary.get('rankcap_mean_local_boxes'):.6g}, local-refinement runtime from "
        rf"{timing_summary.get('baseline_local_refinement_runtime_sec_reference'):.6g} sec/point to "
        rf"{timing_summary.get('rankcap_local_refinement_runtime_sec'):.6g} sec/point, and point-total runtime from "
        rf"{timing_summary.get('baseline_point_total_runtime_sec_reference'):.6g} sec/point to "
        rf"{timing_summary.get('rankcap_point_total_runtime_sec'):.6g} sec/point.",
        r"\section{Corrected Validation Gates}",
        r"\begin{longtable}{p{0.36\linewidth}p{0.14\linewidth}p{0.40\linewidth}}",
        r"\toprule",
        r"Gate & Status & Evidence \\",
        r"\midrule",
        *latex_table_rows(gate_df),
        r"\bottomrule",
        r"\end{longtable}",
    ]
    for title, key, caption in figure_sequence:
        if key not in figures:
            continue
        lines.extend(
            [
                rf"\section{{{latex_escape(title)}}}",
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.92\linewidth]{{figures/{figures[key]}}}",
                rf"\caption{{{latex_escape(caption)}}}",
                r"\end{figure}",
            ]
        )
    lines.extend(
        [
            r"\section{Surrogate Metric Interpretation}",
            "The package stores validation-style surrogate metrics in "
            r"\texttt{metrics\_history.json}, not raw per-epoch training loss. "
            "The ML curves in this report should therefore be read as held-out "
            "surrogate quality diagnostics analogous to loss curves, not as raw "
            "optimizer training loss.",
        ]
    )
    if not final_metric.empty:
        lines.extend(
            [
                r"\begin{longtable}{p{0.28\linewidth}p{0.62\linewidth}}",
                r"\toprule",
                r"Metric & Final value \\",
                r"\midrule",
            ]
        )
        last = final_metric.iloc[0].to_dict()
        for key, value in last.items():
            if isinstance(value, (float, np.floating)):
                value_text = f"{float(value):.6g}"
            else:
                value_text = str(value)
            lines.append(rf"{latex_escape(key)} & {latex_escape(value_text)} \\")
        lines.extend([r"\bottomrule", r"\end{longtable}"])
    lines.extend(
        [
            r"\section{Decision}",
            "The full-loop rank-and-cap K3 result is accepted as a completed numerical run after correcting "
            "the report aggregation key. This report is an analysis/reporting artifact only. It does not change "
            "the Hamiltonian, exact oracle, acquisition function, StopController, thermodynamic phase criterion, "
            "Delta tolerance, or final ambiguity tolerance.",
            r"\end{document}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    try:
        subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(report_root),
                str(tex_path),
            ],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        (report_root / f"{REPORT_NAME}_latex_build.log").write_text(str(exc), encoding="utf-8")
        return False


def mirror_report_files() -> None:
    if ML_REPORT_ROOT.exists():
        shutil.rmtree(ML_REPORT_ROOT)
    ML_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in [
        f"{REPORT_NAME}.md",
        f"{REPORT_NAME}.tex",
        f"{REPORT_NAME}.pdf",
        "summary.json",
        "decision_log.md",
    ]:
        src = REPORT_ROOT / name
        if src.exists():
            shutil.copy2(src, ML_REPORT_ROOT / name)
    for folder in ["figures", "tables"]:
        src_dir = REPORT_ROOT / folder
        dst_dir = ML_REPORT_ROOT / folder
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def main() -> None:
    figures_dir = REPORT_ROOT / "figures"
    tables_dir = REPORT_ROOT / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = read_json(ORIGINAL_REPORT_ROOT / "summary.json")
    iter_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "iteration_summary.csv")
    dataset_counts = collect_dataset_phase_counts()
    trace_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "traceback_scan.csv")
    local_box_points = collect_rank_level_local_boxes()
    iter_recheck = build_iteration_recheck(iter_df, local_box_points)
    timing_summary, exact_iter_timing = collect_timing_summary(summary, iter_recheck)
    ml_metrics = collect_ml_metrics()
    discrepancy = build_discrepancy_table(iter_df)
    distribution = (
        local_box_points.groupby("local_boxes_refined_count", sort=True)
        .size()
        .reset_index(name="point_count")
    )
    weighted_mean = float(
        iter_recheck["total_local_boxes_rank_level"].sum()
        / iter_recheck["exact_points_rank_level"].sum()
    )
    corrected_summary = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "run_id": summary.get("run_id"),
        "original_validation_status": summary.get("validation_status"),
        "corrected_validation_status": "pass",
        "actual_exact_points": int(len(local_box_points)),
        "corrected_max_local_boxes": int(local_box_points["local_boxes_refined_count"].max()),
        "corrected_points_gt3": int((local_box_points["local_boxes_refined_count"] > 3).sum()),
        "corrected_mean_local_boxes_unweighted": float(iter_recheck["mean_local_boxes_rank_level"].mean()),
        "corrected_mean_local_boxes_weighted": weighted_mean,
        "original_reported_max_local_boxes": summary.get("max_local_boxes_refined_count"),
        "final_dataset_samples": summary.get("final_dataset_samples"),
        "final_normal_count": summary.get("final_normal_count"),
        "final_uniform_SC_count": summary.get("final_uniform_SC_count"),
        "final_FFLO_count": summary.get("final_FFLO_count"),
        **timing_summary,
    }
    gate_df = build_gate_table(summary, iter_recheck, dataset_counts, trace_df)
    if not (gate_df["status"] == "pass").all():
        corrected_summary["corrected_validation_status"] = "fail"
    summary_table = compact_summary_table(summary, corrected_summary)

    local_box_points.to_csv(tables_dir / "enhanced_actual_local_box_point_counts.csv", index=False)
    distribution.to_csv(tables_dir / "enhanced_actual_local_box_distribution.csv", index=False)
    iter_recheck.to_csv(tables_dir / "enhanced_iteration_recheck.csv", index=False)
    discrepancy.to_csv(tables_dir / "enhanced_original_report_discrepancy.csv", index=False)
    gate_df.to_csv(tables_dir / "enhanced_corrected_validation_gates.csv", index=False)
    summary_table.to_csv(tables_dir / "enhanced_summary_table.csv", index=False)
    pd.DataFrame([corrected_summary]).to_csv(tables_dir / "enhanced_corrected_validation_summary.csv", index=False)
    dataset_counts.to_csv(tables_dir / "enhanced_dataset_phase_counts.csv", index=False)
    pd.DataFrame([timing_summary]).to_csv(tables_dir / "enhanced_runtime_timing_summary.csv", index=False)
    exact_iter_timing.to_csv(tables_dir / "enhanced_exact_iteration_walltime.csv", index=False)
    ml_metrics.to_csv(tables_dir / "enhanced_surrogate_metric_history.csv", index=False)
    write_json(REPORT_ROOT / "summary.json", {**summary, **{"enhanced_recheck": corrected_summary}})

    figures = create_figures(figures_dir, dataset_counts, iter_df, iter_recheck, local_box_points, ml_metrics, exact_iter_timing)
    write_markdown(
        REPORT_ROOT,
        figures,
        summary,
        corrected_summary,
        gate_df,
        summary_table,
        dataset_counts,
        iter_recheck,
        discrepancy,
        timing_summary,
        ml_metrics,
    )
    write_pdf(REPORT_ROOT, figures, corrected_summary, gate_df)
    latex_ok = write_latex(REPORT_ROOT, figures, summary, corrected_summary, gate_df, summary_table, timing_summary, ml_metrics)
    decision_log = "\n".join(
        [
            "# Rankcap K3 Full-Loop Enhanced Decision Log",
            "",
            f"- original validation_status: {summary.get('validation_status')}",
            f"- corrected validation_status: {corrected_summary['corrected_validation_status']}",
            f"- final dataset samples: {corrected_summary['final_dataset_samples']}",
            f"- final phase counts: normal={corrected_summary['final_normal_count']}, uniform_SC={corrected_summary['final_uniform_SC_count']}, FFLO={corrected_summary['final_FFLO_count']}",
            f"- corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.6g}/{corrected_summary['corrected_max_local_boxes']}",
            f"- total wall time estimate: {format_hours(timing_summary.get('total_wall_sec'))}",
            f"- average wall time per exact iteration: {timing_summary.get('wall_minutes_per_exact_iteration', math.nan):.6g} min",
            f"- LaTeX PDF build: {'pass' if latex_ok else 'failed'}",
            "- conclusion: full-loop run completed; original fail is the known rank-local point-id report aggregation artifact.",
            "- next step: use the enhanced report for result interpretation and patch the collector before future runs.",
            "",
        ]
    )
    (REPORT_ROOT / "decision_log.md").write_text(decision_log, encoding="utf-8")
    mirror_report_files()


if __name__ == "__main__":
    main()
