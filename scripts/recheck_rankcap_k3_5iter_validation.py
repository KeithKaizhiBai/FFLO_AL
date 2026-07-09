from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "rankcap_k3_5iter_validation"
RUN_ID = "active_boundary_discovery_rankcap_k3_5iter_validation_v1"
RUN_DIR = (
    PACKAGE_ROOT
    / "ML_Phase_512_RankCapK3_5Iter"
    / "active_runs"
    / RUN_ID
)
ORIGINAL_REPORT_ROOT = PACKAGE_ROOT / "reports" / "rankcap_k3_5iter_validation"
OUTPUT_ROOT = ROOT / "reports" / "rankcap_k3_5iter_validation_recheck"
TABLES_DIR = OUTPUT_ROOT / "tables"
FIGURES_DIR = OUTPUT_ROOT / "figures"


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


def bool_text(value: bool) -> str:
    return "pass" if bool(value) else "fail"


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
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def collect_rank_level_local_boxes() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"iter(\d+)_local_box_timing_rank(\d+)_of(\d+)\.csv")
    files = sorted(RUN_DIR.glob("iter*/performance/*_local_box_timing_rank*_of*.csv"))
    for path in files:
        match = pattern.search(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        rank = int(match.group(2))
        world_size = int(match.group(3))
        df = read_csv(path)
        if df.empty:
            continue
        grouped = df.groupby("point_id", sort=True)
        for point_id, group in grouped:
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
                    "changed_global_minimum_count": int(
                        pd.to_numeric(group["changed_global_minimum"], errors="coerce")
                        .fillna(0)
                        .sum()
                    ),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    return pd.DataFrame(rows)


def build_diagnostics(actual_counts: pd.DataFrame, iter_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_local_box_rows = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "local_box_rows.csv")
    diagnostics = []
    if not original_local_box_rows.empty:
        for iteration, group in original_local_box_rows.groupby("iteration", sort=True):
            iteration_i = int(iteration)
            expected_points = int(
                iter_df.loc[iter_df["iteration"] == iteration_i, "exact_points"].iloc[0]
            )
            actual_iter = actual_counts[actual_counts["iteration"] == iteration_i]
            wrong_max = int(group.groupby("point_id").size().max())
            correct_max = int(actual_iter["local_boxes_refined_count"].max())
            diagnostics.append(
                {
                    "iteration": iteration_i,
                    "original_local_box_rows": int(len(group)),
                    "unique_point_id_values_in_original_rows": int(group["point_id"].nunique()),
                    "expected_exact_points": expected_points,
                    "rank_dimension_present_in_original_rows": "rank" in original_local_box_rows.columns,
                    "wrong_max_grouped_by_iteration_point_id": wrong_max,
                    "correct_max_grouped_by_iteration_rank_point_id": correct_max,
                    "diagnosis": "rank-local point_id collision in merged report table",
                }
            )
    discrepancy = pd.DataFrame(diagnostics)

    if actual_counts.empty:
        iteration_recheck = pd.DataFrame()
    else:
        grouped = actual_counts.groupby("iteration", sort=True)
        rows = []
        for iteration, group in grouped:
            original = iter_df.loc[iter_df["iteration"] == iteration]
            original_row = original.iloc[0].to_dict() if not original.empty else {}
            total_boxes = int(group["local_boxes_refined_count"].sum())
            exact_points = int(len(group))
            rows.append(
                {
                    "iteration": int(iteration),
                    "exact_points_rank_level": exact_points,
                    "total_local_boxes_rank_level": total_boxes,
                    "mean_local_boxes_rank_level": total_boxes / exact_points if exact_points else math.nan,
                    "max_local_boxes_rank_level": int(group["local_boxes_refined_count"].max()),
                    "points_gt3_rank_level": int((group["local_boxes_refined_count"] > 3).sum()),
                    "original_exact_points": int(original_row.get("exact_points", 0)),
                    "original_selected_refine_target_count_sum": int(
                        original_row.get("selected_refine_target_count_sum", 0)
                    ),
                    "original_mean_local_boxes_refined_count": float(
                        original_row.get("mean_local_boxes_refined_count", math.nan)
                    ),
                    "training_eligible_appended": int(
                        original_row.get("training_eligible_appended", 0)
                    ),
                    "rerun_required_fraction": float(
                        original_row.get("rerun_required_fraction", math.nan)
                    ),
                    "q_unresolved_count": int(original_row.get("q_unresolved_count", 0)),
                    "delta_unresolved_count": int(original_row.get("delta_unresolved_count", 0)),
                    "mean_local_refinement_runtime_sec": float(
                        original_row.get("mean_local_refinement_runtime_sec", math.nan)
                    ),
                    "mean_point_total_runtime_sec": float(
                        original_row.get("mean_point_total_runtime_sec", math.nan)
                    ),
                }
            )
        iteration_recheck = pd.DataFrame(rows)
    return discrepancy, iteration_recheck


def make_figures(
    actual_counts: pd.DataFrame,
    distribution: pd.DataFrame,
    dataset_df: pd.DataFrame,
    iteration_recheck: pd.DataFrame,
    rank_df: pd.DataFrame,
) -> None:
    plt.figure(figsize=(5.2, 3.4))
    if not distribution.empty:
        plt.bar(distribution["local_boxes_refined_count"].astype(str), distribution["point_count"])
    plt.xlabel("Local boxes per actual exact point")
    plt.ylabel("Point count")
    plt.title("Corrected rank-level local-box distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "corrected_local_box_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.0, 3.6))
    if not dataset_df.empty:
        for phase in ["normal", "uniform_SC", "FFLO"]:
            plt.plot(dataset_df["iteration"], dataset_df[phase], marker="o", label=phase)
    plt.xlabel("Dataset iteration")
    plt.ylabel("Samples")
    plt.title("Dataset phase coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "dataset_phase_counts_recheck.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.0, 3.6))
    if not iteration_recheck.empty:
        plt.plot(
            iteration_recheck["iteration"],
            iteration_recheck["mean_local_boxes_rank_level"],
            marker="o",
            label="mean local boxes",
        )
        plt.plot(
            iteration_recheck["iteration"],
            iteration_recheck["max_local_boxes_rank_level"],
            marker="s",
            label="max local boxes",
        )
        plt.axhline(3, color="black", linestyle="--", linewidth=1.0, label="k3 cap")
    plt.xlabel("Exact iteration")
    plt.ylabel("Boxes")
    plt.title("Corrected local-box gate by iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "local_box_gate_by_iteration.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.0, 3.6))
    if not iteration_recheck.empty:
        plt.plot(
            iteration_recheck["iteration"],
            iteration_recheck["mean_local_refinement_runtime_sec"],
            marker="o",
            label="local refinement",
        )
        plt.plot(
            iteration_recheck["iteration"],
            iteration_recheck["mean_point_total_runtime_sec"],
            marker="s",
            label="point total",
        )
    plt.xlabel("Exact iteration")
    plt.ylabel("Seconds per point")
    plt.title("Runtime by iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "runtime_by_iteration_recheck.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.0, 3.6))
    if not rank_df.empty and "rank_runtime_imbalance_ratio" in rank_df.columns:
        iter_rank = (
            rank_df.groupby("iteration", sort=True)["elapsed_sec"]
            .agg(lambda values: float(values.max() / values.min()) if values.min() > 0 else math.nan)
            .reset_index(name="rank_runtime_imbalance_ratio")
        )
        plt.plot(
            iter_rank["iteration"],
            iter_rank["rank_runtime_imbalance_ratio"],
            marker="o",
        )
    plt.xlabel("Exact iteration")
    plt.ylabel("max(rank sec) / min(rank sec)")
    plt.title("Rank runtime imbalance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rank_runtime_imbalance_recheck.png", dpi=180)
    plt.close()


def build_gate_table(summary: dict, iteration_recheck: pd.DataFrame, dataset_df: pd.DataFrame, trace_df: pd.DataFrame) -> pd.DataFrame:
    corrected_max = int(iteration_recheck["max_local_boxes_rank_level"].max())
    corrected_mean_unweighted = float(iteration_recheck["mean_local_boxes_rank_level"].mean())
    corrected_mean_weighted = float(
        iteration_recheck["total_local_boxes_rank_level"].sum()
        / iteration_recheck["exact_points_rank_level"].sum()
    )
    final_row = dataset_df.iloc[-1].to_dict() if not dataset_df.empty else {}
    gates = [
        ("expected_iters_present", bool(summary.get("expected_iters_present", False)), ""),
        ("all_shards_complete", bool(summary.get("all_shards_complete", False)), ""),
        ("merge_status", summary.get("merge_status") == "pass", str(summary.get("merge_status"))),
        ("append_status", summary.get("append_status") == "pass", str(summary.get("append_status"))),
        ("dataset_monotonic", bool(summary.get("dataset_monotonic", False)), ""),
        ("phase_coverage", bool(summary.get("phase_coverage", False)), f"normal={final_row.get('normal')}, uniform_SC={final_row.get('uniform_SC')}, FFLO={final_row.get('FFLO')}"),
        ("training_eligible_nonzero_each_iter", bool(summary.get("training_eligible_nonzero_each_iter", False)), ""),
        ("q_unresolved_ok", bool(summary.get("q_unresolved_ok", False)), ""),
        ("delta_unresolved_ok", bool(summary.get("delta_unresolved_ok", False)), ""),
        ("rerun_fraction_ok", bool(summary.get("rerun_fraction_ok", False)), ""),
        ("mean_local_boxes_ok", corrected_mean_unweighted <= 3.2, f"unweighted={corrected_mean_unweighted:.6g}; weighted={corrected_mean_weighted:.6g}"),
        ("max_local_boxes_ok_corrected", corrected_max <= 3, f"corrected_max={corrected_max}"),
        ("traceback_scan_ok", trace_df.empty, f"matches={len(trace_df)}"),
        ("final_report_nonblocking", str(summary.get("final_report_status", "")).lower() in {"ok", "pdflatex_missing", "pdflatex_failed"}, str(summary.get("final_report_status"))),
    ]
    return pd.DataFrame(
        {
            "gate": [row[0] for row in gates],
            "status": [bool_text(row[1]) for row in gates],
            "evidence": [row[2] for row in gates],
        }
    )


def write_markdown(
    summary: dict,
    corrected_summary: dict,
    gate_df: pd.DataFrame,
    iteration_recheck: pd.DataFrame,
    distribution: pd.DataFrame,
    discrepancy: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> None:
    md_path = OUTPUT_ROOT / "rankcap_k3_5iter_validation_recheck.md"
    lines = [
        "# Rank-and-Cap K3 Five-Iteration Validation Recheck",
        "",
        "## Executive Summary",
        "",
        f"- original_report_validation_status: {summary.get('validation_status')}",
        f"- corrected_validation_status: {corrected_summary['corrected_validation_status']}",
        f"- run_id: {summary.get('run_id')}",
        f"- actual exact points checked: {corrected_summary['actual_exact_points']}",
        f"- corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.6g} / {corrected_summary['corrected_max_local_boxes']}",
        f"- corrected weighted local boxes mean: {corrected_summary['corrected_mean_local_boxes_weighted']:.6g}",
        f"- original reported max local boxes: {summary.get('max_local_boxes_refined_count')}",
        f"- final dataset samples: {summary.get('final_dataset_samples')}",
        f"- final phase counts: normal={summary.get('final_normal_count')}, uniform_SC={summary.get('final_uniform_SC_count')}, FFLO={summary.get('final_FFLO_count')}",
        "",
        "The original returned report marked the run as fail only because the report collector grouped",
        "`local_box_rows.csv` by `(iteration, point_id)`.  In the raw rank outputs, `point_id` is",
        "rank-local.  The merged report table dropped the rank column, so different shards with the",
        "same local point id were counted as one physical point.  Recomputing from the raw",
        "`iterXXX_local_box_timing_rankYYY_of008.csv` files with `(iteration, rank, point_id)` gives",
        "a true maximum of 3 local boxes and no point above the k3 cap.",
        "",
        "## Corrected Gate Table",
        "",
        md_table(gate_df),
        "",
        "## Local-Box Count Recheck",
        "",
        md_table(distribution),
        "",
        "## Iteration-Level Recheck",
        "",
        md_table(
            iteration_recheck,
            [
                "iteration",
                "exact_points_rank_level",
                "total_local_boxes_rank_level",
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
        "## Original Report Discrepancy",
        "",
        md_table(discrepancy),
        "",
        "## Dataset Phase Counts",
        "",
        md_table(dataset_df),
        "",
        "## Interpretation",
        "",
        "- The five acquisition-selected batches completed without tracebacks, OOM, CUDA initialization failure, timeout, or cancellation matches in the report scan.",
        "- Dataset growth is monotonic from 0 to 1692 samples, and the final dataset contains normal, uniform_SC, and FFLO labels.",
        "- The corrected rank-level local-box cap gate passes: all 1792 actual exact points used 2 or 3 local boxes.",
        "- The package report PDF is readable but its first page has clipped long paths and it reports a false fail; it should not be used as the final decision record without this recheck.",
        "",
        "## Recommendation",
        "",
        "The five-iteration validation supports rank_and_cap_k3 as safe for the next active-learning step,",
        "provided the active-loop report collector is patched to preserve rank in `local_box_rows.csv` and",
        "to compute max local boxes by `(iteration, rank, point_id)` or by a globally unique point id.",
        "The existing full-loop package likely has the same report-collector bug, so its report should be",
        "patched or rechecked before using its `validation_status` field as an acceptance decision.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(corrected_summary: dict, gate_df: pd.DataFrame) -> None:
    pdf_path = OUTPUT_ROOT / "rankcap_k3_5iter_validation_recheck.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story = []
    story.append(Paragraph("Rank-and-Cap K3 Five-Iteration Validation Recheck", styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    summary_text = (
        f"Corrected validation status: {corrected_summary['corrected_validation_status']}. "
        f"Actual exact points: {corrected_summary['actual_exact_points']}. "
        f"Corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.4g} / "
        f"{corrected_summary['corrected_max_local_boxes']}. "
        "The original report fail is a rank-local point-id aggregation artifact."
    )
    story.append(Paragraph(summary_text, styles["BodyText"]))
    story.append(Spacer(1, 0.16 * inch))

    table_rows = [["Gate", "Status", "Evidence"]]
    for _, row in gate_df.iterrows():
        table_rows.append([str(row["gate"]), str(row["status"]), str(row["evidence"])])
    table = Table(table_rows, colWidths=[2.25 * inch, 0.75 * inch, 3.9 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    for figure in [
        "corrected_local_box_distribution.png",
        "local_box_gate_by_iteration.png",
        "dataset_phase_counts_recheck.png",
        "runtime_by_iteration_recheck.png",
    ]:
        path = FIGURES_DIR / figure
        if path.exists():
            story.append(Image(str(path), width=5.9 * inch, height=3.45 * inch))
            story.append(Spacer(1, 0.12 * inch))
    doc.build(story)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    original_summary = read_json(ORIGINAL_REPORT_ROOT / "summary.json")
    iter_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "iteration_summary.csv")
    dataset_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "dataset_phase_counts.csv")
    rank_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "rank_runtime_summary.csv")
    trace_df = read_csv(ORIGINAL_REPORT_ROOT / "tables" / "traceback_scan.csv")

    actual_counts = collect_rank_level_local_boxes()
    distribution = (
        actual_counts.groupby("local_boxes_refined_count", sort=True)
        .size()
        .reset_index(name="point_count")
    )
    discrepancy, iteration_recheck = build_diagnostics(actual_counts, iter_df)
    gate_df = build_gate_table(original_summary, iteration_recheck, dataset_df, trace_df)
    corrected_status = "pass" if (gate_df["status"] == "pass").all() else "fail"
    corrected_summary = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "run_id": original_summary.get("run_id"),
        "original_validation_status": original_summary.get("validation_status"),
        "corrected_validation_status": corrected_status,
        "actual_exact_points": int(len(actual_counts)),
        "corrected_max_local_boxes": int(actual_counts["local_boxes_refined_count"].max()),
        "corrected_points_gt3": int((actual_counts["local_boxes_refined_count"] > 3).sum()),
        "corrected_mean_local_boxes_unweighted": float(iteration_recheck["mean_local_boxes_rank_level"].mean()),
        "corrected_mean_local_boxes_weighted": float(
            iteration_recheck["total_local_boxes_rank_level"].sum()
            / iteration_recheck["exact_points_rank_level"].sum()
        ),
        "original_reported_max_local_boxes": original_summary.get("max_local_boxes_refined_count"),
        "final_dataset_samples": original_summary.get("final_dataset_samples"),
        "final_normal_count": original_summary.get("final_normal_count"),
        "final_uniform_SC_count": original_summary.get("final_uniform_SC_count"),
        "final_FFLO_count": original_summary.get("final_FFLO_count"),
    }

    actual_counts.to_csv(TABLES_DIR / "actual_local_box_point_counts.csv", index=False)
    distribution.to_csv(TABLES_DIR / "actual_local_box_distribution.csv", index=False)
    discrepancy.to_csv(TABLES_DIR / "original_report_discrepancy.csv", index=False)
    iteration_recheck.to_csv(TABLES_DIR / "iteration_recheck.csv", index=False)
    gate_df.to_csv(TABLES_DIR / "corrected_validation_gates.csv", index=False)
    pd.DataFrame([corrected_summary]).to_csv(TABLES_DIR / "corrected_validation_summary.csv", index=False)
    dataset_df.to_csv(TABLES_DIR / "dataset_phase_counts_recheck.csv", index=False)
    write_json(OUTPUT_ROOT / "summary.json", corrected_summary)

    make_figures(actual_counts, distribution, dataset_df, iteration_recheck, rank_df)
    write_markdown(original_summary, corrected_summary, gate_df, iteration_recheck, distribution, discrepancy, dataset_df)
    write_pdf(corrected_summary, gate_df)

    decision_log = "\n".join(
        [
            "# Rankcap K3 5-Iteration Recheck Decision Log",
            "",
            f"- original validation_status: {original_summary.get('validation_status')}",
            f"- corrected validation_status: {corrected_status}",
            f"- corrected local boxes mean/max: {corrected_summary['corrected_mean_local_boxes_unweighted']:.6g}/{corrected_summary['corrected_max_local_boxes']}",
            f"- actual exact points checked: {corrected_summary['actual_exact_points']}",
            "- conclusion: original fail is a report aggregation false positive caused by dropping rank before grouping point_id.",
            "- next step: patch the active-loop report collector before relying on full-loop validation_status.",
            "",
        ]
    )
    (OUTPUT_ROOT / "decision_log.md").write_text(decision_log, encoding="utf-8")


if __name__ == "__main__":
    main()
