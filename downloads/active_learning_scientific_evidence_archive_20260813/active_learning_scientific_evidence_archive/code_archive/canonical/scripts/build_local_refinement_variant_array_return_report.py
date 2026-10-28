from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


VARIANTS = [
    "baseline",
    "cluster_only",
    "cluster_optional_k3",
    "cluster_optional_k2",
    "cluster_energy_window",
]
DISPLAY_VARIANTS = {
    "baseline": "baseline",
    "cluster_only": "cluster only",
    "cluster_optional_k3": "optional k3",
    "cluster_optional_k2": "optional k2",
    "cluster_energy_window": "energy window",
}
REPORT_NAME = "local_refinement_variant_array_return_report"
DEFAULT_RETURN_ROOT = Path(
    "local_refinement_refactor_hpc_upload_set/local_refinement_refactor_variant_suite/local_refinement_refactor_variant_suite_run"
)
DEFAULT_OUTPUT_DIR = Path("project_history/reports/report_local_refinement_variant_array_return_20260608")
TWO_HOUR_SEC = 2.0 * 3600.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8", newline="\n")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_value(raw: Any, default: float = float("nan")) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def int_value(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def copy_raw_tables(report_root: Path, regression_root: Path) -> None:
    table_dir = report_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    raw_tables = {
        "raw_task_status.csv": regression_root / "summary" / "task_status.csv",
        "raw_missing_or_failed_tasks.csv": regression_root / "summary" / "missing_or_failed_tasks.csv",
        "raw_equivalence_matrix.csv": regression_root / "summary" / "equivalence_matrix.csv",
        "raw_runtime_summary.csv": regression_root / "performance_report" / "runtime_summary.csv",
        "raw_local_box_summary.csv": regression_root / "performance_report" / "local_box_summary.csv",
    }
    for name, src in raw_tables.items():
        if src.exists():
            shutil.copy2(src, table_dir / name)


def classify_effective_status(row: dict[str, str], logs_root: Path) -> str:
    if row.get("status") == "success":
        return "success"
    task_id = row.get("task_id", "")
    for suffix in (".err", ".out"):
        for path in logs_root.glob(f"variant_point_*_{task_id}{suffix}"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "DUE TO TIME LIMIT" in text or "TIMEOUT" in text or "CANCELLED" in text:
                return "timeout"
    return row.get("status", "unknown") or "unknown"


def build_task_tables(report_root: Path, return_root: Path, regression_root: Path) -> dict[str, Any]:
    task_rows = read_csv_rows(regression_root / "summary" / "task_status.csv")
    logs_root = return_root / "logs"
    for row in task_rows:
        row["effective_status"] = classify_effective_status(row, logs_root)
        row["wall_runtime_sec_float"] = float_value(row.get("wall_runtime_sec"))
        runtime = float_value(row.get("wall_runtime_sec"))
        row["completed_under_2h"] = int(row["effective_status"] == "success" and runtime <= TWO_HOUR_SEC)
        row["completed_after_2h"] = int(row["effective_status"] == "success" and runtime > TWO_HOUR_SEC)

    variant_summary: list[dict[str, Any]] = []
    risk_summary: list[dict[str, Any]] = []
    slow_tasks: list[dict[str, Any]] = []
    for variant in VARIANTS:
        rows = [row for row in task_rows if row.get("variant") == variant]
        success_rows = [row for row in rows if row["effective_status"] == "success"]
        timeout_rows = [row for row in rows if row["effective_status"] == "timeout"]
        other_rows = [row for row in rows if row["effective_status"] not in {"success", "timeout"}]
        runtimes = [float_value(row.get("wall_runtime_sec")) for row in success_rows]
        runtimes = [value for value in runtimes if math.isfinite(value)]
        variant_summary.append(
            {
                "variant": variant,
                "expected_tasks": len(rows),
                "success_count": len(success_rows),
                "timeout_count": len(timeout_rows),
                "other_incomplete_count": len(other_rows),
                "success_rate": len(success_rows) / len(rows) if rows else float("nan"),
                "completed_under_2h": sum(int(row["completed_under_2h"]) for row in rows),
                "completed_after_2h": sum(int(row["completed_after_2h"]) for row in rows),
                "mean_success_runtime_sec": sum(runtimes) / len(runtimes) if runtimes else float("nan"),
                "max_success_runtime_sec": max(runtimes) if runtimes else float("nan"),
            }
        )
        categories = sorted({row.get("risk_tag", "unknown") for row in rows})
        for category in categories:
            subset = [row for row in rows if row.get("risk_tag") == category]
            risk_summary.append(
                {
                    "variant": variant,
                    "risk_tag": category,
                    "expected_tasks": len(subset),
                    "success_count": sum(1 for row in subset if row["effective_status"] == "success"),
                    "timeout_count": sum(1 for row in subset if row["effective_status"] == "timeout"),
                    "other_incomplete_count": sum(
                        1 for row in subset if row["effective_status"] not in {"success", "timeout"}
                    ),
                }
            )
        for row in rows:
            if row["effective_status"] != "success":
                slow_tasks.append(
                    {
                        "task_id": row.get("task_id"),
                        "variant": variant,
                        "point_id": row.get("point_id"),
                        "kT": row.get("kT"),
                        "JA": row.get("JA"),
                        "risk_tag": row.get("risk_tag"),
                        "effective_status": row.get("effective_status"),
                    }
                )

    write_csv(report_root / "tables" / "variant_task_summary.csv", variant_summary)
    write_csv(report_root / "tables" / "risk_category_summary.csv", risk_summary)
    write_csv(report_root / "tables" / "slow_pathology_points.csv", slow_tasks)
    return {
        "task_rows": task_rows,
        "variant_summary": variant_summary,
        "risk_summary": risk_summary,
        "slow_tasks": slow_tasks,
    }


def load_pointwise(regression_root: Path) -> dict[str, dict[int, dict[str, str]]]:
    out: dict[str, dict[int, dict[str, str]]] = {}
    for variant in VARIANTS:
        rows = read_csv_rows(regression_root / variant / f"{variant}_pointwise.csv")
        out[variant] = {int_value(row.get("point_id")): row for row in rows}
    return out


def build_equivalence_and_runtime_tables(report_root: Path, regression_root: Path) -> dict[str, Any]:
    equivalence_rows = read_csv_rows(regression_root / "summary" / "equivalence_matrix.csv")
    write_csv(report_root / "tables" / "equivalence_summary.csv", equivalence_rows)

    pointwise = load_pointwise(regression_root)
    common_ids = sorted(set.intersection(*(set(pointwise[variant]) for variant in VARIANTS)))
    runtime_rows: list[dict[str, Any]] = []
    for point_id in common_ids:
        baseline_runtime = float_value(pointwise["baseline"][point_id].get("point_total_runtime_sec"))
        baseline_boxes = float_value(pointwise["baseline"][point_id].get("local_boxes_refined_count"))
        for variant in VARIANTS:
            row = pointwise[variant][point_id]
            runtime = float_value(row.get("point_total_runtime_sec"))
            boxes = float_value(row.get("local_boxes_refined_count"))
            runtime_rows.append(
                {
                    "variant": variant,
                    "point_id": point_id,
                    "source_category": row.get("source_category", ""),
                    "kT": row.get("kT", ""),
                    "JA": row.get("JA", ""),
                    "phase_candidate": row.get("phase_candidate", ""),
                    "q_opt": row.get("q_opt", ""),
                    "delta_opt": row.get("delta_opt", ""),
                    "DeltaF": row.get("DeltaF", ""),
                    "point_total_runtime_sec": runtime,
                    "local_refinement_runtime_sec": float_value(row.get("local_refinement_runtime_sec")),
                    "local_boxes_refined_count": boxes,
                    "local_minima_detected_count": float_value(row.get("local_minima_detected_count")),
                    "runtime_ratio_vs_baseline_same_point": runtime / baseline_runtime
                    if baseline_runtime > 0
                    else float("nan"),
                    "local_boxes_ratio_vs_baseline_same_point": boxes / baseline_boxes
                    if baseline_boxes > 0
                    else float("nan"),
                }
            )
    write_csv(report_root / "tables" / "common_point_runtime.csv", runtime_rows)
    return {"equivalence_rows": equivalence_rows, "common_runtime_rows": runtime_rows, "common_ids": common_ids}


def write_package_status(report_root: Path, return_root: Path, regression_root: Path) -> dict[str, Any]:
    archive = return_root / "local_refinement_refactor_variant_suite_results.tar.gz"
    array_status_path = regression_root / "summary" / "array_suite_status.json"
    array_status = read_json(array_status_path)
    row = {
        "return_root": str(return_root),
        "return_archive": str(archive),
        "return_archive_exists": int(archive.exists()),
        "return_archive_size_bytes": archive.stat().st_size if archive.exists() else 0,
        "array_status": array_status.get("status"),
        "expected_tasks": array_status.get("expected_tasks"),
        "successful_tasks": array_status.get("successful_tasks"),
        "failed_or_missing_tasks": array_status.get("failed_or_missing_tasks"),
        "task_status": array_status.get("task_status"),
        "comparison_status": array_status.get("comparison_status"),
        "performance_status": array_status.get("performance_status"),
    }
    write_csv(report_root / "tables" / "package_status.csv", [row])
    return {"array_status": array_status, "package_row": row}


def plot_task_outcome(report_root: Path, task_rows: list[dict[str, str]]) -> Path:
    status_map = {"success": 0, "timeout": 1}
    matrix = np.full((len(VARIANTS), 32), 2)
    for i, variant in enumerate(VARIANTS):
        for row in task_rows:
            if row.get("variant") != variant:
                continue
            point_id = int_value(row.get("point_id"))
            matrix[i, point_id] = status_map.get(row.get("effective_status"), 2)
    fig, ax = plt.subplots(figsize=(10, 2.9), dpi=180)
    cmap = ListedColormap(["#2f8f4e", "#d45b4f", "#9aa0a6"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(len(VARIANTS)), [DISPLAY_VARIANTS[v] for v in VARIANTS])
    ax.set_xticks(range(0, 32, 2))
    ax.set_xlabel("fixed point id")
    ax.set_title("Task outcome by variant and fixed point")
    for x in range(32):
        ax.axvline(x - 0.5, color="white", linewidth=0.25)
    for y in range(len(VARIANTS) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.25)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#2f8f4e", markersize=8, label="success"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#d45b4f", markersize=8, label="timeout"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#9aa0a6", markersize=8, label="other"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    path = report_root / "figures" / "fig01_task_outcome_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_success_by_category(report_root: Path, risk_summary: list[dict[str, Any]]) -> Path:
    categories = sorted({row["risk_tag"] for row in risk_summary})
    matrix = np.zeros((len(VARIANTS), len(categories)))
    for i, variant in enumerate(VARIANTS):
        for j, category in enumerate(categories):
            row = next(r for r in risk_summary if r["variant"] == variant and r["risk_tag"] == category)
            expected = int_value(row["expected_tasks"])
            matrix[i, j] = int_value(row["success_count"]) / expected if expected else 0.0
    fig, ax = plt.subplots(figsize=(10, 3.7), dpi=180)
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(VARIANTS)), [DISPLAY_VARIANTS[v] for v in VARIANTS])
    ax.set_xticks(range(len(categories)), categories, rotation=35, ha="right")
    ax.set_title("Success fraction by risk category")
    for i in range(len(VARIANTS)):
        for j in range(len(categories)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="success fraction")
    fig.tight_layout()
    path = report_root / "figures" / "fig02_success_by_risk_category.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_common_runtime(report_root: Path, runtime_rows: list[dict[str, Any]]) -> Path:
    means = []
    mins = []
    maxs = []
    for variant in VARIANTS:
        values = [float_value(row["point_total_runtime_sec"]) / 60.0 for row in runtime_rows if row["variant"] == variant]
        means.append(sum(values) / len(values) if values else float("nan"))
        mins.append(min(values) if values else float("nan"))
        maxs.append(max(values) if values else float("nan"))
    x = np.arange(len(VARIANTS))
    lower = np.asarray(means) - np.asarray(mins)
    upper = np.asarray(maxs) - np.asarray(means)
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=180)
    ax.bar(x, means, color=["#526d82", "#6c8c6b", "#c57b57", "#c57b57", "#c57b57"])
    ax.errorbar(x, means, yerr=[lower, upper], fmt="none", ecolor="#333333", capsize=4, linewidth=1)
    ax.set_xticks(x, [DISPLAY_VARIANTS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel("mean runtime on common points (min)")
    ax.set_title("Runtime on the eight completed common points")
    fig.tight_layout()
    path = report_root / "figures" / "fig03_common_runtime_by_variant.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_common_boxes(report_root: Path, runtime_rows: list[dict[str, Any]]) -> Path:
    means = []
    for variant in VARIANTS:
        values = [float_value(row["local_boxes_refined_count"]) for row in runtime_rows if row["variant"] == variant]
        means.append(sum(values) / len(values) if values else float("nan"))
    fig, ax = plt.subplots(figsize=(8.5, 3.2), dpi=180)
    ax.bar(range(len(VARIANTS)), means, color=["#526d82", "#6c8c6b", "#c57b57", "#c57b57", "#c57b57"])
    ax.set_xticks(range(len(VARIANTS)), [DISPLAY_VARIANTS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel("mean local boxes")
    ax.set_title("Local-box count on common completed points")
    fig.tight_layout()
    path = report_root / "figures" / "fig04_common_local_boxes.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown(
    report_root: Path,
    package: dict[str, Any],
    task: dict[str, Any],
    eq_runtime: dict[str, Any],
    figure_paths: list[Path],
) -> Path:
    array_status = package["array_status"]
    variant_rows = task["variant_summary"]
    eq_rows = eq_runtime["equivalence_rows"]
    common_rows = eq_runtime["common_runtime_rows"]
    lines = [
        "# Local-refinement Variant Array Return Report",
        "",
        f"Source return root: `{package['package_row']['return_root']}`",
        "",
        "## Executive Finding",
        "",
        "The returned array suite produced a diagnostic archive, but the full Stage 2-4 gate did not pass.",
        f"Only {array_status.get('successful_tasks')} of {array_status.get('expected_tasks')} point tasks completed successfully.",
        "The failure mode is dominated by Slurm time-limit termination, not by CUDA initialization or import-path failure.",
        "",
        "The physics-equivalence comparison is strong on completed points: all available candidate rows match the fresh baseline exactly within the configured thresholds.  However, the three optimized candidate variants completed only the clean superconducting control points, so the full boundary/rerun/q-edge/near-degenerate gate remains unresolved.",
        "",
        "## Task Completion",
        "",
        "| variant | success | timeout | success rate | mean success runtime (min) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in variant_rows:
        lines.append(
            "| {variant} | {success} / {expected} | {timeout} | {rate:.1%} | {runtime:.1f} |".format(
                variant=row["variant"],
                success=int_value(row["success_count"]),
                expected=int_value(row["expected_tasks"]),
                timeout=int_value(row["timeout_count"]),
                rate=float_value(row["success_rate"], 0.0),
                runtime=float_value(row["mean_success_runtime_sec"]) / 60.0,
            )
        )
    lines.extend(["", "## Equivalence Summary", ""])
    lines.extend(
        [
            "| variant | common points | missing | flag mismatches | max q diff | max Delta diff | max DeltaF diff | status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in eq_rows:
        lines.append(
            "| {variant} | {common} | {missing} | {mismatch} | {q} | {d} | {df} | {status} |".format(
                variant=row["variant"],
                common=row["n_common_points"],
                missing=row["n_missing_in_candidate"],
                mismatch=row["flag_mismatch_count"],
                q=row["max_q_opt_abs_diff"],
                d=row["max_delta_opt_abs_diff"],
                df=row["max_deltaf_abs_diff"],
                status=row["status"],
            )
        )
    common_ids = sorted({int_value(row["point_id"]) for row in common_rows})
    lines.extend(
        [
            "",
            "## Performance Interpretation",
            "",
            f"The common completed set across all variants contains point ids `{common_ids}`. These are the clean FFLO and clean uniform-SC control points.",
            "On this common set, the optimized variants are roughly an order of magnitude slower than baseline/cluster_only because they refine about 85 local boxes per point rather than 6.",
            "The extended Slurm time limit does not justify a 2h-gate success claim: the 72 incomplete tasks are precisely the boundary-band, q-edge, rerun-required, near-degenerate, previous-correction, and stable-normal representatives.",
            "",
            "## Figures",
            "",
        ]
    )
    captions = [
        "Task outcome heatmap. The optimized variants complete only points 4-11.",
        "Risk-category success fractions. Hard categories are unresolved for all three optimized variants.",
        "Runtime on the eight completed common points.",
        "Local-box count on the eight completed common points.",
    ]
    for path, caption in zip(figure_paths, captions):
        rel = path.relative_to(report_root).as_posix()
        lines.extend([f"![{caption}]({rel})", "", caption, ""])
    lines.extend(
        [
            "## Decision",
            "",
            "Do not accept the Stage 2-4 optimized local-refinement gate as complete.  Accept only the following limited conclusions:",
            "",
            "- `cluster_only` is physics-equivalent to baseline on all 32 fixed points.",
            "- `cluster_optional_k3`, `cluster_optional_k2`, and `cluster_energy_window` are physics-equivalent to baseline on the eight clean completed points.",
            "- The optimized variants show a severe performance regression on the completed clean points and fail to complete all hard-risk categories under the current time limits.",
            "",
            "Recommended next calculation: rerun only the 72 missing optimized-variant tasks with a corrected cap strategy or a smaller diagnostic subset that records local-box selection details before timeout.  Do not rerun baseline or `cluster_only`.",
            "",
        ]
    )
    path = report_root / f"{REPORT_NAME}.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def write_decision_log(report_root: Path, package: dict[str, Any]) -> Path:
    row = package["package_row"]
    text = f"""# Decision Log

- status: fail
- return_archive_exists: {row['return_archive_exists']}
- successful_tasks: {row['successful_tasks']} / {row['expected_tasks']}
- task_status: {row['task_status']}
- comparison_status: {row['comparison_status']}
- performance_status: {row['performance_status']}
- conclusion: cluster_only passes full fixed-point equivalence; optimized variants only pass the eight clean completed points and fail the full hard-point gate.
- next_action: inspect and rerun the 72 optimized-variant timeout tasks, or redesign selective-refinement caps before accepting Stage 2-4.
"""
    path = report_root / "decision_log.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def small_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 8) -> list[list[str]]:
    out = [columns]
    for row in rows[:max_rows]:
        out.append([str(row.get(col, "")) for col in columns])
    return out


def write_pdf(
    report_root: Path,
    package: dict[str, Any],
    task: dict[str, Any],
    eq_runtime: dict[str, Any],
    figure_paths: list[Path],
) -> Path:
    path = report_root / f"{REPORT_NAME}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: list[Any] = []
    story.append(Paragraph("Local-refinement Variant Array Return Report", styles["Title"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Executive finding", styles["Heading2"]))
    story.append(
        Paragraph(
            "The returned array suite produced a diagnostic archive, but the full Stage 2-4 gate did not pass. "
            f"{package['array_status'].get('successful_tasks')} of {package['array_status'].get('expected_tasks')} "
            "point tasks completed successfully. Completed candidate rows are physics-equivalent to baseline, "
            "but the optimized variants are incomplete on the hard-risk categories.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    task_table_data = [
        ["variant", "success", "timeout", "success rate", "mean runtime min"],
    ]
    for row in task["variant_summary"]:
        task_table_data.append(
            [
                row["variant"],
                f"{int_value(row['success_count'])}/{int_value(row['expected_tasks'])}",
                str(int_value(row["timeout_count"])),
                f"{float_value(row['success_rate'], 0.0):.1%}",
                f"{float_value(row['mean_success_runtime_sec']) / 60.0:.1f}",
            ]
        )
    table = Table(task_table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ec")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(Image(str(figure_paths[0]), width=6.9 * inch, height=2.0 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Equivalence and runtime", styles["Heading2"]))
    eq_table_data = [["variant", "common", "missing", "mismatch", "status"]]
    for row in eq_runtime["equivalence_rows"]:
        eq_table_data.append(
            [row["variant"], row["n_common_points"], row["n_missing_in_candidate"], row["flag_mismatch_count"], row["status"]]
        )
    table = Table(eq_table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ec")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(str(figure_paths[2]), width=6.5 * inch, height=2.5 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(str(figure_paths[3]), width=6.5 * inch, height=2.5 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Risk categories and decision", styles["Heading2"]))
    story.append(Image(str(figure_paths[1]), width=6.8 * inch, height=2.8 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Decision: accept the full `cluster_only` equivalence result, but do not accept the optimized-variant gate. "
            "The optimized variants only completed the clean superconducting controls and timed out on all hard-risk categories. "
            "The next calculation should rerun the 72 missing optimized-variant tasks or reduce the selective-refinement search "
            "before any production claim.",
            styles["BodyText"],
        )
    )
    doc.build(story)
    return path


def write_readme(report_root: Path, return_root: Path) -> Path:
    text = f"""# Local-refinement Variant Array Return Report

Source return root:

```text
{return_root}
```

Main files:

```text
{REPORT_NAME}.pdf
{REPORT_NAME}.md
tables/
figures/
decision_log.md
```

Rebuild command:

```bash
python scripts/build_local_refinement_variant_array_return_report.py --return-root "{return_root}" --output-dir "{report_root}"
```
"""
    path = report_root / "README.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def build_report(return_root: Path, output_dir: Path) -> dict[str, Any]:
    report_root = output_dir
    regression_root = return_root / "reports" / "local_refinement_refactor" / "variant_regression"
    if not regression_root.exists():
        raise FileNotFoundError(f"Missing variant regression root: {regression_root}")
    if report_root.exists():
        shutil.rmtree(report_root)
    (report_root / "tables").mkdir(parents=True)
    (report_root / "figures").mkdir(parents=True)
    copy_raw_tables(report_root, regression_root)
    package = write_package_status(report_root, return_root, regression_root)
    task = build_task_tables(report_root, return_root, regression_root)
    eq_runtime = build_equivalence_and_runtime_tables(report_root, regression_root)
    figure_paths = [
        plot_task_outcome(report_root, task["task_rows"]),
        plot_success_by_category(report_root, task["risk_summary"]),
        plot_common_runtime(report_root, eq_runtime["common_runtime_rows"]),
        plot_common_boxes(report_root, eq_runtime["common_runtime_rows"]),
    ]
    md = write_markdown(report_root, package, task, eq_runtime, figure_paths)
    decision = write_decision_log(report_root, package)
    pdf = write_pdf(report_root, package, task, eq_runtime, figure_paths)
    readme = write_readme(report_root, return_root)
    return {
        "report_root": str(report_root),
        "markdown": str(md),
        "pdf": str(pdf),
        "decision_log": str(decision),
        "readme": str(readme),
        "figures": [str(path) for path in figure_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local report for returned variant-array suite results.")
    parser.add_argument("--return-root", type=Path, default=DEFAULT_RETURN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_report(args.return_root.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
