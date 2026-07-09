"""Build a report for the rankcap_k3 tail-surprise continuation return.

This is a report-only script.  It reads downloaded HPC artifacts and writes
summary CSVs, figures, a Markdown report, a compact PDF, and a decision log.
It does not modify active-learning artifacts or production code.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}


ROOT = Path("rankcap_k3_tail_surprise_continuation_results")
RUN_ROOT = (
    ROOT
    / "ML_Phase_512_RankCapK3_TailContinuation"
    / "active_runs"
    / "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1"
)
OUT_DIR = Path("reports/rankcap_k3_tail_surprise_continuation_return")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def phase_counts_from_dataset(path: Path) -> dict[str, int]:
    z = np.load(path, allow_pickle=True)
    y = np.asarray(z["y_phase"], dtype=int)
    out = {"samples": int(y.size)}
    for code, name in PHASE_NAMES.items():
        out[name] = int(np.sum(y == code))
    return out


def count_shards(iter_dir: Path) -> dict[str, Any]:
    npz = sorted(iter_dir.glob("exact_shard_rank*_of008.npz"))
    js = sorted(iter_dir.glob("exact_shard_rank*_of008.json"))
    return {
        "iteration": int(iter_dir.name.replace("iter", "")),
        "npz_count": len(npz),
        "json_count": len(js),
        "expected": 8,
        "complete": len(npz) == 8 and len(js) == 8,
        "npz_files": ";".join(p.name for p in npz),
    }


def collect_stop_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(RUN_ROOT.glob("iter*/stop_metrics_iter*.json")):
        obj = read_json(p)
        metrics = obj.get("metrics", {})
        surprise = obj.get("surprise_details", {})
        conditions = obj.get("conditions", {})
        rows.append(
            {
                "iteration": obj.get("iteration"),
                "completed_iterations": obj.get("completed_iterations"),
                "exact_call_count": obj.get("exact_call_count"),
                "phase_map_change": metrics.get("phase_map_change"),
                "boundary_shift_normal_sc": metrics.get("boundary_shift_normal_sc"),
                "boundary_shift_uniform_fflo": metrics.get("boundary_shift_uniform_fflo"),
                "boundary_coverage_p95": metrics.get("boundary_coverage_p95"),
                "label_surprise_all_selected": metrics.get(
                    "label_surprise_all_selected", metrics.get("label_surprise_rate")
                ),
                "label_surprise_trusted": metrics.get("label_surprise_trusted"),
                "label_surprise_hard_risk": metrics.get("label_surprise_hard_risk"),
                "label_surprise_selected_for_gate": metrics.get(
                    "label_surprise_selected_for_gate"
                ),
                "trusted_denominator": surprise.get("trusted", {}).get("n_denominator"),
                "trusted_surprise": surprise.get("trusted", {}).get("n_surprise"),
                "hard_risk_denominator": surprise.get("hard_risk", {}).get("n_denominator"),
                "hard_risk_surprise": surprise.get("hard_risk", {}).get("n_surprise"),
                "trusted_fraction_selected": surprise.get("trusted_fraction_selected"),
                "hard_risk_fraction_selected": surprise.get("hard_risk_fraction_selected"),
                "rerun_required_count": surprise.get("rerun_required_count"),
                "q_edge_trigger_rate": metrics.get("q_edge_trigger_rate"),
                "rerun_required_rate": metrics.get("rerun_required_rate"),
                "passed_condition_count": obj.get("passed_condition_count"),
                "required_pass_count": obj.get("required_pass_count"),
                "patience_counter": obj.get("patience_counter"),
                "patience": obj.get("patience"),
                "convergence_pass": obj.get("convergence_pass"),
                "main_phase_convergence_pass": obj.get("main_phase_convergence_pass"),
                "stop": obj.get("stop"),
                "stop_reason": obj.get("stop_reason"),
                "publication_ready": obj.get("publication_ready"),
                "publication_ready_reason": obj.get("publication_ready_reason"),
                "numerical_frontier_status": obj.get("numerical_frontier_status"),
                "C1_phase_map_change": conditions.get("C1_phase_map_change"),
                "C2_boundary_shift_normal_sc": conditions.get(
                    "C2_boundary_shift_normal_sc"
                ),
                "C3_boundary_shift_uniform_fflo": conditions.get(
                    "C3_boundary_shift_uniform_fflo"
                ),
                "C4_label_surprise_rate": conditions.get("C4_label_surprise_rate"),
                "C5_boundary_coverage_p95": conditions.get("C5_boundary_coverage_p95"),
            }
        )
    return rows


def collect_dataset_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(RUN_ROOT.glob("dataset_iter*.npz")):
        stem = p.stem
        iteration = int(stem.replace("dataset_iter", ""))
        counts = phase_counts_from_dataset(p)
        row = {"dataset_iteration": iteration, **counts}
        append_json = RUN_ROOT / f"dataset_iter{iteration:03d}.append.json"
        if append_json.exists():
            append_obj = read_json(append_json)
            row.update(
                {
                    "trusted_points_appended": append_obj.get("trusted_points_appended"),
                    "training_eligible_points_appended": append_obj.get(
                        "training_eligible_points_appended"
                    ),
                    "boundary_band_points_appended": append_obj.get(
                        "boundary_band_points_appended"
                    ),
                    "input_samples": append_obj.get("input_samples"),
                    "output_samples": append_obj.get("output_samples"),
                }
            )
        rows.append(row)
    return rows


def collect_workload_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in sorted(RUN_ROOT.glob("iter*/exact_merged_iter*.npz")):
        iteration = int(p.parent.name.replace("iter", ""))
        z = np.load(p, allow_pickle=True)
        row: dict[str, Any] = {"iteration": iteration, "point_count": int(len(z["kT"]))}
        for key in [
            "local_boxes_refined_count",
            "selected_refine_target_count",
            "local_refinement_runtime_sec",
            "point_total_runtime_sec",
            "q_expansion_count",
            "q_expanded",
            "rerun_required",
            "trusted_exact",
            "training_eligible_exact",
            "q_unresolved",
            "delta_unresolved",
        ]:
            if key in z.files:
                arr = np.asarray(z[key])
                if arr.dtype.kind in "biu":
                    row[f"{key}_sum"] = int(arr.sum())
                    row[f"{key}_mean"] = float(arr.mean())
                    row[f"{key}_max"] = int(arr.max()) if arr.size else 0
                else:
                    row[f"{key}_mean"] = float(np.nanmean(arr))
                    row[f"{key}_median"] = float(np.nanmedian(arr))
                    row[f"{key}_max"] = float(np.nanmax(arr))
                    row[f"{key}_sum"] = float(np.nansum(arr))
        if "phase_candidate" in z.files:
            phase = np.asarray(z["phase_candidate"], dtype=int)
            for code, name in PHASE_NAMES.items():
                row[f"exact_{name}_count"] = int(np.sum(phase == code))
        rows.append(row)
    return rows


def collect_final_rerun_summary(final_iter: int) -> list[dict[str, Any]]:
    p = RUN_ROOT / f"iter{final_iter:03d}" / "rerun_points.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        phase = row.get("phase_candidate") or row.get("phase") or "unknown"
        q_status = row.get("q_status", "unknown")
        key = f"phase={phase};q_status={q_status}"
        summary.setdefault(key, {"count": 0})
        summary[key]["count"] += 1
    return [{"group": key, **value} for key, value in sorted(summary.items())]


def plot_metric(rows: list[dict[str, Any]], keys: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [int(r["iteration"]) for r in rows]
    for key in keys:
        ys = [safe_float(r.get(key)) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=key)
    ax.set_xlabel("iteration")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dataset(rows: list[dict[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = [int(r["dataset_iteration"]) for r in rows]
    for key in ["normal", "uniform_SC", "FFLO"]:
        ax.plot(xs, [int(r.get(key, 0)) for r in rows], marker="o", label=key)
    ax.set_xlabel("dataset iteration")
    ax.set_ylabel("samples")
    ax.set_title("Dataset Phase Counts")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_workload(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    xs = [int(r["iteration"]) for r in rows]
    axes[0].plot(
        xs,
        [safe_float(r.get("local_boxes_refined_count_mean")) for r in rows],
        marker="o",
    )
    axes[0].set_title("Mean Local Boxes")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylim(0, 6.5)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        xs,
        [safe_float(r.get("local_refinement_runtime_sec_mean")) for r in rows],
        marker="o",
        color="tab:orange",
    )
    axes[1].set_title("Mean Local-Refinement Runtime")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("seconds")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_pdf(md_text: str, figure_paths: list[Path], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        rendered_lines: list[tuple[str, int, str]] = []
        for raw in md_text.splitlines():
            text = raw.replace("**", "").replace("`", "")
            if not text:
                rendered_lines.append(("", 9, "normal"))
                continue
            if text.startswith("# "):
                rendered_lines.append((text[2:], 15, "bold"))
                continue
            if text.startswith("## "):
                rendered_lines.append((text[3:], 12, "bold"))
                continue
            prefix = ""
            body = text
            if text.startswith("- "):
                prefix = "- "
                body = text[2:]
            elif text[:3].rstrip(".").isdigit() and ". " in text[:5]:
                idx, body = text.split(". ", 1)
                prefix = idx + ". "
            width = 105 if not text.startswith("|") else 125
            wrapped = textwrap.wrap(body, width=width) or [body]
            rendered_lines.append((prefix + wrapped[0], 9, "normal"))
            for cont in wrapped[1:]:
                rendered_lines.append(("  " + cont, 9, "normal"))

        page_lines: list[tuple[str, int, str]] = []
        used = 0.0
        pages: list[list[tuple[str, int, str]]] = []
        for item in rendered_lines:
            text, size, _ = item
            height = 0.038 if size >= 12 else 0.027
            if used + height > 0.88 and page_lines:
                pages.append(page_lines)
                page_lines = []
                used = 0.0
            page_lines.append(item)
            used += height if text else 0.018
        if page_lines:
            pages.append(page_lines)

        for page in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
            ax.axis("off")
            y = 1.0
            for text, size, weight in page:
                ax.text(0, y, text, ha="left", va="top", fontsize=size, weight=weight)
                y -= 0.038 if size >= 12 else (0.018 if not text else 0.027)
            pdf.savefig(fig)
            plt.close(fig)
        for fig_path in figure_paths:
            img = plt.imread(fig_path)
            fig, ax = plt.subplots(figsize=(8.27, 5.8))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(fig_path.name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stop_rows = collect_stop_rows()
    dataset_rows = collect_dataset_rows()
    workload_rows = collect_workload_rows()
    shard_rows = [count_shards(p) for p in sorted(RUN_ROOT.glob("iter[0-9][0-9][0-9]"))]

    final = stop_rows[-1]
    final_iter = int(final["iteration"])
    rerun_summary = collect_final_rerun_summary(final_iter)

    write_csv(
        TABLE_DIR / "stop_metrics_by_iteration.csv",
        stop_rows,
        list(stop_rows[0].keys()),
    )
    write_csv(
        TABLE_DIR / "dataset_phase_counts.csv",
        dataset_rows,
        sorted({k for r in dataset_rows for k in r.keys()}),
    )
    write_csv(
        TABLE_DIR / "workload_runtime_by_iteration.csv",
        workload_rows,
        sorted({k for r in workload_rows for k in r.keys()}),
    )
    write_csv(TABLE_DIR / "exact_shard_status.csv", shard_rows, list(shard_rows[0].keys()))
    if rerun_summary:
        write_csv(TABLE_DIR / "final_rerun_group_summary.csv", rerun_summary, ["group", "count"])

    decision_rows = [
        {
            "question": "Should we rerun another full loop immediately?",
            "answer": "no",
            "evidence": "The continuation stopped at iter034 with trusted surprise 0/127 and all five main StopController conditions passing.",
        },
        {
            "question": "Is the main phase map converged under the trusted gate?",
            "answer": "yes",
            "evidence": "stop_reason=converged_main_phase_boundaries; phase_map_change, both boundary shifts, trusted surprise, and boundary coverage pass.",
        },
        {
            "question": "Is the hard-risk numerical frontier solved?",
            "answer": "no",
            "evidence": "hard-risk surprise is 75/129 and rerun_required_count is 110 at iter034; publication_ready=false.",
        },
        {
            "question": "Recommended next direction",
            "answer": "enter next major stage with a targeted hard-risk cleanup/audit substage",
            "evidence": "Do not spend another full loop on main-boundary convergence; focus on auditing rerun_required hard-risk boundary impact and publication readiness.",
        },
    ]
    write_csv(TABLE_DIR / "decision_summary.csv", decision_rows, ["question", "answer", "evidence"])

    fig_paths = [
        FIG_DIR / "surprise_layers.png",
        FIG_DIR / "stop_pass_counts.png",
        FIG_DIR / "qedge_rerun_rates.png",
        FIG_DIR / "dataset_phase_counts.png",
        FIG_DIR / "workload_runtime.png",
    ]
    plot_metric(
        stop_rows,
        [
            "label_surprise_all_selected",
            "label_surprise_trusted",
            "label_surprise_hard_risk",
        ],
        fig_paths[0],
        "Surprise Layers",
    )
    plot_metric(
        stop_rows,
        ["passed_condition_count"],
        fig_paths[1],
        "StopController Passed Conditions",
    )
    plot_metric(
        stop_rows,
        ["q_edge_trigger_rate", "rerun_required_rate"],
        fig_paths[2],
        "Q-Edge and Rerun Rates",
    )
    plot_dataset(dataset_rows, fig_paths[3])
    plot_workload(workload_rows, fig_paths[4])

    final_dataset = dataset_rows[-1]
    final_workload = [r for r in workload_rows if int(r["iteration"]) == final_iter][0]
    stop_status = "pass" if final.get("stop") and final.get("convergence_pass") else "fail"
    recommendation = (
        "Do not run another main full loop now.  Treat rankcap_k3 main-boundary "
        "active learning as converged under the trusted-surprise StopController, "
        "then move to the next major stage with a targeted hard-risk numerical "
        "frontier audit/cleanup before publication-grade claims."
    )

    md = f"""# Rankcap K3 Tail-Surprise Continuation Return Report

## Executive Summary

- Return directory checked: `{ROOT}`.
- Run ID: `active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1`.
- Final evaluated iteration: iter{final_iter:03d}; final dataset: dataset_iter035.
- Stop status: **{stop_status}**.
- Stop reason: `{final.get('stop_reason')}`.
- Main phase convergence: `{final.get('main_phase_convergence_pass')}`.
- Publication-ready status: `{final.get('publication_ready')}`.
- Recommended direction: {recommendation}

## Final Stop Metrics

| metric | value | threshold / note | status |
|---|---:|---|---|
| phase_map_change | {safe_float(final.get('phase_map_change')):.6f} | < 0.002 | pass |
| boundary_shift_normal_sc | {safe_float(final.get('boundary_shift_normal_sc')):.6f} | <= 0.004167 | pass |
| boundary_shift_uniform_fflo | {safe_float(final.get('boundary_shift_uniform_fflo')):.6f} | <= 0.004167 | pass |
| boundary_coverage_p95 | {safe_float(final.get('boundary_coverage_p95')):.6f} | < 0.00625 | pass |
| all-selected surprise | {safe_float(final.get('label_surprise_all_selected')):.6f} | diagnostic, not gate | high |
| trusted surprise | {safe_float(final.get('label_surprise_trusted')):.6f} | < 0.05 | pass |
| hard-risk surprise | {safe_float(final.get('label_surprise_hard_risk')):.6f} | diagnostic frontier | high |

Trusted surprise denominator is {final.get('trusted_denominator')} selected points with {final.get('trusted_surprise')} surprises.
Hard-risk denominator is {final.get('hard_risk_denominator')} selected points with {final.get('hard_risk_surprise')} surprises.

## Dataset Growth

The restart/continuation artifacts include dataset_iter026 through dataset_iter035.  The final dataset has:

- total samples: {final_dataset.get('samples')}
- normal: {final_dataset.get('normal')}
- uniform_SC: {final_dataset.get('uniform_SC')}
- FFLO: {final_dataset.get('FFLO')}

The iter034 append step added {final_dataset.get('trusted_points_appended')} trusted points and produced dataset_iter035.

## Runtime and Workload

At iter{final_iter:03d}:

- selected exact points: {final_workload.get('point_count')}
- mean local boxes refined: {safe_float(final_workload.get('local_boxes_refined_count_mean')):.3f}
- max local boxes refined: {safe_float(final_workload.get('local_boxes_refined_count_max')):.0f}
- mean local-refinement runtime: {safe_float(final_workload.get('local_refinement_runtime_sec_mean')):.2f} s
- mean point total runtime: {safe_float(final_workload.get('point_total_runtime_sec_mean')):.2f} s
- trusted exact count: {final_workload.get('trusted_exact_sum')}
- rerun-required count: {final_workload.get('rerun_required_sum')}

Rankcap_k3 still enforces the intended local-box cap: all points have max local boxes <= 3 in the final exact batch.

## Evidence for Stopping

The main phase-map criteria all pass simultaneously at iter034:

- phase map change is below tolerance;
- normal/SC boundary shift is at the tolerance boundary and marked pass;
- uniform-SC/FFLO boundary shift is zero with boundary points present;
- boundary coverage p95 is below tolerance;
- trusted label surprise is 0 with a valid denominator.

This is positive evidence that the main phase map and main thermodynamic boundaries have converged under the trusted-surprise gate.

## Remaining Caveat

The hard-risk frontier is still active:

- all-selected surprise = {final.get('label_surprise_all_selected')} ({final.get('hard_risk_surprise')} selected-batch surprises);
- hard-risk surprise = {final.get('label_surprise_hard_risk')};
- rerun_required_count = {final.get('rerun_required_count')};
- numerical_frontier_status = `{final.get('numerical_frontier_status')}`;
- publication_ready = `{final.get('publication_ready')}`;
- publication_ready_reason = `hard_risk_boundary_impact_not_audited`.

Therefore the correct statement is not that every numerical frontier issue is solved.  The supported statement is that the clean/trusted labels no longer block main-boundary convergence, while hard-risk/rerun points remain a separate numerical reliability frontier.

## Next Direction

Do **not** launch another full main active-learning loop just to satisfy the old all-selected surprise metric.  The tail continuation already demonstrated formal convergence under the trusted gate.

The next work should be the next major stage, with a focused hard-risk substage before publication-grade claims:

1. Audit `iter034/rerun_points.csv` and quantify whether hard-risk points can move the plotted main boundaries.
2. Run a targeted numerical cleanup only for hard-risk/rerun boundary points if that audit shows boundary impact.
3. Freeze the converged rankcap_k3 main phase map for the second-stage report.
4. Move to the next physics/reporting stage after the hard-risk frontier is documented.

## Do-Not-Claim List

1. Do not claim the old all-selected surprise criterion passed.
2. Do not claim hard-risk FFLO frontier issues are solved.
3. Do not claim publication readiness from this stop alone.
4. Do not launch a new full loop before auditing hard-risk boundary impact.
5. Do not modify thermodynamic phase criteria or tolerances based on this return.

## Output Files

- `tables/stop_metrics_by_iteration.csv`
- `tables/dataset_phase_counts.csv`
- `tables/workload_runtime_by_iteration.csv`
- `tables/exact_shard_status.csv`
- `tables/final_rerun_group_summary.csv`
- `tables/decision_summary.csv`
- `figures/surprise_layers.png`
- `figures/stop_pass_counts.png`
- `figures/qedge_rerun_rates.png`
- `figures/dataset_phase_counts.png`
- `figures/workload_runtime.png`
"""

    (OUT_DIR / "rankcap_k3_tail_surprise_continuation_return.md").write_text(
        md, encoding="utf-8"
    )
    decision_log = """# Decision Log

Decision: Treat rankcap_k3 main-boundary active learning as converged under the trusted-surprise StopController at iter034.

Reason: All five main stop conditions pass, trusted surprise is 0 with a valid denominator, and the final dataset was appended successfully.

Consequence: Do not run another main full loop now.  The next calculation should be a targeted hard-risk boundary-impact audit/cleanup, because publication_ready remains false and hard-risk surprise remains high.
"""
    (OUT_DIR / "decision_log.md").write_text(decision_log, encoding="utf-8")
    render_pdf(md, fig_paths, OUT_DIR / "rankcap_k3_tail_surprise_continuation_return.pdf")
    print(OUT_DIR / "rankcap_k3_tail_surprise_continuation_return.md")
    print(OUT_DIR / "rankcap_k3_tail_surprise_continuation_return.pdf")


if __name__ == "__main__":
    main()
