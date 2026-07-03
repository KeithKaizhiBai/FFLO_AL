"""Build the organized Phase-II LaTeX discussion reports.

This is report-only glue.  It reads already generated CSV, JSON, Markdown and
PNG artifacts, creates index files and presentation diagrams, writes LaTeX
sources, and leaves compilation to ``pdflatex``.  It does not recompute exact
labels or modify active-learning production code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(".")
PROJECT_REPORT_ROOT = Path("project_history/reports")
SUPPORT_PROJECT_DIR = PROJECT_REPORT_ROOT / "_supporting_reports"
SUPPORT_PHASE2_DIR = Path("reports/_phase2_supporting_reports")
AUDIT_DIR = SUPPORT_PHASE2_DIR / "hard_risk_boundary_impact_audit_v2"
FINAL_DIR = PROJECT_REPORT_ROOT / "report_phase2_robust_al_final_202606"
MANIFEST_PATH = FINAL_DIR / "reproduction_manifest.json"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: Any) -> str:
    text = str(value)
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
    return "".join(repl.get(ch, ch) for ch in text)


def fmt_float(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except Exception:
        return str(value)


def short_hash(value: Any, chars: int = 16) -> str:
    text = str(value or "")
    if len(text) <= chars:
        return text
    return text[:chars] + "..."


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def tex_preamble(title: str) -> str:
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\usepackage{{amsmath}}
\usepackage{{enumitem}}
\usepackage{{fancyvrb}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlist[itemize]{{leftmargin=1.5em}}
\setlist[enumerate]{{leftmargin=1.5em}}
\captionsetup{{font=small, labelfont=bf}}
\sloppy
\title{{{latex_escape(title)}}}
\date{{2026-06-20}}
\begin{{document}}
\maketitle
\tableofcontents
\newpage
"""


def tex_end() -> str:
    return "\n\\end{document}\n"


def key_value_table(rows: list[tuple[str, Any]], caption: str | None = None) -> str:
    lines = []
    if caption:
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}p{0.43\linewidth}p{0.47\linewidth}@{}}")
    lines.append(r"\toprule")
    lines.append(r"Item & Value \\")
    lines.append(r"\midrule")
    for key, val in rows:
        lines.append(f"{latex_escape(key)} & {latex_escape(val)} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if caption:
        lines.append(rf"\caption{{{latex_escape(caption)}}}")
        lines.append(r"\end{table}")
    return "\n".join(lines)


def simple_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    caption: str,
    widths: list[float] | None = None,
) -> str:
    if widths is None:
        widths = [0.9 / len(columns)] * len(columns)
    colspec = "".join(f"p{{{w:.2f}\\linewidth}}" for w in widths)
    out = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"\toprule",
    ]
    out.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    out.append(r"\midrule")
    for row in rows:
        out.append(" & ".join(latex_escape(row.get(c, "")) for c in columns) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(rf"\caption{{{latex_escape(caption)}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def figure(path: str, caption: str, width: str = r"0.84\linewidth") -> str:
    return rf"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{{path}}}
\caption{{{latex_escape(caption)}}}
\end{{figure}}
"""


GATE_LABELS = {
    "single_and_cluster_p95_shift_within_tolerance": "p95 <= tol",
    "significant_local_max_and_hausdorff_no_spike": "sig. Hausdorff <= tol",
    "strict_local_hausdorff_diagnostic": "strict Hausdorff diag.",
    "no_meaningful_topology_change": "topology stable",
    "no_boundary_moving_cluster": "no moving cluster",
    "hard_risk_reason_counts_reconciled": "counts reconciled",
    "critical_metadata_present": "metadata present",
    "targeted_rerun_list_empty": "targeted list empty",
    "publication_boundary_audit": "publication audit",
}


def compact_gate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        threshold = r.get("threshold", "")
        if r.get("gate") == "strict_local_hausdorff_diagnostic":
            threshold = "diagnostic; sig. arc >= 0.05"
        out.append(
            {
                "gate": GATE_LABELS.get(r.get("gate", ""), r.get("gate", "")),
                "status": r.get("status", ""),
                "value": fmt_float(r.get("value", "")),
                "threshold": fmt_float(threshold),
            }
        )
    return out


def compact_boundary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        out.append(
            {
                "boundary": r.get("boundary_type", "").replace("normal_sc", "normal/SC").replace("uniform_fflo", "uniform/FFLO"),
                "points": r.get("boundary_point_count", ""),
                "components": r.get("connected_component_count", ""),
                "note": "exists; StopController crossing rule",
            }
        )
    return out


def compact_cluster_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for r in rows:
        out.append(
            {
                "id": r.get("cluster_id", ""),
                "n": r.get("point_count", ""),
                "boundary": r.get("nearest_boundary_type", "").replace("normal_sc", "normal/SC").replace("uniform_fflo", "uniform/FFLO"),
                "p95": fmt_float(r.get("local_p95_shift", "")),
                "haus": fmt_float(r.get("hausdorff_shift", "")),
                "arc": fmt_float(r.get("affected_arc_length_fraction", "")),
                "class": "local below tol" if r.get("influence_class") == "locally_influential_below_tolerance" else r.get("influence_class", ""),
            }
        )
    return out


def draw_boxes(
    path: Path,
    title: str,
    boxes: list[tuple[str, str]],
    columns: int = 3,
    color: str = "#e8f0fe",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (len(boxes) + columns - 1) // columns
    fig, ax = plt.subplots(figsize=(12, max(3.6, 2.2 * rows)))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=15, pad=16)
    w = min(0.26, 0.86 / columns)
    h = 0.28 / max(1, rows / 2)
    x_gap = (1 - columns * w) / (columns + 1)
    y_gap = (0.82 - rows * h) / max(1, rows + 1)
    rects: list[tuple[float, float, float, float, int, int]] = []
    for i, (head, body) in enumerate(boxes):
        r = i // columns
        c = i % columns
        x = x_gap + c * (w + x_gap)
        y = 0.83 - r * (h + y_gap) - h
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, head, ha="center", va="center", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=8.2, wrap=True)
        rects.append((x, y, w, h, r, c))
    for i in range(len(rects) - 1):
        x1, y1, w1, h1, r1, c1 = rects[i]
        x2, y2, _w2, h2, r2, _c2 = rects[i + 1]
        if r1 == r2:
            arrow = FancyArrowPatch(
                (x1 + w1 + 0.008, y1 + h1 / 2),
                (x2 - 0.008, y2 + h2 / 2),
                arrowstyle="->",
                mutation_scale=12,
                color="#475569",
            )
            ax.add_patch(arrow)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def build_overview_figures() -> None:
    fig_dir = FINAL_DIR / "figures"
    draw_boxes(
        fig_dir / "phase2_from_seed_loop.png",
        "RankCapK3 Active-Learning Loop from Random Seed",
        [
            ("Random seed", "initial exact seed set"),
            ("Train surrogate", "phase classifier + diagnostics"),
            ("Full acquisition", "boundary + uncertainty + repulsion"),
            ("Exact oracle", "q-window + Delta + local boxes"),
            ("Merge / append", "trusted labels into dataset"),
            ("StopController", "phase map, boundaries, trusted surprise"),
        ],
        columns=3,
        color="#e8f7f0",
    )
    draw_boxes(
        fig_dir / "phase2_optimization_timeline.png",
        "Phase-II Optimization Timeline",
        [
            ("Grid baseline", "known exact phase structure"),
            ("Acquisition fixes", "B_delta gating, active pool, repulsion"),
            ("Oracle reliability", "label closure, q/Delta checks"),
            ("Incremental q", "reuse scanned q strips safely"),
            ("Local refinement", "85-box failure diagnosed"),
            ("RankCapK3", "basin risk + hard cap <= 3"),
            ("Validation ladder", "fixed points, 1 iter, 5 iter"),
            ("Full loop", "30 acquisition batches"),
            ("Tail continuation", "trusted surprise stop"),
            ("Publication audit", "hard-risk boundary impact pass"),
        ],
        columns=5,
        color="#eef2ff",
    )
    draw_boxes(
        fig_dir / "phase2_acquisition_oracle_flow.png",
        "Acquisition and Exact-Oracle Responsibility Split",
        [
            ("Acquisition", "selects informative candidates"),
            ("No phase authority", "cannot replace exact BdG"),
            ("Exact BdG", "minimizes free energy"),
            ("Reliability tags", "trusted / rerun / unresolved"),
            ("Training gate", "append trusted exact labels"),
            ("Numerical frontier", "hard-risk points disclosed separately"),
        ],
        columns=3,
        color="#fff7ed",
    )


def stage_summary_rows() -> list[dict[str, str]]:
    return [
        {"stage": "Initial exact grid", "objective": "Reference phase structure", "main_result": "Coarse exact map available; active learning need is boundary refinement", "evidence": "project_history/reports/report_active_learning_r0015_note"},
        {"stage": "Acquisition development", "objective": "Avoid deep-normal oversampling while retaining discovery", "main_result": "Full acquisition retained for FFLO/uniform-SC coverage; simple-phase not used for final discovery", "evidence": "reports/_phase2_supporting_reports/acquisition_profile_smoke"},
        {"stage": "Robust exact oracle", "objective": "Separate phase labels from numerical reliability", "main_result": "Trusted, training-eligible, rerun-required, q/Delta unresolved layers recorded", "evidence": "reports/_phase2_supporting_reports/robust_oracle_label_closure_validation"},
        {"stage": "Incremental q-window", "objective": "Reduce repeated q-window work", "main_result": "Incremental strip computation retained with fallback and reliability audit", "evidence": "project_history/reports/_supporting_reports/report_phase_qwindow_delta_refinement_v1"},
        {"stage": "Local-refinement audit", "objective": "Find why optimized variants timed out", "main_result": "Mandatory overflow produced approximately 85 boxes in failed variants", "evidence": "reports/_phase2_supporting_reports/local_refinement_target_logic_audit"},
        {"stage": "RankCapK3", "objective": "Bound selected refinement targets", "main_result": "Fixed-point acceptance passed; selected targets <= 3 with physics-equivalent labels", "evidence": "reports/_phase2_supporting_reports/local_refinement_rankcap_acceptance"},
        {"stage": "One- and five-iteration validation", "objective": "Check real AL behavior before full run", "main_result": "RankCapK3 remained stable and reduced local boxes in active-loop context", "evidence": "reports/_phase2_supporting_reports/rankcap_k3_one_iter_validation; rankcap_k3_5iter_validation"},
        {"stage": "Full loop", "objective": "Run production discovery with optimized oracle", "main_result": "30 acquisition batches completed; main phase nearly stable but all-selected surprise remained high", "evidence": "reports/_phase2_supporting_reports/rankcap_k3_full_loop"},
        {"stage": "Trusted-surprise tail", "objective": "Use trusted labels as formal main-phase surprise gate", "main_result": "Stopped at iter034; trusted surprise 0/127; dataset_iter035 frozen", "evidence": "reports/_phase2_supporting_reports/rankcap_k3_tail_surprise_continuation_return"},
        {"stage": "Hard-risk publication audit", "objective": "Test whether hard-risk frontier can move main boundaries", "main_result": "Decision A; targeted rerun count 0; publication boundary audit pass", "evidence": "reports/_phase2_supporting_reports/hard_risk_boundary_impact_audit_v2"},
    ]


def write_report_indexes() -> None:
    project_rows = []
    for path in sorted(PROJECT_REPORT_ROOT.iterdir()):
        if path.is_dir():
            role = "main report" if path.name in {"report_active_learning_r0015_note", "report_phase2_robust_al_final_202606"} else "supporting container"
            project_rows.append({"directory": str(path), "role": role})
    phase2_rows = []
    for path in sorted(SUPPORT_PHASE2_DIR.iterdir()):
        if path.is_dir():
            phase2_rows.append({"directory": str(path), "role": "Phase-II supporting report"})
    write_csv(FINAL_DIR / "tables" / "supporting_report_inventory.csv", project_rows + phase2_rows, ["directory", "role"])

    (PROJECT_REPORT_ROOT / "README.md").write_text(
        "# Project Report Entry Points\n\n"
        "Top-level discussion reports are intentionally limited to:\n\n"
        "- `report_active_learning_r0015_note/`: first-stage active-learning report.\n"
        "- `report_phase2_robust_al_final_202606/`: Phase-II robust active-learning final report.\n"
        "- `_supporting_reports/`: archived supporting reports.\n\n",
        encoding="utf-8",
    )
    (SUPPORT_PROJECT_DIR / "README.md").write_text(
        "# Supporting Project-History Reports\n\n"
        "This folder holds older or narrower reports that support the two main "
        "discussion reports but should not clutter the first-level report view.\n",
        encoding="utf-8",
    )
    Path("reports/README.md").write_text(
        "# Reports Directory\n\n"
        "Phase-II audit, validation, and debugging reports are archived under "
        "`_phase2_supporting_reports/`.  The main Phase-II discussion report is "
        "under `project_history/reports/report_phase2_robust_al_final_202606/`.\n",
        encoding="utf-8",
    )
    (SUPPORT_PHASE2_DIR / "README.md").write_text(
        "# Phase-II Supporting Reports\n\n"
        "This folder contains the smaller audit and validation reports used to "
        "build the Phase-II final report.  Keep new narrow reports here unless "
        "they become one of the main discussion reports.\n",
        encoding="utf-8",
    )


def build_audit_tex() -> Path:
    counts = read_csv_rows(AUDIT_DIR / "tables" / "hard_risk_reason_counts.csv")
    gates = read_csv_rows(AUDIT_DIR / "tables" / "audit_gate_summary.csv")
    boundary = read_csv_rows(AUDIT_DIR / "tables" / "boundary_definition_check.csv")
    clusters = read_csv_rows(AUDIT_DIR / "tables" / "hard_risk_cluster_impact.csv")
    count_map = {r["reason"]: r["count"] for r in counts}
    gate_map = {r["gate"]: r for r in gates}
    cluster_rows = sorted(clusters, key=lambda r: float(r.get("hausdorff_shift") or 0), reverse=True)[:8]

    tex = [tex_preamble("Hard-Risk Boundary-Impact Audit V2")]
    tex.append(r"\section{Executive Summary}")
    tex.append(
        key_value_table(
            [
                ("publication_boundary_audit", gate_map["publication_boundary_audit"]["status"]),
                ("audit decision", "Decision A"),
                ("need_new_exact_calculation", "False"),
                ("targeted_rerun_count", "0"),
                ("hard_risk_total", count_map["hard_risk_total"]),
                ("rerun_required", count_map["rerun_required"]),
                ("non_rerun_hard_risk", count_map["non_rerun_hard_risk"]),
                ("boundary-near hard-risk", "88"),
                ("strict local Hausdorff diagnostic", gate_map["strict_local_hausdorff_diagnostic"]["value"]),
                ("significant local Hausdorff gate value", gate_map["significant_local_max_and_hausdorff_no_spike"]["value"]),
            ],
            "Audit v2 decision summary.",
        )
    )
    tex.append(
        "The audit is report-only. It uses copied dense-grid phase maps for "
        "counterfactual label flips and does not modify production code, phase "
        "criteria, tolerances, datasets, or Slurm state."
    )
    tex.append(r"\section{Hard-Risk Definition and Count Reconciliation}")
    tex.append(simple_table(counts, ["reason", "count"], "Hard-risk reason counts.", [0.65, 0.25]))
    tex.append(
        "The difference between rerun-required count 110 and hard-risk total "
        "129 is explained by 19 non-rerun hard-risk points that are untrusted "
        "or training-ineligible."
    )
    tex.append(r"\section{Boundary Reconstruction}")
    tex.append(simple_table(compact_boundary_rows(boundary), ["boundary", "points", "components", "note"], "Final boundary definition check.", [0.18, 0.14, 0.17, 0.43]))
    tex.append(r"\section{Counterfactual Boundary-Impact Gate}")
    tex.append(simple_table(compact_gate_rows(gates), ["gate", "status", "value", "threshold"], "Publication audit gate summary.", [0.31, 0.14, 0.16, 0.31]))
    tex.append(
        "Strict Hausdorff and directed-max outliers are retained as diagnostics. "
        "The publication gate treats an outlier as boundary-moving only when it "
        "affects a significant boundary arc or changes significant main-boundary topology."
    )
    tex.append(r"\section{Cluster-Level Impact}")
    tex.append(simple_table(compact_cluster_rows(cluster_rows), ["id", "n", "boundary", "p95", "haus", "arc", "class"], "Largest strict Hausdorff cluster diagnostics.", [0.06, 0.06, 0.15, 0.10, 0.12, 0.12, 0.23]))
    tex.append(r"\section{Figures}")
    tex.append(figure("figures/hard_risk_points_on_phase_map.png", "Hard-risk points over the final phase map."))
    tex.append(figure("figures/hard_risk_points_by_influence.png", "Hard-risk points colored by influence class."))
    tex.append(figure("figures/hausdorff_shift_distribution.png", "Strict Hausdorff diagnostics across counterfactuals."))
    tex.append(figure("figures/final_uncertainty_band_preview.png", "Final uncertainty-band preview."))
    tex.append(r"\section{Decision}")
    tex.append(
        "Decision A is retained: no new exact calculation is required, the "
        "targeted rerun list is empty, and the final main phase map can be used "
        "with an explicit hard-risk uncertainty layer."
    )
    tex.append(tex_end())
    out = AUDIT_DIR / "hard_risk_boundary_impact_audit_v2.tex"
    out.write_text("\n\n".join(tex), encoding="utf-8")
    return out


def build_final_markdown(stage_rows: list[dict[str, str]]) -> None:
    summary = {r["metric"]: r["value"] for r in read_csv_rows(FINAL_DIR / "tables" / "phase2_summary_metrics.csv")}
    perf = read_csv_rows(FINAL_DIR / "tables" / "performance_summary.csv")[0]
    md = f"""# Phase-II Robust Active-Learning Convergence, Numerical Reliability, and Optimization Report

## Executive Summary

Phase-II converts a sequence of small validation reports into one reproducible
story: starting from exact seed labels, the active-learning loop repeatedly
trained a surrogate, selected a full-acquisition batch, called the robust exact
oracle, appended trusted labels, and stopped only after the main phase map and
main thermodynamic boundaries stabilized.

Final frozen state:

```text
main_phase_converged = True
publication_boundary_audit = pass
publication_ready_for_main_phase_map = True
frozen dataset = dataset_iter035
total samples = 7434
normal / uniform_SC / FFLO = 1867 / 715 / 4852
trusted surprise = 0/127
hard-risk surprise = 75/129
targeted rerun count = 0
```

The hard-risk frontier remains active and must be shown as an uncertainty
layer.  It does not invalidate the main thermodynamic phase map under the
publication-grade boundary-impact audit.

## Random-Seed Active-Learning Loop

![RankCapK3 active-learning loop](figures/phase2_from_seed_loop.png)

The loop is intentionally asymmetric: acquisition proposes points, but only the
exact BdG oracle assigns labels.  The surrogate is a scheduler rather than a
physics replacement.

## Stage Timeline

![Optimization timeline](figures/phase2_optimization_timeline.png)

| Stage | Objective | Main result |
|---|---|---|
"""
    for row in stage_rows:
        md += f"| {row['stage']} | {row['objective']} | {row['main_result']} |\n"
    md += f"""

## Acquisition and Oracle Split

![Acquisition and oracle responsibility split](figures/phase2_acquisition_oracle_flow.png)

Acquisition evolved through B-delta gating, active-pool narrowing, stochastic
sampling, observation repulsion, and batch repulsion.  The final run used full
acquisition because it better retained uniform-SC and FFLO discovery coverage
than a simple phase-only boundary profile.

The robust oracle records phase labels separately from reliability metadata:
`trusted_exact`, `training_eligible_exact`, `rerun_required`, `q_unresolved`,
and `delta_unresolved`.  This separation is what allowed the final
trusted-surprise StopController gate to distinguish main phase-map convergence
from the numerical hard-risk frontier.

## Local-Refinement Optimization

The failed optimized variants were not physically disproven; they suffered a
target-construction bug where mandatory branches could overflow to roughly
85 local boxes.  The corrected `rankcap_k3` path performs basin-level risk
annotation and hard caps total refined basins at three.

```text
baseline local boxes = {perf['baseline_local_boxes']}
rankcap mean local boxes = {perf['rankcap_mean_local_boxes']}
rankcap max local boxes = {perf['rankcap_max_local_boxes']}
local-refinement runtime reduction = {float(perf['local_refinement_runtime_reduction_percent']):.2f}%
point-total runtime reduction = {float(perf['point_total_runtime_reduction_percent']):.2f}%
```

![Local-box distribution](figures/local_box_distribution.png)

## Full Loop and Tail Continuation

```text
phase_map_change = {summary['phase_map_change']}
normal/SC boundary shift = {summary['boundary_shift_normal_sc']}
uniform/FFLO boundary shift = {summary['boundary_shift_uniform_fflo']}
boundary_coverage_p95 = {summary['boundary_coverage_p95']}
```

![Dataset growth and phase counts](figures/dataset_growth_and_phase_counts.png)

![Surrogate metric curves](figures/surrogate_metric_curves.png)

![Runtime curve](figures/runtime_curve.png)

## Final Phase Map

![Clean final phase map](figures/final_phase_map_clean.png)

![Final phase map with uncertainty](figures/final_phase_map_with_uncertainty.png)

![Hard-risk uncertainty band](figures/hard_risk_uncertainty_band.png)

## Hard-Risk Boundary-Impact Audit

The publication audit reconstructs 129 hard-risk points.  Of these, 110 are
rerun-required and 19 are non-rerun hard-risk through untrusted or
training-ineligible metadata.  Counterfactual local flips, continuous clusters,
phase-constrained flips, Hausdorff diagnostics, affected arc length, and
topology checks do not identify a cluster that moves the main thermodynamic
boundaries beyond tolerance.  The targeted rerun list is empty.

![Tail surprise layers](figures/tail_surprise_layers.png)

## Final Decision

Freeze `dataset_iter035`, the `rankcap_k3` production oracle settings, and the
trusted-surprise StopController configuration for the Phase-II main phase-map
result.  Do not launch another full active-learning loop for this result.

Recommended next physics stage: branch-resolved topology classification,
hidden-ground-truth evaluation, multi-seed benchmarking, and final publication
figure polishing.

## Do-Not-Claim List

1. Do not claim the hard-risk frontier has disappeared.
2. Do not treat provisional hard-risk labels as definitive phase labels.
3. Do not claim eta anomalies have all become physical effects.
4. Do not treat topology reference curves as pointwise topology labels.
5. Do not present this single-seed run as a complete multi-seed benchmark.
6. Do not claim hidden-ground-truth benchmarking is complete.
7. Do not treat synchronized global stress-test islands as real boundary instability.
"""
    (FINAL_DIR / "phase2_robust_al_final_report.md").write_text(md, encoding="utf-8")
    (FINAL_DIR / "executive_summary.md").write_text(md.split("## Random-Seed Active-Learning Loop")[0], encoding="utf-8")


def build_final_tex(stage_rows: list[dict[str, str]]) -> Path:
    manifest = load_manifest()
    summary_rows = read_csv_rows(FINAL_DIR / "tables" / "phase2_summary_metrics.csv")
    perf_wide_rows = read_csv_rows(FINAL_DIR / "tables" / "performance_summary.csv")
    perf_rows = [{"metric": k, "value": v} for k, v in (perf_wide_rows[0] if perf_wide_rows else {}).items()]
    phase_counts = read_csv_rows(FINAL_DIR / "tables" / "final_dataset_phase_counts.csv")
    checks = read_csv_rows(FINAL_DIR / "tables" / "report_consistency_checks.csv")
    audit_gates = read_csv_rows(AUDIT_DIR / "tables" / "audit_gate_summary.csv")
    reason_counts = read_csv_rows(AUDIT_DIR / "tables" / "hard_risk_reason_counts.csv")
    summary = {r["metric"]: r["value"] for r in summary_rows}

    tex = [tex_preamble("Phase-II Robust Active-Learning Convergence, Numerical Reliability, and Optimization Report")]
    tex.append(r"\section{Executive Summary}")
    tex.append(
        key_value_table(
            [
                ("main_phase_converged", "True"),
                ("publication_boundary_audit", "pass"),
                ("publication_ready_for_main_phase_map", "True"),
                ("frozen dataset", "dataset_iter035"),
                ("dataset total", "7434"),
                ("normal / uniform-SC / FFLO", "1867 / 715 / 4852"),
                ("trusted surprise", "0/127"),
                ("hard-risk surprise", "75/129"),
                ("targeted rerun count", "0"),
            ],
            "Final Phase-II status.",
        )
    )
    tex.append(
        "Phase-II should be read as one continuous workflow: exact seed labels, "
        "surrogate training, full acquisition, robust exact-oracle labeling, "
        "trusted-label append, StopController review, and hard-risk publication "
        "audit.  The machine-learning model schedules exact BdG calls; it does "
        "not replace the thermodynamic phase criterion."
    )
    tex.append(r"\section{From Random Seed to Closed-Loop Active Learning}")
    tex.append(figure("figures/phase2_from_seed_loop.png", "RankCapK3 active-learning loop from random seed."))
    tex.append(
        "The final loop contains an initial exact seed set followed by "
        "acquisition-selected batches.  Every batch is merged, filtered through "
        "trusted-label criteria, appended to the dataset, and checked against "
        "main phase-map stability and boundary-shift criteria."
    )
    tex.append(r"\section{Optimization Timeline}")
    tex.append(figure("figures/phase2_optimization_timeline.png", "Phase-II optimization and validation timeline.", r"0.96\linewidth"))
    tex.append(simple_table(stage_rows, ["stage", "objective", "main_result"], "Stage-level objective and outcome.", [0.18, 0.30, 0.42]))
    tex.append(r"\section{Acquisition Function Development}")
    tex.append(
        "Acquisition development controlled three competing needs: discover new "
        "FFLO and uniform-SC regions, keep attention near main thermodynamic "
        "boundaries, and avoid wasting batches in deep normal interiors.  The "
        "final full profile retained boundary terms, uncertainty terms, "
        "stochastic sampling, observation repulsion, and batch repulsion."
    )
    tex.append(figure("figures/phase2_acquisition_oracle_flow.png", "Responsibility split between acquisition and exact oracle."))
    tex.append(r"\section{Exact-Oracle Reliability Evolution}")
    tex.append(
        "The robust oracle records phase labels separately from reliability "
        "metadata: trusted exact status, training eligibility, rerun-required "
        "status, q-unresolved status, and Delta-unresolved status.  This "
        "separation lets the report distinguish main phase convergence from "
        "the still-active numerical frontier."
    )
    tex.append(r"\section{Incremental q-Window and Local-Refinement Optimization}")
    tex.append(
        "Incremental q-window computation reduces repeated q-strip work without "
        "changing the thermodynamic phase criterion.  Local refinement was the "
        "larger bottleneck: failed cluster-optional variants overflowed to "
        "about 85 local boxes because mandatory risk targets bypassed the "
        "intended cap.  RankCapK3 fixed this by using basin-level risk "
        "annotation and a hard total cap of three refined basins."
    )
    tex.append(simple_table(perf_rows, ["metric", "value"], "Optimization performance summary.", [0.55, 0.35]))
    tex.append(figure("figures/local_box_distribution.png", "Local-box distribution under RankCapK3."))
    tex.append(figure("figures/local_box_gate.png", "Local-box gate and cap behavior."))
    tex.append(r"\section{Validation Ladder}")
    tex.append(
        "The optimized oracle was not promoted directly into a full run.  It "
        "passed fixed-point acceptance, one-iteration active-learning validation, "
        "five-iteration validation, a full-loop run, tail continuation, and the "
        "publication hard-risk boundary-impact audit."
    )
    tex.append(r"\section{Full-Loop and Tail-Continuation Results}")
    tex.append(simple_table(phase_counts, ["phase", "count"], "Frozen dataset_iter035 phase counts.", [0.45, 0.30]))
    tex.append(
        key_value_table(
            [
                ("phase_map_change", summary.get("phase_map_change", "")),
                ("normal/SC boundary shift", summary.get("boundary_shift_normal_sc", "")),
                ("uniform/FFLO boundary shift", summary.get("boundary_shift_uniform_fflo", "")),
                ("boundary_coverage_p95", summary.get("boundary_coverage_p95", "")),
                ("trusted surprise", "0/127"),
                ("all-selected surprise", "75/256"),
                ("hard-risk surprise", "75/129"),
            ],
            "Final StopController metrics.",
        )
    )
    tex.append(figure("figures/dataset_growth_and_phase_counts.png", "Dataset growth and phase counts."))
    tex.append(figure("figures/surrogate_metric_curves.png", "Surrogate metric curves."))
    tex.append(figure("figures/runtime_curve.png", "Runtime learning curve."))
    tex.append(r"\section{Final Phase Diagram}")
    tex.append(figure("figures/final_phase_map_clean.png", "Clean final main phase map with extracted boundaries."))
    tex.append(figure("figures/final_phase_map_with_uncertainty.png", "Final phase map with hard-risk uncertainty markers."))
    tex.append(figure("figures/final_dataset_coverage.png", "Final exact dataset coverage."))
    tex.append(figure("figures/hard_risk_uncertainty_band.png", "Hard-risk uncertainty band."))
    tex.append(r"\section{Convergence Logic}")
    tex.append(
        "Phase-map change and boundary-shift metrics measure stability of dense "
        "monitor-grid predictions between iterations.  Boundary coverage measures "
        "how densely exact samples cover the current main boundaries.  "
        "All-selected surprise is an acquisition-difficulty diagnostic, whereas "
        "trusted surprise is the clean exact-label gate for main phase convergence."
    )
    tex.append(figure("figures/tail_surprise_layers.png", "Trusted and hard-risk surprise layers at the tail endpoint."))
    tex.append(r"\section{Hard-Risk Boundary-Impact Audit}")
    tex.append(simple_table(reason_counts, ["reason", "count"], "Hard-risk reason counts.", [0.65, 0.25]))
    tex.append(simple_table(compact_gate_rows(audit_gates), ["gate", "status", "value", "threshold"], "Publication audit gate summary.", [0.31, 0.14, 0.16, 0.31]))
    tex.append(
        "The strict Hausdorff diagnostic reaches 0.8125 for isolated dense-grid "
        "uncertainty-marker fragments, but the significant local Hausdorff gate "
        "value is zero.  No meaningful main-boundary topology change is detected, "
        "and the targeted rerun list is empty."
    )
    tex.append(r"\section{Supporting Report Map}")
    inventory = read_csv_rows(FINAL_DIR / "tables" / "supporting_report_inventory.csv")
    inventory_display = [
        {"directory": Path(r["directory"]).name, "role": r["role"]}
        for r in inventory[:16]
    ]
    tex.append(simple_table(inventory_display, ["directory", "role"], "Main report and supporting-report map.", [0.48, 0.36]))
    tex.append(r"\section{Consistency and Reproduction}")
    tex.append(simple_table(checks, ["check", "expected", "actual", "status"], "Report consistency checks.", [0.28, 0.20, 0.24, 0.12]))
    tex.append(
        "The reproduction manifest records source run IDs, dataset paths and "
        "hashes, configuration, tolerance values, input report paths, and output hashes."
    )
    tex.append(
        key_value_table(
            [
                ("git commit", manifest.get("git_commit", "")),
                ("candidate grid hash", short_hash(manifest.get("array_hashes", {}).get("candidate_grid_hash", ""))),
                ("phase map hash", short_hash(manifest.get("array_hashes", {}).get("phase_map_hash", ""))),
            ],
            "Reproduction identifiers.",
        )
    )
    tex.append(r"\section{Do-Not-Claim List}")
    tex.append(
        r"\begin{enumerate}"
        r"\item Do not claim the hard-risk numerical frontier has disappeared."
        r"\item Do not treat provisional hard-risk labels as definitive phase labels."
        r"\item Do not claim eta anomalies have all become physical effects."
        r"\item Do not treat topology reference curves as pointwise topology labels."
        r"\item Do not present a single-seed run as a complete multi-seed benchmark."
        r"\item Do not claim hidden-ground-truth benchmarking is complete."
        r"\item Do not treat synchronized global stress-test islands as physical boundary instabilities."
        r"\end{enumerate}"
    )
    tex.append(r"\section{Final Decision and Next Stage}")
    tex.append(
        "Freeze \\texttt{dataset\\_iter035}, RankCapK3 production oracle settings, and the "
        "trusted-surprise StopController configuration for the Phase-II main "
        "phase-map result.  Do not launch another full active-learning loop for "
        "this result.  The next physics stages are branch-resolved topology "
        "classification, hidden-ground-truth evaluation, multi-seed benchmarking, "
        "and final publication figure polishing."
    )
    tex.append(tex_end())
    out = FINAL_DIR / "phase2_robust_al_final_report.tex"
    out.write_text("\n\n".join(tex), encoding="utf-8")
    return out


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "figures").mkdir(parents=True, exist_ok=True)
    write_report_indexes()
    build_overview_figures()
    stage_rows = stage_summary_rows()
    write_csv(FINAL_DIR / "tables" / "phase2_stage_summary.csv", stage_rows, ["stage", "objective", "main_result", "evidence"])
    build_final_markdown(stage_rows)
    audit_tex = build_audit_tex()
    final_tex = build_final_tex(stage_rows)
    print(json.dumps({"audit_tex": str(audit_tex), "final_tex": str(final_tex)}, indent=2))


if __name__ == "__main__":
    main()
