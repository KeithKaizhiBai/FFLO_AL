from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "project_history" / "reports" / "report_second_stage_discussion"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def copy_asset(src: str, dst_name: str, caption: str) -> dict[str, str]:
    src_path = ROOT / src
    dst_path = FIG_DIR / dst_name
    if not src_path.exists():
        return {
            "figure": dst_name,
            "source": src,
            "status": "missing",
            "caption": caption,
        }
    shutil.copy2(src_path, dst_path)
    return {
        "figure": dst_name,
        "source": src,
        "status": "copied",
        "caption": caption,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_box(ax, xy: tuple[float, float], w: float, h: float, title: str, body: str, fc: str) -> None:
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        fc=fc,
        ec="#2f3440",
        lw=1.2,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.67, title, ha="center", va="center", fontsize=10, weight="bold")
    ax.text(xy[0] + w / 2, xy[1] + h * 0.34, body, ha="center", va="center", fontsize=8.2)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333333")
    ax.add_patch(arr)


def save_project_roadmap() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        ((0.02, 0.55), "Exact grid", "21528 warm-up\nBdG points", "#d9e8fb"),
        ((0.21, 0.55), "AL from exact data", "boundary-focused\nexact calls", "#e8f2df"),
        ((0.40, 0.55), "Numerical audits", "q-window, Delta,\nresponse checks", "#fff2cc"),
        ((0.59, 0.55), "Robust oracle", "label closure and\nbranch metadata", "#f9dfdc"),
        ((0.78, 0.55), "Rankcap K3", "target cap with\nphysics guards", "#e6e1f5"),
        ((0.21, 0.08), "From-scratch AL", "seed + acquisition\nbatches", "#e8f2df"),
        ((0.40, 0.08), "5-iter validation", "closed-loop sanity\nall gates pass", "#e8f2df"),
        ((0.59, 0.08), "Full loop", "6880 final samples\n2.79 boxes mean", "#e6e1f5"),
        ((0.78, 0.08), "Open issue", "formal convergence\nnot yet reached", "#f7e2c6"),
    ]
    for xy, title, body, color in boxes:
        draw_box(ax, xy, 0.15, 0.28, title, body, color)
    for i in range(4):
        arrow(ax, (0.17 + 0.19 * i, 0.69), (0.21 + 0.19 * i, 0.69))
    arrow(ax, (0.285, 0.55), (0.285, 0.36))
    arrow(ax, (0.36, 0.22), (0.40, 0.22))
    arrow(ax, (0.55, 0.22), (0.59, 0.22))
    arrow(ax, (0.74, 0.22), (0.78, 0.22))
    ax.text(0.5, 0.96, "Project route: from exact phase maps to optimized closed-loop active learning", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "discussion_project_roadmap.png", dpi=220)
    plt.close(fig)


def save_validation_funnel() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.axis("off")
    rows = [
        ("Variant-array failure", "85 local boxes on clean controls; timeout on hard-risk points", "fail", "#f8d7da"),
        ("Target-construction audit", "mandatory overflow confirmed; direct rerun rejected", "diagnosis", "#fff2cc"),
        ("Dry-run gate", "rank_and_cap_k3 selected targets <= 3", "pass", "#d9ead3"),
        ("Fixed-point acceptance", "32/32 physics-equivalent to baseline", "pass", "#d9ead3"),
        ("One-iteration AL", "single acquisition batch preserves label closure", "pass", "#d9ead3"),
        ("Five-iteration AL", "1792 exact points, max local boxes = 3", "pass", "#d9ead3"),
        ("Full-loop AL", "8192 exact calls, 6880 final samples, validation pass", "pass", "#d9ead3"),
        ("StopController", "max_iterations; label surprise and coverage still fail", "not converged", "#f7e2c6"),
    ]
    y = 0.86
    for stage, evidence, status, color in rows:
        draw_box(ax, (0.08, y - 0.06), 0.62, 0.09, stage, evidence, color)
        draw_box(ax, (0.75, y - 0.06), 0.16, 0.09, status, "", color)
        if y > 0.18:
            arrow(ax, (0.39, y - 0.065), (0.39, y - 0.115))
        y -= 0.105
    ax.text(0.5, 0.98, "Rank-and-cap K3 validation funnel", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "discussion_validation_funnel.png", dpi=220)
    plt.close(fig)


def save_oracle_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    boxes = [
        ((0.03, 0.58), "Coarse scan", r"$F_{\min}(q)$ and $\Delta^*(q)$", "#d9e8fb"),
        ((0.22, 0.58), "Branch candidates", "local minima and risk flags", "#fff2cc"),
        ((0.41, 0.58), "Basin clustering", "merge duplicate branches", "#e8f2df"),
        ((0.60, 0.58), "Rank-and-cap", "global best + top risk basins\nmax total = 3", "#e6e1f5"),
        ((0.79, 0.58), "Local boxes", "refine selected targets only", "#f9dfdc"),
        ((0.22, 0.14), "No physics change", r"same $\Omega(\Delta,q)$ and thresholds", "#f2f2f2"),
        ((0.50, 0.14), "Metadata", "overflow, q-edge, Delta guardrails", "#f2f2f2"),
    ]
    for xy, title, body, color in boxes:
        draw_box(ax, xy, 0.15, 0.23, title, body, color)
    for i in range(4):
        arrow(ax, (0.18 + 0.19 * i, 0.695), (0.22 + 0.19 * i, 0.695))
    arrow(ax, (0.675, 0.58), (0.575, 0.37))
    arrow(ax, (0.295, 0.37), (0.295, 0.58))
    ax.text(0.5, 0.96, "Where rank_and_cap_k3 changes the exact-oracle workflow", ha="center", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "discussion_exact_oracle_rankcap_flow.png", dpi=220)
    plt.close(fig)


def prepare_figures() -> list[dict[str, str]]:
    assets = [
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig18_fflo_altermagnetic_system_schematic.png", "system_schematic.png", "Conceptual FFLO + altermagnetic system schematic."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig01_original_exact_phase_diagram.png", "exact_warmup_phase_diagram.png", "Original exact warm-up phase diagram."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig02_active_learning_main_boundaries.png", "active_learning_main_boundaries.png", "Active-learning selected points near main boundaries."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig03_combined_eta_phase_diagram.png", "combined_eta_phase_diagram.png", "Combined eta and phase map after active learning."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig11_ml_training_architecture.png", "ml_training_architecture.png", "ML training and active-learning architecture."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig14_label_closed_selection_focus.png", "selection_focus_curve.png", "Selection-focus benchmark against random sampling."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig08_phase_change_map.png", "phase_qwindow_change_map.png", "High-JA phase changes after q-window and Delta audit."),
        ("project_history/reports/report_local_refinement_refactor_note/figures/fig17_free_energy_minimization_flow.png", "free_energy_minimization_flow_old.png", "Existing free-energy minimization flow diagram."),
        ("reports/local_refinement_target_logic_audit/figures/selected_target_count_by_variant.png", "target_count_by_variant.png", "Target explosion in failed optional variants."),
        ("reports/local_refinement_target_logic_audit/figures/target_count_pipeline_sankey_or_bar.png", "target_count_pipeline.png", "Target-construction pipeline counts."),
        ("rankcap_k3_full_loop/reports/local_refinement_rankcap_acceptance/figures/local_boxes_before_after.png", "rankcap_acceptance_local_boxes.png", "Fixed-point local-box count before and after rankcap."),
        ("rankcap_k3_full_loop/reports/local_refinement_rankcap_acceptance/figures/local_refinement_runtime_before_after.png", "rankcap_acceptance_runtime.png", "Fixed-point local-refinement runtime before and after rankcap."),
        ("reports/rankcap_k3_5iter_validation_recheck/figures/local_box_gate_by_iteration.png", "five_iter_local_box_gate.png", "Five-iteration corrected local-box gate."),
        ("reports/rankcap_k3_5iter_validation_recheck/figures/dataset_phase_counts_recheck.png", "five_iter_dataset_phase_counts.png", "Five-iteration dataset phase growth."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_final_phase_diagram.png", "full_loop_final_phase_diagram.png", "Full-loop final exact phase diagram."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_phase_snapshots.png", "full_loop_phase_snapshots.png", "Phase diagram evolution snapshots."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_learning_curve_phase_counts.png", "full_loop_learning_curve_phase_counts.png", "Full-loop dataset growth and phase counts."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_surrogate_metric_curves.png", "full_loop_surrogate_metric_curves.png", "Full-loop surrogate metric curves."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_corrected_local_box_gate.png", "full_loop_corrected_local_box_gate.png", "Full-loop corrected local-box gate."),
        ("rankcap_k3_full_loop/reports/rankcap_k3_full_loop/figures/enhanced_runtime_curve.png", "full_loop_runtime_curve.png", "Full-loop runtime curve."),
        ("rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/last5_selection_stop_audit/figures/last5_failed_stop_metrics.png", "last5_failed_stop_metrics.png", "Last-five failed StopController metrics."),
        ("rankcap_k3_full_loop/ML_Phase_512_RankCapK3_FullLoop/reports/last5_selection_stop_audit/figures/last5_selected_points_map.png", "last5_selected_points_map.png", "Last-five selected points in parameter space."),
    ]
    rows = [copy_asset(*asset) for asset in assets]
    save_project_roadmap()
    save_validation_funnel()
    save_oracle_flow()
    rows.extend(
        [
            {"figure": "discussion_project_roadmap.png", "source": "generated", "status": "generated", "caption": "Project roadmap flow diagram."},
            {"figure": "discussion_validation_funnel.png", "source": "generated", "status": "generated", "caption": "Rankcap K3 validation funnel."},
            {"figure": "discussion_exact_oracle_rankcap_flow.png", "source": "generated", "status": "generated", "caption": "Exact-oracle rankcap insertion point."},
        ]
    )
    write_csv(TABLE_DIR / "figure_manifest.csv", rows)
    return rows


def write_tables() -> None:
    milestones = [
        {"stage": "0", "milestone": "Exact warm-up phase map", "goal": "Build a trustworthy global phase baseline", "evidence": "21528 exact grid points; normal/uniform_SC/FFLO boundaries visible", "status": "complete"},
        {"stage": "1", "milestone": "Active learning from exact data", "goal": "Use ML only to schedule new exact calls", "evidence": "24083-point r=0.015 report; selected points concentrate near boundaries", "status": "complete"},
        {"stage": "2", "milestone": "From-scratch AL workflow", "goal": "Demonstrate seed plus acquisition batches without relying on a dense warm start", "evidence": "rankcap_k3 one-iteration and five-iteration validations", "status": "complete"},
        {"stage": "3", "milestone": "Numerical reliability audits", "goal": "Separate phase boundaries from response artifacts and q-window errors", "evidence": "q-window, Delta, response, and label-closure reports", "status": "complete"},
        {"stage": "4", "milestone": "Target-construction audit", "goal": "Explain why optional variants timed out", "evidence": "local-box count about 85 confirmed as refined boxes; mandatory overflow confirmed", "status": "complete"},
        {"stage": "5", "milestone": "Rank-and-cap K3", "goal": "Cap local-refinement targets without changing physics criteria", "evidence": "selected targets at most 3; fixed-point physics equivalence", "status": "complete"},
        {"stage": "6", "milestone": "Closed-loop validation", "goal": "Check repeated train/select/exact/append behavior", "evidence": "5-iteration validation pass; 1792 exact points; max local boxes = 3", "status": "complete"},
        {"stage": "7", "milestone": "Full-loop validation", "goal": "Run full active-loop with optimized oracle", "evidence": "31 exact iterations; 6880 final samples; corrected validation pass", "status": "complete"},
        {"stage": "8", "milestone": "Formal convergence diagnosis", "goal": "Determine why StopController did not converge", "evidence": "last-five audit: label surprise and coverage fail", "status": "diagnosed"},
    ]
    validation = [
        {"gate": "target_logic_audit", "result": "confirmed failure mode", "evidence": "mandatory branches bypassed old caps; K only limited ordinary optional rows"},
        {"gate": "target_construction_dryrun", "result": "pass", "evidence": "rank_and_cap_k3 selected targets controlled to at most 3"},
        {"gate": "fixed_point_acceptance", "result": "pass", "evidence": "32/32 fixed points complete; labels/trust/training eligibility match baseline"},
        {"gate": "one_iteration_al", "result": "pass", "evidence": "one acquisition batch complete; label closure healthy"},
        {"gate": "five_iteration_al", "result": "pass after rank-level recheck", "evidence": "1792 exact points; max local boxes = 3; all phase classes present"},
        {"gate": "full_loop_al", "result": "pass after rank-level recheck", "evidence": "8192 exact calls; 6880 final samples; max local boxes = 3"},
        {"gate": "formal_stop_convergence", "result": "not passed", "evidence": "stop_reason=max_iterations; label_surprise_rate=0.1836; coverage p95=0.006588"},
    ]
    speedup = [
        {"metric": "mean local boxes", "baseline": "6.0", "rankcap_k3": "2.79297", "change": "-53.45%, 2.15x"},
        {"metric": "mean local-refinement runtime sec/point", "baseline": "189.767", "rankcap_k3": "88.2856", "change": "-53.48%, 2.15x"},
        {"metric": "mean point-total runtime sec/point", "baseline": "234.194", "rankcap_k3": "117.285", "change": "-49.92%, about 2.00x"},
        {"metric": "corrected max local boxes", "baseline": "6", "rankcap_k3": "3", "change": "hard cap respected"},
    ]
    final_state = [
        {"item": "exact iterations", "value": "31", "interpretation": "iter000 seed plus 30 acquisition batches"},
        {"item": "final dataset samples", "value": "6880", "interpretation": "normal=1777, uniform_SC=715, FFLO=4388"},
        {"item": "corrected validation status", "value": "pass", "interpretation": "rank-level local-box aggregation fixes false collector failure"},
        {"item": "formal convergence", "value": "false", "interpretation": "StopController stopped at max_iterations"},
        {"item": "label surprise", "value": "0.18359375", "interpretation": "above 0.05 threshold; predicted normal -> exact FFLO dominates"},
        {"item": "boundary coverage p95", "value": "0.006588078458684216", "interpretation": "slightly above 0.00625 threshold"},
    ]
    write_csv(TABLE_DIR / "milestone_summary.csv", milestones)
    write_csv(TABLE_DIR / "validation_summary.csv", validation)
    write_csv(TABLE_DIR / "speedup_summary.csv", speedup)
    write_csv(TABLE_DIR / "final_state_summary.csv", final_state)


def md_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(out)


def read_csv_rows(name: str) -> list[dict[str, str]]:
    path = TABLE_DIR / name
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_markdown() -> None:
    milestones = read_csv_rows("milestone_summary.csv")
    validation = read_csv_rows("validation_summary.csv")
    speedup = read_csv_rows("speedup_summary.csv")
    final_state = read_csv_rows("final_state_summary.csv")
    text = f"""# Second-Stage Discussion Report

Date: 2026-06-17

## Executive Summary

This report organizes the project history from the first exact BdG phase map to
the optimized rank_and_cap_k3 active-learning full loop.  The central message is
that the workflow has moved from phase-map discovery to a validated
cost-controlled exact oracle.  The rankcap optimization is accepted as an oracle
cost improvement, while formal active-learning convergence remains an open
late-stage selection / stopping-metric issue.

Key results:

- exact warm-up grid established the global normal, uniform-SC, and FFLO phase structure;
- active learning was introduced as a scheduler for exact BdG calls, not as a replacement for the exact oracle;
- q-window, Delta, response, and label-closure audits corrected numerical interpretation without changing the phase rule;
- target-construction audit identified mandatory overflow and explained why early optimized variants timed out;
- rank_and_cap_k3 capped selected local-refinement targets to three while preserving fixed-point physics labels;
- one-iteration, five-iteration, and full-loop validations passed after rank-level report rechecks;
- full-loop local-box mean dropped from 6.0 to 2.79297, and mean local-refinement runtime dropped by about 53.48%;
- the full loop did not formally converge because label surprise and boundary coverage still failed the StopController criteria.

## Project Roadmap

![Project roadmap](figures/discussion_project_roadmap.png)

The report follows a target-completion narrative.  Each stage either
establishes a scientific baseline, fixes a numerical ambiguity, or reduces the
cost of trusted exact labels.

{md_table(milestones)}

## Physical System and Exact Labels

![FFLO altermagnetic system](figures/system_schematic.png)

The physical model is a one-dimensional altermagnetic FFLO BdG system.  The
trusted labels come from minimizing the exact finite-temperature free energy
\\(\\Omega(\\Delta,q;T,J_A)\\), not from the surrogate model.  The phase rule
remains fixed throughout:

```text
normal:     Delta_opt < 1e-3
uniform_SC: Delta_opt >= 1e-3 and abs(q_opt) < 1e-2
FFLO:       Delta_opt >= 1e-3 and abs(q_opt) >= 1e-2
```

No stage in this report changes these thresholds.

## From Exact Grid to Active Learning

![Exact warm-up phase diagram](figures/exact_warmup_phase_diagram.png)

The exact warm-up grid supplied the first global map.  It also created the
training data that made active learning possible.

![Boundary-focused active learning](figures/active_learning_main_boundaries.png)

![Combined eta phase diagram](figures/combined_eta_phase_diagram.png)

Active learning then concentrated expensive exact calls near thermodynamic
boundaries.  This established the basic workflow: ML proposes where to compute,
but exact BdG evaluation decides what is accepted.

![ML training architecture](figures/ml_training_architecture.png)

![Selection focus curve](figures/selection_focus_curve.png)

The selection-focus curve made boundary targeting quantitative.  This was the
first step toward judging acquisition policies by evidence rather than by visual
scatter alone.

## Numerical Audits and the Move to a Robust Oracle

![Phase q-window change map](figures/phase_qwindow_change_map.png)

The q-window and Delta audits showed that some high-JA, low-T labels were
limited by numerical coverage rather than by a new physical criterion.  Expanded
q windows found lower FFLO branches for points that had previously appeared
normal.  Response-function eta claims were therefore separated from the main
phase-boundary stop criterion unless separately validated.

## Exact-Oracle Bottleneck

![Old free-energy minimization flow](figures/free_energy_minimization_flow_old.png)

The robust exact oracle became more reliable but also more expensive: q-window
expansion, Delta refinement, and branch checks increased the number of local
boxes evaluated per point.  The local-refinement refactor therefore targeted
duplicate or excessive local-box refinement while preserving the free-energy
criterion.

![Rankcap insertion flow](figures/discussion_exact_oracle_rankcap_flow.png)

The rank_and_cap_k3 policy acts after coarse candidate detection, risk
annotation, and clustering, but before local-box scans.  It caps selected
refinement basins without changing \\(\\Omega(\\Delta,q)\\), the normal
reference, or the phase thresholds.

## Target Explosion and Rank-and-Cap

![Target count by variant](figures/target_count_by_variant.png)

![Target construction pipeline](figures/target_count_pipeline.png)

The failed optional variants revealed the core bug: top-k caps limited ordinary
optional branches but did not control mandatory overflow.  Completed clean
controls refined about 85 boxes in optimized variants that were supposed to be
cheaper.  The correct response was not to rerun the timeout tasks, but to audit
target construction and introduce a rank-and-cap overflow policy.

![Validation funnel](figures/discussion_validation_funnel.png)

{md_table(validation)}

## Fixed-Point and Closed-Loop Validation

![Rankcap fixed-point local boxes](figures/rankcap_acceptance_local_boxes.png)

![Rankcap fixed-point runtime](figures/rankcap_acceptance_runtime.png)

Fixed-point acceptance verified that rank_and_cap_k3 reduced local boxes while
preserving labels, trust flags, training eligibility, and unresolved flags
against the baseline.

![Five-iteration local-box gate](figures/five_iter_local_box_gate.png)

![Five-iteration dataset phase counts](figures/five_iter_dataset_phase_counts.png)

The five-iteration closed-loop validation then checked repeated train, select,
exact, merge, and append behavior.  After correcting the rank-local point-id
aggregation in the report recheck, the run passed all gates with 1792 actual
exact points and a true max of three local boxes.

## Full-Loop Result

![Full-loop final phase diagram](figures/full_loop_final_phase_diagram.png)

![Full-loop phase snapshots](figures/full_loop_phase_snapshots.png)

The full loop completed 31 exact iterations: one seed iteration plus 30
acquisition-selected batches.  The final dataset contains 6880 exact samples:
1777 normal, 715 uniform_SC, and 4388 FFLO.

![Full-loop learning curve](figures/full_loop_learning_curve_phase_counts.png)

![Full-loop surrogate metrics](figures/full_loop_surrogate_metric_curves.png)

The returned package stores surrogate validation metrics rather than raw
per-epoch training loss.  The plotted curves should therefore be discussed as
held-out surrogate quality diagnostics, not optimizer loss curves.

![Full-loop corrected local-box gate](figures/full_loop_corrected_local_box_gate.png)

![Full-loop runtime curve](figures/full_loop_runtime_curve.png)

{md_table(speedup)}

The key operational result is a roughly twofold reduction in per-point exact
cost while maintaining the robust-oracle safety checks.

## Formal Convergence Status

{md_table(final_state)}

The full loop is a successful optimized-oracle validation, but not a formal
StopController convergence result.  The final stop state was
`stop_reason=max_iterations`, with three of four required conditions passed.
The stable phase-map and boundary-shift checks passed; label surprise and
boundary coverage did not.

![Last-five failed stop metrics](figures/last5_failed_stop_metrics.png)

![Last-five selected points map](figures/last5_selected_points_map.png)

The last-five audit shows that late-stage acquisition still selected many
q-edge-sensitive and rerun-required points.  The dominant label-surprise trace
was predicted normal before exact evaluation but exact FFLO after evaluating
\\(\\Delta_{{\\mathrm{{opt}}}}\\) and \\(q_{{\\mathrm{{opt}}}}\\).

## Discussion Points

1. The physics phase criterion is now better protected than at the start of the project.
2. The exact-oracle cost bottleneck was reduced without changing the criterion.
3. The remaining convergence failure is not evidence that rankcap_k3 breaks labels.
4. A future cleanup run, if needed, should target late-stage label surprise and boundary coverage explicitly.
5. The report collector's local-box aggregation key should be patched before future packages rely on its raw validation status.

## Companion Files

```text
second_stage_discussion_report.md
second_stage_discussion_report.tex
second_stage_discussion_report.pdf
decision_log.md
tables/milestone_summary.csv
tables/validation_summary.csv
tables/speedup_summary.csv
tables/final_state_summary.csv
tables/figure_manifest.csv
figures/*.png
```
"""
    (REPORT_DIR / "second_stage_discussion_report.md").write_text(text, encoding="utf-8")


def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def latex_table(rows: list[dict[str, str]], headers: list[str], widths: list[str]) -> str:
    spec = " ".join([f"p{{{w}}}" for w in widths])
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule"]
    lines.append(" & ".join(escape_latex(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(escape_latex(row[h]) for h in headers) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def fig_tex(name: str, caption: str, width: str = "0.92\\linewidth") -> str:
    safe_caption = escape_latex(caption)
    return rf"""\begin{{figure}}[H]
\centering
\includegraphics[width={width}]{{figures/{name}}}
\caption{{{safe_caption}}}
\end{{figure}}
"""


def write_latex() -> None:
    milestones = read_csv_rows("milestone_summary.csv")
    validation = read_csv_rows("validation_summary.csv")
    speedup = read_csv_rows("speedup_summary.csv")
    final_state = read_csv_rows("final_state_summary.csv")
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.78in]{{geometry}}
\usepackage{{amsmath,amssymb,bm}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\usepackage{{array}}
\usepackage{{caption}}
\usepackage{{longtable}}
\usepackage{{enumitem}}
\hypersetup{{colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!45!black}}
\graphicspath{{{{figures/}}}}
\setlength{{\parskip}}{{0.45em}}
\setlength{{\parindent}}{{0pt}}
\title{{From Exact Phase Maps to Cost-Controlled Active Learning\\A Second-Stage Discussion Report}}
\author{{Research Note}}
\date{{2026-06-17}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This discussion report summarizes the project route from exact BdG phase-map
calculation to active-learning refinement and the rank\_and\_cap\_k3
local-refinement optimization.  The exact oracle remains the source of physical
labels.  The machine-learning model is used as a scheduler for exact calls, not
as a replacement for the free-energy minimization.  The rankcap optimization
has passed fixed-point, one-iteration, five-iteration, and full-loop validation
after rank-level report rechecks.  The full loop did not formally converge
because late-stage label surprise and boundary coverage still failed the
StopController criteria.
\end{{abstract}}
\tableofcontents
\clearpage

\section{{Executive Summary}}
The workflow has advanced from phase-map discovery to a validated
cost-controlled exact oracle.  Rank\_and\_cap\_k3 should be described as an
accepted exact-oracle cost optimization, while formal active-learning
convergence remains open.

\begin{{itemize}}[leftmargin=*]
\item The exact warm-up grid established the normal, uniform-SC, and FFLO phase structure.
\item Active learning was introduced only as a scheduler for exact BdG calls.
\item q-window, Delta, response, and label-closure audits corrected numerical interpretation without changing the phase rule.
\item Target-construction audit identified mandatory overflow and explained the failed optimized variants.
\item Rank\_and\_cap\_k3 caps selected local-refinement basins to three.
\item Full-loop corrected validation passed with 6880 final samples.
\item Formal StopController convergence did not pass because label surprise and boundary coverage remained above tolerance.
\end{{itemize}}

{fig_tex("discussion_project_roadmap.png", "Project route from exact phase maps to optimized closed-loop active learning.", "0.98\\linewidth")}

\section{{Milestone-Oriented Project Narrative}}
{{\small
{latex_table(milestones, ["stage", "milestone", "goal", "evidence", "status"], ["0.07\\linewidth", "0.19\\linewidth", "0.25\\linewidth", "0.34\\linewidth", "0.11\\linewidth"])}
}}

\section{{Physical System and Fixed Phase Criterion}}
{fig_tex("system_schematic.png", "Conceptual one-dimensional altermagnetic FFLO BdG system.", "0.82\\linewidth")}

The trusted labels come from minimizing the exact finite-temperature free
energy \(\Omega(\Delta,q;T,J_A)\), not from the surrogate.  The phase rule is
unchanged throughout the work:
\[
\begin{{aligned}}
\mathrm{{normal}} &: \Delta_{{\mathrm{{opt}}}} < 10^{{-3}},\\
\mathrm{{uniform\_SC}} &: \Delta_{{\mathrm{{opt}}}} \ge 10^{{-3}}\ \mathrm{{and}}\ |q_{{\mathrm{{opt}}}}| < 10^{{-2}},\\
\mathrm{{FFLO}} &: \Delta_{{\mathrm{{opt}}}} \ge 10^{{-3}}\ \mathrm{{and}}\ |q_{{\mathrm{{opt}}}}| \ge 10^{{-2}}.
\end{{aligned}}
\]

\section{{From Exact Grid to Active Learning}}
{fig_tex("exact_warmup_phase_diagram.png", "Original exact warm-up phase diagram.", "0.9\\linewidth")}
{fig_tex("active_learning_main_boundaries.png", "Active-learning selected points concentrate near main phase boundaries.", "0.9\\linewidth")}
{fig_tex("combined_eta_phase_diagram.png", "Combined eta and phase map after active learning.", "0.9\\linewidth")}
{fig_tex("ml_training_architecture.png", "Surrogate-training and active-learning architecture.", "0.86\\linewidth")}
{fig_tex("selection_focus_curve.png", "Selection-focus benchmark against random sampling.", "0.82\\linewidth")}

\section{{Numerical Audits and Robust Oracle}}
{fig_tex("phase_qwindow_change_map.png", "High-JA phase changes after q-window and Delta audits.", "0.86\\linewidth")}

The q-window and Delta audits showed that some high-\(J_A\), low-\(T\) labels
were limited by numerical coverage.  The response-level eta anomalies were
therefore kept separate from the main thermodynamic stop criterion unless
separately validated.

\section{{Exact-Oracle Bottleneck and Rank-and-Cap}}
{fig_tex("free_energy_minimization_flow_old.png", "Original free-energy minimization and local-refinement target path.", "0.86\\linewidth")}
{fig_tex("discussion_exact_oracle_rankcap_flow.png", "Where rank_and_cap_k3 changes the exact-oracle workflow.", "0.98\\linewidth")}

Rank\_and\_cap\_k3 acts after candidate construction, risk annotation, and
clustering, but before expensive local-box scans.  It does not change
\(\Omega(\Delta,q)\), the normal reference, or phase thresholds.

{fig_tex("target_count_by_variant.png", "Target explosion in earlier optimized variants.", "0.84\\linewidth")}
{fig_tex("target_count_pipeline.png", "Target-construction pipeline counts from the audit.", "0.84\\linewidth")}
{fig_tex("discussion_validation_funnel.png", "Validation funnel for rank_and_cap_k3.", "0.92\\linewidth")}

{{\small
{latex_table(validation, ["gate", "result", "evidence"], ["0.25\\linewidth", "0.22\\linewidth", "0.48\\linewidth"])}
}}

\section{{Fixed-Point and Closed-Loop Validation}}
{fig_tex("rankcap_acceptance_local_boxes.png", "Fixed-point local-box counts before and after rankcap.", "0.82\\linewidth")}
{fig_tex("rankcap_acceptance_runtime.png", "Fixed-point local-refinement runtime before and after rankcap.", "0.82\\linewidth")}
{fig_tex("five_iter_local_box_gate.png", "Five-iteration corrected local-box gate.", "0.82\\linewidth")}
{fig_tex("five_iter_dataset_phase_counts.png", "Five-iteration dataset phase growth.", "0.82\\linewidth")}

\section{{Full-Loop Result}}
{fig_tex("full_loop_final_phase_diagram.png", "Full-loop final exact phase diagram.", "0.88\\linewidth")}
{fig_tex("full_loop_phase_snapshots.png", "Full-loop phase diagram evolution snapshots.", "0.98\\linewidth")}
{fig_tex("full_loop_learning_curve_phase_counts.png", "Full-loop dataset growth and phase counts.", "0.88\\linewidth")}
{fig_tex("full_loop_surrogate_metric_curves.png", "Full-loop surrogate validation metrics.  These are not raw optimizer loss curves.", "0.88\\linewidth")}
{fig_tex("full_loop_corrected_local_box_gate.png", "Full-loop corrected local-box gate.", "0.88\\linewidth")}
{fig_tex("full_loop_runtime_curve.png", "Full-loop runtime curve.", "0.88\\linewidth")}

{{\small
{latex_table(speedup, ["metric", "baseline", "rankcap_k3", "change"], ["0.34\\linewidth", "0.18\\linewidth", "0.18\\linewidth", "0.24\\linewidth"])}
}}

\section{{Formal Convergence Status}}
{{\small
{latex_table(final_state, ["item", "value", "interpretation"], ["0.28\\linewidth", "0.22\\linewidth", "0.45\\linewidth"])}
}}

{fig_tex("last5_failed_stop_metrics.png", "Last-five failed StopController metrics.", "0.84\\linewidth")}
{fig_tex("last5_selected_points_map.png", "Last-five selected points in parameter space.", "0.84\\linewidth")}

The full loop is a successful optimized-oracle validation, not a formal
StopController convergence result.  The final stop state was
\(\mathrm{{stop\_reason}}=\mathrm{{max\_iterations}}\).  The stable phase-map
and boundary-shift checks passed, while label surprise and boundary coverage
did not.  The last-five audit shows that late-stage acquisition continued to
select q-edge-sensitive and rerun-required points, with predicted-normal to
exact-FFLO mismatches dominating the surprise rate.

\section{{Discussion Points}}
\begin{{enumerate}}[leftmargin=*]
\item The physics phase criterion is now better protected than at the beginning of the project.
\item The exact-oracle cost bottleneck has been reduced without changing the physical criterion.
\item Rankcap non-convergence should not be confused with rankcap oracle failure.
\item A future cleanup run should target late-stage label surprise and boundary coverage explicitly.
\item The report collector's local-box aggregation key should be patched before future packages rely on its raw validation status.
\end{{enumerate}}

\section{{Companion Files}}
\begin{{verbatim}}
second_stage_discussion_report.md
second_stage_discussion_report.tex
second_stage_discussion_report.pdf
decision_log.md
tables/milestone_summary.csv
tables/validation_summary.csv
tables/speedup_summary.csv
tables/final_state_summary.csv
tables/figure_manifest.csv
figures/*.png
\end{{verbatim}}

\end{{document}}
"""
    (REPORT_DIR / "second_stage_discussion_report.tex").write_text(tex, encoding="utf-8")


def write_decision_log() -> None:
    text = """# Decision Log

Date: 2026-06-17

Decision-level conclusion:

```text
Use the new second-stage discussion report as the high-level narrative for
project discussion.  It supersedes the older local-refinement note for current
status, but it does not replace the underlying detailed reports.
```

Main conclusions:

```text
rank_and_cap_k3 is accepted as a cost-controlled robust-oracle optimization.
It passed fixed-point, one-iteration, five-iteration, and full-loop validation
after rank-level local-box rechecks.

The full-loop active-learning run did not formally converge.  The remaining
failure is attributed to late-stage label surprise and boundary coverage, not
to local-refinement target explosion.
```

Do not claim:

```text
Do not claim formal StopController convergence.
Do not claim raw training loss curves exist in the returned package.
Do not claim rankcap changes the thermodynamic phase criterion.
Do not use the old uncorrected local_box_rows aggregation as a final gate.
```

Next recommended action:

```text
Use this report for discussion of the completed second-stage workflow, then
decide separately whether a late-stage cleanup validation is needed for formal
convergence.
```
"""
    (REPORT_DIR / "decision_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    prepare_figures()
    write_tables()
    write_markdown()
    write_latex()
    write_decision_log()


if __name__ == "__main__":
    main()
