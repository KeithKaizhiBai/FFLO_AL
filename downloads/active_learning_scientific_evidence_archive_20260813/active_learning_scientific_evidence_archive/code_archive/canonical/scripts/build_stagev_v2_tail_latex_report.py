from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path("stagev_v2_multihead_boundary_learning_3d_v1_hpc")
REPORT_ROOT = (
    PACKAGE_ROOT
    / "ML_Phase_StageV_V2_Multihead"
    / "reports"
    / "stagev_v2_tail_latex_report_local"
)
RETURN_ROOT = PACKAGE_ROOT / "ML_Phase_StageV_V2_Multihead" / "reports" / "stagev_v2_return_report_local"
AUDIT_ROOT = PACKAGE_ROOT / "ML_Phase_StageV_V2_Multihead" / "reports" / "stagev_v2_formal_audit_local"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def escape_tex(value: object) -> str:
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


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def copy_fig(src: Path, dst_dir: Path, dst_name: str) -> str:
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    return f"figures/{dst.name}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_table(rows: list[tuple[str, object]], widths: str = r"p{0.42\linewidth}p{0.50\linewidth}") -> str:
    body = "\n".join(f"{escape_tex(k)} & {escape_tex(fmt(v))} \\\\" for k, v in rows)
    return (
        "\\begin{center}\n"
        f"\\begin{{tabular}}{{|{widths}|}}\n"
        "\\hline\n"
        "\\textbf{Quantity} & \\textbf{Value} \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{center}\n"
    )


def figure_block(path: str, caption: str) -> str:
    return (
        "\\begin{figure}[htbp]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.92\\linewidth]{{{path}}}\n"
        f"\\caption{{{escape_tex(caption)}}}\n"
        "\\end{figure}\n"
    )


def main() -> None:
    figures_dir = ensure_dir(REPORT_ROOT / "figures")
    tables_dir = ensure_dir(REPORT_ROOT / "tables")

    return_decision = read_json(RETURN_ROOT / "stagev_v2_return_decision.json")
    audit_decision = read_json(AUDIT_ROOT / "stagev_v2_decision.json")
    surface = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_surface_shift_coverage.csv")
    topo = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_topology_convergence.csv")
    phase = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_phase_convergence.csv")
    surprise = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_surprise_by_iteration.csv")
    learning = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_learning_heads_audit.csv")
    focus = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_final_focus_audit.csv")

    final_iteration = int(audit_decision["final_exact_iteration"])
    final_dataset = audit_decision["final_dataset"]
    surface_col = "boundary" if "boundary" in surface.columns else "surface"
    final_surface = surface[
        (surface["to_dataset_iteration"] == final_iteration + 1) & (surface[surface_col] == "topology")
    ]
    final_surface_row = final_surface.iloc[-1].to_dict()
    latest_learning = learning.iloc[-1].to_dict()

    key_rows = [
        {"metric": "production_loop_completed", "value": audit_decision["production_loop_completed"]},
        {"metric": "final_dataset", "value": final_dataset},
        {"metric": "final_exact_iteration", "value": final_iteration},
        {"metric": "total_samples", "value": audit_decision["total_samples"]},
        {"metric": "trusted_exact", "value": audit_decision["trusted_exact"]},
        {"metric": "decision_class", "value": audit_decision["decision_class"]},
        {"metric": "stagev_v2_converged", "value": audit_decision["stagev_v2_converged"]},
        {"metric": "main_limiting_factor", "value": audit_decision["main_limiting_factor"]},
        {"metric": "topology_surface_coverage_p95_final", "value": audit_decision["topology_surface_coverage_p95_final"]},
        {"metric": "coverage_threshold", "value": 0.00625},
        {"metric": "need_new_exact_calculation", "value": audit_decision["need_new_exact_calculation"]},
        {"metric": "recommended_next_action", "value": audit_decision["recommended_next_action"]},
    ]
    write_csv(tables_dir / "tail_key_decision_summary.csv", key_rows)
    phase.to_csv(tables_dir / "phase_convergence.csv", index=False)
    topo.to_csv(tables_dir / "topology_convergence.csv", index=False)
    surface.to_csv(tables_dir / "surface_shift_coverage.csv", index=False)
    surprise.to_csv(tables_dir / "surprise_by_iteration.csv", index=False)
    learning.tail(20).to_csv(tables_dir / "learning_heads_tail.csv", index=False)
    focus.to_csv(tables_dir / "final_focus_audit.csv", index=False)

    fig_paths = {
        "phase3d": copy_fig(RETURN_ROOT / "figures" / "final_phase_3d.png", figures_dir, "final_phase_3d.png"),
        "jakt": copy_fig(RETURN_ROOT / "figures" / "final_phase_jakt_projection.png", figures_dir, "final_phase_jakt_projection.png"),
        "slices": copy_fig(RETURN_ROOT / "figures" / "phase_mu_slice_atlas.png", figures_dir, "phase_mu_slice_atlas.png"),
        "topo3d": copy_fig(RETURN_ROOT / "figures" / "final_fflo_topology_3d.png", figures_dir, "final_fflo_topology_3d.png"),
        "topomap": copy_fig(AUDIT_ROOT / "figures" / "fig06_topology_volume_map_change.png", figures_dir, "topology_volume_map_change.png"),
        "coverage": copy_fig(AUDIT_ROOT / "figures" / "fig08_topology_surface_shift_coverage.png", figures_dir, "topology_surface_shift_coverage.png"),
        "surprise": copy_fig(AUDIT_ROOT / "figures" / "fig09_trusted_phase_topology_surprise.png", figures_dir, "trusted_phase_topology_surprise.png"),
        "learning": copy_fig(AUDIT_ROOT / "figures" / "fig14_reward_correlations.png", figures_dir, "reward_correlations.png"),
        "focus": copy_fig(AUDIT_ROOT / "figures" / "fig17_final10_selected_boundary_focus.png", figures_dir, "final10_selected_boundary_focus.png"),
        "decision": copy_fig(AUDIT_ROOT / "figures" / "fig19_final_decision_summary_panel.png", figures_dir, "decision_summary_panel.png"),
    }

    phase_counts = audit_decision["phase_counts"]
    topo_counts = audit_decision["topology_counts"]
    last3_topo_change = audit_decision["topology_map_change_last3"]
    last3_phase_change = audit_decision["phase_map_change_last3"]
    last3_shift = audit_decision["topology_surface_shift_p95_last3"]
    last3_surprise = audit_decision["trusted_topology_surprise_last3"]
    component_last3 = audit_decision["topology_component_count_last3"]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{xcolor}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Stage V-v2 Coverage-Tail Return Audit}}
\author{{Fu-FFLO Active-Learning Project}}
\date{{Generated 2026-07-06}}

\begin{{document}}
\maketitle

\section*{{Executive Summary}}
The latest downloaded Stage V-v2 tail completed successfully on the H100
cluster.  The remote log reports completion through exact iteration {final_iteration},
producing \texttt{{{escape_tex(final_dataset)}}}.  The Slurm queue was empty at
inspection time.

The final dataset contains {audit_decision['total_samples']} unique samples,
with {audit_decision['trusted_exact']} trusted exact points.  The phase counts
are normal = {phase_counts.get('normal', 0)}, uniform-SC = {phase_counts.get('uniform_SC', 0)},
and FFLO = {phase_counts.get('FFLO', 0)}.  Within trusted FFLO, the topology
counts are cFFLO/trivial = {topo_counts.get('cFFLO/trivial', 0)} and
tFFLO/topological = {topo_counts.get('tFFLO/topological', 0)}.

The tail stabilizes the map-level metrics: the last-three topology map changes
are {escape_tex(last3_topo_change)}, all below the 0.002 gate; the last-three
topology-surface p95 shifts are {escape_tex(last3_shift)}; trusted topology
surprise remains {escape_tex(last3_surprise)}.  However, the final topology
surface coverage p95 is {audit_decision['topology_surface_coverage_p95_final']:.6f},
well above the 0.00625 coverage gate.  The formal decision therefore remains
\textbf{{Decision {escape_tex(audit_decision['decision_class'])}: not converged}},
limited by \texttt{{{escape_tex(audit_decision['main_limiting_factor'])}}}.

\section*{{Run Completion And Provenance}}
{latex_table([
        ("Remote package", "stagev_v2_multihead_boundary_learning_3d_v1_hpc"),
        ("Run id", audit_decision["run_id"] if "run_id" in audit_decision else return_decision["run_id"]),
        ("Final dataset", final_dataset),
        ("Final exact iteration", final_iteration),
        ("Production loop completed", audit_decision["production_loop_completed"]),
        ("Local report root", REPORT_ROOT.name),
    ])}

The earlier planning note expected a possible \texttt{{dataset\_iter101}} tail.
The actual remote run stopped cleanly at \texttt{{dataset\_iter097}}, with the
log line stating completion through exact iteration 096.  This report audits
the actual completed result and does not infer missing iterations.

\section*{{Final Dataset}}
{latex_table([
        ("Total samples", audit_decision["total_samples"]),
        ("Trusted exact", audit_decision["trusted_exact"]),
        ("Normal", phase_counts.get("normal", 0)),
        ("Uniform-SC", phase_counts.get("uniform_SC", 0)),
        ("FFLO", phase_counts.get("FFLO", 0)),
        ("cFFLO / trivial", topo_counts.get("cFFLO/trivial", 0)),
        ("tFFLO / topological", topo_counts.get("tFFLO/topological", 0)),
    ])}

{figure_block(fig_paths["phase3d"], "Final 3D thermodynamic phase map from dataset_iter097.")}
{figure_block(fig_paths["jakt"], "JA-kT projection of the final 3D phase map, retained for comparison with earlier 2D studies.")}
{figure_block(fig_paths["slices"], "Fixed-mu slice atlas for the returned tail dataset.")}
{figure_block(fig_paths["topo3d"], "Trusted FFLO topology labels in the final dataset.")}

\section*{{Convergence Metrics}}
{latex_table([
        ("Phase map change last 3", last3_phase_change),
        ("Topology map change last 3", last3_topo_change),
        ("Topology surface p95 shift last 3", last3_shift),
        ("Trusted topology surprise last 3", last3_surprise),
        ("Topology component count last 3", component_last3),
        ("Final topology coverage median", final_surface_row.get("coverage_median")),
        ("Final topology coverage p95", final_surface_row.get("coverage_p95")),
        ("Coverage gate", 0.00625),
    ])}

The map-level and shift metrics are stable, but the topology surface remains
coverage limited.  The p95 exact-point coverage of the topology surface is
approximately {float(final_surface_row.get('coverage_p95')):.6f}, about
{float(final_surface_row.get('coverage_p95')) / 0.00625:.1f} times the current
coverage tolerance.

{figure_block(fig_paths["topomap"], "Topology volume map change across the final tail iterations.")}
{figure_block(fig_paths["coverage"], "Topology surface shift and coverage diagnostics.")}
{figure_block(fig_paths["surprise"], "Trusted phase/topology surprise metrics.")}

\section*{{Acquisition Learning Diagnostics}}
{latex_table([
        ("Latest reward iteration", int(latest_learning.get("iteration"))),
        ("lambda_p0", latest_learning.get("lambda_p0")),
        ("lambda_ppi", latest_learning.get("lambda_ppi")),
        ("lambda_ns", latest_learning.get("lambda_ns")),
        ("lambda_uf", latest_learning.get("lambda_uf")),
        ("rank_correlation_model_p0", latest_learning.get("rank_correlation_model_p0")),
        ("rank_correlation_model_ppi", latest_learning.get("rank_correlation_model_ppi")),
        ("rank_correlation_model_ns", latest_learning.get("rank_correlation_model_ns")),
        ("rank_correlation_model_uf", latest_learning.get("rank_correlation_model_uf")),
        ("gap head status", latest_learning.get("status_gap")),
    ])}

The multi-head reward model remains active for the normal-SC, uniform-FFLO,
P0, and Ppi channels.  The P0/Ppi heads continue to carry nonzero lambda
weights, but the final coverage gate indicates that the tail did not fully
resolve sparse topology-surface support.

{figure_block(fig_paths["learning"], "Reward-model rank correlations and learning-head diagnostics.")}
{figure_block(fig_paths["focus"], "Final-10-iteration selected-point focus audit.")}

\section*{{Decision}}
{latex_table([
        ("Decision class", audit_decision["decision_class"]),
        ("stagev_v2_converged", audit_decision["stagev_v2_converged"]),
        ("Main limiting factor", audit_decision["main_limiting_factor"]),
        ("Secondary limiting factors", ", ".join(audit_decision.get("secondary_limiting_factors", []))),
        ("Need new exact calculation", audit_decision["need_new_exact_calculation"]),
        ("Recommended next action", "targeted topology-surface coverage tail or v3 support fix"),
        ("Recommended tail batches", audit_decision.get("recommended_tail_batches")),
    ])}

The scientifically defensible conclusion is that Stage V-v2 is map-stable but
not publication-converged for the 3D cFFLO/tFFLO topology surface.  The next
calculation should be targeted at the remaining topology-surface coverage gaps,
not a blind restart of the full active-learning loop.

{figure_block(fig_paths["decision"], "Final formal audit decision summary panel.")}

\section*{{Generated Artifacts}}
This LaTeX report is accompanied by CSV tables under \texttt{{tables/}} and
copied PNG figures under \texttt{{figures/}}.  The upstream machine-readable
decision files remain:

\begin{{itemize}}
\item \texttt{{stagev\_v2\_return\_report\_local/stagev\_v2\_return\_decision.json}}
\item \texttt{{stagev\_v2\_formal\_audit\_local/stagev\_v2\_decision.json}}
\end{{itemize}}

\end{{document}}
"""

    tex_path = REPORT_ROOT / "stagev_v2_tail_latex_report.tex"
    tex_path.write_text(tex, encoding="utf-8")

    md_path = REPORT_ROOT / "stagev_v2_tail_latex_report.md"
    md_path.write_text(
        "\n".join(
            [
                "# Stage V-v2 Coverage-Tail Return Audit",
                "",
                f"- Final dataset: `{final_dataset}`",
                f"- Final exact iteration: `{final_iteration}`",
                f"- Total samples: `{audit_decision['total_samples']}`",
                f"- Trusted exact: `{audit_decision['trusted_exact']}`",
                f"- Decision class: `{audit_decision['decision_class']}`",
                f"- Main limiting factor: `{audit_decision['main_limiting_factor']}`",
                f"- Final topology surface coverage p95: `{audit_decision['topology_surface_coverage_p95_final']}`",
                f"- Need new exact calculation: `{audit_decision['need_new_exact_calculation']}`",
                f"- Recommended next action: `{audit_decision['recommended_next_action']}`",
                "",
                "The LaTeX source and PDF in this directory contain the full figure-based report.",
            ]
        ),
        encoding="utf-8",
    )

    (REPORT_ROOT / "decision_log.md").write_text(
        "\n".join(
            [
                "# Stage V-v2 Tail Decision Log",
                "",
                f"- Completion: production loop completed through exact iter `{final_iteration}`.",
                f"- Final dataset: `{final_dataset}`.",
                f"- Decision: `{audit_decision['decision_class']}` / converged = `{audit_decision['stagev_v2_converged']}`.",
                f"- Limiting factor: `{audit_decision['main_limiting_factor']}`.",
                f"- Coverage p95: `{audit_decision['topology_surface_coverage_p95_final']}` vs gate `0.00625`.",
                f"- Need new exact calculation: `{audit_decision['need_new_exact_calculation']}`.",
                f"- Recommended next action: `{audit_decision['recommended_next_action']}`.",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report_root": str(REPORT_ROOT),
                "tex": str(tex_path),
                "md": str(md_path),
                "tables": str(tables_dir),
                "figures": str(figures_dir),
                "decision": str(REPORT_ROOT / "decision_log.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
