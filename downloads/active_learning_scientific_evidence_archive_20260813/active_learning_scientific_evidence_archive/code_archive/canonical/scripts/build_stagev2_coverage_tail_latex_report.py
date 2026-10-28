from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path("stagev2_topology_surface_coverage_tail_v1_hpc")
OUTPUT_ROOT = "ML_Phase_StageV_V2_TopologyCoverageTail"
RUN_ID = "stagev2_topology_surface_coverage_tail_v1"
AUDIT_ROOT = PACKAGE_ROOT / OUTPUT_ROOT / "reports" / "stagev2_tail_formal_audit_local"
REPORT_ROOT = PACKAGE_ROOT / OUTPUT_ROOT / "reports" / "stagev2_tail_latex_report_local"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def escape_tex(value: object) -> str:
    text = str(value)
    for old, new in {
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
    }.items():
        text = text.replace(old, new)
    return text


def fmt(value: object, digits: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def latex_table(rows: list[tuple[str, object]]) -> str:
    body = "\n".join(f"{escape_tex(k)} & {escape_tex(fmt(v))} \\\\" for k, v in rows)
    return (
        "\\begin{center}\n"
        "\\begin{tabular}{|p{0.43\\linewidth}|p{0.49\\linewidth}|}\n"
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


def copy_fig(src_name: str, dst_name: str, figures_dir: Path) -> str:
    src = AUDIT_ROOT / "figures" / src_name
    dst = figures_dir / dst_name
    shutil.copy2(src, dst)
    return f"figures/{dst.name}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    figures_dir = ensure_dir(REPORT_ROOT / "figures")
    tables_dir = ensure_dir(REPORT_ROOT / "tables")

    decision = read_json(AUDIT_ROOT / "stagev_v2_decision.json")
    phase = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_phase_convergence.csv")
    topo = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_topology_convergence.csv")
    surface = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_surface_shift_coverage.csv")
    surprise = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_surprise_by_iteration.csv")
    components = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_component_stability.csv")
    focus = pd.read_csv(AUDIT_ROOT / "tables" / "stagev_v2_final_focus_audit.csv")

    topology_surface = surface[surface["surface"].eq("topology")].copy()
    final_surface = topology_surface.iloc[-1].to_dict()
    first_cov = 0.036647239177640754
    final_cov = float(decision["topology_surface_coverage_p95_final"])
    coverage_ratio = final_cov / 0.00625
    coverage_delta = final_cov - first_cov
    coverage_rel = coverage_delta / first_cov

    key_rows = [
        {"metric": "production_loop_completed", "value": decision["production_loop_completed"]},
        {"metric": "final_dataset", "value": decision["final_dataset"]},
        {"metric": "final_exact_iteration", "value": decision["final_exact_iteration"]},
        {"metric": "total_samples", "value": decision["total_samples"]},
        {"metric": "trusted_exact", "value": decision["trusted_exact"]},
        {"metric": "decision_class", "value": decision["decision_class"]},
        {"metric": "stagev2_converged", "value": decision["stagev_v2_converged"]},
        {"metric": "main_limiting_factor", "value": decision["main_limiting_factor"]},
        {"metric": "topology_surface_coverage_p95_final", "value": final_cov},
        {"metric": "coverage_gate", "value": 0.00625},
        {"metric": "coverage_gate_ratio", "value": coverage_ratio},
        {"metric": "trusted_topology_surprise_last3", "value": decision["trusted_topology_surprise_last3"]},
        {"metric": "need_new_exact_calculation", "value": decision["need_new_exact_calculation"]},
        {"metric": "recommended_next_action", "value": decision["recommended_next_action"]},
    ]
    write_csv(tables_dir / "tail_key_decision_summary.csv", key_rows)
    phase.to_csv(tables_dir / "phase_convergence.csv", index=False)
    topo.to_csv(tables_dir / "topology_convergence.csv", index=False)
    topology_surface.to_csv(tables_dir / "topology_surface_coverage.csv", index=False)
    surprise.to_csv(tables_dir / "surprise_by_iteration.csv", index=False)
    components.to_csv(tables_dir / "component_stability.csv", index=False)
    focus.to_csv(tables_dir / "final_focus_audit.csv", index=False)

    fig = {
        "phase3d": copy_fig("fig01_final_3d_thermodynamic_phase_map.png", "final_3d_thermodynamic_phase_map.png", figures_dir),
        "topo3d": copy_fig("fig02_final_3d_fflo_topology_map.png", "final_3d_fflo_topology_map.png", figures_dir),
        "phase_slices": copy_fig("fig03_fixed_mu_thermodynamic_slice_atlas.png", "fixed_mu_thermodynamic_slice_atlas.png", figures_dir),
        "topo_slices": copy_fig("fig04_fixed_mu_topology_slice_atlas.png", "fixed_mu_topology_slice_atlas.png", figures_dir),
        "phase_change": copy_fig("fig05_phase_volume_map_change.png", "phase_volume_map_change.png", figures_dir),
        "topo_change": copy_fig("fig06_topology_volume_map_change.png", "topology_volume_map_change.png", figures_dir),
        "coverage": copy_fig("fig08_topology_surface_shift_coverage.png", "topology_surface_shift_coverage.png", figures_dir),
        "surprise": copy_fig("fig09_trusted_phase_topology_surprise.png", "trusted_phase_topology_surprise.png", figures_dir),
        "components": copy_fig("fig10_topology_component_stability.png", "topology_component_stability.png", figures_dir),
        "decision": copy_fig("fig19_final_decision_summary_panel.png", "decision_summary_panel.png", figures_dir),
    }

    phase_counts = decision["phase_counts"]
    topo_counts = decision["topology_counts"]

    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{array}}
\usepackage{{hyperref}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Stage V-v2 Topology-Surface Coverage Tail Audit}}
\author{{Fu-FFLO Active-Learning Project}}
\date{{Generated 2026-07-06}}

\begin{{document}}
\maketitle

\section*{{Executive Summary}}
The dedicated Stage V-v2 topology-surface coverage tail completed on the H100
cluster with no active Slurm jobs remaining at inspection time.  The tail
started from \texttt{{dataset\_iter097}} and completed exact iterations
097--100, producing \texttt{{dataset\_iter101}} with {decision['total_samples']}
unique samples and {decision['trusted_exact']} trusted exact points.

The tail did \emph{{not}} close the publication convergence gate.  The final
decision remains \textbf{{Decision {escape_tex(decision['decision_class'])}: not
converged}}.  The leading limiter is
\texttt{{{escape_tex(decision['main_limiting_factor'])}}}; secondary limiters are
\texttt{{{escape_tex(', '.join(decision.get('secondary_limiting_factors', [])))}}}.
The topology-surface coverage p95 improved only slightly from the pre-tail
audit value {first_cov:.6f} to {final_cov:.6f}, still {coverage_ratio:.1f} times
the 0.00625 coverage gate.  The last-three trusted topology surprise values are
{escape_tex(decision['trusted_topology_surprise_last3'])}, above the 0.02 gate.

\section*{{Run Completion}}
{latex_table([
        ("Remote package", "stagev2_topology_surface_coverage_tail_v1_hpc"),
        ("Run id", RUN_ID),
        ("Final dataset", decision["final_dataset"]),
        ("Final exact iteration", decision["final_exact_iteration"]),
        ("Production loop completed", decision["production_loop_completed"]),
        ("Downloaded local package", PACKAGE_ROOT.as_posix()),
    ])}

\section*{{Final Dataset Counts}}
{latex_table([
        ("Total samples", decision["total_samples"]),
        ("Trusted exact", decision["trusted_exact"]),
        ("Normal", phase_counts.get("normal", 0)),
        ("Uniform-SC", phase_counts.get("uniform_SC", 0)),
        ("FFLO", phase_counts.get("FFLO", 0)),
        ("cFFLO / trivial", topo_counts.get("cFFLO/trivial", 0)),
        ("tFFLO / topological", topo_counts.get("tFFLO/topological", 0)),
    ])}

{figure_block(fig["phase3d"], "Final 3D thermodynamic phase map after the topology-surface coverage tail.")}
{figure_block(fig["topo3d"], "Final trusted FFLO topology labels after the tail.")}
{figure_block(fig["phase_slices"], "Fixed-mu thermodynamic slice atlas.")}
{figure_block(fig["topo_slices"], "Fixed-mu topology slice atlas.")}

\section*{{Convergence And Coverage Gates}}
{latex_table([
        ("Phase map change last 3", decision["phase_map_change_last3"]),
        ("Topology map change last 3", decision["topology_map_change_last3"]),
        ("Topology surface p95 shift last 3", decision["topology_surface_shift_p95_last3"]),
        ("Topology component count last 3", decision["topology_component_count_last3"]),
        ("Trusted topology surprise last 3", decision["trusted_topology_surprise_last3"]),
        ("Final topology coverage median", final_surface.get("coverage_median")),
        ("Final topology coverage p95", final_surface.get("coverage_p95")),
        ("Coverage gate", 0.00625),
        ("Coverage p95 change from pre-tail", coverage_delta),
        ("Relative coverage p95 change", coverage_rel),
    ])}

The topology map gate fails because the 099--100 transition reaches
0.003135, above the 0.002 threshold.  The final 100--101 transition is below
threshold, but the formal rule evaluates the last three transitions.  The
topology surface shift proxy remains zero at p95, while direct coverage remains
too sparse.

{figure_block(fig["phase_change"], "Phase-volume map change during the coverage tail.")}
{figure_block(fig["topo_change"], "Topology-volume map change during the coverage tail.")}
{figure_block(fig["coverage"], "Topology-surface shift and coverage diagnostics.")}
{figure_block(fig["surprise"], "Trusted surprise diagnostics.")}
{figure_block(fig["components"], "Topology component stability proxy.")}

\section*{{Decision And Next Direction}}
{latex_table([
        ("Decision class", decision["decision_class"]),
        ("stagev2 converged", decision["stagev_v2_converged"]),
        ("Main limiting factor", decision["main_limiting_factor"]),
        ("Secondary limiting factors", ", ".join(decision.get("secondary_limiting_factors", []))),
        ("Need new exact calculation", decision["need_new_exact_calculation"]),
        ("Recommended next action", "localize sparse arcs; revise support-aware acquisition"),
        ("Recommended tail batches", decision["recommended_tail_batches"]),
        ("Do not continue v2 unchanged", decision["do_not_continue_v2_unchanged"]),
    ])}

The result argues against simply extending the same Stage V-v2 tail unchanged.
The next calculation should first localize the remaining topology map-change and
coverage-failure arcs, then use a revised support-aware acquisition or a smaller
targeted exact list.

{figure_block(fig["decision"], "Final decision summary panel for the coverage-tail audit.")}

\section*{{Companion Files}}
This report was generated from downloaded local artifacts only.  It did not run
new exact calculations, Delta-q searches, active-learning loops, or Slurm jobs.
Machine-readable companion files are in:

\begin{{itemize}}
\item \texttt{{tables/}}
\item \texttt{{figures/}}
\item \texttt{{decision\_log.md}}
\item upstream audit: \texttt{{stagev2\_tail\_formal\_audit\_local/}}
\end{{itemize}}

\end{{document}}
"""

    tex_path = REPORT_ROOT / "stagev2_coverage_tail_audit_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    (REPORT_ROOT / "stagev2_coverage_tail_audit_report.md").write_text(
        "\n".join(
            [
                "# Stage V-v2 Topology-Surface Coverage Tail Audit",
                "",
                f"- Final dataset: `{decision['final_dataset']}`",
                f"- Final exact iteration: `{decision['final_exact_iteration']}`",
                f"- Total samples: `{decision['total_samples']}`",
                f"- Trusted exact: `{decision['trusted_exact']}`",
                f"- Decision class: `{decision['decision_class']}`",
                f"- Main limiting factor: `{decision['main_limiting_factor']}`",
                f"- Coverage p95: `{final_cov}`",
                f"- Coverage gate: `0.00625`",
                f"- Trusted topology surprise last3: `{decision['trusted_topology_surprise_last3']}`",
                f"- Need new exact calculation: `{decision['need_new_exact_calculation']}`",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_ROOT / "decision_log.md").write_text(
        "\n".join(
            [
                "# Stage V-v2 Coverage Tail Decision Log",
                "",
                f"- Tail completed through exact iter `{decision['final_exact_iteration']}` and produced `{decision['final_dataset']}`.",
                f"- Decision class: `{decision['decision_class']}`.",
                f"- Converged: `{decision['stagev_v2_converged']}`.",
                f"- Main limiting factor: `{decision['main_limiting_factor']}`.",
                f"- Coverage p95: `{final_cov}` vs gate `0.00625`.",
                f"- Topology map change last3: `{decision['topology_map_change_last3']}`.",
                f"- Trusted topology surprise last3: `{decision['trusted_topology_surprise_last3']}`.",
                f"- Recommended next action: `{decision['recommended_next_action']}`.",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_root": str(REPORT_ROOT),
                "tex": str(tex_path),
                "pdf": str(REPORT_ROOT / "stagev2_coverage_tail_audit_report.pdf"),
                "md": str(REPORT_ROOT / "stagev2_coverage_tail_audit_report.md"),
                "tables": str(tables_dir),
                "figures": str(figures_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
