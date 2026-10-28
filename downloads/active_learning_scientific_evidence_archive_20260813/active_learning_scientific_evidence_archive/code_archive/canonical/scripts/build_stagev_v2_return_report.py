from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PHASE_COLORS = {
    "normal": "#8f8f8f",
    "uniform_SC": "#2f7ed8",
    "FFLO": "#d62728",
}

TOPO_COLORS = {
    "cFFLO/trivial": "#2f7ed8",
    "tFFLO/topological": "#d62728",
    "unresolved/nonapplicable": "#aaaaaa",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_blocks(paths: list[Path]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        active: list[str] = []
        depth = 0
        for line in lines:
            stripped = line.strip()
            if not active and stripped.startswith("{"):
                active = [line]
                depth = line.count("{") - line.count("}")
            elif active:
                active.append(line)
                depth += line.count("{") - line.count("}")
            if active and depth <= 0:
                text = "\n".join(active)
                try:
                    blocks.append(json.loads(text))
                except json.JSONDecodeError:
                    pass
                active = []
                depth = 0
    return blocks


def parse_logs(log_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    logs = sorted(log_dir.glob("*.nohup.log"))
    blocks = load_json_blocks(logs)
    reward_rows: list[dict[str, Any]] = []
    append_rows: list[dict[str, Any]] = []
    completion = {
        "completed": False,
        "completed_line": "",
        "error_lines": [],
        "log_files": [p.name for p in logs],
    }
    for block in blocks:
        if "validation" in block and "lambda_state" in block:
            row: dict[str, Any] = {"iteration": block.get("iteration")}
            for name, value in block.get("lambda_state", {}).items():
                row[f"lambda_{name}"] = value
            for name, value in block.get("alpha_state", {}).items():
                row[f"alpha_{name}"] = value
            for name, value in block.get("model_status_by_boundary", {}).items():
                row[f"status_{name}"] = value
            for name, value in block.get("validation", {}).items():
                row[name] = value
            row["reward_history_rows"] = block.get("reward_history_rows")
            row["new_reward_rows"] = block.get("new_reward_rows")
            row["stageiv_data_used_for_training"] = block.get("stageiv_data_used_for_training")
            row["stagev1_data_used_for_training"] = block.get("stagev1_data_used_for_training")
            reward_rows.append(row)
        if "input_samples" in block and "output_samples" in block:
            append_rows.append(block)
    for path in logs:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "production loop completed" in line:
                completion["completed"] = True
                completion["completed_line"] = line.strip()
            if "[error]" in line or "Traceback" in line or "FAILED" in line:
                completion["error_lines"].append(f"{path.name}: {line.strip()}")
    reward_df = pd.DataFrame(reward_rows).drop_duplicates(subset=["iteration"], keep="last")
    if not reward_df.empty:
        reward_df = reward_df.sort_values("iteration")
    append_df = pd.DataFrame(append_rows)
    if not append_df.empty:
        output_iteration = append_df["output_npz"].astype(str).str.extract(r"dataset_iter(\d+)\.npz")[0]
        append_df["iteration"] = output_iteration.astype(int) - 1
        append_df = append_df.sort_values("iteration")
    return reward_df, append_df, completion


def topology_name(code: int) -> str:
    if code == 0:
        return "cFFLO/trivial"
    if code == 1:
        return "tFFLO/topological"
    return "unresolved/nonapplicable"


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def final_counts(df: pd.DataFrame, tables_dir: Path) -> dict[str, Any]:
    phase_counts = (
        df["phase_name"].value_counts().rename_axis("phase").reset_index(name="count").sort_values("phase")
    )
    save_table(phase_counts, tables_dir / "final_phase_counts.csv")

    fflo = df[df["phase_name"] == "FFLO"].copy()
    fflo["topology_name"] = fflo["topology_label_code"].astype(int).map(topology_name)
    topo_counts = (
        fflo["topology_name"].value_counts().rename_axis("topology").reset_index(name="count").sort_values("topology")
    )
    save_table(topo_counts, tables_dir / "fflo_topology_counts.csv")

    status_rows = [
        {"metric": "total_samples", "value": int(len(df))},
        {"metric": "trusted_exact", "value": int(df["trusted_exact"].sum())},
        {"metric": "training_eligible_exact", "value": int(df["training_eligible_exact"].sum())},
        {"metric": "needs_rerun_exact", "value": int(df["needs_rerun_exact"].sum())},
        {"metric": "q_unresolved", "value": int(df["q_unresolved"].sum())},
        {"metric": "delta_unresolved", "value": int(df["delta_unresolved"].sum())},
        {"metric": "topology_trusted", "value": int(df["topology_trusted"].sum())},
        {"metric": "fflo_points", "value": int(len(fflo))},
    ]
    status_df = pd.DataFrame(status_rows)
    save_table(status_df, tables_dir / "final_status_counts.csv")
    return {
        "phase_counts": dict(zip(phase_counts["phase"], phase_counts["count"])),
        "topology_counts": dict(zip(topo_counts["topology"], topo_counts["count"])),
        "status_counts": dict(zip(status_df["metric"], status_df["value"])),
    }


def plot_phase_3d(df: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for phase, sub in df.groupby("phase_name"):
        ax.scatter(
            sub["kT"],
            sub["JA"],
            sub["mu"],
            s=6,
            alpha=0.34,
            color=PHASE_COLORS.get(phase, "#333333"),
            label=f"{phase} ({len(sub)})",
            depthshade=False,
        )
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final Stage V-v2 thermodynamic phase map")
    ax.view_init(elev=23, azim=-57)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_phase_projection(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for phase, sub in df.groupby("phase_name"):
        ax.scatter(
            sub["kT"],
            sub["JA"],
            s=8,
            alpha=0.28,
            color=PHASE_COLORS.get(phase, "#333333"),
            label=f"{phase} ({len(sub)})",
        )
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_title(r"$J_A$-$k_B T$ projection, colored by thermodynamic phase")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_topology_3d(df: pd.DataFrame, out: Path) -> None:
    fflo = df[df["phase_name"] == "FFLO"].copy()
    fflo["topology_name"] = fflo["topology_label_code"].astype(int).map(topology_name)
    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for topo, sub in fflo.groupby("topology_name"):
        ax.scatter(
            sub["kT"],
            sub["JA"],
            sub["mu"],
            s=8,
            alpha=0.42,
            color=TOPO_COLORS.get(topo, "#888888"),
            label=f"{topo} ({len(sub)})",
            depthshade=False,
        )
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final FFLO topology labels")
    ax.view_init(elev=24, azim=-57)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_mu_slices(df: pd.DataFrame, out: Path) -> None:
    qs = np.quantile(df["mu"], np.linspace(0, 1, 7))
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for i in range(6):
        lo, hi = qs[i], qs[i + 1]
        if i == 5:
            sub = df[(df["mu"] >= lo) & (df["mu"] <= hi)]
        else:
            sub = df[(df["mu"] >= lo) & (df["mu"] < hi)]
        ax = axes[i]
        for phase, psub in sub.groupby("phase_name"):
            ax.scatter(
                psub["kT"],
                psub["JA"],
                s=7,
                alpha=0.38,
                color=PHASE_COLORS.get(phase, "#333333"),
            )
        ax.set_title(f"{lo:.2f} <= mu/t <= {hi:.2f}\nN={len(sub)}", fontsize=9)
        ax.grid(alpha=0.18)
    for ax in axes[3:]:
        ax.set_xlabel(r"$k_B T/t$")
    for ax in axes[::3]:
        ax.set_ylabel(r"$J_A/t$")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=p, markersize=6)
        for p, c in PHASE_COLORS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Phase map by mu quantile slices", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_reward_curves(reward_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    boundaries = ["ns", "uf", "p0", "ppi", "gap"]
    colors_map = {"ns": "#333333", "uf": "#2f7ed8", "p0": "#d62728", "ppi": "#9467bd", "gap": "#2ca02c"}
    for b in boundaries:
        col = f"rank_correlation_model_{b}"
        if col in reward_df:
            axes[0].plot(reward_df["iteration"], reward_df[col], label=f"model {b}", color=colors_map[b])
        a0 = f"rank_correlation_a0_{b}"
        if a0 in reward_df:
            axes[0].plot(reward_df["iteration"], reward_df[a0], linestyle="--", alpha=0.45, color=colors_map[b])
        lam = f"lambda_{b}"
        if lam in reward_df:
            axes[1].plot(reward_df["iteration"], reward_df[lam], label=f"lambda {b}", color=colors_map[b])
    axes[0].set_ylabel("rank correlation")
    axes[0].set_title("Acquisition reward-model validation")
    axes[0].grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("lambda")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=5, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_growth(append_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(append_df["iteration"], append_df["output_samples"], marker="o", markersize=2, linewidth=1.4)
    ax.set_xlabel("exact iteration")
    ax.set_ylabel("cumulative samples")
    ax.set_title("Stage V-v2 dataset growth")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_final_selection(iter_dir: Path, out: Path, tables_dir: Path) -> None:
    meta_path = iter_dir / "selected_points_metadata.csv"
    if not meta_path.exists():
        return
    meta = pd.read_csv(meta_path)
    counts = meta["dominant_boundary"].fillna("unknown").value_counts().rename_axis("dominant_boundary").reset_index(name="count")
    save_table(counts, tables_dir / "iter096_selected_dominant_boundary_counts.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(counts["dominant_boundary"], counts["count"], color="#4c78a8")
    ax.set_ylabel("selected points")
    ax.set_title("Final iteration selected-point boundary focus")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_markdown(
    out: Path,
    counts: dict[str, Any],
    latest_reward: dict[str, Any],
    completion: dict[str, Any],
    figures: list[Path],
    source_root: Path,
) -> None:
    lines = [
        "# Stage V-v2 Multi-head Acquisition Return Report",
        "",
        "## Scope",
        "",
        "This is a report-only local analysis of the returned Stage V-v2 H100 run.",
        "It does not modify the dataset, exact oracle, acquisition code, or numerical tolerances.",
        "",
        "## Source Data",
        "",
        f"- Local source root: `{source_root}`",
        "- Final dataset: `dataset_iter097.npz` / `dataset_iter097.csv`",
        "- Final exact iteration: `iter096`",
        "- Logs: downloaded `*.nohup.log` files",
        "",
        "## Completion Status",
        "",
        f"- Production loop completed: `{completion.get('completed')}`",
        f"- Completion line: `{completion.get('completed_line', '')}`",
        "",
        "## Final Counts",
        "",
        f"- Total samples: `{counts['status_counts'].get('total_samples')}`",
        f"- Trusted exact: `{counts['status_counts'].get('trusted_exact')}`",
        f"- Training eligible exact: `{counts['status_counts'].get('training_eligible_exact')}`",
        f"- Rerun required: `{counts['status_counts'].get('needs_rerun_exact')}`",
        f"- q unresolved: `{counts['status_counts'].get('q_unresolved')}`",
        f"- Delta unresolved: `{counts['status_counts'].get('delta_unresolved')}`",
        "",
        "Phase counts:",
        "",
    ]
    for name, value in counts["phase_counts"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "FFLO topology counts:", ""])
    for name, value in counts["topology_counts"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Acquisition Learning Snapshot", ""])
    for key in [
        "iteration",
        "reward_history_rows",
        "rank_correlation_model_ns",
        "rank_correlation_model_uf",
        "rank_correlation_model_p0",
        "rank_correlation_model_ppi",
        "rank_correlation_model_gap",
        "lambda_ns",
        "lambda_uf",
        "lambda_p0",
        "lambda_ppi",
        "lambda_gap",
    ]:
        if key in latest_reward:
            lines.append(f"- `{key}`: `{latest_reward[key]}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The run completed the configured production loop through exact iteration 096 and produced `dataset_iter097`.",
            "The multi-head acquisition model remained active for the topology P0/Ppi heads, with lambda values at 0.7 in the final update.",
            "The gap head remained marked `insufficient_data` in the final reward update, so gap/nodal learning should be treated as diagnostic rather than a demonstrated learned acquisition channel.",
            "The final phase and topology figures below are scatter/projection diagnostics over the sampled 3D point cloud; they are not smooth fitted phase-boundary surfaces.",
            "",
            "## Figures",
            "",
        ]
    )
    for fig in figures:
        lines.append(f"![{fig.stem}](figures/{fig.name})")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_pdf(md_summary: dict[str, Any], figures: list[Path], out: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=0.55 * inch, leftMargin=0.55 * inch)
    story: list[Any] = []
    story.append(Paragraph("Stage V-v2 Multi-head Acquisition Return Report", styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Completion and Final Dataset", styles["Heading2"]))
    table_data = [["Metric", "Value"]] + [[k, str(v)] for k, v in md_summary.items()]
    table = Table(table_data, colWidths=[2.4 * inch, 4.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "Figures are report-only diagnostics from the downloaded final dataset and logs. "
            "No exact calculation or acquisition step was rerun locally.",
            styles["BodyText"],
        )
    )
    for i, fig in enumerate(figures):
        story.append(PageBreak())
        story.append(Paragraph(fig.stem.replace("_", " ").title(), styles["Heading2"]))
        story.append(Image(str(fig), width=6.8 * inch, height=4.7 * inch))
    doc.build(story)


def write_decision_log(out: Path, decision: dict[str, Any], completion: dict[str, Any]) -> None:
    lines = [
        "# Stage V-v2 Return Decision Log",
        "",
        "## Decision",
        "",
        "- Status: returned production run completed through exact iteration 096.",
        "- Final dataset: `dataset_iter097`.",
        "- Local report is diagnostic only; it does not claim formal 3D boundary convergence.",
        "",
        "## Evidence",
        "",
        f"- Production loop completed: `{decision['production_loop_completed']}`",
        f"- Completion line: `{completion.get('completed_line', '')}`",
        f"- Total samples: `{decision['total_samples']}`",
        f"- Trusted exact samples: `{decision['trusted_exact']}`",
        f"- Phase counts: `{decision['phase_counts']}`",
        f"- FFLO topology counts: `{decision['fflo_topology_counts']}`",
        f"- Latest reward iteration: `{decision['latest_reward_iteration']}`",
        f"- Latest P0 reward-model rank correlation: `{decision['latest_model_rank_correlation_p0']}`",
        f"- Latest Ppi reward-model rank correlation: `{decision['latest_model_rank_correlation_ppi']}`",
        "",
        "## Caveats",
        "",
        "- Figures are sampled-point and projection diagnostics, not smooth final boundary surfaces.",
        "- The gap acquisition head remained diagnostic/insufficient-data in the final reward update.",
        "- Formal convergence still requires a dedicated fixed-grid boundary and topology audit.",
        "",
        "## Recommended Next Step",
        "",
        "- Run a report-only Stage V-v2 convergence audit on the returned cumulative datasets before freezing any 3D boundary surface.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local Stage V-v2 return report.")
    parser.add_argument(
        "--package-root",
        default="stagev_v2_multihead_boundary_learning_3d_v1_hpc",
        help="Local downloaded Stage V-v2 package root.",
    )
    args = parser.parse_args()

    package_root = Path(args.package_root)
    active_run = package_root / "ML_Phase_StageV_V2_Multihead" / "active_runs" / "stagev_v2_multihead_boundary_learning_3d_v1"
    logs_dir = package_root / "logs"
    report_root = ensure_dir(package_root / "ML_Phase_StageV_V2_Multihead" / "reports" / "stagev_v2_return_report_local")
    tables_dir = ensure_dir(report_root / "tables")
    figures_dir = ensure_dir(report_root / "figures")

    df = pd.read_csv(active_run / "dataset_iter097.csv")
    reward_history = pd.read_csv(active_run / "stagev2_reward_history.csv")
    save_table(reward_history.tail(256), tables_dir / "reward_history_tail.csv")
    reward_df, append_df, completion = parse_logs(logs_dir)
    if not reward_df.empty:
        save_table(reward_df, tables_dir / "reward_summary_by_iteration.csv")
    if not append_df.empty:
        save_table(append_df, tables_dir / "append_summary_by_iteration.csv")
    counts = final_counts(df, tables_dir)

    figure_paths = [
        figures_dir / "final_phase_3d.png",
        figures_dir / "final_phase_jakt_projection.png",
        figures_dir / "final_fflo_topology_3d.png",
        figures_dir / "phase_mu_slice_atlas.png",
        figures_dir / "reward_rank_correlation_and_lambda.png",
        figures_dir / "dataset_growth.png",
        figures_dir / "iter096_selected_boundary_focus.png",
    ]
    plot_phase_3d(df, figure_paths[0])
    plot_phase_projection(df, figure_paths[1])
    plot_topology_3d(df, figure_paths[2])
    plot_mu_slices(df, figure_paths[3])
    if not reward_df.empty:
        plot_reward_curves(reward_df, figure_paths[4])
    if not append_df.empty:
        plot_growth(append_df, figure_paths[5])
    plot_final_selection(active_run / "iter096", figure_paths[6], tables_dir)
    figure_paths = [p for p in figure_paths if p.exists()]

    latest_reward = reward_df.iloc[-1].to_dict() if not reward_df.empty else {}
    decision = {
        "run_id": "stagev_v2_multihead_boundary_learning_3d_v1",
        "final_dataset": "dataset_iter097",
        "final_exact_iteration": 96,
        "production_loop_completed": bool(completion.get("completed")),
        "total_samples": counts["status_counts"].get("total_samples"),
        "trusted_exact": counts["status_counts"].get("trusted_exact"),
        "phase_counts": counts["phase_counts"],
        "fflo_topology_counts": counts["topology_counts"],
        "latest_reward_iteration": int(latest_reward.get("iteration", -1)) if latest_reward else None,
        "latest_model_rank_correlation_p0": latest_reward.get("rank_correlation_model_p0"),
        "latest_model_rank_correlation_ppi": latest_reward.get("rank_correlation_model_ppi"),
        "latest_model_rank_correlation_ns": latest_reward.get("rank_correlation_model_ns"),
        "latest_lambda_p0": latest_reward.get("lambda_p0"),
        "latest_lambda_ppi": latest_reward.get("lambda_ppi"),
        "caveat": "This report is diagnostic and does not assert formal 3D boundary convergence.",
    }
    (report_root / "stagev_v2_return_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_decision_log(report_root / "decision_log.md", decision, completion)
    write_markdown(
        report_root / "stagev_v2_return_report.md",
        counts,
        latest_reward,
        completion,
        figure_paths,
        active_run,
    )
    pdf_summary = {
        "run_id": decision["run_id"],
        "final_dataset": decision["final_dataset"],
        "final_exact_iteration": decision["final_exact_iteration"],
        "production_loop_completed": decision["production_loop_completed"],
        "total_samples": decision["total_samples"],
        "trusted_exact": decision["trusted_exact"],
        "normal": counts["phase_counts"].get("normal", 0),
        "uniform_SC": counts["phase_counts"].get("uniform_SC", 0),
        "FFLO": counts["phase_counts"].get("FFLO", 0),
        "cFFLO/trivial": counts["topology_counts"].get("cFFLO/trivial", 0),
        "tFFLO/topological": counts["topology_counts"].get("tFFLO/topological", 0),
        "latest p0 model rho": decision["latest_model_rank_correlation_p0"],
        "latest ppi model rho": decision["latest_model_rank_correlation_ppi"],
    }
    write_pdf(pdf_summary, figure_paths, report_root / "stagev_v2_return_report.pdf")
    print(json.dumps({"report_root": str(report_root), **decision}, indent=2))


if __name__ == "__main__":
    main()
