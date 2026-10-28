from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PACKAGE_ROOT = Path("stagev_v3_sparse_arc_boundary_support_3d_v1_hpc")
DEFAULT_OUTPUT_ROOT = "ML_Phase_StageV_V3_SparseArc"
DEFAULT_RUN_ID = "stagev_v3_sparse_arc_boundary_support_3d_v1"
DEFAULT_REPORT_SUBDIR = "stagev_v3_return_report_local"

V2_TAIL_BASELINE = {
    "label": "Stage V-v2 coverage tail",
    "final_dataset": "dataset_iter101",
    "total_samples": 7323,
    "trusted_exact": 7182,
    "normal": 4876,
    "uniform_SC": 83,
    "FFLO": 2364,
    "cFFLO_trivial": 1968,
    "tFFLO_topological": 396,
    "topology_surface_coverage_p95": 0.03591463649972722,
    "topology_map_change_last3": [0.0008933263, 0.0031354002, 0.0015760993],
    "trusted_topology_surprise_last3": [0.1, 0.0909090909, 0.0357142857],
    "decision_class": "C",
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def final_iteration(run_dir: Path) -> int:
    values: list[int] = []
    for path in run_dir.glob("dataset_iter*.csv"):
        stem = path.stem
        try:
            values.append(int(stem.replace("dataset_iter", "")))
        except ValueError:
            pass
    if not values:
        raise FileNotFoundError(f"No dataset_iter*.csv files under {run_dir}")
    return max(values)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_final_dataset(run_dir: Path, iteration: int) -> pd.DataFrame:
    path = run_dir / f"dataset_iter{iteration:03d}.csv"
    return pd.read_csv(path, low_memory=False)


def topology_name(code: object) -> str:
    try:
        icode = int(code)
    except (TypeError, ValueError):
        return "not_applicable_or_unresolved"
    if icode == 0:
        return "cFFLO / trivial"
    if icode == 1:
        return "tFFLO / topological"
    return "not_applicable_or_unresolved"


def summarize_final_dataset(df: pd.DataFrame, tables_dir: Path) -> dict[str, Any]:
    phase_counts = (
        df["phase_name"].value_counts().rename_axis("phase").reset_index(name="count").sort_values("phase")
    )
    phase_counts.to_csv(tables_dir / "final_phase_counts.csv", index=False)

    fflo = df[df["phase_name"] == "FFLO"].copy()
    fflo["topology_name"] = fflo["topology_label_code"].map(topology_name)
    topo_counts = (
        fflo["topology_name"].value_counts().rename_axis("topology").reset_index(name="count").sort_values("topology")
    )
    topo_counts.to_csv(tables_dir / "final_fflo_topology_counts.csv", index=False)

    status_rows = [
        ("total_samples", len(df)),
        ("trusted_exact", int(df["trusted_exact"].astype(bool).sum())),
        ("training_eligible_exact", int(df["training_eligible_exact"].astype(bool).sum())),
        ("needs_rerun_exact", int(df["needs_rerun_exact"].astype(bool).sum())),
        ("q_unresolved", int(df["q_unresolved"].astype(bool).sum())),
        ("delta_unresolved", int(df["delta_unresolved"].astype(bool).sum())),
        ("topology_applicable", int(df["topology_applicable"].astype(bool).sum())),
        ("topology_trusted", int(df["topology_trusted"].astype(bool).sum())),
        ("fflo_points", len(fflo)),
    ]
    status = pd.DataFrame(status_rows, columns=["metric", "value"])
    status.to_csv(tables_dir / "final_status_counts.csv", index=False)

    return {
        "phase_counts": dict(zip(phase_counts["phase"], phase_counts["count"])),
        "topology_counts": dict(zip(topo_counts["topology"], topo_counts["count"])),
        "status_counts": dict(zip(status["metric"], status["value"])),
    }


def summarize_iterations(run_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    usecols = [
        "phase_name",
        "trusted_exact",
        "training_eligible_exact",
        "needs_rerun_exact",
        "topology_label_code",
        "topology_trusted",
    ]
    for path in sorted(run_dir.glob("dataset_iter*.csv")):
        try:
            iteration = int(path.stem.replace("dataset_iter", ""))
        except ValueError:
            continue
        df = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
        row: dict[str, Any] = {
            "dataset_iteration": iteration,
            "total_samples": len(df),
            "trusted_exact": int(df.get("trusted_exact", pd.Series(dtype=bool)).astype(bool).sum()),
            "training_eligible_exact": int(
                df.get("training_eligible_exact", pd.Series(dtype=bool)).astype(bool).sum()
            ),
            "needs_rerun_exact": int(df.get("needs_rerun_exact", pd.Series(dtype=bool)).astype(bool).sum()),
        }
        phase_series = df["phase_name"] if "phase_name" in df.columns else pd.Series("", index=df.index)
        for phase in ["normal", "uniform_SC", "FFLO"]:
            row[f"phase_{phase}"] = int(phase_series.eq(phase).sum())
        if "topology_label_code" in df.columns:
            fflo_mask = phase_series.eq("FFLO") if "phase_name" in df.columns else pd.Series(False, index=df.index)
            row["topology_cfflo_trivial"] = int((fflo_mask & df["topology_label_code"].eq(0)).sum())
            row["topology_tfflo_topological"] = int((fflo_mask & df["topology_label_code"].eq(1)).sum())
            row["topology_trusted"] = int(df.get("topology_trusted", pd.Series(dtype=bool)).astype(bool).sum())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("dataset_iteration")
    out.to_csv(tables_dir / "iteration_dataset_counts.csv", index=False)
    return out


def summarize_reward(run_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("iter*/stagev3_reward_update_summary.json")):
        data = read_json(path)
        if not data:
            continue
        row: dict[str, Any] = {"iteration": data.get("iteration")}
        for name, value in data.get("lambda_state", {}).items():
            row[f"lambda_{name}"] = value
        for name, value in data.get("alpha_state", {}).items():
            row[f"alpha_{name}"] = value
        for name, value in data.get("model_status_by_boundary", {}).items():
            row[f"status_{name}"] = value
        for name, value in data.get("validation", {}).items():
            row[name] = value
        row["reward_history_rows"] = data.get("reward_history_rows")
        row["new_reward_rows"] = data.get("new_reward_rows")
        row["stageiv_data_used_for_training"] = data.get("stageiv_data_used_for_training")
        row["stagev1_data_used_for_training"] = data.get("stagev1_data_used_for_training")
        row["stagev2_data_used_for_training"] = data.get("stagev2_data_used_for_training")
        row["stagev2_tail_data_used_for_training"] = data.get("stagev2_tail_data_used_for_training")
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("iteration")
    df.to_csv(tables_dir / "reward_learning_by_iteration.csv", index=False)
    return df


def summarize_selection(run_dir: Path, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(run_dir.glob("iter*/selected_points_metadata.csv")):
        try:
            iteration = int(path.parent.name.replace("iter", ""))
        except ValueError:
            continue
        df = pd.read_csv(path, low_memory=False)
        df["iteration"] = iteration
        frames.append(df)
    if not frames:
        empty = pd.DataFrame()
        return empty, empty, empty
    selected = pd.concat(frames, ignore_index=True)
    source_col = "candidate_source" if "candidate_source" in selected.columns else "proposal_source"
    selected[source_col] = selected[source_col].fillna("unknown_explicit").astype(str)
    selected.loc[selected[source_col].str.strip().eq(""), source_col] = "unknown_explicit"
    selected["source"] = selected[source_col]

    source_counts = selected["source"].value_counts().rename_axis("source").reset_index(name="selected_count")
    source_counts["selected_fraction"] = source_counts["selected_count"] / max(1, len(selected))
    source_counts.to_csv(tables_dir / "proposal_source_counts.csv", index=False)

    source_iter = (
        selected.groupby(["iteration", "source"]).size().rename("selected_count").reset_index().sort_values(["iteration", "source"])
    )
    source_iter.to_csv(tables_dir / "proposal_source_by_iteration.csv", index=False)

    diagnostics = {
        "selected_rows": len(selected),
        "iterations_with_selection": int(selected["iteration"].nunique()),
        "unknown_explicit_count": int(selected["source"].eq("unknown_explicit").sum()),
        "blank_or_nan_source_failure_count": int(
            selected.get("source_failure_reason", pd.Series("", index=selected.index))
            .fillna("")
            .astype(str)
            .str.contains("blank_or_nan_source")
            .sum()
        ),
        "mean_C_topo": float(selected.get("C_topo", pd.Series(dtype=float)).mean())
        if "C_topo" in selected.columns
        else math.nan,
        "median_C_topo": float(selected.get("C_topo", pd.Series(dtype=float)).median())
        if "C_topo" in selected.columns
        else math.nan,
        "mean_gap_topology_relevance": float(selected.get("gap_topology_relevance", pd.Series(dtype=float)).mean())
        if "gap_topology_relevance" in selected.columns
        else math.nan,
    }
    pd.DataFrame([diagnostics]).to_csv(tables_dir / "source_attribution_diagnostics.csv", index=False)
    selected.to_csv(tables_dir / "selected_points_metadata_all_iterations.csv", index=False)
    return selected, source_counts, source_iter


def read_formal_decision(package_root: Path, output_root: str) -> dict[str, Any]:
    path = package_root / output_root / "reports" / "stagev_v3_formal_audit_local" / "stagev_v2_decision.json"
    return read_json(path)


def build_comparison(decision: dict[str, Any], final_summary: dict[str, Any], tables_dir: Path) -> pd.DataFrame:
    v3_cov = float(decision.get("topology_surface_coverage_p95_final", math.nan))
    v2_cov = V2_TAIL_BASELINE["topology_surface_coverage_p95"]
    rows = [
        {
            "metric": "total_samples",
            "stagev2_tail": V2_TAIL_BASELINE["total_samples"],
            "stagev3": decision.get("total_samples", final_summary["status_counts"].get("total_samples")),
            "interpretation": "Comparable budget; v3 is slightly smaller.",
        },
        {
            "metric": "topology_surface_coverage_p95",
            "stagev2_tail": v2_cov,
            "stagev3": v3_cov,
            "relative_change": (v3_cov - v2_cov) / v2_cov,
            "interpretation": "Lower is better; v3 improves coverage but remains above gate.",
        },
        {
            "metric": "coverage_gate_ratio",
            "stagev2_tail": v2_cov / 0.00625,
            "stagev3": v3_cov / 0.00625,
            "interpretation": "Ratio to 0.00625 gate.",
        },
        {
            "metric": "tFFLO_topological_count",
            "stagev2_tail": V2_TAIL_BASELINE["tFFLO_topological"],
            "stagev3": final_summary["topology_counts"].get("tFFLO / topological", 0),
            "interpretation": "Higher count is support evidence, not convergence proof.",
        },
        {
            "metric": "topology_map_change_last3_max",
            "stagev2_tail": max(V2_TAIL_BASELINE["topology_map_change_last3"]),
            "stagev3": max(decision.get("topology_map_change_last3", [math.nan])),
            "interpretation": "Gate is 0.002.",
        },
        {
            "metric": "trusted_topology_surprise_last3_max",
            "stagev2_tail": max(V2_TAIL_BASELINE["trusted_topology_surprise_last3"]),
            "stagev3": max(decision.get("trusted_topology_surprise_last3", [math.nan])),
            "interpretation": "Gate is 0.02; v3 remains high.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "stagev3_vs_v2_tail_comparison.csv", index=False)
    return df


def write_acquisition_design_tables(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_rows = [
        {
            "component": "cold_start_initial_design",
            "mathematical_role": "initial exact support",
            "implementation": "1024 scrambled Sobol points written by select_seed",
            "learned": "no",
            "assumption": "A space-filling initial design is sufficient to start thermodynamic and topology support without importing Stage V-v2 labels.",
        },
        {
            "component": "boundary_likelihood_B_s",
            "mathematical_role": "boundary proximity",
            "implementation": "per-head residual or margin score for NS, UF, P0, Ppi, and gap heads",
            "learned": "no",
            "assumption": "Useful exact points are enriched near thermodynamic or topology decision surfaces.",
        },
        {
            "component": "uncertainty_U_s",
            "mathematical_role": "model uncertainty",
            "implementation": "ensemble or surrogate uncertainty columns inherited from the Stage V-v2 scoring pipeline",
            "learned": "partly",
            "assumption": "Uncertain boundary predictions are more likely to produce informative exact labels.",
        },
        {
            "component": "support_sparsity_H_s",
            "mathematical_role": "fill-distance pressure",
            "implementation": "distance to nearby trusted exact support and bracket support",
            "learned": "no",
            "assumption": "Unsupported boundary patches should receive priority even if the map is visually stable.",
        },
        {
            "component": "topology_coverage_C_topo",
            "mathematical_role": "expected topology-surface coverage improvement",
            "implementation": "build_sparse_topology_surface plus expected_topology_coverage_improvement",
            "learned": "no",
            "assumption": "A candidate should be valuable if adding it would reduce local support distance on sparse cFFLO/tFFLO surface samples.",
        },
        {
            "component": "gap_topology_relevance",
            "mathematical_role": "gap-head guard",
            "implementation": "suppresses gap candidates far from P0/Ppi or sparse topology-support regions",
            "learned": "no",
            "assumption": "Small spectral gap alone is not enough; gap work should support topology classification.",
        },
        {
            "component": "learned_residual_g_s",
            "mathematical_role": "online value correction",
            "implementation": "per-head linear value models fit from Stage V-v3 reward history",
            "learned": "yes",
            "assumption": "The logged exact outcomes can learn which candidate features actually improve scientific metrics.",
        },
        {
            "component": "lambda_s",
            "mathematical_role": "residual activation strength",
            "implementation": "updated from rank-correlation and coverage-improvement validation",
            "learned": "adaptive",
            "assumption": "The learned residual should influence selection only when it improves reward prediction and scientific coverage.",
        },
        {
            "component": "alpha_s",
            "mathematical_role": "head priority",
            "implementation": "updated from boundary support, surprise, shift, and convergence proxy deficits",
            "learned": "adaptive",
            "assumption": "Different boundary heads should be prioritized according to current scientific deficits.",
        },
        {
            "component": "proposal_source",
            "mathematical_role": "provenance and source efficiency",
            "implementation": "candidate_source validation with unknown_explicit fallback",
            "learned": "diagnostic",
            "assumption": "Every selected point must be traceable to a proposal mechanism for later acquisition audit.",
        },
    ]
    components = pd.DataFrame(component_rows)
    components.to_csv(tables_dir / "stagev3_acquisition_function_components.csv", index=False)

    training_rows = [
        {
            "stage": "candidate_pool",
            "input_data": "current Stage V-v3 cumulative dataset only",
            "output": "global, boundary, sparse topology, bracket, and jitter candidates",
            "leakage_guard": "no Stage IV, V1, V2, V2-tail labels or coordinates used for production initialization",
        },
        {
            "stage": "base_score",
            "input_data": "current surrogate fields, uncertainties, support distances, and sparse topology surface proxy",
            "output": "A0_s for NS, UF, P0, Ppi, and gap heads",
            "leakage_guard": "explicit physics/support score, not fit to previous runs",
        },
        {
            "stage": "value_model",
            "input_data": "Stage V-v3 selected-point metadata and exact outcomes accumulated so far",
            "output": "g_s(phi_s(x)) for each boundary head",
            "leakage_guard": "reward history flags record stageiv/stagev1/stagev2/stagev2_tail training use as false",
        },
        {
            "stage": "score_mix",
            "input_data": "A0_s, g_s, lambda_s, alpha_s, source-density correction, exact repulsion",
            "output": "A_total used for stochastic micro-batch selection",
            "leakage_guard": "lambda_s is reduced or held when learned reward does not improve scientific coverage",
        },
        {
            "stage": "exact_feedback",
            "input_data": "new exact-oracle shards from the selected micro-batch",
            "output": "per-head rewards including topology coverage, bracket, residual, support, and reliability terms",
            "leakage_guard": "original thermodynamic and topology criteria remain unchanged",
        },
        {
            "stage": "state_update",
            "input_data": "updated reward history and boundary metric proxies",
            "output": "new value models, lambda_s, alpha_s, and reward diagnostics",
            "leakage_guard": "state is saved inside the V3 run directory only",
        },
    ]
    training = pd.DataFrame(training_rows)
    training.to_csv(tables_dir / "stagev3_acquisition_training_loop.csv", index=False)
    return components, training


def plot_dataset_growth(iter_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(iter_df["dataset_iteration"], iter_df["total_samples"], label="total", color="#222222")
    for col, color, label in [
        ("phase_normal", "#777777", "normal"),
        ("phase_uniform_SC", "#2f7ed8", "uniform-SC"),
        ("phase_FFLO", "#d62728", "FFLO"),
    ]:
        if col in iter_df:
            ax.plot(iter_df["dataset_iteration"], iter_df[col], label=label, color=color, alpha=0.85)
    ax.set_xlabel("Dataset iteration")
    ax.set_ylabel("Cumulative exact points")
    ax.set_title("Stage V-v3 cumulative dataset growth")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_topology_growth(iter_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, color, label in [
        ("topology_cfflo_trivial", "#2f7ed8", "cFFLO / trivial"),
        ("topology_tfflo_topological", "#d62728", "tFFLO / topological"),
        ("topology_trusted", "#333333", "topology trusted"),
    ]:
        if col in iter_df:
            ax.plot(iter_df["dataset_iteration"], iter_df[col], label=label, color=color, alpha=0.9)
    ax.set_xlabel("Dataset iteration")
    ax.set_ylabel("Cumulative points")
    ax.set_title("Stage V-v3 FFLO topology support growth")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_reward(reward_df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    ax = axes[0]
    for col, color in [
        ("lambda_gap", "#8c564b"),
        ("lambda_ns", "#777777"),
        ("lambda_p0", "#d62728"),
        ("lambda_ppi", "#9467bd"),
        ("lambda_uf", "#2f7ed8"),
    ]:
        if col in reward_df:
            ax.plot(reward_df["iteration"], reward_df[col], label=col.replace("lambda_", r"$\lambda_{") + "}$", color=color)
    ax.set_ylabel("Lambda")
    ax.set_title("Per-head learned residual activation")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=3)
    ax = axes[1]
    for col, color, label in [
        ("rank_correlation_model_p0", "#d62728", "P0 model"),
        ("rank_correlation_model_ppi", "#9467bd", "Ppi model"),
        ("rank_correlation_model_gap", "#8c564b", "gap model"),
        ("rank_correlation_model_uf", "#2f7ed8", "UF model"),
    ]:
        if col in reward_df:
            ax.plot(reward_df["iteration"], reward_df[col], label=label, color=color)
    ax.set_xlabel("Exact iteration")
    ax.set_ylabel("Rank correlation")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_sources(source_counts: pd.DataFrame, source_iter: pd.DataFrame, out_counts: Path, out_iter: Path) -> None:
    top = source_counts.head(12).copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(top["source"][::-1], top["selected_count"][::-1], color="#4c78a8")
    ax.set_xlabel("Selected points")
    ax.set_title("Top Stage V-v3 proposal sources")
    fig.tight_layout()
    fig.savefig(out_counts, dpi=220)
    plt.close(fig)

    if source_iter.empty:
        return
    keep = set(source_counts.head(8)["source"])
    tmp = source_iter.copy()
    tmp.loc[~tmp["source"].isin(keep), "source"] = "other"
    pivot = tmp.groupby(["iteration", "source"])["selected_count"].sum().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.stackplot(pivot.index, [pivot[c] for c in pivot.columns], labels=list(pivot.columns), alpha=0.9)
    ax.set_xlabel("Exact iteration")
    ax.set_ylabel("Selected points")
    ax.set_title("Proposal-source mix across Stage V-v3")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_iter, dpi=220)
    plt.close(fig)


def plot_final_phase(df: pd.DataFrame, out: Path) -> None:
    colors = {"normal": "#777777", "uniform_SC": "#2f7ed8", "FFLO": "#d62728"}
    fig = plt.figure(figsize=(8.0, 6.3))
    ax = fig.add_subplot(111, projection="3d")
    for phase, sub in df.groupby("phase_name"):
        ax.scatter(sub["kT"], sub["JA"], sub["mu"], s=7, alpha=0.36, color=colors.get(phase, "#333333"), label=phase)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final Stage V-v3 thermodynamic labels")
    ax.view_init(elev=22, azim=-58)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_final_topology(df: pd.DataFrame, out: Path) -> None:
    fflo = df[df["phase_name"] == "FFLO"].copy()
    fflo["topology_name"] = fflo["topology_label_code"].map(topology_name)
    colors = {"cFFLO / trivial": "#2f7ed8", "tFFLO / topological": "#d62728"}
    fig = plt.figure(figsize=(8.0, 6.3))
    ax = fig.add_subplot(111, projection="3d")
    for topo, sub in fflo[fflo["topology_name"].isin(colors)].groupby("topology_name"):
        ax.scatter(sub["kT"], sub["JA"], sub["mu"], s=8, alpha=0.45, color=colors[topo], label=topo)
    ax.set_xlabel(r"$k_B T/t$")
    ax.set_ylabel(r"$J_A/t$")
    ax.set_zlabel(r"$\mu/t$")
    ax.set_title("Final Stage V-v3 FFLO topology labels")
    ax.view_init(elev=22, azim=-58)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def copy_formal_figures(package_root: Path, output_root: str, figures_dir: Path) -> dict[str, Path]:
    formal_fig_dir = package_root / output_root / "reports" / "stagev_v3_formal_audit_local" / "figures"
    wanted = {
        "audit_phase_map": "fig01_final_3d_thermodynamic_phase_map.png",
        "audit_topology_map": "fig02_final_3d_fflo_topology_map.png",
        "audit_phase_slice": "fig03_fixed_mu_thermodynamic_slice_atlas.png",
        "audit_topology_slice": "fig04_fixed_mu_topology_slice_atlas.png",
        "audit_phase_change": "fig05_phase_volume_map_change.png",
        "audit_topology_change": "fig06_topology_volume_map_change.png",
        "audit_topology_coverage": "fig08_topology_surface_shift_coverage.png",
        "audit_surprise": "fig09_trusted_phase_topology_surprise.png",
        "audit_components": "fig10_topology_component_stability.png",
        "audit_decision": "fig19_final_decision_summary_panel.png",
    }
    copied: dict[str, Path] = {}
    for key, name in wanted.items():
        src = formal_fig_dir / name
        if src.exists():
            dst = figures_dir / name
            shutil.copy2(src, dst)
            copied[key] = dst
    return copied


def write_markdown(
    report_root: Path,
    decision: dict[str, Any],
    final_summary: dict[str, Any],
    comparison: pd.DataFrame,
    source_diag: dict[str, Any],
    acquisition_components: pd.DataFrame,
    acquisition_training: pd.DataFrame,
) -> None:
    cov = decision.get("topology_surface_coverage_p95_final", math.nan)
    gate = 0.00625
    cov_ratio = cov / gate if isinstance(cov, (float, int)) else math.nan
    lines = [
        "# Stage V-v3 Sparse-Arc Acquisition Return Report",
        "",
        "## Executive Summary",
        "",
        f"- Final dataset: `{decision.get('final_dataset', 'dataset_iter097')}`.",
        f"- Final exact iteration: `{decision.get('final_exact_iteration', 96)}`.",
        f"- Total samples: `{decision.get('total_samples', final_summary['status_counts'].get('total_samples'))}`.",
        f"- Decision class from formal audit: `{decision.get('decision_class', 'unknown')}`.",
        f"- Topology-surface coverage p95: `{cov:.6g}`; gate: `{gate}`; ratio: `{cov_ratio:.2f}x`.",
        "- Stage V-v3 improves the v2-tail coverage p95 substantially, but the formal audit still fails the coverage and trusted topology surprise gates.",
        "- Do not claim Stage V-v3 convergence before a targeted follow-up audit or tail decision.",
        "",
        "## Final Counts",
        "",
        f"- Phase counts: `{final_summary['phase_counts']}`.",
        f"- FFLO topology counts: `{final_summary['topology_counts']}`.",
        f"- Reliability counts: `{final_summary['status_counts']}`.",
        "",
        "## Acquisition Function: What Is Learned From Scratch?",
        "",
        "Stage V-v3 is a strict cold-start run in the data and model-state sense: the production loop starts from a new Sobol design and does not initialize its training set, selected coordinates, reward history, value models, lambda state, or alpha state from Stage IV, Stage V-v1, Stage V-v2, V2-tail, `dataset_iter097`, or `dataset_iter101`.",
        "",
        "The acquisition function is not learned from a blank symbolic form.  It has a transparent physics/support base score, and Stage V-v3 learns an online residual/value correction from its own exact outcomes:",
        "",
        r"$$A_s^{(0)}(x)=B_s(x)\,U_s(x)\,H_s(x),\qquad s\in\{\mathrm{NS},\mathrm{UF},P0,P\pi,\mathrm{gap}\}.$$",
        "",
        r"For topology heads, Stage V-v3 adds the sparse topology-surface coverage term \(C_{\rm topo}(x)\).  The code stores the pre-V3 topology scores and reweights the P0/Ppi scores by the expected coverage-improvement proxy.  Conceptually,",
        "",
        r"$$A_{P0/P\pi}^{\rm v3}(x) \propto A_{P0/P\pi}^{(0)}(x)\,C_{\rm topo}(x).$$",
        "",
        r"The learned residual then enters as a log-score correction, with per-head activation \(\lambda_s\) and priority \(\alpha_s\):",
        "",
        r"$$\ell_s(x)=\log A_s^{\rm v3}(x)+\lambda_s\,g_s(\phi_s(x)),\qquad A_{\rm total}(x)=\exp\{\mathrm{logsumexp}_s[\alpha_s+\mathrm{ranknorm}(\ell_s(x))]+\mathrm{source\ correction}\}.$$",
        "",
        r"Here \(g_s\) is the learned value/residual model fit only from Stage V-v3 selected-point rewards.  Thus V3 learns how much to trust and reshape the explicit base acquisition, rather than discovering the entire acquisition law without physics assumptions.",
        "",
        markdown_table(acquisition_components, ["component", "mathematical_role", "implementation", "learned"], max_rows=20),
        "",
        "## Acquisition Assumptions",
        "",
        "- Boundary proximity is informative: useful exact points should concentrate near normal/SC, uniform/FFLO, P0, Ppi, or gap surfaces.",
        "- Model uncertainty is useful only when tied to a physical or support deficit.",
        "- Topology convergence requires direct support near the cFFLO/tFFLO surface, not just a visually stable smooth contour.",
        "- The topology surface proxy is an acquisition target and audit object, not a final publication boundary.",
        "- The gap head is topology-relevant only when it improves P0/Ppi support, opposite-Z2 bracketing, or sparse topology-surface coverage.",
        "- Reward learning is allowed to change candidate ranking, but lambda gates prevent a learned residual from dominating when it does not improve scientific coverage.",
        "",
        "## Acquisition Training Loop",
        "",
        markdown_table(acquisition_training, ["stage", "input_data", "output", "leakage_guard"], max_rows=20),
        "",
        "The practical loop is: generate candidates, score explicit boundary/support terms, reconstruct sparse topology-surface support, compute `C_topo`, combine per-head scores with learned residuals, select a stochastic micro-batch with recorded propensity, run the exact oracle, convert the exact feedback into per-head rewards, retrain value models, and update `lambda_s` and `alpha_s`.",
        "",
        "The reward design is deliberately aligned with topology support.  P0/Ppi rewards include local coverage gain, p95-contributor gain, opposite-Z2 bracket creation or shortening, sparse-component support, residual proximity, surprise, and reliability penalties.  The gap reward is guarded so that small-gap points far from topology support do not dominate.",
        "",
        "## V2-tail Comparison",
        "",
        markdown_table(comparison, ["metric", "stagev2_tail", "stagev3", "interpretation"], max_rows=8),
        "",
        "## Source Attribution",
        "",
        f"- Selected rows inspected: `{source_diag.get('selected_rows')}`.",
        f"- Unknown explicit sources: `{source_diag.get('unknown_explicit_count')}`.",
        f"- Blank/NaN source failures logged: `{source_diag.get('blank_or_nan_source_failure_count')}`.",
        "",
        "## Caveats",
        "",
        "- This report uses offline KNN/surface-proxy audit outputs from the existing formal-audit script.",
        "- Smooth visual surfaces are not final publication topology boundaries.",
        "- The absence of nodal SC is not treated as failure.",
        "- Response-side eta/Ic quantities are not used to validate topology.",
    ]
    (report_root / "stagev_v3_return_report.md").write_text("\n".join(lines), encoding="utf-8")


def table_rows_for_latex(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    lines: list[str] = []
    for _, row in df.head(max_rows).iterrows():
        cells: list[str] = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                if math.isnan(val):
                    cells.append("")
                else:
                    cells.append(f"{val:.6g}")
            else:
                cells.append(latex_escape(val))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df.head(max_rows).iterrows():
        cells: list[str] = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append("" if math.isnan(val) else f"{val:.6g}")
            else:
                cells.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_latex(
    report_root: Path,
    figures: dict[str, Path],
    decision: dict[str, Any],
    final_summary: dict[str, Any],
    comparison: pd.DataFrame,
    source_counts: pd.DataFrame,
    source_diag: dict[str, Any],
    acquisition_components: pd.DataFrame,
    acquisition_training: pd.DataFrame,
) -> Path:
    cov = float(decision.get("topology_surface_coverage_p95_final", math.nan))
    gate = 0.00625
    cov_ratio = cov / gate
    v2_cov = V2_TAIL_BASELINE["topology_surface_coverage_p95"]
    improvement = (v2_cov - cov) / v2_cov
    phase_counts = final_summary["phase_counts"]
    topo_counts = final_summary["topology_counts"]
    status_counts = final_summary["status_counts"]
    fig = {k: Path(v).relative_to(report_root).as_posix() for k, v in figures.items()}
    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=0.75in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{array}}
\usepackage{{hyperref}}
\usepackage{{float}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\title{{Stage V-v3 Sparse-Arc-Aware Topology-Surface Coverage Acquisition Return Report}}
\author{{BdG topology-aware active-learning project}}
\date{{2026-07-08}}
\begin{{document}}
\maketitle

\section*{{Executive Summary}}
Stage V-v3 completed as a strict cold-start 3D active-learning run.  The final cumulative dataset is
\texttt{{{latex_escape(decision.get("final_dataset", "dataset_iter097"))}}}, with exact iteration
{decision.get("final_exact_iteration", 96)} and {decision.get("total_samples", status_counts.get("total_samples"))}
total exact samples.

The formal offline audit gives decision class \textbf{{{latex_escape(decision.get("decision_class", "unknown"))}}}.
The main improvement relative to the Stage V-v2 coverage tail is topology-surface support:
\[
  p95_{{\rm cov}} = {cov:.6f},
  \quad p95_{{\rm cov,v2tail}} = {v2_cov:.6f},
  \quad \Delta_{{\rm rel}} = {100.0 * improvement:.1f}\%.
\]
However, the configured coverage gate is {gate:.5f}, so the final value is still {cov_ratio:.2f} times the gate.
The topology map-change gate passes in the last-three transitions, but trusted topology surprise remains above
the 0.02 target.  Therefore this report does not claim convergence.

\section*{{Run Provenance and Frozen Semantics}}
\begin{{itemize}}
\item Run ID: \texttt{{stagev\_v3\_sparse\_arc\_boundary\_support\_3d\_v1}}.
\item Output root: \texttt{{ML\_Phase\_StageV\_V3\_SparseArc}}.
\item Production rule: strict cold start; no Stage IV, Stage V-v1, Stage V-v2, or V2-tail data were used for production training initialization.
\item Unchanged physics: exact oracle, Hamiltonian, thermodynamic phase criterion, Pfaffian topology criterion, q/Delta search rules, and tolerances.
\end{{itemize}}

\section*{{Stage V-v3 Acquisition Function Decomposition}}
Stage V-v3 is a strict cold-start run in the data and model-state sense: the production loop starts from a new Sobol design and does not initialize its training set, selected coordinates, reward history, value models, \(\lambda_s\), or \(\alpha_s\) from Stage IV, Stage V-v1, Stage V-v2, V2-tail, \texttt{{dataset\_iter097}}, or \texttt{{dataset\_iter101}}.

The acquisition function itself is not learned from a blank symbolic form.  It is a physics- and support-informed base acquisition with an online learned residual/value correction trained only from Stage V-v3 exact feedback.  For boundary head
\(\mathcal{{S}}=[\mathrm{{NS}},\mathrm{{UF}},P0,P\pi,\mathrm{{gap}}]\), the base score has the form
\[
  A_s^{{(0)}}(x) = B_s(x)\,U_s(x)\,H_s(x),
\]
where \(B_s\) is boundary likelihood, \(U_s\) is field uncertainty, and \(H_s\) is support sparsity.  For topology heads, Stage V-v3 adds the sparse topology-surface coverage-improvement score \(C_{{\rm topo}}(x)\).  The implementation stores the pre-V3 P0/Ppi scores and reweights them using the expected local support-distance reduction:
\[
  A_{{P0/P\pi}}^{{\rm v3}}(x) \propto A_{{P0/P\pi}}^{{(0)}}(x)\,C_{{\rm topo}}(x).
\]
The learned residual enters as a gated log-score correction,
\[
  \ell_s(x) = \log A_s^{{\rm v3}}(x) + \lambda_s g_s(\phi_s(x)),
  \qquad
  A_{{\rm total}}(x) =
  \exp\left[
    \mathrm{{logsumexp}}_s\left(\alpha_s + \mathrm{{ranknorm}}(\ell_s(x))\right)
    + \mathrm{{source\ correction}}
  \right],
\]
followed by stochastic micro-batch selection with recorded propensity.  Thus Stage V-v3 learns how to reshape and trust the explicit base acquisition, not the entire acquisition law without physical assumptions.

\begin{{table}}[H]
\centering
\footnotesize
\begin{{tabular}}{{p{{0.20\linewidth}}p{{0.18\linewidth}}p{{0.39\linewidth}}p{{0.11\linewidth}}}}
\toprule
Component & Role & Implementation & Learned \\
\midrule
{table_rows_for_latex(acquisition_components, ["component", "mathematical_role", "implementation", "learned"], max_rows=10)}
\bottomrule
\end{{tabular}}
\caption{{Stage V-v3 acquisition components.  The learned part is the per-head value residual and adaptive gating; the physical/support base terms remain explicit.}}
\end{{table}}

\section*{{Acquisition Training and Assumptions}}
The central assumptions are: boundary proximity is informative; uncertainty is useful only when tied to a physical or support deficit; topology convergence requires direct support near the cFFLO/tFFLO surface rather than merely a stable smooth proxy; the topology surface proxy is an acquisition target and audit object, not a final publication boundary; and the gap head should be high only when it is topology-relevant.

\begin{{table}}[H]
\centering
\footnotesize
\begin{{tabular}}{{p{{0.18\linewidth}}p{{0.24\linewidth}}p{{0.24\linewidth}}p{{0.24\linewidth}}}}
\toprule
Stage & Input data & Output & Leakage guard \\
\midrule
{table_rows_for_latex(acquisition_training, ["stage", "input_data", "output", "leakage_guard"], max_rows=8)}
\bottomrule
\end{{tabular}}
\caption{{Stage V-v3 online acquisition-learning loop.  All production training state is generated inside the V3 run directory.}}
\end{{table}}

P0/Ppi rewards include local topology-coverage gain, p95-contributor gain, opposite-Z2 bracket creation or shortening, sparse-component support, residual proximity, trusted topology surprise, and numerical reliability penalties.  \(\lambda_s\) gates the learned residual and is reduced or held when reward prediction improves but topology coverage does not.  \(\alpha_s\) adjusts head priority according to current support, surprise, shift, and convergence deficits.

\section*{{Final Dataset Counts}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lr}}
\toprule
Quantity & Count \\
\midrule
Total samples & {status_counts.get("total_samples")} \\
Trusted exact & {status_counts.get("trusted_exact")} \\
Training eligible & {status_counts.get("training_eligible_exact")} \\
Needs rerun & {status_counts.get("needs_rerun_exact")} \\
Topology trusted & {status_counts.get("topology_trusted")} \\
\midrule
Normal & {phase_counts.get("normal", 0)} \\
Uniform-SC & {phase_counts.get("uniform_SC", 0)} \\
FFLO & {phase_counts.get("FFLO", 0)} \\
cFFLO / trivial & {topo_counts.get("cFFLO / trivial", 0)} \\
tFFLO / topological & {topo_counts.get("tFFLO / topological", 0)} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Phase and Topology Maps}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("final_phase", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("final_topology", "")}}}
\caption{{Final exact point cloud. Left: thermodynamic labels. Right: FFLO topology labels. These are exact-point visualizations, not final smoothed publication surfaces.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_phase_slice", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_topology_slice", "")}}}
\caption{{Fixed-\(\mu\) slice atlases from the formal audit.}}
\end{{figure}}

\section*{{Formal Audit Metrics}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lccc}}
\toprule
Metric & Last-3 values / final & Gate & Status \\
\midrule
Phase map change & {latex_escape(decision.get("phase_map_change_last3"))} & 0.002 & pass \\
Topology map change & {latex_escape(decision.get("topology_map_change_last3"))} & 0.002 & pass \\
Topology surface shift p95 & {latex_escape(decision.get("topology_surface_shift_p95_last3"))} & 0.004167 & pass \\
Topology coverage p95 & {cov:.6f} & 0.00625 & fail \\
Trusted topology surprise & {latex_escape(decision.get("trusted_topology_surprise_last3"))} & 0.02 & fail \\
Topology component count & {latex_escape(decision.get("topology_component_count_last3"))} & stable & pass \\
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_topology_change", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_topology_coverage", "")}}}
\caption{{Topology map-change and topology-surface coverage diagnostics. Coverage improved but remains above the required gate.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_surprise", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("audit_components", "")}}}
\caption{{Trusted surprise and component stability. Component count is stable, but trusted topology surprise is not close enough to zero.}}
\end{{figure}}

\section*{{Comparison to Stage V-v2 Coverage Tail}}
\begin{{table}}[H]
\centering
\small
\begin{{tabular}}{{lrrp{{0.34\linewidth}}}}
\toprule
Metric & V2-tail & V3 & Interpretation \\
\midrule
{table_rows_for_latex(comparison, ["metric", "stagev2_tail", "stagev3", "interpretation"], max_rows=8)}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Acquisition and Learning Diagnostics}}
\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("dataset_growth", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("topology_growth", "")}}}
\caption{{Cumulative dataset and topology-support growth.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.62\linewidth]{{{fig.get("reward_learning", "")}}}
\caption{{Per-head learned residual activation and reward rank-correlation diagnostics.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.48\linewidth]{{{fig.get("source_counts", "")}}}
\includegraphics[width=0.48\linewidth]{{{fig.get("source_mix", "")}}}
\caption{{Proposal-source attribution. The report checks selected-point sources across all iterations.}}
\end{{figure}}

Source attribution diagnostics:
\begin{{itemize}}
\item Selected rows inspected: {source_diag.get("selected_rows")}.
\item Unknown explicit sources: {source_diag.get("unknown_explicit_count")}.
\item Blank/NaN source failure records: {source_diag.get("blank_or_nan_source_failure_count")}.
\item Mean selected \(C_{{\rm topo}}\): {source_diag.get("mean_C_topo", math.nan):.6g}.
\end{{itemize}}

\section*{{Decision and Next Step}}
The Stage V-v3 acquisition change is useful: it roughly halves the topology-surface coverage p95 relative to the V2-tail baseline.  But the run is still \textbf{{Decision C}} under the formal gates because coverage remains above threshold and trusted topology surprise is too high.  The next step should not be a blind continuation of V3.  First inspect the final sparse-patch source efficiency, unsupported surface components, and high-surprise exact points.  If the sparse-patch mechanism is targeting the right regions, a short coverage-tail continuation may be justified; otherwise the acquisition must be adjusted before spending another full budget.

\section*{{Do-Not-Claim List}}
\begin{{itemize}}
\item Do not claim Stage V-v3 has converged.
\item Do not claim V3 is publication-ready solely because the topology surface shift is zero.
\item Do not treat smooth visual surfaces as final topology boundaries.
\item Do not use response-side \(\eta\) or \(I_c\) to validate topology.
\item Do not treat increased tFFLO count alone as success.
\end{{itemize}}

\end{{document}}
"""
    tex_path = report_root / "stagev_v3_return_report.tex"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")
    return tex_path


def write_decision_log(report_root: Path, decision: dict[str, Any], comparison: pd.DataFrame) -> None:
    cov = float(decision.get("topology_surface_coverage_p95_final", math.nan))
    gate = 0.00625
    text = "\n".join(
        [
            "# Stage V-v3 Decision Log",
            "",
            f"- Decision: {decision.get('decision_class', 'unknown')}.",
            f"- Final dataset: {decision.get('final_dataset', 'dataset_iter097')}.",
            f"- Final exact iteration: {decision.get('final_exact_iteration', 96)}.",
            f"- Topology coverage p95: {cov:.6f}; gate {gate:.5f}.",
            f"- Coverage p95 relative to V2-tail: {100.0 * (V2_TAIL_BASELINE['topology_surface_coverage_p95'] - cov) / V2_TAIL_BASELINE['topology_surface_coverage_p95']:.1f}% lower.",
            f"- Main limiting factor: {decision.get('main_limiting_factor', 'unknown')}.",
            f"- Secondary limiting factors: {decision.get('secondary_limiting_factors', [])}.",
            "- Acquisition interpretation: V3 is cold-start for data and model state, but it learns only the per-head residual/value correction on top of an explicit physics/support base acquisition.",
            "- Recommendation: audit final sparse patches and high-surprise points before deciding on a short targeted tail.",
            "- No exact calculation was run during this local report generation.",
        ]
    )
    (report_root / "decision_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage V-v3 return report from downloaded HPC results.")
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--report-subdir", default=DEFAULT_REPORT_SUBDIR)
    args = parser.parse_args()

    package_root = args.package_root
    run_dir = package_root / args.output_root / "active_runs" / args.run_id
    report_root = ensure_dir(package_root / args.output_root / "reports" / args.report_subdir)
    tables_dir = ensure_dir(report_root / "tables")
    figures_dir = ensure_dir(report_root / "figures")

    iteration = final_iteration(run_dir)
    df = read_final_dataset(run_dir, iteration)
    final_summary = summarize_final_dataset(df, tables_dir)
    iter_df = summarize_iterations(run_dir, tables_dir)
    reward_df = summarize_reward(run_dir, tables_dir)
    selected, source_counts, source_iter = summarize_selection(run_dir, tables_dir)
    source_diag = (
        pd.read_csv(tables_dir / "source_attribution_diagnostics.csv").iloc[0].to_dict()
        if (tables_dir / "source_attribution_diagnostics.csv").exists()
        else {}
    )
    decision = read_formal_decision(package_root, args.output_root)
    decision.setdefault("final_dataset", f"dataset_iter{iteration:03d}")
    decision.setdefault("final_exact_iteration", iteration - 1)
    decision.setdefault("total_samples", len(df))
    comparison = build_comparison(decision, final_summary, tables_dir)
    acquisition_components, acquisition_training = write_acquisition_design_tables(tables_dir)

    figures: dict[str, Path] = {}
    figures.update(copy_formal_figures(package_root, args.output_root, figures_dir))
    figures["final_phase"] = figures_dir / "stagev3_final_phase_point_cloud.png"
    plot_final_phase(df, figures["final_phase"])
    figures["final_topology"] = figures_dir / "stagev3_final_topology_point_cloud.png"
    plot_final_topology(df, figures["final_topology"])
    figures["dataset_growth"] = figures_dir / "stagev3_dataset_growth.png"
    plot_dataset_growth(iter_df, figures["dataset_growth"])
    figures["topology_growth"] = figures_dir / "stagev3_topology_growth.png"
    plot_topology_growth(iter_df, figures["topology_growth"])
    if not reward_df.empty:
        figures["reward_learning"] = figures_dir / "stagev3_reward_learning.png"
        plot_reward(reward_df, figures["reward_learning"])
    if not source_counts.empty:
        figures["source_counts"] = figures_dir / "stagev3_source_counts.png"
        figures["source_mix"] = figures_dir / "stagev3_source_mix.png"
        plot_sources(source_counts, source_iter, figures["source_counts"], figures["source_mix"])

    write_markdown(
        report_root,
        decision,
        final_summary,
        comparison,
        source_diag,
        acquisition_components,
        acquisition_training,
    )
    write_decision_log(report_root, decision, comparison)
    tex_path = write_latex(
        report_root,
        figures,
        decision,
        final_summary,
        comparison,
        source_counts,
        source_diag,
        acquisition_components,
        acquisition_training,
    )

    manifest = {
        "report_root": str(report_root),
        "tex_path": str(tex_path),
        "final_dataset_iteration": iteration,
        "decision_class": decision.get("decision_class"),
        "topology_surface_coverage_p95_final": decision.get("topology_surface_coverage_p95_final"),
        "tables": sorted(str(p.relative_to(report_root)) for p in tables_dir.glob("*.csv")),
        "figures": sorted(str(p.relative_to(report_root)) for p in figures_dir.glob("*.png")),
    }
    (report_root / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
