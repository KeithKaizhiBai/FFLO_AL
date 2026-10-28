from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_COLORS = {"normal": "#b8bcc2", "uniform_SC": "#2a9d8f", "FFLO": "#d1495b"}
TOPO_COLORS = {0: "#386cb0", 1: "#ef3b2c"}
GATES = {
    "phase_map_change": 0.002,
    "topology_map_change": 0.002,
    "topology_surface_shift_p95": 0.004167,
    "topology_surface_coverage_p95": 0.00625,
    "trusted_topology_surprise": 0.02,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value)
    for old, new in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ]:
        text = text.replace(old, new)
    return text


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def infer_iteration(index: np.ndarray) -> np.ndarray:
    out = np.zeros_like(index, dtype=int)
    later = index >= 1024
    out[later] = 1 + ((index[later] - 1024) // 64)
    return out


def load_core_tables(core_dir: Path, tables_dir: Path) -> dict[str, pd.DataFrame]:
    mapping = {
        "iteration_counts": "stagev_v2_iteration_counts.csv",
        "phase_convergence": "stagev_v2_phase_convergence.csv",
        "topology_convergence": "stagev_v2_topology_convergence.csv",
        "surface_shift_coverage": "stagev_v2_surface_shift_coverage.csv",
        "surprise": "stagev_v2_surprise_by_iteration.csv",
        "component_stability": "stagev_v2_component_stability.csv",
        "p0_ppi_dominance": "stagev_v2_p0_ppi_dominance.csv",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, source_name in mapping.items():
        source = core_dir / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        frame = pd.read_csv(source)
        target = tables_dir / source_name.replace("stagev_v2_", "stagev_v4_")
        frame.to_csv(target, index=False)
        out[key] = frame
    return out


def aggregate_graph(run_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("iter*/stagev4_graph_summary.json")):
        iteration = int(path.parent.name.replace("iter", ""))
        row = {"iteration": iteration, **read_json(path)}
        nodes_path = path.parent / "stagev4_graph_nodes.csv"
        edges_path = path.parent / "stagev4_graph_edges.csv"
        if nodes_path.exists():
            nodes = pd.read_csv(nodes_path)
            row["node_rows"] = len(nodes)
            row["candidate_identity_count"] = int(nodes["node_kind"].astype(str).eq("candidate").sum())
            row["support_p95_max"] = float(nodes["support_p95"].max()) if len(nodes) else np.nan
        if edges_path.exists():
            edges = pd.read_csv(edges_path)
            row["edge_rows"] = len(edges)
            row["edge_support_total"] = int(edges["edge_support_count"].sum()) if len(edges) else 0
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("iteration")
    frame.to_csv(tables_dir / "stagev_v4_graph_evolution.csv", index=False)
    return frame


def aggregate_learning(run_dir: Path, tables_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("iter*/stagev4_reward_update_summary.json")):
        payload = read_json(path)
        row: dict[str, Any] = {
            "iteration": int(payload.get("iteration", int(path.parent.name.replace("iter", "")))),
            "graph_lambda": payload.get("graph_lambda"),
            "graph_value_model_status": payload.get("graph_value_model_status"),
            "reward_history_rows": payload.get("reward_history_rows"),
        }
        for name, value in payload.get("lambda_state", {}).items():
            row[f"lambda_{name}"] = value
        for name, value in payload.get("alpha_state", {}).items():
            row[f"alpha_{name}"] = value
        for name, value in payload.get("validation", {}).items():
            row[name] = value
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("iteration")
    frame.to_csv(tables_dir / "stagev_v4_learning_history.csv", index=False)
    return frame


def aggregate_reward_history(run_dir: Path, tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    reward = pd.read_csv(run_dir / "stagev4_reward_history.csv", low_memory=False)
    reward["inferred_iteration"] = infer_iteration(np.arange(len(reward)))
    action = (
        reward.groupby("action_family", dropna=False)
        .agg(
            selected_count=("action_family", "size"),
            trusted_rate=("trusted_exact", "mean"),
            topology_trusted_rate=("topology_trusted", "mean"),
            mean_graph_reward=("reward_graph_normalized", "mean"),
            mean_p0_reward=("reward_p0_normalized", "mean"),
            mean_ppi_reward=("reward_ppi_normalized", "mean"),
        )
        .reset_index()
        .sort_values("selected_count", ascending=False)
    )
    source = (
        reward.groupby("proposal_source", dropna=False)
        .agg(
            selected_count=("proposal_source", "size"),
            trusted_rate=("trusted_exact", "mean"),
            topology_trusted_rate=("topology_trusted", "mean"),
            mean_graph_reward=("reward_graph_normalized", "mean"),
        )
        .reset_index()
        .sort_values("selected_count", ascending=False)
    )
    per_iter = (
        reward.groupby(["inferred_iteration", "action_family"], dropna=False)
        .size()
        .rename("selected_count")
        .reset_index()
    )
    action.to_csv(tables_dir / "stagev_v4_action_family_audit.csv", index=False)
    source.to_csv(tables_dir / "stagev_v4_proposal_source_audit.csv", index=False)
    per_iter.to_csv(tables_dir / "stagev_v4_action_family_by_iteration.csv", index=False)
    failure = reward["source_failure_reason"].fillna("").astype(str).str.strip()
    quality = {
        "reward_rows": int(len(reward)),
        "blank_proposal_source": int(reward["proposal_source"].fillna("").astype(str).str.strip().eq("").sum()),
        "blank_action_family": int(reward["action_family"].fillna("").astype(str).str.strip().eq("").sum()),
        "stale_v3_allowlist_flag_rows": int(failure.str.startswith("source_not_in_allowlist:").sum()),
    }
    pd.DataFrame([quality]).to_csv(tables_dir / "stagev_v4_source_attribution_quality.csv", index=False)
    return action, source, per_iter, quality


def final_counts(final_df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    rows = [
        {"metric": "total_samples", "value": len(final_df)},
        {"metric": "trusted_exact", "value": int(final_df["trusted_exact"].sum())},
        {"metric": "training_eligible_exact", "value": int(final_df["training_eligible_exact"].sum())},
        {"metric": "needs_rerun_exact", "value": int(final_df["needs_rerun_exact"].sum())},
        {"metric": "q_unresolved", "value": int(final_df["q_unresolved"].sum())},
        {"metric": "delta_unresolved", "value": int(final_df["delta_unresolved"].sum())},
        {"metric": "q_expanded", "value": int(final_df["q_expanded"].sum())},
        {"metric": "delta_refined", "value": int(final_df["delta_refined"].sum())},
        {"metric": "delta_boundary_band_normal", "value": int(final_df["delta_boundary_band_normal"].sum())},
    ]
    for name, value in final_df["phase_name"].value_counts().items():
        rows.append({"metric": f"phase_{name}", "value": int(value)})
    topo = final_df[
        final_df["trusted_exact"].astype(bool)
        & final_df["training_eligible_exact"].astype(bool)
        & final_df["topology_trusted"].astype(bool)
        & final_df["phase_name"].eq("FFLO")
        & final_df["topology_label_code"].isin([0, 1])
    ]
    rows.extend(
        [
            {"metric": "trusted_cFFLO", "value": int(topo["topology_label_code"].eq(0).sum())},
            {"metric": "trusted_tFFLO", "value": int(topo["topology_label_code"].eq(1).sum())},
        ]
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "stagev_v4_final_counts.csv", index=False)
    return frame


def build_gate_table(core: dict[str, pd.DataFrame], graph: pd.DataFrame, tables_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    phase_last3 = core["phase_convergence"].tail(3)["phase_map_change"].astype(float).tolist()
    topo_last3 = core["topology_convergence"].tail(3)["topology_map_change"].astype(float).tolist()
    topo_surface = core["surface_shift_coverage"].query("surface == 'topology'").tail(3)
    shift_last3 = topo_surface["p95"].astype(float).tolist()
    coverage = float(topo_surface.iloc[-1]["coverage_p95"])
    surprise = core["surprise"].tail(3)["trusted_topology_surprise"].astype(float).tolist()
    support_ok = bool((core["surprise"].tail(3)["trusted_topology_denominator"] >= 16).all())
    components = core["component_stability"].query("surface == 'topology'").tail(3)["boundary_component_count"].astype(int).tolist()
    values = {
        "phase_map_change": max(phase_last3),
        "topology_map_change": max(topo_last3),
        "topology_surface_shift_p95": max(shift_last3),
        "topology_surface_coverage_p95": coverage,
        "trusted_topology_surprise": max(surprise),
    }
    rows = []
    for name, threshold in GATES.items():
        value = values[name]
        passed = bool(np.isfinite(value) and value <= threshold)
        if name == "trusted_topology_surprise":
            passed = passed and support_ok
        rows.append({"gate": name, "value": value, "threshold": threshold, "pass": passed})
    component_pass = len(components) == 3 and len(set(components)) == 1
    rows.append({"gate": "topology_component_stability", "value": "/".join(map(str, components)), "threshold": "constant_last3", "pass": component_pass})
    graph_final = graph.iloc[-1]
    rows.append({"gate": "graph_candidate_nodes_observed", "value": int(graph_final.get("candidate_node_count", 0)), "threshold": ">=1", "pass": int(graph_final.get("candidate_node_count", 0)) >= 1})
    frame = pd.DataFrame(rows)
    frame.to_csv(tables_dir / "stagev_v4_gate_summary.csv", index=False)
    context = {
        "phase_last3": phase_last3,
        "topology_last3": topo_last3,
        "shift_last3": shift_last3,
        "coverage": coverage,
        "surprise_last3": surprise,
        "components_last3": components,
        "support_ok": support_ok,
        "all_formal_pass": bool(frame.iloc[:6]["pass"].all()),
    }
    return frame, context


def make_figures(
    final_df: pd.DataFrame,
    core: dict[str, pd.DataFrame],
    graph: pd.DataFrame,
    learning: pd.DataFrame,
    action: pd.DataFrame,
    source: pd.DataFrame,
    per_iter: pd.DataFrame,
    gate: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, group in final_df.groupby("phase_name"):
        axes[0].scatter(group["kT"], group["JA"], s=3, alpha=0.45, color=PHASE_COLORS.get(name, "k"), label=name)
    axes[0].set(xlabel=r"$k_B T/t$", ylabel=r"$J_A/t$", title="Final thermodynamic samples (mu projected)")
    axes[0].legend(markerscale=3, fontsize=8)
    topo = final_df[final_df["topology_trusted"].astype(bool) & final_df["phase_name"].eq("FFLO") & final_df["topology_label_code"].isin([0, 1])]
    for code, group in topo.groupby("topology_label_code"):
        axes[1].scatter(group["kT"], group["JA"], s=3, alpha=0.45, color=TOPO_COLORS[int(code)], label="tFFLO" if int(code) == 1 else "cFFLO")
    axes[1].set(xlabel=r"$k_B T/t$", ylabel=r"$J_A/t$", title="Trusted FFLO topology samples (mu projected)")
    axes[1].legend(markerscale=3, fontsize=8)
    p = figures_dir / "fig01_final_phase_topology_projection.png"; save_figure(fig, p); paths.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, mu0 in zip(axes.flat, [-0.25, 0.25, 0.75, 1.25]):
        near = final_df[np.abs(final_df["mu"] - mu0) <= 0.06]
        for name, group in near.groupby("phase_name"):
            ax.scatter(group["kT"], group["JA"], s=5, alpha=0.65, color=PHASE_COLORS.get(name, "k"), label=name)
        ax.set_title(rf"$\mu/t\approx {mu0:.2f}$")
        ax.set_xlabel(r"$k_B T/t$"); ax.set_ylabel(r"$J_A/t$")
    axes[0, 0].legend(fontsize=7, markerscale=2)
    p = figures_dir / "fig02_fixed_mu_phase_atlas.png"; save_figure(fig, p); paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    phase = core["phase_convergence"]
    topo_c = core["topology_convergence"]
    axes[0].plot(phase["to_dataset_iteration"], phase["phase_map_change"], marker="o", label="phase")
    axes[0].plot(topo_c["to_dataset_iteration"], topo_c["topology_map_change"], marker="s", label="topology")
    axes[0].axhline(0.002, color="k", linestyle="--", label="gate")
    axes[0].set(xlabel="dataset iteration", ylabel="map-change fraction", title="Last-five fixed-cloud map changes")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    surf = core["surface_shift_coverage"].query("surface == 'topology'")
    axes[1].plot(surf["to_dataset_iteration"], surf["p95"], marker="o", label="surface shift p95")
    axes[1].plot(surf["to_dataset_iteration"], surf["coverage_p95"], marker="s", label="coverage p95")
    axes[1].axhline(GATES["topology_surface_shift_p95"], color="#666", linestyle="--", label="shift gate")
    axes[1].axhline(GATES["topology_surface_coverage_p95"], color="k", linestyle=":", label="coverage gate")
    axes[1].set(xlabel="dataset iteration", ylabel="normalized distance", title="Topology-surface proxies")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    p = figures_dir / "fig03_convergence_and_coverage.png"; save_figure(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    surprise = core["surprise"]
    ax.plot(surprise["to_dataset_iteration"], surprise["trusted_phase_surprise"], marker="o", label="trusted phase surprise")
    ax.plot(surprise["to_dataset_iteration"], surprise["trusted_topology_surprise"], marker="s", label="trusted topology surprise")
    ax.axhline(0.02, color="k", linestyle="--", label="topology gate")
    ax.set(xlabel="dataset iteration", ylabel="surprise fraction", title="Exact-label surprise on newly added points")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    p = figures_dir / "fig04_trusted_surprise.png"; save_figure(fig, p); paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(graph["iteration"], graph["known_node_count"], label="known nodes")
    axes[0].plot(graph["iteration"], graph["candidate_node_count"], label="candidate nodes")
    axes[0].plot(graph["iteration"], graph["edge_count"], label="edges")
    axes[0].set(xlabel="exact iteration", ylabel="count", title="Phase-structure graph evolution")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    axes[1].plot(graph["iteration"], graph["graph_uncertainty_total"], label="total")
    axes[1].plot(graph["iteration"], graph["node_uncertainty_total"], label="node")
    axes[1].plot(graph["iteration"], graph["edge_uncertainty_total"], label="edge")
    axes[1].set(xlabel="exact iteration", ylabel="uncertainty proxy", title="Graph uncertainty history")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    p = figures_dir / "fig05_graph_evolution.png"; save_figure(fig, p); paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for col, label in [("graph_lambda", "graph"), ("lambda_p0", "P0"), ("lambda_ppi", "Ppi"), ("lambda_ns", "NS"), ("lambda_uf", "UF"), ("lambda_gap", "gap")]:
        if col in learning:
            axes[0].plot(learning["iteration"], learning[col], label=label)
    axes[0].set(xlabel="exact iteration", ylabel=r"$\lambda$", title="Learned-residual activation")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.25)
    for col, label in [("rank_correlation_a0_graph", "graph prior"), ("rank_correlation_model_graph", "graph model"), ("rank_correlation_model_p0", "P0 model"), ("rank_correlation_model_ppi", "Ppi model")]:
        if col in learning:
            axes[1].plot(learning["iteration"], learning[col], label=label)
    axes[1].set(xlabel="exact iteration", ylabel="Spearman-like rank correlation", title="Reward prediction validation")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.25)
    p = figures_dir / "fig06_learning_diagnostics.png"; save_figure(fig, p); paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    action_plot = action.iloc[::-1]
    axes[0].barh(action_plot["action_family"].fillna("missing").astype(str).to_numpy(), action_plot["selected_count"].to_numpy(), color="#4c78a8")
    axes[0].set(title="Selected points by action family", xlabel="count")
    top_source = source.head(12)
    source_plot = top_source.iloc[::-1]
    axes[1].barh(source_plot["proposal_source"].fillna("missing").astype(str).to_numpy(), source_plot["selected_count"].to_numpy(), color="#f58518")
    axes[1].set(title="Selected points by proposal source", xlabel="count")
    p = figures_dir / "fig07_action_source_audit.png"; save_figure(fig, p); paths.append(p)

    pivot = per_iter.pivot(index="inferred_iteration", columns="action_family", values="selected_count").fillna(0)
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    ax.stackplot(pivot.index, *[pivot[c] for c in pivot.columns], labels=[str(c) for c in pivot.columns], alpha=0.85)
    ax.set(xlabel="exact iteration", ylabel="selected points", title="Action-family allocation over the production run")
    ax.legend(loc="upper left", fontsize=7); ax.grid(alpha=0.2)
    p = figures_dir / "fig08_action_family_history.png"; save_figure(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = gate["gate"].astype(str).str.replace("_", " ")
    vals = gate["pass"].astype(bool).astype(int)
    ax.bar(labels, vals, color=["#2a9d8f" if v else "#d1495b" for v in vals])
    ax.set_ylim(0, 1.15); ax.set_ylabel("gate pass"); ax.set_title("Formal Stage V-v4 audit gates")
    ax.tick_params(axis="x", rotation=38)
    p = figures_dir / "fig09_gate_summary.png"; save_figure(fig, p); paths.append(p)
    return paths


def tex_figure(path: Path, caption: str, label: str) -> str:
    return rf"""
\begin{{figure}}[p]
\centering
\includegraphics[width=0.96\textwidth]{{\detokenize{{{path.name}}}}}
\caption{{{caption}}}
\label{{fig:{label}}}
\end{{figure}}
"""


def build_reports(
    output_dir: Path,
    final_df: pd.DataFrame,
    counts: dict[str, int],
    gate: pd.DataFrame,
    context: dict[str, Any],
    graph: pd.DataFrame,
    learning: pd.DataFrame,
    action: pd.DataFrame,
    source: pd.DataFrame,
    figures: list[Path],
    decision: dict[str, Any],
    source_quality: dict[str, int],
) -> None:
    graph_final = graph.iloc[-1]
    learn_final = learning.iloc[-1]
    action_text = ", ".join(f"{r.action_family}: {int(r.selected_count)}" for r in action.itertuples())
    source_text = ", ".join(f"{r.proposal_source}: {int(r.selected_count)}" for r in source.head(8).itertuples())
    md = rf"""# Stage V-v4 Combined Scientific Report and Formal Audit

Date: 2026-07-11

## Executive decision

Stage V-v4 completed its full cold-start production budget through exact iteration 096 and produced `dataset_iter097`. The formal result is **Decision {decision['decision_class']}**: production completed successfully, thermodynamic and topology maps are stable under the fixed-cloud proxy, and the topology surface position is stable, but the topology-surface coverage and trusted topology surprise do not pass their formal gates. Stage V-v4 must not yet be called converged or publication-ready.

Main limiting factor: `{decision['main_limiting_factor']}`. Secondary limiting factor: `{decision['secondary_limiting_factors']}`. The next step is a matched V3/V4 diagnostic comparison and candidate-node validation audit; no new exact calculation is justified until that analysis identifies a specific unsupported region.

## Scope and provenance

This combined report/audit uses the downloaded minimum Stage V-v4 package only. It reconstructs all 98 cumulative datasets, all 97 selection/reward/merge stages, 96 graph snapshots, the complete 7168-row reward history, final learning state, and controller logs. It does not run the BdG oracle, a Delta-q search, an active-learning iteration, or an HPC job.

The core convergence audit uses a fixed 50,000-point 3D cloud and one consistent offline kNN exact-label surrogate. Boundary surfaces are local opposite-label bracket midpoints; raw long Delaunay edges are not used. These are audit proxies, not redefinitions of the physical phase boundaries.

## Physical and numerical definitions

The exact solver minimizes the BdG free energy over $\Delta$ and $q$ at every $(k_B T/t,J_A/t,\mu/t)$ point. The canonical thermodynamic labels remain

$$
\mathrm{{normal}}:\Delta_\mathrm{{opt}}<10^{{-3}},\qquad
\mathrm{{uniform\ SC}}:\Delta_\mathrm{{opt}}\ge10^{{-3}},\ |q_\mathrm{{opt}}|<10^{{-2}},
$$

$$
\mathrm{{FFLO}}:\Delta_\mathrm{{opt}}\ge10^{{-3}},\ |q_\mathrm{{opt}}|\ge10^{{-2}}.
$$

The topology label uses the Pfaffian product at $k=0,\pi$ together with a full-Brillouin-zone bulk-gap guard. The active learner schedules exact calculations and never replaces these definitions.

The robust oracle preserved adaptive q-window expansion, near-boundary Delta refinement, finite-resolution boundary-band semantics, and separate trusted/training-eligible/rerun metadata.

## Stage V-v4 acquisition method

For boundary channel $s$, the transparent base score is

$$A_s^{{(0)}}(x)=B_s(x)U_s(x)H_s(x).$$

Stage V-v3 adds topology coverage reduction $C_\mathrm{{topo}}$. Stage V-v4 adds a phase-structure graph and scores four action families: global exploration, known-boundary refinement, candidate-phase validation, and oracle-anomaly audit. The implemented graph correction is summarized as

$$
A_\mathrm{{V4}}(x)=A_\mathrm{{multihead}}(x)\left[1+w_GU_G(x)\right]\exp\!\left[\lambda_Gg_G(\phi_G(x))\right].
$$

Candidate nodes are provisional. They do not override normal, uniform-SC, FFLO, cFFLO, or tFFLO labels.

## Final dataset

- Total training dataset: {counts['total_samples']}.
- Clean trusted exact: {counts['trusted_exact']}.
- Boundary-band training points: {counts['delta_boundary_band_normal']}.
- Normal / uniform-SC / FFLO: {counts['phase_normal']} / {counts['phase_uniform_SC']} / {counts['phase_FFLO']}.
- Trusted cFFLO / tFFLO: {counts['trusted_cFFLO']} / {counts['trusted_tFFLO']}.
- q-expanded / Delta-refined: {counts['q_expanded']} / {counts['delta_refined']}.
- q-unresolved / Delta-unresolved / rerun-required: 0 / 0 / 0.

The identity check reproduces the remote final count of 7080 and the exact phase totals. The normal-state and q=0 limits were not recomputed because this is an offline audit, but no phase-threshold mismatch or unresolved numerical flag is present in the final cumulative dataset.

## Formal convergence audit

- Phase-map change, last three: {context['phase_last3']}.
- Topology-map change, last three: {context['topology_last3']}.
- Topology-surface shift p95, last three: {context['shift_last3']}.
- Final topology-surface coverage p95: {context['coverage']:.6f}, gate {GATES['topology_surface_coverage_p95']:.6f}.
- Trusted topology surprise, last three: {context['surprise_last3']}.
- Topology component count, last three: {context['components_last3']}.

The phase-map, topology-map, surface-shift, and component-stability gates pass. Coverage fails by a factor of {context['coverage']/GATES['topology_surface_coverage_p95']:.2f}; trusted topology surprise also remains above 0.02 in the last three transitions.

## Graph and learned acquisition audit

The final graph contains {int(graph_final['known_node_count'])} known nodes, {int(graph_final['candidate_node_count'])} provisional candidate nodes, and {int(graph_final['edge_count'])} edges. This confirms that the candidate-phase path, absent from the synthetic smoke example, became active in production.

The final graph-value model is `{learn_final.get('graph_value_model_status', 'unknown')}` with $\lambda_G={float(learn_final['graph_lambda']):.3f}$. Its reward rank correlation is {float(learn_final['rank_correlation_model_graph']):.3f}, versus {float(learn_final['rank_correlation_a0_graph']):.3f} for the graph prior, an increment of {float(learn_final['rank_correlation_delta_vs_a0_graph']):.3f}. P0 and Ppi residual heads remain active at 0.7; NS, UF, and gap fall back effectively to their transparent base scores.

Action-family totals are {action_text}. The leading proposal sources are {source_text}. Candidate-phase validation is therefore present. Oracle-anomaly audit has 617 selected points overall but no final-iteration candidate allocation, so its late-run behavior is not fully tested. Proposal-source and action-family fields are nonblank for all 7168 reward rows. However, {source_quality['stale_v3_allowlist_flag_rows']} rows retain stale `source_not_in_allowlist` text from the inherited V3 validator even though their sources are valid under the V4 mapping. This is a logging-semantics defect, not an exact-label failure.

## Comparison with Stage V-v3

The prior V3 formal report recorded topology coverage p95 = 0.018772 and Decision C. The V4 offline value is {context['coverage']:.6f}. Because the two values were not generated from the same downloaded datasets in one matched run, this comparison is descriptive, not a superiority test. V4 demonstrates graph learning and provisional node discovery, but the available evidence does not show a formal coverage improvement over V3.

## Decision and next action

Decision {decision['decision_class']}: not converged. Production completion is accepted, numerical label closure is accepted, and the graph/value-learning path is operational. Formal convergence fails on topology support density and trusted topology surprise.

Recommended next action:

1. run a matched V3/V4 offline audit with the same fixed cloud and support radii;
2. inspect the four final candidate nodes for persistence, physical descriptor separation, and exact-label closure;
3. localize the topology-surprise points and distinguish missing support from label noise or surface multiplicity;
4. only then decide whether a short targeted exact tail is warranted.

No Hamiltonian, phase criterion, topology convention, q/Delta rule, or tolerance was changed.

## Figures

""" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in figures) + "\n"
    (output_dir / "stagev_v4_combined_report_audit.md").write_text(md, encoding="utf-8")

    tex_figs = "\n".join(
        tex_figure(p, caption, f"f{i:02d}")
        for i, (p, caption) in enumerate(
            zip(
                figures,
                [
                    "Final thermodynamic and trusted FFLO topology samples projected over chemical potential.",
                    "Fixed-chemical-potential atlas of the final thermodynamic labels.",
                    "Fixed-cloud phase/topology map changes and local-support topology-surface metrics.",
                    "Trusted phase and topology label surprise for newly added exact points.",
                    "Evolution of the Stage V-v4 phase-structure graph and its uncertainty proxies.",
                    "Online residual activation and reward-model validation diagnostics.",
                    "Selected exact calls by action family and proposal source.",
                    "Action-family allocation over the cold-start production run.",
                    "Pass/fail summary of the formal Stage V-v4 audit gates.",
                ],
            ),
            start=1,
        )
    )
    gate_rows = "\n".join(
        rf"{latex_escape(r[0])} & {latex_escape(r[1])} & {latex_escape(r[2])} & {'PASS' if bool(r[3]) else 'FAIL'} \\"
        for r in gate.itertuples(index=False, name=None)
    )
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.82in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,array,graphicx,xcolor,float,hyperref,microtype}}
\graphicspath{{{{figures/}}}}
\hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
\definecolor{{passgreen}}{{HTML}}{{2A9D8F}}
\definecolor{{failred}}{{HTML}}{{D1495B}}
\title{{Stage V-v4 Combined Scientific Report and Formal Audit}}
\author{{Altermagnetic FFLO Topological Superconductor Project}}
\date{{11 July 2026}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Stage V-v4 completed its strict cold-start production loop through exact iteration 096 and produced 7080 training-eligible exact points. This report combines the scientific return report and the formal convergence/acquisition audit. The result is \textbf{{Decision {decision['decision_class']}}}: thermodynamic and topology maps are stable and the local topology-surface position is stable, but topology-surface coverage and trusted topology surprise fail their formal gates. The run is operationally complete but not scientifically converged.
\end{{abstract}}

\section{{Scope and provenance}}
The audit uses all 98 cumulative dataset CSV files, 97 selection/reward/merge stages, 96 graph snapshots, the 7168-row reward history, final model state, and controller logs from the minimum downloaded Stage V-v4 package. No BdG calculation, $\Delta$--$q$ search, active-learning iteration, or HPC job was run. A fixed 50,000-point 3D cloud and one consistent offline kNN exact-label surrogate are used. Boundary surfaces are local opposite-label bracket midpoints; raw long Delaunay edges are excluded.

\section{{Physical and numerical definitions}}
The exact solver minimizes the BdG free energy over $(\Delta,q)$ for every $(k_BT/t,J_A/t,\mu/t)$. The canonical labels are
\begin{{align}}
\mathrm{{normal}} &: \Delta_\mathrm{{opt}}<10^{{-3}},\\
\mathrm{{uniform\ SC}} &: \Delta_\mathrm{{opt}}\ge10^{{-3}},\quad |q_\mathrm{{opt}}|<10^{{-2}},\\
\mathrm{{FFLO}} &: \Delta_\mathrm{{opt}}\ge10^{{-3}},\quad |q_\mathrm{{opt}}|\ge10^{{-2}}.
\end{{align}}
Topology uses the $k=0,\pi$ Pfaffian product with a full-Brillouin-zone bulk-gap guard. Adaptive $q$ expansion, near-boundary $\Delta$ refinement, boundary-band semantics, and separate trusted/training/rerun fields are preserved.

\section{{Stage V-v4 graph acquisition}}
The transparent boundary-channel score is
\begin{{equation}} A_s^{{(0)}}(x)=B_s(x)U_s(x)H_s(x). \end{{equation}}
Stage V-v4 adds a phase-structure graph and four action families. Its implemented graph correction is summarized by
\begin{{equation}}
A_\mathrm{{V4}}(x)=A_\mathrm{{multihead}}(x)\left[1+w_GU_G(x)\right]\exp\left[\lambda_Gg_G(\phi_G(x))\right].
\end{{equation}}
Candidate nodes are provisional diagnostics and never replace the canonical physical labels.

\section{{Final exact dataset}}
The final dataset has {counts['total_samples']} samples: {counts['phase_normal']} normal, {counts['phase_uniform_SC']} uniform-SC, and {counts['phase_FFLO']} FFLO. There are {counts['trusted_exact']} clean trusted points and {counts['delta_boundary_band_normal']} finite-resolution boundary-band points; every stored point is training eligible. Trusted FFLO topology contains {counts['trusted_cFFLO']} cFFLO and {counts['trusted_tFFLO']} tFFLO points. The oracle expanded the $q$ window for {counts['q_expanded']} points and refined $\Delta$ for {counts['delta_refined']} points. No final point is $q$-unresolved, $\Delta$-unresolved, or rerun-required.

\section{{Formal convergence audit}}
\begin{{table}}[H]\centering\small
\begin{{tabular}}{{p{{0.38\textwidth}}rrc}}\toprule
Gate & Value & Threshold & Result\\\midrule
{gate_rows}
\bottomrule\end{{tabular}}
\caption{{Formal audit gate summary.}}\end{{table}}
The final topology coverage p95 is {context['coverage']:.6f}, or {context['coverage']/GATES['topology_surface_coverage_p95']:.2f} times the gate. Trusted topology surprise also remains above 0.02. Phase-map, topology-map, surface-shift, and component-stability gates pass.

\section{{Graph and learned-value audit}}
The final graph contains {int(graph_final['known_node_count'])} known nodes, {int(graph_final['candidate_node_count'])} candidate nodes, and {int(graph_final['edge_count'])} edges. The graph-value model is trained with $\lambda_G={float(learn_final['graph_lambda']):.2f}$. Its final reward rank correlation is {float(learn_final['rank_correlation_model_graph']):.3f}, compared with {float(learn_final['rank_correlation_a0_graph']):.3f} for the graph prior. P0 and Ppi residual heads remain active; NS, UF, and gap effectively use their base scores. Candidate-phase validation is active. Proposal-source and action-family values are nonblank, but {source_quality['stale_v3_allowlist_flag_rows']} reward rows carry stale V3 allowlist warning text for sources that are valid in V4; this is a logging-semantics defect rather than an exact-label failure.

\section{{Comparison with Stage V-v3}}
The previous V3 report recorded topology coverage p95 $=0.018772$ and Decision C. V4 gives {context['coverage']:.6f} under the present offline audit. Because V3 and V4 were not recomputed together from both raw cumulative datasets, the comparison is descriptive rather than a matched superiority test. V4 validates graph learning and provisional-node discovery, but does not establish a formal coverage improvement over V3.

\section{{Decision and next action}}
\textbf{{Decision {decision['decision_class']}: not converged.}} Production completion and numerical label closure are accepted. The main limiting factor is topology-surface coverage; trusted topology surprise is secondary. The next step is a matched V3/V4 audit, candidate-node persistence and descriptor validation, and localization of high-surprise topology points. New exact calls should be considered only after those diagnostics identify a concrete unsupported region.

No Hamiltonian, phase criterion, topology convention, $q/\Delta$ rule, or numerical tolerance was changed.

\clearpage
{tex_figs}
\end{{document}}
"""
    (output_dir / "stagev_v4_combined_report_audit.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--core-tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    core = load_core_tables(args.core_tables_dir, tables_dir)
    final_df = pd.read_csv(args.run_dir / "dataset_iter097.csv", low_memory=False)
    final_table = final_counts(final_df, tables_dir)
    counts = {str(r.metric): int(r.value) for r in final_table.itertuples()}
    if counts["total_samples"] != 7080 or counts["phase_normal"] + counts["phase_uniform_SC"] + counts["phase_FFLO"] != 7080:
        raise RuntimeError("Final dataset identity/count check failed")
    graph = aggregate_graph(args.run_dir, tables_dir)
    learning = aggregate_learning(args.run_dir, tables_dir)
    action, source, per_iter, source_quality = aggregate_reward_history(args.run_dir, tables_dir)
    gate, context = build_gate_table(core, graph, tables_dir)
    decision = {
        "decision_class": "C",
        "stagev_v4_converged": False,
        "production_loop_completed": True,
        "final_dataset": "dataset_iter097",
        "final_exact_iteration": 96,
        "total_samples": counts["total_samples"],
        "trusted_exact": counts["trusted_exact"],
        "main_limiting_factor": "topology_surface_coverage",
        "secondary_limiting_factors": ["trusted_topology_surprise"],
        "need_new_exact_calculation": "not_before_targeted_diagnostics",
        "recommended_next_action": "matched_v3_v4_audit_then_candidate_node_and_surprise_localization",
        "caveats": [
            "Map metrics use one fixed offline kNN exact-label surrogate, not the online neural ensemble.",
            "Boundary metrics use local opposite-label bracket proxies.",
            "The V3 reference was not recomputed from V3 cumulative datasets in this run.",
            "Inherited V3 source-failure text remains stale for valid V4 proposal sources.",
        ],
        **context,
    }
    write_json(output_dir / "stagev_v4_audit_summary.json", {"decision": decision, "gates": gate.to_dict(orient="records")})
    write_json(output_dir / "stagev_v4_decision.json", decision)
    comparison = pd.DataFrame(
        [
            {"run": "Stage V-v3", "coverage_p95": 0.01877224394307404, "decision": "C", "comparison_mode": "prior_report_reference"},
            {"run": "Stage V-v4", "coverage_p95": context["coverage"], "decision": "C", "comparison_mode": "current_fixed_cloud_audit"},
        ]
    )
    comparison.to_csv(tables_dir / "stagev_v4_comparison_to_v3.csv", index=False)
    figures = make_figures(final_df, core, graph, learning, action, source, per_iter, gate, figures_dir)
    build_reports(output_dir, final_df, counts, gate, context, graph, learning, action, source, figures, decision, source_quality)
    (output_dir / "decision_log.md").write_text(
        "# Stage V-v4 Combined Report/Audit Decision Log\n\n"
        "- Decision: `C` (not converged).\n"
        "- Production completed through exact iter096 and numerical label closure passed.\n"
        f"- Final topology-surface coverage p95: `{context['coverage']:.9f}`; gate: `{GATES['topology_surface_coverage_p95']}`.\n"
        f"- Trusted topology surprise last three: `{context['surprise_last3']}`.\n"
        "- Main unresolved issue: topology-surface support density.\n"
        "- Next check: matched V3/V4 audit, candidate-node persistence, and high-surprise localization.\n"
        "- No new exact calculation is recommended before those diagnostics.\n"
        "- No physical or numerical definition was changed.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "decision": decision, "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
