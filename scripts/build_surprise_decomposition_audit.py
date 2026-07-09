from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "rankcap_k3_full_loop"
    / "ML_Phase_512_RankCapK3_FullLoop"
    / "active_runs"
    / "active_boundary_discovery_rankcap_k3_full_loop_v1"
)
OUT_DIR = ROOT / "reports" / "surprise_decomposition_audit"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_NAME = "surprise_decomposition_audit"

PHASE_NAMES = {
    0: "normal",
    1: "uniform_SC",
    2: "FFLO",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return ""
        return repr(float(value))
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_scalar(row.get(k, "")) for k in fields})


def run_config() -> dict[str, Any]:
    return read_json(RUN_DIR / "run_config.json", {}).get("active_learning_config", {})


def phase_label(delta: np.ndarray, q: np.ndarray, delta_eps: float, q_eps: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    out = np.full(delta.shape, 2, dtype=np.int64)
    out[delta < delta_eps] = 0
    out[(delta >= delta_eps) & (np.abs(q) < q_eps)] = 1
    return out


def npz_frame(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as z:
        cols: dict[str, np.ndarray] = {}
        n = None
        for key in z.files:
            arr = z[key]
            if arr.ndim != 1:
                continue
            if n is None:
                n = int(arr.shape[0])
            if int(arr.shape[0]) == n:
                cols[key] = arr
    return pd.DataFrame(cols)


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_kT_key"] = pd.to_numeric(out["kT"], errors="coerce").round(12)
    out["_JA_key"] = pd.to_numeric(out["JA"], errors="coerce").round(12)
    return out


def phase_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    try:
        return PHASE_NAMES.get(int(value), str(int(value)))
    except Exception:
        return str(value)


def iteration_numbers() -> list[int]:
    numbers = []
    for path in sorted(RUN_DIR.glob("iter[0-9][0-9][0-9]")):
        try:
            it = int(path.name.replace("iter", ""))
        except ValueError:
            continue
        if (path / f"selected_points_by_pool.csv").exists() and (path / f"exact_merged_iter{it:03d}.npz").exists():
            numbers.append(it)
    return numbers


def load_iteration(iteration: int, cfg: dict[str, Any]) -> pd.DataFrame:
    iter_dir = RUN_DIR / f"iter{iteration:03d}"
    selected = add_keys(pd.read_csv(iter_dir / "selected_points_by_pool.csv"))
    exact = add_keys(npz_frame(iter_dir / f"exact_merged_iter{iteration:03d}.npz"))

    delta_eps = float(cfg.get("delta_eps", 1.0e-3))
    q_eps = float(cfg.get("q_eps", 1.0e-2))
    exact["exact_phase_label_stop"] = phase_label(exact["delta_opt"], exact["q_opt"], delta_eps, q_eps)
    exact_cols = [
        "_kT_key",
        "_JA_key",
        "delta_opt",
        "q_opt",
        "eta",
        "phase_candidate",
        "exact_phase_label_stop",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "needs_rerun_exact",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "delta_boundary_ambiguous",
        "delta_refined",
        "delta_unresolved",
        "q_expansion_count",
        "q_expansion_trigger",
        "q_edge_distance",
        "positive_delta_gap",
        "local_boxes_refined_count",
        "selected_refine_target_count",
        "local_refinement_runtime_sec",
        "point_total_runtime_sec",
        "rank",
    ]
    exact_cols = [c for c in exact_cols if c in exact.columns]
    joined = selected.merge(
        exact[exact_cols],
        on=["_kT_key", "_JA_key"],
        how="left",
        validate="one_to_one",
    )
    joined.insert(0, "iteration", iteration)
    joined["predicted_phase_name"] = joined.get("predicted_phase_before_exact", pd.Series(dtype=float)).map(phase_name)
    joined["exact_phase_name"] = joined.get("exact_phase_label_stop", pd.Series(dtype=float)).map(phase_name)
    joined["phase_transition"] = joined["predicted_phase_name"] + "_to_" + joined["exact_phase_name"]
    joined["label_surprise"] = (
        joined["exact_phase_label_stop"].notna()
        & joined["predicted_phase_before_exact"].notna()
        & (joined["exact_phase_label_stop"].astype(float) != joined["predicted_phase_before_exact"].astype(float))
    )
    joined["qedge_or_expanded"] = (
        (pd.to_numeric(joined.get("q_edge_hit", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(joined.get("q_expanded", 0), errors="coerce").fillna(0) > 0)
    )
    joined["rerun_any"] = (
        (pd.to_numeric(joined.get("rerun_required", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(joined.get("needs_rerun_exact", 0), errors="coerce").fillna(0) > 0)
    )
    joined["boundary_distance_bin"] = pd.cut(
        pd.to_numeric(joined.get("selected_to_predicted_boundary_distance", np.nan), errors="coerce"),
        bins=[-np.inf, 0.003125, 0.00625, 0.0125, 0.025, 0.05, np.inf],
        labels=["<=0.5tol", "0.5-1tol", "1-2tol", "2-4tol", "4-8tol", ">8tol"],
    ).astype(str)
    kt_min = float(cfg.get("kt_min", 0.0))
    kt_max = float(cfg.get("kt_max", 0.56))
    ja_min = float(cfg.get("ja_min", 0.0))
    ja_max = float(cfg.get("ja_max", 2.12))
    joined["kT_bin"] = pd.cut(
        joined["kT"],
        bins=np.linspace(kt_min, kt_max, 17),
        include_lowest=True,
        labels=False,
    )
    joined["JA_bin"] = pd.cut(
        joined["JA"],
        bins=np.linspace(ja_min, ja_max, 17),
        include_lowest=True,
        labels=False,
    )
    return joined


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return math.nan
    return float(values.mean())


def safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return math.nan
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def summarize_iterations(all_points: pd.DataFrame, iterations: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for it in iterations:
        group = all_points[all_points["iteration"] == it]
        stop = read_json(RUN_DIR / f"iter{it:03d}" / f"stop_metrics_iter{it:03d}.json", {})
        metrics = stop.get("metrics", {})
        rows.append(
            {
                "iteration": it,
                "selected_count": int(len(group)),
                "label_surprise_count": int(group["label_surprise"].sum()),
                "label_surprise_rate_recomputed": float(group["label_surprise"].mean()) if len(group) else math.nan,
                "label_surprise_rate_stop": metrics.get("label_surprise_rate", math.nan),
                "normal_to_FFLO_count": int(((group["predicted_phase_name"] == "normal") & (group["exact_phase_name"] == "FFLO")).sum()),
                "normal_to_uniform_SC_count": int(((group["predicted_phase_name"] == "normal") & (group["exact_phase_name"] == "uniform_SC")).sum()),
                "uniform_SC_to_FFLO_count": int(((group["predicted_phase_name"] == "uniform_SC") & (group["exact_phase_name"] == "FFLO")).sum()),
                "FFLO_to_normal_count": int(((group["predicted_phase_name"] == "FFLO") & (group["exact_phase_name"] == "normal")).sum()),
                "qedge_or_expanded_rate": safe_rate(group["qedge_or_expanded"]),
                "rerun_rate": safe_rate(group["rerun_any"]),
                "q_unresolved_rate": safe_rate(group.get("q_unresolved", pd.Series(dtype=float))),
                "delta_unresolved_rate": safe_rate(group.get("delta_unresolved", pd.Series(dtype=float))),
                "mean_boundary_distance": safe_mean(group.get("selected_to_predicted_boundary_distance", pd.Series(dtype=float))),
                "median_boundary_distance": float(pd.to_numeric(group.get("selected_to_predicted_boundary_distance", pd.Series(dtype=float)), errors="coerce").median()),
                "mean_A0_main": safe_mean(group.get("A0_main", pd.Series(dtype=float))),
                "mean_A_phase": safe_mean(group.get("A_phase", pd.Series(dtype=float))),
                "mean_A_numerical": safe_mean(group.get("A_numerical", pd.Series(dtype=float))),
                "mean_A_explore": safe_mean(group.get("A_explore", pd.Series(dtype=float))),
                "mean_A_response": safe_mean(group.get("A_response", pd.Series(dtype=float))),
                "boundary_coverage_p95": metrics.get("boundary_coverage_p95", math.nan),
                "phase_map_change": metrics.get("phase_map_change", math.nan),
                "boundary_shift_normal_sc": metrics.get("boundary_shift_normal_sc", math.nan),
                "boundary_shift_uniform_fflo": metrics.get("boundary_shift_uniform_fflo", math.nan),
            }
        )
    return pd.DataFrame(rows)


def group_rate(df: pd.DataFrame, columns: list[str], min_count: int = 1) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, group in df.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(columns, key)}
        row.update(
            {
                "selected_count": int(len(group)),
                "surprise_count": int(group["label_surprise"].sum()),
                "surprise_rate": float(group["label_surprise"].mean()) if len(group) else math.nan,
                "qedge_or_expanded_rate": safe_rate(group["qedge_or_expanded"]),
                "rerun_rate": safe_rate(group["rerun_any"]),
                "mean_boundary_distance": safe_mean(group.get("selected_to_predicted_boundary_distance", pd.Series(dtype=float))),
                "mean_A0_main": safe_mean(group.get("A0_main", pd.Series(dtype=float))),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[out["selected_count"] >= int(min_count)].sort_values(
            ["surprise_count", "surprise_rate", "selected_count"], ascending=[False, False, False]
        )
    return out


def component_contrast(df: pd.DataFrame) -> pd.DataFrame:
    components = [
        "A0_main",
        "A_phase",
        "A_numerical",
        "A_explore",
        "A_response",
        "cls_entropy",
        "cls_margin_uncertainty",
        "U_delta",
        "U_q",
        "U_reg_phase",
        "B_delta",
        "B_q_SC",
        "gradient_score",
        "q_edge_risk_score",
        "E_q_SC",
        "extrapolation_risk_score",
        "selected_to_predicted_boundary_distance",
    ]
    rows = []
    surprise = df[df["label_surprise"]]
    clean = df[~df["label_surprise"]]
    for col in components:
        if col not in df.columns:
            continue
        s = pd.to_numeric(surprise[col], errors="coerce")
        c = pd.to_numeric(clean[col], errors="coerce")
        rows.append(
            {
                "component": col,
                "surprise_mean": float(s.mean()) if s.notna().sum() else math.nan,
                "non_surprise_mean": float(c.mean()) if c.notna().sum() else math.nan,
                "mean_difference": float(s.mean() - c.mean()) if s.notna().sum() and c.notna().sum() else math.nan,
                "surprise_median": float(s.median()) if s.notna().sum() else math.nan,
                "non_surprise_median": float(c.median()) if c.notna().sum() else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_difference", ascending=False, na_position="last")


def cleanup_recommendations(hotspots: pd.DataFrame, transition_table: pd.DataFrame, final_summary: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not transition_table.empty:
        for _, row in transition_table.head(6).iterrows():
            rows.append(
                {
                    "recommendation_id": f"surprise_transition_{len(rows)+1}",
                    "target": row.get("phase_transition", ""),
                    "evidence": f"{int(row['surprise_count'])} surprises across {int(row['selected_count'])} selected points; surprise_rate={float(row['surprise_rate']):.3f}",
                    "suggested_action": "Reserve cleanup acquisition quota for this predicted/exact transition; sample near existing hotspot bins before changing thresholds.",
                    "code_change_required": "yes, opt-in cleanup selector/profile only",
                    "do_not_change": "phase criterion, exact oracle, StopController thresholds, rankcap_k3",
                }
            )
    if not hotspots.empty:
        for _, row in hotspots.head(6).iterrows():
            rows.append(
                {
                    "recommendation_id": f"surprise_hotspot_{len(rows)+1}",
                    "target": f"kT_bin={row.get('kT_bin')}, JA_bin={row.get('JA_bin')}",
                    "evidence": f"{int(row['surprise_count'])} surprises / {int(row['selected_count'])} selected; qedge_rate={float(row['qedge_or_expanded_rate']):.3f}; rerun_rate={float(row['rerun_rate']):.3f}",
                    "suggested_action": "Use report-only hotspot bins to seed a bounded cleanup batch; cap repeated q-edge-heavy selections.",
                    "code_change_required": "yes, opt-in cleanup selector/profile only",
                    "do_not_change": "phase criterion, exact oracle, StopController thresholds, rankcap_k3",
                }
            )
    rows.append(
        {
            "recommendation_id": "coverage_cleanup_1",
            "target": "boundary_coverage_p95",
            "evidence": f"final boundary_coverage_p95={float(final_summary.get('boundary_coverage_p95', math.nan)):.6f}; coverage_tol=0.00625",
            "suggested_action": "Prefer boundary-coverage gaps after excluding already dense exact neighborhoods; this addresses coverage without relaxing StopController.",
            "code_change_required": "yes, opt-in cleanup selector/profile only",
            "do_not_change": "coverage threshold",
        }
    )
    return pd.DataFrame(rows)


def write_tables(all_points: pd.DataFrame, summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_points.to_csv(TABLE_DIR / "surprise_selected_points.csv", index=False)
    summary.to_csv(TABLE_DIR / "surprise_by_iteration.csv", index=False)
    confusion = group_rate(all_points, ["iteration", "predicted_phase_name", "exact_phase_name", "phase_transition"], min_count=1)
    confusion.to_csv(TABLE_DIR / "surprise_confusion_by_iteration.csv", index=False)
    transition = group_rate(all_points, ["phase_transition"], min_count=1)
    transition.to_csv(TABLE_DIR / "surprise_by_phase_transition.csv", index=False)
    boundary = group_rate(all_points, ["boundary_distance_bin"], min_count=1)
    boundary.to_csv(TABLE_DIR / "surprise_by_boundary_distance.csv", index=False)
    qedge = group_rate(all_points, ["qedge_or_expanded", "rerun_any"], min_count=1)
    qedge.to_csv(TABLE_DIR / "surprise_by_qedge_rerun.csv", index=False)
    components = component_contrast(all_points)
    components.to_csv(TABLE_DIR / "surprise_by_acquisition_component.csv", index=False)
    hotspots = group_rate(all_points[all_points["label_surprise"]], ["kT_bin", "JA_bin", "phase_transition"], min_count=2)
    hotspots.to_csv(TABLE_DIR / "surprise_hotspots.csv", index=False)
    recommendations = cleanup_recommendations(hotspots, transition, summary.iloc[-1])
    recommendations.to_csv(TABLE_DIR / "cleanup_target_recommendations.csv", index=False)
    return {
        "all_points": all_points,
        "summary": summary,
        "confusion": confusion,
        "transition": transition,
        "boundary": boundary,
        "qedge": qedge,
        "components": components,
        "hotspots": hotspots,
        "recommendations": recommendations,
    }


def plot_figures(tables: dict[str, pd.DataFrame]) -> list[Path]:
    paths: list[Path] = []
    summary = tables["summary"]
    all_points = tables["all_points"]
    final_it = int(summary["iteration"].max())
    final = all_points[all_points["iteration"] == final_it]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(summary["iteration"], summary["label_surprise_rate_recomputed"], marker="o", label="recomputed")
    ax.plot(summary["iteration"], summary["label_surprise_rate_stop"], marker="s", linestyle="--", label="StopController")
    ax.axhline(0.05, color="tab:red", linestyle=":", label="surprise_tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Label surprise rate")
    ax.set_title("Label surprise remains above formal stop tolerance")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "surprise_rate_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    labels = ["normal", "uniform_SC", "FFLO"]
    mat = np.zeros((3, 3), dtype=int)
    for i, pred in enumerate(labels):
        for j, actual in enumerate(labels):
            mat[j, i] = int(((final["predicted_phase_name"] == pred) & (final["exact_phase_name"] == actual)).sum())
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted before exact")
    ax.set_ylabel("Exact label")
    ax.set_title(f"Final-iteration phase confusion (iter {final_it:03d})")
    for j in range(3):
        for i in range(3):
            ax.text(i, j, str(mat[j, i]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG_DIR / "final_confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    last5 = all_points[all_points["iteration"] >= max(1, final_it - 4)].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    clean = last5[~last5["label_surprise"]]
    surprise = last5[last5["label_surprise"]]
    ax.scatter(clean["kT"], clean["JA"], s=8, c="lightgray", alpha=0.35, label="not surprise")
    for transition, group in surprise.groupby("phase_transition"):
        ax.scatter(group["kT"], group["JA"], s=18, alpha=0.8, label=str(transition))
    ax.set_xlabel("kBT")
    ax.set_ylabel("JA")
    ax.set_title("Last-five selected points and label-surprise transitions")
    ax.legend(fontsize=7, loc="upper right", ncol=1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = FIG_DIR / "surprise_map_last5.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    transition = tables["transition"].head(10)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(transition["phase_transition"].astype(str), transition["surprise_count"], color="tab:orange")
    ax.invert_yaxis()
    ax.set_xlabel("Surprise count")
    ax.set_title("Dominant predicted-to-exact surprise transitions")
    fig.tight_layout()
    path = FIG_DIR / "surprise_by_transition.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    qedge = tables["qedge"].copy()
    qedge["group"] = qedge["qedge_or_expanded"].astype(str) + " / rerun=" + qedge["rerun_any"].astype(str)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(qedge["group"], qedge["surprise_rate"], color="tab:green")
    ax.set_ylabel("Surprise rate")
    ax.set_xlabel("q-edge-or-expanded / rerun group")
    ax.set_title("Surprise by q-edge and rerun status")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "qedge_rerun_vs_surprise.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    components = tables["components"].head(12).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(components["component"], components["mean_difference"], color="tab:purple")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean(surprise) - Mean(non-surprise)")
    ax.set_title("Acquisition component contrast")
    fig.tight_layout()
    path = FIG_DIR / "acquisition_component_surprise.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    boundary = tables["boundary"].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(boundary["boundary_distance_bin"].astype(str), boundary["surprise_rate"], color="tab:blue")
    ax.set_ylabel("Surprise rate")
    ax.set_xlabel("Distance to predicted boundary")
    ax.set_title("Surprise as a function of predicted-boundary distance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "surprise_by_boundary_distance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def markdown_report(tables: dict[str, pd.DataFrame], figures: list[Path]) -> str:
    summary = tables["summary"]
    final = summary.iloc[-1]
    transition = tables["transition"].head(8)
    components = tables["components"].head(8)
    recommendations = tables["recommendations"]
    rel_figures = [p.relative_to(OUT_DIR).as_posix() for p in figures]

    lines = [
        "# Surprise Decomposition Audit",
        "",
        "## Executive Summary",
        "",
        "This is a report-only audit of the rankcap_k3 full loop. It reads the completed active-learning artifacts and does not rerun exact labels, modify acquisition, change StopController, or alter any thermodynamic criteria.",
        "",
        "Main result: the non-convergence is driven by late-stage label surprise and slightly insufficient boundary coverage, not by phase-map or main-boundary drift. The surprise events are dominated by specific predicted-to-exact transition channels and q-edge/rerun-heavy selections.",
        "",
        "Final iteration metrics:",
        "",
        "```text",
        f"iteration = {int(final['iteration'])}",
        f"label_surprise_rate = {float(final['label_surprise_rate_recomputed']):.6f}",
        f"StopController label_surprise_rate = {float(final['label_surprise_rate_stop']):.6f}",
        f"boundary_coverage_p95 = {float(final['boundary_coverage_p95']):.6f}",
        f"phase_map_change = {float(final['phase_map_change']):.6f}",
        f"boundary_shift_normal_sc = {float(final['boundary_shift_normal_sc']):.6f}",
        f"boundary_shift_uniform_fflo = {float(final['boundary_shift_uniform_fflo']):.6f}",
        f"qedge_or_expanded_rate = {float(final['qedge_or_expanded_rate']):.6f}",
        f"rerun_rate = {float(final['rerun_rate']):.6f}",
        "```",
        "",
        "## What Was Audited",
        "",
        "- `selected_points_by_pool.csv`: selected points, predicted phase before exact, acquisition components, boundary distance.",
        "- `exact_merged_iterXXX.npz`: exact labels and q/delta/rerun diagnostics.",
        "- `stop_metrics_iterXXX.json`: StopController metrics and thresholds.",
        "- `run_config.json`: phase-label epsilons and grid ranges.",
        "",
        "The recomputed surprise rate is `exact_phase_label_stop != predicted_phase_before_exact`, where `exact_phase_label_stop` is derived from `Delta_opt` and `q_opt` using the active-loop `delta_eps` and `q_eps` values.",
        "",
        "## Surprise By Iteration",
        "",
        "| iteration | selected | surprise count | surprise rate | stop surprise | q-edge/expanded | rerun | coverage p95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['iteration'])} | {int(row['selected_count'])} | {int(row['label_surprise_count'])} | "
            f"{float(row['label_surprise_rate_recomputed']):.4f} | {float(row['label_surprise_rate_stop']):.4f} | "
            f"{float(row['qedge_or_expanded_rate']):.4f} | {float(row['rerun_rate']):.4f} | {float(row['boundary_coverage_p95']):.6f} |"
        )
    lines += [
        "",
        "## Dominant Surprise Channels",
        "",
        "| phase transition | selected | surprises | surprise rate | q-edge/expanded | rerun | mean boundary distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in transition.iterrows():
        lines.append(
            f"| {row['phase_transition']} | {int(row['selected_count'])} | {int(row['surprise_count'])} | "
            f"{float(row['surprise_rate']):.4f} | {float(row['qedge_or_expanded_rate']):.4f} | "
            f"{float(row['rerun_rate']):.4f} | {float(row['mean_boundary_distance']):.6f} |"
        )
    lines += [
        "",
        "## Acquisition Component Contrast",
        "",
        "| component | surprise mean | non-surprise mean | difference |",
        "|---|---:|---:|---:|",
    ]
    for _, row in components.iterrows():
        lines.append(
            f"| {row['component']} | {float(row['surprise_mean']):.6f} | "
            f"{float(row['non_surprise_mean']):.6f} | {float(row['mean_difference']):.6f} |"
        )
    lines += [
        "",
        "## Cleanup-Oriented Interpretation",
        "",
        "The correct optimization target is not to relax convergence criteria. It is to add a bounded cleanup selection mode that intentionally samples the remaining surprise channels and the sparse boundary-coverage regions after the main phase map has stabilized.",
        "",
        "Minimum safe strategy:",
        "",
        "1. Keep `full` acquisition as the discovery profile.",
        "2. Add a separate opt-in cleanup selector/profile after phase-map and boundary-shift stability pass.",
        "3. Allocate a bounded batch quota to surprise hotspots and boundary-coverage gaps.",
        "4. Keep q-edge/rerun-heavy selections monitored so cleanup does not become another q-edge expansion run.",
        "5. Do not change phase criterion, exact oracle, rankcap_k3, StopController thresholds, or AL stop logic.",
        "",
        "## Recommended Cleanup Targets",
        "",
        "| id | target | evidence | suggested action |",
        "|---|---|---|---|",
    ]
    for _, row in recommendations.iterrows():
        lines.append(
            f"| {row['recommendation_id']} | {row['target']} | {row['evidence']} | {row['suggested_action']} |"
        )
    lines += [
        "",
        "## Figures",
        "",
    ]
    for rel in rel_figures:
        lines.append(f"![{Path(rel).stem}]({rel})")
        lines.append("")
    lines += [
        "## Output Tables",
        "",
        "```text",
        "tables/surprise_by_iteration.csv",
        "tables/surprise_confusion_by_iteration.csv",
        "tables/surprise_by_phase_transition.csv",
        "tables/surprise_by_boundary_distance.csv",
        "tables/surprise_by_qedge_rerun.csv",
        "tables/surprise_by_acquisition_component.csv",
        "tables/surprise_hotspots.csv",
        "tables/surprise_selected_points.csv",
        "tables/cleanup_target_recommendations.csv",
        "```",
        "",
        "## Do-Not-Claim List",
        "",
        "1. Do not claim formal convergence passed.",
        "2. Do not treat high label surprise as evidence that the main phase map is drifting.",
        "3. Do not change thermodynamic phase criteria, tolerances, exact oracle, rankcap_k3, or StopController thresholds.",
        "4. Do not conclude `full` acquisition failed as a discovery method; it succeeded in map stabilization but remains too surprise-heavy for formal stop.",
        "5. Do not run more full-loop iterations before a cleanup-mode validation unless the goal is explicitly continued discovery rather than convergence cleanup.",
        "",
        "## Next Step",
        "",
        "Implement or package a small opt-in cleanup validation run. The cleanup run should start from the completed full-loop dataset, sample only one bounded cleanup batch, and report whether label surprise and boundary coverage move toward the formal stop criteria without changing physics definitions.",
        "",
    ]
    return "\n".join(lines)


def latex_escape(text: Any) -> str:
    s = str(text)
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
    return "".join(repl.get(ch, ch) for ch in s)


def write_pdf(markdown_text: str, figures: list[Path]) -> Path | None:
    tex_path = OUT_DIR / f"{REPORT_NAME}.tex"
    pdf_path = OUT_DIR / f"{REPORT_NAME}.pdf"
    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{hyperref}",
        r"\usepackage{float}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\begin{document}",
        r"\title{Surprise Decomposition Audit}",
        r"\author{report-only analysis}",
        r"\date{2026-06-18}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        "This report-only audit decomposes late-stage label surprise in the rankcap\\_k3 full-loop run. It reads existing artifacts only and does not rerun exact labels or modify acquisition, oracle, StopController, rankcap, or tolerances.",
        r"\section*{Key Metrics}",
        r"\begin{verbatim}",
    ]
    capture = False
    captured: list[str] = []
    for line in markdown_text.splitlines():
        if line.strip() == "```text":
            capture = True
            continue
        if line.strip() == "```" and capture:
            break
        if capture:
            captured.append(line)
    lines.extend(captured)
    lines += [
        r"\end{verbatim}",
        r"\section*{Interpretation}",
        "The main phase map and main boundaries are stable, but the selected batch still has label surprise above the formal StopController tolerance and boundary coverage remains slightly sparse. The safe next step is an opt-in cleanup validation, not a threshold change.",
        r"\section*{Figures}",
    ]
    for fig in figures:
        rel = fig.relative_to(OUT_DIR).as_posix()
        lines += [
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.92\linewidth]{{{rel}}}",
            rf"\caption{{{latex_escape(fig.stem)}}}",
            r"\end{figure}",
        ]
    lines += [
        r"\section*{Companion Files}",
        r"\begin{verbatim}",
        "surprise_decomposition_audit.md",
        "tables/*.csv",
        "figures/*.png",
        "decision_log.md",
        r"\end{verbatim}",
        r"\end{document}",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    if shutil.which("pdflatex") is None:
        return None
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=OUT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not pdf_path.exists():
        (OUT_DIR / f"{REPORT_NAME}_pdflatex.log").write_text(result.stdout, encoding="utf-8")
        return None
    return pdf_path


def write_decision_log(summary: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    final = summary.iloc[-1]
    lines = [
        "# Surprise Decomposition Decision Log",
        "",
        "- Status: report-only audit completed.",
        "- Decision: do not change physics definitions, oracle, rankcap_k3, StopController thresholds, or default full acquisition behavior based on this audit alone.",
        f"- Evidence: final label_surprise_rate={float(final['label_surprise_rate_recomputed']):.6f}; final boundary_coverage_p95={float(final['boundary_coverage_p95']):.6f}.",
        f"- Evidence: final phase_map_change={float(final['phase_map_change']):.6f}, normal/SC shift={float(final['boundary_shift_normal_sc']):.6f}, uniform/FFLO shift={float(final['boundary_shift_uniform_fflo']):.6f}.",
        "- Recommended next calculation: a bounded opt-in cleanup validation from the completed full-loop dataset.",
        f"- Cleanup recommendation rows: {len(recommendations)}.",
        "",
    ]
    (OUT_DIR / "decision_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cfg = run_config()
    iterations = [it for it in iteration_numbers() if it >= 1]
    if not iterations:
        raise SystemExit(f"No completed acquisition iterations found under {RUN_DIR}")
    frames = [load_iteration(it, cfg) for it in iterations]
    all_points = pd.concat(frames, ignore_index=True)
    summary = summarize_iterations(all_points, iterations)
    tables = write_tables(all_points, summary)
    figures = plot_figures(tables)
    md = markdown_report(tables, figures)
    md_path = OUT_DIR / f"{REPORT_NAME}.md"
    md_path.write_text(md, encoding="utf-8")
    pdf_path = write_pdf(md, figures)
    write_decision_log(summary, tables["recommendations"])
    print(f"wrote {md_path}")
    print(f"wrote {pdf_path if pdf_path is not None else 'PDF generation failed or pdflatex missing'}")
    print(f"wrote {TABLE_DIR}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
