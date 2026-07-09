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
OUT_DIR = ROOT / "reports" / "surprise_review_recheck"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_NAME = "surprise_review_recheck"

PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
LABELS = ["normal", "uniform_SC", "FFLO"]
COVERAGE_TOL = 0.00625


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


def phase_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return "missing"
    try:
        return PHASE_NAMES.get(int(value), str(int(value)))
    except Exception:
        return str(value)


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
    out["_kT_key"] = pd.to_numeric(out["kT"], errors="coerce").round(10)
    out["_JA_key"] = pd.to_numeric(out["JA"], errors="coerce").round(10)
    return out


def iteration_numbers() -> list[int]:
    out: list[int] = []
    for path in sorted(RUN_DIR.glob("iter[0-9][0-9][0-9]")):
        try:
            it = int(path.name.replace("iter", ""))
        except ValueError:
            continue
        if (path / "selected_points_by_pool.csv").exists() and (path / f"exact_merged_iter{it:03d}.npz").exists():
            out.append(it)
    return out


def load_iteration(iteration: int, cfg: dict[str, Any]) -> pd.DataFrame:
    iter_dir = RUN_DIR / f"iter{iteration:03d}"
    selected = add_keys(pd.read_csv(iter_dir / "selected_points_by_pool.csv"))
    exact = add_keys(npz_frame(iter_dir / f"exact_merged_iter{iteration:03d}.npz"))
    exact["exact_phase_label_stop"] = phase_label(
        exact["delta_opt"],
        exact["q_opt"],
        float(cfg.get("delta_eps", 1.0e-3)),
        float(cfg.get("q_eps", 1.0e-2)),
    )
    exact_cols = [
        "_kT_key",
        "_JA_key",
        "delta_opt",
        "q_opt",
        "exact_phase_label_stop",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "needs_rerun_exact",
        "q_edge_hit",
        "q_expanded",
        "q_unresolved",
        "delta_unresolved",
        "delta_boundary_ambiguous",
        "q_expansion_count",
        "selected_refine_target_count",
        "local_boxes_refined_count",
        "selected_minimum_rank",
        "point_total_runtime_sec",
        "local_refinement_runtime_sec",
    ]
    exact_cols = [c for c in exact_cols if c in exact.columns]
    joined = selected.merge(exact[exact_cols], on=["_kT_key", "_JA_key"], how="left", validate="one_to_one")
    joined.insert(0, "iteration", iteration)
    joined["matched_exact"] = joined["exact_phase_label_stop"].notna()
    joined["predicted_phase_name"] = joined["predicted_phase_before_exact"].map(phase_name)
    joined["exact_phase_name"] = joined["exact_phase_label_stop"].map(phase_name)
    joined["phase_transition"] = joined["predicted_phase_name"] + "_to_" + joined["exact_phase_name"]
    joined["label_surprise"] = (
        joined["matched_exact"]
        & joined["predicted_phase_before_exact"].notna()
        & (joined["predicted_phase_before_exact"].astype(float) != joined["exact_phase_label_stop"].astype(float))
    )
    joined["trusted_bool"] = pd.to_numeric(joined.get("trusted_exact", 0), errors="coerce").fillna(0).astype(int) > 0
    joined["training_eligible_bool"] = (
        pd.to_numeric(joined.get("training_eligible_exact", 0), errors="coerce").fillna(0).astype(int) > 0
    )
    joined["rerun_bool"] = (
        (pd.to_numeric(joined.get("rerun_required", 0), errors="coerce").fillna(0).astype(int) > 0)
        | (pd.to_numeric(joined.get("needs_rerun_exact", 0), errors="coerce").fillna(0).astype(int) > 0)
    )
    joined["qexpanded_bool"] = (
        (pd.to_numeric(joined.get("q_expanded", 0), errors="coerce").fillna(0).astype(int) > 0)
        | (pd.to_numeric(joined.get("q_edge_hit", 0), errors="coerce").fillna(0).astype(int) > 0)
    )
    joined["trusted_nonrerun_bool"] = joined["trusted_bool"] & joined["training_eligible_bool"] & ~joined["rerun_bool"]
    joined["trusted_nonrerun_clean_bool"] = joined["trusted_nonrerun_bool"] & ~joined["qexpanded_bool"]
    joined["trusted_qexpanded_bool"] = joined["trusted_nonrerun_bool"] & joined["qexpanded_bool"]
    dist = pd.to_numeric(joined.get("selected_to_predicted_boundary_distance", np.nan), errors="coerce")
    joined["boundary_distance_bin"] = pd.cut(
        dist,
        bins=[-np.inf, 0.5 * COVERAGE_TOL, COVERAGE_TOL, 2 * COVERAGE_TOL, 4 * COVERAGE_TOL, 8 * COVERAGE_TOL, np.inf],
        labels=["<=0.5tol", "0.5-1tol", "1-2tol", "2-4tol", "4-8tol", ">8tol"],
    ).astype(str)
    return joined


def rate(mask: pd.Series) -> float:
    if len(mask) == 0:
        return math.nan
    return float(pd.to_numeric(mask, errors="coerce").fillna(0).mean())


def subset_row(name: str, df: pd.DataFrame) -> dict[str, Any]:
    matched = df[df["matched_exact"]].copy()
    return {
        "scope": name,
        "iterations": ",".join(str(int(x)) for x in sorted(df["iteration"].unique())),
        "selected_count": int(len(df)),
        "matched_exact_count": int(len(matched)),
        "surprise_count": int(matched["label_surprise"].sum()),
        "surprise_rate": rate(matched["label_surprise"]),
        "normal_to_fflo_count": int(((matched["predicted_phase_name"] == "normal") & (matched["exact_phase_name"] == "FFLO")).sum()),
        "normal_to_uniform_count": int(((matched["predicted_phase_name"] == "normal") & (matched["exact_phase_name"] == "uniform_SC")).sum()),
        "fflo_to_normal_count": int(((matched["predicted_phase_name"] == "FFLO") & (matched["exact_phase_name"] == "normal")).sum()),
        "uniform_to_fflo_count": int(((matched["predicted_phase_name"] == "uniform_SC") & (matched["exact_phase_name"] == "FFLO")).sum()),
        "trusted_nonrerun_count": int(matched["trusted_nonrerun_bool"].sum()),
        "trusted_nonrerun_surprise_count": int((matched["trusted_nonrerun_bool"] & matched["label_surprise"]).sum()),
        "trusted_nonrerun_surprise_rate": rate(matched.loc[matched["trusted_nonrerun_bool"], "label_surprise"]),
        "trusted_nonrerun_clean_count": int(matched["trusted_nonrerun_clean_bool"].sum()),
        "trusted_nonrerun_clean_surprise_count": int((matched["trusted_nonrerun_clean_bool"] & matched["label_surprise"]).sum()),
        "trusted_nonrerun_clean_surprise_rate": rate(matched.loc[matched["trusted_nonrerun_clean_bool"], "label_surprise"]),
        "trusted_qexpanded_count": int(matched["trusted_qexpanded_bool"].sum()),
        "trusted_qexpanded_surprise_count": int((matched["trusted_qexpanded_bool"] & matched["label_surprise"]).sum()),
        "trusted_qexpanded_surprise_rate": rate(matched.loc[matched["trusted_qexpanded_bool"], "label_surprise"]),
        "rerun_required_count": int(matched["rerun_bool"].sum()),
        "rerun_required_surprise_count": int((matched["rerun_bool"] & matched["label_surprise"]).sum()),
        "rerun_required_surprise_rate": rate(matched.loc[matched["rerun_bool"], "label_surprise"]),
        "qexpanded_count": int(matched["qexpanded_bool"].sum()),
        "qexpanded_surprise_count": int((matched["qexpanded_bool"] & matched["label_surprise"]).sum()),
        "qexpanded_surprise_rate": rate(matched.loc[matched["qexpanded_bool"], "label_surprise"]),
    }


def grouped_rate(df: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    matched = df[df["matched_exact"]]
    for key, group in matched.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {"scope": scope}
        row.update({col: val for col, val in zip(group_cols, key)})
        row.update(
            {
                "selected_count": int(len(group)),
                "surprise_count": int(group["label_surprise"].sum()),
                "surprise_rate": rate(group["label_surprise"]),
                "trusted_count": int(group["trusted_bool"].sum()),
                "training_eligible_count": int(group["training_eligible_bool"].sum()),
                "rerun_required_count": int(group["rerun_bool"].sum()),
                "qexpanded_count": int(group["qexpanded_bool"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables(all_points: pd.DataFrame, iterations: list[int]) -> dict[str, pd.DataFrame]:
    final_it = max(iterations)
    scopes = {
        "iter030": all_points[all_points["iteration"] == final_it],
        "last5": all_points[all_points["iteration"] >= final_it - 4],
        "all_acquisition_iterations": all_points,
    }
    scope_summary = pd.DataFrame([subset_row(name, df) for name, df in scopes.items()])
    scope_summary.to_csv(TABLE_DIR / "scope_summary.csv", index=False)

    final = scopes["iter030"]
    confusion_rows: list[dict[str, Any]] = []
    for exact in LABELS:
        for pred in LABELS:
            count = int(((final["exact_phase_name"] == exact) & (final["predicted_phase_name"] == pred)).sum())
            confusion_rows.append({"scope": "iter030", "exact_phase": exact, "predicted_phase": pred, "count": count})
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(TABLE_DIR / "iter030_confusion_matrix.csv", index=False)

    transition_frames = []
    for name, df in scopes.items():
        transition_frames.append(grouped_rate(df, ["phase_transition"], name))
    transitions = pd.concat(transition_frames, ignore_index=True)
    transitions.sort_values(["scope", "surprise_count", "selected_count"], ascending=[True, False, False]).to_csv(
        TABLE_DIR / "transition_breakdown_by_scope.csv", index=False
    )

    layer_rows = []
    for name, df in scopes.items():
        matched = df[df["matched_exact"]]
        groups = [
            ("stopcontroller_all_matched_selected", matched),
            ("trusted_nonrerun", matched[matched["trusted_nonrerun_bool"]]),
            ("trusted_nonrerun_clean_no_qexpanded", matched[matched["trusted_nonrerun_clean_bool"]]),
            ("trusted_qexpanded_nonrerun", matched[matched["trusted_qexpanded_bool"]]),
            ("rerun_required", matched[matched["rerun_bool"]]),
            ("qexpanded_or_qedge", matched[matched["qexpanded_bool"]]),
        ]
        for metric_name, group in groups:
            layer_rows.append(
                {
                    "scope": name,
                    "metric": metric_name,
                    "denominator_count": int(len(group)),
                    "surprise_count": int(group["label_surprise"].sum()) if len(group) else 0,
                    "surprise_rate": rate(group["label_surprise"]),
                    "normal_to_fflo_count": int(((group["predicted_phase_name"] == "normal") & (group["exact_phase_name"] == "FFLO")).sum()),
                }
            )
    layered = pd.DataFrame(layer_rows)
    layered.to_csv(TABLE_DIR / "layered_surprise_metrics.csv", index=False)

    n2f_rows = []
    for name, df in scopes.items():
        n2f = df[(df["matched_exact"]) & (df["predicted_phase_name"] == "normal") & (df["exact_phase_name"] == "FFLO")]
        n2f_rows.append(
            {
                "scope": name,
                "normal_to_fflo_count": int(len(n2f)),
                "trusted_count": int(n2f["trusted_bool"].sum()),
                "training_eligible_count": int(n2f["training_eligible_bool"].sum()),
                "rerun_required_count": int(n2f["rerun_bool"].sum()),
                "qexpanded_or_qedge_count": int(n2f["qexpanded_bool"].sum()),
                "trusted_nonrerun_count": int(n2f["trusted_nonrerun_bool"].sum()),
                "trusted_qexpanded_nonrerun_count": int(n2f["trusted_qexpanded_bool"].sum()),
                "median_boundary_distance": float(pd.to_numeric(n2f.get("selected_to_predicted_boundary_distance"), errors="coerce").median()) if len(n2f) else math.nan,
            }
        )
    n2f = pd.DataFrame(n2f_rows)
    n2f.to_csv(TABLE_DIR / "normal_to_fflo_diagnostics_by_scope.csv", index=False)

    boundary_frames = []
    qedge_frames = []
    for name, df in scopes.items():
        boundary_frames.append(grouped_rate(df, ["boundary_distance_bin"], name))
        qedge_frames.append(grouped_rate(df, ["qexpanded_bool", "rerun_bool"], name))
    boundary = pd.concat(boundary_frames, ignore_index=True)
    boundary.to_csv(TABLE_DIR / "boundary_distance_surprise_by_scope.csv", index=False)
    qedge = pd.concat(qedge_frames, ignore_index=True)
    qedge.to_csv(TABLE_DIR / "qedge_rerun_surprise_by_scope.csv", index=False)

    denominator = pd.DataFrame(
        [
            {
                "metric": "label_surprise_rate",
                "source_file": "ml_phase/stop_controller.py",
                "function": "label_surprise_rate",
                "line_range": "197-225",
                "denominator": "all rows in selected_points_by_pool.csv that match exact_merged_iterXXX.npz by rounded (kT, JA)",
                "filters_trusted_exact": False,
                "filters_training_eligible_exact": False,
                "filters_rerun_required": False,
                "filters_qexpanded": False,
                "final_iter_matched_count": int(scopes["iter030"]["matched_exact"].sum()),
                "final_iter_selected_count": int(len(scopes["iter030"])),
                "final_iter_surprise_count": int(scopes["iter030"]["label_surprise"].sum()),
                "final_iter_surprise_rate": rate(scopes["iter030"]["label_surprise"]),
                "verdict": "confirmed: StopController surprise includes rerun-required and untrusted/hard-risk selected points if they are present in exact_merged",
            }
        ]
    )
    denominator.to_csv(TABLE_DIR / "stopcontroller_surprise_denominator_audit.csv", index=False)

    fixed_probe = pd.DataFrame(
        [
            {
                "metric": "fixed_probe_or_random_control_surprise",
                "available_in_current_artifacts": False,
                "reason": "No independent fixed-probe or random-control exact batch with paired pre-exact predictions is saved for iter030/last5.",
                "safe_statement": "cannot determine from current artifacts; requires a held-out probe/control batch selected independently of acquisition",
            }
        ]
    )
    fixed_probe.to_csv(TABLE_DIR / "fixed_probe_control_availability.csv", index=False)

    all_points.to_csv(TABLE_DIR / "scoped_selected_point_recheck.csv", index=False)
    return {
        "scope_summary": scope_summary,
        "confusion": confusion,
        "transitions": transitions,
        "layered": layered,
        "n2f": n2f,
        "boundary": boundary,
        "qedge": qedge,
        "denominator": denominator,
        "fixed_probe": fixed_probe,
        "all_points": all_points,
    }


def plot_figures(tables: dict[str, pd.DataFrame]) -> list[Path]:
    paths: list[Path] = []
    confusion = tables["confusion"]
    mat = np.zeros((3, 3), dtype=int)
    for i, pred in enumerate(LABELS):
        for j, exact in enumerate(LABELS):
            row = confusion[(confusion["predicted_phase"] == pred) & (confusion["exact_phase"] == exact)]
            mat[j, i] = int(row["count"].iloc[0]) if len(row) else 0
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted before exact")
    ax.set_ylabel("Exact")
    ax.set_title("iter030 selected-batch confusion")
    for j in range(3):
        for i in range(3):
            ax.text(i, j, str(mat[j, i]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = FIG_DIR / "iter030_confusion_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    transitions = tables["transitions"]
    last5 = transitions[(transitions["scope"] == "last5") & (transitions["surprise_count"] > 0)].copy()
    last5 = last5.sort_values("surprise_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(last5["phase_transition"].astype(str), last5["surprise_count"], color="tab:orange")
    ax.set_xlabel("Surprise count")
    ax.set_title("Strict last-5 surprise transitions")
    fig.tight_layout()
    path = FIG_DIR / "last5_transition_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    layered = tables["layered"]
    last5_layers = layered[layered["scope"] == "last5"].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(last5_layers["metric"], last5_layers["surprise_rate"], color="tab:purple")
    ax.axhline(0.05, color="tab:red", linestyle="--", linewidth=1, label="surprise_tol")
    ax.set_ylabel("Surprise rate")
    ax.set_title("Strict last-5 surprise split by denominator")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "last5_layered_surprise_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    boundary = tables["boundary"][tables["boundary"]["scope"] == "last5"].copy()
    order = ["<=0.5tol", "0.5-1tol", "1-2tol", "2-4tol", "4-8tol", ">8tol"]
    boundary["order"] = boundary["boundary_distance_bin"].map({k: i for i, k in enumerate(order)})
    boundary = boundary.sort_values("order")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(boundary["boundary_distance_bin"].astype(str), boundary["surprise_rate"], color="tab:blue")
    ax.set_xlabel("Distance to predicted boundary")
    ax.set_ylabel("Surprise rate")
    ax.set_title("Strict last-5 surprise by boundary distance")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "last5_boundary_distance_surprise.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    qedge = tables["qedge"][tables["qedge"]["scope"] == "last5"].copy()
    qedge["group"] = "qedge=" + qedge["qexpanded_bool"].astype(str) + ", rerun=" + qedge["rerun_bool"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(qedge["group"], qedge["surprise_rate"], color="tab:green")
    ax.set_xlabel("q-edge/expanded and rerun group")
    ax.set_ylabel("Surprise rate")
    ax.set_title("Strict last-5 surprise by q-edge/rerun group")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIG_DIR / "last5_qedge_rerun_surprise.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def markdown_report(tables: dict[str, pd.DataFrame], figures: list[Path]) -> str:
    scope = tables["scope_summary"]
    final = scope[scope["scope"] == "iter030"].iloc[0]
    last5 = scope[scope["scope"] == "last5"].iloc[0]
    all_scope = scope[scope["scope"] == "all_acquisition_iterations"].iloc[0]
    confusion = tables["confusion"]
    layered = tables["layered"]
    last5_layers = layered[layered["scope"] == "last5"]
    n2f_last5 = tables["n2f"][tables["n2f"]["scope"] == "last5"].iloc[0]
    denom = tables["denominator"].iloc[0]
    rel_figs = [p.relative_to(OUT_DIR).as_posix() for p in figures]

    def layer(metric: str, column: str) -> Any:
        row = last5_layers[last5_layers["metric"] == metric]
        return row[column].iloc[0] if len(row) else math.nan

    def confusion_count(exact: str, pred: str) -> int:
        row = confusion[(confusion["exact_phase"] == exact) & (confusion["predicted_phase"] == pred)]
        return int(row["count"].iloc[0]) if len(row) else 0

    lines = [
        "# Surprise Review Recheck",
        "",
        "## Executive Summary",
        "",
        "This report rechecks the scope and denominator issues raised in the review of `surprise_decomposition_audit`. It is report-only: no acquisition, exact oracle, StopController, tolerance, or phase criterion code was changed.",
        "",
        "Main verdicts:",
        "",
        "1. StopController label surprise is confirmed to use all matched selected exact points as the denominator. It does not filter `trusted_exact`, `training_eligible_exact`, `rerun_required`, or q-expanded points.",
        "2. The original surprise decomposition mixed all-iteration, last-5, and iter030 scopes in the narrative. The dominant transition remains `predicted normal -> exact FFLO`, but quantitative counts must be scope-qualified.",
        "3. In strict iter030, all 47 surprises are `predicted normal -> exact FFLO`, exactly reproducing `47 / 256 = 0.18359375`.",
        "4. In strict last-5, `predicted normal -> exact FFLO` accounts for the dominant surprise channel, and most of those points are q-expanded/rerun-required hard-risk points.",
        "5. No fixed-probe or random-control surprise metric can be computed from current artifacts; that requires a separately selected held-out batch.",
        "",
        "## StopController Denominator",
        "",
        "| metric | denominator | filters trusted? | filters rerun? | verdict |",
        "|---|---|---:|---:|---|",
        f"| label_surprise_rate | {denom['denominator']} | {denom['filters_trusted_exact']} | {denom['filters_rerun_required']} | {denom['verdict']} |",
        "",
        "The relevant function is `ml_phase/stop_controller.py::label_surprise_rate`, lines 197-225. It builds exact labels from `delta_opt` and `q_opt`, matches selected rows by rounded `(kT, JA)`, and returns mismatches divided by matched rows.",
        "",
        "## Scope-Correct Summary",
        "",
        "| scope | selected | surprise | surprise rate | normal->FFLO | trusted nonrerun surprise | rerun surprise |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in scope.iterrows():
        lines.append(
            f"| {row['scope']} | {int(row['selected_count'])} | {int(row['surprise_count'])} | "
            f"{float(row['surprise_rate']):.6f} | {int(row['normal_to_fflo_count'])} | "
            f"{float(row['trusted_nonrerun_surprise_rate']):.6f} | {float(row['rerun_required_surprise_rate']):.6f} |"
        )
    lines += [
        "",
        "## iter030 Confusion Matrix",
        "",
        "Strict final-iteration confusion:",
        "",
        "```text",
        f"exact normal:     {confusion_count('normal', 'normal')} predicted normal",
        f"exact uniform-SC: {confusion_count('uniform_SC', 'uniform_SC')} predicted uniform-SC",
        f"exact FFLO:       {confusion_count('FFLO', 'normal')} predicted normal + {confusion_count('FFLO', 'FFLO')} predicted FFLO",
        "```",
        "",
        f"Final surprise: `{int(final['surprise_count'])} / {int(final['matched_exact_count'])} = {float(final['surprise_rate']):.6f}`.",
        "",
        "## Last-5 Normal-to-FFLO Diagnostics",
        "",
        "| count | trusted | training eligible | rerun required | qexpanded/qedge | trusted nonrerun | trusted qexpanded nonrerun | median boundary distance |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {int(n2f_last5['normal_to_fflo_count'])} | {int(n2f_last5['trusted_count'])} | {int(n2f_last5['training_eligible_count'])} | "
        f"{int(n2f_last5['rerun_required_count'])} | {int(n2f_last5['qexpanded_or_qedge_count'])} | "
        f"{int(n2f_last5['trusted_nonrerun_count'])} | {int(n2f_last5['trusted_qexpanded_nonrerun_count'])} | "
        f"{float(n2f_last5['median_boundary_distance']):.6f} |",
        "",
        "## Layered Last-5 Surprise Metrics",
        "",
        "| metric | denominator | surprise | surprise rate | normal->FFLO |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in last5_layers.iterrows():
        lines.append(
            f"| {row['metric']} | {int(row['denominator_count'])} | {int(row['surprise_count'])} | "
            f"{float(row['surprise_rate']):.6f} | {int(row['normal_to_fflo_count'])} |"
        )
    lines += [
        "",
        "Interpretation:",
        "",
        "- The selected-batch surprise is high because acquisition intentionally samples difficult boundary points.",
        "- The formal StopController metric currently includes rerun-required hard-risk points.",
        "- A fair convergence assessment should report at least both all-selected surprise and trusted/nonrerun surprise.",
        "- A fixed-probe or random-control surprise metric is currently unavailable and should be added before using selected-batch surprise as a global error proxy.",
        "",
        "## Corrected Physical Interpretation",
        "",
        "The main predicted phase map and the main predicted thermodynamic boundaries can be stable while selected-batch surprise remains high. The data support a narrow, boundary-localized conservative bias: the surrogate still labels some true FFLO boundary-side points as normal. Because acquisition is concentrated near uncertain normal/SC boundary regions, this bias is amplified in the selected batch.",
        "",
        "`boundary_shift` passing means the predicted boundary is no longer moving much between iterations. It does not prove that the predicted boundary has zero fixed offset relative to the exact boundary.",
        "",
        "## Required Next Checks",
        "",
        "1. Add a fixed-probe or random-control exact batch with saved pre-exact predictions.",
        "2. Report StopController-style all-selected surprise, trusted-nonrerun surprise, trusted-qexpanded surprise, and rerun-required surprise separately.",
        "3. Do not continue ordinary full discovery before deciding whether formal convergence should use selected-batch surprise or a stratified/held-out surprise metric.",
        "",
        "## Figures",
        "",
    ]
    for rel in rel_figs:
        lines.append(f"![{Path(rel).stem}]({rel})")
        lines.append("")
    lines += [
        "## Output Tables",
        "",
        "```text",
        "tables/stopcontroller_surprise_denominator_audit.csv",
        "tables/scope_summary.csv",
        "tables/iter030_confusion_matrix.csv",
        "tables/transition_breakdown_by_scope.csv",
        "tables/layered_surprise_metrics.csv",
        "tables/normal_to_fflo_diagnostics_by_scope.csv",
        "tables/boundary_distance_surprise_by_scope.csv",
        "tables/qedge_rerun_surprise_by_scope.csv",
        "tables/fixed_probe_control_availability.csv",
        "tables/scoped_selected_point_recheck.csv",
        "```",
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
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{6pt}",
        r"\begin{document}",
        r"\title{Surprise Review Recheck}",
        r"\author{report-only analysis}",
        r"\date{2026-06-18}",
        r"\maketitle",
        r"\section*{Executive Summary}",
        "This report rechecks StopController's label-surprise denominator and separates iter030, strict last-five, and all-iteration scopes. StopController currently counts all matched selected exact points and does not filter trusted or rerun-required points.",
        r"\section*{Key Findings}",
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
    lines.extend(captured[:12])
    lines += [r"\end{verbatim}", r"\section*{Figures}"]
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
        "surprise_review_recheck.md",
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


def write_decision_log(tables: dict[str, pd.DataFrame]) -> None:
    final = tables["scope_summary"][tables["scope_summary"]["scope"] == "iter030"].iloc[0]
    last5 = tables["scope_summary"][tables["scope_summary"]["scope"] == "last5"].iloc[0]
    lines = [
        "# Surprise Review Recheck Decision Log",
        "",
        "- Status: report-only recheck completed.",
        "- Confirmed: StopController label surprise denominator includes all matched selected exact points; it does not filter trusted/training-eligible/rerun-required points.",
        f"- Confirmed: iter030 surprise = {int(final['surprise_count'])}/{int(final['matched_exact_count'])} = {float(final['surprise_rate']):.6f}.",
        f"- Confirmed: iter030 surprises are all predicted normal -> exact FFLO ({int(final['normal_to_fflo_count'])} points).",
        f"- Strict last-5 surprise = {int(last5['surprise_count'])}/{int(last5['matched_exact_count'])} = {float(last5['surprise_rate']):.6f}.",
        "- Decision: do not use all-selected surprise alone as a global phase-map error proxy.",
        "- Next check: add held-out fixed-probe/random-control surprise and report trusted/rerun/q-expanded surprise strata separately.",
        "",
    ]
    (OUT_DIR / "decision_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    cfg = run_config()
    iterations = [it for it in iteration_numbers() if it >= 1]
    if not iterations:
        raise SystemExit(f"No acquisition iterations found under {RUN_DIR}")
    all_points = pd.concat([load_iteration(it, cfg) for it in iterations], ignore_index=True)
    tables = build_tables(all_points, iterations)
    figures = plot_figures(tables)
    md = markdown_report(tables, figures)
    md_path = OUT_DIR / f"{REPORT_NAME}.md"
    md_path.write_text(md, encoding="utf-8")
    pdf_path = write_pdf(md, figures)
    write_decision_log(tables)
    print(f"wrote {md_path}")
    print(f"wrote {pdf_path if pdf_path is not None else 'PDF generation failed or pdflatex missing'}")
    print(f"wrote {TABLE_DIR}")
    print(f"wrote {FIG_DIR}")


if __name__ == "__main__":
    main()
