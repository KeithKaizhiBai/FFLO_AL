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
OUT_DIR = ROOT / "reports" / "trusted_surprise_counterfactual"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
REPORT_NAME = "trusted_surprise_counterfactual"

PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
SCENARIOS = {
    "scenario0_all_selected": "surprise_all_selected",
    "scenario1_nonrerun": "surprise_nonrerun",
    "scenario2_training_eligible": "surprise_training_eligible",
    "scenario3_trusted": "surprise_trusted",
    "scenario4_trusted_boundary": "surprise_trusted_boundary",
}
TRUSTED_SENSITIVITY = {
    "scenario3_trusted_min16": ("surprise_trusted", 16, None),
    "scenario3_trusted_min32": ("surprise_trusted", 32, None),
    "scenario3_trusted_min64": ("surprise_trusted", 64, None),
    "scenario3_trusted_min25pct": ("surprise_trusted", 0, 0.25),
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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
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


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_kT_key"] = pd.to_numeric(out["kT"], errors="coerce").round(10)
    out["_JA_key"] = pd.to_numeric(out["JA"], errors="coerce").round(10)
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


def selected_exact_frame(iteration: int, cfg: dict[str, Any]) -> pd.DataFrame:
    iter_dir = RUN_DIR / f"iter{iteration:03d}"
    selected_path = iter_dir / "selected_points_by_pool.csv"
    exact_path = iter_dir / f"exact_merged_iter{iteration:03d}.npz"
    if not selected_path.exists() or not exact_path.exists():
        return pd.DataFrame()
    selected = add_keys(pd.read_csv(selected_path))
    if selected.empty or "predicted_phase_before_exact" not in selected.columns:
        return pd.DataFrame()
    exact = add_keys(npz_frame(exact_path))
    if exact.empty or not {"delta_opt", "q_opt"}.issubset(exact.columns):
        return pd.DataFrame()
    exact["exact_phase_label"] = phase_label(
        exact["delta_opt"],
        exact["q_opt"],
        float(cfg.get("delta_eps", 1.0e-3)),
        float(cfg.get("q_eps", 1.0e-2)),
    )
    exact_cols = [
        "_kT_key",
        "_JA_key",
        "exact_phase_label",
        "trusted_exact",
        "training_eligible_exact",
        "rerun_required",
        "needs_rerun_exact",
        "q_expanded",
        "q_edge_hit",
        "q_unresolved",
        "delta_unresolved",
        "delta_boundary_ambiguous",
    ]
    exact_cols = [c for c in exact_cols if c in exact.columns]
    joined = selected.merge(exact[exact_cols], on=["_kT_key", "_JA_key"], how="left", validate="one_to_one")
    joined.insert(0, "iteration", iteration)
    joined["matched_exact"] = joined["exact_phase_label"].notna()
    joined["predicted_phase_name"] = joined.get("predicted_phase_before_exact", pd.Series(dtype=float)).map(phase_name)
    joined["exact_phase_name"] = joined.get("exact_phase_label", pd.Series(dtype=float)).map(phase_name)
    joined["phase_transition"] = joined["predicted_phase_name"] + "_to_" + joined["exact_phase_name"]
    joined["label_surprise"] = (
        joined["matched_exact"]
        & joined["predicted_phase_before_exact"].notna()
        & (joined["predicted_phase_before_exact"].astype(float) != joined["exact_phase_label"].astype(float))
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
    joined["q_unresolved_bool"] = pd.to_numeric(joined.get("q_unresolved", 0), errors="coerce").fillna(0).astype(int) > 0
    joined["delta_unresolved_bool"] = pd.to_numeric(joined.get("delta_unresolved", 0), errors="coerce").fillna(0).astype(int) > 0
    joined["trusted_mask"] = (
        joined["trusted_bool"]
        & joined["training_eligible_bool"]
        & ~joined["rerun_bool"]
        & ~joined["q_unresolved_bool"]
        & ~joined["delta_unresolved_bool"]
    )
    joined["hard_risk_mask"] = (
        joined["rerun_bool"]
        | ~joined["trusted_bool"]
        | ~joined["training_eligible_bool"]
        | joined["q_unresolved_bool"]
        | joined["delta_unresolved_bool"]
    )
    selected_diag = read_json(iter_dir / "selection_diagnostics.json", {})
    band_tol = selected_diag.get("boundary_band_width_norm")
    if band_tol is None:
        band_tol = selected_diag.get("boundary_position_tol", 0.008333333333333333)
    joined["boundary_band_tolerance"] = float(band_tol)
    dist = pd.to_numeric(joined.get("selected_to_predicted_boundary_distance", np.nan), errors="coerce")
    joined["distance_to_main_boundary"] = dist
    joined["trusted_boundary_mask"] = joined["trusted_mask"] & np.isfinite(dist) & (dist <= float(band_tol))
    joined["trusted_interior_mask"] = joined["trusted_mask"] & np.isfinite(dist) & (dist > float(band_tol))
    joined["boundary_distance_bin"] = pd.cut(
        dist,
        bins=[-np.inf, 0.5 * float(band_tol), float(band_tol), 2 * float(band_tol), 4 * float(band_tol), np.inf],
        labels=["<=0.5band", "0.5-1band", "1-2band", "2-4band", ">4band"],
    ).astype(str)
    return joined


def metric_from_mask(df: pd.DataFrame, mask: pd.Series, selected_count: int) -> dict[str, Any]:
    denom = df[mask & df["matched_exact"]].copy()
    n = int(len(denom))
    surprise = int(denom["label_surprise"].sum()) if n else 0
    return {
        "n_denominator": n,
        "n_surprise": surprise,
        "surprise_rate": None if n == 0 else float(surprise / n),
        "available": bool(n > 0),
        "denominator_fraction_of_selected": None if selected_count <= 0 else float(n / selected_count),
    }


def build_surprise_metrics(history: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    point_rows: list[pd.DataFrame] = []
    for item in history:
        iteration = int(item["iteration"])
        df = selected_exact_frame(iteration, cfg)
        if not df.empty:
            point_rows.append(df)
        selected_count = int(len(df)) if not df.empty else 0
        if df.empty:
            metric_defs = {
                "surprise_all_selected": pd.Series([], dtype=bool),
                "surprise_nonrerun": pd.Series([], dtype=bool),
                "surprise_training_eligible": pd.Series([], dtype=bool),
                "surprise_trusted": pd.Series([], dtype=bool),
                "surprise_hard_risk": pd.Series([], dtype=bool),
                "surprise_trusted_boundary": pd.Series([], dtype=bool),
                "surprise_trusted_interior": pd.Series([], dtype=bool),
            }
        else:
            metric_defs = {
                "surprise_all_selected": df["matched_exact"],
                "surprise_nonrerun": ~df["rerun_bool"],
                "surprise_training_eligible": df["training_eligible_bool"],
                "surprise_trusted": df["trusted_mask"],
                "surprise_hard_risk": df["hard_risk_mask"],
                "surprise_trusted_boundary": df["trusted_boundary_mask"],
                "surprise_trusted_interior": df["trusted_interior_mask"],
            }
        for metric_name, mask in metric_defs.items():
            result = metric_from_mask(df, mask, selected_count) if not df.empty else {
                "n_denominator": 0,
                "n_surprise": 0,
                "surprise_rate": None,
                "available": False,
                "denominator_fraction_of_selected": None,
            }
            rows.append({"iteration": iteration, "surprise_metric_name": metric_name, **result})
    all_points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return pd.DataFrame(rows), all_points


def bool_condition(value: Any, available: bool, tol: float) -> bool:
    if not available or value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(math.isfinite(v) and v < float(tol))


def current_condition_rows(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in history:
        iteration = int(item["iteration"])
        metrics = item.get("metrics", {})
        availability = item.get("metric_availability", {})
        conditions = item.get("conditions", {})
        thresholds = item.get("thresholds", {})
        rows.append(
            {
                "iteration": iteration,
                "phase_map_change": metrics.get("phase_map_change"),
                "phase_map_pass": bool(conditions.get("C1_phase_map_change", False)),
                "boundary_shift_normal_sc": metrics.get("boundary_shift_normal_sc"),
                "boundary_shift_normal_sc_pass": bool(conditions.get("C2_boundary_shift_normal_sc", False)),
                "boundary_shift_uniform_fflo": metrics.get("boundary_shift_uniform_fflo"),
                "boundary_shift_uniform_fflo_pass": bool(conditions.get("C3_boundary_shift_uniform_fflo", False)),
                "label_surprise_rate_saved": metrics.get("label_surprise_rate"),
                "label_surprise_saved_pass": bool(conditions.get("C4_label_surprise_rate", False)),
                "boundary_coverage_p95": metrics.get("boundary_coverage_p95"),
                "boundary_coverage_pass": bool(conditions.get("C5_boundary_coverage_p95", False)),
                "passed_condition_count_saved": int(item.get("passed_condition_count", 0)),
                "required_pass_count": int(item.get("required_pass_count", 4)),
                "convergence_pass_saved": bool(item.get("convergence_pass", False)),
                "patience_counter_saved": int(item.get("patience_counter", 0)),
                "stop_saved": bool(item.get("stop", False)),
                "stop_reason_saved": item.get("stop_reason", ""),
                "surprise_tol": thresholds.get("surprise_tol", 0.05),
                "max_iterations": item.get("stop_config", {}).get("max_iterations", 31),
                "min_iterations": item.get("stop_config", {}).get("min_iterations", 5),
                "patience": item.get("stop_config", {}).get("patience", 4),
                "phase_map_available": bool(availability.get("phase_map_change", False)),
                "boundary_shift_normal_available": bool(availability.get("boundary_shift_normal_sc", False)),
                "boundary_shift_uniform_available": bool(availability.get("boundary_shift_uniform_fflo", False)),
                "boundary_coverage_available": bool(availability.get("boundary_coverage_p95", False)),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_scenario(
    base: pd.DataFrame,
    metrics: pd.DataFrame,
    scenario_name: str,
    metric_name: str,
    min_denominator: int = 0,
    min_fraction: float | None = None,
) -> list[dict[str, Any]]:
    metric_map = {
        int(row["iteration"]): row
        for _, row in metrics[metrics["surprise_metric_name"] == metric_name].iterrows()
    }
    rows: list[dict[str, Any]] = []
    patience_counter = 0
    for _, base_row in base.sort_values("iteration").iterrows():
        iteration = int(base_row["iteration"])
        metric_row = metric_map.get(iteration)
        if metric_row is None:
            surprise_value = None
            denom = 0
            available = False
            denom_fraction = None
        else:
            surprise_value = metric_row["surprise_rate"] if pd.notna(metric_row["surprise_rate"]) else None
            denom = int(metric_row["n_denominator"])
            available = bool(metric_row["available"])
            denom_fraction = metric_row["denominator_fraction_of_selected"]
        if min_denominator > 0 and denom < int(min_denominator):
            available = False
        if min_fraction is not None and (denom_fraction is None or float(denom_fraction) < float(min_fraction)):
            available = False
        surprise_pass = bool_condition(surprise_value, available, float(base_row["surprise_tol"]))
        phase_pass = bool(base_row["phase_map_pass"])
        normal_pass = bool(base_row["boundary_shift_normal_sc_pass"])
        uniform_pass = bool(base_row["boundary_shift_uniform_fflo_pass"])
        coverage_pass = bool(base_row["boundary_coverage_pass"])
        passed = int(sum([phase_pass, normal_pass, uniform_pass, surprise_pass, coverage_pass]))
        required = int(base_row["required_pass_count"])
        min_iterations = int(base_row["min_iterations"])
        patience = int(base_row["patience"])
        completed_iterations = iteration + 1
        convergence_pass = bool(completed_iterations >= min_iterations and passed >= required)
        patience_counter = patience_counter + 1 if convergence_pass else 0
        convergence_stop = bool(patience_counter >= patience)
        hard_stop = bool(completed_iterations >= int(base_row["max_iterations"]))
        would_stop = bool(convergence_stop or hard_stop)
        if convergence_stop:
            stop_reason = "converged_main_phase_boundaries"
        elif hard_stop:
            stop_reason = "max_iterations"
        else:
            stop_reason = ""
        failed = []
        if not phase_pass:
            failed.append("phase_map_change")
        if not normal_pass:
            failed.append("boundary_shift_normal_sc")
        if not uniform_pass:
            failed.append("boundary_shift_uniform_fflo")
        if not surprise_pass:
            failed.append(metric_name)
        if not coverage_pass:
            failed.append("boundary_coverage_p95")
        rows.append(
            {
                "iteration": iteration,
                "scenario": scenario_name,
                "surprise_metric_name": metric_name,
                "surprise_value": surprise_value,
                "denominator_count": denom,
                "denominator_fraction_of_selected": denom_fraction,
                "min_denominator": min_denominator,
                "min_denominator_fraction": min_fraction,
                "surprise_available_after_min_denominator": available,
                "surprise_pass": surprise_pass,
                "phase_map_pass": phase_pass,
                "normal_sc_shift_pass": normal_pass,
                "uniform_fflo_shift_pass": uniform_pass,
                "boundary_coverage_pass": coverage_pass,
                "passed_condition_count": passed,
                "required_pass_count": required,
                "convergence_pass": convergence_pass,
                "patience_counter": patience_counter,
                "would_stop": would_stop,
                "stop_reason": stop_reason,
                "remaining_failed_conditions": ";".join(failed),
            }
        )
    return rows


def earliest_summary(counterfactual: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, group in counterfactual.groupby("scenario", sort=False):
        g = group.sort_values("iteration")
        req = g[g["passed_condition_count"] >= g["required_pass_count"]]
        conv = g[g["convergence_pass"]]
        stops = g[g["would_stop"] & (g["stop_reason"] == "converged_main_phase_boundaries")]
        first_req = int(req["iteration"].iloc[0]) if len(req) else None
        first_conv = int(conv["iteration"].iloc[0]) if len(conv) else None
        first_stop = int(stops["iteration"].iloc[0]) if len(stops) else None
        at_stop = stops.iloc[0] if len(stops) else g.iloc[-1]
        rows.append(
            {
                "scenario": scenario,
                "surprise_metric_name": at_stop["surprise_metric_name"],
                "earliest_iteration_meeting_required_pass_count": first_req,
                "earliest_iteration_starting_patience": first_conv,
                "earliest_counterfactual_stop_iteration": first_stop,
                "whether_stop_occurs_before_iter030": bool(first_stop is not None and first_stop < 30),
                "remaining_failed_conditions_at_stop_or_final": at_stop["remaining_failed_conditions"],
                "final_passed_condition_count": int(g.iloc[-1]["passed_condition_count"]),
                "final_boundary_coverage_pass": bool(g.iloc[-1]["boundary_coverage_pass"]),
                "final_surprise_pass": bool(g.iloc[-1]["surprise_pass"]),
                "final_surprise_value": g.iloc[-1]["surprise_value"],
                "final_denominator_count": int(g.iloc[-1]["denominator_count"]),
            }
        )
    return pd.DataFrame(rows)


def grouped_breakdown(all_points: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if all_points.empty:
        return {}
    scoped = {
        "last5": all_points[all_points["iteration"] >= 26],
        "all_acquisition_iterations": all_points,
    }
    outputs: dict[str, pd.DataFrame] = {}
    for group_name, group_cols in {
        "surprise_by_transition": ["scope", "phase_transition"],
        "surprise_by_exact_phase": ["scope", "exact_phase_name"],
        "surprise_by_predicted_phase": ["scope", "predicted_phase_name"],
        "surprise_by_qexpanded_rerun_group": ["scope", "qexpanded_bool", "rerun_bool"],
        "surprise_by_boundary_distance": ["scope", "boundary_distance_bin"],
    }.items():
        rows: list[dict[str, Any]] = []
        for scope, df in scoped.items():
            matched = df[df["matched_exact"]].copy()
            matched["scope"] = scope
            for key, gg in matched.groupby(group_cols, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                row = {col: val for col, val in zip(group_cols, key)}
                denom = int(len(gg))
                surprise = int(gg["label_surprise"].sum())
                row.update(
                    {
                        "denominator_count": denom,
                        "surprise_count": surprise,
                        "surprise_rate": None if denom == 0 else float(surprise / denom),
                        "trusted_count": int(gg["trusted_mask"].sum()),
                        "hard_risk_count": int(gg["hard_risk_mask"].sum()),
                        "rerun_required_count": int(gg["rerun_bool"].sum()),
                        "qexpanded_count": int(gg["qexpanded_bool"].sum()),
                    }
                )
                rows.append(row)
        outputs[group_name] = pd.DataFrame(rows)
    return outputs


def hard_risk_contribution(all_points: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "last5": all_points[all_points["iteration"] >= 26],
        "all_acquisition_iterations": all_points,
    }
    rows: list[dict[str, Any]] = []
    for scope, df in scopes.items():
        matched = df[df["matched_exact"]]
        surprise = matched[matched["label_surprise"]]
        total = int(len(surprise))
        rerun = int((surprise["rerun_bool"]).sum())
        trusted = int((surprise["trusted_mask"]).sum())
        hard = int((surprise["hard_risk_mask"]).sum())
        qexp_nonrerun = int((surprise["qexpanded_bool"] & ~surprise["rerun_bool"]).sum())
        untrusted = int((~surprise["trusted_bool"]).sum())
        rows.append(
            {
                "scope": scope,
                "total_surprise_count": total,
                "hard_risk_surprise_count": hard,
                "trusted_surprise_count": trusted,
                "rerun_required_surprise_count": rerun,
                "qexpanded_nonrerun_surprise_count": qexp_nonrerun,
                "untrusted_surprise_count": untrusted,
                "fraction_of_all_surprise_from_rerun_required": None if total == 0 else float(rerun / total),
                "fraction_of_all_surprise_from_untrusted": None if total == 0 else float(untrusted / total),
                "fraction_of_all_surprise_from_trusted": None if total == 0 else float(trusted / total),
                "fraction_of_all_surprise_from_qexpanded_nonrerun": None if total == 0 else float(qexp_nonrerun / total),
            }
        )
    return pd.DataFrame(rows)


def build_definition_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "surprise_all_selected",
                "definition": "All matched selected rows with predicted_phase_before_exact and exact phase from Delta/q.",
                "denominator_mask": "matched_exact",
                "role": "current formal StopController C4",
            },
            {
                "metric": "surprise_nonrerun",
                "definition": "Matched selected rows where rerun_required and needs_rerun_exact are false.",
                "denominator_mask": "matched_exact and not rerun_required",
                "role": "counterfactual",
            },
            {
                "metric": "surprise_training_eligible",
                "definition": "Matched selected rows with training_eligible_exact true.",
                "denominator_mask": "matched_exact and training_eligible_exact",
                "role": "counterfactual",
            },
            {
                "metric": "surprise_trusted",
                "definition": "Matched selected rows that are trusted, training-eligible, non-rerun, q-resolved, and delta-resolved. q_expanded is not excluded.",
                "denominator_mask": "trusted_exact and training_eligible_exact and not rerun_required and not q_unresolved and not delta_unresolved",
                "role": "recommended formal gate candidate if denominator is sufficient",
            },
            {
                "metric": "surprise_hard_risk",
                "definition": "Matched selected rows that are rerun-required, untrusted, not training eligible, q-unresolved, or delta-unresolved.",
                "denominator_mask": "rerun_required or not trusted_exact or not training_eligible_exact or q_unresolved or delta_unresolved",
                "role": "numerical-reliability frontier diagnostic",
            },
            {
                "metric": "surprise_trusted_boundary",
                "definition": "Trusted surprise restricted to selected points within the saved selection_diagnostics boundary_band_width_norm.",
                "denominator_mask": "trusted_mask and distance_to_main_boundary <= boundary_band_width_norm",
                "role": "boundary-local counterfactual",
            },
            {
                "metric": "surprise_trusted_interior",
                "definition": "Trusted surprise outside the saved main-boundary band.",
                "denominator_mask": "trusted_mask and distance_to_main_boundary > boundary_band_width_norm",
                "role": "interior consistency diagnostic",
            },
        ]
    )


def discrepancy_check(base: pd.DataFrame, metrics: pd.DataFrame, current_recon: pd.DataFrame) -> pd.DataFrame:
    metric_map = metrics[metrics["surprise_metric_name"] == "surprise_all_selected"].set_index("iteration")
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        it = int(row["iteration"])
        m = metric_map.loc[it] if it in metric_map.index else None
        recon = current_recon[current_recon["iteration"] == it].iloc[0]
        saved = row["label_surprise_rate_saved"]
        recomputed = None if m is None or pd.isna(m["surprise_rate"]) else float(m["surprise_rate"])
        rows.append(
            {
                "iteration": it,
                "saved_label_surprise_rate": saved,
                "recomputed_label_surprise_rate": recomputed,
                "abs_surprise_discrepancy": None
                if saved is None or pd.isna(saved) or recomputed is None
                else abs(float(saved) - float(recomputed)),
                "saved_passed_condition_count": int(row["passed_condition_count_saved"]),
                "reconstructed_passed_condition_count": int(recon["passed_condition_count"]),
                "passed_condition_count_match": int(row["passed_condition_count_saved"]) == int(recon["passed_condition_count"]),
                "saved_convergence_pass": bool(row["convergence_pass_saved"]),
                "reconstructed_convergence_pass": bool(recon["convergence_pass"]),
                "convergence_pass_match": bool(row["convergence_pass_saved"]) == bool(recon["convergence_pass"]),
                "saved_patience_counter": int(row["patience_counter_saved"]),
                "reconstructed_patience_counter": int(recon["patience_counter"]),
                "patience_counter_match": int(row["patience_counter_saved"]) == int(recon["patience_counter"]),
                "saved_stop": bool(row["stop_saved"]),
                "reconstructed_would_stop": bool(recon["would_stop"]),
                "stop_match": bool(row["stop_saved"]) == bool(recon["would_stop"]),
            }
        )
    return pd.DataFrame(rows)


def write_tables(
    definition: pd.DataFrame,
    metrics: pd.DataFrame,
    all_points: pd.DataFrame,
    base: pd.DataFrame,
    counterfactual: pd.DataFrame,
    earliest: pd.DataFrame,
    discrepancy: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    definition.to_csv(TABLE_DIR / "surprise_definition_audit.csv", index=False)
    metrics.to_csv(TABLE_DIR / "surprise_metrics_by_iteration.csv", index=False)
    metrics[
        [
            "iteration",
            "surprise_metric_name",
            "n_denominator",
            "n_surprise",
            "denominator_fraction_of_selected",
            "available",
        ]
    ].to_csv(TABLE_DIR / "surprise_denominator_by_iteration.csv", index=False)
    grouped = grouped_breakdown(all_points)
    for name, df in grouped.items():
        df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    base.to_csv(TABLE_DIR / "stopcontroller_current_reconstruction.csv", index=False)
    counterfactual.to_csv(TABLE_DIR / "stopcontroller_counterfactual_reconstruction.csv", index=False)
    counterfactual[
        [
            "iteration",
            "scenario",
            "surprise_metric_name",
            "passed_condition_count",
            "convergence_pass",
            "patience_counter",
            "would_stop",
            "stop_reason",
        ]
    ].to_csv(TABLE_DIR / "patience_counterfactual.csv", index=False)
    earliest.to_csv(TABLE_DIR / "earliest_stop_summary.csv", index=False)
    hard = hard_risk_contribution(all_points)
    hard.to_csv(TABLE_DIR / "hard_risk_contribution.csv", index=False)
    discrepancy.to_csv(TABLE_DIR / "discrepancy_check.csv", index=False)
    return {
        "definition": definition,
        "metrics": metrics,
        "all_points": all_points,
        "base": base,
        "counterfactual": counterfactual,
        "earliest": earliest,
        "discrepancy": discrepancy,
        "hard": hard,
        **grouped,
    }


def plot_figures(tables: dict[str, pd.DataFrame]) -> list[Path]:
    paths: list[Path] = []
    metrics = tables["metrics"]
    cf = tables["counterfactual"]
    hard = tables["hard"]

    pivot = metrics.pivot(index="iteration", columns="surprise_metric_name", values="surprise_rate")
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ["surprise_all_selected", "surprise_nonrerun", "surprise_training_eligible", "surprise_trusted", "surprise_hard_risk"]:
        if col in pivot:
            ax.plot(pivot.index, pivot[col], marker="o", linewidth=1.5, label=col.replace("surprise_", ""))
    ax.axhline(0.05, color="tab:red", linestyle="--", label="surprise_tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surprise rate")
    ax.set_title("Surprise definitions by iteration")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "surprise_metrics_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    denom = metrics.pivot(index="iteration", columns="surprise_metric_name", values="n_denominator")
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ["surprise_all_selected", "surprise_nonrerun", "surprise_training_eligible", "surprise_trusted", "surprise_hard_risk"]:
        if col in denom:
            ax.plot(denom.index, denom[col], marker="o", linewidth=1.5, label=col.replace("surprise_", ""))
    ax.axhline(64, color="tab:gray", linestyle=":", linewidth=1, label="64 denominator")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Denominator count")
    ax.set_title("Surprise denominator counts")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIG_DIR / "surprise_denominator_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for col in ["surprise_trusted", "surprise_hard_risk"]:
        if col in pivot:
            ax.plot(pivot.index, pivot[col], marker="o", label=col.replace("surprise_", ""))
    ax.axhline(0.05, color="tab:red", linestyle="--", label="surprise_tol")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surprise rate")
    ax.set_title("Trusted vs hard-risk surprise")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "trusted_vs_hardrisk_surprise.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if "surprise_hard_risk" in denom and "surprise_all_selected" in denom:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        frac = denom["surprise_hard_risk"] / denom["surprise_all_selected"].replace(0, np.nan)
        ax.plot(frac.index, frac, marker="o", color="tab:orange")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Hard-risk denominator fraction")
        ax.set_title("Hard-risk fraction of selected batch")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = FIG_DIR / "hardrisk_fraction_by_iteration.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    main_cf = cf[cf["scenario"].isin(SCENARIOS.keys())]
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, group in main_cf.groupby("scenario", sort=False):
        ax.plot(group["iteration"], group["passed_condition_count"], marker="o", linewidth=1.5, label=scenario.replace("scenario", "S"))
    ax.axhline(4, color="tab:red", linestyle="--", label="required_pass_count")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Passed condition count")
    ax.set_title("Counterfactual passed-condition count")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "counterfactual_passed_condition_count.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario, group in main_cf.groupby("scenario", sort=False):
        ax.plot(group["iteration"], group["patience_counter"], marker="o", linewidth=1.5, label=scenario.replace("scenario", "S"))
    ax.axhline(4, color="tab:red", linestyle="--", label="patience")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Patience counter")
    ax.set_title("Counterfactual patience curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / "counterfactual_patience_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    earliest = tables["earliest"][tables["earliest"]["scenario"].isin(SCENARIOS.keys())].copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = earliest["earliest_counterfactual_stop_iteration"].fillna(31)
    ax.bar(earliest["scenario"].str.replace("scenario", "S", regex=False), values, color="tab:blue")
    ax.axhline(30, color="tab:red", linestyle="--", label="iter030")
    ax.set_ylabel("Earliest stop iteration; 31 means no convergence stop")
    ax.set_title("Earliest counterfactual stop")
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "earliest_stop_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    boundary = tables.get("surprise_by_boundary_distance", pd.DataFrame())
    if not boundary.empty:
        last5 = boundary[boundary["scope"] == "last5"].copy()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(last5["boundary_distance_bin"].astype(str), last5["surprise_rate"], color="tab:green")
        ax.set_xlabel("Distance-to-boundary bin")
        ax.set_ylabel("Surprise rate")
        ax.set_title("Trusted/control context: selected surprise by boundary distance")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = FIG_DIR / "trusted_surprise_by_boundary_distance.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def final_metric(metrics: pd.DataFrame, name: str, iteration: int = 30) -> pd.Series:
    row = metrics[(metrics["iteration"] == iteration) & (metrics["surprise_metric_name"] == name)]
    if row.empty:
        raise KeyError(name)
    return row.iloc[0]


def recommendation(earliest: pd.DataFrame, metrics: pd.DataFrame) -> str:
    trusted_final = final_metric(metrics, "surprise_trusted")
    trusted_ok = (
        bool(trusted_final["available"])
        and int(trusted_final["n_denominator"]) >= 64
        and float(trusted_final["surprise_rate"]) < 0.05
    )
    trusted_row = earliest[earliest["scenario"] == "scenario3_trusted"].iloc[0]
    stop_before_final = bool(trusted_row["whether_stop_occurs_before_iter030"])
    if trusted_ok and stop_before_final:
        return "Decision B"
    if not trusted_ok:
        return "Decision A"
    return "Decision C"


def markdown_report(tables: dict[str, pd.DataFrame], figures: list[Path]) -> str:
    metrics = tables["metrics"]
    earliest = tables["earliest"]
    discrepancy = tables["discrepancy"]
    hard = tables["hard"]
    rec = recommendation(earliest, metrics)
    final_all = final_metric(metrics, "surprise_all_selected")
    final_nonrerun = final_metric(metrics, "surprise_nonrerun")
    final_trusted = final_metric(metrics, "surprise_trusted")
    final_hard = final_metric(metrics, "surprise_hard_risk")
    last5_hard = hard[hard["scope"] == "last5"].iloc[0]
    trusted_stop = earliest[earliest["scenario"] == "scenario3_trusted"].iloc[0]
    rel_figs = [p.relative_to(OUT_DIR).as_posix() for p in figures]
    max_discrepancy = discrepancy["abs_surprise_discrepancy"].dropna().max()
    scenario0_ok = bool(
        (discrepancy["passed_condition_count_match"].all())
        and (discrepancy["convergence_pass_match"].all())
        and (discrepancy["patience_counter_match"].all())
        and (max_discrepancy is None or float(max_discrepancy) == 0.0)
    )

    lines = [
        "# Trusted-Only Label Surprise Counterfactual",
        "",
        "## Executive Summary",
        "",
        "This report is a report-only counterfactual analysis.  It does not modify StopController, acquisition, exact oracle, rankcap_k3, tolerances, Slurm state, or active-learning artifacts.",
        "",
        "Main findings:",
        "",
        f"- Scenario 0 current StopController reconstruction: {'exactly reproduced' if scenario0_ok else 'not exactly reproduced'}.",
        f"- Final all-selected surprise: {int(final_all['n_surprise'])}/{int(final_all['n_denominator'])} = {float(final_all['surprise_rate']):.6f}.",
        f"- Final non-rerun surprise: {int(final_nonrerun['n_surprise'])}/{int(final_nonrerun['n_denominator'])} = {float(final_nonrerun['surprise_rate']):.6f}.",
        f"- Final trusted surprise: {int(final_trusted['n_surprise'])}/{int(final_trusted['n_denominator'])} = {float(final_trusted['surprise_rate']):.6f}.",
        f"- Final hard-risk surprise: {int(final_hard['n_surprise'])}/{int(final_hard['n_denominator'])} = {float(final_hard['surprise_rate']):.6f}.",
        f"- Last-five fraction of all surprise from rerun-required points: {float(last5_hard['fraction_of_all_surprise_from_rerun_required']):.6f}.",
        f"- Trusted-only counterfactual earliest stop iteration: {trusted_stop['earliest_counterfactual_stop_iteration']}.",
        f"- Recommended decision: {rec}.",
        "",
        "## StopController Definition Audit",
        "",
        "The current formal surprise metric is `ml_phase/stop_controller.py::label_surprise_rate`.  It matches selected points to `exact_merged_iterXXX.npz` by rounded `(kT, JA)`, computes exact phase from `Delta_opt` and `q_opt`, and divides mismatches by all matched selected rows.  It does not filter trusted, training-eligible, non-rerun, q-resolved, or delta-resolved points.",
        "",
        "## Final Surprise Values",
        "",
        "| metric | denominator | surprise | rate | denominator fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in [
        "surprise_all_selected",
        "surprise_nonrerun",
        "surprise_training_eligible",
        "surprise_trusted",
        "surprise_hard_risk",
        "surprise_trusted_boundary",
        "surprise_trusted_interior",
    ]:
        row = final_metric(metrics, name)
        rate_value = "" if row["surprise_rate"] is None or pd.isna(row["surprise_rate"]) else f"{float(row['surprise_rate']):.6f}"
        frac = "" if row["denominator_fraction_of_selected"] is None or pd.isna(row["denominator_fraction_of_selected"]) else f"{float(row['denominator_fraction_of_selected']):.3f}"
        lines.append(f"| {name} | {int(row['n_denominator'])} | {int(row['n_surprise'])} | {rate_value} | {frac} |")

    lines += [
        "",
        "## Counterfactual Stop Summary",
        "",
        "| scenario | metric | first required-pass iteration | first patience iteration | earliest stop iteration | final passed | remaining blocker at stop/final |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in earliest.iterrows():
        if not str(row["scenario"]).startswith("scenario"):
            continue
        lines.append(
            f"| {row['scenario']} | {row['surprise_metric_name']} | {row['earliest_iteration_meeting_required_pass_count']} | "
            f"{row['earliest_iteration_starting_patience']} | {row['earliest_counterfactual_stop_iteration']} | "
            f"{int(row['final_passed_condition_count'])} | {row['remaining_failed_conditions_at_stop_or_final']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "`surprise_all_selected` is an acquisition-selected batch difficulty diagnostic. It is useful, but it is not a global phase-map accuracy metric because the selected batch is intentionally biased toward uncertain boundary regions.",
        "",
        "`surprise_trusted` measures model consistency only on points that produced trusted, training-eligible, non-rerun, q-resolved, and delta-resolved exact labels. q-expanded points remain included when they satisfy those trusted conditions.",
        "",
        "`surprise_hard_risk` isolates the numerical-reliability frontier. These points should be reported and queued for targeted numerical follow-up rather than used as clean exact-label errors.",
        "",
        "## Decision",
        "",
    ]
    if rec == "Decision B":
        lines += [
            "Recommended decision: **Decision B - split formal surprise into trusted and hard-risk layers.**",
            "",
            "Evidence: the trusted denominator is large enough in the final batch, trusted surprise is below tolerance, and the current formal non-convergence is driven primarily by rerun-required hard-risk points. Under the trusted-surprise counterfactual, formal convergence would occur before iter030.",
            "",
            "Minimal code design, not implemented here:",
            "",
            "```text",
            "Keep existing label_surprise_rate as surprise_all_selected diagnostic.",
            "Add surprise_trusted, surprise_hard_risk.",
            "Add n_surprise_trusted_denominator, n_surprise_hard_risk_denominator.",
            "Formal gate candidate:",
            "    surprise_trusted <= surprise_tol",
            "    and n_surprise_trusted_denominator >= configured minimum",
            "Hard-risk points:",
            "    record separately",
            "    do not directly veto main phase convergence",
            "    may trigger targeted numerical rerun queue",
            "```",
        ]
    elif rec == "Decision A":
        lines += [
            "Recommended decision: **Decision A - keep current StopController unchanged and run late-stage cleanup acquisition.**",
            "",
            "Evidence: trusted surprise is not yet sufficiently below tolerance or the denominator is not sufficient under the chosen sensitivity condition.",
        ]
    else:
        lines += [
            "Recommended decision: **Decision C - cannot determine.**",
            "",
            "Evidence: the trusted denominator or metadata is insufficient to select A or B safely.",
        ]

    lines += [
        "",
        "## Do-Not-Claim List",
        "",
        "1. Do not treat acquisition-selected surprise as global phase accuracy.",
        "2. Do not treat rerun-required provisional labels as clean exact-label error.",
        "3. Do not exclude all q-expanded points.",
        "4. Do not claim convergence when the trusted denominator is too small.",
        "5. Do not modify surprise_tol.",
        "6. Do not modify StopController code in this report-only step.",
        "7. Do not start cleanup run from this report.",
        "8. Do not conflate boundary_coverage failure with label-surprise failure.",
        "9. Do not claim the hard-risk FFLO frontier is solved.",
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
        "tables/surprise_definition_audit.csv",
        "tables/surprise_metrics_by_iteration.csv",
        "tables/surprise_denominator_by_iteration.csv",
        "tables/surprise_by_transition.csv",
        "tables/surprise_by_exact_phase.csv",
        "tables/surprise_by_predicted_phase.csv",
        "tables/surprise_by_qexpanded_rerun_group.csv",
        "tables/surprise_by_boundary_distance.csv",
        "tables/stopcontroller_current_reconstruction.csv",
        "tables/stopcontroller_counterfactual_reconstruction.csv",
        "tables/patience_counterfactual.csv",
        "tables/earliest_stop_summary.csv",
        "tables/hard_risk_contribution.csv",
        "tables/discrepancy_check.csv",
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
        r"\title{Trusted-Only Label Surprise Counterfactual}",
        r"\author{report-only analysis}",
        r"\date{2026-06-18}",
        r"\maketitle",
        r"\section*{Executive Summary}",
    ]
    summary_lines = []
    for line in markdown_text.splitlines():
        if line.startswith("- ") and len(summary_lines) < 8:
            summary_lines.append(line[2:])
    lines.extend(latex_escape(x) + r"\\" for x in summary_lines)
    lines += [r"\section*{Figures}"]
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
        "trusted_surprise_counterfactual.md",
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
    metrics = tables["metrics"]
    earliest = tables["earliest"]
    hard = tables["hard"]
    rec = recommendation(earliest, metrics)
    final_all = final_metric(metrics, "surprise_all_selected")
    final_trusted = final_metric(metrics, "surprise_trusted")
    final_hard = final_metric(metrics, "surprise_hard_risk")
    trusted_stop = earliest[earliest["scenario"] == "scenario3_trusted"].iloc[0]
    last5_hard = hard[hard["scope"] == "last5"].iloc[0]
    lines = [
        "# Trusted Surprise Counterfactual Decision Log",
        "",
        "- Status: report-only counterfactual completed.",
        f"- Current all-selected final surprise: {int(final_all['n_surprise'])}/{int(final_all['n_denominator'])} = {float(final_all['surprise_rate']):.6f}.",
        f"- Trusted final surprise: {int(final_trusted['n_surprise'])}/{int(final_trusted['n_denominator'])} = {float(final_trusted['surprise_rate']):.6f}.",
        f"- Hard-risk final surprise: {int(final_hard['n_surprise'])}/{int(final_hard['n_denominator'])} = {float(final_hard['surprise_rate']):.6f}.",
        f"- Last-five surprise fraction from rerun-required points: {float(last5_hard['fraction_of_all_surprise_from_rerun_required']):.6f}.",
        f"- Trusted-only counterfactual earliest stop iteration: {trusted_stop['earliest_counterfactual_stop_iteration']}.",
        f"- Recommendation: {rec}.",
        "- No StopController, acquisition, oracle, rankcap, tolerance, Slurm, or active-run artifact was modified.",
        "",
    ]
    (OUT_DIR / "decision_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    history = read_json(RUN_DIR / "stop_metrics_history.json", [])
    if not history:
        raise SystemExit(f"Missing stop history: {RUN_DIR / 'stop_metrics_history.json'}")
    cfg = run_config()
    definition = build_definition_audit()
    metrics, all_points = build_surprise_metrics(history, cfg)
    base = current_condition_rows(history)

    rows: list[dict[str, Any]] = []
    for scenario, metric in SCENARIOS.items():
        rows.extend(reconstruct_scenario(base, metrics, scenario, metric))
    for scenario, (metric, min_den, min_frac) in TRUSTED_SENSITIVITY.items():
        rows.extend(reconstruct_scenario(base, metrics, scenario, metric, min_denominator=min_den, min_fraction=min_frac))
    counterfactual = pd.DataFrame(rows)
    earliest = earliest_summary(counterfactual)
    current_recon = counterfactual[counterfactual["scenario"] == "scenario0_all_selected"]
    discrepancy = discrepancy_check(base, metrics, current_recon)
    max_disc = discrepancy["abs_surprise_discrepancy"].dropna().max()
    if pd.notna(max_disc) and float(max_disc) > 0.0:
        raise SystemExit(f"Scenario 0 surprise reconstruction discrepancy: {max_disc}")
    if not discrepancy["passed_condition_count_match"].all() or not discrepancy["convergence_pass_match"].all():
        raise SystemExit("Scenario 0 StopController condition reconstruction failed")

    tables = write_tables(definition, metrics, all_points, base, counterfactual, earliest, discrepancy)
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
