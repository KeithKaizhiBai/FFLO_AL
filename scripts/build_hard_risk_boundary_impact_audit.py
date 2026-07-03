"""Hard-risk boundary-impact audit for the rankcap_k3 tail continuation.

This is a report-only script.  It reads downloaded tail-continuation artifacts
and performs offline counterfactual label-flip tests on the final dense monitor
phase map.  It does not modify datasets, acquisition, exact oracle,
StopController, phase criteria, tolerances, or Slurm state.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback for minimal environments
    cKDTree = None


PHASE_NAMES = {0: "normal", 1: "uniform_SC", 2: "FFLO"}
PHASE_COLORS = ["#e8e8e8", "#4c78a8", "#f58518"]

INPUT_ROOT = Path("rankcap_k3_tail_surprise_continuation_results")
RUN_DIR = (
    INPUT_ROOT
    / "ML_Phase_512_RankCapK3_TailContinuation"
    / "active_runs"
    / "active_boundary_discovery_rankcap_k3_tail_surprise_continuation_v1"
)
ITERATION = 34
ITER_DIR = RUN_DIR / f"iter{ITERATION:03d}"

OUT_DIR = Path("reports/hard_risk_boundary_impact_audit")
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr).astype(bool)


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "NA"
    return f"{f:.{digits}f}"


def normalized(points: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2) / scale.reshape(1, 2)


def nearest_distances_norm(a: np.ndarray, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.empty((0,), dtype=np.float64)
    an = normalized(a, scale)
    bn = normalized(b, scale)
    if cKDTree is not None:
        dist, _ = cKDTree(bn).query(an, k=1)
        return np.asarray(dist, dtype=np.float64)
    out = np.full(an.shape[0], np.inf, dtype=np.float64)
    for start in range(0, an.shape[0], 512):
        pts = an[start : start + 512]
        d2 = np.sum((pts[:, None, :] - bn[None, :, :]) ** 2, axis=2)
        out[start : start + 512] = np.sqrt(np.min(d2, axis=1))
    return out


def boundary_points_from_phase(
    phase: np.ndarray,
    coords: np.ndarray,
    boundary_type: str,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.int64)
    if boundary_type == "normal_sc":
        h_cross = (phase[:, :-1] == 0) != (phase[:, 1:] == 0)
        v_cross = (phase[:-1, :] == 0) != (phase[1:, :] == 0)
    elif boundary_type == "uniform_fflo":
        h_cross = ((phase[:, :-1] == 1) & (phase[:, 1:] == 2)) | (
            (phase[:, :-1] == 2) & (phase[:, 1:] == 1)
        )
        v_cross = ((phase[:-1, :] == 1) & (phase[1:, :] == 2)) | (
            (phase[:-1, :] == 2) & (phase[1:, :] == 1)
        )
    else:
        raise ValueError(f"unknown boundary_type: {boundary_type}")

    rows: list[np.ndarray] = []
    hi, hj = np.nonzero(h_cross)
    if hi.size:
        rows.append(0.5 * (coords[hi, hj] + coords[hi, hj + 1]))
    vi, vj = np.nonzero(v_cross)
    if vi.size:
        rows.append(0.5 * (coords[vi, vj] + coords[vi + 1, vj]))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def boundary_shift_stats(
    phase_cf: np.ndarray,
    phase_base: np.ndarray,
    coords: np.ndarray,
    scale: np.ndarray,
    boundary_type: str,
) -> dict[str, Any]:
    cur = boundary_points_from_phase(phase_cf, coords, boundary_type)
    base = boundary_points_from_phase(phase_base, coords, boundary_type)
    if cur.size == 0 or base.size == 0:
        return {
            "boundary_type": boundary_type,
            "boundary_point_count_current": int(cur.shape[0]),
            "boundary_point_count_reference": int(base.shape[0]),
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "available": False,
        }
    d_cur = nearest_distances_norm(cur, base, scale)
    d_base = nearest_distances_norm(base, cur, scale)
    dist = np.concatenate([d_cur, d_base])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {
            "boundary_type": boundary_type,
            "boundary_point_count_current": int(cur.shape[0]),
            "boundary_point_count_reference": int(base.shape[0]),
            "mean": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "max": math.nan,
            "available": False,
        }
    return {
        "boundary_type": boundary_type,
        "boundary_point_count_current": int(cur.shape[0]),
        "boundary_point_count_reference": int(base.shape[0]),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "available": True,
    }


def phase_map_change(phase_cf: np.ndarray, phase_base: np.ndarray) -> float:
    return float(np.mean(np.ravel(phase_cf) != np.ravel(phase_base)))


def load_data() -> dict[str, Any]:
    stop = read_json(ITER_DIR / f"stop_metrics_iter{ITERATION:03d}.json")
    monitor = np.load(ITER_DIR / f"monitor_predictions_iter{ITERATION:03d}.npz")
    exact = np.load(ITER_DIR / f"exact_merged_iter{ITERATION:03d}.npz", allow_pickle=True)
    selected = pd.read_csv(ITER_DIR / "selected_points_by_pool.csv")
    rerun_path = ITER_DIR / "rerun_points.csv"
    rerun = pd.read_csv(rerun_path) if rerun_path.exists() else pd.DataFrame()
    dataset = np.load(RUN_DIR / "dataset_iter035.npz", allow_pickle=True)
    return {
        "stop": stop,
        "monitor": monitor,
        "exact": exact,
        "selected": selected,
        "rerun": rerun,
        "dataset": dataset,
    }


def selected_lookup(selected: pd.DataFrame) -> dict[tuple[float, float], dict[str, Any]]:
    out: dict[tuple[float, float], dict[str, Any]] = {}
    for row in selected.to_dict(orient="records"):
        key = (round(float(row["kT"]), 10), round(float(row["JA"]), 10))
        out[key] = row
    return out


def rerun_lookup(rerun: pd.DataFrame) -> dict[tuple[float, float], dict[str, Any]]:
    out: dict[tuple[float, float], dict[str, Any]] = {}
    if rerun.empty:
        return out
    for row in rerun.to_dict(orient="records"):
        key = (round(float(row["kT"]), 10), round(float(row["JA"]), 10))
        out[key] = row
    return out


def dense_grid_cell(point: np.ndarray, grid_points: np.ndarray, scale: np.ndarray) -> int:
    dist = nearest_distances_norm(point.reshape(1, 2), grid_points, scale)
    if dist.size == 0:
        return -1
    return int(np.argmin(np.sum((normalized(grid_points, scale) - normalized(point.reshape(1, 2), scale)) ** 2, axis=1)))


def classify_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stop = data["stop"]
    monitor = data["monitor"]
    exact = data["exact"]
    selected = data["selected"]
    rerun = data["rerun"]
    thresholds = stop["thresholds"]
    boundary_tol = float(thresholds["boundary_shift_tol"])
    phase_map_tol = float(thresholds["map_tol"])
    coverage_tol = float(thresholds["coverage_tol"])

    full_shape = tuple(int(x) for x in np.asarray(monitor["full_shape"]).ravel()[:2])
    grid_points = np.asarray(monitor["grid_points"], dtype=np.float64).reshape(-1, 2)
    phase_base = np.asarray(monitor["phase_pred"], dtype=np.int64).reshape(full_shape)
    coords = grid_points.reshape(full_shape + (2,))
    kt_min, kt_max = float(np.min(grid_points[:, 0])), float(np.max(grid_points[:, 0]))
    ja_min, ja_max = float(np.min(grid_points[:, 1])), float(np.max(grid_points[:, 1]))
    scale = np.array([max(kt_max - kt_min, 1e-12), max(ja_max - ja_min, 1e-12)], dtype=np.float64)

    normal_sc_boundary = boundary_points_from_phase(phase_base, coords, "normal_sc")
    uniform_fflo_boundary = boundary_points_from_phase(phase_base, coords, "uniform_fflo")

    sel_lookup = selected_lookup(selected)
    rr_lookup = rerun_lookup(rerun)

    n = int(np.asarray(exact["kT"]).shape[0])
    hard_mask = (
        as_bool(exact["rerun_required"])
        | ~as_bool(exact["trusted_exact"])
        | ~as_bool(exact["training_eligible_exact"])
        | as_bool(exact["q_unresolved"])
        | as_bool(exact["delta_unresolved"])
    )
    rows: list[dict[str, Any]] = []
    for idx in np.where(hard_mask)[0]:
        kT = float(exact["kT"][idx])
        JA = float(exact["JA"][idx])
        point = np.array([[kT, JA]], dtype=np.float64)
        key = (round(kT, 10), round(JA, 10))
        selected_row = sel_lookup.get(key, {})
        rerun_row = rr_lookup.get(key, {})
        predicted = selected_row.get("predicted_phase_before_exact", "")
        try:
            predicted_phase = int(float(predicted))
        except (TypeError, ValueError):
            predicted_phase = -1
        selected_grid_index = int(float(selected_row.get("grid_index", -1))) if selected_row else -1
        if selected_grid_index < 0:
            selected_grid_index = dense_grid_cell(point.ravel(), grid_points, scale)
        grid_i = int(selected_grid_index // full_shape[1]) if selected_grid_index >= 0 else -1
        grid_j = int(selected_grid_index % full_shape[1]) if selected_grid_index >= 0 else -1
        current_phase_at_grid = int(phase_base[grid_i, grid_j]) if grid_i >= 0 else -1
        dns = float(nearest_distances_norm(point, normal_sc_boundary, scale)[0])
        duf = float(nearest_distances_norm(point, uniform_fflo_boundary, scale)[0])
        nearest_boundary = "normal_sc" if dns <= duf else "uniform_fflo"
        min_dist = min(dns, duf)
        near_ns = dns <= boundary_tol
        near_uf = duf <= boundary_tol
        if near_ns and near_uf:
            region = "boundary_overlap"
        elif near_ns:
            region = "normal_SC_boundary_band"
        elif near_uf:
            region = "uniform_FFLO_boundary_band"
        elif min_dist > 4.0 * boundary_tol:
            region = "far_interior"
        elif current_phase_at_grid == 0:
            region = "normal_interior"
        else:
            region = "SC_interior"

        reason = str(rerun_row.get("reason", ""))
        if not reason:
            reason_bits = []
            if bool(exact["rerun_required"][idx]):
                reason_bits.append("rerun_required")
            if not bool(exact["trusted_exact"][idx]):
                reason_bits.append("untrusted_exact")
            if not bool(exact["training_eligible_exact"][idx]):
                reason_bits.append("not_training_eligible")
            if bool(exact["q_unresolved"][idx]):
                reason_bits.append("q_unresolved")
            if bool(exact["delta_unresolved"][idx]):
                reason_bits.append("delta_unresolved")
            reason = ";".join(reason_bits)

        rows.append(
            {
                "point_id": f"iter034_exact_{idx:03d}",
                "exact_index": int(idx),
                "kBT": kT,
                "JA": JA,
                "provisional_phase_code": int(exact["phase_candidate"][idx]),
                "provisional_phase": PHASE_NAMES.get(int(exact["phase_candidate"][idx]), "unknown"),
                "predicted_phase_code": predicted_phase,
                "predicted_phase": PHASE_NAMES.get(predicted_phase, "unknown"),
                "current_monitor_phase_code": current_phase_at_grid,
                "current_monitor_phase": PHASE_NAMES.get(current_phase_at_grid, "unknown"),
                "rerun_required": bool(exact["rerun_required"][idx]),
                "trusted_exact": bool(exact["trusted_exact"][idx]),
                "training_eligible_exact": bool(exact["training_eligible_exact"][idx]),
                "rerun_required_reason": reason,
                "recommended_action_from_rerun_file": str(rerun_row.get("recommended_action", "")),
                "q_expanded": bool(exact["q_expanded"][idx]),
                "q_edge_hit": bool(exact["q_edge_hit"][idx]),
                "q_edge_distance": float(exact["q_edge_distance"][idx]),
                "q_status": int(exact["q_status"][idx]),
                "q_expansion_trigger": str(exact["q_expansion_trigger"][idx]),
                "q_unresolved": bool(exact["q_unresolved"][idx]),
                "delta_boundary_ambiguous": bool(exact["delta_boundary_ambiguous"][idx]),
                "delta_status": int(exact["delta_status"][idx]),
                "delta_unresolved": bool(exact["delta_unresolved"][idx]),
                "distance_to_normal_sc_boundary": dns,
                "distance_to_uniform_fflo_boundary": duf,
                "distance_to_nearest_boundary": min_dist,
                "nearest_boundary_type": nearest_boundary,
                "region": region,
                "selected_grid_index": selected_grid_index,
                "grid_i": grid_i,
                "grid_j": grid_j,
                "selected_to_predicted_boundary_distance": selected_row.get(
                    "selected_to_predicted_boundary_distance", ""
                ),
            }
        )

    context = {
        "full_shape": full_shape,
        "grid_points": grid_points,
        "phase_base": phase_base,
        "coords": coords,
        "scale": scale,
        "boundary_tol": boundary_tol,
        "phase_map_tol": phase_map_tol,
        "coverage_tol": coverage_tol,
        "normal_sc_boundary": normal_sc_boundary,
        "uniform_fflo_boundary": uniform_fflo_boundary,
        "stop": stop,
    }
    return rows, context


def apply_flips(
    base_phase: np.ndarray,
    rows: list[dict[str, Any]],
    mode: str,
) -> np.ndarray:
    phase = base_phase.copy()
    for row in rows:
        i = int(row["grid_i"])
        j = int(row["grid_j"])
        if i < 0 or j < 0:
            continue
        if mode == "as_normal":
            new_phase = 0
        elif mode == "as_sc_or_fflo":
            provisional = int(row["provisional_phase_code"])
            new_phase = provisional if provisional != 0 else 2
        elif mode == "opposite_current":
            current = int(base_phase[i, j])
            new_phase = 0 if current != 0 else 2
        elif mode == "as_uniform":
            new_phase = 1
        elif mode == "as_fflo":
            new_phase = 2
        else:
            raise ValueError(mode)
        phase[i, j] = int(new_phase)
    return phase


def scenario_stats(
    name: str,
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    mode: str,
    note: str = "",
    scope: str = "global_stress",
) -> dict[str, Any]:
    base = context["phase_base"]
    cf = apply_flips(base, rows, mode)
    ns = boundary_shift_stats(cf, base, context["coords"], context["scale"], "normal_sc")
    uf = boundary_shift_stats(cf, base, context["coords"], context["scale"], "uniform_fflo")
    out = {
        "scenario": name,
        "mode": mode,
        "point_count_flipped": len(rows),
        "phase_map_change": phase_map_change(cf, base),
        "normal_sc_mean": ns["mean"],
        "normal_sc_median": ns["median"],
        "normal_sc_p95": ns["p95"],
        "normal_sc_max": ns["max"],
        "uniform_fflo_mean": uf["mean"],
        "uniform_fflo_median": uf["median"],
        "uniform_fflo_p95": uf["p95"],
        "uniform_fflo_max": uf["max"],
        "normal_sc_boundary_point_count": ns["boundary_point_count_current"],
        "uniform_fflo_boundary_point_count": uf["boundary_point_count_current"],
        "exceeds_boundary_tol": bool(
            (np.isfinite(ns["p95"]) and ns["p95"] > context["boundary_tol"])
            or (np.isfinite(uf["p95"]) and uf["p95"] > context["boundary_tol"])
        ),
        "exceeds_phase_map_tol": bool(phase_map_change(cf, base) > context["phase_map_tol"]),
        "scenario_scope": scope,
        "note": note,
    }
    return out


def build_clusters(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    pts = np.array([[float(r["kBT"]), float(r["JA"])] for r in rows], dtype=np.float64)
    pn = normalized(pts, context["scale"])
    # Cluster at the scale of a short continuous boundary segment, not a single
    # dense-grid edge.  A 4x boundary-shift tolerance radius groups adjacent
    # hard-risk points along the same local boundary band while keeping separated
    # boundary regions distinct.
    threshold = 4.0 * float(context["boundary_tol"])
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if cKDTree is not None:
        tree = cKDTree(pn)
        for i, js in enumerate(tree.query_ball_point(pn, r=threshold)):
            for j in js:
                if j > i:
                    union(i, j)
    else:
        for i in range(n):
            d = np.sqrt(np.sum((pn[i + 1 :] - pn[i]) ** 2, axis=1))
            for off in np.where(d <= threshold)[0]:
                union(i, i + 1 + int(off))

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def point_influence_rows(rows: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scenarios = [
            scenario_stats("single_as_normal", [row], context, "as_normal"),
            scenario_stats("single_as_sc_or_fflo", [row], context, "as_sc_or_fflo"),
            scenario_stats("single_opposite_current", [row], context, "opposite_current"),
        ]
        best_ns = max(scenarios, key=lambda x: float(x["normal_sc_p95"]) if np.isfinite(x["normal_sc_p95"]) else -1.0)
        best_uf = max(scenarios, key=lambda x: float(x["uniform_fflo_p95"]) if np.isfinite(x["uniform_fflo_p95"]) else -1.0)
        out.append(
            {
                "point_id": row["point_id"],
                "exact_index": row["exact_index"],
                "kBT": row["kBT"],
                "JA": row["JA"],
                "region": row["region"],
                "nearest_boundary_type": row["nearest_boundary_type"],
                "distance_to_nearest_boundary": row["distance_to_nearest_boundary"],
                "best_normal_sc_scenario": best_ns["mode"],
                "best_normal_sc_p95": best_ns["normal_sc_p95"],
                "best_normal_sc_max": best_ns["normal_sc_max"],
                "best_uniform_fflo_scenario": best_uf["mode"],
                "best_uniform_fflo_p95": best_uf["uniform_fflo_p95"],
                "best_uniform_fflo_max": best_uf["uniform_fflo_max"],
                "single_point_exceeds_tolerance": bool(
                    best_ns["normal_sc_p95"] > context["boundary_tol"]
                    or best_uf["uniform_fflo_p95"] > context["boundary_tol"]
                ),
            }
        )
    return out


def cluster_rows(clusters: list[list[dict[str, Any]]], context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_out: list[dict[str, Any]] = []
    targeted: list[dict[str, Any]] = []
    for cid, rows in enumerate(clusters):
        scenarios = [
            scenario_stats(f"cluster_{cid}_as_normal", rows, context, "as_normal"),
            scenario_stats(f"cluster_{cid}_as_sc_or_fflo", rows, context, "as_sc_or_fflo"),
            scenario_stats(f"cluster_{cid}_opposite_current", rows, context, "opposite_current"),
        ]
        worst = max(
            scenarios,
            key=lambda x: max(
                float(x["normal_sc_p95"]) if np.isfinite(x["normal_sc_p95"]) else -1.0,
                float(x["uniform_fflo_p95"]) if np.isfinite(x["uniform_fflo_p95"]) else -1.0,
            ),
        )
        reasons = Counter(str(r["rerun_required_reason"]) for r in rows)
        nearest = Counter(str(r["nearest_boundary_type"]) for r in rows)
        phase_counts = Counter(str(r["provisional_phase"]) for r in rows)
        influential = bool(worst["normal_sc_p95"] > context["boundary_tol"] or worst["uniform_fflo_p95"] > context["boundary_tol"])
        representative = min(rows, key=lambda r: float(r["distance_to_nearest_boundary"]))
        rec = "targeted_rerun_representative" if influential else "no_new_exact_needed_for_this_cluster"
        row = {
            "cluster_id": cid,
            "point_count": len(rows),
            "kBT_min": min(float(r["kBT"]) for r in rows),
            "kBT_max": max(float(r["kBT"]) for r in rows),
            "JA_min": min(float(r["JA"]) for r in rows),
            "JA_max": max(float(r["JA"]) for r in rows),
            "dominant_rerun_reason": reasons.most_common(1)[0][0] if reasons else "",
            "dominant_provisional_phase": phase_counts.most_common(1)[0][0] if phase_counts else "",
            "nearest_boundary": nearest.most_common(1)[0][0] if nearest else "",
            "worst_case_scenario": worst["mode"],
            "worst_normal_sc_p95": worst["normal_sc_p95"],
            "worst_uniform_fflo_p95": worst["uniform_fflo_p95"],
            "worst_phase_map_change": worst["phase_map_change"],
            "potentially_boundary_moving": influential,
            "representative_point_id": representative["point_id"],
            "representative_kBT": representative["kBT"],
            "representative_JA": representative["JA"],
            "recommended_action": rec,
        }
        cluster_out.append(row)
        if influential:
            targeted.append(
                {
                    "cluster_id": cid,
                    "point_id": representative["point_id"],
                    "exact_index": representative["exact_index"],
                    "kBT": representative["kBT"],
                    "JA": representative["JA"],
                    "provisional_phase": representative["provisional_phase"],
                    "predicted_phase": representative["predicted_phase"],
                    "rerun_required_reason": representative["rerun_required_reason"],
                    "nearest_boundary_type": representative["nearest_boundary_type"],
                    "distance_to_nearest_boundary": representative["distance_to_nearest_boundary"],
                    "recommended_action": "targeted exact cleanup representative for boundary-impact audit",
                }
            )
    return cluster_out, targeted


def assign_influence(
    classification_rows: list[dict[str, Any]],
    single_rows: list[dict[str, Any]],
    cluster_rows_data: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    single_by_id = {r["point_id"]: r for r in single_rows}
    cluster_by_point: dict[str, dict[str, Any]] = {}
    for c in cluster_rows_data:
        # Filled later from exact cluster membership in main.
        pass
    del cluster_by_point
    tol = float(context["boundary_tol"])
    for row in classification_rows:
        dist = float(row["distance_to_nearest_boundary"])
        s = single_by_id.get(row["point_id"])
        single_exceeds = bool(s and s.get("single_point_exceeds_tolerance"))
        if row.get("_cluster_potentially_boundary_moving", False) or single_exceeds:
            influence = "potentially_boundary_moving"
        elif dist > 4.0 * tol:
            influence = "non_influential"
        elif dist <= tol:
            influence = "locally_influential"
        else:
            influence = "non_influential"
        row["influence_class"] = influence
        row["single_point_exceeds_tolerance"] = single_exceeds


def update_cluster_membership(rows: list[dict[str, Any]], clusters: list[list[dict[str, Any]]], cluster_out: list[dict[str, Any]]) -> None:
    by_id = {r["point_id"]: r for r in rows}
    for cid, members in enumerate(clusters):
        c = cluster_out[cid]
        potential = bool(c["potentially_boundary_moving"])
        for member in members:
            row = by_id[member["point_id"]]
            row["cluster_id"] = cid
            row["_cluster_potentially_boundary_moving"] = potential


def make_figures(
    rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
    cluster_rows_data: list[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    phase = context["phase_base"]
    grid = context["grid_points"]
    extent = [grid[:, 0].min(), grid[:, 0].max(), grid[:, 1].min(), grid[:, 1].max()]
    pts = np.array([[float(r["kBT"]), float(r["JA"])] for r in rows])
    influence_colors = {
        "non_influential": "#888888",
        "locally_influential": "#1f77b4",
        "potentially_boundary_moving": "#d62728",
        "cannot_determine": "#9467bd",
    }
    region_colors = {
        "normal_SC_boundary_band": "#d62728",
        "uniform_FFLO_boundary_band": "#9467bd",
        "boundary_overlap": "#ff7f0e",
        "normal_interior": "#4c4c4c",
        "SC_interior": "#2ca02c",
        "far_interior": "#7f7f7f",
    }

    def base_map(ax: Any) -> None:
        ax.imshow(
            phase,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=ListedColormap(PHASE_COLORS),
            alpha=0.55,
            interpolation="nearest",
        )
        ns = context["normal_sc_boundary"]
        uf = context["uniform_fflo_boundary"]
        ax.scatter(ns[:, 0], ns[:, 1], s=1, c="black", alpha=0.35, label="normal/SC boundary")
        ax.scatter(uf[:, 0], uf[:, 1], s=1, c="#6a3d9a", alpha=0.45, label="uniform/FFLO boundary")
        ax.set_xlabel("kBT")
        ax.set_ylabel("JA")

    fig, ax = plt.subplots(figsize=(8, 6))
    base_map(ax)
    for region, color in region_colors.items():
        sub = [r for r in rows if r["region"] == region]
        if sub:
            arr = np.array([[float(r["kBT"]), float(r["JA"])] for r in sub])
            ax.scatter(arr[:, 0], arr[:, 1], s=18, c=color, label=region, edgecolor="white", linewidth=0.2)
    ax.set_title("Hard-Risk Points on Final Phase Map")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hard_risk_points_on_phase_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    base_map(ax)
    for influence, color in influence_colors.items():
        sub = [r for r in rows if r["influence_class"] == influence]
        if sub:
            arr = np.array([[float(r["kBT"]), float(r["JA"])] for r in sub])
            ax.scatter(arr[:, 0], arr[:, 1], s=22, c=color, label=influence, edgecolor="white", linewidth=0.2)
    ax.set_title("Hard-Risk Points by Boundary-Impact Class")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hard_risk_points_by_influence.png", dpi=180)
    plt.close(fig)

    scenarios = [r for r in counterfactual_rows if r["scenario"] in {"all_boundary_near_as_normal", "all_boundary_near_as_sc_or_fflo"}]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scenarios))
    ax.bar(x - 0.18, [float(r["normal_sc_p95"]) for r in scenarios], width=0.36, label="normal/SC p95")
    ax.bar(x + 0.18, [float(r["uniform_fflo_p95"]) for r in scenarios], width=0.36, label="uniform/FFLO p95")
    ax.axhline(context["boundary_tol"], color="black", linestyle="--", linewidth=1, label="boundary-shift tol")
    ax.set_xticks(x)
    ax.set_xticklabels([r["scenario"].replace("all_boundary_near_", "") for r in scenarios], rotation=20, ha="right")
    ax.set_ylabel("normalized boundary shift")
    ax.set_title("Counterfactual Boundary Shift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boundary_shift_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    base_map(ax)
    near_rows = [r for r in rows if float(r["distance_to_nearest_boundary"]) <= context["boundary_tol"]]
    if near_rows:
        arr = np.array([[float(r["kBT"]), float(r["JA"])] for r in near_rows])
        ax.scatter(arr[:, 0], arr[:, 1], s=18, c="#d62728", edgecolor="white", linewidth=0.2, label="boundary-near hard-risk")
    ax.set_title("Counterfactual Boundary-Overlay Inputs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "counterfactual_boundary_overlays.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    base_map(ax)
    for cluster in cluster_rows_data:
        if not cluster["potentially_boundary_moving"]:
            continue
        ax.scatter(
            float(cluster["representative_kBT"]),
            float(cluster["representative_JA"]),
            s=70,
            marker="*",
            c="#d62728",
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_title("Potentially Influential Cluster Representatives")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "influential_clusters.png", dpi=180)
    plt.close(fig)


def render_pdf(md_text: str, figure_paths: list[Path], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_lines: list[tuple[str, int, str]] = []
    for raw in md_text.splitlines():
        text = raw.replace("**", "").replace("`", "")
        if not text:
            rendered_lines.append(("", 9, "normal"))
            continue
        if text.startswith("# "):
            rendered_lines.append((text[2:], 15, "bold"))
            continue
        if text.startswith("## "):
            rendered_lines.append((text[3:], 12, "bold"))
            continue
        prefix = ""
        body = text
        if text.startswith("- "):
            prefix = "- "
            body = text[2:]
        width = 105 if not text.startswith("|") else 130
        wrapped = textwrap.wrap(body, width=width) or [body]
        rendered_lines.append((prefix + wrapped[0], 9, "normal"))
        for cont in wrapped[1:]:
            rendered_lines.append(("  " + cont, 9, "normal"))

    pages: list[list[tuple[str, int, str]]] = []
    current: list[tuple[str, int, str]] = []
    used = 0.0
    for item in rendered_lines:
        text, size, _ = item
        height = 0.038 if size >= 12 else (0.018 if not text else 0.027)
        if used + height > 0.88 and current:
            pages.append(current)
            current = []
            used = 0.0
        current.append(item)
        used += height
    if current:
        pages.append(current)

    with PdfPages(pdf_path) as pdf:
        for page in pages:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
            ax.axis("off")
            y = 1.0
            for text, size, weight in page:
                ax.text(0, y, text, ha="left", va="top", fontsize=size, weight=weight)
                y -= 0.038 if size >= 12 else (0.018 if not text else 0.027)
            pdf.savefig(fig)
            plt.close(fig)
        for fig_path in figure_paths:
            img = plt.imread(fig_path)
            fig, ax = plt.subplots(figsize=(8.27, 6.0))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(fig_path.name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    rows, context = classify_rows(data)
    boundary_near = [r for r in rows if float(r["distance_to_nearest_boundary"]) <= context["boundary_tol"]]
    deep_interior = [r for r in rows if r["region"] == "far_interior"]

    counterfactual: list[dict[str, Any]] = [
        scenario_stats(
            "all_boundary_near_as_normal",
            boundary_near,
            context,
            "as_normal",
            "All boundary-near hard-risk cells are set to normal.",
            scope="global_stress",
        ),
        scenario_stats(
            "all_boundary_near_as_sc_or_fflo",
            boundary_near,
            context,
            "as_sc_or_fflo",
            "All boundary-near hard-risk cells are set to their provisional non-normal label; provisional normal is set to FFLO as a conservative SC label.",
            scope="global_stress",
        ),
        scenario_stats(
            "all_boundary_near_opposite_current",
            boundary_near,
            context,
            "opposite_current",
            "Each boundary-near cell is flipped to the opposite normal-vs-SC side of the current monitor label.",
            scope="global_stress",
        ),
    ]

    single = point_influence_rows(boundary_near, context)
    clusters = build_clusters(boundary_near, context)
    cluster_table, targeted = cluster_rows(clusters, context)
    update_cluster_membership(rows, clusters, cluster_table)
    assign_influence(rows, single, cluster_table, context)

    # Recompute cluster membership influence after assign, because rows were updated.
    potentially = [r for r in rows if r["influence_class"] == "potentially_boundary_moving"]
    influential_points = [r for r in rows if r["influence_class"] in {"potentially_boundary_moving", "locally_influential"}]

    for cluster in cluster_table:
        counterfactual.append(
            {
                "scenario": f"cluster_{cluster['cluster_id']}_worst_case",
                "mode": cluster["worst_case_scenario"],
                "point_count_flipped": cluster["point_count"],
                "phase_map_change": cluster["worst_phase_map_change"],
                "normal_sc_mean": "",
                "normal_sc_median": "",
                "normal_sc_p95": cluster["worst_normal_sc_p95"],
                "normal_sc_max": "",
                "uniform_fflo_mean": "",
                "uniform_fflo_median": "",
                "uniform_fflo_p95": cluster["worst_uniform_fflo_p95"],
                "uniform_fflo_max": "",
                "normal_sc_boundary_point_count": "",
                "uniform_fflo_boundary_point_count": "",
                "exceeds_boundary_tol": cluster["potentially_boundary_moving"],
                "exceeds_phase_map_tol": bool(float(cluster["worst_phase_map_change"]) > context["phase_map_tol"]),
                "scenario_scope": "cluster_local",
                "note": cluster["recommended_action"],
            }
        )

    def finite_max(values: list[Any]) -> float:
        finite: list[float] = []
        for value in values:
            if value in ("", None):
                continue
            f = float(value)
            if np.isfinite(f):
                finite.append(f)
        return max(finite) if finite else math.nan

    global_stress_rows = [r for r in counterfactual if r.get("scenario_scope") == "global_stress"]
    cluster_counterfactual_rows = [r for r in counterfactual if r.get("scenario_scope") == "cluster_local"]

    strict_worst_ns = finite_max([r["normal_sc_p95"] for r in counterfactual])
    strict_worst_uf = finite_max([r["uniform_fflo_p95"] for r in counterfactual])
    global_worst_ns = finite_max([r["normal_sc_p95"] for r in global_stress_rows])
    global_worst_uf = finite_max([r["uniform_fflo_p95"] for r in global_stress_rows])
    cluster_worst_ns = finite_max([r["normal_sc_p95"] for r in cluster_counterfactual_rows])
    cluster_worst_uf = finite_max([r["uniform_fflo_p95"] for r in cluster_counterfactual_rows])
    single_worst_ns = finite_max([r["best_normal_sc_p95"] for r in single])
    single_worst_uf = finite_max([r["best_uniform_fflo_p95"] for r in single])
    local_worst_ns = finite_max([cluster_worst_ns, single_worst_ns])
    local_worst_uf = finite_max([cluster_worst_uf, single_worst_uf])
    strict_exceeds = bool(strict_worst_ns > context["boundary_tol"] or strict_worst_uf > context["boundary_tol"])
    global_stress_exceeds = bool(global_worst_ns > context["boundary_tol"] or global_worst_uf > context["boundary_tol"])
    local_exceeds = bool(local_worst_ns > context["boundary_tol"] or local_worst_uf > context["boundary_tol"])
    if not rows:
        decision = "Decision C"
        need_exact = "cannot_determine"
        decision_reason = "No hard-risk rows could be reconstructed."
    elif local_exceeds:
        decision = "Decision B"
        need_exact = "yes_targeted_only"
        decision_reason = "At least one single-point or local-cluster counterfactual exceeds the boundary-shift tolerance."
    else:
        decision = "Decision A"
        need_exact = "no"
        if global_stress_exceeds:
            decision_reason = (
                "No single point or continuous local hard-risk cluster exceeds the boundary-shift tolerance; "
                "the only exceedance is the deliberately synchronized all-boundary-near SC/FFLO stress test."
            )
        else:
            decision_reason = "No single-point, cluster-level, or global-stress p95 boundary shift exceeded tolerance."

    distance_rows = [
        {
            "point_id": r["point_id"],
            "kBT": r["kBT"],
            "JA": r["JA"],
            "distance_to_normal_sc_boundary": r["distance_to_normal_sc_boundary"],
            "distance_to_uniform_fflo_boundary": r["distance_to_uniform_fflo_boundary"],
            "distance_to_nearest_boundary": r["distance_to_nearest_boundary"],
            "nearest_boundary_type": r["nearest_boundary_type"],
            "region": r["region"],
            "influence_class": r["influence_class"],
        }
        for r in rows
    ]
    decision_row = {
        "decision": decision,
        "need_new_exact_calculation": need_exact,
        "hard_risk_total": len(rows),
        "boundary_near_hard_risk_count": len(boundary_near),
        "deep_interior_count": len(deep_interior),
        "potentially_boundary_moving_count": len(potentially),
        "targeted_rerun_point_count": len(targeted),
        "worst_normal_sc_boundary_shift_p95": strict_worst_ns,
        "worst_uniform_fflo_boundary_shift_p95": strict_worst_uf,
        "local_single_or_cluster_worst_normal_sc_p95": local_worst_ns,
        "local_single_or_cluster_worst_uniform_fflo_p95": local_worst_uf,
        "global_stress_worst_normal_sc_p95": global_worst_ns,
        "global_stress_worst_uniform_fflo_p95": global_worst_uf,
        "strict_global_stress_exceeds_tolerance": global_stress_exceeds,
        "local_single_or_cluster_exceeds_tolerance": local_exceeds,
        "boundary_shift_tolerance": context["boundary_tol"],
        "phase_map_tolerance": context["phase_map_tol"],
        "reason": decision_reason,
    }

    write_csv(
        TABLE_DIR / "hard_risk_point_classification.csv",
        rows,
        [
            "point_id",
            "exact_index",
            "kBT",
            "JA",
            "provisional_phase",
            "predicted_phase",
            "current_monitor_phase",
            "rerun_required",
            "trusted_exact",
            "training_eligible_exact",
            "rerun_required_reason",
            "q_expanded",
            "q_edge_hit",
            "q_edge_distance",
            "q_status",
            "q_expansion_trigger",
            "q_unresolved",
            "delta_boundary_ambiguous",
            "delta_status",
            "delta_unresolved",
            "distance_to_normal_sc_boundary",
            "distance_to_uniform_fflo_boundary",
            "distance_to_nearest_boundary",
            "nearest_boundary_type",
            "region",
            "influence_class",
            "cluster_id",
        ],
    )
    write_csv(
        TABLE_DIR / "hard_risk_boundary_distance.csv",
        distance_rows,
        list(distance_rows[0].keys()) if distance_rows else ["point_id"],
    )
    write_csv(
        TABLE_DIR / "counterfactual_boundary_shift.csv",
        counterfactual,
        [
            "scenario",
            "mode",
            "point_count_flipped",
            "phase_map_change",
            "normal_sc_mean",
            "normal_sc_median",
            "normal_sc_p95",
            "normal_sc_max",
            "uniform_fflo_mean",
            "uniform_fflo_median",
            "uniform_fflo_p95",
            "uniform_fflo_max",
            "normal_sc_boundary_point_count",
            "uniform_fflo_boundary_point_count",
            "exceeds_boundary_tol",
            "exceeds_phase_map_tol",
            "scenario_scope",
            "note",
        ],
    )
    write_csv(
        TABLE_DIR / "single_point_influence.csv",
        single,
        list(single[0].keys()) if single else ["point_id"],
    )
    write_csv(
        TABLE_DIR / "hard_risk_clusters.csv",
        cluster_table,
        list(cluster_table[0].keys()) if cluster_table else ["cluster_id"],
    )
    write_csv(
        TABLE_DIR / "influential_points.csv",
        influential_points,
        [
            "point_id",
            "exact_index",
            "kBT",
            "JA",
            "provisional_phase",
            "predicted_phase",
            "rerun_required_reason",
            "distance_to_normal_sc_boundary",
            "distance_to_uniform_fflo_boundary",
            "region",
            "influence_class",
            "cluster_id",
        ],
    )
    write_csv(
        TABLE_DIR / "targeted_rerun_points.csv",
        targeted,
        list(targeted[0].keys()) if targeted else [
            "cluster_id",
            "point_id",
            "exact_index",
            "kBT",
            "JA",
            "provisional_phase",
            "predicted_phase",
            "rerun_required_reason",
            "nearest_boundary_type",
            "distance_to_nearest_boundary",
            "recommended_action",
        ],
    )
    write_csv(TABLE_DIR / "audit_decision.csv", [decision_row], list(decision_row.keys()))

    make_figures(rows, counterfactual, cluster_table, context)

    region_counts = Counter(r["region"] for r in rows)
    influence_counts = Counter(r["influence_class"] for r in rows)
    near_ns = sum(1 for r in rows if float(r["distance_to_normal_sc_boundary"]) <= context["boundary_tol"])
    near_uf = sum(1 for r in rows if float(r["distance_to_uniform_fflo_boundary"]) <= context["boundary_tol"])
    overlap = sum(
        1
        for r in rows
        if float(r["distance_to_normal_sc_boundary"]) <= context["boundary_tol"]
        and float(r["distance_to_uniform_fflo_boundary"]) <= context["boundary_tol"]
    )
    md = f"""# Hard-Risk Boundary-Impact Audit

## Executive Summary

This is a report-only audit of existing tail-continuation artifacts.  It does not modify the original dataset, acquisition, exact oracle, StopController, phase criteria, tolerances, or Slurm state.

| question | answer |
|---|---|
| hard-risk total | {len(rows)} |
| boundary-near hard-risk points | {len(boundary_near)} |
| potentially boundary-moving points | {len(potentially)} |
| local single/cluster worst normal/SC p95 shift | {fmt(local_worst_ns)} |
| local single/cluster worst uniform/FFLO p95 shift | {fmt(local_worst_uf)} |
| strict global-stress worst normal/SC p95 shift | {fmt(strict_worst_ns)} |
| strict global-stress worst uniform/FFLO p95 shift | {fmt(strict_worst_uf)} |
| boundary-shift tolerance | {fmt(context['boundary_tol'])} |
| local single/cluster exceeds tolerance | {local_exceeds} |
| strict global-stress exceeds tolerance | {strict_exceeds} |
| decision | {decision} |
| need new exact calculation | {need_exact} |
| targeted rerun point count | {len(targeted)} |

## Scope and Method

The audit uses:

- `dataset_iter035.npz`;
- `iter034/exact_merged_iter034.npz`;
- `iter034/rerun_points.csv`;
- `iter034/selected_points_by_pool.csv`;
- `iter034/monitor_predictions_iter034.npz`;
- `iter034/stop_metrics_iter034.json`.

Boundary points are extracted from the final dense monitor phase map using the same crossing rule as StopController:

- normal/SC boundary: one side normal and the other side superconducting;
- uniform/FFLO boundary: one side uniform_SC and the other side FFLO.

Distances are normalized by the final monitor grid parameter ranges before nearest-neighbor comparisons.  Counterfactual flips are offline changes to copied dense-grid labels only; they are not exact reruns and are not written back to the dataset.

## Spatial Classification

Hard-risk points are selected by:

```text
rerun_required
or not trusted_exact
or not training_eligible_exact
or q_unresolved
or delta_unresolved
```

Counts by region:

```text
{dict(region_counts)}
```

Boundary-near counts:

```text
near normal/SC: {near_ns}
near uniform/FFLO: {near_uf}
boundary overlap: {overlap}
far interior: {len(deep_interior)}
```

## Counterfactual Boundary Impact

The two global worst-case scenarios give:

```text
all boundary-near as normal:
    normal/SC p95 shift = {fmt(counterfactual[0]['normal_sc_p95'])}
    uniform/FFLO p95 shift = {fmt(counterfactual[0]['uniform_fflo_p95'])}

all boundary-near as SC/FFLO:
    normal/SC p95 shift = {fmt(counterfactual[1]['normal_sc_p95'])}
    uniform/FFLO p95 shift = {fmt(counterfactual[1]['uniform_fflo_p95'])}
```

The all-SC/FFLO scenario is deliberately conservative: provisional normal cells are forced to FFLO.  If it creates distant uniform/FFLO islands along the normal/SC frontier, this is evidence for targeted cleanup, not evidence that the present phase map has already moved.

For the actual cleanup decision, this report separates that synchronized global stress test from local single-point and continuous-cluster tests.  The local tests are the relevant criterion for whether a small targeted rerun list is needed, because they ask whether any spatially coherent hard-risk segment can move the extracted main boundaries by more than the existing tolerance.

```text
local single/cluster normal/SC p95 shift = {fmt(local_worst_ns)}
local single/cluster uniform/FFLO p95 shift = {fmt(local_worst_uf)}
local single/cluster exceeds tolerance = {local_exceeds}
global synchronized stress exceeds tolerance = {global_stress_exceeds}
```

## Influence Classes

```text
{dict(influence_counts)}
```

Potentially boundary-moving points are those whose single-point or cluster-level conservative counterfactual p95 shift exceeds the existing boundary-shift tolerance.  They are not automatically accepted as real labels; they are candidates for targeted exact cleanup.

## Decision

Decision: **{decision}**.

Reason: {decision_reason}

If `Decision B`, only the representatives in `tables/targeted_rerun_points.csv` should be considered for a targeted cleanup package.  Do not start a new full active-learning loop.  If `Decision A`, no new exact calculation is required by this audit; the remaining hard-risk frontier should be shown as an uncertainty band/marker layer rather than silently promoted to definitive labels.

## Next Step

1. Review `tables/targeted_rerun_points.csv`.
2. If the list is nonempty, package only those representative points for exact cleanup.
3. After cleanup, rerun this report-only audit and update the hard-risk uncertainty markers on the phase map.
4. If cleanup confirms no boundary-moving effect, mark the main phase map publication-ready with an explicit hard-risk uncertainty band.

## Do-Not-Claim List

1. Do not treat provisional hard-risk labels as definitive phases.
2. Do not claim the all-selected surprise gate passed.
3. Do not modify any tolerance based on this audit.
4. Do not start another full active-learning loop from this audit.
5. Do not rerun every hard-risk point just because hard-risk count is large.

## Figures

![hard-risk points](figures/hard_risk_points_on_phase_map.png)

![influence classes](figures/hard_risk_points_by_influence.png)

![counterfactual overlays](figures/counterfactual_boundary_overlays.png)

![boundary shifts](figures/boundary_shift_distribution.png)

![clusters](figures/influential_clusters.png)
"""
    (OUT_DIR / "hard_risk_boundary_impact_audit.md").write_text(md, encoding="utf-8")
    decision_log = f"""# Decision Log

Decision: {decision}

Need new exact calculation: {need_exact}

Reason: {decision_reason}

Targeted rerun point count: {len(targeted)}

Important caveat: this is an offline counterfactual audit.  It does not modify the dataset and does not validate provisional hard-risk labels as definitive physics.
"""
    (OUT_DIR / "decision_log.md").write_text(decision_log, encoding="utf-8")
    render_pdf(
        md,
        [
            FIG_DIR / "hard_risk_points_on_phase_map.png",
            FIG_DIR / "hard_risk_points_by_influence.png",
            FIG_DIR / "counterfactual_boundary_overlays.png",
            FIG_DIR / "boundary_shift_distribution.png",
            FIG_DIR / "influential_clusters.png",
        ],
        OUT_DIR / "hard_risk_boundary_impact_audit.pdf",
    )
    print(json.dumps(decision_row, indent=2))
    print(OUT_DIR / "hard_risk_boundary_impact_audit.md")
    print(OUT_DIR / "hard_risk_boundary_impact_audit.pdf")


if __name__ == "__main__":
    main()
