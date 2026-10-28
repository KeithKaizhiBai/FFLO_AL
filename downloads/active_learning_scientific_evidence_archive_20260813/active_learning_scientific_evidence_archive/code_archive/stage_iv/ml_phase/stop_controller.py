from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ActiveLearningConfig
from .labels import PHASE_FFLO, PHASE_NORMAL, PHASE_UNIFORM_SC, phase_label


@dataclass
class StopConfig:
    min_iterations: int = 5
    patience: int = 4
    max_iterations: int = 50
    max_exact_calls: int | None = None
    warmup_reference_iters: int = 3
    map_tol: float = 0.002
    boundary_shift_tol: float | None = None
    surprise_tol: float = 0.05
    selected_a0_ratio_tol: float = 0.15
    qedge_rate_tol: float = 0.01
    rerun_rate_tol: float = 0.01
    coverage_tol: float | None = None
    allow_missing_boundary: bool = False
    required_pass_count: int = 4
    stop_surprise_mode: str = "all_selected"
    trusted_surprise_min_denominator: int = 64
    trusted_surprise_min_fraction: float = 0.25


def dense_grid_spacing_norm(cfg: ActiveLearningConfig) -> float:
    dkt = 1.0 / max(int(cfg.n_kt_candidates) - 1, 1)
    dja = 1.0 / max(int(cfg.n_ja_candidates) - 1, 1)
    return float(max(dkt, dja))


def default_stop_config(cfg: ActiveLearningConfig, max_iterations: int = 50) -> StopConfig:
    spacing = dense_grid_spacing_norm(cfg)
    return StopConfig(
        max_iterations=int(max_iterations),
        boundary_shift_tol=1.0 * spacing,
        coverage_tol=1.5 * spacing,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_monitor(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def _load_dataset_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 2), dtype=np.float64)
    with np.load(path, allow_pickle=False) as z:
        if "x" in z.files:
            return np.asarray(z["x"], dtype=np.float64).reshape(-1, 2)
        if "kT" in z.files and "JA" in z.files:
            return np.stack([z["kT"], z["JA"]], axis=1).astype(np.float64)
    return np.empty((0, 2), dtype=np.float64)


def _normalized_points(points: np.ndarray, cfg: ActiveLearningConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    offset = np.array([float(cfg.kt_min), float(cfg.ja_min)], dtype=np.float64)
    scale = np.array(
        [max(float(cfg.kt_max - cfg.kt_min), 1e-12), max(float(cfg.ja_max - cfg.ja_min), 1e-12)],
        dtype=np.float64,
    )
    return (points - offset) / scale


def _nearest_distances_norm(a: np.ndarray, b: np.ndarray, cfg: ActiveLearningConfig) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)
    if a.size == 0 or b.size == 0:
        return np.empty((0,), dtype=np.float64)
    an = _normalized_points(a, cfg)
    bn = _normalized_points(b, cfg)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        dist, _ = cKDTree(bn).query(an, k=1)
        return np.asarray(dist, dtype=np.float64)
    except Exception:
        out = np.full(an.shape[0], np.inf, dtype=np.float64)
        chunk = 512
        for start in range(0, an.shape[0], chunk):
            pts = an[start : start + chunk]
            d2 = np.sum((pts[:, None, :] - bn[None, :, :]) ** 2, axis=2)
            out[start : start + chunk] = np.sqrt(np.min(d2, axis=1))
        return out


def _boundary_points_from_phase(
    grid_points: np.ndarray,
    phase_pred: np.ndarray,
    full_shape: tuple[int, int],
    boundary_type: str,
) -> np.ndarray:
    points = np.asarray(grid_points, dtype=np.float64).reshape(-1, 2)
    phase = np.asarray(phase_pred, dtype=np.int64).reshape(full_shape)
    coords = points.reshape(full_shape + (2,))
    rows: list[np.ndarray] = []

    def crosses(a: int, b: int) -> bool:
        if boundary_type == "normal_sc":
            return (a == PHASE_NORMAL) != (b == PHASE_NORMAL)
        if boundary_type == "uniform_fflo":
            return {int(a), int(b)} == {PHASE_UNIFORM_SC, PHASE_FFLO}
        raise ValueError(f"Unsupported boundary type: {boundary_type}")

    n_ja, n_kt = phase.shape
    for i in range(n_ja):
        for j in range(n_kt - 1):
            if crosses(int(phase[i, j]), int(phase[i, j + 1])):
                rows.append(0.5 * (coords[i, j] + coords[i, j + 1]))
    for i in range(n_ja - 1):
        for j in range(n_kt):
            if crosses(int(phase[i, j]), int(phase[i + 1, j])):
                rows.append(0.5 * (coords[i, j] + coords[i + 1, j]))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(rows).astype(np.float64)


def phase_map_change(current: dict[str, np.ndarray], previous: dict[str, np.ndarray] | None) -> tuple[float | None, bool]:
    if previous is None or "phase_pred" not in previous or "phase_pred" not in current:
        return None, False
    cur = np.asarray(current["phase_pred"], dtype=np.int64).ravel()
    prev = np.asarray(previous["phase_pred"], dtype=np.int64).ravel()
    if cur.shape != prev.shape or cur.size == 0:
        return None, False
    return float(np.mean(cur != prev)), True


def boundary_shift(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray] | None,
    boundary_type: str,
    cfg: ActiveLearningConfig,
) -> dict[str, Any]:
    unavailable = {"value": None, "mean": None, "p95": None, "available": False, "n_current": 0, "n_previous": 0}
    if previous is None:
        return unavailable
    required = {"grid_points", "phase_pred", "full_shape"}
    if not required.issubset(current) or not required.issubset(previous):
        return unavailable
    shape_cur = tuple(int(x) for x in np.asarray(current["full_shape"]).ravel()[:2])
    shape_prev = tuple(int(x) for x in np.asarray(previous["full_shape"]).ravel()[:2])
    cur = _boundary_points_from_phase(current["grid_points"], current["phase_pred"], shape_cur, boundary_type)
    prev = _boundary_points_from_phase(previous["grid_points"], previous["phase_pred"], shape_prev, boundary_type)
    if cur.size == 0 or prev.size == 0:
        return {**unavailable, "n_current": int(cur.shape[0]), "n_previous": int(prev.shape[0])}
    d_cur = _nearest_distances_norm(cur, prev, cfg)
    d_prev = _nearest_distances_norm(prev, cur, cfg)
    dist = np.concatenate([d_cur, d_prev])
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {**unavailable, "n_current": int(cur.shape[0]), "n_previous": int(prev.shape[0])}
    p95 = float(np.percentile(finite, 95))
    return {
        "value": p95,
        "mean": float(np.mean(finite)),
        "p95": p95,
        "available": True,
        "n_current": int(cur.shape[0]),
        "n_previous": int(prev.shape[0]),
    }


def _bool_from_exact(data: dict[str, np.ndarray], key: str, n: int, default: bool) -> np.ndarray:
    if key not in data:
        return np.full(n, bool(default), dtype=bool)
    arr = np.asarray(data[key])
    if arr.ndim == 0:
        return np.full(n, bool(arr.item()), dtype=bool)
    if arr.shape[0] != n:
        return np.full(n, bool(default), dtype=bool)
    return arr.astype(bool)


def _exact_phase_rows(iter_dir: Path, iteration: int, cfg: ActiveLearningConfig) -> pd.DataFrame | None:
    selected_path = iter_dir / "selected_points_by_pool.csv"
    exact_path = iter_dir / f"exact_merged_iter{iteration:03d}.npz"
    if not selected_path.exists() or not exact_path.exists():
        return None
    selected = pd.read_csv(selected_path)
    if selected.empty or "predicted_phase_before_exact" not in selected:
        return None
    with np.load(exact_path, allow_pickle=False) as z:
        if "kT" not in z.files or "JA" not in z.files or "delta_opt" not in z.files or "q_opt" not in z.files:
            return None
        data = {k: z[k].copy() for k in z.files}
    n = int(np.asarray(data["kT"]).shape[0])
    if n <= 0:
        return None
    exact_phase = phase_label(data["delta_opt"], data["q_opt"], cfg.delta_eps, cfg.q_eps)
    exact_rows: dict[tuple[float, float], dict[str, Any]] = {}
    trusted_exact = _bool_from_exact(data, "trusted_exact", n, True)
    training_eligible = _bool_from_exact(data, "training_eligible_exact", n, True)
    rerun_required = _bool_from_exact(
        data,
        "rerun_required",
        n,
        False,
    ) | _bool_from_exact(data, "needs_rerun_exact", n, False)
    q_unresolved = _bool_from_exact(data, "q_unresolved", n, False)
    delta_unresolved = _bool_from_exact(data, "delta_unresolved", n, False)
    q_expanded = _bool_from_exact(data, "q_expanded", n, False)
    for idx, (k, j, p) in enumerate(zip(data["kT"], data["JA"], exact_phase, strict=False)):
        exact_rows[(round(float(k), 10), round(float(j), 10))] = {
            "exact_phase": int(p),
            "trusted_exact": bool(trusted_exact[idx]),
            "training_eligible_exact": bool(training_eligible[idx]),
            "rerun_required": bool(rerun_required[idx]),
            "q_unresolved": bool(q_unresolved[idx]),
            "delta_unresolved": bool(delta_unresolved[idx]),
            "q_expanded": bool(q_expanded[idx]),
        }

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        key = (round(float(row["kT"]), 10), round(float(row["JA"]), 10))
        exact = exact_rows.get(key)
        if exact is None:
            continue
        pred = int(row["predicted_phase_before_exact"])
        out = {
            "kT": float(row["kT"]),
            "JA": float(row["JA"]),
            "predicted_phase": pred,
            **exact,
        }
        out["surprise"] = bool(pred != int(exact["exact_phase"]))
        rows.append(out)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _rate_from_mask(rows: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n_total = int(rows.shape[0])
    n_den = int(np.sum(mask))
    n_surprise = int(np.sum(rows.loc[mask, "surprise"].to_numpy(dtype=bool))) if n_den else 0
    rate = float(n_surprise / n_den) if n_den else None
    return {
        "rate": rate,
        "available": bool(n_den > 0),
        "n_denominator": n_den,
        "n_surprise": n_surprise,
        "denominator_fraction_selected": float(n_den / n_total) if n_total else None,
    }


def label_surprise_metrics(
    iter_dir: Path,
    iteration: int,
    cfg: ActiveLearningConfig,
    stop_cfg: StopConfig | None = None,
) -> dict[str, Any]:
    rows = _exact_phase_rows(iter_dir, iteration, cfg)
    if rows is None or rows.empty:
        empty_metric = {
            "rate": None,
            "available": False,
            "n_denominator": 0,
            "n_surprise": 0,
            "denominator_fraction_selected": None,
        }
        return {
            "selected_count_matched": 0,
            "all_selected": dict(empty_metric),
            "trusted": dict(empty_metric),
            "hard_risk": dict(empty_metric),
            "trusted_surprise_denominator_valid": False,
            "trusted_surprise_min_denominator": int(
                stop_cfg.trusted_surprise_min_denominator if stop_cfg is not None else 0
            ),
            "trusted_surprise_min_fraction": float(
                stop_cfg.trusted_surprise_min_fraction if stop_cfg is not None else 0.0
            ),
            "trusted_fraction_selected": None,
            "hard_risk_fraction_selected": None,
            "q_unresolved_count": 0,
            "delta_unresolved_count": 0,
            "rerun_required_count": 0,
            "numerical_frontier_status": "insufficient_data",
        }

    trusted_mask = (
        rows["trusted_exact"].to_numpy(dtype=bool)
        & rows["training_eligible_exact"].to_numpy(dtype=bool)
        & ~rows["rerun_required"].to_numpy(dtype=bool)
        & ~rows["q_unresolved"].to_numpy(dtype=bool)
        & ~rows["delta_unresolved"].to_numpy(dtype=bool)
    )
    hard_risk_mask = (
        rows["rerun_required"].to_numpy(dtype=bool)
        | ~rows["trusted_exact"].to_numpy(dtype=bool)
        | ~rows["training_eligible_exact"].to_numpy(dtype=bool)
        | rows["q_unresolved"].to_numpy(dtype=bool)
        | rows["delta_unresolved"].to_numpy(dtype=bool)
    )
    all_mask = np.ones(rows.shape[0], dtype=bool)
    all_metric = _rate_from_mask(rows, all_mask)
    trusted_metric = _rate_from_mask(rows, trusted_mask)
    hard_metric = _rate_from_mask(rows, hard_risk_mask)

    min_den = int(stop_cfg.trusted_surprise_min_denominator) if stop_cfg is not None else 0
    min_frac = float(stop_cfg.trusted_surprise_min_fraction) if stop_cfg is not None else 0.0
    trusted_denominator_valid = bool(
        trusted_metric["available"]
        and int(trusted_metric["n_denominator"]) >= min_den
        and (
            trusted_metric["denominator_fraction_selected"] is not None
            and float(trusted_metric["denominator_fraction_selected"]) >= min_frac
        )
    )
    q_unresolved_count = int(np.sum(rows["q_unresolved"].to_numpy(dtype=bool)))
    delta_unresolved_count = int(np.sum(rows["delta_unresolved"].to_numpy(dtype=bool)))
    rerun_required_count = int(np.sum(rows["rerun_required"].to_numpy(dtype=bool)))
    if q_unresolved_count > 0 or delta_unresolved_count > 0:
        frontier_status = "unresolved"
    elif int(hard_metric["n_denominator"]) > 0:
        frontier_status = "active"
    else:
        frontier_status = "closed"
    return {
        "selected_count_matched": int(rows.shape[0]),
        "all_selected": all_metric,
        "trusted": trusted_metric,
        "hard_risk": hard_metric,
        "trusted_surprise_denominator_valid": trusted_denominator_valid,
        "trusted_surprise_min_denominator": min_den,
        "trusted_surprise_min_fraction": min_frac,
        "trusted_fraction_selected": trusted_metric["denominator_fraction_selected"],
        "hard_risk_fraction_selected": hard_metric["denominator_fraction_selected"],
        "q_unresolved_count": q_unresolved_count,
        "delta_unresolved_count": delta_unresolved_count,
        "rerun_required_count": rerun_required_count,
        "numerical_frontier_status": frontier_status,
    }


def label_surprise_rate(iter_dir: Path, iteration: int, cfg: ActiveLearningConfig) -> tuple[float | None, bool]:
    metrics = label_surprise_metrics(iter_dir, iteration, cfg)
    all_metric = metrics["all_selected"]
    return all_metric["rate"], bool(all_metric["available"])


def selected_a0_mean(iter_dir: Path) -> tuple[float | None, bool]:
    selected_path = iter_dir / "selected_points_by_pool.csv"
    if not selected_path.exists():
        return None, False
    selected = pd.read_csv(selected_path)
    if selected.empty or "A0_main" not in selected:
        return None, False
    vals = selected["A0_main"].to_numpy(dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, False
    return float(np.mean(vals)), True


def exact_rates(iter_dir: Path, iteration: int) -> dict[str, Any]:
    exact_path = iter_dir / f"exact_merged_iter{iteration:03d}.npz"
    if not exact_path.exists():
        return {
            "q_edge_trigger_rate": None,
            "rerun_required_rate": None,
            "q_edge_available": False,
            "rerun_available": False,
            "n_exact": 0,
        }
    with np.load(exact_path, allow_pickle=False) as z:
        if "kT" not in z.files:
            n = 0
        else:
            n = int(z["kT"].shape[0])
        if n <= 0:
            return {
                "q_edge_trigger_rate": None,
                "rerun_required_rate": None,
                "q_edge_available": False,
                "rerun_available": False,
                "n_exact": 0,
            }
        q_trigger = np.zeros(n, dtype=bool)
        q_available = False
        for key in ("q_expanded", "q_unresolved", "q_edge_hit"):
            if key in z.files:
                q_trigger |= z[key].astype(bool)
                q_available = True
        if "needs_rerun_exact" in z.files:
            rerun = z["needs_rerun_exact"].astype(bool)
            rerun_available = True
        elif "training_eligible_exact" in z.files:
            rerun = ~z["training_eligible_exact"].astype(bool)
            rerun_available = True
        else:
            rerun = np.zeros(n, dtype=bool)
            rerun_available = False
    return {
        "q_edge_trigger_rate": float(np.mean(q_trigger)) if q_available else None,
        "rerun_required_rate": float(np.mean(rerun)) if rerun_available else None,
        "q_edge_available": q_available,
        "rerun_available": rerun_available,
        "n_exact": n,
    }


def boundary_coverage_p95(
    current: dict[str, np.ndarray],
    dataset_points: np.ndarray,
    cfg: ActiveLearningConfig,
) -> tuple[float | None, bool, dict[str, int]]:
    if not {"grid_points", "phase_pred", "full_shape"}.issubset(current):
        return None, False, {}
    shape = tuple(int(x) for x in np.asarray(current["full_shape"]).ravel()[:2])
    normal_sc = _boundary_points_from_phase(current["grid_points"], current["phase_pred"], shape, "normal_sc")
    uniform_fflo = _boundary_points_from_phase(current["grid_points"], current["phase_pred"], shape, "uniform_fflo")
    counts = {"normal_sc": int(normal_sc.shape[0]), "uniform_fflo": int(uniform_fflo.shape[0])}
    boundary_points = [x for x in (normal_sc, uniform_fflo) if x.size]
    if not boundary_points or dataset_points.size == 0:
        return None, False, counts
    all_boundary = np.vstack(boundary_points)
    dist = _nearest_distances_norm(all_boundary, dataset_points, cfg)
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return None, False, counts
    return float(np.percentile(finite, 95)), True, counts


def _selected_a0_baseline(history: list[dict[str, Any]], current_mean: float | None, stop_cfg: StopConfig) -> float | None:
    vals: list[float] = []
    for item in history:
        mean = item.get("metrics", {}).get("selected_A0_mean")
        if mean is not None and np.isfinite(float(mean)):
            vals.append(float(mean))
        if len(vals) >= int(stop_cfg.warmup_reference_iters):
            break
    if len(vals) < int(stop_cfg.warmup_reference_iters) and current_mean is not None and np.isfinite(float(current_mean)):
        vals.append(float(current_mean))
    if not vals:
        return None
    return float(np.mean(vals[: int(stop_cfg.warmup_reference_iters)]))


def _condition(value: float | None, available: bool, tol: float, allow_missing: bool = False) -> bool:
    if not available or value is None or not np.isfinite(float(value)):
        return bool(allow_missing)
    return bool(float(value) < float(tol))


def evaluate_stop(
    run_dir: Path,
    iteration: int,
    current_dataset: Path,
    cfg: ActiveLearningConfig,
    stop_cfg: StopConfig,
) -> dict[str, Any]:
    iter_dir = run_dir / f"iter{iteration:03d}"
    current_monitor = _load_monitor(iter_dir / f"monitor_predictions_iter{iteration:03d}.npz")
    previous_monitor = None
    if iteration > 0:
        previous_monitor = _load_monitor(run_dir / f"iter{iteration - 1:03d}" / f"monitor_predictions_iter{iteration - 1:03d}.npz")

    history_path = run_dir / "stop_metrics_history.json"
    state_path = run_dir / "stop_state.json"
    history = _read_json(history_path, default=[])
    state = _read_json(state_path, default={"patience_counter": 0})

    if current_monitor is None:
        current_monitor = {}
    dataset_points = _load_dataset_points(current_dataset)

    map_change, map_available = phase_map_change(current_monitor, previous_monitor)
    shift_normal = boundary_shift(current_monitor, previous_monitor, "normal_sc", cfg)
    shift_fflo = boundary_shift(current_monitor, previous_monitor, "uniform_fflo", cfg)
    surprise_layers = label_surprise_metrics(iter_dir, iteration, cfg, stop_cfg)
    all_surprise = surprise_layers["all_selected"]
    trusted_surprise = surprise_layers["trusted"]
    hard_risk_surprise = surprise_layers["hard_risk"]
    stop_surprise_mode = str(stop_cfg.stop_surprise_mode)
    if stop_surprise_mode == "all_selected":
        surprise = all_surprise["rate"]
        surprise_available = bool(all_surprise["available"])
        surprise_denominator_valid = surprise_available
        surprise_condition_name = "label_surprise_all_selected"
    elif stop_surprise_mode == "trusted":
        surprise = trusted_surprise["rate"]
        surprise_available = bool(trusted_surprise["available"])
        surprise_denominator_valid = bool(surprise_layers["trusted_surprise_denominator_valid"])
        surprise_condition_name = "label_surprise_trusted"
    else:
        raise ValueError("stop_surprise_mode must be 'all_selected' or 'trusted'.")
    a0_mean, a0_available = selected_a0_mean(iter_dir)
    baseline = _selected_a0_baseline(history, a0_mean, stop_cfg)
    if a0_mean is not None and baseline is not None and baseline > 0.0:
        a0_ratio = float(a0_mean / baseline)
        a0_ratio_available = True
    else:
        a0_ratio = None
        a0_ratio_available = False
    rates = exact_rates(iter_dir, iteration)
    coverage, coverage_available, coverage_counts = boundary_coverage_p95(current_monitor, dataset_points, cfg)

    boundary_shift_tol = float(stop_cfg.boundary_shift_tol) if stop_cfg.boundary_shift_tol is not None else dense_grid_spacing_norm(cfg)
    coverage_tol = float(stop_cfg.coverage_tol) if stop_cfg.coverage_tol is not None else 1.5 * dense_grid_spacing_norm(cfg)

    c1 = _condition(map_change, map_available, stop_cfg.map_tol)
    c2 = _condition(shift_normal["value"], bool(shift_normal["available"]), boundary_shift_tol, stop_cfg.allow_missing_boundary)
    c3 = _condition(shift_fflo["value"], bool(shift_fflo["available"]), boundary_shift_tol, stop_cfg.allow_missing_boundary)
    c4 = bool(surprise_denominator_valid and _condition(surprise, surprise_available, stop_cfg.surprise_tol))
    q_ok = _condition(rates["q_edge_trigger_rate"], bool(rates["q_edge_available"]), stop_cfg.qedge_rate_tol)
    rerun_ok = _condition(rates["rerun_required_rate"], bool(rates["rerun_available"]), stop_cfg.rerun_rate_tol)
    c5 = _condition(coverage, coverage_available, coverage_tol, stop_cfg.allow_missing_boundary)

    conditions = {
        "C1_phase_map_change": c1,
        "C2_boundary_shift_normal_sc": c2,
        "C3_boundary_shift_uniform_fflo": c3,
        "C4_label_surprise_rate": c4,
        "C5_boundary_coverage_p95": c5,
    }
    diagnostic_conditions = {
        "selected_A0_ratio_below_tol": _condition(a0_ratio, a0_ratio_available, stop_cfg.selected_a0_ratio_tol),
        "q_edge_trigger_rate_below_tol": q_ok,
        "rerun_required_rate_below_tol": rerun_ok,
    }
    passed = int(sum(bool(v) for v in conditions.values()))
    completed_iterations = int(iteration) + 1
    exact_call_count = int(dataset_points.shape[0])

    hard_stop = False
    stop_reason = ""
    if completed_iterations >= int(stop_cfg.max_iterations):
        hard_stop = True
        stop_reason = "max_iterations"
    if stop_cfg.max_exact_calls is not None and exact_call_count >= int(stop_cfg.max_exact_calls):
        hard_stop = True
        stop_reason = "max_exact_calls"

    convergence_pass = bool(
        completed_iterations >= int(stop_cfg.min_iterations)
        and passed >= int(stop_cfg.required_pass_count)
    )
    patience_counter = int(state.get("patience_counter", 0))
    if convergence_pass:
        patience_counter += 1
    else:
        patience_counter = 0

    convergence_stop = bool(patience_counter >= int(stop_cfg.patience))
    stop = bool(hard_stop or convergence_stop)
    if convergence_stop and not stop_reason:
        stop_reason = "converged_main_phase_boundaries"
    numerical_cleanup_warning = bool(
        convergence_stop
        and (
            (rates["q_edge_available"] and not q_ok)
            or (rates["rerun_available"] and not rerun_ok)
        )
    )

    metrics = {
        "phase_map_change": map_change,
        "boundary_shift_normal_sc": shift_normal["value"],
        "boundary_shift_uniform_fflo": shift_fflo["value"],
        "label_surprise_rate": all_surprise["rate"],
        "label_surprise_all_selected": all_surprise["rate"],
        "label_surprise_trusted": trusted_surprise["rate"],
        "label_surprise_hard_risk": hard_risk_surprise["rate"],
        "label_surprise_selected_for_gate": surprise,
        "selected_A0_mean": a0_mean,
        "selected_A0_baseline": baseline,
        "selected_A0_ratio": a0_ratio,
        "q_edge_trigger_rate": rates["q_edge_trigger_rate"],
        "rerun_required_rate": rates["rerun_required_rate"],
        "boundary_coverage_p95": coverage,
    }
    availability = {
        "phase_map_change": map_available,
        "boundary_shift_normal_sc": bool(shift_normal["available"]),
        "boundary_shift_uniform_fflo": bool(shift_fflo["available"]),
        "label_surprise_rate": bool(all_surprise["available"]),
        "label_surprise_all_selected": bool(all_surprise["available"]),
        "label_surprise_trusted": bool(trusted_surprise["available"]),
        "label_surprise_hard_risk": bool(hard_risk_surprise["available"]),
        "label_surprise_selected_for_gate": bool(surprise_available),
        "trusted_surprise_denominator_valid": bool(surprise_layers["trusted_surprise_denominator_valid"]),
        "selected_A0_ratio": a0_ratio_available,
        "q_edge_trigger_rate": bool(rates["q_edge_available"]),
        "rerun_required_rate": bool(rates["rerun_available"]),
        "boundary_coverage_p95": coverage_available,
    }

    out = {
        "iteration": int(iteration),
        "completed_iterations": completed_iterations,
        "exact_call_count": exact_call_count,
        "metrics": metrics,
        "metric_availability": availability,
        "boundary_details": {
            "normal_sc": shift_normal,
            "uniform_fflo": shift_fflo,
            "coverage_boundary_counts": coverage_counts,
        },
        "surprise_details": {
            "stop_surprise_mode": stop_surprise_mode,
            "selected_gate_metric": surprise_condition_name,
            "all_selected": all_surprise,
            "trusted": trusted_surprise,
            "hard_risk": hard_risk_surprise,
            "selected_count_matched": int(surprise_layers["selected_count_matched"]),
            "trusted_surprise_denominator_valid": bool(surprise_layers["trusted_surprise_denominator_valid"]),
            "trusted_surprise_min_denominator": int(surprise_layers["trusted_surprise_min_denominator"]),
            "trusted_surprise_min_fraction": float(surprise_layers["trusted_surprise_min_fraction"]),
            "trusted_fraction_selected": surprise_layers["trusted_fraction_selected"],
            "hard_risk_fraction_selected": surprise_layers["hard_risk_fraction_selected"],
            "rerun_required_count": int(surprise_layers["rerun_required_count"]),
            "q_unresolved_count": int(surprise_layers["q_unresolved_count"]),
            "delta_unresolved_count": int(surprise_layers["delta_unresolved_count"]),
        },
        "conditions": conditions,
        "diagnostic_conditions": diagnostic_conditions,
        "passed_condition_count": passed,
        "required_pass_count": int(stop_cfg.required_pass_count),
        "mandatory_gates": [],
        "mandatory_gates_pass": True,
        "convergence_pass": convergence_pass,
        "main_phase_convergence_pass": convergence_pass,
        "patience_counter": patience_counter,
        "patience": int(stop_cfg.patience),
        "hard_stop": hard_stop,
        "stop": stop,
        "stop_reason": stop_reason,
        "main_phase_converged": bool(convergence_stop),
        "numerical_frontier_status": str(surprise_layers["numerical_frontier_status"]),
        "publication_ready": bool(convergence_stop and c5 and surprise_layers["numerical_frontier_status"] == "closed"),
        "publication_ready_reason": (
            ""
            if bool(convergence_stop and c5 and surprise_layers["numerical_frontier_status"] == "closed")
            else (
                "hard_risk_boundary_impact_not_audited"
                if surprise_layers["numerical_frontier_status"] != "closed"
                else "main_phase_or_boundary_coverage_not_converged"
            )
        ),
        "numerical_cleanup_warning": numerical_cleanup_warning,
        "numerical_cleanup_list": str(iter_dir / "rerun_points.csv") if numerical_cleanup_warning else "",
        "thresholds": {
            "map_tol": float(stop_cfg.map_tol),
            "boundary_shift_tol": boundary_shift_tol,
            "surprise_tol": float(stop_cfg.surprise_tol),
            "trusted_surprise_min_denominator": int(stop_cfg.trusted_surprise_min_denominator),
            "trusted_surprise_min_fraction": float(stop_cfg.trusted_surprise_min_fraction),
            "selected_A0_ratio_tol": float(stop_cfg.selected_a0_ratio_tol),
            "qedge_rate_tol": float(stop_cfg.qedge_rate_tol),
            "rerun_rate_tol": float(stop_cfg.rerun_rate_tol),
            "coverage_tol": coverage_tol,
        },
        "stop_config": asdict(stop_cfg),
    }

    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / f"stop_metrics_iter{iteration:03d}.json").write_text(
        json.dumps(_json_safe(out), indent=2),
        encoding="utf-8",
    )

    history = [item for item in history if int(item.get("iteration", -1)) != int(iteration)]
    history.append(out)
    history.sort(key=lambda item: int(item.get("iteration", -1)))
    history_path.write_text(json.dumps(_json_safe(history), indent=2), encoding="utf-8")

    state = {
        "last_iteration": int(iteration),
        "patience_counter": patience_counter,
        "stop": stop,
        "stop_reason": stop_reason,
        "selected_A0_baseline": baseline,
    }
    state_path.write_text(json.dumps(_json_safe(state), indent=2), encoding="utf-8")
    return out


def print_stop_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    availability = result["metric_availability"]
    conditions = result["conditions"]
    names = [
        ("C1_phase_map_change", "phase_map_change"),
        ("C2_boundary_shift_normal_sc", "boundary_shift_normal_sc"),
        ("C3_boundary_shift_uniform_fflo", "boundary_shift_uniform_fflo"),
        ("C4_label_surprise_rate", "label_surprise_selected_for_gate"),
        ("C5_selected_A0_ratio", "selected_A0_ratio"),
        ("C6_qedge_and_rerun_rates", "q_edge_trigger_rate"),
        ("C7_boundary_coverage_p95", "boundary_coverage_p95"),
    ]
    for condition_key, metric_key in names:
        value = metrics.get(metric_key)
        available = availability.get(metric_key)
        passed = conditions.get(condition_key)
        if condition_key == "C6_qedge_and_rerun_rates":
            value = {
                "q_edge_trigger_rate": metrics.get("q_edge_trigger_rate"),
                "rerun_required_rate": metrics.get("rerun_required_rate"),
            }
            available = availability.get("q_edge_trigger_rate") and availability.get("rerun_required_rate")
        print(f"[stop] {condition_key}: value={value}, available={available}, pass={passed}")
    surprise_details = result.get("surprise_details", {})
    if surprise_details:
        print(
            "[stop] surprise mode="
            f"{surprise_details.get('stop_surprise_mode')}; "
            f"all_selected={metrics.get('label_surprise_all_selected')}; "
            f"trusted={metrics.get('label_surprise_trusted')}; "
            f"hard_risk={metrics.get('label_surprise_hard_risk')}; "
            f"trusted_denominator_valid={surprise_details.get('trusted_surprise_denominator_valid')}"
        )
    print(
        "[stop] convergence_pass="
        f"{result['passed_condition_count']}/5 >= {result['required_pass_count']}, "
        f"mandatory_gates={result['mandatory_gates_pass']}: {result['convergence_pass']}; "
        f"patience={result['patience_counter']}/{result['patience']}; stop={result['stop']}; "
        f"reason={result['stop_reason'] or 'none'}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate active-learning convergence stop metrics.")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--current-dataset", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("ML_Phase"))
    p.add_argument("--min-iterations", type=int, default=5)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--max-exact-calls", type=int, default=None)
    p.add_argument("--warmup-reference-iters", type=int, default=3)
    p.add_argument("--map-tol", type=float, default=0.002)
    p.add_argument("--boundary-shift-tol", type=float, default=None)
    p.add_argument("--surprise-tol", type=float, default=0.05)
    p.add_argument("--selected-a0-ratio-tol", type=float, default=0.15)
    p.add_argument("--qedge-rate-tol", type=float, default=0.01)
    p.add_argument("--rerun-rate-tol", type=float, default=0.01)
    p.add_argument("--coverage-tol", type=float, default=None)
    p.add_argument("--allow-missing-boundary", action="store_true")
    p.add_argument("--stop-surprise-mode", choices=["all_selected", "trusted"], default="all_selected")
    p.add_argument("--trusted-surprise-min-denominator", type=int, default=64)
    p.add_argument("--trusted-surprise-min-fraction", type=float, default=0.25)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = ActiveLearningConfig(output_root=str(args.output_root))
    base = default_stop_config(cfg, max_iterations=args.max_iterations)
    stop_cfg = StopConfig(
        min_iterations=args.min_iterations,
        patience=args.patience,
        max_iterations=args.max_iterations,
        max_exact_calls=args.max_exact_calls,
        warmup_reference_iters=args.warmup_reference_iters,
        map_tol=args.map_tol,
        boundary_shift_tol=args.boundary_shift_tol if args.boundary_shift_tol is not None else base.boundary_shift_tol,
        surprise_tol=args.surprise_tol,
        selected_a0_ratio_tol=args.selected_a0_ratio_tol,
        qedge_rate_tol=args.qedge_rate_tol,
        rerun_rate_tol=args.rerun_rate_tol,
        coverage_tol=args.coverage_tol if args.coverage_tol is not None else base.coverage_tol,
        allow_missing_boundary=bool(args.allow_missing_boundary),
        stop_surprise_mode=str(args.stop_surprise_mode),
        trusted_surprise_min_denominator=int(args.trusted_surprise_min_denominator),
        trusted_surprise_min_fraction=float(args.trusted_surprise_min_fraction),
    )
    result = evaluate_stop(
        run_dir=args.run_dir,
        iteration=args.iteration,
        current_dataset=args.current_dataset,
        cfg=cfg,
        stop_cfg=stop_cfg,
    )
    print_stop_summary(result)


if __name__ == "__main__":
    main()
